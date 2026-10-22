"""Explicit state diagnostic utilities. Object observations are privileged."""
import copy
from collections import deque
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data/can_ph_lowdim_v15.hdf5'
OBS_KEYS = ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object']


def pack_obs(obs):
    return np.concatenate([np.asarray(obs[k if k in obs else 'object-state']).reshape(-1) for k in OBS_KEYS]).astype(np.float32)


def read_episodes(split='train'):
    splits = json.loads((ROOT / 'data/splits.json').read_text())
    episodes = []
    with h5py.File(DATA, 'r') as file:
        for name in splits[split]:
            d = file['data'][name]
            x = np.concatenate([d['obs'][k][:] for k in OBS_KEYS], axis=-1).astype(np.float32)
            xn = np.concatenate([d['next_obs'][k][:] for k in OBS_KEYS], axis=-1).astype(np.float32)
            episodes.append({'name':name, 'x':x, 'xn':xn, 'a':d['actions'][:].astype(np.float32),
                             'reward':d['rewards'][:].astype(np.float32)})
    return episodes


def make_env(visual=False):
    import os
    os.environ.setdefault('MUJOCO_GL','wgl')
    import robosuite
    with h5py.File(DATA,'r') as file:
        meta = json.loads(file['data'].attrs['env_args'])
    kwargs = copy.deepcopy(meta['env_kwargs'])
    kwargs.update(has_renderer=False, has_offscreen_renderer=visual, use_camera_obs=visual,
                  use_object_obs=True, hard_reset=False, ignore_done=True)
    if visual:
        kwargs.update(camera_names=['agentview'], camera_heights=96, camera_widths=96)
    return robosuite.make(meta['env_name'], **kwargs)


def restore_demo(env, demo, index=0, xml=True):
    if xml:
        model = demo.attrs['model_file']
        if isinstance(model,bytes):
            model=model.decode()
        model=env.edit_model_xml(model)
        env.reset_from_xml_string(model)
    else:
        # Match XML resets: random initialization must not leak into controllers.
        previous = env.deterministic_reset
        try:
            env.deterministic_reset = True
            env.reset()
        finally:
            env.deterministic_reset = previous
    env.sim.set_state_from_flattened(demo['states'][index])
    env.sim.forward()
    env.timestep=0
    # MuJoCo state restoration does not update Python-side controller caches.
    for robot in env.robots:
        robot.composite_controller.update_state()
        for controller in robot.part_controllers.values():
            controller.update(force=True)
            controller.reset_goal()
    # Relative-object sensors precede their absolute-pose dependencies in 1.5.1.
    env._obs_cache.clear()
    env._get_observations(force_update=True)
    return env._get_observations(force_update=True)


class ChunkPolicy(nn.Module):
    """Small mixture-density action-chunk BC; train with likelihood, not MSE."""
    def __init__(self, dim, horizon=8, mixtures=5, width=256):
        super().__init__()
        self.horizon=horizon
        self.mixtures=mixtures
        self.net=nn.Sequential(nn.Linear(dim,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU(),
                               nn.Linear(width,width),nn.ReLU())
        self.head=nn.Linear(width,mixtures*(horizon*7*2+1))
        self.register_buffer('mean',torch.zeros(dim))
        self.register_buffer('scale',torch.ones(dim))

    def forward(self,x):
        h=self.head(self.encode(x)).reshape(-1,self.mixtures,self.horizon*7*2+1)
        logits=h[:,:,0]
        mean=torch.tanh(h[:,:,1:1+self.horizon*7]).reshape(-1,self.mixtures,self.horizon,7)
        scale=nn.functional.softplus(h[:,:,1+self.horizon*7:]).reshape_as(mean).clamp(.02,1.)
        return logits,mean,scale

    def encode(self,x):
        return self.net((x-self.mean)/self.scale)

    def loss(self,x,y):
        logits,mean,scale=self(x)
        logp=(-.5*((y[:,None]-mean)/scale).square()-scale.log()-.5*np.log(2*np.pi)).sum((-1,-2))
        return -torch.logsumexp(logits.log_softmax(-1)+logp,-1).mean()/(self.horizon*7)

    @torch.no_grad()
    def best(self,x):
        logits,means,_=self(x)
        return means[torch.arange(len(x),device=x.device),logits.argmax(-1)]


class RecurrentChunkPolicy(ChunkPolicy):
    def __init__(self,dim,horizon=8,mixtures=5,width=256,history=10):
        super().__init__(dim,horizon,mixtures,width)
        self.net=nn.Sequential(nn.Linear(dim,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
        self.rnn=nn.GRU(128,width,batch_first=True)
        self.history=history

    def encode(self,x):
        z=self.net((x-self.mean)/self.scale)
        sequence,_=self.rnn(z)
        return sequence[:,-1]


class OnlineHistoryPolicy:
    def __init__(self,model):
        self.model=model
        self.horizon=model.horizon
        self.reset()

    def reset(self):
        self.frames=deque(maxlen=self.model.history)

    @torch.no_grad()
    def best(self,x):
        assert x.shape[0]==1, 'Online policy accepts one environment at a time.'
        if not self.frames:
            self.frames.extend([x[0]]*self.model.history)
        else:
            self.frames.append(x[0])
        return self.model.best(torch.stack(list(self.frames))[None])


def policy_windows(episodes,horizon):
    xx,aa=[] ,[]
    for ep in episodes:
        n=len(ep['a'])-horizon+1
        if n<=0:
            continue
        xx.append(ep['x'][:n])
        aa.append(np.stack([ep['a'][t:t+horizon] for t in range(n)]))
    return np.concatenate(xx),np.concatenate(aa)


def load_policy(path=None):
    path=path or ROOT/'policy/policy.pt'
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    cfg=checkpoint['config']
    model=(RecurrentChunkPolicy(cfg['dim'],cfg['horizon'],cfg['mixtures'],cfg['width'],cfg['history'])
           if cfg.get('architecture')=='recurrent' else ChunkPolicy(cfg['dim'],cfg['horizon'],cfg['mixtures'],cfg['width']))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return OnlineHistoryPolicy(model) if cfg.get('architecture')=='recurrent' else model
