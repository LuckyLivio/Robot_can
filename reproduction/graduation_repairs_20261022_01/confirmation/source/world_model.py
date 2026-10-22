"""Small action-conditioned state diagnostic; vision encoder is a later stage."""
import numpy as np
import torch
from torch import nn


class WorldModel(nn.Module):
    def __init__(self,dim=23,width=128,future_only=False):
        super().__init__()
        self.init=nn.Sequential(nn.Linear(dim,width),nn.Tanh())
        self.cell=nn.GRUCell(dim+7,width)
        self.delta=nn.Sequential(nn.Linear(width,width),nn.ReLU(),nn.Linear(width,dim))
        self.future_only=future_only
        self.labels=nn.Sequential(nn.Linear(dim if future_only else dim+width,width),nn.ReLU(),nn.Linear(width,2))
        self.register_buffer('mean',torch.zeros(dim))
        self.register_buffer('scale',torch.ones(dim))

    def forward(self,x,actions):
        z=(x-self.mean)/self.scale
        h=self.init(z)
        states=[];labels=[]
        for t in range(actions.shape[1]):
            h=self.cell(torch.cat([z,actions[:,t]],-1),h)
            z=z+self.delta(h)*.1
            # v1: the planner must pass through the predicted physical state.
            # The legacy v0 hidden-state readout remains loadable for diagnosis.
            labels.append(self.labels(z if self.future_only else torch.cat([z,h],-1)))
            states.append(z*self.scale+self.mean)
        return torch.stack(states,1),torch.stack(labels,1)


class DirectScore(nn.Module):
    """No predicted future state: current observation + flattened action chunk."""
    def __init__(self,dim=23,horizon=8,width=128):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dim+horizon*7,width),nn.ReLU(),nn.Linear(width,width),
                               nn.ReLU(),nn.Linear(width,horizon*2))
        self.horizon=horizon
        self.register_buffer('mean',torch.zeros(dim))
        self.register_buffer('scale',torch.ones(dim))

    def forward(self,x,actions):
        inp=torch.cat([(x-self.mean)/self.scale,actions.flatten(1)],-1)
        return None,self.net(inp).reshape(-1,self.horizon,2)


def score(labels,risk_weight=.5):
    # Progress readout is bounded; no deployment simulator reward is queried.
    progress=labels[...,0].sigmoid()
    risk=labels[...,1].sigmoid()
    return progress[:,-1]+.5*progress.mean(-1)-risk_weight*risk.max(-1).values


def target_score(labels,risk_weight=.5):
    return labels[:,-1,0]+.5*labels[:,:,0].mean(-1)-risk_weight*labels[:,:,1].max(-1).values


def grouped_rank_loss(pred,target,groups,k,temperature=.1):
    pred=pred.reshape(groups,k)
    target=target.reshape(groups,k)
    pd=pred[:,:,None]-pred[:,None,:]
    td=target[:,:,None]-target[:,None,:]
    mask=(td.abs()>.025)&torch.triu(torch.ones((k,k),device=pred.device,dtype=torch.bool),diagonal=1)[None]
    if not mask.any():return pred.sum()*0
    return nn.functional.softplus(-td.sign()[mask]*pd[mask]/temperature).mean()


def rank_metrics(pred,target,groups,k):
    pred=np.asarray(pred).reshape(groups,k);target=np.asarray(target).reshape(groups,k)
    choice=pred.argmax(-1)
    chosen=target[np.arange(groups),choice]
    regret=target.max(-1)-chosen
    pd=pred[:,:,None]-pred[:,None,:];td=target[:,:,None]-target[:,None,:]
    mask=(np.abs(td)>.025)&np.triu(np.ones((k,k),bool),1)[None]
    return {'mean_regret':float(regret.mean()),'wrong_pair_rate':float(np.mean((pd[mask]*td[mask])<0)) if mask.any() else None,
            'optimal_within_025':float(np.mean(regret<=.025)), 'groups':groups,'nontrivial_pairs':int(mask.sum())}
