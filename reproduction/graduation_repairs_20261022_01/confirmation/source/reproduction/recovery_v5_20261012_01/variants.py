from dataclasses import asdict
from recovery_controller import RecoveryController,Config
from reproduction.recovery_v2_20261012_01.recovery_controller import RecoveryController as V2,Config as V2Config
from reproduction.recovery_v4_20261012_01.recovery_controller import RecoveryController as V4,Config as V4Config
VERSIONS=['v2','v4','timing_only','side_only','v5']
def controller_class(version):
    if version=='v2':return V2,V2Config
    if version=='v4':return V4,V4Config
    assert version in VERSIONS,version
    return RecoveryController,Config
def config(version):
    if version in ['v2','v4']:return {}
    return asdict(Config(time_budget=version in ['timing_only','v5'],side_clearance=version in ['side_only','v5']))
