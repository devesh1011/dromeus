"""Read-only verification of the frozen image and experiment before launch."""
import hashlib
import json
import platform
import sys
from pathlib import Path
import torch
from benchmarks.noloco.experiment import load_frozen_experiment, RunSelector
from benchmarks.noloco_reference.optimizer import ReferenceNoLoCoConfig, apply_outer_step, outer_gradient

path=Path(sys.argv[1])
assert hashlib.sha256(path.read_bytes()).hexdigest()=='c588c5b389bf0b506723786b351f4bd816608230b29628bc9596021fa57ea5cd'
experiment=load_frozen_experiment(path)
run=experiment.resolve(RunSelector(profile='official',world_size=16,benchmark_seed=29))
assert run.run.round_count==500 and run.algorithm.inner_steps==50
assert run.run.checkpoint.sha256=='1b838fe02f6c2de2b0f54cb6aae51a4c0effcda0edc2c34dd423dd3a784e4353'
assert torch.__version__=='2.12.1+cu130' and torch.version.cuda=='13.0'
assert torch.backends.cudnn.version()==92000 and torch.cuda.nccl.version()==(2,29,7)
assert torch.cuda.get_device_name(0)=='NVIDIA A10G'
fixture=json.loads(Path(__file__).with_name('upstream-fixture.json').read_text())
for node in fixture['nodes']:
    peer=fixture['nodes'][1-node['rank']]
    def tensor(values): return {'w':torch.tensor(values,dtype=torch.float32)}
    result=apply_outer_step(slow_weights=tensor(node['slow_weights']),outer_momentum=tensor(node['outer_momentum_before']),local_outer_gradient=outer_gradient(tensor(node['slow_weights']),tensor(node['fast_weights'])),peer_outer_gradient=outer_gradient(tensor(peer['slow_weights']),tensor(peer['fast_weights'])),peer_slow_weights=tensor(peer['slow_weights']),config=ReferenceNoLoCoConfig(alpha=.5,beta=.7,gamma=.7))
    torch.testing.assert_close(result.slow_weights['w'],tensor(node['slow_weights_after'])['w'],atol=1e-6,rtol=0)
    torch.testing.assert_close(result.outer_momentum['w'],tensor(node['outer_momentum_after'])['w'],atol=1e-6,rtol=0)
files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('benchmarks/noloco_reference').glob('*.py')}
print(json.dumps({'status':'verified','python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'nccl':torch.cuda.nccl.version(),'gpu':torch.cuda.get_device_name(0),'run_config_sha256':run.run_config_sha256,'pairing_digest':run.run.pairing_digest,'initial_checkpoint_sha256':run.run.checkpoint.sha256,'round_count':run.run.round_count,'reference_source_files':files},sort_keys=True))
