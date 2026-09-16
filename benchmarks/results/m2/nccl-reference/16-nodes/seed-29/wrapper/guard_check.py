"""Exercise restart and concurrent-start rejection without invoking training."""
import tempfile,subprocess,sys,fcntl,json,hashlib
from pathlib import Path
script=Path(__file__).with_name('run-reference.sh');base=script.parent.parent
results=[]
for case in ['existing-marker','held-lock']:
 with tempfile.TemporaryDirectory(prefix='guard-check-',dir=base) as directory:
  root=Path(directory);log=root/'host.log';original=b'original log retained\n';log.write_bytes(original)
  marker=root/'started-at.txt';lock=None
  if case=='existing-marker':marker.write_bytes(b'immutable-start\n')
  else:
   lock=(root/'launch.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  with log.open('ab') as stream:
   p=subprocess.run(['bash',str(script),str(root),'0','127.0.0.1','16','29','official'],stdout=stream,stderr=subprocess.STDOUT,timeout=10)
  assert p.returncode==73,(case,p.returncode)
  assert log.read_bytes().startswith(original)
  assert not (root/'final-checkpoints').exists()
  if case=='existing-marker':assert marker.read_bytes()==b'immutable-start\n'
  else:assert not marker.exists();lock.close()
  results.append({'case':case,'exit':p.returncode,'original_log_prefix_preserved':True,'training_not_invoked':True})
print(json.dumps({'status':'verified','guards':results,'wrapper_sha256':hashlib.sha256(script.read_bytes()).hexdigest()}))
