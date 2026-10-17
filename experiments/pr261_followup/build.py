import os,re,shutil,subprocess,hashlib
from pathlib import Path
h=Path(__file__).resolve().parents[2]
r=Path(os.environ.get('INFERENA_ARTIFACT_DIR',str(Path(__file__).parent/'results')));r.mkdir(parents=True,exist_ok=True)
env=os.environ|{'CARGO_BUILD_JOBS':'4','CARGO_INCREMENTAL':'0'}
files=[h/'Cargo.toml',h/'frameworks/meganeura/Cargo.toml',h/'Cargo.lock'];original={p:p.read_bytes() for p in files}
try:
 for name,meganeura,blade in [
  ('base','41a15d31b652f95aa9fb427dea47acf052c1e9bd','96c1dc731b1d5e133a6c0a8a020014619ff2e434'),
  ('head','de90b900fff9f87adfa2ee7c46c30ed0f70f1286','4845d6529c38b07718a624bf3c4f2ab4317c9bf4')]:
  files[0].write_text(re.sub(r'^meganeura = .*$', 'meganeura = { git = "https://github.com/kvark/meganeura", rev = "'+meganeura+'", features = ["models", "hf-hub"] }',original[files[0]].decode(),flags=re.M))
  files[1].write_text(re.sub(r'^blade-graphics = .*$', 'blade-graphics = { version = "0.9", git = "https://github.com/kvark/blade", rev = "'+blade+'" }',original[files[1]].decode(),flags=re.M))
  print('BUILD',name,flush=True)
  with (r/('build-'+name+'.log')).open('w') as f:
   subprocess.run(['cargo','build','--release','-p','inferena-meganeura'],cwd=h,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=900)
  metadata=subprocess.check_output(['cargo','metadata','--no-deps','--format-version','1'],cwd=h,env=env)
  import json
  binary=Path(json.loads(metadata)['target_directory'])/'release/inferena-meganeura'
  shutil.copy2(binary,r/name)
  for p in files:shutil.copy2(p,r/(name+'-'+str(p.relative_to(h)).replace('/','-')))
  print('DONE',name,hashlib.sha256((r/name).read_bytes()).hexdigest(),flush=True)
finally:
 for p,data in original.items():p.write_bytes(data)
