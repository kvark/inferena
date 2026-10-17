import os
import os,subprocess,json,time
from pathlib import Path
r=Path(os.environ.get('INFERENA_ARTIFACT_DIR',str(Path(__file__).parent/'results')))
commands=''.join(json.dumps(x)+'\n' for x in [{'command':'samples','runs':1},{'command':'finish'}]*3)
for arm,temperature in [('base','cold'),('head','cold'),('head','warm'),('base','warm')]:
 name=f'cache-{arm}-{temperature}';out=r/name;out.mkdir(exist_ok=False)
 cache=r/('private-nvidia-cache-'+arm)
 if temperature=='cold':cache.mkdir(exist_ok=False)
 env={k:v for k,v in os.environ.items() if not k.startswith(('MEGANEURA_','INFERENA_','VK_','FRAMEWORK_REV','__GL_SHADER_DISK_CACHE'))}
 env.update(VK_DRIVER_FILES='/usr/share/vulkan/icd.d/nvidia_icd.json',MEGANEURA_DEVICE_ID='12036',MEGANEURA_GPU_TIMING='0',MEGANEURA_TUNE='1',INFERENA_TUNE_SECONDS='30',INFERENA_STRICT='1',INFERENA_REPLAY_INTERACTIVE='1',__GL_SHADER_DISK_CACHE='1',__GL_SHADER_DISK_CACHE_PATH=str(cache),__GL_SHADER_DISK_CACHE_SIZE=str(1<<30))
 print('START',name,flush=True);start=time.monotonic()
 with (out/'record.json').open('w') as stdout,(out/'stderr').open('w') as stderr:
  subprocess.run([str(r/arm),'ResNet-50'],cwd=os.environ.get('INFERENA_MODEL_CWD',str(Path(__file__).resolve().parents[2])),env=env,input=commands,text=True,stdout=stdout,stderr=stderr,check=True,timeout=300)
 record=json.loads((out/'record.json').read_text());duration=lambda x:x['secs']+x['nanos']*1e-9
 rows=[]
 for session in record['optimizer']['sessions']:
  q=session['search'];assert q['trials'][q['selected']]['outcome']['qualified']
  rows.append({'mode':session['mode'],'construction':sum(duration(t['construction_time']) for t in q['trials']),'initial_construction':duration(q['trials'][0]['construction_time']),'search':duration(q['elapsed']),'trials':len(q['trials'])})
 files=[p for p in cache.rglob('*') if p.is_file()]
 summary={'wall_seconds':time.monotonic()-start,'shader_cache_path':str(cache),'shader_cache_files':len(files),'shader_cache_bytes':sum(p.stat().st_size for p in files),'sessions':rows}
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print('DONE',name,json.dumps(summary),flush=True)
