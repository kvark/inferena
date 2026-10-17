import argparse,hashlib,json,os,signal,statistics,subprocess,sys,time,queue,threading
from pathlib import Path
repo = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(repo/'scripts'))
from collection import write_json
r=Path(os.environ.get('INFERENA_ARTIFACT_DIR',str(Path(__file__).parent/'results')));r.mkdir(parents=True,exist_ok=True)
p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--gpu',default='b570');p.add_argument('--model',default='StableDiffusion');p.add_argument('--profile',default='accelerated');p.add_argument('--left',default='base');p.add_argument('--right',default='head');p.add_argument('--left-plans');p.add_argument('--right-plans');p.add_argument('--rounds',type=int,default=20);p.add_argument('--runs',type=int,default=32);p.add_argument('--timestamps',action='store_true');p.add_argument('--cpu',type=int);p.add_argument('--timing-cpu',type=int);a=p.parse_args()
out=r/a.name;out.mkdir(exist_ok=False)
children={};handles=[];events=[];started=time.monotonic()
def launch(label,binary,plans):
 env={k:v for k,v in os.environ.items() if not k.startswith(('MEGANEURA_','INFERENA_','VK_','FRAMEWORK_REV'))}
 env.update(VK_DRIVER_FILES='/usr/share/vulkan/icd.d/'+('intel' if a.gpu=='b570' else 'nvidia')+'_icd.json',MEGANEURA_DEVICE_ID='57868' if a.gpu=='b570' else '12036',MEGANEURA_GPU_TIMING=str(int(a.timestamps)),INFERENA_STRICT=str(int(a.profile=='strict')),INFERENA_TUNE_SECONDS='30',INFERENA_REPLAY_INTERACTIVE='1',INFERENA_DUMP_PLANS_DIR=str(out/(label+'-plans')))
 if plans:env['INFERENA_REPLAY_DIR']=str(Path(plans).resolve());env.pop('INFERENA_DUMP_PLANS_DIR')
 stdout=(out/(label+'.stdout')).open('w');handles.append(stdout)
 command=(["taskset","-c",str(a.cpu)] if a.cpu is not None else [])+[str(r/binary),a.model]
 proc=subprocess.Popen(command,cwd=os.environ.get('INFERENA_MODEL_CWD',str(repo)),env=env,stdin=subprocess.PIPE,stdout=stdout,stderr=subprocess.PIPE,text=True,bufsize=1,start_new_session=True)
 log=(out/(label+'.log')).open('w');handles.append(log)
 q=queue.Queue()
 def reader():
  for line in proc.stderr:
   log.write(line);log.flush();q.put(line)
  q.put(None)
 threading.Thread(target=reader,daemon=True).start()
 children[label]=(proc,log,q)
 write_json(out/(label+'-protocol.json'),{'binary':str(r/binary),'sha256':hashlib.sha256((r/binary).read_bytes()).hexdigest(),'env':{k:v for k,v in env.items() if k.startswith(('INFERENA_','MEGANEURA_','VK_'))},'arguments':vars(a)})
 return event(label,'ready')
def event(label,expected):
 proc,log,q=children[label]
 while True:
  if time.monotonic()-started>1800:raise TimeoutError('campaign watchdog')
  try:line=q.get(timeout=10)
  except queue.Empty:continue
  if line is None:
   if expected in ('eof','next'):return None
   raise RuntimeError(f'{label} disconnected expecting {expected}, exit={proc.poll()}')
  if line.startswith('REPLAY '):
   data=json.loads(line[7:]);assert data['event']==expected or (expected=='next' and data['event']=='ready'),(expected,data)
   data.update(arm=label,time=time.time());events.append(data)
   with (out/'events.jsonl').open('a') as f:f.write(json.dumps(data)+'\n')
   return data

def send(label,command,**kw):
 proc,_,_=children[label];proc.stdin.write(json.dumps({'command':command}|kw)+'\n');proc.stdin.flush()

def summarize():
 rows=[]
 for phase in ['inference','latency','training']:
  blocks=[e for e in events if e.get('event')=='samples' and e['phase']==phase]
  if not blocks:continue
  values={v:[x for b in blocks if b['arm']==v for x in b['samples_ms']] for v in ['base','head']}
  ratios=[]
  for i in range(0,len(blocks),4):
   group=blocks[i:i+4]
   if len(group)!=4:continue
   med={v:statistics.median([statistics.median(b['samples_ms']) for b in group if b['arm']==v]) for v in ['base','head']}
   ratios.append(med['head']/med['base'])
  rows.append({'phase':phase,'samples':{k:len(v) for k,v in values.items()},'medians_ms':{k:statistics.median(v) for k,v in values.items()},'paired_delta_pct':100*(statistics.median(ratios)-1),'round_ratios':ratios,'encode_median_ms':{v:statistics.median([x for b in blocks if b['arm']==v for x in b['encode_ms']]) for v in ['base','head']},'wait_median_ms':{v:statistics.median([x for b in blocks if b['arm']==v for x in b['wait_ms']]) for v in ['base','head']}})
 write_json(out/'summary.json',rows)
 for x in rows:print(x['phase'],x['medians_ms'],'paired delta',round(x['paired_delta_pct'],2),'encode',x['encode_median_ms'],flush=True)

try:
 left=launch('base',a.left,a.left_plans);right=launch('head',a.right,a.right_plans)
 while True:
  phase=left['phase'];assert right['phase']==phase
  if a.timing_cpu is not None:
   for proc,_,_ in children.values():os.sched_setaffinity(proc.pid,{a.timing_cpu})
  print('PHASE',phase,flush=True)
  for v in ['base','head','head','base']:
   send(v,'warmup',seconds=5);event(v,'warm')
  for round_index in range(a.rounds):
   for v in (['base','head','head','base'] if round_index%2==0 else ['head','base','base','head']):
    send(v,'samples',runs=a.runs);event(v,'samples')
  summarize()
  if a.timing_cpu is not None:
   for proc,_,_ in children.values():os.sched_setaffinity(proc.pid,os.sched_getaffinity(0))
  send('base','finish')
  left=event('base','next')
  send('head','finish')
  right=event('head','next')
  if left is None or right is None:
   assert left is None and right is None,'phase count mismatch'
   break
 for label,(proc,log,q) in children.items():
  assert proc.wait(timeout=30)==0
  proc.stdin.close();proc.stderr.close()
 for h in handles:h.close()
 for label in children:write_json(out/(label+'.json'),json.loads((out/(label+'.stdout')).read_text()))
 summarize();write_json(out/'completion.json',{'complete':True,'elapsed':time.monotonic()-started,'args':vars(a)})
except BaseException:
 for proc,_,_ in children.values():
  if proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM)
 raise
finally:
 for h in handles:
  if not h.closed:h.close()
