import os
import json, random, statistics, hashlib
from pathlib import Path
r=Path(os.environ.get('INFERENA_ARTIFACT_DIR',str(Path(__file__).parent/'results')))
rows=[]
def seconds(x):return x['secs']+x['nanos']*1e-9
for path in sorted(r.glob('*/completion.json')):
 d=path.parent
 entry={'case':d.name,'args':json.loads(path.read_text())['args'],'phases':json.loads((d/'summary.json').read_text()),'sessions':{}}
 for phase in entry['phases']:
  ratios=phase['round_ratios']; rng=random.Random(0)
  # Conditional within-process intervals, not between-process reproducibility.
  boot=sorted(statistics.median(rng.choices(ratios,k=len(ratios))) for _ in range(2000))
  phase['round_bootstrap_95_pct']=[100*(boot[50]-1),100*(boot[1949]-1)]
 for arm in ['base','head']:
  data=json.loads((d/(arm+'.json')).read_text());sessions=[]
  for session in data['optimizer']['sessions']:
   search=session.get('search')
   if not search:
    sessions.append({'replay':session.get('replay')});continue
   trial=search['trials'][search['selected']]
   sessions.append({'selected':search['selected'],'description':trial['description'],'trials':len(search['trials']),'search_elapsed':seconds(search['elapsed']),'construction':sum(seconds(x['construction_time']) for x in search['trials']),'input_refresh':sum(seconds(x['input_refresh_time']) for x in search['trials']),'qualified':trial['outcome']['qualified'],'qualification_failures':sum(x['outcome']['decision']=='InvalidOutput' for x in search['trials']),'workgroup_dispatches':sum('workgroup-' in k for k in session['kernel_pipelines']),'search_input_policy':session['search_input_policy']})
  entry['sessions'][arm]=sessions
 rows.append(entry)
 print(d.name,' | '.join(f"{x['phase']}: {x['paired_delta_pct']:+.2f}%" for x in entry['phases']))
(r/'analysis.json').write_text(json.dumps(rows,indent=2)+'\n')
