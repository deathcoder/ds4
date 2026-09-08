from pathlib import Path
import json,collections,re
w=Path(__file__).resolve().parent;p=w if (w/'results').is_dir() else Path(json.loads((w/'paired-location.json').read_text())['work'])
h=[json.loads(x) for x in (p/'host-observations.jsonl').read_text().splitlines()];loads=[json.loads(x) for x in (p/'load-observations.jsonl').read_text().splitlines()]
def mib(s):
 m=re.search(r'used = ([0-9.]+)([KMG])',s);assert m,s
 return float(m[1])*{'K':1/1024,'M':1,'G':1024}[m[2]]
swaps=[mib(x['swap']) for x in h];cpu={}
for x in loads:
 for row in x['cpu_processes']:cpu[row['process']]=max(cpu.get(row['process'],0),row['cpu_percent'])
result={'thermal_swap_samples':len(h),'all_thermal_reads_passed':all(x['thermal_exit']==0 for x in h),'all_report_no_thermal_warnings':all('No thermal warning level has been recorded' in x['thermal'] for x in h),'swap_used_mib_min':min(swaps),'swap_used_mib_max':max(swaps),'load_samples':len(loads),'sampled_cpu_percent_max_by_process':dict(sorted(cpu.items(),key=lambda x:(-x[1],x[0]))),'scope':'Shared interactive host. Periodic thermal/swap samples cover the timed block; load samples also cover the separate profile. Post-pair/profile endpoint snapshots are retained in inputs/.'}
(w/'host-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
