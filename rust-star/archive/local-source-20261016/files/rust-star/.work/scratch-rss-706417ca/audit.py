from pathlib import Path
import json,hashlib,re,statistics
w=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text());plan=read(w/'plan.json');med=statistics.median
progress=read(w/'runs/progress.json');assert len(progress)==25
expected=[f'{i+1}-{m}' for i,order in enumerate(plan['orders']) for m in order];assert [x['name'] for x in progress]==expected
labels={x['label'] for x in plan['inventory']};total=sum(x['bytes'] for x in plan['inventory']);raw={};checked=0
for entry in progress:
 d=w/'runs'/entry['name'];status=read(d/'status.json');assert status['exit_code']==0
 lines=(d/'run.log').read_text().splitlines();headers=[json.loads(x[7:]) for x in lines if x.startswith('HEADER ')];rows=[json.loads(x[9:]) for x in lines if x.startswith('SNAPSHOT ')];ids=[json.loads(x[11:]) for x in lines if x.startswith('IDENTITIES ')];results=[json.loads(x[7:]) for x in lines if x.startswith('RESULT ')];assert len(headers)==1 and headers[0]['mode']==entry['mode'] and headers[0]['device']=='Apple M1 Ultra' and headers[0]['args_bytes']==24
 assert len(rows)==22 and len(ids)==4 and len(results)==1
 mode=entry['mode'];reuse=plan['modes'][mode]['reuse'];clear=0 if mode in 'AC' else (total-536870912 if mode=='E' else total)
 assert results[0]==dict(passed=True,mode=mode,tiles=4,buffers=105,logical_bytes=total,checked_bytes=4*total,copy_destination_bytes=536870912,pool_identities_stable=reuse)
 checked+=results[0]['checked_bytes'];assert rows[0]['phase']=='baseline' and rows[-1]['phase']=='released'
 for tile in range(4):
  ts=rows[1+5*tile:6+5*tile];assert [x['phase'] for x in ts]==['entry','allocated','initialized','gpu_completed','after_tile']
  assert all(x['tile']==tile and x['vm_result']==0 and x['rss']>0 and x['footprint']>0 for x in ts)
  assert ts[1]['live_count']>=105 and ts[1]['live_bytes']>=total
  if reuse:assert ts[1]['live_count']==105 and ts[1]['live_bytes']==total
  assert ts[2]['cpu_clear_bytes']==ts[3]['cpu_clear_bytes']==clear
  assert ids[tile]['tile']==tile and set(ids[tile]['ids'])==labels and len(set(ids[tile]['ids'].values()))==105
  if reuse:assert ids[tile]['ids']==ids[0]['ids']
 text=(d/'run.log').read_text();rss=int(re.search(r'(\d+)\s+maximum resident set size',text)[1]);foot=int(re.search(r'(\d+)\s+peak memory footprint',text)[1]);assert rss>0 and foot>0
 raw[entry['name']]=dict(mode=mode,block=entry['block'],peak_rss=rss,peak_footprint=foot,baseline_rss=rows[0]['rss'],peak_initialized_rss=max(x['rss'] for x in rows if x['phase']=='initialized'),peak_gpu_rss=max(x['rss'] for x in rows if x['phase']=='gpu_completed'),peak_device_allocated=max(x['device_allocated'] for x in rows),max_live_bytes=max(x['live_bytes'] for x in rows),fresh_previous_at_entry=[x['prior_generation_bytes'] for x in rows if x['phase']=='entry'],final_live_bytes=rows[-1]['live_bytes'],final_rss=rows[-1]['rss'])
pairs=[]
for block in range(1,6):
 roles={m:raw[f'{block}-{m}'] for m in 'ABCDE'}
 pairs.append(dict(block=block,order=plan['orders'][block-1],fresh_clear_effect=roles['B']['peak_rss']-roles['A']['peak_rss'],pool_clear_effect=roles['D']['peak_rss']-roles['C']['peak_rss'],selected_saving=roles['D']['peak_rss']-roles['E']['peak_rss'],selected_initialized_saving=roles['D']['peak_initialized_rss']-roles['E']['peak_initialized_rss'],selected_footprint_saving=roles['D']['peak_footprint']-roles['E']['peak_footprint'],no_clear_reuse_effect=roles['C']['peak_rss']-roles['A']['peak_rss']))
criteria=plan['selection'];gates=dict(all_source_and_gpu_proofs_pass=read(w/'source-proof.json')['passed'] and read(w/'proof-status.json')['exit_code']==0,all25_runs_complete_and_exact=True,pool_clear_positive_every_block=all(x['pool_clear_effect']>0 for x in pairs),pool_clear_median=med(x['pool_clear_effect'] for x in pairs)>=criteria['pool_clear_minus_no_clear_median_bytes_at_least'],selected_saving_every_block=all(x['selected_saving']>=criteria['selected_rss_saving_every_block_at_least'] for x in pairs),selected_saving_median=med(x['selected_saving'] for x in pairs)>=criteria['selected_rss_saving_median_at_least'])
summary={m:{k:dict(median=med(v:=[raw[f'{i}-{m}'][k] for i in range(1,6)]),minimum=min(v),maximum=max(v)) for k in ['peak_rss','peak_footprint','baseline_rss','peak_initialized_rss','peak_gpu_rss','peak_device_allocated','max_live_bytes']} for m in 'ABCDE'}
result=dict(passed=True,processes=25,tiles=100,checked_bytes=checked,gates=gates,select_for_future_candidate=all(gates.values()),paired_medians={k:med(x[k] for x in pairs) for k in ['fresh_clear_effect','pool_clear_effect','selected_saving','selected_initialized_saving','selected_footprint_saving','no_clear_reuse_effect']},pairs=pairs,summary=summary,raw=raw,engine_acceptance=False,model_runs=0)
p=w/'audit.json'
if p.exists():assert read(p)==result
else:p.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['raw','summary']},indent=2))
