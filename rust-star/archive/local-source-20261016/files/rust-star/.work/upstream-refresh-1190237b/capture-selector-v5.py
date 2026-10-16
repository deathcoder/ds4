from pathlib import Path
import json,hashlib,subprocess,os,struct,datetime
w=Path(__file__).resolve().parent;root=w.parents[2];src=w.with_name('oracle-v5-source');exe=src/'ds4-bench';model='/Users/deathcodevision/dev/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf';old=root/'rust-star/fixtures/prefill-layer23-continuation-second-tile-4128-v1';rows=[]
for rep in ['a','b']:
 d=w/f'selector-v5-{rep}';d.mkdir(exist_ok=True);env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','RUST_STAR_','METAL_','MTL_'))};env.update(DS4_METAL_GRAPH_DUMP_PREFIX=str(d/'oracle'),DS4_METAL_GRAPH_DUMP_POS='4096',DS4_METAL_GRAPH_DUMP_NAME='indexer_scores,indexer_topk',DS4_METAL_GRAPH_DUMP_LAYER='2')
 command=[str(exe),'--metal','-m',model,'--prompt-file',str(root/'speed-bench/promessi_sposi.txt'),'--ctx-start','8192','--ctx-max','8192','--gen-tokens','0','--prefill-chunk','4096','--csv',str(d/'benchmark.csv')]
 print('START selector',rep,flush=True)
 if not (d/'oracle_indexer_topk-2_pos4096.i32').exists():
  with (d/'console.log').open('w') as f:subprocess.run(command,cwd=src,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 scores=(d/'oracle_indexer_scores-2_pos4096.bin').read_bytes();indices=(d/'oracle_indexer_topk-2_pos4096.i32').read_bytes();assert len(scores)==4096*2048*4 and len(indices)==4096*512*4
 st=scores[32*2048*4:64*2048*4];it=indices[32*512*4:64*512*4];assert st==(old/'layer2-indexer-scores-tile2-bits.i32le.bin').read_bytes();assert struct.unpack('<16384I',it)[1930]==643
 (d/'scores-tile.f32le.bin').write_bytes(st);(d/'topk-tile.i32le.bin').write_bytes(it);r={'scores_sha256':hashlib.sha256(st).hexdigest(),'topk_sha256':hashlib.sha256(it).hexdigest(),'full_scores_sha256':hashlib.sha256(scores).hexdigest(),'full_topk_sha256':hashlib.sha256(indices).hexdigest()};rows.append(r);print('PASS selector',rep,r,flush=True)
assert rows[0]==rows[1]
(w/'selector-v5-acceptance.json').write_text(json.dumps({'passed':True,'source_commit':'ccbd16d36895ef9fe8edf7f793014988ff1ab7f4','executable_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'fresh_repetitions':2,'production_prefill_tokens':8192,'prefill_chunk':4096,'layer':2,'position_range':[4128,4159],'score_inputs_match_historical_fixture':True,'rows':rows,'captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
