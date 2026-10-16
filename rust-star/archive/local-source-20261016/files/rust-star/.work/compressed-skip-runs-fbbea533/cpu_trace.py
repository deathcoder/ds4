"""Exhaustive/seeded integer traversal comparison, including padded boundaries."""
from pathlib import Path
import hashlib,itertools,json,random
w=Path(__file__).resolve().parent

def trace(blocks,padded,masked,skip):
 n=len(blocks);nfull=n-int(padded);ne11=64*nfull+(17 if padded else 0)
 ptr=0;pad=False;events=[];i=0;reads=0;bulk=0
 def block(j):
  nonlocal reads
  assert 0<=j<n;reads+=1;return blocks[j]
 while True:
  if skip and masked:
   first=i
   while i+4<=nfull and (block(i)|block(i+1)|block(i+2)|block(i+3))==0:i+=4;bulk+=1
   while i<nfull and block(i)==0:i+=1
   ptr+=(i-first)*32
  if i*64>=ne11:break
  if padded and i*64+64>ne11:pad=True;ptr=0
  kind=block(i) if masked else 1
  if masked and kind==0:ptr+=32;i+=1;continue
  events.append((i,kind,pad,0 if pad else i*64,ptr))
  if masked:ptr+=32
  i+=1
 return (events,ptr,pad),reads,bulk

def evaluate(blocks,padded,masked):
 global cases,total_bulk,total_reads_original,total_reads_candidate,max_blocks
 original,ra,_=trace(blocks,padded,masked,False);candidate,rb,bulk=trace(blocks,padded,masked,True)
 assert original==candidate,(blocks,padded,masked,original,candidate)
 cases+=1;total_bulk+=bulk;total_reads_original+=ra;total_reads_candidate+=rb;max_blocks=max(max_blocks,len(blocks))
 h.update(json.dumps([list(blocks),padded,masked,original],separators=(',',':')).encode())
cases=total_bulk=total_reads_original=total_reads_candidate=max_blocks=0;h=hashlib.sha256()
for length in range(10):
 for blocks in itertools.product(range(3),repeat=length):
  for padded in ([False,True] if length else [False]):
   for masked in [False,True]:evaluate(blocks,padded,masked)
rng=random.Random(0x51c1f00d)
for length in [15,16,17,31,32,33,63,64,65,66,67,68,69,70,71,127,128,129,255,256,257,4096]:
 for rep in range(24):
  blocks=[0 if rng.randrange(100)<85 else rng.randrange(1,3) for _ in range(length)]
  for padded in [False,True]:evaluate(blocks,padded,True)
result=dict(passed=True,cases=cases,maximum_blocks=max_blocks,exhaustive_lengths=list(range(10)),block_values=[0,1,2],masked_and_unmasked_branches=True,padded_and_unpadded=True,seed=0x51c1f00d,trace_sha256=h.hexdigest(),four_zero_block_jumps=total_bulk,metadata_reads_original=total_reads_original,metadata_reads_candidate=total_reads_candidate,scope='Integer traversal/read bounds only; exact active-block order, mask-pointer and padded-KV state match. Metadata reads may increase near active blocks.')
if (w/'cpu-trace.json').exists():assert json.loads((w/'cpu-trace.json').read_text())==result
else:(w/'cpu-trace.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
