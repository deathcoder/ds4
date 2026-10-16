import struct,json
from pathlib import Path
r=Path.cwd();w=r/'rust-star/.work/remaining-layers-v30'
f=open('/Users/deathcodevision/dev/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf','rb')
def u(fmt):return struct.unpack('<'+fmt,f.read(struct.calcsize('<'+fmt)))[0]
def st():return f.read(u('Q')).decode()
def value(t):
 if t==8:return st()
 if t==9:
  typ,n=u('I'),u('Q')
  for _ in range(n):value(typ)
  return None
 return u({0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}[t])
assert f.read(4)==b'GGUF';assert u('I')==3;nt,nm=u('Q'),u('Q');align=32
for _ in range(nm):
 name=st();v=value(u('I'))
 if name=='general.alignment':align=v
found=None
for _ in range(nt):
 name=st();dims=[u('Q') for _ in range(u('I'))];typ=u('I');off=u('Q')
 if name=='blk.26.exp_probs_b.bias':found=(off,typ,dims)
 if 'blk.26.' in name and ('bias' in name or 'probs' in name):print(name,off,typ,dims)
assert found, 'find bias tensor name';base=(f.tell()+align-1)//align*align;f.seek(base+found[0]);bias=struct.unpack('<256f',f.read(1024))
(w/'layer26-router-bias.json').write_text(json.dumps(bias))
