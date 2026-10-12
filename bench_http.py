import argparse,json,time,urllib.request,pathlib
p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8012);p.add_argument('--lengths',default='512,2048');p.add_argument('--gen',type=int,default=128);p.add_argument('--out',required=True);p.add_argument('--chat',action='store_true');a=p.parse_args()
base=f'http://127.0.0.1:{a.port}'
def post(path,data):
 r=urllib.request.Request(base+path,json.dumps(data).encode(),{'Content-Type':'application/json'})
 return json.load(urllib.request.urlopen(r,timeout=3600))
out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
if a.chat:
 prompts=['What is 17 times 23? Answer with the number only.','Write a Python function that returns the greatest common divisor of two positive integers.','به فارسی در سه جمله توضیح بده چرا آسمان آبی است.']
 for i,prompt in enumerate(prompts):
  t=time.monotonic();r=post('/v1/chat/completions',{'messages':[{'role':'user','content':prompt}],'max_tokens':a.gen,'temperature':0,'seed':42,'cache_prompt':False,'chat_template_kwargs':{'enable_thinking':False}})
  (out/f'chat-{i}.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));print(json.dumps({'test':i,'wall_s':time.monotonic()-t,'result':r},ensure_ascii=False),flush=True)
else:
 text='A database stores records in tables. Each record has a unique identifier. Indexes help find records efficiently, but require additional storage and maintenance when data changes. Transactions group operations into a single unit that either completes or is rolled back. Engineers measure latency and throughput separately when evaluating a service. A benchmark must preserve correctness while measuring performance under clearly stated conditions.\n'
 toks=post('/tokenize',{'content':text*220,'add_special':True})['tokens']
 for n in map(int,a.lengths.split(',')):
  assert len(toks)>=n
  t=time.monotonic();r=post('/completion',{'prompt':toks[:n],'n_predict':a.gen,'temperature':0,'seed':42,'cache_prompt':False,'ignore_eos':True})
  (out/f'pp{n}-tg{a.gen}.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));print(json.dumps({'pp':n,'wall_s':time.monotonic()-t,'timings':r.get('timings'),'content':r.get('content','')[:200]},ensure_ascii=False),flush=True)
