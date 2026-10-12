import json, pathlib, urllib.request, time, os
cap = int(os.environ.get("SPEC_CAP", "1"))

out = pathlib.Path(f'/home/saeed/glm-tuning-20261011/real-{os.environ.get("BENCH_VARIANT","mtp-smallbatch48")}-cap{cap}-results')
out.mkdir(exist_ok=True)
source = pathlib.Path('/home/saeed/glm-tuning-20261011/cpu-bench-source-fixed.c').read_text()
start = source.index('// Threadpool def')
prompts = [
    'Explain how synchronization and memory ordering work in this C code. Identify two specific bottlenecks and explain possible improvements. Keep the response below 180 words.\n```c\n'+source[start:start+7000]+'\n```',
    'کد زیر را بررسی کن و به فارسی توضیح بده barrier چگونه کار می‌کند و کدام قسمت‌ها ممکن است در سیستم دو سوکتی کند شوند. پاسخ را حداکثر در ۱۵۰ کلمه بنویس.\n```c\n'+source[start:start+16000]+'\n```',
]
for i, prompt in enumerate(prompts):
    body = {'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 512,
            'temperature': 0, 'seed': 42, 'cache_prompt': False, 'speculative.n_max': cap,
            'chat_template_kwargs': {'reasoning_effort': 'low'}}
    req = urllib.request.Request('http://127.0.0.1:8017/v1/chat/completions',
        json.dumps(body).encode(), {'Content-Type': 'application/json'})
    t = time.monotonic()
    r = json.load(urllib.request.urlopen(req, timeout=900))
    (out / f'chat-{i}.json').write_text(json.dumps(r, ensure_ascii=False, indent=2))
    print(json.dumps({'i': i, 'draft_cap': cap, 'wall_s': time.monotonic()-t, 'timings': r.get('timings'),
        'choices': r['choices'], 'usage': r.get('usage')}, ensure_ascii=False), flush=True)
