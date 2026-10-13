import json, os, pathlib, time, urllib.request

base = pathlib.Path('/home/saeed/glm-tuning-20261011')
source = (base / 'cpu-bench-source-fixed.c').read_text()
start = source.index('// Threadpool def')
prompts = [
    'Explain how synchronization and memory ordering work in this C code. Identify two specific bottlenecks and explain possible improvements. Keep the response below 180 words.\n```c\n' + source[start:start+7000] + '\n```',
    'کد زیر را بررسی کن و به فارسی توضیح بده barrier چگونه کار می‌کند و کدام قسمت‌ها ممکن است در سیستم دو سوکتی کند شوند. پاسخ را حداکثر در ۱۵۰ کلمه بنویس.\n```c\n' + source[start:start+16000] + '\n```',
]
port = int(os.environ.get('PORT', '8019'))
max_tokens = int(os.environ.get('MAX_TOKENS', '1'))
variant = os.environ.get('BENCH_VARIANT', 'prefill-v2')
dest = base / f'{variant}.jsonl'

with dest.open('w') as out:
    for i, prompt in enumerate(prompts):
        body = {
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': max_tokens,
            'temperature': 0,
            'seed': 42,
            'cache_prompt': False,
            'speculative.n_max': 1,
            'chat_template_kwargs': {'reasoning_effort': 'low'},
        }
        req = urllib.request.Request(
            f'http://127.0.0.1:{port}/v1/chat/completions',
            json.dumps(body).encode(),
            {'Content-Type': 'application/json'},
        )
        begun = time.monotonic()
        response = json.load(urllib.request.urlopen(req, timeout=1200))
        row = {
            'i': i,
            'wall_s': time.monotonic() - begun,
            'timings': response.get('timings'),
            'usage': response.get('usage'),
            'choices': response.get('choices'),
        }
        line = json.dumps(row, ensure_ascii=False)
        print(line, flush=True)
        out.write(line + '\n')
