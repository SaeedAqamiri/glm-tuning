import json, pathlib, time, urllib.request

out = pathlib.Path('/home/saeed/glm-tuning-20261011/quality-mtp-small')
out.mkdir(exist_ok=True)
prompts = [
    'What is 17 times 23? Answer with the number only.',
    'Write a Python function gcd(a, b) for positive integers using Euclid\'s algorithm. Include the function only.',
    'به فارسی در سه جمله توضیح بده چرا آسمان آبی است.',
    'A service receives 120 requests per second. Each worker handles 30 requests per second. To keep utilization at or below 60%, how many workers are needed? Give the integer and a brief calculation.',
]
for i, prompt in enumerate(prompts):
    body = {'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 512,
            'temperature': 0, 'seed': 42, 'cache_prompt': False,
            'chat_template_kwargs': {'reasoning_effort': 'low'}}
    req = urllib.request.Request('http://127.0.0.1:8016/v1/chat/completions',
        json.dumps(body).encode(), {'Content-Type': 'application/json'})
    t = time.monotonic()
    r = json.load(urllib.request.urlopen(req, timeout=900))
    (out / f'chat-{i}.json').write_text(json.dumps(r, ensure_ascii=False, indent=2))
    print(json.dumps({'i': i, 'wall_s': time.monotonic()-t,
        'choices': r['choices'], 'usage': r.get('usage'), 'timings': r.get('timings')},
        ensure_ascii=False), flush=True)
