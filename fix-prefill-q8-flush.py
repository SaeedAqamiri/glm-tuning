from pathlib import Path
import re
p = Path('/home/saeed/glm-tuning-20261011/prefill-400/ggml/src/ggml-cpu/amx/mmq.cpp')
s = p.read_text()
a = s.index('void tinygemm_kernel_amx(int M')
b = s.index('void tinygemm_kernel_amx(int M', a + 10)
part = s[a:b]
part, count = re.subn(r'acc_C<TA, TB, (true|false)>::apply\(([^;]+)\);', r'acc_amx<TA, TB, \1>(\2, signed_q8);', part)
assert count > 0, 'No constant-bool accumulation calls found; inspect manually'
p.with_suffix('.before-flush-fix.cpp').write_text(s)
p.write_text(s[:a] + part + s[b:])
print('Updated constant-bool flush calls:', count)
