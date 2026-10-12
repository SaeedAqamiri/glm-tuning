const $ = (id) => document.getElementById(id);
const state = { history: JSON.parse(localStorage.getItem('glm-history') || '[]'), controller: null, run: 0 };
const defaults = { temperature: 0, maxTokens: 512, seed: 42, reasoning: true, systemPrompt: '' };

function esc(text='') { return text.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c])); }
function formatContent(text='') {
  const parts = text.split(/```([\s\S]*?)```/g);
  return parts.map((part, i) => i % 2 ? `<pre><code>${esc(part.trim())}</code></pre>` : esc(part)).join('');
}
function fmt(n, digits=1) { return Number.isFinite(+n) ? (+n).toFixed(digits) : '—'; }
function seconds(ms) { return Number.isFinite(+ms) ? `${(+ms/1000).toFixed(2)}s` : '—'; }
function estimateTokens(text) { return Math.max(0, Math.round((text || '').trim().length / 3.2)); }
function toast(message, error=false) { const el=$('toast'); el.textContent=message; el.className=`toast show${error?' error':''}`; clearTimeout(toast.timer); toast.timer=setTimeout(()=>el.className='toast',3000); }

async function checkHealth() {
  const el=$('connection');
  try { const res=await fetch('/api/health'); if(!res.ok) throw Error(); el.className='connection online'; el.querySelector('span').textContent='سرور تستی آنلاین'; }
  catch { el.className='connection offline'; el.querySelector('span').textContent='سرور در دسترس نیست'; }
}

function addMessage(role, text, loading=false) {
  $('emptyState')?.classList.add('hidden');
  const node=document.createElement('article'); node.className=`message ${role}`;
  node.innerHTML=`<div class="avatar">${role==='user'?'شما':'GLM'}</div><div class="bubble"><div class="bubble-head"><b>${role==='user'?'پرامپت':'پاسخ مدل'}</b><span>${new Date().toLocaleTimeString('fa-IR',{hour:'2-digit',minute:'2-digit'})}</span></div><div class="bubble-content">${loading?'<span class="loader"><i></i><i></i><i></i></span>':formatContent(text)}</div></div>`;
  $('messages').appendChild(node); $('messages').scrollTop=$('messages').scrollHeight; return node;
}

function readSettings() { return { temperature:+$('temperature').value, maxTokens:+$('maxTokens').value, seed:+$('seed').value, reasoning:$('reasoning').checked, systemPrompt:$('systemPrompt').value.trim() }; }
function saveSettings() { localStorage.setItem('glm-settings',JSON.stringify(readSettings())); }
function applySettings(settings={}) { const s={...defaults,...settings}; for(const key of ['temperature','maxTokens','seed','systemPrompt']) $(key).value=s[key]; $('reasoning').checked=s.reasoning; $('temperatureValue').textContent=(+s.temperature).toFixed(1); $('maxTokensValue').textContent=s.maxTokens; }

function updateMetrics(run) {
  const t=run?.timings||{}; const usage=run?.usage||{};
  $('prefillRate').textContent=fmt(t.prompt_per_second); $('decodeRate').textContent=fmt(t.predicted_per_second);
  $('promptTokens').textContent=t.prompt_n ?? usage.prompt_tokens ?? '—'; $('outputTokens').textContent=t.predicted_n ?? usage.completion_tokens ?? '—';
  $('prefillTime').textContent=seconds(t.prompt_ms); $('decodeTime').textContent=seconds(t.predicted_ms); $('totalTime').textContent=seconds(run?.client_ms);
  $('runId').textContent=run ? `#${String(run.id).padStart(3,'0')}` : '—'; $('rawJson').textContent=run?.raw ? JSON.stringify(run.raw,null,2) : '{}';
}

function renderHistory() {
  const list=$('historyList'); const items=state.history.slice(0,12);
  list.innerHTML=items.length?items.map(r=>`<div class="history-item"><p title="${esc(r.prompt)}">${esc(r.prompt)}</p><b>${fmt(r.timings?.prompt_per_second)}</b><em>${fmt(r.timings?.predicted_per_second)}</em></div>`).join(''):'<p class="muted">هنوز تستی ثبت نشده است.</p>';
  const points=state.history.slice(0,16).reverse(); const svg=$('chart');
  if(points.length<2){svg.innerHTML='<text x="220" y="54" text-anchor="middle" fill="#60758b" font-size="11">برای نمودار حداقل دو تست اجرا کن</text>';return;}
  const all=points.flatMap(r=>[r.timings?.prompt_per_second||0,r.timings?.predicted_per_second||0]); const max=Math.max(...all,1); const xy=(key)=>points.map((r,i)=>`${i*440/(points.length-1)},${96-(r.timings?.[key]||0)*82/max}`).join(' ');
  svg.innerHTML=`<defs><linearGradient id="fill" x1="0" y1="0" x2="0" y2="1"><stop stop-color="#6ee7c2" stop-opacity=".22"/><stop offset="1" stop-color="#6ee7c2" stop-opacity="0"/></linearGradient></defs><polyline points="0,104 ${xy('prompt_per_second')} 440,104" fill="url(#fill)" stroke="none"/><polyline points="${xy('prompt_per_second')}" fill="none" stroke="#6ee7c2" stroke-width="2" vector-effect="non-scaling-stroke"/><polyline points="${xy('predicted_per_second')}" fill="none" stroke="#54c7ec" stroke-width="1.5" stroke-dasharray="4 4" vector-effect="non-scaling-stroke"/>`;
}

async function runPrompt(event) {
  event?.preventDefault(); const prompt=$('prompt').value.trim(); if(!prompt||state.controller)return;
  const settings=readSettings(); saveSettings(); addMessage('user',prompt); const pending=addMessage('assistant','',true);
  state.controller=new AbortController(); $('sendBtn').disabled=true; $('stopBtn').classList.remove('hidden'); const started=performance.now();
  const messages=[]; if(settings.systemPrompt) messages.push({role:'system',content:settings.systemPrompt}); messages.push({role:'user',content:prompt});
  const body={messages,max_tokens:settings.maxTokens,temperature:settings.temperature,seed:settings.seed,cache_prompt:false,stream:false};
  if(settings.reasoning) body.chat_template_kwargs={reasoning_effort:'low'};
  try {
    const res=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),signal:state.controller.signal});
    const raw=await res.json(); if(!res.ok) throw Error(raw?.error?.message||`HTTP ${res.status}`);
    const content=raw.choices?.[0]?.message?.content ?? raw.choices?.[0]?.text ?? '';
    pending.querySelector('.bubble-content').innerHTML=formatContent(content)||'<span class="muted">پاسخ متنی خالی بود.</span>';
    const run={id:Date.now()%1000,prompt,content,settings,timings:raw.timings||{},usage:raw.usage||{},client_ms:performance.now()-started,raw,date:new Date().toISOString()};
    state.history.unshift(run); state.history=state.history.slice(0,50); localStorage.setItem('glm-history',JSON.stringify(state.history)); updateMetrics(run); renderHistory(); $('prompt').value=''; updateEstimate(); toast('تست با موفقیت ثبت شد');
  } catch(err) {
    pending.querySelector('.bubble-content').innerHTML=`<span style="color:var(--danger)">${err.name==='AbortError'?'درخواست متوقف شد.':esc(err.message)}</span>`; if(err.name!=='AbortError')toast(err.message,true);
  } finally { state.controller=null; $('sendBtn').disabled=false; $('stopBtn').classList.add('hidden'); }
}

function updateEstimate(){ $('tokenEstimate').textContent=`حدود ${estimateTokens($('prompt').value).toLocaleString('fa-IR')} توکن ورودی`; $('prompt').style.height='auto'; $('prompt').style.height=Math.min($('prompt').scrollHeight,220)+'px'; }
function exportHistory(){ const blob=new Blob([JSON.stringify(state.history,null,2)],{type:'application/json'}); const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=`glm-tests-${new Date().toISOString().slice(0,10)}.json`;a.click();URL.revokeObjectURL(a.href); }

$('chatForm').addEventListener('submit',runPrompt); $('stopBtn').onclick=()=>state.controller?.abort(); $('prompt').addEventListener('input',updateEstimate); $('prompt').addEventListener('keydown',e=>{if(e.ctrlKey&&e.key==='Enter')runPrompt(e)});
$('clearBtn').onclick=()=>{ $('messages').innerHTML='<div class="empty-state" id="emptyState"><div class="orb"><span></span></div><h2>گفتگو پاک شد</h2><p>برای اجرای تست بعدی، یک پرامپت جدید بنویس.</p></div>'; };
$('exportBtn').onclick=exportHistory; $('resetSettings').onclick=()=>{applySettings(defaults);saveSettings();toast('تنظیمات بازنشانی شد')};
for(const id of ['temperature','maxTokens']) $(id).addEventListener('input',()=>{ $(id+'Value').textContent=id==='temperature'?(+$(id).value).toFixed(1):$(id).value; saveSettings(); });
for(const id of ['seed','reasoning','systemPrompt']) $(id).addEventListener('change',saveSettings);
document.querySelectorAll('[data-prompt]').forEach(b=>b.onclick=()=>{$('prompt').value=b.dataset.prompt;updateEstimate();$('prompt').focus()});
applySettings(JSON.parse(localStorage.getItem('glm-settings')||'{}')); updateEstimate(); renderHistory(); if(state.history[0])updateMetrics(state.history[0]); checkHealth(); setInterval(checkHealth,15000);
