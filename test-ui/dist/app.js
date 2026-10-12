const $ = (id) => document.getElementById(id);
const state = { history: JSON.parse(localStorage.getItem('glm-history') || '[]'), controller: null, conversation: [] };
const defaults = { temperature: 0, maxTokens: 512, seed: 42, reasoning: true, systemPrompt: '', memory: true };

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

async function loadModelTag() {
  try {
    const res=await fetch('/api/models'); const data=await res.json();
    const name=(data.models?.[0]?.name || '').split('/').pop().replace(/\.gguf$/i,'').replace(/-00001-of-\d+$/,'');
    if(name){ $('modelChip').textContent=name; $('modelChip').hidden=false; }
  } catch {}
}

function addMessage(role, text, loading=false) {
  $('emptyState')?.classList.add('hidden');
  const node=document.createElement('article'); node.className=`message ${role}`;
  node.innerHTML=`<div class="avatar">${role==='user'?'شما':'GLM'}</div><div class="bubble"><div class="bubble-head"><b>${role==='user'?'پرامپت':'پاسخ مدل'}</b><span>${new Date().toLocaleTimeString('fa-IR',{hour:'2-digit',minute:'2-digit'})}</span></div><div class="bubble-content">${loading?'<span class="loader"><i></i><i></i><i></i></span>':formatContent(text)}</div></div>`;
  $('messages').appendChild(node); scrollMessages(); return node;
}

function scrollMessages(){ const m=$('messages'); m.scrollTop=m.scrollHeight; }

function createAssistantBubble() {
  const node=document.createElement('article'); node.className='message assistant streaming';
  node.innerHTML=`<div class="avatar">GLM</div><div class="bubble">
    <div class="bubble-head"><b>پاسخ مدل</b><span>${new Date().toLocaleTimeString('fa-IR',{hour:'2-digit',minute:'2-digit'})}</span></div>
    <details class="reasoning hidden"><summary>فرایند تفکر مدل <span class="reasoning-len"></span></summary><div class="reasoning-body"></div></details>
    <div class="bubble-content"><span class="loader"><i></i><i></i><i></i></span></div>
    <div class="bubble-foot"><span class="live-badge hidden"><i></i><span class="live-text"></span></span></div>
  </div>`;
  $('messages').appendChild(node); scrollMessages();
  const reasoningEl=node.querySelector('.reasoning'), reasoningBody=node.querySelector('.reasoning-body'),
        reasoningLen=node.querySelector('.reasoning-len'), contentEl=node.querySelector('.bubble-content'),
        liveEl=node.querySelector('.live-badge'), liveText=node.querySelector('.live-text');
  const view={ reasoning:'', content:'', ntok:0, t0:0, raf:0,
    paint(){
      if(view.raf) return;
      view.raf=requestAnimationFrame(()=>{ view.raf=0;
        if(view.reasoning){ reasoningEl.classList.remove('hidden'); reasoningLen.textContent=`· ${view.reasoning.length.toLocaleString('fa-IR')} نویسه`; reasoningBody.innerHTML=formatContent(view.reasoning); }
        contentEl.innerHTML=view.content?formatContent(view.content):'<span class="loader"><i></i><i></i><i></i></span>';
        if(view.t0&&view.ntok){ liveEl.classList.remove('hidden'); const rate=view.ntok/((performance.now()-view.t0)/1000); liveText.textContent=`${rate.toFixed(1)} tok/s · ${view.ntok} توکن`; }
        scrollMessages();
      });
    },
    finalize(aborted=false){
      node.classList.remove('streaming');
      liveEl.classList.add('hidden');
      if(aborted&&view.content) contentEl.innerHTML=formatContent(view.content)+'<p class="muted">— توقف توسط کاربر —</p>';
    }};
  return view;
}

function readSettings() { return { temperature:+$('temperature').value, maxTokens:+$('maxTokens').value, seed:+$('seed').value, reasoning:$('reasoning').checked, systemPrompt:$('systemPrompt').value.trim(), memory:$('memory').checked }; }
function saveSettings() { localStorage.setItem('glm-settings',JSON.stringify(readSettings())); }
function applySettings(settings={}) { const s={...defaults,...settings}; for(const key of ['temperature','maxTokens','seed','systemPrompt']) $(key).value=s[key]; $('reasoning').checked=s.reasoning; $('memory').checked=s.memory; $('temperatureValue').textContent=(+s.temperature).toFixed(1); $('maxTokensValue').textContent=s.maxTokens; }

function updateMetrics(run) {
  const t=run?.timings||{}; const usage=run?.usage||{};
  $('prefillRate').textContent=fmt(t.prompt_per_second); $('decodeRate').textContent=fmt(t.predicted_per_second);
  $('promptTokens').textContent=t.prompt_n ?? usage.prompt_tokens ?? '—'; $('outputTokens').textContent=t.predicted_n ?? usage.completion_tokens ?? '—';
  $('prefillTime').textContent=seconds(t.prompt_ms); $('decodeTime').textContent=seconds(t.predicted_ms); $('totalTime').textContent=seconds(run?.client_ms);
  const draft=t.draft_n||0;
  $('mtpRate').textContent=draft?`${Math.round(100*(t.draft_n_accepted||0)/draft)}٪`:'—';
  $('cacheTokens').textContent=(t.cache_n!==undefined&&t.cache_n!==null)?t.cache_n:'—';
  $('runId').textContent=run ? `#${String(run.id).padStart(3,'0')}` : '—';
  $('rawJson').textContent=JSON.stringify({timings:t,usage},null,2);
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
  const settings=readSettings(); saveSettings(); addMessage('user',prompt);
  const view=createAssistantBubble();
  state.controller=new AbortController(); $('sendBtn').disabled=true; $('stopBtn').classList.remove('hidden'); const started=performance.now();
  const messages=[]; if(settings.systemPrompt) messages.push({role:'system',content:settings.systemPrompt});
  if(settings.memory) for(const m of state.conversation) messages.push(m);
  messages.push({role:'user',content:prompt});
  const body={messages,max_tokens:settings.maxTokens,temperature:settings.temperature,seed:settings.seed,cache_prompt:settings.memory,stream:true};
  if(settings.reasoning) body.chat_template_kwargs={reasoning_effort:'low'};
  let timings=null, usage=null, aborted=false, failed=null;
  try {
    const res=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),signal:state.controller.signal});
    if(!res.ok){ const raw=await res.json().catch(()=>({})); throw Error(raw?.error?.message||`HTTP ${res.status}`); }
    const reader=res.body.getReader(); const decoder=new TextDecoder(); let buf='';
    while(true){
      const {done,value}=await reader.read(); if(done) break;
      buf+=decoder.decode(value,{stream:true});
      let idx;
      while((idx=buf.indexOf('\n'))>=0){
        const line=buf.slice(0,idx).trim(); buf=buf.slice(idx+1);
        if(!line.startsWith('data:')) continue;
        const data=line.slice(5).trim();
        if(data==='[DONE]') continue;
        let chunk; try{ chunk=JSON.parse(data); }catch{ continue; }
        const delta=chunk.choices?.[0]?.delta||{};
        if(delta.reasoning_content){ if(!view.reasoning&&!view.content) view.t0=performance.now(); view.reasoning+=delta.reasoning_content; view.ntok++; }
        if(delta.content){ if(!view.content) view.t0=performance.now(); view.content+=delta.content; view.ntok++; }
        if(chunk.timings) timings=chunk.timings;
        if(chunk.usage) usage=chunk.usage;
        view.paint();
      }
    }
  } catch(err) {
    if(err.name==='AbortError') aborted=true; else failed=err.message;
  }
  view.finalize(aborted&&!failed);
  const client_ms=performance.now()-started;
  if(failed){
    view.contentEl.innerHTML=`<span style="color:var(--danger)">${esc(failed)}</span>`;
    toast(failed,true);
  } else {
    if(!view.content&&!view.reasoning) view.contentEl.innerHTML='<span class="muted">پاسخ متنی خالی بود.</span>';
    if(settings.memory){ state.conversation.push({role:'user',content:prompt}); state.conversation.push({role:'assistant',content:view.content||''}); }
    const estPerSecond=view.ntok>1?view.ntok/((client_ms-(timings?.prompt_ms||0))/1000):undefined;
    const run={id:Date.now()%1000,prompt:prompt.slice(0,160),content:view.content,reasoning:view.reasoning,settings,
      timings:timings||{predicted_n:view.ntok,predicted_ms:client_ms,predicted_per_second:estPerSecond},
      usage:usage||{},client_ms,aborted,date:new Date().toISOString()};
    state.history.unshift(run); state.history=state.history.slice(0,50); localStorage.setItem('glm-history',JSON.stringify(state.history));
    updateMetrics(run); renderHistory(); $('prompt').value=''; updateEstimate();
    toast(aborted?'تست متوقف شد و ثبت گردید':'تست با موفقیت ثبت شد',aborted);
  }
  state.controller=null; $('sendBtn').disabled=false; $('stopBtn').classList.add('hidden');
}

function updateEstimate(){ $('tokenEstimate').textContent=`حدود ${estimateTokens($('prompt').value).toLocaleString('fa-IR')} توکن ورودی`; $('prompt').style.height='auto'; $('prompt').style.height=Math.min($('prompt').scrollHeight,220)+'px'; }
function exportHistory(){ const blob=new Blob([JSON.stringify(state.history,null,2)],{type:'application/json'}); const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=`glm-tests-${new Date().toISOString().slice(0,10)}.json`;a.click();URL.revokeObjectURL(a.href); }

$('chatForm').addEventListener('submit',runPrompt); $('stopBtn').onclick=()=>state.controller?.abort(); $('prompt').addEventListener('input',updateEstimate); $('prompt').addEventListener('keydown',e=>{if(e.ctrlKey&&e.key==='Enter')runPrompt(e)});
$('clearBtn').onclick=()=>{ state.conversation=[]; $('messages').innerHTML='<div class="empty-state" id="emptyState"><div class="orb"><span></span></div><h2>گفتگوی جدید</h2><p>حافظهٔ کش پاک شد؛ پیام بعدی از صفر prefill می‌شود.</p></div>'; };
$('exportBtn').onclick=exportHistory; $('resetSettings').onclick=()=>{applySettings(defaults);saveSettings();toast('تنظیمات بازنشانی شد')};
for(const id of ['temperature','maxTokens']) $(id).addEventListener('input',()=>{ $(id+'Value').textContent=id==='temperature'?(+$(id).value).toFixed(1):$(id).value; saveSettings(); });
for(const id of ['seed','reasoning','memory','systemPrompt']) $(id).addEventListener('change',saveSettings);
document.querySelectorAll('[data-prompt]').forEach(b=>b.onclick=()=>{$('prompt').value=b.dataset.prompt;updateEstimate();$('prompt').focus()});
applySettings(JSON.parse(localStorage.getItem('glm-settings')||'{}')); updateEstimate(); renderHistory(); if(state.history[0])updateMetrics(state.history[0]); checkHealth(); loadModelTag(); setInterval(checkHealth,15000);
