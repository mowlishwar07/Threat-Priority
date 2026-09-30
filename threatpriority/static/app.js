'use strict';
const $=id=>document.getElementById(id);
let queue=[],page=0,selected=null,replayToken=0,replayTimer=null,replayPending=null,replaying=false;
const pageSize=30;
let previousSignature=null,polling=false;
function node(tag,text,cls){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;}
function clear(n){n.replaceChildren();}
function notice(text,error=false){$('notice').hidden=false;$('notice').textContent=text;$('notice').className=error?'error':'';}
async function api(path,options={}){const headers={'X-Requested-With':'ThreatPriority',...(options.headers||{})};const response=await fetch(path,{...options,headers});let data;try{data=await response.json();}catch{throw new Error('Server did not return JSON. Check the running terminal.');}if(!response.ok)throw new Error(data.error||`Request failed (${response.status}).`);return data;}
function jsonPost(path,body){return api(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});}
function fmt(value){return value===null||value===undefined?'N/A':(100*value).toFixed(1)+'%';}
async function action(button,fn){button.disabled=true;try{await fn();}catch(e){notice(e.message,true);}finally{button.disabled=false;}}
async function refresh(){queue=(await api('/api/queue')).events;const signature=queue.map(e=>e.id).join(',');if(previousSignature!==null&&signature!==previousSignature){if(selected&&!queue.some(e=>e.id===selected))resetExplanation();}previousSignature=signature;$('total').textContent=queue.length;$('high').textContent=queue.filter(e=>e.score>=65).length;$('unreviewed').textContent=queue.filter(e=>e.status==='Unreviewed').length;$('incomplete').textContent=queue.filter(e=>e.imputed_fields.length).length;renderQueue();}
function renderQueue(){const filter=$('severity').value;const search=$('search').value.trim().toLowerCase();const filtered=queue.map((e,i)=>({...e,rank:i+1})).filter(e=>(!filter||e.severity===filter)&&(!search||(e.category+' '+e.status).toLowerCase().includes(search)));page=Math.max(0,Math.min(page,Math.ceil(filtered.length/pageSize)-1));clear($('events'));const current=filtered.slice(page*pageSize,(page+1)*pageSize);if(!current.length){const row=node('tr');const cell=node('td',queue.length?'No matching alerts.':'Load a CSV or the demo to build your investigation queue.','empty');cell.colSpan=7;row.append(cell);$('events').append(row);}for(const e of current){const row=node('tr',undefined,e.id===selected?'active':'');row.tabIndex=0;row.setAttribute('aria-label',`Explain rank ${e.rank}, ${e.category}, risk ${e.score.toFixed(1)}`);row.addEventListener('click',()=>showExplanation(e));row.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();showExplanation(e);}});row.append(node('td',e.rank));const event=node('td',e.category);const reference=node('small',e.event_ref.length>28?e.event_ref.slice(0,28)+'\u2026':e.event_ref);reference.title=e.event_ref;event.append(reference);event.append(node('small',e.timestamp?new Date(e.timestamp).toLocaleString():'Capture time unavailable'));row.append(event,node('td',e.score.toFixed(1),'risk'));const sev=node('td');sev.append(node('span',e.severity,'sev '+e.severity));row.append(sev,node('td',e.priority));const evidence=node('td',`${e.observed_features}/9 fields`);if(e.imputed_fields.length)evidence.append(node('small','Contains imputation'));row.append(evidence,node('td',e.status));$('events').append(row);}$('pages').textContent=filtered.length?`${page*pageSize+1}\u2013${Math.min((page+1)*pageSize,filtered.length)} of ${filtered.length} alerts`:'0 alerts';$('prev').disabled=page===0;$('next').disabled=(page+1)*pageSize>=filtered.length;}
function resetExplanation(){selected=null;clear($('explanation'));$('explanation').append(node('p','Select an alert to see its SHAP explanation.','empty-small'));}
async function showExplanation(event){selected=event.id;renderQueue();clear($('explanation'));$('explanation').append(node('p','Computing Tree SHAP\u2026'));try{const data=await api('/api/explain/'+event.id);if(selected!==event.id)return;clear($('explanation'));const score=node('div',undefined,'ex-score');score.append(node('strong',data.score.toFixed(1)),node('span',event.severity+' / '+event.priority));$('explanation').append(score,node('h3',event.category),node('p',data.summary,'ex-note'));if(event.imputed_fields.length)$('explanation').append(node('div','Missing: '+event.imputed_fields.join(', ')+'. Contributions for these fields describe imputed assumptions.','caution'));$('explanation').append(node('p',`Baseline ${data.base_score.toFixed(2)} + all contributions = ${data.reconstructed_score.toFixed(2)} risk points.`,'ex-note'));const max=Math.max(...data.features.map(f=>Math.abs(f.contribution_points)),1);for(const f of data.features){const row=node('div',undefined,'shap-row');const label=node('div',undefined,'shap-label');const direction=f.contribution_points>=0?'positive':'negative';label.append(node('span',f.feature),node('span',(f.contribution_points>=0?'+':'')+f.contribution_points.toFixed(2)+' pts',direction));const track=node('div',undefined,'bar-track');const bar=node('div',undefined,'bar '+direction);bar.style.width=(Math.abs(f.contribution_points)/max*100)+'%';track.append(bar);row.append(label,track,node('small',f.imputed?'Missing / imputed':`Observed: ${f.value}`));$('explanation').append(row);}$('explanation').append(node('p',data.caveat,'ex-note'),node('p',`SHAP calculation: ${data.latency_ms.toFixed(0)} ms \u00b7 reconstruction error: ${data.additivity_error.toExponential(1)} points`,'ex-note'));const controls=node('div',undefined,'feedback');controls.append(node('h3','Analyst decision'));for(const decision of ['Confirmed threat','Benign','Needs investigation']){const button=node('button',decision,'secondary');button.addEventListener('click',()=>action(button,async()=>{const result=await jsonPost('/api/feedback/'+event.id,{decision});notice(result.note);await refresh();}));controls.append(button);}$('explanation').append(controls);}catch(e){if(selected===event.id){clear($('explanation'));$('explanation').append(node('p',e.message));}notice(e.message,true);}}
async function stopReplay(){replaying=false;replayToken++;clearTimeout(replayTimer);$('stream').textContent='Start recorded replay';if(replayPending)await replayPending.catch(()=>{});replayPending=null;}
async function startReplay(){await stopReplay();await api('/api/queue',{method:'DELETE'});resetExplanation();await refresh();const rows=await api('/api/replay-data');replaying=true;const token=++replayToken;let index=0;$('stream').textContent='Stop replay';notice('Recorded replay started: synthetic events arrive one per second.');async function step(){if(!replaying||token!==replayToken)return;if(index>=rows.length){await stopReplay();notice('Replay complete.');return;}try{replayPending=jsonPost('/api/events',{event:rows[index++]});await replayPending;replayPending=null;await refresh();if(replaying&&token===replayToken)replayTimer=setTimeout(step,1000);}catch(e){await stopReplay();notice(e.message,true);}}replayTimer=setTimeout(step,0);}
$('upload-form').addEventListener('submit',e=>{e.preventDefault();action($('upload'),async()=>{await stopReplay();const data=new FormData();data.append('file',$('file').files[0]);data.append('mapping',$('mapping').value.trim()||'{}');const result=await api('/api/upload',{method:'POST',body:data});notice(`${result.accepted} events scored in ${result.scoring_batch_ms.toFixed(0)} ms.\n`+result.warnings.join('\n'));resetExplanation();page=0;await refresh();});});
$('demo').addEventListener('click',()=>action($('demo'),async()=>{await stopReplay();const result=await api('/api/demo',{method:'POST'});notice('Bundled synthetic demo loaded.\n'+result.warnings.join('\n'));resetExplanation();page=0;await refresh();}));
$('stream').addEventListener('click',()=>action($('stream'),async()=>{if(replaying){await stopReplay();notice('Replay paused. The current queue remains available.');}else await startReplay();}));
$('clear').addEventListener('click',()=>action($('clear'),async()=>{await stopReplay();await api('/api/queue',{method:'DELETE'});resetExplanation();await refresh();notice('Active queue cleared. Feedback history remains stored locally.');}));
$('manual-form').addEventListener('submit',e=>{e.preventDefault();const button=e.target.querySelector('button');action(button,async()=>{const event=JSON.parse($('manual').value);const result=await jsonPost('/api/events',{event});notice('Event scored.\n'+result.warnings.join('\n'));await refresh();});});
$('severity').addEventListener('change',()=>{page=0;renderQueue();});$('search').addEventListener('input',()=>{page=0;renderQueue();});$('prev').addEventListener('click',()=>{page--;renderQueue();});$('next').addEventListener('click',()=>{page++;renderQueue();});

// Tab Switching Logic
$('tab-btn-queue').addEventListener('click', () => {
  $('tab-btn-queue').style.background = 'var(--cyan)';
  $('tab-btn-queue').style.color = '#08111c';
  $('tab-btn-model').style.background = 'transparent';
  $('tab-btn-model').style.color = 'var(--muted)';
  
  $('tab-content-queue').style.display = 'grid';
  $('tab-content-model').style.display = 'none';
});

$('tab-btn-model').addEventListener('click', () => {
  $('tab-btn-model').style.background = 'var(--cyan)';
  $('tab-btn-model').style.color = '#08111c';
  $('tab-btn-queue').style.background = 'transparent';
  $('tab-btn-queue').style.color = 'var(--muted)';
  
  $('tab-content-queue').style.display = 'none';
  $('tab-content-model').style.display = 'block';
});

(async()=>{try{
  const model=await api('/api/model');
  $('source').textContent=model.metadata.provenance;
  const t=model.test_metrics;
  const acc=(t.confusion_matrix.tp+t.confusion_matrix.tn)/t.n_events;
  
  if($('model-info')) {
    $('model-info').textContent = `Model: ${model.metadata.model} \u00b7 Training rows: ${model.metadata.training_rows} \u00b7 Held-out evaluation rows: ${model.metadata.test_rows}`;
  }
  
  clear($('model-metrics'));
  for(const [title,val] of [['Precision',t.precision],['Recall',t.recall],['F1 Score',t.f1],['Accuracy',acc]]){
    const art=node('article');
    art.append(node('span',title), node('strong', fmt(val)));
    $('model-metrics').append(art);
  }
  
  await refresh();
}catch(e){notice(e.message,true);}})();

setInterval(async()=>{if(document.hidden||replaying||polling)return;polling=true;try{await refresh();}catch(e){notice(e.message,true);}finally{polling=false;}},2000);
