import {Timeline, currentPrediction, formatTime} from "./timeline.js?v=20261026-corrections-2";
import {submitCompatibleModels} from "./inference.js?v=20261026-corrections-2";
import {correctionRange, saveCorrection, selectionAt, selectedCorrections} from "./corrections.js?v=20261026-corrections-2";

const $ = id => document.getElementById(id);
const text = (id, value) => { $(id).textContent = value; };
const show = (id, visible = true) => { $(id).hidden = !visible; };
const sessionPath = id => `/api/v1/sessions/${encodeURIComponent(id)}`;
const taskNames = {sleep:"Schlafstadium",sleep_stage:"Schlafstadium",arousal:"Arousal",breathing:"Atmung",desaturation:"Desaturation"};
const classNames = {wake:"W",w:"W",n1:"N1",n2:"N2",n3:"N3",rem:"REM",arousal:"Arousal",no_arousal:"Kein Arousal",apnea:"Apnoe",hypopnea:"Hypopnoe","regular breathing":"Reguläre Atmung",desaturation:"Desaturation","no desaturation":"Keine Desaturation"};
const taskName = task => taskNames[task.task || task.id] || task.name || task.id || task.task;
const className = label => classNames[String(label).toLowerCase()] || label;
const state = {
  connected:false, file:null, models:[], session:null, title:"Aufzeichnung", channels:[], channelDraft:[],
  range:[0,120], time:0, interval:null, taskId:null, tasks:[], predictions:new Map(),
  annotations:{edits:[],labels:[],events:[]}, expanded:new Set(), absolute:false,
  view:makeViewId(), revision:0, frameReady:false, plotLoaded:false, generation:0, rangeRequest:0,
  predictionAbort:null, predictionTimer:null, job:null, jobStream:null, jobTimer:null, jobEpoch:0,
  export:null, busy:false, mutation:false, inputIssues:new Map(), selection:null, mode:"select", correction:null,
};
let profiles = readStorage(localStorage,"sleepwalker.servers",[]);
if (!Array.isArray(profiles)) profiles = [];

function makeViewId() {
  // randomUUID requires HTTPS on LAN hosts; getRandomValues also works on HTTP.
  if (typeof crypto.randomUUID === "function") return crypto.randomUUID();
  return [...crypto.getRandomValues(new Uint8Array(16))].map(value=>value.toString(16).padStart(2,"0")).join("");
}

function element(tag, content, cls) {
  const node = document.createElement(tag);
  if (content !== undefined) node.textContent = content;
  if (cls) node.className = cls;
  return node;
}
function readStorage(storage,key,fallback) { try { return JSON.parse(storage.getItem(key)) ?? fallback; } catch { return fallback; } }
function writeStorage(storage,key,value) { try { if (value === null) storage.removeItem(key); else storage.setItem(key,JSON.stringify(value)); } catch { /* Browsing without persistence is supported. */ } }
function rememberSession() {
  if (!state.session) return;
  writeStorage(sessionStorage,"sleepwalker.session",{id:state.session.id,job:state.job,export:state.export,range:state.range,time:state.time,taskId:state.taskId,channels:state.channels});
}
function notice(message, error=false) {
  text("notice-text",message); $("notice").classList.toggle("error",error); show("notice");
}
function errorMessage(error) {
  const messages = {
    SESSION_NOT_FOUND:"Die Session ist abgelaufen oder wurde gelöscht. EDF oder Workspace erneut öffnen.",
    INVALID_STUDY:"Die Datei ist keine lesbare EDF-Aufzeichnung.", FILE_TOO_LARGE:"Die Datei überschreitet die Uploadgrenze des Servers.",
    MISSING_REQUIRED_CHANNEL:`Modell „${state.models.find(model=>model.id===error.details?.model_id)?.name||error.details?.model_id||"unbekannt"}“: keine passende Kanalzuordnung für ${error.details?.channels?.join(", ")||"die benötigten Eingänge"}. Die Signale können fehlen oder anders heißen.`,
    COMPUTE_UNAVAILABLE:"Die Inferenzumgebung ist auf diesem Server nicht verfügbar.",
    JOB_ALREADY_RUNNING:"Für diese Session läuft bereits ein Job.",
    MODEL_INCOMPATIBLE:"Das Modell ist mit dieser Aufnahme oder den vorhandenen Ergebnissen nicht kompatibel.",
    INVALID_BUNDLE:"Der Workspace ist beschädigt oder ungültig.", UNSUPPORTED_FORMAT_VERSION:"Diese Workspace-Version wird vom Server nicht unterstützt.",
    RESOURCE_EXHAUSTED:"Der Server hat zu wenig Rechenspeicher für diese Inferenz.",
    EXPORT_NOT_READY:"Der Export ist noch nicht fertig.", INVALID_SESSION_STATE:"Die Session ist noch nicht bereit.",
    EDIT_NOT_FOUND:"Diese Änderung kann nicht mehr rückgängig gemacht werden. Den Verlauf aktualisieren.",
  };
  return messages[error.code] || error.message || "Die Anfrage ist fehlgeschlagen.";
}
function handleError(error) {
  if (error.name === "AbortError") return;
  if (error.code === "SESSION_NOT_FOUND" && state.session) {
    closeView(true); writeStorage(sessionStorage,"sleepwalker.session",null); renderResume();
  }
  notice(errorMessage(error),true);
}
function parseError(data,status) {
  const detail = data?.error || {};
  const error = new Error(typeof detail.message === "string" ? detail.message : `Anfrage fehlgeschlagen (HTTP ${status}).`);
  error.code=detail.code; error.details=detail.details; return error;
}
async function api(path,{json,...options}={}) {
  if (json !== undefined) { options.body=JSON.stringify(json); options.headers={...options.headers,"Content-Type":"application/json"}; }
  let response;
  try { response=await fetch(path,{credentials:"same-origin",...options}); }
  catch (error) { if(error.name === "AbortError") throw error; throw new Error("Verbindung zum Server unterbrochen. Die Session bleibt auf dem Server; erneut verbinden oder die Seite neu laden."); }
  if (response.status === 204) return null;
  const data=await response.json().catch(() => null);
  if (!response.ok) throw parseError(data,response.status);
  if (data === null) throw new Error("Der Server hat keine gültige Antwort geliefert.");
  return data;
}
function bind(id,handler) { $(id).addEventListener("click",() => Promise.resolve().then(handler).catch(handleError)); }
function timeLabel(seconds) {
  if (!state.absolute || !state.session?.recording_start) return formatTime(seconds,true);
  const match=state.session.recording_start.match(/T(\d+):(\d+):(\d+(?:\.\d+)?)/);
  if (!match) return formatTime(seconds,true);
  const total=Number(match[1])*3600+Number(match[2])*60+Number(match[3])+seconds;
  const day=Math.floor(total/86400);
  return formatTime(total%86400,true)+(day ? ` (+${day})` : "");
}

function renderProfiles() {
  const select=$("server-profile"); select.replaceChildren(new Option("Aktuelle Instanz","current"));
  const list=$("server-list"); list.replaceChildren();
  for (const profile of profiles.filter(p => p && typeof p.origin === "string").slice(0,10)) {
    select.add(new Option(profile.name || profile.origin,profile.origin));
    const button=element("button",undefined,"saved-server");
    button.append(element("span",profile.name || profile.origin),element("small",profile.origin));
    button.addEventListener("click",() => { $("server-address").value=profile.origin; select.value=profile.origin; });
    list.append(button);
  }
}
async function connect() {
  const address=new URL($("server-address").value || location.origin,location.origin);
  if (!["http:","https:"].includes(address.protocol) || address.username || address.password) throw new Error("Eine HTTP- oder HTTPS-Serveradresse ohne Zugangsdaten eingeben.");
  if (address.origin !== location.origin) {
    profiles=profiles.filter(p => p.origin !== address.origin);
    profiles.unshift({origin:address.origin,name:address.hostname === "localhost" || address.hostname === "127.0.0.1" ? "Lokal" : address.host});
    writeStorage(localStorage,"sleepwalker.servers",profiles.slice(0,10));
    location.assign(`${address.origin}/ui/`);
    return;
  }
  $("connect").disabled=true;
  try {
    const [config,health,models]=await Promise.all([api("/ui/config"),api("/health"),api("/api/v1/models")]);
    if(config.ui !== "sleepwalker" || health.status !== "ok" || !Array.isArray(models.models)) throw new Error("Die Adresse stellt keine passende Sleepwalker-GUI bereit.");
    state.models=models.models; state.connected=true;
    text("connection-status","Verbunden"); $("connection-status").classList.add("connected");
    text("server-indicator",location.host); $("server-indicator").classList.add("connected");
    text("model-count",`${state.models.length} Modelle verfügbar`);
    profiles=profiles.filter(p => p.origin !== location.origin);
    profiles.unshift({origin:location.origin,name:location.hostname === "localhost" || location.hostname === "127.0.0.1" ? "Lokal" : location.host});
    writeStorage(localStorage,"sleepwalker.servers",profiles.slice(0,10)); renderProfiles(); renderModels(); renderResume();
  } catch (error) {
    state.connected=false; text("connection-status","Nicht verbunden"); $("connection-status").classList.remove("connected"); throw error;
  } finally { $("connect").disabled=false; updateOpenButtons(); }
}
function renderModels() {
  const tbody=$("models"); tbody.replaceChildren();
  const seen=new Set();
  for(const model of state.models) {
    const row=element("tr"),cell=element("td"),label=element("label"),input=element("input"),description=element("span",taskName(model));
    input.type="checkbox"; input.value=model.id; input.dataset.task=model.task; input.checked=!seen.has(model.task); seen.add(model.task);
    description.append(element("small",model.name),element("small",`Benötigte Eingänge: ${model.input_channels.join(", ")}`)); label.append(input,description); cell.append(label);
    row.append(cell,element("td",`${model.output_resolution} s`,"mono"),element("td",model.classes.map(className).join(", "),"model-classes"));
    input.addEventListener("change",() => { if(input.checked) for(const other of tbody.querySelectorAll("input")) if(other !== input && other.dataset.task === input.dataset.task) other.checked=false; updateOpenButtons(); });
    tbody.append(row);
  }
  if(!state.models.length) { const row=element("tr"),cell=element("td","Keine Modelle auf diesem Server eingerichtet. Signalansicht und Workspace-Import sind verfügbar.","muted");cell.colSpan=3;row.append(cell);tbody.append(row); }
}
function updateOpenButtons() {
  $("open-signals").disabled=!state.connected || !state.file || state.busy;
  $("open-infer").disabled=!state.connected || !state.file || state.busy || !$("models").querySelector("input:checked");
  $("workspace-file").disabled=!state.connected || state.busy;
  $("edf-file").disabled=state.busy;
}
function setFile(file) {
  if (!file) return;
  if (!/\.edf$/i.test(file.name)) { notice("Bitte eine EDF-Datei auswählen.",true); return; }
  state.file=file; text("file-name",file.name); text("file-size",`${(file.size/1024/1024).toLocaleString("de-DE",{maximumFractionDigits:1})} MiB`); updateOpenButtons();
}
function upload(path,file,field) {
  return new Promise((resolve,reject) => {
    const xhr=new XMLHttpRequest(); xhr.open("POST",path);
    show("upload-progress"); $("upload-meter").value=0; text("upload-status","Upload wird gestartet …");
    xhr.upload.onprogress=event => { if(event.lengthComputable) { const percent=event.loaded/event.total*100;$("upload-meter").value=percent;text("upload-status",percent>=100 ? "Datei übertragen · wird geprüft …" : `Upload ${Math.round(percent)} %`); } };
    xhr.onload=() => { let data;try{data=JSON.parse(xhr.responseText);}catch{reject(new Error("Ungültige Antwort beim Upload."));return;} if(xhr.status>=200&&xhr.status<300)resolve(data);else reject(parseError(data,xhr.status)); };
    xhr.onerror=() => reject(new Error("Upload unterbrochen. Verbindung prüfen. Eine eventuell bereits angelegte Session wird nach Ablauf automatisch bereinigt."));
    const form=new FormData();form.append(field,file);xhr.send(form);
  });
}
async function openEDF(infer) {
  if (state.busy || !state.file) return;
  const file=state.file;state.busy=true; updateOpenButtons();
  try {
    const models=[...$("models").querySelectorAll("input:checked")].map(input=>input.value);
    const session=await upload("/api/v1/sessions",file,"files");
    await activateSession(session,{title:file.name});
    if(infer && models.length) await startInference(models);
  } finally {state.busy=false;show("upload-progress",false);updateOpenButtons();}
}
function renderResume() {
  const saved=readStorage(sessionStorage,"sleepwalker.session",null);
  show("resume-panel",Boolean(saved?.id && !state.session));
  if(saved?.id) text("resume-text","Auf diesem Server ist ein Verweis auf deine letzte Session vorhanden.");
}
function chooseChannels(signals) {
  const priorities=[/EEG|C3|C4|F3|F4/i,/EOG/i,/Airflow|Flow|Nasal/i,/Thorax|Chest/i,/SpO2|SaO2|SPO2/i];
  const chosen=[];
  for(const rule of priorities){const found=signals.find(s=>rule.test(s.label)&&!chosen.includes(s.id));if(found)chosen.push(found.id);}
  if(!chosen.length) chosen.push(...signals.slice(0,3).map(s=>s.id));
  return chosen;
}
function setViewer(visible) {
  show("entry",!visible);show("viewer",visible);
  for(const id of ["export-open","close-open","inspector-toggle"])show(id,visible);
  $("nav-file").classList.toggle("active",!visible);$("nav-view").classList.toggle("active",visible);
  $("nav-view").disabled=!visible;$("nav-history").disabled=!visible;
}
function sendPlot(type,extra={}) { const frame=$("signal-frame"); if(frame.contentWindow) frame.contentWindow.postMessage({source:"sleepwalker-ui",type,...extra},location.origin); }
function requestPlot(action="view") {
  if(!state.frameReady || !state.session || state.session.state!=="ready")return;
  if(!state.channels.length)return;
  sendPlot("view",{request:{session:state.session.id,view:state.view,channels:state.channels,range:state.range,duration:state.session.duration,initialize:!state.plotLoaded,revision:action==="heartbeat"?state.revision:++state.revision,action}});
  sendPlot("cursor",{time:state.time});
  sendPlot("mode",{mode:state.mode});
  sendPlot("interval",{range:state.interval});
}
function loadFrame() {
  const frame=$("signal-frame");
  if(!frame.getAttribute("src")){state.frameReady=false;frame.src="/ui/plot/";}
  else requestPlot();
}
async function activateSession(session,{title,saved}={}) {
  if(state.session && state.session.id!==session.id) sendPlot("close");
  state.generation++; state.predictionAbort?.abort();
  state.selection=null;state.correction=null;closeSelectionMenu();if($("correction-dialog").open)$("correction-dialog").close();
  state.view=makeViewId();
  state.plotLoaded=false;
  state.session=session; state.title=title || "Aufzeichnung"; state.predictions=new Map();state.annotations={edits:[],labels:[],events:[]};state.expanded=new Set();state.inputIssues=new Map();renderInputIssues();
  const available=(session.signals||[]).map(s=>s.id);
  state.channels=saved?.channels?.filter(c=>available.includes(c))||chooseChannels(session.signals||[]);
  if(!state.channels.length&&available.length)state.channels=chooseChannels(session.signals);
  state.range=[0,Math.min(120,session.duration||120)];state.time=0;state.interval=null;state.taskId=saved?.taskId||null;
  if(saved?.range&&Number.isFinite(saved.range[0])&&Number.isFinite(saved.range[1])&&session.duration)state.range=clampRange(saved.range[0],saved.range[1]);
  if(Number.isFinite(saved?.time))state.time=Math.max(0,Math.min(saved.time,(session.duration||1)-1e-6));
  state.tasks=session.tasks||[];setViewer(true);text("study-title",state.title);text("study-meta",`${session.format?.toUpperCase()||"EDF"} · ${formatTime(session.duration||0)} · temporäre Session`);
  text("recording-start",session.recording_start?.replace("T"," ")||"—");text("expires-at",session.expires_at ? new Date(session.expires_at).toLocaleTimeString("de-DE") : "—");
  show("plot-loading");show("signal-frame",session.state==="ready");
  state.job=saved?.job||null;state.export=saved?.export||null;
  updateTaskOptions();renderRange();renderContext();rememberSession();
  if(session.state==="ready") {loadFrame();await Promise.all([refreshTasks(),refreshAnnotations()]);}
  else text("plot-loading","Workspace wird wiederhergestellt …");
  renderResume();
}
async function refreshTasks() {
  if(!state.session)return;const id=state.session.id,generation=state.generation;
  const result=await api(`${sessionPath(id)}/tasks`);
  if(generation!==state.generation)return;
  state.tasks=result.tasks;updateTaskOptions();schedulePredictions();
}
function updateTaskOptions() {
  state.tasks.sort((a,b)=>{const order=["sleep","sleep_stage","arousal","breathing","desaturation"];const ai=order.indexOf(a.id),bi=order.indexOf(b.id);return (ai<0?99:ai)-(bi<0?99:bi)||a.name.localeCompare(b.name);});
  const select=$("task-select");select.replaceChildren();for(const task of state.tasks)select.add(new Option(taskName(task),task.id));
  if(!state.tasks.some(t=>t.id===state.taskId))state.taskId=state.tasks[0]?.id||null;
  if(state.taskId)select.value=state.taskId;
  select.disabled=!state.tasks.length;text("task-count",`${state.tasks.length} Tasks`);renderContext();drawTimeline();
  if(!state.job)text("inference-state",state.tasks.length?`${state.tasks.length} Ergebnisspuren`:"Signale bereit");
}
async function refreshAnnotations() {
  if(!state.session||state.session.state!=="ready")return;
  const generation=state.generation;const data=await api(`${sessionPath(state.session.id)}/annotations`);
  if(generation!==state.generation)return;
  state.annotations=data;
  if(["label","event"].includes(state.selection?.kind)) {
    const selected=state.selection,task=state.tasks.find(t=>t.id===selected.task);
    const current=task?selectionAt(task,state.predictions.get(task.id)||[],data,(selected.start+selected.end)/2):null;
    state.selection=current||{kind:"interval",task:selected.task,start:selected.start,end:selected.end};
    state.interval=[state.selection.start,state.selection.end];sendPlot("interval",{range:state.interval});
  }
  renderContext();renderHistory();drawTimeline();
}
function clampRange(start,end) {
  const duration=state.session?.duration||120,width=Math.max(.05,Math.min(duration,end-start));
  const left=Math.max(0,Math.min(duration-width,start));return[left,left+width];
}
function setRange(start,end,fromPlot=false) {
  if(!state.session||!Number.isFinite(start)||!Number.isFinite(end)||end<=start)return;
  const next=clampRange(start,end);if(next.every((x,i)=>Math.abs(x-state.range[i])<1e-6))return;
  state.range=next;renderRange();schedulePredictions();drawTimeline();rememberSession();
  if(!fromPlot)requestPlot();
}
function renderRange() {
  text("range-label",`${timeLabel(state.range[0])} – ${timeLabel(state.range[1])}`);
  $("goto-time").value=formatTime(state.time);
  const width=state.range[1]-state.range[0];const choice=[30,120,300,900].find(n=>Math.abs(n-width)<1e-4);
  $("window-size").value=choice?String(choice):Math.abs(width-(state.session?.duration||0))<1e-4?"all":"";
  $("previous").disabled=state.range[0]<=1e-6;$("next").disabled=state.range[1]>=(state.session?.duration||0)-1e-6;
}
function navigate(direction) {const width=state.range[1]-state.range[0];setRange(state.range[0]+direction*width,state.range[1]+direction*width);}
function schedulePredictions() {
  clearTimeout(state.predictionTimer);state.predictionAbort?.abort();state.predictionTimer=setTimeout(()=>refreshPredictions().catch(handleError),130);
}
async function refreshPredictions() {
  if(!state.session||state.session.state!=="ready")return;
  const generation=state.generation,request=++state.rangeRequest,controller=new AbortController();state.predictionAbort=controller;
  const [start,end]=state.range,id=state.session.id;show("prediction-loading",state.tasks.length>0);
  try {
    const data=await Promise.all(state.tasks.map(async task=>{
      if(task.kind==="continuous"&&end-start>900)return[task.id,[]];
      const response=await api(`${sessionPath(id)}/predictions/${encodeURIComponent(task.id)}?${new URLSearchParams({start,end})}`,{signal:controller.signal});return[task.id,response.predictions];
    }));
    if(generation!==state.generation||request!==state.rangeRequest||controller.signal.aborted)return;
    state.predictions=new Map(data);renderContext();drawTimeline();
  } finally {if(request===state.rangeRequest&&!controller.signal.aborted)show("prediction-loading",false);}
}
function drawTimeline() {
  timeline.update({tasks:state.tasks,predictions:state.predictions,annotations:state.annotations,expanded:state.expanded,range:state.range,time:state.time,taskId:state.taskId,selection:state.selection,timeLabel,taskName,className});
}
const timeline=new Timeline($("timeline"),{
  select:(task,time,target)=>target?selectRegion(target):selectTime(time,task),
  interval:(task,start,end)=>selectRegion({kind:"interval",task,start,end}),
  context:openSelectionMenu,
  toggle:task=>{if(state.expanded.has(task))state.expanded.delete(task);else state.expanded.add(task);drawTimeline();},
});
function selectTime(time,taskId=state.taskId,keepInterval=false) {
  if(!state.session)return;state.time=Math.max(0,Math.min(time,state.session.duration-1e-6));state.taskId=taskId;
  if(!keepInterval){state.interval=null;state.selection=null;}
  closeSelectionMenu();
  if(taskId)$("task-select").value=taskId;
  $("goto-time").value=formatTime(state.time);
  sendPlot("cursor",{time:state.time});sendPlot("interval",{range:state.interval});renderContext();drawTimeline();rememberSession();
}
function renderContext() {
  const task=state.tasks.find(t=>t.id===state.taskId),items=state.predictions.get(state.taskId)||[];
  const prediction=currentPrediction(task,items,state.time);
  const selected=state.interval||(prediction&&task?.kind==="epoch"?[prediction.start,prediction.end]:null);
  text("selection-range",selected?`${timeLabel(selected[0])} – ${timeLabel(selected[1])}`:timeLabel(state.time));
  const selectionKind={interval:"Markiertes Intervall",event:"Eigenes Ereignis",label:"Eigene Klassenkorrektur"}[state.selection?.kind];
  text("selection-meta",`${selectionKind?selectionKind+" · ":task?.kind==="epoch"?`${task.resolution}-s-Epoche · `:""}Zeitpunkt ${timeLabel(state.time)}`);
  text("prediction-label",prediction?className(prediction.label):"—");
  const confidence=prediction?.probabilities?.[prediction.label];text("prediction-confidence",Number.isFinite(confidence)?`${Math.round(confidence*100)} %`:"");
  const human=prediction&&task?.kind==="epoch"?state.annotations.labels.find(e=>e.task===task.id&&Math.abs(e.start-prediction.start)<1e-6):null;
  text("effective-label",human?`Manuell: ${className(human.label)} · Modell: ${className(prediction.label)}`:prediction?"Modellwahrscheinlichkeit":"Keine Vorhersage an diesem Zeitpunkt.");
  const probabilities=$("probabilities");probabilities.replaceChildren();
  for(const [label,value] of Object.entries(prediction?.probabilities||{})) {
    const row=element("div",undefined,"probability-row"),bar=element("div",undefined,"bar"),fill=element("span");fill.style.width=`${Math.max(0,Math.min(100,value*100))}%`;bar.append(fill);
    row.append(element("span",className(label)),bar,element("span",`${Math.round(value*100)} %`,"value mono"));probabilities.append(row);
  }
  show("probability-details",Boolean(prediction));
  $("selection-edit").disabled=state.mutation||!state.selection||!state.tasks.length;
  text("viewer-status",`Auswahl: ${timeLabel(state.time)}${task?` · ${taskName(task)}`:""}${human?" · manuell korrigiert":""}`);
  text("timeline-summary",task?`${taskName(task)} bei ${timeLabel(state.time)}: ${human?className(human.label):prediction?className(prediction.label):"keine Vorhersage"}`:"Keine Ergebnisspuren verfügbar.");
}
function renderHistory() {
  const container=$("history");container.replaceChildren();const edits=state.annotations.edits;
  if(!edits.length){container.append(element("p","Noch keine Änderungen.","muted small"));return;}
  const reverted=new Set(edits.filter(e=>e.operation==="revert").map(e=>e.reverted_edit_id));
  const shown=container.dataset.expanded==="true"?edits:[...edits].slice(-8);
  for(const edit of [...shown].reverse()) {
    const row=element("div",undefined,"history-item"),heading=element("div");
    heading.append(element("span",Number.isFinite(edit.start)?formatTime(edit.start,true):new Date(edit.created_at).toLocaleTimeString("de-DE"),"mono small"));
    const operation={set_label:`Klasse: ${className(edit.label)}`,add_event:`Ereignis: ${className(edit.label)}`,delete_event:"Ereignis gelöscht",revert:"Rückgängig"}[edit.operation]||edit.operation;
    heading.append(element("span",operation,"small"));row.append(heading);
    row.append(element("p",`${taskNames[edit.task]||edit.task||"Verlauf"}${reverted.has(edit.edit_id)?" · zurückgenommen":""}`,"muted small"));
    if(edit.operation!=="revert"&&!reverted.has(edit.edit_id)) { const button=element("button","Rückgängig","text-button");button.disabled=state.mutation;button.addEventListener("click",()=>annotate({operation:"revert",edit_id:edit.edit_id}).catch(handleError));row.append(button); }
    container.append(row);
  }
  if(edits.length>8&&container.dataset.expanded!=="true"){const button=element("button",`Alle ${edits.length} Änderungen anzeigen`,"text-button small");button.addEventListener("click",()=>{container.dataset.expanded="true";renderHistory();});container.append(button);}
}
async function annotate(payload) {
  if(!state.session||state.mutation)return;
  await mutateAnnotations(post=>post(payload));notice("Änderung in der temporären Session gespeichert.");
}
function selectRegion(selection) {
  if(!state.session)return;
  const start=Math.max(0,selection.start),end=Math.min(state.session.duration,selection.end);
  if(!Number.isFinite(start)||!Number.isFinite(end)||end<=start)return;
  state.selection={...selection,start,end};state.interval=[start,end];
  selectTime((start+end)/2,selection.task||state.taskId,true);
}
function closeSelectionMenu() {show("selection-menu",false);}
function openSelectionMenu({task=state.taskId,time=state.time,target=null,x,y}) {
  if(!state.session||state.mutation)return;
  const selected=state.selection;
  if(!selected||selected.task!==task||time<selected.start||time>=selected.end) {
    const spec=state.tasks.find(t=>t.id===task);
    const hit=target||(spec?selectionAt(spec,state.predictions.get(task)||[],state.annotations,time):null);
    selectRegion(hit||{kind:"interval",task,start:time,end:Math.min(state.session.duration,time+Math.max(spec?.resolution||1,.05))});
  }
  const selection=state.selection;if(!selection)return;
  const spec=state.tasks.find(t=>t.id===selection.task);
  text("selection-menu-summary",`${spec?taskName(spec):"Auswahl"} · ${formatTime(selection.start,true)}–${formatTime(selection.end,true)}`);
  text("context-edit",selection.kind==="event"||selection.kind==="label"?"Änderung bearbeiten …":"Korrigieren …");
  $("context-edit").disabled=!state.tasks.length;
  const changes=selectedCorrections(selection,spec,state.annotations);
  show("context-delete",changes.length>0);text("context-delete",changes.length>1?`${changes.length} Änderungen löschen`:"Änderung löschen");
  show("context-delete-prediction",selection.kind==="prediction");show("context-limit",selection.kind==="prediction");
  const menu=$("selection-menu");show("selection-menu");
  menu.style.left=`${Math.max(8,Math.min(x??innerWidth/2,innerWidth-menu.offsetWidth-8))}px`;
  menu.style.top=`${Math.max(8,Math.min(y??innerHeight/2,innerHeight-menu.offsetHeight-8))}px`;
  $("context-edit").focus({preventScroll:true});
}
function setInteractionMode(mode) {
  state.mode=mode;$("select-tool").setAttribute("aria-pressed",String(mode==="select"));$("pan-tool").setAttribute("aria-pressed",String(mode==="pan"));
  sendPlot("mode",{mode});
}
function openCorrection() {
  closeSelectionMenu();
  if(!state.selection||state.mutation||!state.tasks.length)return;
  state.correction={...state.selection};
  const select=$("correction-task");select.replaceChildren();
  for(const task of state.tasks)select.add(new Option(taskName(task),task.id));
  select.value=state.tasks.some(t=>t.id===state.correction.task)?state.correction.task:state.tasks[0].id;
  select.disabled=["event","label","prediction"].includes(state.correction.kind);
  $("correction-start").value=Number(state.correction.start.toFixed(6));$("correction-end").value=Number(state.correction.end.toFixed(6));
  updateCorrectionClasses();show("correction-error",false);updateCorrectionPreview();
  text("correction-title",state.correction.kind==="event"||state.correction.kind==="label"?"Änderung bearbeiten":"Auswahl korrigieren");
  $("correction-dialog").showModal();
}
function updateCorrectionClasses() {
  const task=state.tasks.find(t=>t.id===$("correction-task").value),select=$("correction-label");select.replaceChildren();
  for(const label of task?.classes||[])select.add(new Option(className(label),label));
  const previous=task?selectionAt(task,state.predictions.get(task.id)||[],state.annotations,(state.correction.start+state.correction.end)/2):null;
  const label=state.correction?.task===task?.id&&state.correction.label?state.correction.label:previous?.label;
  if(task?.classes.includes(label))select.value=label;
}
function dialogRange() {
  const task=state.tasks.find(t=>t.id===$("correction-task").value);
  const start=$("correction-start").valueAsNumber,end=$("correction-end").valueAsNumber;
  return {task,range:correctionRange(task,start,end,state.session.duration)};
}
function updateCorrectionPreview() {
  try {
    const {task,range}=dialogRange();
    text("correction-preview",task.kind==="epoch"?`${range.count} ${range.count===1?"Epoche":"Epochen"} à ${task.resolution} s: ${formatTime(range.start,true)}–${formatTime(range.end,true)}. Berührte Epochen werden vollständig korrigiert.`:`Ereignis: ${formatTime(range.start,true)}–${formatTime(range.end,true)} (${Number((range.end-range.start).toFixed(3))} s).`);
    $("apply-correction").disabled=state.mutation;
  } catch(error) {text("correction-preview",error.message);$("apply-correction").disabled=true;}
}
function closeCorrection() {if(!state.mutation){$("correction-dialog").close();state.correction=null;}}
async function mutateAnnotations(action) {
  if(!state.session||state.mutation)return;
  const id=state.session.id,generation=state.generation;
  state.mutation=true;renderContext();renderHistory();
  const post=payload=>{
    if(generation!==state.generation)throw new DOMException("Die Session wurde gewechselt.","AbortError");
    return api(`${sessionPath(id)}/annotations`,{method:"POST",json:payload});
  };
  try {return await action(post);}
  finally {
    try {if(generation===state.generation)await refreshAnnotations();}
    finally {state.mutation=false;renderContext();renderHistory();}
  }
}
async function applyCorrection() {
  if(!state.correction||state.mutation)return;
  let count=0,total=0;
  try {
    const {task,range}=dialogRange(),label=$("correction-label").value;
    const event=state.correction.kind==="event"?state.correction:null;
    show("correction-error",false);
    for(const control of $("correction-form").elements)control.disabled=true;
    const saved=await mutateAnnotations(post=>saveCorrection({task,range,label,event},post,(done,all)=>{count=done;total=all;text("apply-correction",`Speichern ${done}/${all} …`);}));
    const last=saved?.[saved.length-1];
    selectRegion({kind:task.kind==="epoch"?(range.count===1?"label":"interval"):"event",task:task.id,start:range.start,end:range.end,label,editId:last?.edit_id,eventId:last?.event_id});
    $("correction-dialog").close();state.correction=null;
    notice(`${task.kind==="epoch"?`${range.count} ${range.count===1?"Epoche":"Epochen"} korrigiert`:event?"Ereignis geändert":"Ereignis hinzugefügt"}.`);
  } catch(error) {
    text("correction-error",`${count&&total>1?`${count} von ${total} Epochen wurden gespeichert. `:""}${errorMessage(error)}`);show("correction-error");
  } finally {
    for(const control of $("correction-form").elements)control.disabled=false;
    $("correction-task").disabled=["event","label","prediction"].includes(state.correction?.kind);
    text("apply-correction","Speichern");if($("correction-dialog").open)updateCorrectionPreview();
  }
}
async function deleteSelectedCorrection() {
  closeSelectionMenu();const selection=state.selection;
  const task=state.tasks.find(t=>t.id===selection?.task),changes=selectedCorrections(selection,task,state.annotations);
  if(!changes.length)return;
  let deleted=0;
  try {await mutateAnnotations(async post=>{for(const operation of changes){await post(operation);deleted++;}});}
  catch(error){if(deleted)throw new Error(`${deleted} von ${changes.length} Änderungen wurden gelöscht. ${errorMessage(error)}`);throw error;}
  const time=(selection.start+selection.end)/2;
  const next=task?selectionAt(task,state.predictions.get(task.id)||[],state.annotations,time):null;
  if(next)selectRegion(next);else selectTime(time,selection.task);
  notice(changes.length>1?`${changes.length} Änderungen gelöscht.`:"Änderung gelöscht.");
}

function stopWatchingJob() {state.jobStream?.close();state.jobStream=null;clearTimeout(state.jobTimer);state.jobEpoch++;}
function renderJob(snapshot) {
  const states={queued:"Wartet",running:"Läuft",cancelling:"Wird abgebrochen",succeeded:"Abgeschlossen",failed:"Fehlgeschlagen",cancelled:"Abgebrochen"};
  const phases={loading:"Laden",preprocessing:"Vorverarbeitung",inference:"Inferenz",postprocessing:"Nachverarbeitung",export:"Export",import:"Import"};
  text("job-phase",`${state.job?.type==="export"?"Export":state.job?.type==="import"?"Workspace-Import":"Inferenz"} · ${phases[snapshot.phase]||states[snapshot.state]||snapshot.state}`);
  text("job-message",snapshot.message||states[snapshot.state]||"");
  if(Number.isFinite(snapshot.progress))$("job-meter").value=snapshot.progress;else $("job-meter").removeAttribute("value");
  $("cancel-job").disabled=snapshot.state==="cancelling";
  text("inference-state",state.job?.type==="inference"?`Inferenz: ${states[snapshot.state]||snapshot.state}`:state.tasks.length?`${state.tasks.length} Ergebnisspuren`:"Signale bereit");
}
function watchJob(job) {
  stopWatchingJob();state.job=job;rememberSession();show("job-panel");$("export-start").disabled=true;$("infer-open").disabled=true;
  const epoch=state.jobEpoch,generation=state.generation;let terminal=false;
  const path=`/api/v1/jobs/${encodeURIComponent(job.id)}`;
  async function update(snapshot) {
    if(epoch!==state.jobEpoch||generation!==state.generation||terminal)return;
    renderJob(snapshot);
    if(!["succeeded","failed","cancelled"].includes(snapshot.state))return;
    terminal=true;stopWatchingJob();state.job=null;rememberSession();show("job-panel",false);$("export-start").disabled=false;$("infer-open").disabled=false;
    if(snapshot.state==="failed") {const error=new Error(snapshot.error?.message||"Der Job ist fehlgeschlagen.");error.code=snapshot.error?.code;error.details=snapshot.error?.details;throw error;}
    if(snapshot.state==="cancelled"){notice("Job abgebrochen. Die Session bleibt verfügbar.");if(job.type==="import")text("plot-loading","Workspace-Import abgebrochen. Session schließen und bei Bedarf erneut öffnen.");return;}
    if(job.type==="export") await finishExport();
    else if(job.type==="import") {
      const session=await api(sessionPath(state.session.id));
      if(generation!==state.generation)return;
      const title=state.title;await activateSession(session,{title});notice("Arbeitsstand wiederhergestellt.");
    } else {
      await refreshTasks();notice("Inferenz abgeschlossen.");text("inference-state",`${state.tasks.length} Ergebnisspuren · Inferenz abgeschlossen`);
    }
  }
  async function poll() {
    if(epoch!==state.jobEpoch||terminal)return;
    try {await update(await api(path));}
    catch(error){if(epoch===state.jobEpoch){if(error.code==="JOB_NOT_FOUND"){terminal=true;stopWatchingJob();state.job=null;rememberSession();show("job-panel",false);$("export-start").disabled=false;$("infer-open").disabled=false;}handleError(error);}}
    if(epoch===state.jobEpoch&&!terminal)state.jobTimer=setTimeout(poll,2000);
  }
  if(window.EventSource) {
    const stream=new EventSource(path+"/events");state.jobStream=stream;
    stream.addEventListener("job_update",event=>{let snapshot;try{snapshot=JSON.parse(event.data);}catch{stream.close();poll();return;}update(snapshot).catch(handleError);});
    stream.addEventListener("job_deleted",()=>{stream.close();stopWatchingJob();state.job=null;show("job-panel",false);$("export-start").disabled=false;$("infer-open").disabled=false;rememberSession();notice("Der Job ist nicht mehr verfügbar.",true);});
    stream.onerror=()=>{stream.close();if(epoch===state.jobEpoch&&!terminal){state.jobStream=null;poll();}};
  } else poll();
}
async function startInference(models) {
  if(!state.session||!models.length)return;
  const id=state.session.id,generation=state.generation;
  try {
    const result=await submitCompatibleModels(models,
      selected=>api(`${sessionPath(id)}/inference`,{method:"POST",json:{models:selected}}),
      issue=>{state.inputIssues.set(issue.modelId,issue);renderInputIssues();},
      ()=>generation===state.generation);
    if(result.cancelled)return;
    if(result.accepted) {
      watchJob({id:result.accepted.job_id,type:"inference"});
      if(result.skipped.length)notice(`Inferenz startet für ${result.models.length} Modell${result.models.length===1?"":"e"}. ${result.skipped.length} Modell${result.skipped.length===1?" wurde":"e wurden"} wegen fehlender Kanalzuordnung übersprungen. Details unter „Kanalzuordnung“.`,true);
    } else {
      text("inference-state","Keine passende Kanalzuordnung für die ausgewählten Modelle");
      notice("Keines der ausgewählten Modelle konnte gestartet werden. Die benötigten Eingänge und vorhandenen EDF-Kanäle stehen unter „Kanalzuordnung“. Die Signalansicht bleibt verfügbar.",true);
    }
  } catch(error) {
    if(error.code==="JOB_ALREADY_RUNNING"&&error.details?.job_id)watchJob({id:error.details.job_id,type:"inference"});
    throw error;
  }
}
function renderInputIssues() {
  show("model-input-report",state.inputIssues.size>0);
  const container=$("model-input-issues");container.replaceChildren();
  for(const issue of state.inputIssues.values()) {
    const model=state.models.find(item=>item.id===issue.modelId),row=element("div",undefined,"model-input-issue");
    row.append(element("strong",model?`${taskName(model)} · ${model.name}`:issue.modelId),element("p",`Keine Zuordnung für: ${issue.channels.join(", ")}`,"small"));
    container.append(row);
  }
  text("model-input-signals",(state.session?.signals||[]).map(signal=>signal.label).join(", "));
}
function openInferenceDialog() {
  const container=$("inference-models");container.replaceChildren();const existing=new Set(state.tasks.map(t=>t.id)),seen=new Set();
  const available=element("details",undefined,"available-signals"),summary=element("summary","Vorhandene EDF-Kanäle");available.append(summary,element("p",(state.session.signals||[]).map(signal=>signal.label).join(", "),"small mono"));container.append(available);
  const updateButton=()=>{$("inference-start").disabled=!container.querySelector("input:checked:not(:disabled)")||Boolean(state.job);};
  for(const model of state.models) {
    if(existing.has(model.task))continue;
    const row=element("div",undefined,"channel-row inference-model-row"),label=element("label"),input=element("input"),issue=state.inputIssues.get(model.id),description=element("span",`${taskName(model)} · ${model.name}`);
    input.type="checkbox";input.value=model.id;input.dataset.task=model.task;input.disabled=Boolean(issue);input.checked=!issue&&!seen.has(model.task);if(!issue)seen.add(model.task);
    description.append(element("small",`Benötigte Eingänge: ${model.input_channels.join(", ")}`));
    if(issue)description.append(element("small",`Nicht zugeordnet: ${issue.channels.join(", ")}`,"input-issue-text"));
    label.append(input,description);row.append(label);container.append(row);
    input.addEventListener("change",()=>{if(input.checked)for(const other of container.querySelectorAll("input"))if(other!==input&&other.dataset.task===input.dataset.task)other.checked=false;updateButton();});
  }
  if(!container.querySelector("input"))container.append(element("p","Für alle verfügbaren Aufgaben liegen bereits Ergebnisse vor, oder auf diesem Server sind keine Modelle eingerichtet.","muted"));
  updateButton();$("inference-dialog").showModal();
}
async function importWorkspace(file) {
  if(!file||state.busy)return;state.busy=true;updateOpenButtons();
  try {
    const accepted=await upload("/api/v1/sessions/import",file,"bundle");
    const session=await api(sessionPath(accepted.session_id));
    await activateSession(session,{title:file.name.replace(/\.sleepwalker\.zip$/i,"")});
    watchJob({id:accepted.job_id,type:"import"});
  } finally {state.busy=false;show("upload-progress",false);updateOpenButtons();}
}
function configureExport() {
  const workspace=$("export-mode").value==="workspace";
  for(const input of $("export-options").querySelectorAll("input")) {input.disabled=workspace;if(workspace)input.checked=true;}
  text("export-note",workspace?"Enthält die EDF, alle Vorhersagen, Wahrscheinlichkeiten, Metadaten und Änderungen. Kann später wieder geöffnet werden.":"Ergebnisdatei ohne EDF. Wird auf dem Server erstellt und anschließend heruntergeladen.");
}
function openExport(workspace=false) {
  $("export-mode").value=workspace?"workspace":"results";configureExport();show("export-download",Boolean(state.export?.ready));$("export-start").disabled=Boolean(state.job);$("export-dialog").showModal();
}
async function startExport() {
  if(!state.session||state.job)return;
  $("export-start").disabled=true;show("export-download",false);
  try {
    const mode=$("export-mode").value;
    const include=Object.fromEntries([...$("export-options").querySelectorAll("input")].map(input=>[input.value,input.checked]));
    const accepted=await api(`${sessionPath(state.session.id)}/exports`,{method:"POST",json:{format:mode==="workspace"?"sleepwalker":"json",mode,include}});
    state.export={id:accepted.export_id,job:accepted.job_id,ready:false};watchJob({id:accepted.job_id,type:"export"});text("export-note","Export läuft. Du kannst den Dialog schließen und weiter sichten.");
  } catch(error){$("export-start").disabled=false;throw error;}
}
async function finishExport() {
  if(!state.export)return;const id=state.export.id,generation=state.generation;
  const result=await api(`/api/v1/exports/${encodeURIComponent(id)}`);
  if(generation!==state.generation)return;
  if(result.state!=="ready")throw new Error("Der Export ist noch nicht zum Download bereit.");
  state.export.ready=true;rememberSession();const link=$("export-download");link.href=`/api/v1/exports/${encodeURIComponent(id)}/download`;link.download=result.mode==="workspace"?"arbeitsstand.sleepwalker.zip":"sleepwalker-ergebnisse.json";
  show("export-download");text("export-note","Export fertig. Die Datei wird heruntergeladen; bei Bedarf erneut herunterladen.");link.click();notice("Export fertig. Download gestartet.");
}
function closeView(expired=false) {
  closeSelectionMenu();state.selection=null;state.interval=null;state.correction=null;
  if($("correction-dialog").open)$("correction-dialog").close();
  sendPlot("close");stopWatchingJob();state.predictionAbort?.abort();clearTimeout(state.predictionTimer);state.generation++;
  state.session=null;state.tasks=[];state.predictions=new Map();state.annotations={edits:[],labels:[],events:[]};state.job=null;state.export=null;
  setViewer(false);text("study-title","Aufzeichnung öffnen");text("study-meta","");show("job-panel",false);
  for(const dialog of document.querySelectorAll("dialog[open]"))dialog.close();
  renderResume();if(expired)show("resume-panel",false);
}
async function deleteSession() {
  if(!state.session)return;$("close-delete").disabled=true;
  try {await api(sessionPath(state.session.id),{method:"DELETE"});writeStorage(sessionStorage,"sleepwalker.session",null);closeView();notice("Session gelöscht.");}
  finally {$("close-delete").disabled=false;}
}
function renderChannels() {
  const container=$("channel-list");container.replaceChildren();
  for(const [index,item] of state.channelDraft.entries()) {
    const row=element("div",undefined,"channel-row"),label=element("label"),input=element("input"),name=element("span",item.signal.label);
    name.append(element("small",`  ${item.signal.unit} · ${item.signal.sample_rate} Hz`));input.type="checkbox";input.checked=item.checked;input.addEventListener("change",()=>item.checked=input.checked);label.append(input,name);row.append(label);
    for(const [delta,title,caption] of [[-1,"Kanal nach oben","↑"],[1,"Kanal nach unten","↓"]]) {
      const button=element("button",caption);button.title=title;button.setAttribute("aria-label",`${item.signal.label}: ${title}`);button.disabled=index+delta<0||index+delta>=state.channelDraft.length;
      button.addEventListener("click",()=>{const temp=state.channelDraft[index+delta];state.channelDraft[index+delta]=item;state.channelDraft[index]=temp;renderChannels();});row.append(button);
    }
    container.append(row);
  }
}

window.addEventListener("message",event=>{
  if(event.origin!==location.origin||event.source!==$("signal-frame").contentWindow||event.data?.source!=="sleepwalker-plot")return;
  const data=event.data;
  if(data.type==="ready") {if(!state.frameReady){state.frameReady=true;requestPlot();}return;}
  if(data.type==="dismiss"){closeSelectionMenu();return;}
  if(!state.session)return;
  if(data.type==="geometry") {if(Number.isFinite(data.left)&&Number.isFinite(data.right))timeline.geometry(data.left,data.right);return;}
  if(data.type==="status") {
    const status=data.status;if(status.session!==state.session.id||status.view!==state.view)return;
    if(Number.isInteger(status.revision)&&status.revision<state.revision)return;
    if(status.error){show("plot-loading",false);handleError(Object.assign(new Error(status.error),status.expired?{code:"SESSION_NOT_FOUND"}:{}));return;}
    if(status.range){state.plotLoaded=true;show("plot-loading",false);$("signal-frame").style.height=`${status.height}px`;setRange(status.range[0],status.range[1],true);}
  } else if(data.type==="range"&&data.session===state.session.id&&data.view===state.view&&Array.isArray(data.range)) {
    setRange(data.range[0],data.range[1]);
  } else if(data.type==="interval"&&data.session===state.session.id&&data.view===state.view&&Array.isArray(data.range)) {
    selectRegion({kind:"interval",task:state.taskId,start:data.range[0],end:data.range[1]});
  } else if(data.type==="context"&&data.session===state.session.id&&data.view===state.view) {
    const rect=$("signal-frame").getBoundingClientRect();openSelectionMenu({time:data.time,x:rect.left+data.x,y:rect.top+data.y});
  } else if(data.type==="selection"&&data.session===state.session.id&&data.view===state.view)selectTime(data.time);
  else if(data.type==="error")notice(data.message,true);
});

bind("connect",connect);bind("notice-dismiss",()=>show("notice",false));bind("open-infer",()=>openEDF(true));bind("open-signals",()=>openEDF(false));
$("server-profile").addEventListener("change",()=>$("server-address").value=$("server-profile").value==="current"?location.origin:$("server-profile").value);
$("server-address").value=location.origin;
$("server-address").addEventListener("keydown",event=>{if(event.key==="Enter"){event.preventDefault();connect().catch(handleError);}});
$("edf-file").addEventListener("change",()=>setFile($("edf-file").files[0]));
$("workspace-file").addEventListener("change",()=>importWorkspace($("workspace-file").files[0]).catch(handleError));
$("drop-zone").addEventListener("dragover",event=>{event.preventDefault();$("drop-zone").classList.add("dragging");});
$("drop-zone").addEventListener("dragleave",()=>$("drop-zone").classList.remove("dragging"));
$("drop-zone").addEventListener("drop",event=>{event.preventDefault();$("drop-zone").classList.remove("dragging");if(!state.busy)setFile(event.dataTransfer.files[0]);});
bind("resume",async()=>{
  const saved=readStorage(sessionStorage,"sleepwalker.session",null);if(!saved?.id)return;
  try {const session=await api(sessionPath(saved.id));await activateSession(session,{saved});if(saved.job)watchJob(saved.job);else if(saved.export?.ready){state.export=saved.export;$("export-download").href=`/api/v1/exports/${encodeURIComponent(saved.export.id)}/download`;}}
  catch(error){if(error.code==="SESSION_NOT_FOUND"){writeStorage(sessionStorage,"sleepwalker.session",null);renderResume();}throw error;}
});
bind("forget-resume",()=>{writeStorage(sessionStorage,"sleepwalker.session",null);renderResume();});
bind("previous",()=>navigate(-1));bind("next",()=>navigate(1));
$("window-size").addEventListener("change",()=>{const width=$("window-size").value==="all"?state.session.duration:Number($("window-size").value);setRange(state.range[0],state.range[0]+width);});
$("goto-form").addEventListener("submit",event=>{event.preventDefault();const value=$("goto-time").value;if(!/^\d+:[0-5]\d:[0-5]\d(?:\.\d+)?$/.test(value)){notice("Zeit als HH:MM:SS eingeben.",true);return;}const [h,m,s]=value.split(":").map(Number);const time=h*3600+m*60+s;if(time>=state.session.duration){notice("Der Zeitpunkt liegt außerhalb der Aufnahme.",true);return;}const width=state.range[1]-state.range[0];setRange(time,time+width);selectTime(time);});
bind("time-mode",()=>{state.absolute=!state.absolute;text("time-mode",state.absolute?"Aufnahmeuhrzeit":"Zeit seit Aufnahmebeginn");renderRange();renderContext();drawTimeline();});
$("task-select").addEventListener("change",()=>selectTime(state.time,$("task-select").value));
bind("refresh-annotations",refreshAnnotations);
bind("select-tool",()=>setInteractionMode("select"));bind("pan-tool",()=>setInteractionMode("pan"));
bind("selection-edit",openCorrection);bind("context-edit",openCorrection);bind("context-delete",deleteSelectedCorrection);
bind("correction-close",closeCorrection);bind("correction-cancel",closeCorrection);
$("correction-form").addEventListener("submit",event=>{event.preventDefault();applyCorrection().catch(handleError);});
$("correction-task").addEventListener("change",()=>{updateCorrectionClasses();updateCorrectionPreview();});
for(const id of ["correction-start","correction-end"])$(id).addEventListener("input",updateCorrectionPreview);
$("correction-dialog").addEventListener("cancel",event=>{if(state.mutation)event.preventDefault();else state.correction=null;});
document.addEventListener("pointerdown",event=>{if(!event.target.closest("#selection-menu"))closeSelectionMenu();});
window.addEventListener("resize",closeSelectionMenu);window.addEventListener("scroll",closeSelectionMenu,true);
document.addEventListener("keydown",event=>{
  const menu=$("selection-menu");if(menu.hidden)return;
  if(event.key==="Escape"){event.preventDefault();closeSelectionMenu();$("timeline").focus();}
  if(["ArrowUp","ArrowDown","Home","End"].includes(event.key)){
    event.preventDefault();const items=[...menu.querySelectorAll('button:not(:disabled):not([hidden])')];
    const index=items.indexOf(document.activeElement),next=event.key==="Home"?0:event.key==="End"?items.length-1:(index+(event.key==="ArrowUp"?-1:1)+items.length)%items.length;items[next]?.focus();
  }
});
$("timeline").addEventListener("keydown",event=>{
  if(state.selection&&(event.key==="ContextMenu"||(event.shiftKey&&event.key==="F10"))){event.preventDefault();const rect=$("timeline").getBoundingClientRect();openSelectionMenu({time:state.time,x:rect.left+126,y:rect.top+20});}
});
bind("channels-open",()=>{
  const signals=state.session.signals||[];const order=[...state.channels,...signals.map(s=>s.id).filter(id=>!state.channels.includes(id))];
  state.channelDraft=order.map(id=>({signal:signals.find(s=>s.id===id),checked:state.channels.includes(id)}));renderChannels();$("channels-dialog").showModal();
});
bind("channels-apply",()=>{const channels=state.channelDraft.filter(s=>s.checked).map(s=>s.signal.id);if(!channels.length)throw new Error("Mindestens einen Kanal auswählen.");state.channels=channels;state.plotLoaded=false;$("channels-dialog").close();show("plot-loading");rememberSession();requestPlot();});
bind("infer-open",openInferenceDialog);
bind("inference-start",async()=>{const models=[...$("inference-models").querySelectorAll("input:checked")].map(input=>input.value);if(!models.length)throw new Error("Mindestens ein Modell auswählen.");$("inference-dialog").close();await startInference(models);});
bind("cancel-job",async()=>{if(state.job){await api(`/api/v1/jobs/${encodeURIComponent(state.job.id)}/cancel`,{method:"POST"});$("cancel-job").disabled=true;}});
bind("export-open",()=>openExport());bind("workspace-export",()=>openExport(true));bind("export-start",startExport);
$("export-mode").addEventListener("change",configureExport);
for(const id of ["close-open","delete-open"])bind(id,()=>$("close-dialog").showModal());
bind("close-save",()=>{$("close-dialog").close();openExport(true);});
bind("close-keep",()=>{rememberSession();closeView();});bind("close-delete",deleteSession);
bind("nav-file",()=>{if(state.session)$("close-dialog").showModal();});
bind("nav-view",()=>{$("viewer").scrollIntoView({behavior:"auto"});});
bind("nav-history",()=>{ $("viewer").classList.remove("inspector-hidden");$("inspector-toggle").setAttribute("aria-expanded","true");$("history-section").scrollIntoView({behavior:"auto",block:"center"}); });
bind("nav-help",()=>$("help-dialog").showModal());
bind("inspector-toggle",()=>{const hidden=$("viewer").classList.toggle("inspector-hidden");$("inspector-toggle").setAttribute("aria-expanded",String(!hidden));});
document.addEventListener("keydown",event=>{if(!state.session||document.querySelector("dialog[open]")||event.target.closest("input,select,textarea,button,summary"))return;if(event.key==="ArrowLeft"||event.key==="ArrowRight"){event.preventDefault();navigate(event.key==="ArrowLeft"?-1:1);}});
window.addEventListener("pagehide",()=>{rememberSession();sendPlot("close");state.jobStream?.close();});
setInterval(()=>{if(state.session&&document.visibilityState==="visible")requestPlot("heartbeat");},45000);
document.addEventListener("visibilitychange",()=>{if(state.session&&document.visibilityState==="visible"){requestPlot();refreshAnnotations().catch(handleError);}});
renderProfiles();updateOpenButtons();renderResume();connect().catch(handleError);
