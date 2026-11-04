import {isNegativeClass, labelColor, labelTextColor} from "./colors.js?v=20261026-theme-1";
export {labelColor};
export function currentPrediction(task, predictions, time) {
  if (!task) return null;
  if (task.kind === "continuous") {
    let found = null;
    for (const item of predictions) if (item.time <= time && time < item.time + task.resolution && (!found || item.time > found.time)) found = item;
    if (!found) return null;
    const label = Object.entries(found.probabilities).sort((a,b) => b[1] - a[1])[0]?.[0];
    return {...found, label};
  }
  return predictions.find(p => p.start <= time && time < p.end) || null;
}
export function formatTime(seconds, fractional = false) {
  // Round to tenths first, so 82.97 s reads 00:01:23.0 and not 00:01:22.0.
  const t = fractional ? Math.round(Math.max(0, Number(seconds) || 0) * 10) / 10 : Math.max(0, Number(seconds) || 0);
  const hours = Math.floor(t / 3600), minutes = Math.floor(t / 60) % 60, secs = Math.floor(t) % 60;
  return `${String(hours).padStart(2,"0")}:${String(minutes).padStart(2,"0")}:${String(secs).padStart(2,"0")}${fractional && t % 1 > .001 ? (t % 1).toFixed(1).slice(1) : ""}`;
}

export class Timeline {
  // preview(range) moves the view while a gesture runs; commit(range) settles it, like the signal plot.
  constructor(canvas, {select, interval, toggle, context, preview, commit}) {
    this.canvas = canvas; this.select = select; this.interval = interval; this.toggle = toggle; this.context = context;
    this.preview = preview; this.commit = commit;
    this.left = 126; this.right = 22; this.rows = []; this.data = null; this.drag = null; this.wheelTarget = null; this.wheelTimer = null;
    new ResizeObserver(() => this.draw()).observe(canvas.parentElement);
    canvas.addEventListener("pointerdown", event => {
      if (!this.data || event.button !== 0) return;
      const p = this.position(event), row = this.rows.find(r => p.y >= r.y && p.y <= r.y + r.height);
      if (!row) return;
      if (p.x < this.left) { this.toggle(row.task.id); return; }
      this.drag = {x: p.x, current:p.x, row, pan:this.data.mode === "pan", range:[...this.data.range], moved:false};
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener("pointermove", event => {
      const p = this.position(event), row = this.rows.find(r => p.y >= r.y && p.y <= r.y + r.height);
      if (this.drag?.pan) {
        this.drag.current = p.x; this.drag.moved ||= Math.abs(p.x - this.drag.x) > 2;
        if (this.drag.moved) this.preview(this.panned(p.x));
        return;
      }
      if (this.drag) { this.drag.current = p.x; this.draw(); }
      if (!row || p.x < this.left || !this.data) return;
      const time = this.timeAt(p.x);
      const selected = this.hit(row,time);
      canvas.title = `${this.data.taskName(row.task)} · ${this.data.timeLabel(time)} · ${selected ? this.data.className(selected.label) + (["label","event"].includes(selected.kind)?this.data.t(" · manuell"):"") : this.data.t("Keine Vorhersage")}`;
    });
    canvas.addEventListener("pointerup", event => {
      if (!this.drag) return;
      const {x, row, pan, moved} = this.drag, p = this.position(event);
      if (pan && moved) { const range = this.panned(p.x); this.drag = null; this.commit(range); return; }
      this.drag = null;
      const start = this.timeAt(x), end = this.timeAt(p.x);
      if (!pan && Math.abs(p.x - x) > 5) this.interval(row.task.id, Math.min(start,end), Math.max(start,end));
      else this.select(row.task.id, end, this.hit(row,end));
      this.draw();
    });
    canvas.addEventListener("pointercancel", () => {
      const drag = this.drag; this.drag = null;
      if (drag?.pan && drag.moved) this.commit(this.panned(drag.current, drag)); else this.draw();
    });
    // Same gestures as the signal plot: wheel or pinch zooms around the pointer, horizontal scroll pans.
    canvas.addEventListener("wheel", event => {
      if (!this.data || !this.width) return;
      const p = this.position(event);
      if (p.x < this.left || p.x > this.width - this.right) return;
      event.preventDefault();
      const pair = this.wheelTarget || this.data.range, plotWidth = this.width - this.left - this.right;
      if (!event.ctrlKey && (event.shiftKey || Math.abs(event.deltaX) > Math.abs(event.deltaY))) {
        const delta = (event.shiftKey ? event.deltaY : event.deltaX) * (pair[1] - pair[0]) / plotWidth;
        this.wheelTarget = [pair[0] + delta, pair[1] + delta];
      } else {
        const units = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? this.canvas.clientHeight : 1;
        const factor = Math.exp(Math.max(-1.5, Math.min(1.5, event.deltaY * units * .002)));
        const anchor = Math.max(0, Math.min(1, (p.x - this.left) / plotWidth));
        const width = (pair[1] - pair[0]) * factor, start = pair[0] + (pair[1] - pair[0] - width) * anchor;
        this.wheelTarget = [start, start + width];
      }
      this.wheelTarget = this.preview(this.wheelTarget);
      clearTimeout(this.wheelTimer);
      this.wheelTimer = setTimeout(() => { const range = this.wheelTarget; this.wheelTarget = null; this.commit(range); }, 90);
    }, {passive:false});
    canvas.addEventListener("contextmenu", event => {
      if (!this.data) return;
      const p = this.position(event), row = this.rows.find(r => p.y >= r.y && p.y <= r.y + r.height);
      if (!row || p.x < this.left || p.x > this.width - this.right) return;
      event.preventDefault();
      const time = this.timeAt(p.x);
      this.context({task:row.task.id, time, target:this.hit(row,time), x:event.clientX, y:event.clientY});
    });
    canvas.addEventListener("keydown", event => {
      if (!this.data || !["ArrowUp","ArrowDown"].includes(event.key)) return;
      event.preventDefault();
      const index = this.data.tasks.findIndex(t => t.id === this.data.taskId);
      const next = Math.max(0, Math.min(this.data.tasks.length - 1, index + (event.key === "ArrowUp" ? -1 : 1)));
      if (this.data.tasks[next]) this.select(this.data.tasks[next].id, this.data.time);
    });
  }
  panned(x, drag = this.drag) {
    const [start, end] = drag.range, seconds = (x - drag.x) * (end - start) / (this.width - this.left - this.right);
    return [start - seconds, end - seconds];
  }
  position(event) { const rect = this.canvas.getBoundingClientRect(); return {x:event.clientX - rect.left,y:event.clientY - rect.top}; }
  hit(row,time) { return selectionAt(row.task,this.data.predictions.get(row.task.id)||[],this.data.annotations,time,row.klass); }
  timeAt(x) { const [start,end] = this.data.range; return Math.min(end - 1e-6, Math.max(start,start + (x - this.left) / (this.width - this.left - this.right) * (end - start))); }
  update(data) { this.data = data; this.draw(); }
  geometry(left, right) { this.left = left; this.right = right; this.draw(); }
  draw() {
    if (!this.data) return;
    const {tasks, predictions, annotations, expanded, range, time, taskId, selection, timeLabel, taskName, className, t} = this.data;
    this.width = this.canvas.clientWidth;
    if (!this.width) return;
    this.rows = []; let y = 8;
    for (const task of tasks) {
      const classes = expanded.has(task.id) ? task.classes : [null];
      for (const klass of classes) { this.rows.push({task, klass, y, height:32}); y += 35; }
    }
    const height = Math.max(92, y + 48), ratio = window.devicePixelRatio || 1;
    this.canvas.style.height = `${height}px`; this.canvas.width = Math.round(this.width * ratio); this.canvas.height = Math.round(height * ratio);
    const c = this.canvas.getContext("2d"); c.scale(ratio,ratio); c.clearRect(0,0,this.width,height);
    c.font = "11px Arial, sans-serif"; c.textBaseline = "middle";
    const plotWidth = this.width - this.left - this.right;
    const xx = t => this.left + (t - range[0]) / (range[1] - range[0]) * plotWidth;
    const style=getComputedStyle(this.canvas);
    const ink=style.getPropertyValue("--ink").trim()||"#262a2c",muted=style.getPropertyValue("--muted").trim()||"#6e7476";
    const secondary=style.getPropertyValue("--ink-secondary").trim()||"#626a6c",line=style.getPropertyValue("--line").trim()||"#d7d9d8";
    const track=style.getPropertyValue("--track-line").trim()||"#c9cdcb",selectionColor=style.getPropertyValue("--selection").trim()||"#161a1c";
    const cursorColor=style.getPropertyValue("--cursor").trim()||"#687074";
    const segment=(label,start,end,y,manual=false)=>{
      const x=xx(Math.max(range[0],start)),w=xx(Math.min(range[1],end))-x;
      if(w<=0)return;
      const negative=isNegativeClass(label);
      if(manual)c.clearRect(x,y+5,w,22);
      if(!negative){c.fillStyle=labelColor(label);c.fillRect(x,y+7,Math.max(.5,w),16);}
      else if(manual){c.strokeStyle=track;c.lineWidth=1;c.beginPath();c.moveTo(x,y+15.5);c.lineTo(x+w,y+15.5);c.stroke();}
      c.fillStyle=negative?ink:labelTextColor(label);
      const caption=className(label);
      if(!negative&&w>c.measureText(caption).width+(manual?28:10)){c.textAlign="center";c.fillText(caption,x+w/2,y+15);c.textAlign="left";}
      if(manual){
        if(w>=18){c.save();c.font="bold 9px Arial, sans-serif";c.fillText("M",x+3,y+15);c.restore();}
        else {c.fillStyle=selectionColor;c.beginPath();c.moveTo(x,y+5);c.lineTo(x+Math.min(5,w),y+5);c.lineTo(x,y+10);c.closePath();c.fill();}
      }
    };
    if (!tasks.length) { c.fillStyle = muted; c.fillText(t("Noch keine Vorhersagen. Modelle können über ‘Modelle’ gestartet werden."),16,27,Math.max(100,this.width-32)); return; }
    for (const row of this.rows) {
      const {task, klass, y} = row;
      const items = predictions.get(task.id) || [];
      const label = klass ? `  ${className(klass)}` : taskName(task);
      c.save(); c.beginPath(); c.rect(10,y,this.left - 18,32); c.clip();
      c.fillStyle = task.id === taskId ? ink : secondary;
      c.fillText((klass ? "" : expanded.has(task.id) ? "▾ " : "▸ ") + label,12,y+15); c.restore();
      const baseline=y+(task.kind==="continuous"?25.5:15.5);
      c.strokeStyle=track;c.lineWidth=1;c.beginPath();c.moveTo(this.left,baseline);c.lineTo(this.width-this.right,baseline);c.stroke();
      c.save(); c.beginPath(); c.rect(this.left,y,plotWidth,32); c.clip();
      const visible = items.some(p => (p.end ?? p.time + task.resolution) > range[0] && (p.start ?? p.time) < range[1]);
      if (!klass && !visible && task.kind !== "event" && !(task.kind === "continuous" && range[1]-range[0] > 900)) {
        c.fillStyle = muted; c.fillText(t("Keine Vorhersage in diesem Ausschnitt"), this.left + 8, y + 15);
      }
      if (task.kind === "continuous") {
        const labels = klass ? [klass] : task.classes.filter(label=>!isNegativeClass(label));
        for (const cls of labels) {
          c.strokeStyle = labelColor(cls); c.lineWidth=isNegativeClass(cls)?1:1.8;
          c.beginPath(); let prev=null;
          for (const point of items) {
            const prob=point.probabilities[cls]; if (!Number.isFinite(prob)) continue;
            const x=xx(point.time), py=y+25-prob*20;
            if (prev === null || point.time-prev > task.resolution*1.5) c.moveTo(x,py); else c.lineTo(x,py);
            prev=point.time;
          }
          c.stroke();
        }
        if (range[1]-range[0] > 900) { c.fillStyle=muted; c.fillText(t("Für kontinuierliche Werte auf höchstens 15 min zoomen"),this.left+8,y+15); }
      } else {
        for (const p of items) {
          if (annotations.labels.some(e => e.task===task.id && Math.abs(e.start-p.start)<1e-6)) continue;
          if (klass && p.label !== klass) continue;
          segment(p.label,p.start,p.end,y);
        }
      }
      for (const edit of annotations.labels.filter(e => e.task===task.id)) {
        const end=edit.start+task.resolution;
        if (end<=range[0] || edit.start>=range[1] || (klass && edit.label!==klass)) continue;
        segment(edit.label,edit.start,end,y,true);
      }
      for (const event of annotations.events.filter(e => e.task===task.id)) {
        if (event.end<=range[0] || event.start>=range[1] || (klass && event.label!==klass)) continue;
        segment(event.label,event.start,event.end,y,true);
      }
      const marked = this.drag && this.drag.row.task.id === task.id && Math.abs(this.drag.current-this.drag.x)>5
        ? {start:this.timeAt(Math.min(this.drag.x,this.drag.current)), end:this.timeAt(Math.max(this.drag.x,this.drag.current)),kind:"interval"}
        : selection?.task===task.id ? selection : null;
      if (marked && (!klass || !marked.label || marked.label===klass)) {
        const x=xx(Math.max(range[0],marked.start)), w=xx(Math.min(range[1],marked.end))-x;
        if(w>0) {c.strokeStyle=selectionColor;c.lineWidth=2;c.setLineDash(marked.kind==="interval"?[4,2]:[]);c.strokeRect(x+1,y+2,Math.max(1,w-2),27);c.setLineDash([]);}
      }
      c.restore();
    }
    if (time>=range[0] && time<=range[1]) {
      c.strokeStyle=cursorColor; c.setLineDash([3,3]); c.beginPath(); c.moveTo(xx(time),0); c.lineTo(xx(time),y+2); c.stroke(); c.setLineDash([]);
    }
    c.strokeStyle=line; c.beginPath(); c.moveTo(this.left,y+9); c.lineTo(this.width-this.right,y+9); c.stroke();
    c.font="10px monospace"; c.fillStyle=muted;
    const ticks=plotWidth<450 ? 2 : 4;
    for(let i=0;i<=ticks;i++) { const t=range[0]+(range[1]-range[0])*i/ticks; c.textAlign=i===0?"left":i===ticks?"right":"center"; c.fillText(timeLabel(t),xx(t),y+28); }
    c.textAlign="left";
  }
}
import {selectionAt} from "./corrections.js?v=20261026-theme-1";

