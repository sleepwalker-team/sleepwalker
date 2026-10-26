/* Same-origin bridge between the plain UI and the server-backed Dash graph. */
(() => {
  "use strict";
  // Keep the controller in this asset so running GUIs need only a browser reload.
  const createSignalMotion = ({graph, current, notify, geometry}) => {
    const reduced = matchMedia("(prefers-reduced-motion: reduce)");
    let gesture = null, frame = 0, animation = null, wanted = null, busy = false;
    let active = false, finishing = false, lastRange = null, wheelTarget = null, edgeFrame = 0, wheelTimer = null;
    const edge = document.createElement("div");
    edge.className = "signal-edge";
    edge.hidden = true;
    const clamp = (n, lo, hi) => Math.max(lo, Math.min(n, hi));
    const range = () => graph()?._fullLayout?.xaxis?.range?.slice();
    const bounded = pair => {
      const duration = current()?.duration;
      const width = clamp(pair[1] - pair[0], Math.min(.05, duration), duration);
      const start = clamp(pair[0], 0, duration - width);
      return [start, start + width];
    };
    function feedback(pixels) {
      const gd = graph(), layout = gd?._fullLayout, axis = layout?.xaxis;
      if (!axis) return;
      if (!edge.isConnected) document.body.append(edge);
      const amount = reduced.matches ? 0 : 1 - Math.exp(-Math.abs(pixels) / 110);
      edge.hidden = amount < .01;
      const rect = gd.getBoundingClientRect();
      const rows = Object.entries(layout).filter(([key]) => /^yaxis\d*$/.test(key)).map(([,value]) => value);
      const top = Math.min(...rows.map(row => row._offset));
      const bottom = Math.max(...rows.map(row => row._offset + row._length));
      Object.assign(edge.style, {left:`${rect.left + axis._offset}px`,top:`${rect.top + top}px`,
        width:`${axis._length}px`,height:`${bottom-top}px`,opacity:String(amount)});
      edge.dataset.side = pixels > 0 ? "left" : "right";
      // Keep waveform, clipping, axes and overlays in the same coordinate system.
      // Only the edge cue fades; never scale the SVG or WebGL signal layers.
    }
    function releaseEdge() {
      cancelAnimationFrame(edgeFrame);
      const start = performance.now(), initial = gesture?.overshoot || 0;
      const tick = now => {
        const t = Math.min(1, (now - start) / 160);
        feedback(initial * Math.pow(1-t, 3));
        if (t < 1 && !reduced.matches) edgeFrame = requestAnimationFrame(tick);
        else feedback(0);
      };
      edgeFrame = requestAnimationFrame(tick);
    }
    async function flush() {
      if (busy || !wanted || !graph() || !window.Plotly) return;
      const pair = wanted; wanted = null; busy = true;
      const update = {};
      for (const key of Object.keys(graph().layout).filter(k => /^xaxis\d*$/.test(k))) update[key + ".range"] = pair;
      lastRange = pair;
      try { await window.Plotly.relayout(graph(), update); }
      finally {
        busy = false; geometry();
        if (wanted) schedule();
        else if (finishing) commit();
      }
    }
    function schedule() {
      if (!frame) frame = requestAnimationFrame(() => { frame = 0; flush(); });
    }
    function show(pair) { active = true; wanted = bounded(pair); schedule(); }
    function commit() {
      if (busy || wanted || animation || gesture) return;
      finishing = false; active = false; wheelTarget = null;
      const pair = range();
      if (pair && current()) notify(pair);
    }
    function finish() { finishing = true; if (!busy && !wanted) commit(); }
    function stop(publish = false) {
      const interrupted = active, visible = range();
      cancelAnimationFrame(frame); cancelAnimationFrame(edgeFrame);
      clearTimeout(wheelTimer); wheelTimer = null;
      if (animation) cancelAnimationFrame(animation.id);
      frame = 0; animation = null; wanted = null; gesture = null;
      finishing = false; active = false; wheelTarget = null; feedback(0);
      if (publish && interrupted && visible && current()) notify(visible);
    }
    function animate(to, duration, initial) {
      if (animation) cancelAnimationFrame(animation.id);
      const from = initial || range(), started = performance.now();
      if (!from) return;
      animation = {id:0};
      const tick = now => {
        const t = reduced.matches ? 1 : Math.min(1, (now - started) / duration);
        const ease = 1 - Math.pow(1 - t, 3);
        show(from.map((value, i) => value + (to[i] - value) * ease));
        if (t < 1) animation.id = requestAnimationFrame(tick);
        else { animation = null; finish(); }
      };
      animation.id = requestAnimationFrame(tick);
    }
    function down(event, mode) {
      stop(true);
      if (event.button !== 0 || mode !== "pan" || !current() || !event.target.closest(".nsewdrag")) return false;
      const pair = range(), size = graph()?._fullLayout?._size;
      if (!pair || !size) return false;
      gesture = {pointer:event.pointerId,x:event.clientX,pair,width:size.w,lastX:event.clientX,
        lastAt:performance.now(),velocity:0,overshoot:0,moved:false};
      active = true;
      event.target.setPointerCapture?.(event.pointerId);
      event.preventDefault(); event.stopImmediatePropagation();
      return true;
    }
    document.addEventListener("pointermove", event => {
      if (!gesture || gesture.pointer !== event.pointerId) return;
      event.preventDefault(); event.stopImmediatePropagation();
      const now = performance.now(), delta = event.clientX - gesture.x;
      const secondsPerPixel = (gesture.pair[1] - gesture.pair[0]) / gesture.width;
      const raw = gesture.pair.map(value => value - delta * secondsPerPixel), pair = bounded(raw);
      gesture.overshoot = (pair[0] - raw[0]) / secondsPerPixel;
      gesture.velocity = .65 * gesture.velocity + .35 * (event.clientX - gesture.lastX) / Math.max(1, now - gesture.lastAt);
      gesture.lastX = event.clientX; gesture.lastAt = now;
      gesture.moved ||= Math.abs(delta) > 2;
      feedback(gesture.overshoot); show(pair);
    }, {capture:true,passive:false});
    function up(event) {
      if (!gesture || gesture.pointer !== event.pointerId) return;
      event.preventDefault(); event.stopImmediatePropagation();
      const previous = gesture;
      releaseEdge(); gesture = null;
      // A short, capped coast only for a recent fast movement inside the recording.
      const velocity = performance.now() - previous.lastAt < 70 ? previous.velocity : 0;
      if (event.type !== "pointercancel" && !reduced.matches && previous.moved && !previous.overshoot && Math.abs(velocity) > .25) {
        const base = wanted || range();
        const offset = clamp(velocity * 65, -65, 65) * (previous.pair[1] - previous.pair[0]) / previous.width;
        // Continue from the final requested drag position.
        wanted = null;
        animate(bounded(base.map(value => value - offset)), 180, base);
      } else finish();
    }
    document.addEventListener("pointerup", up, true);
    document.addEventListener("pointercancel", up, true);
    function wheel(event) {
      if (!current() || !event.target.closest(".nsewdrag")) return;
      event.preventDefault(); event.stopImmediatePropagation();
      if (animation) cancelAnimationFrame(animation.id);
      animation = null; finishing = false;
      cancelAnimationFrame(edgeFrame); feedback(0); gesture = null;
      clearTimeout(wheelTimer);
      const gd = graph(), size = gd?._fullLayout?._size, pair = wheelTarget || wanted || range();
      if (!size || !pair) return;
      // Trackpads pan horizontally; ordinary wheel and pinch zoom around the pointer.
      if (!event.ctrlKey && (event.shiftKey || Math.abs(event.deltaX) > Math.abs(event.deltaY))) {
        const delta = (event.shiftKey ? event.deltaY : event.deltaX) * (pair[1] - pair[0]) / size.w;
        wheelTarget = bounded(pair.map(value => value + delta));
      } else {
        const units = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? size.h : 1;
        const factor = Math.exp(clamp(event.deltaY * units * .002, -1.5, 1.5));
        const anchor = clamp((event.clientX - gd.getBoundingClientRect().left - size.l) / size.w, 0, 1);
        const width = clamp((pair[1] - pair[0]) * factor, Math.min(.05,current().duration), current().duration);
        const start = pair[0] + (pair[1] - pair[0] - width) * anchor;
        wheelTarget = bounded([start,start + width]);
      }
      // Apply the latest wheel target in the next frame, without a trailing
      // easing animation fighting subsequent wheel events or direction changes.
      show(wheelTarget);
      wheelTimer = setTimeout(() => {wheelTimer = null; finish();}, 90);
    }
    document.addEventListener("wheel", wheel, {capture:true,passive:false});
    window.addEventListener("blur", () => { if (gesture) {releaseEdge();gesture=null;finish();} });
    return {down,stop,accept:next => {
      const old = current();
      if (next.session !== old?.session || next.view !== old?.view || !lastRange ||
          !next.range?.every((value,i) => Math.abs(value-lastRange[i]) < 1e-6)) stop();
    },owns:pair => active || (lastRange && pair.every((v,i) => Math.abs(v-lastRange[i]) < 1e-6))};
  };

  let request = null, cursor = null, lastStatus = "", pending = null, submitTimer = null, boundGraph = null, lastSelectionAt = 0;
  const send = (type, extra = {}) => {
    if (window.parent !== window) window.parent.postMessage({source: "sleepwalker-plot", type, ...extra}, window.location.origin);
  };
  const graph = () => document.querySelector("#signals .js-plotly-plot");
  const cursorLine = document.createElement("div");
  cursorLine.className = "signal-cursor";
  cursorLine.hidden = true;
  const intervalBox = document.createElement("div");
  intervalBox.className = "signal-interval";
  intervalBox.hidden = true;
  let theme = "light";
  try { if (localStorage.getItem("sleepwalker.theme") === "dark") theme = "dark"; } catch { /* Storage is optional. */ }
  document.documentElement.dataset.theme = theme;
  let interval = null, mode = "select", applyingLayout = false;
  const motion = createSignalMotion({graph, current:() => request, geometry,
    notify:range => { if (request) send("range", {range,session:request.session,view:request.view}); }});
  function geometry() {
    const gd = graph();
    if (gd?._fullLayout?._size) send("geometry", {left: gd._fullLayout._size.l, right: gd._fullLayout._size.r, width: gd._fullLayout.width});
    drawCursor();
    drawInterval();
    const update = {};
    const colors = theme === "dark" ? {paper:"#20262b",ink:"#d5dfe5",muted:"#a3afb7",grid:"#303a42"} : {paper:"#fcfcfb",ink:"#343c40",muted:"#6e7476",grid:"#e9eae8"};
    const recolor = gd?.layout && (gd.layout.paper_bgcolor !== colors.paper || gd.data?.some(trace => trace.line?.color !== colors.ink));
    if (recolor) {
      update["newselection.line.color"] = theme === "dark" ? "#edf3f5" : "#161a1c";
      if (theme === "dark") {
        update["hoverlabel.bgcolor"] = colors.paper; update["hoverlabel.bordercolor"] = "#5b6871"; update["hoverlabel.font.color"] = colors.ink;
      } else {
        update["hoverlabel.bgcolor"] = null; update["hoverlabel.bordercolor"] = null; update["hoverlabel.font.color"] = null;
      }
      update.paper_bgcolor = colors.paper; update.plot_bgcolor = colors.paper; update["font.color"] = colors.ink;
      for (const axis of Object.keys(gd.layout).filter(key => /^[xy]axis\d*$/.test(key))) {
        update[axis+".gridcolor"] = theme === "light" && axis.startsWith("y") ? "#eff0ed" : colors.grid;
      }
      for (const [index, annotation] of (gd.layout.annotations||[]).entries()) {
        update[`annotations[${index}].font.color`] = annotation.font?.size === 10 ? colors.muted : (theme === "light" ? "#262a2c" : colors.ink);
      }
    }
    if (gd?.layout && gd.layout.dragmode !== mode) {update.dragmode=mode;update.selectdirection="h";}
    // Native Plotly limits stop the gesture itself, before a resampler round trip.
    // Apply on every subplot and after channel/session changes. This also upgrades
    // already-running UI instances whose initial Python figure has no limits yet.
    if (gd?.layout && Number.isFinite(request?.duration) && request.duration>0) {
      for (const axis of Object.keys(gd.layout).filter(key=>/^xaxis\d*$/.test(key))) {
        if (gd.layout[axis].minallowed!==0) update[axis+".minallowed"]=0;
        if (gd.layout[axis].maxallowed!==request.duration) update[axis+".maxallowed"]=request.duration;
      }
    }
    if (gd && Object.keys(update).length && !applyingLayout && window.Plotly) {
      applyingLayout = true;
      (recolor ? window.Plotly.update(gd, {"line.color":colors.ink}, update) : window.Plotly.relayout(gd, update)).finally(() => { applyingLayout = false; geometry(); });
    }
  }
  function drawInterval() {
    const gd = graph(), layout = gd?._fullLayout, axis = layout?.xaxis;
    if (!axis || !interval || interval[1] <= axis.range[0] || interval[0] >= axis.range[1]) {intervalBox.hidden = true;return;}
    if (!intervalBox.isConnected) document.body.append(intervalBox);
    const rect = gd.getBoundingClientRect(), start = Math.max(interval[0],axis.range[0]), end = Math.min(interval[1],axis.range[1]);
    intervalBox.hidden = false;
    intervalBox.style.left = `${rect.left + layout._size.l + axis.l2p(start)}px`;
    intervalBox.style.top = `${rect.top + layout._size.t}px`;
    intervalBox.style.width = `${axis.l2p(end) - axis.l2p(start)}px`;
    intervalBox.style.height = `${layout._size.h}px`;
  }
  function drawCursor() {
    const gd = graph();
    const layout = gd?._fullLayout, axis = layout?.xaxis;
    if (!axis || !layout._size || cursor === null) { cursorLine.hidden = true; return; }
    // A cursor must not start another Plotly.relayout while Dash is replacing
    // resampled data. This overlay never changes the figure or emits navigation.
    if (!cursorLine.isConnected) document.body.append(cursorLine);
    const bounds = axis.range;
    cursorLine.hidden = cursor < bounds[0] || cursor > bounds[1];
    const rect = gd.getBoundingClientRect();
    cursorLine.style.left = `${rect.left + layout._size.l + axis.l2p(cursor)}px`;
    cursorLine.style.top = `${rect.top + layout._size.t}px`;
    cursorLine.style.height = `${layout._size.h}px`;
    if (boundGraph !== gd && typeof gd.on === "function") {
      boundGraph = gd;
      gd.on("plotly_afterplot", geometry);
      gd.on("plotly_click", click => {
        if (!request || click?.event?.button === 2 || performance.now() - lastSelectionAt < 100) return;
        const time = click?.points?.[0]?.x;
        if (Number.isFinite(time)) send("selection", {time,session:request.session,view:request.view});
      });
      gd.on("plotly_selected", selected => {
        const pair = Object.entries(selected?.range || {}).find(([axis,pair]) => /^x\d*$/.test(axis) && Array.isArray(pair))?.[1];
        if (!request || !pair || !pair.every(Number.isFinite) || pair[1] <= pair[0]) return;
        interval = pair;
        lastSelectionAt = performance.now();
        drawInterval();
        send("interval", {range:pair,session:request.session,view:request.view});
        // Keep a single selection across all subplots. Remove Plotly's local
        // persistent rectangle/handles after its drag has finished.
        window.setTimeout(() => {
          if (gd.layout.selections?.length) window.Plotly.relayout(gd, {selections:[]});
        }, 0);
      });
    }
  }
  function submit(data, attempts = 0) {
    if (window.dash_clientside?.set_props) {
      window.dash_clientside.set_props("plot-request", {data});
    } else if (attempts < 100) {
      window.setTimeout(() => submit(data, attempts + 1), 100);
    } else send("error", {message: "Signalansicht konnte nicht initialisiert werden."});
  }
  window.addEventListener("message", event => {
    if (event.origin !== window.location.origin || event.source !== window.parent || event.data?.source !== "sleepwalker-ui") return;
    if (event.data.type === "theme") {
      const next = event.data.theme === "dark" ? "dark" : "light";
      if (next !== theme) {motion.stop(true); theme = next; document.documentElement.dataset.theme = theme; geometry();}
    } else if (event.data.type === "view") {
      const next = event.data.request;
      if (!next || !Array.isArray(next.channels)) return;
      if (next.action === "heartbeat") {
        if (!pending) submit(next);
        return;
      }
      motion.accept(next);
      request = next;
      pending = next;
      window.clearTimeout(submitTimer);
      // Coalesce wheel/drag bursts; Dash discards obsolete callback responses,
      // and the parent rejects statuses older than its current revision.
      submitTimer = window.setTimeout(() => { const next = pending; pending = null; if (next) submit(next); }, 80);
    } else if (event.data.type === "cursor" && Number.isFinite(event.data.time)) {
      cursor = event.data.time;
      drawCursor();
    } else if (event.data.type === "interval") {
      interval = Array.isArray(event.data.range) ? event.data.range : null;
      drawInterval();
    } else if (event.data.type === "mode" && ["select","pan"].includes(event.data.mode)) {
      if (mode !== event.data.mode) motion.stop(true);
      mode = event.data.mode;
      geometry();
    } else if (event.data.type === "close" && request) {
      motion.stop();
      window.clearTimeout(submitTimer);
      pending = null;
      submit({...request, action: "close", revision: request.revision + 1});
      request = null;
      cursorLine.hidden = true;
      interval = null; intervalBox.hidden = true;
    }
  });
  window.sleepwalkerBridge = {
    report(status, click, relayout) {
      if (!request) { send("ready"); return; }
      if (status && status.session === request.session && status.view === request.view && (status.revision === undefined || status.revision >= request.revision)) {
        const key = JSON.stringify(status);
        if (key !== lastStatus) {
          lastStatus = key;
          send("status", {status});
        }
        geometry();
      }
      if (relayout) {
        for (const [key, value] of Object.entries(relayout)) {
          const axis = key.split(".")[0];
          if (!/^xaxis\d*$/.test(axis)) continue;
          if (key.endsWith(".autorange") && value && graph()?.data?.length) {
            send("range", {range: [0, request.duration], session: request.session, view: request.view});
            break;
          }
          const pair = key.endsWith(".range") ? value : [relayout[axis + ".range[0]"], relayout[axis + ".range[1]"]];
          if (Array.isArray(pair) && pair.length === 2 && pair.every(Number.isFinite) && pair[1] > pair[0]) {
            if (motion.owns(pair)) return;
            send("range", {range: pair, session: request.session, view: request.view});
            break;
          }
        }
      }
    }
  };
  const announce = window.setInterval(() => {
    if (!request) send("ready");
    else window.clearInterval(announce);
  }, 300);
  window.addEventListener("resize", geometry);
  document.addEventListener("pointerdown", event => {
    if(event.button!==0)return;
    send("dismiss");
    motion.down(event,mode);
  },true);
  document.addEventListener("contextmenu", event => {
    const gd = graph(), layout = gd?._fullLayout;
    if (!request || !layout?.xaxis || !event.target.closest("#signals")) return;
    const x = event.clientX - gd.getBoundingClientRect().left - layout._size.l;
    if (x < 0 || x > layout._size.w) return;
    event.preventDefault();
    motion.stop(true);
    const time = Math.max(0, Math.min(request.duration - 1e-6, layout.xaxis.p2l(x)));
    send("context", {time,x:event.clientX,y:event.clientY,session:request.session,view:request.view});
  });
})();
