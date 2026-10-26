/* Same-origin bridge between the plain UI and the server-backed Dash graph. */
(() => {
  "use strict";
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
  let interval = null, mode = "select", applyingLayout = false, panGesture = null;
  function geometry() {
    const gd = graph();
    if (gd?._fullLayout?._size) send("geometry", {left: gd._fullLayout._size.l, right: gd._fullLayout._size.r, width: gd._fullLayout.width});
    drawCursor();
    drawInterval();
    const update = {};
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
      window.Plotly.relayout(gd, update).finally(() => { applyingLayout = false; geometry(); });
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
    if (event.data.type === "view") {
      const next = event.data.request;
      if (!next || !Array.isArray(next.channels)) return;
      if (next.action === "heartbeat") {
        if (!pending) submit(next);
        return;
      }
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
      mode = event.data.mode;
      geometry();
    } else if (event.data.type === "close" && request) {
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
            // Plotly clips the two endpoints separately at its native limits.
            // Keep a pan from becoming an accidental zoom when an edge is hit.
            const gesture=panGesture;panGesture=null;
            if(gesture?.moved&&gesture.view===request.view&&gesture.session===request.session&&Math.abs(pair[1]-pair[0]-gesture.width)>1e-6){
              const width=Math.min(gesture.width,request.duration),start=Math.max(0,Math.min(pair[0],request.duration-width));
              const gd=graph(),update={};
              for(const axis of Object.keys(gd.layout).filter(key=>/^xaxis\d*$/.test(key)))update[axis+".range"]=[start,start+width];
              window.Plotly.relayout(gd,update);
              return;
            }
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
    send("dismiss");panGesture=null;
    const axis=graph()?._fullLayout?.xaxis;
    if(mode==="pan"&&request&&axis&&event.target.closest(".nsewdrag"))panGesture={pointer:event.pointerId,x:event.clientX,width:axis.range[1]-axis.range[0],session:request.session,view:request.view,moved:false};
  },true);
  document.addEventListener("pointerup",event=>{if(panGesture?.pointer===event.pointerId)panGesture.moved=Math.abs(event.clientX-panGesture.x)>2;},true);
  document.addEventListener("pointercancel",()=>{panGesture=null;},true);
  document.addEventListener("wheel",()=>{panGesture=null;},{capture:true,passive:true});
  document.addEventListener("contextmenu", event => {
    const gd = graph(), layout = gd?._fullLayout;
    if (!request || !layout?.xaxis || !event.target.closest("#signals")) return;
    const x = event.clientX - gd.getBoundingClientRect().left - layout._size.l;
    if (x < 0 || x > layout._size.w) return;
    event.preventDefault();
    const time = Math.max(0, Math.min(request.duration - 1e-6, layout.xaxis.p2l(x)));
    send("context", {time,x:event.clientX,y:event.clientY,session:request.session,view:request.view});
  });
})();
