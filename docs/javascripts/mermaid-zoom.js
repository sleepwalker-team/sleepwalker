// Adds click-to-fullscreen zoom to Mermaid diagrams.
//
// The theme renders each diagram into a <div class="mermaid"> that owns a CLOSED
// shadow root holding the SVG. During rendering Mermaid also creates *temporary*
// elements in <body> whose ids contain "mermaid" (e.g. #__mermaid_0, #d__mermaid_0).
// We must only wrap the final host `div.mermaid` and never touch those temp nodes,
// or we move the SVG out from under Mermaid and break the render.
(function () {
  var FULLSCREEN_ICON =
    '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" ' +
    'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
    '<path d="M8 3H5a2 2 0 0 0-2 2v3"/><path d="M21 8V5a2 2 0 0 0-2-2h-3"/>' +
    '<path d="M3 16v3a2 2 0 0 0 2 2h3"/><path d="M16 21h3a2 2 0 0 0 2-2v-3"/></svg>';

  function enhance(host) {
    if (host.dataset.zoomReady) return;
    host.dataset.zoomReady = "1";

    var wrap = document.createElement("div");
    wrap.className = "mermaid-zoom";
    host.parentNode.insertBefore(wrap, host);
    wrap.appendChild(host);

    var btn = document.createElement("button");
    btn.className = "mermaid-zoom__btn";
    btn.type = "button";
    btn.setAttribute("aria-label", "Toggle fullscreen diagram");
    btn.innerHTML = FULLSCREEN_ICON;
    wrap.appendChild(btn);

    function toggle() {
      if (document.fullscreenElement === wrap) {
        document.exitFullscreen();
      } else if (wrap.requestFullscreen) {
        wrap.requestFullscreen();
      }
    }

    btn.addEventListener("click", toggle);
    host.addEventListener("dblclick", toggle);
    document.addEventListener("fullscreenchange", function () {
      btn.classList.toggle("is-active", document.fullscreenElement === wrap);
    });
  }

  function tryEnhance(node) {
    if (node.nodeType !== 1 || !node.matches) return;
    if (node.matches("div.mermaid")) {
      enhance(node);
    } else if (node.querySelectorAll) {
      node.querySelectorAll("div.mermaid").forEach(enhance);
    }
  }

  var mo = new MutationObserver(function (mutations) {
    mutations.forEach(function (m) {
      m.addedNodes.forEach(tryEnhance);
    });
  });

  function start() {
    mo.observe(document.body, { childList: true, subtree: true });
    tryEnhance(document.body);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
