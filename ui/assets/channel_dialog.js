/* Dialog that connects model inputs (left) to EDF signals (right) with lines. */
import {t} from "./i18n.js?v=20261111-background";
import {palette} from "./colors.js?v=20261111-background";
import {canStart, duplicateInputs, inferenceRequest, initialMapping, isManual, openInputs} from "./channels.js?v=20261111-background";

const SVG = "http://www.w3.org/2000/svg";
const $ = id => document.getElementById(id);

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

// Resolves with {models, channel_map} to start, or null if the user closes the dialog.
export function openChannelDialog(models, signals, describeModel) {
  const dialog = $("mapping-dialog"), board = $("mapping-board"), lines = $("mapping-lines");
  const inputsColumn = $("mapping-inputs"), signalsColumn = $("mapping-signals");
  const mapping = initialMapping(models, signals);
  const color = new Map(models.map((model, index) => [model.id, palette[index % palette.length]]));
  let pending = null, drag = null, suppressClick = false;

  const inputPort = (modelId, input) => inputsColumn.querySelector(`[data-model="${CSS.escape(modelId)}"][data-input="${CSS.escape(input)}"]`);
  const signalPort = id => signalsColumn.querySelector(`[data-signal="${CSS.escape(id)}"]`);

  function anchor(node, side) {
    const box = node.getBoundingClientRect(), origin = board.getBoundingClientRect();
    return [side === "right" ? box.right - origin.left : box.left - origin.left, box.top - origin.top + box.height / 2];
  }
  function curve([x1, y1], [x2, y2]) {
    const bend = Math.max(40, (x2 - x1) / 2);
    return `M${x1},${y1} C${x1 + bend},${y1} ${x2 - bend},${y2} ${x2},${y2}`;
  }

  function drawLines(pointer) {
    lines.replaceChildren();
    lines.setAttribute("width", board.clientWidth);
    lines.setAttribute("height", board.clientHeight);
    for (const model of models) {
      const entry = mapping[model.id];
      if (entry.skip) continue;
      for (const input of model.input_channels) {
        const signal = entry.links[input];
        if (!signal) continue;
        // Start right of the row, so the line does not cross the × button.
        const d = curve(anchor(inputPort(model.id, input).parentElement, "right"), anchor(signalPort(signal), "left"));
        const path = document.createElementNS(SVG, "path");
        path.setAttribute("d", d);
        path.setAttribute("class", "mapping-line" + (isManual(entry, input) ? " manual" : ""));
        path.style.stroke = color.get(model.id);
        // A wide invisible twin makes the thin line easy to click.
        const hit = document.createElementNS(SVG, "path");
        hit.setAttribute("d", d);
        hit.setAttribute("class", "mapping-hit");
        const title = document.createElementNS(SVG, "title");
        title.textContent = t`${input} → ${signal}: klicken zum Lösen`;
        hit.append(title);
        hit.addEventListener("click", () => { entry.links[input] = null; render(); });
        lines.append(path, hit);
      }
    }
    if (drag?.moved && pointer) {
      const path = document.createElementNS(SVG, "path");
      path.setAttribute("d", curve(anchor(inputPort(drag.modelId, drag.input).parentElement, "right"), pointer));
      path.setAttribute("class", "mapping-line manual");
      path.style.stroke = color.get(drag.modelId);
      lines.append(path);
    }
  }

  function connect(modelId, input, signalId) {
    mapping[modelId].links[input] = signalId;
    pending = null;
    render();
  }

  function renderInputs() {
    inputsColumn.replaceChildren();
    for (const model of models) {
      const entry = mapping[model.id], open = new Set(openInputs(model, entry)), duplicates = new Set(duplicateInputs(model, entry));
      const group = element("section", undefined, "mapping-model" + (entry.skip ? " skipped" : ""));
      group.style.setProperty("--model-color", color.get(model.id));
      const heading = element("div", undefined, "mapping-model-heading"), skip = element("label", undefined, "mapping-skip"), box = element("input");
      box.type = "checkbox"; box.checked = entry.skip;
      box.addEventListener("change", () => { entry.skip = box.checked; if (pending?.modelId === model.id) pending = null; render(); });
      skip.append(box, element("span", t("nicht ausführen")));
      heading.append(element("strong", describeModel(model)), skip);
      group.append(heading);
      for (const input of model.input_channels) {
        const row = element("div", undefined, "mapping-input-row");
        const port = element("button", input, "mapping-port input-port");
        port.type = "button"; port.dataset.model = model.id; port.dataset.input = input; port.disabled = entry.skip;
        if (open.has(input)) port.classList.add("open");
        if (duplicates.has(input)) port.classList.add("duplicate");
        if (pending?.modelId === model.id && pending.input === input) port.classList.add("pending");
        port.setAttribute("aria-pressed", String(port.classList.contains("pending")));
        row.append(port);
        if (entry.links[input] && !entry.skip) {
          const unlink = element("button", "×", "mapping-unlink");
          unlink.type = "button";
          unlink.setAttribute("aria-label", t`Verbindung von ${input} lösen`);
          unlink.addEventListener("click", () => { entry.links[input] = null; render(); });
          row.append(unlink);
        }
        group.append(row);
      }
      inputsColumn.append(group);
    }
  }

  function renderSignals() {
    signalsColumn.replaceChildren();
    for (const signal of signals) {
      const port = element("button", undefined, "mapping-port signal-port");
      port.type = "button"; port.dataset.signal = signal.id; port.disabled = !pending;
      port.append(element("span", signal.label), element("small", signal.unit || ""));
      signalsColumn.append(port);
    }
  }

  function render() {
    renderInputs();
    renderSignals();
    const manual = models.some(model => !mapping[model.id].skip && model.input_channels.some(input => isManual(mapping[model.id], input)));
    $("mapping-manual-hint").hidden = !manual;
    const open = models.reduce((sum, model) => sum + openInputs(model, mapping[model.id]).length, 0);
    const duplicates = models.some(model => duplicateInputs(model, mapping[model.id]).length);
    $("mapping-status").textContent = duplicates ? t("Ein EDF-Kanal kann pro Modell nur einen Eingang versorgen.")
      : open ? t`${open} Eingänge noch offen. Verbinden oder Modell nicht ausführen.` : pending ? t("Jetzt rechts einen EDF-Kanal wählen.") : "";
    $("mapping-start").disabled = !canStart(models, mapping);
    requestAnimationFrame(() => drawLines());
  }

  inputsColumn.onclick = event => {
    const port = event.target.closest(".input-port");
    if (!port || suppressClick) return;
    const same = pending?.modelId === port.dataset.model && pending.input === port.dataset.input;
    pending = same ? null : {modelId: port.dataset.model, input: port.dataset.input};
    render();
  };
  signalsColumn.onclick = event => {
    const port = event.target.closest(".signal-port");
    if (port && pending) connect(pending.modelId, pending.input, port.dataset.signal);
  };
  inputsColumn.onpointerdown = event => {
    const port = event.target.closest(".input-port");
    if (!port || port.disabled || event.button !== 0) return;
    drag = {modelId: port.dataset.model, input: port.dataset.input, x: event.clientX, y: event.clientY, moved: false};
  };
  dialog.onpointermove = event => {
    if (!drag) return;
    drag.moved ||= Math.hypot(event.clientX - drag.x, event.clientY - drag.y) > 4;
    if (!drag.moved) return;
    const origin = board.getBoundingClientRect();
    drawLines([event.clientX - origin.left, event.clientY - origin.top]);
  };
  dialog.onpointerup = event => {
    if (!drag) return;
    const {modelId, input, moved} = drag;
    drag = null;
    if (!moved) return;
    // The click that follows this pointerup must not also select an input.
    suppressClick = true;
    setTimeout(() => { suppressClick = false; });
    const target = document.elementFromPoint(event.clientX, event.clientY)?.closest(".signal-port");
    if (target) connect(modelId, input, target.dataset.signal); else drawLines();
  };

  const observer = new ResizeObserver(() => drawLines());
  observer.observe(board);
  render();
  dialog.showModal();

  return new Promise(resolve => {
    const finish = result => {
      observer.disconnect();
      $("mapping-start").onclick = $("mapping-cancel").onclick = dialog.onclose = null;
      if (dialog.open) dialog.close();
      resolve(result);
    };
    $("mapping-start").onclick = () => { if (canStart(models, mapping)) finish(inferenceRequest(models, mapping)); };
    $("mapping-cancel").onclick = () => finish(null);
    dialog.onclose = () => finish(null);
  });
}
