/* Annotation planning uses only the existing annotation operations. */
export function correctionRange(task, start, end, duration) {
  if (!task || !Number.isFinite(start) || !Number.isFinite(end) || start < 0 || end <= start || end > duration + 1e-6) {
    throw new Error("Ein gültiges Intervall innerhalb der Aufzeichnung angeben.");
  }
  if (task.kind !== "epoch") return {start, end, count:1};
  const step = task.resolution;
  if (!Number.isFinite(step) || step <= 0) throw new Error("Ungültige Epochenlänge.");
  const first = Math.floor(start / step + 1e-9);
  const last = Math.min(Math.ceil(end / step - 1e-9), Math.floor(duration / step + 1e-9));
  if (last <= first) throw new Error("Die Auswahl enthält keine vollständige korrigierbare Epoche.");
  return {start:first * step, end:last * step, count:last - first};
}

export function correctionOperations(task, range, label) {
  if (!task.classes.includes(label)) throw new Error("Eine gültige Klasse auswählen.");
  if (task.kind !== "epoch") return [{operation:"add_event", task:task.id, start:range.start, end:range.end, label}];
  return Array.from({length:range.count}, (_, i) => ({operation:"set_label", task:task.id, start:range.start + i * task.resolution, label}));
}

export async function saveCorrection({task, range, label, event = null}, post, progress = () => {}) {
  const operations = correctionOperations(task, range, label), saved = [];
  for (const operation of operations) {
    saved.push(await post(operation));
    progress(saved.length, operations.length);
  }
  if (event && task.kind !== "epoch") {
    // Preserve the original until its replacement is saved. If deleting the
    // original fails, undo the replacement so a failed edit does not duplicate it.
    try {
      await post({operation:"delete_event", task:task.id, event_id:event.eventId});
    } catch (error) {
      try { await post({operation:"revert", edit_id:saved[0].edit_id}); }
      catch { throw new Error("Das neue Ereignis wurde gespeichert, das bisherige aber nicht ersetzt. Beide sind sichtbar; bitte den Verlauf prüfen."); }
      throw error;
    }
  }
  return saved;
}

export function selectionAt(task, predictions, annotations, time, klass = null) {
  const matches = item => item.task === task.id && item.start <= time && time < (item.end ?? item.start + task.resolution) && (!klass || item.label === klass);
  const event = [...annotations.events].reverse().find(matches);
  if (event) return {kind:"event", task:task.id, start:event.start, end:event.end, label:event.label, editId:event.edit_id, eventId:event.event_id};
  const label = annotations.labels.find(item => item.task===task.id && item.start<=time && time<item.start+task.resolution);
  if (label) return klass && label.label!==klass ? null : {kind:"label", task:task.id, start:label.start, end:label.start + task.resolution, label:label.label, editId:label.edit_id};
  const prediction = predictions.find(p => (p.start ?? p.time) <= time && time < (p.end ?? p.time + task.resolution) && (!klass || !p.label || p.label === klass));
  if (!prediction) return null;
  const winner = prediction.label ?? Object.entries(prediction.probabilities).sort((a,b) => b[1] - a[1])[0]?.[0];
  return {kind:"prediction", task:task.id, start:prediction.start ?? prediction.time, end:prediction.end ?? prediction.time + task.resolution, label:klass || winner};
}

export function selectedCorrections(selection, task, annotations) {
  if (!selection || !task) return [];
  const labels = annotations.labels.filter(item => item.task===task.id && (
    selection.kind==="label" ? item.edit_id===selection.editId :
    selection.kind==="interval" && item.start<selection.end && item.start+task.resolution>selection.start));
  const events = annotations.events.filter(item => item.task===task.id && (
    selection.kind==="event" ? item.event_id===selection.eventId :
    selection.kind==="interval" && item.start>=selection.start && item.end<=selection.end));
  return [...labels.map(item=>({operation:"revert",edit_id:item.edit_id})),
    ...events.map(item=>({operation:"delete_event",task:task.id,event_id:item.event_id}))];
}
