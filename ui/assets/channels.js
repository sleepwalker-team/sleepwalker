/* Matching model inputs to EDF signals. No DOM, so it can be tested in Node. */
const key = name => String(name).trim().toLowerCase();

// Same rule as the server: an input matches a signal whose ID or label equals one of its package names.
export function autoMatch(model, signals) {
  const links = {};
  for (const input of model.input_channels) {
    const names = (model.channel_aliases?.[input] || [input]).map(key);
    links[input] = signals.find(signal => names.includes(key(signal.id)) || names.includes(key(signal.label)))?.id ?? null;
  }
  return links;
}

export const isMatched = (model, signals) => Object.values(autoMatch(model, signals)).every(Boolean);

// One entry per model: {skip, links: {input: signalId|null}, auto: {input: signalId|null}}.
export function initialMapping(models, signals) {
  return Object.fromEntries(models.map(model => {
    const auto = autoMatch(model, signals);
    return [model.id, {skip:false, links:{...auto}, auto}];
  }));
}

export function openInputs(model, entry) {
  return entry.skip ? [] : model.input_channels.filter(input => !entry.links[input]);
}

// An EDF signal carries one name in the relabeled copy, so it can feed only one input per model.
export function duplicateInputs(model, entry) {
  if (entry.skip) return [];
  const seen = new Map(), duplicates = new Set();
  for (const input of model.input_channels) {
    const signal = entry.links[input];
    if (!signal) continue;
    if (seen.has(signal)) { duplicates.add(input); duplicates.add(seen.get(signal)); } else seen.set(signal, input);
  }
  return [...duplicates];
}

export function isManual(entry, input) {
  return Boolean(entry.links[input]) && entry.links[input] !== entry.auto[input];
}

export function canStart(models, mapping) {
  const active = models.filter(model => !mapping[model.id].skip);
  return active.length > 0 && active.every(model => !openInputs(model, mapping[model.id]).length && !duplicateInputs(model, mapping[model.id]).length);
}

// Models the user changed are sent with a full map; the others keep the server's own matching.
export function inferenceRequest(models, mapping) {
  const active = models.filter(model => !mapping[model.id].skip);
  const channelMap = {};
  for (const model of active) {
    const entry = mapping[model.id];
    if (model.input_channels.some(input => isManual(entry, input))) channelMap[model.id] = {...entry.links};
  }
  return {models: active.map(model => model.id), channel_map: channelMap};
}
