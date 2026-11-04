import assert from "node:assert/strict";
import {test} from "node:test";
import {readFile} from "node:fs/promises";

// Load the browser ES module without imposing a Node/npm build on the UI.
const source = await readFile(new URL("../assets/channels.js", import.meta.url), "utf8");
const {autoMatch, isMatched, initialMapping, openInputs, duplicateInputs, canStart, inferenceRequest} = await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);

const signal = label => ({id:label, label});
const signals = ["EEG Fpz-Cz", "EOG horizontal", "c4-m1"].map(signal);
const sleep = {id:"sleep", input_channels:["eeg"], channel_aliases:{eeg:["C3-M2", "C4-M1"]}};
const arousal = {id:"arousal", input_channels:["EEG", "EOG"], channel_aliases:{EEG:["C3-M2"], EOG:["E1-M2"]}};

test("automatic matching follows the package names like the server, ignoring case", () => {
  assert.deepEqual(autoMatch(sleep, signals), {eeg:"c4-m1"});
  assert.equal(isMatched(sleep, signals), true);
  assert.deepEqual(autoMatch(arousal, signals), {EEG:null, EOG:null});
  assert.equal(isMatched(arousal, signals), false);
});

test("start needs every active input connected, and a skipped model needs none", () => {
  const mapping = initialMapping([sleep, arousal], signals);
  assert.deepEqual(openInputs(arousal, mapping.arousal), ["EEG", "EOG"]);
  assert.equal(canStart([sleep, arousal], mapping), false);
  mapping.arousal.skip = true;
  assert.equal(canStart([sleep, arousal], mapping), true);
  mapping.sleep.skip = true;
  assert.equal(canStart([sleep, arousal], mapping), false, "at least one model must run");
});

test("one signal cannot feed two inputs of the same model", () => {
  const mapping = initialMapping([arousal], signals);
  mapping.arousal.links = {EEG:"EEG Fpz-Cz", EOG:"EEG Fpz-Cz"};
  assert.deepEqual(duplicateInputs(arousal, mapping.arousal).sort(), ["EEG", "EOG"]);
  assert.equal(canStart([arousal], mapping), false);
});

test("only models with hand-made connections are sent with a channel map", () => {
  const mapping = initialMapping([sleep, arousal], signals);
  mapping.arousal.links = {EEG:"EEG Fpz-Cz", EOG:"EOG horizontal"};
  assert.deepEqual(inferenceRequest([sleep, arousal], mapping), {
    models:["sleep", "arousal"],
    channel_map:{arousal:{EEG:"EEG Fpz-Cz", EOG:"EOG horizontal"}},
  });
  mapping.arousal.skip = true;
  assert.deepEqual(inferenceRequest([sleep, arousal], mapping), {models:["sleep"], channel_map:{}});
});
