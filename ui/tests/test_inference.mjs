import assert from "node:assert/strict";
import {test} from "node:test";
import {readFile} from "node:fs/promises";

// Load the browser ES module without imposing a Node/npm build on the UI.
const source = await readFile(new URL("../assets/inference.js", import.meta.url), "utf8");
const {submitCompatibleModels} = await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);
const missing = (model, channels = ["EEG"]) => Object.assign(new Error("No mapping"), {code:"MISSING_REQUIRED_CHANNEL",details:{model_id:model,channels}});

test("a rejected model does not prevent the other selected models from running", async () => {
  const calls=[], issues=[], accepted={job_id:"job_one"};
  const result=await submitCompatibleModels(["arousal","sleep","desaturation"], async models=>{
    calls.push([...models]);
    if(models.includes("arousal"))throw missing("arousal",["EEG","EOG","Chin EMG","ECG"]);
    return accepted;
  }, issue=>issues.push(issue));
  assert.deepEqual(calls,[["arousal","sleep","desaturation"],["sleep","desaturation"]]);
  assert.equal(result.accepted,accepted);
  assert.deepEqual(issues,[{modelId:"arousal",channels:["EEG","EOG","Chin EMG","ECG"]}]);
});

test("all rejected models leave the study open without claiming a job was started", async () => {
  let calls=0;
  const result=await submitCompatibleModels(["sleep","arousal"],async models=>{calls++;throw missing(models[0]);},()=>{});
  assert.equal(calls,2);
  assert.equal(result.accepted,null);
  assert.equal(result.skipped.length,2);
  assert.equal(result.cancelled,false);
});

test("channel aliases remain the server's decision", async () => {
  const result=await submitCompatibleModels(["alias-model"],async models=>({job_id:"alias-accepted"}),()=>{throw new Error("No local matching should reject this model");});
  assert.equal(result.accepted.job_id,"alias-accepted");
});

test("unrelated API errors and inconsistent rejections are not retried",async () => {
  for(const error of [new Error("Disconnected"),Object.assign(new Error("Compute unavailable"),{code:"COMPUTE_UNAVAILABLE"}),missing("not-selected")]) {
    let calls=0;
    await assert.rejects(()=>submitCompatibleModels(["sleep"],async()=>{calls++;throw error;},()=>{}),value=>value===error);
    assert.equal(calls,1);
  }
});

test("a response for a study that was closed cannot trigger more inference requests",async () => {
  let current=true,calls=0;
  const result=await submitCompatibleModels(["sleep","arousal"],async()=>{calls++;current=false;throw missing("sleep");},()=>{throw new Error("Stale issue");},()=>current);
  assert.equal(result.cancelled,true);
  assert.equal(calls,1);
});
