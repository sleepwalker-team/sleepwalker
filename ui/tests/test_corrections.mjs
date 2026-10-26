import assert from "node:assert/strict";
import {test} from "node:test";
import {readFile} from "node:fs/promises";

const source=await readFile(new URL("../assets/corrections.js",import.meta.url),"utf8");
const {correctionRange,correctionOperations,saveCorrection,selectionAt,selectedCorrections}=await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);
const epoch={id:"sleep",kind:"epoch",resolution:30,classes:["n2","n3"]};
const event={id:"arousal",kind:"event",resolution:1,classes:["arousal"]};

test("intervals cover exactly the touched epochs, including exact boundaries",()=>{
  assert.deepEqual(correctionRange(epoch,14.4,69.6,120),{start:0,end:90,count:3});
  assert.deepEqual(correctionRange(epoch,30,60,120),{start:30,end:60,count:1});
  assert.deepEqual(correctionRange(epoch,59.999,60.001,120),{start:30,end:90,count:2});
  assert.deepEqual(correctionRange(epoch,89,99.9,100),{start:60,end:90,count:1});
  assert.deepEqual(correctionRange(epoch,315,345,690),{start:300,end:360,count:2});
});

test("invalid intervals and incomplete trailing epochs cannot be submitted",()=>{
  for(const [start,end] of [[NaN,30],[-1,30],[20,20],[30,20],[100,121],[90,99.9]])assert.throws(()=>correctionRange(epoch,start,end,100));
  assert.deepEqual(correctionRange(event,12.3,17.8,120),{start:12.3,end:17.8,count:1});
});

test("epoch operations retain model output and use the existing per-epoch annotation API",()=>{
  assert.deepEqual(correctionOperations(epoch,{start:30,end:90,count:2},"n3"),[
    {operation:"set_label",task:"sleep",start:30,label:"n3"},
    {operation:"set_label",task:"sleep",start:60,label:"n3"},
  ]);
  assert.throws(()=>correctionOperations(epoch,{start:0,count:1},"invented"));
});

test("saving several epochs stops at the first failure and reports the saved count",async()=>{
  const calls=[],progress=[];
  await assert.rejects(saveCorrection({task:epoch,range:{start:0,count:3},label:"n3"},async operation=>{
    calls.push(operation);if(calls.length===2)throw new Error("offline");return {edit_id:"saved"};
  },(done,total)=>progress.push([done,total])),/offline/);
  assert.equal(calls.length,2);assert.deepEqual(progress,[[1,3]]);
});

test("event replacement saves the new event before deleting the original",async()=>{
  const calls=[];
  await saveCorrection({task:event,range:{start:15,end:19},label:"arousal",event:{eventId:"old"}},async operation=>{calls.push(operation);return {edit_id:"new"};});
  assert.deepEqual(calls.map(c=>c.operation),["add_event","delete_event"]);
  assert.equal(calls[1].event_id,"old");
});

test("failed event replacement preserves the original and undoes the new event",async()=>{
  const calls=[];
  await assert.rejects(saveCorrection({task:event,range:{start:15,end:19},label:"arousal",event:{eventId:"old"}},async operation=>{
    calls.push(operation);if(operation.operation==="delete_event")throw new Error("failed deletion");return {edit_id:"new"};
  }),/failed deletion/);
  assert.deepEqual(calls.map(c=>c.operation),["add_event","delete_event","revert"]);
  assert.equal(calls[2].edit_id,"new");
});

test("failure to save a replacement never deletes the original",async()=>{
  const calls=[];
  await assert.rejects(saveCorrection({task:event,range:{start:15,end:19},label:"arousal",event:{eventId:"old"}},async operation=>{calls.push(operation);throw new Error("offline");}));
  assert.deepEqual(calls.map(c=>c.operation),["add_event"]);
});

test("manual corrections take precedence when clicking a corrected class",()=>{
  const annotations={labels:[{task:"sleep",start:0,label:"n3",edit_id:"edit_one"}],events:[]};
  const predictions=[{start:0,end:30,label:"n2"}];
  assert.equal(selectionAt(epoch,predictions,annotations,15).kind,"label");
  assert.equal(selectionAt(epoch,predictions,annotations,15,"n2"),null);
  assert.equal(selectionAt(epoch,predictions,annotations,30),null);
  annotations.events=[{task:"arousal",start:10,end:20,label:"arousal",edit_id:"edit_two",event_id:"evt_one"}];
  assert.equal(selectionAt(event,[],annotations,12).eventId,"evt_one");
});

test("interval deletion removes each active correction and only fully contained human events",()=>{
  const annotations={labels:[0,30,60,90].map(start=>({task:"sleep",start,edit_id:`edit_${start}`})),events:[
    {task:"sleep",start:5,end:20,event_id:"inside"},{task:"sleep",start:80,end:100,event_id:"partial"},
  ]};
  const operations=selectedCorrections({kind:"interval",task:"sleep",start:0,end:90},epoch,annotations);
  assert.deepEqual(operations,[{operation:"revert",edit_id:"edit_0"},{operation:"revert",edit_id:"edit_30"},{operation:"revert",edit_id:"edit_60"},{operation:"delete_event",task:"sleep",event_id:"inside"}]);
  assert.equal(selectedCorrections({kind:"label",editId:"edit_30"},epoch,annotations).length,1);
  assert.equal(selectedCorrections({kind:"prediction",start:0,end:90},epoch,annotations).length,0);
});
