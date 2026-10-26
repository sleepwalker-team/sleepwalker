import assert from "node:assert/strict";
import {test} from "node:test";
import {readFile} from "node:fs/promises";
const source=await readFile(new URL("../assets/sessions.js",import.meta.url),"utf8");
const {knownSessions,sessionSnapshot,rememberKnownSession,sessionURL}=await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);

test("recent sessions filter malformed data, deduplicate and retain the newest twenty",()=>{
  assert.deepEqual(knownSessions({}),[]);
  const entries=[null,{}, {id:15}, {id:""},...Array.from({length:25},(_,i)=>({id:`session_${i}`,updatedAt:i})),{id:"session_1",updatedAt:100}];
  const result=knownSessions(entries);
  assert.equal(result.length,20);assert.equal(result[0].id,"session_1");
  assert.equal(new Set(result.map(s=>s.id)).size,20);
});

test("remembering one session preserves the other recordings",()=>{
  assert.deepEqual(rememberKnownSession([{id:"a",title:"one",updatedAt:1},{id:"b",updatedAt:2}],{id:"a",title:"updated",updatedAt:3}),[
    {id:"a",title:"updated",updatedAt:3},{id:"b",updatedAt:2},
  ]);
});

test("snapshots preserve navigation and job identifiers without signal or prediction arrays",()=>{
  const state={session:{id:"one",signals:[1,2]},title:"night.edf",job:{id:"job_one"},export:null,range:[40,70],time:45,taskId:"sleep",channels:["EEG"],mode:"pan",absolute:true,expanded:new Set(["sleep"]),predictions:[1,2,3]};
  const snapshot=sessionSnapshot(state,10);
  assert.deepEqual(snapshot.range,[40,70]);assert.equal(snapshot.title,"night.edf");assert.equal(snapshot.job.id,"job_one");
  assert.equal(snapshot.mode,"pan");assert.deepEqual(snapshot.expanded,["sleep"]);
  assert.ok(!("predictions" in snapshot));assert.ok(!("signals" in snapshot));
  state.range[0]=50;state.channels.push("EOG");assert.equal(snapshot.range[0],40);assert.equal(snapshot.channels.length,1);
});

test("session URLs retain other query parameters and removing a session returns to the start page",()=>{
  const url=sessionURL("http://localhost:6007/ui/?other=1#viewer","session a");
  assert.equal(url.searchParams.get("session"),"session a");assert.equal(url.hash,"#viewer");
  const home=sessionURL(url.href,null);assert.equal(home.searchParams.get("session"),null);assert.equal(home.searchParams.get("other"),"1");
});
