import assert from "node:assert/strict";
import {test} from "node:test";
import {readFile} from "node:fs/promises";
const source=await readFile(new URL("../assets/i18n.js",import.meta.url),"utf8");
const saved=new Map();
globalThis.localStorage={getItem:key=>saved.get(key),setItem:(key,value)=>saved.set(key,value)};
const {t,language,locale,setLanguage,translationFor,renderTranslation,LANGUAGE_KEY}=await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);

test("languages are explicit, persistent and fall back safely",()=>{
  assert.equal(language(),"de");assert.equal(t("Aufzeichnung öffnen"),"Aufzeichnung öffnen");
  assert.equal(setLanguage("en"),true);assert.equal(saved.get(LANGUAGE_KEY),"en");
  assert.equal(t("Aufzeichnung öffnen"),"Open recording");assert.equal(locale(),"en-GB");
  assert.equal(setLanguage("fr"),false);assert.equal(language(),"en");
  assert.equal(t("EDF_Custom_C3-M2"),"EDF_Custom_C3-M2");
  const error=t(t("Mindestens einen Kanal auswählen."));
  setLanguage("de");assert.equal(renderTranslation(translationFor(error)),"Mindestens einen Kanal auswählen.");
});
test("nested messages can change language without losing their parameters",()=>{
  setLanguage("de");
  const caption=t`Manuell: ${t("Apnoe")} · Modell: ${t("Reguläre Atmung")}`;
  const message=translationFor(caption);
  setLanguage("en");assert.equal(renderTranslation(message),"Manual: Apnea · Model: Regular breathing");
  setLanguage("de");assert.equal(renderTranslation(message),"Manuell: Apnoe · Modell: Reguläre Atmung");
});
test("interpolated user data is literal and persistence can be unavailable",()=>{
  setLanguage("en");assert.equal(t`Benötigte Eingänge: ${"C3 ${0} <script>"}`,"Required inputs: C3 ${0} <script>");
  globalThis.localStorage={setItem(){throw new Error("unavailable");}};
  assert.equal(setLanguage("de"),true);assert.equal(locale(),"de-DE");
});
