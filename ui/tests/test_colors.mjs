import assert from "node:assert/strict";
import {test} from "node:test";
import {readFile} from "node:fs/promises";
const source=await readFile(new URL("../assets/colors.js",import.meta.url),"utf8");
const {isNegativeClass,labelColor,labelTextColor}=await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);

test("labels keep the same colors across case changes and task ordering",()=>{
  for(const label of ["wake","n1","n2","n3","rem","arousal","hypopnea","desaturation","custom class"]){
    assert.equal(labelColor(label),labelColor(label.toUpperCase()));
    assert.equal(labelColor(label,0,0),labelColor(label,3,5));
  }
  assert.equal(new Set(["wake","n1","n2","n3","rem"].map(labelColor)).size,5);
});

test("negative classes become baseline labels without hiding sleep stages",()=>{
  for(const label of ["no arousal","no_desaturation","no-apnea","regular breathing","none","Kein Arousal"])assert.equal(isNegativeClass(label),true,label);
  for(const label of ["wake","n1","n2","n3","rem","arousal","apnea"])assert.equal(isNegativeClass(label),false,label);
});

test("class text remains readable on every palette color",()=>{
  const luminance=color=>{
    const rgb=[1,3,5].map(i=>parseInt(color.slice(i,i+2),16)/255).map(v=>v<=.04045?v/12.92:((v+.055)/1.055)**2.4);
    return .2126*rgb[0]+.7152*rgb[1]+.0722*rgb[2];
  };
  for(const label of ["wake","n1","n2","n3","rem","arousal","apnea","hypopnea","desaturation","no arousal",...Array.from({length:25},(_,i)=>`custom ${i}`)]){
    const a=luminance(labelColor(label)),b=luminance(labelTextColor(label));
    assert.ok((Math.max(a,b)+.05)/(Math.min(a,b)+.05)>=4.5,label);
  }
});
