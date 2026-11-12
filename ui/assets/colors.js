export const palette=["#245cf0","#008f96","#d52b9b","#6727c9","#f47916","#3c9b35"];
const stages={wake:"#f2c500",w:"#f2c500",n1:"#36b8ea",n2:"#245cf0",n3:"#6727c9",rem:"#d52b9b"};
export function labelColor(label) {
  const name=String(label).trim().toLowerCase();
  if(stages[name])return stages[name];
  if(/arousal/.test(name))return "#f47916";
  if(/hypopn/.test(name))return "#e69114";
  if(/apn/.test(name))return "#e53b3b";
  if(/desat/.test(name))return "#008f96";
  // Unknown classes retain their color when tasks are reordered or reopened.
  let hash=0;for(const char of name)hash=(hash*31+char.charCodeAt(0))>>>0;
  return palette[hash%palette.length];
}
// The task's background class ("nothing happens") is always gray.
export const taskLabelColor=(task,label)=>label===task.background_class?"#b9bebb":labelColor(label);
export function labelTextColor(label) {
  const color=labelColor(label),rgb=[1,3,5].map(i=>parseInt(color.slice(i,i+2),16)/255);
  const linear=rgb.map(v=>v<=.04045?v/12.92:((v+.055)/1.055)**2.4);
  const luminance=.2126*linear[0]+.7152*linear[1]+.0722*linear[2];
  return (luminance+.05)/.05>1.05/(luminance+.05)?"#000000":"#ffffff";
}
