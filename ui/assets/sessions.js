/* Browser-side references only; recordings and results stay on the server. */
export function knownSessions(value) {
  if (!Array.isArray(value)) return [];
  const seen=new Set();
  return value.filter(item=>item&&typeof item.id==="string"&&item.id.length>0&&item.id.length<=200)
    .sort((a,b)=>(Number(b.updatedAt)||0)-(Number(a.updatedAt)||0))
    .filter(item=>{if(seen.has(item.id))return false;seen.add(item.id);return true;}).slice(0,20);
}

export function sessionSnapshot(state, now=Date.now()) {
  return {id:state.session.id,title:state.title,job:state.job,export:state.export,
    range:[...state.range],time:state.time,taskId:state.taskId,channels:[...state.channels],
    mode:state.mode,absolute:state.absolute,expanded:[...state.expanded],updatedAt:now};
}

export function rememberKnownSession(entries, snapshot) {
  return knownSessions([snapshot,...knownSessions(entries).filter(item=>item.id!==snapshot.id)]);
}

export function sessionURL(href, id) {
  const url=new URL(href);
  if(id)url.searchParams.set("session",id);else url.searchParams.delete("session");
  return url;
}
