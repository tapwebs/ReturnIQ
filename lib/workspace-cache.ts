// Disposable view cache only. The database remains authoritative.
const cache=new Map<string,{data:any;at:number}>();
const pending=new Map<string,Promise<any>>();
let active='nimbus';let generation=0;
export const activeStore=()=>active;
export const setActiveStore=(id:string)=>{active=id;};
export function cachedWorkspace(id:string){const v=cache.get(id);return v&&Date.now()-v.at<300000?v.data:null;}
export function seedWorkspaceCache(id:string,data:any){cache.set(id,{data,at:Date.now()});}
export function clearWorkspaceCache(){generation++;cache.clear();pending.clear();}
export async function fetchWorkspace(id:string,force=false){if(!force&&pending.has(id))return pending.get(id)!;const epoch=generation;const task=fetch('/api/workspace',{headers:{'X-RQ-Workspace':id},cache:'no-store'}).then(async r=>{const d:any=await r.json();if(!r.ok){if(r.status===401)clearWorkspaceCache();throw Object.assign(Error(d.error),{status:r.status})}if(epoch===generation)cache.set(id,{data:d,at:Date.now()});return d}).finally(()=>{if(pending.get(id)===task)pending.delete(id)});pending.set(id,task);return task;}
