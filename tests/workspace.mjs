import assert from 'node:assert/strict';
import {DatabaseSync} from 'node:sqlite';
import fs from 'node:fs';
import ts from 'typescript';
import {pathToFileURL} from 'node:url';
const sql=new DatabaseSync(':memory:');
sql.exec(fs.readFileSync('drizzle/0000_lazy_impossible_man.sql','utf8').replaceAll('--> statement-breakpoint',''));
globalThis.__rqTestEnv = {
 DB: {
  prepare(q) {
   return {bind(...args) {
    const statement=sql.prepare(q);
    return {
     async first(){return statement.get(...args)||null},
     async run(){return {meta:{changes:Number(statement.run(...args).changes)}}}
    };
   }};
  }
 }
};
let source=fs.readFileSync('app/api/workspace/route.ts','utf8').replace("import {env} from 'cloudflare:workers';","const env=globalThis.__rqTestEnv;").replace("'@/lib/model'",JSON.stringify(pathToFileURL(process.cwd()+'/lib/model.ts').href));
const js=ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.ESNext,target:ts.ScriptTarget.ES2022}}).outputText;
const api=await import('data:text/javascript;base64,'+Buffer.from(js).toString('base64'));
let cookie='';
async function req(body,store='nimbus',user='tester',scope=''){const r=new Request('https://example.com/api/workspace'+scope,{method:body?'POST':'GET',headers:{'Content-Type':'application/json','oai-authenticated-user-id':user,'X-RQ-Workspace':store,cookie},...(body?{body:JSON.stringify(body)}:{})});const res=await (body?api.POST:api.GET)(r);if(res.headers.get('set-cookie'))cookie=res.headers.get('set-cookie').split(';')[0];return {status:res.status,data:await res.json()}}
assert.equal((await req({op:'login',role:'SELLER'})).status,200);
const n=await req();assert.equal(n.data.state.orders.length,124);
assert.equal((await req(undefined,'aurora')).data.state.orders.length,62);
assert.equal((await req(undefined,'circuit')).data.state.orders.length,82);
assert.equal((await req(undefined,'nimbus','another-user')).status,401);
await req({op:'decision',id:'act_1',version:1,decision:'REJECT'});
assert.equal((await req()).data.state.actions[0].status,'REJECTED');
assert.equal((await req(undefined,'aurora')).data.state.actions[0].status,'PENDING_APPROVAL');
assert.equal((await req({op:'decision',id:'act_1',version:1,decision:'APPROVE'})).status,409);
await req({op:'switch',role:'RIDER'});
const rider=await req();assert(rider.data.state.orders.every(o=>o.rider==='rid_1'));assert.equal(rider.data.state.actions.length,0);
assert.equal((await req(undefined,'nimbus','tester','?scope=seller')).status,403);
assert.equal((await req({op:'propose',id:'ord_10491'})).status,403);
assert.equal((await req({op:'delivery',id:'ord_10603',outcome:'NDR',reason:'Unavailable'})).status,404);
assert.equal((await req({op:'delivery',id:'ord_10597',outcome:'NDR',reason:''})).status,422);
assert.equal((await req({op:'delivery',id:'ord_10597',outcome:'NDR',reason:'Customer unavailable'})).status,200);
const cache=await import('../lib/workspace-cache.ts');let calls=0;
globalThis.fetch=async()=>{calls++;return Response.json({role:'SELLER',state:{orders:[]}})};
await Promise.all([cache.fetchWorkspace('nimbus'),cache.fetchWorkspace('nimbus')]);assert.equal(calls,1);assert.equal(cache.cachedWorkspace('nimbus').role,'SELLER');assert.equal(cache.cachedWorkspace('aurora'),null);cache.clearWorkspaceCache();assert.equal(cache.cachedWorkspace('nimbus'),null);
console.log('PASS: tenant and store isolation, rider authorization, reason validation, decision conflicts, request deduplication and cache clearing.');
