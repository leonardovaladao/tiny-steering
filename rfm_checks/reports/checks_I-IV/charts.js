const NS='http://www.w3.org/2000/svg',CONCEPTS=['food','valence','env_social'],BLOCKS=[0,1,2,3,4,5];
const tip=document.getElementById('tip');
const f4=v=>v==null?'–':(+v).toFixed(4), f2=v=>v==null?'–':(+v).toFixed(2), f3=v=>v==null?'–':(+v).toFixed(3);
const $=id=>document.getElementById(id);
function E(t,a,p){const e=document.createElementNS(NS,t);for(const k in a)if(a[k]!=null)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e;}
function showTip(ev,h){tip.innerHTML=h;tip.style.display='block';const w=tip.offsetWidth,hh=tip.offsetHeight;
  let x=ev.clientX+14,y=ev.clientY+14;if(x+w>innerWidth-8)x=ev.clientX-w-14;if(y+hh>innerHeight-8)y=ev.clientY-hh-14;tip.style.left=x+'px';tip.style.top=y+'px';}
function hideTip(){tip.style.display='none';}
function bindTip(e,h){e.addEventListener('pointermove',ev=>showTip(ev,h));e.addEventListener('pointerleave',hideTip);e.addEventListener('click',ev=>showTip(ev,h));}
function nice(lo,hi,n=5){const st0=(hi-lo)/n,m=10**Math.floor(Math.log10(st0)),r=st0/m,st=(r<1.5?1:r<3?2:r<7?5:10)*m;
  const a=Math.floor(lo/st)*st,b=Math.ceil(hi/st)*st,t=[];for(let v=a;v<=b+st/2;v+=st)t.push(+v.toFixed(6));return{d:[a,b],t};}
function Panel(host,{w=320,h=236,x,y,title,m={l:48,r:10,t:26,b:34}}){
  const s=E('svg',{viewBox:`0 0 ${w} ${h}`,role:'img','aria-label':title||'chart'},host);
  const iw=w-m.l-m.r,ih=h-m.t-m.b;
  const X=x.band?(v=>m.l+(x.vals.indexOf(v)+.5)*iw/x.vals.length):(v=>m.l+(v-x.d[0])/(x.d[1]-x.d[0])*iw);
  const Y=y.log?(v=>m.t+ih-(Math.log10(v)-Math.log10(y.d[0]))/(Math.log10(y.d[1])-Math.log10(y.d[0]))*ih):(v=>m.t+ih-(v-y.d[0])/(y.d[1]-y.d[0])*ih);
  if(title){const t=E('text',{x:m.l,y:14,class:'t'},s);t.textContent=title;}
  for(const v of y.t){E('line',{x1:m.l,x2:w-m.r,y1:Y(v),y2:Y(v),class:'g'},s);const t=E('text',{x:m.l-5,y:Y(v)+3.5,'text-anchor':'end'},s);t.textContent=(y.f||(q=>q))(v);}
  E('line',{x1:m.l,x2:w-m.r,y1:m.t+ih,y2:m.t+ih,class:'ax'},s);
  for(const v of (x.band?x.vals:x.t)){const t=E('text',{x:X(v),y:m.t+ih+13,'text-anchor':'middle'},s);t.textContent=(x.f||(q=>q))(v);}
  if(x.label){const t=E('text',{x:m.l+iw/2,y:h-5,'text-anchor':'middle'},s);t.textContent=x.label;}
  if(y.label){const t=E('text',{x:9,y:m.t+ih/2,'text-anchor':'middle',transform:`rotate(-90 9 ${m.t+ih/2})`},s);t.textContent=y.label;}
  return{s,X,Y,m,iw,ih,w,h};
}
function line(P,pts,{c='var(--s1)',dash,sw=2}){const p=pts.filter(q=>q[1]!=null);return E('path',{d:'M'+p.map(q=>P.X(q[0]).toFixed(1)+' '+P.Y(q[1]).toFixed(1)).join('L'),fill:'none',stroke:c,'stroke-width':sw,'stroke-dasharray':dash,'stroke-linejoin':'round','stroke-linecap':'round'},P.s);}
function area(P,pts,c='var(--nul-soft)'){const a=pts.map(q=>P.X(q[0]).toFixed(1)+' '+P.Y(q[2]).toFixed(1)),b=pts.slice().reverse().map(q=>P.X(q[0]).toFixed(1)+' '+P.Y(q[1]).toFixed(1));return E('path',{d:'M'+a.concat(b).join('L')+'Z',fill:c,opacity:.75},P.s);}
function mark(s,x,y,k,c,r=4,hollow){const a={fill:hollow?'var(--surface)':c,stroke:hollow?c:'var(--surface)','stroke-width':hollow?1.8:1};
  if(k==='s')return E('rect',{x:x-r+.5,y:y-r+.5,width:2*r-1,height:2*r-1,...a},s);
  if(k==='t')return E('path',{d:`M${x} ${y-r-.6}L${x+r} ${y+r-.6}L${x-r} ${y+r-.6}Z`,...a},s);
  if(k==='d')return E('path',{d:`M${x} ${y-r-.6}L${x+r+.6} ${y}L${x} ${y+r+.6}L${x-r-.6} ${y}Z`,...a},s);
  return E('circle',{cx:x,cy:y,r,...a},s);}
function hit(P,x,y,h,r=11){const e=E('circle',{cx:P.X(x),cy:P.Y(y),r,fill:'transparent'},P.s);bindTip(e,h);}
function hrule(P,v,{label,dash='4 3',c='var(--axis)',right}={}){E('line',{x1:P.m.l,x2:P.w-P.m.r,y1:P.Y(v),y2:P.Y(v),stroke:c,'stroke-width':1.2,'stroke-dasharray':dash},P.s);
  if(label){const t=E('text',{x:right?P.w-P.m.r-2:P.m.l+3,y:P.Y(v)-4,class:'i','text-anchor':right?'end':'start'},P.s);t.textContent=label;}}
function legend(id,items){$(id).innerHTML='';for(const i of items){const sp=document.createElement('span'),s=E('svg',{width:26,height:12,viewBox:'0 0 26 12'},sp);
  if(i.bar)E('rect',{x:9,y:0,width:8,height:12,rx:2,fill:i.c},s);
  else if(i.area)E('rect',{x:1,y:1,width:24,height:10,fill:i.c},s);
  else{if(i.line!==false)E('line',{x1:1,x2:25,y1:6,y2:6,stroke:i.c,'stroke-width':i.sw||2,'stroke-dasharray':i.dash},s);if(i.k)mark(s,13,6,i.k,i.c,4,i.hollow);}
  sp.appendChild(document.createTextNode(i.t));$(id).appendChild(sp);}}
function table(id,cols,rows){const t=typeof id==='string'?$(id):id;t.innerHTML='<table><thead><tr>'+cols.map(c=>`<th class="${c.n?'n':''}">${c.h}</th>`).join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+cols.map(c=>`<td class="${c.n?'n':''}">${c.v(r)}</td>`).join('')+'</tr>').join('')+'</tbody></table>';}
function seg(host,label,opts,val,on){const d=document.createElement('div');d.className='seg';d.setAttribute('role','group');d.setAttribute('aria-label',label);d.innerHTML=`<span>${label}</span>`;const bs=[];
  for(const o of opts){const b=document.createElement('button');b.type='button';b.textContent=o.t??o.v;b.setAttribute('aria-pressed',o.v===val);b.dataset.v=o.v;b.onclick=()=>{bs.forEach(x=>x.setAttribute('aria-pressed',x===b));on(o.v);};bs.push(b);d.appendChild(b);}
  $(host).appendChild(d);d.set=v=>bs.forEach(x=>x.setAttribute('aria-pressed',String(x.dataset.v)===String(v)));return d;}
const unit={d:[0,1],t:[0,.25,.5,.75,1],f:f2};
const bandB={band:true,vals:BLOCKS,label:'block'};
const cls=(c)=>c;
const seqFill=t=>`color-mix(in oklab,var(--seq-hi) ${Math.round(Math.min(1,t)*100)}%,var(--seq-lo))`;
const divFill=t=>{const a=Math.min(1,Math.abs(t));return `color-mix(in oklab,${t<0?'var(--div-neg)':'var(--div-pos)'} ${Math.round(Math.sqrt(a)*100)}%,var(--div-mid))`;};
function heat(host,{title,rows,cols,cell,rowLabel,colLabel,w=320,cw,ch=24,ml=34,mt=26,mb=30,fs=10,scroll}){
  const cwd=cw||(w-ml-8)/cols.length,h=mt+ch*rows.length+mb,W=ml+cwd*cols.length+8,s=E('svg',{viewBox:`0 0 ${W} ${h}`,role:'img','aria-label':title||'heatmap'},host);
  if(title){const t=E('text',{x:ml,y:14,class:'t'},s);t.textContent=title;}
  rows.forEach((r,i)=>{const t=E('text',{x:ml-6,y:mt+ch*i+ch/2+3.5,'text-anchor':'end'},s);t.textContent=r.t??r;
    cols.forEach((c,j)=>{const o=cell(r,c);if(!o)return;const x=ml+cwd*j,y=mt+ch*i;
      const re=E('rect',{x:x+1,y:y+1,width:cwd-2,height:ch-2,rx:2,fill:o.fill,stroke:o.stroke,'stroke-width':o.stroke?1.8:null},s);
      if(o.text!=null){const q=E('text',{x:x+cwd/2,y:y+ch/2+3.5,'text-anchor':'middle',style:`fill:${o.ink||'var(--ink)'};font-size:${fs}px`},s);q.textContent=o.text;}
      if(o.tip)bindTip(re,o.tip);if(o.on){re.style.cursor='pointer';re.addEventListener('click',o.on);}});});
  cols.forEach((c,j)=>{const t=E('text',{x:ml+cwd*j+cwd/2,y:mt+ch*rows.length+13,'text-anchor':'middle'},s);t.textContent=c.t??c;});
  if(colLabel){const t=E('text',{x:ml+cwd*cols.length/2,y:h-4,'text-anchor':'middle'},s);t.textContent=colLabel;}
  return s;}
/* ---------- Check I ---------- */
let regime='A';
function f1a(){const H=$('f1a');H.innerHTML='';const R=regime==='A'?DATA.IA:DATA.IB;
  for(const c of CONCEPTS){const P=Panel(H,{x:bandB,y:{...unit,label:'share s'},title:c});hrule(P,.95,{label:c==='food'?'s = 0.95':null});
    for(const r of R.filter(q=>q.c===c)){const x=P.X(r.b);E('rect',{x:x-7,y:P.Y(r.hi),width:14,height:Math.max(2,P.Y(r.lo)-P.Y(r.hi)),rx:3,fill:'var(--nul-soft)'},P.s);
      E('line',{x1:x-7,x2:x+7,y1:P.Y(r.med),y2:P.Y(r.med),stroke:'var(--nul)','stroke-width':2},P.s);
      mark(P.s,x,P.Y(r.s),'c','var(--s1)',4.5,r.refuted);
      hit(P,r.b,(r.s+r.med)/2,`<b>${c}, block ${r.b}</b><br>real s = ${f3(r.s)}<br>shuffled median ${f3(r.med)}, range ${f3(r.lo)}–${f3(r.hi)}<br>p = ${f3(r.p)} · ${r.refuted?'refuted':'not refuted'}`,16);}}
  table('t1a',[{h:'concept',v:r=>r.c},{h:'block',v:r=>r.b,n:1},{h:'real s',v:r=>f3(r.s),n:1},{h:'shuffled median',v:r=>f3(r.med),n:1},{h:'min',v:r=>f3(r.lo),n:1},{h:'max',v:r=>f3(r.hi),n:1},{h:'p',v:r=>f3(r.p),n:1},{h:'refuted',v:r=>r.refuted?'yes':'no'}],R);}
seg('c1a','Regime',[{v:'A',t:'A: frozen HPs, B = 100'},{v:'B',t:'B: full sweep, B = 20'}],'A',v=>{regime=v;f1a();});
legend('l1a',[{bar:1,c:'var(--nul-soft)',t:'shuffled range'},{c:'var(--nul)',t:'shuffled median'},{k:'c',c:'var(--s1)',line:false,t:'real share, passes'},{k:'c',c:'var(--s1)',hollow:1,line:false,t:'real share, refuted'}]);f1a();
(function(){const H=$('f1b');for(const c of CONCEPTS){const R=DATA.Irho.filter(q=>q.c===c),B=DATA.IB.filter(q=>q.c===c),P=Panel(H,{x:bandB,y:{...unit,label:'val |ρ|'},title:c});
  area(P,R.map(r=>[r.b,r.rand_med,r.rand_q95]));
  const S=[['rho_rfm','var(--s1)','c'],['diffmeans','var(--s2)','s'],['pc1','var(--s3)','t'],['rho_shufA','var(--nul)','d']];
  line(P,B.map(r=>[r.b,r.rho_sh]),{c:'var(--nul)',dash:'2 3',sw:1.6});
  for(const[k,cc,mk]of S){line(P,R.map(r=>[r.b,r[k]]),{c:cc});R.forEach(r=>mark(P.s,P.X(r.b),P.Y(r[k]),mk,cc,3.2));}
  R.forEach((r,i)=>{hit(P,r.b,.5,`<b>${c}, block ${r.b}</b><br>RFM real ${f3(r.rho_rfm)}<br>DiffMeans ${f3(r.diffmeans)}<br>PC1 ${f3(r.pc1)}<br>RFM shuffled, A ${f3(r.rho_shufA)}<br>RFM shuffled, B ${f3(B[i].rho_sh)}<br>random median ${f3(r.rand_med)}, q95 ${f3(r.rand_q95)}`,24);});}
  legend('l1b',[{k:'c',c:'var(--s1)',t:'RFM, real labels'},{k:'s',c:'var(--s2)',t:'DiffMeans'},{k:'t',c:'var(--s3)',t:'PC1, no labels'},{k:'d',c:'var(--nul)',t:'RFM shuffled, regime A'},{c:'var(--nul)',dash:'2 3',t:'RFM shuffled, regime B'},{area:1,c:'var(--nul-soft)',t:'random directions, median to q95'}]);
  table('t1b',[{h:'concept',v:r=>r.c},{h:'block',v:r=>r.b,n:1},{h:'RFM',v:r=>f3(r.rho_rfm),n:1},{h:'DM',v:r=>f3(r.diffmeans),n:1},{h:'PC1',v:r=>f3(r.pc1),n:1},{h:'shuffled A',v:r=>f3(r.rho_shufA),n:1},{h:'random median',v:r=>f3(r.rand_med),n:1},{h:'random q95',v:r=>f3(r.rand_q95),n:1}],DATA.Irho);})();
/* ---------- Check Ia ---------- */
const cell={c:'valence',b:3,nz:0,sg:10};const ctl={};
function storedOf(){return DATA.Iastored.find(r=>r.concept===cell.c&&r.block===cell.b&&r.normalize===cell.nz&&r.sigma===cell.sg);}
function f2a(){const H=$('f2a');H.innerHTML='';const C=DATA.Iacur[`${cell.c}|${cell.b}|${cell.nz}|${cell.sg}`],T=[1,2,3,4,5,6,7,8,9,10],st=storedOf();
  const P=Panel(H,{w:640,h:280,x:{d:[1,10],t:T,label:'iteration T'},y:{...unit,label:'share s(T)'},title:`${cell.c}, block ${cell.b}, σ = ${cell.sg}, ${cell.nz?'normalized':'raw'} rows${st?' · stored HP cell':''}`,m:{l:44,r:16,t:26,b:36}});
  area(P,T.map((t,i)=>[t,C.lo[i],C.hi[i]]));line(P,T.map((t,i)=>[t,C.med[i]]),{c:'var(--nul)'});line(P,T.map((t,i)=>[t,C.real[i]]),{c:'var(--s1)',sw:2.4});
  if(st){const x=P.X(st.T_star);E('line',{x1:x,x2:x,y1:P.m.t,y2:P.m.t+P.ih,stroke:'var(--axis)','stroke-dasharray':'4 3'},P.s);const t=E('text',{x:x+(st.T_star>8?-4:4),y:P.m.t+11,class:'i','text-anchor':st.T_star>8?'end':'start'},P.s);t.textContent='stored T* = '+st.T_star;}
  mark(P.s,P.X(10),P.Y(C.real[9]),'c','var(--s1)',4.5);mark(P.s,P.X(10),P.Y(C.med[9]),'c','var(--nul)',4);
  T.forEach((t,i)=>hit(P,t,(C.real[i]+C.med[i])/2,`<b>T = ${t}</b><br>real ${f3(C.real[i])}<br>shuffled median ${f3(C.med[i])}<br>shuffled range ${f3(C.lo[i])}–${f3(C.hi[i])}`,20));
  table('t2a',[{h:'T',v:r=>r,n:1},{h:'real',v:r=>f3(C.real[r-1]),n:1},{h:'shuffled median',v:r=>f3(C.med[r-1]),n:1},{h:'shuffled min',v:r=>f3(C.lo[r-1]),n:1},{h:'shuffled max',v:r=>f3(C.hi[r-1]),n:1}],T);}
function setCell(o){Object.assign(cell,o);ctl.c.set(cell.c);ctl.b.set(cell.b);ctl.nz.set(cell.nz);ctl.sg.set(cell.sg);f2a();f2b();}
ctl.c=seg('c2a','Concept',CONCEPTS.map(v=>({v})),cell.c,v=>setCell({c:v}));
ctl.b=seg('c2a','Block',BLOCKS.map(v=>({v})),cell.b,v=>setCell({b:v}));
ctl.sg=seg('c2a','σ',[1,10,100].map(v=>({v})),cell.sg,v=>setCell({sg:v}));
ctl.nz=seg('c2a','Rows',[{v:0,t:'raw'},{v:1,t:'normalized'}],cell.nz,v=>setCell({nz:v}));
legend('l2a',[{c:'var(--s1)',t:'real labels'},{c:'var(--nul)',t:'shuffled median'},{area:1,c:'var(--nul-soft)',t:'shuffled range, 20 permutations'}]);
function f2b(){const H=$('f2b');H.innerHTML='';const cols=[[0,1],[0,10],[0,100],[1,1],[1,10],[1,100]].map(([nz,sg])=>({nz,sg,t:(nz?'n':'r')+sg}));
  for(const c of CONCEPTS)heat(H,{title:c,rows:BLOCKS.map(b=>({b,t:'b'+b})),cols,colLabel:'r = raw, n = normalized, number = σ',cell:(r,k)=>{const q=DATA.Iacell.find(x=>x.concept===c&&x.block===r.b&&x.normalize===k.nz&&x.sigma===k.sg),t=q.median_delta10/.6,sel=cell.c===c&&cell.b===r.b&&cell.nz===k.nz&&cell.sg===k.sg;
    return{fill:seqFill(t),text:'+'+q.median_delta10.toFixed(2),ink:t>.55?'var(--surface)':'var(--ink)',stroke:sel?'var(--s2)':null,tip:`<b>${c}, block ${r.b}, σ ${k.sg}, ${k.nz?'normalized':'raw'}</b><br>median growth +${f3(q.median_delta10)}<br>shuffled ${f3(q.shuf_med_s1)} → ${f3(q.shuf_med_s10)}<br>real ${f3(q.real_s1)} → ${f3(q.real_s10)}`,on:()=>{setCell({c,b:r.b,nz:k.nz,sg:k.sg});$('f2a').scrollIntoView({block:'center'});}};}});}
f2a();f2b();
(function(){const H=$('f2c');for(const c of CONCEPTS){const R=DATA.Iastored.filter(q=>q.concept===c),P=Panel(H,{x:bandB,y:{d:[-.2,.6],t:[-.2,0,.2,.4,.6],f:f2,label:'real − shuffled median'},title:c});hrule(P,0,{dash:null,c:'var(--axis)'});
  for(const r of R){const a=r.real_s1-r.shuf_med_s1,b=r.real_sT-r.shuf_med_sT,x=P.X(r.block);E('line',{x1:x,x2:x,y1:P.Y(a),y2:P.Y(b),stroke:'var(--nul-soft)','stroke-width':2},P.s);mark(P.s,x,P.Y(a),'c','var(--s1)',3.8,true);if(r.T_star>1)mark(P.s,x,P.Y(b),'c','var(--s1)',4.2);
    hit(P,r.block,(a+b)/2,`<b>${c}, block ${r.block}, T* = ${r.T_star}</b><br>gap at T = 1: ${a>=0?'+':''}${f3(a)} (p ${f3(r.p_perm_1)})<br>gap at T*: ${b>=0?'+':''}${f3(b)} (p ${f3(r.p_perm_T)})`,16);}}
  legend('l2c',[{k:'c',c:'var(--s1)',hollow:1,line:false,t:'T = 1'},{k:'c',c:'var(--s1)',line:false,t:'T*'}]);
  table('t2c',[{h:'concept',v:r=>r.concept},{h:'block',v:r=>r.block,n:1},{h:'T*',v:r=>r.T_star,n:1},{h:'real s(1)',v:r=>f3(r.real_s1),n:1},{h:'shuffled s(1)',v:r=>f3(r.shuf_med_s1),n:1},{h:'p(1)',v:r=>f3(r.p_perm_1),n:1},{h:'real s(T*)',v:r=>f3(r.real_sT),n:1},{h:'shuffled s(T*)',v:r=>f3(r.shuf_med_sT),n:1},{h:'p(T*)',v:r=>f3(r.p_perm_T),n:1}],DATA.Iastored);})();
/* ---------- Check II ---------- */
(function(){let seed=7;const rnd=()=>(seed=(seed*16807)%2147483647)/2147483647,gs=()=>Math.sqrt(-2*Math.log(rnd()))*Math.cos(6.2832*rnd());
  const pts=[...Array(40)].map(()=>[1.6+.32*gs(),.25+1.15*gs()]),m=[0,1].map(i=>pts.reduce((a,p)=>a+p[i],0)/40);
  function top(P,cen){const a=P.reduce((s,p)=>s+(p[0]-cen[0])**2,0),b=P.reduce((s,p)=>s+(p[0]-cen[0])*(p[1]-cen[1]),0),d=P.reduce((s,p)=>s+(p[1]-cen[1])**2,0),th=.5*Math.atan2(2*b,a-d);return[Math.cos(th),Math.sin(th)];}
  const H=$('f3a'),ax={d:[-3,3],t:[-2,0,2]},mk=(title,Q,u)=>{const P=Panel(H,{w:320,h:280,x:{...ax},y:{...ax},title,m:{l:22,r:10,t:26,b:22}});
    hrule(P,0,{dash:null});E('line',{x1:P.X(0),x2:P.X(0),y1:P.m.t,y2:P.m.t+P.ih,class:'ax'},P.s);
    Q.forEach(p=>E('circle',{cx:P.X(p[0]),cy:P.Y(p[1]),r:2.6,fill:'var(--nul)',opacity:.55},P.s));
    const s=u[0]<0?-1:1,L=2.4;E('line',{x1:P.X(0),y1:P.Y(0),x2:P.X(s*u[0]*L),y2:P.Y(s*u[1]*L),stroke:'var(--s3)','stroke-width':2.6,'stroke-linecap':'round'},P.s);E('line',{x1:P.X(0),y1:P.Y(0),x2:P.X(-s*u[0]*L),y2:P.Y(-s*u[1]*L),stroke:'var(--s3)','stroke-width':2.6,'stroke-linecap':'round'},P.s);return P;};
  const uU=(()=>{let v=[1,0.2];for(let i=0;i<60;i++){const a=pts.reduce((s,p)=>s+p[0]*(p[0]*v[0]+p[1]*v[1]),0),b=pts.reduce((s,p)=>s+p[1]*(p[0]*v[0]+p[1]*v[1]),0),n=Math.hypot(a,b);v=[a/n,b/n];}return v;})();
  const uC=top(pts,m),Pa=mk('Uncentred D (paper): PC1 tilts toward d̄',pts,uU),Pb=mk('Centred D (old code): PC1 ignores d̄',pts.map(p=>[p[0]-m[0],p[1]-m[1]]),uC);
  E('line',{x1:Pa.X(0),y1:Pa.Y(0),x2:Pa.X(m[0]),y2:Pa.Y(m[1]),stroke:'var(--s2)','stroke-width':2.6,'stroke-linecap':'round'},Pa.s);mark(Pa.s,Pa.X(m[0]),Pa.Y(m[1]),'s','var(--s2)',4.5);
  mark(Pb.s,Pb.X(0),Pb.Y(0),'s','var(--s2)',4.5);
  legend('l3a',[{c:'var(--nul)',k:'c',line:false,t:'paired differences'},{k:'s',c:'var(--s2)',t:'mean d̄, the DiffMeans direction'},{c:'var(--s3)',t:'top principal direction'}]);
  const hd=DATA.IIhead;$('t3head').innerHTML='<thead><tr><th>concept</th><th class="n">|ρ| PCA centred</th><th class="n">|ρ| PCA uncentred</th><th class="n">|ρ| DM</th><th class="n">|ρ| RFM</th><th class="n">AUROC PCA centred</th><th class="n">AUROC PCA uncentred</th><th>break-even</th></tr></thead><tbody>'+hd.map((r,i)=>`<tr><td><code>${r.concept}</code></td><td class="n">${f3(r.rho_PCA_c)}</td><td class="n">${f3(r.rho_PCA_u_seq)}</td><td class="n">${f3(r.rho_DM)}</td><td class="n">${f3(r.rho_RFM)}</td><td class="n">${f3(r.auc_PCA_c)}</td><td class="n">${f3(r.auc_PCA_u_seq)}</td><td>${DATA.IIverd[i].break_even?'<span class="pill good">✓ passes</span>':'<span class="pill crit">✕ fails</span>'}</td></tr>`).join('')+'</tbody>';
  let mode='rho';const S={rho:[['RFM','rfm','var(--s1)','c'],['DM','rho_DM','var(--s2)','s'],['PCA uncentred','rho_PCA_u','var(--s3)','t'],['PCA centred','rho_PCA_c','var(--nul)','d']],cos:[['PCA uncentred','cos_PCA_u_DM','var(--s3)','t'],['PCA centred','cos_PCA_c_DM','var(--nul)','d']]};
  const draw=()=>{const H=$('f3b');H.innerHTML='';for(const c of CONCEPTS){const R=DATA.IIblock.filter(q=>q.concept===c).map(r=>({...r,rfm:DATA.Irho.find(x=>x.c===c&&x.b===r.block).rho_rfm})),P=Panel(H,{x:bandB,y:{...unit,label:mode==='rho'?'val |ρ|':'|cos| with DM'},title:c});
      for(const[n,k,cc,mk]of S[mode]){line(P,R.map(r=>[r.block,r[k]]),{c:cc,dash:n==='PCA centred'?'5 3':null});R.forEach(r=>mark(P.s,P.X(r.block),P.Y(r[k]),mk,cc,3.2));}
      R.forEach(r=>hit(P,r.block,.5,`<b>${c}, block ${r.block}</b><br>`+S[mode].map(([n,k])=>`${n}: ${f3(r[k])}`).join('<br>')+`<br>rank-one share ${f3(r.rank1_share)}`,24));}
    legend('l3b',S[mode].map(([t,,c,k])=>({t,c,k,dash:t==='PCA centred'?'5 3':null})));
    table('t3b',[{h:'concept',v:r=>r.concept},{h:'block',v:r=>r.block,n:1},{h:'|ρ| PCA centred',v:r=>f3(r.rho_PCA_c),n:1},{h:'|ρ| PCA uncentred',v:r=>f3(r.rho_PCA_u),n:1},{h:'|ρ| DM',v:r=>f3(r.rho_DM),n:1},{h:'|cos| centred',v:r=>f3(r.cos_PCA_c_DM),n:1},{h:'|cos| uncentred',v:r=>f3(r.cos_PCA_u_DM),n:1},{h:'rank-one share',v:r=>f3(r.rank1_share),n:1}],DATA.IIblock);};
  seg('c3b','Measure',[{v:'rho',t:'val |ρ|'},{v:'cos',t:'|cos| with DiffMeans'}],'rho',v=>{mode=v;draw();});draw();
  const R=[];for(const b of DATA.IIboot){R.push({c:b.concept,t:'RFM',v:b.RFM_minus_PCAu,lo:b.ci_lo,hi:b.ci_hi});R.push({c:b.concept,t:'DM',v:b.DM_minus_PCAu,lo:b.ci_lo_DM,hi:b.ci_hi_DM});}
  const W=640,G=$('f3d'),rowH=24,ht=30+rowH*R.length+24,s=E('svg',{viewBox:`0 0 ${W} ${ht}`,role:'img','aria-label':'Bootstrap intervals'},G),l=130,r=20,xs=v=>l+(v+.05)/(.45)*(W-l-r);
  for(const v of [0,.1,.2,.3,.4]){E('line',{x1:xs(v),x2:xs(v),y1:10,y2:ht-24,class:v?'g':'ax'},s);const t=E('text',{x:xs(v),y:ht-10,'text-anchor':'middle'},s);t.textContent=(v?'+':'')+v.toFixed(1);}
  R.forEach((q,i)=>{const y=20+rowH*i+(i>>1)*2,cc=q.t==='RFM'?'var(--s1)':'var(--s2)';const t=E('text',{x:l-10,y:y+3.5,'text-anchor':'end',class:'i'},s);t.textContent=`${q.c}, ${q.t}`;E('line',{x1:xs(q.lo),x2:xs(q.hi),y1:y,y2:y,stroke:cc,'stroke-width':2.4,'stroke-linecap':'round'},s);mark(s,xs(q.v),y,q.t==='RFM'?'c':'s',cc,4.5);
    const h=E('rect',{x:l,y:y-11,width:W-l-r,height:22,fill:'transparent'},s);bindTip(h,`<b>${q.c}: ${q.t} − PCA uncentred</b><br>${q.v>=0?'+':''}${f3(q.v)}, 95 % CI [${f3(q.lo)}, ${f3(q.hi)}]`);});
  legend('l3d',[{k:'c',c:'var(--s1)',t:'RFM − PCA uncentred'},{k:'s',c:'var(--s2)',t:'DM − PCA uncentred'}]);})();
/* ---------- Check III ---------- */
(function(){const RV=[-4,-2,-1,-.5,-.25,0,.25,.5,1,2,4,8],bandR={band:true,vals:RV,label:'steering strength r (rank spacing)',f:v=>String(v).replace('0.','.')};
  $('t3gate').innerHTML='<thead><tr><th>concept</th><th class="n">unique outputs (pos + neg)</th><th class="n">CV AUROC of judge</th><th>gate G1</th></tr></thead><tbody>'+DATA.IIIjudge.map(r=>`<tr><td><code>${r.concept}</code></td><td class="n">${r.n_pos} + ${r.n_neg}</td><td class="n">${f3(r.cv_auroc_full)}</td><td>${r.G1_pass?'<span class="pill good">✓ passes</span>':'<span class="pill crit">✕ fails: do not interpret</span>'}</td></tr>`).join('')+'</tbody>';
  table($('t3ex').parentElement,[{h:'concept',v:r=>`<code>${r.concept}</code>`},{h:'r',v:r=>r.r,n:1},{h:'output',v:r=>`<span style="font-family:var(--mono);font-size:12.5px">${r.text.replace(/</g,'&lt;')}</span>`},{h:'judge g',v:r=>f2(r.g),n:1},{h:'PPL',v:r=>r.ppl<100?f2(r.ppl):Math.round(r.ppl).toLocaleString('en-US'),n:1}],DATA.IIIex);
  const H=$('f4a');for(const c of CONCEPTS){const P=Panel(H,{x:bandR,y:{d:[-1,1],t:[-1,-.5,0,.5,1],f:f2,label:'judge g'},title:c+(c==='valence'?' (failed G1)':'')});hrule(P,0,{dash:null,c:'var(--nul)'});
    const G=DATA.IIIgen.filter(q=>q.concept===c),R=DATA.IIIr.filter(q=>q.concept===c).sort((a,b)=>a.r-b.r);G.forEach(q=>E('circle',{cx:P.X(q.r),cy:P.Y(q.g),r:2.2,fill:'var(--s1)',opacity:c==='valence'?.22:.35},P.s));
    line(P,R.map(q=>[q.r,q.g_mean]),{c:c==='valence'?'var(--nul)':'var(--s1)',dash:c==='valence'?'4 3':null});R.forEach(q=>hit(P,q.r,q.g_mean,`<b>${c}, r = ${q.r}</b><br>mean g ${f3(q.g_mean)}<br>on-concept ${q.on}/7 · coherent ${q.coherent}/7`,12));}
  legend('l4a',[{k:'c',c:'var(--s1)',line:false,t:'one generation'},{c:'var(--s1)',t:'mean over 7 prompts'}]);
  table('t4a',[{h:'concept',v:r=>r.concept},{h:'r',v:r=>r.r,n:1},{h:'mean g',v:r=>f3(r.g_mean),n:1},{h:'on-concept',v:r=>r.on+'/7',n:1},{h:'coherent',v:r=>r.coherent+'/7',n:1},{h:'both',v:r=>r.joint+'/7',n:1},{h:'median PPL',v:r=>r.ppl_med<100?f2(r.ppl_med):Math.round(r.ppl_med).toLocaleString('en-US'),n:1},{h:'mean distinct-2',v:r=>f2(r.d2_mean),n:1}],DATA.IIIr);
  const H2=$('f4b'),yl={log:true,d:[1,1e7],t:[1,10,100,1e3,1e4,1e5,1e6,1e7],f:v=>v>=1e3?'10'+'⁰¹²³⁴⁵⁶⁷'[Math.log10(v)]:String(v),label:'PPL (log scale)'};
  for(const c of CONCEPTS){const P=Panel(H2,{x:bandR,y:yl,title:c});hrule(P,DATA.pplmax,{label:c==='food'?'coherence limit 2.25':null,right:true,c:'var(--crit)'});
    DATA.IIIgen.filter(q=>q.concept===c).forEach(q=>{E('circle',{cx:P.X(q.r),cy:P.Y(q.ppl),r:2.6,fill:q.coherent?'var(--s3)':'var(--s1)',opacity:.6},P.s);});
    DATA.IIIr.filter(q=>q.concept===c).forEach(q=>hit(P,q.r,q.ppl_med,`<b>${c}, r = ${q.r}</b><br>median PPL ${q.ppl_med<100?f2(q.ppl_med):Math.round(q.ppl_med).toLocaleString('en-US')}<br>coherent ${q.coherent}/7`,14));}
  legend('l4b',[{k:'c',c:'var(--s3)',line:false,t:'coherent'},{k:'c',c:'var(--s1)',line:false,t:'not coherent'},{c:'var(--crit)',dash:'4 3',sw:1.2,t:'q95 of unsteered text'}]);
  const H3=$('f4e');for(const c of CONCEPTS){const R=DATA.IIIr.filter(q=>q.concept===c).sort((a,b)=>a.r-b.r),P=Panel(H3,{x:bandR,y:{d:[0,7],t:[0,1,3,5,7],f:v=>v+'/7',label:'generations counted'},title:c+(c==='valence'?' (failed G1)':'')});
    const S=[['old_success','var(--nul)','s','5 3'],['on','var(--s1)','c',null],['joint','var(--s2)','t',null]];
    for(const[k,cc,mk,d]of S){line(P,R.map(q=>[q.r,q[k]]),{c:cc,dash:d});R.forEach(q=>mark(P.s,P.X(q.r),P.Y(q[k]),mk,cc,2.9));}
    R.forEach(q=>hit(P,q.r,3.5,`<b>${c}, r = ${q.r}</b><br>old metric ${q.old_success}/7<br>blind judge ${q.on}/7<br>blind judge and coherent ${q.joint}/7<br>lexicon hits ${q.lex}/7`,16));}
  legend('l4e',[{k:'s',c:'var(--nul)',dash:'5 3',t:'old metric'},{k:'c',c:'var(--s1)',t:'blind judge above 0'},{k:'t',c:'var(--s2)',t:'blind judge and coherent'}]);
  table('t4e',[{h:'concept',v:r=>r.concept},{h:'old at r = 0',v:r=>r.old_r0+'/7',n:1},{h:'old, best r > 0',v:r=>r.old_best+'/7',n:1},{h:'judge at r = 0',v:r=>r.on_r0+'/7',n:1},{h:'judge, best r > 0',v:r=>r.on_best+'/7',n:1},{h:'both at r = 0',v:r=>r.joint_r0+'/7',n:1},{h:'both, best r > 0',v:r=>r.joint_best+'/7',n:1}],DATA.IIIold);
  const H4=$('f4d');for(const c of CONCEPTS){const h3=DATA.IIIh3.find(q=>q.concept===c),P=Panel(H4,{h:250,x:{d:[0,1],t:[0,.5,1],f:v=>v,label:'probe'},y:{d:[-1,1],t:[-1,0,1],label:'judge g'},title:`${c}: ρ = ${f2(h3.spearman)}, ${Math.round(h3.agree_binary*100)} % agree`});
    E('rect',{x:P.X(.5),y:P.Y(0),width:P.X(1)-P.X(.5),height:P.Y(-1)-P.Y(0),fill:'var(--crit-bg)'},P.s);hrule(P,0,{dash:null});const v=P.X(.5);E('line',{x1:v,x2:v,y1:P.m.t,y2:P.m.t+P.ih,class:'ax'},P.s);
    const t=E('text',{x:P.X(.98),y:P.Y(-.92),'text-anchor':'end',class:'i'},P.s);t.textContent=`probe yes, judge no: ${h3.probe_yes_judge_no}`;
    DATA.IIIgen.filter(q=>q.concept===c).forEach(q=>{const e=E('circle',{cx:P.X(q.probe),cy:P.Y(q.g),r:3,fill:'var(--s1)',opacity:.6,stroke:'var(--surface)','stroke-width':.6},P.s);bindTip(e,`<b>${c}, r = ${q.r}, prompt ${q.i+1}</b><br>probe ${f3(q.probe)}<br>judge ${f3(q.g)}`);});}
  legend('l4d',[{k:'c',c:'var(--s1)',line:false,t:'one generation'},{area:1,c:'var(--crit-bg)',t:'probe says yes, judge says no'}]);
  heat($('f4f'),{w:760,ml:82,ch:30,fs:10.5,mb:30,rows:CONCEPTS,cols:RV.map(v=>({v,t:String(v)})),colLabel:'steering strength r',cell:(c,k)=>{const q=DATA.IIIdup.find(x=>x.concept===c&&x.r===k.v);return{fill:seqFill(q.mean_jaccard),text:f2(q.mean_jaccard),ink:q.mean_jaccard>.55?'var(--surface)':'var(--ink)',tip:`<b>${c}, r = ${q.r}</b><br>Jaccard ${f3(q.mean_jaccard)}<br>${q.n_distinct} distinct texts of 7`};}});})();
/* ---------- Check IV ---------- */
(function(){let blk=3,con='RFM-DM',met='auc';const NV=[1,2,4,8,16,32,'max'],nmax={food:180,valence:60,env_social:180};
  const nk=(c,n)=>n==='max'?nmax[c]:n,sum=(c,b,n,ct,m)=>DATA.IVsum.find(q=>q.concept===c&&q.block===b&&q.n===nk(c,n)&&q.contrast===ct&&q.metric===m);
  const M=[['RFM','var(--s1)','c'],['DM','var(--s2)','s'],['PCA','var(--s3)','t'],['LogReg','var(--s4)','d']],lab={RFM:'RFM',DM:'DiffMeans',PCA:'PCA uncentred',LogReg:'LogReg'};
  const bx=c=>({band:true,vals:NV,label:'examples per class n',f:v=>v==='max'?nmax[c]:v});
  function a(){const H=$('f5a');H.innerHTML='';const all=DATA.IVmean.filter(q=>q.block===blk);const lo=Math.floor(Math.min(...all.map(q=>q.auc))*10)/10;const t=[];for(let v=lo;v<=1.0001;v+=lo<.6?.1:.1)t.push(+v.toFixed(1));
    for(const c of CONCEPTS){const P=Panel(H,{x:bx(c),y:{d:[lo,1],t,f:f2,label:'test AUROC'},title:c});
      for(const[m,cc,k]of M){const R=NV.map(n=>({n,q:all.find(x=>x.concept===c&&x.method===m&&x.n===nk(c,n))}));line(P,R.map(r=>[r.n,r.q.auc]),{c:cc,sw:m==='RFM'?2.6:1.8});R.forEach(r=>mark(P.s,P.X(r.n),P.Y(r.q.auc),k,cc,3.2,r.n==='max'));}
      NV.forEach(n=>{const q=M.map(([m])=>[m,all.find(x=>x.concept===c&&x.method===m&&x.n===nk(c,n)).auc]);hit(P,n,q[0][1],`<b>${c}, block ${blk}, n = ${nk(c,n)}</b><br>`+q.map(([m,v])=>`${lab[m]}: ${f3(v)}`).join('<br>'),18);});}
    legend('l5a',M.map(([m,c,k])=>({t:lab[m],c,k})));
    table('t5a',[{h:'concept',v:r=>r.c},{h:'n',v:r=>r.n,n:1},...M.map(([m])=>({h:lab[m],n:1,v:r=>f3(r[m])}))],CONCEPTS.flatMap(c=>NV.map(n=>{const o={c,n:nk(c,n)};M.forEach(([m])=>o[m]=all.find(x=>x.concept===c&&x.method===m&&x.n===nk(c,n)).auc);return o;})));}
  function b(){const H=$('f5b');H.innerHTML='';const rows=[];const gather=CONCEPTS.flatMap(c=>NV.map(n=>({c,n,q:sum(c,blk,n,con,met)})).filter(r=>r.q));const vs=gather.flatMap(r=>[r.q.ci_lo,r.q.ci_hi,r.q.delta]).filter(v=>v!=null);const N=nice(Math.min(...vs,-.02),Math.max(...vs,.02),4);
    for(const c of CONCEPTS){const R=gather.filter(r=>r.c===c),P=Panel(H,{x:bx(c),y:{d:N.d,t:N.t,f:f2,label:'Δ'},title:c});const ci=R.filter(r=>r.n!=='max');
      area(P,ci.map(r=>[r.n,r.q.ci_lo,r.q.ci_hi]),'color-mix(in oklab,var(--s1) 30%,transparent)');hrule(P,0,{dash:null,c:'var(--nul)'});line(P,ci.map(r=>[r.n,r.q.delta]),{c:'var(--s1)'});
      R.forEach(r=>{mark(P.s,P.X(r.n),P.Y(r.q.delta),'c','var(--s1)',3.4,r.n==='max');hit(P,r.n,r.q.delta,`<b>${c}, block ${blk}, n = ${nk(c,r.n)}</b><br>Δ = ${r.q.delta.toFixed(4)}<br>${r.q.ci_lo==null?'single draw, no interval':`95 % CI [${f4(r.q.ci_lo)}, ${f4(r.q.ci_hi)}]`}${r.q.holm_declared?'<br>Holm declared':''}`,16);rows.push({c,n:nk(c,r.n),...r.q});});}
    table('t5b',[{h:'concept',v:r=>r.c},{h:'n',v:r=>r.n,n:1},{h:'Δ',v:r=>f4(r.delta),n:1},{h:'CI low',v:r=>f4(r.ci_lo),n:1},{h:'CI high',v:r=>f4(r.ci_hi),n:1},{h:'Holm',v:r=>r.holm_declared?'declared':'–'}],rows);}
  function d(){const H=$('f5d');H.innerHTML='';const ns=[2,4,8,16,32],Ts=[10,9,8,7,6,5,4,3,2,1];for(const c of CONCEPTS)heat(H,{title:`${c}, block ${blk}`,rows:Ts.map(t=>({T:t,t:'T='+t})),cols:ns,ml:40,ch:19,fs:9,colLabel:'examples per class n',cell:(r,n)=>{const q=DATA.IVT.find(x=>x.concept===c&&x.block===blk&&x.n===n&&x.T===r.T);if(!q)return null;const t=q.delta/.3;return{fill:divFill(t),text:(q.is_T_star?'★':'')+(Math.abs(q.delta)<.0005?'0':q.delta.toFixed(2).replace('0.','.')),ink:Math.abs(t)>.75?'var(--surface)':'var(--ink)',stroke:q.is_T_frozen?'var(--ink)':null,tip:`<b>${c}, block ${blk}, n = ${n}, T = ${r.T}</b><br>Δ = ${q.delta.toFixed(4)}${q.is_T_frozen?'<br>frozen T':''}${q.is_T_star?'<br>oracle T*':''}`};}});}
  function all(){a();b();d();}
  seg('c5','Block (0–5; 3 is primary)',BLOCKS.map(v=>({v})),3,v=>{blk=v;all();});
  seg('c5b','Contrast',[{v:'RFM-DM',t:'RFM − DM'},{v:'RFM-LogReg',t:'RFM − LogReg'},{v:'RFM-PCA',t:'RFM − PCA'}],con,v=>{con=v;b();});
  seg('c5b','Metric',[{v:'auc',t:'AUROC'},{v:'rho',t:'|ρ|'}],met,v=>{met=v;b();});
  all();
  const H=$('f5c'),ns=[2,4,8,16,32,'max'];for(const c of CONCEPTS)heat(H,{title:c,rows:BLOCKS.map(b=>({b,t:'b'+b})),cols:ns.map(n=>({n,t:n==='max'?nmax[c]:n})),ch:27,fs:10,colLabel:'examples per class n',cell:(r,k)=>{const q=sum(c,r.b,k.n,'RFM-DM','auc');const t=q.delta/.3,one=k.n==='max',s=one?'':(q.ci_hi<0||q.ci_lo>0)?'•':'';return{fill:divFill(t),text:(q.holm_declared?'★':'')+(Math.abs(q.delta)<.0005?'0':q.delta.toFixed(2).replace('0.','.').replace('-.','−.'))+(s&&!q.holm_declared?s:''),ink:Math.abs(t)>.75?'var(--surface)':'var(--ink)',tip:`<b>${c}, block ${r.b}, n = ${nk(c,k.n)}</b><br>Δ = ${q.delta.toFixed(4)}${one?'<br>single draw, no interval':`<br>95 % CI [${q.ci_lo.toFixed(4)}, ${q.ci_hi.toFixed(4)}]`}${q.holm_declared?'<br>Holm declared':''}`};}});
  const hy=['H3a','H-artigo','H3b'];const pill=v=>/REFUTED/.test(v)?'<span class="pill crit">✕ refuted</span>':/SUPPORTED/.test(v)?'<span class="pill good">✓ supported</span>':'<span class="pill warn">◐ not refuted</span>';
  $('t5v').innerHTML='<thead><tr><th>hypothesis</th>'+BLOCKS.map(b=>`<th>block ${b}${b===3?' (primary)':''}</th>`).join('')+'</tr></thead><tbody>'+hy.map(h=>`<tr><td><b>${h}</b></td>`+BLOCKS.map(b=>`<td>${pill(DATA.IVverd.find(q=>q.block===b&&q.hypothesis===h).verdict)}</td>`).join('')+'</tr>').join('')+'</tbody>';})();
