const NS='http://www.w3.org/2000/svg',CONCEPTS=['food','valence','env_social'],BLOCKS=[0,1,2,3,4,5];
const tip=document.getElementById('tip');
const br=x=>x.replace('.',','),f2=v=>v==null?'–':br((+v).toFixed(2)),f3=v=>v==null?'–':br((+v).toFixed(3));
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
const bandB={band:true,vals:BLOCKS,label:'bloco'};
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
const sg=v=>(v>=0?'+':'')+f3(v);
/* ---------- Check I ---------- */
let regime='A';
function f1a(){const H=$('f1a');H.innerHTML='';const R=regime==='A'?DATA.IA:DATA.IB;
  for(const c of CONCEPTS){const P=Panel(H,{x:bandB,y:{...unit,label:'share s'},title:c});hrule(P,.95,{label:c==='food'?'s = 0,95':null});
    for(const r of R.filter(q=>q.c===c)){const x=P.X(r.b);E('rect',{x:x-7,y:P.Y(r.hi),width:14,height:Math.max(2,P.Y(r.lo)-P.Y(r.hi)),rx:3,fill:'var(--nul-soft)'},P.s);
      E('line',{x1:x-7,x2:x+7,y1:P.Y(r.med),y2:P.Y(r.med),stroke:'var(--nul)','stroke-width':2},P.s);
      mark(P.s,x,P.Y(r.s),'c','var(--s1)',4.5,r.refuted);
      hit(P,r.b,(r.s+r.med)/2,`<b>${c}, bloco ${r.b}</b><br>s real = ${f3(r.s)}<br>mediana embaralhada ${f3(r.med)}, faixa ${f3(r.lo)}–${f3(r.hi)}<br>p = ${f3(r.p)} · ${r.refuted?'refutada':'não refutada'}`,16);}}
  table('t1a',[{h:'conceito',v:r=>r.c},{h:'bloco',v:r=>r.b,n:1},{h:'s real',v:r=>f3(r.s),n:1},{h:'mediana embaralhada',v:r=>f3(r.med),n:1},{h:'mín',v:r=>f3(r.lo),n:1},{h:'máx',v:r=>f3(r.hi),n:1},{h:'p',v:r=>f3(r.p),n:1},{h:'refutada',v:r=>r.refuted?'sim':'não'}],R);}
seg('c1a','Regime',[{v:'A',t:'A: HP congelados, B = 100'},{v:'B',t:'B: sweep completo, B = 20'}],'A',v=>{regime=v;f1a();});
legend('l1a',[{bar:1,c:'var(--nul-soft)',t:'faixa embaralhada'},{c:'var(--nul)',t:'mediana embaralhada'},{k:'c',c:'var(--s1)',line:false,t:'s real, célula não refutada'},{k:'c',c:'var(--s1)',hollow:1,line:false,t:'s real, célula refutada'}]);f1a();
(function(){const H=$('f1b');for(const c of CONCEPTS){const R=DATA.Irho.filter(q=>q.c===c),B=DATA.IB.filter(q=>q.c===c),P=Panel(H,{x:bandB,y:{...unit,label:'val |ρ|'},title:c});
  area(P,R.map(r=>[r.b,r.rand_med,r.rand_q95]));
  const S=[['rho_rfm','var(--s1)','c'],['diffmeans','var(--s2)','s'],['pc1','var(--s3)','t'],['rho_shufA','var(--nul)','d']];
  line(P,B.map(r=>[r.b,r.rho_sh]),{c:'var(--nul)',dash:'2 3',sw:1.6});
  for(const[k,cc,mk]of S){line(P,R.map(r=>[r.b,r[k]]),{c:cc});R.forEach(r=>mark(P.s,P.X(r.b),P.Y(r[k]),mk,cc,3.2));}
  R.forEach((r,i)=>{hit(P,r.b,.5,`<b>${c}, bloco ${r.b}</b><br>RFM real ${f3(r.rho_rfm)}<br>DiffMeans ${f3(r.diffmeans)}<br>PC1 ${f3(r.pc1)}<br>RFM embaralhado, A ${f3(r.rho_shufA)}<br>RFM embaralhado, B ${f3(B[i].rho_sh)}<br>aleatória: mediana ${f3(r.rand_med)}, q95 ${f3(r.rand_q95)}`,24);});}
  legend('l1b',[{k:'c',c:'var(--s1)',t:'RFM, rótulos reais'},{k:'s',c:'var(--s2)',t:'DiffMeans'},{k:'t',c:'var(--s3)',t:'PC1, sem rótulos'},{k:'d',c:'var(--nul)',t:'RFM embaralhado, Regime A'},{c:'var(--nul)',dash:'2 3',t:'RFM embaralhado, Regime B'},{area:1,c:'var(--nul-soft)',t:'direções aleatórias, mediana a q95'}]);
  table('t1b',[{h:'conceito',v:r=>r.c},{h:'bloco',v:r=>r.b,n:1},{h:'RFM',v:r=>f3(r.rho_rfm),n:1},{h:'DM',v:r=>f3(r.diffmeans),n:1},{h:'PC1',v:r=>f3(r.pc1),n:1},{h:'embaralhado A',v:r=>f3(r.rho_shufA),n:1},{h:'mediana aleatória',v:r=>f3(r.rand_med),n:1},{h:'q95 aleatória',v:r=>f3(r.rand_q95),n:1}],DATA.Irho);})();
/* ---------- Check Ia: aggregate over 108 cells ---------- */
(function(){const G=DATA.Iaagg,T=[1,2,3,4,5,6,7,8,9,10],H=$('f2z');
  const P=Panel(H,{w:640,h:300,x:{d:[1,10],t:T,label:'iteração T'},y:{...unit,label:'share s(T)'},title:'Mediana nas 108 células',m:{l:44,r:112,t:26,b:36}});
  area(P,T.map((t,i)=>[t,G.shuf_lo[i],G.shuf_hi[i]]));
  area(P,T.map((t,i)=>[t,G.real_lo[i],G.real_hi[i]]),'color-mix(in oklab,var(--s1) 22%,transparent)');
  line(P,T.map((t,i)=>[t,G.shuf[i]]),{c:'var(--nul)'});line(P,T.map((t,i)=>[t,G.real[i]]),{c:'var(--s1)',sw:2.4});
  mark(P.s,P.X(10),P.Y(G.real[9]),'c','var(--s1)',4.5);mark(P.s,P.X(10),P.Y(G.shuf[9]),'c','var(--nul)',4);
  const lab=(y,a,b)=>{const t=E('text',{x:P.X(10)+10,y:P.Y(y)-1,class:'i'},P.s);t.textContent=a;const u=E('text',{x:P.X(10)+10,y:P.Y(y)+12,class:'i'},P.s);u.textContent=b;};
  lab(G.real[9],'reais',f2(G.real[9]));lab(G.shuf[9],'embaralhados',f2(G.shuf[9]));
  T.forEach((t,i)=>hit(P,t,(G.real[i]+G.shuf[i])/2,`<b>T = ${t}</b><br>real: mediana ${f3(G.real[i])}, q05–q95 ${f3(G.real_lo[i])}–${f3(G.real_hi[i])}<br>embaralhado: mediana ${f3(G.shuf[i])}, q05–q95 ${f3(G.shuf_lo[i])}–${f3(G.shuf_hi[i])}<br>diferença mediana ${sg(G.gap[i])}`,20));
  legend('l2z',[{c:'var(--s1)',t:'rótulos reais'},{c:'var(--nul)',t:'rótulos embaralhados'},{area:1,c:'color-mix(in oklab,var(--s1) 22%,transparent)',t:'q05–q95 entre células, reais'},{area:1,c:'var(--nul-soft)',t:'q05–q95 entre células, embaralhados'}]);
  table('t2z',[{h:'T',v:r=>r,n:1},{h:'s real, mediana',v:r=>f3(G.real[r-1]),n:1},{h:'s embaralhado, mediana',v:r=>f3(G.shuf[r-1]),n:1},{h:'diferença mediana por célula',v:r=>sg(G.gap[r-1]),n:1}],T);})();
/* ---------- Check Ia: gap at T=1 vs T* ---------- */
(function(){const H=$('f2c');for(const c of CONCEPTS){const R=DATA.Iastored.filter(q=>q.concept===c),P=Panel(H,{x:bandB,y:{d:[-.2,.6],t:[-.2,0,.2,.4,.6],f:f2,label:'real − mediana embaralhada'},title:c});hrule(P,0,{dash:null,c:'var(--axis)'});
  for(const r of R){const a=r.real_s1-r.shuf_med_s1,b=r.real_sT-r.shuf_med_sT,x=P.X(r.block);E('line',{x1:x,x2:x,y1:P.Y(a),y2:P.Y(b),stroke:'var(--nul-soft)','stroke-width':2},P.s);mark(P.s,x,P.Y(a),'c','var(--s1)',3.8,true);if(r.T_star>1)mark(P.s,x,P.Y(b),'c','var(--s1)',4.2);
    hit(P,r.block,(a+b)/2,`<b>${c}, bloco ${r.block}, T* = ${r.T_star}</b><br>gap em T = 1: ${sg(a)} (p ${f3(r.p_perm_1)})<br>gap em T*: ${sg(b)} (p ${f3(r.p_perm_T)})`,16);}}
  legend('l2c',[{k:'c',c:'var(--s1)',hollow:1,line:false,t:'T = 1'},{k:'c',c:'var(--s1)',line:false,t:'T*'}]);
  table('t2c',[{h:'conceito',v:r=>r.concept},{h:'bloco',v:r=>r.block,n:1},{h:'T*',v:r=>r.T_star,n:1},{h:'s(1) real',v:r=>f3(r.real_s1),n:1},{h:'s(1) embaralhado',v:r=>f3(r.shuf_med_s1),n:1},{h:'p(1)',v:r=>f3(r.p_perm_1),n:1},{h:'s(T*) real',v:r=>f3(r.real_sT),n:1},{h:'s(T*) embaralhado',v:r=>f3(r.shuf_med_sT),n:1},{h:'p(T*)',v:r=>f3(r.p_perm_T),n:1}],DATA.Iastored);})();
/* ---------- Check Ia: any of the 108 cells ---------- */
const cell={c:'valence',b:3,nz:0,sg:10};const ctl={};
function storedOf(){return DATA.Iastored.find(r=>r.concept===cell.c&&r.block===cell.b&&r.normalize===cell.nz&&r.sigma===cell.sg);}
function f2a(){const H=$('f2a');H.innerHTML='';const C=DATA.Iacur[`${cell.c}|${cell.b}|${cell.nz}|${cell.sg}`],T=[1,2,3,4,5,6,7,8,9,10],st=storedOf();
  const P=Panel(H,{w:640,h:280,x:{d:[1,10],t:T,label:'iteração T'},y:{...unit,label:'share s(T)'},title:`${cell.c}, bloco ${cell.b}, σ = ${cell.sg}, linhas ${cell.nz?'normalizadas':'brutas'}${st?' · célula guardada':''}`,m:{l:44,r:16,t:26,b:36}});
  area(P,T.map((t,i)=>[t,C.lo[i],C.hi[i]]));line(P,T.map((t,i)=>[t,C.med[i]]),{c:'var(--nul)'});line(P,T.map((t,i)=>[t,C.real[i]]),{c:'var(--s1)',sw:2.4});
  if(st){const x=P.X(st.T_star);E('line',{x1:x,x2:x,y1:P.m.t,y2:P.m.t+P.ih,stroke:'var(--axis)','stroke-dasharray':'4 3'},P.s);const t=E('text',{x:x+(st.T_star>8?-4:4),y:P.m.t+11,class:'i','text-anchor':st.T_star>8?'end':'start'},P.s);t.textContent='T* guardado = '+st.T_star;}
  mark(P.s,P.X(10),P.Y(C.real[9]),'c','var(--s1)',4.5);mark(P.s,P.X(10),P.Y(C.med[9]),'c','var(--nul)',4);
  T.forEach((t,i)=>hit(P,t,(C.real[i]+C.med[i])/2,`<b>T = ${t}</b><br>real ${f3(C.real[i])}<br>mediana embaralhada ${f3(C.med[i])}<br>faixa embaralhada ${f3(C.lo[i])}–${f3(C.hi[i])}`,20));
  table('t2a',[{h:'T',v:r=>r,n:1},{h:'real',v:r=>f3(C.real[r-1]),n:1},{h:'mediana embaralhada',v:r=>f3(C.med[r-1]),n:1},{h:'mín embaralhado',v:r=>f3(C.lo[r-1]),n:1},{h:'máx embaralhado',v:r=>f3(C.hi[r-1]),n:1}],T);}
function setCell(o){Object.assign(cell,o);ctl.c.set(cell.c);ctl.b.set(cell.b);ctl.nz.set(cell.nz);ctl.sg.set(cell.sg);f2a();f2b();}
ctl.c=seg('c2a','Conceito',CONCEPTS.map(v=>({v})),cell.c,v=>setCell({c:v}));
ctl.b=seg('c2a','Bloco',BLOCKS.map(v=>({v})),cell.b,v=>setCell({b:v}));
ctl.sg=seg('c2a','σ',[1,10,100].map(v=>({v})),cell.sg,v=>setCell({sg:v}));
ctl.nz=seg('c2a','Linhas',[{v:0,t:'brutas'},{v:1,t:'normalizadas'}],cell.nz,v=>setCell({nz:v}));
legend('l2a',[{c:'var(--s1)',t:'rótulos reais'},{c:'var(--nul)',t:'mediana embaralhada'},{area:1,c:'var(--nul-soft)',t:'faixa embaralhada, 20 permutações'}]);
function f2b(){const H=$('f2b');H.innerHTML='';const cols=[[0,1],[0,10],[0,100],[1,1],[1,10],[1,100]].map(([nz,sg])=>({nz,sg,t:(nz?'n':'b')+sg}));
  for(const c of CONCEPTS)heat(H,{title:c,rows:BLOCKS.map(b=>({b,t:'b'+b})),cols,colLabel:'b = bruta, n = normalizada, número = σ',cell:(r,k)=>{const q=DATA.Iacell.find(x=>x.concept===c&&x.block===r.b&&x.normalize===k.nz&&x.sigma===k.sg),t=q.median_delta10/.6,sel=cell.c===c&&cell.b===r.b&&cell.nz===k.nz&&cell.sg===k.sg;
    return{fill:seqFill(t),text:'+'+f2(q.median_delta10),ink:t>.55?'var(--surface)':'var(--ink)',stroke:sel?'var(--s2)':null,tip:`<b>${c}, bloco ${r.b}, σ ${k.sg}, ${k.nz?'normalizada':'bruta'}</b><br>crescimento mediano +${f3(q.median_delta10)}<br>embaralhado ${f3(q.shuf_med_s1)} → ${f3(q.shuf_med_s10)}<br>real ${f3(q.real_s1)} → ${f3(q.real_s10)}`,on:()=>{setCell({c,b:r.b,nz:k.nz,sg:k.sg});$('f2a').scrollIntoView({block:'center'});}};}});}
f2a();f2b();
