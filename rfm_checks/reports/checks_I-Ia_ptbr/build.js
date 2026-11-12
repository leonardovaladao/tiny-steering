// Usage: node build.js <katex-dir>. Renders $..$ and $$..$$ at build time, inlines KaTeX CSS with fonts, data and charts.
const fs=require('fs'),path=require('path'),K=process.argv[2],katex=require(path.join(K,'dist/katex.js'));
let html=fs.readFileSync('src.html','utf8'),css=fs.readFileSync(path.join(K,'dist/katex.min.css'),'utf8');
css=css.replace(/@font-face\{[^}]*\}/g,m=>{const f=m.match(/url\(fonts\/([^)]+\.woff2)\)/)[1],b=fs.readFileSync(path.join(K,'dist/fonts',f)).toString('base64');
  return m.replace(/src:[^;}]*/,`src:url(data:font/woff2;base64,${b}) format("woff2")`);});
const r=(t,d)=>katex.renderToString(t,{displayMode:d,throwOnError:true,strict:false,output:'html'});
const h=html.indexOf('<body>'),s=html.indexOf('<script>const DATA');
let body=html.slice(h,s).replace(/\$\$([\s\S]+?)\$\$/g,(m,t)=>r(t,true)).replace(/\$([^$\n]+?)\$/g,(m,t)=>r(t,false));
html=html.slice(0,h)+body+html.slice(s);
html=html.replace('/*KATEX*/',()=>css).replace('/*DATA*/',()=>fs.readFileSync('data.json','utf8')).replace('/*CHARTS*/',()=>fs.readFileSync('charts.js','utf8'));
fs.writeFileSync('index.html',html);console.log('index.html KB',Math.round(html.length/1024));
