import json
from pathlib import Path

from redundancy_project.src.loader import load_run_result
from redundancy_project.src.metric_definitions import (
    F_C_INTERVENTION,
    METRIC_DEFINITION_VERSION,
    P_C_DEFINITION,
)


_DASHBOARD_TEMPLATE = r"""<!doctype html>
<html lang="en-GB">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Temporal measurement dashboard</title>
<style>
*{box-sizing:border-box}
body{font:14px system-ui,sans-serif;margin:8px;background:#f4f6fa;color:#172033;height:calc(100vh - 16px);overflow:hidden}
h1{font-size:21px;line-height:1.2;margin:0 0 2px}
p{font-size:12px;color:#4c5870;margin:0}
.controls,.grid{max-width:1600px;margin:6px auto}
.controls{height:38px;display:flex;gap:10px;align-items:center;background:white;padding:6px 12px;border-radius:8px}
input[type=range]{flex:1;min-width:80px}
.grid{height:calc(100vh - 108px);display:grid;grid-template-columns:repeat(2,minmax(0,1fr));grid-template-rows:repeat(4,minmax(0,1fr));gap:7px;position:relative}
section{min-width:0;min-height:0;display:flex;flex-direction:column;background:var(--run-tint,#fff);border-radius:8px;padding:6px 8px;box-shadow:0 1px 5px #18284814;overflow:hidden}
h2{font-size:12px;line-height:1.15;margin:0 0 1px}
.measure-panel{position:relative;cursor:zoom-in}
.measure-panel::after{content:"⤢";position:absolute;top:6px;right:9px;color:#71809a;font-size:13px;pointer-events:none}
.measure-panel:focus-visible{outline:2px solid #2166ac;outline-offset:1px}
.measure-panel.enlarged{position:absolute;left:0;z-index:5;width:100%;cursor:zoom-out;box-shadow:0 5px 24px #18284844}
.measure-panel.enlarged::after{content:"↙";z-index:1}
.matrix-panel>h2{position:absolute;top:6px;left:8px;z-index:2;pointer-events:none}
.small{font-size:10px;line-height:1.25;color:#536078}
canvas{display:block;width:100%;height:100%;min-height:0;flex:1}
.split{font-size:10px;margin:1px 8px 0 0}
.matrix-content{display:flex;align-items:stretch;gap:5px;min-height:0;flex:1}
.matrix-panel .matrix-content{position:absolute;inset:3px 8px}
.matrix-help{flex:0 1 22%;min-width:66px;max-width:130px}
.matrix-panel .matrix-help{padding-top:20px}
.rl-help{flex:0 0 66px;max-width:66px;min-width:66px;font-size:9px}
.matrix-content canvas{width:0;height:100%;min-width:0;flex:1}
.split-options{display:flex;flex-direction:column;align-items:flex-start;gap:1px;margin-top:5px;white-space:nowrap}
.split-options .split{margin-right:0}
.split-options input{margin:0 2px 0 0}
.embedded{height:100%;overflow:hidden}
.embedded h1,.embedded #runInfo,.embedded .controls{display:none}
.embedded .grid{height:calc(100vh - 16px);margin:0 auto}
@media(max-width:500px){body{overflow:auto}.grid{height:auto;grid-template-columns:1fr;grid-template-rows:none}.grid section{height:155px}.controls{position:sticky;top:0;z-index:1}}
</style>
</head>
<body>
<h1>Run evolution</h1>
<p id="runInfo"></p>
<div class="controls">
 <label for="epochSlider">Epoch:</label><input id="epochSlider" type="range">
 <strong id="epochLabel"></strong><span class="small" id="epochHint"></span>
 <span class="small" id="representationStatus"></span>
 <span class="small" id="timingLabel"></span>
</div>
<div class="grid">
 <section class="measure-panel" tabindex="0"><h2>Loss</h2><canvas id="lossChart"></canvas></section>
 <section class="measure-panel" tabindex="0"><h2>Accuracy</h2><canvas id="accuracyChart"></canvas></section>
 <section class="measure-panel matrix-panel" tabindex="0"><h2>P_L</h2><div class="matrix-content"><div class="small matrix-help">−1: opposite directions<br>0: neutral score<br>+1: aligned</div><canvas class="heatmap" id="plChart"></canvas></div></section>
 <section class="measure-panel" tabindex="0"><h2>P_C</h2><canvas id="pcChart"></canvas></section>
 <section class="measure-panel matrix-panel" tabindex="0">
  <h2>R_L</h2>
  <div class="matrix-content">
   <div class="small matrix-help rl-help" title="−1: anticorrelated · 0: no linear correlation · +1: correlated · black: undefined">−1 anti<br>0 neutral<br>+1 corr.<br>black N/A
    <div class="split-options"><label class="split"><input type="radio" name="rlSplit" value="train" checked>Train</label>
    <label class="split"><input type="radio" name="rlSplit" value="validation">Validation</label></div>
   </div>
   <canvas class="heatmap" id="rlChart"></canvas>
  </div>
 </section>
 <section class="measure-panel" tabindex="0"><h2>R_C</h2><canvas id="rcChart"></canvas></section>
 <section class="measure-panel" tabindex="0"><h2>F_L</h2><canvas id="flChart"></canvas></section>
 <section class="measure-panel" tabindex="0"><h2>F_C</h2><p class="small fc-interpretation" id="fcInterpretation"></p><canvas id="fcChart"></canvas></section>
</div>
<script>
const DATA=__DATA__;
const epochs=DATA.measures.map(x=>x.training_iteration);
const slider=document.getElementById("epochSlider");
slider.min=0;slider.max=epochs.length-1;slider.step=1;slider.value=epochs.length-1;
const embedded=new URLSearchParams(window.location.search).get("embedded")==="1";
if(embedded){
 document.body.classList.add("embedded");
 const tint=new URLSearchParams(window.location.search).get("tint");
 if(tint&&/^hsl\(\d+(?:\.\d+)? 65% 88%\)$/.test(tint)){
  document.documentElement.style.setProperty("--run-tint",tint);
 }
}
if(epochs.length===1){
 slider.disabled=true;
 document.getElementById("epochHint").textContent=
  "— only one epoch measured (increase training.epochs in experiment_design.yaml)";
}
document.getElementById("runInfo").textContent=
 `Run ${DATA.run_id}${DATA.run_details.length?` — ${DATA.run_details.join(" · ")}`:""} — `
 +`${DATA.sample_info.train_count} training examples and ${DATA.sample_info.validation_count} validation examples per epoch.`;
function setup(canvas){
 const dpr=window.devicePixelRatio||1, rect=canvas.getBoundingClientRect();
 canvas.width=Math.max(1,Math.round(rect.width*dpr));canvas.height=Math.max(1,Math.round(rect.height*dpr));
 const c=canvas.getContext("2d");c.scale(dpr,dpr);return {c,w:rect.width,h:rect.height};
}
function lineChart(id,series,opts={}){
 const {c,w,h}=setup(document.getElementById(id));c.clearRect(0,0,w,h);
 const left=43,right=opts.right_ylabel?48:8,bottom=26,pw=w-left-right;
 const leftSeries=series.filter(s=>s.axis!=="right");
 const rightSeries=series.filter(s=>s.axis==="right");
 const leftValues=leftSeries.flatMap(s=>s.values.filter(Number.isFinite));
 const rightValues=rightSeries.flatMap(s=>s.values.filter(Number.isFinite));
 if(!leftValues.length&&!rightValues.length){c.fillText("No data",left,40);return}
 let ymin=opts.min===undefined?Math.min(...leftValues):opts.min;
 let ymax=opts.max===undefined?Math.max(...leftValues):opts.max;
 if(ymin===ymax){ymin-=.5;ymax+=.5}
 if(opts.min===undefined){const pad=(ymax-ymin)*.08;ymin-=pad;ymax+=pad}
 let rightMin=rightValues.length
  ?(opts.right_min===undefined?Math.min(...rightValues):opts.right_min):0;
 let rightMax=rightValues.length
  ?(opts.right_max===undefined?Math.max(...rightValues):opts.right_max):1;
 if(rightMin===rightMax){rightMin-=.5;rightMax+=.5}
 if(opts.right_min===undefined&&rightValues.length){const pad=(rightMax-rightMin)*.08;rightMin-=pad;rightMax+=pad}
 const xcount=Math.max(...series.map(s=>s.values.length),1);
 c.strokeStyle="#cbd2df";c.fillStyle="#526078";c.lineWidth=1;c.font="11px system-ui";
 c.font=opts.legendFont||"10px system-ui";c.textAlign="left";
 const ylabelWidth=opts.ylabel?c.measureText(opts.ylabel).width:0;
 const rightLabelWidth=opts.right_ylabel?c.measureText(opts.right_ylabel).width:0;
 const legendAvailable=w-6;
 const legendRows=[];
 if(opts.legendGroups){
  opts.legendGroups.forEach(group=>{
   if(opts.legendSamplesBelow){
    legendRows.push(group.map(item=>({series:item,width:0})));
    return;
   }
   let row=[],rowWidth=0;
   group.forEach(item=>{
    const entry={series:item,width:29+c.measureText(item.name).width};
    if(row.length&&rowWidth+entry.width>legendAvailable){
     legendRows.push(row);row=[];rowWidth=0;
    }
    row.push(entry);rowWidth+=entry.width;
   });
   if(row.length)legendRows.push(row);
  });
 }else{
  legendRows.push([]);
  const legendRowWidths=[0];
  series.forEach(s=>{
   const itemWidth=29+c.measureText(s.name).width;
   let row=legendRows.length-1;
   if(legendRowWidths[row]+itemWidth>legendAvailable&&legendRows[row].length){
    legendRows.push([]);legendRowWidths.push(0);row++;
   }
   legendRows[row].push({series:s,width:itemWidth});
   legendRowWidths[row]+=itemWidth;
  });
 }
 const legendRowWidths=legendRows.map(row=>
  row.reduce((width,item)=>width+item.width,0)
 );
 const legendWidth=legendRowWidths[0];
 const oneLine=!opts.legendBelow&&Boolean(opts.ylabel)&&legendRows.length===1
  &&3+ylabelWidth+12+legendWidth
  +(opts.right_ylabel?rightLabelWidth+12:0)<=w-3;
 let top=opts.legendSamplesBelow?43:opts.legendBelow?22+legendRows.length*13:oneLine?21:17+legendRows.length*14+4,ph=h-top-bottom;
 const chartTop=top+4,chartBottom=top+ph-4;
 for(let i=0;i<=4;i++){
  let y=chartTop+(chartBottom-chartTop)*i/4;
  c.beginPath();c.moveTo(left,y);c.lineTo(left+pw,y);c.stroke();
  c.textAlign="left";c.fillText((ymax-(ymax-ymin)*i/4).toPrecision(3),3,y+4);
  if(rightSeries.length){
   c.textAlign="right";
   c.fillText((rightMax-(rightMax-rightMin)*i/4).toPrecision(3),w-3,y+4);
  }
 }
 c.font=opts.legendFont||"10px system-ui";
 if(opts.legendSamplesBelow){
  const row=legendRows[0]||[];
  const slotWidth=(w-12)/Math.max(1,row.length);
  const labelY=24,sampleY=35;
  c.textAlign="center";
  row.forEach(({series:item},index)=>{
   const centre=6+slotWidth*(index+.5);
   c.fillStyle="#354158";
   c.fillText(item.name,centre,labelY,Math.max(1,slotWidth-4));
   c.strokeStyle=item.color;c.lineWidth=2;c.setLineDash(item.dash||[]);
   c.beginPath();c.moveTo(centre-9,sampleY);c.lineTo(centre+9,sampleY);c.stroke();
   c.setLineDash([]);
  });
 }else{
 legendRows.forEach((row,rowIndex)=>{
  let legendX=oneLine
   ?3+ylabelWidth+12
   :Math.max(3,(w-legendRowWidths[rowIndex])/2);
  const legendY=opts.legendBelow?23+rowIndex*13:oneLine?10:22+rowIndex*14;
  row.forEach(({series:item,width})=>{
   c.strokeStyle=item.color;c.lineWidth=2;c.setLineDash(item.dash||[]);
   c.beginPath();c.moveTo(legendX,legendY);c.lineTo(legendX+11,legendY);c.stroke();
   c.setLineDash([]);
   c.fillStyle="#354158";c.fillText(item.name,legendX+18,legendY+2);
   legendX+=width;
  });
 });
 }
 if(opts.ylabel){c.textAlign="left";c.fillStyle="#526078";c.fillText(opts.ylabel,3,12)}
 if(opts.right_ylabel){c.textAlign="right";c.fillStyle="#526078";c.fillText(opts.right_ylabel,w-3,12)}
 c.textAlign="center";
 const tickCount=Math.min(5,xcount);
 for(let i=0;i<tickCount;i++){
  const point=tickCount===1?0:Math.round((xcount-1)*i/(tickCount-1));
  const x=left+pw*point/Math.max(1,xcount-1);
  const label=opts.xTickLabels
   ? opts.xTickLabels[point]
   : opts.xlabel==="Epoch"
    ? DATA.measures[point].training_iteration
    : point;
  c.fillText(String(label),x,h-15);
 }
 c.textAlign="center";
 c.fillText(opts.xlabel||"",left+pw/2,h-3);
 series.forEach(s=>{
  const isRight=s.axis==="right",seriesMin=isRight?rightMin:ymin,seriesMax=isRight?rightMax:ymax;
  c.strokeStyle=s.color;c.lineWidth=2;c.setLineDash(s.dash||[]);c.beginPath();let begun=false;
  s.values.forEach((v,i)=>{if(!Number.isFinite(v))return;const x=left+(xcount<=1?0:pw*i/(xcount-1));const y=chartTop+(chartBottom-chartTop)*(seriesMax-v)/(seriesMax-seriesMin);if(!begun){c.moveTo(x,y);begun=true}else c.lineTo(x,y)});
  c.stroke();
  c.setLineDash([]);
  if(xcount===1){
   const value=s.values[0];
   if(Number.isFinite(value)){
    const x=left+pw/2,y=chartTop+(chartBottom-chartTop)*(seriesMax-value)/(seriesMax-seriesMin);
    c.fillStyle=s.color;c.beginPath();c.arc(x,y,3,0,2*Math.PI);c.fill();
   }
  }
  if(s.errors)s.values.forEach((v,i)=>{if(!Number.isFinite(v))return;const x=left+(xcount<=1?0:pw*i/(xcount-1));const e=s.errors[i]||0;const y1=chartTop+(chartBottom-chartTop)*(seriesMax-Math.min(seriesMax,v+e))/(seriesMax-seriesMin),y2=chartTop+(chartBottom-chartTop)*(seriesMax-Math.max(seriesMin,v-e))/(seriesMax-seriesMin);c.beginPath();c.moveTo(x,y1);c.lineTo(x,y2);c.stroke()});
 });
 if(opts.xlabel==="Epoch"){
  const epochIndex=DATA.measures.findIndex(
   measure=>measure.training_iteration===selected().training_iteration
  );
  if(epochIndex>=0&&epochIndex<xcount){
   const x=left+(xcount<=1?pw/2:pw*epochIndex/(xcount-1));
   c.save();c.strokeStyle="#26354d";c.lineWidth=1;c.setLineDash([4,3]);
   c.beginPath();c.moveTo(x,chartTop);c.lineTo(x,chartBottom);c.stroke();
   c.restore();
  }
 }
}
function heatmap(id,matrix,min,max,unavailableMessage="No pairwise correlation matrix for this sample"){
 const {c,w,h}=setup(document.getElementById(id));c.clearRect(0,0,w,h);
 if(!Array.isArray(matrix)){
  c.fillStyle="#526078";c.font="11px system-ui";
  c.fillText(unavailableMessage,4,24);return;
 }
 const n=matrix.length, left=24,top=2,size=Math.max(0,Math.floor(Math.min(w-70,h-12)));
 if(size===0){c.fillStyle="#526078";c.font="11px system-ui";c.fillText("Matrice trop petite",4,18);return}
 const cell=size/n;
 for(let i=0;i<n;i++)for(let j=0;j<n;j++){
  const v=matrix[i][j];c.fillStyle=v===null?"#111111":color(v,min,max);
  c.fillRect(left+j*cell,top+i*cell,Math.ceil(cell),Math.ceil(cell));
 }
 c.fillStyle="#43506a";c.font="9px system-ui";c.textAlign="center";
 for(let i=0;i<n;i+=5){
  c.fillText(String(i),left+i*cell+cell/2,top+size+7);
  c.textAlign="right";c.fillText(String(i),left-5,top+i*cell+cell/2+3);c.textAlign="center";
 }
 const barX=left+size+12,barWidth=12;
 const gradient=c.createLinearGradient(0,top+size,0,top);
 gradient.addColorStop(0,color(min,min,max));
 gradient.addColorStop(.5,color((min+max)/2,min,max));
 gradient.addColorStop(1,color(max,min,max));
 c.fillStyle=gradient;c.fillRect(barX,top,barWidth,size);
 c.fillStyle="#43506a";c.font="10px system-ui";c.textAlign="left";
 c.fillText("+1",barX+15,top+4);c.fillText("0",barX+15,top+size/2+3);c.fillText("−1",barX+15,top+size);
}
function color(v,min,max){
 const t=Math.max(0,Math.min(1,(v-min)/(max-min)));
 if(t<.5){const q=t*2;return `rgb(${Math.round(45+210*q)},${Math.round(90+155*q)},${Math.round(170+85*q)})`}
 const q=(t-.5)*2;return `rgb(${Math.round(255-30*q)},${Math.round(245-180*q)},${Math.round(255-170*q)})`;
}
function selected(){return DATA.measures[Number(slider.value)]}
function update(){
 const idx=Number(slider.value), m=selected(), ep=m.training_iteration;
 document.getElementById("epochLabel").textContent=ep;
 const statusSummary=split=>{
  const status=m[split],name=status.representation_status||"status unavailable";
  const rank=Number.isFinite(status.numerical_rank)
   ?`, rank ${status.numerical_rank}/${m.activation_width||20}`:"";
  const units=Number.isFinite(status.constant_unit_count)
   ?`, ${status.constant_unit_count} constant units`:"";
  return `${split}: ${name}${rank}${units}`;
 };
 document.getElementById("representationStatus").textContent=
  `${statusSummary("train")}; ${statusSummary("validation")}`;
 const fcContexts=["train","validation"].map(split=>{
  const status=m[split];
  if(!(status.total_activation_variance>0))
   return `${split}: no activation variance; F_C unavailable`;
  if(status.nullity>0)
   return `${split}: ${status.nullity} zero-variance axes; PCA basis non-unique, centred removal has ~0 effect on this sample`;
  return null;
 }).filter(Boolean);
 document.getElementById("fcInterpretation").textContent=fcContexts.join(" · ");
 document.getElementById("timingLabel").textContent=
  `— P/R/F measurements: ${m.measurement_seconds.toFixed(2)} s`;
 lineChart("lossChart",[
  {name:"Training fixed",values:DATA.metrics.map(x=>x.train_eval_loss),color:"#2166ac"},
  {name:"Validation fixed",values:DATA.metrics.map(x=>x.validation_eval_loss),color:"#b2182b"}],{xlabel:"Epoch",ylabel:"NLL (nats/example)"});
 lineChart("accuracyChart",[
  {name:"Training fixed",values:DATA.metrics.map(x=>x.train_eval_acc),color:"#2166ac"},
  {name:"Validation fixed",values:DATA.metrics.map(x=>x.validation_eval_acc),color:"#b2182b"}],{min:0,max:1,xlabel:"Epoch",ylabel:"Accuracy (0–1)"});
 heatmap("plChart",m.P_L,-1,1);
 lineChart("pcChart",[{name:"P_C",values:DATA.measures.map(x=>x.P_C),color:"#4d9221"}],{xlabel:"Epoch",ylabel:"Effective dimensions"});
 const split=document.querySelector('input[name="rlSplit"]:checked').value;
 heatmap(
  "rlChart",m[split].R_L,-1,1,
  `${split}: R_L unavailable (${m[split].representation_status})`
 );
 lineChart("rcChart",[
  {name:"Train",values:DATA.measures.map(x=>x.train.R_C),color:"#2166ac"},
  {name:"Validation",values:DATA.measures.map(x=>x.validation.R_C),color:"#b2182b"}],{xlabel:"Epoch",ylabel:"Effective dimensions"});
 lineChart("flChart",[
  {name:"Training",values:m.train.F_L,color:"#2166ac"},
  {name:"Validation",values:m.validation.F_L,color:"#b2182b"}],{xlabel:"Unit index",ylabel:"KL (nats/example)"});
 const trainFC=m.train.F_C||{},validationFC=m.validation.F_C||{};
 const dims=[...new Set([
  ...Object.keys(trainFC).map(Number),
  ...Object.keys(validationFC).map(Number)
 ])].sort((a,b)=>a-b);
 const trainLow=[],trainRand=[],trainErr=[],valLow=[],valRand=[],valErr=[];
 const trainRemaining=[],valRemaining=[];
 for(const k of dims){
  const a=trainFC[String(k)],b=validationFC[String(k)];
  const meanRandom=x=>x&&Array.isArray(x.random)&&x.random.length
   ?x.random.reduce((sum,value)=>sum+value,0)/x.random.length:null;
  trainLow.push(a?a.low_variance:null);trainRand.push(meanRandom(a));
  trainErr.push(a&&Array.isArray(a.random)?std(a.random):null);
  valLow.push(b?b.low_variance:null);valRand.push(meanRandom(b));
  valErr.push(b&&Array.isArray(b.random)?std(b.random):null);
  trainRemaining.push(a?a.R_C_remaining_lowvar:null);
  valRemaining.push(b?b.R_C_remaining_lowvar:null);
 }
 lineChart("fcChart",[
  {name:"Training — low-variance F_C",values:trainLow,color:"#2166ac"},
  {name:"Training — random F_C",values:trainRand,errors:trainErr,color:"#2166ac",dash:[5,3]},
  {name:"Validation — low-variance F_C",values:valLow,color:"#b2182b"},
  {name:"Validation — random F_C",values:valRand,errors:valErr,color:"#b2182b",dash:[5,3]},
  {name:"Training — remaining R_C",values:trainRemaining,color:"#2166ac",axis:"right",dash:[1,3]},
  {name:"Validation — remaining R_C",values:valRemaining,color:"#b2182b",axis:"right",dash:[1,3]}
 ],{
  min:0,
  xlabel:"Subspace dimension k",
  ylabel:"KL (nats/example)",
  right_ylabel:"Remaining R_C (···)",
  right_min:0,
  xTickLabels:dims,
  legendFont:"9px system-ui",
  legendBelow:true,
  legendSamplesBelow:true,
  legendGroups:[[
   {name:"Train",color:"#2166ac"},
   {name:"Validation",color:"#b2182b"},
   {name:"Low-variance F_C",color:"#354158"},
   {name:"Random F_C",color:"#354158",dash:[5,3]}
  ]]
 });
}
function std(xs){const mean=xs.reduce((a,b)=>a+b,0)/xs.length;return Math.sqrt(xs.reduce((a,b)=>a+(b-mean)**2,0)/(xs.length-1))}
let enlargedPanel=null;
function setPanelExpanded(panel,expanded){
 if(expanded){
  if(enlargedPanel&&enlargedPanel!==panel){
   setPanelExpanded(enlargedPanel,false);
  }
  const grid=document.querySelector(".grid");
  const style=getComputedStyle(grid);
  const rowGap=parseFloat(style.rowGap)||0;
  const rows=style.gridTemplateRows.split(" ").length;
  const columns=style.gridTemplateColumns.split(" ").length;
  const rowHeight=(grid.clientHeight-rowGap*(rows-1))/rows;
  const expandedRows=Math.min(2,rows);
  const expandedHeight=rowHeight*expandedRows+rowGap*(expandedRows-1);
  const row=Math.floor([...grid.querySelectorAll(".measure-panel")].indexOf(panel)/columns);
  const top=Math.min(row*(rowHeight+rowGap),grid.clientHeight-expandedHeight);
  panel.classList.add("enlarged");
  panel.style.top=`${Math.max(0,top)}px`;
  panel.style.height=`${expandedHeight}px`;
  panel.setAttribute("aria-expanded","true");
  enlargedPanel=panel;
 }else{
  panel.classList.remove("enlarged");
  panel.style.removeProperty("top");
  panel.style.removeProperty("height");
  panel.setAttribute("aria-expanded","false");
  enlargedPanel=null;
 }
 requestAnimationFrame(()=>window.dispatchEvent(new Event("resize")));
}
document.querySelectorAll(".measure-panel").forEach(panel=>{
 const title=panel.querySelector("h2").textContent;
 panel.setAttribute("aria-label",`${title} — click to expand or collapse`);
 panel.setAttribute("aria-expanded","false");
 panel.addEventListener("click",event=>{
  if(event.target.closest("input,label,button"))return;
  setPanelExpanded(panel,panel!==enlargedPanel);
 });
 panel.addEventListener("keydown",event=>{
  if(event.key==="Enter"||event.key===" "){
   event.preventDefault();
   setPanelExpanded(panel,panel!==enlargedPanel);
  }
 });
});
window.addEventListener("keydown",event=>{
 if(event.key==="Escape"&&enlargedPanel)setPanelExpanded(enlargedPanel,false);
});
slider.addEventListener("input",update);
document.querySelectorAll('input[name="rlSplit"]').forEach(x=>x.addEventListener("change",update));
window.addEventListener("message",event=>{
 if(event.data?.type!=="experiment-epoch")return;
 const index=epochs.indexOf(event.data.epoch);
 if(index<0)return;
 slider.value=index;
 update();
});
window.addEventListener("resize",update);
update();
</script>
</body></html>"""


def _run_details(metadata):
    config = metadata.get("requested_config", {})
    dataset = config.get("dataset", {})
    training = config.get("training", {})
    values = [
        ("arch", config.get("architecture")),
        ("opt", config.get("optimizer")),
        ("act", config.get("activation")),
        ("init", config.get("initializer")),
        ("lr", config.get("learning_rate")),
        ("wd", config.get("weight_decay")),
        (
            "mom",
            config.get("momentum")
            if config.get("optimizer") in {"sgd", "rms_prop"}
            else None,
        ),
        ("drop", config.get("dropout")),
        ("batch", training.get("batch_size")),
        ("epochs", training.get("epochs")),
        ("train", dataset.get("train_fraction")),
        ("seed", metadata.get("seed", config.get("seed"))),
    ]
    details = []
    for label, value in values:
        if value is None:
            continue
        if isinstance(value, float):
            value = f"{value:.3g}"
        elif label == "opt":
            value = {
                "adam": "Adam",
                "sgd": "SGD",
                "rms_prop": "RMSProp",
            }.get(str(value), str(value).replace("_", " ").title())
        if label == "train":
            value = f"{float(value) * 100:g}%"
        if label in {"arch", "opt", "act", "init"}:
            details.append(str(value))
        else:
            details.append(f"{label}={value}")
    return details


def _validate_temporal_sample_identities(measurements, metadata):
    sample_metadata = metadata.get("temporal_measurement_samples") or {}
    train_subset_indices = metadata.get("train_subset_indices")
    train_positions = sample_metadata.get("train_positions_in_training_subset")
    train_dataset_indices = sample_metadata.get("train_dataset_indices")
    validation_positions = sample_metadata.get("validation_positions")
    if not isinstance(train_subset_indices, list) or not isinstance(
        train_positions, list
    ) or not isinstance(validation_positions, list):
        raise ValueError(
            "temporal results must retain exact train and validation sample indices"
        )
    if (
        any(type(index) is not int or index < 0 for index in train_subset_indices)
        or len(set(train_subset_indices)) != len(train_subset_indices)
        or any(type(index) is not int for index in train_positions)
        or len(train_positions) != sample_metadata.get("train_count")
        or len(set(train_positions)) != len(train_positions)
        or any(index < 0 or index >= len(train_subset_indices) for index in train_positions)
    ):
        raise ValueError("invalid training measurement sample indices")
    if train_dataset_indices is not None and train_dataset_indices != [
        train_subset_indices[position] for position in train_positions
    ]:
        raise ValueError(
            "training dataset indices do not match the recorded subset positions"
        )
    validation_split_indices = sample_metadata.get("validation_split_indices")
    if (
        validation_split_indices is not None
        and validation_split_indices != validation_positions
    ):
        raise ValueError(
            "validation split indices do not match the recorded positions"
        )
    if (
        any(type(index) is not int for index in validation_positions)
        or len(validation_positions) != sample_metadata.get("validation_count")
        or len(set(validation_positions)) != len(validation_positions)
        or any(
            not isinstance(index, int) or index < 0
            for index in validation_positions
        )
    ):
        raise ValueError("invalid validation measurement sample indices")

    for split in ("train", "validation"):
        expected_identity = sample_metadata.get(f"{split}_identity")
        if expected_identity is None:
            raise ValueError(
                f"run metadata is missing the exact {split} sample identity"
            )
        for record in measurements:
            split_record = record.get(split, {})
            if split_record.get("reference_sample") != expected_identity:
                raise ValueError(
                    f"{split} measurement sample identity changed or does not "
                    "match the exact indices in run metadata"
                )


def write_dashboard(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    metadata_path = run_dir / "run_metadata.json"
    measurements_path = run_dir / "temporal_measurements.jsonl"
    if not metadata_path.is_file() or not measurements_path.is_file():
        raise FileNotFoundError(
            "dashboard requires run_metadata.json and temporal_measurements.jsonl"
        )
    with open(metadata_path) as metadata_file:
        metadata = json.load(metadata_file)
    with open(measurements_path) as measurements_file:
        measurements = [
            json.loads(line) for line in measurements_file if line.strip()
        ]
    metrics = load_run_result(run_dir)
    if not measurements:
        raise ValueError("no temporal measurement records were found")
    _validate_temporal_sample_identities(measurements, metadata)
    for record in measurements:
        if (
            record.get("metric_definition_version")
            != METRIC_DEFINITION_VERSION
            or record.get("p_c_definition") != P_C_DEFINITION
            or record.get("fc_intervention") != F_C_INTERVENTION
        ):
            raise ValueError(
                "temporal dashboard requires metric definition version 2 "
                "with spectral-energy P_C and centered F_C; legacy records "
                "cannot be compared and must be regenerated"
            )
        for split in ("train", "validation"):
            split_record = record.get(split, {})
            fc_record = split_record.get("F_C")
            if (
                split_record.get("reference_sample") is None
                or split_record.get("reference_activation_mean") is None
                or split_record.get("activation_covariance_eigenvalues") is None
                or (
                    fc_record is not None
                    and (
                        not isinstance(fc_record, dict)
                        or "0" not in fc_record
                    )
                )
                or any(
                    "R_C_remaining_lowvar" not in value
                    for value in (fc_record or {}).values()
                )
            ):
                raise ValueError(
                    "temporal dashboard requires centered F_C results with "
                    "reference-sample metadata, PCA spectrum and remaining "
                    "effective dimensions; regenerate incomplete results"
                )
    metrics_by_epoch = {
        record["training_iteration"]: record for record in metrics
    }
    aligned_metrics = [
        metrics_by_epoch[record["training_iteration"]]
        for record in measurements
        if record["training_iteration"] in metrics_by_epoch
    ]
    if len(aligned_metrics) != len(measurements):
        raise ValueError("temporal measurements and epoch metrics do not align")
    payload = {
        "run_id": metadata["run_id"],
        "run_details": _run_details(metadata),
        "sample_info": metadata["temporal_measurement_samples"],
        "metrics": aligned_metrics,
        "measures": measurements,
    }
    encoded_data = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    encoded_data = encoded_data.replace("</", "<\\/")
    html = _DASHBOARD_TEMPLATE.replace("__DATA__", encoded_data)
    output_path = run_dir / "dashboard.html"
    output_path.write_text(html, encoding="utf-8")
    return output_path


_EXPERIMENT_DASHBOARD_TEMPLATE = r"""<!doctype html>
<html lang="en-GB">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Vue d’ensemble des runs</title>
<style>
*{box-sizing:border-box}
body{font:14px system-ui,sans-serif;margin:8px;background:#f4f6fa;color:#172033;height:calc(100vh - 16px);overflow:hidden}
.controls{height:42px;display:flex;gap:10px;align-items:center;background:white;padding:6px 12px;border-radius:8px;margin:0 auto 8px;position:sticky;top:0;z-index:2;box-shadow:0 1px 5px #18284814}
input[type=range]{flex:1;min-width:80px}
.runs{height:calc(100vh - 66px);width:100%;margin:0 auto;display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));grid-auto-rows:170px;gap:8px;overflow:auto;align-items:stretch;padding-bottom:8px}
.run{min-width:0;min-height:0;display:flex;flex-direction:column;border-radius:8px;padding:7px;box-shadow:0 1px 5px #18284814}
.run.expanded{grid-column:span 2;grid-row:span 4}
.accuracy-card{display:flex;width:100%;flex-direction:column;text-align:left;font:inherit;color:inherit;background:transparent;border:0;padding:0;cursor:pointer}
.accuracy-card:focus-visible{outline:2px solid #2166ac;outline-offset:2px}
.run-heading{display:flex;align-items:baseline;gap:6px;flex-wrap:wrap;min-width:0}
.run-heading h2{font-size:12px;white-space:nowrap;margin:0}
.run-config{font-size:10px;line-height:1.25;color:#536078;margin:0}
.accuracy-chart{display:block;width:100%;height:112px;min-height:0;flex:1}
.run.expanded .accuracy-chart{display:none}
.run-panel{display:none;width:100%;min-height:0;flex:1;margin-top:8px;border:1px solid #dce2ec;border-radius:6px}
.run.expanded .run-panel{display:block}
iframe{width:100%;height:100%;border:0}
.hint{font-size:11px;color:#536078}
@media(max-width:700px){.runs{grid-template-columns:repeat(2,minmax(0,1fr));grid-auto-rows:150px}.run.expanded{grid-column:span 2;grid-row:span 4}}
</style>
</head>
<body>
<div class="controls">
 <label for="epochSlider">Shared epoch:</label><input id="epochSlider" type="range">
 <strong id="epochLabel"></strong><span class="hint" id="epochHint"></span>
</div>
<div class="runs" id="runs"></div>
<script>
const DATA=__DATA__;
const epochs=DATA.epochs;
const slider=document.getElementById("epochSlider");
slider.min=0;slider.max=epochs.length-1;slider.step=1;slider.value=epochs.length-1;
if(epochs.length===1){
 slider.disabled=true;
 document.getElementById("epochHint").textContent="only one epoch measured";
}
const container=document.getElementById("runs");
const runElements=[];
for(const run of DATA.runs){
 const tintHue=(runElements.length*137.508)%360;
 const tint=`hsl(${tintHue} 65% 88%)`;
 const section=document.createElement("article");
 section.className="run";
 section.style.backgroundColor=tint;
 const heading=document.createElement("h2");
 heading.textContent=run.run_id;
 const runHeading=document.createElement("header");
 runHeading.className="run-heading";
 runHeading.append(heading);
 if(run.details.length){
  const details=document.createElement("p");
  details.className="run-config";
  details.textContent=run.details.join(" · ");
  runHeading.append(details);
 }
 const button=document.createElement("button");
 button.type="button";
 button.className="accuracy-card";
 button.setAttribute("aria-expanded","false");
 button.setAttribute("aria-label",`Show detailed panel for ${run.run_id}`);
 const chart=document.createElement("canvas");
 chart.className="accuracy-chart";
 chart.setAttribute("aria-label",`Accuracy train et validation de ${run.run_id}`);
 button.append(runHeading,chart);
 const frame=document.createElement("iframe");
 frame.className="run-panel";
 frame.style.backgroundColor=tint;
 frame.title=`Dashboard for ${run.run_id}`;
 frame.dataset.dashboard=run.dashboard;
 frame.dataset.runId=run.run_id;
 section.append(button,frame);
 button.addEventListener("click",()=>{
  const expanded=section.classList.toggle("expanded");
  button.setAttribute("aria-expanded",String(expanded));
  button.setAttribute("aria-label",
   `${expanded?"Close":"Show"} detailed panel for ${run.run_id}`);
  if(expanded&&!frame.dataset.loaded){
   frame.src=frame.dataset.dashboard
    +`?embedded=1&tint=${encodeURIComponent(tint)}`;
   frame.dataset.loaded="true";
  }
  requestAnimationFrame(()=>{
   for(const item of runElements){
    drawAccuracy(item.run,item.chart,Number(slider.value));
   }
  });
 });
 container.append(section);
 runElements.push({run,chart,frame});
}
function drawAccuracy(run,canvas,selectedIndex){
 const rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
 if(!rect.width||!rect.height)return;
 canvas.width=Math.max(1,Math.round(rect.width*dpr));
 canvas.height=Math.max(1,Math.round(rect.height*dpr));
 const c=canvas.getContext("2d");c.scale(dpr,dpr);
 const w=rect.width,h=rect.height,left=34,right=7,top=17,bottom=22;
 const plotTop=top+13,plotBottom=h-bottom-13;
 const pw=w-left-right,ph=plotBottom-plotTop;
 c.clearRect(0,0,w,h);
 const series=[
  {name:"Train",key:"train_eval_acc",color:"#3478c5"},
  {name:"Validation",key:"validation_eval_acc",color:"#e07a32"}
 ];
 c.font="9px system-ui";c.fillStyle="#526078";c.textAlign="right";
 for(let tick=0;tick<=4;tick++){
  const value=tick/4,y=plotBottom-ph*value;
  c.strokeStyle="#e1e6ee";c.beginPath();c.moveTo(left,y);c.lineTo(left+pw,y);c.stroke();
  c.fillText(`${Math.round(value*100)}%`,left-4,y+3);
 }
 const count=Math.max(1,run.accuracy.length);
 const xAt=i=>left+pw*(count<=1?0:i/(count-1));
 for(const item of series){
  c.strokeStyle=item.color;c.lineWidth=1.8;c.beginPath();let started=false;
  run.accuracy.forEach((point,index)=>{
   const value=point[item.key];
   if(!Number.isFinite(value))return;
   const x=xAt(index),y=plotBottom-ph*value;
   if(!started){c.moveTo(x,y);started=true}else c.lineTo(x,y);
  });
  c.stroke();
 }
 const epoch=epochs[selectedIndex];
  const epochIndex=run.accuracy.findIndex(
   point=>point.training_iteration===epoch
  );
  if(epochIndex>=0){
   const x=xAt(epochIndex);
   c.save();c.strokeStyle="#26354d";c.lineWidth=1;c.setLineDash([4,3]);
   c.beginPath();c.moveTo(x,plotTop);c.lineTo(x,plotBottom);c.stroke();
   c.restore();
  }
  c.font="9px system-ui";c.textAlign="left";let legendX=left;
 for(const item of series){
  c.fillStyle=item.color;c.fillRect(legendX,5,10,2);
  c.fillStyle="#354158";c.fillText(item.name,legendX+13,9);
  legendX+=45;
 }
}
function updateFrames(){
 const selectedIndex=Number(slider.value),epoch=epochs[selectedIndex];
 document.getElementById("epochLabel").textContent=epoch;
 for(const {run,chart,frame} of runElements){
  drawAccuracy(run,chart,selectedIndex);
  frame.contentWindow?.postMessage({type:"experiment-epoch",epoch},"*");
 }
}
slider.addEventListener("input",updateFrames);
for(const {frame} of runElements)
 frame.addEventListener("load",updateFrames)
window.addEventListener("resize",updateFrames);
updateFrames();
</script>
</body>
</html>"""


def write_experiment_dashboard(output_root: Path) -> Path:
    output_root = Path(output_root)
    runs_root = output_root / "runs"
    runs = []
    common_epochs = None
    if runs_root.is_dir():
        for run_dir in sorted(runs_root.iterdir()):
            dashboard_path = run_dir / "dashboard.html"
            measurements_path = run_dir / "temporal_measurements.jsonl"
            status_path = run_dir / "status.json"
            if not (
                run_dir.is_dir()
                and dashboard_path.is_file()
                and measurements_path.is_file()
                and status_path.is_file()
            ):
                continue
            with status_path.open(encoding="utf-8") as status_file:
                if json.load(status_file).get("status") != "completed":
                    continue
            with measurements_path.open(encoding="utf-8") as measurements_file:
                measurements = [
                    json.loads(line)
                    for line in measurements_file
                    if line.strip()
                ]
            epochs = [
                item["training_iteration"] for item in measurements
            ]
            if not epochs:
                continue
            metadata_path = run_dir / "run_metadata.json"
            if metadata_path.is_file():
                with metadata_path.open(encoding="utf-8") as metadata_file:
                    run_details = _run_details(json.load(metadata_file))
            else:
                run_details = []
            metrics_by_epoch = {
                record["training_iteration"]: record
                for record in load_run_result(run_dir)
            }
            accuracy = [
                {
                    "training_iteration": epoch,
                    "train_eval_acc": metrics_by_epoch[epoch].get(
                        "train_eval_acc"
                    ),
                    "validation_eval_acc": metrics_by_epoch[epoch].get(
                        "validation_eval_acc"
                    ),
                }
                for epoch in sorted(metrics_by_epoch)
            ]
            if not accuracy:
                raise ValueError(
                    f"no epoch accuracy metrics found for run {run_dir.name}"
                )
            if common_epochs is None:
                common_epochs = epochs
            elif epochs != common_epochs:
                raise ValueError(
                    "completed runs have different measurement epochs; "
                    "a shared epoch slider cannot align them"
                )
            runs.append(
                {
                    "run_id": run_dir.name,
                    "details": run_details,
                    "accuracy": accuracy,
                    "dashboard": f"runs/{run_dir.name}/dashboard.html",
                }
            )
    if not runs or common_epochs is None:
        raise ValueError(
            "no completed temporal run dashboards found in "
            f"{runs_root}"
        )
    payload = {"epochs": common_epochs, "runs": runs}
    encoded_data = json.dumps(payload, separators=(",", ":"))
    encoded_data = encoded_data.replace("</", "<\\/")
    html = _EXPERIMENT_DASHBOARD_TEMPLATE.replace("__DATA__", encoded_data)
    output_path = output_root / "dashboard.html"
    output_path.write_text(html, encoding="utf-8")
    return output_path
