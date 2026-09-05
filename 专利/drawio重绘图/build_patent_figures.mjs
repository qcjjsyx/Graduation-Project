import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';

const outputDir = resolve('专利/drawio重绘图');
const font = 'fontFamily=Microsoft YaHei;';
const base = `whiteSpace=wrap;html=1;${font}fontColor=#000000;`;
const rectStyle = `${base}rounded=1;arcSize=12;fillColor=#ffffff;strokeColor=#000000;strokeWidth=1.4;align=center;verticalAlign=middle;`;
const plainRectStyle = `${base}rounded=0;fillColor=#ffffff;strokeColor=#000000;strokeWidth=1.2;align=center;verticalAlign=middle;`;
const textStyle = `${base}fillColor=none;strokeColor=none;align=center;verticalAlign=middle;`;
const dashedStyle = `${base}rounded=0;fillColor=none;strokeColor=#000000;strokeWidth=1.4;dashed=1;dashPattern=10 8;align=left;verticalAlign=top;spacingTop=8;spacingLeft=10;`;
const arrowStyle = `${font}edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeColor=#000000;strokeWidth=1.4;`;
const dashedArrowStyle = `${arrowStyle}dashed=1;dashPattern=7 6;`;
const dottedArrowStyle = `${arrowStyle}dashed=1;dashPattern=2 7;`;

function esc(value = '') {
  return String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
}

function createDiagram(name, width, height) {
  let nextId = 2;
  const cells = [];
  const newId = (prefix = 'c') => `${prefix}${nextId++}`;
  const vertex = (value, style, x, y, w, h, prefix) => {
    const id = newId(prefix);
    cells.push(`<mxCell id="${id}" value="${esc(value)}" style="${style}" parent="1" vertex="1"><mxGeometry x="${x}" y="${y}" width="${w}" height="${h}" as="geometry" /></mxCell>`);
    return id;
  };
  const edge = (source, target, options = {}) => {
    const id = newId('e');
    const {
      value = '', style = arrowStyle, points = [], exitX, exitY, entryX, entryY,
      sourcePoint, targetPoint,
    } = options;
    let styled = style;
    if (exitX !== undefined) styled += `exitX=${exitX};exitY=${exitY};exitPerimeter=1;`;
    if (entryX !== undefined) styled += `entryX=${entryX};entryY=${entryY};entryPerimeter=1;`;
    const attrs = `${source ? ` source="${source}"` : ''}${target ? ` target="${target}"` : ''}`;
    const pointXml = [
      sourcePoint ? `<mxPoint x="${sourcePoint[0]}" y="${sourcePoint[1]}" as="sourcePoint" />` : '',
      targetPoint ? `<mxPoint x="${targetPoint[0]}" y="${targetPoint[1]}" as="targetPoint" />` : '',
      points.length ? `<Array as="points">${points.map(([x, y]) => `<mxPoint x="${x}" y="${y}" />`).join('')}</Array>` : '',
    ].join('');
    cells.push(`<mxCell id="${id}" value="${esc(value)}" style="${styled}" parent="1" edge="1"${attrs}><mxGeometry relative="1" as="geometry">${pointXml}</mxGeometry></mxCell>`);
    return id;
  };
  const out = () => `<?xml version="1.0" encoding="UTF-8"?>\n<mxfile host="app.diagrams.net" modified="2026-08-28T00:00:00.000Z" agent="Codex" version="28.2.9" type="device"><diagram id="${name}" name="${name}"><mxGraphModel dx="${width}" dy="${height}" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="${width}" pageHeight="${height}" math="0" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>${cells.join('')}</root></mxGraphModel></diagram></mxfile>\n`;
  return { vertex, edge, out };
}

function fig1() {
  const d = createDiagram('图1 加速器总体结构', 1600, 900);
  const v = d.vertex;
  const e = d.edge;
  v('加速器', dashedStyle, 30, 30, 1510, 820, 'box');
  v('因子写回单元', rectStyle + 'fontSize=15;', 760, 55, 150, 55, 'u');
  v('外部存储区域', plainRectStyle + 'fontSize=16;fontStyle=1;', 110, 170, 250, 480, 'box');
  const taskArea = v('任务区', plainRectStyle + 'fontSize=14;', 165, 235, 135, 55, 'm');
  const mapArea = v('映射表区', plainRectStyle + 'fontSize=14;', 165, 315, 135, 55, 'm');
  const matrixArea = v('矩阵数据区', plainRectStyle + 'fontSize=14;', 165, 395, 135, 55, 'm');
  const factorArea = v('因子区', plainRectStyle + 'fontSize=14;', 165, 475, 135, 55, 'm');
  const updateArea = v('更新矩阵区', plainRectStyle + 'fontSize=14;', 165, 555, 135, 55, 'm');
  const scatter = v('散射装配单元', rectStyle + 'fontSize=15;', 140, 700, 180, 70, 'u');
  v('全局控制单元', plainRectStyle + 'fontSize=16;fontStyle=1;fillColor=none;', 430, 155, 650, 410, 'box');
  const taskFetch = v('节点任务读取', rectStyle + 'fontSize=15;', 480, 245, 155, 70, 'u');
  const scoreboard = v('依赖记分牌', rectStyle + 'fontSize=15;', 750, 245, 155, 70, 'u');
  const bufferMgr = v('缓冲区管理', rectStyle + 'fontSize=15;', 480, 380, 155, 70, 'u');
  const scheduler = v('节点内阶段调度', rectStyle + 'fontSize=15;', 845, 380, 175, 70, 'u');
  const loader = v('前沿数据加载', rectStyle + 'fontSize=15;', 480, 495, 155, 70, 'u');
  const core = v('计算核心', rectStyle + 'fontSize=16;', 1210, 360, 255, 90, 'u');
  const hpu = v('层级主元单元', rectStyle + 'fontSize=15;', 1265, 530, 145, 65, 'u');
  const atu = v('地址变换单元', rectStyle + 'fontSize=15;', 1265, 635, 145, 65, 'u');
  v('双缓冲存储管理单元', dashedStyle + 'fontSize=15;fontStyle=1;', 410, 660, 375, 160, 'box');
  const buffer1 = v('缓冲区一', rectStyle + 'fontSize=15;', 440, 720, 130, 60, 'u');
  const buffer2 = v('缓冲区二', rectStyle + 'fontSize=15;', 625, 720, 130, 60, 'u');
  v('线型说明', textStyle + 'fontSize=15;fontStyle=1;align=left;', 1070, 725, 185, 25, 't');
  v('实线箭头：数据、任务及结果传输', textStyle + 'fontSize=13;align=left;', 1070, 752, 350, 25, 't');
  v('虚线箭头：控制与状态反馈', textStyle + 'fontSize=13;align=left;', 1070, 778, 300, 25, 't');
  v('虚线框：功能模块边界', textStyle + 'fontSize=13;align=left;', 1070, 804, 260, 25, 't');
  e(taskArea, taskFetch, { exitX: 1, exitY: .5, entryX: 0, entryY: .5 });
  e(taskFetch, scoreboard, { value: '任务依赖状态', exitX: 1, exitY: .5, entryX: 0, entryY: .5 });
  e(taskFetch, bufferMgr, { exitX: .5, exitY: 1, entryX: .5, entryY: 0 });
  e(mapArea, bufferMgr, { exitX: 1, exitY: .5, entryX: 0, entryY: .5, points: [[400, 342], [400, 415]] });
  e(matrixArea, loader, { exitX: 1, exitY: .5, entryX: 0, entryY: .5, points: [[385, 422], [385, 530]] });
  e(bufferMgr, scheduler, { value: '缓冲区状态', exitX: 1, exitY: .5, entryX: 0, entryY: .5 });
  e(bufferMgr, loader, { exitX: .5, exitY: 1, entryX: .5, entryY: 0 });
  e(scoreboard, scheduler, { style: dashedArrowStyle, value: '就绪信号', exitX: 1, exitY: .5, entryX: .2, entryY: 0, points: [[940, 280], [940, 350]] });
  e(loader, buffer1, { style: dashedArrowStyle, value: '预取数据', exitX: .5, exitY: 1, entryX: .5, entryY: 0 });
  e(loader, buffer2, { style: dashedArrowStyle, exitX: .8, exitY: 1, entryX: .5, entryY: 0, points: [[602, 620], [690, 620]] });
  e(scheduler, core, { value: '调度信号', exitX: 1, exitY: .5, entryX: 0, entryY: .5 });
  e(core, hpu, { exitX: .5, exitY: 1, entryX: .5, entryY: 0 });
  e(hpu, atu, { exitX: .5, exitY: 1, entryX: .5, entryY: 0 });
  e(atu, buffer2, { style: dashedArrowStyle, value: '地址映射', exitX: 0, exitY: .5, entryX: 1, entryY: .5, points: [[1170, 667], [1170, 750], [785, 750]] });
  e(factorArea, scatter, { exitX: .5, exitY: 1, entryX: .5, entryY: 0, points: [[232, 670]] });
  e(scatter, loader, { exitX: 1, exitY: .5, entryX: 0, entryY: .7, points: [[365, 735], [365, 545]] });
  e(updateArea, scatter, { style: dashedArrowStyle, value: '更新数据', exitX: 1, exitY: .5, entryX: 0, entryY: .4, points: [[340, 582], [340, 725]] });
  e(core, null, { value: '分解结果', exitX: .45, exitY: 0, targetPoint: [835, 110], points: [[1335, 135], [835, 135]] });
  return d.out();
}

function fig2() {
  const d = createDiagram('图2 双缓冲流水线', 1120, 500);
  const v = d.vertex;
  const e = d.edge;
  v('', plainRectStyle + 'strokeColor=#b7b7b7;fillColor=#ffffff;', 170, 55, 910, 360, 'box');
  v('双缓冲', `${base}fillColor=#4472c4;strokeColor=#4472c4;fontColor=#ffffff;fontSize=20;fontStyle=1;rotation=270;align=center;verticalAlign=middle;`, 30, 55, 70, 360, 'b');
  v('缓冲区一', `${base}fillColor=#4472c4;strokeColor=#ffffff;fontColor=#ffffff;fontSize=17;fontStyle=1;rotation=270;align=center;verticalAlign=middle;`, 100, 55, 70, 180, 'b');
  v('缓冲区二', `${base}fillColor=#4472c4;strokeColor=#ffffff;fontColor=#ffffff;fontSize=17;fontStyle=1;rotation=270;align=center;verticalAlign=middle;`, 100, 235, 70, 180, 'b');
  const cycleXs = [260, 380, 500, 620, 740, 860];
  cycleXs.forEach((x, i) => v(`周期 ${i + 1}`, plainRectStyle + 'fontSize=15;', x, 28, 120, 45, 'c'));
  v('', `${base}fillColor=none;strokeColor=#b7b7b7;strokeWidth=1;`, 170, 235, 910, 1, 'line');
  const top = ['加载任务', '加载数据', '计算', '空闲', '结果写回', '空闲'];
  const bottom = ['空闲', '加载任务', '加载数据', '计算', '空闲', '结果写回'];
  const stateStyle = `${base}shape=chevron;perimeter=chevronPerimeter;fillColor=#ffffff;strokeColor=#000000;strokeWidth=1.2;fontSize=15;align=center;verticalAlign=middle;`;
  top.forEach((label, i) => v(label, stateStyle, 258 + i * 120, 115, 105, 58, 's'));
  bottom.forEach((label, i) => v(label, stateStyle, 258 + i * 120, 285, 105, 58, 's'));
  e(null, null, { style: dottedArrowStyle + 'endArrow=none;startArrow=none;', sourcePoint: [558, 183], targetPoint: [558, 278], value: '待处理子节点数＝0' });
  return d.out();
}

function fig3() {
  const d = createDiagram('图3 节点处理流程', 1100, 620);
  const v = d.vertex;
  const e = d.edge;
  const read = v('读取节点任务', rectStyle + 'fontSize=19;', 65, 160, 250, 75, 'p');
  const condition = v('待处理子节点数<br>是否为 0', `${base}shape=rhombus;perimeter=rhombusPerimeter;fillColor=#ffffff;strokeColor=#000000;strokeWidth=1.4;fontSize=18;align=center;verticalAlign=middle;`, 430, 125, 220, 130, 'd');
  const load = v('加载节点数据', rectStyle + 'fontSize=19;', 775, 160, 250, 75, 'p');
  const lu = v('主元矩阵分解', rectStyle + 'fontSize=19;', 775, 360, 250, 75, 'p');
  const triangular = v('三角求解', rectStyle + 'fontSize=19;', 430, 360, 250, 75, 'p');
  const gemm = v('矩阵乘法更新', rectStyle + 'fontSize=19;', 65, 360, 250, 75, 'p');
  const writeback = v('结果写回', rectStyle + 'fontSize=19;', 65, 540, 250, 75, 'p');
  const parent = v('更新父节点待处理子节点数', rectStyle + 'fontSize=18;', 450, 540, 320, 75, 'p');
  e(read, condition, { exitX: 1, exitY: .5, entryX: 0, entryY: .5 });
  e(condition, load, { value: '是', exitX: 1, exitY: .5, entryX: 0, entryY: .5 });
  e(condition, read, { value: '否', exitX: .5, exitY: 0, entryX: .5, entryY: 0, points: [[540, 55], [190, 55]] });
  e(load, lu, { exitX: .5, exitY: 1, entryX: .5, entryY: 0 });
  e(lu, triangular, { exitX: 0, exitY: .5, entryX: 1, entryY: .5 });
  e(triangular, gemm, { exitX: 0, exitY: .5, entryX: 1, entryY: .5 });
  e(gemm, writeback, { exitX: .5, exitY: 1, entryX: .5, entryY: 0 });
  e(writeback, parent, { exitX: 1, exitY: .5, entryX: 0, entryY: .5 });
  return d.out();
}

function fig4() {
  const d = createDiagram('图4 依赖记分牌流程', 1280, 860);
  const v = d.vertex;
  const e = d.edge;
  const child = v('已完成的子节点任务', rectStyle + 'fontSize=19;', 440, 40, 310, 85, 'p');
  const board = v('依赖记分牌', rectStyle + 'fontSize=20;', 440, 210, 310, 110, 'p');
  const queue = v('任务进入队列', rectStyle + 'fontSize=18;', 70, 405, 210, 90, 'p');
  const check = v('检查子节点<br>是否全部完成', `${base}shape=rhombus;perimeter=rhombusPerimeter;fillColor=#ffffff;strokeColor=#000000;strokeWidth=1.4;fontSize=17;align=center;verticalAlign=middle;`, 470, 390, 245, 135, 'd');
  const wait = v('等待子节点任务完成', rectStyle + 'fontSize=18;', 855, 405, 240, 90, 'p');
  const update = v('更新父节点', rectStyle + 'fontSize=18;', 855, 570, 240, 90, 'p');
  const take = v('提取任务', rectStyle + 'fontSize=19;', 465, 580, 250, 90, 'p');
  const load = v('加载节点数据', rectStyle + 'fontSize=19;', 465, 735, 250, 90, 'p');
  const execute = v('执行', rectStyle + 'fontSize=19;', 855, 735, 180, 90, 'p');
  const done = v('任务完成', rectStyle + 'fontSize=19;', 1080, 735, 150, 90, 'p');
  e(child, board, { value: '更新待处理子节点数', exitX: .5, exitY: 1, entryX: .5, entryY: 0 });
  e(board, check, { exitX: .5, exitY: 1, entryX: .5, entryY: 0 });
  e(queue, check, { exitX: 1, exitY: .5, entryX: 0, entryY: .5 });
  e(check, take, { value: '是', exitX: .5, exitY: 1, entryX: .5, entryY: 0 });
  e(check, wait, { value: '否', exitX: 1, exitY: .5, entryX: 0, entryY: .5 });
  e(wait, update, { exitX: .5, exitY: 1, entryX: .5, entryY: 0 });
  e(update, board, { value: '更新待处理子节点数', exitX: 1, exitY: .5, entryX: 1, entryY: .5, points: [[1135, 615], [1135, 265]] });
  e(take, load, { exitX: .5, exitY: 1, entryX: .5, entryY: 0 });
  e(load, execute, { exitX: 1, exitY: .5, entryX: 0, entryY: .5 });
  e(execute, done, { exitX: 1, exitY: .5, entryX: 0, entryY: .5 });
  e(queue, load, { value: '缓冲区就绪', style: dottedArrowStyle, exitX: .5, exitY: 1, entryX: 0, entryY: .5, points: [[175, 790], [410, 790]] });
  e(update, execute, { style: dottedArrowStyle, exitX: .5, exitY: 1, entryX: .5, entryY: 0, points: [[975, 695], [945, 695]] });
  return d.out();
}

function fig5() {
  const d = createDiagram('图5 异步控制组件及先进先出缓冲器', 920, 1100);
  const v = d.vertex;
  const e = d.edge;
  v('异步控制组件符号', dashedStyle + 'fontSize=17;fontStyle=1;', 50, 35, 820, 630, 'box');
  const source = v('', `${base}shape=ellipse;perimeter=ellipsePerimeter;fillColor=#ffffff;strokeColor=#000000;strokeWidth=2;`, 205, 100, 75, 75, 's');
  v('源', textStyle + 'fontSize=20;', 195, 178, 95, 35, 't');
  v('', `${base}shape=ellipse;perimeter=ellipsePerimeter;fillColor=#ffffff;strokeColor=#000000;strokeWidth=2;`, 500, 100, 75, 75, 's');
  v('', `${base}shape=ellipse;perimeter=ellipsePerimeter;fillColor=none;strokeColor=#000000;strokeWidth=2;`, 520, 120, 35, 35, 's');
  v('阱', textStyle + 'fontSize=20;', 490, 178, 95, 35, 't');
  e(null, null, { sourcePoint: [90, 295], targetPoint: [210, 295], style: arrowStyle });
  e(null, null, { sourcePoint: [90, 385], targetPoint: [210, 385], style: arrowStyle });
  const merge = v('', `${base}shape=ellipse;perimeter=ellipsePerimeter;fillColor=#ffffff;strokeColor=#000000;strokeWidth=2;`, 210, 270, 65, 140, 's');
  e(merge, null, { sourcePoint: [275, 340], targetPoint: [360, 340], style: arrowStyle });
  v('汇聚', textStyle + 'fontSize=20;', 185, 420, 115, 35, 't');
  e(null, null, { sourcePoint: [410, 295], targetPoint: [520, 295], style: arrowStyle });
  e(null, null, { sourcePoint: [410, 385], targetPoint: [520, 385], style: arrowStyle });
  const arbitration = v('', `${base}shape=ellipse;perimeter=ellipsePerimeter;fillColor=#ffffff;strokeColor=#000000;strokeWidth=2;`, 520, 270, 65, 140, 's');
  e(arbitration, null, { sourcePoint: [585, 340], targetPoint: [670, 340], style: arrowStyle });
  e(null, null, { sourcePoint: [552, 300], targetPoint: [552, 385], style: `${font}endArrow=none;startArrow=none;strokeColor=#000000;strokeWidth=1.6;` });
  e(null, null, { sourcePoint: [520, 340], targetPoint: [585, 340], style: `${font}endArrow=none;startArrow=none;strokeColor=#000000;strokeWidth=1.6;` });
  v('仲裁', textStyle + 'fontSize=20;', 495, 420, 115, 35, 't');
  v('', `${base}shape=rectangle;fillColor=none;strokeColor=#000000;strokeWidth=1;`, 80, 470, 740, 1, 'line');
  // 分流、择路均沿用原图的凹透镜轮廓：上下直边与两侧向内收拢的凹边均可单独编辑。
  const splitter = v('', `${base}fillColor=none;strokeColor=none;`, 220, 480, 65, 145, 'p');
  const lensLineStyle = `${font}edgeStyle=none;rounded=0;html=1;endArrow=none;startArrow=none;strokeColor=#000000;strokeWidth=2;`;
  // curved=1 让控制点生成连续贝塞尔弧线，而非前一版的折线转角。
  const curvedLensLineStyle = `${lensLineStyle}curved=1;`;
  e(null, null, { sourcePoint: [220, 480], targetPoint: [285, 480], style: lensLineStyle });
  e(null, null, { sourcePoint: [220, 625], targetPoint: [285, 625], style: lensLineStyle });
  e(null, null, { sourcePoint: [220, 480], targetPoint: [220, 625], points: [[245, 552]], style: curvedLensLineStyle });
  e(null, null, { sourcePoint: [285, 480], targetPoint: [285, 625], points: [[260, 552]], style: curvedLensLineStyle });
  v('分流', textStyle + 'fontSize=20;', 195, 635, 115, 35, 't');
  e(null, splitter, { sourcePoint: [85, 552], entryX: 0, entryY: .5 });
  e(splitter, null, { exitX: 1, exitY: .25, targetPoint: [465, 515] });
  e(splitter, null, { exitX: 1, exitY: .75, targetPoint: [465, 590] });
  const selector = v('', `${base}fillColor=none;strokeColor=none;`, 595, 480, 65, 145, 'p');
  e(null, null, { sourcePoint: [595, 480], targetPoint: [660, 480], style: lensLineStyle });
  e(null, null, { sourcePoint: [595, 625], targetPoint: [660, 625], style: lensLineStyle });
  e(null, null, { sourcePoint: [595, 480], targetPoint: [595, 625], points: [[620, 552]], style: curvedLensLineStyle });
  e(null, null, { sourcePoint: [660, 480], targetPoint: [660, 625], points: [[635, 552]], style: curvedLensLineStyle });
  v('择路', textStyle + 'fontSize=20;', 570, 635, 115, 35, 't');
  e(null, selector, { sourcePoint: [460, 552], entryX: 0, entryY: .5 });
  e(selector, null, { exitX: 1, exitY: .25, targetPoint: [840, 515] });
  e(selector, null, { exitX: 1, exitY: .75, targetPoint: [840, 590] });
  e(null, selector, { value: '选择信号', sourcePoint: [627, 425], entryX: .5, entryY: 0, style: dashedArrowStyle });
  v('先进先出缓冲器示意图', textStyle + 'fontSize=20;fontStyle=1;', 300, 725, 320, 35, 't');
  const fifo = v('', rectStyle + 'fontSize=15;', 210, 780, 500, 135, 'f');
  const slots = ['数据 1', '数据 2', '数据 3', '…'];
  slots.forEach((label, i) => v(label, plainRectStyle + 'fontSize=16;', 285 + i * 88, 818, 70, 45, 'q'));
  e(null, fifo, { value: '输入', sourcePoint: [90, 847], entryX: 0, entryY: .5 });
  e(fifo, null, { value: '输出', exitX: 1, exitY: .5, targetPoint: [830, 847] });
  v('按进入顺序依次输出', textStyle + 'fontSize=15;', 330, 872, 260, 28, 't');
  return d.out();
}

await mkdir(outputDir, { recursive: true });
const files = [
  ['图1_加速器总体结构.drawio', fig1()],
  ['图2_双缓冲流水线.drawio', fig2()],
  ['图3_节点处理流程.drawio', fig3()],
  ['图4_依赖记分牌流程.drawio', fig4()],
  ['图5_异步控制组件及先进先出缓冲器.drawio', fig5()],
];
for (const [name, content] of files) {
  await writeFile(resolve(outputDir, name), content, 'utf8');
}
console.log(`Generated ${files.length} editable draw.io files in ${outputDir}`);
