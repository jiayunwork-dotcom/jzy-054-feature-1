# 傅里叶变换教学工具（时域 / 频域实验室）

一个跑在浏览器里的傅里叶变换教学工具：学生在前端叠加波形、手绘信号、
拨采样率与补零长度、换窗、框选频段滤波、拖动采样率观察混叠；所有 DFT /
IDFT / 窗函数 / 滤波 / 混叠判定算术都由一个可独立检验的 Python 后端完成，
关键恒等式由自动化测试逐条守住。新增的短时傅里叶分析（STFT）把信号切成
互相重叠的小段、逐段加窗逐段变换，在一张二维时频热力图上展示
"哪个频率在哪个时刻、有多强"，并支持把帧频域表示逆变换拼回原始信号。

- 前端：TypeScript + Vite + 原生 Canvas（无第三方 UI 框架），Node.js 20 构建
- 后端：Python 3.12 + FastAPI，手写 Cooley–Tukey FFT 内核（不依赖 numpy.fft）
- 编排：Docker Compose（前端 nginx 静态托管并反代 `/api`）

## 一键启动（Docker Compose）

```bash
docker compose up --build
# 浏览器打开 http://localhost:8080
# 后端 API（含 /docs Swagger）：http://localhost:8000
```

## 本地开发

后端（Python 3.12）：

```bash
cd backend
python3.12 -m pip install -r requirements-dev.txt
python -m uvicorn app.main:app --reload --port 8000
python -m pytest          # 145 个测试
```

前端（Node.js 20）：

```bash
cd frontend
npm install
npm run dev               # http://localhost:5173 ，/api 已代理到 :8000
npm run build             # 类型检查 + 产物构建到 dist/
```

## 功能地图

| 区域 | 交互 |
| --- | --- |
| ① 信号构造 | 叠加最多 8 个正弦/余弦/方波/三角波/锯齿波，调振幅/频率/初相；或手绘任意波形并按当前 N、fs 离散 |
| ② DFT 与窗 | 选 N（64–1024）、fs、补零长度 N′；矩形/汉宁/汉明/布莱克曼/β 可调凯泽窗；幅度谱/相位谱/功率谱（Hz 横轴）多窗彩色叠加 + 主瓣宽度/旁瓣衰减表 |
| ③ 采样定理 | 拖 f 与 fs 滑块；密集原始信号、离散采样点、sinc 重建信号同图；f>fs/2 时按后端判据标红并给出表观频率 |
| ④ 频域滤波 | 在幅度谱上拖选频段或手填截止频率，低通/高通/带通；后端置零带外 bin 并 IDFT，红色虚线叠加对比 |
| ⑤ 短时傅里叶分析 | 拖段长 N（32–512 档位）、挪动步长 H（1–N）、选窗（含凯泽 β）；Canvas 二维热力图（时间×频率×能量），悬停读数；页面实时显示 Δt=N/fs 与 Δf=fs/N 两个分辨率数字；一键载入线性扫频信号看对角线，并可检验分帧往返重建误差 |

## HTTP 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/dft` | 加窗（可多窗）+ 补零后的 DFT，返回实部/虚部/幅度/相位/功率/dB |
| POST | `/api/idft` | 频谱逆变换回时域 |
| POST | `/api/windows` | 窗系数序列 + 主瓣宽度(bin)/最高旁瓣(dB) |
| POST | `/api/filter` | 频域置零 + IDFT，附能量报告 |
| POST | `/api/sampling` | 采样定理演示的三条曲线 + 混叠判据与表观频率 |
| POST | `/api/stft` | 重叠分帧、逐帧加窗 DFT，返回时间×频率幅度/功率矩阵、时间中心与真实频率轴，以及 Δt/Δf |
| POST | `/api/istft` | 喂入整串帧的完整频域表示，加权 overlap-add 拼回时域序列 |

约定：正向变换不归一化、逆变换除以 N（与 numpy.fft 一致）；
帕塞瓦尔关系为 `Σ|x[n]|² = (1/N)Σ|X[k]|²`。

非法输入一律返回 HTTP 400 并在 `detail` 中说明原因：N 不在允许集合、
fs ≤ 0、未知窗名、凯泽缺 β、滤波频段越界/上界不大于下界、
补零短于原信号等。全零信号等合法退化情形正常返回全零频谱。

时频分析的分帧规则：段长必须落在允许档位 `{32, 64, 128, 256, 512}`；
挪动步长必须为正整数且不超过段长（否则帧间漏掉采样点）；信号短到连
一整段都凑不满时返回 400。信号正好只够一帧（L=N）是合法退化：
热力图只有一列。为让逆变换在信号两端也成立，多帧布局会自动补一个
头部上下文帧与若干尾部零填充帧；逆变换用分析窗作为合成窗、按
逐样本累加的窗平方归一化（weighted overlap-add）。汉宁窗 50% 重叠
（以及更密的重叠、矩形窗不重叠等 COLA 组合）可机器精度重建原信号。

## 代码结构

```
backend/app/
  dft.py          # 手写 radix-2 FFT / 直接 DFT / IDFT / analyze(加窗+补零)
  windows.py      # 五种窗 + I0 贝塞尔级数 + 主瓣/旁瓣实测
  spectrogram.py  # 重叠分帧(frame_starts/extract_frames) + STFT + 加权重叠相加 ISTFT
  filtering.py    # 频段校验、共轭对称置零、IDFT、能量报告
  aliasing.py     # 奈奎斯特判据、表观频率折叠、sinc 重建
  services.py     # 产品级规则校验与编排
  api(main.py)    # FastAPI 接口层（薄）
  schemas.py      # Pydantic 模型
  config.py errors.py
backend/tests/    # test_dft / test_windows / test_filtering / test_aliasing
                  # / test_spectrogram / test_api

frontend/src/
  components/
    SignalBuilder.ts     # ① 分量叠加 + 手绘离散
    WindowCompare.ts     # ② 窗选择与指标表
    SpectrumView.ts      # ② 三谱绘制 + 频段框选
    AliasDemo.ts         # ③ 采样定理
    FilterPanel.ts       # ④ 频域滤波
    SpectrogramPanel.ts  # ⑤ 段长/步长/窗滑块 + 热力图 + 悬停读数 + 往返检验
  util/plot.ts           # 纯 Canvas 线图绘图器
  util/heatmap.ts        # 纯 Canvas 二维时频热力图（ImageData + 自配色带）
  util/heat.ts colors.ts # 热力 LUT / 窗颜色
  util/builder.ts        # 波形合成 / 手绘重采样
  api.ts store.ts types.ts main.ts
```

## 测试守住的恒等式与判据

- 正变换 → 逆变换数值还原（含非 2 的幂长度的直接 DFT，并与 numpy.fft 对拍）
- 帕塞瓦尔能量守恒
- 实信号频谱共轭对称（DC/Nyquist bin 为实数）
- 单频正弦落在 bin 上时仅该 bin 与镜像 bin 有峰
- 全零信号频谱处处为零；补零是对同一 DTFT 的插值（粗网格点精确再现）
- 混叠判据 `f > fs/2` 与表观频率折叠公式
- 低通后高于截止频率的谱能量与残余能量为零
- **单频正弦**逐帧能量集中在同一条频率行（热力图水平亮线，不东飘西荡）
- **线性扫频**各帧峰值频率行随帧序号单调抬升（斜线走向由程序判定，
  且斜率对应扫频速度、峰值频率跟踪瞬时频率）
- **分帧往返恒等式**：汉宁窗 50%/75%/87.5% 重叠、矩形窗不重叠、
  布莱克曼 75% 重叠、凯泽/汉明等组合下，STFT → 逐帧 IDFT →
  加权重叠相加，在 1e-10 容差内重建原始信号（含非整倍长度）
- 时间中心按 H/fs 等间隔、首帧中心在 t=0；频率行按 fs/N 换算到 fs/2
- 信号正好一帧时只有一列；段长非档位、步长非正或超过段长、
  信号短于一帧、重建帧数/帧宽对不上均返回 400
- 全部非法输入返回 400
