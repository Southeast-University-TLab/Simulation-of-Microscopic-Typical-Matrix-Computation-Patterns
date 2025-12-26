# Simulation-of-Microscopic-Typical-Matrix-Computation-Patterns
1. 摘要：这是一种基于微观典型矩阵计算模式的矩阵计算仿真算法。
2. 背景：大规模矩阵乘法是高性能计算与人工智能领域的核心计算内核，其效率直接决定大模型训练等应用的性能上限。专用 AI 加速器理论峰值算力虽高，但受“内存墙”影响，实测效率与理论值差距显著，传统静态优化策略难以适配复杂场景。通过构建硬件性能仿真模型，可在软件层面优化矩阵运算的计算与数据调度策略，充分释放昇腾等处理器的硬件潜能。
3. 方法：
   - 基于NPU多层硬件结构模型分批次建模算法模块：NPU多层硬件结构模块、矩阵分块策略、分批次调度机制、双缓冲数据传输优化、各层级带宽效率模型、计算与传输的延迟计算、FixPipe层的简化处理。
   - 离散事件并行流水线仿真算法模块：模块化分层、数据与计算的时间模型、分层切块与核映射、L2缓存管理与命中口径、离散时间仿真DES与并集时间、策略搜索与Tile选择。
1. Abstract: This is a matrix computation simulation algorithm based on the microcosmic typical matrix computation pattern.
2. Background: Large-scale matrix multiplication is the core computing kernel in the fields of high-performance computing and artificial intelligence, and its efficiency directly determines the performance upper limit of applications such as large model training. Although the theoretical peak computing power of dedicated AI accelerators is quite high, affected by the "memory wall" problem, there is a significant gap between the measured efficiency and the theoretical value. Traditional static optimization strategies are difficult to adapt to complex scenarios with variable matrix dimensions and limited hardware resources. By constructing an accurate hardware performance simulation model, the computation and data scheduling strategies of matrix operations can be optimized at the software level, thereby fully releasing the hardware potential of processors such as Ascend series.
3. Methodology:
   - Batch-wise modeling algorithm module based on the NPU multi-layer hardware structure model: NPU multi-layer hardware structure module, matrix tiling strategy, batch-wise scheduling mechanism, double-buffered data transmission optimization, bandwidth efficiency model of each level, latency calculation of computation and transmission, simplified processing of the FixPipe layer.
   - Discrete event parallel pipeline simulation algorithm module: modular hierarchical design, time model of data and computation, hierarchical tiling and core mapping, L2 cache management and hit criterion, discrete event simulation (DES) and union time, strategy search and tile size selection.
