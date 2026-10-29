# 多卡边界与验证顺序

本版本新增的是单节点TP实验控制，不是新的分叉算法。`tp_size`改变模型如何分布到GPU，`max_running_requests`控制整个实例的并发上限；TP8、BS32表示8张卡共同处理最多32个请求，不是每张卡各32个。

## 保留的计算路径

原快照中，Llama EAGLE3使用`ParallelLMHead`和`LogitsProcessor`。后者根据TP配置先聚合词表logits，之后draft循环才调用Top1/Top2后处理。自定义Triton后处理没有在本次适配中改成分片概率。这里“完整词表”指各自模型的词表：EAGLE3可能使用缩小的draft词表，阈值概率是在完整draft词表内归一化，不是target词表概率，也不是单卡分片内的概率。节点数、原Top1阈值判断、候选排名和KV路径全部保留。

这项源代码检查不能替代多卡运行验证。预检会记录`LogitsProcessor.forward`的logits宽度并要求完整词表宽度；只有初始化成功还不够。该检查本身不证明所有概率数值与TP1逐字一致。

## Graph

独立pilot对Target Decode、Draft Decode和Target Prefill Graph的实际执行做记录，并要求相关Graph在每个TP工作进程中被观察到。Chain1在seed之后没有额外draft扩展，AR没有draft，不能强制要求它们有Draft Graph调用。真实BS记录在`pilot_*_audit.json`，不把配置32当作每轮32。

审计只在pilot安装，正式运行不会写逐阶段事件或进行profile。Graph捕获仍可能因显存不足失败；不要为了使一边跑通而单独关闭某种方法的Graph。

## 必须在A100节点补充的验证

1. `nvidia-smi`确认40GB/80GB规格，`nvidia-smi topo -m`记录NVLink/PCIe连接。
2. 检查依赖版本、驱动及CUDA兼容性。快照报告`0.0.0.dev0`，不要凭目录名安装另一个版本。
3. 先运行`--pilot-only`，检查每个TP进程、完整词表宽度、Graph调用和接受计数上限。
4. 用同一小批固定上下文，另行比较TP1/TP2（8B模型）以及TP4/TP8。浮点计算次序可能改变greedy边界，文本不一致需要解释，不能直接归为算法错误或宣称任意输入无损。
5. 正式对照始终保持相同模型、TP、设备组、输入和Graph设置。TP4与TP8属于另一条实验轴，不能混算同一均值。
6. 分析吞吐和整批耗时时同时报告输出量、达到输出上限的数量；有代码/数学正确率需求时另做任务评分。

## 工程限制

当前只覆盖同节点、单实例TP，`dp_size=pp_size=nnodes=1`；不支持跨节点、多副本吞吐或流水线并行。默认attention backend保持原实验的`triton`，更换backend必须建立新的对照组。源码、设备或输入不同不能复用已有输出目录。

`--resume`不会改有效输出。失败/干扰运行需要用户先看日志，再显式恢复；不会自动从头重跑。`Ctrl+C`是停止本次运行，不会恢复之前被停止的单卡实验。
