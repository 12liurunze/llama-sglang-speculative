# Llama：SGLang 多卡 Chain / Adaptive2 实验

在**单节点多卡**上，对比 Chain 与原方法 Adaptive2。一个主进程调用 `Engine(tp_size=N)`，由 SGLang 创建 TP 工作进程；**直接用 `python main.py`，不需要 `torchrun`，也不要按卡数启动多份 main.py。**

本项目准备于2026-10-29。CPU测试和源码校验已经完成；**尚未在8卡A100上完成模型运行或性能验证**。运行前先做下面的预检，不能把“支持TP参数”当成“已经测得加速”。

## 1. 保存了哪一版代码

- `vendor/original_runtime.tar.gz`：改动前实际使用的单卡 Llama 实验 SGLang 完整快照，4,886个文件，其中3,812个Python文件。
- `vendor/runtime_manifest.json`：每个文件的SHA256。解压、每次运行前后都检查。
- `reference/`：与当时公用源码相比不同的6份Python文件，便于直接阅读。它们属于**原版本**，不是这次新增的多卡改动。
- `llama_speculative/`：新写的多卡实验入口、数据处理、指标重算、设备监控与报告。
- 原始快照包含既有Top2/候选选择融合和KV优化。**这次没有重写建树算法，也没有再修改KV实现。**

“原始”指本次改动前、单卡测过的原方法版本，并非官方零改动SGLang。源码目录名曾含`0.5.20`，但该开发快照无Git提交信息、包版本为`0.0.0.dev0`；不能据目录名假定它等于PyPI某个发行版。

## 2. 模型和显存

推荐组合：

- Target：`meta-llama/Llama-3.3-70B-Instruct`
- Draft：`lmsys/SGLang-EAGLE3-Llama-3.3-70B-Instruct-SpecForge`

两者必须配套。旧8B实验使用`Llama-3.1-8B-Instruct`及对应8B SpecForge draft，不能将8B draft直接用于70B。

70B的BF16权重约140GB，还需KV、draft及Graph显存。8×A100 40GB建议先TP8；8×A100 80GB可比较TP4/TP8。实际是否容纳目标BS和上下文，由预检确认。多卡不一定比更少卡快，要计入通信开销。

## 3. 环境与源码安装

实际推理需要Linux、NVIDIA驱动、兼容的CUDA/PyTorch环境。Windows可看代码、跑CPU测试及`--dry-run`，不在Windows上解压长路径的内核配置文件。

```bash
git clone https://github.com/12liurunze/llama-sglang-speculative.git
cd llama-sglang-speculative

# 先在已配置的SGLang/CUDA环境中操作
python -m pip install -e .
python tools/install_runtime.py --verify-only
python tools/install_runtime.py
python tools/doctor.py --gpus 0,1,2,3,4,5,6,7
```

`vendor/environment_versions.json`和`vendor/requirements.observed.txt`记录旧环境版本；`vendor/pyproject.upstream.toml`保存上游依赖说明。旧环境是开发版，**不是执行`pip install sglang`就一定能复现**。新节点需要先匹配依赖和CUDA，再预检；不要直接升级生产环境。安装脚本只解压本仓库快照，拒绝覆盖已有`runtime/`，不会替换其他SGLang目录。

原运行时仍按原逻辑在TP通信后处理完整词表logits，多卡预检额外检查输出宽度；没有新增“每卡局部Top2直接当全局Top2”的捷径。更详细的检查边界见[多卡说明](docs/MULTIGPU.md)。

## 4. 准备八个子集

已有数据可直接使用：每个JSONL文件每行须有`prompt`字符串，其他字段不作为答案输入。也可下载上一轮相同版本的公开数据：

```bash
python tools/download_redhat.py --output data/raw

python main.py prepare \
  --files data/raw/HumanEval.jsonl data/raw/math_reasoning.jsonl \
          data/raw/qa.jsonl data/raw/question.jsonl data/raw/rag.jsonl \
          data/raw/summarization.jsonl data/raw/translation.jsonl data/raw/writing.jsonl \
  --model /path/to/Llama-3.3-70B-Instruct \
  --samples 80 --copies 2 --context 4096 --max-new-tokens 1024 \
  --output data/prepared70b.json
```

每文件前80条重复一遍，共160请求，**不是160道独立题**。`question`与`writing`原始输入重复，prepare会记录重合。默认`--copies`为1，复现上一轮才显式设2。使用70B自己的tokenizer重新编码，不复用8B的token IDs；不截断超长输入。

## 5. 八卡预检，再正式测试

先只预检Chain7/Adaptive7：

```bash
python main.py run \
  --data data/prepared70b.json \
  --model /path/to/Llama-3.3-70B-Instruct \
  --draft /path/to/SGLang-EAGLE3-Llama-3.3-70B-Instruct-SpecForge \
  --gpus 0,1,2,3,4,5,6,7 --tp 8 --bs 32 \
  --methods chain7 adaptive7 --threshold 0.5 --repeat 3 \
  --context 4096 --max-new-tokens 1024 --memory-fraction 0.8 \
  --output results/70b_tp8 --pilot-only
```

预检用最多32请求、最多128输出token，只检查兼容性、各TP进程的Graph执行及完整词表logits。预检不作为正式性能结果。通过后，**原命令去掉`--pilot-only`并加`--resume`**，会跳过完成的预检，继续48次正式实验。

不想加载模型，只检查参数与计划：将`--pilot-only`改成`--dry-run`，并使用新的输出目录。dry-run不需要GPU，也不需要加载SGLang或权重。

支持的方法：

|参数|含义|S/N|前沿宽度|
|---|---|---|---:|
|`chain3`|单链投3个|3/4|1|
|`chain5`|单链投5个|5/6|1|
|`chain7`|单链投7个|7/8|1|
|`adaptive3/5/7`|原Adaptive2，相应节点预算|预算/预算+1|2|
|`ar`|自回归，只加载target|不适用|不适用|

Adaptive保持原规则：每层对有效父节点计算Top1概率，`p1<threshold`时允许替代孩子，按累计路径概率选择候选并扣节点预算；已经分叉不代表后续必定分叉。N包含额外的target验证位置，所以投5个设S5/N6。

正式运行使用BF16、greedy、seed42，Decode Graph和breakable Prefill Graph开启。BS为并发上限，不代表每轮实际都是32。每个子集/方法/遍次使用独立引擎，方法顺序交替。

## 6. 停止、恢复、报告

- `Ctrl+C`会停止本次实验自己的进程组，保留已完成结果；不会杀其他用户进程。
- 默认所选GPU有进程就报错。需要等空闲时添加`--wait-free`，没有固定60秒等待。
- 运行中监控所有选中GPU，发现外部进程时只停止本实验并保留记录，不自动抢占或重试。
- 恢复必须显式使用`--resume`，且模型、输入、方法、TP、设备与参数保持一致。未完成目录先归档，不覆盖有效结果。

```bash
python main.py report --output results/70b_tp8
# 未完成时只看已有结果
python main.py report --output results/70b_tp8 --allow-partial
```

报告包含每次原始计数和各方法的均值/样本标准差、吞吐、接受长度、接受率、整批耗时、输出token数、触及上限的请求数及重复输出一致性。

- 吞吐=`输出token总数/整批生成耗时`。
- 接受长度=`接受draft总数/请求验证轮次数`，不含target额外token。
- 接受率=`接受draft总数/(节点预算×请求验证轮次数)`。不同预算的分母不同。
- 耗时包含prefill、调度和decode；不含加载、Graph捕获、预热和写盘。
- AR的接受率/接受长度为不适用；输出重复不等于答案正确。

## 7. 从哪里读代码

1. `llama_speculative/common.py`：Chain/Adaptive预算如何转成SGLang参数。
2. `llama_speculative/runner.py`：多卡选择、独立运行、停止和恢复。
3. `llama_speculative/worker.py`：一轮完整生成如何进入`sgl.Engine`。
4. `reference/srt/speculative/eagle_worker_v2.py`：原draft循环和target验证流程。
5. `reference/srt/speculative/spec_utils.py`：原Adaptive父节点分叉、候选选择和预算扣除。
6. `reference/kernels/ops/speculative/topk1.py`：原Top1/Top2后处理融合。
7. `reference/kernels/ops/kvcache/cache_move.py`：原有KV搬移优化。

解压后的`runtime/sglang/`保存其余完整源码。实验入口通过单独`PYTHONPATH`使用它，不更改系统安装。

## 8. 验证状态

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

CPU测试覆盖TP4/TP8参数、相同节点预算、错误设备配置、AR不加载draft、数据重复/超长输入、计数公式、干净环境、dry-run和不完整报告。完整归档也经过逐文件SHA校验。GPU多卡数值一致性、70B显存容量与端到端性能必须在目标A100节点另行验证。

快照中的SGLang代码沿用Apache-2.0，见`vendor/LICENSE.sglang`及原文件版权声明。本仓库不包含模型权重、数据集、SSH凭据或历史生成输出。
