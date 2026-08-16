# TCIM backend 按算子增量实现调研

日期：2026-08-16<br>
DS4 源码基线：`84cc882352757baf628a1776badf7cc54d584e28`

## 范围与口径

本文只使用以下一手资料推导方案：DS4 的实际执行图 `ds4.c`、通用后端契约 `ds4_gpu.h`、仓库 Git 历史，以及本机 Houmo/TCIM 示例源码。当前暂存方案中 Makefile 罗列的 TCIM 算子和测试被明确排除，**不作为需求、实现顺序或验收证据**。

“已支持的 3 个 backend”按项目 README 的口径指 Metal、NVIDIA CUDA 和 ROCm（`README.md:12-17`）。CPU 虽出现在公共枚举中（`84cc882:ds4.h:19-23`），项目规则明确把它定位为 reference/debug，而不是生产性能后端（`AGENT.md:9-13`；`CONTRIBUTING.md:69-77`）。

## 结论

1. **可以一个算子一个算子地开发、验证和合入 TCIM。** TCIM runtime 可以从任意模型文件加载 `Module`，设置输入，异步 `Run(false)` 后 `Sync`（`/work/houmo-examples/hmatc/hmatc/python/tcim_perf.cpp:497-552`）；多个加载实例还能复用同一 `WeightManager` 并绑定 stream（同文件 `:673-715`）。因此，小 HMM module 作为 bring-up 和正确性载体在技术上可行。
2. **“逐算子合入”不等于“每个中间提交都能公开一个完整 `--tcim` backend”。** DS4 的 tensor、KV 和 scratch 在整个 prefill/decode 序列中保持 device-resident（`84cc882:ds4_gpu.h:15-18`）。只有标为 optional fused hook 的函数允许返回 0 后走 portable primitive 回退（`84cc882:ds4_gpu.h:634-640`）；缺失必需 primitive 会令 `ok` 链失败并中止图执行（例如输出头 `84cc882:ds4.c:25148-25234`，session 失败处理 `84cc882:ds4.c:59597-59604`）。
3. **现有三个生产后端首次进入 canonical `main` 都不是逐算子小提交。** Metal 随初始发布一次进入，CUDA 和 ROCm 也分别以完整大提交进入；不过 GitHub 补充审计表明，ROCm 在进入 `main` 前已有公开 fork/PR 孵化史，之后三个后端也都继续按算子/子系统修正确性、扩展功能、增加融合并回滚失败实验（提交证据见下表及补充审计）。
4. **建议路线是“私有算子孵化 → 通用 ABI 闭环 → 单层纵切 → 最小端到端 → 公开激活 → 扩功能/做融合”。** 不应先把 TCIM 宣称为公共 backend，再用大量返回 0 的 stub 填满符号表；返回 0 只对明确的 optional hook 有回退语义。
5. **生产形态不必永久保持“一算子一 module”。** Houmo 的 LLM 示例使用共享 `WeightManager` 的 prefill/decode 两个大 module（`/work/houmo-examples/tools/llm_perf/src/llm/HmllmInfer.cc:34-80`），并用 `GetDevInput`/`SetDevInput` 共享设备 KV（同文件 `:107-127`）。逐算子 module 适合 bring-up；性能稳定后应按调度、数据驻留和 dispatch 成本决定是否合并为更大的图段。

## DS4 架构允许什么、不允许什么

### 1. 通用 ABI 已经给出了真实的工作分解

`ds4_gpu.h` 不是面向任意算子的注册表，而是一组直接调用的 C 符号。其真实契约按执行职责分为：

- tensor/command 生命周期（`84cc882:ds4_gpu.h:11-80`）；
- embedding/indexer（`:397-403`）；
- dense projection、norm、RoPE、KV rounding/store（`:553-559`）；
- KV compression/attention（`:1776-1782`）；
- router/shared expert/routed MoE（`:2228-2234`）；
- HyperConnection（`:2586-2592`）。

这组职责可以作为覆盖面核对表，但具体 PR 单元应定义为 **`op × dtype × shape × mode`**，而不是笼统地写“已支持某算子”。同一个 matmul 的 F16/Q8、4096×512/4096×256、decode 单行/prefill 多行是不同的实现和验收单元。

### 2. optional hook 可以后补，required primitive 不可以假装成功

仓库已经有三个清晰的 portable fallback 例子：

- QKV pair/quad 融合未命中时退回 pair 或普通 dense projection（`84cc882:ds4.c:22224-22345`）；
- compressor 的 fused store 未命中时退回 pair，再退回两个 F16 matmul（`84cc882:ds4.c:22660-22703`）；
- RMS-folded F16 matmul 未命中时退回 plain RMSNorm rows 加普通 F16 matmul（`84cc882:ds4.c:27793-27825`）。

这也是 ROCm 现状的直接先例：不可选择的 CUDA-only/fused/multi-device hook 集中返回 0（`ds4_rocm_unavailable.cu:1-39`），能够由基础 primitive 组合出的接口则在兼容层实现，例如 exact rows 复用普通 matmul、QKV norm+RoPE 由两个基础调用组合（`ds4_rocm_compat.cu:217-259`）。该分层由提交 `d924de3ab0c3d316c172d7fd550d6e698faa7905` 明确化，提交标题即 “Refactor fused GPU ops as optional backend hooks”。

相反，输出头连续要求 plain RMSNorm、F16 projection、HC weights、HC weighted sum、weighted RMSNorm 和 vocab projection；任一步失败都会让整个函数返回 false（`84cc882:ds4.c:25148-25234`）。因此，“为所有缺失函数放一个 return 0”只能让链接通过，不能得到可用 backend。

### 3. 实际图决定依赖前沿

现有 layer-0 CPU/GPU 对照测试已经把单 token 的真实依赖顺序写清楚：

`embedding → HC pre → attention norm → Q/KV projection → RoPE/KV rounding → attention → attention output → HC post → FFN HC pre/norm → shared expert + router/routed MoE → FFN HC post → output head`

CPU reference 的顺序见 `84cc882:ds4.c:26426-26479`；同一测试随后执行通用 GPU 图（`:26481-26512`），读回各中间 stage（`:26514-26531`）并逐 stage 打印差异（`:26534-26552`）。这比人为列一份待办算子表更适合作为 TCIM 的纵向集成判据。

### 4. decode-only 不是当前 session 的完整 backend

正常 session 会对新 prompt 或较长 checkpoint suffix 走 batched/chunked prefill，对较短 suffix 才逐 token decode（`84cc882:ds4.c:60500-60628`）。所以只完成单 token decode 可以作为里程碑，却不能在不改 capability/调度的情况下被描述为完整公共 backend。若希望早期仅支持 decode，必须增加显式能力判断并强制 prompt 逐 token 执行；这会改变产品语义和性能，应作为单独设计决策，而不是隐式 fallback。

## 三个现有 backend 的历史路径

| Backend | 首次进入 canonical `main` | 进入 `main` 之后如何逐步演进 | 对 TCIM 的含义 |
|---|---|---|---|
| Metal | 初始提交 `d997b56c151184bcff469dd8302ed97f23481024` 已包含 14,491 行 `ds4_metal.m`、dense/HC/KV/RoPE/attention/MoE/norm 等 kernel，以及整机推理和测试；该提交总计 59,189 行新增。仓库里没有更早的逐算子 Metal 落地历史。 | 随后按问题域修复和优化：debug buffer 校验 `f9e8715`、short-prompt prefill `5cbee23`、NAX 初版/加速 `63ceed6`/`18c2d4b`；routed MoE TensorOps 因不稳定被限制和关闭 `3d14d1c`/`d4fba7b`；wide-token MoE 实验后来回滚 `5224654`/`f183c19`/`072bc0f`；更晚才逐个增加 decode 融合，如 `71d8c2a`、`b64d3e0`、`ad4d05c`。 | 历史模式是先有完整 correctness floor，再逐算子优化，并允许回滚不稳定快路径。 |
| CUDA | `48beef81a017a3785fd449a6f3cb34eb849b0e1f` 一次加入 9,666 行 `ds4_cuda.cu`，连同 engine/API/CLI/server 共 10,003 行新增；不是逐算子公开。 | 完整 CUDA 进入后，`0ac5df3e65fbc6d8ad4ce21bf66bd1d790a718a6` 才把 `ds4_metal.h` 重构成通用 `ds4_gpu.h`。之后逐项修 long context `320b779`、Q8 FP16 cache 内存 `a97e7a3`、compressor 清零 `ad8a926`；MoE block16 实验先加入后移除 `e85c051`/`0230891`；再扩 Q4 routed MoE `2791d27`/`dc51d64` 和单项 prefill/indexed-attention 优化 `08fecd9`/`45f4ef9`/`cfaee47`。 | 通用 ABI 是完整 CUDA port 后沉淀出的接口；TCIM 可直接利用它，但仍要先闭合 required surface。 |
| ROCm | `586abbfb625434a95372303319a66c90cf94ba12` 一次新增 17,612 行，已同时包含 runtime、matmul、norm/RoPE、KV、attention、compressor、indexer、router、MoE、HC 等模块。 | 次日通过 `d924de3` 把 fused op 明确成 optional hook；之后逐项补 SSD/MTP `526dad2`/`6273c8a`/`bbd069d`、GLM `ef8d923`，并修 IQ2 MoE、indexer、batched HC、SSD deadlock `507fd7c`/`fa8b0b9`/`45a9dcb`/`2f49b27`。`ef8d923` 还新增集中式 compat/unavailable 文件；无效 compact-cache 路径也经历加入和删除 `312934c`/`5e61ad5`。 | ROCm 最直接证明了“基础 primitive + 兼容组合 + optional stub + 后续单项增强”的可维护性，但它的首次公开提交仍然是端到端大提交。 |

历史结论不是“TCIM 也必须一次写完”。Metal/CUDA 的 canonical history 没有留下首发前的增量实现轨迹；ROCm 的 `main` 也表现为一次性落地，但 GitHub 的关闭 PR 与实现作者 fork 实际保留了较细的孵化史，详见下节。TCIM 应保留现有后端后期有效的 correctness/fallback 方法，并把 ROCm 孵化期已经出现的细粒度提交方式做得更干净、更可审查。

## GitHub 公共历史补充审计

本节补充核验 GitHub 上的默认分支、公开 branch/tag、关闭 PR、PR commits 和实现作者公开 fork。它修正了只看本地 `main` 时容易得到的过度简化结论：**canonical `main` 看起来是三次大落地；ROCm 在合入前确实存在公开、细粒度的重构、正确性和性能演进历史，但可见起点仍是完整 backend 快照，并不是从空白逐算子实现；Metal 和 CUDA 没有找到同等级的首发前轨迹。**

复现工作树位于 `/work/ds4.upstream`：它检出官方 `main` 的 `84cc882352757baf628a1776badf7cc54d584e28`，是 clean、non-shallow 仓库，`main` 有 501 个可达提交，并保存审计时可见的五个 upstream branch refs。由于当前环境代理拒绝直接 `git clone` GitHub，工作树由 `/work/ds4` 已完整同步的 `upstream/*` 对象创建，再把 `origin` 设为 `https://github.com/antirez/ds4.git`；未复制任何未提交 TCIM 改动。commit/PR/API 页面则直接从 GitHub 交叉核验。

### canonical 仓库：三个首发提交本身仍然都是“大提交”

- Metal 的 [`d997b56`](https://github.com/antirez/ds4/commit/d997b56c151184bcff469dd8302ed97f23481024) 是零父提交，直接加入 49 个文件、59,189 行；因此 `antirez/ds4` 的 Git 图本身不可能包含更早的 Metal 逐算子祖先。
- CUDA 的 [`48beef8`](https://github.com/antirez/ds4/commit/48beef81a017a3785fd449a6f3cb34eb849b0e1f) 是单父提交，一次加入完整 `ds4_cuda.cu`。维护者此前在 [issue #34](https://github.com/antirez/ds4/issues/34) 公开说明 CUDA 正在优化、验证 KV cache，并将在可行后整体落地，但没有公开中间实现提交。GitHub 的 [commit-associated-PR API](https://api.github.com/repos/antirez/ds4/commits/48beef81a017a3785fd449a6f3cb34eb849b0e1f/pulls) 对该提交返回空数组，说明它不是 GitHub 当前可关联到的 PR merge/cherry-pick。
- ROCm 的 [`586abbf`](https://github.com/antirez/ds4/commit/586abbfb625434a95372303319a66c90cf94ba12) 也仍是单父提交，一次加入 32 个文件、17,612 行；其 [associated-PR API](https://api.github.com/repos/antirez/ds4/commits/586abbfb625434a95372303319a66c90cf94ba12/pulls) 同样为空。当前 canonical 仓库的 [公开 branches](https://api.github.com/repos/antirez/ds4/branches?per_page=100) 只有 `main` 和若干后期 feature branch，[tags API](https://api.github.com/repos/antirez/ds4/tags?per_page=100) 为空；已经删除的旧 ROCm 分支不能从普通 branch 列表恢复。

所以，只看 `main` 的结论仍成立：三个**被支持版本**首次出现时都是完整落地，不能从这三条 canonical commit 直接观察逐算子开发过程。

### ROCm：关闭 PR 保留了第一条公开孵化链，但不是 landing commit 的祖先链

ROCm 在正式进入 `main` 前，GitHub 已经有多轮公开试验：

1. [PR #79](https://github.com/antirez/ds4/pull/79) 最初用一个提交增加 HIP/ROCm 兼容；维护者明确提出先维护独立 `rocm` branch，并在评论中说明 README 已指向该分支。该 PR 随后关闭而非合并。
2. 同一工作经过 [PR #133](https://github.com/antirez/ds4/pull/133)、[PR #180](https://github.com/antirez/ds4/pull/180) 和 [PR #290](https://github.com/antirez/ds4/pull/290) 多次 rebase、force-push、修性能和修 correctness。#180 的作者因 logprob 回归主动关闭；#290 又因 expert-selection 精度问题关闭，说明这条历史同时保存了失败实验和验收门槛。
3. [PR #311](https://github.com/antirez/ds4/pull/311) 最终集中修正 gfx1151 expert-selection 精度、indexer 和 agent 路径，于 2026-06-07 关闭；同日晚些时候 `586abbf` 才进入 `main`。

同期还有独立的 [PR #118](https://github.com/antirez/ds4/pull/118)：它用 9 个提交维护一份约 10k 行的单体 `ds4_hip.cpp`，同样关闭且未合并。它说明 GitHub 保存了不止一种 ROCm port 尝试，但其代码组织与最终模块化 landing 不同，不能混进正式实现的线性历史。

这条 PR 链值得研究，但不能误写成 `586abbf` 的 Git ancestry：这些 PR 均显示为 closed/unmerged，且 `586abbf` 没有关联 PR。早期 [PR #79 的文件范围](https://github.com/antirez/ds4/pull/79/files) 主要是 `ds4_rocm.h` 加少量 CUDA/HIP 兼容修改，而正式 landing 采用独立 `ds4_rocm.cu` 和模块化 `rocm/*.cuh`；它们是孵化输入和设计演进证据，不是一条可直接 `git log 586abbf^..586abbf` 得到的线性实现序列。

### ROCm：实现作者 fork 保留了细粒度模块演进，但不是从零 bring-up

`586abbf` 的作者 Nick Parrin 对应 GitHub 用户 `ejpir`；其公开 fork [`ejpir/ds4-hip`](https://github.com/ejpir/ds4-hip) 的仓库元数据表明它 fork 自 `antirez/ds4`。其中 [`rocm-upstream-shape-cyberneurova` branch](https://github.com/ejpir/ds4-hip/commits/rocm-upstream-shape-cyberneurova/) 保留了正式落地前的详细过程，代表性提交包括：

先明确这条历史的边界：该 fork 更早的 [`c91ea47`](https://github.com/ejpir/ds4-hip/commit/c91ea47230e3c93c606d1fe5a9ea691829ea351c) `draft` 已一次改动 21 个文件、增加 9,938 行，并加入完整 `ds4_hip.cpp` 路径；下述提交是在完整/近完整 HIP baseline 之上迁移为 upstream-shaped 模块、修 correctness 和增加窄能力。它证明“小提交维护 backend”可行，但不能作为“从第一个 primitive 开始逐算子诞生 backend”的历史证据。

- [`a99e540`](https://github.com/ejpir/ds4-hip/commit/a99e5408fdf0) 建立上游形态的 ROCm MoE/quality 基线；
- [`acbf121`](https://github.com/ejpir/ds4-hip/commit/acbf121d4a5a) 把 ROCm backend 拆成独立 translation unit；
- 随后连续把 indexer、compressor/router、norm/matmul、attention、shared expert、HC output、MoE launch 和剩余 wrapper 分离到模块：[`9fb3c25`](https://github.com/ejpir/ds4-hip/commit/9fb3c2542759)、[`d5d6f1e`](https://github.com/ejpir/ds4-hip/commit/d5d6f1eb2086)、[`8e98030`](https://github.com/ejpir/ds4-hip/commit/8e98030b4b5e)、[`87751cb`](https://github.com/ejpir/ds4-hip/commit/87751cb98fb0)、[`906f2b6`](https://github.com/ejpir/ds4-hip/commit/906f2b68985d)、[`42ceaf9`](https://github.com/ejpir/ds4-hip/commit/42ceaf936796)、[`5a65561`](https://github.com/ejpir/ds4-hip/commit/5a6556113a44)、[`30efb3a`](https://github.com/ejpir/ds4-hip/commit/30efb3a5b7b0)；
- 再以较小提交改进具体能力：decode Q norm+RoPE [`3648a29`](https://github.com/ejpir/ds4-hip/commit/3648a29cc680)、MoE decode gate/up [`0f50ab6`](https://github.com/ejpir/ds4-hip/commit/0f50ab6dc4f5)、shared-expert overlap [`c9f8820`](https://github.com/ejpir/ds4-hip/commit/c9f88209edfb)、mixed IQ2/Q2 prefill [`9502d16`](https://github.com/ejpir/ds4-hip/commit/9502d167ea80)、F16 decode matvec [`44e5cd7`](https://github.com/ejpir/ds4-hip/commit/44e5cd7f3dfb)、GPU API hardening [`a3c47db`](https://github.com/ejpir/ds4-hip/commit/a3c47db3d34c)，以及 Q4_K MoE prefill [`1b37bd9`](https://github.com/ejpir/ds4-hip/commit/1b37bd929ec3)/[`5d6b1eb`](https://github.com/ejpir/ds4-hip/commit/5d6b1eb23082)；
- Donato Capitella 最后又以独立提交移植 indexer/router correctness 修复 [`21fcba2`](https://github.com/ejpir/ds4-hip/commit/21fcba2e9015)，并修正 indexer namespace [`50eb72b`](https://github.com/ejpir/ds4-hip/commit/50eb72b8628c)。

这不是仅仅“名字相似”的旁支。该 fork 在 [`3490c2e` 的 `rocm/` tree](https://github.com/ejpir/ds4-hip/tree/3490c2e46c91331323dc0f2bfb7d3018e227fdff/rocm) 已具有正式 landing 的模块结构；与 [`586abbf` 的 `rocm/` tree](https://github.com/antirez/ds4/tree/586abbfb625434a95372303319a66c90cf94ba12/rocm) 对比，`ds4_rocm.h`、common、compressor、embedding launch、FP8 KV、FP8 KV launch、misc launch、output 等至少八个 Git blob 完全相同，其余核心文件是在 landing 前后继续调整的版本。结合 `586abbf` 对 Nick、Donato、alantsev 和 antirez 的 co-author 标注，可以确认该 fork/PR 孵化史与最终 ROCm backend 直接相关；只是 canonical 合入时被整理成了一个提交。

### 对 TCIM 提交策略的修正

GitHub 历史给出的更准确先例是：

- Metal/CUDA 没有公开的首发前细粒度历史；只能借鉴其落地后的逐项修正。
- ROCm 证明了完整 backend 在进入 `main` 前，可以在公开 fork/孵化分支上按模块、能力和 correctness 问题逐步重构、验证和增强，最终再以端到端版本落地；它**没有**证明初始 backend 曾从零按算子逐个实现。
- ROCm 孵化史也展示了应避免的做法：反复 force-push/rebase 的超大 PR 会混入大量 upstream commits，使 PR 的 “Commits/Files changed” 难以审查；最后 squash/direct landing 又让 canonical `git log` 丢失设计过程。

TCIM 因此可以采用更清晰的两层历史：在长期可访问的 incubation branch/stacked PR 中，每个提交只实现一个 `op × dtype × shape × mode` 或一个窄 runtime seam，并保留失败/回滚证据；达到最小端到端 feature matrix 后再激活公共 backend。合入 `main` 时应尽量保留这些提交或在 landing commit 中链接完整 PR stack，而不是只留下一个无法追溯的整包提交。

## TCIM runtime 对增量实现的支持与约束

### 可行点

- `Module::LoadFromFile` 不要求模型必须是完整 LLM；标准性能工具对传入的模型文件统一执行输入绑定、Run 和 Sync（`/work/houmo-examples/hmatc/hmatc/python/tcim_perf.cpp:491-552`）。据此可以推断每个 `op × dtype × shape × mode` 都能先以独立 HMM artifact bring-up。
- `WeightManager`、`Option` 和 stream 可复用到多个加载/执行线程（同文件 `:673-715`），所以不必为每个算子重复建立整套设备上下文。
- 官方 LLM 示例能在 prefill/decode module 间绑定同一设备 KV tensor（`/work/houmo-examples/tools/llm_perf/src/llm/HmllmInfer.cc:107-127`），说明跨 module 保持设备状态有现成 API 先例。

### 必须验证的约束

- 设备 tensor 的 ownership、view/offset、输入输出重绑定和生命周期必须先映射到 `ds4_gpu_tensor`，因为 DS4 明确要求中间状态跨整段命令序列 device-resident（`84cc882:ds4_gpu.h:15-18,47-61`）。
- `Run(false)` 是异步的，而示例紧接着显式 `Sync`（`tcim_perf.cpp:525-552`）。bring-up 时可以每算子同步以定位误差；生产路径若每算子都 `Run + Sync`，会引入同步/dispatch 开销，必须量化，并优先尝试同 stream 排队、只在图边界同步。
- Houmo LLM 示例选择两个大 module，并在每个 prefill/decode module 级别 Run/Sync（`HmllmInfer.cc:301-305,358-360`）。这不是“小 module 不可行”的证据，但说明最终 production 分块应由性能数据决定，而不是把开发提交粒度固化为运行时粒度。

### 当前暂存实现适合做 spike，不适合直接作为首个 backend 提交

即使完全不看 Makefile 的算子/测试清单，代码形态本身也说明当前切片过大：它同时加入公共 backend 枚举（`ds4.h:23`）、CLI/server/bench 选择入口，以及一份 3,843 行的 `ds4_tcim.c`。该文件还直接依赖当前仓库中不存在的 runtime/attention/compressor/indexer/MoE/output-head 头文件（`ds4_tcim.c:3-10`），所以本机无法据此完成链接或设备行为验证。

它也暴露了后续必须解决的性能问题。以 F16 matmul 为例，每次调用都会把 graph tensor 拷到 module-local input、重新上传权重、执行 `Run + Sync`，再把 output 拷回 graph tensor（`ds4_tcim.c:2088-2112`）；当前 command begin/end 只是布尔状态，而全局 synchronize 能直接成功，是因为每个 HMM owner 已在返回前同步（`:3363-3383`）。这适合作为逐算子正确性 bring-up，但不能直接证明生产流水线可接受。建议保留其中已经验证过的契约知识，将提交重切为下面的阶段，而不是把整份暂存实现作为一个 PR。

## 建议的分阶段路线

### 阶段 0：只落 runtime/tensor substrate，不公开 backend

建立 TCIM runtime 初始化/清理、共享 `WeightManager`/stream、tensor alloc/view/read/write/copy、command begin/end/synchronize 的最小适配层。先用无模型或单一小 module 验证 ownership、错误传播、重复执行和资源释放。此阶段不增加公共 backend 枚举、CLI、bench 或 server 入口。

验收依据是 `ds4_gpu.h` 的 tensor/command 契约（`84cc882:ds4_gpu.h:11-80`），不是任何暂存的 TCIM 目标清单。

### 阶段 1：按实际图前沿逐算子合入

每个 PR 只覆盖一个明确的 `op × dtype × shape × mode`，包含：

1. 对应 HMM artifact 及最薄 runtime wrapper；
2. 固定输入、CPU/host reference、误差阈值；
3. 支持 shape 的成功用例和不支持 shape 的明确拒绝用例；
4. 设备输入到设备输出的链式用例，避免“算子正确但只能 host round-trip”；
5. 异步 enqueue 与最终 sync 的错误/生命周期用例。

实现顺序按 `ds4.c` 的真实依赖前沿推进：先让 embedding/HC/norm/dense 等生产出第一个可验证中间张量，再闭合 Q/KV、RoPE/KV、attention，然后 FFN/MoE 和 output head（`84cc882:ds4.c:26426-26479`）。这是依赖顺序，不是要求每个大类只能有一个 PR。

现有仓库已有可复用的测试形态：F16 matvec 测试构造确定性权重/输入，通过通用 `ds4_gpu_*` API执行，再与 host reference 比较（`84cc882:tests/ds4_test.c:299-375`）；当前 kernel group 也按具体 primitive 聚合测试（`:4545-4569`）。TCIM 应抽取/参数化这些真实 contract fixture，而不是另造一套与执行图脱节的测试命名体系。

### 阶段 2：闭合通用 ABI

将已验证算子接入 `ds4_gpu.h` adapter：

- required primitive 必须由 TCIM 原生实现，或由已正确的基础 primitive 组合实现；
- optional fused hook，以及经过 engine capability 校验后保证不可达的 multi-device/其他模型 hook，可以集中放入 unavailable 层并返回 0，仿照 ROCm（`ds4_rocm_unavailable.cu:1-39`）；
- 做链接面完整性检查，但不把“所有符号能链接”误当作图可运行；
- 每个组合 wrapper 都要有与拆分 primitive 相同的数值/顺序测试，仿照 ROCm compat（`ds4_rocm_compat.cu:217-259`）。

### 阶段 3：单层纵切

把现有 layer-0 CPU/GPU 对照测试参数化到 TCIM。必须从 embedding 走完一层 attention、FFN/MoE、HC 和 output head，并逐 stage 对比，不只比较最终 token（`84cc882:ds4.c:26353-26552`）。这是第一次证明“required primitive 集合真的闭合”。

### 阶段 4：最小端到端但仍受控

先定义并验证一个明确的最小 feature matrix，例如单设备、resident weights、一个目标模型/量化、decode-style prefill + 单 token decode + logits readback。仓库已经有逐 token prefill 路径，它循环调用真实的 token evaluator（`84cc882:ds4.c:30970-30988`），入口也已有选择 seam（`:35198-35221`）；TCIM 可以新增显式 capability 来强制走这条 correctness-first 路径，先复用 decode 的 required closure，暂缓 multi-row/batched prefill。不能直接借用当前 SSD/短 prompt 条件（`:30897-30908`），因为那不是 TCIM capability。

这一阶段至少覆盖：完整 prompt 的逐 token prefill、首 token logits、多 token decode、KV 复用/重置，以及较长 suffix。代价是长 prompt 很慢，但功能闭包显著更小且语义明确。未支持的 batched prefill、SSD streaming、multi-device、server batching、额外模型必须在 engine validation 阶段明确拒绝，不能进入图深处才失败；当前 session 的实际分支见 `84cc882:ds4.c:60500-60628`。

### 阶段 5：公开激活 backend

只有阶段 4 的 feature matrix 全部通过后，才增加公共 enum/CLI 选择；bench 和 server 分开接入并各自回归。这样每个此前的算子 PR 都可独立审查和回退，而“支持 TCIM backend”仍保持可验证的产品含义。

### 阶段 6：逐项扩展与融合

先把 decode-style prefill 替换或补充为 batched/layer-major prefill，按独立提交加入 multi-row shape；再扩 SSD expert streaming、server batching、多设备和额外模型。融合只作为 optional acceleration hook 增加，保持 portable primitive 路径始终正确；这正是 `d924de3` 后三个后端共同采用的可回退结构。

## 如果要求每个中间提交都能运行 `--tcim`

技术上可以先做 hybrid fallback，但这不是零成本方案。必须新增按 `op × dtype × shape × mode` 查询的 capability 层；未支持的 required primitive 需要 device→host 同步、调用 CPU reference、再 host→device，并维护 KV/alias/lifetime。它与“中间状态全程 device-resident”的现有设计相冲突（`84cc882:ds4_gpu.h:15-18`），也会在每个 fallback 边界产生同步和拷贝。

因此建议默认采用私有算子 harness 保证每次合入可运行、可比较；只有产品明确要求“半成品 backend 也要走 CLI”时，才把 hybrid fallback 作为独立基础项目评审。无论哪种方式，都不能把 required primitive 的 return 0 当作 fallback。

## 验证局限

- 本调研是源码和 Git 历史审计，没有在 TCIM 硬件上编译 HMM、跑数值误差或测量 dispatch/sync 成本；“小 module 可加载/可共享资源”有源码证据，但“每算子 module 的生产性能可接受”尚未被证明。
- 本调研没有采用当前暂存 Makefile 中的 TCIM 算子/测试清单，也没有据其名称推断覆盖面或优先级。
- layer-0 对照覆盖了关键纵切，但不能替代 long-context、prefill shape、压缩 KV 边界、MoE dtype、SSD 和 server 并发回归；公开激活前仍需按最小 feature matrix 补齐这些路径。

## 最终建议

批准“按算子增量实现 TCIM”，但把交付语义写成两条独立轨道：

- **开发轨道：** 每个 PR 是一个 `op × dtype × shape × mode` 的 HMM + wrapper + reference test，持续合入主干；
- **产品轨道：** 只有 required ABI 闭合、layer-0 纵切通过、最小 prefill/decode feature matrix 通过后，才公开激活 TCIM backend。

这既避免当前“一次完成整个 backend”的高风险大提交，也不把链接成功或单算子 smoke test误报成完整 backend 支持。
