# TCIM backend 添加路线图

范围：只依据官方仓库历史、`ds4.c` 实际执行图和 `ds4_gpu.h` 契约。分析基线为 `84cc882352757baf628a1776badf7cc54d584e28`。

## 三个平台留下的模式

| 平台 | 首次公开落地 | 后续演进模式 | 对 TCIM 的启示 |
|---|---|---|---|
| Metal | 根提交 [`d997b56`](https://github.com/antirez/ds4/commit/d997b56c151184bcff469dd8302ed97f23481024) 已包含完整 backend（`ds4_metal.m` 14,491 行；整个初始提交 +59,189 行），没有更早的公开逐算子历史。 | 完整正确性基线后，再做小范围修复/优化，如 debug 校验 [`f9e8715`](https://github.com/antirez/ds4/commit/f9e8715421bc31a5c9250bf5c338b91f89a3b344)、short-prompt prefill [`5cbee23`](https://github.com/antirez/ds4/commit/5cbee233e79343fbb05232daf5dc091cdca9d0c3) 和 decode 融合 [`71d8c2a`](https://github.com/antirez/ds4/commit/71d8c2a072c44ca714f8ece20f14f481472d7ff3)；不稳定快路径会被关闭/回滚。 | 先守住 correctness floor，融合永远后置且可回退。 |
| CUDA | [`48beef8`](https://github.com/antirez/ds4/commit/48beef81a017a3785fd449a6f3cb34eb849b0e1f) 一次加入完整 9,666 行 `ds4_cuda.cu`；没有关联的公开实现 PR。 | 完整 port 后才抽出通用 GPU ABI [`0ac5df3`](https://github.com/antirez/ds4/commit/0ac5df3e65fbc6d8ad4ce21bf66bd1d790a718a6)，随后按 long-context [`320b779`](https://github.com/antirez/ds4/commit/320b7793bdfa3928750050df58406aac6ac8015d)、prefill/indexer [`08fecd9`](https://github.com/antirez/ds4/commit/08fecd939e7733ecb934df76db25d9f18def7e38) 等窄问题迭代。 | TCIM 可复用成熟 ABI，但仍须证明 required surface 已闭合。 |
| ROCm | 先在关闭的 [PR #79](https://github.com/antirez/ds4/pull/79)、[#133](https://github.com/antirez/ds4/pull/133)、[#180](https://github.com/antirez/ds4/pull/180)、[#290](https://github.com/antirez/ds4/pull/290)、[#311](https://github.com/antirez/ds4/pull/311) 中公开孵化，进入 `main` 时仍整理为完整大提交 [`586abbf`](https://github.com/antirez/ds4/commit/586abbfb625434a95372303319a66c90cf94ba12)（+17,612 行）。 | 落地后立即把 fused op 改成 optional hook [`d924de3`](https://github.com/antirez/ds4/commit/d924de3ab0c3d316c172d7fd550d6e698faa7905)，以后再按 MoE、indexer、HC、模型兼容逐项补齐；compat/unavailable 层见 [`ef8d923`](https://github.com/antirez/ds4/commit/ef8d923ad7290ef0b93dd26a364bdb048ab4428a)。 | 最可取的是“孵化分支 + 基础 primitive + 兼容组合 + optional 加速”，但应避免巨型 PR、反复 force-push 和最终丢失细粒度历史。 |

**历史结论：** 三个平台都不是把“半成品 backend”公开给用户后再逐算子补齐。TCIM 可以逐算子**开发、测试、合入孵化分支**；只有 required 图闭包通过后，才应公开为一个可选择的完整 backend。

## 按 PR / 验收门推进

| 阶段 | PR 边界 | 合入验收门 |
|---|---|---|
| 0. 定义最小产品闭包 | 一份 feature matrix：固定首个设备、模型/量化、单设备、prefill/decode 方式；从真实执行图列出 required、可组合、optional、明确不支持四类接口。 | 每个可达调用都有归类；不支持组合能在初始化时明确拒绝，而不是运行到图中失败。依赖顺序以 [`ds4.c` layer-0 图](https://github.com/antirez/ds4/blob/84cc882352757baf628a1776badf7cc54d584e28/ds4.c#L26426-L26552) 为准。 |
| 1. Runtime / tensor 基座 | 一个独立 PR：runtime 生命周期、共享 context/stream、tensor alloc/view/copy、enqueue/sync、错误传播；**不加 enum、CLI、bench 或 server 入口**。 | 重复 init/run/free、alias/view、异步错误和 device→device 链接通过；满足 [`ds4_gpu.h` tensor/command 契约](https://github.com/antirez/ds4/blob/84cc882352757baf628a1776badf7cc54d584e28/ds4_gpu.h#L11-L80)。 |
| 2. 逐算子 PR 栈 | 每个 PR 只交付一个 `op × dtype × shape × mode`：HMM artifact、薄 wrapper、CPU/host golden、支持与拒绝用例。按 `embedding → HC/norm/dense → Q/KV/RoPE/attention → FFN/MoE → output head` 的依赖前沿推进。 | 固定输入数值误差达标；不支持 shape 明确失败；至少一个 device-input→device-output 链式测试；bring-up 可同步，生产路径不得把每算子 host round-trip 当成完成。 |
| 3. ABI 闭合 | 一个集成 PR：把已验证 primitive 接入 `ds4_gpu.h`；能组合的放 compat 层，只有引擎已有 portable fallback 的 fused hook 才可放 unavailable/返回 0。 | 符号完整且所有 required primitive 都有真实实现或已验证组合；“链接成功”不算通过。optional 语义以 [`d924de3`](https://github.com/antirez/ds4/commit/d924de3ab0c3d316c172d7fd550d6e698faa7905) 为准。 |
| 4. 单层纵切 | 一个 correctness PR：参数化现有 layer-0 CPU/GPU 对照，让 TCIM 从 embedding 连续跑过 attention、FFN/MoE、HC 到 output head。 | 每个中间 stage 与 CPU reference 对齐；tensor/KV 全程 device-resident；没有隐式 host fallback。 |
| 5. 受控端到端 | 一个 experimental PR：只开放阶段 0 的 feature matrix，先用 correctness-first prefill（必要时逐 token）+ 多 token decode。仍不成为公共 `--tcim`。 | prompt、首 token logits、多 token decode、KV 复用/重置、较长 suffix 全通过；矩阵外能力在入口拒绝。正常 session 分支以 [`ds4.c`](https://github.com/antirez/ds4/blob/84cc882352757baf628a1776badf7cc54d584e28/ds4.c#L60500-L60628) 为准。 |
| 6. 公开激活 | 小 PR 增加 backend enum 和 CLI；bench、server 各自单独 PR，不能与 kernel 大包混合。 | 最小 feature matrix 端到端回归通过；Metal/CUDA/ROCm、SSD 路径无回归；帮助文本准确说明能力边界。此门之后才可称“支持 TCIM backend”。 |
| 7. 扩能力与性能 | batched/layer-major prefill、更多量化/模型、SSD、server batching、多设备分别提交；融合保持 optional、每项可独立回滚。 | 每个 PR 先过 reference/correctness，再给端到端性能与显存数据；不能用无法解释的 logits、attention 或 KV 漂移换速度。 |

提交策略：保留长期可访问的 TCIM incubation branch/stacked PR；主干能安全接收的 runtime 与窄 primitive 可以早合入，但不暴露公共选择入口。公开 landing 尽量保留细粒度提交，或至少链接完整 PR stack，避免重演三个 backend 首发历史中“代码可见、实现过程不可追溯”的问题。
