# XH2 纯 C 算子 runtime 可行性

更新：2026-08-31

## 结论

**Go：纯 C + `libhal_xh2a` 可作为 TCIM/XH2 算子后端的执行底座。** hq50 已验证原生参数 ABI、内嵌 kernel、DDR 读写、BO 生命周期和批量执行；这不代表 DS4 算子覆盖、数值和性能已经完成。

- D3 完成态故障信号门禁仍未关闭：需要厂商认可的 `sync rc==0 && result!=0` 实机结果。当前 V1.4.0 驱动路径将 result 固定为零，fake HAL 或 timeout 不能替代该门禁。
- 后续仍需完成目标模型所需算子及全模型集成，验证数值、GGUF/SSD 连续 decode、长稳态、故障恢复、性能，以及现有后端回归。

阶段验收记录见 [Issue #6](https://github.com/fixedpointworks/ds4/issues/6)；环境验证脚本与结果只保留在 ignored 的 `misc/`。

## 方案

- TCIM 独立构建，复用 DS4 的 GPU 接口和 CUDA backend identity，但不复用 CUDA runtime、kernel 或 capability，见 [ADR-0004](../adr/0004-tcim-reuses-cuda-backend-identity.md)。
- DS4 宿主代码管理模型、权重、KV、GGUF/SSD cache 和算子调度；`xh2rt` 只负责设备资源与执行，不承担模型策略。
- host 保持纯 C，直接调用 HAL；设备 kernel 使用 HDPL/Primitive 构建后内嵌，运行时不依赖 HDPL 编译器或外部 kernel 文件。
- 采用 SDK 原生参数 ABI，由 host wrapper 与设备入口共同维护参数契约，不引入自定义描述块、模块容器或签名匹配。调用方负责算子参数正确性；runtime 保留资源安全和 HAL 错误处理，设备故障后不得复用无效结果或静默 fallback。
- 官方 Dots TCIM 路径仅作为 ABI/数值参考，entry-only relink 不作为生产路线或验收前提。
