# XH2 纯 C 算子 runtime 可行性

更新：2026-08-31

## 结论

**纯 C + `libhal_xh2a` 可作为 TCIM/XH2 算子后端的执行底座；当前采用 SDK 原生参数 ABI 与内嵌 kernel。** 这不代表 Issue #6 或 DS4 算子覆盖、数值和性能验收已经完成。

- 从单 DDR descriptor pointer / `.xh2k` 改为原生参数 ABI 后，D0–D3 必须重新验证。旧 evidence 保留供追溯，不能作为当前实现的通过依据；native ABI 与本轮 review 修复的结果须另行记录。
- D3 完成态故障信号门禁仍未关闭：需要厂商认可的 `sync rc==0 && result!=0` 实机结果。当前 V1.4.0 驱动路径将 result 固定为零，fake HAL 或 timeout 不能替代该门禁。
- D4 基础 DS4 adapter 未开始；D3 得到 GO 前不扩展到该阶段。
- 后续仍需完成目标模型所需算子及全模型集成，验证数值、GGUF/SSD 连续 decode、长稳态、故障恢复、性能，以及现有后端回归。

契约与阶段验收记录见 [Issue #6](https://github.com/fixedpointworks/ds4/issues/6)；环境验证脚本与结果保留在 ignored 的 `misc/`，正式 evidence 记录完整命令、版本、hash 与最终输出。

## 方案

- TCIM 独立构建，复用 DS4 的 GPU 接口和 CUDA backend identity，但不复用 CUDA runtime、kernel 或 capability，见 [ADR-0004](../adr/0004-tcim-reuses-cuda-backend-identity.md)。
- DS4 宿主代码管理模型、权重、KV、GGUF/SSD cache 和算子调度；`xh2rt` 只负责设备资源与执行，不承担模型策略。
- host 保持纯 C，直接调用 HAL；设备 kernel 使用 HDPL/Primitive 构建，由 `xxd` 生成完整 C byte array 后内嵌，payload 必须为 1..65535 bytes。每个 context 在 open 时分配并上传 code BO，后续 launch 复用；运行时不依赖 HDPL 编译器或外部 kernel 文件。
- host typed wrapper 与设备入口共同维护 SDK 原生参数契约，以编译、envelope known-vector 和实机测试约束漂移。SPM envelope 为 8-byte zero header、8-byte argc 与各 8-byte little-endian 参数；fill 为 3 参数/40 bytes，add 为 4 参数/48 bytes。KLD 固定 2 cores × 4 tiles、SPM、`ilm_mode=1`，不引入自定义描述块、模块容器、签名匹配或动态 module cache。
- wrapper 在提交 HAL 前必须校验 buffer handle/lifetime、`count * sizeof(float)` overflow、全部 view 的访问范围和 4-byte alignment，并拒绝 add 超出 `UINT32_MAX` 的 count。buffer 地址只能由 registry 解析；非法参数不能依赖 caller 自查或 device early-return 后静默成功。
- command group 是阻塞 batch；SPM envelope 在 launch 调用内由 driver 复制，code/read/write BO 则 pin 到 sync 确认完成。设备故障后 poison context，禁止复用无效结果或静默 fallback；未证明 quiescent 的 BO 与其 allocator 必须 quarantine，不能因为 destroy 成功或进程退出就假定安全释放。
- 官方 Dots TCIM 路径仅作为 ABI/数值参考，entry-only relink 不作为生产路线或验收前提。
