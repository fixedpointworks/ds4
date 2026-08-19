# DS4 Inference

DS4 Inference 是面向 DeepSeek V4 Flash 模型的推理领域。本术语表统一模型、权重供给、增量推理状态与执行后端的称谓。

## Language

### 模型与权重

**GGUF 模型（GGUF model）**:
供 DS4 推理使用、以 GGUF 格式封装模型元数据和量化权重的模型。
_Avoid_: TCIM 模型

**SSD 权重路径（SSD weight path）**:
DS4 中以 SSD 为权重来源的权重供给路径。
_Avoid_: TCIM staging path

**SSD expert cache**:
SSD 权重路径中的 routed-expert 权重缓存。
_Avoid_: TCIM expert cache

### 增量推理

**逐 token prefill（decode-style prefill）**:
由单步 decode 序列构成的 prefill 形态。

**单步 decode（single-step decode）**:
以一个新 token 为输入粒度的 decode 步骤。

**连续多 token decode（multi-token decode sequence）**:
由多个单步 decode 顺序组成的生成序列。

**实时 KV 状态（live KV state）**:
增量推理中与已处理 token 对应的当前 KV cache 状态。

### 执行后端

**TCIM backend（TCIM 后端）**:
DS4 中通过 TCIM 执行模型推理的后端。
