# 领域文档

本仓库采用 **single-context** 布局。本文件规定工程技能在探索代码库时应如何使用领域文档。

## 开始探索前读取

- 仓库根目录的 **`CONTEXT.md`**。
- **`docs/adr/`** 中与即将处理区域相关的 ADR。

如果这些文件或目录尚不存在，**静默继续**：不要把缺失视为问题，也不要预先建议创建。`/domain-modeling` 技能（可由 `/grill-with-docs` 和 `/improve-codebase-architecture` 间接调用）会在术语或决策真正明确时按需创建它们。

## 文件结构

本仓库使用 single-context 结构：

```text
/
├── CONTEXT.md
├── docs/
│   └── adr/
│       ├── 0001-example-decision.md
│       └── 0002-another-decision.md
├── ds4.c
└── ...
```

`CONTEXT.md` 保存整个仓库共享的领域术语；`docs/adr/` 保存架构决策。

## 使用术语表中的词汇

当输出中命名领域概念时（例如 issue 标题、重构提案、假设或测试名称），使用 `CONTEXT.md` 定义的术语。不要改用术语表明确排除的同义词。

如果需要的概念尚未出现在术语表中，这是一个信号：要么正在发明项目并未使用的语言（应重新考虑），要么确实存在需要交给 `/domain-modeling` 补充的缺口。

## 标明与 ADR 的冲突

如果输出与现有 ADR 冲突，必须明确指出，而不是静默覆盖：

> _与 ADR-0007（示例决策）冲突，但值得重新讨论，因为……_
