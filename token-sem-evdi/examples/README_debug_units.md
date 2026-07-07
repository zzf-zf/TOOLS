# TriPE unit-level 调试样本

## 目的

这组小样本用于对 TriPE 三层预测误差的 unit-level 输出进行 sanity check：

- `PE_token` 检查原始 `unit_answer` 的 token-level uncertainty。
- `PE_sem` 检查同一上下文下多次采样的语义稳定性。
- `PE_evid` 检查原始 `unit_answer` 与外部证据的一致性。

每条样本都是一个简短的中间回答单元，主要突出一种现象，不代表完整长回答。

## 文件

- `debug_units.jsonl`：8 个待评估 unit，包含统一格式的生成上下文、原始回答和预期行为。
- `debug_evidence_corpus.json`：人工编写的最小调试证据库，所有 passage 的 `metadata.source` 均为 `debug_manual`。
- `README_debug_units.md`：数据用途、case 设计和使用限制说明。

## Case 预期

| Case | 主要现象 | PE_token | PE_sem | PE_evid |
|---|---|---|---|---|
| case-001 | 稳定且正确的地点事实 | 低或中 | 低 | 低 |
| case-002 | 表述稳定但地点错误的高置信幻觉 | 低或中 | 低 | 高 |
| case-003 | 带犹豫措辞的正确事实 | 中或高 | 低 | 低 |
| case-004 | 开放问题引发语义立场摇摆 | 中 | 高 | 不可用或中 |
| case-005 | 较弱可验证的解释性观点 | 中 | 中 | 不可用或中 |
| case-006 | 检索条件下得到支持的架构事实 | 低或中 | 低 | 低 |
| case-007 | 检索条件下与证据不符的架构事实 | 低或中 | 低 | 高 |
| case-008 | 难以原子化验证的抽象观点 | 任意 | 中 | 不可用或中 |

这些模式用于观察三层指标能否区分语言置信度、采样语义分歧和证据支持程度。例如，case-002 与 case-007 预期呈现较低的 token/semantic error 和较高的 evidence error。

## 使用限制

- 这是 sanity check，不是正式 benchmark。
- `PE_evid` 依赖 retriever、AFG 和 AFV 的实际表现。
- 证据库很小，检索遗漏可能使 `PE_evid` 不稳定。
- `expected_pattern` 是调试预期，不是 ground-truth label。
- 该证据库是人工构造的调试材料，不声称来自任何真实官网或正式数据源。
- 后续正式实验需要使用真实数据集和人工标注，并报告检索与验证误差。
