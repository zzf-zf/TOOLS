# Evidence 模块

这是一个与模型、提示词、RAG 框架和强化学习框架解耦的证据解析模块。

它只负责两件事：

1. 从**模型生成的响应**中解析 `<original_evidence>...</original_evidence>`；
2. 可选地检查证据是否是检索文档中的原文片段。

它不负责调用检索器、调用评估模型或计算最终奖励。这些职责应当分开，方便在不同项目中替换。

## 快速使用

```python
from evidence import EvidenceExtractor

response = """
<original_evidence>
Albert Einstein was born in Ulm, Germany.
</original_evidence>
<answer>Ulm</answer>
"""

extractor = EvidenceExtractor()
result = extractor.extract(response)

if not result.valid:
    print(result.error)
else:
    print(result.last_text)
```

注意：只向 `extract()` 传入模型生成的 response，不要传入 system prompt。这样不会出现 system prompt 中的示例标签被误判为模型证据的问题。

## 配置不同标签

新项目如果规定：

```text
<evidence>...</evidence>
```

使用：

```python
extractor = EvidenceExtractor(tag="evidence")
```

允许模型输出多个证据块：

```python
extractor = EvidenceExtractor(
    tag="evidence",
    allow_multiple=True,
)

result = extractor.extract(response)
for evidence in result.texts:
    print(evidence)
```

如果可以控制新项目的输出协议，结构复杂时建议使用 JSON，而不是继续扩展正则表达式。本模块适合简单、非嵌套的 XML 风格标签。

## 检查证据是否来自检索文档

```python
grounding = extractor.check_grounding(
    evidence=result.texts,
    sources=[
        "Document 1 text...",
        "Albert Einstein was born in Ulm, Germany.",
    ],
)

print(grounding.grounded)       # True
print(grounding.source_indices) # (1,)
```

这是严格的“原文子串”检查，只进行大小写和空白归一化。它不会接受语义相同但经过改写的证据。

如果项目要求“必须复制原文”，建议将 `grounding.grounded` 作为获得 evidence reward 的硬条件。这能阻止模型编造一段包含正确答案的伪证据。

## 推荐的数据流

```text
模型响应
  → EvidenceExtractor.extract()
  → EvidenceExtractor.check_grounding()
  → EvidenceEvaluator.evaluate()
  → RewardComposer.compose()
```

解析、真实性检查、充分性评估和奖励组合应当是四个独立步骤。

## 新项目需要配置的 Evidence evaluator

Evaluator 判断“仅根据这些证据，能否回答原问题”。推荐定义稳定接口：

```python
from typing import Protocol, Sequence


class EvidenceEvaluator(Protocol):
    def evaluate(
        self,
        *,
        question: str,
        evidence: Sequence[str],
        reference_answers: Sequence[str],
    ) -> float:
        """Return a score in [0, 1]."""
        ...
```

可选实现：

- **Answer reconstruction**：把 question 和 evidence 交给固定 LLM，让它生成短答案，再与标准答案计算 EM/F1。适合有标准答案的 QA。
- **LLM judge**：让评估模型判断证据是否蕴含答案。适合开放式任务，但必须固定评分 rubric，并测试 judge 偏差。
- **NLI evaluator**：判断 evidence 是否蕴含 claim。适合事实核验。
- **规则评分**：实体、数值或引用位置可以确定时，优先使用确定性规则。

配置 evaluator 时需要明确：

1. 输入 evidence 是原文还是允许改写；
2. 是否有 reference answer；
3. 评分是 EM、token F1、蕴含概率还是 judge 分数；
4. 失败、超时和空输出返回多少分；
5. 是否支持批量评估和缓存——RL 训练中逐条调用 LLM 会很慢；
6. evaluator 模型必须与被训练模型分离，并在训练期间冻结。

Evaluator 不应负责解析标签，也不应从 ChatML 完整轨迹中猜测 question。

## 最终奖励组合建议

推荐把每个子奖励限制在 `[0, 1]`，再显式配置权重：

```python
total_reward = (
    answer_weight * answer_score
    + evidence_weight * evidence_score
    + format_weight * format_score
)
```

一个保守示例：

```python
if not extraction.valid:
    evidence_score = 0.0
elif not grounding.grounded:
    evidence_score = 0.0
else:
    evidence_score = evaluator.evaluate(
        question=question,
        evidence=extraction.texts,
        reference_answers=reference_answers,
    )

total_reward = (
    1.0 * answer_score
    + 0.5 * evidence_score
    + 0.1 * format_score
)
```

实践提示：

- 先记录各子奖励的独立指标，再调整权重，不要只观察总奖励。
- Evidence reward 不应明显压过 answer reward，否则模型可能复制大量文档而不专注回答。
- 对 evidence 长度设置上限或惩罚，避免模型把所有 observation 全部复制出来。
- 不要把“搜索次数至少为 2”写进通用解析器；是否鼓励多步搜索应由项目奖励策略决定。
- 分开设置 `format_valid`、`grounded` 和 `sufficient`，便于定位模型究竟在哪一步失败。
- 在正式训练前制作几十条人工样例，验证 evaluator 分数是否符合人的判断。

## 测试

在该目录执行：

```bash
python -m unittest -v test_extractor.py
```

模块仅使用 Python 标准库。
