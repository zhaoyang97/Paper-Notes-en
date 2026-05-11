---
title: >-
  [Paper Note] On Evaluating LLM Alignment by Evaluating LLMs as Judges
description: >-
  [NeurIPS 2025][LLM Evaluation][LLM alignment] This paper systematically investigates the consistency between LLMs' generation capability and evaluation capability (GE-consistency)…
tags:
  - "NeurIPS 2025"
  - "LLM Evaluation"
  - "LLM alignment"
  - "LLM-as-Judge"
  - "evaluation benchmark"
  - "generation-evaluation consistency"
  - "preference oracle"
date: 2026-05-08
content_hash: 172557375e3c076b
---

# On Evaluating LLM Alignment by Evaluating LLMs as Judges

**Conference**: NeurIPS 2025
**arXiv**: [2511.20604](https://arxiv.org/abs/2511.20604)
**Code**: [yale-nlp/AlignEval](https://github.com/yale-nlp/AlignEval)
**Area**: LLM Evaluation
**Keywords**: LLM alignment, LLM-as-Judge, evaluation benchmark, generation-evaluation consistency, preference oracle

## TL;DR

This paper systematically investigates the consistency between LLMs' generation capability and evaluation capability (GE-consistency), finding a strong correlation between the two rankings under a strong preference oracle (Spearman $\rho = 0.96$). Based on this finding, the authors propose the AlignEval benchmark, which measures LLM alignment by assessing LLMs' ability as judges—without directly invoking LLM-as-Judge to evaluate model outputs—achieving performance comparable to or better than AlpacaEval and Arena-Hard.

## Background & Motivation

LLM alignment evaluation—measuring whether models conform to human preferences, instructions, and values—is a core task in modern NLP. Existing evaluation paradigms face the following challenges:

**Human evaluation is costly**: ChatBot Arena, while the gold standard, relies on crowdsourced annotations that are expensive, slow, and difficult to scale.

**LLM-as-Judge is not cheap either**: Automated benchmarks such as AlpacaEval and Arena-Hard depend on GPT-4 as a judge, incurring tens of dollars in API calls per newly evaluated model, with repeated calls required each time.

**The relationship between generation and evaluation capability is underexplored**: Prior work (Generative AI Paradox, GV-consistency) has studied generation–verification inconsistency within individual LLMs, but whether the generation ranking and evaluation ranking across multiple LLMs are consistent (GE-consistency) has not been systematically explored.

**Demand for evaluation efficiency**: If GE-consistency holds, it becomes possible to construct a benchmark that is annotated once and reused many times, substantially reducing evaluation cost.

Core insight: If an LLM is better at judging whether a response aligns with human preferences, its own generated responses are also more likely to be aligned—implying that an LLM's "judging capability" can serve as an indirect proxy for its "generation quality."

## Method

### Overall Architecture

The paper proceeds in two steps: (1) systematically measuring the existence and conditions of GE-consistency; and (2) constructing the AlignEval benchmark based on this finding.

**Formal definition of GE-consistency**: Given a set of LLMs $\mathcal{M} = \{M_1, \dots, M_N\}$, a preference oracle $J$, and an instruction set $\mathcal{I}$:

- Generation ranking $R^{(g)}$: obtained by having $J$ evaluate the response quality of each LLM on $\mathcal{I}$.
- Evaluation ranking $R^{(e)}$: obtained by measuring the agreement between each LLM acting as a judge and $J$.

$$c(\mathcal{M}; J, \mathcal{I}) = \mathcal{C}(R^{(g)}, R^{(e)})$$

where $\mathcal{C}$ denotes the Spearman rank correlation coefficient.

### Key Designs

**Experimental setup for GE-consistency measurement**:

- **Instruction sets**: AlpacaEval (805 instances) and Arena-Hard (500 instances).
- **Preference oracle**: GPT-4o (gpt-4o-2024-08-06).
- **Evaluated LLMs**: 15 instruction-tuned models spanning various scales and model families.
- **Generation ranking**: Each LLM generates responses; GPT-4o performs pairwise comparisons against a GPT-4 baseline to compute win rates.
- **Evaluation ranking**: Each LLM acts as a judge on the same pairwise comparison task; agreement with GPT-4o is measured using Cohen's Kappa.

**Consistency Filtering**: A critical denoising step. For each output pair $(y_1, y_2)$, GPT-4o performs two evaluations with swapped order. Instances where the two judgments disagree are discarded: 58.3% of AlpacaEval instances and 50.7% of Arena-Hard instances are filtered out. This filtering improves GE-consistency from 0.793 to 0.971 on Arena-Hard.

**Effect of oracle strength**: When a weaker LLM (e.g., llama-3-8b) is used as the oracle, GE-consistency drops substantially, demonstrating that a strong oracle is a necessary condition for high GE-consistency.

### AlignEval Benchmark Construction

Using the Arena-Hard instruction set with GPT-4o as the preference oracle, the authors construct a benchmark containing 2,671 evaluation instances. Each instance consists of an instruction, two outputs, and the oracle's preference label.

Two versions are provided:
- **AlignEval-gpt**: annotated using GPT-4o.
- **AlignEval-claude**: annotated using Claude-3.7-Sonnet.

Core advantage: once constructed, evaluating a new model requires no further LLM judge calls, incurring zero API cost.

### Loss & Training / Evaluation Combination

**AlignEval+**: Combines AlignEval with IFEval—AlignEval evaluates "understanding what constitutes a good response" (analogous to planning), while IFEval evaluates "precisely following instructions" (analogous to execution). The two are complementary, and the final ranking averages the scores from both benchmarks.

## Key Experimental Results

### Main Results: GE-Consistency Measurement

| Condition | AlpacaEval | Arena-Hard |
|-----------|-----------|------------|
| Without filtering | 0.743 | 0.793 |
| With consistency filtering | 0.839 | **0.971** |

GE-consistency on Arena-Hard is substantially higher than on AlpacaEval, likely because Arena-Hard contains more technical and challenging instructions, making evaluation more objective and stable.

### Main Results: Spearman Correlation with ChatBot Arena

| Benchmark | Standalone | Combined with IFEval |
|-----------|-----------|----------------------|
| IFEval-Loose | 0.919 | 0.919 |
| Arena-Hard | 0.905 | 0.946 |
| Arena-Hard-SC | 0.882 | 0.936 |
| AlpacaEval-LC | 0.746 | 0.925 |
| GPT4o-Judge | 0.911 | 0.958 |
| MixEval | 0.816 | 0.900 |
| HelpSteer3 | 0.813 | 0.904 |
| **AlignEval-gpt** | **0.856** | **0.946** |
| **AlignEval-claude** | **0.885** | **0.946** |

### Ablation Study

| Ablation | Result |
|----------|--------|
| No consistency filtering | Arena-Hard GE-consistency drops from 0.971 to 0.793 |
| Weak oracle (llama-3-8b) | GE-consistency ≈ 0.3–0.5 |
| Medium oracle (llama-3-70b) | Arena-Hard GE-consistency ≈ 0.9 |
| WildBench instruction set | GE-consistency = 0.938 |

### Key Findings

1. **GE-consistency is broadly observed**: High correlations (0.84–0.97) are found across Arena-Hard, AlpacaEval, and WildBench, indicating a general regularity rather than a dataset-specific artifact.
2. **Consistency filtering is critical**: Removing inconsistent instances improves correlation by 15–18 percentage points, eliminating noisy cases where the oracle is uncertain or outputs are too similar.
3. **AlignEval achieves top-tier performance without an LLM judge**: AlignEval-claude alone reaches 0.885; combined with IFEval it reaches 0.946, matching Arena-Hard (0.946) which requires LLM judge calls.
4. **Self-preference bias exists but is manageable**: AlignEval-gpt favors GPT-4o-family models and AlignEval-claude favors Claude-family models, yet both consistently rank Gemini-2.0-Flash highly.
5. **A strong oracle is a necessary condition**: GE-consistency is strongly dependent on oracle quality and degrades substantially when weak models serve as oracles.

## Highlights & Insights

- **Paradigm innovation**: The paper proposes measuring alignment quality indirectly by evaluating LLMs' judging capability, establishing a low-cost, reusable evaluation paradigm.
- **Theoretical contribution**: This is the first work to systematically validate GE-consistency at the level of cross-model rankings, distinguishing itself from prior single-model GV-consistency research.
- **High practical value**: AlignEval is constructed once and evaluated indefinitely; the API cost per newly evaluated model is $0, compared to approximately $20 for Arena-Hard.
- **A deep distinction between GE-consistency and GV-consistency**: Even when individual LLMs exhibit inconsistency between generation and verification, the relative rankings across multiple LLMs can remain highly consistent—better evaluators tend to also be better generators.
- **Complementarity with IFEval**: The combination of "planning" (understanding what constitutes a good response) and "execution" (precisely following instructions) offers a template for constructing comprehensive evaluation systems.

## Limitations & Future Work

1. **Vulnerability to adversarial attacks**: Fine-tuning an LLM to become a better judge can artificially inflate AlignEval scores without genuinely improving alignment.
2. **Oracle dependency**: The validity of the entire framework depends on the strength and fairness of the oracle; any preference bias in the oracle propagates into the benchmark.
3. **ChatBot Arena is not a perfect gold standard**: Arena rankings are used as validation targets, yet Arena itself suffers from limited transparency in data collection and potential biases.
4. **Self-preference bias is not fully resolved**: Different oracle choices produce AlignEval versions that favor models from the same family.
5. **Limited coverage**: Instances are derived from 500 Arena-Hard instructions, which may not cover all dimensions of alignment.
6. **Only pairwise comparison is assessed**: Pointwise scoring and more fine-grained evaluation formats are not explored.

## Related Work & Insights

- **AlpacaEval / Arena-Hard**: The dominant LLM-as-Judge benchmarks and the primary baselines for this work.
- **MixEval**: Reduces reliance on LLM judges by matching user queries to existing benchmarks; this paper demonstrates that evaluating "evaluation capability" is more effective.
- **RewardBench**: A benchmark for evaluating reward models, conceptually related to AlignEval's assessment of LLMs-as-judges.
- **Generative AI Paradox** (West et al.): Finds that LLMs sometimes generate better than they evaluate; this paper provides a complementary ranking-level perspective through GE-consistency.
- **Inspiration**: Evaluation capability is itself an important dimension of LLM ability; future work should routinely include "whether a model can accurately judge output quality" as a standard evaluation metric.

## Rating

- Novelty: ⭐⭐⭐⭐ — Studying generation–evaluation consistency at the ranking level is a novel perspective; the zero-cost evaluation paradigm of AlignEval is genuinely innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 23 LLMs, multiple instruction sets, multiple oracles, and extensive ablations; the experimental design is rigorous and comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ — Concepts are clearly defined, arguments are logically tight, and figures are informative; the paper is highly readable.
- Value: ⭐⭐⭐⭐ — Provides both a practical tool and theoretical insights for the LLM evaluation community, though vulnerability to adversarial attacks limits its applicability in certain deployment scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] PARROT: A Benchmark for Evaluating LLMs in Cross-System SQL Translation](parrot_a_benchmark_for_evaluating_llms_in_cross-system_sql_translation.md)
- [\[NeurIPS 2025\] Leveraging Robust Optimization for LLM Alignment under Distribution Shifts](leveraging_robust_optimization_for_llm_alignment_under_distribution_shifts.md)
- [\[NeurIPS 2025\] LTD-Bench: Evaluating Large Language Models by Letting Them Draw](ltd-bench_evaluating_large_language_models_by_letting_them_draw.md)
- [\[NeurIPS 2025\] Beyond the Surface: Enhancing LLM-as-a-Judge Alignment with Human via Internal Representations](beyond_the_surface_enhancing_llm-as-a-judge_alignment_with_human_via_internal_re.md)
- [\[AAAI 2026\] Where Norms and References Collide: Evaluating LLMs on Normative Reasoning](../../AAAI2026/llm_evaluation/where_norms_and_references_collide_evaluating_llms_on_normative_reasoning.md)

</div>

<!-- RELATED:END -->
