---
title: >-
  [Paper Note] From Human-Level AI Tales to AI Leveling Human Scales
description: >-
  [ICML 2026][AI evaluation] This paper uses LLMs as population extrapolators, calibrating 18 ability dimensions on a "world population accuracy" logarithmic scale $L=-\log_B p_W$. It finds that the true base for Volume /…
tags:
  - "ICML 2026"
  - "AI evaluation"
  - "psychometrics"
  - "ADeLe"
  - "world population calibration"
  - "LLM as annotator"
date: 2026-05-08
content_hash: 4b5efae7a16af8a1
---

# From Human-Level AI Tales to AI Leveling Human Scales

**Conference**: ICML 2026  
**arXiv**: [2602.18911](https://arxiv.org/abs/2602.18911)  
**Code**: None  
**Area**: AI Evaluation / Psychometrics  
**Keywords**: AI evaluation, psychometrics, ADeLe, world population calibration, LLM as annotator

## TL;DR
This paper uses LLMs as population extrapolators, calibrating 18 ability dimensions on a "world population accuracy" logarithmic scale $L=-\log_B p_W$. It finds that the true base for Volume / Attention dimensions is $B \gg 10$, while for Comprehension $B \approx 1$, revealing that current AI-human comparisons are fundamentally misaligned.

## Background & Motivation

**Background**: Mainstream AI evaluation relies on benchmarking—comparing "human-level" performance using average scores on a single benchmark. This approach compresses varying task difficulties, sample populations, and ability dimensions into a single number, leading to contradictory conclusions: LLMs achieve 90% on MMLU but only 50-70% on real-world software engineering tasks; on GPQA Diamond, PhDs score 70% while models reach 88%.

**Limitations of Prior Work**: (1) Benchmarks are not comparable, and "human-level" is entirely dependent on the sampled reference population (often WEIRD: Western / Educated / Industrialized / Rich / Democratic); (2) Existing criterion-referenced frameworks like ADeLe provide dimension-level rubrics but use $B=10$ by convention, not calibration, so cross-dimension comparison is still invalid; (3) Large-scale human measurement is prohibitively expensive and infeasible for new benchmarks.

**Key Challenge**: Referencing "human-level" requires human samples, but available samples are always biased subsets; without calibration, conclusions about "surpassing" or "falling short of" humans are entirely sample-dependent.

**Goal**: (1) Annotate benchmark items with ADeLe's 18-dimensional demand levels; (2) Extrapolate any small-sample human performance to the global population (WWP); (3) Infer the true logarithmic base for each dimension from WWP accuracy; (4) Validate the reliability of the entire extrapolation process.

**Key Insight**: Psychometrics has long used equating/post-stratification to extrapolate from small to large samples; modern LLMs, trained on vast demographic data, can serve as cheap, repeatable population extrapolators.

**Core Idea**: Use LLMs to translate "focal-group success rate + demographic description of that group + target group demographic description" into "target group success rate." Then, for each ability dimension, perform linear regression to obtain the true base $B = 10^m$, establishing a demographically anchored ability ruler.

## Method

### Overall Architecture
A 5-step pipeline: (1) Aggregate item pools (PISA 2009 / TIMSS 2003+2011 G4&G8 / ICAR / UKBioBank / ReliabilityBench); (2) Annotate each item with ADeLe's 18-dimensional demand level $d_{i,c} \in \{0,1,2,3,4,5+\}$; (3) Use LLMs to extrapolate focal group accuracy $p_i^g$ to WWP accuracy $p_i^W$; (4) Convert to logarithmic difficulty $L_i = -\log_B p_i^W$; (5) Validate predictions from sub-group to full-sample (MAE / RMSE / Pearson / Spearman).

### Key Designs

1. **LLM as Population Extrapolator**:

    - **Function**: Translates any small-sample item accuracy to "global population" accuracy, avoiding human annotation bias.
    - **Mechanism**: Prompt contains 6 parts—(a) dataset and test domain introduction; (b) focal group demographic description (e.g., "15-year-old students in OECD countries, PISA 2009"); (c) item stem + options + correct answer; (d) focal group observed accuracy $p_i^g$; (e) reference group (world population) demographic description; (f) request for LLM to output predicted reference group accuracy $\hat p_i^W$ with rationale. The prompt explicitly lists 7 adjustment factors: global age distribution, educational access and quality, post-graduation forgetting, fluid/crystallized ability lifespan curves, specialization and exposure, health and cognitive decline, language factors. Robustness is tested with 27 paraphrased versions.
    - **Design Motivation**: Traditional IRT requires large numbers of participants, which is infeasible for new benchmarks; LLMs' training data implicitly encode demographic statistics, providing a cheap proxy, and rationales allow for auditing.

2. **Dimension-Specific Base Calibration (Optimal Base)**:

    - **Function**: Calibrates the logarithmic difficulty base from the default $B=10$ to the true steepness for each ability dimension.
    - **Mechanism**: Empirical difficulty $L_{\text{emp},i} = -\log_{10}(p_i^W / \sqrt{10})$; applies "primary bottleneck" filtering—only items with $d_{i,c} \ge \max_k d_{i,k}$ are used for regression on dimension $c$, avoiding interference from other bottlenecks. Then, for each level $l \in \{1,..,5\}$, averages to obtain $\bar y_l$, and performs linear regression on $(l, \bar y_l)$; the slope $m$ yields $B = 10^m$. Results fall into three categories: High-base (Volume $B\approx 32$, Attention $B\approx 17$, difficulty increases more steeply than annotated); Standard (Metacognition $B\approx 6.7$, Knowledge $B\approx 5.1$); Invariant (Comprehension / Spatial $B\approx 1$, difficulty increase has almost no effect).
    - **Design Motivation**: The single $B=10$ assumption does not hold across dimensions, leading to misleading cross-dimension interpretations such as "AI surpasses humans in knowledge but lags far behind in reasoning"; dimension-specific bases are key to aligning different rulers to a common unit.

3. **Dominance Filter + Means-based Regression**:

    - **Function**: Extracts pure dimension signals from items with multiple bottlenecks and counters regression bias due to sparse high-difficulty items.
    - **Mechanism**: First applies dominance filter to retain bottleneck items; then averages by level (not raw point regression) to counteract mean shifts caused by the overrepresentation of low-level items. Finally, fits a line to the five mean points, with the slope representing $\log_{10} B$.
    - **Design Motivation**: Raw data is crowded with level 1 items, and direct regression would flatten the slope; averaging by level before regression is a fair-weight compromise.

### Loss & Training
No training involved. LLMs used: GPT-5 Chat, GPT-4.1, Llama-4, DeepSeek-v3.1, GROK-3 (five commercial models), low temperature, no tool use; each item × 27 paraphrases. Validation uses sub-group → full-sample design on ICAR, TIMSS, and UKBioBank.

## Key Experimental Results

### Main Results (LLM Extrapolation Quality)

| Model | ICAR MAE ↓ | ICAR RMSE ↓ | ICAR Pearson ↑ | ICAR Spearman ↑ |
|-------|------------|-------------|----------------|-----------------|
| gpt-5-chat | **0.030** | **0.044** | **0.976** | **0.968** |
| llama-4 | 0.033 | 0.052 | 0.971 | 0.963 |
| gpt-4.1 | 0.040 | 0.058 | 0.958 | 0.944 |
| deepseek-v3.1 | 0.043 | 0.085 | 0.922 | 0.914 |
| grok-3 | 0.043 | 0.068 | 0.939 | 0.920 |

On TIMSS, MAE rises to $0.12$-$0.16$ and Pearson drops to $0.5$-$0.7$, reflecting greater difficulty in extrapolation amid cross-country heterogeneity.

### Ablation Study (Dimension-Specific Base Calibration)

| Dimension Group | Calibrated $B$ | Interpretation |
|-----------------|----------------|---------------|
| Volume | $\approx 32$ | Much steeper than $B=10$; higher levels should be upscaled |
| Attention | $\approx 17$ | Same as above |
| Metacognition | $\approx 6.7$ | Close to $B=10$, well-calibrated |
| Knowledge | $\approx 5.1$ | Same as above |
| Comprehension & Expression | $\approx 1$ | Difficulty barely increases, levels should be downscaled |
| Spatial Reasoning & Navigation | $\approx 1$ | Same as above |

### Key Findings
- The single $B=10$ does not hold across dimensions—Volume and Comprehension differ by about $30\times$ in true base, meaning that "AI leads humans in Knowledge" and "AI still lags far behind in Volume" cannot be compared without calibration.
- LLM extrapolation achieves MAE of just $0.030$ (Pearson 0.976) on structurally homogeneous ICAR, proving that LLMs indeed encode substantial demographic priors; but errors rise sharply on TIMSS, indicating persistent Western bias.
- After calibrating each dimension with its own base, current LLMs show a clear capability profile of "strong in Knowledge, weak in Volume/Attention," providing policymakers with more interpretable comparisons.

## Highlights & Insights
- Redefines the often-misused "AI vs. human" comparison from "benchmark score comparison" to "position on a logarithmic scale of population distribution," representing a philosophical shift in evaluation.
- Using LLMs as population extrapolators is a clever loop of "using AI to calibrate AI-human comparison," and sub-group → full-sample validation shows LLMs have indeed learned demographic adjustment.
- The dimension-specific base calibration results (Volume $\approx 32$, Comprehension $\approx 1$) directly challenge all scalar "AI reaches X% of human level" conclusions from recent years—a striking negative finding.

## Limitations & Future Work
- Only five data sources, all text-only; multimodal and agentic tasks are not covered.
- LLM extrapolator shows large MAE and clear Western/Anglosphere bias on TIMSS; population estimates for non-Western cultures may be systematically biased.
- Assumes dominance filter sufficiently "purifies" dimension signals, but actual items may have multiple bottlenecks, and filtered samples may themselves be valuable.
- Base calibration uses linear regression on five mean points, with weak statistical significance; for some dimensions (e.g., Mind Modeling) even yields negative slopes.

## Related Work & Insights
- **vs ADeLe (Zhou 2025)**: ADeLe provides demand rubrics but $B=10$ is by convention; this work offers empirical calibration.
- **vs METR time-horizon (Kwa 2025)**: Anchors on single-dimension human hours; this work anchors on multidimensional population distributions.
- **vs IRT psychometrics**: Classic IRT requires dense real response data; this work uses LLMs to bypass that step.
- **vs MMLU / GPQA**: Scalar accuracy + single reference population; this work provides a decomposable, comparable profile.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ "LLM as population extrapolator + dimension-specific base calibration" is a rare methodological innovation.
- Experimental Thoroughness: ⭐⭐⭐ Only five data sources, all text-only; large errors on TIMSS, insufficient cross-cultural validation.
- Writing Quality: ⭐⭐⭐⭐ Motivation is compelling, technical exposition is clear.
- Value: ⭐⭐⭐⭐⭐ Paradigm-level reflection for the AI evaluation community; both policymakers and researchers should read.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ChatBench: From Static Benchmarks to Human-AI Evaluation](../../ACL2025/llm_evaluation/chatbench_from_static_benchmarks_to_human-ai_evaluation.md)
- [\[ACL 2025\] CulturalBench: A Robust, Diverse, and Challenging Cultural Benchmark by Human-AI CulturalTeaming](../../ACL2025/llm_evaluation/culturalbench_a_robust_diverse_and_challenging_cultural_benchmark_by_human-ai_cu.md)
- [\[ICLR 2026\] Unpacking Human Preference for LLMs: Demographically Aware Evaluation with the HUMAINE Framework](../../ICLR2026/llm_evaluation/unpacking_human_preference_for_llms_demographically_aware_evaluation_of_long-fo.md)
- [\[ACL 2026\] AutoReproduce: Automatic AI Experiment Reproduction with Paper Lineage](../../ACL2026/llm_evaluation/autoreproduce_automatic_ai_experiment_reproduction_with_paper_lineage.md)
- [\[ICLR 2026\] AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite](../../ICLR2026/llm_evaluation/astabench_benchmarking_ai_agents.md)

</div>

<!-- RELATED:END -->
