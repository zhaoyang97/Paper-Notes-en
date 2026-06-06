---
title: >-
  [Paper Note] From Human-Level AI Tales to AI Leveling Human Scales
description: >-
  [ICML 2026][LLM Evaluation][AI evaluation] This paper utilizes LLMs as population extrapolators to calibrate 18 ability dimensions according to the "world population accuracy" logarithmic scale $L=-\log_B p_W$. It finds…
tags:
  - "ICML 2026"
  - "LLM Evaluation"
  - "AI evaluation"
  - "psychometrics"
  - "ADeLe"
  - "world population calibration"
  - "LLM as annotator"
date: 2026-05-08
content_hash: b8dbd75a0eb5b348
---

# From Human-Level AI Tales to AI Leveling Human Scales

**Conference**: ICML 2026  
**arXiv**: [2602.18911](https://arxiv.org/abs/2602.18911)  
**Code**: None  
**Area**: AI evaluation / psychometrics  
**Keywords**: AI evaluation, psychometrics, ADeLe, world population calibration, LLM as annotator

## TL;DR
This paper utilizes LLMs as population extrapolators to calibrate 18 ability dimensions according to the "world population accuracy" logarithmic scale $L=-\log_B p_W$. It finds that the real base $B \gg 10$ for Volume/Attention dimensions, while $B \approx 1$ for the Comprehension dimension, revealing that current comparisons between AI and humans are severely misaligned.

## Background & Motivation

**Background**: The mainstream of AI evaluation is benchmarking—using a single average benchmark score to compare against "human level." This approach compresses different task difficulties, sample populations, and dimensional abilities into one number, leading to contradictory conclusions: LLMs score 90% on MMLU but only 50-70% on real-world software engineering tasks; on GPQA Diamond, PhDs score 70% while models reach 88%.

**Limitations of Prior Work**: (1) Benchmarks are incomparable, and "human level" depends entirely on the sampled reference population (mostly WEIRD: Western / Educated / Industrialized / Rich / Democratic); (2) existing criterion-referenced frameworks like ADeLe provide dimension-level rubrics, but setting the base $B=10$ is a convention rather than calibration, making dimensions still incomparable; (3) large-scale human measurement is extremely expensive and impossible to conduct for newly emerging benchmarks.

**Key Challenge**: To use "humans as a reference," human samples must be used, but available human samples are always biased small subsets; without calibration, conclusions of "surpassing humans" or "falling short of humans" are entirely sample-dependent.

**Goal**: (1) Map benchmark items to 18-dimensional ADeLe demand levels; (2) extrapolate arbitrary small-sample human performance to the World Wide Population (WWP); (3) back-calculate the true logarithmic base for each dimension based on WWP accuracy; (4) verify the reliability of the entire extrapolation suite.

**Key Insight**: Psychometrics has long used equating and post-stratification to handle extrapolation from small to large samples; modern LLMs have compressed massive demographic data and knowledge during training, allowing them to serve as cheap, repeatable population extrapolators.

**Core Idea**: Use LLMs to translate "focal-group success rate + demographic description of that group + demographic description of the target group" into "target group success rate," then perform linear regression for each ability dimension to obtain the real base $B = 10^m$, establishing a demographically anchored capability ruler.

## Method

### Overall Architecture
A 5-step pipeline: (1) Aggregate item pools (PISA 2009 / TIMSS 2003+2011 G4&G8 / ICAR / UKBioBank / ReliabilityBench); (2) use ADeLe rubrics to label each item with 18-dimensional demand levels $d_{i,c} \in \{0,1,2,3,4,5+\}$; (3) use LLMs to extrapolate focal group accuracy $p_i^g$ to WWP accuracy $p_i^W$; (4) transform to logarithmic difficulty as $L_i = -\log_B p_i^W$; (5) verify using sub-group to full-sample predictions (MAE / RMSE / Pearson / Spearman).

### Key Designs

1. **LLM as Population Extrapolator**:

    - **Function**: Translates item accuracy from any small sample to a "global population" baseline, avoiding human annotation bias.
    - **Mechanism**: The prompt contains 6 blocks: (a) dataset and test domain intro; (b) focal group demographic description (e.g., "15-year-old students in OECD countries from PISA 2009"); (c) item stem + options + correct answer; (d) measured focal group accuracy $p_i^g$; (e) reference group (world population) demographic description; (f) request for LLM output of predicted reference group accuracy $\hat p_i^W$ with rationale. The prompt explicitly lists 7 adjustment factors: global age distribution, education accessibility/quality, post-graduation forgetting, fluid/crystallized ability lifecycles, specialization/exposure, health/cognitive decline, and language factors. Robustness is ensured using 27 paraphrase versions.
    - **Design Motivation**: Traditional IRT requires many subjects, which is impossible for new benchmarks; the demographic statistics implicit in LLM training data provide a cheap proxy path, and rationales allow for auditing.

2. **Dimension-specific base calibration (Optimal Base)**:

    - **Function**: Calibrates the log-difficulty base from the default $B=10$ to the true steepness of each ability dimension.
    - **Mechanism**: Empirical difficulty $L_{\text{emp},i} = -\log_{10}(p_i^W / \sqrt{10})$; filter by "primary bottleneck"—keeping only items where $d_{i,c} \ge \max_k d_{i,k}$ for dimension $c$ to avoid interference. Take the mean $\bar y_l$ for each level $l \in \{1,..,5\}$, perform linear regression on $(l, \bar y_l)$, where slope $m$ gives $B = 10^m$. Results fall into three categories: High-base (Volume $B\approx 32$, Attention $B\approx 17$, steeper than expected); Standard (Metacognition $B\approx 6.7$, Knowledge $B\approx 5.1$); Invariant (Comprehension / Spatial $B\approx 1$, difficulty increase has almost no effect).
    - **Design Motivation**: The $B=10$ assumption fails across dimensions, making conclusions like "AI Knowledge exceeds humans but Reasoning is far inferior" difficult to compare horizontally; dimension-specific bases are key to aligning different rulers to the same unit.

3. **Dominance Filter + Means-based Regression**:

    - **Function**: Extracts pure dimensional signals from multi-bottleneck items and counters regression bias caused by high-difficulty sample scarcity.
    - **Mechanism**: Apply dominance filter for bottleneck items, then average by level (rather than raw point regression) to counter mean shifts caused by level 1 items far outnumbering high-level ones. Fit a line through 5 mean points; the slope is $\log_{10} B$.
    - **Design Motivation**: Level 1 items dominate the raw data; direct regression would flatten the slope. Regression on level means is a fair-weight compromise.

### Loss & Training
No training. LLMs used include GPT-5 Chat, GPT-4.1, Llama-4, DeepSeek-v3.1, and GROK-3. Inference at low temperature without tool use; each item × 27 paraphrases. Validation uses sub-group to full-sample designs from ICAR, TIMSS, and UKBioBank.

## Key Experimental Results

### Main Results (Verification of LLM Extrapolation Quality)

| Model | ICAR MAE ↓ | ICAR RMSE ↓ | ICAR Pearson ↑ | ICAR Spearman ↑ |
|------|------------|--------------|----------------|------------------|
| gpt-5-chat | **0.030** | **0.044** | **0.976** | **0.968** |
| llama-4 | 0.033 | 0.052 | 0.971 | 0.963 |
| gpt-4.1 | 0.040 | 0.058 | 0.958 | 0.944 |
| deepseek-v3.1 | 0.043 | 0.085 | 0.922 | 0.914 |
| grok-3 | 0.043 | 0.068 | 0.939 | 0.920 |

On TIMSS, MAE increases to $0.12$-$0.16$ and Pearson drops to $0.5$-$0.7$, reflecting greater difficulty in extrapolation when transnational heterogeneity is high.

### Ablation Study (Dimension-specific base calibration)

| Dimension Group | Calibrated $B$ | Interpretation |
|--------|------------|------|
| Volume | $\approx 32$ | Much steeper than $B=10$; high levels should be pushed up |
| Attention | $\approx 17$ | Same as above |
| Metacognition | $\approx 6.7$ | Close to $B=10$, well-calibrated |
| Knowledge | $\approx 5.1$ | Same as above |
| Comprehension & Expression | $\approx 1$ | Difficulty barely increases, levels should be pushed down |
| Spatial Reasoning & Navigation | $\approx 1$ | Same as above |

### Key Findings
- A single $B=10$ is invalid across dimensions—the real bases of Volume and Comprehension differ by $\approx 30\times$, meaning "AI leads humans in Knowledge" and "AI is far behind in Volume" cannot be compared without calibration.
- LLM extrapolation achieved an MAE of only $0.030$ (Pearson 0.976) on structurally homogeneous ICAR, proving LLMs compress significant demographic priors; however, error spikes on heterogeneous data like TIMSS (60 countries), indicating a Western bias remains.
- After applying calibrated bases, current LLM capability profiles show a distinct "Strong Knowledge, Weak Volume/Attention" shape, providing more interpretable comparisons for policymakers.

## Highlights & Insights
- Redefines "AI vs. Human" comparison from "benchmark score comparison" to "position on a logarithmic population distribution scale," a proposal at the level of evaluation philosophy.
- Using LLMs as population extrapolators is a clever "AI calibrating AI comparing humans" loop; the sub-group to full-sample validation proves it has learned demographic adjustment capabilities.
- Dimension-specific base results (Volume $\approx 32$, Comprehension $\approx 1$) directly undermine previous scalar conclusions of "AI reaching X% human level," serving as an impactful negative finding.

## Limitations & Future Work
- Only 5 data sources, all text-only; multimodal and agentic tasks are not covered.
- LLM extrapolators show larger MAE on TIMSS and obvious Western/Anglosphere bias; population estimates for non-Western cultures may be systematically biased.
- Assumes the dominance filter is sufficient to "purify" dimensional signals, but real items may have co-existing bottlenecks; filtered samples might still hold value.
- Calibrating bases using linear regression on 5 mean points has weak statistical significance, even yielding negative slopes for some dimensions (e.g., Mind Modeling).

## Related Work & Insights
- **vs. ADeLe (Zhou 2025)**: ADeLe provides demand rubrics but uses $B=10$ as a convention; this paper performs empirical calibration.
- **vs. METR time-horizon (Kwa 2025)**: METR uses human hours as a single-dimension anchor; this paper uses multi-dimensional population distributions.
- **vs. IRT psychometrics**: Classical IRT requires dense real response data; this paper uses LLMs to bypass that step.
- **vs. MMLU / GPQA**: These use scalar accuracy with a single reference population; this paper provides decomposable and comparable profiles.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ "LLM as population extrapolator + dimension-specific base calibration" is a rare methodological innovation.
- Experimental Thoroughness: ⭐⭐⭐ Only 5 data sources (text-only), large errors on TIMSS, and insufficient cross-cultural validation.
- Writing Quality: ⭐⭐⭐⭐ The motivation is compelling, and the technical narrative is clear.
- Value: ⭐⭐⭐⭐⭐ A paradigm-level reflection for the AI evaluation community; both policymakers and researchers should read it.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] CoCoReviewBench: A Completeness- and Correctness-Oriented Benchmark for AI Reviewers](cocoreviewbench_a_completeness-_and_correctness-oriented_benchmark_for_ai_review.md)
- [\[ACL 2026\] HoWToBench: Holistic Evaluation for LLM's Capability in Human-level Writing using Tree of Writing](../../ACL2026/llm_evaluation/howtobench_holistic_evaluation_for_llms_capability_in_human-level_writing_using_.md)
- [\[ICML 2026\] When AI Benchmarks Plateau: A Systematic Study of Benchmark Saturation](when_ai_benchmarks_plateau_a_systematic_study_of_benchmark_saturation.md)
- [\[ICML 2026\] When Hallucination Costs Millions: Benchmarking AI Agents in High-Stakes Adversarial Financial Markets (CAIA)](when_hallucination_costs_millions_benchmarking_ai_agents_in_high-stakes_adversar.md)
- [\[ICML 2026\] REAL: Integrating Regression-Aware Rewards into RL for Fine-Grained Human-Centric LLM Evaluation](real_regression-aware_reinforcement_learning_for_llm-as-a-judge.md)

</div>

<!-- RELATED:END -->
