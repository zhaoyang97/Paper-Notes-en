---
title: >-
  [Paper Note] BiasScope: Towards Automated Detection of Bias in LLM-as-a-Judge Evaluation
description: >-
  [ICLR 2026][LLM Evaluation][LLM-as-a-Judge] This paper proposes BiasScope, a fully LLM-driven iterative framework that automatically discovers previously unknown biases in LLM-as-a-Judge evaluation at scale. Based on the discovered biases, the authors construct JudgeBench-Pro, a more challenging benchmark on which even powerful LLM judges exceed 50% error rate.
tags:
  - ICLR 2026
  - LLM Evaluation
  - LLM-as-a-Judge
  - bias discovery
  - evaluation robustness
  - automated bias mining
  - JudgeBench-Pro
date: 2026-05-08
content_hash: dda25c9c9d673905
---

# BiasScope: Towards Automated Detection of Bias in LLM-as-a-Judge Evaluation

**Conference**: ICLR 2026
**arXiv**: [2602.09383](https://arxiv.org/abs/2602.09383)
**Code**: [https://github.com](https://github.com) (available, includes code and JudgeBench-Pro data)
**Area**: LLM Evaluation
**Keywords**: LLM-as-a-Judge, bias discovery, evaluation robustness, automated bias mining, JudgeBench-Pro

## TL;DR
This paper proposes BiasScope, a fully LLM-driven iterative framework that automatically discovers previously unknown biases in LLM-as-a-Judge evaluation at scale. Based on the discovered biases, the authors construct JudgeBench-Pro, a more challenging benchmark on which even powerful LLM judges exceed 50% error rate.

## Background & Motivation

### State of the Field
LLM-as-a-Judge has been widely adopted in benchmark construction, data filtering, and model performance evaluation, leveraging LLMs as "judges" to automatically assess model outputs at scale.

### Limitations of Prior Work

**Bias research limited to known types**: Existing work primarily focuses on verifying the impact of known biases (e.g., position bias, length bias, self-preference bias) on evaluation results, lacking systematic exploration of unknown latent biases.

**Manual discovery does not scale**: Identifying new bias types by hand is costly and has limited coverage.

**Passive discovery paradigm**: Conventional methods rely on researchers predefining a list of biases and then validating them one by one, precluding proactive mining.

### Root Cause
LLM-as-a-Judge is widely used, yet the reliability and robustness of its evaluations cannot be guaranteed—unknown biases may have a greater impact than known ones, and automated, systematic means of discovering them are currently absent.

### Paper Goals
How can unknown biases that arise during LLM evaluation be discovered automatically and at scale?

### Starting Point
A teacher model injects known biases into data to "stimulate" the target model into revealing new bias tendencies. A cascaded-error strategy (DeeperExplain) then further mines deeper biases, forming an iterative self-expanding bias-space exploration mechanism.

### Core Idea
Through an iterative pipeline of "bias injection → misprediction collection → error cascading → bias identification → validation and archiving," bias discovery is transformed from passive manual exploration into active automated mining.

## Method

### Overall Architecture
BiasScope is a two-stage iterative framework:
- **Input**: target model $M$, target dataset $\mathcal{D}$ with ground-truth preference labels, initial bias library $\mathcal{B}_0$
- **Stage 1 — Bias Discovery**: perturb data with known biases → target model evaluation → collect mispredictions and explanations → error cascading for deeper explanations → teacher model identifies new biases → merge and deduplicate
- **Stage 2 — Bias Validation**: validate candidate biases on an independent test set to confirm their efficacy → valid biases are added to the library
- **Iteration**: repeat both stages until convergence (no new bias validated, library stable, or maximum iteration count reached)
- **Output**: final bias library $\mathcal{B}_T$

### Key Designs

1. **Bias Injection and Misprediction Collection**:

    - **Function**: Sample a bias $b_k$ from the bias library; the teacher model perturbs the rejected response $y_i^r$ to produce $\tilde{y}_i^r$, leaving the chosen response unchanged.
    - **Mechanism**: Construct the perturbed dataset $\tilde{\mathcal{D}}_t$, have the target model evaluate it, and collect all mispredicted samples (where the model selects the rejected response) along with their explanations $E_i$.
    - **Design Motivation**: Known biases can "leverage" the model to expose deeper latent bias tendencies.

2. **Cascaded-Error Strategy (DeeperExplain)**:

    - **Function**: Follow up on the model's erroneous explanations by prompting it to "explain its own faulty reasoning," inducing further exposure of deeper biases.
    - **Mechanism**: $E_i' = \text{DeeperExplain}(x_i, y_i^c, \tilde{y}_i^r, E_i; M)$; erroneous reasoning is used to elicit additional biased behavior.
    - **Design Motivation**: The model's original erroneous explanations are insufficient to fully reveal biases; further prompting can trigger additional latent biases. Ablation results show that this strategy increases the number of discovered biases for Qwen2.5-7B from 25 to 27 and for Qwen2.5-1.5B from 43 to 48.

3. **Bias Identification and Merge-Deduplication**:

    - **Function**: The teacher model identifies new biases from misprediction data, then performs pairwise similarity comparisons against existing biases to merge redundant ones.
    - **Mechanism**: First identify $\tilde{\mathcal{B}}_t$, then construct $\mathcal{B}_t^{\text{temp}} = \tilde{\mathcal{B}}_t \cup \mathcal{B}_t$ and merge pairwise.
    - **Design Motivation**: Ensures the final bias set consists of independent, non-overlapping entries, avoiding double-counting.

4. **Test-Set Validation**:

    - **Function**: Validate the efficacy of each candidate bias using an independent test set.
    - **Mechanism**: For each candidate bias $b_j$, the teacher model perturbs the entire test set; the target model's error rate on the perturbed data is compared against that on the original data. If $\text{Err}(\tilde{\mathcal{D}}_j^{\text{test}}) > \text{Err}(\mathcal{D}^{\text{test}})$, the bias is deemed valid and added to the library.
    - **Design Motivation**: An independently and objectively annotated test set (JudgeBench) is used to eliminate subjective preference noise.

### Loss & Training
- Pair-wise evaluation is adopted for bias identification.
- Answer positions are randomly swapped during evaluation to neutralize position bias.
- Greedy decoding with fixed random seeds is used to ensure reproducibility.
- The initial bias library contains 7 known biases.
- The maximum number of iterations is set to 4 (most models converge near this limit).

## Key Experimental Results

### Main Results
BiasScope is applied to 7 target models of varying scales and families, with JudgeBench as the test set:

| Target Model | Validated Biases | Original Err (%) | BiasScope Err (%) | Gain |
|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 48 | 48.6 | 53.1 | +4.5 |
| InternLM3-8B-Instruct | 19 | 45.3 | 50.7 | +5.4 |
| Mistral-7B-Instruct-v0.3 | 41 | 43.9 | 51.2 | +7.3 |
| Qwen2.5-7B-Instruct | 27 | 43.4 | 48.1 | +4.7 |
| LLaMA-3.1-8B-Instruct | 29 | 41.7 | 52.5 | +10.8 |
| Qwen2.5-14B-Instruct | 19 | 37.7 | 47.8 | +10.1 |
| Qwen3-8B (Non-Thinking) | 14 | 36.9 | 42.7 | +5.8 |
| **Average** | - | - | - | **+6.9** |

### Ablation Study

| Configuration | Key Metric | Note |
|---|---|---|
| Early-Validate (default) | LLaMA: 29 biases, Err 52.5% | Per-round validation discovers more biases |
| Late-Validate | LLaMA: 27 biases, Err 52.2% | Deferred validation yields slightly fewer biases |
| With DeeperExplain | Qwen2.5-7B: 27, 1.5B: 48 | Error cascading uncovers more biases |
| Without DeeperExplain | Qwen2.5-7B: 25, 1.5B: 43 | ~10% fewer biases discovered |
| GPT-OSS-120B as Teacher | LLaMA: 19 biases, Err 53.8% | Stronger teacher finds more valid biases |
| GPT-OSS-20B as Teacher | LLaMA: 9 biases, Err 47.7% | Weaker teacher halves bias count |

### Key Findings
- **Simpler domains are more susceptible to bias**: The math domain has the lowest baseline error rate, yet shows the largest increase after bias injection (+11.1%), indicating that simpler tasks are more prone to judgment disruption by biases.
- **Stronger models exhibit fewer discoverable biases**: Within the Qwen2.5 family, the number of discoverable biases decreases as model size increases, suggesting that larger models yield more stable evaluation processes.
- **Length is not the root cause of error rate increases**: Truncation experiments show that multi-bias perturbations maintain higher error rates even after length is controlled (+2.2%), whereas pure length bias drops below baseline after truncation (−2.5%).
- **Discovered biases transfer across models**: Biases discovered on Qwen2.5-1.5B and used to construct JudgeBench-Pro also significantly degrade the performance of closed-source models such as GPT-4o.

## JudgeBench-Pro Benchmark
- Built from 620 JudgeBench samples, each generating 10 bias variants (6,200 total); after adversarial filtering by strong models and manual review, 1,178 high-quality samples are retained.
- Four out of five mainstream strong models perform no better than random chance on JudgeBench-Pro.
- GPT-4o reaches an error rate of 74.7%; only Doubao-Seed-1-6 performs relatively well (20.4%).
- Rejected responses are only 8.4% longer than originals on average, ruling out a pure length effect.
- Inter-annotator agreement (Fleiss' Kappa) = 0.92.

## Bias Mitigation Validation
Discovered biases are used to construct augmented preference data for DPO training:
- Standard UltraFeedback DPO training actually increases error rate (Mistral: 14.3→20.6%).
- Bias-augmented DPO training reduces error rate (Mistral: 14.3→13.3%; LLaMA: 21.5→20.3%).

## Highlights & Insights
- **Cascaded-error strategy**: Exploiting the model's "explanation of its own errors" to induce further error exposure is a clever "error-begets-error" design that can transfer to other red-teaming scenarios.
- **Biases mined from small models transfer to large/closed-source models**: Running the framework on cost-friendly small open-source models yields biases that also expose weaknesses in closed-source models such as GPT-4o, substantially lowering the practical barrier to adoption.
- **Closed-loop from bias discovery to bias mitigation**: The work goes beyond identifying problems by using bias-augmented data in DPO training to address them.

## Limitations & Future Work
- The bias discovery pipeline remains computationally expensive, requiring a powerful teacher model and multiple iterations.
- Bias validation depends on test sets with objectively correct answers, limiting applicability to subjective evaluation scenarios.
- The maximum iteration count of 4 may leave deeper biases undiscovered.
- The framework currently focuses solely on the pair-wise evaluation paradigm and does not cover point-wise or reference-based evaluation.

## Related Work & Insights
- **vs. CALM (Ye et al., 2024)**: CALM constructs benchmarks from known biases to quantify their degree of influence—a "passive validation" approach; BiasScope takes an "active discovery" approach capable of mining unknown biases.
- **vs. JudgeBench (Tan et al., 2025)**: JudgeBench provides an objectively annotated evaluation benchmark; BiasScope builds upon it to construct the more challenging JudgeBench-Pro.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The shift from passively validating known biases to actively mining unknown ones represents a genuine methodological innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 7 models, multiple ablations, reliability verification, length-controlled experiments, DPO mitigation validation, and JudgeBench-Pro construction and evaluation—highly comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, rigorous formalization, and well-integrated figures and tables.
- **Value**: ⭐⭐⭐⭐ Provides practical tools and a new benchmark for robustness evaluation of LLM-as-a-Judge.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Contrastive Decoding Mitigates Score Range Bias in LLM-as-a-Judge](../../ACL2026/llm_evaluation/contrastive_decoding_mitigates_score_range_bias_in_llm-as-a-judge.md)
- [\[AAAI 2026\] LLM-as-a-Judge for Scalable Test Coverage Evaluation](../../AAAI2026/llm_evaluation/llm-as-a-judge_for_scalable_test_coverage_evaluation_accuracy_operational_reliab.md)
- [\[ICLR 2026\] Preference Leakage: A Contamination Problem in LLM-as-a-judge](preference_leakage_a_contamination_problem_in_llm-as-a-judge.md)
- [\[ICLR 2026\] Talk, Evaluate, Diagnose: User-aware Agent Evaluation with Automated Error Analysis](talk_evaluate_diagnose_user-aware_agent_evaluation_with_automated_error_analysis.md)
- [\[ICLR 2026\] LLM Unlearning with LLM Beliefs](llm_unlearning_with_llm_beliefs.md)

</div>

<!-- RELATED:END -->
