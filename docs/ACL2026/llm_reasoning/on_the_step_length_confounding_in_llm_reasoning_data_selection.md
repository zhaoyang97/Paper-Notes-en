---
title: >-
  [Paper Note] On the Step Length Confounding in LLM Reasoning Data Selection
description: >-
  [ACL 2026][LLM Reasoning][Reasoning Data Selection] This paper identifies the "step length confounding" problem in naturalness-based LLM reasoning data selection—a systematic preference for samples with longer steps rath…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Reasoning Data Selection"
  - "Step Length Confounding"
  - "Naturalness"
  - "First Token"
  - "Causal Debiasing"
date: 2026-05-08
content_hash: cf42c53efd589f69
---

# On the Step Length Confounding in LLM Reasoning Data Selection

**Conference**: ACL 2026  
**arXiv**: [2604.06834](https://arxiv.org/abs/2604.06834)  
**Code**: [GitHub](https://github.com/wangbing1416/ASLEC)  
**Area**: Social Computing  
**Keywords**: Reasoning Data Selection, Step Length Confounding, Naturalness, First Token, Causal Debiasing

## TL;DR

This paper identifies the "step length confounding" problem in naturalness-based LLM reasoning data selection—a systematic preference for samples with longer steps rather than higher quality. The root cause is that the low probability of the first token in a reasoning step is diluted by long steps. The authors propose two correction methods, Aslec-drop (dropping first token probability) and Aslec-casl (causal regression debiasing), achieving an average accuracy improvement of 6-9%.

## Background & Motivation

**Background**: Constructing high-quality SFT data is central to training large reasoning models (e.g., DeepSeek-R1). Existing data selection methods are divided into heuristic rules (answer correctness, diversity, difficulty) and naturalness-based methods (using LLM log-probabilities/perplexity to score and select samples with the highest model fitness).

**Limitations of Prior Work**: Naturalness-based methods (e.g., GRACE, Local LP) exhibit severe bias on long CoT datasets—they systematically prefer samples with more tokens per step rather than truly high-quality ones. The step length distribution of selected data significantly differs from unselected data.

**Key Challenge**: The first token of a reasoning step often branches into different reasoning paths, thus possessing higher entropy and lower log-probability. In long steps, the proportion of first tokens is smaller, and their low probability is diluted by more non-first tokens. This leads to higher average log-probabilities for long steps, making them easier to be selected.

**Goal**: To quantify and eliminate this step length confounding effect so that data selection is unaffected by step length bias.

**Key Insight**: Start from the first-token probability—since the root cause is that the low probability of the first token exerts different impacts across different step lengths, directly intervene in the contribution of the first token.

**Core Idea**: Two methods—Aslec-drop directly discards first-token probabilities from score calculation; Aslec-casl treats the first-token ratio as a confounding factor and removes its influence using causal debiasing regression.

## Method

### Overall Architecture

Given $N$ questions with $K$ candidate answers each, a high-quality subset must be selected for SFT. Traditional methods score using average log-probability and select the highest performers. This paper intervenes in the contribution of the first token during the scoring stage to generate debiased scores for selection.

### Key Designs

1.  **Aslec-drop (Drop First Token)**:
    - **Function**: Eliminates step length confounding by excluding first-token probabilities.
    - **Mechanism**: Splits an answer $\mathbf{o}_i$ into $L$ reasoning steps and skips the first token of each step when calculating the average log-probability: $s_i^{drop} = \frac{1}{|\mathbf{o}_i| - |\mathcal{S}_i|} \sum_{\mathbf{s}_i^l} \sum_{t=2}^{|\mathbf{s}_i^l|} \log P_\theta(s_{i,t}^l | \text{context})$. The denominator is adjusted to the total token count excluding first tokens.
    - **Design Motivation**: The most direct way to eliminate the issue—since the first token is the source of confounding, it is removed from scoring. The disadvantage is that useful information carried by the first token is also discarded.

2.  **Aslec-casl (Causal Debiasing Regression)**:
    - **Function**: Removes step length confounding while retaining first-token information.
    - **Mechanism**: Decomposes the log-probability using linear regression: $s_i^{logp} = \beta_1 s_i^{first} + \beta_2 s_i^{drop} + \gamma \mathcal{Z}_i + \epsilon$, where $\mathcal{Z}_i = |\mathcal{S}_i| / |\mathbf{o}_i|$ is the first-token ratio (confounding factor). After estimating $\gamma$ via OLS, the final debiased score is $s_i^{casl} = s_i^{logp} - \gamma \mathcal{Z}_i$.
    - **Design Motivation**: The causal debiasing framework treats step length as a confounding factor and adjusts for its impact through regression. This is more refined than direct dropping and preserves useful signals from the first token.

3.  **Quantitative Analysis of Step Length Confounding**:
    - **Function**: Establishes the causal chain of the confounding effect.
    - **Mechanism**: Three-step verification—(1) Selected data has significantly longer step lengths; (2) Average log-probabilities of long steps increase monotonically; (3) First tokens consistently have the lowest log-probabilities across all steps, and long steps dilute their impact.
    - **Design Motivation**: Diagnose before treatment; find the intervention point by quantifying the causal chain.

### Loss & Training

Aslec is a data selection method and does not involve training itself. After selecting the data, the target model is trained using standard SFT (cross-entropy loss).

## Key Experimental Results

### Main Results (LIMO-v2, Qwen3-4B-Base)

| Method | AIME24 | AIME25 | MATH500 | Average |
| :--- | :--- | :--- | :--- | :--- |
| GRACE | 16.66 | 15.83 | 59.40 | 31.42 |
| Local LP | 19.16 | 20.83 | 71.60 | 36.50 |
| **Aslec-drop** | **30.00** (+10.84) | **28.33** (+7.50) | **77.80** (+6.20) | **44.64** |
| **Aslec-casl** | **31.66** (+12.50) | **30.83** (+10.00) | **80.00** (+8.40) | **47.54** |

### Ablation Study

| Analysis | Findings |
| :--- | :--- |
| Step length vs. Total length | The step length confounding effect is much stronger than the total length effect |
| Aslec-drop vs. Aslec-casl | Aslec-casl is consistently superior because it retains first-token information |
| Cross-model consistency | Consistently effective across Qwen3-4B, 8B, 32B, and Llama-3.1-8B |

### Key Findings
- Aslec-casl improves by approximately 9.08% on average compared to the SOTA method Local LP, while Aslec-drop improves by about 6.28%.
- The confounding effect consistently exists across all four naturalness-based methods (GRACE, Local LP, Min Entropy, Min Perplex).
- The low probability of the first token is the root cause of confounding, aligning with previous research on the branching behavior of reasoning step first tokens.
- Aslec-casl's causal regression has a closed-form solution with negligible computational overhead.
- Effects are consistent across various model sizes (4B-32B) and different datasets (LIMO-v2, AceReason).

## Highlights & Insights
- The **discovery of the "Step Length Confounding" phenomenon** is a major contribution: it reveals a systematic bias in LLM reasoning data selection that is widely overlooked but significant, with clear and reproducible explanations.
- The **application of the causal debiasing framework** is ingenious: by taking the first-token ratio as a confounder and using classic linear regression for causal debiasing, the method is both elegant and effective.
- **Insights into "first token branching behavior"** connect reasoning data selection with the understanding of the reasoning process.

## Limitations & Future Work
- The linear regression assumes step length confounding is linear, potentially missing non-linear confounding effects.
- Step segmentation depends on "\n\n" or sentence boundaries; different segmentation methods may affect results.
- Only mathematical reasoning tasks were verified; the performance on other tasks like code reasoning or natural language reasoning remains unknown.
- The "branching behavior" hypothesis of first tokens might not apply to all reasoning modes.
- Future work could explore incorporating step length information as a regularization target during training.

## Related Work & Insights
- **vs. GRACE / Local LP**: These naturalness-based methods suffer from step length confounding; Aslec directly corrects this by intervening in first-token probabilities.
- **vs. Heuristic Data Selection**: Heuristic methods (correctness, difficulty, etc.) do not directly consider model fitness; Aslec removes bias while retaining the advantages of naturalness-based methods.
- **vs. IFD / Deita**: These methods use perplexity differences between models or reward model scores; they are orthogonal to naturalness-based methods.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The discovery of step length confounding is a significant contribution; the causal debiasing method is concise and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Verified across multiple models, datasets, and benchmarks with thorough analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain from problem diagnosis to causal analysis to solution is very clear.
- Value: ⭐⭐⭐⭐⭐ Directly and significantly impacts the practice of LLM reasoning data selection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Stabilizing Efficient Reasoning with Step-Level Advantage Selection](stabilizing_efficient_reasoning_with_step-level_advantage_selection.md)
- [\[ACL 2026\] Budget-Aware Anytime Reasoning with LLM-Synthesized Preference Data](budget-aware_anytime_reasoning_with_llm-synthesized_preference_data.md)
- [\[ACL 2026\] LLM Reasoning as Trajectories: Step-Specific Representation Geometry and Correctness Signals](llm_reasoning_as_trajectories_step-specific_representation_geometry_and_correctn.md)
- [\[ACL 2026\] Reasoning Fails Where Step Flow Breaks](reasoning_fails_where_step_flow_breaks.md)
- [\[ACL 2026\] Step-GRPO: Internalizing Dynamic Early Exit for Efficient Reasoning](step-grpo_internalizing_dynamic_early_exit_for_efficient_reasoning.md)

</div>

<!-- RELATED:END -->
