---
title: >-
  [Paper Note] Ranked Voting based Self-Consistency of Large Language Models
description: >-
  [ACL 2025][Reasoning][Self-Consistency] Upgrades the majority voting in Self-Consistency to ranked voting, allowing the LLM to generate a preference ranking of multiple candidate answers for each reasoning path instead of a single answer. It uses three ranked voting methods (IRV/BCV/MRRV) to aggregate ranking information across multiple reasoning paths, consistently performing better than traditional SC on six datasets with a maximum improvement of 12.46%.
tags:
  - "ACL 2025"
  - "Reasoning"
  - "Self-Consistency"
  - "Ranked Voting"
  - "Borda Count"
  - "Instant Runoff Voting"
  - "MRR"
date: 2026-05-08
content_hash: a039632e614bee38
---

# Ranked Voting based Self-Consistency of Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2505.10772](https://arxiv.org/abs/2505.10772)  
**Code**: [https://github.com/szu-tera/RankedVotingSC](https://github.com/szu-tera/RankedVotingSC)  
**Area**: LLM Reasoning  
**Keywords**: Self-Consistency, Ranked Voting, Borda Count, Instant Runoff Voting, MRR

## TL;DR
Upgrades the majority voting in Self-Consistency to ranked voting, allowing the LLM to generate a preference ranking of multiple candidate answers for each reasoning path instead of a single answer. It uses three ranked voting methods (IRV/BCV/MRRV) to aggregate ranking information across multiple reasoning paths, consistently performing better than traditional SC on six datasets with a maximum improvement of 12.46%.

## Background & Motivation

**Background**: Self-Consistency (SC) allows LLMs to perform Chain-of-Thought (CoT) reasoning multiple times and select the final answer via majority voting, representing a mainstream method for improving reasoning accuracy.

**Limitations of Prior Work**: Traditional SC only selects one answer per reasoning path for voting, discarding the model's preference information over other candidate answers, which wastes "second-choice" and "third-choice" information.

**Key Challenge**: Majority voting only utilizes top-1 information, ignoring the correct answer if it frequently appears in top-2/3 but not top-1.

**Goal**: Fully utilize the preference ranking information of multiple candidate answers from the model in each reasoning run.

**Key Insight**: Drawing inspiration from ranked voting methods in social choice/voting theory (IRV, Borda Count, MRR), the voting granularity is upgraded from "single choice" to "ranking" in SC.

**Core Idea**: Have the LLM generate a ranking of candidate answers instead of a single answer, replacing majority voting with ranked voting for Self-Consistency aggregation.

## Method

### Overall Architecture
The differences from standard SC lie in only two aspects: (1) using few-shot prompts to guide the LLM to output $m$ ranked candidates $\mathcal{A}^r = \{\mathcal{A}^{r_1} \succ \mathcal{A}^{r_2} \succ ... \succ \mathcal{A}^{r_m}\}$; (2) aggregating the results of $k$ reasoning runs using ranked voting instead of majority voting.

### Key Designs

1. **Three Ranked Voting Methods**:

    - **IRV (Instant Runoff Voting)**: Iteratively eliminates the candidate with the fewest first-choice votes until a candidate exceeds 50%. Suitable for scenarios with dispersed votes, finding the most universally accepted answer through multiple rounds of screening.
    - **Borda Count (BCV)**: $\text{BordaCount}(\mathcal{A}) = \sum_{i=1}^{k}(m - \text{rank}_\mathcal{A}(\mathcal{A}_i^r) + 1)$, weighting by assigned points based on rank. The 1st place gets $m$ points, the 2nd place gets $m-1$ points, etc., decaying linearly.
    - **MRRV (Mean Reciprocal Rank Voting)**: $\text{MRR}(\mathcal{A}) = \frac{1}{k}\sum_{i=1}^{k}\frac{1}{\text{rank}_\mathcal{A}(\mathcal{A}_i^r)}$, assigning higher weights to top ranks. The 1st place has a weight of 1.0, 2nd has 0.5, 3rd has 0.33, etc. This non-linear decay emphasizes top position information.

2. **Ranked Answer Generation**:

    - Function: Guide the LLM to output multiple candidates and their ranking using few-shot examples.
    - Mechanism: Show examples in prompt, requiring the model to reason first, then output "The ranking of options by likelihood is: A > B > D > C".
    - Design Motivation: Simple yet effective—no changes to the model architecture, only modifying the prompt and the voting method.
    - For multiple-choice questions: Rank all options; for open-ended QA: Generate $m$ most likely answers and rank them.

3. **Tie-Breaking**:

    - Function: Decision-making mechanism when multiple candidates obtain identical scores.
    - Mechanism: Calculate the token probability confidence score $\mathcal{S}_i = \sum_{t=1}^{n} \log(p(\mathcal{C}_{i,t}))$ for each candidate and select the one with the highest confidence.
    - Result: Ranked voting itself significantly reduces ties (from 5.08% to 2.29%).

4. **Few-Shot Prompt Construction**:

    - Core Guideline: Ensure strong semantic correlation between questions and candidate answers.
    - Expansion Strategy: Manually construct template examples, then use LLM to automatically generate more examples with minimal human validation required.

### Loss & Training
- Zero-training method, applied purely at inference time.
- Applicable to both open-source and closed-source LLMs.

## Key Experimental Results

### Main Results

| Model | Method | AQUA-RAT | CommonsenseQA | ARC-C | Average |
|------|------|----------|-------------|-------|------|
| LLaMA-3.2-3B | SC | 61.81% | 73.46% | 80.54% | 62.47% |
| | **MRRV** | **71.26%** | 74.45% | **81.40%** | **65.79%** |
| Qwen-2.5-3B | SC | 77.95% | 77.89% | 76.96% | 60.13% |
| | **IRV** | **79.13%** | **78.95%** | **83.45%** | **64.76%** |
| LLaMA-3-8B | SC | 66.93% | 78.71% | 86.77% | 68.04% |
| | **MRRV** | **75.20%** | 79.36% | **87.63%** | **71.55%** |
| Phi-3-4B | SC | 73.62% | 75.84% | 90.13% | 67.73% |
| | **MRRV** | **75.20%** | **78.95%** | 90.44% | **69.53%** |
| GPT-3.5-turbo | SC | - | - | - | 71.36% |
| | **MRRV** | - | - | - | **76.69%** |

### Ablation Study

| Configuration | Finding |
|------|------|
| k=2→16 | Ranked voting consistently outperforms majority voting across all $k$ values |
| Single ranked answer (no voting) | No consistent improvement $\rightarrow$ shows that the improvement stems from ranked voting aggregation |
| Randomly shuffle few-shot | Ranked voting exhibits lower variance and is more robust |
| c=1→5 candidate count | Outperforms SC when $c \ge 4$ |

### Key Findings
- **Largest improvement on AQUA-RAT (+8-9%)**: Distractors in multiple-choice questions easily appear in top-2/3 positions.
- **Diminishing gains on stronger models**: Only +0.48% on GPT-4-turbo, while +5.33% on GPT-3.5.
- **MRRV performs the best overall**: Non-linear decay weights make the most full use of ranking information.
- **Reduced tie rate**: Ranked voting reduces tie rates from 5.08% to 2.29% (using IRV).

## Highlights & Insights
- **Cross-domain transfer from voting theory to LLM reasoning**: Introducing established voting methods from social choice theory to SC. The strategy is simple yet effective.
- **Zero-cost improvement**: No modifications to the model, no extra training required, only modifications to the prompt and post-processing.
- **Reveals information waste in SC**: Majority voting only utilizes top-1 information; this finding is inspiring for all methods using SC.

## Limitations & Future Work
- **Limited candidates**: The diversity of ranked candidates in open-ended QA depends heavily on the model's generation capacity.
- **Small gains on stronger models**: GPT-4-level models already have very high top-1 accuracy, leading to diminishing marginal returns from ranking information.
- **Only three voting methods explored**: More complex voting methods, such as the Schulze method or Copeland method, remain unexplored.
- Potential improvements: Combining weighted or adaptive voting strategies.

## Related Work & Insights
- **vs Self-Consistency (Wang et al., 2023)**: SC only uses top-1 votes, whereas this work utilizes complete ranking information. It achieves an average gain of 2-5% on SC benchmarks without increasing extra inference cost (sample size $k$ remains identical).
- **vs Universal SC (Chen et al., 2023)**: USC also improves SC but focuses on answer equivalence determination (e.g., treating "1/2" and "0.5" as identical), which is orthogonal to and can be combined with ranked voting.
- **vs Adaptive-SC**: Dynamically adjusts sampling frequency but still relies on majority voting. Ranked voting is consistently better under the same sampling budget.
- **vs Best-of-N**: Selects the response with the highest model probability directly without voting. Ranked voting outperforms Best-of-N across all models.

## Rating
- Novelty: ⭐⭐⭐⭐ Simple and effective cross-domain transfer of voting theory.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 6 datasets, multiple model sizes, and detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear formulas and comprehensive experiments.
- Value: ⭐⭐⭐⭐ Highly practical, plug-and-play upgrade of SC.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Revisiting Self-Consistency from Dynamic Distributional Alignment Perspective on Answer Aggregation](revisiting_self-consistency_from_dynamic_distributional_alignment_perspective_on.md)
- [\[ICML 2025\] Self-Consistency Preference Optimization](../../ICML2025/llm_reasoning/self-consistency_preference_optimization.md)
- [\[ACL 2025\] Large Language and Reasoning Models are Shallow Disjunctive Reasoners](large_language_and_reasoning_models_are_shallow_disjunctive_reasoners.md)
- [\[NeurIPS 2025\] Scalable Best-of-N Selection for Large Language Models via Self-Certainty](../../NeurIPS2025/llm_reasoning/scalable_best-of-n_selection_for_large_language_models_via_self-certainty.md)
- [\[NeurIPS 2025\] A Theoretical Study on Bridging Internal Probability and Self-Consistency for LLM Reasoning](../../NeurIPS2025/llm_reasoning/a_theoretical_study_on_bridging_internal_probability_and_sel.md)

</div>

<!-- RELATED:END -->
