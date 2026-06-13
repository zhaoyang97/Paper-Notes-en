---
title: >-
  [Paper Note] Towards Bridging the Reward-Generation Gap in Direct Alignment Algorithms
description: >-
  [ACL 2026][LLM Alignment][Direct Alignment Algorithms] This paper identifies the "reward-generation gap" in direct alignment algorithms (DAAs)—a mismatch between training objectives and autoregressive decoding dynamics.…
tags:
  - "ACL 2026"
  - "LLM Alignment"
  - "Direct Alignment Algorithms"
  - "Prefix Importance"
  - "Equal-length Training"
  - "Reward-Generation Gap"
  - "DPO/SimPO"
date: 2026-05-08
content_hash: b690451acea1119d
---

# Towards Bridging the Reward-Generation Gap in Direct Alignment Algorithms

**Conference**: ACL 2026 Findings  
**arXiv**: [2506.09457](https://arxiv.org/abs/2506.09457)  
**Code**: [GitHub](https://github.com/sustech-nlp/POET)  
**Area**: LLM Alignment / Preference Optimization  
**Keywords**: Direct Alignment Algorithms, Prefix Importance, Equal-length Training, Reward-Generation Gap, DPO/SimPO

## TL;DR

This paper identifies the "reward-generation gap" in direct alignment algorithms (DAAs)—a mismatch between training objectives and autoregressive decoding dynamics. It proposes POET (Prefix-Oriented Equal-length Training), which implicitly constrains the token-level MDP to converge across all time steps by truncating preference response pairs to the length of the shorter response, resulting in a maximum improvement of 11.8 percentage points on AlpacaEval 2.

## Background & Motivation

**Background**: Direct alignment algorithms (DAAs) such as DPO and SimPO have become efficient alternatives to RLHF. DAAs optimize directly on preference datasets via implicit reward functions, eliminating the need for explicit reward models and reinforcement learning.

**Limitations of Prior Work**: (1) DAAs may decrease the absolute reward of preferred responses while increasing the preference-non-preference reward gap; (2) higher preference rewards and larger reward gaps do not necessarily translate to better generation quality; (3) the implicit rewards in DAAs assign equal weight to every token, ignoring the critical importance of prefix tokens in autoregressive generation.

**Key Challenge**: DAAs optimize $r(x, y_w) \gg r(x, y_l)$ at the sequence level but cannot guarantee that the prefix level also satisfies $r(x, y_{w,<k}) \gg r(x, y_{l,<k})$. In autoregressive generation, errors in early tokens are amplified through accumulated exposure bias, making prefix quality the determinant of overall generation quality.

**Goal**: To analyze the limitations of DAAs from a token-level MDP perspective and design methods to bridge the gap between training objectives and generation dynamics.

**Key Insight**: Empirical observations indicate that the entropy of prefix tokens is significantly higher than that of subsequent tokens, yet their log-probabilities are diluted by the mean of numerous subsequent tokens. Consequently, DAAs fail to sufficiently focus on quality differences in the prefixes.

**Core Idea**: Truncate preference response pairs to an equal length (taking the length of the shorter one) to implicitly constrain the training objectives of DAAs to converge across all time steps, thereby enhancing the focus on prefix tokens.

## Method

### Overall Architecture

POET is a simple data augmentation method: given a preference response pair $(y_w, y_l)$, both are truncated to the length of the shorter one $k = \min(|y_w|, |y_l|)$. Standard DPO or SimPO training is then performed on these truncated, equal-length pairs. The method does not modify any optimization objectives and is compatible with all DAAs.

### Key Designs

1.  **Theoretical Foundation of the Equal-length Sub-trajectory BT Model**:
    *   **Function**: Proves that optimizing on equal-length sub-trajectories produces the same optimal policy as sequence-level optimization.
    *   **Mechanism**: Defines an equal-length sub-trajectory BT model $p_k^*(y_{w,\leq k} \succeq y_{l,\leq k})$, which incorporates the optimal state-value function $V^*$ following the truncation point. Theorem 1 proves that the optimal policy derived from this model is equivalent to the one derived from the original sequence-level BT model.
    *   **Design Motivation**: Provides rigorous theoretical guarantees that truncation does not alter the optimal policy while offering finer-grained reward signals for prefix tokens by training across various truncation lengths.

2.  **Empirical Validation of Prefix Quality Disparity**:
    *   **Function**: Verifies whether full-sequence preference labels remain valid after truncation.
    *   **Mechanism**: Computes the prefix quality difference $\Delta Q(k) = Q(y_{w,\leq k}) - Q(y_{l,\leq k})$ for different prefix lengths $k$ across 1,000 samples. Results show that quality differences emerge very early in the prefix and grow with length, albeit with diminishing marginal returns. This indicates high consistency (98.5%) in preference rankings post-truncation.
    *   **Design Motivation**: If the preference ranking remains unchanged after truncation, using full-sequence preference labels to train equal-length pairs is safe; this empirical validation serves as a key pillar for the feasibility of POET.

3.  **POET Data Augmentation Strategy**:
    *   **Function**: Naturally directs DAAs to focus on prefix quality without introducing additional hyperparameters.
    *   **Mechanism**: Setting $k = \min(|y_w|, |y_l|)$ means one response remains intact while only the suffix of the longer response is truncated. Since $k$ varies across different samples in the dataset, training implicitly occurs over multiple truncation lengths, constraining DAAs to converge across all MDP time steps.
    *   **Design Motivation**: Offers three major advantages: (1) universal compatibility with any DAA; (2) no additional hyperparameters; (3) minimized risk of data noise (as only suffixes are truncated, minimizing impact on overall quality).

### Loss & Training

The optimization objectives of DPO/SimPO are not modified; only the input data is changed. Training follows the hyperparameter settings of Meng et al. (2024). It supports both Base (starting from an SFT model) and Instruct (starting from an instruction-tuned model) configurations.

## Key Experimental Results

### Main Results

**AlpacaEval 2 & Arena-Hard Instruction Following Evaluation**

| Method | Mistral-7B LC% | Llama-3-8B LC% | Llama-3-Inst LC% | Gemma-2-9B LC% |
| :--- | :--- | :--- | :--- | :--- |
| DPO | 12.9 | 16.9 | 65.9 | 78.4 |
| DPO + POET | **24.7** (+11.8) | **28.4** (+11.5) | **70.4** (+4.5) | **79.7** (+1.3) |
| SimPO | 20.0 | 28.0 | 68.1 | 78.5 |
| SimPO + POET | **24.2** (+4.2) | **33.8** (+5.8) | **70.1** (+2.0) | **80.1** (+1.6) |

### Ablation Study

**Comparison of Truncation Strategies (Mistral-7B, AlpacaEval 2 LC%)**

| Truncation Strategy | Retain 25% | 50% | 75% | 100% |
| :--- | :--- | :--- | :--- | :--- |
| Original Length | 14.1 | 17.2 | 16.2 | 12.9 |
| POET Length | 23.5 | 24.9 | 26.7 | 24.7 |

### Key Findings

*   POET consistently improves AlpacaEval 2 LC across all 8 settings (4 models × 2 DAAs), with a maximum gain of +11.8 percentage points.
*   Ablation studies demonstrate that the equal-length truncation strategy (POET Len.) significantly outperforms truncation based on original lengths across all retention ratios, indicating that "equal length" itself is the critical factor.
*   POET does not increase the alignment tax—it maintains or slightly improves performance on downstream tasks in the HuggingFace Open Leaderboard.
*   Compared to token-level methods, POET significantly outperforms SamPO and D2PO; the latter even proved counterproductive when applied to SimPO.
*   Safety alignment evaluations also show a significant increase in safety rates.

## Highlights & Insights

*   Precise problem identification—the "reward-generation gap" captures a fundamental issue in DAAs: the mismatch between sequence-level optimization objectives and autoregressive generation dynamics.
*   Minimalist yet effective—significant improvements are achieved without modifying objectives or adding hyperparameters, simply by truncating data, suggesting that the root of the problem lies in the data format rather than the optimization algorithm.
*   Empirical analysis of prefix quality differences (Figure 2) is the most compelling part of the paper—using an oracle model to verify the crucial assumption that "preference ranking remains almost unchanged after truncation."

## Limitations & Future Work

*   Equal-length truncation relies on the assumption that "suffixes beyond the shorter response length have little impact on quality," which may not hold for extremely asymmetric pairs.
*   Theoretical guarantees assume the existence of an optimal state-value function $V^*$, which cannot be computed in practice.
*   Validation was limited to DPO and SimPO; it has not been extended to other DAAs like IPO or KTO.
*   Although preference consistency after truncation is high (91.4%-98.5%), some noise is still introduced.

## Related Work & Insights

*   **vs SamPO (Lu et al., 2024)**: SamPO calculates rewards using random subsets of tokens to mitigate length bias but does not focus on prefixes; POET focuses on prefixes and is effective for both DPO and SimPO.
*   **vs D2PO (Shao et al., 2025)**: D2PO emphasizes prefixes using exponentially decaying weights, but the decay factor introduces extra hyperparameters and can be counterproductive for SimPO; POET is hyperparameter-free and consistently effective.
*   **vs Token-level DPO (Rafailov et al., 2024b)**: Theoretically, token-level DPO can learn the optimal policy, but in practice, only sequence-level preference labels are available, preventing direct training.

## Rating

*   Novelty: ⭐⭐⭐⭐ Clear problem identification; simple method backed by strong theoretical motivation and empirical validation.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 models × 2 DAAs × detailed ablations + comparison with token-level methods + safety evaluation.
*   Writing Quality: ⭐⭐⭐⭐⭐ Tight logical chain from theory to empirical evidence to methodology.
*   Value: ⭐⭐⭐⭐⭐ High practical value as a plug-and-play, hyperparameter-free method compatible with all DAAs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] BACH-V: Bridging Abstract and Concrete Human-Values in Large Language Models](bach-v_bridging_abstract_and_concrete_human-values_in_large_language_models.md)
- [\[ICLR 2026\] Is On-Policy Data always the Best Choice for Direct Preference Optimization-based LM Alignment?](../../ICLR2026/llm_alignment/is_on-policy_data_always_the_best_choice_for_direct_preference_optimization-base.md)
- [\[ACL 2026\] RbtAct: Rebuttal as Supervision for Actionable Review Feedback Generation](rbtact_rebuttal_as_supervision_for_actionable_review_feedback_generation.md)
- [\[ACL 2026\] ModeX: Evaluator-Free Best-of-N Selection for Open-Ended Generation](modex_evaluator-free_best-of-n_selection_for_open-ended_generation.md)
- [\[AAAI 2026\] LaF-GRPO: In-Situ Navigation Instruction Generation for the Visually Impaired via GRPO with LLM-as-Follower Reward](../../AAAI2026/llm_alignment/laf-grpo_in-situ_navigation_instruction_generation_for_the_visually_impaired_via.md)

</div>

<!-- RELATED:END -->
