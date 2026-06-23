---
title: >-
  [Paper Note] Towards Bridging the Reward-Generation Gap in Direct Alignment Algorithms
description: >-
  [ACL 2026][Alignment & RLHF][DPO/SimPO] This paper identifies the "reward-generation gap" in Direct Alignment Algorithms (DAAs)—a mismatch between training objectives and autoregressive decoding dynamics. The authors propose POET (Prefix-Oriented Equal-length Training), which implicitly constrains the token-level MDP to converge at all timesteps by truncatin
tags:
  - ACL 2026
  - Alignment & RLHF
  - DPO/SimPO
date: 2026-05-08
content_hash: d835a6aea4b94184
---
# Towards Bridging the Reward-Generation Gap in Direct Alignment Algorithms

**Conference**: ACL 2026 Findings  
**arXiv**: [2506.09457](https://arxiv.org/abs/2506.09457)  
**Code**: [GitHub](https://github.com/sustech-nlp/POET)  
**Area**: LLM Alignment / Preference Optimization  
**Keywords**: Direct Alignment Algorithms, Prefix Importance, Equal-length Training, Reward-Generation Gap, DPO/SimPO

## TL;DR

This paper identifies the "reward-generation gap" in Direct Alignment Algorithms (DAAs)—a mismatch between training objectives and autoregressive decoding dynamics. The authors propose POET (Prefix-Oriented Equal-length Training), which implicitly constrains the token-level MDP to converge at all timesteps by truncating preference pairs to the length of the shorter response, achieving up to an 11.8 percentage point improvement on AlpacaEval 2.

## Background & Motivation

**Background**: Direct Alignment Algorithms (DAAs) such as DPO and SimPO have become efficient alternatives to RLHF. DAAs optimize directly on preference datasets via implicit reward functions, bypassing the need for explicit reward models and reinforcement learning.

**Limitations of Prior Work**: (1) DAAs may decrease the absolute reward of preferred responses while increasing the margin between preferred and non-preferred ones; (2) higher preference rewards and larger margins do not necessarily translate to better generation quality; (3) the implicit rewards in DAAs assign equal weight to every token, ignoring the critical importance of prefix tokens in autoregressive generation.

**Key Challenge**: DAAs optimize $r(x, y_w) \gg r(x, y_l)$ at the sequence level but cannot guarantee $r(x, y_{w,<k}) \gg r(x, y_{l,<k})$ at the prefix level. In autoregressive generation, errors in early tokens are amplified through cumulative exposure bias; thus, prefix quality determines the overall generation quality.

**Goal**: To analyze the limitations of DAAs from a token-level MDP perspective and design a method to bridge the gap between training objectives and generation dynamics.

**Key Insight**: Empirical observations reveal that the entropy of prefix tokens is significantly higher than that of subsequent tokens, yet their log probabilities are diluted by the mean of numerous subsequent tokens, preventing DAAs from adequately focusing on prefix quality differences.

**Core Idea**: Truncate preference response pairs to equal length (the length of the shorter response) to implicitly constrain the DAA training objective to converge across all timesteps, thereby enhancing focus on prefix tokens.

## Method

### Overall Architecture

POET aims to bridge the "reward-generation gap" in Direct Alignment Algorithms (DAAs). While DPO/SimPO optimizes $r(x,y_w)\gg r(x,y_l)$ at the sequence level, they fail to regulate the prefix level—where early errors in autoregressive generation are magnified by exposure bias. The mechanism is minimalist: given a preference pair $(y_w,y_l)$, both are truncated to the length of the shorter response $k=\min(|y_w|,|y_l|)$. The intermediate product is a batch of equal-length preference pairs, and the output is the result of running standard DPO or SimPO on this data. The method does not modify the optimization objective, making it compatible with all DAAs.

### Key Designs

**1. Theoretical Foundation of Equal-length Sub-trajectory BT Model: Proving Truncation Does Not Shift the Optimal Policy**

The validity of POET depends on whether truncating responses biases the learned policy. The authors define an equal-length sub-trajectory BT model $p_k^*(y_{w,\leq k} \succeq y_{l,\leq k})$, which explicitly incorporates the optimal state-value function $V^*$ after the truncation point. Theorem 1 proves that the optimal policy derived from this equal-length sub-trajectory model is equivalent to the one derived from the original sequence-level BT model.

This provides a rigorous theoretical guarantee that truncation itself does not alter the optimal policy. Training on various truncation lengths provides more fine-grained reward signals for prefix tokens, addressing the insufficient prefix focus in DAAs.

**2. Empirical Validation of Prefix Quality Differences: Confirming Preference Consistency After Truncation**

Beyond theoretical guarantees, it is essential to confirm that using full-sequence preference labels to supervise truncated equal-length pairs is safe. The authors calculated the prefix quality difference $\Delta Q(k) = Q(y_{w,\leq k}) - Q(y_{l,\leq k})$ for various prefix lengths $k$ across 1,000 samples. Results show that quality differences emerge very early and grow with length, but with diminishing marginal returns. The preference ranking remains highly consistent (98.5%) after truncation.

Because preference rankings remain largely unchanged after truncation, training on equal-length pairs with full-sequence labels is safe. This empirical validation is a key support for POET's feasibility, as it prevents truncation from introducing significant label noise.

**3. POET Data Augmentation Strategy: Shortest-Length Truncation with No Extra Hyperparameters**

By setting $k = \min(|y_w|, |y_l|)$, one response in a pair always remains complete, while only the suffix of the longer response is removed. Since $k$ varies across samples in the dataset, training implicitly occurs over various truncation lengths, constraining the DAA to converge at all MDP timesteps and naturally pulling attention back to prefix quality.

This design offers three advantages: universal compatibility with any DAA; no introduction of additional hyperparameters; and minimized data noise since suffixes beyond the shorter response length typically have less impact on overall quality.

### Loss & Training

The optimization objectives of DPO/SimPO are not modified; only the input data changes. Training follows the hyperparameter settings of Meng et al. (2024). It supports both Base (starting from an SFT model) and Instruct (starting from an instruction-tuned model) settings.

## Key Experimental Results

### Main Results

**AlpacaEval 2 & Arena-Hard Instruction Following Evaluation**

| Method | Mistral-7B LC% | Llama-3-8B LC% | Llama-3-Inst LC% | Gemma-2-9B LC% |
|------|----------------|----------------|-------------------|----------------|
| DPO | 12.9 | 16.9 | 65.9 | 78.4 |
| DPO + POET | **24.7** (+11.8) | **28.4** (+11.5) | **70.4** (+4.5) | **79.7** (+1.3) |
| SimPO | 20.0 | 28.0 | 68.1 | 78.5 |
| SimPO + POET | **24.2** (+4.2) | **33.8** (+5.8) | **70.1** (+2.0) | **80.1** (+1.6) |

### Ablation Study

**Comparison of Truncation Strategies (Mistral-7B, AlpacaEval 2 LC%)**

| Truncation Strategy | Keep 25% | 50% | 75% | 100% |
|---------|---------|-----|-----|------|
| Original Length | 14.1 | 17.2 | 16.2 | 12.9 |
| POET Length | 23.5 | 24.9 | 26.7 | 24.7 |

### Key Findings

- POET consistently improves AlpacaEval 2 LC across all 8 settings (4 models × 2 DAAs), with a maximum Gain of +11.8 percentage points.
- Ablation experiments demonstrate that the equal-length truncation strategy (POET Len.) significantly outperforms truncation based on original lengths across all retention ratios, indicating that "equal length" itself is the key.
- POET does not increase the alignment tax—it maintains or slightly improves performance on downstream tasks in the HuggingFace Open Leaderboard.
- Comparison with token-level methods: POET significantly outperforms SamPO and D2PO; the latter even showed counterproductive results on SimPO.
- Safety alignment evaluations also show significant improvements in safety rates.

## Highlights & Insights

- Precise problem identification—The "reward-generation gap" captures a fundamental issue in DAAs: the mismatch between sequence-level optimization and autoregressive generation dynamics.
- Minimalist yet effective—Large improvements are achieved without modifying objectives or adding hyperparameters, suggesting the root cause lies in the data format rather than the optimization algorithm.
- The empirical analysis of prefix quality differences (Figure 2) is the most compelling part of the paper, validating the key assumption that "preference rankings remain largely unchanged after truncation" using an oracle model.

## Limitations & Future Work

- Equal-length truncation relies on the assumption that suffixes longer than the shorter response have minimal impact on quality, which may not hold for samples with extreme length asymmetry.
- The theoretical guarantee assumes the existence of an optimal state-value function $V^*$, which is uncomputable in practice.
- The method was only validated on DPO and SimPO and has not been extended to other DAAs like IPO or KTO.
- Although the preference consistency after truncation is high (91.4%-98.5%), some noise is still introduced.

## Related Work & Insights

- **vs SamPO (Lu et al., 2024)**: SamPO calculates rewards using random token subsets to mitigate length bias but does not focus on prefixes. POET focuses on prefixes and is effective for both DPO and SimPO.
- **vs D2PO (Shao et al., 2025)**: D2PO emphasizes prefixes using exponentially decaying weights, but the decay factor introduces extra hyperparameters and yields negative results on SimPO. POET has no hyperparameters and is consistently effective.
- **vs Token-level DPO (Rafailov et al., 2024b)**: Theoretically, token-level DPO can learn the optimal policy, but in practice, only sequence-level preference labels are available, making direct training difficult.

## Rating

- Novelty: ⭐⭐⭐⭐ Clear problem identification; minimalist method with sufficient theoretical motivation and empirical validation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 models × 2 DAAs × detailed ablations + comparison with token-level methods + safety evaluation.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous logical chain from theory to empirical evidence to methodology.
- Value: ⭐⭐⭐⭐⭐ Extremely high practical value as a plug-and-play, hyperparameter-free method compatible with all DAAs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MPO: Multilingual Safety Alignment via Reward Gap Optimization](../../ACL2025/llm_alignment/mpo_multilingual_safety_alignment.md)
- [\[ACL 2026\] BACH-V: Bridging Abstract and Concrete Human-Values in Large Language Models](bach-v_bridging_abstract_and_concrete_human-values_in_large_language_models.md)
- [\[ACL 2026\] RbtAct: Rebuttal as Supervision for Actionable Review Feedback Generation](rbtact_rebuttal_as_supervision_for_actionable_review_feedback_generation.md)
- [\[CVPR 2026\] Bridging Human Evaluation to Infrared and Visible Image Fusion](../../CVPR2026/llm_alignment/bridging_human_evaluation_to_infrared_and_visible_image_fusion.md)
- [\[ICLR 2026\] Is On-Policy Data always the Best Choice for Direct Preference Optimization-based LM Alignment?](../../ICLR2026/llm_alignment/is_on-policy_data_always_the_best_choice_for_direct_preference_optimization-base.md)

</div>

<!-- RELATED:END -->
