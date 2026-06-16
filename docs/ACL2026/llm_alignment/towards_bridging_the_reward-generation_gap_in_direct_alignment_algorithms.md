---
title: >-
  [Paper Note] Towards Bridging the Reward-Generation Gap in Direct Alignment Algorithms
description: >-
  [ACL 2026][Alignment & RLHF][DPO/SimPO] This paper identifies the "reward-generation gap" in Direct Alignment Algorithms (DAAs)—a mismatch between training objectives and autoregressive decoding dynamics. It proposes POET (Prefix-Oriented Equal-length Training), which implicitly constrains the token-level MDP to converge across all timesteps by truncating pr
tags:
  - ACL 2026
  - Alignment & RLHF
  - DPO/SimPO
date: 2026-05-08
content_hash: fc6503755b520c3b
---
# Towards Bridging the Reward-Generation Gap in Direct Alignment Algorithms

**Conference**: ACL 2026 Findings  
**arXiv**: [2506.09457](https://arxiv.org/abs/2506.09457)  
**Code**: [GitHub](https://github.com/sustech-nlp/POET)  
**Area**: LLM Alignment / Preference Optimization  
**Keywords**: Direct Alignment Algorithms, Prefix Importance, Equal-length Training, Reward-Generation Gap, DPO/SimPO

## TL;DR

This paper identifies the "reward-generation gap" in Direct Alignment Algorithms (DAAs)—a mismatch between training objectives and autoregressive decoding dynamics. It proposes POET (Prefix-Oriented Equal-length Training), which implicitly constrains the token-level MDP to converge across all timesteps by truncating preference pairs to the length of the shorter response, achieving up to an 11.8 percentage point improvement on AlpacaEval 2.

## Background & Motivation

**Background**: Direct Alignment Algorithms (DAAs) such as DPO and SimPO have become efficient alternatives to RLHF. DAAs optimize directly on preference datasets via implicit reward functions, bypassing the need for explicit reward models and reinforcement learning.

**Limitations of Prior Work**: (1) DAAs may increase the preference-dispreference reward gap while simultaneously decreasing the absolute reward of the preferred response; (2) Higher preference rewards and larger reward gaps do not necessarily lead to better generation quality; (3) The implicit rewards in DAAs assign equal weight to every token, ignoring the critical importance of prefix tokens in autoregressive generation.

**Key Challenge**: DAAs optimize at the sequence level $r(x, y_w) \gg r(x, y_l)$, but they cannot guarantee that $r(x, y_{w,<k}) \gg r(x, y_{l,<k})$ at the prefix level. In autoregressive generation, errors in early tokens accumulate and amplify through exposure bias, meaning prefix quality determines overall generation quality.

**Goal**: To analyze the limitations of DAAs from a token-level MDP perspective and design methods to bridge the gap between training objectives and generation dynamics.

**Key Insight**: Empirical observations show that the entropy of prefix tokens is significantly higher than subsequent tokens, yet their log-probabilities are diluted by the mean of numerous subsequent tokens, causing DAAs to fail to focus sufficiently on prefix quality differences.

**Core Idea**: Truncate preference response pairs to equal length (taking the length of the shorter response) to implicitly constrain the DAA training objective to converge across all timesteps, thereby enhancing focus on prefix tokens.

## Method

### Overall Architecture

POET aims to bridge the "reward-generation gap" in direct alignment algorithms (DAAs). While DPO/SimPO optimize $r(x, y_w) \gg r(x, y_l)$ at the sequence level, they fail to regulate the prefix level—where early errors in autoregressive generation are amplified by exposure bias. The mechanism is minimalist: given a pair of preference responses $(y_w, y_l)$, both are truncated to the length of the shorter one $k = \min(|y_w|, |y_l|)$. The intermediate product is a batch of equal-length preference pairs, and the final output is obtained by running standard DPO or SimPO on this equal-length data. This method does not modify any optimization objectives and is therefore compatible with all DAAs.

### Key Designs

**1. Theoretical Foundation of Equal-length Sub-trajectory BT Model: Proving Truncation Preserves Optimal Policy**

The validity of POET depends on whether truncating responses biases the learned policy. The authors define an equal-length sub-trajectory BT model $p_k^*(y_{w,\leq k} \succeq y_{l,\leq k})$, which explicitly includes the optimal state-value function $V^*$ after the truncation point. Theorem 1 proves that the optimal policy derived from this equal-length sub-trajectory model is strictly equivalent to the optimal policy derived from the original sequence-level BT model. 

This provides a rigorous theoretical guarantee: truncation itself does not alter the optimal policy. Furthermore, training on multiple truncation lengths provides finer-grained reward signals for prefix tokens, addressing the insufficient focus on prefixes in standard DAAs.

**2. Empirical Verification of Prefix Quality Differences: Confirming Preference Ranking Consistency After Truncation**

Beyond theoretical guarantees, it is necessary to check if using full-sequence preference labels to supervise truncated equal-length pairs is safe. On 1000 samples, the authors calculated the prefix quality difference between preferred and dispreferred responses $\Delta Q(k) = Q(y_{w,\leq k}) - Q(y_{l,\leq k})$ for different prefix lengths $k$. Results show that the quality gap appears very early and grows with length but with diminishing marginal returns, and preference rankings remain highly consistent (98.5%) after truncation.

Because preference rankings barely change after truncation, supervised training of equal-length pairs with full-sequence labels is safe—this empirical verification is a key support for the feasibility of POET, as truncation would otherwise introduce significant label noise.

**3. POET Data Augmentation Strategy: Truncating to Shortest Length with No Extra Hyperparameters**

Specifically, setting $k = \min(|y_w|, |y_l|)$ ensures that one response in each pair remains complete, only truncating the suffix of the longer one. Since $k$ varies across samples in the dataset, training implicitly occurs over various truncation lengths, thereby constraining the DAAs to converge at all MDP timesteps and naturally shifting attention back to prefix quality.

This design offers three advantages: universal compatibility with any DAA; no introduction of additional hyperparameters; and minimal risk of data noise as only suffixes are truncated—suffixes beyond the length of the shorter response typically have minimal impact on overall quality.

### Loss & Training

The optimization objectives of DPO/SimPO are not modified; only the input data is changed. Training follows the hyperparameter settings of Meng et al. (2024). It supports both Base (starting from SFT models) and Instruct (starting from instruction-tuned models) settings.

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

- POET consistently improves AlpacaEval 2 LC across all 8 settings (4 models × 2 DAAs), with gains up to +11.8 percentage points.
- Ablation studies prove that the equal-length truncation strategy (POET Len.) significantly outperforms truncation based on original lengths across all retention ratios, indicating that equal length itself is the key factor.
- POET does not introduce an alignment tax—performance on HuggingFace Open Leaderboard downstream tasks is maintained or slightly improved.
- Comparison with token-level methods: POET significantly outperforms SamPO and D2PO; the latter even proved counterproductive when applied to SimPO.
- Safety alignment evaluations also show a significant increase in safety rates.

## Highlights & Insights

- Precise problem identification—The "reward-generation gap" captures a fundamental issue in DAAs: the mismatch between sequence-level optimization objectives and autoregressive generation dynamics.
- Minimalist yet effective—Substantial improvements are achieved without modifying objectives or adding hyperparameters, just by truncating data, suggesting the root cause lies in the data format rather than the optimization algorithms.
- The empirical analysis of prefix quality differences (Figure 2) is the most compelling part of the paper, using an oracle model to validate the key assumption that "preference ranking remains almost unchanged after truncation."

## Limitations & Future Work

- Equal-length truncation relies on the assumption that "suffixes longer than the shorter response have little impact on quality," which may not hold for specimens with extreme length asymmetry.
- Theoretical guarantees assume the existence of an optimal state-value function $V^*$, which cannot be computed in practice.
- Validation was only performed on DPO and SimPO; it has not been extended to other DAAs like IPO or KTO.
- Although preference ranking consistency is high after truncation (91.4%-98.5%), some noise is still introduced.

## Related Work & Insights

- **vs SamPO (Lu et al., 2024)**: SamPO computes rewards on random subsets of tokens to mitigate length bias but does not focus on prefixes; POET focuses on prefixes and is effective for both DPO and SimPO.
- **vs D2PO (Shao et al., 2025)**: D2PO uses exponentially decaying weights to emphasize prefixes, but the decay factor introduces extra hyperparameters and yields negative results with SimPO; POET has no hyperparameters and is consistently effective.
- **vs Token-level DPO (Rafailov et al., 2024b)**: Theoretically, token-level DPO can learn the optimal policy, but in practice, only sequence-level preference labels are available, preventing direct training.

## Rating

- Novelty: ⭐⭐⭐⭐ Clear problem identification; the method is simple but supported by sufficient theoretical motivation and empirical validation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across 4 models × 2 DAAs + detailed ablations + comparison with token-level methods + safety evaluation.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous logical chain from theory to empirical analysis to methodology.
- Value: ⭐⭐⭐⭐⭐ Extremely high practical value as it is plug-and-play, hyperparameter-free, and compatible with all DAAs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MPO: Multilingual Safety Alignment via Reward Gap Optimization](../../ACL2025/llm_alignment/mpo_multilingual_safety_alignment.md)
- [\[ACL 2026\] BACH-V: Bridging Abstract and Concrete Human-Values in Large Language Models](bach-v_bridging_abstract_and_concrete_human-values_in_large_language_models.md)
- [\[ACL 2026\] RbtAct: Rebuttal as Supervision for Actionable Review Feedback Generation](rbtact_rebuttal_as_supervision_for_actionable_review_feedback_generation.md)
- [\[ICLR 2026\] Is On-Policy Data always the Best Choice for Direct Preference Optimization-based LM Alignment?](../../ICLR2026/llm_alignment/is_on-policy_data_always_the_best_choice_for_direct_preference_optimization-base.md)
- [\[ACL 2025\] DiffPO: Diffusion Alignment with Direct Preference Optimization](../../ACL2025/llm_alignment/diffpo_diffusion_alignment.md)

</div>

<!-- RELATED:END -->
