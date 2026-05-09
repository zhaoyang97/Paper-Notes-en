---
title: >-
  [Paper Note] Towards Bridging the Reward-Generation Gap in Direct Alignment Algorithms
description: >-
  [ACL 2026][LLM Alignment][Direct Alignment Algorithms] This paper identifies the "reward-generation gap" in Direct Alignment Algorithms (DAAs)—a mismatch between training objectives and autoregressive decoding dynamics—and proposes POET (Prefix-Oriented Equal-length Training), which truncates preference response pairs to the length of the shorter response to implicitly constrain token-level MDP convergence across all timesteps, achieving up to 11.8 percentage point improvement on AlpacaEval 2.
tags:
  - ACL 2026
  - LLM Alignment
  - Direct Alignment Algorithms
  - Prefix Importance
  - Equal-length Training
  - Reward-Generation Gap
  - DPO/SimPO
date: 2026-05-08
content_hash: bcb4208a8ccb0785
---

# Towards Bridging the Reward-Generation Gap in Direct Alignment Algorithms

**Conference**: ACL 2026
**arXiv**: [2506.09457](https://arxiv.org/abs/2506.09457)
**Code**: [GitHub](https://github.com/sustech-nlp/POET)
**Area**: LLM Alignment / Preference Optimization
**Keywords**: Direct Alignment Algorithms, Prefix Importance, Equal-length Training, Reward-Generation Gap, DPO/SimPO

## TL;DR

This paper identifies the "reward-generation gap" in Direct Alignment Algorithms (DAAs)—a mismatch between training objectives and autoregressive decoding dynamics—and proposes POET (Prefix-Oriented Equal-length Training), which truncates preference response pairs to the length of the shorter response to implicitly constrain token-level MDP convergence across all timesteps, achieving up to 11.8 percentage point improvement on AlpacaEval 2.

## Background & Motivation

**State of the Field**: DAAs such as DPO and SimPO have emerged as efficient alternatives to RLHF. These methods optimize directly on preference datasets via implicit reward functions, without requiring explicit reward models or reinforcement learning.

**Limitations of Prior Work**: (1) DAAs may reduce the absolute reward of preferred responses while widening the gap between preferred and dispreferred rewards; (2) higher preferred rewards and larger reward gaps do not necessarily lead to better generation quality; (3) the implicit rewards in DAAs assign equal weight to every token, ignoring the critical importance of prefix tokens in autoregressive generation.

**Root Cause**: DAAs optimize $r(x, y_w) \gg r(x, y_l)$ at the sequence level but cannot guarantee $r(x, y_{w,<k}) \gg r(x, y_{l,<k})$ at the prefix level. In autoregressive generation, errors in early tokens accumulate and amplify through exposure bias, making prefix quality determinative of overall generation quality.

**Paper Goals**: To analyze the limitations of DAAs from a token-level MDP perspective and develop a method to bridge the gap between training objectives and generation dynamics.

**Starting Point**: Empirical observations reveal that the entropy of prefix tokens is significantly higher than that of subsequent tokens, yet their log-probabilities are diluted by the mean over a large number of subsequent tokens, preventing DAAs from adequately attending to quality differences at the prefix level.

**Core Idea**: Truncating preference response pairs to equal length (the length of the shorter response) implicitly constrains the DAA training objective to converge at all timesteps, thereby enhancing focus on prefix token quality.

## Method

### Overall Architecture

POET is a simple data augmentation method: given a preference response pair $(y_w, y_l)$, both responses are truncated to the length of the shorter one, $k = \min(|y_w|, |y_l|)$, and standard DPO or SimPO training is then applied to the equal-length data. The method requires no modification to any optimization objective and is compatible with all DAAs.

### Key Designs

1. **Theoretical Foundation of the Equal-length Sub-trajectory BT Model**:

    - Function: Proves that optimization over equal-length sub-trajectories yields the same optimal policy as sequence-level optimization.
    - Mechanism: An equal-length sub-trajectory BT model $p_k^*(y_{w,\leq k} \succeq y_{l,\leq k})$ is defined, incorporating the optimal state value function $V^*$ beyond the truncation point. Theorem 1 proves that the optimal policy derived from this model is equivalent to that derived from the original sequence-level BT model.
    - Design Motivation: Provides a rigorous theoretical guarantee that truncation does not alter the optimal policy, while training across multiple truncation lengths supplies finer-grained reward signals for prefix tokens.

2. **Empirical Validation of Prefix Quality Differences**:

    - Function: Verifies that full-sequence preference labels remain valid after truncation.
    - Mechanism: On 1,000 samples, the prefix quality gap $\Delta Q(k) = Q(y_{w,\leq k}) - Q(y_{l,\leq k})$ is computed for varying prefix lengths $k$. Results show that the quality gap emerges at very early prefixes and grows with length but with diminishing marginal returns, indicating highly consistent preference ordering after truncation (98.5%).
    - Design Motivation: If preference ordering is preserved after truncation, training on equal-length pairs with full-sequence preference labels is safe—this empirical validation is a critical pillar supporting the feasibility of POET.

3. **POET Data Augmentation Strategy**:

    - Function: Naturally directs DAA attention toward prefix quality without introducing additional hyperparameters.
    - Mechanism: Setting $k = \min(|y_w|, |y_l|)$ means one response remains intact while only the suffix of the longer response is truncated. Since $k$ varies across samples in the dataset, training implicitly proceeds over multiple truncation lengths, constraining DAAs to converge at all MDP timesteps.
    - Design Motivation: Three key advantages—(1) universally compatible with any DAA; (2) requires no additional hyperparameters; (3) minimizes the risk of data noise (only suffixes are truncated, with minimal impact on overall quality).

### Loss & Training

The optimization objectives of DPO/SimPO are not modified; only the input data is changed. Hyperparameter settings follow Meng et al. (2024). Both Base (starting from an SFT model) and Instruct (starting from an instruction-tuned model) settings are supported.

## Key Experimental Results

### Main Results

**AlpacaEval 2 & Arena-Hard Instruction-Following Evaluation**

| Method | Mistral-7B LC% | Llama-3-8B LC% | Llama-3-Inst LC% | Gemma-2-9B LC% |
|--------|----------------|----------------|------------------|----------------|
| DPO | 12.9 | 16.9 | 65.9 | 78.4 |
| DPO + POET | **24.7** (+11.8) | **28.4** (+11.5) | **70.4** (+4.5) | **79.7** (+1.3) |
| SimPO | 20.0 | 28.0 | 68.1 | 78.5 |
| SimPO + POET | **24.2** (+4.2) | **33.8** (+5.8) | **70.1** (+2.0) | **80.1** (+1.6) |

### Ablation Study

**Truncation Strategy Comparison (Mistral-7B, AlpacaEval 2 LC%)**

| Truncation Strategy | Retain 25% | 50% | 75% | 100% |
|--------------------|-----------|-----|-----|------|
| Original Length | 14.1 | 17.2 | 16.2 | 12.9 |
| POET Length | 23.5 | 24.9 | 26.7 | 24.7 |

### Key Findings

- POET consistently improves AlpacaEval 2 LC across all 8 settings (4 models × 2 DAAs), with a maximum gain of +11.8 percentage points.
- Ablation experiments confirm that the equal-length truncation strategy (POET Len.) substantially outperforms truncation at original lengths under all retention ratios, demonstrating that equal length itself is the key factor.
- POET does not incur an alignment tax—performance on HuggingFace Open Leaderboard downstream tasks is maintained or marginally improved.
- Compared to token-level methods, POET significantly outperforms SamPO and D2PO; the latter even yields adverse effects when applied to SimPO.
- Safety alignment evaluations also show significant improvements in safety rates.

## Highlights & Insights

- The problem framing is precise—the "reward-generation gap" captures a fundamental issue in DAAs: the mismatch between sequence-level optimization objectives and autoregressive generation dynamics.
- The method is minimal yet highly effective—substantial gains are achieved merely by truncating data, without modifying objectives or adding hyperparameters, indicating that the root cause lies in the data format rather than the optimization algorithm.
- The empirical analysis of prefix quality differences (Figure 2) is the most compelling part of the paper—an oracle model is used to validate the key assumption that "preference ordering is nearly preserved after truncation."

## Limitations & Future Work

- Equal-length truncation relies on the assumption that suffixes beyond the shorter response's length have minimal impact on quality, which may not hold for samples with extremely asymmetric lengths.
- The theoretical guarantee assumes the existence of an optimal state value function $V^*$, which is not computable in practice.
- Validation is limited to DPO and SimPO; extension to other DAAs such as IPO and KTO has not been explored.
- Although post-truncation preference ordering is highly consistent (91.4%–98.5%), some degree of noise is still introduced.

## Related Work & Insights

- **vs SamPO (Lu et al., 2024)**: SamPO randomly samples token subsets to compute rewards to mitigate length bias but does not focus on prefixes; POET focuses on prefixes and is effective for both DPO and SimPO.
- **vs D2PO (Shao et al., 2025)**: D2PO uses exponential decay weights to emphasize prefixes, but the decay factor introduces an additional hyperparameter and produces adverse effects on SimPO; POET requires no hyperparameters and consistently improves performance.
- **vs Token-level DPO (Rafailov et al., 2024b)**: Token-level DPO can theoretically learn the optimal policy, but in practice only sequence-level preference labels are available, precluding direct training.

## Rating

- Novelty: ⭐⭐⭐⭐ — The problem is clearly identified; the method is simple but supported by thorough theoretical motivation and empirical validation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 4 models × 2 DAAs × detailed ablations + comparisons with token-level methods + safety evaluation.
- Writing Quality: ⭐⭐⭐⭐⭐ — The logical chain from theory to empirics to method is rigorous and coherent.
- Value: ⭐⭐⭐⭐⭐ — Plug-and-play, hyperparameter-free, and compatible with all DAAs; extremely high practical value.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Alignment through Meta-Weighted Online Sampling: Bridging the Gap between Data Generation and Preference Optimization](../../ICLR2026/llm_alignment/alignment_through_meta-weighted_online_sampling_bridging_the_gap_between_data_ge.md)
- [\[ACL 2026\] Reward Modeling for Scientific Writing Evaluation](reward_modeling_for_scientific_writing_evaluation.md)
- [\[ACL 2026\] Aligning Agents via Planning: A Benchmark for Trajectory-Level Reward Modeling](aligning_agents_via_planning_a_benchmark_for_trajectory-level_reward_modeling.md)
- [\[ACL 2026\] ConsistRM: Improving Generative Reward Models via Consistency-Aware Self-Training](consistrm_improving_generative_reward_models_via_consistency-aware_self-training.md)
- [\[ACL 2026\] Alignment Data Map for Efficient Preference Data Selection and Diagnosis](alignment_data_map_for_efficient_preference_data_selection_and_diagnosis.md)

<!-- RELATED:END -->
