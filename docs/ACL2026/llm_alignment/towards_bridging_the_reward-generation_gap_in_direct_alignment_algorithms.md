---
title: >-
  [Paper Note] Towards Bridging the Reward-Generation Gap in Direct Alignment Algorithms
description: >-
  [ACL 2026][LLM Alignment][Direct Alignment Algorithms] This paper identifies the "reward-generation gap" in Direct Alignment Algorithms (DAAs)—a mismatch between training objectives and autoregressive decoding dynamics—and proposes POET (Prefix-Oriented Equal-length Training), which truncates preference response pairs to the length of the shorter response to implicitly constrain token-level MDP convergence across all timesteps, achieving up to an 11.8 percentage point improvement on AlpacaEval 2.
tags:
  - ACL 2026
  - LLM Alignment
  - Direct Alignment Algorithms
  - Prefix Importance
  - Equal-length Training
  - Reward-Generation Gap
  - DPO/SimPO
date: 2026-05-08
content_hash: dc70b1c003c448b0
---

# Towards Bridging the Reward-Generation Gap in Direct Alignment Algorithms

**Conference**: ACL 2026
**arXiv**: [2506.09457](https://arxiv.org/abs/2506.09457)
**Code**: [GitHub](https://github.com/sustech-nlp/POET)
**Area**: LLM Alignment / Preference Optimization
**Keywords**: Direct Alignment Algorithms, Prefix Importance, Equal-length Training, Reward-Generation Gap, DPO/SimPO

## TL;DR

This paper identifies the "reward-generation gap" in Direct Alignment Algorithms (DAAs)—a mismatch between training objectives and autoregressive decoding dynamics—and proposes POET (Prefix-Oriented Equal-length Training), which truncates preference response pairs to the length of the shorter response to implicitly constrain token-level MDP convergence across all timesteps, achieving up to an 11.8 percentage point improvement on AlpacaEval 2.

## Background & Motivation

**State of the Field**: DAAs such as DPO and SimPO have emerged as efficient alternatives to RLHF. They optimize directly on preference datasets via implicit reward functions, eliminating the need for explicit reward models and reinforcement learning.

**Limitations of Prior Work**: (1) DAAs may increase the reward gap between preferred and dispreferred responses while reducing the absolute reward of the preferred response; (2) higher preferred reward and larger reward margins do not necessarily yield better generation quality; (3) the implicit rewards in DAAs assign equal weight to every token, neglecting the critical importance of prefix tokens in autoregressive generation.

**Root Cause**: DAAs optimize $r(x, y_w) \gg r(x, y_l)$ at the sequence level but cannot guarantee $r(x, y_{w,<k}) \gg r(x, y_{l,<k})$ at the prefix level. In autoregressive generation, errors in early tokens accumulate and amplify through exposure bias, meaning prefix quality fundamentally determines overall generation quality.

**Paper Goals**: To analyze the limitations of DAAs from a token-level MDP perspective and design a method that bridges the gap between training objectives and generation dynamics.

**Starting Point**: Empirical observations reveal that the entropy of prefix tokens is significantly higher than that of subsequent tokens, yet their log-probabilities are diluted by the mean over a large number of later tokens, preventing DAAs from adequately attending to prefix-level quality differences.

**Core Idea**: Truncating preference response pairs to equal length (i.e., the length of the shorter response) implicitly constrains the DAA training objective to converge at all timesteps, thereby enhancing focus on prefix token quality.

## Method

### Overall Architecture

POET is a simple data preprocessing method: given a preference response pair $(y_w, y_l)$, both responses are truncated to the length of the shorter one, $k = \min(|y_w|, |y_l|)$, and standard DPO or SimPO training is applied on the resulting equal-length data. The method requires no modification to any optimization objective and is compatible with all DAAs.

### Key Designs

1. **Theoretical Grounding for Equal-length Sub-trajectory BT Model**:

   - Function: Proves that optimizing on equal-length sub-trajectories yields the same optimal policy as sequence-level optimization.
   - Mechanism: An equal-length sub-trajectory BT model $p_k^*(y_{w,\leq k} \succeq y_{l,\leq k})$ is defined, incorporating the optimal state-value function $V^*$ beyond the truncation point. Theorem 1 establishes that the optimal policy derived from this model is equivalent to that derived from the original sequence-level BT model.
   - Design Motivation: This provides a rigorous theoretical guarantee that truncation does not alter the optimal policy, while training across varied truncation lengths supplies a finer-grained reward signal for prefix tokens.

2. **Empirical Validation of Prefix Quality Differences**:

   - Function: Verifies that full-sequence preference labels remain valid after truncation.
   - Mechanism: On 1,000 samples, the prefix quality difference $\Delta Q(k) = Q(y_{w,\leq k}) - Q(y_{l,\leq k})$ is computed for varying prefix lengths $k$. Results show that the quality gap emerges very early and grows with length, but with diminishing marginal returns, indicating high consistency (98.5%) in preference ordering after truncation.
   - Design Motivation: If preference ordering is preserved after truncation, training on equal-length pairs with full-sequence preference labels is safe. This empirical validation is a critical foundation for the feasibility of POET.

3. **POET Data Augmentation Strategy**:

   - Function: Naturally focuses DAAs on prefix quality without introducing additional hyperparameters.
   - Mechanism: Setting $k = \min(|y_w|, |y_l|)$ preserves one response in full while truncating only the suffix of the longer one. Since $k$ varies across samples, training implicitly operates over diverse truncation lengths, constraining DAAs to converge across all MDP timesteps.
   - Design Motivation: Three key advantages—(1) universally compatible with any DAA; (2) requires no additional hyperparameters; (3) minimizes the risk of data noise, as only suffixes are removed with minimal impact on overall quality.

### Loss & Training

The optimization objectives of DPO/SimPO are not modified; only the input data changes. Hyperparameter settings follow Meng et al. (2024). Both Base (starting from an SFT model) and Instruct (starting from an instruction-tuned model) configurations are supported.

## Key Experimental Results

### Main Results

**AlpacaEval 2 & Arena-Hard Instruction-Following Evaluation**

| Method | Mistral-7B LC% | Llama-3-8B LC% | Llama-3-Inst LC% | Gemma-2-9B LC% |
|--------|----------------|----------------|-------------------|----------------|
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
- Ablation studies confirm that the equal-length truncation strategy (POET Len.) substantially outperforms truncation at original lengths across all retention ratios, demonstrating that equal-length alignment itself is the critical factor.
- POET does not incur alignment tax—performance on HuggingFace Open Leaderboard downstream tasks is maintained or slightly improved.
- Compared to token-level methods, POET substantially outperforms SamPO and D2PO; the latter even degrades performance when applied to SimPO.
- Safety alignment evaluations also show significant improvements in safety rates.

## Highlights & Insights

- The problem identification is precise—the "reward-generation gap" captures a fundamental issue in DAAs: the mismatch between sequence-level optimization objectives and autoregressive generation dynamics.
- The method is remarkably simple yet effective—achieving substantial gains by only modifying input data, without changing the objective or introducing hyperparameters, suggesting that the root cause lies in data formulation rather than the optimization algorithm.
- The empirical analysis of prefix quality differences (Figure 2) is the most compelling part of the paper—validating via an oracle model the key assumption that preference ordering is almost entirely preserved after truncation.

## Limitations & Future Work

- Equal-length truncation relies on the assumption that suffixes beyond the shorter response length have minimal impact on quality, which may not hold for samples with extremely asymmetric lengths.
- The theoretical guarantee assumes the existence of an optimal state-value function $V^*$, which is not computable in practice.
- Validation is limited to DPO and SimPO; extension to other DAAs such as IPO and KTO has not been explored.
- Although post-truncation preference ordering consistency is high (91.4%–98.5%), a non-trivial level of noise is still introduced.

## Related Work & Insights

- **vs. SamPO (Lu et al., 2024)**: SamPO randomly subsamples tokens to compute rewards in order to mitigate length bias, but does not focus on prefixes; POET focuses on prefixes and is effective on both DPO and SimPO.
- **vs. D2PO (Shao et al., 2025)**: D2PO applies exponentially decaying weights to emphasize prefixes, but the decay factor introduces an additional hyperparameter and degrades SimPO performance; POET requires no hyperparameters and yields consistent gains.
- **vs. Token-level DPO (Rafailov et al., 2024b)**: Token-level DPO can theoretically learn the optimal policy, but in practice only sequence-level preference labels are available, making direct training infeasible.

## Rating

- Novelty: ⭐⭐⭐⭐ — Problem identification is clear; the method is simple yet supported by sufficient theoretical motivation and empirical validation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 4 models × 2 DAAs × detailed ablations + comparison with token-level methods + safety evaluation.
- Writing Quality: ⭐⭐⭐⭐⭐ — The logical chain from theory to empirical analysis to method is rigorous and well-structured.
- Value: ⭐⭐⭐⭐⭐ — Plug-and-play, hyperparameter-free, and compatible with all DAAs; extremely high practical value.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Alignment through Meta-Weighted Online Sampling: Bridging the Gap between Data Generation and Preference Optimization](../../ICLR2026/llm_alignment/alignment_through_meta-weighted_online_sampling_bridging_the_gap_between_data_ge.md)
- [\[ACL 2026\] Reward Modeling for Scientific Writing Evaluation](reward_modeling_for_scientific_writing_evaluation.md)
- [\[ACL 2026\] Aligning Agents via Planning: A Benchmark for Trajectory-Level Reward Modeling](aligning_agents_via_planning_a_benchmark_for_trajectory-level_reward_modeling.md)
- [\[ACL 2026\] ConsistRM: Improving Generative Reward Models via Consistency-Aware Self-Training](consistrm_improving_generative_reward_models_via_consistency-aware_self-training.md)
- [\[ACL 2026\] Alignment Data Map for Efficient Preference Data Selection and Diagnosis](alignment_data_map_for_efficient_preference_data_selection_and_diagnosis.md)

<!-- RELATED:END -->
