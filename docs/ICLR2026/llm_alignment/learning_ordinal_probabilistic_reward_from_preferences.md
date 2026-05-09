---
title: >-
  [Paper Note] Learning Ordinal Probabilistic Reward from Preferences (OPRM)
description: >-
  [ICLR 2026][LLM Alignment][Ordinal Reward] This paper proposes the Ordinal Probabilistic Reward Model (OPRM), which discretizes response quality into ordinal grades from 1 to 9 and learns the full probability distribution over these grades. Combined with Region Flooding Tuning (RgFT), it enables data-efficient training. OPRM achieves 89.3% on RewardBench, outperforming existing reward models by 2.9%–7.4%, while also providing uncertainty estimation and annotation disagreement detection.
tags:
  - ICLR 2026
  - LLM Alignment
  - Ordinal Reward
  - Probability Distribution
  - Region Flooding Tuning
  - Reward Model
  - Uncertainty Estimation
date: 2026-05-08
content_hash: 0c99cc904c67ae01
---

# Learning Ordinal Probabilistic Reward from Preferences (OPRM)

**Conference**: ICLR 2026
**arXiv**: [2602.12660](https://arxiv.org/abs/2602.12660)

**Code**: [https://github.com/ritzz-ai/OPRM](https://github.com/ritzz-ai/OPRM)

**Area**: LLM Alignment / Reward Modeling
**Keywords**: Ordinal Reward, Probability Distribution, Region Flooding Tuning, Reward Model, Uncertainty Estimation

## TL;DR

This paper proposes the Ordinal Probabilistic Reward Model (OPRM), which discretizes response quality into ordinal grades from 1 to 9 and learns the full probability distribution over these grades. Combined with Region Flooding Tuning (RgFT), it enables data-efficient training. OPRM achieves 89.3% on RewardBench, outperforming existing reward models by 2.9%–7.4%, while also providing uncertainty estimation and annotation disagreement detection.

## Background & Motivation

**Background**: Reward models fall into two categories: generative reward models (GRM, requiring pointwise supervision at high annotation cost) and discriminative reward models (DRM, using pairwise preferences but producing uncalibrated scores).

**Limitations of Prior Work**: DRM scores lack probabilistic interpretation, making uncertainty assessment infeasible; GRM requires precise quality labels.

**Core Idea**: Ordinal discretization + full distribution = efficiency of DRM + interpretability of GRM.

## Method

### Overall Architecture

OPRM transforms reward modeling from "learning a scalar" to "learning a probability distribution." Given a prompt–response pair, the LLM backbone processes the input and extracts vocabulary probabilities at the final token position via softmax. The probabilities corresponding to digit tokens '1'–'9' are extracted and normalized to yield an ordinal quality distribution $p_\psi(s|x,y)$. At inference time, a scalar reward is obtained via the weighted average of this distribution.

### Key Designs

1. **Probabilistic Reward Model (PRM → OPRM)**: Quality score is modeled as a random variable $S$, and the model learns the conditional PDF $p_\psi(s|x,y)$.
The preference probability is defined as $P(y_c \succ y_r|x) = \int\int \mathbb{1}(s_c > s_r) p_\psi(s_c|x,y_c) p_\psi(s_r|x,y_r) ds_r ds_c$.
Since the continuous distribution admits no closed-form solution, discretizing into grades 1–9 yields a tractable closed-form summation.

2. **Relationship between Ordinal Probability and Bradley-Terry**: The paper proves that the BT model is a special case of OPRM—specifically, when the quality distribution degenerates to a fixed-shape Gumbel distribution. By learning the full distribution, OPRM supports multimodal preferences and uncertainty estimation.

3. **Gradient Dynamics**: $\partial J / \partial p_c(k) = P(s_r < k)$ and $\partial J / \partial p_r(k) = P(s_c > k)$. These gradients push probability mass of the chosen response toward higher grades and that of the rejected response toward lower grades, producing sustained contrastive optimization pressure.

4. **Region Flooding Tuning (RgFT)**: Quality-level annotations (good/normal/bad) are used to constrain the learned distribution to corresponding sub-regions. Naive interval constraints cause gradient vanishing; RgFT converts them into a triangular probability landscape that restores gradient incentives—simultaneously anchoring the correct region and maximizing the preference margin. RgFT supports semi-supervised training with a mixture of annotated and preference-only data.

### Implementation Details

- **No additional parameters**: The vocabulary probabilities from the existing LM head are reused directly, with no value head required.
- **Flexible input**: Supports both single-response scoring and multi-response comparison, suitable for Best-of-N (BoN) scenarios.
- **Uncertainty quantification**: The variance of the distribution serves as a confidence indicator—a wide distribution indicates ambiguous preference, while a peaked distribution indicates clear preference.

## Key Experimental Results

### Main Results (4 benchmarks, 10+ tasks)

| Model | RewardBench | RMB-Chat | RMB-Safety | RMB-Code | Overall* |
|-------|-------------|----------|------------|----------|----------|
| Skywork-Reward-V2 (8B) | 92.0 | 70.7 | 76.2 | 67.8 | 71.6 |
| ArmoRM (8B) | 89.5 | 72.1 | 74.8 | 65.3 | 70.7 |
| **OPRM-Qwen2.5-14B** | **89.3** | **76.4** | **78.5** | **70.1** | **73.8** |

### Ablation Study on RgFT

| Configuration | RMB Overall | Notes |
|---------------|-------------|-------|
| OPRM (w/o RgFT) | 71.2 | Baseline probabilistic reward |
| + RgFT (good/bad only) | 72.8 | Binary annotation |
| + RgFT (good/normal/bad) | **73.8** | Three-level annotation is optimal |
| + Full quality annotations | 73.5 | More annotations yield slight regression |

### Key Findings

- OPRM consistently outperforms BT and GRM baselines across three benchmarks beyond RewardBench, with an average improvement of 2.9%–7.4%.
- RgFT effectively calibrates the distribution using only a small proportion of quality-annotated data (20% of the dataset), demonstrating high data efficiency.
- Multimodal distributions can detect annotation disagreement: inconsistent preference pairs yield bimodal distributions, which can be used for data quality filtering.
- OPRM is more sensitive to subtle margin differences—its advantage is largest on difficult samples where the chosen response is only marginally better than the rejected one.

## Highlights & Insights

- **Unifying DRM and GRM advantages**: No additional value head is required (vs. DRM), and no CoT critique generation is needed (vs. GRM); the distribution is obtained directly from the LM head.
- Ordinal discretization preserves the ordering of quality while avoiding the cost of precise pointwise annotations.
- The "flooding" design of RgFT is elegant—it softens hard interval constraints into a gradient-friendly triangular landscape.
- Uncertainty estimation enables risk-aware selection in BoN sampling: responses with high mean and low variance are preferred.

## Limitations & Future Work

- The choice of granularity for the 1–9 grading scale lacks theoretical justification; excessively coarse or fine partitioning may degrade performance.
- Integration with RLHF/DPO training pipelines remains unexplored—how OPRM's distributional rewards can be used in PPO requires additional design work.
- The approach relies on the LLM's intrinsic ordinal understanding of digit tokens, which may be insufficient in smaller models.
- In the semi-supervised RgFT setting, sensitivity analysis of the proportion of annotated data on overall performance is not thoroughly characterized.

## Related Work & Insights

- **vs. Skywork-Reward-V2**: Skywork focuses on data curation, while OPRM focuses on model architecture innovation—the two approaches are complementary and can be combined.
- **vs. Bradley-Terry**: BT is a special case of OPRM under the Gumbel distribution assumption; OPRM relaxes this constraint by learning distributions with greater degrees of freedom.
- **vs. CLoud/Critic-RM**: GRM requires time-consuming CoT critique generation, whereas OPRM obtains the full distribution in a single forward pass.
- **vs. Ordinal Regression Literature (SORD/ALDL)**: This work transfers deep ordinal regression ideas into preference learning, representing productive cross-domain knowledge transfer.

## Rating

- Novelty: ⭐⭐⭐⭐ The unified perspective of ordinal probabilistic reward is novel; the proof of BT as a special case is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four benchmarks + extensive ablations + distribution visualizations.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are complete and motivation is clearly articulated.
- Value: ⭐⭐⭐⭐ Establishes a new paradigm for reward modeling; distributional outputs open up new application scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Capturing Individual Human Preferences with Reward Features](../../NeurIPS2025/llm_alignment/capturing_individual_human_preferences_with_reward_features.md)
- [\[ICLR 2026\] Skywork-Reward-V2: Scaling Preference Data Curation via Human-AI Synergy](skywork-reward-v2_scaling_preference_data_curation_via_human-ai_synergy.md)
- [\[ICLR 2026\] Swap-guided Preference Learning for Personalized RLHF (SPL)](swap-guided_preference_learning_for_personalized_reinforcement_learning_from_hum.md)
- [\[ICLR 2026\] AlphaSteer: Learning Refusal Steering with Principled Null-Space Constraint](alphasteer_learning_refusal_steering_with_principled_null-space_constraint.md)
- [\[NeurIPS 2025\] ResponseRank: Data-Efficient Reward Modeling through Preference Strength Learning](../../NeurIPS2025/llm_alignment/responserank_data-efficient_reward_modeling_through_preference_strength_learning.md)

</div>

<!-- RELATED:END -->
