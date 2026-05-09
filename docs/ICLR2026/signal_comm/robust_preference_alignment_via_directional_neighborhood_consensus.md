---
title: >-
  [Paper Note] Robust Preference Alignment via Directional Neighborhood Consensus
description: >-
  [ICLR 2026][Signal & Communication][preference alignment] This paper proposes Robust Preference Selection (RPS), a training-free inference-time method for improving preference alignment robustness. By sampling multiple candidate directions from the local neighborhood of a target preference and generating responses accordingly, then selecting the best response according to the original target preference, RPS achieves up to 69% win rate over baselines on OOD preferences.
tags:
  - ICLR 2026
  - "Signal & Communication"
  - preference alignment
  - robustness
  - inference-time adjustment
  - directional neighborhood consensus
  - out-of-distribution preferences
date: 2026-05-08
content_hash: 3ad5db4efeeeea47
---

# Robust Preference Alignment via Directional Neighborhood Consensus

**Conference**: ICLR 2026
**arXiv**: [2510.20498](https://arxiv.org/abs/2510.20498)
**Code**: [rcmao/robust-preference-alignment](https://github.com/rcmao/robust-preference-alignment)
**Area**: Signal & Communication
**Keywords**: preference alignment, robustness, inference-time adjustment, directional neighborhood consensus, out-of-distribution preferences

## TL;DR
This paper proposes Robust Preference Selection (RPS), a training-free inference-time method for improving preference alignment robustness. By sampling multiple candidate directions from the local neighborhood of a target preference and generating responses accordingly, then selecting the best response according to the original target preference, RPS achieves up to 69% win rate over baselines on OOD preferences.

## Background & Motivation

Aligning large language models (LLMs) with human preferences is essential for building reliable and controllable AI systems. User preferences can be modeled as directional vectors in a multi-dimensional space, where each dimension represents a trade-off between different attributes (e.g., helpfulness vs. verbosity). Existing preference alignment methods (RLHF, DPO, DPA, etc.) typically optimize toward the dominant "average" preference present in training data.

**Core Pain Point**: Training data covers a limited range of preferences, concentrated in a narrow region — the *Preference Coverage Gap*. When a user's true preference deviates from the concentrated tendency of the training distribution (i.e., OOD preferences), model performance degrades unpredictably. This represents a fundamental out-of-distribution (OOD) challenge.

**Limitations of Prior Work**:
1. Training-time methods (e.g., data augmentation, distributionally robust optimization/DRO) require expensive retraining and may still fail to generalize across the full preference spectrum.
2. Inference-time methods (e.g., token-level guidance, activation steering) require direct manipulation of model internal states or auxiliary models.

**Key Insight**: The authors identify a key insight — **rather than forcing the model to generate a response directly from a specific, rare preference direction (which is inherently fragile), one should instead explore the local neighborhood of that preference, generate a candidate pool from more reliable neighboring directions, and then select the response that best satisfies the original target preference.** This paradigm shifts from "direct generation" to "neighborhood consensus selection."

## Method

### Overall Architecture
RPS is a three-stage inference-time pipeline:
- **Input**: user prompt $x$, target preference vector $\mathbf{v}_{target}$, neighborhood size $k$, angular threshold $\theta_{max}$
- **Output**: optimal response $y^*$

The three stages proceed as: Neighborhood Construction → Multi-Directional Generation → Consensus Selection.

### Key Designs

1. **Preference Space Formalization**: User preferences are modeled as normalized direction vectors on the unit circle $\mathbf{v} = (\cos\theta, \sin\theta)$, where $\theta$ parameterizes the trade-off between helpfulness and verbosity. The reward model maps prompt-response pairs to reward vectors $\mathbf{r}(x,y) = (r_h(x,y), r_v(x,y))$. The objective is to maximize the projected reward $\mathbf{v}_{target}^T \mathbf{r}(x,y)$. The paper uses RewardModel-Mistral-7B-for-DPA-v1 as the reward model.

2. **Formal Definition of the Preference Coverage Gap**: The paper defines the user preference space $\mathcal{V}_{user}$ (the full preference spectrum) and the training preference set $\mathcal{V}_{train}$ (the subset of preferences used during training); their difference constitutes the preference coverage gap. When $\mathbf{v}_{target}$ falls within this gap, model performance becomes unreliable.

3. **Phase 1: Neighborhood Construction**: Rather than directly using the potentially fragile $\mathbf{v}_{target}$, the method samples $k$ neighboring preference directions within angular threshold $\theta_{max}$, forming a local neighborhood $\mathcal{N}_k$. These neighboring directions are closer to the training distribution, yielding more reliable model behavior. Experiments use $\theta_{max} = 30°$.

4. **Phase 2: Multi-Directional Generation**: For each preference vector $\mathbf{v}_i$ in the neighborhood, the LLM generates an independent response $y_i$. Each response reflects a slightly different attribute trade-off, yet all are generated from preference regions where the model performs reliably. This produces a diverse pool of high-quality candidates.

5. **Phase 3: Consensus Selection**: All $k$ candidates are evaluated using the **original target preference** $\mathbf{v}_{target}$, and the response maximizing the projected reward $s_i = \mathbf{v}_{target}^T \mathbf{r}(x,y_i)$ is selected as the final output. The key distinction is: generation uses neighborhood directions (more reliable), while evaluation uses the target direction (faithful to user intent).

6. **Theoretical Guarantee (Theorem 1)**: Under the OOD performance degradation assumption (Assumption 1), the paper proves that the RPS candidate pool stochastically first-order dominates the baseline (repeated sampling from the target direction), such that $\mathbb{E}[\max(S_{RPS})] > \mathbb{E}[\max(S_{Baseline})]$. A corollary further establishes that robustness gains increase with neighborhood size $k$ and the quality gap.

### Loss & Training
RPS is a fully training-free inference-time method — **no training or fine-tuning of any kind is involved**. It is a post-hoc adjustment technique applicable to any existing preference-aligned model.

## Key Experimental Results

### Main Results
A 3×3 experimental matrix covering 3 models × 3 datasets; all combinations exceed the 50% baseline win rate.

| Model | Dataset | RPS Win Rate | Notes |
|-------|---------|-------------|-------|
| DPA (DPA-v1-Mistral-7B) | UltraFeedback | ~60% | Strongest OOD gain |
| DPA | HelpSteer | ~60% | Consistent advantage |
| DPA | HelpSteer2 | ~61% | Consistent advantage |
| DPO (Zephyr-7B-Beta) | UltraFeedback | ~52% | Stable but modest |
| DPO | HelpSteer | ~53% | DPO already has inherent robustness |
| DPO | HelpSteer2 | ~54% | Modest improvement |
| SFT (Mistral-7B-Instruct-v0.2) | UltraFeedback | 52% | Lowest improvement |
| SFT | HelpSteer | ~57% | Notable improvement |
| SFT | HelpSteer2 | 67.3% | Largest improvement — SFT benefits most |

### Directional Robustness (Preference Angle vs. Win Rate)

| Preference Direction | DPA/UltraFeedback | DPA/HelpSteer | SFT/HelpSteer2 |
|---------------------|-------------------|---------------|-----------------|
| v1 (10°) | 55.1% | 56.1% | 52.1% |
| v3 (20°) | 53.4% | 58.0% | 58.9% |
| v5 (30°) | 59.3% | 60.2% | 66.7% |
| v7 (40°) | 64.9% | 62.8% | 83.2% |
| v8 (45°) | 69.1% | 64.3% | 94.3% |

### Ablation Study

| Configuration | Key Metric | Notes |
|--------------|-----------|-------|
| k=5 (neighborhood size) | Reference setting | Strictly compute-equivalent to baseline |
| θ_max=30° (angular threshold) | Optimal balance | Too small → insufficient diversity; too large → deviation from target |

### Key Findings
- RPS exceeds the 50% baseline win rate across all 9 model-dataset pairs, demonstrating that neighborhood consensus is a broadly effective post-hoc enhancement.
- **RPS gains amplify significantly as the preference angle increases (more OOD)**: DPA reaches 69.1% at 45°; SFT on HelpSteer2 reaches 94.3% at 45°.
- Different training paradigms benefit to varying degrees: SFT benefits most (lacking explicit preference training), DPO is relatively robust (inherent robustness), and DPA improves most significantly in OOD directions.
- Qualitative analysis shows that RPS-generated responses are more detailed, more targeted, and better aligned with user intent.

## Highlights & Insights
- **Paradigm Shift**: A clear and compelling inference-time transition from "direct generation" to "neighborhood sampling + selection."
- **Solid Theory**: The stochastic first-order dominance framework elegantly proves the method's superiority.
- **Zero-Cost Deployment**: A purely inference-time method — no retraining required, model-agnostic, and plug-and-play.
- **Compute Parity**: RPS and the baseline generate the same number of candidates; the only difference is the source of preference directions used for generation.
- **Deep Insight**: The paper reveals the OOD problem in preference alignment and quantifies the impact of the preference coverage gap.
- **SFT Models Benefit Most**: This suggests RPS can serve as an effective inference-time preference steering mechanism, substituting for expensive RLHF training.

## Limitations & Future Work
- The preference space is limited to 2 dimensions (helpfulness and verbosity); generalization to higher-dimensional preference spaces remains unverified.
- A reward model is required to evaluate candidates, introducing additional inference overhead.
- $k=5$ implies a 5× inference cost (generating 5 responses), which may be unacceptable in latency-sensitive scenarios.
- The selection of neighborhood size $k$ and angular threshold $\theta_{max}$ relies on prior knowledge, with no adaptive adjustment mechanism.
- The theoretical framework depends on Assumption 1 (model performance is better for nearby directions), which, while reasonable, may not hold in extreme OOD cases.
- No direct comparison with other inference-time alignment methods (e.g., activation steering, ARGS) is provided.
- The use of GPT-4o-mini as an evaluation judge introduces potential bias from model-based assessment.

## Related Work & Insights
- **DPA (Directional Preference Alignment)**: This paper builds upon DPA's multi-dimensional preference space formalization.
- **Self-Consistency (Wang et al., 2022)**: Improves reliability by sampling multiple reasoning paths and aggregating consensus — conceptually analogous to RPS's neighborhood consensus approach.
- **DRO (Distributionally Robust Optimization)**: A training-time robustness method; RPS provides a complementary inference-time alternative.
- **Best-of-N Sampling**: RPS can be viewed as a directional generalization of Best-of-N — rather than repeatedly sampling from the same direction, it samples once from each of several different directions.
- Inspiration: The neighborhood consensus idea may generalize to OOD condition handling in other conditional generation tasks (e.g., image style control, music generation).

## Rating
- Novelty: ⭐⭐⭐⭐ (Clear contribution, though essentially a clever generalization of Best-of-N)
- Experimental Thoroughness: ⭐⭐⭐⭐ (3×3 matrix + multi-angle analysis + qualitative cases)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clean formalization, intuitive visualization, tight theory-experiment integration)
- Value: ⭐⭐⭐⭐ (Plug-and-play inference-time enhancement with high practical utility)

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] ChartNet: A Million-Scale, High-Quality Multimodal Dataset for Robust Chart Understanding](../../CVPR2026/signal_comm/chartnet_a_million-scale_high-quality_multimodal_dataset_for_robust_chart_unders.md)
- [\[ICLR 2026\] Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies](multi-agent_design_optimizing_agents_with_better_prompts_and_topologies.md)
- [\[ICLR 2026\] Deterministic Bounds and Random Estimates of Metric Tensors on Neuromanifolds](deterministic_bounds_and_random_estimates_of_metric_tensors_on_neuromanifolds.md)
- [\[ICLR 2026\] FASA: Frequency-Aware Sparse Attention](fasa_frequency-aware_sparse_attention.md)
- [\[ICLR 2026\] Spectrum Tuning: Post-Training for Distributional Coverage and In-Context Steerability](spectrum_tuning_post-training_for_distributional_coverage_and_in-context_steerab.md)

<!-- RELATED:END -->
