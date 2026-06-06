---
title: >-
  [Paper Note] Inference-time Alignment in Continuous Space
description: >-
  [NeurIPS 2025][LLM Alignment][inference-time alignment] This paper proposes Simple Energy Adaptation (SEA), which shifts the inference-time alignment paradigm from discrete-space search to continuous-space optimization.…
tags:
  - "NeurIPS 2025"
  - "LLM Alignment"
  - "inference-time alignment"
  - "energy-based model"
  - "Langevin dynamics"
  - "RLHF"
  - "continuous optimization"
  - "reward model"
date: 2026-05-08
content_hash: 2ee97db8cb83c8e5
---

# Inference-time Alignment in Continuous Space

**Conference**: NeurIPS 2025
**arXiv**: [2505.20081](https://arxiv.org/abs/2505.20081)  
**Code**: [GitHub](https://github.com/YigeYuan/SEA)  
**Area**: LLM Alignment
**Keywords**: inference-time alignment, energy-based model, Langevin dynamics, RLHF, continuous optimization, reward model

## TL;DR

This paper proposes Simple Energy Adaptation (SEA), which shifts the inference-time alignment paradigm from discrete-space search to continuous-space optimization. By performing gradient-based Langevin sampling over the continuous logit space, SEA approximates the optimal RLHF policy, achieving a 77.51% relative improvement over the strongest baseline on AdvBench and a 16.36% improvement on MATH.

## Background & Motivation

Inference-time alignment has attracted growing attention due to its training-free, plug-and-play nature. Existing methods predominantly follow a discrete-space search paradigm:

**Best-of-N (BoN)**: Generates $N$ candidate responses from the base policy and selects the one with the highest reward.

**ARGS**: Selects output tokens one at a time guided by reward signals.

**CBS**: Performs beam search at the chunk level.

Core limitations of these approaches:

- **Dependence on base policy quality**: When the base policy is weak, the candidate set is unlikely to contain high-quality responses.
- **Exponential sampling requirements**: If the base policy generates the optimal response with probability $\sigma$, the probability that BoN includes at least one optimal response is $1-(1-\sigma)^N$, requiring exponentially growing $N$ when $\sigma$ is small.
- **Constrained to discrete search space**: Gradient information from the reward model cannot be exploited to actively move toward high-reward regions.

## Method

### Overall Architecture

The core idea of SEA is to forgo search in the discrete response space and instead iteratively optimize initial responses along the reward gradient direction within the continuous logit space.

**Three-step pipeline**:
1. Generate an initial response from the base policy $\pi_{\text{ref}}$ and obtain its continuous logit representation.
2. Define an energy function grounded in the optimal RLHF policy.
3. Iteratively optimize the logits in continuous space via Langevin dynamics, then decode into discrete text.

### Key Designs

**Energy function definition**:

The optimal RLHF policy admits the following closed-form solution:

$$\pi^*(y|x) = \frac{1}{Z(x)} \exp(E(x, y))$$

where the energy function is defined as:

$$E(x, y) = \log \pi_{\text{ref}}(y|x) + \alpha \cdot r(x, y)$$

Here $\alpha$ controls the trade-off between reward maximization and KL penalization, and $Z(x)$ is the partition function.

**Langevin MCMC sampling**:

Since directly sampling from the optimal policy requires computing the intractable partition function, SEA employs gradient-based Langevin sampling:

$$y^{(n+1)} \leftarrow y^{(n)} - \eta \nabla_y E(x, y^{(n)}) + \epsilon^{(n)}$$

where $\epsilon^{(n)} \sim \mathcal{N}(0, I)$ is Gaussian noise and $\eta$ is the step size.

Crucially, $\nabla_y \log \pi^*(y|x) = -\nabla_y E(x, y)$ (since the gradient of the partition function with respect to $y$ is zero), so $Z(x)$ need not be computed.

**Continuization**:

- Since discrete token sequences are non-differentiable, SEA uses the continuous logits of the LLM as the representation of $y$.
- A Straight-Through Estimator is adopted: argmax (discrete) is used in the forward pass, while softmax (continuous) is used in the backward pass.
- Continuous logits are directly fed as input tokens to both the reference model and the reward model.

**Multiple initialization strategy**:

Multiple Langevin chains (4 by default) are run in parallel, each initialized from a different sample of the base policy. The response with the highest reward is selected at the end.

### Loss & Training

SEA is an inference-time method and involves no training. Its optimization objective is to minimize the energy function:

$$\min_y E(x, y) = -[\log \pi_{\text{ref}}(y|x) + \alpha \cdot r(x, y)]$$

This is equivalent to maximizing the reward while remaining close to the reference policy.

## Key Experimental Results

### Main Results

**AdvBench safety evaluation** (Harmful Rate ↓):

| Method | LLaMA-3.2-1B | LLaMA-3.2-3B | LLaMA-3-8B | LLaMA-3.2-1B-Instruct |
|--------|-------------|-------------|-----------|----------------------|
| SFT | 65.96% | 50.77% | 14.42% | 0.77% |
| BoN-64 | 43.85% | 28.27% | 8.85% | 0.77% |
| ARGS | 25.96% | 22.50% | 8.27% | 0.19% |
| CBS | 24.81% | 23.65% | 6.35% | 0.96% |
| **SEA** | **5.58%** | **6.92%** | **3.85%** | **0.19%** |

SEA reduces the harmful rate on LLaMA-3.2-1B-Base by 91.54% relative to SFT, far surpassing BoN-64.

**TruthfulQA truthfulness evaluation** (TR ↑):

| Method | LLaMA-3.2-1B | LLaMA-3.2-3B | LLaMA-3-8B | LLaMA-3.2-1B-Instruct |
|--------|-------------|-------------|-----------|----------------------|
| SFT | 59.0% | 64.0% | 62.0% | 72.0% |
| BoN-64 | 78.0% | 74.0% | 72.0% | 77.0% |
| **SEA** | **78.0%** | **80.0%** | **76.0%** | **89.0%** |

SEA improves truthfulness while maintaining informativeness and diversity (also achieving the highest Diversity scores).

**Mathematical reasoning evaluation** (LLaMA-3.2-1B-Instruct):

| Method | GSM8K Acc | MATH Acc |
|--------|----------|---------|
| SFT | 32.00% | 27.50% |
| BoN-64 | 57.00% | 16.00% |
| **SEA** | **58.00%** | **32.00%** |

Search-based methods on MATH perform even below SFT (BoN-64 achieves only 16%), whereas SEA raises accuracy to 32%.

### Ablation Study

**Ablation on LLaMA-3.2-1B-Base** (AdvBench HR ↓):

| Variant | HR (%) |
|---------|--------|
| SEA (4 chains) | 5.58 |
| SEA (1 chain) | 13.65 |
| w/ Random Init | 4.04 |
| w/o Reward | 19.62 |
| w/o Reference | 12.69 |
| w/o Noise | 6.73 |

- Even a single chain yields substantial improvements over SFT (65.96%).
- Random initialization performs better on safety tasks, as starting from an already harmful response makes optimization harder.
- Removing the reward model still yields gains, as the stochastic exploration of Langevin dynamics expands the effective search space.

### Key Findings

1. **Deep alignment**: SEA distributes its KL budget uniformly across all token positions, unlike conventional methods that concentrate modifications in the first few tokens (shallow alignment).
2. **Robustness to prefilling attacks**: On LLaMA-3.2-1B-Instruct, even when 7 harmful prefix tokens are injected, SEA maintains a 0% attack success rate, while BoN-32 is completely compromised.
3. **Rapid reward convergence**: Rewards stabilize after approximately 30 iterations, at which point response quality reaches a high level.

## Highlights & Insights

1. **Paradigm innovation**: The shift from discrete search to continuous optimization is elegant and powerful, naturally exploiting gradient information from the reward model.
2. **Theoretical elegance**: Modeling the optimal RLHF policy as an EBM and sampling via Langevin dynamics provides a solid theoretical foundation.
3. **Deep alignment**: SEA performs alignment simultaneously across all token positions, inherently avoiding the fragility of shallow alignment.
4. **Plug-and-play**: No model parameter modification is required; the method is applicable to any combination of a base LLM and a reward model.
5. **Effective even for weak models**: Performance is not constrained by base policy capability—gradient-based optimization can still navigate to high-reward regions even when the base model is weak.

## Limitations & Future Work

1. **Computational overhead**: Backpropagation through both the reward model and the reference model is required for gradient computation, consuming more GPU memory than BoN.
2. **Approximation error of the Straight-Through Estimator**: Approximating discrete tokens with continuous logits may introduce bias.
3. **Reward model dependence**: Performance is upper-bounded by the quality of the reward model, and reward hacking remains a risk.
4. **Evaluation limited to the LLaMA-3 family**: Closed-source models such as GPT and Claude cannot be directly used, as gradient access is required.
5. **Multi-chain overhead**: The default 4-chain setup quadruples computational resource consumption.

## Related Work & Insights

- **Best-of-N / ARGS / CBS**: Discrete search methods whose performance is bounded by the base policy and the candidate set.
- **COLD (Qin et al., 2022)**: Employs gradient-based sampling in vocabulary space for controllable generation; SEA adapts this idea to the alignment setting.
- **MuCoCO / MuCoLa**: Pioneer works on optimizing over intermediate representations for controllable inference.
- **COLD-Attack**: Uses energy functions for jailbreak attacks; SEA operates in the opposite direction—for defense and alignment.
- **Insight**: Continuous optimization methods are severely underexplored for inference-time alignment; SEA opens this direction. Future work may investigate more efficient samplers (e.g., HMC) or adaptive step-size strategies.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The discrete-to-continuous paradigm shift represents a genuinely novel direction for inference-time alignment.
- **Technical Depth**: ⭐⭐⭐⭐ — The EBM + Langevin dynamics framework is theoretically sound, though the continuization approximation has limitations.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers 4 model variants, 3 task categories, 7 baselines, and includes rich ablations and in-depth analyses.
- **Value**: ⭐⭐⭐⭐ — The plug-and-play formulation has practical potential, though computational overhead limits large-scale deployment.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clear, the method is presented intuitively, and visualizations are informative.
- **Overall**: ⭐⭐⭐⭐⭐ (9/10)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] W2S-AlignTree: Weak-to-Strong Inference-Time Alignment for Large Language Models via Monte Carlo Tree Search](../../AAAI2026/llm_alignment/w2s-aligntree_weak-to-strong_inference-time_alignment_for_large_language_models_.md)
- [\[NeurIPS 2025\] Limited Preference Data? Learning Better Reward Model with Latent Space Synthesis](limited_preference_data_learning_better_reward_model_with_latent_space_synthesis.md)
- [\[NeurIPS 2025\] What Makes a Reward Model a Good Teacher? An Optimization Perspective](what_makes_a_reward_model_a_good_teacher_an_optimization_perspective.md)
- [\[NeurIPS 2025\] ResponseRank: Data-Efficient Reward Modeling through Preference Strength Learning](responserank_data-efficient_reward_modeling_through_preference_strength_learning.md)
- [\[ICLR 2026\] GuardAlign: Test-time Safety Alignment in Multimodal Large Language Models](../../ICLR2026/llm_alignment/guardalign_test-time_safety_alignment_in_multimodal_large_language_models.md)

</div>

<!-- RELATED:END -->
