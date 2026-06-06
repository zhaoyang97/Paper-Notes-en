---
title: >-
  [Paper Note] Enhancing Robustness of Offline RL Under Data Corruption via SAM
description: >-
  [AAAI 2026 (Student Abstract, Oral)][Reinforcement Learning][Offline RL] This paper is the first to apply Sharpness-Aware Minimization (SAM) as a plug-and-play optimizer for offline RL. It hypothesizes that data corrupti…
tags:
  - "AAAI 2026 (Student Abstract, Oral)"
  - "Reinforcement Learning"
  - "Offline RL"
  - "Data Corruption"
  - "SAM Optimizer"
  - "Flat Minima"
  - "Robustness"
date: 2026-05-08
content_hash: dd9ea7652ce09ffb
---

# Enhancing Robustness of Offline RL Under Data Corruption via SAM

**Conference**: AAAI 2026 (Student Abstract, Oral)
**arXiv**: [2511.17568](https://arxiv.org/abs/2511.17568)  
**Code**: None  
**Area**: Reinforcement Learning
**Keywords**: Offline RL, Data Corruption, SAM Optimizer, Flat Minima, Robustness

## TL;DR
This paper is the first to apply Sharpness-Aware Minimization (SAM) as a plug-and-play optimizer for offline RL. It hypothesizes that data corruption induces sharp minima in the loss landscape, leading to poor generalization, and demonstrates that SAM improves robustness by seeking flat minima. On the D4RL benchmark, IQL+SAM improves average score from 34.47 to 44.40.

## Background & Motivation

**Background**: Offline RL learns policies from static datasets, avoiding online interaction. IQL learns Q-functions and V-functions via expectile regression and exhibits inherent robustness to partial data corruption. RIQL augments IQL with Huber loss and quantile estimators to specifically handle dynamic corruption.

**Limitations of Prior Work**: Even though IQL and RIQL perform reasonably well under dynamic corruption, their performance degrades significantly under observation corruption and mixed corruption.

**Core Hypothesis**: Data corruption creates sharp, unreliable minima in the loss landscape. Models that converge to such sharp minima are insufficiently robust—small perturbations in the input data can cause large errors in value estimation.

**Key Insight**: Rather than modifying the algorithmic loss function, the paper proposes replacing the optimizer. SAM seeks flat regions via a two-step minimax procedure: it first finds an adversarial weight perturbation that locally maximizes the loss, then computes the gradient at this "worst-case" point to update the original parameters.

**Core Idea**: Replace Adam with SAM as the value function optimizer to seek flat minima rather than sharp ones.

## Method

### Overall Architecture
SAM is implemented as a custom PyTorch `Optimizer` class wrapping Adam. Ablation studies show that applying SAM exclusively to the value function network yields the most stable performance.

### Key Designs

1. **SAM Two-Step Optimization**:

    - **Ascent step**: Compute adversarial perturbation $\hat{\epsilon}(\theta) = \rho \frac{\nabla_\theta L(\theta)}{\|\nabla_\theta L(\theta)\|_2}$, where $\rho$ controls the neighborhood size.
    - **Descent step**: Compute the gradient at the perturbed parameters $\theta' = \theta + \hat{\epsilon}(\theta)$, and use this gradient to update the original $\theta$.
    - Intuition: Penalize sharpness by minimizing the highest loss value within the neighborhood.

2. **Applied to Value Function Only**:

    - Ablation experiments show that applying SAM simultaneously to the policy network leads to instability.
    - Sharp minima in the value function directly degrade policy extraction quality.

### Loss & Training
The loss functions of IQL/RIQL are unchanged; only the optimizer is replaced. Experiments use 3 random seeds with a 30% data corruption rate.

## Key Experimental Results

### Main Results (D4RL medium-replay, Random Corruption)

| Environment | Corruption Type | IQL | IQL+SAM | RIQL | RIQL+SAM |
|-------------|----------------|-----|---------|------|----------|
| HalfCheetah | Observation | 21.01 | **33.33** | 26.03 | **33.74** |
| HalfCheetah | Mixed | 20.93 | **33.02** | 22.08 | **32.06** |
| Walker2d | Observation | 24.74 | **31.75** | 30.48 | **30.93** |
| Hopper | Observation | 58.42 | **73.21** | 44.09 | **53.19** |
| **Average** | | 34.47 | **44.40** | 33.97 | **39.47** |

### Adversarial Corruption Results

| Method | Average Score |
|--------|--------------|
| IQL | 22.45 |
| **IQL+SAM** | **36.03** (+60%) |
| RIQL | 38.20 |
| **RIQL+SAM** | **40.09** |

### Reward Landscape Visualization
IQL converges to regions with sharp peaks and deep valleys, whereas IQL+SAM learns a noticeably smoother and flatter reward landscape—visually confirming that SAM guides the agent toward more robust solutions.

### Key Findings
- SAM yields particularly pronounced improvements under mixed corruption, which is the most challenging setting.
- Under adversarial corruption, IQL+SAM achieves a 60%+ gain, a remarkably large improvement.
- Only the optimizer is replaced without modifying any algorithmic logic, making this a true plug-and-play approach.
- Computational overhead is approximately 2× due to two forward–backward passes per step.

## Highlights & Insights
- The perspective of **addressing robustness through optimizer geometry** is distinctive—rather than designing new algorithms or loss functions, the approach directly replaces the optimizer.
- The causal hypothesis **"data corruption → sharp minima → poor generalization"** is validated through visualization, lending strong persuasive force.
- The **plug-and-play property** allows the method to be combined with any offline RL algorithm.

## Limitations & Future Work
- As a Student Abstract, the evaluation is limited to only 3 environments on medium-replay datasets.
- The 2× computational overhead of SAM warrants consideration in latency-sensitive applications.
- Sensitivity analysis of the $\rho$ hyperparameter is insufficient.
- No comprehensive comparison against other robust RL methods such as UWMSG, TRACER, or ADG.

## Related Work & Insights
- **vs. RIQL**: RIQL modifies the loss function to handle specific corruption types; SAM provides a general enhancement at the optimization level—the two approaches are complementary.
- **vs. ADG**: ADG recovers clean data via diffusion models (a preprocessing approach); SAM addresses corruption at the training optimization stage and can be combined with ADG.
- **vs. SAF**: SAF offers a more efficient SAM variant and could be substituted to reduce computational overhead.

## Rating
- Novelty: ⭐⭐⭐⭐ First application of SAM to offline RL; highly original perspective.
- Experimental Thoroughness: ⭐⭐⭐ Limited environments due to Student Abstract page constraints.
- Writing Quality: ⭐⭐⭐⭐ Clear and concise; visualizations are persuasive.
- Value: ⭐⭐⭐⭐ A general plug-and-play solution with strong inspirational value.

## Supplementary Analysis
- The proposed method represents a meaningful technical advance within its specific sub-domain.
- The core innovation lies in encoding domain-specific structural priors into the model design rather than relying entirely on data-driven end-to-end learning.
- Compared to concurrent top-venue work, this paper demonstrates a high level of research maturity in both problem formulation and the systematic design of the methodology.
- For real-world deployment, additional engineering considerations such as computational efficiency, latency requirements, data privacy, and system scalability must be addressed.
- The core idea of the method exhibits transferability—similar design paradigms may prove effective on related but distinct tasks and data modalities.
- The ablation study design is well-conceived, providing a clear analytical perspective for understanding each component's contribution to overall performance.
- Future work could explore integration with large-scale pretrained models (LLMs/VLMs/foundation models) to further raise the performance ceiling by leveraging their powerful representation learning capabilities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Trajectory-Level Data Augmentation for Offline Reinforcement Learning](../../ICML2026/reinforcement_learning/trajectory-level_data_augmentation_for_offline_reinforcement_learning.md)
- [\[ICLR 2026\] ReFORM: Reflected Flows for On-support Offline RL via Noise Manipulation](../../ICLR2026/reinforcement_learning/reform_reflected_flows_for_on-support_offline_rl_via_noise_manipulation.md)
- [\[ACL 2026\] A Survey of Reinforcement Learning for Large Language Models under Data Scarcity: Challenges and Solutions](../../ACL2026/reinforcement_learning/a_survey_of_reinforcement_learning_for_large_language_models_under_data_scarcity.md)
- [\[ICML 2026\] Towards Efficient and Expressive Offline RL via Flow-Anchored Noise-conditioned Q-Learning](../../ICML2026/reinforcement_learning/towards_efficient_and_expressive_offline_rl_via_flow-anchored_noise-conditioned_.md)
- [\[CVPR 2026\] AnyDoc: Enhancing Document Generation via Large-Scale HTML/CSS Data Synthesis and Height-Aware Reinforcement Optimization](../../CVPR2026/reinforcement_learning/anydoc_enhancing_document_generation_via_large-scale_htmlcss_data_synthesis_and_.md)

</div>

<!-- RELATED:END -->
