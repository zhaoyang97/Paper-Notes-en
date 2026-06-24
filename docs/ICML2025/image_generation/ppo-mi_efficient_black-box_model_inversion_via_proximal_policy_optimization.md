---
title: >-
  [Paper Note] PPO-MI: Efficient Black-Box Model Inversion via Proximal Policy Optimization
description: >-
  [ICML 2025][Image Generation][Model Inversion Attack] Formulating black-box model inversion attack as an MDP, PPO-MI uses PPO reinforcement learning to navigate and search the latent space of a generative model. Relying solely on the target model's prediction probabilities, it reconstructs training samples efficiently, achieving state-of-the-art attack success rates with fewer queries and less class data.
tags:
  - "ICML 2025"
  - "Image Generation"
  - "Model Inversion Attack"
  - "Black-Box Attack"
  - "Reinforcement Learning"
  - "PPO"
  - "Privacy and Security"
date: 2026-05-08
content_hash: 958334cf7593305f
---

# PPO-MI: Efficient Black-Box Model Inversion via Proximal Policy Optimization

**Conference**: ICML 2025  
**arXiv**: [2502.14370](https://arxiv.org/abs/2502.14370)  
**Code**: None  
**Area**: Image Generation  
**Keywords**: Model Inversion Attack, Black-Box Attack, Reinforcement Learning, PPO, Privacy and Security

## TL;DR
Formulating black-box model inversion attack as an MDP, PPO-MI uses PPO reinforcement learning to navigate and search the latent space of a generative model. Relying solely on the target model's prediction probabilities, it reconstructs training samples efficiently, achieving state-of-the-art attack success rates with fewer queries and less class data.

## Background & Motivation
**Background**: Model inversion attacks aim to reconstruct private training data from the predictions of a trained model. White-box methods (GMI, KED-MI) perform well but require model gradients, whereas black-box methods (VMI, MIRROR) require gradient estimation or a massive number of queries.

**Limitations of Prior Work**: (a) White-box methods are impractical for deployed scenarios; (b) existing black-box methods suffer from low query efficiency (100K+ queries) and require a large amount of target category data; (c) gradient-estimation-based methods are inherently unstable in high-dimensional spaces.

**Key Challenge**: How to efficiently search the high-dimensional latent space of generative models without gradient information?

**Goal**: To design a query-efficient black-box model inversion method with minimal information requirements.

**Key Insight**: Modeling the latent space search as a sequential decision-making process, substituting gradient estimation with the policy optimization capability of PPO.

**Core Idea**: Leveraging a PPO agent to navigate the StyleGAN2 latent space, utilizing momentum-based state transitions and a balanced dual reward mechanism to efficiently reconstruct target class faces.

## Method

### Overall Architecture
Given a pre-trained generator $G$ and a target model $T$, the PPO agent navigates within the latent space of $G$. Both the state $s_t$ and the action $a_t$ are vectors in the latent space. The state is updated via the momentum transition $s_{t+1} = \alpha s_t + (1-\alpha)a_t$, and the policy is trained using the classification probabilities of the target model as the reward signal.

### Key Designs

1. **MDP Formulation**:

    - Function: Defining model inversion as a learnable sequential decision-making problem.
    - Mechanism: Both the state space $S \in \mathbb{R}^{z_{dim}}$ and action space $A \in \mathbb{R}^{z_{dim}}$ are vectors in the latent space.
    - Design Motivation: RL is inherently well-suited for gradient-free iterative search problems.

2. **Momentum State Transition**:

    - Function: Smoothing latent space exploration and preventing abrupt jumps.
    - Mechanism: $s_{t+1} = \alpha s_t + (1-\alpha)a_t$, where $\alpha$ controls inertial momentum.
    - Design Motivation: Instantly jumping to a new position results in discontinuous generated images; momentum ensures smooth transitions.

3. **Balanced Reward Function**:

    - Function: Simultaneously driving classification accuracy and spatial exploration.
    - Mechanism: $R = \lambda_1 R_{\text{class}}(s_t) + \lambda_2 R_{\text{class}}(a_t) + \lambda_3 R_{\text{explore}}(s_t, a_t)$, where $R_{\text{explore}} = \beta \cdot \mathbf{1}[T(G(s_t)) \neq T(G(a_t))]$.
    - Design Motivation: Pure classification rewards lead to premature convergence to local optima; the exploration reward encourages discovering diverse regions.

## Key Experimental Results

### Main Results

| Dataset | Method | Attack Success Rate ↑ | Query Count |
|--------|------|-------------|--------|
| CelebA | KED-MI (White-box) | 72.4% | - |
| CelebA | RLB-MI (Black-box) | 76.3% | 40K |
| CelebA | **PPO-MI** | **79.7%** | **20K** |
| PubFig83 | RLB-MI | 41.5% | 40K |
| PubFig83 | **PPO-MI** | **44.3%** | **20K** |
| FaceScrub | KED-MI (White-box) | 47.8% | - |
| FaceScrub | **PPO-MI** | **48.5%** | **20K** |

### Ablation Study

| Configuration | Target Model | PPO-MI Success Rate |
|------|----------|-------------|
| VGG16 | CelebA | 72.6% |
| ResNet-152 | CelebA | 82.3% |
| Face.evoLVe | CelebA | 79.7% |

### Key Findings
- PPO-MI achieves the performance of RLB-MI with only 20K queries (halving the query budget).
- Black-box PPO-MI outperforms white-box KED-MI across multiple configurations.
- PPO-MI exhibits the best performance (52.5%) in cross-dataset transfer scenarios (FFHQ→CelebA).

## Highlights & Insights
- **Query Efficiency Gain**: Halves the queries compared to RLB-MI (SAC), demonstrating that PPO's trust region constraint is better suited for this task.
- **Data Efficiency**: Surpasses methods requiring 300+ classes with training on only 100 classes.
- **Transferable Concept**: The latent space search framework combining MDP and momentum transitions can be applied to other optimization tasks in latent spaces.

## Related Work & Insights
- **vs GMI/KED-MI (White-box)**: White-box methods directly optimize latent vectors via model gradients, achieving strong results but requiring complete model access. PPO-MI substitutes gradients with an RL policy, outperforming certain white-box methods under black-box settings.
- **vs RLB-MI (SAC)**: Both being RL-based black-box attacks, PPO is more stable than SAC's maximum entropy policy due to trust region optimization, halving the query counts.
- **vs MIRROR (Mirror Descent)**: MIRROR uses mirror descent for gradient estimation, requiring 100K queries; PPO-MI's continuous policy optimization is substantially more efficient.
- The proposed framework is highly abstract and theoretically adaptable to any black-box optimization problem centered around latent space searching (not limited to model inversion).

## Limitations & Future Work
- Evaluation is restricted to facial datasets; other sensitive data modalities (e.g., medical imaging, identification documents) have not been verified.
- The selection of the momentum coefficient $\alpha$ and reward weights $\lambda_{1,2,3}$ is heuristic and lacks theoretical guidance.
- Comparison with label-only attack scenarios (returning target labels only, without probabilities) is absent.
- The impact of PPO's actor-critic network architecture on attack performance has not been ablated.
- The impact of defense methods (e.g., adversarial training, output perturbations) on PPO-MI has not been evaluated.

## Rating
- Novelty: ⭐⭐⭐ Using RL for model inversion is not pioneering (RLB-MI); substituting SAC with PPO represents an incremental improvement.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across multiple datasets, model architectures, and cross-domain setups.
- Writing Quality: ⭐⭐⭐ The writing is acceptable, with occasional spelling errors.
- Value: ⭐⭐⭐⭐ Highlights the privacy risks inherent in deployed models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Support-Proximity Augmented Diffusion Estimation for Offline Black-Box Optimization](../../ICML2026/image_generation/support-proximity_augmented_diffusion_estimation_for_offline_black-box_optimizat.md)
- [\[NeurIPS 2025\] Transferable Black-Box One-Shot Forging of Watermarks via Image Preference Models](../../NeurIPS2025/image_generation/transferable_black-box_one-shot_forging_of_watermarks_via_image_preference_model.md)
- [\[CVPR 2025\] InPO: Inversion Preference Optimization with Reparametrized DDIM for Efficient Diffusion Model Alignment](../../CVPR2025/image_generation/inpo_inversion_preference_optimization_diffusion_alignment.md)
- [\[ICML 2025\] Discriminative Policy Optimization for Token-Level Reward Models](discriminative_policy_optimization_for_token-level_reward_models.md)
- [\[ICML 2025\] Smoothed Preference Optimization via ReNoise Inversion for Aligning Diffusion Models with Varied Human Preferences](smoothed_preference_optimization_via_renoise_inversion_for_aligning_diffusion_mo.md)

</div>

<!-- RELATED:END -->
