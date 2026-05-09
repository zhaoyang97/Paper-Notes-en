---
title: >-
  [Paper Note] PADiff: Predictive and Adaptive Diffusion Policies for Ad Hoc Teamwork
description: >-
  [AAAI 2026][Image Generation][Ad Hoc Teamwork] This work is the first to apply diffusion models to the Ad Hoc Teamwork (AHT) problem. The proposed PADiff framework achieves real-time adaptation to dynamic teammates via an Adaptive Feature Modulation Net (AFM-Net), and injects teammate intent prediction into the denoising process through a Predictive Guidance Block (PGB), achieving an average improvement of 35.25% over existing methods in multimodal cooperative scenarios.
tags:
  - AAAI 2026
  - Image Generation
  - Ad Hoc Teamwork
  - Diffusion Models
  - Multimodal Policy
  - Predictive Guidance
  - Adaptive Feature Modulation
date: 2026-05-08
content_hash: 274ed7ca192d6f8a
---

# PADiff: Predictive and Adaptive Diffusion Policies for Ad Hoc Teamwork

**Conference**: AAAI 2026  
**arXiv**: [2511.07260](https://arxiv.org/abs/2511.07260)  
**Code**: None  
**Area**: Image Generation / Multi-Agent Collaboration  
**Keywords**: Ad Hoc Teamwork, Diffusion Models, Multimodal Policy, Predictive Guidance, Adaptive Feature Modulation

## TL;DR

This work is the first to apply diffusion models to the Ad Hoc Teamwork (AHT) problem. The proposed PADiff framework achieves real-time adaptation to dynamic teammates via an Adaptive Feature Modulation Net (AFM-Net), and injects teammate intent prediction into the denoising process through a Predictive Guidance Block (PGB), achieving an average improvement of 35.25% over existing methods in multimodal cooperative scenarios.

## Background & Motivation

### State of the Field

**Ad Hoc Teamwork (AHT)** is a core challenge in multi-agent systems: agents are required to collaborate effectively with **previously unseen teammates** in the absence of **predefined coordination protocols**. This is critical in real-world scenarios such as disaster rescue (collaborating with unfamiliar responders) and autonomous driving (interacting with vehicles exhibiting unknown driving styles).

Existing AHT methods are primarily based on reinforcement learning (e.g., LIAM, GPL-SPI, ODITS), which learn teammate representations to adapt to diverse teammate behaviors.

### Limitations of Prior Work

**Policy collapse to a single mode**: RL methods optimize a fixed expected return, causing policies to collapse to a single dominant behavior. In AHT, multiple valid cooperative modes may exist for the same state (e.g., passing to teammate A, passing to teammate B, or shooting independently), yet RL can only learn one. Even maximum-entropy RL (e.g., SAC) disperses behavior without direction, failing to structurally model multimodal distributions.

**Lack of predictive capability in diffusion models**: Diffusion models are naturally suited to modeling multimodal distributions, but standard diffusion models are designed for distribution reconstruction and **lack the ability to predict teammate intent**, whereas AHT requires anticipating teammate behavior for proactive decision-making.

**Insufficient adaptability in conventional denoising networks**: Traditional denoising architectures such as MLPs and UNets lack mechanisms to adapt to dynamically changing teammate behaviors. Transformers (e.g., DiT) offer attention mechanisms but incur prohibitive computational overhead, making them unsuitable for fast-paced AHT scenarios.

### Root Cause

AHT simultaneously demands three capabilities: (1) multimodal policy representation — capturing diverse cooperative modes; (2) predictive capability — anticipating teammate intent for proactive decisions; and (3) real-time adaptation — rapid response to changes in teammate behavior. Existing methods satisfy at most one of these.

### Core Idea

The paper uses a diffusion model as the policy backbone to naturally capture multimodal cooperative patterns, and designs two dedicated modules to address the predictive and adaptive requirements of AHT: (1) **AFM-Net** — an adaptive feature modulation network based on the FiLM mechanism, which dynamically modulates intermediate features during denoising using teammate context; and (2) **PGB** — a predictive guidance block that, during training, predicts teammates' collaborative return and collaborative goals, internalizing teammate intent prediction into the denoising process via gradient backpropagation, with zero additional computation at inference.

## Method

### Overall Architecture

The PADiff training pipeline proceeds as follows:
1. Sample teammates from a diverse teammate policy pool
2. The ego agent interacts with teammates in the environment to collect trajectories
3. Trajectories are stored in an offline dataset
4. The ego agent is optimized using a diffusion policy

The framework comprises three core components: diffusion policy representation, the Teammates Adaptation Block (containing AFM-Net), and the Predictive Guidance Block (PGB).

### Key Designs

#### 1. **Discrete Diffusion Policy Representation**

**Function**: Models the ego agent's policy as a conditional diffusion process, performing denoising over a discrete action space.

**Mechanism**: For discrete action spaces, the D3PM (Discrete Denoising Diffusion Probabilistic Models) framework is adopted. The forward process employs a categorical distribution:

$$q(\mathbf{a}_t^k | \mathbf{a}_t^{k-1}) = \text{Cat}(\mathbf{a}_t^k; p = \mathbf{a}_t^{k-1} Q_k)$$

where $Q_k$ is a transition matrix with a Uniform schedule. Training minimizes the variational lower bound:

$$\mathcal{L}_{Diff} = \mathbb{E}[D_{KL}[q(\mathbf{a}^{k-1}|\mathbf{a}^k, \mathbf{a}^0) \| p_\theta(\mathbf{a}^{k-1}|\mathbf{a}^k, s, k)]]$$

**Design Motivation**: Diffusion models can represent any normalizable distribution, making them naturally suited for modeling multimodal cooperative policies in AHT. Through iterative denoising, the policy can simultaneously "prepare" multiple cooperative strategies.

#### 2. **State Encoder and Teammates Adaptation Block**

**State Encoder**: Uses a historical window $s_{t-m:t}$ to capture teammate cooperative context, encoding it as a latent variable from a multivariate Gaussian distribution:

$$(μ_{z_t}, σ_{z_t}) = f_ξ(s_{t-m:t}), \quad z_t \sim \mathcal{N}(μ_{z_t}, σ_{z_t})$$

**AFM-Net (Adaptive Feature Modulation Net)**: A denoising network based on the FiLM (Feature-wise Linear Modulation) mechanism, with three core characteristics:

**(a) Conditional feature modulation**: Scale/shift parameters $\gamma, \beta$ are generated from teammate context $z_t$ and diffusion step $k$ to dynamically modulate intermediate features:

$$\gamma_1, \beta_1, \gamma_2, \beta_2 = MLP(\mathbf{z}_t + k)$$
$$\text{AFM}(\mathbf{x}, z_t, k) = \gamma_2 \cdot (MLP(\gamma_1 \cdot LN(\mathbf{x}) + \beta_1)) + \beta_2 + \mathbf{x}$$

**(b) Residual connections**: Ensure training stability and robust representations.

**(c) Dropout regularization**: Improves generalization to unseen teammates.

**Design Motivation**: UNet lacks teammate-awareness, while DiT's attention mechanism is computationally expensive. The FiLM mechanism achieves conditional modulation at minimal computational cost (a single MLP generating scale/shift parameters), while the combination of Layer Norm, Residual connections, and Dropout ensures robustness. The framework stacks two AFM-Net layers in cascade.

#### 3. **Predictive Guidance Block (PGB)**

**Function**: During training, guides the denoising process to learn to predict teammate intent, enabling the generation of teammate-aware actions at inference without additional modules.

**Mechanism**: PGB encompasses two prediction tasks:

**(a) Collaborative Return (CoReturn)** — predicts the expected cumulative team reward:
$$L_{\text{CoReturn}} = \mathbb{E}_{\tau \sim \mathcal{D}}\left[\sum_{t=1}^{T}\|R_\phi(h_t^k, s_t) - R_t\|^2\right]$$

**(b) Collaborative Goal (CoGoal)** — predicts future team states (subgoals):
$$L_{\text{CoGoal}} = \mathbb{E}_{\tau \sim \mathcal{D}}\left[-\frac{1}{N}\sum_{t}\sum_{i}(G_{t,i}\log\hat{G}_{t,i} + (1-G_{t,i})\log(1-\hat{G}_{t,i}))\right]$$

Both prediction tasks take the intermediate features $h_t^k$ from AFM-Net and state $s_t$ as input, optimizing the intermediate representations via gradient backpropagation:

$$\nabla_{h_t^k}L_{\text{total}} = \nabla_{h_t^k}L_{\text{Diffusion}} + \alpha\nabla_{h_t^k}L_{\text{CoReturn}} + \beta\nabla_{h_t^k}L_{\text{CoGoal}}$$

**PGB is entirely removed at inference** — as AFM-Net has internalized the predictive capability through training, its intermediate features are inherently teammate-aware.

**Design Motivation**: (1) CoReturn provides "overall direction" — which action sequences yield higher team returns; (2) CoGoal provides "specific intent" — where teammates are heading and what they will do next. This hierarchical prediction (Return followed by Goal) emulates the natural "evaluation → planning" process in teamwork. Zero overhead at inference is a key advantage.

### Loss & Training

Total training loss:
$$L_{\text{total}} = L_{\text{Diffusion}} + \alpha L_{\text{CoReturn}} + \beta L_{\text{CoGoal}}$$

Teammate pool construction: The Soft-Value Diversity (SVD) method from the CSP framework is used to train 4 independent multi-agent populations — 3 for training interactions and 1 (containing 12 policy checkpoints) for testing. Training runs for 20 epochs with evaluation every 2 epochs.

## Key Experimental Results

### Main Results

Comparison of evaluation returns across three environments (key figures extracted from Figure 5 and Appendix Table 1):

| Environment | Method | Avg. Return | Notes |
|------|------|---------|------|
| Predator-Prey | LIAM | ~18 | Teammate-modeling RL |
| Predator-Prey | Diffusion-BC | ~16 | Diffusion behavioral cloning |
| Predator-Prey | Diffusion-QL | ~20 | Diffusion Q-learning |
| Predator-Prey | **PADiff** | **~26** | +30% vs Diffusion-QL |
| LBF | ODITS | ~0.45 | Online adaptive AHT |
| LBF | **PADiff** | **~0.65** | +44% vs ODITS |
| Overcooked | Diffusion-QL | ~80 | Diffusion Q-learning |
| Overcooked | **PADiff** | **~130** | +62% vs Diffusion-QL |

PADiff significantly outperforms all baselines across all environments, achieving an **average performance improvement of 35.25%**.

### Ablation Study

Ablation on denoising network architecture (replacing AFM-Net):

| Architecture | PP Return | LBF Return | Overcooked Return | Notes |
|------|---------|---------|----------------|------|
| MLP | ~20 | ~0.50 | ~90 | No conditional modulation |
| UNet | ~22 | ~0.55 | ~95 | Image-oriented design, ill-suited |
| **AFM-Net** | **~26** | **~0.65** | **~130** | FiLM + Residual + Dropout |

Ablation on PGB modules:

| Configuration | PP Return | LBF Return | Overcooked Return | Notes |
|------|---------|---------|----------------|------|
| **Full PADiff** | **~26** | **~0.65** | **~130** | Complete model |
| w/o CoReturn | ~23 | ~0.58 | ~110 | Remove collaborative return prediction |
| w/o CoGoal | ~22 | ~0.55 | ~105 | Remove collaborative goal prediction |
| w/o PGB | ~20 | ~0.50 | ~95 | Remove entire predictive guidance |

**Both prediction tasks are indispensable**; removing CoGoal causes a slightly larger performance drop, indicating that intent prediction is more critical than return prediction.

### Key Findings

1. **Multimodal policy visualization**: When the same state is fed into the policy multiple times, the ego agent produces distinct cooperative trajectories (e.g., surrounding prey from different directions), confirming that the diffusion policy has learned a genuinely multimodal distribution.
2. **AFM-Net outperforms UNet**: UNet's multi-scale feature aggregation, designed for images, is ill-suited to the low-dimensional decision-making setting of AHT; the FiLM mechanism is more efficient.
3. **Zero inference overhead from PGB**: By internalizing predictive capability into AFM-Net through gradient backpropagation during training, removing PGB at inference does not degrade performance.
4. **Largest advantage in the most challenging environment (Overcooked)**: The complex spatial layout and high cooperative demands of Overcooked amplify PADiff's advantage (+62%).

## Highlights & Insights

1. **First application of diffusion models to AHT**: Demonstrates the value of diffusion models in multi-agent collaboration — multimodal policy representation is a natural requirement of AHT.
2. **PGB's "predict during training, free at inference" design**: Optimizing intermediate representations via auxiliary prediction gradients is an inspiring design pattern, reminiscent of world model prediction in DreamerV3 but tailored to cooperative scenarios.
3. **Elegant use of the FiLM mechanism**: Replacing complex attention with minimal scale/shift modulation achieves conditional denoising efficiently, well-suited for real-time decision-making.
4. **Consistent 35.25% average improvement**: The method uniformly and substantially outperforms baselines across all environments, demonstrating strong generality.

## Limitations & Future Work

1. **Validation limited to grid-world and kitchen environments**: PP, LBF, and Overcooked are relatively simple discrete environments; the method has not been evaluated on continuous control or high-dimensional visual input settings.
2. **Restricted to discrete action spaces**: D3PM models discrete actions; continuous action spaces (e.g., robotic joint control) are not addressed.
3. **Teammate pool construction relies on SVD**: Pretraining a diverse teammate policy pool incurs additional preparation costs.
4. **Denoising step count not reported**: The number of denoising steps at inference is not explicitly stated; multi-step sampling in diffusion models may still introduce latency in real-time AHT scenarios.
5. **Missing comparison with decision transformer-based methods**: TAGET is a recent DT-based AHT method, but the experimental comparison with it is insufficiently prominent.

## Related Work & Insights

- **Diffusion Policy (Wang et al. 2022)**: Provides the foundational framework for diffusion-model-based policies; PADiff builds upon it by introducing predictive and adaptive capabilities for AHT.
- **FiLM (Perez et al. 2018)**: The conditional modulation mechanism in AFM-Net is directly inspired by FiLM, which has also demonstrated success in visual reasoning.
- **DreamerV3**: The idea of enhancing decision-making through environment prediction in world models is analogous to PGB, though PGB focuses specifically on teammate cooperation prediction.
- **Implications for multi-agent learning**: The paradigm of diffusion model multimodality combined with predictive auxiliary tasks is broadly applicable to cooperative and competitive multi-agent scenarios.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First application of diffusion models to AHT; three core components are each novel in design
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three environments, multiple baselines, complete architecture and module ablations
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clearly argued; method is described in detail
- **Value**: ⭐⭐⭐⭐ — Establishes a new technical direction for AHT; 35% improvement is substantial

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Improving Discrete Diffusion Unmasking Policies Beyond Explicit Reference Policies (UPO)](../../ICLR2026/image_generation/improving_discrete_diffusion_unmasking_policies_beyond_explicit_reference_polici.md)
- [\[ICLR 2026\] Compose Your Policies! Improving Diffusion-based or Flow-based Robot Policies via Test-time Distribution-level Composition](../../ICLR2026/image_generation/compose_your_policies_improving_diffusion-based_or_flow-based_robot_policies_via.md)
- [\[NeurIPS 2025\] Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge](../../NeurIPS2025/image_generation/towards_general_modality_translation_with_contrastive_and_predictive_latent_diff.md)
- [\[ICLR 2026\] Offline Reinforcement Learning with Generative Trajectory Policies](../../ICLR2026/image_generation/offline_reinforcement_learning_with_generative_trajectory_policies.md)
- [\[ICLR 2026\] Contractive Diffusion Policies: Robust Action Diffusion via Contractive Score-Based Sampling with Differential Equations](../../ICLR2026/image_generation/contractive_diffusion_policies_robust_action_diffusion_via_contractive_score-bas.md)

</div>

<!-- RELATED:END -->
