---
title: >-
  [Paper Note] Reward Fine-Tuning Two-Step Diffusion Models via Learning Differentiable Latent-Space Surrogate Reward
description: >-
  [CVPR 2025][Image Generation][Reward Fine-Tuning] This paper proposes LaSRO, which learns a differentiable surrogate reward model in the latent space to transform any (including non-differentiable) reward signal into differentiable gradient guidance. This achieves efficient reward fine-tuning for two-step diffusion models, significantly outperforming mainstream reinforcement learning methods such as DDPO and DPO.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Reward Fine-Tuning"
  - "Step-Distilled Diffusion Models"
  - "Surrogate Reward"
  - "Latent-Space Optimization"
  - "Reinforcement Learning"
date: 2026-05-08
content_hash: e9ff39c8f6cdd7f6
---

# Reward Fine-Tuning Two-Step Diffusion Models via Learning Differentiable Latent-Space Surrogate Reward

**Conference**: CVPR 2025  
**arXiv**: [2411.15247](https://arxiv.org/abs/2411.15247)  
**Code**: None  
**Area**: Diffusion Models / Image Generation  
**Keywords**: Reward Fine-Tuning, Step-Distilled Diffusion Models, Surrogate Reward, Latent-Space Optimization, Reinforcement Learning

## TL;DR

This paper proposes LaSRO, which learns a differentiable surrogate reward model in the latent space to transform any (including non-differentiable) reward signal into differentiable gradient guidance. This achieves efficient reward fine-tuning for two-step diffusion models, significantly outperforming mainstream reinforcement learning methods such as DDPO and DPO.

## Background & Motivation

**Background**: Diffusion models (DMs) demonstrate exceptional performance in text-to-image generation but suffer from slow inference speeds. Consequently, step-distillation methods (such as LCM, SDXL-Turbo, etc.) compress sampling to $\le 2$ steps to achieve ultra-fast generation. On the other hand, using reinforcement learning (RL) for reward fine-tuning of diffusion models, aligning their outputs with human preferences (e.g., aesthetics, text alignment, etc.), has emerged as an important research direction.

**Limitations of Prior Work**: Direct application of existing policy gradient-based RL methods (such as PPO/DDPO, DPO) to $\le 2$-step distilled diffusion models faces three major obstacles: (1) **Exploration difficulty**—the stochasticity of two-step DMs is extremely low (noise is injected only once), restricting the exploration space of on-policy methods; (2) **Degeneration of RL objectives**—the second step of a two-step LCM is deterministic, causing most policy gradient objectives to degenerate into optimizing only the first half of the sampling process; (3) **Non-smooth mapping**—the distilled two-step mapping has a very large Lipschitz constant, leading to a highly volatile reward surface, which makes the variance of policy gradient estimators extremely high, while DPO-based reward-weighted regression methods lead to blurry images.

**Key Challenge**: While step distillation brings speed advantages, it simultaneously destroys the stochasticity and smoothness of multi-step diffusion sampling, rendering traditional RL methods completely ineffective in few-step generation scenarios.

**Goal**: To design a reward fine-tuning method suitable for $\le 2$-step distilled diffusion models that can handle arbitrary (including non-differentiable) reward signals while maintaining training stability and sample efficiency.

**Key Insight**: The authors observe that if arbitrary reward signals can be transformed into a differentiable surrogate reward in the latent space, model optimization can be guided directly through backpropagation, avoiding the issues associated with policy gradient estimation. Furthermore, using a pretrained latent diffusion model (such as the UNet encoder of SDXL) as the backbone of the surrogate reward model yields excellent generalization ability and computational efficiency.

**Core Idea**: Learn a latent-space surrogate reward model to convert non-differentiable rewards into differentiable gradients, replacing policy gradient estimation with direct reward gradient guidance, and address the exploration difficulty of two-step DMs through off-policy exploration.

## Method

The core idea of LaSRO is to avoid direct optimization of the diffusion model using policy gradients. Instead, it first learns a differentiable surrogate reward, and then uses the gradient of this surrogate reward to directly guide model parameter updates. The entire method is divided into two phases: the surrogate reward pre-training phase and the alternating fine-tuning phase.

### Overall Architecture

The input is a pre-trained two-step LCM (e.g., LCM-SSD-1B) and a target reward function (which can be non-differentiable, e.g., Image Reward). The output is the fine-tuned LCM, which achieves higher reward scores in $\le 2$-step generation. This process involves two phases: (1) Pre-training phase: Generate samples using the two-step LCM, construct winning/losing sample pairs based on the target reward, and train a latent-space surrogate reward model based on the Bradley-Terry model; (2) Fine-tuning phase: Alternately perform model reward optimization (updating LCM parameters using the gradient of the surrogate reward) and surrogate reward online adaptation (updating the surrogate reward with new online samples to address distribution shift).

### Key Designs

1. **Latent-space Surrogate Reward Model $\mathcal{R}_\psi$**:

    - **Function**: Transform arbitrary reward signals (including non-differentiable ones) into differentiable surrogate rewards in the latent space.
    - **Mechanism**: Utilize the UNet encoder of a pre-trained SDXL as the backbone and append a CNN prediction head. During training, multiple images are sampled for each prompt, winning/losing pairs are obtained by ranking based on the target reward, and the surrogate reward is trained using the Bradley-Terry preference loss: $\mathcal{L}_{surr}(\psi;r) = -\mathbb{E}[\log(y_\psi^c(z^w, z^l))]$, where $y_\psi^c$ is the softmax-normalized surrogate reward difference. The surrogate reward operates directly in the latent space without VAE decoding, substantially saving VRAM and computation.
    - **Design Motivation**: To avoid policy gradient estimation (which has high variance) and reward-weighted regression (which causes blurriness) by directly providing gradient guidance. The authors compared three backbone architectures—CLIP, BLIP, and the SDXL UNet encoder—and found that the UNet encoder delivers the best generalization and highest efficiency.

2. **Off-Policy Exploration Strategy**:

    - **Function**: Resolve the restricted exploration space of two-step DMs.
    - **Mechanism**: In each iteration, sample $N_s$ different initial noises $z_{\tau_0}$ for the same prompt to generate diverse outputs for the first and second steps. Consequently, exploration does not rely on the on-policy distribution under a fixed initial noise; instead, off-policy coverage is achieved by varying initial conditions.
    - **Design Motivation**: The sampling process of a two-step LCM injects noise only once, resulting in extremely limited exploration for a given initial noise. By simultaneously varying the initial noise, the model essentially samples from a broader distribution, which greatly enhances exploration efficiency. This aligns with the off-policy concept in value-based RL.

3. **Alternating Fine-Tuning Mechanism (Reward Optimization + Online Adaptation)**:

    - **Function**: Maintain the accuracy of the surrogate reward while continuously optimizing the LCM.
    - **Mechanism**: During the fine-tuning phase, two subprocesses are executed alternately: (a) Reward fine-tuning: Update LCM parameters using the normalized and clipped gradients of the surrogate reward, while incorporating the original LCM distillation loss as a regularization term. The total loss is formulated as: $\mathcal{L}_{lasro} = c \cdot \mathcal{L}_{lcm} + c_1 \cdot \mathcal{S}[\mathcal{R}_\psi(z_1, c)] + c_2 \cdot \mathcal{S}[\mathcal{R}_\psi(z_2, c)]$; (b) Online adaptation: Update the surrogate reward model using newly collected sample pairs in a replay buffer to address the drift in the LCM's output distribution.
    - **Design Motivation**: The output distribution of the LCM constantly shifts during fine-tuning. If the surrogate reward remains fixed, it leads to reward hacking. Alternating updates ensure the surrogate reward consistently aligns with the output distribution of the current model.

### Loss & Training

The pre-training phase uses the Bradley-Terry preference loss to train the surrogate reward, with separate training for one-step and two-step outputs. The total loss during the fine-tuning phase comprises three components: (1) The LCM distillation regularization loss $\mathcal{L}_{lcm}$ to prevent the model from deviating too far; (2) The surrogate reward for the first-step output $\mathcal{S}[\mathcal{R}_\psi(z_1, c)]$; and (3) The surrogate reward for the second-step output $\mathcal{S}[\mathcal{R}_\psi(z_2, c)]$. Here, $\mathcal{S}$ is a normalization and clipping function that stabilizes training by tracking moving averages and maximum values.

## Key Experimental Results

### Main Results

| Model | Steps | Resolution | Image Reward |
|------|------|--------|-------------|
| SSD-1B-LCM (baseline)| 2    | 1024²  | 0.781       |
| + GORS-LCM           | 2    | 1024²  | ~0.85       |
| + RLCM (DDPO Variant)| 2    | 1024²  | Unstable/Failed |
| + PSO (DPO Variant)  | 2    | 1024²  | Blurry/Failed |
| + **LaSRO (Ours)**   | **2**| **1024²**| **~1.05**   |
| + LaSRO              | 1    | 1024²  | ~0.95       |
| SDXL-Turbo           | 2    | 512²   | 0.839       |
| + LaSRO              | 2    | 512²   | 0.957       |

### Ablation Study

The authors validated several key design choices: (1) SDXL UNet encoder vs. CLIP/BLIP as backbones—the UNet encoder is superior in both generalization and reward prediction accuracy; (2) Off-policy vs. on-policy-only exploration—the off-policy strategy significantly improves training stability and the final reward; (3) Online adaptation vs. fixed surrogate reward—online adaptation prevents reward hacking; (4) Simultaneously optimizing one-step and two-step vs. optimizing only two-step—optimizing both simultaneously produces better results.

### Key Findings

- DDPO and DPO-like methods fail almost completely on two-step DMs, validating the authors' analysis on RL objective degeneration and exploration obstacles.
- LaSRO is effective across three different rewards: Image Reward, Attribute Binding Score, and Text Alignment Score.
- LaSRO not only improves two-step generation quality but also simultaneously enhances one-step generation quality.
- The method is also applicable to other distilled models such as SDXL-Turbo, and is not limited to LCM.

## Highlights & Insights

- **Remarkably Deep Problem Analysis**: The paper devotes a significant portion to systematically analyzing the three major obstacles of RL on two-step DMs (exploration difficulty, objective degeneration, and non-smooth mapping). Each issue is supported by theoretical derivations and experimental validation, providing a solid motivational foundation for the proposed design.
- **Theoretical Connection to Value-Based RL**: The authors establish a correspondence between LaSRO and value-based RL—surrogate reward $\approx$ Q-function, LCM optimization $\approx$ policy guided by value, and off-policy sampling corresponding to Q-learning exploration. This imbues the method with theoretical depth.
- **Simple and Efficient Surrogate Reward Design**: Leveraging the pre-existing pre-trained UNet encoder and using only a lightweight CNN head avoids training a large reward model from scratch.

## Limitations & Future Work

- Currently, validation is primarily conducted on the SDXL series of models; whether it generalizes to newer architectures like Flux or SD3 remains to be explored.
- Pre-training the surrogate reward model incurs additional computational overhead, although it is relatively small compared to the fine-tuning process itself.
- The paper does not discuss multi-objective reward optimization scenarios.
- Future directions include exploring the application of LaSRO to more-step (e.g., 4-step, 8-step) distilled models, as well as reward fine-tuning for video generation models.

## Related Work & Insights

- **DDPO/RLCM**: DDPO is a classic policy gradient method, and RLCM is its adaptation for LCMs. Both fail in the two-step scenario due to objective degeneration.
- **Diffusion-DPO/PSO**: Preference-ranking-based methods fail on distilled models because they rely on reward-weighted regression with diffusion loss, which disrupts the mapping and leads to blurriness.
- **ReFL/DRaFT**: Backpropagation methods based on differentiable rewards, which require the rewards themselves to be differentiable and entail propagating gradients back through the sampling process, making them computationally expensive.
- **Insights**: The concept of surrogate rewards can be generalized to other scenarios requiring the fine-tuning of generative models, such as video diffusion models and 3D generation.

## Rating

- Novelty: ⭐⭐⭐⭐ — Highly original problem analysis and surrogate reward scheme.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Tested across three rewards, multiple baselines, and complete ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear problem analysis and rigorous logic.
- Value: ⭐⭐⭐⭐ — Establishes a viable path for the alignment and fine-tuning of ultra-fast generative models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] ReNeg: Learning Negative Embedding with Reward Guidance](reneg_learning_negative_embedding_with_reward_guidance.md)
- [\[CVPR 2025\] Personalized Preference Fine-tuning of Diffusion Models](personalized_preference_fine-tuning_of_diffusion_models.md)
- [\[CVPR 2025\] Trust Your Critic: Robust Reward Modeling and Reinforcement Learning for Faithful Image Editing and Generation](trust_your_critic_robust_reward_modeling_and_reinforcement_learning_for_faithful.md)
- [\[CVPR 2026\] Reward Sharpness-Aware Fine-Tuning for Diffusion Models](../../CVPR2026/image_generation/reward_sharpness-aware_fine-tuning_for_diffusion_models.md)
- [\[CVPR 2025\] Focus-N-Fix: Region-Aware Fine-Tuning for Text-to-Image Generation](focus-n-fix_region-aware_fine-tuning_for_text-to-image_generation.md)

</div>

<!-- RELATED:END -->
