---
title: >-
  [Paper Note] AdaGen: Learning Adaptive Policy for Image Synthesis
description: >-
  [ECCV 2024 (Extended version of AdaNAT)][Image Generation][Reinforcement Learning] This paper unifies step-level parameter scheduling (temperature, mask ratio, CFG scale, timestep, etc.) of multi-step generative models (MaskGIT/AR/Diffusion/Rectified Flow) as an MDP. A lightweight RL policy network is used to achieve sample-adaptive scheduling, and an adversarial reward design is proposed to prevent policy overfitting, consistently improving performance across four generative…
tags:
  - "ECCV 2024 (Extended version of AdaNAT)"
  - "Image Generation"
  - "Reinforcement Learning"
  - "Adaptive Scheduling Policy"
  - "Adversarial Reward"
  - "Multi-paradigm Generative Models"
  - "Inference Optimization"
date: 2026-05-08
content_hash: c9128344647776d4
---

# AdaGen: Learning Adaptive Policy for Image Synthesis

**Conference**: ECCV 2024 (Extended version of AdaNAT)  
**arXiv**: [2603.06993](https://arxiv.org/abs/2603.06993)  
**Code**: [https://github.com/LeapLabTHU/AdaGen](https://github.com/LeapLabTHU/AdaGen)  
**Area**: Image Generation / Generative Model Optimization  
**Keywords**: Reinforcement Learning, Adaptive Scheduling Policy, Adversarial Reward, Multi-paradigm Generative Models, Inference Optimization  

## TL;DR

This paper unifies step-level parameter scheduling (temperature, mask ratio, CFG scale, timestep, etc.) of multi-step generative models (MaskGIT/AR/Diffusion/Rectified Flow) as an MDP. A lightweight RL policy network is used to achieve sample-adaptive scheduling, and an adversarial reward design is proposed to prevent policy overfitting, consistently improving performance across four generative paradigms (e.g., VAR FID $1.92 \rightarrow 1.59$, and reducing the inference cost of DiT-XL by 3x with superior performance).

## Background & Motivation

Modern image generative models (MaskGIT, Autoregressive VAR, Diffusion DiT, Rectified Flow SiT) share the common characteristic of decomposing the generation process into multi-step iterations, where each step requires configuring multiple parameters (such as mask ratio, sampling temperature, CFG scale, ODE timestep, etc.). For instance, a 32-step MaskGIT generation requires configuring 128 policy parameters.

Existing practices rely on manually designed static scheduling rules (such as cosine schedules or fixed constants), which suffer from two core pain points:
1. **High Cost of Manual Tuning**: The parameter combination space grows exponentially with the number of steps, requiring expert knowledge and extensive trial-and-error.
2. **Suboptimal Static Scheduling**: All samples share the same schedule, making it impossible to adaptively adjust based on the characteristics of each individual sample (such as simple vs. complex structures).

## Core Problem

Can we use a **universal, learnable, and sample-adaptive** framework to automatically configure the iterative policy of multi-step generative models? The key challenges lie in: (1) end-to-end backpropagation is computationally prohibitive, and some operations are non-differentiable; (2) how to design an effective reward signal to guide policy learning, as simple FIDs or pre-trained reward models are easily "hacked".

## Method

### Overall Architecture

Input: Pre-trained generative model (frozen) + current intermediate generation state $\rightarrow$ lightweight policy network outputs generation policy parameters for the current step $\rightarrow$ generative model executes one generation step $\rightarrow$ iterations repeat until completion $\rightarrow$ adversarial reward model evaluates final image quality and provides RL training signals.

Core Idea: **Keep the generator intact and only train a side-pass policy network to "direct" the generation process**. The policy network is trained via PPO, and the reward model is trained adversarially with the policy network.

### Key Designs

1. **Unified MDP Modeling**: The scheduling problem of four generative paradigms (MaskGIT, AR, Diffusion ODE, Rectified Flow) is unified into an MDP. State = (current step $t$, intermediate generation result), Action = policy parameter vector of the current step, Reward is only provided at the final step. Differences among paradigms lie in the state transition function—MaskGIT/AR uses stochastic transitions, while Diffusion/Flow uses deterministic ODE solvers. The policy network architecture consists of Conv+MLP+AdaLN (injecting step information) and uses intermediate features of the generative model instead of raw intermediate results as input, making it extremely lightweight (comprising only 0.07% to 0.40% of the generator's computation).

2. **Adversarial Reward Model**: This is the key design of the paper. The authors observe that using FID or pre-trained reward models (e.g., ImageReward) directly as RL rewards leads to policy overfitting—resulting in improved FIDs but poor actual image quality, or collapsed generation diversity with a single style. The solution is to introduce a GAN-style discriminator as the reward model, which is trained adversarially with the policy network: while the policy network maximizes the reward, the discriminator is updated simultaneously to better distinguish real and fake images. This makes the reward signal dynamic, effectively preventing the policy from overfitting to a static target.

3. **Action Smoothing**: When the number of generation steps increases (e.g., $T=32$), the action sequences of the policy network exhibit severe fluctuations, leading to unstable training. The authors find that these high-frequency fluctuations are unnecessary—simple linear interpolation actually yields better results. This is analyzed to be caused by PPO's Gaussian exploration adding noise independently at each step, creating unreasonable exploration trajectories. The solution is to apply an EMA (Exponential Moving Average, $\beta=0.8$) filter to the policy output, which acts as a causal low-pass filter that smooths the action sequence while maintaining the Markovian property of the MDP.

4. **Inference-Time Refinement**: The auxiliary networks after training can be reused: (a) repeated sampling using the adversarial reward model (generating multiple times and selecting the one with the highest score); (b) for stochastic transition models (like MaskGIT), lookahead sampling using the value network $V$ (sampling $K$ candidate states at each step and proceeding with the one with the highest $V$-value). The combination of both reduces the FID of MaskGIT-L from 2.28 to 1.81.

5. **Controllable Fidelity-Diversity Trade-off**: A second policy network focusing on fidelity is introduced. The outputs of the two policies are linearly blended with a parameter $\lambda$, while combining the adversarial reward with ImageReward. $\lambda=0$ biases toward diversity, while $\lambda=1$ biases toward fidelity.

### Loss & Training

- The policy network is optimized using PPO (clipped surrogate objective), including advantage estimation and value function loss.
- The adversarial reward model is optimized using the minimax objective of a standard GAN:
$$\max_\phi \min_\psi \mathbb{E}[\log r_\psi(x_{\text{fake}})] + \mathbb{E}[\log(1-r_\psi(x_{\text{real}}))]$$
- The two are optimized alternately: first, the policy network is used to sample and update the policy, then the real/fake images are used to update the discriminator.
- Exploration noise $\sigma=0.6$, PPO clip $\epsilon=0.2$, value function coefficient $c=0.5$.
- Action smoothing with $\beta=0.8$ is activated when $T>10$.

## Key Experimental Results

| Dataset | Model | Metric | Baseline | AdaGen | Gain |
|--------|------|------|----------|--------|------|
| ImageNet 256 | DiT-XL (16 steps) | FID-50K | 3.31 | 2.19 | -1.12 |
| ImageNet 256 | DiT-XL (8 steps) | FID-50K | 5.18 | 2.82 | -2.36 |
| ImageNet 256 | SiT-XL (16 steps) | FID-50K | 2.99 | 2.12 | -0.87 |
| ImageNet 256 | VAR-d30 (10 steps) | FID-50K | 1.92 | 1.59 | -0.33 |
| ImageNet 256 | MaskGIT-L (16 steps) | FID-50K | 3.79 | 2.41 | -1.38 |
| ImageNet 512 | MaskGIT-L (32 steps) | FID-50K | 7.32 | 2.46 | -4.86 |
| MS-COCO | MaskGIT-S (16 steps) | FID-30K | 5.78 | 4.92 | -0.86 |
| LAION-5B→COCO | Stable Diffusion (32 steps) | FID-30K | 9.03 | 8.14 | -0.89 |

System-level Comparison: AdaGen-DiT-XL with 16 steps (4.1 TFLOPs) achieves FID=2.19, which is superior to the original DiT-XL with 50 steps (12.2 TFLOPs) which gets FID=2.29, reducing the inference cost by approximately 3x.

Inference-Time Refinement: MaskGIT-L + repeated + lookahead sampling: FID $2.28 \rightarrow 1.81$.

### Ablation Study

- **Incremental Contribution of Learnable vs. Adaptive**: Learnable (non-adaptive) already yields a clear improvement (MaskGIT FID $7.65 \rightarrow 5.40$), and adaptive scheduling further boosts the performance (FID $5.40 \rightarrow 4.54$). Both designs are essential.
- **Reward Design**: Using FID as the reward leads to blurry images; using PretrainedRewardModel results in a loss of diversity; whereas the adversarial reward achieves the best balance between quality and diversity.
- **Action Smoothing**: $\beta = 0.8$ is optimal; $\beta = 0$ (no smoothing) yields FID = 3.97, while $\beta=0.8$ yields FID = 3.36.
- **Policy Network Input**: Using the generator's intermediate features (FID 4.54) is far superior to using raw intermediate results (FID 6.55).
- **Step Conditioning**: Removing step conditioning degrades the FID from 4.54 to 6.13, indicating that the policy network must know "which step it currently is".
- **Policy Network Architecture**: Conv+MLP (12M parameters) is already sufficient, and scaling up the model size yields minimal marginal gains.
- **Discriminator Architecture**: Transformer-based (FID 4.54) is significantly better than Conv-based (FID 5.86).
- **PPO Hyperparameters**: The policy is insensitive to $\epsilon$, $c$, and $\sigma$, with all settings significantly outperforming the baseline.

## Highlights & Insights

- The **"keep the model, only change the scheduling" paradigm** is highly elegant—the policy network consumes only 0.07% to 0.40% of the generator's computation, yet delivers 17% to 54% performance improvements or 1.6x to 3.6x inference acceleration, offering tremendous practical value.
- The **adversarial reward design** precisely addresses the reward overfitting problem when using RL to optimize generative models. This insight provides valuable reference for the broader field of RLHF for vision.
- The **signal processing perspective of Action Smoothing** is highly clever—solving high-frequency noise issues in RL policy exploration with a classical IIR filter, which is both simple and effective.
- The design choice of **unifying four generative paradigms** into an MDP is elegant, and the summaries in Table I and Table II are extremely well-structured.
- Reusing the training byproducts (discriminator + value network) for refinement during inference avoids wasting any training artifacts.

## Limitations & Future Work

- Although the policy network itself is lightweight, the RL training process requires intensive sampling, and the training cost is not discussed in detail.
- Whether the training stability of the adversarial reward still holds on large-scale models (such as SD-XL, FLUX) remains to be validated.
- Currently, it only handles inference-side scheduling parameters (such as CFG scale, timestep) and does not explore deeper adaptive model architectures (such as layer skipping, token selection, etc.).
- The integration and comparison with recent distillation methods (such as consistency distillation) are insufficient.

## Related Work & Insights

- **vs AdaNAT (ECCV 2024)**: AdaGen is a direct extension of AdaNAT—AdaNAT was only evaluated on MaskGIT, while AdaGen scales to four paradigms and introduces action smoothing, inference-time refinement, and fidelity-diversity control mechanisms.
- **vs Diffusion RLHF methods like DDPO/DRaFT**: These methods fine-tune the parameters of the generator itself using RL, whereas AdaGen freezes the generator and only trains a side-by-channel policy network, incurring much lower computational costs without degrading original model capability. Additionally, AdaGen's adversarial reward addresses the diversity loss issue caused by pre-trained reward models.
- **vs Automatic Sampler Search (AutoDiffusion, USF, etc.)**: These methods learn a globally shared static schedule and lack sample-adaptive capability; indeed, most of them are only designed for diffusion models.

## Insights & Connections

- The concept of "keep the model, only change the scheduling" can be orthogonally combined with token/cache-level inference acceleration methods such as [DPCache](../../CVPR2026/image_generation/dpcache_denoising_path_planning_diffusion_accel.md).

## Rating

- Novelty: ⭐⭐⭐⭐ The adversarial reward and unified MDP modeling are innovative, but the core RL-based scheduling idea is already established in AdaNAT.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely thorough, evaluated on 4 paradigms $\times$ 5 datasets along with extensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ Well-structured with excellent unified modeling visualization in Table I/II, and a step-by-step methodology presentation.
- Value: ⭐⭐⭐⭐ High practical value (0.07% overhead for significant performance gains), though its performance on larger models remains to be verified.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] AdaNAT: Exploring Adaptive Policy for Token-Based Image Generation](adanat_exploring_adaptive_policy_for_token-based_image_generation.md)
- [\[ECCV 2024\] Editable Image Elements for Controllable Synthesis](editable_image_elements_for_controllable_synthesis.md)
- [\[ECCV 2024\] Powerful and Flexible: Personalized Text-to-Image Generation via Reinforcement Learning](powerful_and_flexible_personalized_texttoimage_generation_vi.md)
- [\[CVPR 2026\] AHS: Adaptive Head Synthesis via Synthetic Data Augmentations](../../CVPR2026/image_generation/ahs_adaptive_head_synthesis.md)
- [\[ECCV 2024\] Rejection Sampling IMLE: Designing Priors for Better Few-Shot Image Synthesis](rejection_sampling_imle_designing_priors_for_better_few-shot_image_synthesis.md)

</div>

<!-- RELATED:END -->
