---
title: >-
  [Paper Note] Instant Adversarial Purification with Adversarial Consistency Distillation
description: >-
  [CVPR 2025][Image Generation][adversarial purification] This work proposes the One Step Control Purification (OSCP) framework, which integrates Gaussian Adversarial Noise Distillation (GAND) and Controlled Adversarial Purification (CAP) to achieve adversarial purification within a single U-Net inference step (~0.1 seconds), yielding a 100x acceleration compared to traditional diffusion-based purification methods.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "adversarial purification"
  - "consistency distillation"
  - "OSCP"
  - "GAND"
  - "ControlNet"
  - "one-step defense"
date: 2026-05-08
content_hash: c63d53764839af09
---

# Instant Adversarial Purification with Adversarial Consistency Distillation

**Conference**: CVPR 2025  
**arXiv**: [2408.17064](https://arxiv.org/abs/2408.17064)  
**Code**: To be confirmed  
**Area**: image_generation (adversarial defense / diffusion models)  
**Keywords**: adversarial purification, consistency distillation, OSCP, GAND, ControlNet, one-step defense

## TL;DR

This work proposes the One Step Control Purification (OSCP) framework, which integrates Gaussian Adversarial Noise Distillation (GAND) and Controlled Adversarial Purification (CAP) to achieve adversarial purification within a single U-Net inference step (~0.1 seconds), yielding a 100x acceleration compared to traditional diffusion-based purification methods.

## Background & Motivation

**Background**: Diffusion-based adversarial purification (e.g., DiffPure) defends against attacks by mapping adversarial examples back to the natural distribution. Although showing promising performance, multiple denoising steps result in heavy computational overhead (~10 seconds per image).

**Limitations of Prior Work**:
1. **Diffusion purification is too slow**: DiffPure and GDMP require 100+ iterative denoising steps, taking 9-11 seconds per image, making them impractical for real-time applications.
2. **Adversarial training overfits seen attacks**: Adversarial training has poor generalization capabilities to unseen attacks.
3. **Semantic loss from large purification steps**: Increasing the diffusion step $t^*$ can remove adversarial noise more thoroughly but causes the generated images to deviate from the original semantics.

**Key Challenge**: The core assumption of the Consistency Model is that $f_\theta(z_t, t) = f_\theta(z_{t'}, t')$ holds for all $t, t' \in [0,T]$. However, the latent distribution shift in adversarial images violates this consistency constraint, preventing direct purification with LCM from converging to clean images.

**Goal**: To achieve single-step inference while maintaining the effectiveness of diffusion purification, overcoming the difficulties in consistency distillation caused by the distribution discrepancy between adversarial and Gaussian noise.

**Key Insight**: To address the problem from both the distillation objective and the inference pipeline: GAND modifies the distillation objective to adapt to adversarial noise, while CAP leverages edge guidance to preserve semantics.

## Method

### Overall Architecture

OSCP consists of two components:
1. **GAND (Gaussian Adversarial Noise Distillation)**: Training phase—modifies the LCM distillation objective so that the model learns to remove both Gaussian noise and adversarial noise simultaneously.
2. **CAP (Controlled Adversarial Purification)**: Inference phase—uses Canny edge maps to guide the purification process through ControlNet, preventing semantic drift under large purification steps.

Inference Pipeline: Adversarial image $\mathbf{x}_{adv}$ → VAE encoding $\mathbf{z}_{adv}$ → Forward diffusion adding noise up to $t^*$ → One-step denoising via GAND-LCM (with ControlNet edge guidance) → VAE decoding → Purified image.

### Key Designs

**1. Gaussian Adversarial Noise Distillation (GAND)**
- **Function**: Based on Latent Consistency Distillation, this explicitly introduces adversarial noise into the noise term of the forward process.
- **Mechanism**: Define a new diffusion forward process:
  $$z_t^* = \sqrt{\bar{\alpha}_t} z + \sqrt{1 - \bar{\alpha}_t}(\epsilon + \delta_{adv})$$
  where $\delta_{adv}$ is the adversarial perturbation generated via PGD attacks in the latent space. Key observation: $z_t^* \to z$ as $t \to 0$ (converging to the clean image), and $z_t^* \to \epsilon + \delta_{adv}$ as $t \to T$ (converging to mixed noise). In this way, the consistency constraint of LCM is re-established within $[0, t]$.
- **Distillation loss**:
  $$\mathcal{L}_{Total} = \mathcal{L}_{GAND}(\theta, \theta^-) + \lambda_{CIG} \mathcal{L}_{CIG}(\theta)$$
  - $\mathcal{L}_{GAND}$: Consistency distillation loss, acting on the diffusion trajectory containing adversarial noise.
  - $\mathcal{L}_{CIG}$: Clear Image Guide loss, which directly constrains the output to converge toward the clean latent.
- **Efficient Training**: Employs Parameter-Efficient Fine-Tuning with LoRA, avoiding the substantial computational overhead of full-parameter training.
- **Design Motivation**: Deploying LCM directly for adversarial purification fails because the distribution shift of the adversarial latent $z_{adv}(t)$ violates the convergence condition of the consistency function. By explicitly incorporating $\delta_{adv}$ into the forward diffusion process, the distillation trajectory directly models the mapping from "mixed noise to the clean image".

**2. Controlled Adversarial Purification (CAP)**
- **Function**: Extract the edge map of the adversarial image using a non-learnable Canny edge detector during inference, guiding the purification process via ControlNet.
- **Mechanism**: 
    - The edge map provides a structural prior to prevent the output image from deviating from the geometric structure of the original image under large purification steps.
    - Remove the skip connection term $c_{skip}(t)z_{adv}(t)$ of LCM (i.e., setting $c_{skip} \equiv 0$) because this term retains adversarial noise.
    - Final purification formulation: $\hat{z}_{adv}^0 = c_{out}(t) \cdot \frac{z_{adv} - \sqrt{1-\bar{\alpha}_t}\hat{\epsilon}_\theta(z_{adv}, c_{ce}, t)}{\sqrt{\bar{\alpha}_t}}$
- **Design Motivation**: A non-learnable edge detector is preferred over text prompt guidance since text can be easily manipulated by semantic caption attacks. The skip connection is removed because it directly propagates adversarial noise.

**3. Latent Space Adversarial Training**
- **Function**: The adversarial perturbation $\delta_{adv}$ during GAND training is generated in the latent space rather than the pixel space.
- **Mechanism**: $\delta_{adv} = \arg\max_\delta \mathcal{L}(C(\mathcal{D}(\mathcal{E}(x) + \delta)), y)$, using PGD-10 attacks.
- **Design Motivation**: Latent-space attacks align with the operational workspace of the diffusion model and avoid changes in distribution after passing the pixel-space attack through VAE encoding.

### Loss & Training

- Distillation data: First 40K images of the ImageNet validation set ($512 \times 512$).
- Base model: Stable Diffusion v1.5.
- Training details: 20K iterations, batch size of 4, learning rate of 8e-6, 500-step warm-up.
- Fine-tuning: LoRA fine-tuning, leveraging DDIM solver as the PF-ODE solver, with skip step $k=20$.
- Adversarial perturbation: PGD-10, $\epsilon=0.03$, targeting ResNet50.
- Hyperparameter: $\lambda_{CIG} = 0.001$.
- Inference settings: $t^* = 200$ (the optimal purification intensity).

## Key Experimental Results

### Main Results (ImageNet)

| Method | Category | Attack Method | Standard Acc↑ | Robust Acc↑ |
|---|---|---|---|---|
| Without defense | - | AutoAttack | 80.55% | 0.00% |
| DiffPure | DBP | AutoAttack | 75.77% | 73.02% |
| Amini et al. | Adv Train | AutoAttack | 77.96% | 59.64% |
| **OSCP (Ours)** | **Hybrid** | **AutoAttack** | **77.63%** | **74.19%** |
| **OSCP (Ours)** | **Hybrid** | **PGD-100** | **77.63%** | **73.89%** |

### Inference Speed Comparison

| Method | Dataset | Time per Image |
|---|---|---|
| GDMP | ImageNet | ~9s |
| DiffPure | ImageNet | ~11s |
| **OSCP (Ours)** | **ImageNet** | **~0.1s** |

**100x acceleration**, with an inference time independent of $t^*$ (one-step inference).

### Cross-Architecture Generalization (PGD-100 Attack)

| Architecture | Clean ASR | Robust Acc↑ |
|---|---|---|
| ResNet-50 | 100% | 73.89% |
| WRN-50-2 | 100% | 75.2% |
| ViT-b-16 | 100% | 71.6% |
| Swin-b | 100% | 77.8% |

### Adaptive Attack Defense (Diff-PGD-10)

| Method | ResNet-50 | ViT-b-16 | Swin-b |
|---|---|---|---|
| DiffPure | 53.8% | 16.6% | 45.1% |
| **OSCP** | **59.0%** | **34.1%** | **53.9%** |

### Key Findings

1. **Single-step inference without sacrificing defense performance**: OSCP achieves better robust accuracy on AutoAttack (74.19%) than DiffPure (73.02%), while being 100x faster.
2. **Excellent cross-architecture transferability**: GAND trained only using ResNet50 adversarial examples generalizes well to other architectures like ViT and Swin (achieving at least 71.6%).
3. **Robust against adaptive attacks**: When facing Diff-PGD attacks specifically targeting diffusion purification, OSCP still outperforms DiffPure by 5% to 17%.
4. **LoRA fine-tuning is sufficient**: Eliminating the need for full-parameter training, parameter-efficient fine-tuning achieves optimal performance.

## Highlights & Insights

- **Deep core insight**: Correctly identifies the root cause of consistency constraint violation in LCM under adversarial scenarios (the distribution shift of $z_{adv}$), resolving it elegantly by modifying the forward process.
- **Exquisite design of GAND**: $z_t^* = \sqrt{\bar{\alpha}_t}z + \sqrt{1-\bar{\alpha}_t}(\epsilon + \delta_{adv})$ gracefully forces the two boundaries of the distillation trajectory to converge to the clean image and the mixed noise, respectively, successfully restoring consistency.
- **Robust conditional guidance**: CAP uses non-learnable edge detection instead of neural network conditioning, effectively preventing the adversarial examples from "poisoning" the guidance signal.
- **Hybrid defense paradigm**: Addresses the task from a hybrid perspective of "adversarial training + purification", merging the advantages of both defense paradigms.

## Limitations & Future Work

- The adversarial perturbation during training is only generated with PGD-10; it may need extension when confronting stronger or more diverse attack strategies.
- ControlNet edge guidance might be less effective against texture-based attacks (where the edge detector is unaffected but texture information is lost).
- Grounded on SD 1.5, newer diffusion backbones (e.g., SDXL) have not yet been evaluated.
- Standard accuracy drops from 80.55% to 77.63%, representing a clean accuracy loss of around 3%.
- Defense efficacy against $L_2$ norm attacks remains uninvestigated, as the evaluation focuses primarily on $L_\infty$.

## Related Work & Insights

- DiffPure establishes the theoretical foundation of diffusion purification (reducing KL divergence), while OSCP compresses it from multi-step inference to a single step.
- LCM/LCM-LoRA realizes rapid inference for diffusion models, which OSCP integrates for the first time into adversarial defense.
- The conditional control capability of ControlNet is creatively applied to maintain semantic consistency post-purification.
- **Inspiration**: The paradigm of "adversarial-aware calibration" of distillation targets could be generalized to other novel scenarios, such as video adversarial purification and 3D adversarial defense.

## Rating

⭐⭐⭐⭐ — Outstanding theoretical analysis on why LCM consistency fails under adversarial settings. The design of GAND+CAP is theoretically sound with excellent experimental results, and the 100x speedup holds significant practical value. The main limitations lie in attack coverage and clean accuracy loss.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Divide and Conquer: Heterogeneous Noise Integration for Diffusion-based Adversarial Purification](divide_and_conquer_heterogeneous_noise_integration_for_diffusion-based_adversari.md)
- [\[CVPR 2025\] Detecting Adversarial Data Using Perturbation Forgery](detecting_adversarial_data_using_perturbation_forgery.md)
- [\[CVPR 2025\] IDProtector: An Adversarial Noise Encoder to Protect Against ID-Preserving Image Generation](idprotector_an_adversarial_noise_encoder_to_protect_against_id-preserving_image_.md)
- [\[ECCV 2024\] Learning Differentially Private Diffusion Models via Stochastic Adversarial Distillation](../../ECCV2024/image_generation/learning_differentially_private_diffusion_models_via_stochastic_adversarial_dist.md)
- [\[ICML 2026\] Adversarial Flow Models](../../ICML2026/image_generation/adversarial_flow_models.md)

</div>

<!-- RELATED:END -->
