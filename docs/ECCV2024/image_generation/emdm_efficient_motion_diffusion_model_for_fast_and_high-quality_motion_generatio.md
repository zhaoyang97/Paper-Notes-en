---
title: >-
  [Paper Note] EMDM: Efficient Motion Diffusion Model for Fast and High-Quality Motion Generation
description: >-
  [ECCV 2024][Image Generation][Human motion generation] EMDM is proposed to capture complex denoising distributions under large step sizes through a conditional denoising diffusion GAN. It enables real-time generation of high-quality human motions with no more than 10 sampling steps, improving inference speed by approximately 200 times compared to MDM.
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Human motion generation"
  - "diffusion models"
  - "GAN"
  - "efficient sampling"
  - "text-driven motion"
date: 2026-05-08
content_hash: 9fde258e571eb87d
---

# EMDM: Efficient Motion Diffusion Model for Fast and High-Quality Motion Generation

**Conference**: ECCV 2024  
**arXiv**: [2312.02256](https://arxiv.org/abs/2312.02256)  
**Code**: [GitHub](https://github.com/Frank-ZY-Dou/EMDM)  
**Area**: Image Generation  
**Keywords**: Human motion generation, diffusion models, GAN, efficient sampling, text-driven motion

## TL;DR

EMDM is proposed to capture complex denoising distributions under large step sizes through a conditional denoising diffusion GAN. It enables real-time generation of high-quality human motions with no more than 10 sampling steps, improving inference speed by approximately 200 times compared to MDM.

## Background & Motivation

While current diffusion-model-based human motion generation methods (such as MDM, MotionDiffuse) perform exceptionally well in generation quality, their inference speed is extremely slow: MDM takes about 12 seconds to generate a motion sequence corresponding to a text description, which severely limits practical applications (such as online motion synthesis, game development).

Existing acceleration schemes have obvious limitations:

**Latent Diffusion (MLD)**: Learns the latent space of motion first and then performs latent diffusion. However, in this two-stage approach, the quality of the latent space directly limits downstream generation performance, and end-to-end training is not possible.

**DDIM Accelerated Sampling**: When simply increasing the sampling step size, the denoising distribution transitions from a Gaussian distribution to a complex multimodal distribution. Consequently, the Gaussian assumption no longer holds, leading to a significant drop in generation quality.

The core problem is that when the number of sampling steps is reduced, the denoising distribution at each step becomes non-Gaussian and complex, which existing methods fail to model effectively. The motivation of EMDM is to capture this complex distribution using conditional GANs, thereby maintaining high-quality generation under fewer sampling steps.

## Method

### Overall Architecture

The core idea of EMDM is to replace the Gaussian denoising assumption in standard diffusion models with a **conditional denoising diffusion GAN**. The overall framework consists of two core components:

1. **Conditional Generator** $G_\theta(\mathbf{x}_t, \mathbf{z}, \mathbf{c}, t)$: Takes the noisy motion $\mathbf{x}_t$, a random variable $\mathbf{z} \sim \mathcal{N}(0, I)$ (64-dimensional), control signals $\mathbf{c}$ (text or action labels), and time step $t$ as inputs, and outputs the predicted clean motion $\hat{\mathbf{x}}_0$.
2. **Conditional Discriminator** $D_\phi(\mathbf{x}_{t-1}, \mathbf{x}_t, \mathbf{c}, t)$: Determines whether $\mathbf{x}_{t-1}$ is a reasonable denoising result of $\mathbf{x}_t$.

During inference, the model requires no more than 10 steps to generate high-quality motion sequences from noise.

### Key Designs

#### Distribution Modeling of the Conditional Generator

The core formulation constructs the denoising distribution through the generator and posterior sampling:

$$p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t) = \int p(\mathbf{z}) q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0 = G_\theta(\mathbf{x}_t, \mathbf{z}, \mathbf{c}, t)) d\mathbf{z}$$

The key here is that the generator models the diversity of the multimodal distribution using an additional random variable $\mathbf{z}$. Unlike standard diffusion models that assume the distribution of each denoising step is Gaussian, EMDM implicitly captures arbitrarily complex distribution shapes through the adversarial training of the GAN.

The workflow is as follows:
1. The generator first predicts $\hat{\mathbf{x}}_0 = G_\theta(\mathbf{x}_t, \mathbf{z}, \mathbf{c}, t)$.
2. It then samples $\hat{\mathbf{x}}_{t-1}$ through the posterior distribution $q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)$.
3. The discriminator distinguishes the real $({\mathbf{x}_{t-1}}, {\mathbf{x}_t})$ pair from the generated $(\hat{\mathbf{x}}_{t-1}, {\mathbf{x}_t})$ pair.

#### Conditional Discriminator

The discriminator depends not only on the time step $t$ but is also conditioned on the control signal $\mathbf{c}$. The training objective is:

$$\min_\phi \sum_{t \geq 1} \mathbb{E}_{q(\mathbf{x}_t)} [\mathbb{E}_{q(\mathbf{x}_{t-1}|\mathbf{x}_t)} [F(-D_\phi)] + \mathbb{E}_{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)} [F(D_\phi)]]$$

where $F(\cdot) = \text{softplus}(\cdot)$. The discriminator adopts a 7-layer MLP architecture.

#### Geometric Loss Functions

The authors found that relying solely on the adversarial loss of the GAN is insufficient for generating high-quality human motion, and motion-domain-specific geometric constraints must be introduced. The geometric loss consists of four parts:

| Loss Name | Formulation Meaning | Function |
|---------|---------|------|
| $\mathcal{L}_{\text{recon}}$ | L2 distance between predicted and ground-truth motion | Overall reconstruction quality |
| $\mathcal{L}_{\text{pos}}$ | Joint position error transformed via forward kinematics | Joint position accuracy |
| $\mathcal{L}_{\text{foot}}$ | Displacement constraint during foot contact | Reduce foot sliding artifacts |
| $\mathcal{L}_{\text{vel}}$ | Consistency constraint on joint velocity | Motion smoothness |

The total geometric loss is $\mathcal{L}_{\text{geo}} = \mathcal{L}_{\text{recon}} + \lambda(\mathcal{L}_{\text{pos}} + \mathcal{L}_{\text{vel}} + \mathcal{L}_{\text{foot}})$, where $\lambda$ is set to 1 in action-to-motion tasks and 0 in text-to-motion tasks.

#### Classifier-free Guidance

EMDM employs classifier-free guidance: during training, the condition is randomly set to empty $\mathbf{c} = \emptyset$ with a 10% probability. During inference, diversity and fidelity are balanced by interpolating between the two generation results:

$$G_s = G(\mathbf{x}_t, \mathbf{z}, \emptyset, t) + s \cdot (G(\mathbf{x}_t, \mathbf{z}, \mathbf{c}, t) - G(\mathbf{x}_t, \mathbf{z}, \emptyset, t))$$

### Loss & Training

The final training objective is: $\min_\theta (\mathcal{L}_{\text{disc}} + R \cdot \mathcal{L}_{\text{geo}})$

- Generator: 12-layer Transformer, 32 attention heads, with skip connections
- Text Encoder: Frozen CLIP-ViT-L-14
- Optimizer: AdamW, learning rate of $2 \times 10^{-5}$ (text-to-motion) / $3 \times 10^{-5}$ (action-to-motion)
- EMA decay is utilized
- Batch size: 64
- End-to-end training, no stages required

## Key Experimental Results

### Main Results

**Text-to-motion (HumanML3D)**

| Method | FID↓ | R-Prec Top1↑ | MM Dist↓ | Time per frame (ms)↓ | End-to-End |
|------|------|-------------|----------|-------------|----------|
| MDM | 0.508 | 0.418 | 3.630 | 62.505 | ✓ |
| MLD | 0.473 | 0.481 | 3.196 | 0.598 | ✗ |
| MotionDiffuse | 0.630 | 0.491 | 3.113 | 38.235 | ✓ |
| ReMoDiffuse | 0.103 | 0.510 | 2.974 | 0.959 | ✗ |
| **EMDM** | **0.112** | **0.498** | **3.110** | **0.280** | **✓** |

**Action-to-motion (HumanAct12)**

| Method | FID↓ | ACC↑ | Time per frame (ms)↓ | End-to-End |
|------|------|------|-------------|--------|
| MDM | 0.100 | 0.990 | 41.154 | ✓ |
| MLD | 0.077 | 0.964 | 1.998 | ✗ |
| ACTOR | 0.120 | 0.955 | 0.523 | ✓ |
| **EMDM** | **0.084** | **0.991** | **0.337** | **✓** |

### Ablation Study

**Effect of Sampling Steps (HumanML3D)**:

| Sampling Steps | FID↓ | R-Prec Top1↑ | Time per frame (ms) |
|---------|------|-------------|------------|
| 1 (Pure GAN) | 5.640 | 0.345 | 0.004 |
| 5 | 1.306 | 0.368 | 0.152 |
| 10 (Default) | 0.112 | 0.498 | 0.280 |
| 20 | - | - | - |
| 50 | - | - | - |

Sampling with 1 step degenerates into a pure GAN, leading to a substantial drop in quality; 10 steps serves as the optimal balance point.

### Key Findings

1. **Significant Speed Advantage**: EMDM requires only 0.280 ms per frame on HumanML3D, which is about 200 times faster than MDM (62.5 ms).
2. **No Compromise on Quality**: The FID is only 0.112, outperforming MLD (0.473) and approaching ReMoDiffuse (0.103), which utilizes an extra retrieval database.
3. **Geometric Loss is Crucial**: Without the geometric loss, GAN training becomes unstable, and the generated motion quality significantly declines.
4. **Conditional Signals Enhance GAN Performance**: The conditional control signals provide a strong prior, making the modeling of complex distributions more efficient.

## Highlights & Insights

1. **Deep Theoretical Insight**: It clearly reveals the fundamental issue that the denoising distribution becomes non-Gaussian under large step sizes, and proposes the implicit distribution modeling capability of GANs as the solution.
2. **End-to-End Training**: Compared to the two-stage scheme of MLD, the end-to-end training pipeline of EMDM significantly simplifies practical deployment.
3. **Integration Paradigm of Conditional GAN + Diffusion Model**: Instead of simply applying DDGAN, it introduces motion-specific conditional signals and geometric constraints to adapt it to the motion generation task.
4. **High Practical Value**: The real-time generation capability makes it directly applicable to real-time scenarios such as gaming and VR.

## Limitations & Future Work

1. It is only validated on relatively small motion datasets, without testing its generalization capability on large-scale datasets.
2. Although 10-step sampling is already very fast, whether it can be further compressed to 1–2 steps without quality degradation is worth exploring.
3. The discriminator uses a simple MLP, which may limit its ability to distinguish complex motion distributions.
4. Human joint position and foot contact losses are not used in text-to-motion ($\lambda=0$), leaving potential room for further improvement.
5. Discussion on long-sequence motion generation is lacking.

## Related Work & Insights

- **DDGAN**: The core inspiration source of this work, transferring the diffusion GAN paradigm from image to motion generation domain.
- **MDM**: Pioneering work in motion diffusion models, but the 1000-step sampling is too slow.
- **MLD**: Accelerates speed via latent diffusion, but the two-stage training remains a bottleneck.
- **Insight**: The concept of combining GANs with diffusion models can be extended to other sequence generation tasks requiring fast sampling (such as dance generation and gesture generation).

## Rating

| Dimension | Score (1-5) |
|------|-----------|
| Novelty | 4 |
| Theoretical Depth | 4 |
| Experimental Thoroughness | 4 |
| Practical Value | 5 |
| Writing Quality | 4 |
| Overall Rating | 4.2 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Local Action-Guided Motion Diffusion Model for Text-to-Motion Generation](local_action-guided_motion_diffusion_model_for_text-to-motion_generation.md)
- [\[ECCV 2024\] SMooDi: Stylized Motion Diffusion Model](smoodi_stylized_motion_diffusion_model.md)
- [\[ECCV 2024\] MotionLCM: Real-time Controllable Motion Generation via Latent Consistency Model](motionlcm_real-time_controllable_motion_generation_via_latent_consistency_model.md)
- [\[ECCV 2024\] Prompting Future Driven Diffusion Model for Hand Motion Prediction](prompting_future_driven_diffusion_model_for_hand_motion_prediction.md)
- [\[ECCV 2024\] Realistic Human Motion Generation with Cross-Diffusion Models](realistic_human_motion_generation_with_cross-diffusion_models.md)

</div>

<!-- RELATED:END -->
