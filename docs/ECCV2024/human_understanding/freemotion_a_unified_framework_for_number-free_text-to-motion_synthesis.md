---
title: >-
  [Paper Note] FreeMotion: A Unified Framework for Number-free Text-to-Motion Synthesis
description: >-
  [ECCV 2024][Human Understanding][Text-to-Motion Generation] The FreeMotion framework is proposed to recursively decompose the joint distribution of multi-person motions into conditional single-person motion generation through conditional probability decomposition. This achieves text-driven motion synthesis for an arbitrary number of individuals for the first time, while supporting multi-person spatial control.
tags:
  - "ECCV 2024"
  - "Human Understanding"
  - "Text-to-Motion Generation"
  - "Multi-person Motion Synthesis"
  - "Diffusion Models"
  - "Conditional Motion Modeling"
  - "Spatial Control"
date: 2026-05-08
content_hash: 614fea0c9d7d7823
---

# FreeMotion: A Unified Framework for Number-free Text-to-Motion Synthesis

**Conference**: ECCV 2024  
**arXiv**: [2405.15763](https://arxiv.org/abs/2405.15763)  
**Code**: Yes ([https://VankouF.github.io/FreeMotion](https://VankouF.github.io/FreeMotion))  
**Area**: Human Understanding  
**Keywords**: Text-to-Motion Generation, Multi-person Motion Synthesis, Diffusion Models, Conditional Motion Modeling, Spatial Control

## TL;DR

The FreeMotion framework is proposed to recursively decompose the joint distribution of multi-person motions into conditional single-person motion generation through conditional probability decomposition. This achieves text-driven motion synthesis for an arbitrary number of individuals for the first time, while supporting multi-person spatial control.

## Background & Motivation

Text-driven human motion synthesis (Text-to-Motion, T2M) is an important task in computer vision with broad applications in robot control, film production, and animation. Existing methods have three key limitations:

**High Model Specialization**: Existing methods are designed either for single-person motion (e.g., MDM, MLD) or for two-person motion (e.g., InterGen), failing to support both single-person and multi-person inference simultaneously. Single-person methods fit the marginal distribution $p(\mathbf{x}^1)$, whereas two-person methods fit the joint distribution $p(\mathbf{x}^1, \mathbf{x}^2)$. The network architectures of the two are incompatible.

**Inscalable Number of Persons**: Due to the lack of multi-person motion datasets with textual annotations (the largest current dataset, InterHuman, only contains two-person data) and the non-scalable nature of existing network designs (e.g., InterGen uses a fixed shared-weight dual-stream network with cross-attention), existing methods cannot generate motion for more than two people.

**Difficulties in Multi-person Spatial Control**: Although methods like OmniControl have achieved precise trajectory control for single-person motion, introducing spatial control signals into multi-person motion generation remains a non-trivial challenge because it requires coordinating the spatial positions of multiple individuals simultaneously.

**Key Insight**: The authors observe that the joint distribution of multi-person motions can be decomposed using the conditional probability formula:

$$p(\mathbf{x}^1, \ldots, \mathbf{x}^n) = p(\mathbf{x}^1) \prod_{i=1}^{n-1} p(\mathbf{x}^{i+1} | \mathbf{x}^i, \ldots, \mathbf{x}^1)$$

This implies that if conditional motion distributions can be modeled, the motion of an arbitrary number of persons can be generated through a recursive process. The first step generates the motion of the first person (marginal distribution), the second step generates the motion of the second person conditioned on the already generated motion, and so on. This approach essentially reduces the multi-person motion synthesis problem to a series of conditional single-person motion generation tasks.

## Method

### Overall Architecture

The FreeMotion framework consists of two decoupled modules: the **Generation Module** and the **Interaction Module**. The former is responsible for generating diverse single-person motions, while the latter injects conditional signals (the motions of other individuals) into the generation process of the current motion.

The motion representation adopts the non-canonical representation proposed by InterGen, preserving global coordinate information to maintain spatial relationships among multiple people:

$$x^p(i) = [\mathbf{j}_{pg}, \mathbf{j}_{gv}, \mathbf{j}_r, \mathbf{c}_f]$$

where it includes global joint positions $\mathbf{j}_{pg} \in \mathbb{R}^{3J}$, global velocities $\mathbf{j}_{gv} \in \mathbb{R}^{3J}$, local rotations $\mathbf{j}_r \in \mathbb{R}^{6J}$, and foot contact features $\mathbf{c}_f \in \mathbb{R}^4$.

### Key Designs

1. **Generation Module**: This is a Transformer-based diffusion denoising network responsible for generating single-person motion from text. During training, noise is added to the motion $\mathbf{x}$ to obtain $\mathbf{x}_t$, and the module learns to denoise it to $\mathbf{x}_{t-1}$. Text features are extracted by a pre-trained CLIP-ViT-L-14 and injected into all attention layers via Adaptive LayerNorm. To obtain single-person descriptions, an LLM (ChatGPT) is utilized to decompose multi-person interaction descriptions into individual descriptions for each person.

2. **Interaction Module**: Designed inspired by ControlNet, this module is used to model interactions between conditional motions. The core component is the **Interactive Block**:

    - The noisy motion to be generated $\mathbf{x}_t^1$ and $N-1$ conditional motions are encoded into latent states through a shared linear layer.
    - The Interactive Block contains two sequential Self-Attention (SA) modules and a masking module.
    - The first SA processes the latent state of the noisy motion, while the second SA concatenates the output of the first SA with the (randomly masked) conditional motion latent states to perform global self-attention.
    - Calculation formula: $\mathbf{h}_t^{1,k}, \mathbf{h}^{2,k}, \ldots, \mathbf{h}^{N,k} = SA(SA(\mathbf{h}_t^{1,k-1}), Mask(\mathbf{h}^{2,k-1}, \ldots, \mathbf{h}^{N,k-1}))$
    - **Design Motivation**: Global self-attention is used instead of cross-attention because self-attention imposes no limit on the number of conditional motions (length-independent property), naturally supporting a variable number of motion conditions. Random masking training scales the model to adapt to an arbitrary number of conditions.

3. **Spatial Control Module**: Combines explicit and implicit spatial guidance to achieve precise position control for multi-person motion.

    - **Explicit Guidance**: Given a target position $\mathbf{s}$, classifier guidance is utilized to correct deviations at each denoising step: $\mathbf{x}_t = \mathbf{x}_t - \eta \nabla_{\mathbf{x}_t} \|\mathbf{s}_{nj} - \mathbf{x}_{nj}\|_2$
    - **Implicit Guidance**: Spatial signals are encoded through independent linear layers and added to the noisy motion latent states. During training, a portion of frames and joints are randomly selected.

### Loss & Training

Training is conducted in two stages:

**Stage 1** (Single-person Motion Generation): Train the Generation Module
$$\mathcal{L}_1 = \mathcal{L}_{rec} + \lambda_1 \mathcal{L}_{foot} + \lambda_2 \mathcal{L}_{vel} + \lambda_3 \mathcal{L}_{bl}$$

**Stage 2** (Conditional Motion Generation): Freeze the Generation Module and initialize the Interaction Module using its parameters, incorporating the DM distance loss
$$\mathcal{L}_2 = \mathcal{L}_{rec} + \lambda_1 \mathcal{L}_{foot} + \lambda_2 \mathcal{L}_{vel} + \lambda_3 \mathcal{L}_{bl} + \lambda_4 \mathcal{L}_{dm}$$

where $\mathcal{L}_{foot}$ is the contact loss, $\mathcal{L}_{vel}$ is the joint velocity loss, $\mathcal{L}_{bl}$ is the bone length loss, and $\mathcal{L}_{dm}$ is the masked joint distance map loss.

## Key Experimental Results

### Main Results

Two-person motion generation results on the InterHuman test set:

| Method | R-Prec Top1↑ | FID↓ | MM Dist↓ | Diversity→|
|------|-------------|------|----------|-----------|
| Real | 0.452 | 0.273 | 3.755 | 7.748 |
| TEMOS | 0.224 | 17.375 | 5.342 | 6.939 |
| MDM | 0.153 | 9.167 | 6.125 | 7.602 |
| ComMDM | 0.223 | 7.069 | 5.212 | 7.244 |
| InterGen* | 0.264 | 13.404 | 3.882 | 7.770 |
| **FreeMotion** | **0.326** | **6.740** | **3.848** | **7.828** |

Single-person motion generation (using LLM re-annotated text):

| Method | R-Prec Top1↑ | FID↓ | MM Dist↓ | Diversity→|
|------|-------------|------|----------|-----------|
| InterGen* | 0.206 | 23.415 | 3.925 | 7.514 |
| **FreeMotion** | **0.264** | **12.975** | **3.885** | **7.702** |

FreeMotion comprehensively outperforms InterGen* on both single-person and two-person generation, reducing the FID by 44.6% and 49.7%, respectively.

### Ablation Study

| Configuration | InterDes | R-Prec1↑ | FID↓ | MM Dist↓ | Diversity→|
|------|----------|----------|------|----------|-----------|
| GM* (Generation Module only) | ✗ | 0.300 | 8.842 | 3.863 | 7.761 |
| GM (Generation Module only) | ✓ | 0.259 | 10.749 | 3.883 | 7.645 |
| FreeMotion* | ✗ | 0.300 | 8.792 | 3.865 | 7.750 |
| **FreeMotion** | **✓** | **0.326** | **6.740** | **3.848** | **7.828** |

### Key Findings

- **Necessity of Decoupled Design**: Utilizing GM alone with InterDes actually leads to performance degradation (FID: 8.842 to 10.749), as GM needs to adapt to both single-person and interaction texts, making parameter updates difficult. In contrast, FreeMotion effectively utilizes both types of text through its decoupled design.
- **Three-person Motion Capability**: Despite being trained solely on two-person data, FreeMotion can directly perform inference on three-person motions, benefiting from the length-independent property of global self-attention.
- **Non-destructive Spatial Control**: The spatial control module is only trained on single persons. Once mounted onto the interaction module, it does not cause any noticeable degradation in multi-person spatial control.

## Highlights & Insights

- The idea of conditional probability decomposition is simple yet effective, transforming the non-scalable multi-person joint distribution modeling problem into a recursively scalable conditional single-person generation task.
- The Interaction Module is connected using ControlNet-style zero-initialized linear layers, ensuring that the pre-trained Generation Module is not disrupted during the initial stages of training.
- Leveraging LLMs to automatically decompose multi-person descriptions into single-person descriptions solves the problem of insufficient data annotation.

## Limitations & Future Work

- Text splitting by LLMs may cause the single-person descriptions to not perfectly match the motions.
- Since the model is only trained on two-person data, inter-penetration may occur when generating movements for a large number of individuals.
- The recursive generation approach causes the inference time to scale linearly with the number of persons.
- Non-autoregressive parallel multi-person generation methods have not been explored.

## Related Work & Insights

- The ControlNet paradigm of "parameter replication + zero-initialization" is equally applicable in the motion generation domain.
- The concept of conditional probability decomposition can be generalized to other multi-entity generation tasks (e.g., multi-object 3D generation).
- Compared with the dual-stream shared-weight design of InterGen, the decoupled scheme exhibits superior scalability.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The concept of using conditional probability decomposition to achieve number-free generation is highly novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers quantitative and qualitative experiments on single/double/triple-person motion alongside ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ The mathematical derivations are clear and the motivations are thoroughly explained.
- **Value**: ⭐⭐⭐⭐ Achieves text-to-motion generation for an arbitrary number of persons for the first time, offering clear practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Unified Number-Free Text-to-Motion Generation Via Flow Matching](../../CVPR2026/human_understanding/unified_number-free_text-to-motion_generation_via_flow_matching.md)
- [\[ECCV 2024\] Large Motion Model for Unified Multi-Modal Motion Generation](large_motion_model_for_unified_multi-modal_motion_generation.md)
- [\[ICLR 2026\] Event-T2M: Event-level Conditioning for Complex Text-to-Motion Synthesis](../../ICLR2026/human_understanding/event-t2m_event-level_conditioning_for_complex_text-to-motion_synthesis.md)
- [\[ECCV 2024\] Human Motion Forecasting in Dynamic Domain Shifts: A Homeostatic Continual Test-Time Adaptation Framework](human_motion_forecasting_in_dynamic_domain_shifts_a_homeostatic_continual_test-t.md)
- [\[ECCV 2024\] Towards Unified Representation of Invariant-Specific Features in Missing Modality Face Anti-Spoofing](towards_unified_representation_of_invariant-specific_features_in_missing_modalit.md)

</div>

<!-- RELATED:END -->
