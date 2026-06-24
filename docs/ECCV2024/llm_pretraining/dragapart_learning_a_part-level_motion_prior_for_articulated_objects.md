---
title: >-
  [Paper Note] DragAPart: Learning a Part-Level Motion Prior for Articulated Objects
description: >-
  [ECCV 2024][LLM Pretraining][Part-Level Motion Prior] DragAPart proposes an image generator that uses dragging as an interactive interface, capable of responding to part-level interactions (such as opening/closing drawers/doors) rather than merely moving the entire object. Through the new synthetic dataset Drag-a-Move, multi-resolution drag encoding, and domain randomization strategies, the model generalizes well to real images and unseen categories despite being trained sole…
tags:
  - "ECCV 2024"
  - "LLM Pretraining"
  - "Part-Level Motion Prior"
  - "Drag Control"
  - "Articulated Objects"
  - "Diffusion Models"
  - "Domain Randomization"
date: 2026-05-08
content_hash: d0b2455e2102d654
---

# DragAPart: Learning a Part-Level Motion Prior for Articulated Objects

**Conference**: ECCV 2024  
**arXiv**: [2403.15382](https://arxiv.org/abs/2403.15382)  
**Code**: Yes (dragapart.github.io)  
**Area**: LLM Pre-training  
**Keywords**: Part-Level Motion Prior, Drag Control, Articulated Objects, Diffusion Models, Domain Randomization

## TL;DR

DragAPart proposes an image generator that uses dragging as an interactive interface, capable of responding to part-level interactions (such as opening/closing drawers/doors) rather than merely moving the entire object. Through the new synthetic dataset Drag-a-Move, multi-resolution drag encoding, and domain randomization strategies, the model generalizes well to real images and unseen categories despite being trained solely on synthetic data.

## Background & Motivation

### Why Part-Level Motion Prior is Needed

Existing deformable object modeling typically relies on category-specific parametric templates (such as SMPL for humans, SMAL for mammals), which lack universality. An ideal **general-purpose motion foundation model** should understand the motion of any object—be it humans, jellyfish, towels, or furniture. However, no universal template exists that can represent the poses of all objects in a unified manner.

**Core Hypothesis**: A motion model does not require reference templates; it only needs to understand the possible physical configurations of objects and their transformations. **Dragging** provides a template-free way to probe motion priors—by specifying how a single physical point moves, allowing the model to "fill in" the plausible motion of the remaining parts.

### Failure of Existing Dragging Methods

Existing drag-controlled image/video generation methods (such as DragGAN, DragDiffusion, DragonDiffusion, DragNUWA) perform well in object repositioning but fail completely at **part-level motion**:
- They tend to **move the entire object** (such as translation/scaling) rather than opening drawers or rotating doors.
- Reason: Large-scale pre-training data inevitably entangles camera viewpoint changes and multi-object interactions, making it difficult to distinguish the independent motion of different parts.

### Fundamental Problems of Drag Encoding

Prior methods use sparse optical flow maps to represent drag operations, which suffer from two fatal drawbacks:

**Ambiguous Destination Localization**: The Transformer can perceive the drag handle start point $u$ through spatial positions, but the destination point $v$ is only encoded as a displacement vector at the start position, leading to weak spatial awareness.

**Conflict under Multiple Drags**: All drag instructions are encoded in the same optical flow map. After downsampling, different drag representations may overlap, resulting in mutual interference between different part motion signals.

## Method

### Overall Architecture

DragAPart is fine-tuned based on pre-trained Stable Diffusion (SD):
- **Input**: An object-centric RGB image $y$ + a set of drag instructions $\mathcal{D}$.
- **Output**: A new image reflecting the drag effects, $x \sim \mathbb{P}(x|y, \mathcal{D})$.
- **Training Data**: Synthetic dataset Drag-a-Move (based on 3D articulated objects from GAPartNet).
- **Architecture**: Injecting drag encoding into each Transformer block of the SD U-Net.

Condition injection mechanism: The reference image $y$ undergoes a noiseless forward pass through the same network $\Phi$. Its keys/values replace the self-attention keys/values in the generation process, achieving cross-attention between the reference and the generated images.

The training objective is the standard denoising loss:

$$\min_\Phi \mathbb{E}_{(x,y,\mathcal{D}),t,\epsilon \sim \mathcal{N}(0,1)} \left[\|\epsilon - \Phi(z_t, t, y, \mathcal{D})\|_2^2\right]$$

### Key Designs

#### 1. **Multi-Resolution Drag Encoding**

Core innovation. Unlike prior methods, this work assigns **independent channels** to each drag and **separately encodes the start and end points**:

For a single drag $d = (u, v)$, under the resolution $(h_l, w_l)$ corresponding to the $l$-th block of the LDM, it is encoded as $F_l(u,v) \in \mathbb{R}^{2 \times h_l \times w_l}$:
- Filled entirely with $-1$ (indicating no-drag mask), recording sub-pixel offsets only at the latent pixel position corresponding to the start point $u$.
- The end point $v$ is similarly encoded independently as $F_l(v,u)$.
- Final drag encoding: $F_l(\mathcal{D}) = \bigoplus_{(u,v) \in \mathcal{D}} F_l(u,v) \oplus F_l(v,u) \in \mathbb{R}^{4N \times h \times w}$

**Design Advantages**:
- Each drag occupies independent channels $\rightarrow$ motion signals for different parts **do not overlap**.
- Start and end points are **spatially encoded separately** $\rightarrow$ the Transformer can accurately perceive the positions of both ends.
- Injected in each block at the **corresponding resolution** $\rightarrow$ replacing the prior approach of Gaussian blurring + convolutional downsampling.
- Padded with $-1$ when there are fewer than $N$ drag instructions.

#### 2. **Domain Randomization**

The textures of synthetic objects in GAPartNet are unrealistic and lack variation. Direct training on these causes the model to "cheat"—generating pixels of the same color everywhere instead of truly understanding the pixel sources.

Solution: During training, both original texture renderings and **random solid color texture** renderings (where each part is randomly colored) are used **simultaneously**.

**Design Motivation**: Random textures prevent the model from relying on color consistency to replicate pixels, forcing it to implicitly understand 3D geometric structures and part correspondences. Ablation studies confirm that domain randomization increases PSNR from 18.03 to 19.74 on OOD data, significantly enhancing generalization ability.

#### 3. **Drag-a-Move Dataset**

Built upon GAPartNet, containing 763 3D models across 16 categories (objects labeled with "Hinge Handle", "Slider Drawer", "Hinge Lid", and "Hinge Door").

Data generation pipeline:
- Renders 48 articulated animations per object, with 36 frames per animation $\rightarrow$ totaling 40 million unique image pairs.
- Two animation modes: ① certain joints are locked (closed state), while remaining joints transition between extreme states; ② all joints are in random states.
- Drag generation: A part is sampled from the subtree of moving joints. Surface points are sampled based on displacement probability and projected onto the pixel space.
- A random-textured version is simultaneously rendered for domain randomization.

### Downstream Applications

#### Motion Analysis
Given a 3D mesh $\mathcal{M} = \mathcal{M}_\text{static} \cup \mathcal{M}_\text{moving}$ and 3D drag instructions, motion parameters (revolute/prismatic joints, rotation axes/translation directions) are estimated by minimizing the discrepancy between multi-view renderings and images generated by DragAPart:

$$\arg\min_{t, p_{\text{motion}}} \frac{1}{K} \sum_{k=1}^{K} \|R(\mathcal{M}_\text{static} \cup T(\mathcal{M}_\text{moving}; t, p_{\text{motion}}), C_k) - x_k\|_2^2$$

Grid search is utilized instead of gradient optimization to avoid local optima.

#### Articulated Part Segmentation
Extractor internal features of the denoiser $\Phi$: By performing forward passes with and without drag conditioning on the same image, feature differences are calculated. This difference represents the drag effect within the internal representations of the diffusion model, which can be used to coarsely segment movable parts responsive to the drag instruction.

## Key Experimental Results

### Main Results

Quantitative comparison on Drag-a-Move and Human3.6M:

| Method | Drag-a-Move PSNR↑ | SSIM↑ | LPIPS↓ | Human3.6M PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------------------|-------|--------|------------------|-------|--------|
| iPoke | 16.79 | 0.883 | 0.150 | 21.43 | 0.856 | 0.258 |
| DragDiffusion | 15.30 | 0.773 | 0.226 | 18.06 | 0.707 | 0.280 |
| DragonDiffusion | 17.63 | 0.852 | 0.162 | 19.45 | 0.765 | 0.258 |
| DragNUWA | 13.58 | 0.765 | 0.277 | 15.16 | 0.668 | 0.292 |
| **DragAPart (Ours)** | **21.38** | **0.925** | **0.066** | **23.82** | **0.870** | **0.091** |

DragAPart leads significantly across all metrics: PSNR +3.75 (vs DragonDiffusion), LPIPS reduced by 59% (-0.096).

### Ablation Study

**Architecture & Drag Encoding Ablation**:

| Encoding Method | DiT PSNR↑ | SSIM↑ | LPIPS↓ | SD PSNR↑ | SSIM↑ | LPIPS↓ |
|---------|-----------|-------|--------|----------|-------|--------|
| Convolution only on input layer | 19.56 | 0.910 | 0.095 | 19.97 | 0.914 | 0.077 |
| Convolution injected at each layer | 20.58 | 0.922 | 0.078 | 21.10 | 0.925 | 0.067 |
| **Multi-resolution encoding injected at each layer** | **21.11** | **0.925** | **0.074** | **21.38** | **0.925** | **0.066** |

**Domain Randomization Ablation**:

| Training Data | I.D. PSNR↑ | SSIM↑ | LPIPS↓ | O.O.D. PSNR↑ | SSIM↑ | LPIPS↓ |
|---------|-----------|-------|--------|-------------|-------|--------|
| Without Domain Randomization | 21.38 | 0.925 | 0.066 | 18.03 | 0.897 | 0.104 |
| **With Domain Randomization** | **21.82** | **0.928** | **0.066** | **19.74** | **0.920** | **0.083** |

Domain randomization also yields a slight improvement on I.D. (+0.44 PSNR), while the improvement on O.O.D. is prominent (+1.71 PSNR).

### Key Findings

1. **Multi-Resolution Encoding > Convolutional Encoding**: Consistent improvements are observed across both architectures (SD and DiT), proving that channel separation + spatially precise encoding is more effective than blurring + convolution.
2. **Layer-wise Injection Outperforms Entry-level Injection by a Wide Margin**: The PSNR gap exceeds 1.4 (SD). Injecting at multiple layers allows motion information to propagate accurately across different resolutions.
3. **Pre-trained U-Net (SD) Slightly Outperforms DiT Trained from Scratch**: This benefits from the generalization priors learned via large-scale (hundreds of millions of images) pre-training.
4. **Domain Randomization is Key to Real-World Generalization**: Without it, the model degrades severely on real-world images and unseen categories.
5. **Qualitative Observations**: Prior dragging methods tend to replicate the appearance of the handle point to the target point, or scale/translate the object globally. In contrast, DragAPart produces physically plausible, part-level motions.

## Highlights & Insights

1. **Dragging as a Motion Probe**: Dragging is redefined as an interface to probe general-purpose motion priors, rather than simple image editing. A single drag can implicitly prompt a plausible response across the entire kinematic chain.
2. **Power of Limited Synthetic Data + Pre-trained Models**: With only 763 synthetic 3D objects (far fewer than the millions of objects in NVS tasks), coupled with the visual priors of pre-trained SD, the model successfully generalizes to real images and unseen categories.
3. **Simplicity and Effectiveness of Domain Randomization**: The extremely simple strategy of random coloring unexpectedly bridges the sim-to-real gap, forcing the model to learn structures rather than textures.
4. **A New Paradigm for Motion Analysis**: Instead of directly predicting motion parameters, it optimizes 3D motion parameters through generated multi-view drag results, cleverly exploiting the view consistency of generative models.

## Limitations & Future Work

1. **Cross-View Inconsistency**: It lacks explicit constraints on the generation consistency of the same object under different viewpoints/drag conditions.
2. **Separate Training for Humans and Objects**: Currently, separate models are trained for daily objects and humans, lacking a unified general-purpose motion prior.
3. **Limited Synthetic Data Scale**: Comprising only 763 objects and 16 categories, leading to potentially insufficient generalization on tail/niche classes.
4. **Fixed Upper Bound (N) of Drags**: The maximum number of drags $N$ is pre-determined, failing to handle cases where it is exceeded.
5. **Static Image Generation Only**: It has not been extended to video generation, failing to demonstrate continuous motion trajectories.

## Related Work & Insights

- The conditional injection pipeline akin to **ControlNet** is creatively applied to drag encoding.
- Part-level annotations from GAPartNet provide a crucial data foundation for learning articulated object motion.
- The domain randomization strategy, originating from the sim-to-real transfer literature (Tobin et al.), proves equally effective in diffusion model fine-tuning.
- Articulated part segmentation exploits internal feature discrepancies of the diffusion model, inspiring a new paradigm of using "differential features" for semantic analysis.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The problem formulation of redefining drags as part-level motion probes is highly novel, and the multi-resolution encoding design is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Quantitative and qualitative experiments are comprehensive, and ablations are solid, though real-world evaluation remains predominantly qualitative.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The motivation of the problem is clearly articulated, with exquisite visualizations and rigorous argumentation logic.
- **Value**: ⭐⭐⭐⭐ — Motion analysis and part segmentation showcase promising application prospects, though the coverage of part categories is limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Plan, Posture and Go: Towards Open-Vocabulary Text-to-Motion Generation](plan_posture_and_go_towards_open-vocabulary_text-to-motion_generation.md)
- [\[ICML 2025\] Position: The Future of Bayesian Prediction Is Prior-Fitted](../../ICML2025/llm_pretraining/position_the_future_of_bayesian_prediction_is_prior-fitted.md)
- [\[ICLR 2026\] Token-level Data Selection for Safe LLM Fine-tuning](../../ICLR2026/llm_pretraining/token-level_data_selection_for_safe_llm_fine-tuning.md)
- [\[ICML 2025\] LLM Data Selection and Utilization via Dynamic Bi-level Optimization](../../ICML2025/llm_pretraining/llm_data_selection_and_utilization_via_dynamic_bi-level_optimization.md)
- [\[NeurIPS 2025\] Deep Compositional Phase Diffusion for Long Motion Sequence Generation](../../NeurIPS2025/llm_pretraining/deep_compositional_phase_diffusion_for_long_motion_sequence_generation.md)

</div>

<!-- RELATED:END -->
