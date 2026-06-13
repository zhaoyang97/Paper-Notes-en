---
title: >-
  [Paper Note] FOVI: Bio-inspired Foveated Interface for Deep Vision Models
description: >-
  [ICML 2026][Foveated sampling] Inspired by the human retina—V1 pathway, the authors construct FOVI, a foveated input interface with non-uniform pixel distribution but uniform density on the sensor manifold…
tags:
  - "ICML 2026"
  - "Foveated sampling"
  - "cortical magnification function"
  - "kNN convolution"
  - "kernel mapping"
  - "LoRA adaptation"
date: 2026-05-08
content_hash: fe65d6872216f658
---

# FOVI: Bio-inspired Foveated Interface for Deep Vision Models

**Conference**: ICML 2026  
**arXiv**: [2602.03766](https://arxiv.org/abs/2602.03766)  
**Code**: https://github.com/nblauch/fovi  
**Area**: Efficient Vision Architectures / Bio-inspired  
**Keywords**: Foveated sampling, cortical magnification function, kNN convolution, kernel mapping, LoRA adaptation

## TL;DR
Inspired by the human retina—V1 pathway, the authors construct FOVI, a foveated input interface with non-uniform pixel distribution but uniform density on the sensor manifold, using a "cortical magnification function + local isotropic sampling." Combined with a novel kNN convolution + kernel mapping technique, FOVI is compatible with both CNNs and ViTs, allowing a DINOv3-ViT to approach the ImageNet accuracy of a full-resolution baseline using only about 1/16 of the pixels.

## Background & Motivation

**Background**: Mainstream deep vision models encode the world as rectangular pixel maps with uniform resolution, typically fixed at 224×224. For processing high-resolution full-field images (e.g., robotics, autonomous driving, first-person vision), CNN computation scales linearly with pixels, whereas ViT computation expands **doubly-quadratically** with side length due to self-attention, making the costs prohibitive.

**Limitations of Prior Work**: Historical attempts at foveated vision generally fell into two dead ends. First, log-polar image models, which take the logarithm of the radius and sample equiangularly at each radius; second, warped Cartesian models, which project the log-polar representation back into a "squeezed" rectangular map. Both approaches essentially force a rectangular grid, leading to **local anisotropic sampling**—where sampling intervals differ along the radial and polar angle directions within the same neighborhood—causing distorted receptive fields that are biologically inaccurate and break the geometric isomorphism required for standard convolution. A few works achieving isotropic sampling (e.g., Killick et al.) rely on fixed Gaussian derivative bases and do not support end-to-end learning.

**Key Challenge**: Foveated sampling requires that "sampling density varies only with eccentricity," naturally producing a **curved, non-rectangular** sample set in visual space. Standard convolution/patching operators require a **regular rectangular grid** for weight sharing. These two are fundamentally incompatible.

**Goal**: (1) Provide a locally isotropic sampler strictly aligned with primate retinal-V1 mapping; (2) Invent a weight-sharing, end-to-end trainable convolution operator for non-regular sampling; (3) Demonstrate that this interface can train CNNs from scratch or adapt pre-trained large ViT foundation models (e.g., DINOv3) to realize gains in FLOPs, latency, and memory.

**Key Insight**: The authors leverage the mathematical model of Rovamo & Virsu (1984): if sampling density decreases with eccentricity strictly according to the cortical magnification function $M(r)=1/(r+a)$, and the number of angular samples at each radius is determined by "local isotropy," then the resulting set of discrete points—dense at the center and sparse at the periphery—is **uniformly dense on a 3D curved surface manifold (sensor manifold)**. From the perspective of this sensor manifold, the problem returns to "performing convolution on uniformly dense point clouds," which can be solved by using k-nearest neighbor (kNN) neighborhoods instead of rectangular windows.

**Core Idea**: Reformulate "foveated sampling" as "uniform sampling on a cortical-like sensor manifold," and use **kNN convolution + kernel mapping** to unify the geometric incompatibilities of CNN and ViT architectures while maintaining weight sharing and end-to-end learning.

## Method

### Overall Architecture

The FOVI interface consists of two tightly coupled components. The first is the **foveated sensor**: given a field-of-view radius $r_{\max}$ and magnification parameter $a$, a set of radii $\{r_i\}$ is sampled equidistantly along the "cortical distance" $w(r)=\log(r+a)+C$. At each radius, the angular sample count $n_s(r_i)$ is automatically determined by the local isotropy constraint ("tangential spacing $\approx$ radial spacing"), resulting in a sample point set that is dense centrally and sparse peripherally. This set is uniformly dense on the Rovamo–Virsu manifold and can be flattened via Schwartz’s complex logarithmic model. The second component is the **operator for regular processing on the sensor manifold**: the authors replace rectangular windows with k-nearest neighbor (kNN) neighborhoods. "Convolution" is redefined as a weighted sum over the kNN neighborhood of each output unit on the sensor manifold. Through a kernel mapping technique, a "reference kernel $W$" learned on a Cartesian grid is sampled onto each kNN in an orientation-aligned manner, achieving true weight sharing. The full pipeline involves: **Input image → Foveated sampler → Features on sensor manifold (uniform dense point cloud) → kNN convolutions (for CNN) or kNN patching (for ViT) → Classification head**.

### Key Designs

1.  **Locally Isotropic Sampler based on Cortical Magnification Function**:
    -   **Function**: Generates a set of foveated sample points within the FOV while ensuring equal radial and tangential spacing within neighborhoods.
    -   **Mechanism**: Uses the classic cortical magnification function $M(r)=1/(r+a)$, whose integral $w(r)=\log(r+a)+C$ provides the "cortical distance." Radii $r_i$ are solved from equidistant $w$ values, and the angular count $n_s(r_i)$ is determined such that target spacing is maintained. The parameter $a$ controls foveation intensity: $a\to 0$ for extreme foveation and $a\to\infty$ for uniform sampling. A normalization factor $k_a=(\int_0^{r_{\max}}1/(r+a)\,dr)^{-1}$ is used to ensure different $a$ values match in "total resolution."
    -   **Design Motivation**: To strictly match primate V1 retinal mapping while avoiding the local anisotropy issues of log-polar/warped Cartesian models, ensuring receptive field shapes are not artificially distorted.

2.  **kNN Convolution + Kernel Mapping**:
    -   **Function**: Implements a weight-sharing, orientation-aligned, and end-to-end trainable "convolution" operator on the sensor manifold.
    -   **Mechanism**: For each output unit, its $k$ nearest neighbors are identified on the sensor manifold. Each neighbor is assigned polar coordinates $(r,\theta)$, where $r$ is the geodesic distance to the center unit **on the manifold**, and $\theta$ is its polar angle relative to the center in **visual space**. These are projected to a common Cartesian reference frame $(x=r\cos\theta, y=r\sin\theta)$ and aligned with a reference kernel $W$ learned on a standard grid (default side length $s=2\sqrt{k}$ for antialiasing). The actual kernel weights for the neighborhood are obtained by spatial sampling from $W$. Thus, the geometric size of the "convolution kernel" scales with eccentricity, but **feature orientation** (e.g., "vertical stripes") remains consistent across the field; all neighborhoods share the same reference kernel $W$.
    -   **Design Motivation**: Solves the geometric challenge of convolving irregular point sets and utilizes higher-resolution reference kernels ($s=2\sqrt{k}$, providing ~+3% ImageNet gain) to mitigate aliasing caused by irregular neighborhood distributions.

3.  **kNN-based Patching + LoRA for Adapting Pre-trained ViTs**:
    -   **Function**: Transforms a pre-trained ViT (e.g., DINOv3) into a foveated version without destroying its feature space by replacing only the patch embedding layer.
    -   **Mechanism**: Two foveated grids are defined: a dense "sensor array" and a sparse "patch center array." A patch is defined as the kNN of a patch center on the sensor array. By constraining $a$, the patch count is kept equal to the baseline ViT (e.g., 64), requiring no changes to the transformer backbone weights. Fine-tuning uses **LoRA over the patch embedding + early transformer layers**.
    -   **Design Motivation**: For ViTs, geometric transformation is completed via patch embedding; subsequent attention is naturally independent of patch regularity. LoRA provides an optimal balance between capacity and regularization—avoiding overfitting while allowing the model to adapt to new input geometries.

### Loss & Training
Both CNNs and ViTs are trained with standard supervised classification. During training, 4 fixations are randomly sampled from the central region of each image (radius is 0.25 of image size). Each fixation independently produces logits, which are then **averaged** before calculating the cross-entropy loss. Inference allows for expansion to 20 fixations, with performance typically saturating around that count. The optimizer uses cosine decay.

## Key Experimental Results

### Main Results

| Model | fix. | Pixels | Patches | GFLOPs | Top-1 | Val Latency (ms) | Val Memory (GB) |
|---|---|---|---|---|---|---|---|
| ViT-H+ uniform @224 | 1 | 50176 | 196 | 172.4 | **0.871** | 289.6 | 40.1 |
| FOVI-ViT-H+ @64 (a=2.79) | 1 | 3976 | 64 | 58.4 | 0.844 | 120.0 | 19.1 |
| FOVI-ViT-H+ @64 (a=2.79) | 3 | 11928 | 192 | 175.3 | 0.853 | 303.7 | 40.9 |
| ViT-S+ uniform @224 | 1 | 50176 | 196 | 6.16 | 0.794 | 37.9 | 4.1 |
| FOVI-ViT-S+ @64 (a=2.79) | 1 | 3976 | 64 | 2.04 | 0.700 | 27.6 | 1.7 |
| ViT-S+ uniform @64 | 1 | 4096 | 64 | 2.02 | 0.693 | 27.6 | 1.7 |
| ViT-S+ log-polar @64 (a=2.79) | 1 | 4096 | 64 | 2.02 | 0.643 | 27.9 | 1.7 |
| FOVI-ViT-S+ @64 (a=2.79) | 3 | 11928 | 192 | 6.12 | **0.735** | 46.8 | 4.3 |

(Excerpt from Table 1; results grouped by fixation/accuracy.)

### Ablation Study

| Configuration | ImageNet-1K Top-1 | Description |
|---|---|---|
| FOVI-CNN, $a=0.5$ (Moderate) | Optimal | Under a fixed 64×64 budget, moderate foveation outperforms the pure uniform model ($a=500$). |
| FOVI-CNN, $a=50$ (Near-uniform) | Lower | Loses radial RF gradient; RF no longer grows linearly with eccentricity. |
| Ref kernel res $s=\sqrt{k}$ | -3% | Increasing to $s=2\sqrt{k}$ yields ~+3% top-1 gain. |
| DINOv3 LoRA (patch emb + early layers) | Baseline | Outperforms full fine-tuning by ~10% and frozen adaptation by ~30%. |
| FOVI-ViT-S+ vs log-polar | 0.700 vs 0.643 | +6% advantage at 1 fixation, validating local isotropy. |

### Key Findings

-   **The "Sweet Spot" of Foveation**: Under restricted pixel budgets, moderate foveation ($a=0.5$) beats uniform sampling, showing an inverted U-shape. This aligns with the "center bias + mid-scale bias" of ImageNet data.
-   **Emergent Biological Alignment**: The RF diameters of FOVI-CNN layers grow approximately linearly with eccentricity. Higher layers show larger slopes and intercepts, matching the pRF patterns measured via fMRI in human V1–V3 (Dumoulin & Wandell, 2008). 
-   **Plug-and-play for Large Models**: FOVI-ViT-H+ with 3 fixations (~11.9k pixels, ~1/4 of 224×224) reaches 0.853 top-1, trailing the full-resolution baseline by only 1.8% while maintaining equivalent FLOPs, latency, and memory.

## Highlights & Insights

-   **Translating "Irregular Sampling" to "Regular Manifold Operations"**: This perspective shift is elegant, preserving biological interpretability while maintaining weight sharing and end-to-end learnability—overcoming the long-standing flaws of log-polar and warped Cartesian methods.
-   **Kernel Mapping is a Transferable Trick**: The technique of learning a reference kernel on a standard grid and mapping it to arbitrary geometric neighborhoods is a versatile tool applicable to point cloud, spherical, or graph convolutions.
-   **Logit Averaging for Fixations is Sufficient**: Simple averaging of logits across multiple fixations is nearly optimal, with learnable integrators offering no significant advantage, providing a cost-effective interface for active vision.
-   **LoRA + Early Layer Adaptation Strategy**: For transfer scenarios where input geometry changes but semantic priors should remain, adapting patch embeddings and early layers is the robust choice, as later layers closer to the semantic level should be protected.

## Limitations & Future Work

-   Current fixations are randomly sampled from the center; to fully realize "active perception," the saccade strategy itself needs to be a learnable module (e.g., via RL).
-   Training is limited to 4 fixations due to gradient accumulation costs, which may limit the model's ability to learn how to combine long sequences of gazes.
-   Evaluations are centered on ImageNet, which has a strong center bias. For more uniform fields (e.g., autonomous driving), the optimal foveation point $a$ will likely differ.
-   The need for $s=2\sqrt{k}$ suggests that neighbor distributions are still not perfectly regular, hinting that further antialiasing or interpolation could improve accuracy.

## Related Work & Insights

-   **vs. log-polar / warped Cartesian (Weiman & Chaikin, 1979)**: These force foveated sampling into rectangular grids, causing anisotropic distortion. FOVI eliminates this via uniform sampling on the sensor manifold.
-   **vs. Killick et al. (2023)**: Both pursue isotropy, but Killick requires fixed Gaussian bases. FOVI enables end-to-end learning via kernel mapping.
-   **vs. Architectural Foveation**: Prior works prioritize resources but use uniform pixel inputs. FOVI realizes savings at the sensor level, aligning closer to true "efficient active vision."

## Rating
-   Novelty: ⭐⭐⭐⭐⭐ 
-   Experimental Thoroughness: ⭐⭐⭐⭐ 
-   Writing Quality: ⭐⭐⭐⭐⭐ 
-   Value: ⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Do Vision Models Perceive Illusory Motion in Static Images Like Humans?](../../CVPR2026/others/do_vision_models_perceive_illusory_motion_in_static_images_like_humans.md)
- [\[ICML 2026\] Possibilistic Predictive Uncertainty for Deep Learning](possibilistic_predictive_uncertainty_for_deep_learning.md)
- [\[ICML 2026\] Advantages of Non-Smooth Components in Vision Transformer Fine-Tuning](vision_transformer_finetuning_benefits_from_non-smooth_components.md)
- [\[NeurIPS 2025\] Deep Continuous-Time State-Space Models for Marked Event Sequences](../../NeurIPS2025/others/deep_continuous-time_state-space_models_for_marked_event_sequences.md)
- [\[ICML 2026\] Sequential Group Composition: A Window into the Mechanics of Deep Learning](sequential_group_composition_a_window_into_the_mechanics_of_deep_learning.md)

</div>

<!-- RELATED:END -->
