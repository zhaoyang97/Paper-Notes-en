---
title: >-
  [Paper Note] PhysGM: Large Physical Gaussian Model for Feed-Forward 4D Synthesis
description: >-
  [CVPR 2026][3D Vision][3D Gaussian Splatting] PhysGM proposes the first feed-forward framework that simultaneously predicts 3D Gaussian representations and physical properties (stiffness, mass, etc.) from a single image in one inference pass. Combined with MPM simulation, it generates high-fidelity, physically plausible 4D animations within one minute, requiring no per-scene optimization.
tags:
  - CVPR 2026
  - 3D Vision
  - 3D Gaussian Splatting
  - Physical Simulation
  - Feed-Forward 4D Synthesis
  - MPM
  - DPO
  - Physical Property Prediction
date: 2026-05-08
content_hash: 0344b23e265edaa6
---

# PhysGM: Large Physical Gaussian Model for Feed-Forward 4D Synthesis

**Conference**: CVPR 2026
**arXiv**: [2508.13911](https://arxiv.org/abs/2508.13911)
**Code**: [Project Page](https://hihixiaolv.github.io/PhysGM.github.io/)
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, Physical Simulation, Feed-Forward 4D Synthesis, MPM, DPO, Physical Property Prediction

## TL;DR

PhysGM proposes the first feed-forward framework that simultaneously predicts 3D Gaussian representations and physical properties (stiffness, mass, etc.) from a single image in one inference pass. Combined with MPM simulation, it generates high-fidelity, physically plausible 4D animations within one minute, requiring no per-scene optimization.

## Background & Motivation

**3DGS Reconstruction Bottleneck**: Existing physics-driven 4D synthesis relies on reconstructing 3DGS from dense multi-view images, requiring time-consuming per-scene optimization that precludes scalable deployment.

**Manual Physical Parameter Specification**: Methods such as PhysGaussian require physical properties (stiffness, mass, etc.) to be manually specified via configuration files, lacking automation and generalization capability.

**High Cost of SDS Optimization**: OmniPhysGS, DreamPhysics, and similar methods distill physical priors from video models via SDS, requiring gradient backpropagation through differentiable physics simulators, with per-scene optimization taking 0.5–12 hours.

**Decoupling of Appearance and Physics**: Existing methods naively concatenate pre-built 3DGS with physics modules, neglecting physical cues embedded in appearance (e.g., metallic sheen implying material stiffness), leading to suboptimal performance.

**Lack of Training Data**: No large-scale dataset previously existed pairing 3D assets with physical property annotations and reference simulation videos.

**Core Problem**: Can per-scene optimization be entirely bypassed to generate complete physics-driven 4D simulations through a single feed-forward inference pass?

## Method

### Overall Architecture

PhysGM is a physics-aware reconstruction model based on a Transformer architecture. Given posed RGB images (1 or 4 views), it produces in a single feed-forward pass: (1) 3D Gaussian representation parameters $\psi$; and (2) a physical property vector $\theta$. These parameters directly initialize an MPM simulator to generate dynamic sequences. For single-view input, MVAdapter synthesizes three auxiliary views (back, left, right).

**Multimodal Tokenization**:

- The image encoder uses DINOv3 (ViT-L/16) to patchify input images and project them into 1024-dimensional features.
- Camera geometry is represented via Plücker ray coordinates, processed by a dense representation encoder and concatenated with image tokens.
- Three learnable global tokens $\mathbf{g}_1, \mathbf{g}_2, \mathbf{g}_3$ are introduced to aggregate global scene information.

**Transformer Backbone**: A 24-layer Transformer collects intermediate-layer output tokens to provide multi-scale representations.

### Key Design 1: Dual-Head Prediction

**DPT Head (3DGS Parameters)**: A Dense Prediction Transformer head progressively upsamples multi-scale features to output per-pixel 3DGS parameter maps (position $\mu$, rotation $\mathbf{q}$, scale $\mathbf{s}$, opacity $\alpha$, color $\mathbf{c}$). Gaussians predicted from each view are aggregated into a complete 3D scene.

**Physics Head**: Three physical property types are predicted from the 3 global tokens:

- Classification head $f_{\text{material}}$: predicts material category $C$ (determines the choice of constitutive model).
- Regression head $f_{\text{phys}}$: outputs the mean and log-variance of Young's modulus $E$ and Poisson's ratio $\nu$, defining the conditional probability distribution $P(\theta|I) = \mathcal{N}(\theta|\mu_\theta, \text{diag}(\sigma_\theta^2))$.

Probabilistic modeling enables sampling of diverse physical parameters, providing the foundation for constructing DPO preference pairs.

### Key Design 2: Two-Stage Training

**Stage 1 — Supervised Pre-training**: Joint optimization on the PhysAssets dataset over 3DGS reconstruction (MSE + Alpha + LPIPS losses) and physical property prediction, establishing a strong generalizable prior. The key advantage of joint optimization is the mutual reinforcement between geometry and physics.

**Stage 2 — DPO Fine-tuning**: The backbone is frozen; only the physics prediction head is fine-tuned.

- For each scene, $K$ candidate physical parameter sets are sampled: $\phi_k \sim \pi_\omega(\cdot|\mathbf{z})$.
- Each candidate undergoes MPM simulation and video rendering to produce $V_k$.
- SAM-2 segmentation and CoTracker-3 trajectory extraction are used to compute perceptual distance from the GT video.
- The candidate closest to GT is designated the "winner" $\phi_w$; the farthest is the "loser" $\phi_l$.
- DPO loss: $L_{\text{DPO}} = -\mathbb{E}[\log\sigma(\beta\log\frac{\pi_\omega(\phi_w|\mathbf{z})}{\pi_{\text{ref}}(\phi_w|\mathbf{z})} - \beta\log\frac{\pi_\omega(\phi_l|\mathbf{z})}{\pi_{\text{ref}}(\phi_l|\mathbf{z})})]$

Key advantage of DPO: it entirely bypasses differentiable physics engines and SDS, aligning physical plausibility using only ranking signals.

### Key Design 3: MPM Physical Simulation

The Material Point Method (MPM) drives dynamic simulation, with a one-to-one correspondence between 3D Gaussians and MPM particles:

- Particle position $\mathbf{x}_p$ directly defines the Gaussian mean $\mu$.
- The deformation gradient $\mathbf{F}_p$ is polar-decomposed to obtain rotation $\mathbf{R}_p$ and stretch tensor $\mathbf{S}_p$, which update Gaussian rotation and scale respectively.
- The constitutive model is selected based on the predicted material category: Neo-Hookean (jelly/rubber), Fixed Corotational (metal), Drucker-Prager (sand/snow/plasticine).

Simulation parameters: grid $200^3$, substep $2\times10^{-5}$ s, frame time $4\times10^{-2}$ s, 50 frames per sequence.

### Loss & Training

- Pre-training stage: $L = L_{\text{MSE}} + L_{\alpha} + L_{\text{LPIPS}}$ (rendered image vs. GT) + physical property supervision loss.
- Fine-tuning stage: $L_{\text{DPO}}$ (preference pair ranking loss).

## Key Experimental Results

### PhysAssets Dataset

A large-scale dataset constructed by the authors, containing 50K+ 3D assets sourced from Objaverse, OmniObject3D, ABO, and HSSD. Each asset is annotated with material category, Young's modulus, Poisson's ratio, and reference simulation videos. The material distribution covers 46 classes including plastic (27.3%), wood (16.8%), metal (14.6%), and fabric (14.5%).

### Main Results

| Method | Training | Generalizable | Inference Time | CLIP$_{\text{sim}}$ |
|--------|----------|---------------|----------------|---------------------|
| OmniPhysGS | SDS | ✗ | >12h | 0.2091 |
| DreamPhysics | SDS | ✗ | >0.5h | 0.2291 |
| **PhysGM (w/o DPO)** | Supervised | ✓ | <1min | 0.2693 |
| **PhysGM (w/ DPO)** | DPO | ✓ | <1min | **0.2748** |

- PhysGM outperforms all baselines across 5 material categories (metal / jelly / plasticine / snow / sand).
- User Preference Rate (UPR): PhysGM w/ DPO achieves 42.8% (1-of-4 selection; random baseline 25%), far exceeding OmniPhysGS at 10% and DreamPhysics at 17.2%.

### Multi-View Reconstruction Quality (GSO Dataset)

| Method | Resolution | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|------------|-------|-------|--------|
| LGM | 256 | 21.44 | 0.832 | 0.122 |
| **PhysGM** | 256 | **25.47** | **0.916** | **0.071** |
| GS-LRM | 512 | 30.52 | 0.952 | 0.050 |
| **PhysGM** | 512 | 28.95 | **0.953** | **0.039** |

PhysGM surpasses GS-LRM on LPIPS using only 10% of the training data.

### Ablation Study

Validation of the DPO fine-tuning stage:

- Without DPO → with DPO: CLIP$_{\text{sim}}$ improves from 0.2693 → 0.2748; UPR improves from 30% → 42.8%.
- DPO yields consistent gains across all 5 material categories, with particularly notable improvements for metal (UPR 30%→49%) and snow (26%→47%).
- These results demonstrate that DPO effectively transforms statistically plausible physical priors into a generator with superior perceptual quality.

### Key Findings

1. **Feed-forward inference is fully viable**: End-to-end 4D synthesis is achieved via a single forward pass (<30 s) plus MPM simulation, more than 720× faster than SDS-based methods.
2. **Joint training outperforms modular concatenation**: Joint prediction of 3DGS and physical properties provides mutual reinforcement, avoiding information fragmentation.
3. **DPO as a replacement for SDS**: No differentiable physics engine is required; ranking feedback alone effectively aligns physical realism.
4. **Strong generalization**: The method handles diverse scenarios including stretching, twisting, multi-object multi-material interactions, and real-world images.

## Highlights & Insights

- **First feed-forward physical 4D synthesis framework**: From a single image to a complete physics-simulation animation in under 1 minute, realizing a paradigm shift from per-scene optimization to amortized inference.
- **Innovative DPO training**: The preference alignment idea from RLHF is introduced into the physical simulation domain; SAM-2 and CoTracker-3 automatically construct preference pairs, entirely bypassing the constraints of differentiable simulators.
- **Probabilistic physical prediction**: The physics head outputs a distribution rather than a point estimate, enabling uncertainty quantification while providing the basis for DPO sampling.
- **PhysAssets large-scale dataset**: 50K+ 3D assets with physical annotations fill a critical gap in the field.
- **Speed–quality win-win**: Quality is not sacrificed for speed; PhysGM comprehensively outperforms SDS methods requiring hours of optimization across all metrics.

## Limitations & Future Work

- **High MPM simulation cost**: MPM itself remains the primary computational bottleneck, limiting large-scale real-time applications.
- **Sim-to-Real Gap**: Training data is synthetic, and simplified constitutive models have an inherent gap from real-world physics, affecting generalization to real scenes.
- **GT video dependency on FramePack**: Reference videos are generated by FramePack rather than captured from real physical recordings, which may introduce bias.
- **Limited material categories**: The current framework supports 5 major constitutive model classes and does not yet cover complex physical phenomena such as fluids and fracture.
- **Physical annotation quality**: Physical properties are automatically annotated by an MLLM (Qwen3VL), introducing annotation noise.

## Related Work & Insights

| Method | Requires Pre-built 3DGS | Automatic Physical Params | Generalizable | Inference Time |
|--------|------------------------|--------------------------|---------------|----------------|
| PhysGaussian | ✗ | Manual | ✓ | — |
| DreamPhysics | ✗ | Young's modulus only | ✓ | >0.5h |
| PhysDreamer | ✗ | Young's modulus only | ✓ | >1h |
| OmniPhysGS | ✗ | Material category only | ✓ | >12h |
| PhysSplat | ✗ | ✓ (requires LLM) | ✗ | <2min |
| **PhysGM** | **✓ (not required)** | **✓ (fully automatic)** | **✓** | **<30s** |

PhysGM is the only method that simultaneously satisfies "no pre-built 3DGS required," "fully automatic physical parameters," "strong generalization," and "fast inference."

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Both the first feed-forward physical 4D synthesis framework and the application of DPO for physics simulation alignment are entirely novel contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comparisons across 5 material categories, ablations, user studies, and multi-view reconstruction are all covered, though quantitative evaluation on real-world scenes is lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ — Motivation is clear, method description is complete, and supplementary materials are thorough.
- Value: ⭐⭐⭐⭐⭐ — Advances physical 4D synthesis from hour-level optimization to minute-level inference, accompanied by a large-scale dataset, providing significant impetus for future work.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] AnchorSplat: Feed-Forward 3D Gaussian Splatting with 3D Geometric Priors](anchorsplat_feed-forward_3d_gaussian_splatting_with_3d_geometric_priors.md)
- [\[CVPR 2026\] Off The Grid: Detection of Primitives for Feed-Forward 3D Gaussian Splatting](off_the_grid_detection_of_primitives_for_feed-forward_3d_gaussian_splatting.md)
- [\[CVPR 2026\] Pano3DComposer: Feed-Forward Compositional 3D Scene Generation from Single Panoramic Image](pano3dcomposer_feed-forward_compositional_3d_scene_generation_from_single_panora.md)
- [\[CVPR 2026\] EmbodiedSplat: Online Feed-Forward Semantic 3DGS for Open-Vocabulary 3D Scene Understanding](embodiedsplat_online_feed-forward_semantic_3dgs_for_open-vocabulary_3d_scene_und.md)
- [\[CVPR 2026\] PhysGS: Bayesian-Inferred Gaussian Splatting for Physical Property Estimation](physgs_bayesian-inferred_gaussian_splatting_for_physical_property_estimation.md)

<!-- RELATED:END -->
