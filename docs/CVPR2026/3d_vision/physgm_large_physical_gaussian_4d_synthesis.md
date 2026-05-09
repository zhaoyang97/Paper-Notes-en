---
title: >-
  [Paper Note] PhysGM: Large Physical Gaussian Model for Feed-Forward 4D Synthesis
description: >-
  [CVPR 2026][3D Vision][4D synthesis] The first framework for feed-forward prediction of 3DGS and physical attributes (material category, Young's modulus, Poisson's ratio) from a single image. A two-stage training pipeline (supervised pretraining + DPO preference fine-tuning) entirely bypasses SDS and differentiable physics engines. Combined with the 50K+ PhysAssets dataset, the method generates high-fidelity 4D physical simulations within one minute, surpassing per-scene optimization methods in both CLIP similarity and human preference rate.
tags:
  - CVPR 2026
  - 3D Vision
  - 4D synthesis
  - physics-aware Gaussian
  - feed-forward inference
  - DPO alignment
  - single image to 4D
  - MPM simulation
date: 2026-05-08
content_hash: 283dceb7fa7062ad
---

# PhysGM: Large Physical Gaussian Model for Feed-Forward 4D Synthesis

**Conference**: CVPR 2026
**arXiv**: [2508.13911](https://arxiv.org/abs/2508.13911)
**Code**: [Project Page](https://hihixiaolv.github.io/PhysGM.github.io/)
**Area**: 3D Vision / Physical Simulation
**Keywords**: 4D synthesis, physics-aware Gaussian, feed-forward inference, DPO alignment, single image to 4D, MPM simulation

## TL;DR
The first framework for feed-forward prediction of 3DGS and physical attributes (material category, Young's modulus, Poisson's ratio) from a single image. A two-stage training pipeline (supervised pretraining + DPO preference fine-tuning) entirely bypasses SDS and differentiable physics engines. Combined with the 50K+ PhysAssets dataset, the method generates high-fidelity 4D physical simulations within one minute, surpassing per-scene optimization methods in both CLIP similarity and human preference rate.

## Background & Motivation
**Background**: Physical 4D synthesis conventionally requires multi-view 3DGS reconstruction (hours), manual physical parameter specification, and subsequent simulation. SDS-based methods (OmniPhysGS/DreamPhysics) attempt to distill physical priors from video models, but require differentiable physics engines, making them computationally expensive and unstable.

**Limitations of Prior Work**: Three key bottlenecks: (a) dependence on pre-reconstructed 3DGS (dense multi-view inputs and per-scene optimization); (b) physical attributes either manually specified or SDS-optimized (inflexible/unstable); (c) naive coupling of 3DGS and physics modules that ignores physical cues embedded in appearance.

**Key Challenge**: Per-scene optimization inherently lacks generalizability — each new scene must be processed from scratch. SDS is data-driven but requires a differentiable physics engine and remains unstable.

**Goal**: Can per-scene optimization be entirely bypassed by learning a generative model that directly synthesizes complete physical 4D simulations from sparse inputs via feed-forward inference?

**Key Insight**: Reframe the problem from "slow iterative reconstruction" to "amortized feed-forward inference" — train a large Transformer on large-scale data to learn generalizable physical priors.

**Core Idea**: A feed-forward Transformer that jointly predicts 3DGS and physical attributes, combined with probabilistic physical modeling and DPO preference fine-tuning (instead of SDS), completing 4D inference in a single forward pass.

## Method

### Overall Architecture
Input: 1–4 RGB images + camera parameters → DINOv3 image encoding + Plücker ray camera encoding → token concatenation + 3 global tokens → 24-layer Transformer → dual-branch prediction: DPT Head → 3DGS parameters $\psi$ + Physics Head → physical attribute distribution $\theta$ → MPM simulator → 4D dynamic sequence.

### Key Designs

1. **Multi-Modal Tokenization and Global Physics Tokens**:

    - Function: Uniformly encode image and geometric information; introduce global tokens to aggregate scene-level physical information.
    - Mechanism: DINOv3 encodes image patches; Plücker ray coordinates encode per-pixel primary rays; concatenated tokens are appended with 3 learnable global tokens (used by the physics head). For single-image inference, MVAdapter synthesizes auxiliary back/left/right views.
    - Design Motivation: Global tokens prevent physical attribute prediction from relying on local features, enabling inference of material properties from holistic scene appearance cues.

2. **Probabilistic Physical Attribute Prediction Head**:

    - Function: Predict material category (classification) and probability distributions over continuous physical parameters (regression) from global tokens.
    - Mechanism: Classification head $f_{material}(g_k) \to C$; regression head outputs mean and variance $(\mu_\theta, \log\sigma_\theta^2) = f_{phys}(g_k)$, defining the conditional distribution $P(\theta|I) = \mathcal{N}(\theta|\mu_\theta, \text{diag}(\sigma_\theta^2))$; physical parameters are obtained by sampling at inference time.
    - Design Motivation: Probabilistic modeling captures the inherent uncertainty that "the same appearance may correspond to multiple physical parameters" and enables sampling of multiple candidates for DPO.

3. **DPO Preference Fine-Tuning as a Replacement for SDS**:

    - Function: Align simulation outputs with ground-truth videos via preference learning, entirely bypassing differentiability requirements.
    - Mechanism: The pretrained policy is frozen as $\pi_{ref}$; $K$ candidate physical parameter sets are sampled from $\pi_\omega$, each passed through MPM simulation and rendering; perceptual distances to ground truth are computed using SAM-2 segmentation and CoTracker-3 trajectory extraction; the closest/farthest candidates become winner/loser pairs; the DPO loss is minimized: $L_{DPO} = -\mathbb{E}[\log\sigma(\beta\log\frac{\pi_\omega(\phi_w|z)}{\pi_{ref}(\phi_w|z)} - \beta\log\frac{\pi_\omega(\phi_l|z)}{\pi_{ref}(\phi_l|z)})]$
    - Design Motivation: SDS requires gradients to flow through the physics engine. DPO treats simulation and rendering as a black box — learning from output quality comparisons alone — substantially simplifying training.

4. **PhysAssets Dataset (50K+)**:

    - Function: Construct the first large-scale dataset pairing 3D assets with physical annotations and reference simulation videos.
    - Mechanism: Assets are aggregated from Objaverse, OmniObject3D, ABO, and HSSD; Qwen3VL, a multimodal LLM, infers material categories and physical parameters from multi-view images; Framepack generates ground-truth simulation videos.
    - Design Motivation: Supports both supervised pretraining (with GT physical parameters) and DPO fine-tuning (with GT simulation videos), filling a critical data gap in the field.

### Loss & Training
Two stages: Stage 1 performs large-scale supervised pretraining with joint optimization of reconstruction loss (MSE + Alpha + LPIPS) and physical prediction loss. Stage 2 freezes the backbone and fine-tunes only the physics head via DPO. Training uses 32 × A800 GPUs for 3 days with batch size 8 per GPU. MPM simulation parameters: sub-step time $2\times10^{-5}$ s, frame time $4\times10^{-2}$ s, 50 frames per sequence.

## Key Experimental Results

### Main Results (5 Material Categories)

| Method | metal CLIP | jelly CLIP | plast. CLIP | snow CLIP | sand CLIP | avg CLIP | avg UPR |
|------|-----------|-----------|------------|----------|----------|---------|---------|
| OmniPhysGS | 0.215 | 0.229 | 0.214 | 0.183 | 0.205 | 0.209 | 10% |
| DreamPhysics | 0.227 | 0.246 | 0.244 | 0.207 | 0.222 | 0.229 | 17.2% |
| PhysGM (w/o DPO) | 0.270 | 0.270 | 0.255 | 0.254 | 0.298 | 0.269 | 30% |
| **PhysGM (w/ DPO)** | **0.273** | **0.277** | **0.269** | **0.255** | **0.300** | **0.275** | **42.8%** |

### Ablation Study

| Configuration | avg CLIP_sim | avg UPR | Note |
|------|-------------|---------|------|
| PhysGM w/o DPO | 0.269 | 30% | Pretraining only |
| **PhysGM w/ DPO** | **0.275** | **42.8%** | DPO significantly improves UPR (+12.8%) |

### Key Findings
- **Feed-forward outperforms per-scene optimization**: PhysGM surpasses SDS baselines (requiring hours per scene) on both CLIP_sim and UPR across all material types — demonstrating that feed-forward inference does not sacrifice quality.
- **DPO improves perceptual quality rather than metric scores**: Post-DPO CLIP_sim improvement is marginal, but UPR increases substantially by 12.8 percentage points — indicating that preference fine-tuning primarily enhances human-perceived physical realism.
- **Probabilistic modeling is the foundation for DPO**: Replacing the probability distribution with a point estimate prevents effective multi-candidate sampling for DPO, causing fine-tuning to fail.
- **Joint training outperforms disentangled modules**: Jointly predicting appearance and physics outperforms separate modules — validating the hypothesis that appearance encodes physical cues.
- **Speed**: Complete 4D simulation in under 1 minute vs. hours for SDS-based methods.

## Highlights & Insights
- **Feed-forward physical inference paradigm** — A paradigm shift from "per-scene optimization" to "amortized inference." PhysGM demonstrates that large models trained on large-scale data can learn generalizable physical priors, replacing hours of optimization with a single forward pass.
- **Novel application of DPO in generative modeling** — Transferring DPO from language model preference alignment to physical simulation quality alignment. The idea of constructing preference pairs from non-differentiable simulator outputs is highly innovative.
- **Elegant probabilistic physical modeling** — Predicting distributions rather than point estimates simultaneously captures uncertainty and provides the sampling basis for DPO — a design that serves two purposes at once.

## Limitations & Future Work
- **LLM-dependent dataset annotation**: Physical parameters inferred by Qwen3VL may lack precision; measurements from dedicated physical experiments would be more reliable.
- **GT video quality**: Reference simulation videos generated by Framepack may not themselves be fully physically realistic.
- **Limited material categories**: Only 5 material categories are covered; composite materials and fluids are not addressed.
- **Predominantly single-object scenes**: The capability to handle multi-object interaction scenes remains to be validated.
- **Future directions**: Incorporating real physical experiment videos as ground truth; expanding material categories to include fluids and cloth; supporting interactive physical manipulation by users.

## Related Work & Insights
- **vs PhysGaussian**: PhysGaussian pioneered 3DGS + MPM coupling but requires manual parameter specification per scene; PhysGM automatically predicts physical attributes without pre-reconstruction.
- **vs OmniPhysGS/DreamPhysics**: These SDS-based methods require hours of per-scene optimization; PhysGM completes inference in 1 minute with superior results.
- **vs LGM/GS-LRM**: These feed-forward 3D reconstruction methods handle only static scenes; PhysGM is the first to embed physical inference for dynamic 4D generation.
- **Insight**: The DPO + non-differentiable simulation paradigm is generalizable to any generative task requiring feedback from black-box simulators (robotic control, fluid simulation, etc.).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First feed-forward physics-aware 4D generation framework; DPO as a replacement for SDS is a genuinely novel contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five-material comparison + ablations + user study; more quantitative ablations would strengthen the paper.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, systematic methodology, and naturally motivated two-stage training.
- Value: ⭐⭐⭐⭐⭐ Pioneering value for 4D synthesis and physics-aware 3D vision.

## Experiments

### Table 1: Physical Simulation Quality Comparison (5 Material Categories)

| Method | metal CLIPsim | jelly CLIPsim | plasticine CLIPsim | snow CLIPsim | sand CLIPsim | avg CLIPsim | avg UPR |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| OmniPhysGS | 0.2149 | 0.2291 | 0.2135 | 0.1834 | 0.2047 | 0.2091 | 10% |
| DreamPhysics | 0.2273 | 0.2459 | 0.2437 | 0.2071 | 0.2217 | 0.2291 | 17.2% |
| PhysGM (w/o DPO) | 0.2698 | 0.2700 | 0.2547 | 0.2541 | 0.2980 | 0.2693 | 30% |
| **PhysGM (w/ DPO)** | **0.2732** | **0.2774** | **0.2691** | **0.2548** | **0.2997** | **0.2748** | **42.8%** |

### Table 2: Multi-View Reconstruction Quality (GSO Dataset)

| Method | Resolution | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|:---:|:---:|:---:|:---:|
| LGM | 256 | 21.44 | 0.832 | 0.122 |
| PhysGM (ours) | 256 | **25.47** | **0.916** | **0.071** |
| GS-LRM | 512 | **30.52** | 0.952 | 0.050 |
| PhysGM (ours) | 512 | 28.95 | **0.953** | **0.039** |

### Table 3: Efficiency and Generalization Comparison

| Method | Training Paradigm | Generalizable | Inference Time | CLIPsim |
|------|:---:|:---:|:---:|:---:|
| OmniPhysGS | SDS | ✗ | >12h | 0.2091 |
| DreamPhysics | SDS | ✗ | >0.5h | 0.2291 |
| **PhysGM** | **DPO** | **✓** | **<1min** | **0.2748** |

## Key Findings

- **Feed-forward vs. per-scene optimization**: PhysGM completes single-image-to-4D simulation in under 1 minute (inference <30s + MPM simulation), compared to >12h for OmniPhysGS and >0.5h for DreamPhysics.
- **DPO substantially improves simulation quality**: Adding DPO raises CLIPsim from 0.2693 to 0.2748 and UPR from 30% to 42.8% (a 12.8 percentage point gain in human preference rate).
- **Reconstruction quality is competitive with specialized methods**: PSNR exceeds LGM by 4.03 dB at 256 resolution; at 512 resolution, LPIPS is superior to GS-LRM using only 10% of its training data.
- **Only fully automatic solution**: PhysGM is the only method that simultaneously requires no pre-optimized 3DGS, no predefined physical parameters, generalizes across scenes, does not depend on an LLM at inference time, and completes inference in under 30 seconds.

## Highlights

- **Paradigm innovation**: Transforms physical 4D synthesis from per-scene optimization to feed-forward inference, achieving a speedup of over 720× compared to OmniPhysGS (12h).
- **DPO for physical simulation alignment**: The first application of DPO in the physical simulation domain, bypassing the requirement for differentiable physics engines by constructing preference pairs from black-box simulator outputs.
- **Probabilistic physical prediction**: Outputting distributions over physical attributes rather than point estimates naturally supports DPO sampling and uncertainty modeling.
- **SAM-2 + CoTracker-3 for preference label construction**: An automated preference annotation pipeline that quantifies simulation fidelity relative to ground truth via instance segmentation and trajectory tracking.
- **Large-scale physical annotation dataset PhysAssets**: 50K+ assets spanning metal, jelly, plasticine, snow, sand, and other material categories, filling a critical data gap in the field.

## Limitations & Future Work

- **MPM simulation as a computational bottleneck**: MPM simulation (200³ grid resolution) remains the primary time-consuming step in 4D synthesis, limiting real-time applicability; efficient alternatives for fluid and fracture simulation are lacking.
- **Sim-to-real gap**: Training data is based on synthetic simulation videos (generated by Framepack), and simplified constitutive models introduce an inherent discrepancy with real physics, limiting robustness in real-world deployment.
- **SH degree limitation**: Spherical harmonics are set to degree 0 (diffuse only), precluding modeling of view-dependent specular effects.
- **Single-image depth ambiguity**: 3D reconstruction accuracy from a single image is limited by occlusion and depth uncertainty.
- **Material coverage**: Although 50K assets are included, physical attribute annotations are inferred by an MLLM rather than physically measured, limiting annotation accuracy.

## Related Work & Insights

- **vs PhysGaussian**: Pioneered 3DGS + MPM coupling but requires manual per-scene parameter tuning; PhysGM predicts physical attributes automatically.
- **vs DreamPhysics/OmniPhysGS**: Distill physical parameters from video models via SDS, requiring a differentiable simulator and hours of optimization per scene; PhysGM uses DPO to bypass differentiability.
- **vs PhysDreamer**: Also optimizes Young's modulus via SDS but lacks generalizability; PhysGM is the first to achieve cross-scene generalization.
- **vs PhysSplat**: Uses an LLM to infer physical parameters but depends on pre-reconstructed 3DGS; PhysGM operates end-to-end in a feed-forward manner.
- **vs LGM/GS-LRM**: Feed-forward 3D reconstruction methods that predict only static geometry, without physical attributes.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First feed-forward physics-aware 4D synthesis framework; DPO for physical alignment is a field-first contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Quantitative comparison across 5 material categories + multi-view reconstruction ablation + user study; real-world quantitative evaluation is lacking.
- Writing Quality: ⭐⭐⭐⭐ Clear logic, well-motivated two-stage training, and detailed method description.
- Value: ⭐⭐⭐⭐⭐ Fundamentally transforms the paradigm of physical 4D synthesis — from hour-scale optimization to second-scale inference.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Particulate: Feed-Forward 3D Object Articulation](particulate_feed-forward_3d_object_articulation.md)
- [\[CVPR 2026\] MoRe: Motion-aware Feed-forward 4D Reconstruction Transformer](more_motion-aware_feed-forward_4d_reconstruction_transformer.md)
- [\[CVPR 2026\] Off The Grid: Detection of Primitives for Feed-Forward 3D Gaussian Splatting](off_the_grid_detection_of_primitives_for_feed-forward_3d_gaussian_splatting.md)
- [\[CVPR 2026\] AnchorSplat: Feed-Forward 3D Gaussian Splatting with 3D Geometric Priors](anchorsplat_feed-forward_3d_gaussian_splatting_with_3d_geometric_priors.md)
- [\[CVPR 2026\] PanoVGGT: Feed-Forward 3D Reconstruction from Panoramic Imagery](panovggt_feed-forward_3d_reconstruction_from_panoramic_imagery.md)

</div>

<!-- RELATED:END -->
