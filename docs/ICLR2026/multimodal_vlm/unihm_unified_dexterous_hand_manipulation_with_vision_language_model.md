---
title: >-
  [Paper Note] UniHM: Unified Dexterous Hand Manipulation with Vision Language Model
description: >-
  [ICLR 2026][Multimodal VLM][dexterous hand manipulation] This paper proposes UniHM, the first unified language-conditioned dexterous hand manipulation framework. It maps heterogeneous robotic hands into a shared discrete space via a morphology-agnostic VQ codebook, leverages a VLM for instruction-driven manipulation sequence generation, and ensures physical feasibility through physics-guided dynamic refinement.
tags:
  - ICLR 2026
  - Multimodal VLM
  - dexterous hand manipulation
  - VLM
  - unified tokenizer
  - physics-guided dynamic refinement
  - cross-morphology generalization
date: 2026-05-08
content_hash: cc79b657cd613f4b
---

# UniHM: Unified Dexterous Hand Manipulation with Vision Language Model

**Conference**: ICLR 2026
**arXiv**: [2603.00732](https://arxiv.org/abs/2603.00732)
**Code**: [GitHub](https://unihm.github.io/)
**Area**: Multimodal VLM
**Keywords**: dexterous hand manipulation, VLM, unified tokenizer, physics-guided dynamic refinement, cross-morphology generalization

## TL;DR

This paper proposes UniHM, the first unified language-conditioned dexterous hand manipulation framework. It maps heterogeneous robotic hands into a shared discrete space via a morphology-agnostic VQ codebook, leverages a VLM for instruction-driven manipulation sequence generation, and ensures physical feasibility through physics-guided dynamic refinement.

## Background & Motivation

Dexterous hand manipulation requires perceiving, grasping, and reconfiguring objects in complex environments. Generating diverse, long-horizon, and physically feasible manipulation sequences is critical for advancing humanoid robot applications.

Limitations of prior work:
- **Object-centric methods** (UniDexGrasp, DexGraspNet, etc.): lack open-vocabulary instruction guidance and can only handle fixed sequences.
- **Language-guided grasping methods** (SemGrasp, AffordDexGrasp, etc.): primarily generate static grasp poses, neglect temporal structure, and cannot produce smooth, continuous manipulation sequences.
- **Existing VLM-based manipulation methods** (MotionGPT, HOIGPT, etc.): mainly target digital hands or low-DoF grippers, lacking cross-morphology generalization and physical feasibility guarantees.

**Goal**: Generate dynamic dexterous manipulation sequences directly from images and open-vocabulary instructions, supporting multiple hand morphologies without relying on teleoperation data.

## Method

### Overall Architecture

A three-stage pipeline: (1) morphology-agnostic motion tokenization; (2) language-guided VLM manipulation sequence generation; (3) physics-aware decoding and dynamic refinement.

### Key Designs

1. **Unified Hand-Dexterous Tokenizer**: A shared VQ-VAE codebook $\mathcal{Z} = \{\mathbf{e}_k\}_{k=1}^K$ maps five different hand morphologies (MANO, Shadow Hand, Allegro Hand, etc.) into a unified discrete space. Each morphology has a dedicated encoder $E_h$ and decoder $D_h$, with quantization $c = \arg\min_k \|E_h(\mathbf{x}^{(h)}) - \mathbf{e}_k\|_2^2$. New morphologies align their encoders via knowledge distillation: $\mathcal{L}_{\text{distill}} = \|E_{\text{new}}(\mathbf{x}_{\text{new}}) - E_{\text{ref}}(\mathbf{x}_{\text{ref}})\|_2^2$, circumventing the non-differentiability of the quantization step. Cross-morphology translation reduces to encode–quantize–decode: $\hat{\mathbf{x}}^{(j)} = D_j(\mathbf{e}_{Q(E_i(\mathbf{x}^{(i)}))})$.

2. **VLM-driven Manipulation Generation**: A decoupled architecture is adopted — a CLIPort perception module infers target trajectory $\mathcal{T}_{\text{tar}}$ from RGB-D input and instructions, while Point-SAM segments the target object point cloud $\mathcal{P}_{\text{obj}}$. Using Qwen3-0.6B as the backbone, the encoded initial hand pose, target trajectory, object point cloud, and text tokens are concatenated and fed into the VLM to autoregressively generate the manipulation token sequence. A progressive masking curriculum is applied, gradually increasing the masking ratio from full teacher forcing to fully autoregressive generation.

3. **Physics-guided Dynamic Refinement**: Frame-by-frame Gauss–Newton optimization with three energy terms:

    - **Contact energy** $\mathcal{E}_{\text{contact}}$: based on signed point-to-plane distances from fingertips to the object surface, using an asymmetric smooth penalty.
    - **Generation prior** $\mathcal{E}_{\text{gen}}$: penalizes deviation from VLM-generated configurations to preserve semantic intent.
    - **Temporal prior** $\mathcal{E}_{\text{time}}$: regularizes first-order (velocity) and second-order (acceleration) temporal differences to ensure smooth and coherent motion.

### Loss & Training

VQ-VAE training: reconstruction loss + codebook loss $\mathcal{L}_{\text{vq}} = \|\text{sg}[\mathbf{z}_e] - \mathbf{z}_q\|_2^2 + \beta\|\mathbf{z}_e - \text{sg}[\mathbf{z}_q]\|_2^2$

Physical optimization uses Levenberg–Marquardt damped Gauss–Newton iterations:
$$(J_t^T J_t + \mathbf{W}_{\text{gen}} + \mathbf{W}_{\text{vel}} + \mathbf{W}_{\text{acc}} + \lambda I)\Delta q_t = -J_t^T r_{\text{contact}}(q_t) - \tilde{\mathbf{W}}$$

Data annotation: GPT-4o generates five open-vocabulary instructions per keyframe; Dex-Retargeting maps MANO poses to five robotic hand morphologies.

## Key Experimental Results

### Main Results

| Method | DexYCB Seen MPJPE↓ | FID↓ | Diversity (GT=125.53) | DexYCB Unseen MPJPE↓ | FID↓ |
|--------|-------------------|------|-----------------------|----------------------|------|
| TM2T | 85.33 | 54.83 | 37.12 | 94.22 | 55.94 |
| MDM | 88.06 | 52.33 | 33.95 | 93.05 | 55.13 |
| FlowMDM | 82.75 | 48.05 | 61.25 | 86.13 | 51.33 |
| MotionGPT3 | 74.80 | 43.35 | 72.51 | 77.93 | 46.14 |
| **UniHM** | **61.40** | **31.24** | 39.62 | **63.56** | **41.03** |

| Real-World Success Rate | Grab | Pick&Place | Pull&Push | Open&Close |
|------------------------|------|------------|-----------|------------|
| MDM+Retarget (Seen) | 20% | 10% | 0% | 5% |
| MotionGPT3+Retarget (Seen) | 30% | 15% | 25% | 25% |
| **UniHM (Seen)** | **65%** | **50%** | **60%** | **55%** |
| **UniHM (Unseen)** | **60%** | **35%** | **55%** | **45%** |

### Ablation Study

| Configuration | DexYCB Seen MPJPE↓ | FID↓ | DexYCB Unseen MPJPE↓ | FID↓ | Note |
|--------------|-------------------|------|----------------------|------|------|
| w/o Depth Input | 85.47 | 56.36 | 90.12 | 77.38 | Severe degradation with RGB only |
| w/o Masked Training | 73.41 | 44.87 | 74.63 | 43.09 | Progressive masking is important |
| w/o Physical Refinement | 65.78 | 33.57 | 65.39 | 45.06 | Physical optimization improves feasibility |
| **Full UniHM** | **61.40** | **31.24** | **63.56** | **41.03** | All modules are indispensable |

### Key Findings

- UniHM consistently outperforms state-of-the-art methods on DexYCB and OakInk, reducing MPJPE by 18% in both seen and unseen settings.
- Real-world grasping success rates substantially exceed baselines (Grab: 65% vs. 30%), with good generalization to unseen objects.
- Depth input is critical for 3D scene understanding; removing it increases MPJPE by approximately 40%.
- Physics-guided refinement significantly reduces interpenetration and improves motion stability.
- The unified codebook enables plug-and-play transfer across five hand morphologies.

## Highlights & Insights

- UniHM is the first fully unified language-conditioned dexterous hand manipulation framework, extending the paradigm from static pose generation to dynamic sequence manipulation.
- The morphology-agnostic codebook design is elegant: knowledge distillation circumvents the non-differentiability of VQ, and new morphologies require only training new encoder–decoder pairs.
- The model is trained solely on human video data, eliminating the need for costly teleoperation data collection.
- Physics-guided optimization unifies the generation prior, temporal prior, and contact constraints within a single coherent framework.

## Limitations & Future Work

- The method relies on RGB-D input and lacks tactile and force feedback.
- The contact and friction energy terms are relatively simplified.
- Bimanual coordination and tool-use scenarios are not addressed.
- The Qwen3-0.6B backbone is relatively small; larger models may yield further improvements.
- CLIPort requires fine-tuning for new scenes; end-to-end unification of perception and generation is a promising future direction.

## Related Work & Insights

- This work extends the VQ-VAE tokenization paradigm from human motion generation to multi-morphology manipulation; the codebook sharing strategy has broad applicability.
- The progressive masking training curriculum is an effective strategy for mitigating exposure bias in autoregressive generation.
- Physics-guided post-processing maintains a balance between generation flexibility and physical feasibility.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First unified language-conditioned dexterous hand manipulation framework with multiple novel designs.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated on DexYCB, OakInk, and real-world settings with comprehensive ablations; quantitative evaluation of cross-morphology generalization is somewhat limited.
- **Writing Quality**: ⭐⭐⭐⭐ Method descriptions are detailed and the physical optimization derivations are clearly presented.
- **Value**: ⭐⭐⭐⭐⭐ Addresses core pain points in dexterous hand manipulation with substantial practical application potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Unified Vision-Language Modeling via Concept Space Alignment](unified_vision-language_modeling_via_concept_space_alignment.md)
- [\[ICLR 2026\] EgoHandICL: Egocentric 3D Hand Reconstruction with In-Context Learning](egohandicl_egocentric_3d_hand_reconstruction_with_in-context_learning.md)
- [\[ICLR 2026\] Spatial-DISE: A Unified Benchmark for Evaluating Spatial Reasoning in Vision-Language Models](spatial-dise_a_unified_benchmark_for_evaluating_spatial_reasoning_in_vision-lang.md)
- [\[ICLR 2026\] Enhanced Continual Learning of Vision-Language Models with Model Fusion](enhanced_continual_learning_of_vision-language_models_with_model_fusion.md)
- [\[CVPR 2026\] UniGame: Turning a Unified Multimodal Model Into Its Own Adversary](../../CVPR2026/multimodal_vlm/unigame_turning_a_unified_multimodal_model_into_its_own_adversary.md)

</div>

<!-- RELATED:END -->
