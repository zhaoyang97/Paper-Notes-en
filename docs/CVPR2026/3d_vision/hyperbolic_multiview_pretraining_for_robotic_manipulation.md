---
title: >-
  [Paper Note] HyperMVP: Hyperbolic Multiview Pretraining for Robotic Manipulation
description: >-
  [CVPR 2026][3D Vision][Hyperbolic space] This paper proposes HyperMVP, the first framework for 3D multiview self-supervised pretraining in hyperbolic space. It learns hyperbolic multiview representations via a GeoLink en…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "Hyperbolic space"
  - "multiview pretraining"
  - "robotic manipulation"
  - "self-supervised learning"
  - "3D representation"
date: 2026-05-08
content_hash: 1df62fd3fe80ab4e
---

# HyperMVP: Hyperbolic Multiview Pretraining for Robotic Manipulation

**Conference**: CVPR 2026 Findings  
**arXiv**: [2603.04848](https://arxiv.org/abs/2603.04848)  
**Code**: To be confirmed  
**Area**: 3D Vision
**Keywords**: Hyperbolic space, multiview pretraining, robotic manipulation, self-supervised learning, 3D representation

## TL;DR

This paper proposes HyperMVP, the first framework for 3D multiview self-supervised pretraining in hyperbolic space. It learns hyperbolic multiview representations via a GeoLink encoder and transfers them to robotic manipulation tasks, achieving a 2.1× performance improvement on the most challenging All Perturbations setting of COLOSSEUM.

## Background & Motivation

Visual pretraining for 3D perception has been shown to effectively improve downstream robotic manipulation performance, yet critical limitations remain:

- Existing methods (e.g., 3D-MVP) are confined to **Euclidean embedding spaces**, whose flat geometry limits the ability to model structural relationships among embeddings.
- Distance metrics in Euclidean space grow linearly, making them ill-suited for representing hierarchical and nested relationships.
- Hyperbolic space distances grow exponentially, making it naturally suited for tree-like/nested structures, yet it remains **entirely unexplored** in robotic manipulation pretraining.

The core idea is to extend visual self-supervised pretraining from Euclidean space to hyperbolic space (Lorentz model), leveraging the geometric properties of hyperbolic space to learn more structured representations and thereby improve the robustness and generalization of manipulation policies.

## Method

### Overall Architecture

HyperMVP follows a pretrain-then-finetune paradigm: (1) pretrain the GeoLink encoder on the 3D-MOV dataset to learn hyperbolic multiview representations; (2) jointly finetune the pretrained encoder with the Robotic View Transformer (RVT) to learn manipulation policies.

### Key Designs

1. **GeoLink Encoder**: Extends the MAE paradigm by rendering 3D point clouds into 5 orthographic view images. The encoder consists of $N=8$ ViT blocks (hidden dimension 768, 8 attention heads), outputting CLS embeddings $\mathbf{f}^{\text{cls}} \in \mathbb{R}^{5 \times 1 \times D}$ and patch embeddings $\mathbf{f}^{\mathrm{p}} \in \mathbb{R}^{5 \times P \times D}$. The core operation lifts Euclidean embeddings onto the Lorentz hyperboloid via the exponential map:
$$\mathbf{x}_s^* = \frac{\sinh(\sqrt{c}\|\mathbf{f}^*\|)}{\sqrt{c}\|\mathbf{f}^*\|}\mathbf{f}^*$$
During finetuning, embeddings are mapped back to Euclidean space via the logarithmic map for compatibility with downstream policies. **Design Motivation**: The exponential distance expansion in hyperbolic space captures semantic hierarchical relationships among patches.

2. **Patch-aware Top-K Neighbor Rank Correlation Loss $L_{\text{corr}}$**: Preserves the semantic topological consistency of patch embeddings across Euclidean and hyperbolic spaces. For each patch, Top-K nearest neighbors are identified in both spaces, and ranking discrepancies are minimized. An ordinal formulation (focusing on "who is closer" rather than "how much closer") is used to avoid convergence issues caused by geometric differences:
$$L_{\text{corr}} = 1 - \frac{1}{5}\sum_{i=1}^{5} g\left(|\mathbf{R}_i^{\mathcal{E}}_{\pi_i^K}|_z \odot |\mathbf{R}_i^{\mathcal{L}}_{\pi_i^K}|_z\right)$$
**Design Motivation**: Direct distance alignment fails to converge due to geometric discrepancies; rank alignment is geometry-agnostic.

3. **Entailment Loss $L_{\text{etl}}$ + Multiview Reconstruction**: Defines entailment cones around hyperbolic CLS embeddings and constrains patch embeddings to fall within these cones, modeling local-to-global semantic alignment. Intra-view reconstruction (standard MAE decoder reconstructing the source view) and inter-view reconstruction (using cross-attention over features from other views to predict the anchor view) are also incorporated to learn multiview consistency.

### Loss & Training

- Pretraining loss: $L_{\text{pretrain}} = L_{\text{hyper}} + L_{\text{recon}}$
    - $L_{\text{hyper}} = \lambda_c L_{\text{corr}} + \lambda_{e1} L_{\text{etl}}(\mathbf{x}^{\text{cls}}, \mathbf{x}^{\mathrm{p}}) + \lambda_{e2} L_{\text{etl}}(\mathbf{x}^{\text{cls}}, \mathbf{x}^{\mathrm{msk}})$ ($\lambda_c=1, \lambda_{e1}=0.5, \lambda_{e2}=0.1$)
    - $L_{\text{recon}} = \lambda_{\text{ita}} L_{\text{intra}} + \lambda_{\text{ite}} L_{\text{inter}}$ ($\lambda_{\text{ita}}=1, \lambda_{\text{ite}}=0.5$)
- Pretrained for 100 epochs, batch size 64, masking ratio 0.75, AdamW (lr=5.12e-4), 8×4090 GPUs.
- Finetuned for 50K steps (simulation) / 4K steps (real-world), LAMB optimizer, lr=2e-3.

### 3D-MOV Dataset

A large-scale dataset of ~200K high-quality 3D point clouds is constructed: 180K objects (Objaverse-XL) + 6,052 scene partitions (ScanNet) + 3,999 vanilla tabletop + 10,001 crowd tabletop (TO-Scene), yielding approximately 1M multiview rendered images in total.

## Key Experimental Results

### Main Results

| Dataset | Metric | HyperMVP | Prev. SOTA | Gain |
|--------|------|----------|----------|------|
| COLOSSEUM Avg (all perturbations) | Success Rate | 47.5% | 35.6% (3D-MVP) | +33.4% |
| COLOSSEUM All Perturbations | Success Rate | 11.2% | 5.3% (3D-MVP) | **2.1×** |
| RLBench 18-task Avg | Success Rate | **71.1%** | 68.0% (SAM2Act) | +3.1% |
| RLBench vs. scratch | Success Rate | 71.1% | 62.9% (RVT) | +13.0% relative |
| Real-world Avg | Success Rate | **60.0%** | 32.9% (RVT) | +27.1% |
| Real-world All Perturbations | Success Rate | 50.0% | 22.2% (RVT) | +27.8% |

### Ablation Study

| Configuration | Key Metric (Avg Success %) | Note |
|------|---------|------|
| HyperMVP (full) | 71.11 | Full model |
| MVT (3D-MVP style) | OOM | Quadratic attention + large-scale pretraining causes out-of-memory |
| MAE* (Euclidean) | 68.22 | Hyperbolic space is genuinely beneficial (+2.89) |
| w/o ScanNet (~194K) | 65.06 | Real-scene data is the most critical component |
| w/o TO-Scene (~186K) | 68.44 | Data diversity > data scale |
| w/o $L_{\text{corr}}$ | 67.72 | Rank correlation loss contributes most (−3.39) |
| w/o $L_{\text{etl}}(\mathbf{x}^{\text{cls}}, \mathbf{x}^{\mathrm{p}})$ | 70.06 | Entailment loss provides a marginal contribution |
| w/o $L_{\text{inter}}$ | 71.00 | Inter-view reconstruction contributes minimally |

### Key Findings

- Hyperbolic representations genuinely outperform Euclidean ones (68.22 → 71.11), with the advantage being more pronounced under perturbation scenarios.
- Data diversity (inclusion of real-scene data) is more important than data scale: 194K with scene data outperforms 186K without.
- The Top-K rank correlation loss $L_{\text{corr}}$ is the most critical loss component, with the largest performance drop upon removal.
- Orthographic projection ensures geometric consistency across views, reducing the marginal benefit of inter-view reconstruction.

## Highlights & Insights

- **Strong novelty**: This is the first work to introduce hyperbolic space into visual pretraining for robotic manipulation, opening a new direction for non-Euclidean geometry in embodied intelligence.
- **Elegant Top-K rank correlation loss design**: Replacing distance alignment with rank correlation elegantly resolves the incomparability of distances between Euclidean and hyperbolic spaces.
- **Thoughtful 3D-MOV dataset design**: Ablations reveal the importance of scene-level data, rather than simply scaling up data volume.
- **Flexible and extensible GeoLink encoder**: Unlike 3D-MVP, it can be adapted to an arbitrary number of input views during finetuning.

## Limitations & Future Work

- Improvement on high-precision tasks (e.g., Place Cups) is limited, constrained by the capabilities of the downstream RVT policy itself.
- Orthographic projection may discard perspective information, whereas real robot cameras typically use perspective imaging.
- The mechanism underlying the benefits of hyperbolic space lacks deeper theoretical analysis (why does hyperbolic space help manipulation?).
- Real-world experiments are conducted at a relatively small scale (only 50 demonstrations and 10 evaluation trials per task).

## Related Work & Insights

- MERU's approach to hyperbolic vision-language alignment is generalized here to an unsupervised multiview setting, suggesting broad potential for hyperbolic space in self-supervised learning.
- The multiview pretraining paradigm of 3D-MVP is extended, demonstrating that the choice of embedding space has a non-trivial impact on downstream tasks.
- The finding that data diversity > data scale offers important guidance for pretraining data engineering.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First work to conduct 3D multiview pretraining in hyperbolic space for robotic manipulation; highly novel direction.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive evaluation across simulation (COLOSSEUM + RLBench), real-world, and ablations; real-world scale is somewhat limited.
- **Writing Quality**: ⭐⭐⭐⭐ — Mathematical derivations are clear, hyperbolic space preliminaries are well-presented, and the paper is well-structured.
- **Value**: ⭐⭐⭐⭐ — Opens a new direction for non-Euclidean representation learning in embodied intelligence; the 3D-MOV dataset has strong reuse potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Ada3Drift: Adaptive Training-Time Drifting for One-Step 3D Visuomotor Robotic Manipulation](ada3drift_adaptive_trainingtime_drifting_for_onest.md)
- [\[ICCV 2025\] RoboTron-Mani: All-in-One Multimodal Large Model for Robotic Manipulation](../../ICCV2025/3d_vision/robotron-mani_all-in-one_multimodal_large_model_for_robotic_manipulation.md)
- [\[NeurIPS 2025\] DynaRend: Learning 3D Dynamics via Masked Future Rendering for Robotic Manipulation](../../NeurIPS2025/3d_vision/dynarend_learning_3d_dynamics_via_masked_future_rendering_for_robotic_manipulati.md)
- [\[CVPR 2026\] Real2Edit2Real: Generating Robotic Demonstrations via a 3D Control Interface](real2edit2real_generating_robotic_demonstrations_via_a_3d_control_interface.md)
- [\[CVPR 2026\] Stepper: Stepwise Immersive Scene Generation with Multiview Panoramas](stepper_stepwise_immersive_scene_generation_with_multiview_panoramas.md)

</div>

<!-- RELATED:END -->
