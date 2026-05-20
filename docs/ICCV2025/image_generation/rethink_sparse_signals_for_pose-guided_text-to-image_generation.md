---
title: >-
  [Paper Note] Rethink Sparse Signals for Pose-guided Text-to-Image Generation
description: >-
  [ICCV 2025][Image Generation][ControlNet] This paper proposes SP-Ctrl (Spatial-Pose ControlNet), which replaces the fixed RGB encoding of OpenPose with learnable Spatial-Pose Representations (SPR) and introduces a Keypoi…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "ControlNet"
  - "OpenPose"
  - "Sparse Signal"
  - "Keypoint Concept Learning"
  - "Pose-guided Generation"
date: 2026-05-08
content_hash: 4ccc8667481b4ca0
---

# Rethink Sparse Signals for Pose-guided Text-to-Image Generation

**Conference**: ICCV 2025
**arXiv**: [2506.20983](https://arxiv.org/abs/2506.20983)  
**Code**: [GitHub](https://github.com/WenjieXuan/SP-Ctrl) (noted as "Codes are available" in the paper)  
**Area**: Image Generation / Pose-guided Generation / Spatially Controllable Generation
**Keywords**: ControlNet, OpenPose, Sparse Signal, Keypoint Concept Learning, Pose-guided Generation

## TL;DR
This paper proposes SP-Ctrl (Spatial-Pose ControlNet), which replaces the fixed RGB encoding of OpenPose with learnable Spatial-Pose Representations (SPR) and introduces a Keypoint Concept Learning (KCL) strategy that leverages cross-attention heatmap constraints to improve keypoint alignment. The method enables sparse pose signals to achieve pose control accuracy comparable to dense signals (depth maps / DensePose), while preserving image diversity and cross-species generation capability.

## Background & Motivation

### Problem Background
Pose-guided text-to-image (T2I) generation is a fundamental task in controllable image synthesis. Recent work has favored **dense signals** (depth maps, normal maps, DensePose, SMPL) for precise spatial guidance, as sparse signals (e.g., OpenPose) are generally considered insufficient in control fidelity.

### Drawbacks of Dense Signals

**Poor flexibility**: Dense signals are typically extracted from reference images, which limits the creation and editing of new conditions.

**Conflict with text prompts**: Dense signals impose strong constraints on object shape and silhouette; when these conflict with textual descriptions, generation quality degrades.

### Three Key Advantages of Sparse Signals

**Shape-agnostic**: Keypoints are anatomical abstractions of objects and do not prescribe specific shapes.

**Category-agnostic**: Keypoint definitions are shared across species (e.g., mammalian keypoint topology is structurally similar).

**Operability**: Creation and editing are unconstrained by reference images, offering high degrees of freedom.

### Two Critical Bottlenecks
The authors identify two fundamental causes of insufficient accuracy in sparse signals:

**Limitations of the OpenPose representation itself**: OpenPose was designed for visualization; its RGB color encoding conveys limited semantic information and can even confuse the perception of different keypoints.

**Spatial alignment difficulty due to sparsity**: The point-like and line-like nature of sparse signals makes it difficult for models to perceive and follow spatial instructions.

## Method

### Overall Architecture
SP-Ctrl is built upon Stable Diffusion v1.5 and the ControlNet architecture, comprising three core components:
1. A Spatial-Pose Embedding module (SPR) — extends OpenPose's fixed RGB encoding into learnable embeddings.
2. Keypoint Concept Learning (KCL) — enhances keypoint spatial alignment using text embeddings and heatmap constraints via cross-attention maps.
3. A frozen SD backbone with a ControlNet adapter as the base controllable diffusion architecture.

### Key Design 1: Spatial-Pose Representation (SPR)
The fixed RGB keypoint encoding is extended into **learnable embeddings**:

$$\boldsymbol{E}_{kpt} = \mathcal{G}(\boldsymbol{E}_0; \phi)$$

where $\boldsymbol{E}_0 = \{\boldsymbol{e}_k \in \mathbb{R}^{1 \times C}\}_{k=1}^N$ are randomly initialized fixed vectors and $\mathcal{G}(\cdot; \phi)$ is a parameterized MLP embedding module. The skeleton embedding is represented by an all-ones vector $\boldsymbol{e}_{sks} = \mathbf{1}^{1 \times C'}$.

Key findings:
- **Random initialization outperforms text-embedding initialization** (by 0.88% mAP) — the text embedding space and the spatial pose representation space differ significantly.
- **Fixed input + learnable mapping** outperforms a fully learnable design — optimization is more stable.
- **One-dimensional embeddings already achieve competitive performance** — SPR possesses sufficient expressive capacity.

The learned embeddings are ultimately rendered as a multi-channel ($C'$-channel) skeleton pose image $\boldsymbol{I}_{sp} \in \mathbb{R}^{H \times W \times C'}$, which serves as the spatial conditioning input.

### Key Design 2: Keypoint Concept Learning (KCL)
Motivated by the observed correlation between cross-attention maps and the spatial response of noun tokens, KCL introduces new text tokens to learn keypoint concepts:

1. **Introduce keypoint tokens**: A new learnable text embedding $\mathcal{V}_{kpt} = \{\boldsymbol{v}_i^* \in \mathbb{R}^{768}\}_{i=1}^N$ is added for each keypoint and appended to the text prompt.
2. **Heatmap constraint**: The cross-attention maps $\mathcal{M}_{kpt}$ corresponding to the keypoint tokens are extracted and constrained to align with the keypoint position heatmaps $\mathcal{H}$:

$$\mathcal{L}_{ht} = \frac{1}{|\mathcal{M}_{kpt}|} \cdot \frac{1}{H'W'} \sum_{v_i \geq 1} \|(\mathcal{M}_i - \mathcal{H}_i)\|^2$$

3. **Critical design**: Gradients from the noisy image query $Q$ are detached to prevent appearance collapse.

### Training Objective
The SPR embedding module, keypoint text embeddings, and ControlNet adapter are jointly optimized:

$$\phi^*, \boldsymbol{V}_{kpt}^*, \Theta^* = \arg\min_{\phi, v_i^*, \Theta} \mathcal{L}_{ldm} + \eta \cdot \mathcal{L}_{ht}$$

where $\eta = 0.1$. The heatmap constraint is applied at the 3rd Transformer block and over timesteps 250–500, which are found to be most critical.

## Experiments

### Experimental Setup
- **Model**: SD v1.5 + ControlNet (adapter trained from scratch)
- **Datasets**: AP-10K (animals, 54 species, 17 keypoints), Human-Art (humans, 50K images)
- **Metrics**: Pose mAP (OKS-mAP via ViTPose++), CLIP-Score, FID, Detection AP.75

### Main Results: AP-10K Animal Pose-guided Generation

| Method | Pose mAP↑ | FID↓ | CLIP-Score↑ | Det AP.75↑ |
|--------|-----------|------|-------------|-----------|
| T2I-Adapter | 48.16 | 27.29 | 25.52 | 24.23 |
| ControlNet | 44.25 | 19.40 | 24.77 | 24.35 |
| **SP-Ctrl (Ours)** | **55.63** | **18.52** | 23.86 | **25.10** |

SP-Ctrl surpasses ControlNet in pose alignment by **11.38 mAP** while also achieving lower FID.

### Main Results: Human-Art Human Pose-guided Generation

| Method | Pose mAP↑ | FID↓ | CLIP-Score↑ | Det AP.75↑ |
|--------|-----------|------|-------------|-----------|
| ControlNet | 45.26 | 26.69 | 27.84 | 8.18 |
| HumanSD† | 49.92 | 35.18 | 27.35 | 8.29 |
| GRPose† | 50.93 | 28.85 | 27.95 | 6.51 |
| **SP-Ctrl (Ours)** | **51.11** | 29.30 | 25.94 | **9.11** |

SP-Ctrl achieves state-of-the-art mAP (on par with GRPose) **without requiring any additional pre-trained pose estimator**.

### Ablation Study: Contribution of Each Module

| Method | Pose mAP↑ | FID↓ |
|--------|-----------|------|
| ControlNet (baseline) | 44.25 | 19.40 |
| + Spatial Pose | 52.85 | 19.67 |
| + KCL | 51.34 | 18.94 |
| **SP-Ctrl (combined)** | **55.63** | **18.52** |

The two modules individually contribute +8.60 and +7.09 mAP, respectively; their combination yields the optimal result of 55.63 with the lowest FID.

### Ablation Study: Critical Timesteps for KCL

| Timestep Range | Pose mAP↑ | CLIP-Score↑ |
|---------------|-----------|-------------|
| 0–250 | ~48 | ~24.2 |
| **250–500** | **~53** | **~24.8** |
| 500–750 | ~49 | ~24.0 |
| 750–999 | ~44 | ~24.0 |

Timesteps 250–500 are most critical for keypoint concept formation, while 750–999 contribute almost nothing — revealing the temporal dynamics of concept formation in the denoising process.

### Comparison with Dense Signals
SP-Ctrl, using sparse signals, achieves pose control accuracy close to that of dense signal methods (e.g., depth maps), while demonstrating clear advantages in **image diversity** and **cross-species generalization**.

## Highlights & Insights
- **Reassessing the value of sparse signals**: Demonstrates that sparse signals, when properly designed, can match dense signals in control precision while retaining flexibility and generalizability.
- **Replacing fixed RGB with learnable embeddings**: A simple yet effective modification that reveals the fixed RGB encoding of OpenPose to be even less informative than random initialization.
- **Elegant design of KCL**: Injects keypoint semantics into the text space via textual inversion, then uses heatmap constraints to enforce spatial alignment.
- **Cross-species generation capability**: The category-agnostic nature of sparse signals enables a single model to transfer across different animal species.

## Limitations & Future Work
- Experiments are conducted on SD v1.5; generalizability to newer architectures (e.g., SDXL, FLUX) is not validated.
- The newly introduced keypoint tokens cause a slight drop in CLIP-Score, as the evaluation model does not recognize these tokens at inference time.
- Validation is limited to the 17-keypoint setting; generalization to more or fewer keypoints is unexplored.
- The skeleton embedding uses a simple all-ones vector, which may limit the utilization of skeletal topology information.

## Related Work & Insights
- **Spatially controllable diffusion models**: Adapter-injection methods such as ControlNet, T2I-Adapter, and GLIGEN.
- **Pose-guided generation**: HumanSD (perceptual loss) and GRPose (graph learning) for enhancing sparse signals.
- **Dense conditioning methods**: DensePose, SMPL, and depth maps for precise spatial constraints.

## Rating
- Novelty: ⭐⭐⭐⭐ — Offers original insights into rethinking sparse signals; KCL design is elegant.
- Technical Depth: ⭐⭐⭐⭐ — SPR and KCL designs are well-grounded and supported by thorough ablation studies.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers two datasets (animal and human) with comprehensive ablation analysis.
- Practical Value: ⭐⭐⭐⭐ — Negligible additional inference cost; compatible with ControlNet; strong cross-species generalization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Dual Recursive Feedback on Generation and Appearance Latents for Pose-Robust Text-to-Image Diffusion](dual_recursive_feedback_on_generation_and_appearance_latents_for_pose-robust_tex.md)
- [\[ICCV 2025\] TeRA: Rethinking Text-guided Realistic 3D Avatar Generation](tera_rethinking_text-guided_realistic_3d_avatar_generation.md)
- [\[NeurIPS 2025\] SceneDesigner: Controllable Multi-Object Image Generation with 9-DoF Pose Manipulation](../../NeurIPS2025/image_generation/scenedesigner_controllable_multi-object_image_generation_with_9-dof_pose_manipul.md)
- [\[ICCV 2025\] LUSD: Localized Update Score Distillation for Text-Guided Image Editing](lusd_localized_update_score_distillation_for_text-guided_image_editing.md)
- [\[ICCV 2025\] Text Embedding Knows How to Quantize Text-Guided Diffusion Models](text_embedding_knows_how_to_quantize_text-guided_diffusion_models.md)

</div>

<!-- RELATED:END -->
