---
title: >-
  [Paper Note] EPSegFZ: Efficient Point Cloud Semantic Segmentation for Few- and Zero-Shot Scenarios
description: >-
  [AAAI 2026][3D Vision][Point Cloud Semantic Segmentation] This paper proposes EPSegFZ, a pretraining-free framework for few- and zero-shot 3D point cloud semantic segmentation. It extracts high-frequency features via Pro…
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Point Cloud Semantic Segmentation"
  - "Few-Shot Learning"
  - "Zero-Shot Learning"
  - "Language Guidance"
  - "Attention Mechanism"
date: 2026-05-08
content_hash: daabdef2dfa622f2
---

# EPSegFZ: Efficient Point Cloud Semantic Segmentation for Few- and Zero-Shot Scenarios

**Conference**: AAAI 2026
**arXiv**: [2511.11700](https://arxiv.org/abs/2511.11700)
**Code**: None
**Area**: 3D Vision
**Keywords**: Point Cloud Semantic Segmentation, Few-Shot Learning, Zero-Shot Learning, Language Guidance, Attention Mechanism

## TL;DR

This paper proposes EPSegFZ, a pretraining-free framework for few- and zero-shot 3D point cloud semantic segmentation. It extracts high-frequency features via ProERA, updates prototypes with textual information via LGPE, and establishes accurate query-prototype correspondences via DRPE. EPSegFZ surpasses the state of the art by 5.68% on S3DIS and 3.82% on ScanNet.

## Background & Motivation

Few-shot semantic segmentation (FS-SemSeg) on 3D point clouds faces three core challenges:

**Over-reliance on pretraining**: Existing methods (AttMPTI, COSeg, etc.) heavily depend on fully supervised pretrained backbones, introducing domain-shift bias. Moreover, 3D datasets are small, making pretraining prone to overfitting and computationally expensive.

**Loss of high-frequency information**: Pretraining-free methods such as Seg-PN discard high-frequency information to maintain robustness, yet high-frequency features contain critical edge details that are essential for precise segmentation.

**Insufficient exploitation of support-set information**: Existing methods rely solely on point cloud labels, ignoring complementary information such as textual annotations, which limits both performance and zero-shot capability.

The authors' core idea is to simultaneously leverage high-frequency visual features and low-frequency textual features to achieve efficient few-/zero-shot segmentation without pretraining.

## Method

### Overall Architecture

EPSegFZ consists of three core modules forming an end-to-end pretraining-free framework:

1. **ProERA (Prototype-Enhanced Registers Attention)**: Enhances feature extraction by capturing high-frequency information.
2. **LGPE (Language-Guided Prototype Embedding)**: Updates prototypes with textual information to enable zero-shot inference.
3. **DRPE (Dual Relative Positional Encoding)-based cross-attention**: Establishes accurate query-prototype correspondences.

Pipeline: DGCNN (trained from scratch) extracts point cloud features → MPS samples multiple prototypes → ProERA refines features (with appended register and prototype tokens) → LGPE updates prototypes (incorporating text embeddings) → DRPE cross-attention establishes correspondences → dot-product yields predictions.

### Key Designs

#### ProERA Module

The core innovation lies in progressively focusing on high-frequency information by subtracting low-frequency components:

- Learnable register tokens $\mathbf{r}_t \in \mathbb{R}^{n_r \times D}$ and prototype tokens are appended to the input token sequence.
- Since self-attention is inherently a low-pass filter, the mean of the input features (i.e., the low-frequency component) is subtracted from the attention output, yielding high-frequency-dominant features:

$$\tilde{\mathbf{X}}_j^i = \text{Res}(\text{SA}([\hat{\mathbf{X}}_j^{i-1}; \mathbf{r}_j; \hat{\mathbf{p}}^{i-1}])) - \frac{1}{n_j}\sum_{n_j}\hat{\mathbf{X}}_j^{i-1}$$

- Register tokens learn to attend to different regions: one focuses on background/object-free areas, while another focuses on regions containing multiple targets, implicitly alleviating foreground-background imbalance.

#### LGPE Module

This module addresses prototype quality issues, particularly the lack of discriminability caused by random backbone initialization in early training:

- A pretrained CLIP text encoder extracts class text embeddings $\mathbf{T}^c$, which are mapped to a shared space via a projection network.
- Prototype updates fuse four sources: previous-layer prototype token $\tilde{\mathbf{p}}^i$, raw prototype $\mathbf{p}_{raw}$, dynamic prototype $\mathbf{p}_{dyn}^i$, and text prototype $\mathbf{p}_{text}$:

$$\mathbf{p}^i = \lambda_1 \tilde{\mathbf{p}}^i + \lambda_2 \mathbf{p}_{raw} + \lambda_3 \mathbf{p}_{dyn}^i + \lambda_4 \mathbf{p}_{text}$$

- **Dynamic weight scheduling**: The text weight $\lambda_4(t) = \lambda_4^* e^{-0.5t}$ decays exponentially, while visual weights $\lambda_i(t) = \lambda_i^*(1 - e^{-0.5t})$ grow gradually, enabling a smooth transition from text-driven to visually-balanced prototype construction.
- Zero-shot capability: prototypes can be built from text embeddings alone, without any support-set point clouds.

#### DRPE Module

This is the first work to introduce query-prototype correlations in the latent space as positional encoding signals for cross-attention:

- The **Euclidean distance** $d_E^{i,j,c}$ between each query point and each prototype is computed and encoded via sinusoidal positional encoding functions to obtain $\mathbf{R}_E^i$.
- The **cosine similarity** $d_C^{i,j,c}$ between query and prototype vectors is similarly encoded to obtain $\mathbf{R}_C^i$.
- The dual encodings are summed: $\mathbf{R}^i = \mathbf{R}_C^i + \mathbf{R}_E^i$.
- Advantage: no additional trainable parameters are introduced; query-prototype correlations are efficiently captured as prior knowledge.

### Loss & Training

Three loss functions are jointly optimized:

1. **Segmentation loss** $\mathcal{L}_{seg} = \text{CE}(\mathbf{Y}_q, \hat{\mathbf{Y}}_q)$: the primary supervision signal.
2. **Foreground consistency loss** $\mathcal{L}_{con} = \text{InfoNCE}(\mathbf{x}_q, \mathbf{x}_s)$: encourages same-class foreground features to cluster in the embedding space, compensating for the limited representational capacity of a pretraining-free backbone.
3. **Foreground-aware alignment loss** $\mathcal{L}_{align}$: minimizes the cross-entropy between text-visual similarity scores and text labels, strengthening the joint text-visual space.

$$\mathcal{L} = \mathcal{L}_{seg} + \lambda_{con}\mathcal{L}_{con} + \lambda_{align}\mathcal{L}_{align}$$

Training strategy: episodic learning, 30,000 iterations; the backbone uses a relatively high learning rate with rapid decay.

## Key Experimental Results

### Main Results

**S3DIS dataset (2-way 1-shot)**:

| Method | S⁰ | S¹ | Mean | Δ |
|--------|-----|-----|------|---|
| AttMPTI | 53.77 | 55.94 | 54.86 | −18.56 |
| PAPFZS3D | 59.45 | 66.08 | 62.76 | −10.66 |
| Seg-PN | 64.84 | 67.98 | 66.41 | −7.01 |
| SDSimPoint | 68.73 | 70.61 | 69.67 | −3.75 |
| **EPSegFZ** | **73.08** | **73.75** | **73.42** | **—** |

**ScanNet dataset (2-way 1-shot)**:

| Method | Mean | Δ |
|--------|------|---|
| Seg-PN | 63.74 | −5.10 |
| SDSimPoint | 65.19 | −3.65 |
| **EPSegFZ** | **68.84** | **—** |

Efficiency analysis: EPSegFZ has only 2.02M parameters, 2.11 GFLOPs, and 0.36s inference time — comparable to the lightweight Seg-PN (241K / 1.95 GFLOPs / 0.32s) and far superior to COSeg (7.69M / 9.71 GFLOPs / 1.35s).

### Ablation Study

| Configuration | mIoU | Δ |
|--------------|------|---|
| No modules | 31.55 | −41.53 |
| ProERA only | 64.84 | −8.24 |
| ProERA + LGPE | 70.48 | −2.60 |
| ProERA + DRPE | 70.17 | −2.91 |
| Full model | **73.08** | — |

Among prototype components, the dynamic prototype $\mathbf{p}_{dyn}$ contributes most. Removing $\mathcal{L}_{con}$ or $\mathcal{L}_{align}$ each causes approximately 4–5% degradation. DRPE outperforms both learnable and standard sinusoidal positional encodings.

### Key Findings

- Zero-shot evaluation (S3DIS, CLIP, 2-way 1-shot): EPSegFZ achieves 63.84%, surpassing PAPFZS3D's 61.09%.
- t-SNE visualizations show that EPSegFZ produces more compact intra-class feature distributions with clearer inter-class separation.
- $N+1$ register tokens (where $N$ is the number of classes) yields optimal performance; 3 decoder blocks offer the best performance-efficiency trade-off.

## Highlights & Insights

1. **Elegant high-frequency extraction**: Self-attention is naturally a low-pass filter; subtracting the mean from its output extracts high-frequency components in a simple yet effective manner.
2. **Dynamic weight scheduling**: Adaptive text-visual weighting across training phases elegantly resolves the cold-start problem inherent to pretraining-free settings.
3. **Zero-parameter overhead in DRPE**: Sinusoidal encoding injects query-prototype relationships into attention without introducing any trainable parameters.
4. **Unified few-/zero-shot framework**: LGPE enables prototype construction from text alone, naturally supporting zero-shot inference.

## Limitations & Future Work

- Zero-shot performance still lags behind large-scale pretrained models (e.g., SegPoint), though training resources and data scales are not comparable.
- Text embeddings are derived from a frozen CLIP encoder; finer-grained textual descriptions remain unexplored.
- Validation is limited to indoor scene datasets (S3DIS, ScanNet); experiments on large-scale outdoor scenes are absent.
- The dynamic weight schedule uses fixed exponential decay/growth functions; adaptive learning of these schedules is worth exploring.

## Related Work & Insights

- Seg-PN (a pretraining-free method) inspired the pretraining-free design; its discarding of high-frequency information serves as the key motivation for improvement in this work.
- Research on register tokens in ViT (Darcet et al.) inspired the use of registers in ProERA.
- CLIP's text-visual alignment inspired the LGPE module.
- Implications for future work: this framework could be extended to other 3D tasks (detection, instance segmentation) or explored with stronger language models as a replacement for CLIP.

## Rating

- Novelty: ⭐⭐⭐⭐ — Each of the three modules is innovative; the high-frequency extraction idea in ProERA is particularly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multiple benchmarks and settings; comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure; frequency-analysis visualizations are persuasive.
- Value: ⭐⭐⭐⭐ — Pretraining-free, parameter-efficient, and fast at inference.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CLIPoint3D: Language-Grounded Few-Shot Unsupervised 3D Point Cloud Domain Adaptation](../../CVPR2026/3d_vision/clipoint3d_language-grounded_few-shot_unsupervised_3d_point_cloud_domain_adaptat.md)
- [\[CVPR 2026\] SCOPE: Scene-Contextualized Incremental Few-Shot 3D Segmentation](../../CVPR2026/3d_vision/scope_scenecontextualized_incremental_fewshot_3d_s.md)
- [\[ICCV 2025\] BUFFER-X: Towards Zero-Shot Point Cloud Registration in Diverse Scenes](../../ICCV2025/3d_vision/bufferx_towards_zeroshot_point_cloud_registration_in_diverse.md)
- [\[CVPR 2026\] Lite Any Stereo: Efficient Zero-Shot Stereo Matching](../../CVPR2026/3d_vision/lite_any_stereo_efficient_zero-shot_stereo_matching.md)
- [\[AAAI 2026\] DAPointMamba: Domain Adaptive Point Mamba for Point Cloud Completion](dapointmamba_domain_adaptive_point_mamba_for_point_cloud_completion.md)

</div>

<!-- RELATED:END -->
