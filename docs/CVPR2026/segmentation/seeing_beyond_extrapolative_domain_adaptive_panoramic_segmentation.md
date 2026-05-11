---
title: >-
  [Paper Note] Seeing Beyond: Extrapolative Domain Adaptive Panoramic Segmentation
description: >-
  [CVPR 2026][Segmentation][Panoramic semantic segmentation] This paper proposes EDA-PSeg, a framework that introduces two core modules — a Graph Matching Adapter (GMA) and an Euler-Margin Attention (EMA) — to achieve…
tags:
  - "CVPR 2026"
  - "Segmentation"
  - "Panoramic semantic segmentation"
  - "open-set domain adaptation"
  - "field-of-view transfer"
  - "graph matching"
  - "Euler attention"
date: 2026-05-08
content_hash: 94f09541deef0ea0
---

# Seeing Beyond: Extrapolative Domain Adaptive Panoramic Segmentation

**Conference**: CVPR 2026
**arXiv**: [2603.15475](https://arxiv.org/abs/2603.15475)
**Code**: [Available](https://github.com/zyfone/EDA-PSeg)
**Area**: Semantic Segmentation
**Keywords**: Panoramic semantic segmentation, open-set domain adaptation, field-of-view transfer, graph matching, Euler attention

## TL;DR

This paper proposes EDA-PSeg, a framework that introduces two core modules — a Graph Matching Adapter (GMA) and an Euler-Margin Attention (EMA) — to achieve, for the first time, open-set unsupervised domain adaptive semantic segmentation from pinhole to 360° panoramic images, simultaneously addressing geometric FoV distortion and unknown category discovery.

## Background & Motivation

**Importance of panoramic vision**: Panoramic images provide a 360° field of view (FoV), enabling complete scene perception without occlusion, with broad applications in autonomous driving and robotics.

**Challenges in cross-domain panoramic segmentation**: Existing methods train on labeled pinhole images (source domain) and transfer to unlabeled panoramic images (target domain), facing severe geometric FoV distortion and inconsistent semantic distributions.

**Limitations of the closed-set assumption**: Most existing cross-domain panoramic segmentation (CPS) methods assume that only categories seen during training appear at test time. This closed-set assumption fails in open-world scenarios when encountering unknown objects, posing safety risks.

**Inadequacy of pixel-level prototype methods**: Existing open-set domain adaptation methods (e.g., BUS, UniMAP) rely on pixel-level prototype mapping, but stylistic inconsistency and geometric distortion in panoramic images limit their effectiveness.

**Interference from adverse weather conditions**: Transfer from single or multiple weather conditions to diverse adverse weather further degrades cross-domain alignment.

**First open-set panoramic segmentation work**: This paper is the first to define the open-set cross-domain panoramic semantic segmentation task, requiring the model to generalize to unseen categories while adapting to different FoV scenes and weather conditions.

## Method

### Overall Architecture

EDA-PSeg is built upon the DAFormer architecture, using a MiT-B5 encoder-decoder network as the backbone. Source-domain (pinhole) and target-domain (panoramic) inputs are randomly cropped and fed into the network for feature extraction, then processed sequentially through:

- **Euler-Margin Attention (EMA)**: Projects features into angle-aware embeddings in complex vector space, mitigates cross-view geometric distortion via angular margin constraints, and enhances known/unknown category separability through amplitude and phase modulation.
- **Graph Matching Adapter (GMA)**: Constructs high-order semantic graph relations, aligns graph nodes of shared categories across domains, and separates unknown categories via regularization.

### Key Designs

**1. Graph Matching Adapter (GMA)**

- **Node sampling**: Local node sampling is performed based on confidence, entropy, and prototype distance to select nodes representing local semantics, which are then aggregated into class-level global prototypes. For known categories, positive/negative sample sets are filtered by confidence threshold $\tau_p$ and entropy threshold $\tau_e$; for unknown categories, positive/negative samples are split by median entropy $\tau_m$. The nearest $K$ nodes are retained and a global memory bank $\mathcal{M}$ is updated via EMA.
- **Graph construction**: Shared categories between source and target domains are identified; missing category nodes are completed using memory bank $\mathcal{M}$ (with added Gaussian noise); node sets are updated via multi-head self-attention to generate node features and edge affinity matrices that form the semantic graph.
- **Graph matching and regularization**: The Sinkhorn algorithm computes node matching matrices, and open-set matching labels that ignore unknown classes are constructed. The loss function comprises three terms: graph matching loss (node alignment), graph edge affinity loss (structural consistency), and unknown category regularization loss (Frobenius norm penalty on known/unknown node pairs).

**2. Euler-Margin Attention (EMA)**

- **Euler-margin projection**: Input features undergo channel descending reordering (via a soft permutation matrix to ensure gradient flow); even and odd channels of the reordered features serve as real and imaginary parts, respectively, projected into complex space via the Euler formula $\mathcal{F}(\mathbf{V}) = \Lambda \cdot e^{i\theta}$. Channel reordering constrains the range of phase angle $\theta$, enhancing intra-class cohesion to mitigate cross-view discrepancy.
- **Amplitude and phase modulation**: Learnable parameters are introduced into the self-attention dot product: $\delta_1$ (exponential amplitude scaling) regulates feature importance; $\delta_2$ (phase scaling factor) and $b$ (phase bias) regulate semantic direction. The final attention score is $\mathcal{E}_{\text{Euler}} = (e^{2\delta_1}(\Lambda_q \odot \Lambda_k))^\top \text{Re}[\exp(i[\delta_2(\theta_q - \theta_k) + b])]$.

### Loss & Training

Total training objective: $\mathcal{L}_{\text{total}} = \ell_{\text{seg}} + \ell_{\text{mixup}} + \gamma \cdot \ell_{\text{graph}}$

- $\ell_{\text{seg}}$: Supervised segmentation loss on the source domain
- $\ell_{\text{mixup}}$: Pseudo-label loss for source–target mixed training
- $\ell_{\text{graph}}$: Graph matching loss from the GMA module (comprising node matching, edge affinity, and unknown category regularization)
- Weight $\gamma = 0.1$ (balancing common/private class performance)
- MobileSAM is used for pseudo-label mask refinement in the target domain
- Training runs for 40k iterations with $512 \times 512$ random crops; original panoramic resolution is used at test time

## Key Experimental Results

### Main Results

**Open-set domain adaptation results on four benchmarks (mIoU %):**

| Benchmark | Type | Common | Private | H-Score |
|-----------|------|--------|---------|---------|
| C2D (Cityscapes→DensePASS) | Pin2Pan, Real2Real | **56.81** | **18.86** | **28.32** |
| S2D (SynPASS→DensePASS) | Syn2Real, Pan2Pan | **35.07** | **7.48** | **12.33** |
| G2S (GTA→SynPASS) | Pin2Pan + Weather | **44.96** | **10.20** | **16.63** |
| S2A (SynPASS→ACDC) | Syn2Real + Weather | **30.17** | **9.18** | **14.08** |

**Comparison with best baselines (C2D benchmark):**

| Method | Common | Private | H-Score |
|--------|--------|---------|---------|
| HRDA | 53.42 | 0.00 | 0.00 |
| BUS (SAM) | 49.47 | 3.10 | 5.84 |
| **EDA-PSeg (Ours)** | **56.81** | **18.86** | **28.32** |

Closed-set methods (DAFormer/HRDA/MIC) all achieve Private mIoU of 0, completely failing to recognize unknown categories.

### Ablation Study

**Module ablation (C2D):**

| GMA | EMA | Common | Private | H-Score |
|-----|-----|--------|---------|---------|
| ✗ | ✗ | 52.56 | 8.57 | 14.74 |
| ✓ | ✗ | 55.15 | 14.67 | 23.18 |
| ✗ | ✓ | 56.12 | 13.00 | 21.11 |
| ✓ | ✓ | **56.81** | **18.86** | **28.32** |

**EMA vs. other attention mechanisms (C2D):**

| Method | Common | Private | H-Score |
|--------|--------|---------|---------|
| Self-Attention | 55.45 | 10.95 | 18.28 |
| EulerFormer | 55.09 | 7.20 | 12.74 |
| Deformable MLP | 55.89 | 7.68 | 13.51 |
| **Euler-Margin (Ours)** | **56.12** | **13.00** | **21.11** |

**GMA loss component ablation**: Removing the graph matching term causes the largest performance drop (H-Score: 23.18→8.73); the unknown category regularization term significantly improves Private mIoU (7.78→14.67).

### Key Findings

1. **Closed-set methods completely fail under the open-set setting**: All closed-set UDA methods achieve Private mIoU of 0 and H-Score of 0, failing to recognize any unknown categories.
2. **GMA and EMA are complementary**: GMA primarily improves Private category recognition (+6.10), while EMA primarily enhances Common category representation (+3.56); their combination raises H-Score from 14.74 to 28.32.
3. **Graph matching is the core of GMA**: Among the three GMA loss terms, the graph matching term contributes the most; removing it causes H-Score to drop sharply from 23.18 to 8.73.
4. **Weight sensitivity**: A large $\gamma$ (1.0) benefits Private but harms Common performance; a small $\gamma$ (0.01) has the opposite effect; $\gamma = 0.1$ yields the optimal balance.

## Highlights & Insights

- **First definition of open-set panoramic segmentation**: FoV geometric transformation and unknown category discovery are jointly modeled, more closely reflecting real-world scenarios than conventional closed-set CPS.
- **Elegant application of Euler's formula**: The amplitude-phase decomposition in complex space is leveraged such that amplitude encodes feature importance and phase encodes semantic direction; channel ordering constrains the angular range to achieve view invariance.
- **Graph matching as an alternative to pixel prototypes**: High-order graph relational modeling is more robust than conventional pixel-level prototype alignment, simultaneously handling node matching and structural consistency.
- **Comprehensive benchmark coverage**: Multiple domain transfer scenarios are covered, including Pin↔Pan, Syn→Real, and adverse weather conditions, with systematic comparison of closed-set and open-set methods.

## Limitations & Future Work

- Random cropping introduces sampling sensitivity, occasionally causing training instability.
- Graph matching increases model parameters and computational overhead; EMA also adds architectural complexity.
- Performance improvement is limited on certain fine-grained categories (e.g., Traffic Light, Traffic Sign), approaching 0 mIoU on some benchmarks.
- Absolute Private mIoU values remain low (7–9%) on the S2D and S2A benchmarks, indicating that open-set discovery capability warrants further improvement.

## Related Work & Insights

- **Cross-domain panoramic segmentation**: CFA (distortion-aware attention), DPPASS (tangential projection), Trans4PASS (deformable patch embedding), OmniSAM/GoodSAM (SAM-assisted alignment)
- **Open-set domain adaptation**: BUS (SAM masks + prototype matching), UniMAP (prototype weight scaling), OSBP/UAN/UniOT (conventional OSDA)
- **Positional encoding**: RoPE (rotary position embedding), EulerFormer (unified semantic-positional representation in Euler space)
- **Graph matching**: Graph relational reasoning in cross-domain named entity recognition, medical image analysis, and object detection

## Rating

- Novelty: ⭐⭐⭐⭐ — First to define the open-set panoramic segmentation problem; EMA and GMA designs are creative
- Experimental Thoroughness: ⭐⭐⭐⭐ — Four benchmarks, multi-scenario coverage, detailed ablations, though absolute performance on some categories remains low
- Writing Quality: ⭐⭐⭐⭐ — Problem definition is clear and the methodology is systematically presented; equations are numerous but logically coherent
- Value: ⭐⭐⭐⭐ — Fills a gap in open-set panoramic segmentation with practically meaningful methods

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Masked Representation Modeling for Domain-Adaptive Segmentation](mrm_masked_representation_modeling_domain_adaptive.md)
- [\[CVPR 2026\] REL-SF4PASS: Panoramic Semantic Segmentation with REL Depth Representation and Spherical Fusion](rel-sf4pass_panoramic_semantic_segmentation_with_rel_depth_representation_and_sp.md)
- [\[CVPR 2026\] Heuristic Self-Paced Learning for Domain Adaptive Semantic Segmentation under Adverse Conditions](heuristic_self-paced_learning_for_domain_adaptive_semantic_segmentation_under_ad.md)
- [\[CVPR 2026\] Seeing Through the Tool: A Controlled Benchmark for Occlusion Robustness in Foundation Segmentation Models](occsam_bench_occlusion_robustness_segmentation.md)
- [\[ICCV 2025\] Exploring Probabilistic Modeling Beyond Domain Generalization for Semantic Segmentation](../../ICCV2025/segmentation/exploring_probabilistic_modeling_beyond_domain_generalization_for_semantic_segme.md)

</div>

<!-- RELATED:END -->
