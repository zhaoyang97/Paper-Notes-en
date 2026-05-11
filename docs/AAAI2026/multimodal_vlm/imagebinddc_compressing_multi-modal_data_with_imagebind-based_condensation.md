---
title: >-
  [Paper Note] ImageBindDC: Compressing Multi-modal Data with ImageBind-based Condensation
description: >-
  [AAAI 2026][Multimodal VLM][Dataset distillation] This paper proposes ImageBindDC, the first framework for multimodal data compression in the unified feature space of ImageBind. It replaces the conventional MMD with Char…
tags:
  - "AAAI 2026"
  - "Multimodal VLM"
  - "Dataset distillation"
  - "multimodal compression"
  - "ImageBind"
  - "characteristic function distance"
  - "distribution matching"
date: 2026-05-08
content_hash: 590f3747d8ddd856
---

# ImageBindDC: Compressing Multi-modal Data with ImageBind-based Condensation

**Conference**: AAAI 2026  
**arXiv**: [2511.08263](https://arxiv.org/abs/2511.08263)  
**Code**: None  
**Area**: Multimodal VLM  
**Keywords**: Dataset distillation, multimodal compression, ImageBind, characteristic function distance, distribution matching

## TL;DR

This paper proposes ImageBindDC, the first framework for multimodal data compression in the unified feature space of ImageBind. It replaces the conventional MMD with Characteristic Function Distance (CFD) and introduces a three-level distribution alignment loss covering uni-modal, cross-modal, and joint-modal objectives. On NYU-v2, the method achieves performance comparable to full-data training (97.30%) using only 5 synthetic samples per class, surpassing the previous SOTA by an absolute margin of 8.2% while reducing compression time by 4.6×.

## Background & Motivation

### State of the Field
The success of modern AI relies on the interplay between large-scale data and large models, yet the computational, storage, and financial costs of training continue to rise. Dataset condensation/distillation addresses this by synthesizing a small set of representative samples to replace full-scale training. Significant progress has been made in the unimodal (image) setting through strategies such as gradient matching (DC, DSA), trajectory matching (MTT), and distribution matching (DM, NCFM).

### Limitations of Prior Work
Existing data compression methods perform poorly in **multimodal settings**:

**Independent compression loses cross-modal correspondence**: Conventional methods compress each modality (e.g., vision, audio) independently. Although per-modality statistics are preserved, **the semantic correspondence across modalities is destroyed**. For instance, independently condensed synthetic images and synthetic audio may no longer be semantically aligned.

**Insufficiency of existing multimodal methods**:
   - AVDD performs distribution matching in separate modal feature spaces, risking inter-modal misalignment.
   - LoRS reduces cross-modal relationships to scalar similarity, which is overly simplistic.
   - RepBlend prevents modal collapse through representation blending, but this is a heuristic approach that cannot guarantee joint semantics.

**Limitations of distribution matching metrics**: The widely used MMD (Maximum Mean Discrepancy) relies on kernel function selection (typically a heuristic Gaussian kernel), which may fail to capture all statistical differences between distributions.

### Core Idea
Perform data compression within the **unified multimodal embedding space** provided by ImageBind, leveraging **Characteristic Function Distance (CFD)** for precise distribution matching (equivalent to matching infinite-order moments), and design a three-level alignment loss to preserve the structural integrity of multimodal data.

## Method

### Overall Architecture
The pipeline of ImageBindDC is as follows:
1. Map real multimodal data (e.g., image + audio) into the unified embedding space via a frozen ImageBind encoder.
2. Initialize synthetic multimodal data using the herding strategy.
3. Map synthetic data into the same embedding space.
4. Optimize synthetic data via the three-level distribution matching loss.
5. Output the compressed small-scale synthetic dataset.

### Key Designs

#### 1. **CFD Replacing MMD**
- Conventional distribution matching methods use MMD as the distribution divergence metric, whose performance is highly sensitive to kernel selection.
- CFD is based on the characteristic function (Fourier transform) of a probability distribution. By Lévy's uniqueness theorem, two distributions are identical if and only if their characteristic functions are identical.
- CFD operates in the frequency domain and implicitly matches all statistical moments of a distribution (not merely the first and second order).

$$\text{CFD}(x, \tilde{x}) = |\Phi(x;t) - \Phi(\tilde{x};t)|^2$$

where the characteristic function $\Phi(z;t) = \mathbb{E}_{z \sim P}[e^{jt^\top z}]$, and $t$ is a frequency vector sampled from a Gaussian distribution.

**Design Motivation**: CFD requires no user-defined kernel functions, providing a more principled framework for distribution matching.

#### 2. **Three-Level Distribution Alignment Loss**
This constitutes the core innovation of ImageBindDC — ensuring distributional consistency between synthetic and real data at three levels:

**(i) Uni-modal Alignment $\mathcal{L}_{\text{uni}}$**:
- Matches the distribution of synthetic and real data within each modality (audio and visual) separately.
- $\mathcal{L}_{\text{uni}} = \text{CFD}(e_a, \tilde{e}_a) + \text{CFD}(e_v, \tilde{e}_v)$
- Preserves intra-modal statistical properties.

**(ii) Cross-modal Alignment $\mathcal{L}_{\text{cross}}$**:
- Captures inter-modal relationships between real/synthetic pairs via element-wise multiplication.
- $\rho_{\text{cross}} = \frac{\langle e_a \odot e_v, \tilde{e}_a \odot \tilde{e}_v \rangle}{\|e_a \odot e_v\|_2 \|\tilde{e}_a \odot \tilde{e}_v\|_2}$
- $\mathcal{L}_{\text{cross}} = 1 - \rho_{\text{cross}}$
- Ensures paired semantic correspondence.

**(iii) Joint-modal Alignment $\mathcal{L}_{\text{joint}}$**:
- Captures the full multivariate data structure through matrix multiplication of mean embeddings.
- $\rho_{\text{joint}} = E_a \odot \tilde{E}_v^\top \times E_v \odot \tilde{E}_a^\top$
- $\mathcal{L}_{\text{joint}} = 1 - \rho_{\text{joint}}$
- Preserves the global structure of the joint distribution.

#### 3. **ImageBind Unified Embedding Space**
A pretrained ImageBind model maps different modalities into a shared feature space, enabling cross-modal distribution matching without operating in the heterogeneous raw data space. ImageBind remains frozen during compression, avoiding the complexity of bi-level optimization.

### Loss & Training

$$\mathcal{L} = \lambda_{\text{uni}} \mathcal{L}_{\text{uni}} + \lambda_{\text{cross}} \mathcal{L}_{\text{cross}} + \lambda_{\text{joint}} \mathcal{L}_{\text{joint}}$$

Training details:
- Synthetic data initialization: herding strategy
- Optimizer: SGD (momentum 0.5), synthetic data learning rate 0.5
- 30 training epochs with evaluation every 10 steps
- Batch size: 32 (synthetic), 128 (real)
- Data augmentation: differentiable Siamese augmentation strategy

## Key Experimental Results

### Main Results

**VGGS-10K dataset (ConvNet backbone, classification accuracy)**:

| Method | 1 DPC | 10 DPC | 20 DPC |
|------|-------|--------|--------|
| Random | 15.44% | 32.01% | 45.10% |
| DM | 36.54% | 43.85% | 49.01% |
| AVDD | 40.41% | 48.08% | 48.86% |
| **ImageBindDC** | **42.66%** | **55.23%** | **55.30%** |
| Full data | 68.24% | — | — |

**NYU-v2 dataset (depth–text classification)**:

| DPC | Random | DM | AVDD | ImageBindDC | Full data |
|-----|--------|-----|------|-------------|--------|
| 1 | 60.60% | 67.97% | 72.22% | **80.43%** | 98.62% |
| 5 | 73.85% | 89.08% | 95.92% | **97.30%** | — |
| 10 | 88.38% | 96.89% | 98.62% | **98.73%** | — |

On NYU-v2, 5 DPC achieves 97.30%, approaching the full-data result of 98.62%, effectively realizing lossless compression.

### Ablation Study

**CFD vs. MMD (AVE dataset, ImageBind backbone)**:

| Configuration | 1 DPC | 10 DPC |
|------|-------|--------|
| MMD (Audio only) | 27.87% | — |
| CFD (Audio only) | 32.33% | — |
| MMD (Video+Audio) | — | 69.26% |
| CFD (Video+Audio) | — | 70.34% |

**Alignment component ablation (AVE, 10 DPC)**:

| Configuration | Accuracy | Note |
|------|--------|------|
| Uni-modal only | 70.34% | Baseline |
| + Joint-modal | Improved | Positive gain |
| + Cross-modal | Improved | Positive gain |
| All three | **73.67%** | +3.33%, synergistic effect |

**Computational efficiency (VGGS-10K)**:

| Method | 1 DPC Time (s) | 20 DPC Time (s) | 1 DPC Memory (GB) |
|------|-------------|-------------|--------------|
| DM | 140.3 | 707.21 | 8.96 |
| AVDD | 158.11 | 700.10 | 8.96 |
| **ImageBindDC** | **57.46** | **123.74** | **5.60** |

ImageBindDC is more than 2.4× faster than DM, with a 37.5% reduction in memory usage.

### Key Findings

1. The three-level loss exhibits a synergistic effect: the components are complementary rather than simply additive. Uni-modal alignment ensures intra-modal integrity, while cross-modal and joint-modal alignment preserve inter-modal relational structure.
2. ImageBindDC generalizes across architectures: synthetic data distilled using ImageBind also achieves the best performance when used to train a ConvNet.
3. UMAP visualizations show that the embedding distribution of ImageBindDC's synthetic data most closely approximates that of the real data.

## Highlights & Insights

1. The idea of **performing multimodal compression in a unified space** is both elegant and effective — it leverages ImageBind's pre-aligned feature space, avoiding the need to handle cross-modal relationships in a heterogeneous raw data space.
2. **CFD as a replacement for MMD** offers a more principled distribution matching framework: no kernel selection is required, and all statistical moments are implicitly matched — a contribution of value to the broader dataset distillation community.
3. The **three-level alignment** design provides complete coverage of multimodal data structure, and is methodologically clean.
4. Achieving lossless performance on NYU-v2 with only 5 samples per class demonstrates the practical utility of the proposed method.

## Limitations & Future Work

- Validation is currently limited to three bimodal combinations (audio–visual, depth–text, audio–text); applicability to three or more modalities remains unexplored.
- The quality of ImageBind embeddings directly determines the upper bound of compression performance — the method may be limited in domains where ImageBind's embedding quality is suboptimal.
- The visual quality of synthetic data still has room for improvement (Figure 5 shows that AVDD's synthetic images are noticeably degraded, and ImageBindDC also does not reach the clarity of real data).
- A comprehensive comparison with more recent strong baselines (e.g., RepBlend, LoRS) across all datasets is absent.

## Related Work & Insights

- NCFM first introduced characteristic functions into unimodal dataset distillation; this work extends that idea to the multimodal setting.
- The unified embedding space provided by ImageBind is foundational to this approach — future improvements in multimodal alignment models could be directly substituted.
- The three-level alignment design is generalizable to other scenarios requiring the preservation of multimodal structure, such as multimodal knowledge distillation and data augmentation.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Multi-modal Data Spectrum: Multi-modal Datasets are Multi-dimensional](../../ICLR2026/multimodal_vlm/multi-modal_data_spectrum_multi-modal_datasets_are_multi-dimensional.md)
- [\[AAAI 2026\] RMAdapter: Reconstruction-based Multi-Modal Adapter for Vision-Language Models (Oral)](rmadapter_reconstructionbased_multimodal_adapter_for_visionlanguage.md)
- [\[AAAI 2026\] InEx: Hallucination Mitigation via Introspection and Cross-Modal Multi-Agent Collaboration](inex_hallucination_mitigation_via_introspection_and_cross-mo.md)
- [\[CVPR 2026\] CRIT: Graph-Based Automatic Data Synthesis to Enhance Cross-Modal Multi-Hop Reasoning](../../CVPR2026/multimodal_vlm/crit_graph-based_automatic_data_synthesis_to_enhance_cross-modal_multi-hop_reaso.md)
- [\[AAAI 2026\] Revisiting the Data Sampling in Multimodal Post-training from a Difficulty-Distinguish View](revisiting_the_data_sampling_in_multimodal_post-training_from_a_difficulty-disti.md)

</div>

<!-- RELATED:END -->
