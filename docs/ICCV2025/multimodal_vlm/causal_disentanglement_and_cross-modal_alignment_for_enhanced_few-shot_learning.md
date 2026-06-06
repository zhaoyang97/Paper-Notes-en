---
title: >-
  [Paper Note] Causal Disentanglement and Cross-Modal Alignment for Enhanced Few-Shot Learning
description: >-
  [ICCV 2025][Multimodal VLM][Few-shot learning] This paper proposes the Causal CLIP Adapter (CCA), which applies ICA to causally disentangle CLIP visual features…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "Few-shot learning"
  - "CLIP"
  - "causal disentanglement"
  - "independent component analysis"
  - "cross-modal alignment"
date: 2026-05-08
content_hash: 10e50406a85340fe
---

# Causal Disentanglement and Cross-Modal Alignment for Enhanced Few-Shot Learning

**Conference**: ICCV 2025  
**arXiv**: [2508.03102](https://arxiv.org/abs/2508.03102)  
**Code**: [github.com/tianjiao-j/CCA](https://github.com/tianjiao-j/CCA)  
**Area**: Multimodal VLM  
**Keywords**: Few-shot learning, CLIP, causal disentanglement, independent component analysis, cross-modal alignment

## TL;DR

This paper proposes the Causal CLIP Adapter (CCA), which applies ICA to causally disentangle CLIP visual features, and enhances cross-modal alignment via unidirectional text classifier fine-tuning and bidirectional cross-attention, achieving state-of-the-art few-shot classification performance across 11 benchmark datasets.

## Background & Motivation

### Problem Definition

Few-shot learning (FSL) requires models to rapidly adapt to new tasks with limited labeled data. Most current approaches perform transfer learning on pre-trained vision-language models such as CLIP, but directly using CLIP-extracted features presents a fundamental challenge: **the features are entangled**.

### Limitations of Prior Work

**Entangled features lead to overfitting**: The features $\mathbf{z}'$ output by the CLIP visual encoder are linear mixtures of true latent variables $\mathbf{z}$. Existing methods (e.g., Tip-Adapter, TaskRes) directly use these entangled features, requiring the classifier to implicitly learn a dense weight matrix $\mathbf{W}_{\text{dense}} = \mathbf{W}_{\text{unmix}} \cdot \mathbf{W}'_{\text{sparse}}$, which has a large parameter count and is prone to overfitting in few-shot settings.

**Cache model alone is insufficient**: Methods such as Tip-Adapter construct similarity matching solely within the image modality, failing to fully exploit CLIP's cross-modal alignment capability.

**Prompt-tuning methods incur high training cost**: Approaches like CoOp require backpropagation through the CLIP text encoder, resulting in significantly longer training times compared to adapter-based methods.

### Core Motivation

Theoretical work (Daunhawer et al.) has demonstrated that multimodal contrastive learning models such as CLIP can recover true latent variables up to a linear transformation, i.e., $\mathbf{z}' = \mathbf{A}\mathbf{z} + \mathbf{c}$. Since such a linear transformation is known to exist, ICA can be directly applied to remove it and obtain disentangled features, substantially reducing the number of parameters that need to be learned for downstream tasks.

**Core Idea**: First apply ICA to disentangle CLIP features and reduce parameter count; then restore the intra-modal alignment lost by disentanglement through cross-modal alignment.

## Method

### Overall Architecture

CCA consists of three core components: (1) a FastICA-based disentangled cache model for intra-modal image matching; (2) a fine-tuned CLIP text classifier for unidirectional cross-modal alignment; and (3) a cross-attention fusion module for bidirectional cross-modal information interaction. The logits from all three components are linearly combined to produce the final prediction.

### Key Designs

#### 1. **FastICA Disentanglement**

- **Function**: Extracts an unmixing matrix $\mathbf{U} \in \mathbb{R}^{C \times M}$ from an ImageNet subset and applies it to disentangle all CLIP visual features.
- **Mechanism**: CLIP features lie on a hypersphere with $\mathbf{z}' = \mathbf{A}\mathbf{z} + \mathbf{c}$; ICA recovers independent latent variables by maximizing non-Gaussianity. The disentangled cache keys are:

$$\mathbf{F}_{\text{cache}}^d = \mathbf{F}_{\text{cache}} \mathbf{U} \mathbf{W}_c, \quad \mathbf{F}_{\text{query}}^d = \mathbf{F}_{\text{query}} \mathbf{U}$$

where $\mathbf{W}_c$ is a trainable cache adapter (initialized as an identity matrix, with $\ell_1$ regularization to induce sparsity).

- **Design Motivation**: After disentanglement, the classifier only needs to learn a sparse weight matrix, greatly reducing parameter count and mitigating few-shot overfitting. Since $\mathbf{U}$ is extracted from ImageNet once and shared across all datasets, the computational overhead is minimal.

#### 2. **Unidirectional Cross-Modal Alignment (Text Classifier Fine-tuning)**

- **Function**: Fine-tunes the CLIP text classifier $\mathbf{W}_t \in \mathbb{R}^{N \times C}$ (obtained by encoding class names through the text encoder) to better align text features with image features.
- **Mechanism**: Text weights are fine-tuned at a small learning rate (0.0001), preserving CLIP's prior knowledge while improving downstream alignment.
- **Design Motivation**: ICA disentanglement may disrupt CLIP's original intra-modal alignment; fine-tuning the text classifier compensates for this loss from a cross-modal perspective.

#### 3. **Bidirectional Cross-Attention Fusion**

- **Function**: Generates two types of hybrid features via cross-attention — image-enhanced text classifiers and text-enhanced image features.
- **Core Formulas**:

$$\mathbf{W}_t^* = (\text{softmax}(\mathbf{W}_t \mathbf{F}_{\text{query}}^\top) \mathbf{F}_{\text{query}})^\top$$

$$\mathbf{F}_{\text{query}}^* = \text{softmax}(\mathbf{F}_{\text{query}} \mathbf{W}_t^\top) \mathbf{W}_t$$

Final cross-modal logits: $\mathbf{l}_2 = \mathbf{F}_{\text{query}} \mathbf{W}_t^\top + \gamma \mathbf{F}_{\text{query}} \mathbf{W}_t^* + \eta \mathbf{F}_{\text{query}}^* \mathbf{W}_t^\top$

- **Design Motivation**: Unidirectional fine-tuning can only align from text to image; bidirectional cross-attention allows the two modalities to mutually complement each other, capturing richer semantic relationships.

### Loss & Training

- **Loss Function**: Standard cross-entropy loss + $\ell_1$ regularization on the cache adapter.
- **Final Logits**: $\mathbf{l} = \alpha \mathbf{l}_1 + \mathbf{l}_2$
- **Training Details**: SGD optimization; cache adapter learning rate 0.001; text classifier learning rate 0.0001; training for 20 epochs on most datasets (100 epochs for EuroSAT).
- **Hyperparameters**: $\alpha, \gamma, \eta, \beta$ are optimized via grid search on a validation set.

## Key Experimental Results

### Main Results

Average classification accuracy across 11 datasets (16-shot):

| Method | Type | ImageNet | Avg. (11 datasets) |
|--------|------|----------|--------------------|
| Zero-shot CLIP | Zero-shot | 60.33 | - |
| CoOp | Prompt-tuning | 62.95 | - |
| Tip-Adapter | Training-free | 61.80 | - |
| Tip-Adapter-F | Fine-tuned | 65.45 | - |
| CCA | Training-free | 63.00 | - |
| **CCA-FT** | **Fine-tuned** | **66.04** | **Best** |

CCA-FT outperforms Tip-Adapter-F across all 11 datasets under all shot settings, and achieves the best average accuracy among all compared methods.

### Ablation Study

| Configuration | 1-shot | 4-shot | 16-shot | Notes |
|---------------|--------|--------|---------|-------|
| Full model | 66.00 | 72.10 | 77.60 | All components |
| w/o ICA | 64.94 | 70.34 | 75.95 | Disentanglement contributes most |
| Fixed cache adapter | 65.16 | 70.78 | 75.75 | Intra-modal alignment matters |
| Fixed text classifier | 65.10 | 71.32 | 77.00 | Text fine-tuning helps but less so |
| w/o fusion features | 65.81 | 72.02 | 77.43 | Cross-attention provides marginal gain |

### Key Findings

- **ICA disentanglement is the most critical component**: Its removal causes the largest performance drops across all shot settings (−1.65% at 16-shot).
- **Strong out-of-distribution robustness**: CCA-FT outperforms all baselines on both ImageNet-V2 and ImageNet-Sketch.
- **Robustness to Gaussian noise and adversarial attacks**: CCA/CCA-FT substantially outperforms Tip-Adapter/-F under both settings (16-shot accuracy 35.48% vs. 33.21% under adversarial attack).
- **High computational efficiency**: 20 epochs, 4.9 min training (vs. CoOp's 200 epochs, 7.5 h), with superior accuracy.
- **Consistent effectiveness across backbones**: Outperforms Tip-Adapter-F on ResNet-50/101, ViT-B/16, and ViT-B/32.

## Highlights & Insights

1. **Theory-driven practical method**: Grounded in causal disentanglement theory, ICA's linear disentanglement capability is leveraged to address overfitting in FSL, achieving a tight coupling of theory and practice.
2. **Complementary uni- and bidirectional alignment design**: Recognizing that disentanglement may disrupt intra-modal alignment, two complementary cross-modal alignment mechanisms are introduced to compensate, yielding a logically coherent overall design.
3. **Unmixing matrix shared across datasets**: Extracted once from ImageNet and reused for all downstream datasets, offering strong practical utility.
4. **Sparsity inductive bias**: $\ell_1$ regularization induces sparsity in the cache adapter, consistent with the theoretical expectations of disentangled representations.

## Limitations & Future Work

1. **ICA assumption constraints**: The method assumes at most one Gaussian latent variable, which may not hold under certain data distributions.
2. **Hypersphere constraint**: Since CLIP features lie on a hypersphere, ICA operates in an effective dimensionality of $M-1$, potentially losing information.
3. **Simple text classifier design**: Only linear fine-tuning is explored; more flexible text adaptation strategies are not investigated.
4. **Larger-scale CLIP not explored**: Validation is limited to ResNet-50/101 and ViT-B; larger backbones such as ViT-L are not tested.
5. **Train-test inconsistency in cross-attention**: During training, both directions use query features; during inference, the reverse path switches to cache features.

## Related Work & Insights

- The causal disentanglement theory for CLIP (Daunhawer et al.) provides the theoretical foundation for this work; this theoretical perspective merits exploration in other CLIP applications.
- The lightweight nature of FastICA makes it particularly suitable for few-shot settings; similar unsupervised disentanglement methods may be applicable to other transfer learning tasks.
- The complementary design of uni- and bidirectional alignment can be generalized to other multimodal fusion scenarios.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Introducing causal disentanglement theory into CLIP-based few-shot learning is a novel angle; the combination of ICA and cross-modal alignment is well-motivated.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive evaluation across 11 datasets, multiple shot settings, OOD testing, noise/adversarial robustness, and ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Theoretical motivation is clearly articulated, method descriptions are detailed, and figures are intuitive.
- **Value**: ⭐⭐⭐⭐ — Provides a lightweight and efficient CLIP adapter method whose theoretical insights offer meaningful contributions to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Sparsity Outperforms Low-Rank Projections in Few-Shot Adaptation](sparsity_outperforms_low-rank_projections_in_few-shot_adaptation.md)
- [\[CVPR 2026\] Noise-Aware Few-Shot Learning through Bi-directional Multi-View Prompt Alignment](../../CVPR2026/multimodal_vlm/noise-aware_few-shot_learning_through_bi-directional_multi-view_prompt_alignment.md)
- [\[NeurIPS 2025\] VT-FSL: Bridging Vision and Text with LLMs for Few-Shot Learning](../../NeurIPS2025/multimodal_vlm/vt-fsl_bridging_vision_and_text_with_llms_for_few-shot_learning.md)
- [\[CVPR 2026\] Mind the Discriminability Trap in Source-Free Cross-domain Few-shot Learning](../../CVPR2026/multimodal_vlm/mind_the_discriminability_trap_in_source-free_cross-domain_few-shot_learning.md)
- [\[NeurIPS 2025\] Causal-LLaVA: Causal Disentanglement for Mitigating Hallucination in Multimodal Large Language Models](../../NeurIPS2025/multimodal_vlm/causalllava_causal_disentanglement_for_mitigating_hallucinat.md)

</div>

<!-- RELATED:END -->
