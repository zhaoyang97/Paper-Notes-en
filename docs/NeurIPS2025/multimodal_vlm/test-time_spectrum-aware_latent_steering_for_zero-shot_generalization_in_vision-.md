---
title: >-
  [Paper Note] Test-Time Spectrum-Aware Latent Steering for Zero-Shot Generalization in Vision-Language Models
description: >-
  [NeurIPS 2025][Multimodal VLM][Test-time adaptation] This paper proposes STS (Spectrum-Aware Test-Time Steering), a lightweight test-time adaptation method that extracts a low-dimensional semantic subspace via SVD decomposition of text embeddings, and learns a small set of coefficients to steer text prototypes within this subspace to handle distribution shift. STS requires no backpropagation through large encoders, runs 8× faster than TPT with 12× less memory…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "Test-time adaptation"
  - "VLM zero-shot generalization"
  - "SVD spectral decomposition"
  - "text prototype steering"
  - "parameter efficiency"
date: 2026-05-08
content_hash: a3da63a6fa28df82
---

# Test-Time Spectrum-Aware Latent Steering for Zero-Shot Generalization in Vision-Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2511.09809](https://arxiv.org/abs/2511.09809)  
**Code**: [GitHub](https://github.com/kdafnis/STS)  
**Area**: Multimodal VLM
**Keywords**: Test-time adaptation, VLM zero-shot generalization, SVD spectral decomposition, text prototype steering, parameter efficiency

## TL;DR

This paper proposes STS (Spectrum-Aware Test-Time Steering), a lightweight test-time adaptation method that extracts a low-dimensional semantic subspace via SVD decomposition of text embeddings, and learns a small set of coefficients to steer text prototypes within this subspace to handle distribution shift. STS requires no backpropagation through large encoders, runs 8× faster than TPT with 12× less memory, and substantially outperforms existing TTA methods on OOD datasets.

## Background & Motivation

CLIP and similar VLMs achieve strong zero-shot performance, but suffer significant degradation under distribution shift (OOD data) at test time. Test-time adaptation (TTA) has emerged to dynamically adapt models to unlabeled test samples during inference. However, existing TTA methods share three fundamental issues:

**High computational overhead**: TPT (Test-Time Prompt Tuning) and its variants require **backpropagation through large text encoders** to update prompt parameters, leading to substantial inference time and memory costs. TPT consumes 0.75 seconds and 17.6 GB per sample.

**Requires internal model modification**: Parameter-efficient fine-tuning methods such as TTL (based on LoRA) require access to and modification of internal model architecture (e.g., attention layers), departing from a truly black-box paradigm and rendering them inapplicable to proprietary or fixed-architecture models.

**Unconstrained representation shift**: TPS (Test-Time Prototype Shifting) directly learns shift vectors in the embedding space to avoid encoder backpropagation, but its shifts are **unconstrained high-dimensional vectors**—arbitrary directions in high-dimensional space may deviate from semantically meaningful regions and are sensitive to noise.

The authors' core insight is that embeddings from pretrained deep networks exhibit **low intrinsic dimensionality**—their essential information lies on a low-dimensional manifold. Adaptation should therefore be performed within this low-dimensional semantic subspace rather than via arbitrary shifts in the full high-dimensional space. This motivates STS: identifying principal semantic directions via SVD and performing constrained steering along those directions.

## Method

### Overall Architecture

STS operates in two stages: precomputation and test-time adaptation.
- **Precomputation stage** (one-time): All class text descriptions are encoded to obtain initial prototypes $Z_{T_{init}} \in \mathbb{R}^{C \times D}$, which are then decomposed via SVD; the top-$k_t$ right singular vectors are extracted as the semantic adaptation basis $B_T$.
- **Test-time stage** (per sample): $k_t$ coefficients $\gamma$ (initialized to zero) are learned to generate a steering vector applied to all class prototypes, minimizing the marginal entropy over augmented views.

### Key Designs

1. **Spectral subspace identification**: SVD is applied to the initial text prototypes $Z_{T_{init}}$:
    $$Z_{T_{init}} = U_T S_T V_T^\top$$

    The right singular vectors corresponding to the top-$k_t$ largest singular values are selected as $B_T = [v_1, v_2, \ldots, v_{k_t}] \in \mathbb{R}^{D \times k_t}$. These vectors capture the most prominent axes of semantic variation across class concepts. The value of $k_t$ is determined automatically using the Gavish–Donoho optimal hard threshold, based on the matrix aspect ratio and the median singular value.

    A key property is that a small number of singular vectors captures $>90\%$ of total energy (e.g., for 1,000 ImageNet classes in a 512-dimensional space, $k_t$ is typically only in the tens), meaning adaptation operates in an extremely low-dimensional space.

2. **Subspace coefficient steering mechanism**: For each test sample, a **shared** $k_t$-dimensional coefficient vector $\gamma \in \mathbb{R}^{k_t}$ (shared across all classes) is learned to reconstruct and apply a steering vector:
    $$\Delta z_T = B_T \gamma$$
    $$(z_{T_{adapted}})_c = \text{normalize}((z_{T_{init}})_c + \Delta z_T)$$

    This design has two key properties: (a) all classes share the same steering vector, keeping the parameter count minimal (only $k_t$ coefficients); (b) steering is constrained to the semantic subspace, providing implicit regularization against overfitting to noisy augmentations.

3. **Marginal entropy-based test-time optimization**:
    - 63 augmented views (random crop + horizontal flip only) are generated for the test image.
    - Views are filtered by prediction entropy using initial prototypes; the most confident top-10% are retained.
    - The marginal probability distribution over retained views is computed using the adapted prototypes:
    $$\bar{P}_{adapted}(c) = \frac{1}{N_{filt}}\sum_{j'} \text{softmax}_c(L_{adapted}^{(j')}(c))$$
    - The objective minimizes marginal entropy with L2 regularization:
    $$\mathcal{L}_{STS} = H(\bar{P}_{adapted}) + \lambda_R \|\Delta z_T\|_2$$
    - Optimization uses AdamW with learning rate 5e-3, for **a single gradient step**.

### Loss & Training

- **Objective**: Marginal entropy minimization—encourages consistent, peaked predictions across high-confidence augmented views.
- **Regularization**: L2 regularization constrains the magnitude of the shift.
- **Initialization**: $\gamma$ is initialized to the zero vector (no shift at the start).
- **Optimization steps**: Only 1 step—extremely efficient.
- **Per-sample independence**: $\gamma$ is reset to zero after processing each sample.

## Key Experimental Results

### Main Results — Natural Distribution Shift (ImageNet and OOD Variants)

| Method | ImageNet | ImgNet-A | ImgNet-V2 | ImgNet-R | ImgNet-Sketch | OOD Avg |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Zero-Shot CLIP | 66.73 | 47.87 | 60.86 | 73.98 | 46.09 | 57.20 |
| TPT | 68.97 | 54.39 | 63.37 | 77.07 | 48.01 | 60.71 |
| TPS | 67.96 | 57.46 | 62.95 | 74.90 | 46.03 | 60.34 |
| **STS** | 68.85 | 61.23 | 64.15 | 77.13 | 48.06 | **62.64** |
| **STS_Ensemble** | **70.81** | **64.29** | **64.82** | **80.53** | **50.19** | **64.96** |

### Ablation Study — Efficiency Comparison

| Method | Time/sample (s) | Memory (GB) | ImageNet Acc | Gain |
|------|:---:|:---:|:---:|:---:|
| Zero-Shot | 0.02 | 0.83 | 66.73 | - |
| TPT | 0.75 | 17.6 | 68.97 | +2.24 |
| **STS_Ensemble** | **0.09** | **1.4** | **70.81** | **+4.08** |
| Speedup vs. TPT | **8.3×** | **12.6×** | - | **1.8×** |

### Fine-Grained Classification (Average over 10 Datasets)

| Method | Mean Accuracy | Best Datasets |
|------|:---:|------|
| Zero-Shot | 63.58% | - |
| TPT | 64.78% | - |
| TPS | 63.49% | - |
| **STS_Ensemble** | **65.06%** | Caltech101, EuroSAT, Cars, Food, SUN |

### Key Findings

- **Large OOD gains**: STS outperforms TPT by 6.84% on ImageNet-A (61.23% vs. 54.39%) and by 1.93% on average OOD accuracy—while being 8× faster.
- **TPS vs. STS validates the value of subspace constraints**: Unconstrained high-dimensional shifting in TPS underperforms TPT, whereas constrained steering in STS's low-dimensional subspace substantially outperforms both.
- **A single optimization step suffices**: The SVD basis provides well-structured semantic directions that eliminate the need for multi-step iteration.
- **Prompt ensemble further improves performance**: STS_Ensemble with 7 generic templates achieves the best results across all settings.

## Highlights & Insights

1. **"Adapt within a low-dimensional subspace" is the key insight**: Text embeddings of pretrained models exhibit low-rank structure; a small number of singular vectors captures the dominant semantic variation. Steering within this subspace amounts to shifting along "semantically meaningful directions" rather than taking arbitrary steps in high-dimensional space.
2. **Truly black-box adaptation**: STS operates entirely in the encoder's output embedding space without accessing any internal components, making it applicable to API-only VLM services.
3. **Extreme efficiency–performance trade-off**: Only $k_t$ parameters (typically tens), one optimization step, 0.09 s/sample, 1.4 GB memory, and state-of-the-art performance—a simultaneous breakthrough in both efficiency and accuracy within the TTA paradigm.
4. **Methodological simplicity**: The entire approach can be described in a few lines—SVD + linear combination + one gradient step.

## Limitations & Future Work

- **Limitation of linear steering**: When domain shift is highly nonlinear, shifts within the linear SVD subspace may be insufficient to capture complex distributional changes.
- **Linear cost in augmentation views**: Computational cost scales linearly with the number of augmented views; 64 views is a reasonable trade-off but still incurs overhead.
- **Shared steering vector**: Applying the same shift direction to all classes may be suboptimal when different classes require distinct adaptation directions.
- **Dependence on SVD quality**: With very few classes, SVD may fail to extract meaningful semantic directions.

## Related Work & Insights

- **Comparison with TPS**: TPS also operates in embedding space but uses unconstrained shift vectors; STS introduces a critical inductive bias via SVD subspace projection.
- **Comparison with TPT**: TPT requires encoder access to optimize prompts; STS operates at the encoder output, achieving truly black-box adaptation.
- **Implications for VLM adaptation paradigms**: The structural properties of embedding spaces (low rank, principal components) constitute valuable priors; exploiting such structure is more effective than brute-force optimization.
- **Potential extension**: The visual-side embeddings may similarly benefit from spectral analysis and subspace steering.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Applying SVD spectral decomposition to text prototype adaptation for TTA is a novel and insightful idea.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers ImageNet OOD variants + 10 fine-grained datasets + CIFAR10-C + efficiency analysis + multiple backbones + detailed ablations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear motivation, rigorous method derivation, and excellent experimental presentation.
- **Value**: ⭐⭐⭐⭐ High practical value (8× speedup + better performance), elegant and simple design, though impact is scoped to the TTA subfield.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] TOMCAT: Test-time Comprehensive Knowledge Accumulation for Compositional Zero-Shot Learning](tomcat_test-time_comprehensive_knowledge_accumulation_for_compositional_zero-sho.md)
- [\[NeurIPS 2025\] DOTA: DistributiOnal Test-time Adaptation of Vision-Language Models](dota_distributional_testtime_adaptation_of_visionlanguage_mo.md)
- [\[NeurIPS 2025\] The Illusion of Progress? A Critical Look at Test-Time Adaptation for Vision-Language Models](the_illusion_of_progress_a_critical_look_at_testtime_adaptat.md)
- [\[ICCV 2025\] Multi-Cache Enhanced Prototype Learning for Test-Time Generalization of Vision-Language Models](../../ICCV2025/multimodal_vlm/multi-cache_enhanced_prototype_learning_for_test-time_generalization_of_vision-l.md)
- [\[NeurIPS 2025\] Mint: A Simple Test-Time Adaptation of Vision-Language Models against Common Corruptions](mint_a_simple_testtime_adaptation_of_visionlanguage_models_a.md)

</div>

<!-- RELATED:END -->
