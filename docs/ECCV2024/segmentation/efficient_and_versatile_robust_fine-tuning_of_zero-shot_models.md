---
title: >-
  [Paper Note] Efficient and Versatile Robust Fine-Tuning of Zero-shot Models
description: >-
  [ECCV 2024][Segmentation][Robust Fine-Tuning] R-Adapter introduces a lightweight adapter module into the CLIP model along with three self-ensemble strategies (Adapter Dropping, Weight Accumulation, and Weight-Scaling Re-parameterization). While fine-tuning only 13% of the parameters, it simultaneously achieves high ID accuracy and strong OOD robustness, and extends robust fine-tuning to cross-modal retrieval and open-vocabulary segmentation tasks for the first time.
tags:
  - "ECCV 2024"
  - "Segmentation"
  - "Robust Fine-Tuning"
  - "Parameter-Efficient Fine-Tuning"
  - "Self-Ensemble"
  - "CLIP"
  - "Out-of-Distribution Generalization"
date: 2026-05-08
content_hash: 2296f3ce8a9b997f
---

# Efficient and Versatile Robust Fine-Tuning of Zero-shot Models

**Conference**: ECCV 2024  
**arXiv**: [2408.05749](https://arxiv.org/abs/2408.05749)  
**Code**: Yes (cvlab.postech.ac.kr/research/R-Adapter)  
**Area**: Image Segmentation / Vision-Language Model Fine-Tuning  
**Keywords**: Robust Fine-Tuning, Parameter-Efficient Fine-Tuning, Self-Ensemble, CLIP, Out-of-Distribution Generalization

## TL;DR

R-Adapter introduces a lightweight adapter module into the CLIP model along with three self-ensemble strategies (Adapter Dropping, Weight Accumulation, and Weight-Scaling Re-parameterization). While fine-tuning only 13% of the parameters, it simultaneously achieves high ID accuracy and strong OOD robustness, and extends robust fine-tuning to cross-modal retrieval and open-vocabulary segmentation tasks for the first time.

## Background & Motivation

### Key Challenge

Large-scale vision-language pre-trained models (such as CLIP) exhibit zero-shot classification and cross-distribution generalization capabilities. However, fine-tuning them on downstream tasks faces two conflicting challenges:

**Robustness Degradation**: Full fine-tuning disrupts pre-trained knowledge, leading to a substantial drop in out-of-distribution (OOD) accuracy. For example, CLIP ViT-B/16 achieves a zero-shot average OOD accuracy of 58.4%, which drops to 52.8% after full fine-tuning—where the ID accuracy gains 18.4% but the OOD accuracy only improves by 2.0%.

**Prohibitive Computational Cost**: Full fine-tuning requires updating all parameters (e.g., 305M parameters in ViT-L/14), making the memory and storage overhead unsustainable as the model scale grows.

### Limitations of Prior Work

| Method Type | Representative | Pros | Cons |
|---------|------|------|------|
| Robust Fine-Tuning (WiSE-FT, Mask-Fill) | Weight-space interpolation / regularization | OOD Robust | Requires full fine-tuning (80M+ parameters); limited to classification tasks |
| Parameter-Efficient Fine-Tuning (PEFT) (AdaptFormer, MaPLe) | Fine-tuning only a few parameters | Efficient | Poor OOD robustness, performance drops sharply under distribution shifts |

**Key Insight**: No existing method simultaneously addresses both challenges. Robust fine-tuning methods are inefficient, while PEFT methods lack robustness. The Core Idea of R-Adapter is to **introduce a self-ensemble mechanism into the PEFT framework to achieve the robustness benefits of weight-space ensembling at a low cost**.

### Inspiration from Ensembling

Weight-space ensembling (such as WiSE-FT, which linearly interpolates pre-trained and fine-tuned weights) has been shown to improve OOD generalization. However, traditional methods require storing two complete models. The novelty of R-Adapter lies in compressing the weight-space ensemble into a **single model** via adapter re-parameterization, requiring only the storage of the adapter weights (approx. 13% of the total parameters).

## Method

### Overall Architecture

R-Adapter is built on the Vision Transformer and Text Transformer of CLIP:
- A lightweight adapter module is inserted after the MHA and FFN of each Transformer layer.
- All pre-trained parameters are frozen, and only the adapter weights are trained.
- Three self-ensemble strategies are combined to enhance OOD robustness.
- An MPM-NCE loss function is used to replace the standard InfoNCE.

### Key Designs

#### 1. **Adapter Module Design**

A simplified version of the Houlsby Adapter (without non-linear layers and bias terms) is adopted, structured as a residual linear transformation:

$$h(X) = X W_{\text{adp}} + X$$

where $W_{\text{adp}} \in \mathbb{R}^{d \times d}$ is the adapter weight matrix. For full data, a full-rank structure is used, while for few-shot learning, a low-rank decomposition $W_{\text{adp}} = BA$ ($B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times d}, r \ll d$) can be used.

Since the adapter has no non-linear layers, it can be merged with the preceding pre-trained weight through **re-parameterization**, incurring zero extra computational overhead during inference:

$$W_{\text{rep}} = W_{\text{org}}(W_{\text{adp}} + I), \quad b_{\text{rep}} = b_{\text{org}}(W_{\text{adp}} + I)$$

**Design Motivation**: The linear adapter enables re-parameterization, which is the cornerstone for the subsequent weight-space ensemble—eliminating the need to store two separate models.

#### 2. **Three Self-Ensemble Strategies**

**(a) Dynamic Ensemble by Adapter Dropping**

During training, the adapter module is randomly deactivated with a probability $p$:

$$h(X) = \frac{\gamma}{1-p} \cdot X W_{\text{adp}} + X, \quad \gamma \sim \text{Bernoulli}(1-p)$$

No dropping is applied during inference. This is equivalent to implicitly ensembling multiple subnetworks (with different combinations of active/inactive adapters).

**Fundamental Difference from Dropout/Drop-path**: Dropout creates feature sparsity, and Drop-path reduces model depth. In contrast, Adapter Dropping uniquely and randomly switches between **pre-trained features** and **adapted features**, ensuring that pre-trained knowledge is always preserved. Ablation studies show that using AD alone improves OOD by 1.9% (E3 vs E0).

**(b) Temporal Ensemble by Accumulation**

The historical versions of the adapter weights are accumulated using the Exponential Moving Average (EMA):

$$\tilde{W}_{\text{adp}} \leftarrow m \cdot \tilde{W}_{\text{adp}} + (1-m) \cdot W_{\text{adp}}$$

The accumulated weights $\tilde{W}_{\text{adp}}$ are used during inference. This represents an ensemble in the **temporal dimension**, capturing the combined information of all models during training at almost zero extra cost.

**Design Motivation**: Performing EMA only on the adapter parameters (rather than the entire model) is extremely memory-efficient.

**(c) Weight-space Ensemble by Re-scaling**

Linear interpolation between pre-trained and fine-tuned weights is performed using a scaling coefficient $\alpha$:

$$W_{\text{ens}} = \alpha \tilde{W}_{\text{rep}} + (1-\alpha) W_{\text{org}} = W_{\text{org}}(\alpha \tilde{W}_{\text{adp}} + I)$$

**Core Advantage**: While traditional WiSE-FT requires storing two full models to perform weight interpolation, R-Adapter only needs to store the adapter weights, achieving an equivalent weight-space ensemble **within a single model** through re-parameterization. $\alpha$ controls the degree of pre-trained knowledge preservation (using RS alone boosts OOD by 5.8%, E5 vs E0).

#### 3. **MPM-NCE Loss Function**

Designed for downstream vision-language tasks, addressing two issues in the standard InfoNCE:

**(a) Multi-Positive Soft Labels**: In classification tasks where multiple text templates correspond to the same category, InfoNCE considers only a single positive pair, which causes semantic conflict. MPM-NCE addresses this with soft label assignment:

$$\tilde{y}_{ij} = \frac{(1-\epsilon) \cdot y_{ij}}{|P(i)|} + \frac{\epsilon \cdot (1-y_{ij})}{B - |P(i)|}$$

where $P(i)$ is the set of positive samples for sample $i$, and $\epsilon$ is the label smoothing noise.

**(b) Angular Margin**: An angular margin $\delta$ is applied to negative pairs to enhance discriminative power:

$$\mathcal{L}(\mathcal{B}) = -\sum_{i,j=1}^{B} \left(\tilde{y}_{ij} \log \frac{e^{(f_i \cdot g_j + \delta_{ij})/\tau}}{\sum_{k=1}^{B} e^{(f_i \cdot g_k + \delta_{ik})/\tau}} + \tilde{y}_{ji} \log \frac{e^{(f_j \cdot g_i + \delta_{ji})/\tau}}{\sum_{k=1}^{B} e^{(f_k \cdot g_i + \delta_{ki})/\tau}}\right)$$

where $\delta_{ij} = 0$ (for positive pairs) or $\delta$ (for negative pairs), and $\tau = 0.01$.

### Training Strategy

- Optimizer: AdamW (no weight decay)
- Trained for 10 epochs with a learning rate of $5 \times 10^{-4}$ using a cosine schedule with 500 warmup steps
- Hyperparameters: $p=0.2$ (drop probability), $m=0.999$ (EMA momentum), $\delta=0.05$ (margin), $\alpha=0.5$ (for classification tasks)

## Key Experimental Results

### Main Results

ImageNet classification + 5 OOD datasets (CLIP ViT-B/16):

| Method | Trainable Parameters | IN (ID) | OOD avg | IN-V2 | IN-R | IN-Sketch | ObjectNet | IN-A |
|------|-----------|---------|---------|-------|------|-----------|-----------|------|
| Zero-Shot | 0 | 68.3 | 58.4 | 61.9 | 77.6 | 48.3 | 54.0 | 50.1 |
| Fine-Tuning | 86.7M | 80.7 | 52.8 | 70.4 | 64.0 | 45.1 | 49.1 | 35.2 |
| WiSE-FT | 86.7M | 81.7 | 63.0 | 72.8 | 78.7 | 53.9 | 57.3 | 52.2 |
| Mask-Fill | 86.7M | 82.4 | 63.3 | 73.4 | 78.1 | 53.4 | 57.9 | 53.5 |
| **R-Adapter (Ours)** | **20.5M** | 82.0 | **64.8** | 73.6 | **79.1** | **53.9** | **59.7** | **57.5** |

R-Adapter outperforms all robust fine-tuning methods in OOD performance (+1.5% OOD avg) with only **1/4 of the parameters**, showing a particularly pronounced advantage (+4.0%) on IN-A, which represents the largest distribution shift.

### Ablation Study

Contribution of each component to ImageNet classification (ViT-B/32):

| Configuration | ID | OOD avg | Description |
|------|-------|---------|------|
| E0: Base Adapter | 77.5 | 47.7 | Baseline |
| E3: + Adapter Dropping | 77.8 (+0.3) | 49.6 (+1.9) | Dynamic ensembling significantly improves OOD |
| E5: + Re-scaling | 76.5 (-1.0) | 53.5 (+5.8) | Weight interpolation substantially boosts OOD |
| E7: AD + AC + RS | 76.6 (-0.9) | 53.7 (+6.0) | Combined three-strategy |
| E9: + MPM-NCE | 77.5 (±0) | 53.9 (+6.2) | Loss function improvement |
| E10: + Label Smooth | **77.7 (+0.2)** | **54.3 (+6.6)** | Final configuration |
| B1: AdaptFormer | 77.2 | 48.5 | Existing PEFT baseline |
| B2: RepAdapter | 77.2 | 48.3 | Existing PEFT baseline |

Hyperparameter sensitivity analysis (ViT-B/32):

| margin δ | ID | OOD | Description |
|----------|-----|-----|------|
| 0 | 77.1 | 54.0 | No margin |
| 0.05 (Default) | 77.7 | 54.3 | Optimal balance |
| 0.1 | 77.8 | 53.8 | Excessive margin drops OOD |
| w/ Single Positive | 77.2 | 47.0 | Poor performance with single positive + margin |

### Key Findings

1. **Adapter Dropping ≠ Dropout**: Dropout (E1, +1.1 OOD) and Drop-path (E2, +0.2 OOD) are far less effective than Adapter Dropping (E3, +1.9 OOD), as the latter switches between pre-trained and adapted knowledge.
2. **Re-scaling is Key to OOD Robustness**: Using RS alone yields a +5.8% OOD improvement, but slightly decreases ID performance (-1.0%), necessitating its combination with AD/AC to balance.
3. **Synergy of MPM-NCE + Label Smoothing**: InfoNCE + LS degrades ID accuracy due to semantic conflicts, but MPM-NCE's multi-positive mechanism avoids this, with the combination improving both ID and OOD.
4. **Broad Generalizability Across Tasks**: R-Adapter is effective in few-shot classification (outperforming CLIPood by +1.2% OOD avg), cross-modal retrieval, and open-vocabulary segmentation.
5. **Insensitivity to α**: Compared to WiSE-FT, R-Adapter's performance curve remains flatter across different $\alpha$ values.

## Highlights & Insights

1. **Conceptual Unification**: Unified parameter-efficient fine-tuning and robust fine-tuning into a single framework for the first time, proving they are complementary rather than conflicting.
2. **Lightweight Ensemble via Re-parameterization**: Through the linear structure of the adapter combined with re-parameterization, the dual-model weight interpolation of WiSE-FT is compressed into a single-model operation, reducing the storage cost from 2× models to the adapter size.
3. **Benchmark Expansion**: Extended the evaluation of robust fine-tuning from classification to cross-modal retrieval and open-vocabulary segmentation for the first time, driving systematic evaluation in the field.
4. **Generality of MPM-NCE**: The design of multi-positive + margin losses can be widely applied to vision-language tasks with many-to-many mapping relationships.

## Limitations & Future Work

1. The full-rank adapter still has 64.5M parameters on ViT-L (relatively large), and the performance of the low-rank version drops.
2. The $\alpha$ value needs to be manually tuned for different tasks (0.5 for classification, 0.8 for retrieval, 0.4 for segmentation).
3. More complex adapter architectures (e.g., with gating mechanisms) were not explored, which might further improve the ID-OOD balance.
4. The evaluation is primarily based on CLIP, leaving its effectiveness on other VLMs (e.g., ALIGN, SigLIP) unverified.
5. OOD evaluation is limited to natural distribution shifts, without addressing adversarial attack scenarios.

## Related Work & Insights

- The weight interpolation concept of **WiSE-FT** is elegantly integrated into the PEFT framework here.
- The Adapter Dropping strategy can inspire robustness improvements in other PEFT methods (such as LoRA).
- The MPM-NCE loss has potential applications in CLIP-based open-world detection/segmentation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The three self-ensemble strategies are highly creative, and the idea of achieving weight-space ensembling via re-parameterization is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Extensive evaluations across 4 ViT scales, 5 different tasks, with comprehensive ablated and hyperparameter analyses.
- **Writing Quality**: ⭐⭐⭐⭐ — Clearly structured with complete mathematical derivations, though some formula notations are slightly heavy.
- **Value**: ⭐⭐⭐⭐⭐ — Parameter-efficient and robust, highly suited for real-world deployment scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Robust 3D Shape Reconstruction in Zero-Shot from a Single Image in the Wild](../../CVPR2025/segmentation/robust_3d_shape_reconstruction_in_zero-shot_from_a_single_image_in_the_wild.md)
- [\[NeurIPS 2025\] Fast and Fluent Diffusion Language Models via Convolutional Decoding and Rejective Fine-tuning](../../NeurIPS2025/segmentation/fast_and_fluent_diffusion_language_models_via_convolutional_decoding_and_rejecti.md)
- [\[ICLR 2026\] Object-Centric Refinement for Enhanced Zero-Shot Segmentation](../../ICLR2026/segmentation/object-centric_refinement_for_enhanced_zero-shot_segmentation.md)
- [\[ICML 2025\] InfoSAM: Fine-Tuning the Segment Anything Model from An Information-Theoretic Perspective](../../ICML2025/segmentation/infosam_fine-tuning_the_segment_anything_model_from_an_information-theoretic_per.md)
- [\[ICCV 2025\] ZIM: Zero-Shot Image Matting for Anything](../../ICCV2025/segmentation/zim_zero-shot_image_matting_for_anything.md)

</div>

<!-- RELATED:END -->
