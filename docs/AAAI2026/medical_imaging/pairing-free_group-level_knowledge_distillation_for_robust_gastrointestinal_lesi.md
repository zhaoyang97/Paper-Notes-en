---
title: >-
  [Paper Note] Pairing-free Group-level Knowledge Distillation for Robust Gastrointestinal Lesion Classification in White-Light Endoscopy
description: >-
  [AAAI 2026][Medical Imaging][Knowledge Distillation] This paper proposes PaGKD, a pairing-free group-level knowledge distillation framework that eliminates the dependency on paired data in conventional NBI→WLI cross-moda…
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "Knowledge Distillation"
  - "Cross-modal Learning"
  - "White-Light Endoscopy"
  - "Narrow-Band Imaging"
  - "Pairing-free"
  - "Group-level Distillation"
  - "Gastrointestinal Lesion Classification"
date: 2026-05-08
content_hash: 810ee0f6212928c1
---

# Pairing-free Group-level Knowledge Distillation for Robust Gastrointestinal Lesion Classification in White-Light Endoscopy

**Conference**: AAAI 2026
**arXiv**: [2601.09209](https://arxiv.org/abs/2601.09209)
**Code**: [Huster-Hq/PaGKD](https://github.com/Huster-Hq/PaGKD)
**Area**: Medical Imaging / Endoscopy
**Keywords**: Knowledge Distillation, Cross-modal Learning, White-Light Endoscopy, Narrow-Band Imaging, Pairing-free, Group-level Distillation, Gastrointestinal Lesion Classification

## TL;DR

This paper proposes PaGKD, a pairing-free group-level knowledge distillation framework that eliminates the dependency on paired data in conventional NBI→WLI cross-modal distillation. It introduces group-level prototype distillation (GKD-Pro, which extracts modality-invariant semantic prototypes via a shared lesion query Transformer) and group-level dense distillation (GKD-Den, which achieves dense spatial alignment through activation map-guided semantic relation cross-attention). PaGKD improves AUC by 3.3%/1.1%/2.8%/3.2% across four clinical datasets.

## Background & Motivation

**Background**: Endoscopy is a critical tool for early detection of gastrointestinal (GI) cancers. White-light imaging (WLI) is the standard clinical modality, while narrow-band imaging (NBI) enhances vascular and mucosal details through spectral filtering, providing superior lesion visibility and classification performance. However, NBI is often unavailable or underutilized in routine clinical deployment.

**Limitations of Prior Work**:
- **Poor WLI classification performance**: WLI images lack the fine-grained vascular and mucosal information present in NBI, limiting standalone classification performance.
- **Scarcity and high cost of paired data**: Existing cross-modal distillation methods (ADD, CPC-Trans, PolypsAlign) require paired NBI-WLI images of the same lesion, which are difficult to acquire.
- **Underutilization of abundant unpaired data**: Large quantities of NBI and WLI images are independently collected from different lesions and patients, yet existing methods cannot leverage them.
- **Semantic mismatch in instance-level distillation**: When instance-level alignment is applied to unpaired images, individual lesion images capture only partial disease characteristics, resulting in cross-modal feature incompatibility.

**Key Challenge**: NBI knowledge benefits WLI classification, yet existing distillation methods require paired data, leaving vast amounts of unpaired data unused and limiting WLI classification performance.

**Goal**: To leverage abundant unpaired NBI and WLI data for effective cross-modal knowledge distillation, thereby improving WLI-only lesion classification performance.

**Key Insight**: Rather than aligning individual images, same-class lesion images are organized into "groups" for group-level distillation — multiple images within a group provide a more complete disease representation, mitigating bias and noise from individual samples.

**Core Idea**: Group-level distillation = prototype-level global semantic alignment (GKD-Pro) + dense-level local spatial alignment (GKD-Den), without requiring image-level pairing.

## Method

### Overall Architecture

PaGKD consists of three components:
1. A **pretrained frozen NBI classifier** (teacher)
2. A **trainable WLI classifier** (student)
3. **Two group-level knowledge distillation modules**: GKD-Pro and GKD-Den

At each training iteration, same-class image groups $\mathcal{G}_c^{mod}$ are constructed, where $c$ denotes the class and $mod \in \{WLI, NBI\}$. Each image independently passes through its corresponding classifier to extract feature maps $\mathcal{F}_c^{mod} \in \mathbb{R}^{N_c \times d \times h \times w}$, which are then flattened and aggregated into a unified representation.

### Key Design 1: Group-level Prototype Knowledge Distillation (GKD-Pro)

**Function**: Extracts modality-invariant lesion semantic prototypes for cross-modal alignment at the global high-level representation.

**Lesion-Relevant Query Transformer (LR-QFormer)**:
- A set of shared learnable lesion queries $\mathcal{Q} \in \mathbb{R}^{N_q \times d}$ is designed ($N_q=12$, far smaller than the feature sequence length $L_c$).
- Queries are shared across all groups and modalities, serving as category- and modality-agnostic "lesion concept anchors."
- Within each of $T$ Transformer blocks, queries first undergo self-attention (SA), then cross-attention (CA) with group features:
$$\mathcal{Q}_{t,c}^{mod} = \text{CA}(\text{SA}(\mathcal{Q}_{t-1,c}^{mod}),\ \mathcal{F}_c^{mod} + \mathbf{E}_{pos})$$
- After $T$ iterations, each query accumulates modality-specific evidence for specific disease attributes.

**Group-level Contrastive Loss**:
- Similarity is defined as the mean cosine similarity between queries of the same index:
$$S_{\mathcal{Q}_c^{WLI}, \mathcal{Q}_{c'}^{mod'}} = \frac{1}{N_q}\sum_{i=1}^{N_q} \frac{(\mathbf{q}_{i,c}^{WLI})^\top \mathbf{q}_{i,c'}^{mod'}}{\|\mathbf{q}_{i,c}^{WLI}\| \|\mathbf{q}_{i,c'}^{mod'}\|}$$
- A symmetric contrastive objective pulls same-class WLI-NBI prototypes together while pushing apart prototypes of different classes.

### Key Design 2: Group-level Dense Knowledge Distillation (GKD-Den)

**Function**: Aligns cross-modal features at a fine-grained local spatial level, complementing the spatial details missed by GKD-Pro.

**Semantic Relation Generation**:
1. Class activation maps (CAMs) are computed for each group's features, followed by pixel-adaptive refinement to enhance spatial consistency.
2. A dual-threshold scheme ($\tau_1=0.3, \tau_2=0.7$) discretizes CAMs into: background (0), lesion (1), and ambiguous (∅).
3. A semantic relation matrix $\mathbf{R}_c \in \{0, -\infty\}^{L_c \times L_c}$ is constructed:
   - Set to 0 when two positions share the same non-ambiguous label in their respective CAMs (allowing attention to pass).
   - Set to $-\infty$ otherwise (blocking attention).

**Semantic Relation-guided Cross-Attention (SRCA)**:
$$\mathbf{A} = \text{Softmax}\left(\mathbf{R}_c + \frac{(\mathcal{F}_c^{WLI} W_q)(\mathcal{F}_c^{NBI} W_k)^\top}{\sqrt{d/4}}\right)$$

- The relation matrix $\mathbf{R}_c$ serves as an attention mask to guide spatial reconstruction.
- Bidirectional reconstruction: NBI→WLI and WLI→NBI.
- A bidirectional consistency loss $\mathcal{L}_{den}$ constrains the L2 distance between reconstructed and original features.

### Loss & Training

$$\mathcal{L}_{total} = \mathcal{L}_{pro} + \mathcal{L}_{den} + \mathcal{L}_{cls}$$

where $\mathcal{L}_{cls}$ is the cross-entropy loss of the WLI classifier. Only the WLI classifier is retained at inference.

## Key Experimental Results

### Datasets

| Dataset | Classes | Paired | Unpaired WLI | Unpaired NBI | Type |
|--------|------|------|-----------|-----------|------|
| PICCOLO | 3 | 1,055 | 1,065 | 214 | Colorectal Polyp |
| PolypSet | 2 | 165 | 450 | 450 | Colorectal Polyp |
| IH-Polyp | 2 | 556 | 3,730 | 921 | Colorectal Polyp |
| IH-GC | 2 | 264 | 469 | 303 | Gastric Cancer |

### Main Results (vs. 8 SOTA Methods)

**PICCOLO Dataset (3-class)**:

| Method | Training Data | Acc | F1 | AUC |
|------|----------|-----|-----|-----|
| ADD (paired) | $\mathcal{D}_p$ | 79.1 | 76.4 | 83.8 |
| ADD (paired+unpaired) | $\mathcal{D}_p + \mathcal{D}_{unp}$ | 77.9 | 74.2 | 84.2 |
| **PaGKD (unpaired)** | $\mathcal{D}_{unp}$ | 80.8 | 78.8 | 86.6 |
| **PaGKD (all)** | $\mathcal{D}_p + \mathcal{D}_{unp}$ | **81.9** | **81.1** | **90.1** |

Key observations:
1. **PaGKD using only unpaired data** (80.8/78.8/86.6) already surpasses all SOTA methods trained on paired data.
2. **Adding unpaired data degrades existing CDC methods**: ADD improves marginally from 83.8→84.2 (AUC only), CPC-Trans from 86.6→87.2, with negligible gains.
3. **PaGKD with full data** achieves AUC improvements of at least 3.3%, 1.1%, 2.8%, and 3.2% across the four datasets.

**AUC Summary Across Four Datasets**:

| Method | PICCOLO | PolypSet | IH-Polyp | IH-GC |
|------|---------|----------|----------|-------|
| NBI Classifier (upper bound) | 86.9 | 97.6 | 87.0 | 86.3 |
| Strongest Baseline | 87.2 | 93.7 | 82.8 | 81.4 |
| **PaGKD** | **90.1** | **94.7** | **85.1** | **84.0** |

### Ablation Study

**Component Ablation**:

| GKD-Pro | GKD-Den | PICCOLO AUC | IH-GC AUC |
|---------|---------|-------------|-----------|
| ✗ | ✗ | 71.2 | 66.9 |
| ✓ | ✗ | 83.5 | 75.5 |
| ✗ | ✓ | 85.0 | 77.3 |
| ✓ | ✓ | **90.1** | **84.0** |

- Each module individually yields substantial improvements (+12.3/+13.8 and +8.6/+10.4).
- Joint use provides further gains, indicating complementarity rather than redundancy.

**Sub-component Ablation**:
- Removing LR-QFormer (replaced by average pooling): PICCOLO AUC drops from 83.5 to 78.3.
- Removing SRCA (replaced by standard cross-attention): AUC drops from 85.0 to 83.2.
- Removing bidirectional consistency (unidirectional NBI→WLI only): AUC drops from 90.1 to 87.8.

**Group-level vs. Image-level Distillation**:

| Distillation Level | GKD-Pro AUC | GKD-Den AUC | Joint AUC |
|----------|-------------|-------------|---------|
| Image-level | 78.6 | 79.9 | 84.3 |
| **Group-level** | **83.5** | **85.0** | **90.1** |

Group-level distillation consistently outperforms image-level distillation across all settings (joint AUC on PICCOLO: 90.1 vs. 84.3), validating that group-level aggregation mitigates noise in unpaired data.

### Key Findings

1. Group-level distillation is the key to exploiting unpaired data — image-level distillation introduces noise and semantic mismatch when applied to unpaired data.
2. GKD-Pro (global semantics) and GKD-Den (local spatial) are complementary — their joint improvement substantially exceeds either module used alone.
3. PaGKD using only unpaired data already matches the strongest baselines trained on paired data.
4. Unpaired data not only does not hurt PaGKD but provides additional training signal.
5. The shared query design in LR-QFormer ensures consistent lesion concept extraction across modalities and categories.

## Highlights & Insights

1. **Paradigm shift**: From "pairing required" to "pairing-free" cross-modal distillation — this unlocks the large volume of idle unpaired clinical data available in practice.
2. **The wisdom of group-level aggregation**: A single lesion image captures only a fragment of the disease; group-level aggregation provides a more complete and robust disease representation — a principle generalizable to other medical multi-modal tasks.
3. **Shared queries as semantic anchors**: The shared learnable queries in LR-QFormer act as a "disease attribute dictionary," establishing a unified semantic reference across different modalities and categories.
4. **CAM-guided dense alignment**: Leveraging CAMs to determine semantic correspondences for guiding cross-attention is more fine-grained than global alignment and more robust than pixel-level alignment.
5. **Fair experimental design**: CDC methods are also trained on paired plus unpaired data for comparison, fairly demonstrating the degradation of existing methods on unpaired data.

## Limitations & Future Work

1. **Group construction strategy**: Organizing same-class image groups within a batch requires hyperparameter tuning based on the class distribution of each dataset.
2. **Computational overhead**: The spatial complexity of group-level operations scales with group size ($L_c = N_c \cdot h \cdot w$), which may become memory-constrained for large groups.
3. **Evaluation limited to binary/ternary classification**: Fine-grained classification of GI lesions (e.g., multiple polyp subtypes) has not been tested.
4. **Backbone limitation**: Only ResNet-50 is used as the backbone; stronger architectures such as ViT are not explored.
5. **Label assumption**: The method requires class labels for unpaired data — unlabeled unpaired data remains unexploitable.
6. **Dependence on CAM quality**: Semantic relation generation in GKD-Den relies on CAM quality; weak classifiers may produce noisy CAMs.

## Related Work & Insights

- **Cross-modal Independent Classification (CIC)**: SSL-CPCD (self-supervised patch-image clustering), SSL-WCE (adaptive aggregation attention), FFCNet (Fourier transform denoising)
- **Cross-modal Distillation Classification (CDC)**:
  - PolypsAlign: discriminator + contrastive loss for global alignment
  - CPC-Trans: Transformer cross-attention for patch/global alignment
  - ADD: pixel-level image-to-image distillation (current strongest paired method)
  - SAMD: semantic attention distillation
- **Knowledge Distillation**: FitNets, RKD, CRD, PKT, original Hinton KD

## Rating

⭐⭐⭐⭐⭐ (5/5)

- **Novelty**: ⭐⭐⭐⭐⭐ — Pairing-free cross-modal distillation represents a significant paradigm innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Four datasets, comparisons against 8 SOTA methods, thorough ablations, and fair experimental setup.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Rigorously structured; problem motivation and method design are tightly connected.
- **Value**: ⭐⭐⭐⭐⭐ — Addresses a real clinical pain point, provides code, and requires only the WLI classifier at inference.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Momentum Memory for Knowledge Distillation in Computational Pathology](../../CVPR2026/medical_imaging/momentum_memory_for_knowledge_distillation_in_computational_pathology.md)
- [\[AAAI 2026\] Error Correction in Radiology Reports: A Knowledge Distillation-Based Multi-Stage Framework](error_correction_in_radiology_reports_a_knowledge_distillation-based_multi-stage.md)
- [\[CVPR 2026\] Diffusion-Based Feature Denoising and Using NNMF for Robust Brain Tumor Classification](../../CVPR2026/medical_imaging/diffusion-based_feature_denoising_and_using_nnmf_for_robust_brain_tumor_classifi.md)
- [\[AAAI 2026\] A Principle-Driven Adaptive Policy for Group Cognitive Stimulation Dialogue for Elderly with Cognitive Impairment](a_principle-driven_adaptive_policy_for_group_cognitive_stimu.md)
- [\[NeurIPS 2025\] FGBench: A Dataset and Benchmark for Molecular Property Reasoning at Functional Group-Level in Large Language Models](../../NeurIPS2025/medical_imaging/fgbench_a_dataset_and_benchmark_for_molecular_property_reasoning_at_functional_g.md)

</div>

<!-- RELATED:END -->
