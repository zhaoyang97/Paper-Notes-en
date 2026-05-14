---
title: >-
  [Paper Note] MultiModalPFN: Extending Prior-Data Fitted Networks for Multimodal Tabular Learning
description: >-
  [CVPR 2026][Medical Imaging][Tabular Learning] This paper proposes MMPFN, the first method to extend the pretrained tabular foundation model TabPFN to multimodal settings (tabular + image/text). By introducing a Multi-he…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "Tabular Learning"
  - "Multimodal Fusion"
  - "TabPFN"
  - "Attention Imbalance"
  - "Modality Projection"
date: 2026-05-08
content_hash: 9b56c7ba4de9a106
---

# MultiModalPFN: Extending Prior-Data Fitted Networks for Multimodal Tabular Learning

**Conference**: CVPR 2026
**arXiv**: [2602.20223](https://arxiv.org/abs/2602.20223)
**Code**: [Available](https://github.com/too-z/MultiModalPFN)
**Area**: Medical Imaging
**Keywords**: Tabular Learning, Multimodal Fusion, TabPFN, Attention Imbalance, Modality Projection

## TL;DR

This paper proposes MMPFN, the first method to extend the pretrained tabular foundation model TabPFN to multimodal settings (tabular + image/text). By introducing a Multi-head Gated MLP (MGM) and a Cross-Attention Pooler (CAP), MMPFN addresses two failure modes — over-compression of non-tabular embeddings and token-count imbalance — and achieves state-of-the-art performance on both medical and general-purpose datasets.

## Background & Motivation

### 1. State of the Field
TabPFN is a tabular foundation model that performs Bayesian inference via pretraining on synthetic tabular data, enabling prediction in a single forward pass on small-to-medium-scale datasets. However, TabPFN's pretraining is restricted to synthetic tabular data and cannot handle unstructured modalities such as images or text.

### 2. Limitations of Prior Work
(1) Real-world domains such as healthcare and marketing frequently require the fusion of structured and unstructured data (e.g., diagnostic records + medical images, sales logs + product reviews); (2) gradient-boosted trees offer limited gains when extended to heterogeneous data; (3) deep multimodal models suffer from poor performance and slow training in data-scarce scenarios.

### 3. Root Cause
TabPFN possesses strong tabular priors but cannot incorporate unstructured modalities. Naive fusion leads to two failure modes: over-compression of non-tabular embeddings (a single [CLS] token carries insufficient information) and attention imbalance caused by token-count mismatch (when non-tabular tokens greatly outnumber tabular tokens, attention is monopolized by the non-tabular modality).

### 4. Starting Point
The paper designs lightweight modality projectors that transform non-tabular embeddings into "tabular-compatible tokens," injecting multimodal information without disrupting TabPFN's tabular priors.

## Method

### Overall Architecture

MMPFN consists of three components:
1. **Per-Modality Encoders**: TabPFN v2 encoder (frozen) for tabular data, DINOv2 ViT-B/14 (frozen) for images, and ELECTRA (frozen) for text.
2. **Modality Projector**: MGM + CAP, which transform image/text [CLS] embeddings into $K$ tabular-compatible tokens of dimension $d$.
3. **TabPFN Backbone**: Receives the concatenated multimodal token sequence as a unified tabular input and performs prediction via 2D attention (feature attention + sample attention).

During training, all encoders are frozen; only the modality projector, TabPFN backbone, and decoder head are trained.

### Key Designs

#### 1. **Multi-Head Gated MLP (MGM)**

**Function**: Expands a single [CLS] embedding into $N$ tokens of dimension $d$, alleviating over-compression.

**Mechanism**: The [CLS] embedding is simultaneously fed into $N$ MLP heads, each projecting the encoder output to dimension $d$. A Gated Linear Unit (GLU) modulates each head's contribution, encouraging inter-head specialization and preserving diverse aspects of the original non-tabular representation.

$$\text{MGM}(h_{\text{CLS}}) = \{g_i \odot \text{MLP}_i(h_{\text{CLS}})\}_{i=1}^{N}$$

where $g_i = \sigma(W_g^{(i)} h_{\text{CLS}} + b_g^{(i)})$ denotes the GLU gate.

**Design Motivation**: A single [CLS] token excessively compresses image/text information; multi-head expansion captures complementary, high-resolution information, while GLU gating prevents redundant heads.

#### 2. **Cross-Attention Pooler (CAP)**

**Function**: Compresses $N$ MGM tokens into $K$ compact representations to balance the token count across modalities.

**Mechanism**: $K$ learnable query vectors are introduced; cross-attention is applied to extract information from the $N$ MGM tokens (as keys/values), with the output further refined by an MLP. The $K$ tokens are then concatenated with tabular tokens along the feature dimension and fed into TabPFN.

**Design Motivation**: Without compression, an excessive number of non-tabular tokens dominates the attention budget in TabPFN's feature attention, suppressing tabular signals.

#### 3. **Theoretical Analysis of Attention Imbalance**

The paper provides a mathematical analysis of attention imbalance. Let $N_I$ and $N_T$ denote the number of non-tabular and tabular tokens, respectively. When per-token quality is comparable ($c_I \approx c_T$), the total attention allocated to the non-tabular modality is approximately:

$$\mathbb{E}[a_I] \approx \frac{N_I c_I}{N_I c_I + N_T c_T}$$

When $N_I \gg N_T$, $a_I \to 1$, and the tabular signal is completely overwhelmed. CAP mitigates this by compressing $N_I$ to a fixed $K$ (e.g., 4 or 24).

### Loss & Training

- **Loss**: Cross-entropy loss
- **Optimizer**: AdamW, $\text{lr}=1 \times 10^{-5}$, batch size = 1
- **Training**: Fine-tuned for 100 iterations
- **Freezing Strategy**: All modality encoders (TabPFN v2, DINOv2, ELECTRA) are frozen; only MGM, CAP, the TabPFN backbone, and decoder head are trained
- **Inference**: Follows TabPFN's in-context inference protocol

## Key Experimental Results

### Main Results

**Table 2: Image-Tabular Datasets (Accuracy)**

| Method | PU20 | Mass | Calc | PetFinder | Avg. Rank |
|--------|------|------|------|-----------|-----------|
| TabPFN | 82.17 | 71.27 | 73.31 | 36.33 | 4.25 |
| CatBoost | 80.43 | **78.31** | 72.09 | 38.69 | 3.25 |
| AutoGluon | 81.09 | 76.28 | 71.04 | 38.81 | 3.50 |
| MMCL | 76.61 | 57.62 | 60.12 | 36.61 | 7.25 |
| TIP | 78.75 | 73.12 | 67.96 | 37.28 | 5.50 |
| HEALNet | 74.65 | 68.10 | 71.83 | 37.03 | 6.25 |
| **MMPFN** | **85.22** | 74.53 | **75.40** | **40.74** | **1.50** |

**Table 3: Text-Tabular Datasets (Accuracy)**

| Method | Airbnb | Salary | Cloth | PetFinder | Avg. Rank |
|--------|--------|--------|-------|-----------|-----------|
| TabPFN | 46.96 | 44.96 | 55.07 | 36.33 | 5.50 |
| AutoGluon | 44.60 | 45.24 | **72.07** | 37.96 | 3.00 |
| TTT | 38.3 | **47.2** | 65.5 | 38.9 | 3.00 |
| **MMPFN** | **47.78** | 46.17 | 66.26 | **39.04** | **1.75** |

MMPFN achieves an average rank of 1.50 on image-tabular benchmarks and 1.75 on text-tabular benchmarks, outperforming all baselines. The largest improvement is observed on the medical dataset PU20 (+3.05% vs. TabPFN).

### Ablation Study

| Configuration | Avg. Accuracy | Note |
|---------------|---------------|------|
| Single-head Linear | 53.86 | Simplest projection; over-compression |
| Single-head MLP | 54.14 | Marginally better than linear |
| Multi-head MLP | 55.81 | Multi-head expansion is effective |
| Multi-head MoE | 53.23 | Sparse routing unstable on small data |
| **Multi-head MGM** | **57.37** | GLU gating achieves best performance |

### Key Findings

1. **Attention imbalance is empirically confirmed**: With non-tabular input only, increasing the number of MGM heads consistently improves performance; however, when mixing tabular and non-tabular inputs, adding more non-tabular tokens actually degrades performance (non-monotonic behavior), corroborating the theoretical analysis.
2. **MGM and CAP are synergistic**: MGM first richly extracts features, then CAP compresses them to a fixed token count; their combination eliminates the non-monotonic degradation.
3. **Cross-modal potential of TabPFN**: When only image/text inputs are used (no tabular data), MMPFN still matches DINOv2+MLP, indicating that TabPFN's synthetic tabular prior transfers to non-tabular features.
4. **Low-data robustness**: MMPFN outperforms TIP (which leverages all unlabeled data for self-supervised pretraining) even with only 10% of the data, demonstrating the advantage of PFN priors in few-shot settings.
5. **More modalities consistently help**: On PetFinder, accuracy increases monotonically as modalities are added: T → T+t → T+I → T+I+t.

## Highlights & Insights

1. **First work to extend TabPFN to multimodal settings**: The conceptual novelty lies in "tabularizing" unstructured data to leverage powerful tabular priors.
2. **Theoretical analysis of attention imbalance**: The paper derives a mathematical formulation of softmax attention behavior under token-count mismatch, providing a principled motivation for CAP.
3. **Extremely low training cost**: Only 100 iterations of fine-tuning with batch size 1, and all encoders frozen — well-suited for resource-constrained medical settings.
4. **Cross-modal embedding correlation analysis**: Cosine similarity visualizations confirm that MMPFN learns cross-modal interactions rather than merely intra-modal structures.

## Limitations & Future Work

1. **Limited medical evaluation beyond PU20**: Results on the two CBIS-DDSM subsets are inconsistent (Mass ranks 3rd, Calc ranks 1st), raising concerns about stability.
2. **Classification only**: Regression, survival analysis, and other common medical tasks are not evaluated.
3. **Non-tabular encoders remain frozen**: End-to-end fine-tuning of DINOv2/ELECTRA is not explored.
4. **Manual tuning of CAP's $K$**: $K = 24$ for PU20 and $K = 4$ for Cloth; dataset-specific hyperparameter selection is required.
5. **No comparison with large multimodal models**: Baselines such as GPT-4V and LLaVA in multimodal medical settings are absent.

## Related Work & Insights

- **Prior transfer from TabPFN**: The "tabular distributional prior" learned by TabPFN on synthetic data can be ingeniously repurposed for non-tabular features — the paradigm of "tabularizing everything" is conceptually inspiring.
- **Attention imbalance**: This issue is not unique to multimodal TabPFN; it also arises broadly in VLMs when long text sequences are paired with short image token sequences.
- **Connection between MGM and Q-Former**: CAP's cross-attention pooling is conceptually analogous to the Q-Former in BLIP-2, but considerably more lightweight.

## Rating

- Novelty: ⭐⭐⭐⭐ — First extension of TabPFN to multimodal settings; MGM+CAP design is theoretically grounded.
- Experimental Thoroughness: ⭐⭐⭐ — Six datasets spanning medical and general domains, but medical evaluation lacks depth.
- Writing Quality: ⭐⭐⭐⭐ — Theoretical analysis is clear; experimental presentation is well-organized.
- Value: ⭐⭐⭐⭐ — Opens a new direction for multimodal extension of tabular foundation models; highly practical for small-data medical scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Extending ZACH-ViT to Robust Medical Imaging: Corruption and Adversarial Stress Testing in Low-Data Regimes](extending_zach-vit_to_robust_medical_imaging_corruption_and_adversarial_stress_t.md)
- [\[AAAI 2026\] ConSurv: Multimodal Continual Learning for Survival Analysis](../../AAAI2026/medical_imaging/consurv_multimodal_continual_learning_for_survival_analysis.md)
- [\[CVPR 2026\] Multimodal Classification of Radiation-Induced Contrast Enhancements and Tumor Recurrence Using Deep Learning](multimodal_classification_of_radiation-induced_contrast_enhancements_and_tumor_r.md)
- [\[CVPR 2026\] PGR-Net: Prior-Guided ROI Reasoning Network for Brain Tumor MRI Segmentation](pgr-net_prior-guided_roi_reasoning_network_for_brain_tumor_mri_segmentation.md)
- [\[CVPR 2026\] GLEAM: A Multimodal Imaging Dataset and HAMM for Glaucoma Classification](gleam_a_multimodal_imaging_dataset_and_hamm_for_gl.md)

</div>

<!-- RELATED:END -->
