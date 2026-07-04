---
title: >-
  [Paper Note] Unsupervised Foundation Model-Agnostic Slide-Level Representation Learning
description: >-
  [CVPR 2025][Computational Biology][Whole Slide Image representation learning] This work proposes Cobra, an unsupervised foundation model-agnostic (FM-agnostic) whole slide image (WSI)-level representation learning framework. It leverages embeddings from multiple pre-trained patch-level foundation models as feature-space augmentations, training a slide-level encoder using a Mamba-2 encoder and contrastive learning. Pre-trained on only 3,048 WSIs…
tags:
  - "CVPR 2025"
  - "Computational Biology"
  - "Whole Slide Image representation learning"
  - "self-supervised learning"
  - "contrastive learning"
  - "foundation model-agnostic"
  - "Mamba-2"
date: 2026-05-08
content_hash: 82b7d9fe6837c601
---

# Unsupervised Foundation Model-Agnostic Slide-Level Representation Learning

**Conference**: CVPR 2025  
**arXiv**: [2411.13623](https://arxiv.org/abs/2411.13623)  
**Code**: [github](https://github.com/KatherLab/COBRA)  
**Area**: Medical Imaging  
**Keywords**: Whole Slide Image representation learning, self-supervised learning, contrastive learning, foundation model-agnostic, Mamba-2

## TL;DR

This work proposes Cobra, an unsupervised foundation model-agnostic (FM-agnostic) whole slide image (WSI)-level representation learning framework. It leverages embeddings from multiple pre-trained patch-level foundation models as feature-space augmentations, training a slide-level encoder using a Mamba-2 encoder and contrastive learning. Pre-trained on only 3,048 WSIs, Cobra outperforms existing slide encoders by at least +4.4% in average AUC across 15 downstream tasks.

## Background & Motivation

In computational pathology, whole slide images (WSIs) can reach up to $150,000 \times 150,000$ pixels, making them too large to be directly processed by ViT architectures. The dominant workflow crops WSIs into small patches, extracts patch embeddings using pre-trained foundation models (FMs), and then aggregates them into slide-level predictions via MIL. However, MIL is **supervised and task-specific**.

Unsupervised slide representation learning attempts to generate task-agnostic slide embeddings, but faces a core challenge: **how to generate effective data augmentations for slide-level learning?** Traditional image augmentations are largely ineffective on modern FMs (as FMs are already invariant to these transformations), while multi-staining or multi-modal approaches are constrained by data availability.

The key insight of Cobra: **different patch-level FMs inherently constitute feature-space augmentations**. Encoding the same slide with different FMs produces different yet semantically consistent patch embedding sequences. Combined with embeddings at different magnifications, this allows direct contrastive learning in the feature space without requiring any pixel-level augmentations.

## Method

### Overall Architecture

During the preprocessing stage, Cobra extracts patch embeddings using four pre-trained FMs (CTransPath, UNI, Virchow2, H-Optimus-0) at three magnifications (0.5, 1.14, 2 MPP). The slide encoder consists of three components: (1) an Embedding MLP that maps FM embeddings of different dimensions into a shared space; (2) a two-layer Mamba-2 (SSD) that encodes patch sequences; and (3) a Multi-head Gated Attention layer that aggregates them into a single slide vector. The model is trained using a MoCo-style contrastive loss.

### Key Designs

**1. Feature Space Augmentation via Multiple FMs — Multi-FM Feature Space Augmentation**

- **Function**: Generates positive pairs for contrastive learning without pixel-level augmentations.
- **Mechanism**: A patient's WSI is encoded into patch embeddings using different FMs ($fe_n \in \{CTP, UNI, V2, H0\}$) and different magnifications, which serve as different "views" of the same slide. The query $q$ and positive key $k^+$ are drawn from different FM/magnification embeddings of the same patient.
- **Design Motivation**: Different FMs feature distinct architectures, pre-training data, and training objectives, thereby capturing complementary morphological characteristics. Various magnifications provide multi-scale context. This feature-space (rather than pixel-space) augmentation is immune to the invariance of pre-trained FMs, offering a more effective positive pair generation strategy for contrastive learning than traditional augmentations.

**2. Mamba-2 + Multi-head Gated Attention Architecture**

- **Function**: Efficiently encodes long sequences of patch embeddings and aggregates them into slide-level vectors.
- **Mechanism**: The architecture is formulated as $z = f_A(f_S(f_E(H^{fe_n})))$. The embedding module $f_E$ uses an MLP with SiLU to map different embedding dimensions to a shared $d$-dimensional space; the State Space module $f_S$ utilizes two Mamba-2 SSD layers with residual connections to encode the sequence; the aggregation module $f_A$ employs an $M$-head gated attention mechanism to compute a weighted average $z = \sum_k a_k \cdot H_{S,k}$, where $a_k$ is computed via a tanh-sigmoid gating mechanism.
- **Design Motivation**: Mamba-2 is more efficient than Transformer architectures over long sequences, making it suitable for handling the thousands or tens of thousands of patches in a WSI. Gated attention aggregation focuses more on diagnostically relevant regions compared to simple average pooling.

**3. Flexible Inference Modes (Single-FM / Multi-FM / Unseen-FM)**

- **Function**: Enables inference using any FM, whether seen or unseen during training.
- **Mechanism**: In Single-FM mode, encoded embeddings $H_S$ are used to compute attention weights, but the weighted average is performed over the original patch embeddings $H^{fe_n}$ (Eq. 6). In Multi-FM mode, encoded embeddings from multiple FMs are averaged and then processed (Eq. 8). In Unseen-FM mode, FMs unseen during training are still mapped to the shared space via the embedding module.
- **Design Motivation**: The mapping learned by the embedding module generalizes well, enabling Cobra to transform even **unseen new FMs** into superior slide-level feature extractors. This is highly valuable given the continuous emergence of new FMs.

### Loss & Training

- **Loss Function**: InfoNCE contrastive loss $\mathcal{L}_q = -\log \frac{\psi(q, k^+)}{\sum_i \psi(q, k_i)}$, where $\psi(x_1, x_2) = \exp(\text{sim}(x_1, x_2)/\tau)$.
- **MoCo-style Training**: The key encoder is updated via momentum: $\theta_k \leftarrow m\theta_k + (1-m)\theta_q$.
- **Pre-training Data**: Only 3,048 WSIs from TCGA (across 4 tissue types), which is significantly fewer than GigaPath's 171K or PRISM's 587K.
- **Model Scale**: Only 15M parameters.

## Key Experimental Results

### Main Results

15 downstream classification tasks (TCGA training, CPTAC external validation), average AUC:

| Slide Encoder | Pre-training Data | Parameters | Average AUC |
|---------------|-----------|--------|----------|
| Mean CTransPath | - | - | 62.1 |
| Mean Virchow2 | - | - | 73.8 |
| GigaPath-SE | 171K WSI | 86M | 71.5 |
| CHIEF | 60K WSI | 1M | - |
| MADELEINE | 21K WSI | 5M | - |
| **Cobra (V2)** | **3K WSI** | **15M** | **78.2** |

### Ablation Study

Contribution of each Cobra component (comparison of inference modes):

| Inference Mode | Average AUC | Description |
|----------|----------|------|
| Mean patch embedding (V2) | 73.8 | No slide encoder |
| Cobra Single-FM (V2) | **78.2** | Uses original embeddings for weighted average |
| Cobra Multi-FM (4 FMs) | 77.5 | Fuses all training FMs |
| Cobra + unseen FM | Shows improvement | Also effective on unseen FMs |

### Key Findings

1. **Extreme Data Efficiency**: Pre-training on only 3K WSIs outperforms GigaPath-SE, which was trained on 171K WSIs (Average AUC 78.2 vs 71.5).
2. **FM Agnosticism**: Cobra can transform even unseen FMs (such as newly released FMs) into better slide encoders.
3. **Single-FM Inference Outperforms Multi-FM**: Performing a weighted average on the original patch embeddings (Eq. 6) outperforms using the encoded embeddings (Eq. 4), as it preserves FM-specific information.
4. **Works Well at Low Magnification**: Offers a strong trade-off between computational efficiency and overall performance.

## Highlights & Insights

1. **The concept of "FMs as augmentations" is elegant and simple**: It bypasses the hurdle of designing manual augmentation strategies in traditional SSL.
2. **Achieving SOTA with only 3K WSIs + 15M parameters**: The data efficiency is remarkable, making it highly accessible to resource-constrained institutions.
3. **Generalization capability on unseen FMs**: This gives Cobra forward-looking value, eliminating the need to retrain the model whenever new FMs emerge.

## Limitations & Future Work

1. The mapping quality of the embedding module for unseen FMs relies on dimension matching and feature-space similarity.
2. Pre-trained on only 4 tissue types, and its generalization to rare tumor types has yet to be verified.
3. Mamba-2's sequence modeling assumes a fixed order of patches, which might not be optimal.
4. Future work could explore more FM combinations and adaptive FM selection strategies.

## Related Work & Insights

- **GigaPath**: Trains a slide encoder with a masked autoencoder, requiring 171K WSIs. Cobra is more efficient, utilizing contrastive learning and multi-FM augmentations.
- **MADELEINE**: Uses multi-modal (H&E + IHC) contrastive learning, which is restricted by data availability. Cobra requires only a single modality.
- **PRISM**: Trained on 587K WSIs + multi-modal data including text and genomics. Cobra outperforms it with single-modality and very limited data.
- **MambaMIL**: Integrates Mamba into MIL, inspiring Cobra's architectural design.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The idea of using multiple FMs as feature-space augmentations is highly original and elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across 15 downstream tasks, external validation, multiple baselines, and ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear method descriptions and rigorous mathematical formulations.
- **Value**: ⭐⭐⭐⭐⭐ — Highly significant for computational pathology; its exceptional data efficiency and FM agnosticism offer high practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CARE: A Molecular-Guided Foundation Model with Adaptive Region Modeling for Whole Slide Image Analysis](../../CVPR2026/computational_biology/care_a_molecular-guided_foundation_model_with_adaptive_region_modeling_for_whole.md)
- [\[ICML 2026\] Learning the Interaction Prior for Protein-Protein Interaction Prediction: A Model-Agnostic Approach](../../ICML2026/computational_biology/learning_the_interaction_prior_for_protein-protein_interaction_prediction_a_mode.md)
- [\[ICML 2025\] SToFM: a Multi-scale Foundation Model for Spatial Transcriptomics](../../ICML2025/computational_biology/stofm_a_multi-scale_foundation_model_for_spatial_transcriptomics.md)
- [\[ICML 2025\] Global Context-aware Representation Learning for Spatially Resolved Transcriptomics](../../ICML2025/computational_biology/global_context-aware_representation_learning_for_spatially_resolved_transcriptom.md)
- [\[ICLR 2026\] Learning Brain Representation with Hierarchical Visual Embeddings](../../ICLR2026/computational_biology/learning_brain_representation_with_hierarchical_visual_embeddings.md)

</div>

<!-- RELATED:END -->
