---
title: >-
  [Paper Note] GENIUS: A Generative Framework for Universal Multimodal Search
description: >-
  [CVPR 2025][Multimodal VLM][Generative Retrieval] The first universal generative multimodal retrieval framework, which encodes multimodal data into discrete IDs through modal-decoupled semantic quantization and utilizes an autoregressive decoder to directly generate target IDs from queries. It outperforms preceding generative methods by over 25 points on Flickr30K Text-to-Image retrieval, while reducing storage overhead by 99% compared to CLIP.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Generative Retrieval"
  - "Multimodal Search"
  - "Modal-Decoupled Quantization"
  - "Residual Quantization"
  - "Universal Retrieval"
date: 2026-05-08
content_hash: 85502650f445d8d0
---

# GENIUS: A Generative Framework for Universal Multimodal Search

**Conference**: CVPR 2025  
**arXiv**: [2503.19868](https://arxiv.org/abs/2503.19868)  
**Code**: None  
**Area**: Information Retrieval  
**Keywords**: Generative Retrieval, Multimodal Search, Modal-Decoupled Quantization, Residual Quantization, Universal Retrieval

## TL;DR
The first universal generative multimodal retrieval framework, which encodes multimodal data into discrete IDs through modal-decoupled semantic quantization and utilizes an autoregressive decoder to directly generate target IDs from queries. It outperforms preceding generative methods by over 25 points on Flickr30K Text-to-Image retrieval, while reducing storage overhead by 99% compared to CLIP.

## Background & Motivation

**Background**: Multimodal retrieval (such as text-to-image, image-to-text, and image-text-to-image) typically employs the "encode-and-search" paradigm, where all candidates are first encoded into vectors followed by nearest neighbor retrieval. In contrast, generative retrieval directly generates the discrete identifiers (IDs) of target items from queries, avoiding expensive similarity searches.

**Limitations of Prior Work**: Existing generative retrieval methods (e.g., GRACE, IRGen) only support single tasks (such as text-to-image) and cannot handle unified multimodal retrieval where queries and candidates of different modalities are mixed together. Furthermore, quantized discrete IDs lose modal information, leading to degraded cross-modal retrieval performance.

**Key Challenge**: A significant gap still remains between the efficiency advantages of generative retrieval (where retrieval complexity is independent of database size) and its retrieval accuracy, particularly in universal retrieval scenarios that require handling various modality combinations simultaneously.

**Goal**: To design a unified generative retrieval framework capable of supporting any combination of queries and retrieval among three modalities (text, image, and image-text pairs) while significantly boosting the accuracy of generative retrieval.

**Key Insight**: To introduce modal codes (three codes representing image, text, and image-text respectively) in the first layer of Residual Quantization (RQ), while subsequent layers encode semantic information in a coarse-to-fine manner, achieving hierarchical ID generation that "determines the modality first, then the semantics."

**Core Idea**: Explicitly decoupling "modality identification" and "semantic encoding" during the quantization stage, allowing the autoregressive decoder to determine the target modality first before progressively refining the semantics, thereby achieving universal multimodal retrieval.

## Method

### Overall Architecture
Three-stage training: (1) Pre-trained CLIP encoders align query-target embeddings; (2) Modal-decoupled semantic quantization—using contrastive learning and residual quantization to convert multimodal data into discrete IDs (1st code = modality, subsequent 8 codes = semantics); (3) Training a T5-small autoregressive decoder to generate target IDs from query embeddings. During inference, Trie-constrained beam search is used to generate valid IDs.

### Key Designs

1. **Modal-Decoupled Semantic Quantization**:

    - **Function**: Converts multimodal data into structured discrete IDs, where the first code encodes the modality ownership.
    - **Mechanism**: The codebook size of the first layer is $K_1=3$ (for three modalities: image, text, and image-text), and the subsequent 8 layers have a codebook size of $K=4096$, encoding progressively finer semantic information layer-by-layer via residual quantization. During training, the contrastive loss $\mathcal{L}_{cl}$ (aligning query and target embeddings), quantization loss $\mathcal{L}_{rq}$, and reconstruction loss $\mathcal{L}_{mse}$ are optimized simultaneously.
    - **Design Motivation**: Ablation studies show that without modal decoupling, COCO T→I performance plummets from 55.4 to 20.2 (-63.5%), because the decoder cannot search effectively without knowing the target modality. Contrastive loss is especially critical—removing it drops performance to zero across all tasks.

2. **Fusion Module**:

    - **Function**: Fuses the two embeddings of image-text pairs into a unified representation.
    - **Mechanism**: $h(x,y) = \lambda \cdot x + (1-\lambda) \cdot y + \text{MLP}([x;y])$, where $\lambda$ is dynamically predicted by another MLP + sigmoid. This allows the model to adaptively determine the relative weights of image and text features.
    - **Design Motivation**: Simple element-wise addition or concatenation fails to capture interaction relationships between image and text. The dynamic weight $\lambda$ allows different samples to have different modal preferences.

3. **Query Augmentation**:

    - **Function**: Increases training data diversity and improves generalization capability.
    - **Mechanism**: Interpolates between query embedding $\mathbf{z}_q$ and target embedding $\mathbf{z}_c$: $\mathbf{z}'_q = \mu \cdot \mathbf{z}_q + (1-\mu) \cdot \mathbf{z}_c$, where $\mu \sim \text{Beta}(2,2)$. This is equivalent to "moving towards the target" in the embedding space to generate new queries.
    - **Design Motivation**: Ablation shows that removing augmentation drops performance on the CIRR task (composed image-text-to-image retrieval) from 20.5 to 11.7 (-43%), indicating that augmentation is highly effective for complex relational reasoning tasks.

### Loss & Training
Quantization stage: $\mathcal{L} = \mathcal{L}_{cl} + 100 \cdot \mathcal{L}_{rq} + 100 \cdot \mathcal{L}_{mse}$. Decoder stage: standard cross-entropy. Inference uses Trie-constrained beam search (beam=50). Optional embedding re-ranking performs nearest-neighbor search on the beam candidates, which incurs minimal overhead but yields significant returns.

## Key Experimental Results

### Main Results

| Method | Flickr30K T→I R@1 | COCO T→I R@1 | COCO I→T R@1 | Type |
|------|-------------------|-------------|-------------|------|
| GRACE | 37.4 | 16.7 | - | Generative |
| IRGen | 49.0 | 29.6 | - | Generative |
| **GENIUS** | **60.6** | **40.1** | **83.2** | Generative |
| **GENIUS+R** | **74.1** | **46.1** | **91.1** | Generative + Re-ranking |
| CLIP-SF | - | 81.1 (R@5) | 92.3 (R@5) | Vector Retrieval |

### Ablation Study

| Configuration | COCO T→I | COCO I→T | Description |
|------|---------|---------|------|
| Full GENIUS | 55.4 | 82.7 | Baseline |
| W/o Modal Decoupling | 20.2 | 73.2 | Cross-modal retrieval collapses |
| W/o Query Augmentation | 47.8 | 67.7 | Degraded generalization capability |
| W/o Contrastive Loss | **0.0** | **0.1** | Complete failure |
| W/o MSE Loss | 45.5 | 83.1 | Minor impact |

### Key Findings
- **Contrastive loss is crucial**: Performance drops to zero when removed; UMAP visualization shows that query-target features are completely misaligned without contrastive loss.
- **Modal decoupling codes are vital**: Allowing the decoder to determine the target modality in the very first step significantly reduces the search space (55.4 vs. 20.2).
- **Clear efficiency advantages**: Retrieval speed remains nearly constant regardless of database growth (O(M) vs O(N)), and storage requires only ~12 bytes per data point (compared to ~3KB for CLIP).
- **Gap still remains compared to vector retrieval**: On COCO T→I, GENIUS+R (46.1 R@1) still lags behind CLIP-SF (~55 R@1), indicating that the accuracy ceiling of generative methods still needs to be breached.

## Highlights & Insights
- **The "modality first, semantics second" quantization design** is highly elegant. Embedding structured priors into the first dimension of discrete IDs significantly simplifies the decoder's search space.
- **99% storage compression** (12 bytes vs. 3KB) has tremendous practical value for large-scale retrieval systems, particularly for mobile and edge deployments.
- **Query augmentation via Beta interpolation** performs data augmentation within the embedding space, which is more universal than traditional text/image augmentations.

## Limitations & Future Work
- A clear gap in accuracy still exists compared to vector retrieval methods, particularly in knowledge-intensive tasks (e.g., WebQA, InfoSeek).
- There is a trade-off between speed and accuracy depending on the beam size of the beam search (beam=1: 24.2 R@5 vs. beam=50: 68.2 R@5).
- The quantization process inevitably loses information; the optimal choices for the number of residual quantization layers and codebook sizes remain dataset-dependent.
- Only evaluated on M-BEIR; the approach has not been validated on more diverse retrieval scenarios (such as video retrieval and cross-lingual retrieval).

## Related Work & Insights
- **vs. GRACE / IRGen**: Single-task generative retrieval methods. GENIUS is the first to extend this to universal multimodal scenarios, outperforming GRACE by 23+ points on COCO T→I.
- **vs. CLIP-SF / BLIP-FF**: Vector retrieval methods achieve higher accuracy but require storing entire embeddings and executing nearest-neighbor search. The re-ranked version of GENIUS approaches or even surpasses them on certain tasks.
- **vs. DSI / NCI**: Pioneers of generative retrieval in the text domain. GENIUS is the first to extend their ideas to multimodal scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The first universal multimodal generative retrieval framework, where modal-decoupled quantization is the key innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Full-task evaluation on M-BEIR with clear ablations, but lacks comparison with a broader range of vector retrieval methods.
- Writing Quality: ⭐⭐⭐⭐ The framework description is clear, and the three-stage training logic is coherent.
- Value: ⭐⭐⭐⭐ Offers important insights for large-scale multimodal retrieval systems, though the accuracy gap limits immediate deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] CART: A Generative Cross-Modal Retrieval Framework with Coarse-To-Fine Semantic Modeling](../../ACL2025/multimodal_vlm/cart_a_generative_cross-modal_retrieval_framework_with_coarse-to-fine_semantic_m.md)
- [\[ICML 2025\] Universal Retrieval for Multimodal Trajectory Modeling](../../ICML2025/multimodal_vlm/universal_retrieval_for_multimodal_trajectory_modeling.md)
- [\[CVPR 2025\] Improving Personalized Search with Regularized Low-Rank Parameter Updates](improving_personalized_search_with_regularized_low-rank_parameter_updates.md)
- [\[CVPR 2025\] Global-Local Tree Search in VLMs for 3D Indoor Scene Generation](global-local_tree_search_in_vlms_for_3d_indoor_scene_generation.md)
- [\[NeurIPS 2025\] Generalized Contrastive Learning for Universal Multimodal Retrieval](../../NeurIPS2025/multimodal_vlm/generalized_contrastive_learning_for_universal_multimodal_re.md)

</div>

<!-- RELATED:END -->
