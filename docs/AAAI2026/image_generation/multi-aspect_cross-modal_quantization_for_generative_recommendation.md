---
title: >-
  [Paper Note] Multi-Aspect Cross-modal Quantization for Generative Recommendation
description: >-
  [AAAI 2026][Image Generation][Generative Recommendation] This paper proposes MACRec, which introduces multi-aspect cross-modal interaction at both the semantic ID learning stage and the generative model training stage. Through cross-modal quantization (contrastive learning-enhanced residual quantization) and multi-aspect alignment (implicit + explicit), MACRec significantly improves recommendation performance while reducing ID collision rates.
tags:
  - AAAI 2026
  - Image Generation
  - Generative Recommendation
  - Cross-modal Quantization
  - Residual Quantization
  - Contrastive Learning
  - Semantic ID
date: 2026-05-08
content_hash: 013e765a723b4ad3
---

# Multi-Aspect Cross-modal Quantization for Generative Recommendation

**Conference**: AAAI 2026
**arXiv**: [2511.15122](https://arxiv.org/abs/2511.15122)
**Code**: [github.com/zhangfw123/MACRec](https://github.com/zhangfw123/MACRec)
**Area**: Image Generation / Recommender Systems
**Keywords**: Generative Recommendation, Cross-modal Quantization, Residual Quantization, Contrastive Learning, Semantic ID

## TL;DR

This paper proposes MACRec, which introduces multi-aspect cross-modal interaction at both the semantic ID learning stage and the generative model training stage. Through cross-modal quantization (contrastive learning-enhanced residual quantization) and multi-aspect alignment (implicit + explicit), MACRec significantly improves recommendation performance while reducing ID collision rates.

## Background & Motivation

### State of the Field

Generative Recommendation (GR) is an emerging recommendation paradigm that reformulates the recommendation task as a next-token prediction problem. The pipeline proceeds as follows: (1) item embeddings are discretized into **Semantic ID** sequences via Residual Quantization (RQ-VAE); (2) user interaction histories are represented as sequences of Semantic IDs; (3) a sequence generation model (e.g., T5) predicts the Semantic ID of the next item. Representative works include TIGER, LC-Rec, and LETTER.

### Limitations of Prior Work

**Insufficient unimodal information**: Existing GR methods primarily use text embeddings to construct Semantic IDs, but **a single modality offers limited semantic discriminability**. For instance, different instruments from the same brand may be close in the text embedding space (dominated by brand information), making it difficult to distinguish between distinct products.

**Hierarchical semantic loss during quantization**: RQ-VAE suffers from significant semantic loss at deeper quantization layers, causing the model to lack clear semantic guidance when assigning tokens, leading to near-random ID assignments.

**Absence of cross-modal interaction**: Existing multimodal GR methods (e.g., MQL4GRec) encode each modality independently to obtain Semantic IDs, without considering cross-modal interaction during quantization, thus failing to exploit inter-modal complementarity.

### Root Cause

How to effectively leverage the complementarity of multimodal information during both the **learning stage** and the **utilization stage** (generative model training) of Semantic IDs, so as to construct high-quality Semantic IDs with clear hierarchical semantics, low collision rates, and effective generative model training?

### Core Idea

**Introduce cross-modal interaction at two stages**: (1) **Quantization stage**: Contrastive learning is applied to enhance residual representations at each RQ-VAE layer, using visual pseudo-labels to optimize text residuals and text pseudo-labels to optimize visual residuals, thereby reducing per-layer semantic loss and lowering ID collision rates. (2) **Generative model training stage**: Implicit alignment (latent-space contrastive learning) and explicit alignment (cross-modal generation tasks) are employed to help the model learn shared features across modality-specific Semantic IDs.

## Method

### Overall Architecture

MACRec consists of two primary modules:
1. **Cross-modal Item Quantization**: Generates high-quality multimodal Semantic IDs.
2. **Generative Recommendation with Multi-aspect Alignment**: Trains the GR model using alignment strategies.

### Key Designs

#### 1. **Dual-modality Pseudo-label Generation**

**Function**: Constructs positive sample pairs for cross-modal contrastive learning.

**Mechanism**: K-means clustering ($K$=512) is applied separately to text embeddings $\{\mathbf{t}_i\}$ and visual embeddings $\{\mathbf{v}_i\}$ of all items, yielding text cluster labels $\mathcal{C}_{text}$ and visual cluster labels $\mathcal{C}_{vision}$.

**Design Motivation**: Clustering results from different modalities reflect different aspects of item similarity—text clusters by brand/description, while vision clusters by appearance/shape. Using the clustering labels of one modality as positive-sample guidance for the other enables complementary information injection.

#### 2. **Cross-modal Quantization with Contrastive Learning**

**Function**: Introduces cross-modal contrastive learning at each quantization layer of RQ-VAE to enhance the discriminability of residual representations.

**Mechanism**:

Residual quantization is performed over $L$ layers separately for text and visual embeddings:
$$c_l^t = \arg\min_k \|\mathbf{r}_l^t - \mathbf{e}_{l,k}^t\|_2, \quad \mathbf{r}_{l+1}^t = \mathbf{r}_l^t - \mathbf{e}_{l,c_k^t}^t$$

A cross-modal contrastive loss is applied to the residual representation at each layer—**visual pseudo-labels** are used to construct positives for text residuals, and **text pseudo-labels** for visual residuals:

$$\mathcal{L}_{con}^{l,v\rightarrow t} = -\frac{1}{B}\sum_{i=1}^{B}\log\frac{\exp(\langle\mathbf{r}_i^t, \mathbf{r}_{i,pos}^t\rangle/\tau)}{\sum_{j=1}^{B}\exp(\langle\mathbf{r}_i^t, \mathbf{r}_j^t\rangle/\tau)}$$

**Layer-onset strategy**: The contrastive loss is applied starting from layer 3 ($\lambda_{con}^{0,1}=0$, $\lambda_{con}^{2,3}=0.1$), allowing the first two layers to preserve modality-specific information.

**Design Motivation**: (1) Independent quantization of text and vision leads to codebook collapse (similar embeddings mapped to the same codeword) and low utilization; (2) Cross-modal contrastive learning enables each residual layer to capture complementary features from different modalities, reducing hierarchical semantic loss.

#### 3. **Cross-modal Reconstruction Alignment**

**Function**: Applies cross-modal alignment constraints on quantized representations to further optimize the codebook.

**Mechanism**: The quantized representation is obtained by summing codewords across layers: $\hat{\mathbf{z}}^t = \sum_{l=0}^{L-1}\mathbf{e}_{l,c_k^t}^t$. Bidirectional contrastive alignment is then applied between text and visual quantized representations of the same item:

$$\mathcal{L}_{align}^{t\rightarrow v} = -\frac{1}{B}\sum_{i=1}^{B}\log\frac{\exp(\langle\hat{\mathbf{z}}_i^t, \hat{\mathbf{z}}_i^v\rangle/\tau)}{\sum_{j=1}^{B}\exp(\langle\hat{\mathbf{z}}_i^t, \hat{\mathbf{z}}_j^v\rangle/\tau)}$$

**Design Motivation**: Ensures that the quantized representations of the same item remain consistent across modalities in the latent space, balancing codebook utilization.

#### 4. **Implicit Alignment**

**Function**: Aligns text and visual Semantic ID representations of the same item in the latent space of the GR model.

**Mechanism**: Text and visual Semantic IDs are separately encoded by the T5 Encoder followed by mean pooling, and then aligned via bidirectional InfoNCE loss:

$$\mathcal{L}_{implicit} = \mathcal{L}_{implicit}^{t\rightarrow v} + \mathcal{L}_{implicit}^{v\rightarrow t}$$

**Design Motivation**: Enables the GR model to recognize shared properties across modality-specific Semantic IDs, providing a stronger feature foundation for subsequent prediction.

#### 5. **Explicit Alignment**

**Function**: Performs alignment in the output space through cross-modal generation tasks.

**Mechanism**: Two types of cross-modal generation tasks are designed:
- **Item-level alignment**: Text ID → Visual ID, and Visual ID → Text ID.
- **Sequence-level alignment**: Text ID sequence → next item's Visual ID, and Visual ID sequence → next item's Text ID.

These auxiliary tasks are jointly trained with the standard sequential recommendation task.

**Design Motivation**: While implicit alignment operates only on the encoder side, explicit alignment further reinforces cross-modal associations on the decoder side, allowing the model to learn shared inter-modal features from multiple perspectives.

### Loss & Training

**Semantic ID learning stage**:
$$\mathcal{L}_{ID} = \mathcal{L}_{RQ-VAE} + \lambda_{con}^l \sum_{l=0}^{L-1}\mathcal{L}_{con}^l + \lambda_{align}\mathcal{L}_{align}$$

where $\mathcal{L}_{RQ-VAE}$ includes reconstruction loss and quantization loss.

**GR model training stage**:
$$\mathcal{L}_{rec} = -\sum_{t=1}^{|y|}\log P_\theta(y_t | y_{<t}, x) + \lambda_{implicit}\mathcal{L}_{implicit}$$

During inference, constrained beam search is used to generate candidate Semantic IDs, and the average scores of both modalities are taken as the final result.

Hyperparameter settings: codebook size $M$=256, 4 quantization layers, AdamW optimizer, batch size 1024, learning rate 0.001, $\lambda_{align}=0.001$, $\lambda_{implicit}=0.01$, $\tau=0.1$.

## Key Experimental Results

### Main Results

Recommendation performance comparison on three Amazon datasets:

| Dataset | Metric | TIGER | MQL4GRec | **MACRec** | Gain |
|--------|------|-------|----------|---------|------|
| Instruments | HR@1 | 0.0754 | 0.0763 | **0.0819** | +7.3% |
| Instruments | NDCG@10 | 0.0950 | 0.0997 | **0.1046** | +4.9% |
| Arts | HR@1 | 0.0532 | 0.0626 | **0.0685** | +9.4% |
| Arts | NDCG@10 | 0.0806 | 0.0898 | **0.0953** | +6.1% |
| Games | HR@10 | 0.0857 | 0.1007 | **0.1078** | +7.1% |
| Games | NDCG@10 | 0.0453 | 0.0538 | **0.0565** | +5.0% |

MACRec achieves the best results on all metrics across all three datasets (p < 0.05), statistically significantly outperforming the strongest baseline MQL4GRec.

### Ablation Study

Per-module ablation (HR@10):

| Configuration | Instruments | Arts | Games | Note |
|------|------------|------|-------|------|
| **MACRec (Full)** | **0.1363** | **0.1329** | **0.1078** | All components |
| w/o $\mathcal{L}_{con}^l$ | 0.1289 | 0.1283 | 0.1018 | Remove cross-modal quantization contrastive |
| w/o $\mathcal{L}_{align}$ | 0.1310 | 0.1301 | 0.1026 | Remove reconstruction alignment |
| w/o $\mathcal{L}_{implicit}$ | 0.1312 | 0.1296 | 0.1042 | Remove implicit alignment |
| w/o Explicit Alignment | 0.1296 | 0.1299 | 0.1037 | Remove explicit alignment |

**Key observation**: Removing $\mathcal{L}_{con}$ leads to the largest performance drop, indicating that cross-modal contrastive learning during quantization is the most critical component.

ID collision rate comparison (%):

| Dataset | Modality | MQL4GRec | MACRec | Reduction |
|--------|------|----------|--------|------|
| Instruments | Text | 2.76 | **2.38** | -14% |
| Instruments | Image | 3.71 | **3.23** | -13% |
| Arts | Text | 5.15 | **4.24** | -18% |
| Games | Image | 26.10 | **25.24** | -3.3% |

MACRec effectively reduces ID collision rates for both modalities.

### Key Findings

1. **Cross-modal quantization is most critical**: $\mathcal{L}_{con}$ is the most impactful component, demonstrating that introducing cross-modal interaction during the quantization stage is essential for Semantic ID quality.
2. **Visualization validates modality complementarity**: Text embeddings excel at clustering by brand, while visual embeddings better distinguish product categories (e.g., different instrument shapes), confirming their complementary nature.
3. **Improved codebook utilization**: MACRec achieves more uniform codeword assignment, avoiding codebook collapse.
4. **Optimal onset layer for contrastive loss**: Applying the contrastive loss starting from layer 3 yields the best results, with the first two layers preserving modality-specific information and the latter two leveraging cross-modal signals to compensate for semantic loss.

## Highlights & Insights

1. **First to introduce cross-modal interaction during quantization**: Prior multimodal GR methods process each modality independently during quantization; MACRec is the first to incorporate cross-modal interaction at every layer of RQ-VAE.
2. **Comprehensive multi-aspect alignment strategy**: The approach covers four levels—per-layer quantization contrastive loss, reconstruction alignment, encoder-side implicit alignment, and decoder-side explicit alignment.
3. **Elegant pseudo-label mechanism**: Clustering results from one modality serve as positive samples for the other, eliminating the need for cross-modal annotation.
4. **In-depth ID collision rate analysis**: Beyond recommendation performance, the paper analyzes quantization quality (collision rates, codeword distributions), providing multi-angle evidence for the method's effectiveness.

## Limitations & Future Work

1. **Only text and image modalities are used**: Other potentially useful modalities (e.g., price, category labels, user reviews) are not considered.
2. **Fixed clustering number K=512**: Different datasets may require different clustering granularities.
3. **Constrained GR model backbone**: T5-small (4-layer encoder+decoder) is used; performance on larger models remains to be validated.
4. **Fairness regarding MQL4GRec**: The paper notes that MQL4GRec's million-scale pretraining data was not used to ensure fair comparison, which also means MQL4GRec's full potential may not be demonstrated.
5. **Inference efficiency**: Dual-modality inference requires two generation paths plus score fusion, incurring higher inference overhead than unimodal GR.

## Related Work & Insights

- **TIGER**: Pioneered the paradigm of generating Semantic IDs via RQ-VAE; MACRec extends this with multimodal support.
- **MQL4GRec**: The most direct predecessor, employing multimodal quantization language but lacking cross-modal interaction during quantization.
- **CLIP / Contrastive Learning**: InfoNCE loss is applied at multiple positions, demonstrating the broad applicability of contrastive learning in multimodal recommendation.
- **Implications for recommender systems**: The quality of RQ-VAE quantization is a key bottleneck in GR performance; optimizing from the quantization stage may be a more effective improvement path.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Cross-modal interaction during quantization is the core innovation; the multi-aspect alignment strategy is comprehensive.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three datasets, multiple ablations, collision rate analysis, and codeword distribution visualization.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with detailed method descriptions.
- **Value**: ⭐⭐⭐⭐ — Significant contribution to the GR field; the cross-modal quantization idea is generalizable.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] StyleMotif: Multi-Modal Motion Stylization using Style-Content Cross Fusion](../../ICCV2025/image_generation/stylemotif_multi-modal_motion_stylization_using_style-content_cross_fusion.md)
- [\[ACL 2026\] From Past To Path: Masked History Learning for Next-Item Prediction in Generative Recommendation](../../ACL2026/image_generation/from_past_to_path_masked_history_learning_for_next-item_prediction_in_generative.md)
- [\[AAAI 2026\] Multi-Metric Preference Alignment for Generative Speech Restoration](multi-metric_preference_alignment_for_generative_speech_restoration.md)
- [\[CVPR 2026\] Cross-Modal Emotion Transfer for Emotion Editing in Talking Face Video](../../CVPR2026/image_generation/cross-modal_emotion_transfer_for_emotion_editing_in_talking_face_video.md)
- [\[CVPR 2026\] Quantization with Unified Adaptive Distillation to enable multi-LoRA based one-for-all Generative Vision Models on edge](../../CVPR2026/image_generation/quantization_with_unified_adaptive_distillation_to_enable_multi-lora_based_one-f.md)

<!-- RELATED:END -->
