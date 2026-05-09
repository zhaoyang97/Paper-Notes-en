---
title: >-
  [Paper Note] NOSE: Neural Olfactory-Semantic Embedding with Tri-Modal Orthogonal Contrastive Learning
description: >-
  [ACL 2026][olfactory representation learning] This paper proposes NOSE, a tri-modal olfactory representation learning framework that uses molecules as a pivot to align three modalities—molecular structure, receptor sequences, and natural language descriptions—via an orthogonal injection mechanism. Combined with an LLM-driven weak positive augmentation strategy to address description sparsity, NOSE achieves state-of-the-art performance on 11 downstream tasks and demonstrates strong zero-shot generalization.
tags:
  - ACL 2026
  - olfactory representation learning
  - tri-modal alignment
  - orthogonal disentanglement
  - contrastive learning
  - weak positive samples
date: 2026-05-08
content_hash: a61a08142eb10a9e
---

# NOSE: Neural Olfactory-Semantic Embedding with Tri-Modal Orthogonal Contrastive Learning

**Conference**: ACL 2026
**arXiv**: [2604.10452](https://arxiv.org/abs/2604.10452)
**Code**: [GitHub](https://github.com/Xianyusyy/NOSE)
**Area**: Interpretability
**Keywords**: olfactory representation learning, tri-modal alignment, orthogonal disentanglement, contrastive learning, weak positive samples

## TL;DR
This paper proposes NOSE, a tri-modal olfactory representation learning framework that uses molecules as a pivot to align three modalities—molecular structure, receptor sequences, and natural language descriptions—via an orthogonal injection mechanism. Combined with an LLM-driven weak positive augmentation strategy to address description sparsity, NOSE achieves state-of-the-art performance on 11 downstream tasks and demonstrates strong zero-shot generalization.

## Background & Motivation

**Background**: Olfaction is the most difficult sense to digitize—vision has pixels, hearing has spectrograms, but olfaction lacks a stable mapping from physical quantities to perception. The olfactory perception chain is: molecular structure → receptor binding → neural signals → linguistic description.

**Limitations of Prior Work**: (1) Existing methods model only fragments of the olfactory pathway (molecular structure alone, or molecule–description/receptor pairs in isolation), without capturing the complete molecule → receptor → semantics chain in a unified framework. (2) Mainstream methods cast odor prediction as a classification problem ("floral" or "fruity"), which disrupts the continuity of odor space—"mint" and "cool" are highly related yet treated as independent labels under a classification paradigm. (3) Classification objectives force the model to fit label boundaries, discarding information that is structurally important but classification-irrelevant.

**Key Challenge**: Complete tri-modal data (molecule–receptor–description triplets) is extremely scarce, whereas bimodal data (molecule–receptor and molecule–description pairs) can be obtained separately. The key challenge is achieving tri-modal alignment without triplet-level annotations.

**Goal**: Construct a continuous representation space covering the complete olfactory perception pathway, such that molecular representations jointly encode receptor and semantic information without mutual interference.

**Key Insight**: Molecules are the sole intersection of the two bimodal datasets and can serve as a pivot to bridge receptor and semantic information. The critical challenge is preventing the two signals from overwriting each other during injection—addressed by orthogonal injection.

**Core Idea**: Receptor features and semantic features are added as orthogonal increments to the molecular representation. Gram-Schmidt orthogonalization ensures modality independence, while an LLM is used to mine semantic neighborhood relationships among odor descriptors to expand sparse labels.

## Method

### Overall Architecture
NOSE centers on molecules for tri-modal pre-training: Uni-Mol extracts molecular 3D structural features $z_{mol}$ (frozen); ESM-2 extracts receptor sequence features $z_{rec}$ (with a trainable projection layer); Qwen3 Embedding extracts odor descriptor features $z_{desc}$ via LoRA fine-tuning. The molecular embedding is decomposed into a receptor-aligned component $a_r$ and a description-aligned component $a_d$ via dual adapters, which are orthogonalized via Gram-Schmidt and trained with multiple InfoNCE losses. At inference, only the molecular encoder and adapters are required.

### Key Designs

1. **Orthogonal Injection Mechanism**:

    - Function: Independently inject receptor and semantic features into the molecular representation, preventing cross-modal information overwriting.
    - Mechanism: Hard orthogonalization (geometric disentanglement) projects the adapter output $a_{adapter}$ onto the orthogonal complement of $z_{mol}$ via Gram-Schmidt: $z_{adapter} = a_{adapter} - \frac{a_{adapter} \cdot z_{mol}}{\|z_{mol}\|^2 + \epsilon} z_{mol}$. Soft orthogonalization (optimization regularization) drives the three subspaces to remain mutually decorrelated via the loss $\mathcal{L}_{orth} = \sum_{(i,j)} \|\frac{z_i}{\|z_i\|} \cdot \frac{z_j}{\|z_j\|}\|^2$.
    - Design Motivation: Naive multimodal fusion leads to feature redundancy and overwriting; orthogonal constraints ensure that each modality contributes unique and irreplaceable information.

2. **LLM-Driven Weak Positive Augmentation**:

    - Function: Mitigate the false negative problem caused by sparse odor descriptions.
    - Mechanism: DeepSeek is used to mine semantic neighborhood relationships among 1,086 odor descriptors, expanding isolated labels into continuous olfactory semantic neighborhoods. In contrastive learning, positive samples receive weight 1.0, weak positives 0.5, and negatives 0.0, yielding a softened InfoNCE loss.
    - Design Motivation: In standard contrastive learning, "lemon" and "sour" would be treated as negatives and repelled, yet they should be adjacent in olfactory space. The weak positive strategy transforms a discrete label space into a continuous semantic manifold.

3. **Differentiated Adapter Design**:

    - Function: Accommodate the large scale disparity between the two bimodal datasets (3,877 receptor pairs vs. 88,512 description pairs).
    - Mechanism: The description adapter uses a 12-layer inverted-bottleneck ResMLP (high capacity to fit rich textual data); the receptor adapter uses a bottleneck structure with high dropout (to prevent overfitting on sparse data).
    - Design Motivation: A more than 20× data scale difference means a unified architecture would cause overfitting on one side or underfitting on the other.

### Loss & Training
The total loss comprises: receptor–molecule InfoNCE, description–molecule soft-weighted InfoNCE, intra-modal InfoNCE, and orthogonal constraint loss. The molecular encoder (Uni-Mol) is frozen; ESM-2 uses a trainable projection; Qwen3 Embedding is fine-tuned with LoRA. The final representation is $Z = w_1 \cdot z_{mol} + w_2 \cdot a_r + w_3 \cdot a_d$.

## Key Experimental Results

### Main Results (Basic Perceptual Attribute Prediction, Pearson Correlation)

| Method | Threshold (Abraham) | Pleasantness (Keller) | Pleasantness (Sagar) | Intensity (Keller) | Intensity (Sagar) | Intensity (Ravia) |
|--------|--------------------|-----------------------|----------------------|--------------------|-------------------|-------------------|
| Uni-Mol | 0.78 | 0.68 | 0.14 | 0.27 | 0.37 | 0.31 |
| ChemBERTa | 0.81 | 0.65 | 0.15 | 0.39 | 0.45 | 0.47 |
| **NOSE** | **0.84** | **0.71** | **0.40** | **0.42** | **0.47** | **0.49** |

### Ablation Study

| Configuration | Key Metric | Note |
|--------------|------------|------|
| NOSE (full) | SOTA | Tri-modal + orthogonal + weak positives |
| w/o receptor modality | Significant drop | Bimodal only; lacks biological grounding |
| w/o orthogonal constraint | Drop | Modality feature redundancy |
| w/o weak positives | Drop | False negatives cause representation collapse |

### Key Findings
- NOSE matches or surpasses SOTA across all 11 downstream tasks, with the largest gains on sparse datasets (Sagar), where Pearson correlation improves from 0.14 to 0.40.
- Strong zero-shot generalization validates the learned representation space's strong alignment with human olfactory intuition.
- Good performance on mixture perception tasks indicates that the learned representations capture nonlinear intermolecular interactions.

## Highlights & Insights
- Using molecules as a pivot to achieve tri-modal alignment without triplet annotations is the core innovation—the intersection of bimodal datasets indirectly bridges the third modality.
- The design philosophy of orthogonal injection is broadly transferable: in any multimodal fusion scenario where different signal sources provide complementary rather than redundant information, orthogonal constraints prevent information overwriting.
- The weak positive strategy "softens" a discrete label space into a continuous manifold, offering a general technique for handling label sparsity in contrastive learning.

## Limitations & Future Work
- The receptor dataset contains only 3,877 pairs; performance may further improve as more receptor–ligand data become available.
- The current framework addresses single-molecule odor prediction; real-world mixture odors involve more complex combinatorial effects.
- The inherent subjectivity of olfactory descriptions cannot be fully resolved, and cross-cultural variation in odor language remains a fundamental challenge.

## Related Work & Insights
- **vs. POM**: POM models only the molecule–description bimodal alignment and lacks biological grounding from receptor information; NOSE's tri-modal alignment consistently outperforms POM on perceptual attribute prediction.
- **vs. Uni-Mol**: Uni-Mol already provides strong molecular representations, but NOSE further improves all tasks by injecting receptor and semantic information.
- **vs. Classification Methods**: Traditional classification approaches cannot capture the continuity of odor space; NOSE's representation learning paradigm fundamentally addresses this limitation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First tri-modal framework covering the complete olfactory pathway; orthogonal injection mechanism is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 11 downstream tasks, 6 datasets, extensive ablation and zero-shot experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is clearly derived, figures are well-crafted, and background is accessible.
- Value: ⭐⭐⭐⭐ Computational olfaction is an emerging interdisciplinary area; the framework design is transferable to other multimodal scenarios.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning](../../NeurIPS2025/interpretability/self-supervised_contrastive_learning_is_approximately_supervised_contrastive_lea.md)
- [\[ICLR 2026\] Modal Logical Neural Networks for Financial AI](../../ICLR2026/interpretability/modal_logical_neural_networks_for_financial_ai.md)
- [\[AAAI 2026\] Explainable Melanoma Diagnosis with Contrastive Learning and LLM-based Report Generation](../../AAAI2026/interpretability/explainable_melanoma_diagnosis_with_contrastive_learning_and_llm-based_report_ge.md)
- [\[ACL 2026\] LLM-Guided Semantic Bootstrapping for Interpretable Text Classification with Tsetlin Machines](llm-guided_semantic_bootstrapping_for_interpretable_text_classification_with_tse.md)
- [\[AAAI 2026\] Adaptive Evidential Learning for Temporal-Semantic Robustness in Moment Retrieval](../../AAAI2026/interpretability/adaptive_evidential_learning_for_temporal-semantic_robustnes.md)

<!-- RELATED:END -->
