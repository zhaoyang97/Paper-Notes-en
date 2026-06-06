---
title: >-
  [Paper Note] NOSE: Neural Olfactory-Semantic Embedding with Tri-Modal Orthogonal Contrastive Learning
description: >-
  [ACL 2026][Interpretability][Olfactory representation learning] This paper proposes NOSE, a tri-modal olfactory representation learning framework that uses molecules as a hub to align molecular structures…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "Olfactory representation learning"
  - "Tri-modal alignment"
  - "Orthogonal decoupling"
  - "Contrastive learning"
  - "Weak positives"
date: 2026-05-08
content_hash: be1dac7bf28d8a77
---

# NOSE: Neural Olfactory-Semantic Embedding with Tri-Modal Orthogonal Contrastive Learning

**Conference**: ACL 2026  
**arXiv**: [2604.10452](https://arxiv.org/abs/2604.10452)  
**Code**: [GitHub](https://github.com/Xianyusyy/NOSE)  
**Area**: Interpretability  
**Keywords**: Olfactory representation learning, Tri-modal alignment, Orthogonal decoupling, Contrastive learning, Weak positives

## TL;DR
This paper proposes NOSE, a tri-modal olfactory representation learning framework that uses molecules as a hub to align molecular structures, receptor sequences, and natural language descriptions via an orthogonal injection mechanism. Coupled with an LLM-driven weak positive strategy to alleviate description sparsity, NOSE achieves SOTA results across 11 downstream tasks and demonstrates superior zero-shot generalization capabilities.

## Background & Motivation

**Background**: Olfaction is the most challenging sense to digitize—vision has pixels and audition has spectra, but olfaction lacks a stable mapping from physical quantities to perception. The olfactory perception chain consists of: molecular structure → receptor binding → neural signals → linguistic description.

**Limitations of Prior Work**: (1) Existing methods only model fragments of the olfactory pathway (either molecular structure alone, or just molecule-description/receptor correspondences), failing to capture the complete molecule-receptor-semantic chain in a unified framework; (2) Mainstream approaches model odor prediction as classification (e.g., "floral" vs. "fruity"), which breaks the continuity of the odor space—descriptors like "minty" and "cool" are highly correlated but treated as independent labels in classification; (3) Classification objectives force models to fit label boundaries, discarding structural information that is crucial for molecular identity but irrelevant for specific classification.

**Key Challenge**: Complete tri-modal data (molecule-receptor-description triplets) are extremely scarce, although bi-modal data (molecule-receptor and molecule-description) can be obtained separately. How can tri-modal alignment be achieved without triplet annotations?

**Goal**: Construct a continuous representation space covering the full olfactory perception pathway, ensuring that molecular embeddings encode both receptor and semantic information without mutual interference.

**Key Insight**: The molecule serves as the sole intersection of the two bi-modal datasets, acting as a hub to bridge receptor and semantic information. The critical problem is preventing the two signals from overriding each other during injection—the solution is orthogonal injection.

**Core Idea**: Treat receptor and semantic features as orthogonal increments superimposed on the molecular representation. Use Gram-Schmidt orthogonalization to ensure modal independence, while leveraging LLMs to mine semantic neighbor relationships between odor descriptors to expand sparse labels.

## Method

### Overall Architecture
NOSE performs tri-modal pre-training focused on the molecule: Uni-Mol extracts 3D molecular features $z_{mol}$ (frozen), ESM-2 extracts receptor sequence features $z_{rec}$ (with a trainable projection layer), and Qwen3 Embedding extracts odor description features $z_{desc}$ via LoRA fine-tuning. Molecular embeddings are decomposed through dual adapters into a receptor-alignment component $a_r$ and a description-alignment component $a_d$. After Gram-Schmidt orthogonalization, the model is trained with multiple InfoNCE losses. Only the molecular encoder and adapters are required during inference.

### Key Designs

1. **Orthogonal Injection Mechanism**:

    - **Function**: Independently injects receptor and semantic features into molecular representations to prevent information overlap between modalities.
    - **Mechanism**: Hard orthogonalization (geometric decoupling) is achieved via Gram-Schmidt by projecting adapter output $a_{adapter}$ onto the orthogonal complement of $z_{mol}$: $z_{adapter} = a_{adapter} - \frac{a_{adapter} \cdot z_{mol}}{\|z_{mol}\|^2 + \epsilon} z_{mol}$. Soft orthogonalization (optimization regularization) uses the loss function $\mathcal{L}_{orth} = \sum_{(i,j)} \|\frac{z_i}{\|z_i\|} \cdot \frac{z_j}{\|z_j\|}\|^2$ to keep the three subspaces decorrelated.
    - **Design Motivation**: Simple multi-modal fusion leads to feature redundancy and information overriding; orthogonal constraints ensure that each modality contributes unique and irreplaceable information.

2. **LLM-driven Weak Positive Augmentation**:

    - **Function**: Alleviates false negative issues caused by the sparsity of odor descriptions.
    - **Mechanism**: DeepSeek is used to mine semantic neighbor relationships among 1,086 odor descriptors, expanding isolated labels into a continuous olfactory semantic neighborhood. In contrastive learning, positive samples receive a weight of 1.0, weak positives 0.5, and negatives 0.0, implementing a softened InfoNCE loss.
    - **Design Motivation**: In traditional contrastive learning, "lemon" and "sour" would be treated as negatives and repelled, yet they should be adjacent in olfactory space. The weak positive strategy transforms discrete label spaces into continuous semantic manifolds.

3. **Differential Adapter Design**:

    - **Function**: Adapts to the massive scale disparity between the two bi-modal datasets (3,877 receptor pairs vs. 88,512 description pairs).
    - **Mechanism**: The description adapter uses a 12-layer inverted bottleneck ResMLP structure (high capacity for rich text data), while the receptor adapter uses a bottleneck structure with high dropout (to prevent overfitting on sparse data).
    - **Design Motivation**: A data volume difference exceeding 20x means a unified architecture would cause overfitting on one side or underfitting on the other.

### Loss & Training
The total loss includes receptor-molecule InfoNCE, description-molecule soft-weighted InfoNCE, intra-modal InfoNCE, and orthogonal constraint losses. The molecular encoder (Uni-Mol) is frozen, ESM-2 has a trainable projection, and Qwen3 Embedding is fine-tuned with LoRA. The final representation is $Z = w_1 \cdot z_{mol} + w_2 \cdot a_r + w_3 \cdot a_d$.

## Key Experimental Results

### Main Results (Basic Perceptual Attribute Prediction, Pearson Correlation)

| Method | Threshold (Abraham) | Pleasantness (Keller) | Pleasantness (Sagar) | Intensity (Keller) | Intensity (Sagar) | Intensity (Ravia) |
|------|-------------|---------------|---------------|-------------|-------------|-------------|
| Uni-Mol | 0.78 | 0.68 | 0.14 | 0.27 | 0.37 | 0.31 |
| ChemBERTa | 0.81 | 0.65 | 0.15 | 0.39 | 0.45 | 0.47 |
| **NOSE** | **0.84** | **0.71** | **0.40** | **0.42** | **0.47** | **0.49** |

### Ablation Study

| Configuration | Key Indicator | Description |
|------|---------|------|
| NOSE (Full) | SOTA | Tri-modal + Orthogonal + Weak Positives |
| w/o Receptor Modality | Significant Drop | Bi-modal only, missing biological grounding |
| w/o Orthogonal Constraint | Drop | Modal feature redundancy |
| w/o Weak Positives | Drop | False negatives causing representation degradation |

### Key Findings
- NOSE consistently reaches or exceeds SOTA across 11 downstream tasks, with the largest gains on sparse datasets (Sagar) (Pearson jumping from 0.14 to 0.40).
- Excellent zero-shot generalization performance validates the strong alignment between the representation space and human olfactory intuition.
- Performance on mixture perception tasks is also strong, indicating the learned representations capture non-linear interactions between molecules.

## Highlights & Insights
- Implementing tri-modal alignment without triplet labels by using the molecule as a hub is a core innovation—leveraging the intersection of bi-modal data to bridge the third modality indirectly.
- The philosophy of orthogonal injection is transferable: in any multi-modal fusion where signal sources provide complementary rather than redundant information, orthogonal constraints prevent information overriding.
- The weak positive strategy "softens" discrete label spaces into continuous manifolds, a generalizable technique for handling label sparsity in contrastive learning.

## Limitations & Future Work
- Receptor data remains limited (3,877 pairs); performance may improve as more receptor-ligand data accumulates.
- Currently only considers single-molecule odor prediction; combinatorial effects of mixed odors in real-world scenarios are more complex.
- The subjectivity of olfactory descriptions is fundamentally difficult to resolve, as descriptions vary significantly across different cultural backgrounds.

## Related Work & Insights
- **vs POM**: POM only models molecule-description bi-modality and lacks biological grounding from receptors; NOSE tri-modal alignment consistently outperforms POM in perceptual attribute prediction.
- **vs Uni-Mol**: While Uni-Mol is a strong molecular encoder, NOSE further enhances performance across all tasks by injecting receptor and semantic information.
- **vs Classification Methods**: Traditional classification fails to capture the continuity of the odor space; NOSE's representation learning paradigm fundamentally addresses this issue.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First tri-modal framework covering the full olfactory pathway; novel orthogonal injection mechanism.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 11 downstream tasks, 6 datasets, comprehensive ablation and zero-shot experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear derivation of motivation, high-quality figures, and accessible background introduction.
- Value: ⭐⭐⭐⭐ Olfactory computing is an emerging cross-disciplinary field; the framework design is transferable to other multi-modal scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning](../../NeurIPS2025/interpretability/self-supervised_contrastive_learning_is_approximately_supervised_contrastive_lea.md)
- [\[ICLR 2026\] Modal Logical Neural Networks for Financial AI](../../ICLR2026/interpretability/modal_logical_neural_networks_for_financial_ai.md)
- [\[AAAI 2026\] Explainable Melanoma Diagnosis with Contrastive Learning and LLM-based Report Generation](../../AAAI2026/interpretability/explainable_melanoma_diagnosis_with_contrastive_learning_and_llm-based_report_ge.md)
- [\[AAAI 2026\] Adaptive Evidential Learning for Temporal-Semantic Robustness in Moment Retrieval](../../AAAI2026/interpretability/adaptive_evidential_learning_for_temporal-semantic_robustnes.md)
- [\[ACL 2026\] Interpretable Semantic Gradients in SSD: A PCA Sweep Approach and a Case Study on AI Discourse](interpretable_semantic_gradients_in_ssd_a_pca_sweep_approach_and_a_case_study_on.md)

</div>

<!-- RELATED:END -->
