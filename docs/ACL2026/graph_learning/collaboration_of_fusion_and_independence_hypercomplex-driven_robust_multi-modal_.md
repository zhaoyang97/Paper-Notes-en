---
title: >-
  [Paper Note] Collaboration of Fusion and Independence: Hypercomplex-driven Robust Multi-Modal Knowledge Graph Completion
description: >-
  [ACL 2026][Graph Learning][Multi-modal Knowledge Graphs] M-Hyper encodes multi-modal knowledge graph (MMKG) entities as four orthogonal bases of a biquaternion, carrying three independent modalities (structural, visual…
tags:
  - "ACL 2026"
  - "Graph Learning"
  - "Multi-modal Knowledge Graphs"
  - "Hypercomplex Space"
  - "Biquaternion"
  - "Modal Fusion"
  - "Link Prediction"
date: 2026-05-08
content_hash: ddec34ad1ab909e4
---

# Collaboration of Fusion and Independence: Hypercomplex-driven Robust Multi-Modal Knowledge Graph Completion

**Conference**: ACL 2026  
**arXiv**: [2509.23714](https://arxiv.org/abs/2509.23714)  
**Code**: https://github.com/zjukg/M-Hyper (Available)  
**Area**: Multi-modal Fusion / Knowledge Graph Completion / Representation Learning  
**Keywords**: Multi-modal Knowledge Graphs, Hypercomplex Space, Biquaternion, Modal Fusion, Link Prediction

## TL;DR
M-Hyper encodes multi-modal knowledge graph (MMKG) entities as four orthogonal bases of a biquaternion, carrying three independent modalities (structural, visual, and textual) along with one fused modality. Through the Hamilton product, it simultaneously achieves "modal independence preservation" and "sufficient pairwise interaction," outperforming 18 baselines on DB15K, MKG-W, and MKG-Y datasets with the lowest VRAM usage and shortest training time.

## Background & Motivation
**Background**: Current Multi-modal Knowledge Graph Completion (MMKGC) follows two main routes: fusion-based (e.g., IKRL, OTKGE, AdaMF, MyGO), which compresses multi-modal features into a unified representation via explicit fusion modules or cross-modal losses; and ensemble-based (e.g., MoSE, IMF, MoMoK), which trains independent sub-models for each modality and performs joint decision-making.

**Limitations of Prior Work**: Fusion-based methods rely on fixed strategies, inevitably losing modality-specific information and failing to dynamically adjust modal weights for different relations. Ensemble-based methods preserve independence but lack deep cross-modal interaction mechanisms, making it difficult to capture subtle dependencies under complex relations.

**Key Challenge**: Modal contributions in MMKGs are dynamic, context-aware, and task-dependent. It is necessary to both maintain modal independence (to avoid information loss) and ensure sufficient interaction (to capture synergy). Satisfying both requirements simultaneously in traditional Euclidean vector spaces is nearly impossible.

**Goal**: Design a representation space where "independent modalities" and "fused modalities" coexist, naturally supporting pairwise interaction while possessing translation and rotation capabilities for relation modeling.

**Key Insight**: The authors observe that quaternion algebra consists of four linearly independent orthogonal bases $\{\mathbf{1}, \mathbf{i}, \mathbf{j}, \mathbf{k}\}$, and the Hamilton product naturally generates all pairwise cross-terms. This perfectly accommodates "3 independent modalities + 1 fused modality." Furthermore, using biquaternions (quaternions with complex coefficients) allows for the simultaneous modeling of both translation and rotation transformations.

**Core Idea**: Map structural, visual, textual, and fused modalities to the four orthogonal bases of a biquaternion and use the Hamilton product as the scoring function. Independence is guaranteed by the orthogonality of the bases, while interaction is provided by the cross-terms of the product.

## Method

### Overall Architecture
Input: A triplet $(h, r, t)$ along with structural embeddings $\mathbf{e}^s$, visual embeddings $\mathbf{e}^v$ (VGG), and textual embeddings $\mathbf{e}^t$ (BERT) for entities $h$ and $t$. The process consists of four steps: (1) The **FERF module** decomposes each independent modality into "modality-specific" and "task-specific" subspaces to obtain robust $\hat{\mathbf{e}}^s, \hat{\mathbf{e}}^v, \hat{\mathbf{e}}^t$; (2) The **R2MF module** uses relation-aware gated fusion to obtain the fused representation $\hat{\mathbf{e}}^j$, incorporating noise self-distillation for robustness; (3) The four modalities are mapped to the four orthogonal bases of a biquaternion $Q = \hat{\mathbf{e}}^j + \hat{\mathbf{e}}^s \mathbf{i} + \hat{\mathbf{e}}^v \mathbf{j} + \hat{\mathbf{e}}^t \mathbf{k}$; (4) The **Biquaternion Scoring Function** $\phi(h,r,t) = \langle (Q_h \oplus Q_r^T) \otimes Q_r^R, Q_t \rangle$ simultaneously models translation (addition $\oplus$) and rotation (Hamilton product $\otimes$).

### Key Designs

1.  **FERF: Fine-grained Entity Representation Decomposition**:
    *   **Function**: Resolves noise caused by missing modalities and cross-modal semantic ambiguity to obtain robust independent representations.
    *   **Mechanism**: For each modality $m$, the representation is decomposed into modality-specific $\mathbf{e}^m_m$ (capturing raw info via pretrained encoders + MLP) and task-specific $\mathbf{e}^m_t$ (learnable embeddings initialized with PCA). A reconstruction loss $\mathcal{L}_{recon} = \sum_m \|\mathcal{E}^m(\mathbf{e}^m_t; \{\mathbf{e}^{\hat{m}}_m: \hat{m} \neq m\}) - \mathbf{e}^m_m\|^2$ is introduced, requiring "task embedding + raw embeddings of other modalities" to reconstruct the original information, forcing the task embedding to preserve modal characteristics while collaborating. Finally, $\hat{\mathbf{e}}^m = \mathbf{e}^m_m + \mathbf{e}^m_t$.
    *   **Design Motivation**: Using only pretrained encoder outputs is susceptible to noise; using only learned embeddings loses pretrained semantics. Adding both pathways with reconstruction constraints denoises while preserving semantics.

2.  **R2MF: Relation-aware Gated Fusion + Noise Self-distillation**:
    *   **Function**: Obtains a fused modality representation $\hat{\mathbf{e}}^j$ that is sensitive to relation context and robust to noise.
    *   **Mechanism**: (a) Relation-aware gating—uses a 1-layer MLP based on $[\hat{\mathbf{e}}^m; \mathbf{r}^T; \mathbf{r}^R]$ to calculate modal weights $w^m$, followed by a relation-level learnable temperature $\tau_r$ in a softmax to get $\hat{w}^m(e,r) = \exp(w^m/\tau_r) / \sum_i \exp(w^i/\tau_r)$. (b) Noise self-distillation—adds Gaussian noise $\tilde{\mathbf{e}}^m \sim \mathcal{N}(\bm{\varphi}^m, \bm{\mu}^m)$ to get student fused representations $\hat{\mathbf{e}}^{j'}$, using the noise-free $\hat{\mathbf{e}}^j$ as a teacher to enforce consistency via $\mathcal{L}_{distill} = \frac{1}{n}\sum \|\hat{\mathbf{e}}^j_i - \hat{\mathbf{e}}^{j'}_i\|^2$.
    *   **Design Motivation**: Fixed fusion cannot adapt to the fact that different relations require different modalities. Noise distillation makes the gating robust to missing or perturbed modalities.

3.  **Biquaternion Scoring Function**:
    *   **Function**: Implements "independent modality preservation + pairwise interaction + relation translation + relation rotation" within a unified algebraic structure.
    *   **Mechanism**: $\hat{\mathbf{e}}^j, \hat{\mathbf{e}}^s, \hat{\mathbf{e}}^v, \hat{\mathbf{e}}^t$ are placed on the $\mathbf{1}, \mathbf{i}, \mathbf{j}, \mathbf{k}$ bases of a biquaternion. Relations learn two sets of embeddings $Q_r^T$ (translation) and $Q_r^R$ (rotation). The score is calculated as $Q_{h'} = Q_h \oplus Q_r^T$ followed by $Q_{h''} = Q_{h'} \otimes Q_r^R$, finally taking the inner product with $Q_t$. The Hamilton product expansion naturally produces all pairwise cross-terms $\hat{\mathbf{e}}^m_h \cdot \hat{\mathbf{e}}^{m'}_t$ (Theorem 2).
    *   **Design Motivation**: Mathematically provides an "independent yet interactive" structure. Theorem 1 uses the Information Bottleneck framework to prove this representation is strictly superior to pure fusion $T_f$ and pure ensemble $T_{ens}$: $\mathcal{L}_{IB}(Q) < \min(\mathcal{L}_{IB}(T_f), \mathcal{L}_{IB}(T_{ens}))$.

### Loss & Training
Total loss: $\mathcal{L}_{total} = \mathcal{L}_{recon} + \mathcal{L}_{distill} + \mathcal{L}_{triple} + \mathcal{L}_{reg}$, where $\mathcal{L}_{triple}$ is cross-entropy (with 1-vs-all candidates) and $\mathcal{L}_{reg}$ is N3 regularization. Optimizer: Adagrad, batch size: 1000, key hyperparameters: $d=128$, $\lambda=0.005$, noise rate $\beta=0.2$, learning rate $\alpha=0.1$. Training includes reverse triplets $(t, r^{-1}, h)$.

## Key Experimental Results

### Main Results
Compared against 18 baselines (6 Uni-modal KGE and 12 MMKGC methods) on DB15K, MKG-W, and MKG-Y:

| Dataset | Metric | M-Hyper | Prev. SOTA (MoMoK) | Gain |
| :--- | :--- | :--- | :--- | :--- |
| DB15K | MRR | **41.25** | 39.57 | +1.68 |
| DB15K | Hit@10 | **56.09** | 54.14 | +1.95 |
| MKG-W | MRR | **37.02** | 36.10 (MyGO) | +0.92 |
| MKG-W | Hit@10 | **48.84** | 47.75 (MyGO) | +1.09 |
| MKG-Y | MRR | **39.46** | 38.44 (MyGO) | +1.02 |
| MKG-Y | Hit@10 | 45.22 | 45.48 (AdaMF) | -0.26 |

Average MRR increased by ~4.25% and Hit@10 by ~3.89%. Efficiency analysis shows M-Hyper has the **shortest training time per epoch and near-optimal memory usage** among compared methods.

### Ablation Study

| Configuration | DB15K MRR | MKG-W MRR | MKG-Y MRR | Description |
| :--- | :--- | :--- | :--- | :--- |
| M-Hyper (Full) | **41.25** | **37.02** | **39.46** | — |
| w/o fused modality $\hat{\mathbf{e}}^j$ | 36.36 | 35.09 | 36.71 | Most significant drop |
| w/o visual $\hat{\mathbf{e}}^v$ | 35.09 | 36.46 | 37.95 | Visual info critical for DB15K |
| w/o structural $\hat{\mathbf{e}}^s$ | 39.77 | 34.62 | 38.03 | Structural info critical for MKG-W |
| w/o FERF | 39.24 | 35.93 | 37.93 | Robust decomp. contribution |
| w/o Noise Distill | 39.64 | 36.10 | 38.16 | Distillation helps ~1.6 MRR |
| w/o Relation Gating | 40.18 | 36.18 | 38.21 | Dynamic fusion contribution |
| w/o Rotation $\mathbf{r}^R$ | 38.91 | 36.46 | 37.78 | Biquaternion degrades to quaternion |
| M-Hyper-fusion variant | 39.23 | 35.54 | 37.52 | Pure fusion loss |
| M-Hyper-ensemble variant | 39.31 | 34.75 | 37.58 | Pure ensemble loss |

### Key Findings
- Removing the **fused modality $\hat{\mathbf{e}}^j$** causes the largest drop (DB15K -4.89 MRR), proving the real part of the biquaternion carries essential cross-modal synergy.
- Removing **rotation $\mathbf{r}^R$** (falling back to quaternion) leads to performance loss across all datasets, validating the expressive power of complex-domain rotation.
- In scenarios involving missing modalities, noise, or sparse links, M-Hyper outperforms AdaMF and MoMoK.
- t-SNE visualization shows the fused modality provides the highest discriminative power for city-country relations.

## Highlights & Insights
- **Algebraic Structure as Representation Constraint**: Using the four orthogonal bases of a biquaternion to encode "3 independent + 1 fused" is an elegant design. Orthogonality ensures independence, and the Hamilton product generates interactions without extra regularization.
- **FERF Decomposition**: Uses reconstruction loss to separate "information that must be contributed by this modality" from "collaborative cross-modal information," providing finer grain than pure decoupling.
- **Unified Triple Scoring**: Combining DualE’s bi-directional transformations with BiQUE’s biquaternion algebra while layering multi-modal semantics represents a culmination of KGE scoring function evolution.
- The theoretical proof via Information Bottleneck (Theorem 1) provides a formal explanation for why biquaternions outperform standard fusion/ensemble approaches.

## Limitations & Future Work
- The study is limited to **transductive** static MMKGC and cannot handle dynamic scenarios with new entities or relations.
- The 8d dimensionality of the biquaternion space naturally doubles the parameter count compared to quaternions; advantages may diminish as $d$ increases.
- Robustness experiments only considered random noise/omission, not adversarial perturbations.
- Potential exists to transfer the "coexistence and collaboration" idea to entity alignment, KGQA, and NER.

## Related Work & Insights
- **vs MoMoK (ICLR 2025)**: MoMoK uses MoE to decouple modalities and minimizes mutual information, but lacks explicit interaction between sub-models. M-Hyper uses biquaternion algebra to handle both.
- **vs MyGO (AAAI 2025)**: MyGO uses fine-grained tokenization for fusion, losing modal independence. M-Hyper preserves independent modalities as imaginary parts.
- **vs BiQUE (EMNLP 2021)**: BiQUE embeds uni-modal KGs in biquaternion space for rotation and translation. M-Hyper is the first to extend this to multi-modal scenarios.
- **vs AdaMF (LREC-COLING 2024)**: AdaMF uses adversarial training for noise enhancement; M-Hyper utilizes self-distillation and task embeddings for more stable robustification.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to introduce biquaternions to MMKGC with a "bases carry modalities" design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 3 datasets, 18 baselines, 3D ablation, and robustness/efficiency/visualization analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear methodology, though algebraic derivations may be challenging for some readers.
- Value: ⭐⭐⭐⭐ New SOTA for MMKGC; strategies for algebraic representation constraints are insightful for the multi-modal field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] GS-Quant: Granular Semantic and Generative Structural Quantization for Knowledge Graph Completion](gs-quant_granular_semantic_and_generative_structural_quantization_for_knowledge_.md)
- [\[AAAI 2026\] MyGram: Modality-aware Graph Transformer with Global Distribution for Multi-modal Entity Alignment](../../AAAI2026/graph_learning/mygram_modality-aware_graph_transformer_with_global_distribution_for_multi-modal.md)
- [\[ACL 2026\] ComplianceNLP: Knowledge-Graph-Augmented RAG for Multi-Framework Regulatory Gap Detection](compliancenlp_knowledge-graph-augmented_rag_for_multi-framework_regulatory_gap_d.md)
- [\[CVPR 2026\] Graph-to-Frame RAG: Visual-Space Knowledge Fusion for Training-Free and Auditable Video Reasoning](../../CVPR2026/graph_learning/graph-to-frame_rag_visual-space_knowledge_fusion_for_training-free_and_auditable.md)
- [\[NeurIPS 2025\] Uncertain Knowledge Graph Completion via Semi-Supervised Confidence Distribution Learning](../../NeurIPS2025/graph_learning/uncertain_knowledge_graph_completion_via_semi-supervised_confidence_distribution.md)

</div>

<!-- RELATED:END -->
