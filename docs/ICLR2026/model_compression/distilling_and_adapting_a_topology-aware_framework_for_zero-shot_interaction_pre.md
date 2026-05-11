---
title: >-
  [Paper Note] Distilling and Adapting: A Topology-Aware Framework for Zero-Shot Interaction Prediction in Multiplex Biological Networks
description: >-
  [ICLR 2026][Model Compression][multiplex biological networks] This paper proposes CAZI-MBN, a framework that integrates domain-specific LLM sequence embeddings, a topology-aware unified graph tokenizer…
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "multiplex biological networks"
  - "zero-shot prediction"
  - "knowledge distillation"
  - "graph Transformer"
  - "multimodal representation learning"
date: 2026-05-08
content_hash: ed557d0030ef7523
---

# Distilling and Adapting: A Topology-Aware Framework for Zero-Shot Interaction Prediction in Multiplex Biological Networks

**Conference**: ICLR 2026
**arXiv**: [2603.06618](https://arxiv.org/abs/2603.06618)
**Code**: [Available](https://github.com/alanadeng/CAZI-MBN)
**Area**: Model Compression
**Keywords**: multiplex biological networks, zero-shot prediction, knowledge distillation, graph Transformer, multimodal representation learning

## TL;DR

This paper proposes CAZI-MBN, a framework that integrates domain-specific LLM sequence embeddings, a topology-aware unified graph tokenizer, context-aware cross-layer attention, and teacher-student distillation to enable zero-shot interaction prediction for unseen entities in multiplex biological networks, achieving AUROC improvements of 3.1–20.4% over the best baseline across 5 benchmark datasets.

## Background & Motivation

Multiplex Biological Networks (MBNs) represent different interaction types among the same set of entities through multi-layer structures (e.g., multiple drug–gene action mechanisms, protein interactions across cell types), and are critical tools for understanding complex biological systems. Existing methods suffer from three fundamental limitations:

**Insufficient multiplexity handling**: Most approaches rely on single-layer network analysis, discarding relational heterogeneity and semantic distinctions across interaction types.

**Weak multimodal integration**: Effectively combining biological/chemical sequence features with network topology remains challenging.

**Difficulty with zero-shot prediction**: Existing GNN methods struggle to generalize to new entities unseen during training (i.e., entities with no prior neighborhood information).

This paper proposes CAZI-MBN (Context-Aware and Zero-shot Interaction prediction in MBNs) to systematically address all three challenges.

## Method

### Overall Architecture

CAZI-MBN consists of four core modules:

1. **Feature Representation**: Sequence embeddings (domain-specific LLMs) + topology embeddings (Unified Graph Tokenizer, UGT)
2. **Context-Aware Enhancement (CAE)**: Cross-layer attention + contrastive learning
3. **Mixture of Experts (MoE)**: Adaptive prediction for multi-label classification
4. **Knowledge Distillation**: Transfer of zero-shot generalization capability from teacher to student

Training is driven by a composite loss: $\mathcal{L} = \mathcal{L}_{disc} + \mathcal{L}_{reg} + \mathcal{L}_{cls} + \beta\|\Theta\|^2$

### Key Designs

**1. Feature Representation Layer**

**Sequence Embeddings**: Domain-specific pretrained LLMs are applied according to biological entity type:
- ChemBERTa → SMILES representations of drugs/metabolites
- DNABERT-2 → Gene sequences
- ESM-2 → Protein sequences

**Unified Graph Tokenizer (UGT)**: Generates topology-, multiplexity-, and higher-order connectivity-aware node embeddings from the supra-adjacency matrix $\hat{A}$:
- Constructs a smoothed higher-order matrix: $\tilde{A} = \bar{A} + \bar{A}^2 + \cdots + \bar{A}^O$, where $\bar{A} = D^{-1/2}\hat{A}D^{-1/2}$
- Performs SVD decomposition on $\tilde{A}$ to obtain $U, \Sigma, V$
- Node embeddings: $e_v = \tilde{A}_{v,:} \cdot \text{LN}(U\sqrt{\Sigma} \| V\sqrt{\Sigma})$

The supra-adjacency matrix $\hat{A}$ encodes intra-layer connections via direct sum and incorporates inter-layer connection matrix $C$, fully preserving the multiplex network structure.

**2. Context-Aware Enhancement (CAE) Module**

The CAE module refines multiplex embeddings through a two-level attention mechanism:

**Node-level cross-layer attention**: Enables entities to adaptively aggregate information across their representations in different layers:

$$a_n^{(p \leftarrow q)} = \text{softmax}\left(\frac{\sigma(\theta^{(p)} \cdot (H_n^{(p)} \otimes H_n^{(q)}))}{\sum_{l \neq p} \sigma(\theta^{(p)} \cdot (H_n^{(p)} \otimes H_n^{(l)}))}\right)$$

**Layer-level attention aggregation**: Aggregates per-layer features into a unified representation $H$ via an attention module.

**Contrastive learning framework**:
- Negative sampling is applied to each layer graph to generate perturbed graphs $\tilde{G}_i$
- A discriminator distinguishes real edge embeddings from negative ones
- A consensus regularizer learns a consensus embedding $Z$: maximizing alignment with $H$ while minimizing alignment with $\tilde{H}$
- Regularization loss: $\mathcal{L}_{reg} = 1 + \text{CosineSim}(H, Z) - \text{CosineSim}(\tilde{H}, Z)$

**3. Mixture of Experts (MoE)**

Interaction prediction in MBNs is formulated as a multi-label classification task. In the MoE framework:
- $K$ experts $f_k$ each capture distinct interaction patterns
- A gating network assigns weights based on the input: $\boldsymbol{a} = \text{softmax}(W_g \mathbf{h} + b_g)$
- Final prediction: $\hat{\mathbf{Y}} = \sum_{k=1}^{K} a_k f_k(\mathbf{h})$

**4. Knowledge Distillation for Zero-Shot Generalization**

- **Teacher model**: Uses both sequence and topology embeddings, relying on neighborhood context
- **Student model**: Uses sequence data only, topology-agnostic
- Distillation loss: MSE alignment of teacher and student latent representations + classification loss
- At inference, the student model can make predictions for entirely unseen entities (with no graph structural information)

### Loss & Training

- **Teacher loss**: $\mathcal{L} = \mathcal{L}_{disc} + \mathcal{L}_{reg} + \mathcal{L}_{cls} + \beta\|\Theta\|^2$
    - $\mathcal{L}_{disc}$: Discriminator binary cross-entropy loss
    - $\mathcal{L}_{reg}$: Consensus regularization loss (cosine similarity)
    - $\mathcal{L}_{cls}$: Multi-label soft-margin loss
- **Student loss**: $\mathcal{L}_{distill}(\text{MSE}) + \mathcal{L}_{cls}$

## Key Experimental Results

### Main Results (5 Multiplex Biological Networks, 13 Baselines)

**DGIdb (Drug–Gene, 1,846 nodes, 5 interaction types)**:

| Setting | Model | AUROC | AUPRC | HS | SA |
|---------|-------|-------|-------|-----|-----|
| Transductive | Graph Transformer | 0.505 | 0.514 | 0.493 | 0.508 |
| Transductive | HDMI | 0.551 | 0.557 | 0.540 | 0.511 |
| Transductive | **CAZI-MBN** | **0.715** | **0.729** | **0.687** | **0.684** |
| Zero-shot | DMGI | 0.524 | 0.528 | 0.529 | 0.502 |
| Zero-shot | **CAZI-MBN** | **0.671** | **0.709** | **0.688** | **0.663** |

**ChEMBL (Compound–Bacteria, 9,368 nodes, 3 interaction types)**:

| Setting | Model | AUROC | AUPRC | HS | SA |
|---------|-------|-------|-------|-----|-----|
| Transductive | HDMI | 0.663 | 0.762 | 0.789 | 0.730 |
| Transductive | **CAZI-MBN** | **0.812** | **0.863** | **0.889** | **0.757** |
| Zero-shot | DMGI | 0.652 | 0.745 | 0.756 | 0.711 |
| Zero-shot | **CAZI-MBN** | **0.791** | **0.839** | **0.857** | **0.723** |

**PINNACLE (Protein–Protein, 7,044 nodes, 12 interaction types)**:

| Setting | Model | AUROC | AUPRC |
|---------|-------|-------|-------|
| Transductive | xCAPT5 | 0.781 | 0.804 |
| Transductive | **CAZI-MBN** | **0.831** | **0.845** |
| Zero-shot | xCAPT5 | 0.785 | 0.791 |
| Zero-shot | **CAZI-MBN** | **0.812** | **0.820** |

### Ablation Study

Average performance degradation upon removal of each module (across 5 datasets):
- **LLMs**: AUROC drop of 15–20% (largest contribution)
- **CAE**: AUROC drop of 7–10%
- **UGT**: AUROC drop of 5–8%
- **MoE**: AUROC drop of 5–8%

### Key Findings

1. CAZI-MBN consistently outperforms all 13 baselines on AUROC and AUPRC across all 5 datasets, with improvements of **3.1–20.4%**.
2. Performance degradation under the zero-shot setting is modest, demonstrating that knowledge distillation effectively transfers topology knowledge into sequence space.
3. Domain-specific LLM embeddings are the most critical component (15–20% contribution), underscoring the importance of sequence-level semantic information for biological interaction prediction.
4. Performance remains stable on rare interaction types, indicating that MoE effectively handles label imbalance.
5. In an IBD case study, the model recovers 82.7% (DGIdb) and 85.7% (TRRUST) of known interactions.

## Highlights & Insights

1. **First systematic zero-shot MBN prediction framework**: Knowledge distillation compresses topology information into sequence space, enabling prediction for entirely unseen entities.
2. **Elegant UGT design**: SVD decomposition and higher-order smoothing of the supra-adjacency matrix jointly encode topology, multiplexity, and higher-order connectivity in a single step.
3. **Principled modular design**: The sequence → topology → cross-layer → prediction pipeline allows each component's contribution to be isolated and ablated.
4. **Five high-quality benchmark datasets constructed from scratch**: Filling the gap in standardized MBN evaluation.

## Limitations & Future Work

1. **3D structural data** (protein/compound three-dimensional structure) is not incorporated, limiting fine-grained modeling.
2. Zero-shot generalization is demonstrated only at the entity level; generalization to entirely novel interaction types remains unverified.
3. The scalability of multi-layer attention computation in the CAE module on large-scale networks requires further investigation.
4. Selection of domain-specific LLMs requires prior knowledge, and the generality of the framework depends on the availability of suitable pretrained models.
5. Transfer learning across cross-species networks is not explored.

## Related Work & Insights

- **Single-layer GNNs (GCN/GraphSAGE)**: Limited capacity to handle multiplex data.
- **Multiplex network models (DMGI/HDMI)**: Preserve intra- and inter-layer dependencies but exhibit weak multimodal integration.
- **Knowledge graph methods**: Flattening of layer structure leads to loss of interaction-type specificity.
- **Knowledge distillation** is widely applied in NLP and CV but remains largely unexplored in MBNs and zero-shot generalization settings.
- Key insight: The topology-to-sequence distillation paradigm is generalizable to other graph learning tasks requiring zero-shot generalization.

## Rating

- **Novelty**: ★★★★☆ — Zero-shot prediction in multiplex networks is a novel problem; the framework design is systematic and comprehensive.
- **Technical Depth**: ★★★★☆ — The combination of UGT + CAE + MoE + distillation demonstrates substantial technical sophistication.
- **Experimental Soundness**: ★★★★★ — Coverage of 5 datasets × 13 baselines × 2 settings is thorough.
- **Practical Value**: ★★★★☆ — Direct applicability to drug discovery and precision medicine.
- **Clarity**: ★★★☆☆ — The large number of modules makes the presentation somewhat fragmented, with key details scattered across the appendix.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Boomerang Distillation Enables Zero-Shot Model Size Interpolation](boomerang_distillation_enables_zero-shot_model_size_interpolation.md)
- [\[ICLR 2026\] Topology and Geometry of the Learning Space of ReLU Networks: Connectivity and Size](topology_and_geometry_of_the_learning_space_of_relu_networks_connectivity_and_si.md)
- [\[ICCV 2025\] Perspective-Aware Teaching: Adapting Knowledge for Heterogeneous Distillation](../../ICCV2025/model_compression/perspective-aware_teaching_adapting_knowledge_for_heterogeneous_distillation.md)
- [\[ICLR 2026\] Parallel Token Prediction for Language Models](parallel_token_prediction_for_language_models.md)
- [\[ICLR 2026\] Token Distillation: Attention-Aware Input Embeddings for New Tokens](token_distillation_attention-aware_input_embeddings_for_new_tokens.md)

</div>

<!-- RELATED:END -->
