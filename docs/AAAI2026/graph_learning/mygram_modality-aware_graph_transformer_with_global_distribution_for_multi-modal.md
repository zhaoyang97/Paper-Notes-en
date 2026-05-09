---
title: >-
  [Paper Note] MyGram: Modality-aware Graph Transformer with Global Distribution for Multi-modal Entity Alignment
description: >-
  [AAAI 2026][Graph Learning][Multi-modal Entity Alignment] This paper proposes MyGram, which captures deep structural contextual information within each modality via a Modality-aware Graph Diffusion (MGD) module, and introduces a global distribution alignment loss (Gram Loss) based on the determinant of the Gram matrix to enforce cross-modal semantic consistency in high-dimensional space, achieving more robust multi-modal entity alignment.
tags:
  - AAAI 2026
  - Graph Learning
  - Multi-modal Entity Alignment
  - Knowledge Graph
  - Gram Matrix
  - Graph Diffusion Learning
  - Transformer
date: 2026-05-08
content_hash: 1f4498c0f635e8d3
---

# MyGram: Modality-aware Graph Transformer with Global Distribution for Multi-modal Entity Alignment

**Conference**: AAAI 2026
**arXiv**: [2601.11885](https://arxiv.org/abs/2601.11885)
**Code**: [https://github.com/HubuKG/MyGram](https://github.com/HubuKG/MyGram)
**Area**: Graph Learning / Knowledge Graphs
**Keywords**: Multi-modal Entity Alignment, Knowledge Graph, Gram Matrix, Graph Diffusion Learning, Transformer

## TL;DR

This paper proposes MyGram, which captures deep structural contextual information within each modality via a Modality-aware Graph Diffusion (MGD) module, and introduces a global distribution alignment loss (Gram Loss) based on the determinant of the Gram matrix to enforce cross-modal semantic consistency in high-dimensional space, achieving more robust multi-modal entity alignment.

## Background & Motivation

### State of the Field
Multi-modal knowledge graphs (MMKGs) enhance entity semantic representations by integrating multiple modalities such as text and images. However, MMKGs from different sources often represent the same real-world entity inconsistently. **Multi-modal Entity Alignment (MMEA)** aims to identify equivalent entities across different MMKGs that refer to the same real-world object, and is a core task in knowledge fusion.

### Two Key Challenges in Existing Methods

**Limitations of contrastive learning**: Existing methods primarily adopt intra-modal contrastive learning frameworks, optimizing feature distances between positive and negative entity pairs. However, these methods **neglect distributional discrepancies between modalities in the global feature space**, focusing only on local point-to-point alignment and failing to guarantee global cross-modal feature consistency.

**Shallow feature interference**: Existing methods overlook the **structural contextual information** within each modality, making it difficult to distinguish entities that are visually or attributively similar but semantically distinct. A typical example is Anne Hathaway and Kirsten Dunst, whose visual and attribute features are highly similar and thus interfere with alignment, yet accurate alignment can still be achieved by leveraging structural information.

### Mechanism
- Obtain modality features rich in structural context via graph diffusion learning (addressing shallow feature interference)
- Use the volume of a high-dimensional parallelepiped constructed from the Gram matrix as a geometric metric to constrain cross-modal distributional consistency (addressing the absence of global alignment)

## Method

### Overall Architecture

MyGram consists of three main modules:
1. **Multi-modal Feature Extraction**: Extracts uni-modal embeddings independently from each modality
2. **Modality-aware Diffusion Learning**: Obtains modality features rich in structural context via graph convolutional diffusion
3. **Multi-modal Training and Learning**: Establishes alignment between equivalent entities using Gram Loss

### Key Designs

#### 1. **Multi-modal Feature Extraction**: Builds independent embeddings for each modality

- **Structural modality**: Aggregates neighbors using a Relation-Reflection Graph Attention Network (RRGAT) to preserve relational structure: $\mathbf{h}_g = RRGAT(\omega, \mathbf{M}_g, x_g)$
- **Relation/Attribute/Visual modalities**: Project into a shared feature space via linear transformation: $\mathbf{h}_m = \mathbf{W}_m x_m + b_m, \quad m \in \{r, a, v\}$
    - Attributes and relations use bag-of-words feature representations
    - Visual features are extracted using a pre-trained image encoder (VGG-16)

#### 2. **Modality-aware Graph Convolutional Diffusion (MGD) Module**: Captures deep structural context

**Design Motivation**: Conventional methods ignore the modal information of neighboring entities, and relying solely on shallow features makes the model susceptible to interference from similar but distinct entities. MGD performs multi-hop neighborhood information aggregation independently for each modality.

**Graph Convolutional Diffusion Process**:
- Construct a normalized adjacency matrix with self-loops: $\hat{A} = D^{-1/2}(A+I)D^{-1/2}$
- Iterative propagation ($k$ rounds) with residual connections to prevent over-smoothing:

$$\mathbf{H}_m^{(l)} = \beta \cdot \hat{A}\mathbf{H}_m^{(l-1)} + \alpha \cdot \mathbf{H}_m^{(0)}, \quad (l=1,2,...,k)$$

- The final output is normalized and subjected to Dropout: $\mathbf{H}_m = \text{Dropout}\left(\frac{1}{\gamma}\mathbf{H}_m^{(k)}\right)$, where $\gamma = \beta^k + \alpha \sum_{c=0}^{k-1}\beta^c$ prevents gradient explosion.

**Transformer Self-attention Fusion**:
- Multi-head cross-attention is applied to the diffused modality features:

$$head_m^i = \beta_m^{(i)} V_m^{(i)}, \quad \beta_m = \text{softmax}\left(\frac{Q_m^\top K_m}{\sqrt{d_h}}\right)$$

- Cross-modal weights are computed for adaptive fusion:

$$\omega_m = \frac{\exp\left(\sum_{j \in M}\sum_{i=0}^{N_h}\beta_{m,j}^{(i)} / \sqrt{|M| \times N_h}\right)}{\sum_{k \in M} \exp\left(\sum_{k \in M}\sum_{i=0}^{N_h}\beta_{m,k}^{(i)} / \sqrt{|M| \times N_h}\right)}$$

- Joint embedding: $\mathbf{H}_o = \mathbf{H}_g \oplus_{m \in M}[\omega_m \mathbf{H}_m]$

#### 3. **Gram-based Global Distribution Alignment**: Geometric constraints via high-dimensional volume

**Core Idea**: The volume of a 4-dimensional parallelepiped formed by multi-modal vectors is used as a geometric metric for cross-modal consistency. A smaller volume indicates that the embeddings reside in a more compact subspace, reflecting stronger cross-modal semantic consistency.

**Implementation**:
- Top-K candidate entities are first selected via a similarity matrix
- A multi-modal matrix is constructed from the structural features of source entities and the visual/attribute/relation features of target entities: $\mathcal{M} = [\tilde{\mathbf{H}}_g^s, \tilde{\mathbf{H}}_v^t, \tilde{\mathbf{H}}_a^t, \tilde{\mathbf{H}}_r^t] \in \mathbb{R}^{d_h \times 4}$
- The Gram matrix is computed as $G = \mathcal{M}^\top \mathcal{M} \in \mathbb{R}^{4 \times 4}$
- Volume of the 4-dimensional parallelepiped: $Vol = \sqrt{|\det(G)| + \epsilon}$
- Gram Loss (sparse contrastive loss):

$$\mathcal{L}_{Gram} = -\frac{1}{M}\sum_{m=1}^{M} \log \frac{\exp(-Vol^{(m,p)}/\tau)}{\sum_{k=1}^{K} \exp(-Vol^{(m,k)}/\tau)}$$

**Distinction from Conventional Methods**: Conventional methods optimize point-to-point feature distances (local), whereas Gram Loss constrains the overall geometric relationship among multi-modal vectors in high-dimensional space (global), promoting cross-modal semantic structural consistency.

### Loss & Training

The total loss combines an InfoNCE contrastive loss and a weighted Gram Loss:

$$\mathcal{L}_{total} = \mathcal{L}_{InfoNCE} + \lambda \mathcal{L}_{Gram}$$

where InfoNCE maximizes the similarity of truly aligned entity pairs while pushing apart negative samples:

$$\mathcal{L}_{InfoNCE} = \sum_{(e_i,e_j) \in \mathcal{S}} -\log \frac{\exp(\text{sim}(e_i, e_j)/\mathcal{T})}{\sum_{e_k \in \mathcal{N}_i^{neg}} \exp(\text{sim}(e_i, e_k)/\mathcal{T})}$$

## Key Experimental Results

### Experimental Setup
- **Datasets**:
    - Cross-KG: FB15K-DB15K, FB15K-YG15K (seed ratios: 20%/50%/80%)
    - Bilingual: DBP15K (ZH-EN, JA-EN, FR-EN, seed ratio 30%)
- **Metrics**: Hits@1, Hits@10, MRR
- **Image features**: VGG-16, $d_v = 4096$
- **Hidden dimension**: 300, self-attention heads: 5, Transformer intermediate layer: 400

### Main Results

| Dataset | Metric | MyGram | Prev. SOTA | Gain |
|--------|------|--------|---------|------|
| FBDB15K (80%) | Hit@1 | **0.842** | IBMEA: 0.821 | +2.6% |
| FBDB15K (80%) | MRR | **0.879** | SimDiff: 0.865 | +1.6% |
| FBYG15K (80%) | Hit@1 | **0.783** | PMF: 0.756 | +3.6% |
| FBYG15K (20%) | Hit@1 | **0.629** | SimDiff: 0.530 | **+18.7%** |
| DBP15K ZH-EN | Hit@1 | **0.833** | DESAlign: 0.810 | +2.8% |
| DBP15K JA-EN | Hit@1 | **0.836** | DESAlign: 0.811 | +3.1% |
| DBP15K FR-EN | Hit@1 | **0.869** | DESAlign: 0.826 | **+5.2%** |

Maximum Hit@1 improvement of 4.8% on FBDB15K, 9.9% on FBYG15K, and 4.3% on DBP15K.

### Ablation Study

| Configuration | FBDB15K MRR | FBDB15K Hit@1 | FBYG15K MRR | FBYG15K Hit@1 | Notes |
|------|-------------|---------------|-------------|---------------|------|
| MyGram (full) | 0.879 | 0.842 | 0.836 | 0.783 | Full model |
| w/o Relation | 0.842 | 0.822 | 0.811 | 0.761 | **Largest drop; relation modality most critical** |
| w/o Attributes | 0.859 | 0.834 | 0.818 | 0.768 | Significant attribute contribution |
| w/o Image | 0.851 | 0.829 | 0.824 | 0.772 | Visual information beneficial |
| w/o MGD | Significant drop | Significant drop | Significant drop | Significant drop | MGD module is critical |
| w/o Gram | Drop | Drop | Drop | Drop | Gram Loss is effective |

### Key Findings

1. **Relation modality is the most important**: Removing relational information causes the largest performance drop, indicating that structural information plays a central role in multi-modal entity alignment.
2. **Pronounced advantage in low-resource settings**: Under low-resource experiments with seed ratios of 5%–30%, MyGram consistently outperforms MEAformer and SimDiff.
3. **Case study**: In the alignment of the "Shanghai" entity, MEAformer and PMF rank the correct entity poorly, while MyGram achieves accurate matching, demonstrating its capacity to capture deep-level information.
4. **MGD contributes more than Gram Loss**: The modality-aware graph convolutional diffusion module has a more significant impact on performance than Gram Loss.

## Highlights & Insights

1. **Global alignment from a geometric perspective**: Using the Gram matrix determinant as a multi-modal consistency metric is elegant—a volume of zero implies that all modality vectors are linearly dependent (perfect consistency), and larger volume indicates greater inconsistency. This provides a more global constraint than pairwise comparisons.
2. **Complementarity of graph diffusion and Transformer**: Graph diffusion captures local structural context, while the Transformer captures global cross-modal dependencies; the two are complementary.
3. **Adaptive computation of modality weights**: Determining modality weights by normalizing attention scores is more flexible than manual assignment.
4. **Comprehensive experimental coverage**: 5 datasets, 9 groups of comparative experiments, low-resource analysis, and case studies.

## Limitations & Future Work

1. **Using VGG-16 as the image encoder** is relatively outdated; stronger multi-modal encoders such as CLIP could be explored.
2. **The 4-dimensional parallelepiped is a hardcoded design**; if the number of modalities changes, the design must be revised accordingly.
3. **A fixed hidden dimension of 300** may limit the model's capacity to capture complex semantics.
4. **Deep textual representations are not explored** (attributes and relations are represented using only bag-of-words features).
5. **Potential improvements**: Incorporating LLMs to enhance textual understanding (noted in the paper), and exploring incremental alignment in dynamic knowledge graph scenarios.

## Related Work & Insights

- **SimDiff** employs diffusion-enhanced alignment, and **IBMEA** uses information bottleneck to suppress spurious cues, but both remain at the level of local alignment.
- The volume-constraint idea based on the Gram matrix is transferable to scenarios such as **multi-modal retrieval** and **cross-modal generation quality assessment**.
- The modality-aware design of graph diffusion learning is applicable to **multi-modal recommendation** and **social network analysis**.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The geometric perspective of Gram Loss is novel, though the MGD component is relatively conventional.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 5 datasets, 9 experimental groups, modality ablation + component ablation + low-resource analysis + case studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, though notation in formulas is occasionally inconsistent.
- **Value**: ⭐⭐⭐⭐ — Open-source code and feasible methodology, though feature extractors are somewhat dated.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] GSAP-ERE: Fine-Grained Scholarly Entity and Relation Extraction Focused on Machine Learning](gsap-ere_fine-grained_scholarly_entity_and_relation_extraction_focused_on_machin.md)
- [\[ICLR 2026\] Relational Graph Transformer](../../ICLR2026/graph_learning/relational_graph_transformer.md)
- [\[AAAI 2026\] S-DAG: A Subject-Based Directed Acyclic Graph for Multi-Agent Heterogeneous Reasoning](s-dag_a_subject-based_directed_acyclic_graph_for_multi-agent.md)
- [\[AAAI 2026\] NTSFormer: A Self-Teaching Graph Transformer for Multimodal Isolated Cold-Start Node Classification](ntsformer_a_self-teaching_graph_transformer_for_multimodal_isolated_cold-start_n.md)
- [\[AAAI 2026\] PCoKG: Personality-aware Commonsense Reasoning with Debate](pcokg_personality-aware_commonsense_reasoning_with_debate.md)

</div>

<!-- RELATED:END -->
