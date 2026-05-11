---
title: >-
  [Paper Note] NTSFormer: A Self-Teaching Graph Transformer for Multimodal Isolated Cold-Start Node Classification
description: >-
  [AAAI 2026][Graph Learning][Cold-start node classification] This paper proposes NTSFormer (Neighbor-to-Self Graph Transformer), a unified Graph Transformer framework that implements a **self-teaching paradigm** via a col…
tags:
  - "AAAI 2026"
  - "Graph Learning"
  - "Cold-start node classification"
  - "Graph Transformer"
  - "self-teaching"
  - "missing modalities"
  - "mixture of experts"
date: 2026-05-08
content_hash: 343f71b549463949
---

# NTSFormer: A Self-Teaching Graph Transformer for Multimodal Isolated Cold-Start Node Classification

**Conference**: AAAI 2026
**arXiv**: [2507.04870](https://arxiv.org/abs/2507.04870)
**Code**: [https://github.com/CrawlScript/NTSFormer](https://github.com/CrawlScript/NTSFormer)
**Area**: Graph Learning / Graph Transformer
**Keywords**: Cold-start node classification, Graph Transformer, self-teaching, missing modalities, mixture of experts

## TL;DR

This paper proposes NTSFormer (Neighbor-to-Self Graph Transformer), a unified Graph Transformer framework that implements a **self-teaching paradigm** via a cold-start attention mask. Within the same model, a "student" prediction is derived solely from the node's own features while a "teacher" prediction leverages neighbor information, enabling end-to-end self-teaching without degrading to an MLP. The framework handles missing modalities in multimodal graphs through MoE input projection and multimodal graph pre-computation.

## Background & Motivation

### Problem Definition

**Multimodal isolated cold-start node classification**: Newly added nodes in a multimodal graph (e.g., new users in a social network) face two simultaneous challenges:

**Structural isolation**: The node has no edges, making graph structure inaccessible to GNNs.

**Missing modalities**: Certain data (e.g., text or images) may be absent (e.g., a new user has a profile picture but no bio).

### Limitations of Prior Work

**GNN failure under cold-start**: Experiments show that GNN models including GraphSAGE, MMGCN, and MGAT perform extremely poorly in isolated cold-start scenarios, **even underperforming simple MLPs**. GNNs are trained with graph structure but evaluated on fully isolated nodes, inducing a severe train-test distribution shift.

**Capacity bottleneck of MLP-student methods**:
- Methods such as GLNN, SGKD, and SimMLP adopt a **teacher-student paradigm**: a GNN teacher distills structural knowledge into an MLP student.
- The MLP student avoids distribution shift since it operates without graph structure at both train and test time.
- However, **limited MLP capacity** makes it difficult to jointly handle missing modalities in multimodal settings.

### Core Idea

**Degrading to an MLP is unnecessary.** By designing a cold-start attention mask within a unified Graph Transformer, the same model can simultaneously produce predictions based only on self-features (simulating cold-start) and predictions leveraging neighbor information (providing supervision signals), achieving end-to-end self-teaching that fully exploits Transformer capacity to handle missing modalities.

## Method

### Overall Architecture

NTSFormer consists of three key modules:
1. **Multimodal Graph Pre-computation**: A one-time transformation of multi-hop neighbor information into fixed-length token sequences.
2. **MoE Input Projection**: Dynamic routing of different token types to expert networks.
3. **Neighbor-to-Self Teaching via Cold-Start Mask**: A cold-start attention mask that separates the student and teacher contexts.

### Key Designs

#### 1. **Multimodal Graph Pre-computation**: Converting graph structure into Transformer input sequences

**Objective**: Organize multi-hop neighbor information by modality and hop depth into fixed-length token sequences suitable for Transformer processing.

**Procedure**:
- Text features $X^{(t)}$ and visual features $X^{(v)}$ are zero-padded to align dimensions to $d_{in} = \max(d_t, d_v)$.
- $K$-hop neighbor aggregations are computed: $\{\hat{A}^k X^{(t)} | k=1,...,K\}$ and $\{\hat{A}^k X^{(v)} | k=1,...,K\}$.

**Token sequence construction**:
- **Self tokens**:

$$\mathcal{X}_{\text{self}} = [X^{(t)} \text{ or } \langle\text{MISS}\rangle, \ X^{(v)} \text{ or } \langle\text{MISS}\rangle, \ \langle\text{CLS}_S\rangle]$$

Missing modalities are replaced by a learnable $\langle\text{MISS}\rangle$ placeholder. During training, modalities are randomly replaced with probability $p_{\text{miss}}$ to simulate missingness.

- **Neighbor tokens**:

$$\mathcal{X}_{\text{nbr}} = [\hat{A}X^{(t)}, \ \hat{A}X^{(v)}, \ ..., \ \hat{A}^K X^{(t)}, \ \hat{A}^K X^{(v)}, \ \langle\text{CLS}_T\rangle]$$

- Full sequence $S = \mathcal{X}_{\text{self}} \oplus \mathcal{X}_{\text{nbr}}$, with length $L = 2K + 4$.

**Key advantage**: Pre-computation is performed once on CPU without gradients, enabling standard mini-batch training thereafter.

#### 2. **MoE Input Projection**: Differentiated processing of heterogeneous tokens

**Design Motivation**: Input tokens originate from diverse sources (self/neighbor, different modalities, special tokens); a shared MLP projection loses discriminative information.

**Position-aware gating network**:
- Token features and one-hot position vectors are concatenated: $\tilde{S}[i] = [S[i] \| \mathbf{1}_N e_i^\top]$
- Gating scores are computed: $\gamma = \text{softmax}(\tilde{S}[i] \cdot W_{\text{gate}}) \in \mathbb{R}^{N \times M}$
- Each token selects the top-$\hat{k}$ experts; outputs are combined as a weighted sum:

$$S'_{\text{RE}}[i]_j = \sum_{m=1}^{M} \mathcal{T}(S[i])_{j,m} \cdot \gamma_{j,m} \cdot \text{LN}(\text{MLP}_{\text{RE}_m}(S[i]_j))$$

- A shared expert output is added: $S'[i]_j = S'_{\text{RE}}[i]_j + \text{LN}(\text{MLP}_{\text{SE}}(S[i]_j))$
- MoE load-balancing loss: $\mathcal{L}_{\text{MoE}} = \sum_{m=1}^{M} P_m \cdot f_m$

#### 3. **Self-Teaching via Cold-Start Attention Mask**: Core Innovation

**Key problem**: Standard self-attention allows all tokens to attend to one another, which causes neighbor information to leak into $\langle\text{CLS}_S\rangle$, violating the cold-start assumption.

**Cold-start attention mask design**:

$$\mathcal{M} = \begin{pmatrix} \mathbf{1}^{3 \times 3} & \mathbf{0}^{3 \times (L-3)} \\ \mathbf{1}^{(L-3) \times 3} & \mathbf{1}^{(L-3) \times (L-3)} \end{pmatrix}$$

- Self tokens (first 3 positions) **can only attend to one another**, with no access to neighbor tokens whatsoever.
- Neighbor tokens can attend to all tokens, including self tokens.
- As a result, the representation of $\langle\text{CLS}_S\rangle$ is derived purely from self-features, suitable for isolated cold-start inference; $\langle\text{CLS}_T\rangle$ integrates the full context and serves as the teacher supervision signal.

**Two prediction branches**:
- **Student prediction** (from $\langle\text{CLS}_S\rangle$): $Z_S = \text{softmax}(\text{MLP}_S(H[:,3]))$
- **Teacher prediction** (averaging $\langle\text{CLS}_T\rangle$ and the second-to-last token): $Z_T = \frac{1}{2}(Z_{T_1} + Z_{T_2})$

**Self-teaching loss** (KL divergence with stop-gradient on the teacher):

$$\mathcal{L}_{\text{ST}} = \text{KL}(\text{stopgrad}(Z_T) \| Z_S)$$

### Loss & Training

Overall training objective:

$$\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda \mathcal{L}_{\text{ST}} + \gamma \mathcal{L}_{\text{MoE}}$$

- $\mathcal{L}_{\text{CE}}$: Cross-entropy loss on teacher predictions.
- $\mathcal{L}_{\text{ST}}$: Self-teaching KL divergence loss ($\lambda=1.0$).
- $\mathcal{L}_{\text{MoE}}$: MoE load-balancing loss ($\gamma=0.1$).
- Optimizer: AdamW, learning rate $2 \times 10^{-3}$.
- During cold-start inference, only $\mathcal{X}_{\text{self}}$ is used, and predictions are drawn from $\langle\text{CLS}_S\rangle$.

## Key Experimental Results

### Experimental Setup
- **Datasets**: Movies (16K nodes), Ele-fashion (97K nodes), Goodreads-NC (685K nodes).
- **Splits**: 20% labeled training, 60% unlabeled training (participating in message passing), 10% validation / 10% test (fully isolated).
- **Missing modality setting**: The test set is evenly divided into three subsets: Text-Miss, Visual-Miss, and No-Miss.
- **Hidden dimension**: 512, $K=2$, 6 routing experts + 1 shared expert, 2-layer Transformer.

### Main Results

| Method | Movies All | Ele-fashion All | Goodreads-NC All | Notes |
|------|-----------|----------------|-----------------|------|
| MLP | 41.85 | 75.15 | 55.56 | Baseline |
| GraphSAGE | 39.02 | 66.56 | 41.65 | GNN, **worse than MLP** |
| MMGCN | 40.77 | 68.52 | 43.69 | Multimodal GNN, poor under cold-start |
| GLNN | 43.04 | 74.41 | 54.01 | MLP-student |
| MUSE | 43.44 | **80.66** | 48.49 | Handles missing modalities |
| **NTSFormer** | **46.12** | **83.37** | **61.58** | **Best overall** |

NTSFormer **consistently and substantially outperforms** all baselines across all datasets and all missing modality settings.

### Ablation Study

| Configuration | Movies | Ele-fashion | Goodreads-NC | Note |
|------|--------|-------------|-------------|------|
| NTSFormer (full) | **46.12** | **83.37** | **61.58** | Full model |
| w/o MMPre | Drop | Drop | Drop | Multimodal pre-computation is effective |
| w/o MoE | Drop | Drop | Drop | MoE projection outperforms shared linear |
| w/o SelfTeach | Significant drop | Significant drop | Significant drop | **Self-teaching is essential**; removing it degrades performance substantially |

Effect of number of Transformer layers (accuracy % under All setting):

| Layers $L^{(tf)}$ | Movies | Ele-fashion | Goodreads-NC |
|---|--------|-------------|-------------|
| 1 | 45.22 | 83.54 | 61.36 |
| 2 | **46.12** | 83.37 | 61.58 |
| 3 | 45.27 | 83.51 | **61.68** |
| 4 | 45.32 | **83.67** | 61.51 |

### Key Findings

1. **GNNs universally underperform MLPs under isolated cold-start**: GraphSAGE falls below MLP on all three datasets, validating the severe impact of cold-start on GNNs.
2. **Self-teaching outperforms MLP-student**: Removing self-teaching (reducing to a two-branch MLP-student setup) causes significant performance degradation, demonstrating the capacity advantage of Transformers.
3. **Superior missing modality handling**: NTSFormer leads on both Text-Miss and Visual-Miss subsets; on Goodreads-NC Text-Miss it achieves 50.99% versus MLP's 40.25%.
4. **Training efficiency**: On the large-scale Goodreads-NC dataset, NTSFormer requires only 260 seconds, significantly faster than MUSE (1437s) and GLNN (778s), owing to one-time pre-computation.
5. **MoE expert count**: Performance degrades when $M<3$, indicating that diverse token types benefit from multiple specialized expert networks.

## Highlights & Insights

1. **Elegant self-teaching design**: A single attention mask matrix achieves teacher-student role separation within the same Transformer, offering a cleaner and more efficient alternative to conventional two-stage teacher-student methods.
2. **Simplicity of the cold-start mask**: The core idea is extremely straightforward — preventing self tokens from attending to neighbor tokens — yet the effect is substantial. This binary mask design introduces virtually no additional computational overhead.
3. **Scalability through one-time pre-computation**: Converting graph structure into regular tensors eliminates repeated message passing during training, achieving a training time of only 260 seconds on Goodreads-NC with 685K nodes.
4. **Position-aware MoE design**: Concatenating one-hot position vectors enables the gating network to be aware of each token's semantic role (self/neighbor, text/visual), which is more principled than a naive shared projection.
5. The **$\langle\text{MISS}\rangle$ placeholder combined with random dropout during training** naturally equips the model with robust missing modality handling.

## Limitations & Future Work

1. **Only two modalities (text + visual) are supported**; extending to additional modalities requires modifications to the pre-computation and mask design.
2. **Pre-computation assumes a static graph during training**, making the approach unsuitable for dynamic graph settings (acknowledged by the authors).
3. **Teacher signal quality depends on label homophily among neighbors**, which may limit effectiveness on heterophilic graphs.
4. **Dataset scale is relatively modest** (up to 685K nodes); efficiency and effectiveness on larger-scale graphs remain to be validated.
5. **Future directions**: dynamic graph adaptation, adaptive mask design, integration with LLM-based features, and extension to semi-supervised settings.

## Related Work & Insights

- **NAGphormer** pre-computes neighbor features as Transformer inputs but only handles single-modality graphs; NTSFormer extends this to multimodal settings with MoE.
- The teacher-student paradigm of **GLNN/SGKD** inspired the self-teaching design, but NTSFormer achieves end-to-end training via attention masking.
- The cold-start attention mask concept is transferable to **recommender systems** (new users/items) and **knowledge graph completion** (new entities).

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The self-teaching paradigm via cold-start masking is concise and effective, representing a genuine contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three datasets, comprehensive ablations, and efficiency analysis, though dataset scale is somewhat limited.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Motivation is developed progressively: GNN failure → MLP-student capacity bottleneck → self-teaching Graph Transformer.
- **Value**: ⭐⭐⭐⭐⭐ — Open-source, training-efficient, generalizable methodology, scalable via one-time pre-computation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MoToRec: Sparse-Regularized Multimodal Tokenization for Cold-Start Recommendation](motorec_sparse-regularized_multimodal_tokenization_for_cold-start_recommendation.md)
- [\[AAAI 2026\] Posterior Label Smoothing for Node Classification](posterior_label_smoothing_for_node_classification.md)
- [\[AAAI 2026\] GT-SNT: A Linear-Time Transformer for Large-Scale Graphs via Spiking Node Tokenization](gt-snt_a_linear-time_transformer_for_large-scale_graphs_via_spiking_node_tokeniz.md)
- [\[AAAI 2026\] Self-Adaptive Graph Mixture of Models](self-adaptive_graph_mixture_of_models.md)
- [\[AAAI 2026\] Beyond Fixed Depth: Adaptive Graph Neural Networks for Node Classification Under Varying Homophily](beyond_fixed_depth_adaptive_graph_neural_networks_for_node_classification_under_.md)

</div>

<!-- RELATED:END -->
