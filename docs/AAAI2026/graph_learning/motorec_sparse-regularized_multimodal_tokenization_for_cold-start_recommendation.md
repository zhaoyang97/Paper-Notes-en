---
title: >-
  [Paper Note] MoToRec: Sparse-Regularized Multimodal Tokenization for Cold-Start Recommendation
description: >-
  [AAAI 2026][Graph Learning][Cold-start recommendation] MoToRec reformulates multimodal recommendation as a discrete semantic tokenization task. By leveraging a sparsely-regularized Residual Quantization VAE (RQ-VAE), raw multimodal features are transformed into composable discrete semantic codes. Combined with adaptive rarity amplification and a hierarchical multi-source graph encoder, the framework effectively addresses the item cold-start problem.
tags:
  - AAAI 2026
  - Graph Learning
  - Cold-start recommendation
  - multimodal recommendation
  - discrete semantic tokenization
  - residual quantization VAE
  - graph neural networks
date: 2026-05-08
content_hash: 6615a735d49517bb
---

# MoToRec: Sparse-Regularized Multimodal Tokenization for Cold-Start Recommendation

**Conference**: AAAI 2026
**arXiv**: [2602.11062](https://arxiv.org/abs/2602.11062)
**Code**: N/A
**Area**: Graph Learning / Recommender Systems
**Keywords**: Cold-start recommendation, multimodal recommendation, discrete semantic tokenization, residual quantization VAE, graph neural networks

## TL;DR

MoToRec reformulates multimodal recommendation as a discrete semantic tokenization task. By leveraging a sparsely-regularized Residual Quantization VAE (RQ-VAE), raw multimodal features are transformed into composable discrete semantic codes. Combined with adaptive rarity amplification and a hierarchical multi-source graph encoder, the framework effectively addresses the item cold-start problem.

## Background & Motivation

### State of the Field
Graph neural networks (GNNs) have become a cornerstone of modern recommender systems, yet their success relies heavily on dense historical interaction data. In data-sparse scenarios—particularly the **item cold-start problem** (new items with few or no interaction histories)—GNN performance degrades sharply.

### Limitations of Prior Work
Multimodal information (visual and textual) offers a promising avenue for alleviating cold-start, but existing methods share common shortcomings:

**Semantic Fog**: Existing methods perform multimodal alignment in high-dimensional continuous spaces, essentially mapping a concept such as "red T-shirt" from pixel vectors and text vectors to a single coherent point in high-dimensional space. This process is highly sensitive to noise and inherently unreliable.

**Evolution of Prior Approaches**: From the simple concatenation in VBPR, modality-specific graphs in MMGCN, item–item semantic graphs in LATTICE, to contrastive learning in FREEDOM/BM3—despite architectural diversity, all approaches **fundamentally perform noisy alignment in continuous spaces**.

**OOD Representation Problem**: Even when LLMs are used as feature extractors, aligning these noisy continuous embeddings still produces suboptimal out-of-distribution (OOD) representations, particularly for cold-start items.

### Core Idea
**Discrete representations are superior to continuous alignment.** The authors propose transforming multimodal features into structured discrete token sequences, where each token represents a disentangled semantic concept (e.g., style: minimalist; color: red), fundamentally avoiding the alignment noise inherent to continuous spaces.

## Method

### Overall Architecture

MoToRec comprises three core components:
1. **Adaptive Rarity Amplification (ARA)**: Dynamically reweights learning signals to prioritize cold-start items.
2. **Sparsely-Regularized Multimodal Tokenizer**: Transforms raw multimodal features into discrete semantic codes via RQ-VAE.
3. **Hierarchical Multi-Source Graph Encoder**: Integrates semantic codes with collaborative signals.

As input, each item is associated with visual features $\mathbf{f}_i^v$ (from BEiT) and textual features $\mathbf{f}_i^t$ (from BGE). The objective is to learn user embeddings $\mathbf{e}_u$ and item embeddings $\mathbf{e}_i$, with relevance scores predicted via dot product.

### Key Designs

#### 1. **Adaptive Rarity Amplification (ARA)**: Mitigating popularity bias and amplifying cold-start learning signals

Recommendation datasets exhibit inherent popularity bias, causing models to underfit long-tail rare items. ARA addresses this through a degree-aware dynamic weighting scheme.

**Steps**:
- Compute the interaction degree of each item: $d_i = \sum_{u \in \mathcal{U}} R_{ui}$
- Set a domain threshold $\tau$; items with $d_i < \tau$ are marked as cold-start items.
- Define item weights (inverse-logarithmic weighting):

$$w_i = \begin{cases} (\log_2(d_i + 2))^{-1} & \text{if } c_i = 1 \text{ and } d_i > 0 \\ 1.0 & \text{otherwise} \end{cases}$$

**Design Motivation**: Inverse-logarithmic weighting compresses the degree range; the $+2$ offset stabilizes small values. This assigns higher weights to items with fewer interactions, while items with zero interactions (zero-shot) are not additionally upweighted—they rely on the overall learning quality of the content features.

#### 2. **Sparsely-Regularized RQ-VAE Tokenizer**: Transforming continuous features into interpretable discrete codes

This is the core module of MoToRec.

**Residual Quantization Process**:
- For each modality $m \in \{v, t\}$, encoder $E_m$ (an MLP) projects raw features into a latent space: $\mathbf{z}_{e,i}^m = E_m(\mathbf{f}_i^m)$
- $N_q$ quantizers are applied in cascade for progressive residual quantization:

$$q_i^{(k)} = \arg\min_{c \in C_m^{(k)}} \|r_i^{(k-1)} - c\|_2^2, \quad r_i^{(k)} = r_i^{(k-1)} - q_i^{(k)}$$

- The final quantized representation $\mathbf{z}_{q,i}^m = \sum_{k=1}^{N_q} \mathbf{q}_i^{(k)}$ is the sum of all quantized codebook vectors.

**Sparsity-Inducing Regularization (Key Innovation)**:
- To prevent the codebook from producing entangled representations, a KL divergence penalty encourages the aggregate posterior distribution of codebook usage to approximate a sparse Bernoulli prior with mean $\rho$:

$$\mathcal{L}_{\text{sparse}} = \sum_{j=1}^{K} \text{KL}(\rho \| \hat{\rho}_j) = \sum_{j=1}^{K} \left(\rho \log \frac{\rho}{\hat{\rho}_j} + (1-\rho) \log \frac{1-\rho}{1-\hat{\rho}_j}\right)$$

- Theoretical basis: The KL penalty promotes disentangled representations by minimizing mutual information between codebook activations, analogous to nonlinear independent component analysis in a discrete latent space.

**Tokenizer Training Objective**:

$$\mathcal{L}_{\text{RQ-VAE}}^m = \underbrace{\|\mathbf{f}_i^m - D_m(\mathbf{z}_{q,i}^m)\|_2^2}_{\text{reconstruction}} + \beta \underbrace{\|\mathbf{z}_{e,i}^m - \text{sg}(\mathbf{z}_{q,i}^m)\|_2^2}_{\text{commitment}} + \gamma \underbrace{\mathcal{L}_{\text{sparse}}}_{\text{sparsity}}$$

#### 3. **Hierarchical Multi-Source Graph Encoder**: Aligning semantic codes with collaborative preferences

**Intra-Modality Decoupled Propagation**: Three parallel decoupled propagation channels are maintained:
- Visual channel: initialized with quantized visual embeddings $\{\mathbf{z}_{q,i}^v\}$, capturing aesthetic preferences.
- Textual channel: initialized with quantized textual embeddings $\{\mathbf{z}_{q,i}^t\}$, learning item attributes.
- Collaborative channel: initialized with standard learnable ID embeddings, modeling pure collaborative signals.

Within each channel, $L$ layers of embedding refinement are performed using the LightGCN propagation rule: $\mathbf{E}^{(l+1)} = (\mathbf{D}^{-1/2}\tilde{\mathbf{A}}\mathbf{D}^{-1/2})\mathbf{E}^{(l)}$

**Cross-Source Fusion**: A hybrid fusion strategy is adopted:

$$\mathbf{e}_i^m = \alpha \cdot \text{CONCAT}(\mathbf{i}_v, \mathbf{i}_t) + (1-\alpha) \cdot \text{Attention}(\mathbf{i}_v, \mathbf{i}_t)$$

The hyperparameter $\alpha$ balances static feature preservation against dynamic context-aware reweighting; collaborative embeddings are subsequently integrated via a gated residual connection.

### Loss & Training

The final loss integrates four components:

$$\mathcal{L} = \mathcal{L}_{\text{BPR}} + \lambda_{cl} \mathcal{L}_{\text{CL}} + \lambda_{rq} \sum_{m \in \{v,t\}} \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} w_i \cdot \mathcal{L}_{\text{RQ-VAE},i}^m + \lambda_{reg} \|\Theta\|_2^2$$

- **BPR ranking loss**: Optimizes relative ranking between positive and negative items for each user.
- **InfoNCE contrastive loss**: Pulls augmented views of the same node closer while pushing away negatives.
- **Weighted RQ-VAE loss**: Cold-start items receive higher weight $w_i$, ensuring tokenization quality for these items.
- **L2 regularization**: Prevents overfitting.

## Key Experimental Results

### Experimental Setup
- **Datasets**: Amazon Baby, Sports, Clothing (all with sparsity >99.88%)
- **Evaluation Protocol**: 8:1:1 train/validation/test split; the cold-start group consists of test items with fewer than 10 interactions in the training set.
- **Metrics**: Recall@N and NDCG@N ($N = 10, 20$)

### Main Results

| Dataset | Metric | MoToRec | LGMRec (SOTA) | LPIC (SOTA) | Max Gain |
|---------|--------|---------|---------------|-------------|---------|
| Baby | R@20 | **0.1077** | 0.0989 | 0.0977 | +8.57% |
| Baby | N@20 | **0.0473** | 0.0430 | 0.0422 | +10.00% |
| Sports | R@20 | **0.1163** | 0.1068 | 0.1113 | +4.49% |
| Sports | N@20 | **0.0529** | 0.0477 | 0.0485 | +9.07% |
| Clothing | R@20 | **0.1014** | 0.0828 | 0.0928 | +7.76% |
| Clothing | N@20 | **0.0456** | 0.0371 | 0.0405 | +8.57% |

Compared to ID-only models (LightGCN), improvements reach up to **88%**. Under cold-start conditions, N@20 improves by **12.58%**.

### Ablation Study

| Configuration | Baby N@20 | Baby Cold N@20 | Sports N@20 | Clothing N@20 | Note |
|---------------|-----------|---------------|-------------|---------------|------|
| MoToRec (full) | 0.0473 | 0.0147 | 0.0529 | 0.0456 | Full model |
| w/o RQ-VAE | 0.0398 | 0.0092 | 0.0422 | 0.0362 | **Largest drop**, validates the core value of discrete tokenization |
| w/o ARA | 0.0437 | 0.0111 | 0.0466 | 0.0397 | Significant cold-start performance degradation |
| w/o Sparsity | 0.0430 | 0.0109 | 0.0455 | 0.0389 | Sparse constraint is critical for disentangled representations |
| w/o CL | 0.0455 | 0.0118 | 0.0515 | 0.0438 | Contrastive loss improves the embedding space |
| w/o HF | 0.0449 | 0.0120 | 0.0468 | 0.0401 | Hybrid fusion outperforms single-strategy fusion |

### Key Findings

1. **Removing RQ-VAE causes the most severe performance degradation** (cold-start N@20 drops from 0.0147 to 0.0092), directly validating the central claim that discrete semantic tokenization outperforms continuous feature mapping.
2. **Hyperparameter sensitivity varies by dataset**: The sparse Baby dataset favors moderate sparsity ($\gamma=0.05$) and a compact codebook ($K=512$), while the visually rich Clothing dataset requires lower sparsity ($\gamma=0.01$) and a larger codebook ($K=1024$).
3. **t-SNE visualizations** confirm that the full model learns a more organized semantic manifold; cold-start items are no longer isolated outliers but are seamlessly integrated into the structure.
4. **Case studies** verify that the codebook learns human-interpretable concepts—e.g., code `<c_121>` corresponds to "red" and `<a_34>` to "T-shirt"—enabling new items to be represented by composing these codes.

## Highlights & Insights

1. **Paradigm shift**: Reformulating recommendation from "continuous-space alignment" to "discrete semantic tokenization" is a highly novel and intuitively clear perspective. Discretization inherently confers denoising and interpretability advantages.
2. **Sparsity regularization promotes disentanglement**: KL divergence penalties drive codebook usage toward a sparse prior, achieving an independent component analysis effect in the discrete latent space.
3. **Three-channel decoupled propagation**: Early modality interference is avoided by separately preserving the semantic purity of visual preferences, textual attributes, and pure collaborative signals.
4. **Acceptable efficiency**: Training costs 11.33s/epoch, only 74% more overhead than LightGCN, with inference efficiency comparable to other high-performance models.

## Limitations & Future Work

1. **The cold-start threshold $\tau=10$ is a hard setting**; different datasets may require different thresholds, and no adaptive adjustment mechanism is provided.
2. **Only item cold-start is addressed**; user cold-start is not considered.
3. **Codebook size and the number of quantization levels require extensive hyperparameter tuning**, leading to high tuning costs in practical deployment.
4. **Validation is limited to Amazon datasets**; generalization to more diverse recommendation scenarios (e.g., news recommendation, short-video recommendation) remains untested.
5. **Promising future directions** include combining discrete tokenization with LLM-based recommender systems, exploring multi-codebook sharing mechanisms, and introducing discretized representations of user profiles.

## Related Work & Insights

- **VQ-Rec** pioneered vector quantization in recommendation but targeted sequential recommendation; MoToRec is the first to use it for learning disentangled multimodal compositional representations to address cold-start.
- **Meta-learning methods such as MeLU** are limited to few-shot scenarios and cannot handle zero-interaction zero-shot cold-start.
- The discrete tokenization paradigm is transferable to tasks such as **multimodal retrieval and cross-modal generation**.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The discretization perspective for cold-start recommendation is pioneering.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three datasets, comprehensive ablations, visualizations, and efficiency analysis are provided, though the dataset types are homogeneous.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Motivation is clearly articulated; the "semantic fog" metaphor is intuitive and precise.
- **Value**: ⭐⭐⭐⭐ — Training efficiency is acceptable, though hyperparameter tuning costs are relatively high.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] NTSFormer: A Self-Teaching Graph Transformer for Multimodal Isolated Cold-Start Node Classification](ntsformer_a_self-teaching_graph_transformer_for_multimodal_isolated_cold-start_n.md)
- [\[AAAI 2026\] GT-SNT: A Linear-Time Transformer for Large-Scale Graphs via Spiking Node Tokenization](gt-snt_a_linear-time_transformer_for_large-scale_graphs_via_spiking_node_tokeniz.md)
- [\[ICLR 2026\] Graph Tokenization for Bridging Graphs and Transformers](../../ICLR2026/graph_learning/graph_tokenization_for_bridging_graphs_and_transformers.md)
- [\[AAAI 2026\] Self-Adaptive Graph Mixture of Models](self-adaptive_graph_mixture_of_models.md)
- [\[AAAI 2026\] Commonality in Few: Few-Shot Multimodal Anomaly Detection via Hypergraph-Enhanced Memory](commonality_in_few_few-shot_multimodal_anomaly_detection_via_hypergraph-enhanced.md)

</div>

<!-- RELATED:END -->
