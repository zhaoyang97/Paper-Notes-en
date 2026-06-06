---
title: >-
  [Paper Note] GILT: An LLM-Free, Tuning-Free Graph Foundational Model for In-Context Learning
description: >-
  [ICML 2026][Graph Learning][Graph Foundation Models] GILT unifies few-shot classification for nodes, edges, and graphs into a token-based in-context learning problem. Using a pure numerical architecture consisting of "Li…
tags:
  - "ICML 2026"
  - "Graph Learning"
  - "Graph Foundation Models"
  - "Graph ICL"
  - "Few-shot Graph Learning"
  - "Prototypical Classification"
  - "Transformer"
date: 2026-05-08
content_hash: f2fc5cafed0b7fbb
---

# GILT: An LLM-Free, Tuning-Free Graph Foundational Model for In-Context Learning

**Conference**: ICML 2026  
**arXiv**: [2510.04567](https://arxiv.org/abs/2510.04567)  
**Code**: https://github.com/yiming421/inductnode/ (Available)  
**Area**: Graph Learning / Graph Foundation Models / In-Context Learning  
**Keywords**: Graph Foundation Models, Graph ICL, Few-shot Graph Learning, Prototypical Classification, Transformer

## TL;DR
GILT unifies few-shot classification for nodes, edges, and graphs into a token-based in-context learning problem. Using a pure numerical architecture consisting of "Linear GCN for structural extraction + asymmetric prototype tokens + two-stage attention Transformer + prototype head," it achieves zero-dependency on LLMs and requires no downstream tuning. In 5-shot settings, it outperforms both LLM-based and tuning-based GFMs while being 1 to 4 orders of magnitude faster.

## Background & Motivation

**Background**: While general GNNs perform strongly on single graphs, their cross-graph transferability is limited, leading to the emergence of "Graph Foundation Models" (GFM). Current GFMs follow two main paths: 1) mapping text attributes of nodes/categories to a unified semantic space via LLMs (e.g., ZeroG, GOFA), or 2) pre-training a structural encoder on large-scale graphs and using graph prompting for parameter fine-tuning on downstream graphs (e.g., GCOPE, RiemannGFM).

**Limitations of Prior Work**: The LLM approach is inherently text-dependent and fails on graphs dominated by numerical, categorical, or pure structural features (e.g., molecular graphs, social networks) unless manual text descriptions are provided. The prompting approach, while graph-native, requires gradient descent for every new graph, creating an efficiency bottleneck and violating the "off-the-shelf" foundational model philosophy.

**Key Challenge**: The extreme heterogeneity of graph data—where feature dimensions, semantics, label sets, and topologies vary across graphs—naturally ties traditional GNN parameters to the training graph. Breaking this bond usually requires either a text bridge (limited by text availability) or tuning (limited by efficiency).

**Goal**: Construct a unified GFM that is simultaneously LLM-free, tuning-free, multi-domain, multi-task, and few-shot capable, allowing the model to process arbitrary N-way K-shot node/edge/graph classification tasks during inference by observing only a few support samples.

**Key Insight**: The authors draw inspiration from the success of TabPFN on tabular data, where Transformers with causal attention demonstrate excellent ICL on structured data. By "translating" graph tasks into a unified set of tokens, Transformers can perform ICL on graphs just as they do on tables, completely bypassing the need for text or tuning.

**Core Idea**: In one sentence: "Unify few-shot graph classification as token reasoning, using prototype-aware asymmetric tokens and two-stage attention to let the Transformer 'understand' task semantics from the support set, followed by a cosine prototype head for on-the-fly N-way classification."

## Method

### Overall Architecture
GILT is a two-stage pipeline that takes an N-way K-shot task as input: a support set $\mathcal{S} = \{(x_i, y_i)\}_{i=1}^{N \times K}$ (labeled) and a query set $\mathcal{Q} = \{x_j\}_{j=1}^{Q}$ (to be predicted), where $x_i$ can be a node, edge, or graph. The output is a probability distribution over classes for the queries.

- **Phase 1: Graph-Native Tokenization (Syntactic Unification)**: A structural encoder converts heterogeneous graphs into structure-aware embeddings, which are grouped into item representations according to the task type and concatenated with class prototypes to form a unified-dimension support/query token set.
- **Phase 2: In-Context Inference (Semantic Unification)**: The ICL Transformer uses two-stage attention—internal self-attention for the support set to refine task context, followed by cross-attention where queries "learn" from the refined support set—to produce context-aware query representations for the non-parametric prototypical head to calculate cosine similarity for final classification.

The model is trained on "meta-tasks" across 22 multi-domain graph datasets, learning the meta-skill of "how to infer task rules from support samples" rather than memorizing labels of specific graphs.

### Key Designs

1. **Graph-Native Tokenization: Linear GCN + Asymmetric Prototype Tokens**:
    - **Function**: Converts local topology and raw features of arbitrary heterogeneous graphs into a unified-dimension token set consumable by the ICL Transformer; this serves as the entry point for the LLM-free design.
    - **Mechanism**: The structural encoder utilizes a 4–6 layer linear GCN **stripped of learnable weights and non-linear activations** (similar to SGC/APPNP), where each layer performs $H^{(l+1)} = \mathrm{LayerNorm}(\tilde{A} H^{(l)})$ to aggregate multi-hop neighborhood information. Item representations $h$ are then formed based on the task type (node embeddings for node tasks, element-wise products of endpoint embeddings for edge tasks, pooling for graph tasks). Class prototypes $p_c$ are calculated via mean-pooling + L2 normalization. Support tokens are **asymmetrically** constructed as $t_s = [h_i \,\|\, p_{y_i}]$ and query tokens as $t_q = [h_j \,\|\, \mathbf{0}]$.
    - **Design Motivation**: Learnable weights are disabled because "semantic projection" during pre-training tends to overfit the feature distribution of training graphs, damaging cross-graph generalization. Semantic reasoning is deferred entirely to the Transformer. The asymmetric token design elegantly manages the tension between fixed token dimensions and the need for the model to see all category concepts simultaneously—outperforming one-hot encodings (which vary with class count) or decomposition into multiple binary classifications.

2. **Two-Stage ICL Transformer + Prototypical Head**:
    - **Function**: Injects task semantics into each query representation and completes N-way classification without any parameter updates, while maintaining a unidirectional information flow from support to query.
    - **Mechanism**: Each layer consists of two steps—**Stage 1: Context Refinement**, which performs multi-head self-attention only among support tokens $T_\mathcal{S}' = \mathrm{SelfAttention}(T_\mathcal{S})$ to allow interaction and formation of task semantics; **Stage 2: Information Aggregation**, where queries extract necessary context from refined supports via multi-head cross-attention $T_\mathcal{Q}' = \mathrm{CrossAttention}(Q{=}T_\mathcal{Q}, K{=}T_\mathcal{S}', V{=}T_\mathcal{S}')$. Final prediction uses the "class space" segment of the token embedding to calculate prototypes (mean of support samples in the same class) and computes class distributions via softmax over cosine similarity. The separation of item and class spaces allows natural support for arbitrary $N$.
    - **Design Motivation**: The two-stage attention strictly ensures that queries do not influence each other and do not contaminate the support set—a key for stable ICL on structured data (removing the Transformer drops performance to ~13%). The prototypical head is another key to the tuning-free property, as it is non-parametric and decoupled from $N$.

3. **Test-Time Augmentation (TTA) + Link Prediction Node Labeling**:
    - **Function**: Further reduces prediction variance and addresses 1-WL expressive bottlenecks without modifying the shared backbone, primarily for structural-sensitive tasks like link prediction.
    - **Mechanism**: A unified **test-time augmentation** is applied across tasks—multiple views are generated via random rotation of raw features, and predictions are averaged. For link prediction, an MPLP-inspired node labeling strategy is introduced to provide structural cues for target edge pairs (compensating for MPNN limitations in distinguishing different edge pairs in isomorphic subgraphs).
    - **Design Motivation**: Following the experience from TabPFN that ensembles are effective for ICL models, the authors migrate this to graphs and add specificity for link-level tasks to stabilize the unified backbone on challenging tasks.

### Loss & Training
The pre-training corpus covers 22 cross-domain graphs (citation, social, molecule) with 450k+ nodes and 4M+ edges. At each step, a few-shot task is randomly sampled, and the model is supervised using standard cross-entropy loss:

$$\mathcal{L} = -\frac{1}{|\mathcal{Q}|} \sum_{x_j \in \mathcal{Q}} \log P(y = y_j \mid x_j)$$

The test sets are completely disjoint from the pre-training sets, ensuring the model learns the meta-skill of task inference rather than categorical memorization.

## Key Experimental Results

### Main Results

Tasks cover three major graph learning types; evaluation strictly separates train/test splits.

| Dataset | Task | Metric | Setting | Ours (GILT) | Prev. SOTA | Gain |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Cora | Node | Acc | 5-shot | 73.22 | GraphAny 72.68 | +0.54 |
| Citeseer | Node | Acc | 5-shot | 66.17 | GCOPE 63.90 | +2.27 |
| Pubmed | Node | Acc | 5-shot | 71.86 | GCN 69.88 | +1.98 |
| Node Mean (4 sets) | Node | Acc | 5-shot | **69.51** | GAT 66.21 | +3.30 |
| ogbl-collab | Link | Hits@K | 5-shot | 67.83 | MaskGAE-sup 65.84 | +1.99 |
| ogbg-molhiv | Graph | ROC-AUC | 5-shot | 65.81 | GCN 55.56 | +10.25 |

Highlights: Under the 5-shot setting, GILT not only beats all ICL/tuning baselines but also outperforms supervised methods like SEAL/MaskGAE on link prediction tasks in Cora/Citeseer/ogbl-collab.

### Ablation Study

| Configuration | Cora 5-shot Acc | Description |
| :--- | :--- | :--- |
| Full model | 73.22 | Complete model |
| w/o ICL Transformer | 13.00 | Performance collapses, proving the ICL module is the core |
| w/ Full Token for Pred | 72.97 | No item/class space separation; minor drop on Cora but ~10% drop on WikiCS |
| w/o Graph Encoder | 57.50 | 15+ point drop without structural encoding |
| w/ Non-linear GCN | 70.76 | Switching to standard GCN with non-linearity drops performance |
| w/ 2-layer Encoder | 70.52 | Shallow encoder lacks expressiveness; 4–6 layers is the sweet spot |
| Base (w/o TTA) | 68.68 | Backbone only; TTA adds ~4 points on average |

### Key Findings
- **ICL Transformer is the absolute core**: Performance collapses to near-random without it, making it even more vital than the graph encoder—consistent with TabPFN findings.
- **"Linear is better than Non-linear"**: Stripping learnable weights from GCN actually outperforms standard GCN across 4 node datasets. The authors attribute this to "fewer parameters = less overfitting to pre-training semantics = better generalization."
- **Efficiency Gap**: On the same hardware (RTX 4090), GILT is ~20× faster than GAT, 180× faster than tuning-based GCOPE, and **14,000×** faster than LLM-based GOFA.
- **Beating Zero-shot LLMs**: GILT with only 5 numerical samples outperforms ZeroG/GOFA/LLaGA on Planetoid datasets, suggesting that inferring semantics from supports is more effective for graphs than "knowledge retrieval" from text category names.

## Highlights & Insights
- The **asymmetric token + prototype concatenation** is the most ingenious design: it encodes "what the item is" and "its current class estimate" in a fixed dimension, allowing the Transformer to perform both instance reasoning and cross-class comparison.
- The philosophy of **"Semantics to Transformer, Structure to simple Encoder"** is a valuable design principle for LLM-free GFMs: concentrating learnable parameters where they "understand" logic aids generalization.
- **Migration of TabPFN logic (causal mask + ensemble) to graphs**: This recognizes graphs as structured data, suggesting we can borrow aggressively from tabular ICL rather than reinventing graph-specific mechanisms from scratch.
- **Inference-only MPLP labeling for Link Prediction**: Fixing the 1-WL expressiveness gap during inference rather than in the backbone preserves the simplicity of a "one-model-multi-task" architecture.

## Limitations & Future Work
- **Task Scope**: Currently limited to N-way K-shot classification; regression, generation, and node ranking are not yet covered.
- **Complexity with Support Set Size**: Attention complexity is $O((NK+Q)^2)$; while efficient for 1/5-shot, it may become a bottleneck for large K.
- **Performance on WikiCS**: GILT's 1-shot performance is lower than GraphAny, which might suggest that ICL is less stable on extremely noisy or highly heterophilic graphs compared to simpler non-parametric methods.
- **Heterophily Assumption**: Linear GCNs (SGC/APPNP) act as the structural prior, which may fail on heterophilic graphs where more complex priors are needed.

## Related Work & Insights
- **vs. OFA**: OFA uses prompt graphs to connect support samples into virtual nodes for GNN forward inference; GILT uses Transformers on token sets, offering stronger cross-task unification (Node 1-shot Cora 30.52 vs 56.36).
- **vs. GraphAny**: GraphAny is tuning-free but limited to node classification; GILT generalizes ICL to node/link/graph tasks with a more expressive end-to-end trainable deep network.
- **vs. GCOPE/RiemannGFM**: These rely on downstream tuning; GILT is stronger on many datasets without any tuning and is 100+ times faster.
- **vs. ZeroG/GOFA**: These rely on LLMs for zero-shot text-based inference; GILT's 5-shot numerical performance suggests LLMs might be a burden rather than a necessity for text-poor graphs.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First GFM to achieve LLM-free + tuning-free across node/link/graph tasks simultaneously.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers various tasks and baselines, though lacks stress tests for extreme N/K or heterophily.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation and well-justified design choices.
- **Value**: ⭐⭐⭐⭐ Directly benefits latency-sensitive industrial graph tasks and provides a reusable template for graph ICL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Quantile-Free Uncertainty Quantification in Graph Neural Networks](quantile-free_uncertainty_quantification_in_graph_neural_networks.md)
- [\[ICML 2026\] Message Tuning Outshines Graph Prompt Tuning: A Prismatic Space Perspective](message_tuning_outshines_graph_prompt_tuning_a_prismatic_space_perspective.md)
- [\[ICML 2026\] Are Common Substructures Transferable? Riemannian Graph Foundation Model with Neural Vector Bundles](are_common_substructures_transferable_riemannian_graph_foundation_model_with_neu.md)
- [\[CVPR 2026\] Graph-to-Frame RAG: Visual-Space Knowledge Fusion for Training-Free and Auditable Video Reasoning](../../CVPR2026/graph_learning/graph-to-frame_rag_visual-space_knowledge_fusion_for_training-free_and_auditable.md)
- [\[ICML 2026\] Structure-Centric Graph Foundation Model via Geometric Bases](structure-centric_graph_foundation_model_via_geometric_bases.md)

</div>

<!-- RELATED:END -->
