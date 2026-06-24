---
title: >-
  [Paper Note] Multi-Hop Reasoning for Question Answering with Hyperbolic Representations
description: >-
  [NLP Understanding] By simply inserting a single Poincaré hyperbolic layer into a T5 encoder-decoder model, this work maps Euclidean embeddings to hyperbolic space for multi-hop reasoning with minimal model modifications. Experiments across four datasets consistently outperform Euclidean counterparts, demonstrating the effectiveness of $\delta$-hyperbolicity-based curvature initialization and showing that hyperbolic space is more advantageous on datasets with stronger hierarc…
tags:
  - "NLP Understanding"
date: 2026-05-08
content_hash: 0b4661227350d397
---

# Multi-Hop Reasoning for Question Answering with Hyperbolic Representations

- **Conference**: ACL 2025
- **arXiv**: [2507.03612](https://arxiv.org/abs/2507.03612)
- **Code**: Closed-source (uses open-source [2WikiMultiHop evaluation code](https://github.com/Alab-NII/2wikimultihop) and Poincaré layer implementation)
- **Area**: NLP Understanding / Multi-Hop Reasoning / Hyperbolic Representation Learning
- **Keywords**: Multi-Hop Reasoning, Hyperbolic Space, Poincaré Ball, Knowledge Graphs, Question Answering, Curvature Initialization

## TL;DR

By simply inserting a single Poincaré hyperbolic layer into a T5 encoder-decoder model, this work maps Euclidean embeddings to hyperbolic space for multi-hop reasoning with minimal model modifications. Experiments across four datasets consistently outperform Euclidean counterparts, demonstrating the effectiveness of $\delta$-hyperbolicity-based curvature initialization and showing that hyperbolic space is more advantageous on datasets with stronger hierarchical structures.

## Background & Motivation

Multi-hop reasoning requires a model to integrate multiple steps of evidence to derive an answer. For example, answering "Which country does the composer of Cloudburst come from?" requires first finding the composer (Eric Whitacre) and then looking up his nationality (American). This reasoning process is inherently a path traversal over the hierarchical or tree-like structure of a knowledge graph.

Traditional language models represent concepts in Euclidean space. Although they can capture hierarchical structures to some extent, hyperbolic space possesses inherent advantages in modeling tree-like and graph-like structures, as its volume grows exponentially with distance, making it more suitable for encoding hierarchical relationships. However, existing work in hyperbolic reasoning suffers from two main limitations:

1. **Lack of Controlled Comparison**: Prior works introduce extensive architectural modifications and extra parameters alongside hyperbolic geometry, making it impossible to separate whether performance improvements stem from geometric properties or from model modifications.
2. **Excessive Architectural Modifications**: These approaches require redesigning model architectures, which increases complexity and computational overhead.

The core contribution of this paper is to provide a **strictly controlled** comparative experiment. By adding only a single hyperbolic layer (compared against a parameter-matched Euclidean layer), the influence of the geometric space itself is successfully isolated.

## Method

### Design 1: Two-Stage Soft Prompt Reasoning Based on the PaTH Framework

The method is based on the PaTH (Prompt-Aided T5 for multi-Hop reasoning) framework, which comprises two core stages:

**Stage 1: Knowledge Integration**

The T5 model is fine-tuned using knowledge graph triples $(e_1, r_1, e_2)$ to internalize entity-relation structures. Only subgraph triples related to 2-hop questions are used.

**Stage 2: Soft Prompt Tuning**

Two types of soft prompts are trained:
- **Parsing Prompt**: Parses natural language questions into incomplete path sequences $(e_1, r_1, r_2, \ldots, r_n)$.
- **Hopping Prompt**: Given an incomplete sequence, trains the model to predict the complete path $(e_1, r_1, e_2, r_2, \ldots, r_{n-1}, e_n)$ using random walks over the knowledge graph.

### Design 2: Integration of the Poincaré Ball Hyperbolic Layer

The model inserts a hyperbolic processing pipeline after the T5 encoder output (see Figure 2):

$$\text{T5 Encoder} \xrightarrow{\text{exp}_0^c} \text{Poincaré Ball} \xrightarrow{\text{Poincaré Layer}} \text{Poincaré Ball} \xrightarrow{\text{log}_0^c} \text{T5 Decoder}$$

**Specific Steps**:

1. **Exponential Map**: Maps the Euclidean embedding $v$ onto the Poincaré ball:

$$\exp_0^c(v) = \frac{\tanh(\|v\| \cdot \sqrt{c})}{\|v\| \cdot \sqrt{c}} \cdot v$$

2. **Poincaré Linear Layer**: Performs operations within the hyperbolic space using the transformation formula of hyperbolic multinomial logistic regression:

$$v_k(x) = \frac{2}{\sqrt{c}} \|z_k\| \sinh^{-1}\left(\lambda_x^c \left\langle \sqrt{c}x, \frac{z_k}{\|z_k\|} \right\rangle \cosh(2\sqrt{c}r_k) - (\lambda_x^c - 1)\sinh(2\sqrt{c}r_k)\right)$$

where $\lambda_x^c = \frac{2}{1 - c\|x\|^2}$ is the conformal factor, and $Z = \{z_k\}$ and $r = \{r_k\}$ denote the trainable weights and biases, respectively.

3. **Logarithmic Map**: Maps the hyperbolic embedding back to Euclidean space to maintain compatibility with the T5 decoder:

$$\log_0^c(y) = \frac{\tanh^{-1}(\|y\| \cdot \sqrt{c})}{\|y\| \cdot \sqrt{c}} \cdot y$$

### Design 3: Curvature Initialization Based on $\delta$-Hyperbolicity

The $\delta$-hyperbolicity of the dataset is computed via the Gromov product. A relative metric, $\delta_{rel}(X) = \frac{2\delta(X)}{\text{diam}(X)}$, is used to eliminate scale effects. The curvature parameter is initialized based on this metric:

$$c(X) = \left(\frac{0.144}{\delta_{rel}(X)}\right)^2$$

This curvature parameter is learnable and updated during training. Graph properties are estimated by sampling 1,500 points and repeating the process 5 times.

## Key Experimental Results

### Experimental Setup

- **Model**: T5-Large (770M parameters), with frozen backbone, training only the added layers and soft prompts.
- **Optimizer**: AdaFactor, learning rate 0.001, batch size 64.
- **Evaluation Metrics**: Exact Match (EM).
- **Datasets**: 4 closed-book QA datasets (only using 2-hop questions).

### Table 1: Dataset Statistics

| Dataset | Nodes | Edges | Relations | Train/Val/Test |
|:---:|:---:|:---:|:---:|:---:|
| 2WikiMultiHopQA | 97,298 | 95,116 | 29 | 72,760/8,085/6,768 |
| MetaQA | 31,374 | 58,974 | 9 | 47,108/5,951/5,942 |
| MLPQ | 51,402 | 53,327 | 72 | 57,283/7,160/7,161 |
| PQ | 1,056 | 1,211 | 13 | 1,698/210/191 |

### Table 2: EM Scores (%) in the Hopping Prompt Stage

| Split | Model | 2WikiHop | MetaQA | MLPQ | PQ |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Dev | Euclidean | 44.36 | 22.92 | 81.03 | 18.28 |
| Dev | **Hyperbolic** | **46.93** | **28.33** | **82.60** | **29.03** |
| Test | Euclidean | 14.88 | 19.76 | 72.10 | 11.90 |
| Test | **Hyperbolic** | **15.20** | **25.40** | **74.58** | **23.21** |

### Table 3: Test EM Scores (%) with Different Combinations of Hyperbolic/Euclidean Layers across Stages

| Parsing | Hopping | 2WikiHop | MetaQA | MLPQ | PQ |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Euclidean | Euclidean | 13.39 | 19.20 | 72.59 | 12.04 |
| Hyperbolic | Euclidean | 13.56 | 19.08 | 72.74 | 12.04 |
| Euclidean | **Hyperbolic** | **13.40** | **24.74** | **73.48** | **23.04** |
| Hyperbolic | Hyperbolic | 13.65 | 24.72 | 73.40 | 22.51 |

**Key Insights**: The majority of the performance gains originate from deploying the hyperbolic layer in the Hopping stage, as this stage directly depends on the hierarchical structure of the knowledge graph.

## Key Findings

1. **Hyperbolic Consistently Outperforms Euclidean**: Across all 4 datasets, the hyperbolic layer outperforms the parameter-matched Euclidean layer during the Hopping stage, yielding the most significant improvement on the PQ dataset (dev: +10.75%, test: +11.31%).
2. **Curvature Initialization is Crucial**: Curvature initialized based on $\delta$-hyperbolicity significantly outperforms random initialization. Extremely high curvature values (e.g., $c=10$) lead to performance collapse (e.g., 2WikiHop EM drops to 2.21%, MetaQA drops to 0.22%).
3. **Stronger Hierarchies Yield Larger Gains**: The MLPQ dataset shows the smallest gain (+1.57%) since 80% of its nodes have an out-degree of 1 (making it close to linear). MetaQA and PQ present more complex hierarchical structures, showcasing more substantial improvements.
4. **Hyperbolic Space Expands Distances**: In almost 100% of cases on 2WikiHop, MetaQA, and PQ, the hyperbolic geodesic distance is larger than the Euclidean distance, which facilitates path disambiguation.
5. **Negligible Computational Overhead**: The inference time and memory overhead introduced by the hyperbolic layer are virtually zero.
6. **Effective Even Without Soft Prompts**: The hyperbolic layer consistently outperforms the Euclidean layer even when soft prompts are removed (Table 6).

## Highlights & Insights

- **Rigorous Experimental Design**: By keeping the geometry space type as the only variable and using a parameter-matched baseline, this work presents the first truly controlled comparative study in hyperbolic multi-hop reasoning.
- **Minimalist Integration**: Only a single hyperbolic layer together with exponential/logarithmic maps is added. It leaves the backbone T5 architecture intact and keeps the encoder/decoder parameters frozen.
- **$\delta$-hyperbolicity Driven Curvature Initialization**: It aligns the data structure properties with model geometry, offering a theoretically grounded approach for hyperparameter selection.
- **Multi-faceted Analysis**: Incorporates performance, computational efficiency, embedding distances, and dataset difficulty to explain the underlying "why" rather than just showing "what".

## Limitations & Future Work

1. **Limited to Closed-book QA**: Open-domain or retrieval-augmented setups are not evaluated, restricting the information strictly to the model's pre-trained knowledge.
2. **Frozen Model Backbone**: Since only the additional layer is fine-tuned (~1M parameters), the relative impact of the single hyperbolic layer might be diluted when full-parameter fine-tuning is applied to the main model (billions of parameters).
3. **Confined to Encoder-Decoder Architectures**: Has not been extended to decoder-only models (e.g., the GPT series), making generalizability uncertain.
4. **Only Evaluated on 2-hop Tasks**: The performance on longer reasoning chains (3-hop and beyond) is not verified.
5. **Limited Dataset Scale**: The PQ dataset only contains 1,908 questions, which may impact statistical reliability.

## Related Work & Insights

- **Path-based Multi-hop Reasoning**: Lao et al. (2011) utilize predefined rules to execute reasoning over knowledge bases.
- **Neural Embedding Methods**: TransE (Bordes et al., 2013), RotatE (Sun et al.), etc., vectorize entities and relations.
- **Graph Neural Network Inference**: R-GCN (Schlichtkrull et al., 2018) and GAT (Veličković et al., 2018) propagate information across multi-hop relational structures.
- **Hyperbolic Knowledge Graph Embeddings**: Poincaré Embedding (Nickel & Kiela, 2017), MuRP (Balažević et al., 2019), and ATTH (Chami et al., 2020) model hierarchical relationships using hyperbolic spaces.
- **Hyperbolic Graph Neural Networks**: HGCN (Chami et al., 2019) and DeepHGCN (Liu et al., 2024) perform message passing within hyperbolic spaces.
- **PaTH Framework**: Misra et al. (2023) propose a T5 multi-hop reasoning baseline combining soft prompts and random walks.

## Rating

- **Novelty**: ⭐⭐⭐ — While the method itself is simple (adding a single layer), the experimental design (isolating the exact impact of the geometric space) is rigorous and fills a gap in controlled testing.
- **Effectiveness**: ⭐⭐⭐⭐ — Consistently outperforms baselines across all four datasets, with comprehensive ablation studies and convincing multi-angle analyses.
- **Practicality**: ⭐⭐⭐ — Simple integration and small overhead, but constrained to specific architectures and closed-book settings, resulting in a narrow deployment scope.
- **Recommended Reading**: ⭐⭐⭐⭐ — Suitable for researchers interested in the intersection of Geometric Deep Learning $\times$ NLP, offering valuable insights into understanding the role of hyperbolic space in reasoning tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] RISE: Reasoning Enhancement via Iterative Self-Exploration in Multi-hop Question Answering](rise_reasoning_enhancement_via_iterative_self-exploration_in_multi-hop_question_.md)
- [\[ACL 2025\] BELLE: A Bi-Level Multi-Agent Reasoning Framework for Multi-Hop Question Answering](belle_a_bi-level_multi-agent_reasoning_framework_for_multi-hop_question_answerin.md)
- [\[ACL 2025\] Self-Critique Guided Iterative Reasoning for Multi-hop Question Answering](self-critique_guided_iterative_reasoning_for_multi-hop_question_answering.md)
- [\[ACL 2025\] Active LLMs for Multi-hop Question Answering](active_llms_for_multi-hop_question_answering.md)
- [\[ACL 2025\] ReSCORE: Label-free Iterative Retriever Training for Multi-hop Question Answering with Relevance-Consistency Supervision](rescore_multihop_qa.md)

</div>

<!-- RELATED:END -->
