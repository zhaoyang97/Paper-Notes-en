---
title: >-
  [Paper Note] SSUF: A Semi-supervised Scalable Unified Framework for E-commerce Query Classification
description: >-
  [ACL 2025][LLM (Other)][Query Classification] A unified framework for e-commerce query classification, SSUF, is proposed. It utilizes three pluggable modules—Label Enhancement (BERT semantic label encoding), Knowledge Enhancement (LLM world knowledge + posterior clicks + semi-supervised label generation), and Structure Enhancement (co-occurrence/semantic/hierarchical multi-graph fusion GCN)—to address insufficient information in short queries and the vicious cycle of the Matt…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Query Classification"
  - "E-commerce Search"
  - "Semi-supervised Learning"
  - "Knowledge Enhancement"
  - "Graph Neural Networks"
date: 2026-05-08
content_hash: afccf0ce0c0e0a36
---

# SSUF: A Semi-supervised Scalable Unified Framework for E-commerce Query Classification

**Conference**: ACL 2025  
**arXiv**: [2506.21049](https://arxiv.org/abs/2506.21049)  
**Code**: None  
**Area**: Others  
**Keywords**: Query Classification, E-commerce Search, Semi-supervised Learning, Knowledge Enhancement, Graph Neural Networks

## TL;DR

A unified framework for e-commerce query classification, SSUF, is proposed. It utilizes three pluggable modules—Label Enhancement (BERT semantic label encoding), Knowledge Enhancement (LLM world knowledge + posterior clicks + semi-supervised label generation), and Structure Enhancement (co-occurrence/semantic/hierarchical multi-graph fusion GCN)—to address insufficient information in short queries and the vicious cycle of the Matthew effect. SSUF achieves Macro F1 scores of 49.46 and 41.22 on JD.COM intent and category classification tasks, respectively (outperforming SOTAs like SMGCN), and has been deployed online, bringing significant commercial value.

## Background & Motivation

**Background**: Query classification (predicting intent, category, and brand) in e-commerce platforms (such as JD.com, Taobao, and Amazon) is the core of search systems. Significant progress has been made using deep learning methods (XML-CNN, LSAN, DPHA) and recent hierarchy-aware methods (HCL4QC, SMGCN, HQC).

**Limitations of Prior Work**: (1) **Short and ambiguous queries**—E-commerce queries average only 6-8 characters (e.g., "Black 16pro"), suffering from extremely deficient semantic information where direct encoding fails to associate them with the "Mobile Phone" category. (2) **Vicious cycle of the Matthew effect**—Industrial methods rely on user click behavior to construct training samples, giving excessive attention to popular queries. Such biased data leads to poor generalization on long-tail queries. (3) **Isolated subtasks**—Intent, category, and brand prediction are modeled independently without a unified framework to support shared optimization.

**Key Challenge**: How to improve classification performance under the dual constraints of extreme information scarcity (short queries) and severely skewed labels (the Matthew effect)?

**Goal**: Construct a unified framework that breaks the over-dependence on posterior click labels through prior knowledge injection and structural information propagation.

**Key Insight**: A three-pronged approach—generating world knowledge with LLMs to supplement query semantics, encoding label semantics to generate semi-supervised signals, and utilizing graph structures to propagate gradients to long-tail labels.

**Core Idea**: Knowledge enhancement addresses "insufficient information", semi-supervised labeling addresses the "Matthew effect", and structural graph enhancement addresses "long-tail labels".

## Method

### Overall Architecture

The core of SSUF is a shared BERT text encoder, on top of which three highly pluggable enhancement modules are stacked. Online inference only requires the query text and GCN label embeddings, with heavy computations of the knowledge-enhanced branch completed offline.

### Key Designs

1. **Label-Enhanced Module**:

    - **Function**: Encodes the semantic representation of labels using BERT, replacing traditional label index embeddings.
    - **Mechanism**: Label input = label name $n$ + enhanced side information $m$ (product words, high-frequency search terms, LLM knowledge descriptions). Encoded via shared BERT: $\mathbf{C}_j = \text{BERT}_{\text{CLS}}([n_1,...,n_L, m_1,...,m_{L_m}])$
    - **Design Motivation**: Traditional index embeddings fail to capture semantic relationships among labels. Semantic encoding enables similarity computation between labels, promoting knowledge transfer.

2. **Knowledge-Enhanced Module**:

    - **Function**: Supplements semantic information of short queries with external knowledge and generates semi-supervised training signals.
    - **Mechanism**:
        - **Knowledge Sources**: (1) Posterior knowledge—highly clicked/purchased product labels by users, (2) World knowledge—sending queries and relevant products into open-source LLMs to generate short descriptions (containing related queries, categories, and products).
        - **Knowledge Fusion**: Query representations and knowledge embeddings are fused via an attention mechanism: $\alpha = \text{softmax}(\mathbf{Q}_i \mathbf{K}^T)$, $\mathbf{q}'_i = \mathbf{Q}_i + \sum_j \alpha_j \mathbf{K}_j$
        - **Semi-supervised Label Generation**: Computes the cosine similarity between the fused query and labels; those exceeding a threshold $\tau$ are treated as semi-supervised labels: $y^{semi}_{ij} = s_{ij} \cdot \mathbb{1}_{s_{ij} \geq \tau}$
        - **Key Design**: Applying `stop_gradient` to the semi-supervised branch prevents model collapse caused by cyclic dependencies.
    - **Design Motivation**: For instance, the query "Black 16pro" can be supplemented by LLM knowledge as "Apple mobile phone iPhone 16 Pro Black", thereby matching the "Mobile Phone" category.

3. **Structure-Enhanced Module**:

    - **Function**: Propagates gradients to long-tail labels via a label relationship graph.
    - **Mechanism**—Three graph constructions:
        - **Co-occurrence Graph** $\mathbf{A}^{coo}$: Conditional probability of label co-occurrence $a_{ij} = N(c_i, c_j) / N(c_i)$, with threshold $\alpha$ filtering low-frequency edges.
        - **Semantic Similarity Graph** $\mathbf{A}^{sim}$: Cosine similarity of label BERT embeddings, filtered by threshold $\beta$.
        - **Hierarchical Structure Graph** $\mathbf{A}^{hier}$: Parent-child label relations, where edge weight = $\max(1/|Child(k)|, m_i / \sum_{j \in Child(k)} m_j)$
    - **Graph Fusion and Learning**: $\mathbf{A} = \frac{1}{2}(\mathbf{A}^{coo} + \mathbf{A}^{sim}) \rightarrow \mathbf{A}^{hier}$; after normalization, GCN is applied to learn label representations.
    - **Design Motivation**: Although long-tail labels have few training samples, they can connect with popular labels through graph connections, thereby receiving gradient propagation.

### Loss & Training

- Final Prediction: $\hat{\mathbf{y}}_i = \text{sigmoid}(\mathbf{q}_i \mathbf{H}_l^T + \mathbf{b})$, predicting only leaf labels.
- Label Fusion: $\mathbf{y}_i = \min(\mathbf{y}_i^{click} + \mathbf{y}_i^{semi}, 1.0)$, combining posterior and semi-supervised labels.
- Loss Function: Binary Cross-Entropy Loss
- Inference Optimization: The knowledge enhancement branch (LLM world knowledge generation + attention fusion) is precomputed offline, requiring only query encoding and label embedding interactions online.

## Key Experimental Results

### Main Results (JD.COM Dataset, Micro/Macro F1)

| Model | Intent Task Micro F1 | Intent Task Macro F1 | Category Task Micro F1 | Category Task Macro F1 |
|------|------------------|------------------|------------------|------------------|
| XML-CNN | 45.58 | 27.24 | 38.34 | 20.16 |
| LSAN | 47.98 | 31.71 | 37.15 | 22.84 |
| SMGCN | 59.72 | 48.54 | 53.92 | 40.15 |
| HQC | 49.58 | 36.77 | 44.85 | 33.98 |
| **SSUF** | **61.81** | **49.46** | **56.45** | **41.22** |

### Ablation Study

| Configuration | Intent Macro F1 | Category Macro F1 | Description |
|------|--------------|--------------|------|
| Full SSUF | 49.46 | 41.22 | Baseline |
| w/o SE (Without all structure enhancement) | 43.30 (-6.16) | 38.52 (-2.70) | Graph propagation contributes significantly |
| w/o KE (Knowledge enhancement) | 45.82 (-3.64) | 39.24 (-1.98) | Knowledge enhancement primarily improves Macro (long-tail) |
| w/o LE & KE | 42.36 (-7.10) | 36.47 (-4.75) | Joint removal of labels and knowledge has the greatest impact |
| w/o SE-S (Without semantic graph) | 45.21 (-4.25) | 39.72 | Semantic graph contributes the most |
| w/o SE-C (Without co-occurrence graph) | 44.92 (-4.54) | 39.24 | Co-occurrence graph is equally important |
| w/o SE-H (Without hierarchical graph) | 47.29 (-2.17) | 39.95 | Hierarchical graph contributes relatively less |
| Pure BERT | 36.84 | 33.80 | Joint contribution of all three modules yields +12.62 Macro F1 |

### Key Findings

- Each of the three modules contributes independently, and their combination dramatically outperforms individual use—from pure BERT to SSUF, the intent Macro F1 improves from 36.84 to 49.46 (+34%).
- The knowledge-enhanced module yields a larger improvement on Macro F1 (long-tail labels) than on Micro F1 (popular labels), validating the design goal of breaking the Matthew effect.
- Validated through online A/B testing on JD.COM, demonstrating significant commercial value.
- Among the three graphs, both the semantic and co-occurrence graphs contribute ~4-5 percentage points to Macro F1, while the hierarchical graph contributes around 2 points.

## Highlights & Insights

- Modular design of a unified framework—The three modules are highly pluggable and can be flexibly combined based on the data characteristics of subtasks. Such an engineered framework design is highly practical in industry.
- Offline injection of LLM knowledge into small models is a practical paradigm for knowledge distillation—avoiding online LLM API calls by precomputing LLM world knowledge into query features keeps costs manageable.
- Clever semi-supervised design using `stop_gradient` to prevent cyclic dependencies—Since queries and labels share the encoder, directly propagating gradients would lead to mutual reinforcement and collapse between the semi-supervised signal and the encoder.

## Limitations & Future Work

- The quality of LLM-generated world knowledge is difficult to control, and incorrect information might backfire, contaminating the classification.
- Sensitivity analysis for the semi-supervised threshold $\tau$ and graph filtering thresholds $\alpha$/$\beta$ is not sufficiently thorough.
- Only validated on JD.COM Chinese e-commerce data; tests on other platforms and multilingual scenarios are yet to be conducted.
- The fusion strategy for the three graphs is relatively simple (mean pooling + hierarchical assignment); more complex attention-based fusion mechanisms could perform better.
- With a label space of 6,634 categories in the category task, the scalability of GCN under larger label spaces needs to be verified.

## Related Work & Insights

- **vs HCL4QC/SMGCN**: Utilize hierarchical structures but lack knowledge enhancement and semi-supervised signals; SSUF combines all three.
- **vs Pure LLM Classification**: Direct classification via LLMs suffers from unacceptable latency and costs; SSUF distills LLM knowledge into offline features.
- **vs LEAM/LSAN**: Label-aware but lack graph structures and external knowledge; SSUF provides richer label representations.

## Rating

- Novelty: ⭐⭐⭐ The three modules themselves are not entirely novel (LLM knowledge enhancement, semi-supervised learning, and GCN), but the combined design and the engineered unified framework offer practical contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Large-scale real-world data (67M+ training samples), complete ablation studies, and online A/B testing.
- Writing Quality: ⭐⭐⭐⭐ Clear framework description, complete formula derivations, but the motivation section leans heavily towards industrial perspectives.
- Value: ⭐⭐⭐⭐ Successfully deployed in online industrial systems, providing direct value to e-commerce search, and its modular design offers transferable reference value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Learning Textual Prompts for Open-World Semi-Supervised Learning](../../CVPR2025/llm_nlp/learning_textual_prompts_for_open-world_semi-supervised_learning.md)
- [\[ACL 2025\] Lost in Literalism: How Supervised Training Shapes Translationese in LLMs](lost_in_literalism_how_supervised_training_shapes_translationese_in_llms.md)
- [\[ACL 2025\] MDCure: A Scalable Pipeline for Multi-Document Instruction-Following](mdcure_a_scalable_pipeline_for_multi-document_instruction-following.md)
- [\[CVPR 2025\] MG-MotionLLM: A Unified Framework for Motion Comprehension and Generation across Multiple Granularities](../../CVPR2025/llm_nlp/mg-motionllm_a_unified_framework_for_motion_comprehension_and_generation_across_.md)
- [\[ACL 2025\] ScaleQuest: Unleashing LLM Reasoning Capability via Scalable Question Synthesis from Scratch](unleashing_llm_reasoning_capability_via_scalable.md)

</div>

<!-- RELATED:END -->
