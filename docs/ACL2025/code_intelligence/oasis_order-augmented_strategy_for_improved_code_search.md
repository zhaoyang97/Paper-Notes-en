---
title: >-
  [Paper Note] OASIS: Order-Augmented Strategy for Improved Code Search
description: >-
  [ACL 2025][Code Intelligence][Code Search] OASIS is proposed to capture subtle nuances in code semantics by introducing order-based similarity labels for negative pairs. By training code embedding models with a dual loss function combining InfoNCE and CoSENT, OASIS consistently outperforms existing state-of-the-art (SOTA) models on NL2Code and Code2Code search tasks across three benchmarks: CoSQA, AdvTest, and CodeSearchNet.
tags:
  - "ACL 2025"
  - "Code Intelligence"
  - "Code Search"
  - "Code Embedding"
  - "Contrastive Learning"
  - "Order Label"
  - "Negative Pairs"
  - "Similarity Refinement"
date: 2026-05-08
content_hash: ab4ab1f41a705689
---

# OASIS: Order-Augmented Strategy for Improved Code Search

**Conference**: ACL 2025  
**arXiv**: [2503.08161](https://arxiv.org/abs/2503.08161)  
**Code**: [HuggingFace](https://huggingface.co/Kwaipilot/OASIS-code-embedding-1.5B)  
**Area**: Other (Code Search / Code Embeddings)  
**Keywords**: Code Search, Code Embedding, Contrastive Learning, Order Label, Negative Pairs, Similarity Refinement

## TL;DR

OASIS is proposed to capture subtle nuances in code semantics by introducing order-based similarity labels for negative pairs. By training code embedding models with a dual loss function combining InfoNCE and CoSENT, OASIS consistently outperforms existing state-of-the-art (SOTA) models on NL2Code and Code2Code search tasks across three benchmarks: CoSQA, AdvTest, and CodeSearchNet.

## Background & Motivation

Code search aims to retrieve the most relevant code snippets based on natural language queries, serving as a core foundation for code-related LLM applications (such as RAG and code completion). Current state-of-the-art approaches typically train code embedding models using contrastive learning, employing the InfoNCE loss to pull positive pairs closer and push negative pairs apart.

However, existing methods possess several critical limitations:

**Over-reliance on the "primary differences" between positive and negative samples**: Current training only focuses on the large discrepancies between positive and negative pairs, ignoring the subtle differences among negative pairs. Due to the sparsity of code contexts, even a minor alteration can lead to significant changes in function and semantics.

**The trap of surface-level semantic matching**: Focusing solely on positive-negative discrepancies easily biases the model toward learning superficial features. For instance, given an NL query, the model might match it with code that has high lexical overlap but different functionality, while ignoring semantically correct code with low lexical overlap.

**Difficulty in similarity annotation in the code domain**: Unlike the text embedding domain which enjoys readily available STS (Semantic Textual Similarity) datasets, annotating similarity in code is significantly more challenging due to the sparsity of context, hindering development in this field.

The core idea of OASIS is to **leverage subtle differences between negative pairs (captured via order labels) to learn deeper code semantics**, rather than solely relying on the coarse-grained distinction between positive and negative samples.

## Method

### Overall Architecture

OASIS comprises three core steps: (1) Docstring generation and similarity annotation; (2) Similarity refinement; and (3) Mixed loss training. The data is sourced from open-source GitHub repositories, yielding a synthesized dataset of 53 million NL-Code pairs across 9 programming languages.

### Key Designs

#### 1. Docstring Generation and Similarity Annotation

- **Repository-level program analysis**: For each function, caller and callee information is extracted and combined with the source code to construct a prompt. An LLM is then utilized to generate high-quality docstrings for the code.
- **In-repository negative sample construction**: For a given docstring A, $K$ code snippets from other functions are randomly selected from the same repository to form negative pairs. Code snippets within the same repository often exhibit similar semantics or lexical overlaps, naturally serving as high-quality "hard negatives."
- **Similarity label computation**: Another embedding model (Text-Embedding-3-Large) is used to compute a similarity score $sim \in [0,1)$ for each negative pair, which acts as an order label providing auxiliary training signals.

#### 2. Similarity Refinement

The initial similarity labels can contain misannotations and require further refinement:

- **GMM thresholding**: A Gaussian Mixture Model (GMM) is used to fit the similarity distribution of all sample pairs (which exhibits a bimodal distribution). The intersection of the two distributions is taken as the positive-negative dividing threshold $s^*$. If the similarity of a negative pair exceeds $s^*$ or surpasses that of the corresponding positive pair, it is flagged as a potential misannotation.
- **AST edit distance method**: The Abstract Syntax Trees (ASTs) of the code are parsed to select candidate pairs with a low ratio of AST edit distance to the total node count. These represent pairs that are highly structurally similar but lexically distinct.
- **LLM binary judgment**: For candidate pairs, an LLM is used to judge whether "the candidate code also satisfies the description of the docstring." If so, the similarity of this negative pair is positively adjusted by $\Delta s$ (optimized via grid search).

#### 3. Mixed Loss Training

OASIS employs two complementary loss functions:

**InfoNCE loss** — A traditional contrastive learning objective that focuses on the overall distinction between positive and negative pairs within a batch:

$$\mathcal{L}_{ibn} = -\sum_b \sum_{i=1}^m \log \frac{\exp(\cos(h_i, h_i^+) / \tau)}{\sum_{j=1}^N \exp(\cos(h_i, h_j) / \tau)}$$

**CoSENT loss** — An order-based optimization objective focusing on the relative ranking of similarity pairs:

$$\mathcal{L}_{cos} = \log \left[1 + \sum_{s_{ij} > s_{mn}} \exp\left(\frac{\cos_{nm} - \cos_{ij}}{\tau}\right)\right]$$

CoSENT does not force the model to predict precise similarity values; instead, it ensures that the predicted similarity ranks align with the label ranks. A loss is incurred when the labels indicate that pair $(i,j)$ has higher similarity than $(m,n)$ but the model predicts otherwise.

**The total loss** is a weighted combination of both: $L = w_1 \cdot L_{ibn} + w_2 \cdot L_{cos}$

### Information-Theoretic Perspective

InfoNCE focuses on the "coarse-grained" distinction of overall embeddings (positive vs. negative), while CoSENT concentrates on the "fine-grained" relative relationships between embeddings. They are complementary: InfoNCE establishes the global structure of the embedding space, while CoSENT refines local ranking relationships on top of it.

## Key Experimental Results

### Main Results - NL2Code Search (MRR@1000)

| Method | CoSQA | AdvTest | CSN Python | CSN Java | CSN JS | CSN PHP | CSN Go | CSN Ruby | CSN Avg |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| OpenAI-ada-002 | 44.23 | 38.08 | 68.02 | 71.49 | 67.50 | 60.62 | 85.63 | 74.72 | 71.33 |
| Text-Embed-3-Large | 55.38 | 46.84 | 70.84 | 72.92 | 68.13 | 59.59 | 87.64 | 75.25 | 72.40 |
| CodeSage-large | 47.53 | 52.67 | 70.77 | 70.21 | 69.50 | 61.33 | 83.71 | 71.92 | 71.24 |
| **OASIS** | **55.77** | **57.27** | **73.69** | **73.97** | **69.80** | **63.84** | **88.21** | **75.47** | **74.16** |

### Main Results - Code2Code Search (MAP)

| Method | Python | Java | JS | TS | C# | C | Ruby | PHP | Go | Avg |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Text-Embed-3-Large | 41.51 | 25.75 | 22.40 | 22.45 | 11.56 | 32.82 | 41.70 | 43.47 | 21.57 | 29.25 |
| CodeSage-large | 46.70 | 33.13 | 37.16 | 41.18 | 16.81 | 32.89 | 54.12 | 52.13 | 32.48 | 38.51 |
| **OASIS** | **66.27** | **37.26** | **47.71** | **51.15** | **22.18** | **49.38** | **58.60** | **64.06** | **34.18** | **47.87** |

On the Code2Code task, OASIS achieves a relative improvement of 24.31% (9.36% absolute) compared to CodeSage-large, far exceeding the gains observed in the NL2Code task.

### Ablation Study (NL2Code Average MRR@1000)

| Configuration | MRR |
|------|:---:|
| **OASIS (Full)** | **69.75** |
| w/o Similarity Refinement | 69.15 |
| Order-only objective (CoSENT) | 67.33 |
| Contrastive-only objective (InfoNCE) | 65.49 |
| AST candidate strategy only | 69.46 |
| Threshold candidate strategy only | 69.26 |

### Hard Sample Subset - CSN Python (MRR@1000)

| Method | MRR |
|------|:---:|
| CodeSage-Large | 45.67 |
| Text-Embedding-3-Large | 45.78 |
| **OASIS** | **51.13** |

### Key Findings

1. **Order labels contribute more than contrastive loss**: Using CoSENT alone (67.33) outperforms using InfoNCE alone (65.49), indicating that ordering relationship information between negative pairs is critical for code semantic learning.
2. **The improvements on Code2Code are significantly larger than on NL2Code**: Code2Code tasks are more challenging and rely more heavily on fine-grained semantic understanding, highlighting OASIS's distinct advantages in such scenarios.
3. **Outperforming the teacher labeling model**: Although OASIS's similarity labels are generated by Text-Embedding-3-Large, its final performance surpasses that of the labeling model itself, demonstrating that the efficacy of the proposed method does not depend on the ceiling of the labeling model's capability.
4. **Significant advantage on hard samples**: On the hard subset where all baseline models perform poorly, OASIS improves MRR by more than 5%, which suggests that order labels help the model acquire more fundamental semantic features.
5. **Complementarity of the two refinement strategies**: The GMM thresholding method targets numerical anomalies in annotations, whereas the AST-based method resolves structural similarity. Combining both yields the optimal result.

## Highlights & Insights

1. **Paradigm shift from "distinguishing positives/negatives" to "ranking negatives"**: This is the first work in code embedding to explore subtle differences among negative pairs. Unlike traditional hard negative mining (which focuses on individual hard negatives), OASIS establishes a total ordering among negatives via order labels.
2. **Clever design of repository-level data augmentation**: Code snippets within the same repository naturally exhibit high similarity but distinct functionalities, allowing the collection of code without manual annotations to yield massive volumes of high-quality negative pairs.
3. **Complementarity of program analysis and LLMs**: Structural information from ASTs captures similarity at the code level, while LLMs assess equivalence at the semantic level. They refine the annotation quality along different dimensions.
4. **Robust visual evidence**: MDS dimensionality reduction visualization reveals that, within OASIS's embedding space, queries are positioned closer to their target code, and retrieval spaces of different queries show less overlap.

## Limitations & Future Work

- The initial similarity labels depend on an external embedding model (Text-Embedding-3-Large), which bounds the labeling quality to the baseline capability of that model.
- Code search is only validated at the function level, leaving file-level or project-level search scenarios unexplored.
- The GMM thresholding method assumes a bimodal Gaussian distribution for similarity, and its applicability to multimodal or highly skewed distributions remains unverified.
- The cost of using LLMs for docstring generation and equivalence judgment in the data synthesis pipeline is relatively high, which may limit scalability.

## Related Work & Insights

- **CodeSage**: A pioneer in code embeddings using contrastive learning and large-scale pre-training. OASIS builds on CodeSage by introducing order labels to achieve further improvements.
- **CoSENT**: A prior work in text embedding that employs STS labels for order optimization. OASIS ports this paradigm to the code domain and successfully addresses the challenge of annotating code similarities.
- **Implications for NLP Embeddings**: The concept of order labels can be extended to other embedding tasks characterized by sparse context, such as SQL query embeddings, configuration file embeddings, etc.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The integration of order labels with dual losses is innovative, being the first to systematically investigate fine-grained variations among negative pairs in code embeddings.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 2 task categories (NL2Code & Code2Code) × 3 benchmarks, combined with rigorous ablation studies, hard-subset analysis, and visualization.
- **Writing Quality**: ⭐⭐⭐⭐ The methodology and motivation are clearly framed, with Figure 1's illustration serving as an intuitive example.
- **Value**: ⭐⭐⭐⭐ High practical value; the 53M training dataset and 1.5B model weights are open-sourced and ready for immediate deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] CoRet: Improved Retriever for Code Editing](coret_improved_retriever_for_code_editing.md)
- [\[ACL 2025\] GALLa: Graph Aligned Large Language Models for Improved Source Code Understanding](galla_graph_aligned_large_language_models.md)
- [\[ICLR 2026\] RESCUE: Retrieval Augmented Secure Code Generation](../../ICLR2026/code_intelligence/rescue_retrieval_augmented_secure_code_generation.md)
- [\[ACL 2026\] MARS2: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation](../../ACL2026/code_intelligence/mars2_scaling_multi-agent_tree_search_via_reinforcement_learning_for_code_genera.md)
- [\[AAAI 2026\] EquaCode: A Multi-Strategy Jailbreak Approach for Large Language Models via Equation Solving and Code Completion](../../AAAI2026/code_intelligence/equacode_a_multi-strategy_jailbreak_approach_for_large_language_models_via_equat.md)

</div>

<!-- RELATED:END -->
