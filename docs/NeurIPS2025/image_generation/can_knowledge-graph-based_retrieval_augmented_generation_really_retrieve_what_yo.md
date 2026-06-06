---
title: >-
  [Paper Note] Can Knowledge-Graph-based Retrieval Augmented Generation Really Retrieve What You Need?
description: >-
  [NeurIPS 2025][Image Generation][Knowledge Graph] This paper proposes GraphFlow, a framework that models retrieval over knowledge graphs as a flow-matching problem under GFlowNet…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Knowledge Graph"
  - "Retrieval-Augmented Generation"
  - "GFlowNet"
  - "Process Reward Model"
  - "Optimal Transport"
date: 2026-05-08
content_hash: cf6fa95cc8f53521
---

# Can Knowledge-Graph-based Retrieval Augmented Generation Really Retrieve What You Need?

**Conference**: NeurIPS 2025
**arXiv**: [2510.16582](https://arxiv.org/abs/2510.16582)  
**Code**: [GitHub](https://github.com/Samyu0304/GraphFlow)  
**Area**: Image Generation
**Keywords**: Knowledge Graph, Retrieval-Augmented Generation, GFlowNet, Process Reward Model, Optimal Transport

## TL;DR

This paper proposes GraphFlow, a framework that models retrieval over knowledge graphs as a flow-matching problem under GFlowNet, jointly training a retrieval policy and flow estimator via a detailed balance objective and local exploration strategy. On the STaRK benchmark, GraphFlow surpasses GPT-4o by approximately 10% in both retrieval accuracy and diversity.

## Background & Motivation

Knowledge-graph (KG)-based retrieval-augmented generation (RAG) is an important approach for enhancing LLM accuracy, as KGs provide structured relational information and interpretability. However, existing KG-RAG methods face two key challenges when handling **complex queries**:

**Fusion of structural and textual information**: Simple relational queries (e.g., "Who is Alice's father?") can be answered by retrieving relational triples alone, whereas complex queries (e.g., "List papers published at University A related to topic B") require simultaneously understanding author–institution relationships and paper content.

**Retrieval diversity**: Complex queries correspond to **multiple** retrieval targets, yet existing methods tend to maximize likelihood while neglecting diversity.

Process reward models (PRMs) can provide step-wise guidance for multi-step retrieval, but training PRMs requires expensive **process-level supervision**. In KG retrieval, only **outcome-level rewards** (whether a retrieval trajectory supports the query) are readily available, while step-by-step annotation of intermediate steps is prohibitively costly.

## Method

### Overall Architecture

GraphFlow models KG retrieval as an **energy-based trajectory distribution**:

$$P(\tau) = \prod_{t=0}^{T} P(s_{t+1} \mid s_t) \propto R(\tau)$$

That is, retrieval trajectories with higher rewards should be sampled with higher probability. This differs from the likelihood-maximization objective of conventional methods—GraphFlow inherently promotes a balance between **diversity and accuracy** in retrieval results.

Core components:
- **Retrieval policy** $P(s_{t+1} \mid s_t)$: A forward policy that determines which neighboring node to move to from the current node on the KG.
- **Flow estimator** $F(s): \mathcal{S} \to \mathbb{R}_{\geq 0}$: Assigns non-negative flow values to each intermediate state, decomposing terminal rewards into intermediate steps.
- **LLM encoder**: A shared LLM backbone (with LoRA) that encodes states and state–action pairs.

### Key Designs

**Detailed Balance Condition**:

$$F(s_t) \cdot P(s_{t+1} \mid s_t) = F(s_{t+1}) \cdot P_B(s_t \mid s_{t+1})$$

When this condition holds across all transitions, the trajectories induced by policy $P(s_{t+1} \mid s_t)$ satisfy the energy-based distribution objective. Since KG retrieval is non-reversible, the backward policy is set to $P_B(s_t \mid s_{t+1}) = 1$.

**Local Exploration Strategy**: Enforcing detailed balance over the full KG is computationally prohibitive. For each non-terminal state $s_t$, GraphFlow performs $k$ exploratory steps to generate $k$ alternative actions $\{a'_{t,1}, \ldots, a'_{t,k}\}$, forming $k+1$ candidate transitions. The optimization objective is:

$$\mathcal{L}_{\text{DBLE}}(s_t) = \sum_{i=0}^{k} \left[\log F(s_t) - \log F(s'_{t+1,i}) + r_\theta(s_t, a'_{t,i}) - \log \sum_{i=0}^{k} e^{r_\theta(s_t, a'_{t,i})}\right]^2$$

**Self-loop Termination Mechanism**: A special self-loop action is introduced, allowing the policy to adaptively decide when to stop retrieval rather than relying on a fixed number of steps.

**Relationship to SFT/PRM**: When $\log F(s_t) = 1$ and $\log F(s'_{t+1,i}) = 0$, GraphFlow reduces to behavior cloning. The full flow estimator endows the model with stronger generalization and diversity capabilities.

### Loss & Training

The detailed balance loss $\mathcal{L}_{\text{DBLE}}$ is computed independently at each non-terminal state along every sampled trajectory. Boundary conditions $\log F(s_0) = \log F(s_T) = 1$ ensure flow consistency at the start and end of each trajectory. The policy head and flow head share the LLM encoder and are efficiently fine-tuned via LoRA.

## Key Experimental Results

### Main Results

Retrieval accuracy on the STaRK benchmark (STaRK-AMAZON, STaRK-MAG, STaRK-PRIME):

| Method | AMAZON Hit@1 | AMAZON Hit@5 | MAG Hit@1 | MAG Hit@5 | PRIME Hit@1 | PRIME Hit@5 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| DenseRetriever | 6.10 | 15.85 | 24.44 | 40.23 | 5.43 | 13.07 |
| ToG+GPT4o | 20.67 | 41.38 | 23.33 | 56.67 | 16.67 | 39.77 |
| SFT | 8.16 | 15.30 | 26.53 | 28.57 | 27.50 | 40.07 |
| PRM | 20.09 | 26.25 | 26.05 | 28.00 | 21.01 | 46.72 |
| **GraphFlow** | **19.63** | **44.17** | **29.32** | **58.64** | **39.84** | **71.71** |
| **GraphFlow (Rerank)** | **47.85** | **65.03** | **39.09** | **57.51** | **51.39** | **72.11** |

### Ablation Study

Retrieval diversity metrics (Recall@20 and Diverse-Recall@20):

| Method | AMAZON R@20 | AMAZON D-R@20 | MAG R@20 | MAG D-R@20 | PRIME R@20 | PRIME D-R@20 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| ToG+GPT4o | 25.81 | 23.70 | 48.03 | 47.71 | 54.35 | 54.35 |
| PRM | 35.72 | 18.94 | 36.73 | 36.73 | 45.97 | 45.97 |
| **GraphFlow** | **36.15** | **36.15** | **57.18** | **57.18** | **79.71** | **79.59** |

GraphFlow's R@20 and D-R@20 are nearly equal, indicating that its retrieved results are not only numerous but also diverse. By contrast, PRM achieves R@20 = 35.72 on AMAZON but D-R@20 of only 18.94, revealing a severe redundancy problem.

### Key Findings

1. GraphFlow achieves an average Hit@1 improvement of approximately 10% in the reranking setting, surpassing the GPT-4o baseline.
2. On the PRIME dataset, GraphFlow's R@20 reaches 79.71%, far exceeding the second-best method ToG+GPT4o at 54.35%.
3. The flow estimator effectively decomposes outcome-level rewards into process-level signals without requiring explicit process-level annotations.
4. The local exploration strategy avoids ineffective exploration in low-reward regions, significantly improving training efficiency.

## Highlights & Insights

- ⭐ GFlowNet is elegantly introduced into KG-RAG: the principle of "sampling probability proportional to reward" naturally resolves the accuracy–diversity trade-off.
- ⭐ The flow estimator serves as a "free" process reward model, circumventing the need for costly process-level annotations.
- The self-loop termination mechanism enables the retrieval policy to adaptively determine search depth, offering greater flexibility than a fixed number of steps.
- A unified perspective with SFT/PRM (GraphFlow as their generalization) provides clear theoretical understanding.

## Limitations & Future Work

- Experiments are conducted solely on three datasets from the STaRK benchmark; generalization to other mainstream KG-QA benchmarks (e.g., WebQSP, CWQ) remains unverified.
- The LLM encoder incurs substantial computational overhead, as long documents must be encoded at each retrieval step, raising scalability concerns.
- The analysis of how the local exploration hyperparameter $k$ affects performance is insufficiently thorough.
- Training relies only on positive trajectories that reach target documents, which may result in inadequate coverage of negative examples.

## Related Work & Insights

GraphFlow bridges two research directions: GFlowNet (generative flow networks) and KG-RAG. GFlowNet was originally proposed for diversity-oriented sampling in molecular generation; this paper extends it to multi-step retrieval decisions over graph structures. Future work may consider scaling this approach to larger KGs (e.g., Wikidata) or combining it with community detection methods from GraphRAG. The flow estimator as a surrogate for process rewards also offers insights into credit assignment in LLM reasoning.

## Rating

⭐⭐⭐⭐ (4/5)

| Dimension | Rating |
|:--|:--:|
| Novelty | ⭐⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |

The introduction of GFlowNet into KG-RAG is highly innovative, the theoretical framework is clear, and the experimental gains are substantial. The primary limitations are the restricted coverage of experimental datasets and the insufficient analysis of computational efficiency.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] TruthfulRAG: Resolving Factual-level Conflicts in Retrieval-Augmented Generation with Knowledge Graphs](../../AAAI2026/image_generation/truthfulrag_resolving_factual-level_conflicts_in_retrieval-augmented_generation_.md)
- [\[NeurIPS 2025\] Graph Diffusion that can Insert and Delete](graph_diffusion_that_can_insert_and_delete.md)
- [\[NeurIPS 2025\] Graph Distance as Surprise: Free Energy Minimization in Knowledge Graph Reasoning](graph_distance_as_surprise_free_energy_minimization_in_knowledge_graph_reasoning.md)
- [\[NeurIPS 2025\] Highlighting What Matters: Promptable Embeddings for Attribute-Focused Image Retrieval](highlighting_what_matters_promptable_embeddings_for_attribute-focused_image_retr.md)
- [\[AAAI 2026\] Improved Masked Image Generation with Knowledge-Augmented Token Representations](../../AAAI2026/image_generation/improved_masked_image_generation_with_knowledge-augmented_token_representations.md)

</div>

<!-- RELATED:END -->
