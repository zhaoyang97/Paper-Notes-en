---
title: >-
  [Paper Note] SPOT-Trip: Dual-Preference Driven Out-of-Town Trip Recommendation
description: >-
  [NeurIPS 2025][Graph Learning][Trip recommendation] This paper proposes SPOT-Trip, the first framework to systematically address out-of-town trip recommendation. By integrating knowledge graph-enhanced static preference learning, neural ODE-driven dynamic preference learning, and a preference fusion module, the framework achieves up to 17.01% improvement over state-of-the-art baselines on two real-world datasets.
tags:
  - NeurIPS 2025
  - Graph Learning
  - Trip recommendation
  - knowledge graph
  - neural ODE
  - temporal point process
  - static-dynamic preference fusion
date: 2026-05-08
content_hash: e4daa391f207a6a6
---

# SPOT-Trip: Dual-Preference Driven Out-of-Town Trip Recommendation

**Conference**: NeurIPS 2025
**arXiv**: [2506.01705](https://arxiv.org/abs/2506.01705)
**Code**: Not available
**Area**: Graph Learning
**Keywords**: Trip recommendation, knowledge graph, neural ODE, temporal point process, static-dynamic preference fusion

## TL;DR

This paper proposes SPOT-Trip, the first framework to systematically address out-of-town trip recommendation. By integrating knowledge graph-enhanced static preference learning, neural ODE-driven dynamic preference learning, and a preference fusion module, the framework achieves up to 17.01% improvement over state-of-the-art baselines on two real-world datasets.

## Background & Motivation

When users travel from their home city to an unfamiliar destination, a recommender system must suggest a complete itinerary comprising multiple intermediate points of interest (POIs). This problem poses three key challenges:

**Data Sparsity**: Users typically have few or no historical check-in records at the destination, rendering conventional recommendation models ineffective.

**Preference Duality**: User preferences encompass stable static interests (long-term behavioral patterns) and context-sensitive dynamic preferences (varying with time and location). Existing methods conflate the two.

**Preference Transfer**: Check-in behaviors in the home city may diverge from out-of-town behaviors (interest drift), making effective knowledge transfer a critical challenge.

**Core Innovation**: This work is the first to formally define out-of-town trip recommendation as the task of recommending an intermediate POI sequence given a departure point, destination, and number of stops, while explicitly disentangling static and dynamic preferences.

## Method

### Overall Architecture

SPOT-Trip consists of three modules: Knowledge-enhanced Static Preference Learning (KSPL), ODE-based Dynamic Preference Learning (ODPL), and a Static-Dynamic Preference Fusion module. All three modules are trained end-to-end via a joint loss function.

### Key Designs

1. **Knowledge-enhanced Static Preference Learning (KSPL)**

   A POI attribute knowledge graph $\mathcal{G}_k = (v, r, e)$ is constructed to encode semantic relationships between POIs and attribute entities (e.g., "Balboa Park → Located in → San Diego"). Relation-aware attentive aggregation generates enriched POI embeddings:

    $\bar{\mathbf{v}} = \mathbf{v} + \sum_{e \in \mathcal{N}_v} \alpha(e, r_{e,v}, v) \mathbf{e}$

   where the attention weight $\alpha$ captures entity- and relation-specific semantic associations. Static preference alignment maps home-city preferences to the destination via an MLP:

    $\bar{\mathbf{P}}^o = \phi(\mathbf{W}_S \bar{\mathbf{u}}^h + \mathbf{b}_S)$

   An Euclidean distance loss bridges the inferred and ground-truth out-of-town preferences: $\mathcal{L}_S = \sum_u \|\bar{\mathbf{P}}^o - \bar{\mathbf{u}}^o\|_2^2$

   **Design Motivation**: External semantic knowledge from the KG alleviates data sparsity at the destination, while treating static and spatial information separately avoids conflicting interactions.

2. **ODE-based Dynamic Preference Learning (ODPL)**

   A neural ODE models the continuous-time evolution of user dynamic preferences:

    $\frac{d\tilde{\mathbf{p}}_t^o}{dt} = f(\tilde{\mathbf{p}}_{t_1}^o)$

   A temporal point process characterizes the instantaneous probability of each behavioral event via the intensity function $\lambda(\cdot)$ of a non-homogeneous Poisson process:

    $t_n \sim \text{NHPP}(\lambda(\tilde{\mathbf{p}}_t^o))$

   Variational inference is used to estimate the posterior over the latent initial state, with variational parameters derived by aggregating home-city behavioral sequences through a Transformer encoder. The dynamic learning loss is based on the ELBO:

    $\mathcal{L}_D = -\sum_{u \in \mathcal{U}} \left[\underbrace{\text{reconstructed behavior log-likelihood}}_{(i)} + \underbrace{\text{temporal point process log-likelihood}}_{(ii)} - \underbrace{\text{KL divergence}}_{(iii)}\right]$

   **Design Motivation**: Check-in data are irregularly sampled in time; neural ODEs naturally handle continuous-time modeling, while point processes quantify the probability of behavioral occurrence.

3. **Static-Dynamic Preference Fusion**

   The query representation, static preference, and dynamic preference sequence are concatenated and passed through a nonlinear prediction head to compute per-POI logits:

    $z_{u,n,i} = \phi(\mathbf{W}_R [\mathbf{Q}^o \| \bar{\mathbf{P}}^o \| \tilde{\mathbf{P}}^o] + \mathbf{b}_R)$

### Loss & Training

Three losses are jointly optimized:

$$\mathcal{L} = \beta_1 \mathcal{L}_S + \beta_2 \mathcal{L}_D + \beta_3 \mathcal{L}_R$$

The recommendation loss $\mathcal{L}_R$ is cross-entropy. During inference, Top-p sampling is used to generate the intermediate POI sequence, with normalized positions substituted for actual timestamps.

## Key Experimental Results

### Main Results

| Method | Foursquare F₁ | Foursquare PairsF₁ | Yelp F₁ | Yelp PairsF₁ |
|------|-------------|-------------------|---------|-------------|
| AR-Trip | 0.0304 | 0.0045 | 0.0307 | 0.0153 |
| Base+KDDC | 0.0375 | 0.0079 | 0.0341 | 0.0156 |
| Base+CNN-ODE | 0.0367 | 0.0094 | 0.0326 | 0.0168 |
| **SPOT-Trip** | **0.0400** | **0.0109** | **0.0399** | **0.0190** |
| Gain | +6.67% | +15.96% | +17.01% | +13.90% |

### Ablation Study

| Variant | Yelp F₁ | Yelp PairsF₁ | Note |
|------|---------|-------------|------|
| SPOT-Trip | 0.0399 | 0.0190 | Full model |
| w/o KS | ↓10.02% | ↓21.79% | Remove knowledge-enhanced static preference |
| w/o OD | Degraded | Degraded | Remove ODE dynamic preference |
| w/o SI | Significant drop | Significant drop | Remove spatial information |

### Data Sparsity Experiment

| Data Ratio | SPOT-Trip F₁ (Yelp) | Base+KDDC F₁ | Base+CNN-ODE F₁ |
|---------|--------------------|--------------| ----------------|
| 40% | 0.0327 | 0.0321 | 0.0313 |
| 60% | 0.0343 | 0.0331 | 0.0327 |
| 80% | 0.0384 | 0.0338 | 0.0333 |
| 100% | 0.0399 | 0.0341 | 0.0326 |

### Key Findings

1. Methods that incorporate home-city information consistently outperform trip recommendation baselines that do not, validating the necessity of preference transfer.
2. Removing the KSPL module causes a sharp 21.79% drop in PairsF₁, demonstrating the critical role of semantic knowledge graphs in sequential recommendation.
3. SPOT-Trip with only 40% of home-city data still outperforms baselines trained on 100% of the data.
4. The model produces reasonable recommendations even when only the departure point or only the destination is provided (e.g., predicted endpoints correspond to train stations), demonstrating an intrinsic understanding of travel patterns.

## Highlights & Insights

1. **Novel Problem Formulation**: The first systematic formalization of out-of-town trip recommendation as the task of generating intermediate POI sequences given departure and destination points.
2. **Decoupled Preference Design**: Static preferences are modeled via KG semantics while dynamic preferences are modeled via neural ODEs in continuous time, preventing conflicting interactions.
3. **Alternating TransE Training**: KG embeddings and the recommendation model are optimized alternately, enriching the multi-relational semantic space.

## Limitations & Future Work

- KG construction relies on attribute information provided by the dataset, limiting applicability to settings with sparse attribute coverage.
- During inference, actual timestamps are unavailable; using normalized positions as a proxy may introduce bias in dynamic preference modeling.
- Top-p sampling during sequence generation may introduce non-deterministic uncertainty.

## Related Work & Insights

- **Out-of-town recommendation**: KDDC, CNN-ODE, PPROC
- **Trip recommendation**: GraphTrip, AR-Trip, MatTrip
- **Neural ODE applications**: Continuous-time modeling advantages of Neural ODEs in spatiotemporal tasks
- **Knowledge-enhanced recommendation**: KG-based POI recommendation

## Rating

- Novelty: ⭐⭐⭐⭐ — First to define the out-of-town trip recommendation problem and propose a complete framework
- Experimental Thoroughness: ⭐⭐⭐⭐ — Two datasets, nine baselines, ablation, sparsity, and parameter sensitivity analyses
- Writing Quality: ⭐⭐⭐⭐ — Clear problem definition with logically progressive module design
- Value: ⭐⭐⭐⭐ — High practical value for real-world trip recommendation applications

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] PKD: Preference-driven Knowledge Distillation for Few-shot Node Classification](preference-driven_knowledge_distillation_for_few-shot_node_classification.md)
- [\[NeurIPS 2025\] DuetGraph: Coarse-to-Fine Knowledge Graph Reasoning with Dual-Pathway Global-Local Fusion](duetgraph_coarse-to-fine_knowledge_graph_reasoning_with_dual-pathway_global-loca.md)
- [\[NeurIPS 2025\] ReMindRAG: Low-Cost LLM-Guided Knowledge Graph Traversal for Efficient RAG](remindrag_low-cost_llm-guided_knowledge_graph_traversal_for_efficient_rag.md)
- [\[NeurIPS 2025\] GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation](gfm-rag_graph_foundation_model_for_retrieval_augmented_generation.md)
- [\[AAAI 2026\] MoToRec: Sparse-Regularized Multimodal Tokenization for Cold-Start Recommendation](../../AAAI2026/graph_learning/motorec_sparse-regularized_multimodal_tokenization_for_cold-start_recommendation.md)

</div>

<!-- RELATED:END -->
