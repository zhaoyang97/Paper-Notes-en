---
title: >-
  [Paper Note] Causally Reliable Concept Bottleneck Models
description: >-
  [NeurIPS 2025][AI Safety][concept bottleneck models] This paper proposes C2BM (Causally reliable Concept Bottleneck Models), which organizes the concept bottleneck as a causal graph structure. By combining observational data with background knowledge, C2BM automatically learns causal relationships, achieving significantly improved causal reliability, intervention responsiveness, and fairness while maintaining classification accuracy.
tags:
  - NeurIPS 2025
  - AI Safety
  - concept bottleneck models
  - causal reasoning
  - structural causal model
  - interpretability
  - fairness
date: 2026-05-08
content_hash: 3cf10cd4e9e79046
---

# Causally Reliable Concept Bottleneck Models

**Conference**: NeurIPS 2025
**arXiv**: [2503.04363](https://arxiv.org/abs/2503.04363)
**Code**: Supplementary material included at submission
**Area**: AI Safety
**Keywords**: concept bottleneck models, causal reasoning, structural causal model, interpretability, fairness

## TL;DR

This paper proposes C2BM (Causally reliable Concept Bottleneck Models), which organizes the concept bottleneck as a causal graph structure. By combining observational data with background knowledge, C2BM automatically learns causal relationships, achieving significantly improved causal reliability, intervention responsiveness, and fairness while maintaining classification accuracy.

## Background & Motivation

**State of the Field**: Concept Bottleneck Models (CBMs) represent a prominent paradigm in interpretable deep learning, achieving transparency by forcing models to reason through a layer of human-understandable concepts. CBMs decompose prediction into two steps: an encoder maps inputs to concepts, and a decoder predicts task labels from those concepts.

**Limitations of Prior Work**: Existing CBMs adopt a bipartite graph structure, assuming all concepts are independent and directly influence the output. This assumption is overly simplistic: (1) it ignores causal dependencies among concepts, potentially producing misleading explanations (e.g., attributing lung cancer to both "coughing" and "smoking," implying that reducing coughing could lower cancer risk); (2) the independence assumption prevents intervention effects from propagating across related concepts; (3) models learn statistical correlations rather than causal relationships, making them susceptible to spurious associations.

**Root Cause**: CBMs are inherently associative models whose decision-making reflects statistical correlations in data rather than real-world causal mechanisms. This prevents them from supporting causal inference, limits out-of-distribution generalization, and hinders the enforcement of fairness constraints.

**Paper Goals**: (a) How can the concept bottleneck be structured according to causal mechanisms? (b) How can concepts and the causal graph be discovered automatically without human expert annotation? (c) How can causal reliability be improved without sacrificing accuracy?

**Starting Point**: Structural Causal Models (SCMs) are employed to organize the concept bottleneck as a causal graph, with causal relationships automatically discovered from unstructured background knowledge via LLM + RAG.

**Core Idea**: A causal graph structure is embedded within the concept bottleneck; hypernetworks adaptively parameterize the structural equations, aligning the model's reasoning process with true causal mechanisms.

## Method

### Overall Architecture

C2BM consists of three core modules: (1) **Concept Discovery and Annotation** — an LLM is used to identify relevant concepts, which are then annotated using CLIP; (2) **Causal Graph Discovery** — the GES causal discovery algorithm is combined with LLM+RAG queries to determine causal relationships among concepts; (3) **The C2BM Model** — comprising a neural encoder $\mathbf{g}(\cdot)$ and a parameterized SCM $\mathcal{M}_{\boldsymbol{\Theta}}$.

The raw input $X$ is processed by the encoder to predict exogenous variables (high-dimensional embeddings) $\mathcal{U} = \{U_i\}_{i=1}^C$; information then flows along the causal graph from source nodes to sink nodes. The value of each endogenous variable $V_i$ is predicted from its causal parents via structural equations.

### Key Designs

1. **Causal Bottleneck**:

    - *Function*: Replaces the flat bipartite graph structure of CBMs with a DAG that organizes causal relationships among concepts.
    - *Mechanism*: C2BM is defined as $\langle \mathbf{g}, \mathcal{M}_{\boldsymbol{\Theta}} \rangle$, where the SCM is $\langle \mathcal{V}, \mathcal{U}, \mathcal{F}_{\boldsymbol{\Theta}}, P(\mathcal{U}|X) \rangle$. Exogenous variables are first obtained from the encoder, and each concept is then computed sequentially according to the topological order of the DAG.
    - *Design Motivation*: The causal structure enables interventions to propagate along the graph, blocking spurious correlation paths and supporting fairness constraints.

2. **Adaptive Structural Equations**:

    - *Function*: Learns an interpretable causal mechanism for each concept.
    - *Mechanism*: The structural equations take the linear form $V_i = \sum_{V_j \in \text{PA}_i} [\boldsymbol{\theta}_{f_i}]_j V_j$, but the parameters $\boldsymbol{\theta}_{f_i}$ are not fixed — they are adaptively predicted for each input by a hypernetwork $\mathbf{r}_i(U_i)$, achieving locally linear yet globally nonlinear expressive capacity.
    - *Design Motivation*: Linear structural equations ensure interpretability (each concept is a weighted linear combination of its parents), while hypernetworks ensure expressive power (the paper proves C2BM is a universal approximator).

3. **Automated Causal Graph Construction Pipeline**:

    - *Function*: Automatically discovers concepts and the causal graph without requiring human experts.
    - *Mechanism*: (a) Relevant concepts are queried from an LLM and annotated with CLIP; (b) the GES algorithm learns a CPDAG (a partially directed graph with undirected edges) from observational data; (c) LLM+RAG queries over background knowledge orient the undirected edges and remove spurious ones, with each query repeated 10 times and a majority vote taken.
    - *Design Motivation*: Data-driven causal discovery alone cannot uniquely identify a DAG; incorporating background knowledge (e.g., scientific literature) effectively narrows the candidate graph space.

### Loss & Training

The training objective is to maximize the empirical log-likelihood:

$$\boldsymbol{\phi}^* = \arg\max_{\boldsymbol{\phi}} \sum_{\mathcal{D}} \sum_{i=1}^C \log P(V_i \mid \text{PA}_i, U_i; \mathbf{r}(\mathcal{E}, \mathcal{U})_i)$$

The joint conditional distribution factorizes into a product of independent factors according to the Markov condition of the causal graph. The encoder and hypernetwork parameters are learned jointly in an end-to-end manner.

## Key Experimental Results

### Main Results

| Dataset | OpaqNN | CBM+lin | CEM | SCBM | C2BM |
|--------|--------|---------|-----|------|------|
| Asia | 71.0 | 71.2 | 71.1 | 70.7 | **71.4** |
| Sachs | 65.83 | 65.44 | 65.93 | **66.30** | 65.33 |
| Hailfinder | 72.0 | 72.2 | 71.5 | 73.4 | **74.1** |
| cMNIST | 91.24 | 93.92 | 93.72 | 94.02 | **94.18** |
| CelebA | 74.97 | 71.07 | **74.72** | 72.15 | **74.73** |
| Pneumoth. | 80.0 | 76.6 | **80.1** | 78.4 | **80.5** |

C2BM matches or outperforms the strongest baselines (OpaqNN, CEM) on most datasets, while being the only model that provides causal reliability.

### Ablation Study — Causal Graph Quality

| Metric | Flat CBM | CD Only | CD + LLM |
|------|---------|-------------|----------|
| Hamming (Asia) | 6.5 | 0.7 | **0.3** |
| Incorrect edges (Sachs, 17 edges total) | 23 | 17 | **7** |
| Incorrect edges (Alarm, 46 edges total) | 78 | 13 | **10** |

LLM background knowledge effectively reduces erroneous edges in the causal graph, correctly recovering 10 additional edges on Sachs.

### Key Findings
- **Intervention Experiments**: C2BM achieves the fastest and largest accuracy gains when concepts are intervened upon layer by layer, as the causal structure naturally propagates intervention effects to child nodes.
- **Debiasing**: On biased cMNIST, C2BM correctly removes the spurious Color→Parity edge via the causal graph; after intervening on the *number* concept, accuracy reaches ~90%.
- **Fairness**: C2BM is the only model capable of reducing CaCE (Attractive→Should be Hired) to **0.0%** via do-intervention; all other CBM/CEM/SCBM variants fail to fully block the bias pathway.

## Highlights & Insights
- **Causal Structure = Better Interventions**: The causal graph allows concept intervention effects to propagate along the DAG, rather than being confined to individual nodes as in bipartite structures. Intervening on just 1–2 high-level concepts substantially improves accuracy across all downstream nodes.
- **Hypernetworks + Linear Equations = Interpretable Nonlinearity**: The structural equations are linear (interpretable), but their parameters are dynamically predicted by hypernetworks (expressive), achieving an elegant balance between interpretability and representational capacity.
- **LLM+RAG for Automated Causal Graph Construction**: The challenging problem of orienting undirected edges in causal discovery is reframed as an LLM background knowledge query task, substantially reducing dependence on human experts.

## Limitations & Future Work
- Causal graph quality depends on the accuracy of LLM background knowledge; biases in the knowledge base propagate into the model.
- The scalability of causal structure learning is constrained by existing causal discovery algorithms (GES incurs high computational cost in high-dimensional settings).
- Out-of-distribution generalization of the encoder is not guaranteed, potentially affecting OOD performance of the SCM.
- Hidden confounders are not accounted for; the causal graph is assumed to be a DAG.
- Experiments are conducted on relatively small datasets; performance on large-scale visual tasks (e.g., ImageNet) remains unvalidated.

## Related Work & Insights
- **vs. CBM/CEM**: CBM/CEM adopt a flat bipartite structure in which all concepts are independent and directly connected to the task; C2BM introduces a causal graph structure with hierarchical causal relationships among concepts.
- **vs. SCBM**: SCBM relaxes the concept independence assumption but captures only associative relationships; C2BM captures causal relationships, enabling more precise intervention effects.
- **vs. DiConStruct**: DiConStruct is a post-hoc method that may be misaligned with DNN outputs and relies solely on observational data; C2BM is an by-design method that incorporates background knowledge.
- This work is highly relevant to interpretable AI and fairness constraint research, as the causal bottleneck can be directly applied to enforce algorithmic fairness.

## Rating
- Novelty: ⭐⭐⭐⭐ Integrating causal reasoning into concept bottlenecks is a natural yet important new direction.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers five dimensions — accuracy, causal reliability, intervention, debiasing, and fairness — with thorough ablations.
- Writing Quality: ⭐⭐⭐⭐ Rigorous formalization and clear pipeline description.
- Value: ⭐⭐⭐⭐⭐ Causally reliable interpretable models carry significant practical importance for safety-critical domains such as healthcare and law.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Preserving Task-Relevant Information Under Linear Concept Removal](preserving_task-relevant_information_under_linear_concept_removal.md)
- [\[NeurIPS 2025\] Factor Decorrelation Enhanced Data Removal from Deep Predictive Models](factor_decorrelation_enhanced_data_removal_from_deep_predictive_models.md)
- [\[CVPR 2026\] LogitDynamics: Reliable ViT Error Detection from Layerwise Logit Trajectories](../../CVPR2026/ai_safety/logitdynamics_vit_error_detection.md)
- [\[NeurIPS 2025\] Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond)](artificial_hivemind_the_open-ended_homogeneity_of_language_models_and_beyond.md)
- [\[ICCV 2025\] A Framework for Double-Blind Federated Adaptation of Foundation Models](../../ICCV2025/ai_safety/a_framework_for_double-blind_federated_adaptation_of_foundation_models.md)

<!-- RELATED:END -->
