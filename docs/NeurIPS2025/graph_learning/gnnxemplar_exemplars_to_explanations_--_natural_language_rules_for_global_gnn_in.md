---
title: >-
  [Paper Note] GnnXemplar: Exemplars to Explanations -- Natural Language Rules for Global GNN Interpretability
description: >-
  [NeurIPS 2025][Graph Learning][graph neural network interpretability] This paper proposes GnnXemplar, a framework grounded in the cognitive-science Exemplar Theory. It selects representative nodes (exemplars) in the GNN embedding space and employs an LLM with iterative self-refinement to generate natural-language Boolean rules, achieving global interpretability for node-classification GNNs on large-scale graphs.
tags:
  - NeurIPS 2025
  - Graph Learning
  - graph neural network interpretability
  - global explanation
  - exemplar theory
  - natural language rules
  - coverage maximization
date: 2026-05-08
content_hash: da9e43a5543fb599
---

# GnnXemplar: Exemplars to Explanations -- Natural Language Rules for Global GNN Interpretability

**Conference**: NeurIPS 2025
**arXiv**: [2509.18376](https://arxiv.org/abs/2509.18376)
**Code**: [GitHub](https://github.com/idea-iitd/GnnXemplar.git)
**Area**: Graph Learning / GNN Interpretability
**Keywords**: graph neural network interpretability, global explanation, exemplar theory, natural language rules, coverage maximization

## TL;DR

This paper proposes GnnXemplar, a framework grounded in the cognitive-science Exemplar Theory. It selects representative nodes (exemplars) in the GNN embedding space and employs an LLM with iterative self-refinement to generate natural-language Boolean rules, achieving global interpretability for node-classification GNNs on large-scale graphs.

## Background & Motivation

**Background**: GNNs are widely applied to node classification tasks, yet their decision processes remain opaque. Existing local explanation methods (GNNExplainer, PGExplainer) explain only individual predictions, while global explanation methods are still underdeveloped.

**Limitations of Prior Work**:
- Existing global explainers (GNNInterpreter, GLGExplainer) primarily target motif discovery in small-scale graph classification tasks.
- Exact subgraph recurrence is exceedingly rare in large real-world graphs, and subgraph isomorphism is NP-hard, preventing scalability.
- High-dimensional continuous node attributes invalidate classical motif definitions.
- Subgraph visualization on large graphs exceeds human cognitive capacity.

**Key Challenge**: How can one provide global explanations on large-scale, high-dimensional attributed graphs that are simultaneously faithful to model decisions (high fidelity) and comprehensible to humans?

**Goal**: To deliver global explanations for node-classification GNNs on large-scale graphs, requiring scalability, high fidelity, and human interpretability.

**Key Insight**: The framework draws on the Exemplar Theory from cognitive science — humans categorize new objects by comparing them against representative exemplars stored in memory.

**Core Idea**: Identify representative exemplars in the GNN embedding space, then leverage an LLM with iterative self-refinement to generate natural-language Boolean rules as explanations for each exemplar.

## Method

### Overall Architecture

GnnXemplar consists of two major steps:
1. **Exemplar Identification**: Select a budget-constrained set of representative nodes in the GNN embedding space.
2. **Signature Discovery**: Employ an LLM with iterative self-refinement to generate natural-language Boolean rules for each exemplar.

### Key Designs

1. **Reverse k-NN and Representativeness Measure**:

    - **Function**: Quantifies the representativeness of each node in the embedding space.
    - **Design Motivation**: A node that frequently appears in the k-nearest-neighbor sets of other same-class nodes occupies a dense region of the embedding space and is therefore a strong exemplar candidate.
    - **Mechanism**: The reverse k-NN is defined as $\text{Rev-}k\text{-NN}(v) = \{u \in \mathcal{V}_{tr} \mid v \in k\text{-NN}(u), \Phi(v)=\Phi(u)\}$, and representativeness is defined as:
    $$\Pi(v) = \frac{|\text{Rev-}k\text{-NN}(v)|}{|\{u \in \mathcal{V}_{tr} \mid \Phi(v)=\Phi(u)\}|}$$
    - **Scalable Approximation**: Rev-k-NN is approximated by sampling $z$ nodes; a Chernoff bound guarantees that the required sample size is independent of total node count, reducing computational complexity from $\mathcal{O}(n^2)$ to $\mathcal{O}(n)$.

2. **Coverage Maximization**:

    - **Function**: Selects an exemplar set within budget $b$ to maximize coverage of the training set.
    - **Design Motivation**: A small number of exemplars should suffice to capture the behavioral patterns of a large number of same-class nodes.
    - **Mechanism**: The objective is $\Pi(\mathbb{A}) = |\bigcup_{v \in \mathbb{A}} \text{Rev-}k\text{-NN}(v)| / |\mathcal{V}_{tr}|$. The problem is proven NP-hard, but the objective function is monotone submodular; a greedy algorithm yields a $(1-1/e)$ approximation guarantee.
    - **Distinction**: Unlike motif-based methods that rely on subgraph isomorphism, this approach operates directly in the embedding space, naturally handling continuous attributes.

3. **LLM Self-Refinement for Signature Discovery**:

    - **Function**: Generates interpretable Boolean rules in natural-language form for each exemplar.
    - **Design Motivation**: Traditional subgraph visualization is infeasible on large graphs; natural language better aligns with human cognition.
    - **Mechanism**: The self-refine paradigm proceeds as follows:
      1. Sample positive examples (same-class nodes) and negative examples from Rev-k-NN.
      2. Provide node attributes, per-hop neighbor GNN prediction class distributions, and attribute distance statistics.
      3. The LLM first generates Python code implementing Boolean logic, then translates it into natural language.
      4. Iterative feedback: misclassified node information is returned to the LLM to improve the rules.
      5. Iteration continues until validation accuracy exceeds a threshold or a maximum iteration count is reached.
    - The global explanation for class $i$ is the disjunction (OR) of all exemplar signatures: $f_i(v) = \bigvee_{e \in \mathcal{E}_i} \sigma_e(v)$.

### Loss & Training

- This work explains pre-trained GNNs rather than training new models.
- GNN training: GAT is used for TAGCora; GCN is used for all other datasets (standard training settings).
- Exemplar selection: greedy algorithm with budget $b$ as a hyperparameter.
- LLM: iterative self-refinement until validation accuracy meets the threshold or the iteration limit is reached.

## Key Experimental Results

### Main Results

**Fidelity Comparison**:

| Method | TAGCora | Citeseer | WikiCS | ogbn-arxiv | Amazon-R | Questions | Minesweeper | BA-Shapes |
|--------|---------|----------|--------|-----------|----------|-----------|-------------|-----------|
| GNNInterpreter | NA | 0.50 | NA | NA | NA | NA | 0.50 | 0.47 |
| GCNeuron | 0.51 | 0.50 | OOM | OOM | 0.56 | OOM | 0.54 | 0.50 |
| GLGExplainer | NF | NF | OOM | OOM | NF | OOM | 0.22 | 0.30 |
| **GnnXemplar** | **0.83** | **0.92** | **0.78** | **0.84** | **0.82** | **0.92** | **0.86** | **0.93** |

NA = not applicable (requires discrete attributes), NF = formula generation failed, OOM = out of memory.

### Ablation Study

**Rev-k-NN vs. Random Selection and Self-Refinement vs. Zero-Shot**:

| Ablation | Effect |
|----------|--------|
| Rev-k-NN exemplars → random sampling | Fidelity drops noticeably; random nodes fail to cover semantically dense regions. |
| Self-refinement → zero-shot (one-shot LLM) | Fidelity drops significantly with higher variance. |

### Key Findings

- Existing global explainers almost entirely fail on large-scale graph node classification tasks (OOM / NF / NA).
- GnnXemplar achieves fidelity >0.78 across all 8 datasets, substantially outperforming baselines (which typically achieve ≈0.50, i.e., random-level performance).
- Natural-language rules are strongly preferred over subgraph visualization in a user study with 60 participants (200/300 participants chose text-based explanations, $p < 0.0001$).
- **Diagnostic capability**: On the Questions dataset, explanations from GnnXemplar that appeared "incorrect" in fact exposed that the GNN had learned spurious homophily patterns (rather than the expected heterophily), revealing systematic model failure on imbalanced classes.

## Highlights & Insights

- **Cognitive-science grounding**: A novel application of Exemplar Theory to AI interpretability, bringing psychological categorization theory into GNN explanation.
- **Theoretical guarantees**: NP-hardness proof for Rev-k-NN coverage maximization, submodularity proof, and greedy $(1-1/e)$ approximation ratio.
- **Effective use of LLMs**: Rather than asking the LLM to directly explain the GNN, the framework has the LLM iteratively generate Python logic rules from data features — decoupling reasoning capability from language expression.
- **Scalability**: Sampling-based approximation of Rev-k-NN linearizes computational complexity, enabling successful processing of ogbn-arxiv with 170k nodes.
- **Diagnostic value**: The framework not only explains correct predictions but also uncovers systematic model failures.

## Limitations & Future Work

- The method only accesses the GNN embedding space and cannot probe the feature–topology interaction mechanisms within internal model layers.
- The quality of LLM-generated rules depends on the design of the node summary information provided as input.
- Boolean rules may oversimplify extremely complex decision boundaries.
- The framework has not been extended to node classification explanation on dynamic or temporal graphs.
- The choice of exemplar budget $b$ requires manual tuning, with no automated selection criterion.

## Related Work & Insights

- **GNNExplainer / PGExplainer**: Classic local explainers that explain individual predictions by optimizing subgraph masks.
- **GLGExplainer**: The only prior global logical explainer, which distills local explanation subgraphs via clustering combined with Boolean formula learning, but scales poorly.
- **GraphTrail**: Translates GNN predictions into human-interpretable logical rules, but supports only discrete labels.
- **Self-Refine (Madaan et al.)**: The LLM self-refinement paradigm, which this work applies to GNN rule discovery.
- **Insights**: The combination of Exemplar Theory and LLM self-refinement is generalizable to global explanation of other black-box models (e.g., recommender systems, text classifiers).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The combination of Exemplar Theory, Rev-k-NN, and LLM self-refinement is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Eight datasets (homophilic/heterophilic), scalability validation, 60-participant user study, and diagnostic case analysis.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear, methodology is rigorously presented, and theoretical proofs are complete.
- Value: ⭐⭐⭐⭐⭐ First work to achieve high-fidelity global GNN node classification explanation at large scale.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] GraphFaaS: Serverless GNN Inference for Burst-Resilient, Real-Time Intrusion Detection](graphfaas_serverless_gnn_inference_for_burst-resilient_real-time_intrusion_detec.md)
- [\[NeurIPS 2025\] DuetGraph: Coarse-to-Fine Knowledge Graph Reasoning with Dual-Pathway Global-Local Fusion](duetgraph_coarse-to-fine_knowledge_graph_reasoning_with_dual-pathway_global-loca.md)
- [\[NeurIPS 2025\] Deliberation on Priors: Trustworthy Reasoning of Large Language Models on Knowledge Graphs](deliberation_on_priors_trustworthy_reasoning_of_large_language_models_on_knowled.md)
- [\[ICLR 2026\] Learning Concept Bottleneck Models from Mechanistic Explanations](../../ICLR2026/graph_learning/learning_concept_bottleneck_models_from_mechanistic_explanations.md)
- [\[NeurIPS 2025\] Dynamic Bundling with Large Language Models for Zero-Shot Inference on Text-Attributed Graphs](dynamic_bundling_with_large_language_models_for_zero-shot_inference_on_text-attr.md)

<!-- RELATED:END -->
