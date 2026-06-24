---
title: >-
  [Paper Note] GraphCheck: Breaking Long-Term Text Barriers with Extracted Knowledge Graph-Powered Fact-Checking
description: >-
  [Graph Learning][Fact-Checking] GraphCheck proposes a graph-enhanced fact-checking framework that leverages LLMs to extract knowledge graph triples from documents and claims. These triples are encoded by GNNs as graph structures and injected as soft prompts into a frozen LLM validator. This achieves fine-grained fact-checking in a single inference call, yielding an average improvement of $7.1\%$ across 7 benchmarks and showing strong generalization capability in the medical d…
tags:
  - "Graph Learning"
  - "Fact-Checking"
  - "Knowledge Graphs"
  - "Graph Neural Networks"
  - "Long-Text Understanding"
  - "Hallucination Detection"
date: 2026-05-08
content_hash: dff6c6e9ad7fbf5a
---

# GraphCheck: Breaking Long-Term Text Barriers with Extracted Knowledge Graph-Powered Fact-Checking

| Conference | Area | arXiv | Code |
|------|------|-------|------|
| ACL2025 | Graph Learning / Fact-Checking | [2502.16514](https://arxiv.org/abs/2502.16514) | [GitHub](https://github.com/Yingjian-Chen/GraphCheck) |

**Keywords**: Fact-Checking, Knowledge Graphs, Graph Neural Networks, Long-Text Understanding, Hallucination Detection

## TL;DR

GraphCheck proposes a graph-enhanced fact-checking framework that leverages LLMs to extract knowledge graph triples from documents and claims. These triples are encoded by GNNs as graph structures and injected as soft prompts into a frozen LLM validator. This achieves fine-grained fact-checking in a single inference call, yielding an average improvement of $7.1\%$ across 7 benchmarks and showing strong generalization capability in the medical domain.

## Background & Motivation

LLMs are widely used but frequently generate subtle factual errors (hallucinations), especially in long texts. In professional fields such as medicine, factual errors can lead to misdiagnosis and inappropriate treatments.

Existing grounding document-based fact-checking methods face two major challenges:

**Difficulty in Multi-Hop Relational Understanding**: Directly feeding long documents and claims into an LLM for checking (Naive Check) easily overlooks complex entity relationships and subtle factual inconsistencies.

**Low Efficiency**: Specialized methods (e.g., FactScore, MiniCheck) decompose long documents into atomic facts for one-by-one verification (Atomic Check), requiring multiple model calls and incurring high computational costs.

The goal of GraphCheck is to achieve fine-grained fact-checking in a single inference run while capturing complex multi-hop logical relationships in long texts.

## Method

### Overall Architecture

GraphCheck consists of three main steps:

1. **Graph Construction**: Extract knowledge triples from the document $D$ and claim $C$ to construct the corresponding knowledge graphs $G_D$ and $G_C$.
2. **Graph Encoding**: Encode graph structures using a trainable GNN to obtain graph embeddings.
3. **Fact-Checking**: Concatenate the graph embeddings with text embeddings and feed them into a frozen LLM to make the judgment.

### Graph Construction

An LLM is used to automatically extract $\{source, relation, target\}$ triples from the text to construct a directed graph $G = (\mathcal{V}, \mathcal{E})$:
- $\mathcal{V} = \{\mathbf{v}_i\}_{i=1,\dots,n}$: Set of node (entity) features
- $\mathcal{E} = \{\mathbf{e}_{ij}\}$: Set of edge (relation) features

The textual attributes of nodes and edges are encoded into feature vectors using Sentence-Transformers (all-roberta-large-v1).

### Graph Encoding (GNN)

A message-passing mechanism is adopted to update node features:

$$\mathbf{v}_i^{l+1} = \text{UPDATE}\left(\mathbf{v}_i^l, \sum_{j \in \mathcal{N}_i} \text{MESSAGE}(\mathbf{v}_j^l, \mathbf{e}_{ji})\right)$$

The final graph embedding is obtained via a READOUT function (which includes sum aggregation):

$$\mathbf{h}_g = \text{READOUT}(\{\mathbf{v}_i^L\}_{i=1,\dots,n})$$

### Graph Projection

A projection module $P$ is used to map the graph features $\mathbf{h}_g^C$ and $\mathbf{h}_g^D$ into the text embedding space of the LLMs, yielding $\tilde{\mathbf{h}}_g^C$ and $\tilde{\mathbf{h}}_g^D$.

### Fact-Checking

The projected graph embeddings are concatenated with the text embeddings and input to the self-attention layers of the LLM:

$$y = \text{LLM}(\tilde{\mathbf{h}}_g^C, \tilde{\mathbf{h}}_g^D, \mathbf{h}_t)$$

Output $y \in \{\text{"support"}, \text{"unsupport"}\}$.

### Loss & Training

- **Training Data**: 14K synthetic samples based on the MiniCheck dataset.
- **Graph Extraction**: Using Claude-3.5-Sonnet to extract triples.
- **Training Approach**: Only the GNN and projection layers are trained; the LLM parameters are frozen.
- **Key Findings**: Although trained on general-domain data, the graph-enhanced reasoning capability generalizes well to the medical domain.

## Key Experimental Results

### Evaluation Benchmarks

The evaluation covers 7 benchmarks across both general and medical domains:

| Dataset | Domain | Size | Avg Document Length | Avg Claim Length |
|--------|------|------|-------------|-------------|
| AggreFact-Xsum | General | 558 | 324 | 23 |
| AggreFact-CNN | General | 558 | 500 | 55 |
| SummEval | General | 1600 | 359 | 63 |
| ExpertQA | General | 3702 | 432 | 26 |
| COVID-Fact | Medical | 4086 | 72 | 12 |
| SCIFact | Medical | 809 | 249 | 12 |
| PubHealth | Medical | 1231 | 77 | 14 |

### Main Results (Balanced Accuracy %)

| Method | Overall Avg |
|------|-------------|
| GPT-4 | 70.8 |
| GPT-4o | 70.1 |
| OpenAI o1 | 72.9 |
| Claude 3.5-Sonnet | 73.6 |
| DeepSeek-V3 671B | 71.7 |
| MiniCheck | 68.1 |
| GraphEval | 65.1 |
| **GraphCheck-Llama3.3 70B** | **71.1** |
| **GraphCheck-Qwen 72B** | **70.7** |

Key Observations:
- GraphCheck outperforms GPT-4 and GPT-4o, and approaches the strongest large models (o1, Claude-3.5).
- It significantly outperforms all specialized fact-checking methods (+3% compared to MiniCheck, +10.5% compared to ACUEval).
- Completed in a single call, it is far more efficient than methods requiring multiple calls.

### Domain Analysis

- **General Domain**: Performs on par with MiniCheck and GraphEval.
- **Medical Domain**: Outperforms MiniCheck by **8.1%**, demonstrating strong cross-domain generalization.

### Ablation Study

1. **Incremental Contribution of Graph Information**: Significant improvements are observed in both lightweight models (Llama3 8B, Qwen2.5 7B) and large models (70B+).
2. **KG as Text vs. Graph Embeddings**: Directly adding the KG text to the prompt yields only a marginal improvement (66.4% vs. 65.3%), whereas encoding it with GNN achieves robust results (71.1%), indicating that LLMs cannot efficiently extract structural information from pure text KGs.
3. **Training Data Size**: Performance scales upward with the volume of training data, e.g., AggreFact-Xsum improves from 60.1% to 72.9%.
4. **Impact of KG Quality**: The differences in KG quality extracted by various models are minimal for short texts; however, for long texts, KG completeness is critical, and low-quality KGs (e.g., extracted by Llama 8B) introduce noise.

### Interpretability Case Study

By visualizing the edge attention weights of the GCN, one can observe the triple relationships that the model prioritizes during fact-checking. For instance, in a medical case study, the model focuses on key triples such as "(Dr. Erica Pan, is, California state epidemiologist)" and "(Dr. Erica Pan, recommended pause of, Moderna COVID-19 vaccine)", aligning with the verification requirements.

## Highlights & Insights

1. **Single-Inference Fine-Grained Checking**: Compared to FactScore (multiple atomic checks) and MiniCheck (pairwise comparisons), GraphCheck completes verification in a single call, dramatically boosting efficiency.
2. **KG-Enhanced Cross-Domain Generalization**: GNNs trained on general-domain data can generalize to the medical domain, suggesting that graph-structured reasoning is largely domain-independent.
3. **GNN as a Soft Prompt**: Graph embeddings are injected through a projection module without modifying the LLM's parameters, offering a flexible and reusable approach.
4. **Strong Interpretability**: Edge attention weights allow tracing the model's reasoning process, which is particularly vital in the medical domain.
5. **KG Quality Experiments**: Systematic comparison of the impact of KG quality extracted by different models provides practical guidance for deployment.

## Limitations & Future Work

1. **Dependence on KG Quality**: The quality of the extracted KG directly affects fact-checking performance; however, standard automatic evaluation methods for KG quality are currently lacking.
2. **Poor Performance on Long Claims**: Performance is weaker on AggreFact-CNN and SummEval (average claim length > 50) because long claims make triple extraction more challenging.
3. **Training Data Bottleneck**: The work relies on a limited size of 14K synthetic samples.
4. **Computational Cost of KG Extraction**: Requiring an extra LLM call to extract triples introduces additional preprocessing overhead.

## Related Work & Insights

- **Hallucination Detection Methods**: RAG methods (external knowledge base verification), LLM + grounding document methods (direct checking).
- **Long-Text Fact-Checking**: FactScore, MiniCheck, and ACUEval decompose text into atomic units for sequential verification.
- **Graph-based Methods**: GraphEval (triple-by-triple NLI evaluation without using global graph structures), FactGraph (a pre-LLM method), and AMRFact (AMR graph-guided summary generation).
- **G-Retriever**: Graph-enhanced retrieval and question answering; this study finds that directly prompting with KG text yields limited gains.

## Rating

⭐⭐⭐⭐ (4/5)

The methodology is clear and elegant—using KGs to capture multi-hop relations in long text and GNNs to encode them before injecting them as soft prompts into the LLM. Single-inference verification is a major practical advantage for real-world deployment. The cross-domain generalization capability (general to medical) is impressive. The primary limitations lie in the heavy reliance on KG extraction quality and the performance bottlenecks with long claim inputs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Fast-and-Frugal Text-Graph Transformers are Effective Link Predictors](fast-and-frugal_text-graph_transformers_are_effective_link_predictors.md)
- [\[ACL 2025\] Can Graph Neural Networks Learn Language with Extremely Weak Text Supervision?](can_graph_neural_networks_learn_language.md)
- [\[NeurIPS 2025\] Unifying Text Semantics and Graph Structures for Temporal Text-attributed Graphs with LLMs](../../NeurIPS2025/graph_learning/unifying_text_semantics_and_graph_structures_for_temporal_text-attributed_graphs.md)
- [\[ICML 2025\] On Measuring Long-Range Interactions in Graph Neural Networks](../../ICML2025/graph_learning/on_measuring_long-range_interactions_in_graph_neural_networks.md)
- [\[NeurIPS 2025\] Sketch-Augmented Features Improve Learning Long-Range Dependencies in Graph Neural Networks](../../NeurIPS2025/graph_learning/sketch-augmented_features_improve_learning_long-range_dependencies_in_graph_neur.md)

</div>

<!-- RELATED:END -->
