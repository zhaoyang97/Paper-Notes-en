---
title: >-
  [Paper Note] Assessing LLMs for Serendipity Discovery in Knowledge Graphs: A Case for Drug Repurposing
description: >-
  [AAAI 2026][Graph Learning][Serendipity] This paper proposes SerenQA, the first framework to formally define the serendipity discovery task in knowledge graph question answering. It introduces an information-theoretic RNS metric, an expert-annotated drug repurposing benchmark dataset, and a three-stage LLM evaluation pipeline. The work reveals that current LLMs perform reasonably on retrieval tasks but have substantial room for improvement in serendipitous exploration.
tags:
  - "AAAI 2026"
  - "Graph Learning"
  - "Serendipity"
  - "Knowledge Graph Question Answering"
  - "LLM Evaluation"
  - "Drug Repurposing"
  - "information theory"
date: 2026-05-08
content_hash: d3a28ae3d0ea7716
---

# Assessing LLMs for Serendipity Discovery in Knowledge Graphs: A Case for Drug Repurposing

**Conference**: AAAI 2026
**arXiv**: [2511.12472](https://arxiv.org/abs/2511.12472)  
**Code**: [cwru-db-group/serenQA](https://cwru-db-group.github.io/serenQA)  
**Area**: Graph Learning
**Keywords**: Serendipity, Knowledge Graph Question Answering, LLM Evaluation, Drug Repurposing, information theory

## TL;DR

This paper proposes SerenQA, the first framework to formally define the serendipity discovery task in knowledge graph question answering. It introduces an information-theoretic RNS metric, an expert-annotated drug repurposing benchmark dataset, and a three-stage LLM evaluation pipeline. The work reveals that current LLMs perform reasonably on retrieval tasks but have substantial room for improvement in serendipitous exploration.

## Background & Motivation

- **Background**: Existing LLM-augmented KGQA systems focus on returning highly relevant yet predictable answers, lacking the ability to discover unexpected but valuable connections.
- **Scientific significance of serendipity**: Many major breakthroughs in the history of science originated from serendipitous discoveries (e.g., penicillin). Enabling LLMs to mine surprising findings from existing knowledge bases is a key step toward truly AI-driven scientific discovery.
- **Demand for drug repurposing**: Drug repurposing—identifying new indications for existing drugs—is a core task in medical research and naturally suits serendipity as an application scenario. For example, Journavx, the first non-opioid drug for severe acute pain, achieves analgesia via a novel mechanism (NaV1.8 sodium channel), representing a canonical case of serendipitous discovery.
- **Limitations of Prior Work**: Existing serendipity research (in recommender systems and web search) relies primarily on subjective human annotation or LLM self-evaluation, lacking interpretable, scalable, and reproducible quantification methods.
- **Evaluation gap**: The community lacks a dedicated benchmark dataset and systematic evaluation scheme for serendipitous discovery capabilities in scientific KGQA.
- **Key Challenge**: Serendipity itself is a composite experience of relevance, novelty, and unexpectedness. Discovering genuinely novel and surprising answers while maintaining query relevance poses both theoretical and practical challenges.

## Method

### Overall Architecture

SerenQA consists of three core components: (1) the RNS metric—a graph-based, information-theoretic serendipity quantification measure; (2) a Serendipity-aware Benchmark—an expert-annotated drug repurposing dataset based on the Clinical Knowledge Graph (1,529 queries, 15M+ entities, 201M+ relations); and (3) an Assessment Pipeline—a three-stage LLM evaluation workflow covering knowledge retrieval, subgraph reasoning, and serendipity exploration. For a given query $Q$, the system returns an ordered partition $\mathcal{A} = (\mathcal{A}_e, \mathcal{A}_s)$, where $\mathcal{A}_e$ is the set of known answers directly derivable from the graph, and $\mathcal{A}_s$ is the set of serendipitous discoveries that go beyond direct knowledge.

### Key Design 1: RNS Serendipity Metric

- **Function**: Quantifies the degree of serendipity of answer set $\mathcal{A}_s$ relative to $\mathcal{A}_e$.
- **Mechanism**: Serendipity is decomposed into a weighted combination of three information-theoretic dimensions: $\text{RNS}(\mathcal{A}_e, \mathcal{A}_s) = \alpha R + \beta N + \gamma S$. Relevance $R$ measures contextual similarity via normalized Euclidean distance over GCN embeddings; Novelty $N = 1 - MI(\mathcal{A}_e, \mathcal{A}_s)$ quantifies the informational increment of $\mathcal{A}_s$ relative to $\mathcal{A}_e$ via mutual information; Surprise $S$ measures unpredictability of the entity distribution via Jensen–Shannon divergence.
- **Design Motivation**: Compared to subjective methods relying on human annotation or LLM self-evaluation, the information-theoretic approach grounded in a graph probabilistic model offers interpretability, scalability, and reproducibility. The authors verify through axiomatic analysis that RNS satisfies four properties: scale invariance, consistency, non-monotonicity, and independence.

### Key Design 2: 3-Hop Graph Probabilistic Modeling

- **Function**: Establishes an efficient probabilistic model for RNS computation.
- **Mechanism**: A 3-hop conditional probability matrix is constructed as $P_k = \sum_{h=1}^k \alpha_h P_1^h$, where $P_1$ is the normalized single-hop transition probability matrix and weights $\alpha_h$ increase with hop distance (prioritizing more distant connections). Marginal probabilities are approximated via PageRank-style damped iteration, reducing complexity from $O(V^3)$ to $O(V^2 \log V)$.
- **Design Motivation**: Empirical analysis shows that 99% of serendipitous answers are reachable within 3 hops from known answers, making 3-hop sufficient. The probability matrix is computed once and reused across queries, enabling adaptation to graphs from different domains.

### Key Design 3: Benchmark Construction and Three Partitioning Strategies

- **Function**: Constructs a ground-truth dataset with serendipity annotations based on the Clinical Knowledge Graph.
- **Mechanism**: For the complete candidate answer set $\mathcal{A}_c$ of each query, three complementary strategies are employed to partition $(\mathcal{A}_e, \mathcal{A}_s)$: (1) LLM Ensemble—four SOTA LLMs score candidates and the top 20% are assigned to $\mathcal{A}_s$; (2) Expert Crowdsourced—six domain experts (three physicians, one pharmacist, and two annotators) refine and rank the partitions; (3) RNS Guided—a greedy swap algorithm (Algorithm 1) optimizes the RNS score. Pearson correlation among the three strategies exceeds 85%, with the expert and RNS-guided partitions reaching ~99% correlation.
- **Design Motivation**: The three strategies cross-validate one another, ensuring evaluation robustness. An evaluation graph $\mathcal{G}$ is constructed by removing selected edges from $\mathcal{G}_c$, making $\mathcal{A}_e$ derivable while rendering $\mathcal{A}_s$ unreachable, thereby simulating a realistic discovery scenario.

### Key Design 4: Three-Stage LLM Evaluation Pipeline

- **Function**: Systematically evaluates LLM capabilities at each stage of serendipity discovery.
- **Mechanism**: (T1) Knowledge Retrieval—LLMs translate natural language queries into Cypher queries and retrieve $\mathcal{A}_e$ from the KG; (T2) Subgraph Reasoning—LLMs summarize the retrieved subgraph's structured information into domain-aware natural language; (T3) Serendipity Exploration—LLMs explore $\mathcal{A}_s$ from $\mathcal{A}_e$ via beam search (width 30, depth 3), selecting top-$w$ nodes at each step based on evidence strength, interaction force, biological effect direction, and expression level.
- **Design Motivation**: The three tasks respectively evaluate the foundational LLM capabilities of precise knowledge retrieval, structured reasoning, and creative exploration, forming a comprehensive capability profile.

## Loss & Training

This paper presents an evaluation framework rather than a training methodology; no new model training is involved. The weights $\alpha, \beta, \gamma$ of the RNS metric are calibrated by alignment with expert-annotated partitions. The system is deployed on five AWS c6a.24xlarge instances for distributed computation, supporting 500 concurrent LLM inference tasks.

## Key Experimental Results

### Table 2: Knowledge Retrieval (T1) — Model Performance Across Query Patterns

| Model | One-Hop F1(%) | Two-Hop F1(%) | 3+-Hop F1(%) | Intersection F1(%) |
|------|:---:|:---:|:---:|:---:|
| DeepSeek-V3 | **78.71** | 10.71 | 6.22 | 7.15 |
| GPT-4o | 77.16 | 6.36 | 4.20 | 4.65 |
| Llama-3.3-70B | 70.67 | **44.34** | **10.16** | **9.60** |
| DeepSeek-R1-70B | 69.07 | 37.00 | 8.06 | 6.16 |
| Med42-V2-70B | 69.43 | 19.12 | 0.51 | 0.13 |
| Qwen3-8B | 37.24 | 2.87 | 2.01 | 1.91 |

**Key Findings**: Frontier large models achieve F1 ~78% on simple single-hop queries, but performance drops sharply to <10% on multi-hop queries (3+ hops), exposing insufficient reasoning depth. The 70B models (Llama, DeepSeek-R1) substantially outperform closed-source frontier models on multi-hop queries.

### Table 3: Serendipity Exploration (T3) — Expert Crowdsourced Partition Results

| Model | Relevance | TypeMatch | SerenHit |
|------|:---:|:---:|:---:|
| Llama-3.3-70B | **2.559** | **0.483** | 0.067 |
| DeepSeek-V3 | 2.494 | 0.462 | 0.061 |
| Gamma-2-27B | 2.379 | 0.414 | 0.057 |
| Qwen-2.5-72B | 2.345 | 0.406 | 0.041 |
| Qwen-2.5-32B | 2.331 | 0.426 | 0.045 |
| DeepSeek-R1-70B | 2.000 | 0.409 | 0.034 |
| Mixtral-8x7B | 2.033 | 0.254 | 0.015 |

**Key Findings**: SerenHit (match rate against the ground-truth serendipity set) is extremely low (<10%) across all models, indicating a substantial gap in genuine serendipitous exploration. Removing subgraph summarization (w.o. summary) actually improves performance for most models, suggesting that hallucinations introduced during summarization may mislead the exploration trajectory.

## Highlights & Insights

- **Novelty of problem formulation**: This work is the first to formally define the serendipity-aware task in scientific KGQA, filling a critical evaluation gap in the field.
- **Theoretically grounded metric**: The RNS metric is founded on information theory and validated through axiomatic analysis, making it more rigorous and reproducible than subjective evaluation methods.
- **Rigorous dataset construction**: Three complementary partitioning strategies (LLM ensemble, expert annotation, and RNS-guided) cross-validate one another with correlations exceeding 85%.
- **Insightful findings**: The work reveals a trade-off between faithfulness and serendipity coverage in subgraph reasoning, as well as the negative impact of summarization hallucinations on exploration.

## Limitations & Future Work

- **Domain limitation**: Validation is conducted solely on drug repurposing (Clinical Knowledge Graph); generalizability to other scientific domains (e.g., materials science, physics) has not been demonstrated.
- **Dependence on embedding quality**: The Relevance dimension of RNS relies on GCN embedding quality, and the choice of embedding method may influence the metric outcomes.
- **Limitations of the 3-hop assumption**: Although 99% of serendipitous answers are reachable within 3 hops, truly breakthrough discoveries may require longer paths.
- **Pharmacological factors not considered**: As the authors acknowledge, the framework does not account for key drug feasibility factors (e.g., physicochemical properties), and discovered results require clinical validation.
- **High evaluation cost**: The framework requires five large AWS instances and 500 concurrent inference tasks, limiting reproducibility due to computational resource constraints.

## Related Work & Insights

- **Serendipity in recommender systems**: Prior work (Bordino et al., Fu et al.) studied serendipity primarily in recommender systems and web search, relying on subjective annotation. This paper shifts the focus to scientific KGQA and proposes an objective metric.
- **LLM + KGQA**: Methods such as GraphLingo improve KGQA accuracy via RAG and prompt engineering but focus solely on "correct" answers. This paper targets answers that are "unexpected yet valuable."
- **LLM for scientific discovery**: Works such as AI4Science and Si et al. explore LLMs' ability to generate scientific hypotheses. This paper provides the first systematic evaluation framework for such capabilities.
- **Drug repurposing**: Traditional approaches are based on similarity networks or knowledge graph reasoning. This paper reframes drug repurposing as a serendipity discovery problem.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First to formally define the serendipity problem in scientific KGQA, with original contributions in both problem formulation and metric design.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers 12+ models, 3 partitioning strategies, and 3 evaluation tasks with detailed analysis; however, validation is limited to a single domain.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, rigorous axiomatic analysis, and intuitive examples; some notation is relatively heavy.
- Value: ⭐⭐⭐⭐ — Opens a new research direction in serendipity-aware KGQA with important implications for the AI4Science community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] AtlasKV: Augmenting LLMs with Billion-Scale Knowledge Graphs in 20GB VRAM](../../ICLR2026/graph_learning/atlaskv_augmenting_llms_with_billion-scale_knowledge_graphs_in_20gb_vram.md)
- [\[AAAI 2026\] NOTAM-Evolve: A Knowledge-Guided Self-Evolving Optimization Framework with LLMs for NOTAM Interpretation](notam-evolve_a_knowledge-guided_self-evolving_optimization_framework_with_llms_f.md)
- [\[ACL 2025\] Can LLMs Evaluate Complex Attribution in QA? Automatic Benchmarking using Knowledge Graphs](../../ACL2025/graph_learning/paper_2401_14640.md)
- [\[ICLR 2026\] Graphon Cross-Validation: Assessing Models on Network Data](../../ICLR2026/graph_learning/graphon_cross-validation_assessing_models_on_network_data.md)
- [\[ACL 2026\] LLMs Underperform Graph-Based Parsers on Supervised Relation Extraction for Complex Graphs](../../ACL2026/graph_learning/llms_underperform_graph-based_parsers_on_supervised_relation_extraction_for_comp.md)

</div>

<!-- RELATED:END -->
