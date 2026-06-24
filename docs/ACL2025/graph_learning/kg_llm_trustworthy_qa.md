---
title: >-
  [Paper Note] Can Knowledge Graphs Make Large Language Models More Trustworthy? An Empirical Study Over Open-ended Question Answering
description: >-
  [ACL 2025][Graph Learning][Knowledge Graph] This work proposes OKGQA, an open-ended knowledge graph question answering benchmark, and its perturbed variant OKGQA-P. Through a unified graph-guided retrieval-generation framework, it systematically demonstrates that KG augmentation effectively reduces LLM hallucination rates (boosting FActScore by ~20 percentage points), with subgraph retrieval achieving optimal performance across all query types and exhibiting robustness to KG…
tags:
  - "ACL 2025"
  - "Graph Learning"
  - "Knowledge Graph"
  - "Large Language Model Hallucination"
  - "Open-ended Question Answering"
  - "Retrieval-Augmented Generation"
  - "Subgraph Retrieval"
  - "Benchmark"
date: 2026-05-08
content_hash: a651db78fba77b94
---

# Can Knowledge Graphs Make Large Language Models More Trustworthy? An Empirical Study Over Open-ended Question Answering

**Conference**: ACL 2025  
**arXiv**: [2410.08085](https://arxiv.org/abs/2410.08085)  
**Code**: [https://github.com/Y-Sui/OKGQA](https://github.com/Y-Sui/OKGQA)  
**Area**: Graph Learning  
**Keywords**: Knowledge Graph, Large Language Model Hallucination, Open-ended Question Answering, Retrieval-Augmented Generation, Subgraph Retrieval, Benchmark  

## TL;DR

This work proposes OKGQA, an open-ended knowledge graph question answering benchmark, and its perturbed variant OKGQA-P. Through a unified graph-guided retrieval-generation framework, it systematically demonstrates that KG augmentation effectively reduces LLM hallucination rates (boosting FActScore by ~20 percentage points), with subgraph retrieval achieving optimal performance across all query types and exhibiting robustness to KG noise.

---

## Background & Motivation

**Severe LLM hallucination issues**: Contemporary LLMs are prone to generating plausible-sounding but factually incorrect outputs (hallucinations), which is particularly detrimental in high-risk areas like healthcare and scientific research.

**Potential of KG augmentation**: Knowledge graphs provide structured, traceable factual information, which theoretically can enhance the reliability and trustworthiness of LLM outputs via external knowledge injection.

**Limitations of existing benchmarks**: Existing KGQA benchmarks (e.g., WebQSP, CWQ) primarily focus on **closed-ended** tasks, where model outputs are restricted to fixed entity/relation sets, failing to effectively detect hallucinations—conventional metrics like accuracy cannot distinguish retrieval errors from fabricated answers.

**Necessity of open-ended evaluation**: In open-ended settings, LLMs are required to generate paragraph-length answers containing reasoning paths and supporting facts. The expanded output space exposes hallucinations more easily, and also allows direct quantification of hallucination rates using metrics like FActScore/SAFE.

**Practical challenges in KG quality**: Real-world KGs frequently contain noise, such as labeling errors and missing relations. It is essential to evaluate model robustness when KGs are noisy.

**Lack of systematic comparison in method designs**: There is a lack of systematic comparative analysis on the differences in hallucination mitigation effects under various KG retrieval granularities (triplets vs. paths vs. subgraphs).

---

## Method

### Overall Architecture

This work proposes a unified KG augmentation framework based on the RAG paradigm, consisting of two core components: **Graph-guided Retrieval (G-Retrieval)**, which extracts query-related knowledge subsets $\mathcal{Z}^*$ from the KG, and **Graph-guided Generation (G-Generator)**, which leverages the retrieved knowledge to generate paragraph-length answers. Formally:

$$p(a|q) = \sum_{\mathcal{Z} \subseteq \mathcal{G}} p_\phi(a|q, \mathcal{Z}) \cdot p_\theta(\mathcal{Z}|q, \mathcal{G}) \approx p_\phi(a|q, \mathcal{Z}^*) \cdot p_\theta(\mathcal{Z}^*|q, \mathcal{G})$$

where $\mathcal{Z}^* = \arg\max_{\mathcal{Z} \in \mathcal{G}} p_\theta(\mathcal{Z}|q, \mathcal{G})$, approximating the summation by selecting the optimal knowledge subset.

### Key Designs

#### Module 1: OKGQA Benchmark Construction

- **Function**: Build a benchmark dataset for open-ended KGQA containing 850→2050 multi-type queries.
- **Mechanism**: Utilize a template-driven LLM generation method to produce five categories of queries (descriptive, explanatory, predictive, comparative, and critical), and optimize query quality through iterative alignment between automatic scores $s_{\text{auto}}$ and human scores $s_{\text{human}}$. KG subgraphs are extracted from the 2-hop neighborhood of DBpedia, compressed from an average of 348,715 tokens to 2,452 using Personalized PageRank (PPR) pruning.
- **Design Motivation**: Closed-ended benchmarks cannot detect hallucinations, requiring open-ended settings where models generate long text to expose factual errors; PPR pruning controls the subgraph scale while preserving relevant information.

#### Module 2: OKGQA-P Perturbation Benchmark

- **Function**: Simulate real-world scenarios of unreliable KG quality via four edge perturbation methods.
- **Mechanism**: Four perturbation strategies are designed: Relation Swapping (RS) randomly swaps the relations of individual edge pairs; Relation Replacement (RR) replaces a relation with its semantically least similar counterpart ("harder negatives"); Edge Reconnection (ER) replaces the target entity with an entity outside the 1-hop neighborhood; Edge Deletion (ED) directly deletes edges. The noise level is controlled dynamically by adjusting the perturbation ratio (0%–100%), and deviation degrees are quantified using ATS (semantic similarity), SC2D, and SD2 (structural similarity).
- **Design Motivation**: Although platforms like Wikidata have community quality control, labeling errors still persist. OKGQA-P evaluates method robustness through systematic perturbations to guide practical application.

#### Module 3: Graph-guided Retrieval (G-Retrieval)

- **Function**: Extract the most relevant knowledge subset from the KG for a given query, supporting three retrieval granularities.
- **Mechanism**: Encode the query and KG elements into a unified embedding space (using text-embedding-3-small), rank them using cosine similarity, and employ a **prize-cost trade-off strategy**—assigning decreasing rewards $p_v = \max(0, k - \text{rank}(v) + 1)$ to top-$k$ nodes/edges while imposing a cost $C_e$ during expansion:
    - **Triplet Retrieval**: Selects a fixed number of triplets with the highest total reward.
    - **Path Retrieval**: Starts from high-reward nodes and greedily expands paths to maximize $S(\mathcal{P}) = \sum p_{v_i} + \sum p_{e_i} - \sum c_e$.
    - **Subgraph Retrieval**: Discovers a connected subgraph that maximizes the total score based on the Prize-Collecting Steiner Tree (PCST) algorithm.
- **Design Motivation**: Different retrieval granularities yield varying levels of structural information. Triplets are the simplest but lack context, paths preserve reasoning chains, and subgraphs offer the most complete relational structure.

#### Module 4: Evaluation System

- **Function**: Establish a multi-dimensional evaluation system covering both hallucination rates and answer quality.
- **Mechanism**: Hallucination evaluation employs FActScore (breaking answers into atomic facts and verifying them against Wikipedia) and SAFE (utilizing an LLM agent to iteratively search and verify); quality evaluation uses the G-Eval framework across four dimensions—context relevance, comprehensiveness, correctness, and empowerment.
- **Design Motivation**: A single metric cannot comprehensively characterize the effectiveness of KG augmentation. It is necessary to evaluate both accuracy ("saying the right things" / hallucination) and quality ("saying things well").

### Loss & Training

This is an empirical benchmark study and **does not involve model training**. All experiments utilize the inference capabilities of pre-trained LLMs with $\text{temperature} = 0.7$ and $\text{top\_p} = 1.0$. The evaluation backbone is gpt-4o-mini, which has been verified to be highly consistent with human judgment through manual evaluation.

---

## Key Experimental Results

### Main Results: Influence of Different KG Retrieval Strategies on Hallucinations (Taking GPT-4o as an Example)

| Method | Context Rel. | Comprehensive. | Correctness | Empowerment | SAFE | FActScore |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Zero-shot (No KG) | 68.12% | 65.41% | 60.41% | 62.41% | 82.47% | 55.34% |
| 4-shot (No KG) | 70.61% | 67.43% | 62.33% | 64.51% | 83.39% | 57.45% |
| IRCoT (Wikipedia) | 73.12% | 69.23% | 66.33% | 65.51% | 87.39% | 69.45% |
| CoT+SC (No KG) | 75.81% | 71.62% | 66.55% | 68.74% | **79.03%** ↓ | **53.23%** ↓ |
| KG-Triplet | 74.62% | 70.44% | 65.37% | 67.12% | 89.20% | 72.53% |
| KG-Path | 78.71% | 74.53% | 69.42% | 71.63% | 90.20% | **75.61%** |
| KG-Subgraph | 80.81% | 76.63% | 71.57% | 73.70% | **90.83%** | 75.33% |
| KG-Subgraph+CoT+SC | **82.90%** | **78.72%** | **73.64%** | **75.80%** | 89.12% | 75.42% |

### Ablation Study: Impact of Perturbation Level on FActScore (GPT-4o, Subgraph Retrieval vs. Baseline)

| Perturbation Method | 0% | 10% | 30% | 50% | 70% | 100% |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|
| Edge Deletion-Subgraph | 75.33% | ~73% | ~68% | ~60% | ~53% | ~40% |
| Relation Replacement-Subgraph | 75.33% | ~71% | ~65% | ~56% | ~48% | ~38% |
| Edge Deletion-Triplet | 72.53% | ~69% | ~62% | ~52% | <CoT | <CoT |
| CoT+SC Baseline | 53.23% | — | — | — | — | — |

> Note: Data for the perturbation experiments are approximate values estimated from Figure 6. Key threshold: At **50% perturbation**, triplet/path retrieval degrades to the baseline level, whereas subgraph retrieval still outperforms the baseline.

### Key Findings

- **KG information significantly reduces hallucinations**: KG-triplets improve GPT-4o's FActScore from 55.34% to 72.53% (+17.19pp), and subgraph retrieval further boosts it to 75.33% (+19.99pp).
- **Internal reasoning step can actually exacerbate hallucinations**: CoT+SC decreases GPT-4o's SAFE from 82.47% to 79.03%, and FActScore from 55.34% to 53.23%, indicating that relying solely on internal reasoning fails to mitigate hallucinations and may even introduce bias.
- **Subgraph retrieval is globally optimal**: Subgraph retrieval HTML outperforms triplet/path retrieval across all five query classes on both G-Eval and FActScore, showing particularly pronounced advantages in simple queries (descriptive and event-descriptive).
- **KG augmentation outperforms conventional RAG**: The FActScore of KG-Triplet (72.53%) already exceeds that of IRCoT (69.45%), and the SAFE score of subgraph retrieval (90.83%) also surpasses IRCoT (87.39%).
- **Subgraph retrieval is most robust to noise**: Under 50% perturbation, subgraph retrieval still outperforms the CoT baseline, while triplet/path retrieval degrades to the baseline level at 50% perturbation.
- **Open-source models also benefit**: FActScores of Llama-3.1-8B and Mistral-7B increase by ~20pp and ~19pp, respectively, when utilizing KG-Subgraph.

---

## Highlights & Insights

1. **Unique value of the open-ended perspective**: Shifting KGQA from closed-ended to open-ended settings enables hallucination detection—a simple yet highly impactful experimental transition.
2. **"CoT intensifies hallucinations" counter-intuitive finding**: CoT+SC improves answer quality (G-Eval↑) but simultaneously increases the hallucination rate (FActScore↓), revealing that LLMs may write coherent but fabricated facts during the reasoning process to "self-justify."
3. **Retrieval design based on Prize-Cost trade-off**: Formulating graph retrieval as a prize-cost optimization problem provides an elegant, scalable, and unified framework to compare retrieval strategies of different granularities.
4. **Practical value of OKGQA-P**: Systematically quantifying the impact of KG noise on downstream performance yields a practical reference value—a 50% perturbation threshold—whereas real-world platform error rates inside Wikidata are far below this threshold.
5. **Structural advantages of subgraphs**: Subgraphs preserve the connectivity structures among entities via the PCST algorithm, providing LLMs with richer reasoning contexts, which explains their superior performance on complex queries.

---

## Limitations & Future Work

1. **Single knowledge source**: Using only DBpedia as the knowledge source leaves generalization to domain-specific KGs (e.g., biomedical KGs) unverified.
2. **Static KG assumption**: The application of this framework is constrained in scenarios requiring real-time knowledge, as it does not account for dynamically updated knowledge graphs.
3. **Lack of training-level integration**: All methods focus on in-context inference-time augmentation (prompting KG info) without exploring knowledge fusion during model training or fine-tuning.
4. **Unanalyzed retrieval overhead**: The study lacks comparisons regarding the computational costs and latencies of triplets, paths, and subgraph retrieval methods, leaving efficiency-effectiveness tradeoffs in practical deployment unclear.
5. **LLM-dependent evaluation**: Both G-Eval and SAFE rely on LLMs as evaluators, introducing potential circular dependency risks despite verification through human alignment.
6. **Extensibility to multimodal KGs**: Future work can extend the framework to knowledge graphs containing multimodal information such as images and tables.

---

## Related Work & Insights

- **G-Retriever** (He et al., 2024): The basis of the subgraph retrieval method used in this paper, which applies the PCST algorithm to graph-text QA, although the original work did not focus on evaluating hallucinations.
- **FActScore** (Min et al., 2023) and **SAFE** (Wei et al., 2024): Two complementary hallucination metrics, with the former based on knowledge-base verification and the latter on search-engine verification.
- **IRCoT** (Trivedi et al., 2022): An interleaved retrieval-reasoning method based on Wikipedia paragraphs. This paper proves that structured KG retrieval outperforms such unstructured RAG systems.
- **GraphRAG** (Edge et al., 2024): A local-to-global Graph RAG method, similar in concept to this framework but focusing on text-centric graphs rather than KGs.
- **Insight**: The fundamental difference between KG augmentation and traditional RAG lies in the utilization of **structured relations**. Future work could explore encoding KG structures into reasoning chains comprehensible to LLMs, rather than merely injecting them as textual prompts.

---

## Rating

- **Novelty**: ⭐⭐⭐ — The algorithmic innovation is relatively limited (mostly benchmark construction and systematic comparison of existing methods), but the open-ended perspective and perturbation evaluations are valuable contributions.
- **Technical Depth**: ⭐⭐⭐ — The framework is clearly formalized and the retrieval methods are comprehensively covered, but it lacks theoretical analysis and novel algorithm design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Highly comprehensive experiments with 5 LLMs × multiple retrieval strategies × various query types × 4 perturbation methods × multi-level perturbation granularities.
- **Value**: ⭐⭐⭐⭐ — The OKGQA benchmark and key insights, such as the 50% perturbation threshold, provide direct guidance for designing practical KG-LLM integrated systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Ontology-Guided Reverse Thinking Makes Large Language Models Stronger on Knowledge Graph Question Answering](ontology-guided_reverse_thinking_makes_large_language_models_stronger_on_knowled.md)
- [\[ACL 2025\] FiDeLiS: Faithful Reasoning in Large Language Model for Knowledge Graph Question Answering](fidelis_faithful_reasoning_in_large_language_model_for_knowledge_graph_question_.md)
- [\[ACL 2025\] The Role of Exploration Modules in Small Language Models for Knowledge Graph Question Answering](the_role_of_exploration_modules_in_small_language_models_for_knowledge_graph_que.md)
- [\[NeurIPS 2025\] Deliberation on Priors: Trustworthy Reasoning of Large Language Models on Knowledge Graphs](../../NeurIPS2025/graph_learning/deliberation_on_priors_trustworthy_reasoning_of_large_language_models_on_knowled.md)
- [\[ACL 2025\] Agent Steerable Search for Knowledge Graph Question Answering](agent_steerable_search_for_knowledge_graph_question_answering.md)

</div>

<!-- RELATED:END -->
