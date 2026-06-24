---
title: >-
  [Paper Note] TARGA: Targeted Synthetic Data Generation for Practical Reasoning over Structured Data
description: >-
  [ACL 2025][Semantic parsing] TARGA proposes a targeted synthetic data generation framework that dynamically generates highly relevant synthetic examples for in-context learning in KBQA without requiring any human annotation. Using only a 7B model, it significantly outperforms all non-fine-tuned methods on GrailQA (+7.7 F1) and KBQA-Agent (+12.2 F1).
tags:
  - "ACL 2025"
  - "Semantic parsing"
  - "Knowledge Base Question Answering (KBQA)"
  - "Synthetic Data Generation"
  - "In-Context Learning"
  - "Structured Reasoning"
date: 2026-05-08
content_hash: 77c33ce3bf900e53
---

# TARGA: Targeted Synthetic Data Generation for Practical Reasoning over Structured Data

**Conference**: ACL 2025  
**arXiv**: [2412.19544](https://arxiv.org/abs/2412.19544)  
**Code**: [https://github.com/cdhx/TARGA](https://github.com/cdhx/TARGA)  
**Area**: Other  
**Keywords**: Semantic parsing, Knowledge Base Question Answering (KBQA), Synthetic Data Generation, In-Context Learning, Structured Reasoning

## TL;DR

TARGA proposes a targeted synthetic data generation framework that dynamically generates highly relevant synthetic examples for in-context learning in KBQA without requiring any human annotation. Using only a 7B model, it significantly outperforms all non-fine-tuned methods on GrailQA (+7.7 F1) and KBQA-Agent (+12.2 F1).

## Background & Motivation

1. **Background**: Semantic parsing (translating natural language questions to logical forms) is the dominant approach for reasoning in structured environments. Existing methods either require fine-tuning on large-scale human-annotated datasets (e.g., Pangu) or retrieve similar examples from annotated datasets for in-context learning (e.g., KB-Binder, KB-Coder).

2. **Limitations of Prior Work**: Two core challenges—(1) **Annotation Reliance**: Collecting human-annotated data in specific environments is time-consuming and labor-intensive, and pre-annotated data is typically unavailable in real-world scenarios; (2) **Limited Generalization**: Methods based on static training sets suffer from drastic performance drops (up to 20%+) when encountering unseen KB elements or query structures (non-I.I.D. settings).

3. **Key Challenge**: In complex environments such as Freebase with over 3 billion triples, representing all potential query schemas using pre-collected static datasets is impossible. Expanding annotation coverage leads to exponential growth in training and retrieval costs.

4. **Goal**: How can highly relevant examples be dynamically generated for each test question without any human annotation, while achieving outstanding performance with small models?

5. **Key Insight**: Starting from KB entities and relations related to the test question, **valid query graphs are constructed directly on the knowledge base** through layer-wise expansion and cross-layer combination. These queries are then translated into natural language questions to serve as examples for in-context learning.

6. **Core Idea**: Instead of retrieving examples from static datasets, **targeted query-question pairs are dynamically synthesized** on the KB for each test question to serve as in-context learning demonstrations.

## Method

### Overall Architecture

Given a natural language query $nlq$, TARGA operates in four steps: (1) retrieving candidate KB entities and relations; (2) constructing synthetic query graphs via layer-wise expansion (multi-hop) and cross-layer combination (multi-constraint); (3) ranking the synthetic queries using a bge-reranker to select the most relevant examples; (4) performing in-context learning with the ranked synthetic (NLQ, Query) pairs as demonstrations to yield the target query.

### Key Designs

1. **Layer-wise Expansion**:

    - **Function**: Build multi-hop chained query structures (depth of the query graph).
    - **Mechanism**: Starting from the simplest single-hop structure $\mathcal{L}_1 = \{(s,p,o) | s \in E_{nlq}, p \in R_{nlq}, \text{Exec}((s,p,o), \mathcal{G}) \neq \emptyset\}$, the query graph is expanded outward layer by layer — connecting terminal variable nodes to new variables via new relations to construct $\mathcal{L}_2, \mathcal{L}_3$. The expansion stops when a complexity threshold (3 hops) is reached, as the entity-to-answer distance in reasonable questions typically does not exceed 3 hops. Key constraint: only queries with **non-empty execution results** are expanded to prevent exponential growth of invalid combinations.
    - **Design Motivation**: An exploration strategy moving from simple to complex guarantees coverage while controlling computational overhead; execution validation automatically achieves joint entity-relation disambiguation.

2. **Cross-layer Combination**:

    - **Function**: Build multi-constraint query structures (width of the query graph).
    - **Mechanism**: Select two queries from different layers $q \in \mathcal{L}_x$ and $q' \in \mathcal{L}_y$, find their respective variables whose execution results overlap ($\mathcal{E}(o_i) \cap \mathcal{E}(o_j) \neq \emptyset$), and merge the two queries via this shared variable. The combination terminates at 5 edges, covering the vast majority of question structures in current datasets. Starting with simple combinations ($\mathcal{L}_{1\times 1}$), it gradually scales to more complex schemas ($\mathcal{L}_{2\times 3}$, $\mathcal{L}_{1\times(1\times 2)}$).
    - **Design Motivation**: Multi-constraint queries are crucial in KBQA (e.g., "people who were born in X and attended Y"), which cannot be modeled solely by chained expansion.

3. **Hierarchical Ranking + Query Textification (Re-ranking)**:

    - **Function**: Filter the most relevant demonstrations from a potentially large volume of synthetic queries.
    - **Mechanism**: First, **Query Textification** is used to translate SPARQL queries into quasi-natural language text using heuristic rules (bridging the semantic gap between embedding models and queries). Then, bge-reranker-v2-m3 is applied to calculate the similarity to the question. **Hierarchical Ranking** is adopted — keeping only the top-n sub-queries derived from the same parent query to prevent imbalance caused by the exponential growth of complex queries.
    - **Design Motivation**: Many valid query structures in synthetic data are irrelevant to the current question and must be efficiently filtered; hierarchical ranking ensures queries across different complexity levels have a chance to be selected.

### Loss & Training

TARGA is a zero-annotation, training-free framework where all core components run dynamically during inference:
- Entity Linking: Directly reuse the results from Gu et al. (2023).
- Relation Retrieval: Use text-embedding-ada-002 to compute similarity between the question and Freebase relations, selecting the top-20.
- Number of examples: 10 demonstrations by default.
- Backbone model: Qwen-2.5-7B-Instruct by default.

## Key Experimental Results

### Main Results

| Dataset | Metric (F1) | TARGA (7B) | Prev. SOTA (non-fine-tuned) | Gain |
|--------|----------|------------|-------------------|------|
| GrailQA | F1 | 69.0 | 61.3 (KB-Coder-R, GPT-3.5) | +7.7 |
| GraphQ | F1 | 50.6 | 50.8 (QueryAgent, GPT-3.5) | -0.2 |
| KBQA-Agent | F1 | 46.5 | 34.3 (FUXI, GPT-3.5) | +12.2 |
| MetaQA | F1 | 85.7 | 99.5 (KB-Binder-R) | -13.8 |

**Stronger Backbone Models**:

| Model | GrailQA | GraphQ | KBQA-Agent |
|------|---------|--------|------------|
| Qwen-2.5-7B | 69.0 | 50.6 | 46.5 |
| Qwen-2.5-72B | 70.6 | 54.1 | 57.3 |
| GPT-3.5-turbo | 68.9 | 51.0 | 52.7 |
| GPT-4-turbo | 69.8 | 52.5 | 51.4 |

### Ablation Study

**Generalization Capability (Three settings in GrailQA)**:

| Method | I.I.D. | Compositional | Zero-shot | Avg |
|------|--------|--------------|-----------|-----|
| KB-Binder-R | 80.6 | 53.6 | 50.7 | 58.5 |
| KB-Coder-R | 81.0 | 57.8 | 54.1 | 61.3 |
| TARGA | 68.4 | 62.2 | 71.7 | 69.0 |
| TARGA-R | 80.8 | 63.6 | 71.6 | 71.9 |

**Efficiency Analysis (GrailQA)**:

| Method | TPQ (sec) | QPQ (queries) | CPQ (cost $) |
|------|---------|-------------|-----------|
| KB-BINDER | 51.2 | 3,297.7 | 0.010 |
| QueryAgent | 16.6 | 5.2 | 0.019 |
| TARGA | 4.5 | 256.8 | 0.000 |

### Key Findings

- **Extremely High Sample Efficiency**: Utilizing just 1 synthetic demonstration leaks stronger performance than the 20-shot random/similarity retrieval settings, demonstrating that synthetic data quality far exceeds static dataset retrieval.
- **Substantial Generalization**: Under the Zero-shot setting (fully unseen KB elements/structures), TARGA achieves a 71.7 F1, outperforming KB-Coder-R (54.1) by 17.6 points. Pre-collected training sets offer almost no help for Zero-shot scenarios.
- **Strong Robustness**: In adversarial settings (randomly replacing relations in demonstrations), TARGA's performance drops by only about 25%, compared to 75% for the random setting, proving that informational redundancy among synthetic examples provides robustness.
- **No Reliance on Powerful Closed-source Models**: The 7B open-source model outperforms all non-fine-tuned methods based on GPT-3.5-turbo, owing to high-quality examples that simplify the task.
- **Fastest Response Speed**: 4.5 seconds per question, which is 11× faster than KB-BINDER, with zero closed-source LLM call cost.

## Highlights & Insights

- **"Dynamic Synthesis vs. Static Retrieval" Paradigm Shift**: Instead of retrieving similar examples from static datasets, targeted examples are constructed dynamically on the KB in real-time based on the current question. This bypasses the root cause of generalization issues (where the training distribution cannot cover all testing scenarios).
- **Cleverly Leveraging KB Structural Constraints for Query Construction**: Joint entity-relation disambiguation is automatically achieved via non-empty execution validation, constraining the combinatorial explosion to only dozens of valid candidate queries per question on average.
- **Query Textification Bridges Semantic Gap**: Translating SPARQL queries to natural language via simple heuristic rules improves embedding ranking accuracy; it is a straightforward yet highly effective design.
- **From Simple to Complex Construction Strategy**: Moving from $\mathcal{L}_1$ to $\mathcal{L}_3$, and from single-layer to cross-layer combinations, ensures that complex queries are invariably assembled from validated simple sub-queries.

## Limitations & Future Work

- Performance on MetaQA is lower than KB-Binder-R which requires a full training set (85.7 vs 99.5), indicating that dynamic synthesis might not offer advantages in simple environments with sufficient annotations.
- It relies on existing entity linking methods; poor linking quality directly affects subsequent query construction.
- The number of KB queries during query construction remains high (256.8 QPQ), which may pose a performance bottleneck on large-scale KBs.
- It is currently limited to KBQA (Freebase); its efficacy on other structured reasoning tasks like Text2SQL has not been fully verified (though preliminary WikiSQL results are mentioned).
- The top-n parameter in hierarchical ranking and the maximum complexity threshold require manual tuning.

## Related Work & Insights

- **vs. KB-Binder/KB-Coder**: These retrieve examples from annotated training sets for ICL, and their performance drops by 20%+ in non-I.I.D. settings. TARGA dynamically generates examples and is entirely unaffected by training set distribution. In I.I.D. settings, incorporating training sets (TARGA-R) achieves competitive results.
- **vs. BYOKG (Agarwal et al.)**: BYOKG also requires no annotations but uses offline synthesis and static retrieval, which still suffers from generalization issues. TARGA synthesizes online, obtaining customized examples for every question.
- **vs. Agent-based (QueryAgent, AgentBench)**: Agent methods decompose questions progressively. Although they generalize well, they incur high computational costs (multiple LLM calls) and rely heavily on strong model capabilities. TARGA accomplishes the task with a single ICL call using a 7B model.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The dynamic, targeted synthesis of examples is highly ingenious, transforming KB structural constraints into data quality guarantees.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive analysis across 4 datasets, evaluating generalization, robustness, efficiency, model size, etc.
- Writing Quality: ⭐⭐⭐⭐ Well-structured methodology descriptions with clear formulations.
- Value: ⭐⭐⭐⭐⭐ Zero annotations and a small model outperforming closed-source LLM methods make it highly practical and directly applicable to real-world KBQA systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] STRICTA: Structured Reasoning in Critical Text Assessment for Peer Review and Beyond](stricta_structured_reasoning_in_critical_text_assessment_for_peer_review_and_bey.md)
- [\[ACL 2025\] Generating Synthetic Relational Tabular Data via Structural Causal Models](generating_synthetic_relational_tabular_data_via_structural_causal_models.md)
- [\[ICLR 2026\] Using maximal information auxiliary variables to improve synthetic data generation based on TabPFN foundation models](../../ICLR2026/others/using_maximal_information_auxiliary_variables_to_improve_synthetic_data_generati.md)
- [\[ICML 2025\] Provably Improving Generalization of Few-Shot Models with Synthetic Data](../../ICML2025/others/provably_improving_generalization_of_few-shot_models_with_synthetic_data.md)
- [\[ACL 2025\] AIDE: Attribute-Guided Multi-Hop Data Expansion for Data Scarcity in Task-Specific Fine-tuning](aide_attribute-guided_multi-hop_data_expansion_for_data_scarcity_in_task-specifi.md)

</div>

<!-- RELATED:END -->
