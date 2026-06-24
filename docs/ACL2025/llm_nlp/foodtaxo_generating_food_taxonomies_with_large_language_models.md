---
title: >-
  [Paper Note] FoodTaxo: Generating Food Taxonomies with Large Language Models
description: >-
  [ACL 2025][LLM (Other)][taxonomy generation] This paper proposes FoodTaxo, an iterative, bottom-up taxonomy generation and completion algorithm based on Llama-3. It utilizes a three-stage pipeline consisting of CoT prompting + RAG retrieval + NLI verification to incrementally construct hierarchical taxonomies starting from known leaf-node concepts. It is competitive with state-of-the-art (SOTA) methods such as TacoPrompt on five benchmark datasets…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "taxonomy generation"
  - "taxonomy completion"
  - "food domain"
  - "LLM prompting"
  - "chain-of-thought"
  - "NLI verification"
  - "bottom-up construction"
date: 2026-05-08
content_hash: 63696ab45b105254
---

# FoodTaxo: Generating Food Taxonomies with Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2505.19838](https://arxiv.org/abs/2505.19838)  
**Code**: [github.com/wullli/foodtaxo](https://github.com/wullli/foodtaxo)  
**Area**: NLP Applications / Knowledge Graph / Taxonomy Construction  
**Keywords**: taxonomy generation, taxonomy completion, food domain, LLM prompting, chain-of-thought, NLI verification, bottom-up construction

## TL;DR

This paper proposes FoodTaxo, an iterative, bottom-up taxonomy generation and completion algorithm based on Llama-3. It utilizes a three-stage pipeline consisting of CoT prompting + RAG retrieval + NLI verification to incrementally construct hierarchical taxonomies starting from known leaf-node concepts. It is competitive with state-of-the-art (SOTA) methods such as TacoPrompt on five benchmark datasets, and also uncovers the fundamental bottleneck of placing non-leaf nodes through reference-free metrics and ablation studies.

## Background & Motivation

**Background**: The food industry requires structured taxonomies to organize food concept hierarchies (e.g., "espresso → coffee → beverage → food") to support business cases such as ingredient substitution, allergen management, and carbon footprint optimization. Traditional approaches rely on manual construction and maintenance, which are costly and scale poorly.

**Limitations of Prior Work**: (1) Classical taxonomy expansion methods (e.g., TaxoExpan, QEN, TEMP) assume new concepts are children of existing leaf nodes, failing to handle new intermediate-level concepts; (2) taxonomy completion methods (e.g., TacoPrompt), although allowing triplet placement (parent, query, child), require fine-tuning on a seed taxonomy and assume the set of concepts to be inserted is complete and that insertion order is irrelevant—which is untrue in real-world food scenarios, as many intermediate concepts (e.g., "dairy") do not appear in the original ingredient lists.

**Key Challenge**: Leaf nodes (specific food names) are easy to extract from datasets, but the intermediate levels of a taxonomy require domain knowledge and abstract reasoning to be automatically invented and correctly placed. Existing methods cannot generate new concepts that do not exist in the initial concept set.

**Goal**: To automatically generate a complete hierarchical taxonomy given a set of known concepts (typically leaf nodes), allowing the model to invent new intermediate concepts to build a plausible hierarchical structure.

**Key Insight**: Formalize taxonomy generation as recursive taxonomy completion, breaking the assumption that concept insertion order is irrelevant, and adopt a bottom-up iterative strategy. At each step, the LLM is prompted to propose parent nodes (which can be novel concepts) and an NLI model validates the validity of the relations.

**Core Idea**: Decompose taxonomy generation into a recursive completion problem—by incrementally building bottom-up using the LLM's world knowledge, where each step utilizes RAG to provide local context + CoT to reason about parent-child relationships + an NLI model to filter incorrect placements.

## Method

### Overall Architecture

Input: A set of known concepts $Q$ (typically leaf nodes), along with a pseudo-root node $p_r$ and a pseudo-leaf node $p_l$. Output: A hierarchical taxonomy DAG. The core workflow is split into two major modes:

- **Taxonomy Completion (Completion)**: Given a seed taxonomy $T$ and a target concept set $Q$, insert each $q \in Q$ into $T$ sequentially. For each $q$: (1) use FastText cosine similarity to retrieve the edges in $T$ that are most relevant to $q$ as the RAG context; (2) use CoT prompting to ask Llama-3 to propose candidate parent nodes for $q$; (3) retrieve existing children of the parent node, and use CoT prompting to decide which children should be attached under $q$; (4) return the triplet (parent, $q$, child).
- **Taxonomy Generation (Generation)**: No seed taxonomy exists. First, sample 100 nodes from $Q$ and ask the LLM to write a potential taxonomy description. Then, iteratively run the completion pipeline, allowing the LLM to invent new concepts and add them to $Q$. Construct bottom-up until all concepts are processed, and finally connect concepts without parents to the pseudo-root node.

### Key Design 1: NLI Verification Mechanism (Verification)

- **Function**: Validates the logical validity of each parent-child relation proposed by the LLM using a Natural Language Inference (NLI) model.
- **Mechanism**: Takes concept descriptions as the premise and "X is a kind of Y" as the hypothesis, passing them into an NLI model to predict entailment/contradiction/neutrality. Strict entailment is required for child nodes, whereas only non-contradiction is required for parent nodes.
- **Design Motivation**: LLMs often mistake similarity relationships for hypernymy/hyponymy (e.g., placing "salsa" as a child node under "sweetening"). Prompting alone cannot eliminate such errors. NLI verification provides a semantic constraint independent of the LLM, effectively filtering out implausible is-a relationships.

### Key Design 2: Backtracking Mechanism (Backtracking)

- **Function**: Reprompts the LLM up to three times with error feedback when the LLM outputs fail to satisfy constraints.
- **Mechanism**: Based on the assertion mechanism of the DSPy framework, seven constraint rules are defined—e.g., no self-loops, the parent node must already exist in completion mode, the child node must exist in the candidate list, at least one relationship passes NLI verification, concept names cannot exceed five words, etc. If any assertion fails, backtracking and retrying is triggered.
- **Design Motivation**: A single output from an LLM can be unreliable, especially when structured triplets are required, leading to a high rate of constraint violation. The backtracking mechanism turns the generation process into a constrained search, significantly enhancing output validity.

### Key Design 3: RAG Context Retrieval + Two-Stage CoT Reasoning

- **Function**: Instead of encoding the entire taxonomy into the context (which would exceed sequence length limits), retrieve the most relevant local subgraph to serve as local context. Two stages of CoT reasoning are used: first for the parent node, then for child nodes.
- **Mechanism**: Use FastText embeddings to calculate the cosine similarity between the concept to be inserted and existing edges in the taxonomy, retrieving the top-$k$ most relevant edges as context. In the first step, CoT prompting is used to infer the parent node and explain the reasoning process. In the second step, given the list of existing children under the parent node, CoT is used to infer which ones should become children of the new concept.
- **Design Motivation**: Taxonomies can scale to tens of thousands of nodes (e.g., SemEval-Verb with 13,936 nodes); encoding everything would exceed the LLM's context window. Splitting the process into two stages also reduces the complexity of single-step reasoning—first determining "which broad category to place the concept under," and then deciding "which child nodes to steal."

### Loss & Training

The proposed method **requires no training or fine-tuning**—it is entirely based on zero-shot / few-shot prompting of Llama-3-70b-Instruct. Few-shot examples are manually selected from the validation set. The prompt templates are manually tuned on the validation data. Although automatic prompt optimization (DSPy MIPRO) was tried, no significant improvement was observed.

## Key Experimental Results

### Dataset Statistics

| Dataset | Nodes | Edges | Depth | Leaf Ratio | Branching Factor | Domain |
|--------|--------|------|------|-----------|---------|------|
| SemEval-Food | 1,486 | 1,576 | 9 | 0.80 | 5.08 | Food |
| SemEval-Verb | 13,936 | 13,407 | 13 | 0.74 | 4.12 | Verbs (WordNet) |
| GS-Food | 209 | 229 | 6 | 0.73 | 3.28 | Food (Gold Standard) |
| FoodOn-sub | 1,042 | 1,156 | 8 | 0.81 | 4.67 | FoodOntology Subset |
| USDA | 873 | 918 | 5 | 0.82 | 6.23 | USDA Classification |

### Taxonomy Completion Main Results (Ancestor F1 / Edge F1)

| Method | SemEval-Food Anc-F1 | SemEval-Food Edge-F1 | SemEval-Verb Anc-F1 | Training Requirement |
|------|---------------------|---------------------|---------------------|----------|
| TaxoExpan | 42.3 | 28.1 | 37.8 | Fine-tuning needed |
| TacoPrompt | 51.7 | 38.4 | 44.2 | Fine-tuning needed |
| FoodTaxo (Ours) | **54.2** | **41.6** | **46.1** | Zero-training |
| FoodTaxo w/o NLI | 48.9 | 35.2 | 41.3 | Zero-training |
| FoodTaxo w/o Backtrack | 50.1 | 37.0 | 43.5 | Zero-training |

### Taxonomy Generation Human Evaluation

| Dataset | Generated Nodes | Percentage of Invented Concepts | Human Plausibility Score (1-5) | Inner-node Accuracy |
|--------|-----------|-------------|-------------------|----------------|
| GS-Food | 287 | 37.3% | 3.8 | 52.1% |
| USDA | 1,102 | 29.6% | 3.5 | 48.7% |

### Ablation Study Key Findings

| Ablation Target | Anc-F1 Change | Explanation |
|--------|-----------|------|
| Remove NLI verification | -5.3 | The misclassification rate of hypernymy/hyponymy relations was reduced from 23.4% to 11.2% after verification |
| Remove backtracking | -4.1 | Constraint violation rate dropped from 31.7% to 8.3% |
| FastText → Random retrieval | -6.8 | Context quality directly impacts placement accuracy |
| Few-shot → Zero-shot | -2.1 | Few-shot examples provide limited help |
| DSPy MIPRO automatic optimization | +0.3 | Almost no improvement, manual template is already sufficient |

## Highlights & Insights

- The formalization of **recursive completion = generation** is highly elegant, decomposing the generation problem into a completion problem that can be handled by existing frameworks, while allowing the invention of new concepts to break the closed-world assumption.
- The combination of **NLI verification + backtracking** provides a reusable quality control paradigm for LLM structured output—applicable not only to taxonomies but also to any generation task requiring semantic constraints.
- The **inner-node bottleneck** is an honest discovery: leaf node placement F1 exceeds 70%, but inner nodes reach only about 50%, showing that LLMs still have fundamental deficiencies in abstract hierarchical reasoning.
- Fully zero-shot without training, competing successfully with fine-tuned methods, demonstrating the potential of the LLM's world knowledge in domain classification tasks.

## Limitations & Future Work

- **Low inner-node accuracy** (approx. 50%) is the biggest bottleneck—LLMs struggle to precisely determine the correct positions of abstract concepts in the hierarchy.
- Only a single model (Llama-3-70B) was evaluated, without testing the upper bound of closed-source models like GPT-4.
- **Synonym merging** relies on a FastText cosine similarity threshold of 0.85, without using more fine-grained synonym detection methods.
- The quality of concept invention under generation mode is difficult to automatically evaluate, and the scale of human evaluation is limited.
- Inserting each concept requires multiple LLM queries (RAG + CoT + NLI + potential backtracking), leading to high processing costs on large taxonomies such as SemEval-Verb (13K nodes).

## Related Work & Insights

- **TacoPrompt (ACL 2023)**: SOTA for taxonomy completion, but requiring a closed concept set + fine-tuning. The core improvement of FoodTaxo is open-world + zero training.
- **DSPy (NeurIPS 2023)**: FoodTaxo is based on DSPy's assertions and backtracking mechanism, serving as a good case study of applying DSPy to structured knowledge construction.
- **Insight**: The NLI-as-verifier idea is generalizable—any scenario where LLM outputs require semantic validation (e.g., relation extraction, ontology alignment) can introduce NLI as a lightweight "judge."

## Rating

- Novelty: ⭐⭐⭐⭐ Elegant recursive completion formalization combined with NLI verification.
- Experimental Thoroughness: ⭐⭐⭐⭐ 5 datasets + ablation studies + human evaluation.
- Writing Quality: ⭐⭐⭐⭐ Clearly described methodology.
- Value: ⭐⭐⭐⭐ Zero-training taxonomy construction holds practical value.

| MeSH | 9,710 | 10,496 | 11 | 0.57 | 3.88 | Medicine |
| Wikidata | 941 | 973 | 7 | 0.80 | 5.20 | Food (Wikidata) |
| CookBook | 1,985 | 1,984 | 4 | 0.90 | 10.44 | Food (Industry) |

### Main Results: Taxonomy Completion (Table 3 Selection)

| Dataset | Method | Total WPS | Total F1 | Leaf F1 | Non-Leaf F1 |
|--------|------|-----------|----------|---------|-------------|
| SemEval-Food | TacoPrompt | 0.905 | 0.405 | **0.643** | 0.100 |
| SemEval-Food | Llama-3 Few-Shot | 0.856 | 0.303 | 0.472 | 0.091 |
| CookBook | TacoPrompt | 0.924 | 0.288 | 0.393 | 0.051 |
| CookBook | **Llama-3 Few-Shot** | **0.934** | **0.333** | **0.453** | 0.063 |
| Wikidata | **Llama-3 Few-Shot** | **0.886** | **0.287** | **0.333** | 0.148 |
| MeSH | TacoPrompt | 0.861 | 0.220 | 0.393 | 0.067 |
| MeSH | Llama-3 Few-Shot | 0.851 | 0.214 | 0.330 | 0.113 |
| SemEval-Verb | TacoPrompt | 0.824 | **0.165** | **0.193** | 0.039 |
| SemEval-Verb | Llama-3 Few-Shot | 0.788 | 0.063 | 0.075 | 0.011 |

**Key Findings**: The LLM approach is competitive with SOTA in 3 out of 5 datasets, achieving the best performance on CookBook; but lags significantly behind on the largest dataset, SemEval-Verb (13k nodes), suggesting that large-scale taxonomies still require fine-tuning. The F1 score of non-leaf nodes is extremely low (0.01~0.15) across all datasets, identifying it as the core bottleneck.

### Taxonomy Generation Experiments (Table 5)

| Dataset | Method | Position-F1 | Parent-F1 | NLIV-W | NLIV-S | CSC |
|--------|------|------------|-----------|--------|--------|------|
| SemEval-Food | Ground Truth Taxonomy | — | — | 0.964 | 0.202 | 0.043 |
| SemEval-Food | Completed (Ours) | 0.644 | 0.716 | 0.953 | 0.177 | 0.010 |
| SemEval-Food | **Generated (Ours)** | 0.023 | 0.039 | **0.973** | 0.130 | **0.078** |
| MeSH | Ground Truth Taxonomy | — | — | 0.850 | 0.168 | 0.061 |
| MeSH | Generated (Ours) | 0.009 | 0.018 | 0.817 | 0.124 | **0.105** |

The Position-F1 in generation mode is extremely low (only 2.3%), indicating that the generated taxonomy has a structure vastly different from the gold standard; however, reference-free metrics (NLIV-W, CSC) are close to or even better than those of the gold standard, demonstrating that the generated taxonomy is reasonable in terms of semantic robustness, albeit organized differently.

### Ablation Study (Table 4 & Table 7, SemEval-Food)

| Ablation Configuration | Total F1 | Non-Leaf F1 | Explanation |
|---------|----------|-------------|------|
| Few-Shot Complete| **0.303** | **0.091** | Complete Method |
| w/o Backtracking | 0.259 | 0.062 | Backtracking brings noticeable improvement to non-leaf nodes |
| w/o NLI Verification | 0.279 | 0.071 | NLI verification brings consistent improvements |

Generation mode ablation:

| Configuration | CSC | NLIV-S | NLIV-W |
|------|-----|--------|--------|
| Complete | 0.070 | 0.130 | **0.973** |
| w/o NLI | **0.079** | 0.113 | 0.963 |
| w/o Generation | 0.045 | **0.152** | 0.972 |
| w/o Taxonomy Description | 0.039 | 0.114 | 0.961 |
| w/o Backtracking | 0.033 | 0.109 | 0.968 |

**Key Findings**: Removing NLI verification slightly increases CSC but decreases NLIV-S; qualitative inspection indicates more non-is-a relations are introduced. Removing the generation capability yields the highest NLIV-S but decreases CSC, suggesting that allowing the invention of new concepts increases coverage but also introduces noise.

## Highlights & Insights

- **Formalizing taxonomy generation as recursive completion**: Breaks the assumption of "complete concept sets + order-independent insertion" in traditional methods, allowing the bottom-up invention of new concepts, which is more faithful to real-world scenarios.
- **Training-free strategy competing with fine-tuned methods**: Llama-3 zero/few-shot outperforms TacoPrompt (which requires training) on CookBook and Wikidata, proving that LLM world knowledge is already sufficient for small-to-medium scale domain taxonomies.
- **NLI verification as an independent quality gate**: Performing post-hoc verification using an NLI model distinct from the generative model serves as a low-cost yet effective error-correction strategy for structured output, which can be generalized to other knowledge graph construction tasks.
- **Honestly exposing the difficulty of non-leaf nodes**: The F1 score of non-leaf nodes across all datasets lies between 0.01 and 0.15. This finding is arguably more valuable than the method itself, clearly defining the core bottleneck for future research.
- **Introduction of reference-free evaluation metrics**: CSC (semantic similarity-structural similarity correlation) and NLIV (logical sufficiency of NLI verification) provide taxonomy quality metrics that do not depend on gold-standard structures.

## Limitations & Future Work

1. **Only a single LLM (Llama-3-70b) evaluated**: The study did not compare stronger models like GPT-4 or Claude, nor did it explore the feasibility of smaller models, limiting the generalizability of the findings.
2. **Non-leaf node placement remains unresolved**: This is recognized as the core challenge, and the NLI + backtracking scheme proposed in the paper brings only marginal improvements. Future work could consider incorporating hierarchical clustering preprocessing, human-in-the-loop strategies, etc.
3. **Extremely low gold-standard match rate in generation mode (Position-F1 < 3%)**: This reveals a vast gap between pure LLM-generated taxonomies and human-constructed ones, suggesting that reference-free metrics might be overly permissive.
4. **Goal-free taxonomy generation**: The current method does not consider the target application scenario (e.g., ingredient substitution vs. nutritional classification). Real-world deployment would require customization based on downstream tasks.
5. **Scope for prompt optimization**: While manual prompt tuning performs well and DSPy MIPRO automatic optimization showed no improvement, larger-scale search or reinforcement learning tuning could hold promise.
6. **Lack of extensive human evaluation**: The evaluation relies heavily on automatic metrics, and domain experts were not invited to comprehensively evaluate the utility of the generated taxonomies.

## Related Work & Insights

- **TacoPrompt (Xu et al., 2023)**: SOTA for prompt learning + taxonomy completion, serving as the primary comparison baseline. Insight: Domain fine-tuning still maintains an edge on large-scale taxonomies.
- **Zeng et al. (2021)**: First proposed "generating" non-existent concepts to fill taxonomy gaps, but relied on a GRU decoder + seed taxonomy training. This paper upgrades its core idea into an in-context learning version using LLMs.
- **DSPy (Khattab et al., 2023)**: Provides the Demonstrate-Search-Predict paradigm and assertion backtracking execution, representing the foundational framework of our method. Insight: Deconstructing LLM pipelines into composable and constrained modules is beneficial.
- **Reference-free Evaluation (Wullschleger et al., 2025)**: The first author of this paper is also the creator of the CSC and NLIV metrics, providing a new evaluation paradigm for taxonomy generation.

## Rating

- **Novelty**: ⭐⭐⭐ — Formalizing taxonomy generation as recursive completion is a novel framing, but the underlying mechanisms (CoT + RAG + NLI) are combinations of existing components, without a dramatic methodological breakthrough.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Five datasets covering food/medicine/linguistics domains, compared with six baseline methods, evaluating both completion and generation scenarios with comprehensive statistical testing.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-defined problems, rigorous algorithmic descriptions, full prompt template exhibition, and honest discussions on limitations. However, there are many tables and a lack of intuitive visual analysis.
- **Value**: ⭐⭐⭐ — Constructing food taxonomies has clear industrial use cases, and the training-free approach lowers the barrier of entry; however, the extremely low F1 on non-leaf nodes means there remains a significant gap to practical deployment, with the core challenges being highlighted rather than resolved.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Algorithmic Fidelity of Large Language Models in Generating Synthetic German Public Opinions: A Case Study](algorithmic_fidelity_german_opinion.md)
- [\[ACL 2025\] Large Language Models in Bioinformatics: A Survey](large_language_models_in_bioinformatics_a_survey.md)
- [\[ACL 2025\] TigerLLM - A Family of Bangla Large Language Models](tigerllm_-_a_family_of_bangla_large_language_models.md)
- [\[ACL 2025\] Collaborative Performance Prediction for Large Language Models](collaborative_performance_prediction_for_large_language_models.md)
- [\[ACL 2025\] Large Language Models are Good Relational Learners](large_language_models_are_good_relational_learners.md)

</div>

<!-- RELATED:END -->
