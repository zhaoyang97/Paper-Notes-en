---
title: >-
  [Paper Note] Enabling LLM Knowledge Analysis via Extensive Materialization
description: >-
  [ACL 2025][LLM (Other)][Knowledge Base Construction] This paper proposes a methodology to materialize the factual knowledge of LLMs into a knowledge base on a large scale through recursive querying and result consolidation. Leveraging this, the authors construct GPTKB, which contains 101 million triples and 2.9 million entities, and conduct the first comprehensive analysis of the scale, accuracy, bias, timeliness, and consistency of GPT-4o-mini's knowledge.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Knowledge Base Construction"
  - "LLM Knowledge Materialization"
  - "Knowledge Graphs"
  - "Recursive Querying"
  - "Factual Knowledge Analysis"
date: 2026-05-08
content_hash: 5bf5b64a41cd31d9
---

# Enabling LLM Knowledge Analysis via Extensive Materialization

**Conference**: ACL 2025  
**arXiv**: [2411.04920](https://arxiv.org/abs/2411.04920)  
**Code**: [gptkb.org](https://gptkb.org)  
**Area**: LLM/NLP  
**Keywords**: Knowledge Base Construction, LLM Knowledge Materialization, Knowledge Graphs, Recursive Querying, Factual Knowledge Analysis

## TL;DR

This paper proposes a methodology to materialize the factual knowledge of LLMs into a knowledge base on a large scale through recursive querying and result consolidation. Leveraging this, the authors construct GPTKB, which contains 101 million triples and 2.9 million entities, and conduct the first comprehensive analysis of the scale, accuracy, bias, timeliness, and consistency of GPT-4o-mini's knowledge.

## Background & Motivation

LLMs internalize a vast amount of factual knowledge, which is a key factor in their success. Since Petroni et al. (2019), analyzing LLM knowledge has emerged as an independent subfield. However, existing approaches suffer from a fundamental issue—**availability bias (Availability Bias)**:

**Transient Single-point Exploration**: Probing with only one question at a time, discarding the results immediately after use.

**Constrained by Predefined Samples**: Only capable of discovering knowledge within the scope pre-defined by the experimenters.

**Superficial Coverage**: Typically using only hundreds to hundreds of thousands of samples, failing to touch the breadth and depth of LLM knowledge.

For instance, the authors find that GPT-4o-mini possesses a wealth of knowledge about art movements, hobbies, and other fields that are largely uncovered in existing knowledge bases. Consequently, the authors propose to **persistently materialize LLM knowledge into a knowledge base**—transitioning from one-off probing to constructing reusable structured resources.

This task faces three major challenges: (1) execution time and cost—inference is slow and expensive; (2) variance, hallucination, and scoping—requiring high output but discouraging hallucinations; and (3) global inconsistency—successive prompts may generate duplicate relations and entities.

## Method

### Overall Architecture

The construction of GPTKB consists of two phases:

**Phase I: Knowledge Elicitation**
- Starting from a seed entity (Vannevar Bush)
- Prompting the LLM to return knowledge in the form of triples about the entity
- Identifying new named entities in the triple objects using NER
- Adding the new entities to a queue for breadth-first search (BFS) to iteratively expand the knowledge graph

**Phase II: Knowledge Consolidation**
- Relation clustering: Merging duplicate relation names
- Category clustering: Merging duplicate category names
- Taxonomy construction: Building a coherent hierarchical taxonomy for categories
- Entity deduplication: Eliminating duplicate entities

### Key Designs

**Knowledge Prompting Design**:
- Instead of fixing the number of returned triples, elastic guidance is provided based on the popularity of the entity, allowing Einstein to return far more triples than common entities.
- Requiring at least one `instance_of` triple to facilitate structural categorization.
- Using OpenAI's Structured Outputs feature to reduce parsing errors and hallucinations.

**Named Entity Recognition (NER) Processing**:
- Early experiments suffered from topic drift caused by linguistic facts, translations, and similar content.
- Existing NER frameworks struggle to handle short entity labels without context.
- Ultimately, the LLM itself is utilized for NER to batch-process multiple candidates.

**Relation Clustering Algorithm** (Algorithm 1):
- Based on a greedy strategy that merges low-frequency relations into their most similar high-frequency counterparts.
- Using SentenceTransformers to compute cosine similarity.
- Applying adaptive thresholds: lower thresholds for low-frequency relations (more aggressive merging) and higher thresholds for high-frequency relations (more conservative).
- Parameters: $\alpha=1.4$, highest threshold $H=0.95$, lowest threshold $L=0.75$.

**Taxonomy Construction Algorithm** (Algorithm 2):
- First prompting the LLM to generate a high-level taxonomy skeleton.
- Computing and ranking the generality score of each existing category.
- Using depth-first search to find the lowest matching node, and then prompting the LLM to update the sub-taxonomy.
- Potentially generating intermediate category nodes automatically.

**Entity Deduplication**:
- Using standard deduplication methods based on blocking keys.
- Focusing on the person category, using the date of birth as the blocking key.
- Criteria within the same block: label embedding cosine similarity $> 0.85$ and $30\%$ of the triples match exactly.

### Loss & Training

This work does not involve model training. The core strategy is **batch API calling + post-processing consolidation**:
- Leveraging the batch request capability of GPT-4o-mini, 100 batches (10,000 entities per batch) can be sent in parallel upon startup.
- BFS is executed to a depth of 10 layers, totaling 2,200 batches, with 5.8 million entities prompted.
- Total execution time is 27 hours, and the total API cost is \$3,500.

## Key Experimental Results

### Main Results

**GPTKB Scale Statistics**:
- 2.9M entities, 101M triples
- 567K relations (788K before clustering), 4,715 categories (103K before clustering)
- Average of 35 triples per entity
- 37M triple objects are entities, 64M are literals

**Accuracy Evaluation (Based on 1,000 Samples)**:
- Entity level: 74% verifiable, 9% plausible, 17% unverifiable
- Triple level: 31% true, 61% plausible, 1% implausible, 7% incorrect
- Taxonomy: 64% of subclass-superclass edges are judged correct, 70% of selected superclasses are considered optimal

**Comparison with Wikidata**:
- 37% of GPTKB entities exist in Wikidata, while 63% are newly discovered
- Among the 41 triples of Vannevar Bush, more than 10 are not present in Wikidata
- GPTKB contains relations not modeled in Wikidata: `historical_significance` (342K), `art_style` (84K), `hobbies` (24K)

### Ablation Study

**Accuracy Comparison of Different LLMs**:

| Model | Verifiable Triples | Wikidata Entities | Verifiable Entities |
|------|-------------|-------------|-----------|
| GPT-4o-mini | 0.38 | 0.78 | 0.80 |
| Llama 3.1 70B | 0.69 | 0.83 | 0.95 |
| GPT-4o | 0.78 | 0.88 | 0.98 |

**Knowledge Consistency Test (Repeating Vannevar Bush 100 times)**:
- Two distinct clusters: 52 runs average 21 triples, 32 runs average 38 triples
- Within the first cluster, 79 out of 1,116 total triples are unique, with each triple shared by 14 runs on average
- Average overlap rate of exact match set intersections is 0.67

### Key Findings

1. **The scale of LLM factual knowledge far exceeds expectations**: GPT-4o-mini ($\approx 8\text{B}$ parameters) can yield 101M triples, translating to approximately 79 parameters/triple.
2. **Highly complementary to traditional KBs**: 63% of entities do not exist in Wikidata, spanning new domains like digital media, art movements, and personal hobbies.
3. **Significant geographical and cultural bias exists**: 119K Americans vs. only 3K Chinese, reflecting the English-centric bias of the training corpora.
4. **Inconsistent inverse relations**: Only 8K out of 318K spouse triples are symmetric, and only 6K out of 61K parent company triples have corresponding subsidiary triples.
5. **Clear knowledge recency cutoff**: The frequency drops sharply after 2023, coinciding with the known knowledge cutoff.
6. **Gender bias shows improvement**: 15K females vs. 8K males in specified gender attributes, reflecting LLM debiasing efforts.

## Highlights & Insights

- **Paradigm Innovation**: Shifting from "temporary probing" to "persistent materialization," pioneering a new paradigm for LLM knowledge analysis.
- **Stunning Cost Efficiency**: The API cost per correct triple is only \$0.0001, which is over 100 times lower than traditional automated KB construction.
- **One Resource, Multi-Analysis**: As a persistent resource, GPTKB simultaneously supports multi-dimensional analyses of scale, accuracy, bias, recency, consistency, and more.
- **Recursive Graph Expansion Avoids Availability Bias**: It does not rely on predefined question sets, enabling the discovery of knowledge unexpected by the researchers.

## Limitations & Future Work

- **Prompt Dependency**: Different prompts generate different KBs; current results only represent a lower bound of LLM knowledge.
- **Reproducibility Risk**: Being based on closed-source LLMs, the service is subject to potential disruption.
- **Hallucination Issue Unresolved**: Particularly within fictional character categories, substantial fabrication occurs (e.g., Officer K.I.T.T. XV).
- **Ample Room for Deduplication and Normalization**: Such as entity normalization, literal typing, and organizing relations/sub-relations.
- **Precision-Recall Trade-off**: Hallucinations are difficult to define in long-tail knowledge.
- **Budget constraints**: A full run on GPT-4o would cost approximately \$825,000, which exceeds academic budgets.

## Related Work & Insights

- Inherits ideas from classic iterative information extraction like DIPRE (Brin 1998) and Snowball, but applies them to internal LLM knowledge for the first time.
- Compared to the small-scale proposal of Cohen et al. (2023) to extract knowledge bases, this work addresses practical engineering challenges and achieves large-scale construction.
- Provides an empirical foundation for LLM epistemology: debates about what LLMs "know" can be empirically analyzed using materialized resources.
- While innovation in traditional KBs (Wikidata/YAGO/DBpedia) has stagnated, this paper provides a brand-new construction paradigm.

## Rating

- **Novelty**: ★★★★★ (Proposes a brand-new paradigm for LLM knowledge materialization)
- **Experimental Thoroughness**: ★★★★★ (Large-scale construction of 101M triples and multi-dimensional analysis)
- **Value**: ★★★★☆ (GPTKB is readily usable, though its precision still has room for improvement)
- **Writing Quality**: ★★★★★ (Clear structure, explicit contributions, in-depth discussions)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Pre³: Enabling Deterministic Pushdown Automata for Faster Structured LLM Generation](pre3_deterministic_pda_structured_gen.md)
- [\[ACL 2025\] Condor: Enhance LLM Alignment with Knowledge-Driven Data Synthesis and Refinement](condor_enhance_llm_alignment_with_knowledge-driven_data_synthesis_and_refinement.md)
- [\[ACL 2025\] INTERACT: Enabling Interactive, Question-Driven Learning in Large Language Models](interact_enabling_interactive_question-driven_learning_in_large_language_models.md)
- [\[ACL 2025\] Re-TASK: Revisiting LLM Tasks from Capability, Skill, and Knowledge Perspectives](re-task_revisiting_llm_tasks_from_capability_skill_and_knowledge_perspectives.md)
- [\[ACL 2025\] Acquisition and Application of Novel Knowledge in Large Language Models](acquisition_and_application_of_novel_knowledge_in_large_language_models.md)

</div>

<!-- RELATED:END -->
