---
title: >-
  [Paper Note] Hierarchical Memory Organization for Wikipedia Generation
description: >-
  [ACL 2025][Wikipedia Generation] Proposes the Memory Organization-based Generation (MOG) framework, which extracts fine-grained memory units (factoids) from web documents and organizes them into a hierarchical Wikipedia outline structure via a recursive clustering-summarization algorithm. This ensures that every section is directly supported by memory. It comprehensively outperforms RAG and STORM baselines in terms of informativeness, citation rate…
tags:
  - "ACL 2025"
  - "Wikipedia Generation"
  - "Hierarchical Memory"
  - "RAG"
  - "Outline Generation"
  - "Citation Traceability"
date: 2026-05-08
content_hash: 95ee6dc6c8f74c70
---

# Hierarchical Memory Organization for Wikipedia Generation

**Conference**: ACL 2025  
**arXiv**: [2506.23393](https://arxiv.org/abs/2506.23393)  
**Code**: [https://github.com/eugeneyujunhao/mog](https://github.com/eugeneyujunhao/mog)  
**Area**: Other  
**Keywords**: Wikipedia Generation, Hierarchical Memory, RAG, Outline Generation, Citation Traceability  

## TL;DR
Proposes the Memory Organization-based Generation (MOG) framework, which extracts fine-grained memory units (factoids) from web documents and organizes them into a hierarchical Wikipedia outline structure via a recursive clustering-summarization algorithm. This ensures that every section is directly supported by memory. It comprehensively outperforms RAG and STORM baselines in terms of informativeness, citation rate, and verifiability on the FreshWiki and WikiStart datasets.

## Background & Motivation
**Background**: Automatically generating Wikipedia articles requires retrieving, integrating, and structuring information from multiple web sources. Methods like STORM collect information through multi-turn dialogues and then generate outlines from the dialogue history; RAG methods directly use document chunks as context.

**Limitations of Prior Work**: (a) There is a **misalignment** between outlines and memory (retrieved documents)—generated outlines may contain sections unsupported by evidence in the memory (leading to hallucinations), or miss valuable information present in the memory (reducing comprehensiveness); (b) using entire document chunks as memory units is too coarse-grained, containing irrelevant details and resulting in low utilization (only ~30% of RAG is cited).

**Key Challenge**: Outline generation and information retrieval are separated—generating the outline first and then retrieving information leads to outlines either "fabricating out of thin air" or "missing important content."

**Goal**: How to make outline structures directly emerge from the memory content, rather than generating them independently and then matching.

**Key Insight**: Treat memory units as "atomic facts" and recursively build a hierarchical structure through clustering and summarization—the outline is the organizational form of memory, naturally aligning the two.

**Core Idea**: Outlines are not generated a priori, but emerge bottom-up from the hierarchical organization of memory units.

## Method

### Overall Architecture
A three-stage pipeline: (1) **Memory Construction**—subtopic exploration + document retrieval + factoid extraction to build a set of fine-grained memory units; (2) **Memory Organization**—recursive clustering + summarization + heading generation to organize memory units into a hierarchical outline; (3) **Article Generation**—section-by-section generation + citation module to associate memory units with source documents.

### Key Designs

1. **Fine-grained Memory Units**:

    - **Function**: Extract independent atomic facts (factoids) from retrieved web documents to serve as memory units.
    - **Mechanism**: Define five memory operations—[save], [recall], [extract], [cluster], [summarize]. [extract] uses LLMs to extract independent, topic-related facts from documents.
    - **Design Motivation**: Entire document chunks contain a large amount of irrelevant information and suffer from coreference ambiguity; atomic facts are more precise, with utilization reaching 75%+ (vs 30% for RAG).

2. **Recursive Hierarchical Organization Algorithm**:

    - **Function**: Organize memory units bottom-up into a Wikipedia-style hierarchical outline.
    - **Mechanism**:
        - Perform K-Means clustering on current-level memory units
        - Summarize each cluster
        - Use the set of summaries to prompt the LLM to generate section headings
        - Each memory unit is assigned to the nearest section according to semantic similarity $l^* = \arg\max_{l'} \text{sim}(m, l')$
        - Recursively perform the above process on information-dense sections to generate sub-sections
    - **Design Motivation**: Bottom-up construction ensures that every section is supported by memory units, avoiding "unsupported sections"; the recursion depth is dynamically adjusted based on information density.

3. **Subtopic Explorer**:

    - **Function**: Expand search coverage to span multiple facets of the topic.
    - **Mechanism**: After the initial retrieval, summarize the collected memory units, discover subtopics from the summaries, and conduct secondary retrieval for these subtopics.
    - **Design Motivation**: A single search offers limited coverage; subtopic exploration boosts the number of effective web pages by 3x.

4. **Post-hoc Citation Module**:

    - **Function**: Associate each generated sentence with a source memory unit to achieve traceable citations.
    - **Mechanism**: After sentence segmentation with NLTK, use an LLM to map each sentence to the most relevant memory unit. Memory units retain links to their source documents.
    - **Design Motivation**: The benefit of fine-grained memory units is that citations point to precise facts rather than long documents, making verification significantly easier.

### Loss & Training
- Training-free: A pure prompt engineering framework, using GPT-4o for outlining/generation, and GPT-4o-mini for other tasks.
- Fine-tuned all-miniLM-L6-v2 as the vector database encoder, trained with Wikipedia section heading-sentence pairs.

## Key Experimental Results

### Main Results

| Dataset | Method | Word Count | Entity Count | Citation Recall (↑) | Citation Precision (↑) | Citation Rate (↑) |
|--------|------|------|--------|-----------|-----------|----------|
| FreshWiki | RAG | 1712 | 96.00 | 67.72 | 66.50 | 86.09 |
| FreshWiki | STORM | 1647 | 73.66 | 58.10 | 67.06 | 72.81 |
| FreshWiki | **MOG** | **2049** | **131.40** | **82.14** | **76.57** | **95.91** |
| WikiStart | RAG | 1271 | 58.97 | 54.70 | 46.80 | 76.80 |
| WikiStart | STORM | 965 | 39.47 | 47.75 | 47.40 | 85.66 |
| WikiStart | **MOG** | **1451** | **105.83** | **77.23** | **68.64** | **95.89** |

### Ablation Study

| Configuration | Section Count | Word Count | Entity Count | Effective Web Pages |
|------|--------|------|--------|----------|
| Full MOG | 21.46 | 2046 | 141 | 54.0 |
| w/o Subtopic Explorer | 9.78 | 1042 | 84 | 18.3 |
| w/o Memory Organization | 8.18 | 1360 | 105 | 55.3 |

### Key Findings
- Articles generated by MOG have 37% more entities and 14.4 percentage points higher citation recall than RAG, proving to be more comprehensive and verifiable.
- Memory utilization is a core advantage—MOG achieves 75%+ vs RAG's ~30%, as fine-grained memory units prevent waste of information.
- MOG is more robust in low-resource scenarios (WikiStart)—citation recall drops only by 2.93%, compared to a drop of ~10% for RAG/STORM.
- The subtopic explorer boosts the number of effective web pages by 3x and the entity count by 40%.
- MOG is comparable to or slightly outperforms baselines on LLM evaluation metrics (Organization/Interest/Focus), proving that the increase in informativeness does not come at the cost of quality.

## Highlights & Insights
- **"Outlines emerge from memory"** is the core insight—rather than planning first and searching for evidence, evidence is collected first and the plan is built from it. This avoids the fundamental issue of outline-memory misalignment.
- **75%+ memory utilization** is exceptionally high—while most retrieved documents in traditional RAG are wasted, MOG's fine-grained extraction ensures almost every memory unit is utilized.
- **The recursive organization algorithm** naturally adapts to the volume of information—information-rich sections automatically spawn more sub-sections, while those with sparse information remain concise.
- Post-hoc citation combined with fine-grained memory makes verification costs extremely low—users can jump directly to precise source facts rather than long documents.

## Limitations & Future Work
- Lack of a robust fact-checking mechanism when multi-source information conflicts.
- Factoids as memory units may lose temporal or narrative context.
- High cost due to dependency on GPT-4o.
- Evaluated only on English Wikipedia; multilingual capabilities remain unknown.
- Choice of K in K-Means clustering significantly affects outline quality, requiring a more adaptive approach.

## Related Work & Insights
- **vs STORM**: STORM constructs outlines from dialogue history, which may not align with actual memory; MOG constructs them directly from memory, ensuring natural alignment.
- **vs RAG**: RAG uses entire document chunks, which are coarse-grained with low utilization; MOG's factoid-level memory is more precise and efficient.
- **vs Sauper & Barzilay (2009)**: Early approaches used template outlines, which were inflexible; MOG's recursive organization adapts to any topic.
- This framework can be applied to other long-form structured text generation tasks (e.g., survey reports, technical documentation).

## Rating
- Novelty: ⭐⭐⭐⭐ The paradigm shift of "outlines emerging from memory" is valuable, and the recursive organization algorithm is elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation using two datasets, ablations, utilization analysis, and LLM evaluations.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, intuitive representation, and well-defined memory operations.
- Value: ⭐⭐⭐⭐⭐ Highly valuable for automated knowledge base generation; the MOG framework possesses strong generalizability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Detecting Sockpuppetry on Wikipedia Using Meta-Learning](detecting_sockpuppetry_on_wikipedia_using_meta-learning.md)
- [\[ACL 2025\] If Attention Serves as a Cognitive Model of Human Memory Retrieval, What is the Plausible Memory Representation?](if_attention_serves_as_a_cognitive_model_of_human_memory_retrieval_what_is_the_p.md)
- [\[ACL 2025\] Hierarchical Bracketing Encodings for Dependency Parsing as Tagging](hierarchical_bracketing_dep_parsing.md)
- [\[AAAI 2026\] LeanRAG: Knowledge-Graph-Based Generation with Semantic Aggregation and Hierarchical Retrieval](../../AAAI2026/others/leanrag_knowledge-graph-based_generation_with_semantic_aggregation_and_hierarchi.md)
- [\[ACL 2025\] Evaluating the Evaluation of Diversity in Commonsense Generation](evaluating_the_evaluation_of_diversity_in_commonsense_generation.md)

</div>

<!-- RELATED:END -->
