---
title: >-
  [Paper Note] LAQuer: Localized Attribution Queries in Content-grounded Generation
description: >-
  [ACL 2025][Attribution] Proposes the Localized Attribution Queries (LAQuer) task—precisely localizing user-selected segments in generated text to corresponding segments in the source documents. This achieves a finer granularity of provenance than sentence-level attribution and is more user-directed than sub-sentence-level attribution, significantly reducing the attributed text length in multi-document summarization and long-form question answering.
tags:
  - "ACL 2025"
  - "Attribution"
  - "Fine-grained Attribution"
  - "Generative Explainability"
  - "Sub-sentence Localization"
  - "Content Provenance"
date: 2026-05-08
content_hash: 6e7456b9a6741d62
---

# LAQuer: Localized Attribution Queries in Content-grounded Generation

**Conference**: ACL 2025  
**arXiv**: [2506.01187](https://arxiv.org/abs/2506.01187)  
**Code**: [https://github.com/eranhirs/LAQuer](https://github.com/eranhirs/LAQuer)  
**Area**: Other  
**Keywords**: Attribution, Fine-grained Attribution, Generative Explainability, Sub-sentence Localization, Content Provenance  

## TL;DR

Proposes the Localized Attribution Queries (LAQuer) task—precisely localizing user-selected segments in generated text to corresponding segments in the source documents. This achieves a finer granularity of provenance than sentence-level attribution and is more user-directed than sub-sentence-level attribution, significantly reducing the attributed text length in multi-document summarization and long-form question answering.

## Background & Motivation

**Background**: Attributed Text Generation appends citation sources to LLM outputs to help users verify factuality. Existing methods primarily perform attribution at the sentence level—associating each generated sentence with an entire paragraph or the whole source document.

**Limitations of Prior Work**: (a) Sentence-level attribution is too coarse—a sentence often contains multiple facts, and users might only care about one but are forced to read all associated source documents; (b) Existing sub-sentence-level methods (e.g., based on hidden state similarity) automatically select the attribution granularity, which does not necessarily match the facts the user actually wants to verify; (c) Both methods are "fixed attributions"—predetermined and unable to respond to dynamic user queries.

**Key Challenge**: Users want to verify specific factual segments, but the attribution scope provided by the system is either too large (sentence-level) or uncontrollable (automatic sub-sentence-level).

**Goal**: Define a user-initiated attribution query task—where the user highlights an output segment of interest, and the system automatically localizes the precise supporting segment in the source documents.

**Key Insight**: Shift attribution from "system-preset" to "user-driven"—allowing users to choose what to verify, with the system returning only the precise relevant evidence.

**Core Idea**: User highlight $\rightarrow$ decontextualization $\rightarrow$ localizing supporting segments in source documents, achieving precise, on-demand attribution.

## Method

### Overall Architecture
A two-stage pipeline: (1) Content generation stage—generating output text based on source documents, optionally containing sentence-level attribution metadata; (2) LAQuer stage—users highlight segments to verify $\rightarrow$ Step A: Decontextualization (transforming highlighted segments into self-contained statements) $\rightarrow$ Step B: Query-oriented attribution (localizing supporting segments in the source documents).

### Key Designs

1. **Decontextualization**:

    - **Function**: Transforms user-highlighted segments (which may contain pronouns or ellipses) into self-contained independent statements.
    - **Mechanism**: For example, if a user highlights "They deserve to know", where "They" refers to "consumers" in the preceding context, the decontextualized version becomes "Consumers deserve to know what they are eating".
    - **Design Motivation**: Without decontextualization, "they" in the source documents might refer to different entities, leading to incorrect attribution. Independent statements ensure disambiguation in attribution.

2. **Query-oriented Attribution**:

    - **Function**: Localizes precise supporting segments in source documents for the decontextualized statements.
    - **Two methods**:
        - **LLM Prompting**: Directly prompts the LLM to output aligned segments from the source documents.
        - **Internal Representations**: Calculates the cosine similarity between source tokens and output tokens in the hidden states of different LLM layers, using the source tokens with the highest similarity to construct the supporting segment.
    - **Design Motivation**: Leverages existing attribution metadata to narrow the search space—if sentence-level attribution has already associated a sentence with two source paragraphs, the search is restricted to these two paragraphs.

3. **User Query Simulation**:

    - **Function**: Constructs LAQuer inputs for evaluation.
    - **Mechanism**: Decomposes each output sentence into atomic facts using FActScore, where each fact corresponds to a set of highlighted segments, simulating user selection behavior.
    - On average, each sentence is decomposed into 2.6 atomic facts.

### Loss & Training
- **No training required**—both methods are inference-time approaches (prompting or hidden state analysis).
- Uses GPT-4o for decontextualization.

## Key Experimental Results

### Main Results

| Task | Method | Attributed Text Length (↓) | Attribution Accuracy | Description |
|------|------|---------------|----------|------|
| MDS | Sentence-level | Long (entire paragraph) | High | Too much irrelevant information |
| MDS | **LAQuer-Prompt** | **Short (precise segment)** | Medium-High | Significantly reduces user reading load |
| MDS | **LAQuer-Internals** | **Short** | **Highest** | Internal representations are more accurate |
| LFQA | Sentence-level | Long | High | Same as above |
| LFQA | **LAQuer** | **Short** | Effective | Effective in both scenarios |

### Ablation Study

| Configuration | Effect | Description |
|------|------|------|
| w/o Decontextualization | Drop in attribution accuracy | Pronouns/ellipses cause ambiguity |
| w/ Sentence-level metadata | Search space reduced + accuracy improved | Two-stage synergy |
| w/o Sentence-level metadata | Still functional, but slight drop in precision | Requires searching the entire document |

### Key Findings
- LAQuer reduces the length of the attributed text users need to read by several fold—from an entire paragraph to a precise segment.
- The internal representation approach performs best when existing attribution metadata is available—token-level alignment from hidden states is highly precise.
- Decontextualization is a crucial step—attribution error rates increase significantly without it.
- LAQuer is effective for various types of output segments (phrases, clauses, complete sentences).
- Attribution accuracy correlates with the syntactic complexity of the segment—simple phrases are easier to attribute.

## Highlights & Insights

- **The paradigm shift from "system-fixed" to "user-driven"** is the core contribution—attribution should not be one-size-fits-all, but precisely provided based on user needs. This aligns with the trend of interactive RAG.
- **The two-step method of decontextualization + attribution** elegantly addresses coreference resolution—transforming the segment into a self-contained statement first makes searching the source documents much simpler.
- **The internal representation approach** showcases the token-alignment capability of LLM intermediate layers—achieving fine-grained alignment without extra training, which has broader potential applications.
- The framework can be directly integrated into any RAG system—serving as a post-processing module for "verification enhancement".

## Limitations & Future Work

- Decontextualization relies on GPT-4o, which may introduce errors.
- The decomposition of atomic facts used for evaluation also relies on LLMs, representing an approximate evaluation.
- Not validated in real user interaction scenarios—all queries are simulated.
- Search efficiency for long source documents could become a bottleneck.
- Only validated in English scenarios.

## Related Work & Insights

- **vs. Sentence-level Attribution (Gao et al. 2023)**: Sentence-level is too coarse, requiring users to read a large amount of irrelevant context; LAQuer provides precise segments.
- **vs. Phukan et al. (2024) Sub-sentence-level Attribution**: Automatically selects attribution granularity but is uncontrollable; LAQuer lets the user decide.
- **vs. FActScore**: FActScore decomposes facts and checks if they are supported; LAQuer not only checks support but also localizes the precise source segment.
- This localization capability can be applied to scenarios such as fact-checking and academic writing verification.

## Rating

- Novelty: ⭐⭐⭐⭐ Defines a meaningful new task and framework; the paradigm shift from "system-fixed attribution" to "user-driven attribution" has conceptual depth.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two tasks (MDS/LFQA) + two methods (prompting/internal representations) + ablations + multiple generators, though it lacks a real user study.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous task definition (with three clear desiderata), intuitive illustrations (the comparison in Fig. 1 is persuasive), and complete formalization.
- Value: ⭐⭐⭐⭐ Improves the verifiability and user experience of RAG/generative systems, allowing direct integration into existing systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] VAQUUM: Are Vague Quantifiers Grounded in Visual Data?](vaquum_are_vague_quantifiers_grounded_in_visual_data.md)
- [\[ACL 2025\] CiteEval: Principle-Driven Citation Evaluation for Source Attribution](citeeval_principle-driven_citation_evaluation_for_source_attribution.md)
- [\[ACL 2025\] Hierarchical Memory Organization for Wikipedia Generation](hierarchical_memory_wikipedia_gen.md)
- [\[ACL 2025\] Evaluating the Evaluation of Diversity in Commonsense Generation](evaluating_the_evaluation_of_diversity_in_commonsense_generation.md)
- [\[CVPR 2026\] Convolutional Neural Networks Driven by Content Similarity](../../CVPR2026/others/convolutional_neural_networks_driven_by_content_similarity.md)

</div>

<!-- RELATED:END -->
