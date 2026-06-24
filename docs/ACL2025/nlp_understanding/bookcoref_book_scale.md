---
title: >-
  [Paper Note] BookCoref: Coreference Resolution at Book Scale
description: >-
  [ACL 2025][NLP Understanding][Coreference Resolution] This work proposes BookCoref, the first book-scale coreference resolution benchmark. By employing an automatic annotation pipeline integrating character linking, LLM filtering, and window expansion, it generates high-quality silver annotation data across 50 full novels, with an average document length exceeding 200k tokens.
tags:
  - "ACL 2025"
  - "NLP Understanding"
  - "Coreference Resolution"
  - "Long Document"
  - "Book-Scale"
  - "Automatic Annotation Pipeline"
  - "Character Linking"
date: 2026-05-08
content_hash: 320caf2253887fcf
---

# BookCoref: Coreference Resolution at Book Scale

**Conference**: ACL 2025  
**arXiv**: [2507.12075](https://arxiv.org/abs/2507.12075)  
**Code**: [GitHub](https://github.com/sapienzanlp/bookcoref)  
**Area**: NLP Understanding  
**Keywords**: Coreference Resolution, Long Document, Book-Scale, Automatic Annotation Pipeline, Character Linking

## TL;DR

This work proposes BookCoref, the first book-scale coreference resolution benchmark. By employing an automatic annotation pipeline integrating character linking, LLM filtering, and window expansion, it generates high-quality silver annotation data across 50 full novels, with an average document length exceeding 200k tokens.

## Background & Motivation

**Background**: Coreference resolution systems are typically evaluated on short or medium-length documents (OntoNotes averages 467 tokens, while LitBank is truncated to 2000 tokens).

**Limitations of Prior Work**: There is a lack of book-scale benchmarks, and existing systems cannot effectively process coreference relations spanning hundreds of thousands of tokens. LongtoNotes consists of only 679 tokens per document, and MovieCoref contains only 9 documents.

**Key Challenge**: Manual annotation of long texts is extremely costly (requiring incremental reading of the entire book), whereas automatic annotation systems (such as Maverick) experience dramatic performance degradation on long texts (achieving only 36% CoNLL-F1 on Animal Farm).

**Goal**: To design a reliable automatic annotation pipeline and construct the first book-scale training and evaluation resource for coreference resolution.

**Key Insight**: Utilizing character lists to initialize coreference clusters, improving precision through LLM filtering, and then utilizing windowed CR models to expand coreference chains to pronouns and other mentions.

**Core Idea**: To achieve high-quality automatic coreference annotation at the book scale through a four-step pipeline: Character Linking → LLM Filtering → Window-level CR Expansion → Grouped-window Expansion.

## Method

### Overall Architecture

The BookCoref pipeline follows a four-step process: (1) character linking to initialize explicit mention clusters; (2) LLM filtering to eliminate erroneous links; (3) using a CR model within small windows to expand clusters to non-explicit mentions such as pronouns; (4) a grouping window step for a second expansion to improve recall.

### Key Designs

1. **Character Linking (Cluster Initialization)**: The ReLiK entity linking system is fine-tuned on LitBank to link character name mentions in the text to a predefined list of characters. Compared to simple pattern matching, this improves the F1 score from 29.2% to 44.5%.
2. **LLM Filtering (Cluster Refinement)**: Qwen2-7B is utilized to verify whether each mention is correctly associated with a character based on context, improving precision by +5.2% and mitigating error propagation.
3. **Windowed CR Expansion**: The book is divided into 1500-word windows. Maverick is applied within each window to expand character clusters (adding pronouns, noun phrases, etc.), and these clusters are then merged across windows based on character names.
4. **Grouped-Window Expansion**: Ten consecutive windows are merged into a group, and Maverick_xl is employed to perform a second expansion in a larger context, resolving missed coreferences across window boundaries.

### Evaluation Strategy

Standard CR metrics are used: MUC, $B^3$, $CEAF_{\phi_4}$, and CoNLL-F1. The pipeline and existing systems are evaluated on the manually annotated BookCoref_gold dataset (consisting of 3 books: *Animal Farm*, *Siddhartha*, and *Pride and Prejudice*).

## Key Experimental Results

### Pipeline Evaluation (BookCoref_gold)

| Pipeline Step | Character Linking F1 | CoNLL-F1 |
|---------|-----------|----------|
| Pattern Matching | 29.2 | 17.9 |
| Character Linking | 44.5 | 34.2 |
| + LLM Filtering | 43.5 | 33.9 |
| + Window CR | 84.7 | 77.7 |
| + Grouping Step | **86.3** | **80.5** |

### Comparison with Existing Systems (BookCoref_gold)

| System | CoNLL-F1 (Full Book) | CoNLL-F1 (Split Version) |
|------|----------------|-----------------|
| BookNLP | 42.2 | 50.6 |
| Longdoc | 46.6 | 61.2 |
| Dual cache | 42.5 | 64.8 |
| Maverick_xl | 41.2 | - |

### Key Findings

- The pipeline achieves a CoNLL-F1 of 80.5 and a MUC score of 93.3, which is close to the human inter-annotator agreement (96.1).
- Existing systems perform significantly worse under the full-book setting compared to the split version (with gaps of up to 20+ CoNLL-F1).
- BookCoref_silver contains 10.8M tokens and 968k mentions, exponentially exceeding the scale of OntoNotes.
- The average coreference chain span at the book level reaches an unprecedented 73,432 tokens.

## Highlights & Insights

- Sophisticated pipeline design: high-precision initialization followed by gradual expansion, effectively preventing error propagation.
- Utilizing LLMs as filters (rather than generators) is a highly practical paradigm.
- The two-stage paradigm of "character linking → CR expansion" can be generalized to other long-document NER/CR tasks.

## Limitations & Future Work

- Annotations are restricted to character coreference, excluding entities such as objects and locations.
- The gold annotations only cover 3 books, limiting the size of the evaluation set.
- Window boundaries in the pipeline may cause missed coreferences across windows (which is only partially mitigated by the grouping step).
- The dataset covers only English classical literature.

## Related Work & Insights

- Complementary to LitBank and MovieCoref, pushing coreference resolution to a true book scale.
- The methodology of the automatic annotation pipeline can be adapted to construct other long-document NLP resources.
- Incremental CR systems (e.g., Longdoc, Dual cache) require further optimization on BookCoref.
- The pipeline's strategy of "high-precision initialization followed by gradual expansion" serves as a valuable reference for other long-document annotation tasks.

## Additional Technical Details

- Base resources: Project Gutenberg full texts + Wikidata character names + LiSCU character profiles, covering 53 books with an average of 27 characters per book.
- Window size: 1500 words (close to Maverick's training configuration), with a grouping size of $G=10$.
- Annotation effort: 3 experts annotated for approximately 120 hours, covering 194,280 words.
- Inter-annotator agreement: MUC score of 96.1%, higher than LitBank (95.5%) and OntoNotes (83.0%).
- BookCoref_silver scale: 50 books, 10.8M tokens, 968k mentions, which is 6.75 times larger than OntoNotes.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The first book-scale CR benchmark with a novel and practical pipeline design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Detailed pipeline evaluation, though comparisons with existing systems are somewhat limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear exposition, with methods closely integrated with evaluation.
- Value: ⭐⭐⭐⭐⭐ Fills the resource gap in book-scale CR, possessing long-term research value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] End-to-End Dialog Neural Coreference Resolution: Balancing Efficiency and Accuracy in Large-Scale Systems](end-to-end_dialog_neural_coreference_resolution_balancing_efficiency_and_accurac.md)
- [\[ACL 2025\] Adapting Psycholinguistic Research for LLMs: Gender-Inclusive Language in a Coreference Context](adapting_psycholinguistic_research_for_llms_gender-inclusive_language_in_a_coref.md)
- [\[ACL 2025\] Disambiguate First, Parse Later: Generating Interpretations for Ambiguity Resolution in Semantic Parsing](disambiguate_first_parse_later_generating_interpretations_for_ambiguity_resoluti.md)
- [\[ACL 2025\] RISE: Reasoning Enhancement via Iterative Self-Exploration in Multi-hop Question Answering](rise_reasoning_enhancement_via_iterative_self-exploration_in_multi-hop_question_.md)
- [\[ICML 2025\] Cover Learning for Large-Scale Topology Representation](../../ICML2025/nlp_understanding/cover_learning_for_large-scale_topology_representation.md)

</div>

<!-- RELATED:END -->
