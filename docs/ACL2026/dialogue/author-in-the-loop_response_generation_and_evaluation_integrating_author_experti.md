---
title: >-
  [Paper Note] Author-in-the-Loop Response Generation and Evaluation: Integrating Author Expertise and Intent in Responses to Peer Review
description: >-
  [ACL 2026][Dialogue Systems][author rebuttal generation] This paper reframes academic rebuttal generation as an "author-in-the-loop" task, contributing the Re3Align dataset (3.4K papers…
tags:
  - "ACL 2026"
  - "Dialogue Systems"
  - "author rebuttal generation"
  - "peer review"
  - "human-in-the-loop"
  - "controllable text generation"
  - "evaluation framework"
date: 2026-05-08
content_hash: 900d71c1d1932609
---

# Author-in-the-Loop Response Generation and Evaluation: Integrating Author Expertise and Intent in Responses to Peer Review

**Conference**: ACL 2026
**arXiv**: [2602.11173](https://arxiv.org/abs/2602.11173)
**Code**: [https://github.com/UKPLab/acl2026-respgen-respeval](https://github.com/UKPLab/acl2026-respgen-respeval)
**Area**: Dialogue / Scientific Document Processing
**Keywords**: author rebuttal generation, peer review, human-in-the-loop, controllable text generation, evaluation framework

## TL;DR

This paper reframes academic rebuttal generation as an "author-in-the-loop" task, contributing the Re3Align dataset (3.4K papers, 440K sentence-level edit annotations, 15K review–rebuttal–revision triples), the REspGen controllable generation framework, and the REspEval evaluation suite comprising 20+ metrics. The framework is systematically validated across 5 state-of-the-art LLMs, demonstrating the effectiveness of author input, controllability, and evaluation-guided refinement.

## Background & Motivation

**Background**: Writing rebuttals is a critical and labor-intensive step in academic peer review. NLP-assisted automatic rebuttal generation (ARG) is an emerging yet underexplored research direction.

**Limitations of Prior Work**: (1) Existing ARG methods use only review comments as input, disregarding authors' domain expertise, proprietary information, and response strategies—yet many review concerns can only be addressed by the authors themselves (e.g., specific experimental design decisions, clarification of definitions). (2) No dataset provides fine-grained author signals—existing datasets lack sentence-level edit annotations, review–rebuttal paragraph alignments, or revision mappings. (3) Evaluation is limited to surface-level similarity metrics (ROUGE/BLEU), with no multidimensional assessment of controllability, input utilization, response quality, or discourse structure.

**Key Challenge**: Rebuttal writing inherently requires integrating author-exclusive signals (planned revisions, domain knowledge, response strategies), yet existing NLP methods treat it as a generic "review → rebuttal" text generation problem, producing responses that lack specificity and author-unique information.

**Goal**: (1) Formally define the "author-in-the-loop" ARG paradigm; (2) construct a large-scale triple-aligned dataset to support this paradigm; (3) provide a generation framework supporting flexible author input and multi-attribute control; (4) establish a comprehensive evaluation framework with 20+ metrics.

**Key Insight**: Paper revisions serve as a proxy for author signals—in conference settings, rebuttals describe planned revisions, and actual edits in revised manuscripts can retrospectively proxy author intent and expertise.

**Core Idea**: Sentence-level edits extracted from paper revisions serve as proxies for author-exclusive information. By constructing a triple-aligned dataset of review–rebuttal–edit correspondences, ARG models can leverage authors' actual revision intent to generate higher-quality rebuttals.

## Method

### Overall Architecture

Three components work in concert: (1) Re3Align constructs sentence-level triples from review–rebuttal–revision records via citation matching, a state-of-the-art edit analysis model, and a bidirectional alignment strategy; (2) REspGen takes review comments as the core input and optionally incorporates author edit signals, retrieved paper context, response plans, and length constraints, supporting evaluation-guided iterative refinement; (3) REspEval provides 20+ metrics across four dimensions: discourse, controllability, input utilization, and response quality.

### Key Designs

1. **Re3Align Triple Dataset Construction**:

    - **Function**: Provides the first large-scale dataset with review–rebuttal–edit alignment, supporting the author-in-the-loop paradigm.
    - **Mechanism**: Complete paper records are collected from EMNLP24 (679 papers) and PeerJ (2,715 papers). A three-step pipeline is applied: (a) a citation-matching algorithm extracts review–rebuttal paragraph pairs (16,071 pairs; 98% accuracy verified manually); (b) a state-of-the-art edit analysis model annotates 439,798 sentence-level edits (alignment F1 > 90%, intent classification 84.3 F1); (c) a bidirectional alignment strategy (review → edit + rebuttal → edit, using a fine-tuned LLM classifier with >90% accuracy) produces 15,521 triples.
    - **Design Motivation**: Active collection of author signals is ethically and practically infeasible; using revised paper versions as post-hoc proxies offers a practical and scalable alternative.

2. **REspGen Controllable Generation Framework**:

    - **Function**: Supports flexible author input configurations and multi-attribute rebuttal control.
    - **Mechanism**: Three layers of control are implemented: (a) **Response plan control**: review comments are classified into Criticism/Question/Request; each category is associated with 16 response action labels across 5 stance classes (Cooperative, Defensive, Hedging, Social, Other); authors can specify a response strategy sequence per review item. (b) **Length constraint**: an upper-bound word count can be set (in experiments, set to human rebuttal length + 50). (c) **Input configuration**: author edits can be provided at two granularities—as "edit strings" (rough ideas) or as "edit strings + paragraph context + section headings" (fine-grained revision localization); retrieval-reranking of v1 paper paragraphs is additionally supported.
    - **Design Motivation**: Rebuttal writing in practice requires control over tone, strategy, and length, yet prior ARG work has entirely lacked controllability research.

3. **REspEval Multidimensional Evaluation Suite**:

    - **Function**: Provides 20+ metrics for comprehensive evaluation of rebuttal generation quality.
    - **Mechanism**: Four dimensions are covered: (a) **Discourse analysis**: proportion of five stance types (%Coop, %Defe, %Hed, %Soc, %Other), ArgumentLoad, and transition flow. (b) **Controllability**: length compliance rate (%met + median diff) and plan fidelity (P/R/F1 + LCS-based Order Fidelity). (c) **Input utilization**: atomic fact verification-based Generated Fact Precision (GFP = proportion of generated facts supported by input) and Input Coverage Recall (ICR = proportion of author edit facts appearing in the rebuttal). (d) **Response quality**: GPT-5 review based on evaluation criteria, assessing Targeting (Targ), Specificity (Spec), and Convincingness (Conv) on a 5-point scale.
    - **Design Motivation**: ROUGE/BLEU measure only surface similarity and cannot capture whether a rebuttal genuinely addresses review concerns, integrates author information, or complies with plan constraints. Human validation (12 researchers, 1,365 judgments) yields agreement scores > 4.17/5 and Krippendorff $\alpha$ = 0.81–0.89.

### Loss & Training

REspGen is built on prompt-driven large language models and does not involve parameter training. Input configurations and attribute control are achieved through carefully designed prompt templates. Evaluation-guided iterative refinement feeds REspEval's returned metrics, rationales, and improvement suggestions—together with the original input and initial draft—back into REspGen to produce an improved rebuttal.

## Key Experimental Results

### Main Results

**Rebuttal quality comparison across LLMs and settings (GPT-4o and DeepSeek selected)**

| Setting | GFP %sup | ICR %sup | Targ | Spec | Conv |
|---|---|---|---|---|---|
| Human baseline | .458 | .200 | .788 | .575 | .575 |
| GPT-4o noAIx (no author input) | .443 | .033 | .842 | .508 | .554 |
| GPT-4o wAIx(S) | .689 | .668 | .826 | .638 | .654 |
| GPT-4o wAIx(+v1) | .781 | .432 | .847 | .721 | .717 |
| GPT-4o +Refine(planC) | .695 | — | .938 | .771 | .742 |
| DeepSeek noAIx | .412 | .046 | .779 | .433 | .496 |
| DeepSeek wAIx(+v1) | .738 | .452 | .861 | .692 | .700 |
| DeepSeek +Refine(planC) | .734 | — | .913 | .746 | .742 |

### Ablation Study

**Incremental effect of author input granularity on fact utilization (Phi-4 model)**

| Setting | GFP %sup ↑ | GFP %unsup ↓ | GFP %con | ICR %sup ↑ |
|---|---|---|---|---|
| noAIx (no author input) | .362 | .542 | .096 | .300 |
| wAIx edit string | .575 | .374 | .051 | .509 |
| + paragraph context | .577 | .364 | .059 | .470 |
| + v1 retrieval | .705 | .236 | .059 | .358 |

**Interaction effects of length and plan control (Llama-3.3)**

| Setting | lenC %met | planC F1 | Targ | Conv |
|---|---|---|---|---|
| +lenC only | 1.00 | — | .771 | .638 |
| +lenC & planC | 1.00 | .619 | .850 | .638 |
| +planC only | — | .486 | .892 | .671 |

### Key Findings

- Author input substantially improves factual precision (GFP %sup increases from .36–.44 to .58–.78), with a marked reduction in unsupported fact ratios.
- Evaluation-guided refinement effectively improves targeting (Targ from .85 to .94) and convincingness, but may reduce factual precision—revealing a quality–factuality trade-off.
- Simultaneously applying length and plan control introduces a quality–controllability trade-off: jointly constraining both attributes yields slightly lower quality than constraining either alone.
- ICR decreases when additional context is provided, suggesting that information overload prevents models from prioritizing core edit content.
- All models generate a large proportion of unsupported facts (>50%) without author input, confirming the necessity of the author-in-the-loop paradigm.

## Highlights & Insights

- The "author-in-the-loop" paradigm represents an essential reconceptualization of the ARG task—from generic generation to human–machine collaboration.
- Using paper revisions as a proxy for author signals is a methodologically elegant innovation that circumvents the ethical and practical barriers of real-time signal collection.
- The atomic fact verification-based GFP/ICR metrics in REspEval more meaningfully measure how well rebuttals leverage author information compared to ROUGE.
- The LCS-based design of the Order Fidelity metric is both concise and principled, and is generalizable to other sequence control evaluation scenarios.
- Table 1's three-dimensional comparison (data / generation / evaluation) of prior work clearly illustrates the systematic nature of the contributions.

## Limitations & Future Work

- There is an inherent gap between the proxy signal (paper edits) and actual author intent—not all revisions correspond to review concerns.
- Validation is limited to English academic text; other languages and domains are untested.
- Evaluation-guided refinement may lead to overfitting REspEval metrics rather than genuine quality improvement.
- Future work may explore interactive multi-turn refinement, user studies with actual authors, and more fine-grained author control interfaces.

## Related Work & Insights

- **vs. Jiu-Jitsu (2023)**: Provides only paragraph-level alignment with no sentence-level edit annotations, no author input, and evaluation limited to ROUGE/BERTScore.
- **vs. ReviewMT (2024)**: Provides only document-level alignment with evaluation limited to ROUGE/BLEU/METEOR.
- **vs. Re2 (2025)**: Provides only document-level alignment with basic similarity and quality metrics; no controllability research.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First systematic definition of the "author-in-the-loop" ARG paradigm, with dataset, framework, and evaluation forming an integrated whole.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Five LLMs, nine settings, 20+ metrics, and 12-person human validation—exceptionally comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with sufficient technical detail, though the extremely high information density raises the reading barrier.
- **Value**: ⭐⭐⭐⭐⭐ Substantially advances NLP-assisted academic writing; the dataset and tools carry high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Auto-PRE: An Automatic and Cost-Efficient Peer-Review Framework for Language Generation Evaluation](../../AAAI2026/dialogue/auto-pre_an_automatic_and_cost-efficient_peer-review_framework_for_language_gene.md)
- [\[ACL 2026\] Discourse Coherence and Response-Guided Context Rewriting for Multi-Party Dialogue Generation](discourse_coherence_and_response-guided_context_rewriting_for_multi-party_dialog.md)
- [\[ICLR 2026\] AQuA: Toward Strategic Response Generation for Ambiguous Visual Questions](../../ICLR2026/dialogue/aqua_toward_strategic_response_generation_for_ambiguous_visual_questions.md)
- [\[ACL 2026\] APEX-MEM: Agentic Semi-Structured Memory with Temporal Reasoning for Long-Term Conversational AI](apex-mem_agentic_semi-structured_memory_with_temporal_reasoning_for_long-term_co.md)
- [\[CVPR 2026\] Evolutionary Multimodal Reasoning via Hierarchical Semantic Representation for Intent Recognition](../../CVPR2026/dialogue/evolutionary_multimodal_reasoning_via_hierarchical_semantic_representation_for_i.md)

</div>

<!-- RELATED:END -->
