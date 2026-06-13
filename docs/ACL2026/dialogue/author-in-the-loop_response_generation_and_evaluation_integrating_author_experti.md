---
title: >-
  [Paper Note] Author-in-the-Loop Response Generation and Evaluation: Integrating Author Expertise and Intent in Responses to Peer Review
description: >-
  [ACL 2026][Dialogue Systems][Author response generation] This paper redefines academic paper author response (rebuttal) generation as an "author-in-the-loop" task. The authors propose the Re3Align dataset (3.4K papers…
tags:
  - "ACL 2026"
  - "Dialogue Systems"
  - "Author response generation"
  - "Peer review"
  - "Human-in-the-loop"
  - "Controllable text generation"
  - "Evaluation framework"
date: 2026-05-08
content_hash: f2977e49f9443016
---

# Author-in-the-Loop Response Generation and Evaluation: Integrating Author Expertise and Intent in Responses to Peer Review

**Conference**: ACL 2026  
**arXiv**: [2602.11173](https://arxiv.org/abs/2602.11173)  
**Code**: [https://github.com/UKPLab/acl2026-respgen-respeval](https://github.com/UKPLab/acl2026-respgen-respeval)  
**Area**: Dialogue/Scientific Document Processing  
**Keywords**: Author response generation, Peer review, Human-in-the-loop, Controllable text generation, Evaluation framework

## TL;DR

This paper redefines academic paper author response (rebuttal) generation as an "author-in-the-loop" task. The authors propose the Re3Align dataset (3.4K papers, 440K sentence-level edit annotations, 15K review-response-revision triplets), the REspGen controllable generation framework, and the REspEval evaluation suite with 20+ metrics. The effectiveness of author input, controllability, and evaluation-guided refinement is systematically validated across five SOTA LLMs.

## Background & Motivation

**Background**: Writing author responses (rebuttals) is a critical step in academic peer review requiring significant effort. NLP-assisted automated response generation (ARG) is an emerging but under-explored research direction.

**Limitations of Prior Work**: (1) Existing ARG work only uses review comments as input, ignoring the author's domain expertise, unique information, and response strategies—yet in practice, many reviewer concerns can only be addressed by the author (e.g., specific experimental designs, clarifying definitions); (2) Lack of datasets providing fine-grained author signals—existing datasets lack sentence-level edit annotations, review-response paragraph alignment, and revision mappings; (3) Evaluation is limited to surface similarity metrics (ROUGE/BLEU), lacking multi-dimensional assessment of controllability, input utilization, response quality, and discourse structure.

**Key Challenge**: Author response writing inherently requires integrating author-specific signals (revision plans, domain knowledge, response strategies), but existing NLP methods treat it as a generic "review $\rightarrow$ response" text generation problem, resulting in responses that lack specific details and author-unique information.

**Goal**: (1) Formally define the "author-in-the-loop" ARG paradigm; (2) Construct a large-scale triplet dataset supporting this paradigm; (3) Provide a generation framework supporting flexible author input and multi-attribute control; (4) Establish a comprehensive evaluation system with 20+ metrics.

**Key Insight**: Utilize paper revisions as a proxy for author signals—in conference scenarios, responses describe planned modifications, and actual edits in the revised paper can retrospectively proxy the author's intent and expertise.

**Core Idea**: By using sentence-level edits from paper revisions as a proxy for author-exclusive information, the authors construct a review-response-revision aligned triplet dataset. This allows ARG models to leverage actual author revision intents to generate high-quality responses.

## Method

### Overall Architecture

Three components work in synergy: (1) The Re3Align dataset extracts sentence-level triplets from review-response-revision records using citation matching, a SOTA revision analysis model, and a bidirectional alignment strategy; (2) REspGen uses review comments as core input with optional access to author edit signals, paper context retrieval, response plans, and length constraints, supporting evaluation-guided iterative refinement; (3) REspEval provides comprehensive evaluation across four dimensions: discourse, controllability, input utilization, and response quality via 20+ metrics.

### Key Designs

1.  **Re3Align Triplet Dataset Construction**:
    *   **Function**: Provides the first large-scale dataset containing review-response-edit alignments to support the "author-in-the-loop" paradigm.
    *   **Mechanism**: Complete paper records were collected from EMNLP24 (679 papers) and PeerJ (2,715 papers). A three-step process was used: (a) Extracting review-response paragraph pairs via citation matching (16,071 pairs, 98% manual accuracy); (b) Annotating 439,798 sentence-level edits using a SOTA revision analysis model (alignment F1 > 90%, intent classification 84.3 F1); (c) Generating 15,521 triplets via a bidirectional alignment strategy (review $\rightarrow$ edit + response $\rightarrow$ edit, using a fine-tuned LLM classifier with >90% accuracy).
    *   **Design Motivation**: Active collection of author signals is ethically and practically infeasible; utilizing paper revisions as a post-hoc proxy is a practical and scalable alternative.

2.  **REspGen Controllable Generation Framework**:
    *   **Function**: Supports flexible author input configurations and multi-attribute response control.
    *   **Mechanism**: Includes a three-layer control mechanism: (a) **Response Plan Control**: Reviews are categorized into Criticism/Question/Request, each associated with 16 response action labels (across 5 stances: Cooperative, Defensive, Hedging, Social, Other), allowing authors to specify strategy sequences; (b) **Length Constraints**: Supports setting an upper word-count bound (set to human length + 50 in experiments); (c) **Input Configuration**: Author edits can be provided as "edit strings" (raw ideas) or "edit strings + paragraph context + section headings" (precise location), with additional support for RAG-based v1 paper paragraph retrieval.
    *   **Design Motivation**: In practice, authors need to control tone, strategy, and length; previous ARG work completely lacked controllability research.

3.  **REspEval Multi-dimensional Evaluation Suite**:
    *   **Function**: Provides 20+ metrics for comprehensive assessment of author response quality.
    *   **Mechanism**: Four dimensions: (a) **Discourse Analysis**: Extracts 5 stance proportions (%Coop, %Defe, %Hed, %Soc, %Other), ArgumentLoad, and transition flows; (b) **Controllability**: Length compliance (%met + median diff) and plan fidelity (P/R/F1 + LCS-based Order Fidelity); (c) **Input Utilization**: Factual precision based on atomic fact-checking (GFP = proportion of generated facts supported by input) and input coverage recall (ICR = proportion of author edit facts appearing in the response); (d) **Response Quality**: GPT-5 review based on criteria assessing targetedness (Targ), specificity (Spec), and persuasiveness (Conv) on a 5-point scale.
    *   **Design Motivation**: ROUGE/BLEU only measure surface similarity and fail to capture if a response truly addresses reviewer concerns, integrates author info, or follows plan constraints. Manual validation (12 researchers, 1,365 judgments) showed consistency scores > 4.17/5 and Krippendorff α = 0.81-0.89.

### Loss & Training

REspGen is based on prompt-driven LLMs and does not involve model parameter training. Input configurations and attribute controls are implemented through carefully designed prompt templates. Evaluation-guided iterative refinement feeds back REspEval metrics, rationales, and suggestions along with original inputs to REspGen to generate improved responses.

## Key Experimental Results

### Main Results

**Comparison of response quality under different LLMs and settings (Selected GPT-4o and DeepSeek)**

| Setting | GFP %sup | ICR %sup | Targ | Spec | Conv |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Human baseline | .458 | .200 | .788 | .575 | .575 |
| GPT-4o noAIx (No author input) | .443 | .033 | .842 | .508 | .554 |
| GPT-4o wAIx(S) | .689 | .668 | .826 | .638 | .654 |
| GPT-4o wAIx(+v1) | .781 | .432 | .847 | .721 | .717 |
| GPT-4o +Refine(planC) | .695 | — | .938 | .771 | .742 |
| DeepSeek noAIx | .412 | .046 | .779 | .433 | .496 |
| DeepSeek wAIx(+v1) | .738 | .452 | .861 | .692 | .700 |
| DeepSeek +Refine(planC) | .734 | — | .913 | .746 | .742 |

### Ablation Study

**Incremental impact of author input granularity on factual utilization (Phi-4 model)**

| Setting | GFP %sup ↑ | GFP %unsup ↓ | GFP %con | ICR %sup ↑ |
| :--- | :--- | :--- | :--- | :--- |
| noAIx (No author input) | .362 | .542 | .096 | .300 |
| wAIx Edit string | .575 | .374 | .051 | .509 |
| +Paragraph context | .577 | .364 | .059 | .470 |
| +v1 Retrieval | .705 | .236 | .059 | .358 |

**Interaction effects of length and plan control (Llama-3.3)**

| Setting | lenC %met | planC F1 | Targ | Conv |
| :--- | :--- | :--- | :--- | :--- |
| +lenC only | 1.00 | — | .771 | .638 |
| +lenC & planC | 1.00 | .619 | .850 | .638 |
| +planC only | — | .486 | .892 | .671 |

### Key Findings

*   Author input significantly improves factual precision (GFP %sup increases from .36-.44 to .58-.78), and the proportion of unsupported facts decreases sharply.
*   Evaluation-guided refinement effectively improves targetedness (Targ increases from .85 to .94) and persuasiveness, though it may decrease factual precision—revealing a quality-factuality trade-off.
*   A quality-controllability trade-off exists when applying both length and plan controls—quality is slightly lower when controlling two attributes than just one.
*   ICR actually decreases after adding more context, suggesting information overload prevents models from prioritizing core edit content.
*   All models generate a high volume of unsupported facts (>50%) when no author input is provided, confirming the necessity of the "author-in-the-loop" approach.

## Highlights & Insights

*   The proposal of the "author-in-the-loop" paradigm is a fundamental redefinition of the ARG task—moving from generic generation to human-AI collaboration.
*   Using paper revisions as a proxy for author signals is a clever methodological innovation that bypasses ethical and practical barriers of real-time collection.
*   The GFP/ICR metrics in REspEval based on atomic fact-checking measure utilization of author information more meaningfully than ROUGE.
*   The Order Fidelity metric designed based on LCS is both elegant and reasonable, and can be generalized to other sequence control evaluation scenarios.
*   Table 1 clearly demonstrates the systematic nature of the contributions by comparing gaps in data, generation, and evaluation relative to prior work.

## Limitations & Future Work

*   Intrinsic gap between proxy signals (paper edits) and actual author intent—not all revisions correspond to reviewer concerns.
*   Validation was only performed on English academic texts; other languages and domains remain untested.
*   Evaluation-guided refinement might lead to overfitting REspEval metrics rather than genuine quality improvement.
*   Future work could explore interactive multi-turn refinement, user studies with actual authors, and more fine-grained author control interfaces.

## Related Work & Insights

*   **vs Jiu-Jitsu (2023)**: Only paragraph-level alignment, no sentence-level edit annotations, no author input, evaluation limited to ROUGE/BERTScore.
*   **vs ReviewMT (2024)**: Only document-level alignment, evaluation limited to ROUGE/BLEU/METEOR.
*   **vs Re2 (2025)**: Only document-level alignment and basic similarity/quality metrics, no study on controllability.

## Rating

*   Novelty: ⭐⭐⭐⭐⭐ First systematic definition of "author-in-the-loop" ARG; trifold contribution of dataset, framework, and evaluation.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive with 5 LLMs, 9 settings, 20+ metrics, and 12-person manual validation.
*   Writing Quality: ⭐⭐⭐⭐ Complete structure with sufficient technical detail, though high information density makes for a demanding read.
*   Value: ⭐⭐⭐⭐⭐ Highly significant for advancing NLP-assisted academic writing; high practical value for both datasets and tools.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Auto-PRE: An Automatic and Cost-Efficient Peer-Review Framework for Language Generation Evaluation](../../AAAI2026/dialogue/auto-pre_an_automatic_and_cost-efficient_peer-review_framework_for_language_gene.md)
- [\[ACL 2026\] Discourse Coherence and Response-Guided Context Rewriting for Multi-Party Dialogue Generation](discourse_coherence_and_response-guided_context_rewriting_for_multi-party_dialog.md)
- [\[ICLR 2026\] AQuA: Toward Strategic Response Generation for Ambiguous Visual Questions](../../ICLR2026/dialogue/aqua_toward_strategic_response_generation_for_ambiguous_visual_questions.md)
- [\[ACL 2026\] Codebook-Injected Dialogue Segmentation for Multi-Utterance Constructs Annotation: LLM-Assisted and Gold-Label-Free Evaluation](codebook-injected_dialogue_segmentation_for_multi-utterance_constructs_annotatio.md)
- [\[CVPR 2026\] Evolutionary Multimodal Reasoning via Hierarchical Semantic Representation for Intent Recognition](../../CVPR2026/dialogue/evolutionary_multimodal_reasoning_via_hierarchical_semantic_representation_for_i.md)

</div>

<!-- RELATED:END -->
