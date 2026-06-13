---
title: >-
  [Paper Note] Beyond Literal Mapping: Benchmarking and Improving Non-Literal Translation Evaluation
description: >-
  [ACL 2026][Multilingual & Machine Translation][Machine Translation Evaluation] Ours constructs a non-literal translation meta-evaluation dataset MENT (7,530 human annotations)…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "Machine Translation Evaluation"
  - "Non-Literal Translation"
  - "Meta-Evaluation Benchmark"
  - "Agent Evaluation Framework"
  - "LLM-as-Judge"
date: 2026-05-08
content_hash: 21fbb44bc582507b
---

# Beyond Literal Mapping: Benchmarking and Improving Non-Literal Translation Evaluation

**Conference**: ACL 2026  
**arXiv**: [2601.07338](https://arxiv.org/abs/2601.07338)  
**Code**: [GitHub](https://github.com/BITHLP/RATE)  
**Area**: Multilingual / MT Evaluation  
**Keywords**: Machine Translation Evaluation, Non-Literal Translation, Meta-Evaluation Benchmark, Agent Evaluation Framework, LLM-as-Judge

## TL;DR

Ours constructs a non-literal translation meta-evaluation dataset MENT (7,530 human annotations), revealing the unreliability of traditional metrics and LLM-as-Judge in non-literal translation evaluation, and proposes the RATE agentic evaluation framework. By using a core agent for reflection and dynamic invocation of sub-agents, RATE improves correlation with human judgment by 3.2+ points.

## Background & Motivation

**Background**: LLMs have significantly expanded the application scope of machine translation (MT), making non-literal translation scenarios such as social media and literature increasingly important. Translation quality evaluation is critical for the iteration of MT systems and reinforcement learning reward signals.

**Limitations of Prior Work**: (1) Traditional metrics (BLEU, COMET) lack deep semantic understanding and are severely decoupled from human judgment in non-literal translation—overestimating literal but semantically incorrect translations and underestimating idiomatic but non-literal ones; (2) LLM-as-Judge is affected by knowledge cutoff (unable to evaluate emerging internet slang) and scoring inconsistency; (3) Existing meta-evaluation datasets primarily cover formal domains like news/Wikipedia, lacking non-literal translation coverage.

**Key Challenge**: The core challenge of non-literal translation is that errors often stem from a misunderstanding of global semantics (e.g., slang, cultural allusions) rather than isolated lexical errors. Traditional metrics based on word-level matching or surface semantics are inherently unable to capture such issues.

**Goal**: Systematically evaluate the reliability of MT metrics on non-literal translation and propose more accurate evaluation methods.

**Key Insight**: First construct a large-scale non-literal translation meta-evaluation benchmark (covering four domains: SNS, cross-cultural, poetry, and literature), then design an agentic framework to address the limitations of LLM evaluation.

**Core Idea**: Reflective agentic evaluation—a Core Agent dynamically decides whether to invoke a Search Agent for background knowledge retrieval, an Evaluation Agent for scoring, or a Comparison Agent for score calibration.

## Method

### Overall Architecture

RATE centers on a reflective Core Agent. Based on the needs of the translation evaluation, it dynamically invokes three sub-agents: Search Agent (retrieves external knowledge to bridge the knowledge cutoff), Evaluation Agent (scores based on context), and Comparison Agent (calibrates score consistency through multi-translation comparison).

### Key Designs

1.  **MENT Meta-Evaluation Dataset**:
    - **Function**: Systematically evaluates the reliability of MT metrics on non-literal translation.
    - **Mechanism**: Covers 4 domains (SNS, cross-cultural, poetry, and literature), 10 MT systems (from NLLB-3.3B to GPT-4o), and includes 7,530 human annotations (SQM 5-point scale), with at least 2 professional translators per entry.
    - **Design Motivation**: Existing datasets have fewer than 1,000 annotations and are limited to formal domains, which cannot support systematic validation of metrics for non-literal translation.

2.  **Reflective Reasoning of Core Agent**:
    - **Function**: Provides dynamic decision-making in the evaluation process to address the limitations of static LLM evaluation.
    - **Mechanism**: After analyzing the source text and translation, the Core Agent reasons whether external knowledge is needed (invoking Search Agent), how to construct evaluation instructions (passing them to the Evaluation Agent), and whether score calibration is required (invoking Comparison Agent).
    - **Design Motivation**: Different translation scenarios require different evaluation strategies—SNS needs retrieval of emerging slang meanings, while poetry requires understanding of rhythm and imagery. Static LLM evaluation cannot adapt flexibly.

3.  **Search Agent Knowledge Retrieval**:
    - **Function**: Mitigates LLM knowledge cutoff limitations.
    - **Mechanism**: When the Core Agent determines external knowledge is needed, it constructs search queries to retrieve background information (e.g., internet slang meanings, cultural allusion explanations) and passes the retrieval results as context to the Evaluation Agent.
    - **Design Motivation**: LLM training data has cutoff dates, preventing accurate evaluation of internet slang or cultural phenomena appearing after the cutoff.

### Loss & Training

RATE is a training-free, zero-shot LLM agentic framework. The evaluation protocol uses an SQM 5-point scale. The Comparison Agent calibrates scores by comparing multiple translations of the same source text, alleviating the score drift problem in LLM evaluation.

## Key Experimental Results

### Main Results

| Method Category | Representative Method | Integrated Correlation (System + Segment) |
| :--- | :--- | :--- |
| Traditional Metrics | BLEU, COMET | Baseline |
| LLM-as-Judge | GEMBA, AutoMQM | +X |
| **RATE (Ours)** | Core + Sub-agents | **+3.2+ points** |

### Ablation Study

| Configuration | Findings |
| :--- | :--- |
| No Search Agent | Significant performance drop in SNS and cross-cultural domains |
| No Comparison Agent | Reduced score consistency |
| Different Backbone LLMs | RATE is consistently effective across different LLMs |
| General Domain Test | RATE is also robust in general MT evaluation |

### Key Findings

- Traditional metrics are severely inaccurate for non-literal translation: they overestimate literal mappings and underestimate idiomatic translations.
- LLM-as-Judge has two major limitations: knowledge cutoff (unable to evaluate new slang) and score inconsistency (assigning different scores for the same quality).
- RATE is not only effective in non-literal scenarios but also maintains robustness in general MT evaluation.
- Among 10 MT systems, large-scale foundation models (GPT-4o, Gemini-1.5-Pro) perform the best.

## Highlights & Insights

- First systematic evaluation of the reliability of MT metrics for non-literal translation, filling a significant gap.
- MENT dataset scale (7,530 annotations) far exceeds similar works (<1,000) and covers 4 challenging domains.
- The agentic design of RATE is elegant—core reasoning combined with on-demand sub-capability invocation aligns with the cognitive process of human evaluation.
- Identified knowledge cutoff as a key bottleneck for LLM evaluation, and the Search Agent provides a concise and effective mitigation.

## Limitations & Future Work

- MENT only covers Chinese-English and has not yet been extended to more language pairs.
- RATE depends on the quality of the search engine.
- The number and types of sub-agents could be further expanded (e.g., style analysis agents).
- The SQM evaluation protocol might miss fine-grained error information.

## Related Work & Insights

- WMT Meta-evaluation (Freitag et al., 2023; Moghe et al., 2025): Benchmarks for formal domains.
- COMET / BLEURT: Model-based metrics, but still limited in non-literal scenarios.
- Agent-as-a-Judge (You et al., 2026): Agentic evaluation paradigm.
- The dynamic sub-agent invocation of RATE can be generalized to other evaluation scenarios requiring external knowledge.

## Rating

- Novelty: ⭐⭐⭐⭐ First focus on non-literal translation evaluation, clever agentic framework design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 7,530 human annotations, 10 MT systems, comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear data construction process, intuitive case analysis.
- Value: ⭐⭐⭐⭐ Direct guidance for MT evaluation research and practice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] XQ-MEval: A Dataset with Cross-lingual Parallel Quality for Benchmarking Translation Metrics](xq-meval_a_dataset_with_cross-lingual_parallel_quality_for_benchmarking_translat.md)
- [\[ACL 2026\] PEAR: Pairwise Evaluation for Automatic Relative Scoring in Machine Translation](pear_pairwise_evaluation_for_automatic_relative_scoring_in_machine_translation.md)
- [\[ACL 2026\] Prosody as Supervision: Bridging the Non-Verbal–Verbal for Multilingual Speech Emotion Recognition](prosody_as_supervision_bridging_the_non-verbal--verbal_for_multilingual_speech_e.md)
- [\[NeurIPS 2025\] Reflective Translation: Improving Low-Resource Machine Translation via Structured Self-Reflection](../../NeurIPS2025/multilingual_mt/reflective_translation_improving_low-resource_machine_translation_via_structured.md)
- [\[ICLR 2026\] ASSESS: A Semantic and Structural Evaluation Framework for Statement Similarity](../../ICLR2026/multilingual_mt/assess_a_semantic_and_structural_evaluation_framework_for_statement_similarity.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ACL 2026\] XQ-MEval: A Dataset with Cross-lingual Parallel Quality for Benchmarking Translation Metrics](xq-meval_a_dataset_with_cross-lingual_parallel_quality_for_benchmarking_translat.md)
- [\[ACL 2025\] Beyond N-Grams: Rethinking Evaluation Metrics and Strategies for Multilingual Abstractive Summarization](../../ACL2025/multilingual_mt/beyond_n-grams_rethinking_evaluation_metrics_and_strategies_for_multilingual_abs.md)
- [\[ACL 2026\] Prosody as Supervision: Bridging the Non-Verbal–Verbal for Multilingual Speech Emotion Recognition](prosody_as_supervision_bridging_the_non-verbal--verbal_for_multilingual_speech_e.md)
- [\[ACL 2026\] PEAR: Pairwise Evaluation for Automatic Relative Scoring in Machine Translation](pear_pairwise_evaluation_for_automatic_relative_scoring_in_machine_translation.md)
- [\[ACL 2025\] Accessible Machine Translation Evaluation For Low-Resource Languages](../../ACL2025/multilingual_mt/accessible_machine_translation_evaluation_for_low-resource_languages.md)

</div>

<!-- RELATED:END -->
