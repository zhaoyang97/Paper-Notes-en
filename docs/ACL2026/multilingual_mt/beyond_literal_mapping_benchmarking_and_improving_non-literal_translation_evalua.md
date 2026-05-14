---
title: >-
  [Paper Note] Beyond Literal Mapping: Benchmarking and Improving Non-Literal Translation Evaluation
description: >-
  [ACL 2026][Multilingual & Machine Translation][machine translation evaluation] This paper constructs MENT, a non-literal translation meta-evaluation dataset comprising 7,530 human-annotated instances…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "machine translation evaluation"
  - "non-literal translation"
  - "meta-evaluation benchmark"
  - "agentic evaluation framework"
  - "LLM-as-Judge"
date: 2026-05-08
content_hash: 6331057c0c523241
---

# Beyond Literal Mapping: Benchmarking and Improving Non-Literal Translation Evaluation

**Conference**: ACL 2026
**arXiv**: [2601.07338](https://arxiv.org/abs/2601.07338)
**Code**: [GitHub](https://github.com/BITHLP/RATE)
**Area**: Multilingual / MT Evaluation
**Keywords**: machine translation evaluation, non-literal translation, meta-evaluation benchmark, agentic evaluation framework, LLM-as-Judge

## TL;DR

This paper constructs MENT, a non-literal translation meta-evaluation dataset comprising 7,530 human-annotated instances, reveals the unreliability of traditional metrics and LLM-as-Judge approaches on non-literal translation evaluation, and proposes RATE, an agentic evaluation framework in which a reflective Core Agent dynamically invokes sub-agents to improve correlation with human judgments by 3.2+ points.

## Background & Motivation

**Background**: LLMs have greatly expanded the scope of machine translation applications, making non-literal translation scenarios such as social media and literary texts increasingly important. Translation quality evaluation is critical for MT system iteration and reinforcement learning reward signals.

**Limitations of Prior Work**: (1) Traditional metrics (BLEU, COMET) lack deep semantic understanding and diverge significantly from human judgments on non-literal translations—overrating literal but semantically incorrect translations while underrating idiomatic but non-literal ones; (2) LLM-as-Judge approaches are constrained by knowledge cutoffs (preventing evaluation of emerging internet slang) and scoring inconsistency; (3) Existing meta-evaluation datasets primarily cover formal domains such as news and Wikipedia, lacking coverage of non-literal translation.

**Key Challenge**: The core difficulty of non-literal translation evaluation lies in the fact that translation errors typically stem from misunderstanding holistic semantics (e.g., slang, cultural allusions) rather than isolated lexical errors, making metrics based on word-level matching or surface semantics inherently incapable of capturing such phenomena.

**Goal**: To systematically assess the reliability of MT metrics on non-literal translation and to propose a more accurate evaluation methodology.

**Key Insight**: The paper first constructs a large-scale non-literal translation meta-evaluation benchmark covering four domains (SNS, cross-cultural, poetry, and literary), then designs an agentic framework to address the limitations of LLM-based evaluation.

**Core Idea**: Reflective agentic evaluation—a Core Agent dynamically decides whether to invoke a Search Agent for background knowledge retrieval, an Evaluation Agent for scoring, or a Comparison Agent for score calibration.

## Method

### Overall Architecture

RATE centers on a reflective Core Agent that dynamically invokes three types of sub-agents according to evaluation needs: a Search Agent (retrieving external knowledge to compensate for knowledge cutoffs), an Evaluation Agent (scoring based on context), and a Comparison Agent (calibrating score consistency through multi-translation comparison).

### Key Designs

1. **MENT Meta-Evaluation Dataset**:

    - Function: Systematically evaluate the reliability of MT metrics on non-literal translation.
    - Mechanism: Covers 4 domains (SNS, cross-cultural, poetry, literary), 10 MT systems (from NLLB-3.3B to GPT-4o), and 7,530 human-annotated instances (SQM 5-point scale), each annotated by at least 2 professional translators.
    - Design Motivation: Existing datasets contain no more than 1,000 annotations and are limited to formal domains, making them insufficient for systematic validation of metrics on non-literal translation.

2. **Core Agent Reflective Reasoning**:

    - Function: Dynamically orchestrate the evaluation pipeline to address the limitations of static LLM evaluation.
    - Mechanism: After analyzing the source text and translation, the Core Agent reasons whether external knowledge is needed (invoking the Search Agent), how to construct evaluation instructions (passed to the Evaluation Agent), and whether score calibration is required (invoking the Comparison Agent).
    - Design Motivation: Different translation scenarios demand different evaluation strategies—SNS content requires retrieval of emerging slang meanings, while poetry requires understanding of prosody and imagery; static LLM evaluation cannot adapt flexibly.

3. **Search Agent Knowledge Retrieval**:

    - Function: Compensate for LLM knowledge cutoff limitations.
    - Mechanism: When the Core Agent determines that external knowledge is necessary, it formulates search queries to retrieve relevant background information (e.g., internet slang definitions, cultural allusion explanations), passing the retrieved content as context to the Evaluation Agent.
    - Design Motivation: LLMs have a training data cutoff and cannot accurately evaluate internet slang or cultural phenomena that emerged afterward.

### Loss & Training

RATE requires no training and operates as a zero-shot agentic framework based on LLMs. The evaluation protocol adopts the SQM 5-point scale. The Comparison Agent calibrates scores by comparing multiple translations of the same source text, mitigating score drift in LLM-based evaluation.

## Key Experimental Results

### Main Results

| Method Category | Representative Method | System + Segment-Level Aggregate Correlation |
|----------------|----------------------|---------------------------------------------|
| Traditional Metrics | BLEU, COMET | Baseline |
| LLM-as-Judge | GEMBA, AutoMQM | +X |
| **RATE (Ours)** | Core + Sub-agents | **+3.2+ points** |

### Ablation Study

| Configuration | Finding |
|---------------|---------|
| w/o Search Agent | Significant performance drop on SNS and cross-cultural domains |
| w/o Comparison Agent | Reduced scoring consistency |
| Different backbone LLMs | RATE remains consistently effective across different LLMs |
| General-domain test | RATE is also robust on general MT evaluation |

### Key Findings

- Traditional metrics are severely unreliable on non-literal translation: they overestimate literal mappings and underestimate idiomatic translations.
- Two key limitations of LLM-as-Judge: knowledge cutoffs (inability to evaluate new slang) and scoring inconsistency (same quality receiving different scores).
- RATE is effective not only in non-literal scenarios but also remains robust on general MT evaluation.
- Among the 10 MT systems evaluated, large-scale foundation models (GPT-4o, Gemini-2.5Pro) achieve the best performance.

## Highlights & Insights

- This work provides the first systematic evaluation of MT metric reliability on non-literal translation, filling an important gap in the field.
- The MENT dataset (7,530 annotations) substantially exceeds comparable prior work (<1,000) and covers four challenging domains.
- The agentic design of RATE is elegant—central reasoning combined with on-demand invocation of sub-capabilities mirrors the cognitive process of human evaluation.
- The paper identifies knowledge cutoff as a critical bottleneck for LLM-based evaluation, and the Search Agent offers a concise and effective mitigation strategy.

## Limitations & Future Work

- MENT covers only the Chinese–English language pair and does not extend to additional language pairs.
- RATE is dependent on the quality of the underlying search engine.
- The number and types of sub-agents could be further expanded (e.g., a style analysis agent).
- The SQM evaluation protocol may omit fine-grained error information.

## Related Work & Insights

- WMT Meta-Evaluation (Freitag et al., 2023; Moghe et al., 2025): formal-domain benchmarks.
- COMET / BLEURT: model-based metrics, still limited in non-literal scenarios.
- Agent-as-a-Judge (You et al., 2026): agentic evaluation paradigm.
- RATE's dynamic sub-agent invocation is generalizable to other evaluation scenarios requiring external knowledge.

## Rating

- Novelty: ⭐⭐⭐⭐ First work focused on non-literal translation evaluation, with an elegant agentic framework design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 7,530 human annotations, 10 MT systems, comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Dataset construction pipeline clearly described; case analyses are intuitive.
- Value: ⭐⭐⭐⭐ Directly informative for both MT evaluation research and practice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Reflective Translation: Improving Low-Resource Machine Translation via Structured Self-Reflection](../../NeurIPS2025/multilingual_mt/reflective_translation_improving_low-resource_machine_translation_via_structured.md)
- [\[ACL 2026\] Prosody as Supervision: Bridging the Non-Verbal–Verbal for Multilingual Speech Emotion Recognition](prosody_as_supervision_bridging_the_non-verbal--verbal_for_multilingual_speech_e.md)
- [\[ACL 2026\] LQM: Linguistically Motivated Multidimensional Quality Metrics for Machine Translation](lqm_linguistically_motivated_multidimensional_quality_metrics_for_machine_transl.md)
- [\[ICLR 2026\] ASSESS: A Semantic and Structural Evaluation Framework for Statement Similarity](../../ICLR2026/multilingual_mt/assess_a_semantic_and_structural_evaluation_framework_for_statement_similarity.md)
- [\[ACL 2026\] Just Use XML: Revisiting Joint Translation and Label Projection](just_use_xml_revisiting_joint_translation_and_label_projection.md)

</div>

<!-- RELATED:END -->
