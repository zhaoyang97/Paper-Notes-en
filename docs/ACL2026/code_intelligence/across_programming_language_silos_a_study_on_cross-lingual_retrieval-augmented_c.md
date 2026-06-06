---
title: >-
  [Paper Note] Across Programming Language Silos: A Study on Cross-Lingual Retrieval-Augmented Code Generation
description: >-
  [ACL 2026][Code Intelligence][cross-lingual code generation] This paper presents the first systematic study of cross-programming-language retrieval-augmented code generation (RACG)…
tags:
  - "ACL 2026"
  - "Code Intelligence"
  - "cross-lingual code generation"
  - "retrieval-augmented generation"
  - "knowledge transfer"
  - "multilingual programming"
  - "code retrieval"
date: 2026-05-08
content_hash: 1af2ddb8a026d47e
---

# Across Programming Language Silos: A Study on Cross-Lingual Retrieval-Augmented Code Generation

**Conference**: ACL 2026
**arXiv**: [2506.03535](https://arxiv.org/abs/2506.03535)  
**Code**: [GitHub](https://github.com/icip-cas/Cross-Lingual-RACG)  
**Area**: Code Intelligence / Cross-Lingual Code Generation
**Keywords**: cross-lingual code generation, retrieval-augmented generation, knowledge transfer, multilingual programming, code retrieval

## TL;DR

This paper presents the first systematic study of cross-programming-language retrieval-augmented code generation (RACG), constructing a 14K-instance dataset spanning 13 programming languages, and reveals the asymmetry of cross-lingual knowledge transfer and its relationship to language family relatedness and pretraining data diversity.

## Background & Motivation

**Background**: Retrieval-augmented code generation (RACG) enhances LLM code generation by retrieving relevant code snippets, but existing research has primarily focused on monolingual settings such as Python and Java.

**Limitations of Prior Work**: Code knowledge is highly unevenly distributed across programming languages — Python benefits from rich documentation and community resources, whereas niche languages such as Scala suffer from severe resource scarcity. Enterprise technology stack migrations also generate substantial demand for cross-lingual code conversion.

**Key Challenge**: Can RACG effectively transfer code knowledge from one programming language to another? Is such transfer equally effective across all language pairs?

**Goal**: To systematically investigate the mechanisms of cross-programming-language knowledge transfer in RACG, addressing three key research questions.

**Key Insight**: Three retrieval experimental settings are designed (oracle injection, practical retrieval, and retrieval without natural language), enabling controlled-variable analysis of cross-lingual transfer effectiveness.

**Core Idea**: Cross-lingual code knowledge transfer is feasible but asymmetric; its effectiveness depends on the linguistic relatedness of the language pair and the diversity of the LLM's pretraining corpus.

## Method

### Overall Architecture

A large-scale dataset of approximately 14K instances covering 13 programming languages is constructed, comprising NL prompts, verified reference solutions, and executable test cases. Systematic evaluation is conducted across three retrieval settings and five code LLMs.

### Key Designs

1. **Three Retrieval Experimental Settings**:

    - Function: Evaluate cross-lingual knowledge transfer from different perspectives.
    - Mechanism: (1) *Golden Solution Document* — oracle retrieval simulates ideal conditions and measures the upper bound of cross-lingual transfer; (2) *Top-k Retrieved Documents* — end-to-end evaluation of the complete RACG pipeline; (3) *Top-k without NL* — removes natural language descriptions to simulate real-world pure code snippet scenarios.
    - Design Motivation: By controlling variables, the retrieval and generation stages are decoupled to identify the bottleneck of cross-lingual transfer.

2. **Large-Scale Multilingual Code Dataset**:

    - Function: Provides a unified evaluation benchmark across 13 programming languages.
    - Mechanism: Each instance contains an NL description, a reference solution, and test cases, covering 13 languages including C++, Go, Java, JavaScript, Python, and Rust.
    - Design Motivation: Existing datasets cover only 2–5 languages, which is insufficient to support large-scale cross-lingual research.

3. **Multilingual vs. Python-Specialized LLM Comparison**:

    - Function: Reveals the impact of pretraining diversity on cross-lingual transfer capability.
    - Mechanism: Multilingual LLMs (CodeLlama, DeepSeek-Coder, Qwen2.5-Coder) are compared against Python-specialized LLMs (Phi-1, Phi-1.5) on cross-lingual RACG performance.
    - Design Motivation: Distinguishes the source of cross-lingual transfer ability — whether it stems from model architecture or the diversity of pretraining data.

### Loss & Training

This paper is an empirical study and involves no model training. Greedy decoding (temperature = 0.0) is used to ensure reproducibility, and the evaluation metric is Pass@1.

## Key Experimental Results

### Main Results (Oracle Injection, Averaged over Multilingual LLMs)

| Source → Target | C++ | Go | Java | JS | Python | Avg. Gain |
|----------------|------|-----|------|-----|--------|-----------|
| C++ | — | +4.47 | +20.33 | +18.90 | +15.04 | +14.68 |
| Go | +9.15 | — | — | — | — | — |
| Baseline (no retrieval) | 54.27 | 42.68 | 61.79 | 58.33 | 59.35 | 55.28 |

### Ablation Study

| Configuration | Key Metric | Notes |
|--------------|------------|-------|
| Remove NL information | Only marginal performance drop | Code retrievers do not heavily rely on natural language |
| Python-specialized LLM | Poor cross-lingual transfer | Pretraining diversity is critical for cross-lingual transfer |
| Code-specialized retriever | Significantly outperforms general retrieval | Better bridges NL intent and code semantics |

### Key Findings
- Cross-lingual knowledge transfer is non-trivial even under oracle conditions, indicating that a cross-lingual gap exists in the generation stage itself.
- Transfer effectiveness is asymmetric and correlates with the syntactic relatedness of language pairs (e.g., Java→JavaScript yields better results than Java→Go).
- Python-specialized LLMs are nearly unable to leverage cross-lingual context, underscoring the importance of pretraining diversity.
- Removing NL leads to only a marginal drop in retrieval performance, suggesting that code semantics alone are sufficient to support retrieval.

## Highlights & Insights
- This work is the first to extend the concept of "cross-lingual" from natural languages to programming languages in the RACG setting, opening a new research direction.
- The experimental design is rigorous: the three retrieval settings form a gradient from ideal to realistic conditions, clearly illuminating the transfer mechanism.
- The finding that Python-specialized LLMs cannot transfer across programming languages carries important implications for model training strategies.

## Limitations & Future Work
- Only LLMs of approximately 7B parameters are evaluated; larger models may exhibit different cross-lingual transfer capabilities.
- Dataset construction relies on translation from existing benchmarks, which may introduce bias.
- The effect of fine-tuning on cross-lingual transfer ability is not explored.
- Future work could investigate optimization of cross-lingual retrieval strategies and mixed-language retrieval.

## Related Work & Insights
- **vs. Monolingual RACG**: This work reveals unique challenges in the cross-lingual setting — asymmetric transfer and the role of language family relatedness.
- **vs. Code Translation**: RACG does not perform direct translation but rather leverages source-language knowledge to augment target-language generation.
- **vs. Multilingual NLP**: The "cross-lingual" phenomenon in programming languages shares similar mechanisms with natural languages (relatedness influences transfer) but also exhibits unique characteristics.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic study of cross-programming-language RACG
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Large-scale experiments across 13 languages × 5 models × 3 settings
- Writing Quality: ⭐⭐⭐⭐ Three RQs provide a clear organizational structure
- Value: ⭐⭐⭐⭐ Provides empirical guidance for multilingual code tool design

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Inference-Time Safety for Code LLMs via Retrieval-Augmented Revision](../../ICLR2026/code_intelligence/inference-time_safety_for_code_llms_via_retrieval-augmented_revision.md)
- [\[AAAI 2026\] SPAN: Benchmarking and Improving Cross-Calendar Temporal Reasoning of Large Language Models](../../AAAI2026/code_intelligence/span_benchmarking_and_improving_cross-calendar_temporal_reasoning_of_large_langu.md)
- [\[ACL 2026\] CollabCoder: Plan-Code Co-Evolution via Collaborative Decision-Making for Efficient Code Generation](collabcoder_plan-code_co-evolution_via_collaborative_decision-making_for_efficie.md)
- [\[ACL 2026\] CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment](coderl_improving_code_generation_via_reinforcement_with_execution_semantics_alig.md)
- [\[ACL 2026\] DeepGuard: Secure Code Generation via Multi-Layer Semantic Aggregation](deepguard_secure_code_generation_via_multi-layer_semantic_aggregation.md)

</div>

<!-- RELATED:END -->
