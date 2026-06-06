---
title: >-
  [Paper Note] Across Programming Language Silos: A Study on Cross-Lingual Retrieval-Augmented Code Generation
description: >-
  [ACL 2026][Code Intelligence][Cross-Lingual Code Generation] The first systematic study of cross-programming-language Retrieval-Augmented Code Generation (RACG)…
tags:
  - "ACL 2026"
  - "Code Intelligence"
  - "Cross-Lingual Code Generation"
  - "Retrieval-Augmented Generation"
  - "Knowledge Transfer"
  - "Multilingual Programming"
  - "Code Retrieval"
date: 2026-05-08
content_hash: 07ddb40190b984a5
---

# Across Programming Language Silos: A Study on Cross-Lingual Retrieval-Augmented Code Generation

**Conference**: ACL 2026  
**arXiv**: [2506.03535](https://arxiv.org/abs/2506.03535)  
**Code**: [GitHub](https://github.com/icip-cas/Cross-Lingual-RACG)  
**Area**: Code Intelligence / Cross-Lingual Code Generation  
**Keywords**: Cross-Lingual Code Generation, Retrieval-Augmented Generation, Knowledge Transfer, Multilingual Programming, Code Retrieval

## TL;DR

The first systematic study of cross-programming-language Retrieval-Augmented Code Generation (RACG), constructing a 14K-instance dataset covering 13 programming languages, revealing the asymmetry of cross-lingual knowledge transfer and its relationship with language kinship and pre-training diversity.

## Background & Motivation

**Background**: Retrieval-Augmented Code Generation (RACG) enhances the code generation capabilities of LLMs by retrieving relevant code snippets. However, existing research primarily focuses on monolingual settings such as Python and Java.

**Limitations of Prior Work**: Code knowledge is heavily imbalanced across programming languages—Python possesses rich documentation and community resources, while niche languages like Scala suffer from resource scarcity. Furthermore, enterprise tech stack migrations generate substantial demand for cross-lingual code conversion.

**Key Challenge**: Can RACG effectively transfer code knowledge from one programming language to another? Is this transfer equally effective across all language pairs?

**Goal**: Systematically investigate the mechanism of cross-programming-language knowledge transfer in RACG to answer three key research questions.

**Key Insight**: Design three retrieval experimental settings (oracle injection, actual retrieval, and code retrieval without natural language) to analyze cross-lingual transfer effects through controlled variables.

**Core Idea**: Cross-lingual code knowledge transfer is feasible but asymmetric, with effectiveness depending on the kinship of language pairs and the diversity of the LLM's pre-training corpora.

## Method

### Overall Architecture

A large-scale dataset covering 13 programming languages (~14K instances) was constructed, containing NL prompts, verified reference solutions, and executable test cases. Systematic evaluations were conducted using three retrieval settings and 5 code LLMs.

### Key Designs

1.  **Three Retrieval Experimental Settings**:
    *   **Function**: To evaluate cross-lingual knowledge transfer from different perspectives.
    *   **Mechanism**: (1) Golden Solution Document—Oracle retrieval simulating ideal conditions to measure the upper bound of cross-lingual transfer; (2) Top-k Retrieved Documents—End-to-end evaluation of the full RACG pipeline; (3) Top-k without NL—Removal of natural language descriptions to simulate real-world scenarios involving pure code snippets.
    *   **Design Motivation**: To isolate the impact of the retrieval and generation stages through controlled variables, identifying the bottlenecks in cross-lingual transfer.

2.  **Large-Scale Multilingual Code Dataset**:
    *   **Function**: To provide a unified evaluation benchmark across 13 programming languages.
    *   **Mechanism**: Each instance includes an NL description, reference solutions, and test cases, covering 13 languages including C++, Go, Java, JavaScript, Python, and Rust.
    *   **Design Motivation**: Existing datasets only cover 2-5 languages, which is insufficient for large-scale cross-lingual research.

3.  **Multilingual vs. Python-specialized LLM Comparison**:
    *   **Function**: To reveal the impact of pre-training diversity on cross-lingual transfer capabilities.
    *   **Mechanism**: Comparing multilingual LLMs (CodeLlama, DeepSeek-Coder, Qwen2.5-Coder) with Python-specialized LLMs (Phi-1, Phi-1.5) in cross-lingual RACG tasks.
    *   **Design Motivation**: To distinguish whether cross-lingual transfer capability stems from architecture or the diversity of pre-training data.

### Loss & Training

This is an empirical study and does not involve model training. Greedy decoding ($temperature=0.0$) was used to ensure reproducibility, and the primary evaluation metric is $Pass@1$.

## Key Experimental Results

### Main Results (Oracle Injection, Multilingual LLM Average)

| Source $\rightarrow$ Target | C++ | Go | Java | JS | Python | Avg Gain |
|------------|------|-----|------|-----|--------|--------|
| C++ | - | +4.47 | +20.33 | +18.90 | +15.04 | +14.68 |
| Go | +9.15 | - | - | - | - | - |
| Baseline (No Retrieval) | 54.27 | 42.68 | 61.79 | 58.33 | 59.35 | 55.28 |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| Remove NL Info | Performance drop is minor | Code retrievers do not strongly rely on natural language. |
| Python-specialized LLM | Poor cross-lingual transfer | Pre-training diversity is key to cross-lingual transfer. |
| Code-specific Retriever | Significantly outperforms general retrieval | Specialized retrievers bridge NL intent and code semantics more effectively. |

### Key Findings
*   Cross-lingual knowledge transfer is non-trivial even under oracle conditions, indicating a cross-lingual gap within the generation stage itself.
*   Transfer effectiveness is asymmetric and correlates with the syntactic kinship of language pairs (e.g., Java $\rightarrow$ JavaScript is more effective than Java $\rightarrow$ Go).
*   Python-specialized LLMs are almost unable to utilize cross-lingual context, emphasizing the importance of pre-training diversity.
*   The performance drop after removing NL is minimal, suggesting that code semantics alone are sufficient to support retrieval.

## Highlights & Insights
*   The study extends the concept of "cross-lingual" from natural language to the RACG scenario for programming languages, opening a new research direction.
*   Rigorous experimental design: The three retrieval settings form a gradient from ideal to realistic, clearly revealing the migration mechanism.
*   The discovery that "Python-specialized LLMs cannot perform cross-lingual transfer" provides important guidance for model training strategies.

## Limitations & Future Work
*   Only LLMs with approximately 7B parameters were tested; the cross-lingual capabilities of larger models may differ.
*   Dataset construction relies on the translation of existing benchmarks, which may introduce bias.
*   The impact of fine-tuning on cross-lingual transfer capabilities was not explored.
*   Future research could investigate the optimization of cross-lingual retrieval strategies and hybrid-language retrieval.

## Related Work & Insights
*   **vs. Monolingual RACG**: Reveals unique challenges in cross-lingual scenarios, such as asymmetric transfer and language kinship.
*   **vs. Code Translation Tasks**: RACG is not direct translation but rather utilizing source language knowledge to enhance target language generation.
*   **vs. Multilingual NLP**: "Cross-lingual" in programming languages shares similar mechanisms with natural language (kinship affecting transfer) but also possesses unique characteristics.

## Rating
*   Novelty: ⭐⭐⭐⭐ First systematic study of cross-programming-language RACG.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Large-scale experiments involving 13 languages × 5 models × 3 settings.
*   Writing Quality: ⭐⭐⭐⭐ Clear organization around three RQs.
*   Value: ⭐⭐⭐⭐ Provides empirical guidance for the design of multilingual code tools.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Inference-Time Safety for Code LLMs via Retrieval-Augmented Revision](../../ICLR2026/code_intelligence/inference-time_safety_for_code_llms_via_retrieval-augmented_revision.md)
- [\[AAAI 2026\] SPAN: Benchmarking and Improving Cross-Calendar Temporal Reasoning of Large Language Models](../../AAAI2026/code_intelligence/span_benchmarking_and_improving_cross-calendar_temporal_reasoning_of_large_langu.md)
- [\[ACL 2026\] SWE-QA: Can Language Models Answer Repository-level Code Questions?](swe-qa_can_language_models_answer_repository-level_code_questions.md)
- [\[ACL 2026\] ReCode: Reinforcing Code Generation with Reasoning-Process Rewards](recode_reinforcing_code_generation_with_reasoning-process_rewards.md)
- [\[ACL 2026\] Ro-SLM: Onboard Small Language Models for Robot Task Planning and Operation Code Generation](ro-slm_onboard_small_language_models_for_robot_task_planning_and_operation_code_.md)

</div>

<!-- RELATED:END -->
