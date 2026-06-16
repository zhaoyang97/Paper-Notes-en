---
title: >-
  [Paper Note] Across Programming Language Silos: A Study on Cross-Lingual Retrieval-Augmented Code Generation
description: >-
  [ACL 2026][Code Intelligence][Paper Note] This paper presents the first systematic study of cross-programming-language Retrieval-Augmented Code Generation (RACG). By constructing a 14K-instance dataset across 13 languages, the study reveals the asymmetry of cross-lingual knowledge transfer and its relationship with language affinity and pre-training diversity.
tags:
  - ACL 2026
  - Code Intelligence
date: 2026-05-08
content_hash: 628c7abb725b12cd
---
# Across Programming Language Silos: A Study on Cross-Lingual Retrieval-Augmented Code Generation

**Conference**: ACL 2026 Findings  
**arXiv**: [2506.03535](https://arxiv.org/abs/2506.03535)  
**Code**: [GitHub](https://github.com/icip-cas/Cross-Lingual-RACG)  
**Area**: Code Intelligence / Cross-Lingual Code Generation  
**Keywords**: Cross-Lingual Code Generation, Retrieval-Augmented Generation, Knowledge Transfer, Multilingual Programming, Code Retrieval

## TL;DR

This paper presents the first systematic study of cross-programming-language Retrieval-Augmented Code Generation (RACG). By constructing a 14K-instance dataset across 13 languages, the study reveals the asymmetry of cross-lingual knowledge transfer and its relationship with language affinity and pre-training diversity.

## Background & Motivation

**Background**: Retrieval-Augmented Code Generation (RACG) enhances LLM code generation by retrieving relevant snippets. However, existing research primarily focuses on monolingual settings like Python and Java.

**Limitations of Prior Work**: Code knowledge is severely imbalanced across programming languages—Python enjoys rich documentation and community resources, while niche languages like Scala face scarcity. Furthermore, enterprise tech stack migrations create significant demand for cross-lingual code conversion.

**Key Challenge**: Can RACG effectively transfer code knowledge from one programming language to another? Is this transfer equally effective for all language pairs?

**Goal**: To systematically investigate the mechanisms of cross-programming-language knowledge transfer in RACG and answer three key research questions.

**Key Insight**: Three retrieval experimental settings (oracle injection, practical retrieval, and retrieval without natural language) were designed to analyze cross-lingual transfer effects through controlled variables.

**Core Idea**: Cross-lingual code knowledge transfer is feasible but asymmetrical; its effectiveness depends on the affinity between language pairs and the diversity of the LLM’s pre-training corpus.

## Method

### Overall Architecture

This is an empirical study centered on "Cross-Lingual Retrieval-Augmented Code Generation." The core workflow involves data construction, experimental setup, and comparative analysis. Given an NL requirement in a specific language, the system retrieves relevant code in another language from a unified dataset of 14K instances (each containing an NL description, verified reference solutions, and executable test cases). This retrieved code serves as context for the target language generation, evaluated via Pass@1. The study decomposes the transferability and its origins using three retrieval settings across five code LLMs.

### Key Designs

**1. Three Retrieval Settings: Disentangling Retrieval and Generation Stages**  
To locate the true bottleneck of cross-lingual transfer, the study designs three levels of retrieval reaching from ideal to realistic conditions. **Golden Solution Document** uses oracle injection of target solution documents to measure the upper bound of transfer capability. **Top-k Retrieved Documents** provides an end-to-end evaluation of the full RACG pipeline. **Top-k without NL** removes natural language descriptions from documents, leaving only pure code snippets to simulate "code-only" enterprise repositories.

**2. Large-Scale Multilingual Code Dataset: Supporting Cross-Lingual Research Across 13 Languages**  
Since existing RACG datasets cover only 2–5 languages, this work builds a dataset spanning 13 languages (C++, Go, Java, JavaScript, Python, Rust, etc.). Each instance includes an NL description, a reference solution, and executable test cases, ensuring that "Source $\rightarrow$ Target" pairings can be measured under a unified standard.

**3. Multilingual vs. Python-Specific LLM Comparison: Attributing Transfer Capability to Pre-training Diversity**  
To determine if transfer capability stems from architecture or data diversity, the study compares multilingual code LLMs (CodeLlama, DeepSeek-Coder, Qwen2.5-Coder) against Python-specific models (Phi-1, Phi-1.5). Results showing that Python-specific models fail to benefit from cross-lingual context attribute transfer capability clearly to pre-training corpus diversity.

### Loss & Training

This paper is a purely empirical study and does not involve model training or fine-tuning. All models use greedy decoding (temperature=0.0) for reproducibility, and the evaluation metric is Pass@1.

## Key Experimental Results

### Main Results (Oracle Injection, Multilingual LLM Average)

| Source $\rightarrow$ Target | C++ | Go | Java | JS | Python | Average Gain |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| C++ | - | +4.47 | +20.33 | +18.90 | +15.04 | +14.68 |
| Go | +9.15 | - | - | - | - | - |
| Baseline (No Retrieval) | 54.27 | 42.68 | 61.79 | 58.33 | 59.35 | 55.28 |

### Ablation Study

| Configuration | Key Metrics | Description |
| :--- | :--- | :--- |
| Remove NL Info | Performance drop is minor | Code retrievers do not rely heavily on natural language. |
| Python-specific LLM | Poor cross-lingual transfer | Pre-training diversity is the key to cross-lingual transfer. |
| Code-specific Retriever | Significantly outperforms general retrievers | Specialized retrievers better bridge NL intent and code semantics. |

### Key Findings
- Cross-lingual knowledge transfer is non-trivial even under oracle conditions, indicating a cross-lingual gap in the generation stage itself.
- Transfer effectiveness is asymmetrical and correlates with the syntactic affinity of language pairs (e.g., Java $\rightarrow$ JavaScript is more effective than Java $\rightarrow$ Go).
- Python-specific LLMs are nearly unable to utilize cross-lingual context, emphasizing the importance of pre-training diversity.
- Removing NL results in minimal retrieval performance loss, suggesting code semantics alone can support retrieval.

## Highlights & Insights
- Extends the "cross-lingual" concept from natural language to programming languages in RACG for the first time.
- Rigorous experimental design: The three retrieval settings create a gradient from ideal to realistic, clearly revealing transfer mechanisms.
- The finding that "Python-specific LLMs cannot perform cross-lingual transfer" serves as important guidance for model training strategies.

## Limitations & Future Work
- Only LLMs with approximately 7B parameters were tested; the cross-lingual capabilities of larger models may differ.
- Dataset construction relies on translations of existing benchmarks, which may introduce bias.
- The impact of fine-tuning on cross-lingual transfer capability has not been explored.
- Future work could investigate the optimization of cross-lingual retrieval strategies and hybrid-language retrieval.

## Related Work & Insights
- **vs. Monolingual RACG**: Reveals unique challenges of cross-lingual scenarios, such as asymmetrical transfer and linguistic affinity.
- **vs. Code Translation**: RACG is not direct translation but utilizes source language knowledge to augment target language generation.
- **vs. Multilingual NLP**: "Cross-lingual" programming shares similar mechanisms with natural language (affinity affects transfer) but possesses unique characteristics.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First systematic study of cross-programming-language RACG.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Large-scale experiments across 13 languages, 5 models, and 3 settings.
- **Writing Quality**: ⭐⭐⭐⭐ Clear organization around three Research Questions.
- **Value**: ⭐⭐⭐⭐ Provides empirical guidance for designing multilingual code tools.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Inference-Time Safety for Code LLMs via Retrieval-Augmented Revision](../../ICLR2026/code_intelligence/inference-time_safety_for_code_llms_via_retrieval-augmented_revision.md)
- [\[AAAI 2026\] SPAN: Benchmarking and Improving Cross-Calendar Temporal Reasoning of Large Language Models](../../AAAI2026/code_intelligence/span_benchmarking_and_improving_cross-calendar_temporal_reasoning_of_large_langu.md)
- [\[ACL 2026\] SWE-QA: Can Language Models Answer Repository-level Code Questions?](swe-qa_can_language_models_answer_repository-level_code_questions.md)
- [\[ACL 2026\] RepoShapley: Shapley-Enhanced Context Filtering for Repository-Level Code Completion](reposhapley_shapley-enhanced_context_filtering_for_repository-level_code_complet.md)
- [\[ACL 2026\] CollabCoder: Plan-Code Co-Evolution via Collaborative Decision-Making for Efficient Code Generation](collabcoder_plan-code_co-evolution_via_collaborative_decision-making_for_efficie.md)

</div>

<!-- RELATED:END -->
