---
title: >-
  [Paper Note] X-WebAgentBench: A Multilingual Interactive Web Benchmark for Evaluating Global Agentic System
description: >-
  [ACL 2025 Findings][Multilingual & Machine Translation][Multilingual Agents] X-WebAgentBench is proposed—a multilingual interactive web benchmark designed to evaluate the planning and interaction capabilities of language agents across various languages. Multiple LLMs and cross-lingual alignment methods are evaluated, revealing that even GPT-4o combined with cross-lingual techniques fails to achieve satisfactory results.
tags:
  - "ACL 2025 Findings"
  - "Multilingual & Machine Translation"
  - "Multilingual Agents"
  - "Web Interaction Benchmark"
  - "Cross-Lingual Alignment"
  - "Global Evaluation"
  - "Planning and Interaction"
date: 2026-05-08
content_hash: 1092781e1885e142
---

# X-WebAgentBench: A Multilingual Interactive Web Benchmark for Evaluating Global Agentic System

**Conference**: ACL 2025 Findings  
**arXiv**: [2505.15372](https://arxiv.org/abs/2505.15372)  
**Code**: None  
**Area**: Multilingual Translation  
**Keywords**: Multilingual Agents, Web Interaction Benchmark, Cross-Lingual Alignment, Global Evaluation, Planning and Interaction

## TL;DR

X-WebAgentBench is proposed—a multilingual interactive web benchmark designed to evaluate the planning and interaction capabilities of language agents across various languages. Multiple LLMs and cross-lingual alignment methods are evaluated, revealing that even GPT-4o combined with cross-lingual techniques fails to achieve satisfactory results.

## Background & Motivation

**Background**: LLM-based agents have achieved remarkable success in interactive environments, particularly in web operation tasks where agents complete tasks by executing actions such as clicking, typing, and navigating on web pages based on natural language instructions. Benchmarks like WebArena and Mind2Web have driven rapid development in this field.

**Limitations of Prior Work**: Currently, almost all web agent research focuses on English scenarios. However, with over 7,000 languages spoken worldwide, users of all languages require comparable agent services. A Chinese user needs the agent to complete tasks on Chinese websites, and an Arabic user needs action on Arabic web pages—yet there is a lack of benchmarks to evaluate the multilingual capabilities of agents.

**Key Challenge**: The multilingual capabilities of LLMs are severely unbalanced—showing excellent performance in high-resource languages (e.g., English) but significantly degrading in low-resource languages. When this imbalance is transferred to interactive agent scenarios, the problem is amplified: agents must not only understand multilingual instructions but also correctly identify elements, make decisions, and execute actions on multilingual web pages.

**Goal**: (1) Construct an interactive web agent benchmark covering multiple languages; (2) evaluate the actual performance of existing LLMs and cross-lingual alignment methods in multilingual agent scenarios; (3) reveal the core challenges faced by multilingual agents.

**Key Insight**: Perform multilingual expansion based on existing English web agent environments, construct task instructions and web environments encompassing various languages, and introduce cross-lingual alignment techniques to narrow the linguistic gap.

**Core Idea**: Extend mature English web agent evaluation frameworks to multilingual scenarios. By comparing the combined performance of different languages, LLMs, and cross-lingual methods, the bottlenecks of multilingual web agents are systematically revealed.

## Method

### Overall Architecture

The construction of X-WebAgentBench comprises three parts: (1) Multilingual task construction—translating English task instructions into target languages and adjusting the language settings of the web environments accordingly; (2) Interactive evaluation environment—constructing interactive web environments based on real website architectures, where agents interact with web pages through actions (such as click, type, scroll, etc.); (3) Cross-lingual method evaluation—evaluating the combined performance of various LLMs and cross-lingual alignment strategies on the benchmark.

### Key Designs

1. **Multilingual Task and Environment Construction**:

    - Function: Provide high-quality web interaction tasks covering multiple languages.
    - Mechanism: Based on the English version of web agent tasks, task instructions are translated into multiple languages—such as Chinese, Japanese, Korean, Arabic, French, German, Spanish, and Portuguese—through professional translation and localization. Translation is not just text conversion; it also requires adjusting localization elements like language settings, currency units, and date formats within the web environment. Every language version undergoes review by native speakers to ensure naturalness and correctness.
    - Design Motivation: Simple machine translation leads to unnatural tasks, whereas localization ensures that tasks for each language appear as if they were natively designed for that language. This makes the evaluation results reflect agent performance in realistic multilingual scenarios more accurately.

2. **Dual-Dimensional Evaluation of Planning and Interaction**:

    - Function: Separately evaluate the high-level planning capability and low-level interaction execution capability of agents.
    - Mechanism: The completion of web tasks is decomposed into two dimensions—Planning scores measure whether the agent can formulate the correct sequence of steps (e.g., "first search for product $\rightarrow$ select the first result $\rightarrow$ add to cart $\rightarrow$ checkout"), while Interaction scores measure whether the agent can correctly execute specific actions in each step (e.g., locating the search box, inputting the correct content, and clicking the correct button). The overall metric is the Success Rate (SR).
    - Design Motivation: Evaluating planning and interaction separately allows for more precise localization of bottlenecks in multilingual scenarios—whether the agent does not know what to do in non-English environments (planning failure), or knows what to do but cannot operate correctly (interaction failure).

3. **Integration of Cross-Lingual Alignment Methods**:

    - Function: Evaluate whether multiple methods can close the language gap for multilingual agents.
    - Mechanism: Three categories of cross-lingual alignment methods are evaluated—(1) Translate-then-Act: translating non-English instructions into English for the agent to execute, which depends on translation quality; (2) Cross-lingual Prompting: adding multilingual examples or cross-lingual instructions to the prompt to guide the model to leverage its cross-lingual capabilities; (3) Multilingual Fine-tuning: fine-tuning agent models on multilingual data. Each method is evaluated in combination with multiple LLMs (GPT-4o, Qwen, LLaMA, etc.).
    - Design Motivation: If simple cross-lingual methods could eliminate the gap, the multilingual agent problem would be less urgent. Experimental results show that even when combining the strongest models with the best cross-lingual methods, the gap remains significant, highlighting the urgency and importance of this issue.

### Loss & Training

X-WebAgentBench is an evaluation benchmark and does not involve model training. The evaluated multilingual fine-tuning methods utilize a standard instruction fine-tuning pipeline, fine-tuning base models on multilingual task instructions and their corresponding action sequences.

## Key Experimental Results

### Main Results

| Model | English SR | Chinese SR | Japanese SR | Arabic SR | Average SR |
|------|--------|--------|--------|-----------|--------|
| GPT-4o | 42.3 | 31.5 | 28.7 | 22.4 | 31.2 |
| GPT-4o + Translate | 42.3 | 34.8 | 31.2 | 26.1 | 33.6 |
| GPT-4o + XL-Prompt | 42.3 | 33.2 | 30.5 | 24.8 | 32.7 |
| Qwen-Max | 38.7 | 35.2 | 26.3 | 19.8 | 30.0 |
| LLaMA-3-70B | 35.1 | 24.8 | 21.5 | 16.2 | 24.4 |
| LLaMA-3-70B + XL-FT | 35.1 | 28.6 | 25.1 | 20.7 | 27.4 |

### Planning vs. Interaction Analysis

| Model | English Planning | Non-English Planning | English Interaction | Non-English Interaction |
|------|---------|----------|---------|----------|
| GPT-4o | 68.5 | 52.3 | 61.7 | 48.1 |
| Qwen-Max | 63.2 | 54.8 | 58.4 | 43.6 |
| LLaMA-3-70B | 58.4 | 41.7 | 53.2 | 37.5 |

### Key Findings

- The performance of all models on non-English languages is significantly lower than that on English; GPT-4o achieves 42.3% SR in English but only 22.4% in Arabic, a gap of nearly 20 percentage points.
- The translation-assisted method (Translate-then-Act) is the simplest and most effective cross-lingual strategy, but its improvement is limited (average gain of only +2.4%) and heavily relies on translation quality.
- Cross-lingual alignment methods fail to eliminate the language gap: even when GPT-4o is combined with the best cross-lingual method, the average non-English SR is only 33.6%, which is far below the English SR of 42.3%.
- Both planning and interaction are affected by multilingual factors, but degradation in the interaction dimension is more severe—the agent's ability to locate and operate elements on non-English web pages decreases more significantly.
- Qwen-Max performs close to English in Chinese (35.2 vs 38.7), potentially due to its strong pre-training in Chinese, but exhibits obvious degradation in other languages.

## Highlights & Insights

- **The first multilingual interactive web agent benchmark** fills an important gap—prior web agent evaluations assumed an English-only environment, yet practical application scenarios are far more diverse than English.
- The design of the dual-dimensional evaluation (planning and interaction) is highly valuable: it reveals that multilingual degradation mainly occurs at the interaction layer (failing to locate correct web elements) rather than the planning layer (not knowing what to do). This points toward practical solutions—improving web element understanding is more urgent than improving task planning.
- The conclusion that "the strongest model + the best method is still insufficient" serves as a crucial warning—it indicates that multilingual web agents cannot be solved through simple tricks and require more fundamental technical breakthroughs.

## Limitations & Future Work

- Multilingual tasks are primarily constructed by translating existing English tasks, which may not capture web interaction patterns unique to certain languages.
- Although many languages are covered, the scope remains limited; many low-resource languages (e.g., Swahili, Hindi dialects) are not included.
- The web environments are simulated and may exhibit discrepancies compared to real-world websites.
- There is no in-depth analysis on which specific linguistic features (e.g., text direction, morphological complexity, word segmentation difficulty) lead to agent degradation.

## Related Work & Insights

- **vs WebArena**: WebArena is the most influential English web agent benchmark. X-WebAgentBench can be seen as its multilingual extension, with the core difference being the introduction of the cross-lingual dimension.
- **vs XTREME / XGLUE**: These are multilingual NLU evaluation benchmarks, but they only evaluate text understanding/generation capabilities without involving planning and action capabilities in interactive environments.
- **vs AgentBench**: AgentBench evaluates LLM capabilities in multiple interactive environments, but all tasks are in English. X-WebAgentBench adds a multilingual dimension within the specific environment of web pages.

## Rating

- Novelty: ⭐⭐⭐⭐ Multilingual web agent evaluation is a novel and important direction.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across multiple models, languages, and methods.
- Writing Quality: ⭐⭐⭐⭐ Clearly defined problems and clear presentation of experimental results.
- Value: ⭐⭐⭐⭐ Reveals significant gaps in multilingual agents and encourages the community to focus on this overlooked direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MORPHOGEN: A Multilingual Benchmark for Evaluating Gender-Aware Morphological Generation](../../ACL2026/multilingual_mt/morphogen_a_multilingual_benchmark_for_evaluating_gender-aware_morphological_gen.md)
- [\[ACL 2025\] 7 Points to Tsinghua but 10 Points to Qinghua? Assessing Agentic Large Language Models in Multilingual National Bias](assessing_agentic_large_language_models_in_multilingual_national_bias.md)
- [\[ACL 2025\] M-RewardBench: Evaluating Reward Models in Multilingual Settings](m_rewardbench.md)
- [\[ACL 2025\] Towards Global AI Inclusivity: A Large-Scale Multilingual Terminology Dataset (GIST)](towards_global_ai_inclusivity_a_large-scale_multilingual_terminology_dataset_gis.md)
- [\[ACL 2025\] EXECUTE: A Multilingual Benchmark for LLM Token Understanding](execute_a_multilingual_benchmark_for_llm_token_understanding.md)

</div>

<!-- RELATED:END -->
