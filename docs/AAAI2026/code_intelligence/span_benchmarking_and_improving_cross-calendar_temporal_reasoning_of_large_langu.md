---
title: >-
  [Paper Note] SPAN: Benchmarking and Improving Cross-Calendar Temporal Reasoning of Large Language Models
description: >-
  [AAAI 2026][Code Intelligence][cross-calendar reasoning] This paper proposes SPAN, a cross-calendar temporal reasoning benchmark (6 calendars × 10 reasoning directions × 100-year range × 37,380 instances). Baseline LLMs achieve an average accuracy of only 34.5% (none exceeding 80%), revealing two systematic failure modes—Future-Date Degradation and Calendar Asymmetry Bias. A tool-augmented Time Agent achieves 95.31%, demonstrating that cross-calendar reasoning requires extern…
tags:
  - "AAAI 2026"
  - "Code Intelligence"
  - "cross-calendar reasoning"
  - "temporal reasoning"
  - "six calendar systems"
  - "tool-augmented agent"
  - "evaluation benchmark"
date: 2026-05-08
content_hash: 22bcf3872752780a
---

# SPAN: Benchmarking and Improving Cross-Calendar Temporal Reasoning of Large Language Models

**Conference**: AAAI 2026  
**arXiv**: [2511.09993](https://arxiv.org/abs/2511.09993)  
**Code**: [GitHub](https://github.com/miaozhongjian/span.git)  
**Area**: Code Intelligence  
**Keywords**: cross-calendar reasoning, temporal reasoning, six calendar systems, tool-augmented agent, evaluation benchmark

## TL;DR
This paper proposes SPAN, a cross-calendar temporal reasoning benchmark (6 calendars × 10 reasoning directions × 100-year range × 37,380 instances). Baseline LLMs achieve an average accuracy of only 34.5% (none exceeding 80%), revealing two systematic failure modes—Future-Date Degradation and Calendar Asymmetry Bias. A tool-augmented Time Agent achieves 95.31%, demonstrating that cross-calendar reasoning requires external tools rather than parametric knowledge.

## Background & Motivation
**Background**: Temporal reasoning evaluation for LLMs is limited to the Gregorian calendar, overlooking the importance of 20+ global calendar systems for multicultural applications.

**Limitations of Prior Work**: (a) No cross-calendar reasoning benchmark exists; (b) conversion between calendar systems involves complex astronomical, religious, and cultural rules (e.g., the Islamic calendar is lunar-based and approximately 11 days shorter than the Gregorian calendar per year); (c) LLMs' temporal knowledge is predominantly derived from Gregorian-calendar corpora.

**Key Challenge**: Cross-calendar conversion requires precise mathematical computation and domain knowledge, yet parametric memory in LLMs cannot cover all calendar–date combinations, especially for future dates.

**Goal**: Establish a systematic benchmark for cross-calendar temporal reasoning and design a tool-augmented solution.

**Key Insight**: Template-driven dynamic instantiation to avoid data contamination; six calendars covering the world's major cultural spheres.

**Core Idea**: A systematic benchmark of 6 calendars × 10 reasoning directions × 100 years, combined with a tool-augmented Time Agent achieving 95.31% accuracy.

## Method

### Overall Architecture
Six calendars: Gregorian, Chinese Lunar, Saka, Hebrew, Islamic, and Persian. Ten reasoning directions (intra-calendar → cross-calendar × 2 directions × date/festival × polarity/content questions). One-hundred-year range: 1960–2060.

### Key Designs

1. **Template-Driven Dynamic Generation**:

    - Function: Runtime instantiation to prevent data contamination.
    - Four stages: calendar conversion → template matching → variable instantiation ($n_d\in[1,10]$ days / $n_w\in[1,10]$ weeks / $n_y\in[1,5]$ years) → code-execution verification.
    - Design Motivation: Static datasets are susceptible to contamination from LLM pretraining data.

2. **Time Agent**:

    - Function: LLM + `search_calendar` tool interface.
    - Three-step pipeline: few-shot prompting to generate executable code → code execution → GPT-4o generates the final answer based on execution results.
    - The `search_calendar` interface supports `{calendar_name, year, month, day}` and `{calendar_name, year, festival_name}`.
    - Design Motivation: Precise calendar conversion requires algorithmic computation rather than memorization.

3. **Two Reasoning Types**:

    - Date reasoning: reasoning given a specific date.
    - Festival reasoning: computing a date given a festival name in a particular calendar.
    - Each type includes polarity questions (yes/no) and content questions (specific date).

## Key Experimental Results

### Main Results (37,380 instances)

| Model | Average Accuracy | Notes |
|-------|-----------------|-------|
| GPT-4o | ~45% | Strongest closed-source |
| Claude-3.7-Sonnet | ~43% | Competitive |
| DeepSeek-V3 | ~45% | Matches closed-source |
| Gemini-1.5-Pro | <30% | Worst |
| **All LLMs (average)** | **34.5%** | **None exceeds 80%** |
| OpenAI-o1 (reasoning) | 59.29% | 2nd place |
| GPT-4o + RAG | 43.69% | Only +0.68% |
| **Time Agent** | **95.31%** | **Tool-augmented wins decisively** |

### Systematic Failure Mode Analysis

| Failure Mode | Manifestation | Magnitude |
|-------------|--------------|-----------|
| Future-Date Degradation | Past ~40% → Future ~25% | −15 pp |
| Calendar Asymmetry Bias | Gregorian→others vs. reverse | 3.97–17.49% gap |
| Polarity vs. Content | Polarity > Content | +18.86% on average |
| Date vs. Festival | Festival > Date | +2.87–12.60% |

### Key Findings
- **Cross-calendar reasoning is a systematic blind spot for LLMs**—34.5% accuracy approaches random chance.
- **Future-Date Degradation**: Accuracy on future dates is 10–15 pp lower than on past dates—future events are absent from training data.
- **Calendar Asymmetry Bias**: Accuracy in the Gregorian→other direction is 15–17 pp higher—pretraining data is predominantly Gregorian-centric.
- **Tool augmentation is the only effective solution**: Time Agent (95.31%) vs. o1 reasoning (59.29%) vs. RAG (43.69%)—neither reasoning nor retrieval alone suffices.
- RAG provides negligible benefit (+0.68%) because retrieved content is also grounded in parametric knowledge sources.

## Highlights & Insights
- **Two systematic biases in LLMs' temporal knowledge are revealed**—Future-Date Degradation and Calendar Asymmetry Bias are generalizable findings. Other knowledge domains (e.g., non-Western legal systems, non-English literature) may exhibit similar center-periphery biases.
- **Clear stratification among tools, reasoning, and retrieval**: Time Agent (95%) >> o1 (59%) >> RAG (44%) >> base LLMs (34%). Precise computation tasks require external tools rather than stronger reasoning or more retrieval.
- **Contamination-resistant dynamic instantiation** is a simple yet critical design choice that ensures the long-term validity of the benchmark.

## Limitations & Future Work
- Only six calendars are covered; the benchmark could be extended to the Japanese, Thai Buddhist, Indian national, and other calendars.
- The Time Agent depends on the coverage of the `search_calendar` API.
- Only calendar conversion is tested; more complex temporal reasoning (e.g., "day of the week from Islamic to Lunar calendar") is not evaluated.
- Internalizing temporal reasoning capabilities into model parameters is a direction worth exploring.

## Related Work & Insights
- **vs. TimeQA/TempReason and similar temporal benchmarks**: These cover only the Gregorian calendar; SPAN is the first to encompass multiple calendar systems, filling the gap in multicultural temporal reasoning evaluation.
- **vs. tool-augmented LLMs**: The Time Agent demonstrates the irreplaceability of external tools for precise computation tasks—the gap between 95.31% and reasoning model's 59.29% indicates that certain capabilities must be externalized.
- **vs. RAG approaches**: GPT-4o + RAG improves accuracy by only 0.68% (43.01→43.69%), showing that retrieval augmentation is largely ineffective for computation-oriented tasks, since retrieved content is also derived from parametric knowledge sources.
- **Insight**: Multicultural and multi-system evaluation is critical for globally deployed AI. LLMs' knowledge biases (Gregorian-first, past-first) reflect structural imbalances in training data.
- **Future directions**: The SPAN evaluation paradigm can be extended to other culturally dependent reasoning tasks (e.g., non-Western legal systems, traditional medicine).
- **Practical application scenarios**: International meeting scheduling, cross-national holiday computation, and public services for multicultural communities all rely on accurate cross-calendar conversion.
- **Connection to other tool-augmented work**: SPAN's findings are consistent with those of RoutingGen in code generation—certain tasks inherently require external tools rather than pure model reasoning.
- **Generalizability of contamination prevention**: The template-driven dynamic instantiation approach is transferable to other benchmark designs that need to prevent pretraining data leakage.
- **Cultural coverage of six calendars**: Gregorian (Western), Chinese Lunar (East Asian), Saka (Indian), Hebrew (Jewish culture), Islamic (Muslim world), Persian (Iran)—covering the world's major cultural spheres.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First cross-calendar reasoning benchmark; discovery of two systematic failure modes.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 37,380 instances × 6 calendars × 10 directions × multiple models + tool comparisons.
- Writing Quality: ⭐⭐⭐⭐ Systematic benchmark design.
- Value: ⭐⭐⭐⭐ Significant contribution to multicultural LLM evaluation and tool-augmented reasoning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] CrossPL: Systematic Evaluation of Large Language Models for Cross Programming Language Interoperating Code Generation](../../ICLR2026/code_intelligence/crosspl_systematic_evaluation_of_large_language_models_for_cross_programming_lan.md)
- [\[ACL 2025\] CodeIF: Benchmarking the Instruction-Following Capabilities of Large Language Models for Code Generation](../../ACL2025/code_intelligence/codeif_benchmarking_the_instruction-following_capabilities_of_large_language_mod.md)
- [\[AAAI 2026\] EquaCode: A Multi-Strategy Jailbreak Approach for Large Language Models via Equation Solving and Code Completion](equacode_a_multi-strategy_jailbreak_approach_for_large_language_models_via_equat.md)
- [\[ICLR 2026\] Local Success Does Not Compose: Benchmarking Large Language Models for Compositional Formal Verification](../../ICLR2026/code_intelligence/local_success_does_not_compose_benchmarking_large_language_models_for_compositio.md)
- [\[ICLR 2026\] Training Large Language Models To Reason In Parallel With Global Forking Tokens](../../ICLR2026/code_intelligence/training_large_language_models_to_reason_in_parallel_with_global_forking_tokens.md)

</div>

<!-- RELATED:END -->
