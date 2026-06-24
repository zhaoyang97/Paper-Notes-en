---
title: >-
  [Paper Note] Explain-then-Process: Using Grammar Prompting to Enhance Grammatical Acceptability Judgments
description: >-
  [ACL 2025][LLM (Other)][grammar prompting] This paper proposes the "explain-then-process" paradigm for grammar prompting, where an LLM first generates an explanation of the target grammatical phenomenon, and this explanation is subsequently fed back to the target model (LLM or SLM) as context to assist in minimal-pair grammatical acceptability judgments. This paradigm significantly improves accuracy across three multilingual benchmarks (English BLiMP, Chinese SLING…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "grammar prompting"
  - "grammatical acceptability"
  - "minimal pairs"
  - "explain-then-process"
  - "multilingual"
date: 2026-05-08
content_hash: cb638ccbe74e3651
---

# Explain-then-Process: Using Grammar Prompting to Enhance Grammatical Acceptability Judgments

**Conference**: ACL 2025  
**arXiv**: [2506.02302](https://arxiv.org/abs/2506.02302)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: grammar prompting, grammatical acceptability, minimal pairs, explain-then-process, multilingual

## TL;DR

This paper proposes the "explain-then-process" paradigm for grammar prompting, where an LLM first generates an explanation of the target grammatical phenomenon, and this explanation is subsequently fed back to the target model (LLM or SLM) as context to assist in minimal-pair grammatical acceptability judgments. This paradigm significantly improves accuracy across three multilingual benchmarks (English BLiMP, Chinese SLING, and Russian RuBLiMP). Pairing SLMs with GP+CoT narrows the average LLM-SLM performance gap from 13.0 percentage points (pp) to 5.8pp (a 56% reduction).

## Background & Motivation

**Background**: LLMs exhibit strong language usage capabilities (functional competence) but reveal unexpected weaknesses in explicit grammatical judgment tasks. For instance, Claude Sonnet, when judging NPI (Negative Polarity Item) licensing, tends to paraphrase sentences first, which loses critical syntactic constraint information during paraphrasing and leads to incorrect judgments.

**Limitations of Prior Work**: When performing grammatical judgments, LLMs tend to paraphrase or translate before analyzing, which systematically obscures critical grammatical features. This creates a critical gap between "knowing the rule" and "applying the rule." LLMs can explain grammatical rules but often fail to apply them correctly during judgment tasks.

**Key Challenge**: LLMs possess implicit grammatical knowledge (enabling them to generate fluent text) but fail to systematically retrieve and apply this knowledge in explicit judgments—resulting in a mismatch between formal competence (knowing rules) and functional competence (using rules).

**Goal**: How to help models focus on linguistic structure rather than paraphrasing when making grammatical judgments?

**Key Insight**: Drawing inspiration from psycholinguistics and the success of MTOB (Machine Translation from One Book): explicitly providing grammatical knowledge can activate the intrinsic linguistic competence of the model.

**Core Idea**: "Explain-then-process"—feeding the self-generated grammatical explanations back to the LLM itself or to an SLM to bridge the gap between "knowing rules" and "using rules."

## Method

### Overall Architecture

A two-step explain-then-process pipeline:
1. **Explain**: using instruction templates to guide an LLM (e.g., Sonnet, GPT-o1) to generate concise grammatical explanations for specific grammatical phenomena (e.g., NPI licensing, filler-gap dependency), specifically avoiding full example sentences to prevent pattern matching.
2. **Process**: feeding the generated grammatical explanations as contextual prompts to the target model for minimal pair judgment (selecting the grammatically correct sentence from a pair that differs by only one syntactic feature).

### Key Designs

1. **Grammatical Explanation Generation (Explain)**:
    - **Function**: Designing instruction templates to guide the LLM to generate explanations of specific grammatical paradigms.
    - **Mechanism**: The template contains the name of the grammatical paradigm (e.g., "NPI licensing"), exemplar minimal pairs, and instructions (demanding the exclusion of full example sentences and specifying the target audience) to generate explanations tailored for either "beginners" or "experts".
    - **Design Motivation**: (1) Avoiding full example sentences prevents the model from relying on superficial pattern matching instead of actual reasoning; (2) Beginner explanations emphasize practical detection heuristics (e.g., "check using who/what"), whereas expert explanations leverage technical terminology (e.g., "long-distance dependency," "selectional restrictions").
    - **Finding**: Beginner explanations outperform expert explanations overall by a small but statistically significant margin ($-1.9\% \pm 5.7\%$, $p=0.002$) in macro-level analysis.

2. **Prompting Strategy Combination (Process)**:
    - **Function**: Testing multiple prompting strategies and their combinations.
    - **Mechanism**:
        - **Base**: Directly asks "which sentence is more grammatically correct".
        - **CoT**: Requires step-by-step reasoning before outputting the answer.
        - **GP (Grammar Prompting)**: Feeds the grammatical explanation as a contextual prefix.
        - **GP+CoT**: First provides the grammatical explanation, then requires step-by-step reasoning.
    - **Control Conditions**: "Control" (providing explanations of unrelated grammatical phenomena) and "Textbook" (providing multiple grammatical explanations and letting the model choose).
    - **Design Motivation**: GP and CoT target different bottlenecks—GP provides the missing rule knowledge, while CoT activates rule application capabilities. Combining them addresses both bottlenecks simultaneously.

3. **Multilingual Minimal Pair Evaluation**:
    - **Function**: Evaluating across English BLiMP (hard subset of 8 categories selected from 67 paradigms), Chinese SLING (6 categories from 38 paradigms), and Russian RuBLiMP (7 categories from 45 paradigms), selecting the top 50 pairs per paradigm.
    - **Mechanism**: Averaging results over three A/B presentation orders (forward order, reverse order, and randomized order) to eliminate positional bias; using a prompt-based approach rather than perplexity.
    - **Design Motivation**: The multilingual setup validates the language-agnostic nature of the method; extracting the hard subset focuses on grammatical phenomena where models genuinely struggle.

## Key Experimental Results

### Main Results (GPT-4o + Grammar Prompting, Hard Subsets of Benchmarks)

| Benchmark | Base | CoT | GPb (Sonnet) | GPb+CoT (o1) |
|---|---|---|---|---|
| BLiMP (English) | 77.0 | 79.9 | 85.2 | **96.7** |
| SLING (Chinese) | 93.1 | 96.7 | 97.1 | **99.2** |
| RuBLiMP (Russian) | 93.3 | 97.6 | 98.0 | **100.0** |

### SLM Experiments (Haiku + Grammar Prompting)

| Benchmark | Base | CoT | GPb+CoT (Sonnet) | GPb+CoT (o1) |
|---|---|---|---|---|
| BLiMP (English) | 61.2 | 72.0 | 82.3 | **86.5** |
| SLING (Chinese) | 78.3 | 83.6 | 89.2 | **93.3** |
| RuBLiMP (Russian) | 78.3 | 86.3 | 93.2 | **95.8** |

### Ablation Study: Control Conditions vs. GP (GPT-4o, BLiMP)

| Condition | gpt-3.5 Avg | gpt-4o Avg |
|---|---|---|
| Control (Unrelated Explanation) | 64.1 | 75.8 |
| Textbook (Mixed Multiple Rules) | 61.3 | 77.8 |
| GPb (Target Rule Explanation) | **72.5** | **90.2** |

### Key Findings
- Grammar Prompting alone improves gpt-3.5 from 67.9% to 73.6% (+5.7pp) and gpt-4o from 77.0% to 85.2% (+8.2pp) on BLiMP.
- The GP+CoT combination yields the strongest performance: gpt-4o achieves 96.7% on BLiMP, and Sonnet reaches 100% on RuBLiMP.
- Control conditions (unrelated explanations) sometimes degrade performance, proving that the improvement stems from targeting specific grammatical knowledge rather than general instruction-following.
- Beginner explanations generally outperform expert explanations overall ($p=0.002$), though expert explanations excel in specific paradigms like filler-gap dependencies.
- Combining SLMs (Haiku) with GP+CoT narrows the gap with LLMs from 13.0pp to 5.8pp—with GP alone narrowing it by 20%, and GP+CoT by 56%.
- 3-shot prompting performs poorly on SLMs (potentially triggering shortcut-based pattern matching), demonstrating that GP is a more principled approach.

## Highlights & Insights
- **Insight on the "knowing" vs. "applying" rules gap**: LLMs can explain grammar but struggle with grammatical judgments because they tend to rely on paraphrasing rather than structural analysis. GP offers an elegant solution by explicitly providing rules to guide the model's attention back to the structural level.
- **Practical value of empowering SLMs**: GP elevates low-cost SLMs to near-frontier LLM performance on grammatical judgments, carrying significant practical value for resource-constrained scenarios and educational applications. The GP+CoT combination is particularly powerful.
- **Zero-cost multilingual generalization**: The method is effective across English, Chinese, and Russian—three typologically diverse languages—and the grammatical explanations can be prompted in English even when target sentences are in Chinese or Russian. This demonstrates the language-agnostic nature of the approach.

## Limitations & Future Work
- Grammatical explanations must be generated once for each paradigm, but the paradigm identification process itself is not automated (paradigm labels are known in testing).
- Evaluations were restricted to only 5 models from the GPT, Claude, and Llama families.
- The selection of hard paradigms was based on the baseline performance of gpt-4o, which may introduce selection bias.
- The method was not tested on broader real-world applications (such as grammatical error correction or writing assistance) and remains limited to minimal pair judgment tasks.
- The quality of grammatical explanations relies on the metalinguistic knowledge of the LLM, which may fail for low-resource languages or rare grammatical phenomena.

## Related Work & Insights
- **vs. MTOB (Tanzer et al., 2024)**: MTOB leverages a grammar book to improve zero-resource translation; GP is conceptually similar but utilizes the LLM itself as the source of the "grammar book."
- **vs. CoT**: CoT activates the reasoning process without introducing new knowledge; GP provides the missing domain-specific knowledge. The two approaches are orthogonal and complementary.
- **vs. Few-shot**: Few-shot prompting might encourage pattern matching instead of rule comprehension; GP provides abstract rules rather than examples, rendering it more principled.

## Rating
- Novelty: ⭐⭐⭐⭐ The explain-then-process paradigm and the orthogonal, complementary relationship between GP and CoT serve as valuable contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Rigorously designed with 3 languages × 5 models × multiple conditions × control experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ Vivid introductory examples and clearly structured experimental design.
- Value: ⭐⭐⭐⭐ Offers practical contributions to LLM linguistic evaluation and prompting methodology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Comparing Linguistic Acceptability Judgments of Autoregressive Language Models](comparing_linguistic_acceptability_judgments_of_autoregressive_language_models.md)
- [\[ACL 2025\] On the Acquisition of Shared Grammatical Representations in Bilingual Language Models](on_the_acquisition_of_shared_grammatical_representations_in_bilingual_language_m.md)
- [\[ACL 2025\] HyGenar: An LLM-Driven Hybrid Genetic Algorithm for Few-Shot Grammar Generation](hygenar_an_llm-driven_hybrid_genetic_algorithm_for_few-shot_grammar_generation.md)
- [\[ACL 2025\] Divide-Then-Aggregate: An Efficient Tool Learning Method via Parallel Tool Invocation](dta_llama_parallel_tool_invocation.md)
- [\[ACL 2025\] Can Input Attributions Explain Inductive Reasoning in In-Context Learning?](can_input_attributions_explain_inductive_reasoning_in_in-context_learning.md)

</div>

<!-- RELATED:END -->
