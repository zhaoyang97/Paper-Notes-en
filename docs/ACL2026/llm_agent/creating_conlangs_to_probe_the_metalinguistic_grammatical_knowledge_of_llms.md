---
title: >-
  [Paper Note] Creating ConLangs to Probe the Metalinguistic Grammatical Knowledge of LLMs
description: >-
  [ACL 2026][LLM Agent][constructed languages] This paper introduces IASC (Interactive Agentic System for ConLangs), a modular constructed-language building system that probes LLMs' metalinguistic knowledge by requiring them to perform morphosyntactic transformations according to linguistic specifications. The findings reveal that LLMs handle common typological patterns far better than rare ones, and that capability gaps across different LLMs are substantial.
tags:
  - ACL 2026
  - LLM Agent
  - constructed languages
  - metalinguistic knowledge
  - morphosyntactic transformation
  - LLM linguistic probing
  - linguistic typology
date: 2026-05-08
content_hash: 282ae7269bcb4093
---

# Creating ConLangs to Probe the Metalinguistic Grammatical Knowledge of LLMs

**Conference**: ACL 2026
**arXiv**: [2510.07591](https://arxiv.org/abs/2510.07591)
**Code**: [https://github.com/SakanaAI/IASC](https://github.com/SakanaAI/IASC)
**Area**: LLM Agent
**Keywords**: constructed languages, metalinguistic knowledge, morphosyntactic transformation, LLM linguistic probing, linguistic typology

## TL;DR

This paper introduces IASC (Interactive Agentic System for ConLangs), a modular constructed-language building system that probes LLMs' metalinguistic knowledge by requiring them to perform morphosyntactic transformations according to linguistic specifications. The findings reveal that LLMs handle common typological patterns far better than rare ones, and that capability gaps across different LLMs are substantial.

## Background & Motivation

**Background**: A large body of research has examined the linguistic capabilities of LLMs—including translation and syntactic annotation—but these tasks evaluate knowledge of specific languages rather than understanding of linguistic concepts per se. A key open question is whether LLMs genuinely "understand" abstract linguistic concepts such as word order, case marking, and agreement, or merely memorize patterns from training data associated with particular languages.

**Limitations of Prior Work**: (1) Existing evaluations of LLM linguistic competence tend to focus on encyclopedic tests (i.e., knowing facts about a given language), and lack systematic probing of metalinguistic reasoning; (2) evaluations on natural languages are susceptible to training data leakage, since LLMs may retrieve memorized answers rather than apply genuine rule-based understanding.

**Key Challenge**: Although LLMs are exposed to extensive linguistic literature and multilingual data during training, this exposure does not guarantee the ability to manipulate linguistic structures according to novel abstract grammatical rules. For instance, reordering an English sentence from SVO to OVS (an extremely rare word order) is no more complex in principle than reordering it to SOV, yet LLM performance on these two cases may differ dramatically.

**Goal**: (1) Provide a flexible and engaging constructed-language building tool; (2) systematically probe LLMs' metalinguistic knowledge across typologically diverse features using morphosyntactic transformation tasks.

**Key Insight**: Constructing a conlang requires LLMs not merely to translate, but to restructure sentence constituents and add morphological markers according to abstract grammatical specifications—directly testing the depth of their understanding of linguistic concepts.

**Core Idea**: A modular constructed-language building system serves as a benchmark. By requiring LLMs to transform English sentences according to varied morphosyntactic parameters (word order, case system, tense marking, etc.), the system quantifies their metalinguistic competence.

## Method

### Overall Architecture

IASC is a complete constructed-language building pipeline comprising five modules: phonology, morphosyntax, lexicon, orthography, and a grammatical handbook. This paper focuses on the morphosyntax module as a probe of LLMs' metalinguistic knowledge. The input consists of English source sentences paired with target grammatical parameters; the output is a gloss-annotated string transformed according to the target grammar. A cumulative morphosyntax strategy is adopted, applying different grammatical features incrementally via multi-step prompting.

### Key Designs

1. **Cumulative Morphosyntax Transformation**:

    - Function: Incrementally transforms source sentences into forms conforming to a target grammatical specification.
    - Mechanism: Rather than providing all grammatical specifications at once (preliminary experiments showed this performs poorly), the system applies one grammatical feature per step (e.g., first reordering words, then adding case markers, then adding tense markers), iterating via $s_i = M(s_{i-1}; G; t_i)$, where each prompt $t_i$ addresses only one grammatical feature.
    - Design Motivation: One-shot transformation produces excessively long and complex prompts, making it difficult for LLMs to satisfy multiple constraints simultaneously. The step-by-step approach reduces the cognitive load at each stage.

2. **Nine Typologically Diverse Grammar Configurations**:

    - Function: Constructs an evaluation dataset spanning common to rare linguistic types.
    - Mechanism: Eight grammar configurations inspired by real languages (Arabic, Fijian, French, Hixkaryana, Mizo, Turkish, Vietnamese, and Welsh) are designed alongside one "hard" configuration representing an extremely rare typological combination. Each configuration specifies parameters including word order, case system, agreement marking, and tense marking. The dataset comprises 45 source sentences × 9 configurations = 405 test instances, with gold data annotated by a trained linguist.
    - Design Motivation: By controlling typological frequency, the design tests whether LLMs genuinely understand abstract rules or can only handle patterns that are common in their training data.

3. **Agentic Self-Refinement Mechanism**:

    - Function: Iteratively improves outputs through automatically generated feedback.
    - Mechanism: Certain modules (e.g., phonology) employ an agentic approach in which the LLM generates an initial output, automatically produces critique and feedback on that output, and then revises accordingly, iterating until convergence.
    - Design Motivation: Initial LLM outputs may not fully conform to the specifications; self-review and correction improve output quality.

## Key Experimental Results

### Main Results

| Model | 'french' (common) | 'turkish' (common) | 'mizo' (rare) | 'hard' (extremely rare) | Overall |
|-------|-------------------|--------------------|---------------|-------------------------|---------|
| GPT-4.1 | low TER | low TER | moderate TER | relatively high TER | best |
| Claude 3.7 | low TER | low TER | moderate–high TER | high TER | 2nd |
| Gemini 2.5 | moderate TER | moderate TER | high TER | very high TER | moderate |
| Smaller models | high TER | high TER | very high TER | extremely high TER | weakest |

### Ablation Study

| Configuration | Result | Notes |
|---------------|--------|-------|
| Cumulative vs. one-shot transformation | Cumulative far superior | One-shot fails to satisfy multiple simultaneous constraints |
| Common vs. rare typological features | Common far superior | LLMs handle SVO/SOV well; OVS/OSV poorly |
| Morphological marking (prefix vs. suffix) | Suffixes better | Consistent with greater suffix frequency in training data |
| With vs. without agentic refinement | Occasionally beneficial | Not all modules benefit equally |

### Key Findings

- LLMs handle common typological patterns (e.g., SVO/SOV word orders, suffixal morphology) substantially better than rare ones (e.g., OVS word order, prefixal morphology), with performance strongly correlated with the cross-linguistic frequency of each feature.
- Capability gaps across LLMs are pronounced: GPT-4.1 performs best on most configurations, while smaller models nearly completely fail on rare configurations.
- The "hard" configuration—combining extremely rare typological features—is challenging for all models, demonstrating that LLMs' metalinguistic knowledge remains strongly constrained by the distributional properties of their training data.

## Highlights & Insights

- **Constructed languages as a probing tool**: The experimental design is highly elegant—conlangs eliminate the risk of training data leakage while enabling precise control over linguistic variables, making evaluation results highly interpretable.
- **Revealing the nature of LLM "linguistic knowledge"**: LLMs do not genuinely understand linguistic concepts; rather, they rely on the distributional patterns present in training data. Their superior performance on common typological patterns and failure on rare ones indicates that their competence is fundamentally a reflection of statistical associations rather than abstract rule comprehension.
- **Cumulative transformation strategy**: Decomposing a complex multi-constraint problem into sequential single-constraint transformations is a generalizable prompt engineering strategy transferable to other multi-step reasoning scenarios.

## Limitations & Future Work

- The evaluation dataset (405 instances) is relatively small and may be insufficient to capture interaction effects across all grammatical features.
- English is the sole source language; transformations from other source languages remain unexplored.
- Gold data for the morphosyntax module was annotated by a single linguist, potentially introducing annotator bias.
- Attempts to apply the framework to low-resource language translation yielded mostly negative results, indicating that practical applications remain distant.
- The 53-page paper contains extensive appendices; the core contributions could be presented more concisely.

## Related Work & Insights

- **vs. ConlangCrafter (Alper et al., 2025)**: Also addresses LLM-driven constructed-language building, but IASC's morphosyntax module operates at a finer granularity, enabling feature-level probing.
- **vs. conventional LLM linguistic benchmarks**: Benchmarks such as BLiMP and SyntaxGym evaluate LLM judgments about specific linguistic phenomena; IASC requires LLMs to actively manipulate linguistic structure, posing a substantially higher challenge.
- **vs. Diamond (2023)**: Uses ChatGPT with simple prompts to generate conlangs, without systematic modular control or typological evaluation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Using constructed-language building to probe metalinguistic knowledge is a highly original and substantive research perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Nine grammar configurations provide rich typological diversity, though sample size is limited.
- Writing Quality: ⭐⭐⭐⭐ The paper is highly detailed (53 pages) with thorough linguistic background, but is overly lengthy.
- Value: ⭐⭐⭐⭐⭐ Provides critical insight into the nature of LLMs' linguistic knowledge; the IASC tool itself also has independent value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Scaling External Knowledge Input Beyond Context Windows of LLMs via Multi-Agent Collaboration](scaling_external_knowledge_input_beyond_context_windows_of_llms_via_multi-agent_.md)
- [\[ACL 2026\] StructMem: Structured Memory for Long-Horizon Behavior in LLMs](structmem_structured_memory_for_long-horizon_behavior_in_llms.md)
- [\[ICLR 2026\] Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning for Web Agents](../../ICLR2026/llm_agent/web-cogreasoner_towards_knowledge-induced_cognitive_reasoning_for_web_agents.md)
- [\[ICLR 2026\] Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning in Web Agents](../../ICLR2026/llm_agent/web-cogreasoner_towards_knowledge-induced_cognitive_reasoning_in_web_agents.md)
- [\[AAAI 2026\] LLMTM: Benchmarking and Optimizing LLMs for Temporal Motif Analysis in Dynamic Graphs](../../AAAI2026/llm_agent/llmtm_benchmarking_and_optimizing_llms_for_temporal_motif_analysis_in_dynamic_gr.md)

</div>

<!-- RELATED:END -->
