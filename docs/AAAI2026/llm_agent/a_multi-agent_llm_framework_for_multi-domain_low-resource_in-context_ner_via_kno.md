---
title: >-
  [Paper Note] KDR-Agent: A Multi-Agent LLM Framework for Multi-Domain Low-Resource In-Context NER via Knowledge Retrieval
description: >-
  [AAAI 2026][LLM Agent][Named Entity Recognition] This paper proposes KDR-Agent, a multi-agent framework in which a central planner coordinates three specialized agents—knowledge retrieval, contextual disambiguation…
tags:
  - "AAAI 2026"
  - "LLM Agent"
  - "Named Entity Recognition"
  - "Multi-Agent Collaboration"
  - "Knowledge Retrieval"
  - "Low-Resource NER"
  - "Entity Disambiguation"
  - "Contrastive Demonstrations"
  - "Reflective Error Correction"
date: 2026-05-08
content_hash: 2b47571ce79ae715
---

# KDR-Agent: A Multi-Agent LLM Framework for Multi-Domain Low-Resource In-Context NER via Knowledge Retrieval

**Conference**: AAAI 2026
**arXiv**: [2511.19083](https://arxiv.org/abs/2511.19083)
**Code**: [GitHub](https://github.com/MWXGOD/KDR-Agent)
**Area**: LLM Agent
**Keywords**: Named Entity Recognition, Multi-Agent Collaboration, Knowledge Retrieval, Low-Resource NER, Entity Disambiguation, Contrastive Demonstrations, Reflective Error Correction

## TL;DR

This paper proposes KDR-Agent, a multi-agent framework in which a central planner coordinates three specialized agents—knowledge retrieval, contextual disambiguation, and reflective error correction—combined with natural language type definitions and entity-level positive/negative contrastive demonstrations. Without any fine-tuning, KDR-Agent comprehensively outperforms zero-shot and few-shot baselines across 10 low-resource NER datasets spanning 5 domains (BC5CDR F1=82.47, WNUT-17 F1=80.78 on GPT-4o).

## Background & Motivation

Named Entity Recognition (NER) is a foundational task in information extraction, supporting downstream applications such as relation extraction and knowledge graph construction. Traditional supervised methods rely on large annotated corpora and fine-tuning, resulting in poor generalization in low-resource or new-domain scenarios. In-context learning (ICL) with LLMs enables NER via a small number of in-prompt examples without parameter updates; however, existing ICL-based NER approaches suffer from three critical limitations:

1. **Reliance on large annotated sets for retrieval** (Issue 1): Few-shot methods retrieve demonstrations from labeled corpora, which degrades under low-resource conditions due to insufficient annotations.
2. **Insufficient domain knowledge** (Issue 2): Zero-shot methods rely on the LLM's internal knowledge to interpret entity types, which is inadequate for emerging or specialized domains (e.g., biomedicine).
3. **Lack of external knowledge and disambiguation** (Issue 3): Existing methods focus solely on demonstration selection, overlooking the incorporation of external knowledge and the resolution of entity ambiguity (e.g., whether "Apple" refers to a company or a fruit).

## Core Problem

How can a multi-agent collaborative system introduce external knowledge and disambiguation mechanisms to improve LLM-based in-context NER performance across multiple domains under extremely scarce annotation conditions?

## Method

### Overall Architecture

KDR-Agent operates in two stages: **Stage 1: Knowledge-Enhanced Context Construction** and **Stage 2: Reflective Error Correction**. A central LLM planner identifies knowledge gaps and ambiguous mentions, then coordinates three specialized agents to perform knowledge retrieval, disambiguation, and self-correction.

### Key Designs

1. **Natural Language Type Definitions**: Concise natural language descriptions (with inclusion/exclusion criteria) are authored for each entity type as prompt inputs, replacing conventional label names. These definitions can be automatically distilled from annotation guidelines using an LLM, offering strong scalability and substantially reducing reliance on large annotated sets.

2. **Static Few-Shot Contrastive Demonstrations**: Rather than retrieval-based approaches, KDR-Agent employs a static set of demonstrations in which each example contains both correct annotations and deliberately constructed negative instances. Negatives are generated according to four error categories:
   - Boundary errors (e.g., "Barack" instead of "Barack Obama")
   - Type errors (e.g., labeling "Apple" as LOC rather than ORG)
   - Hallucinated entities (entities not present in the input text)
   - Omitted entities (valid entities that are missed)

   This contrastive design enables the model to explicitly learn to distinguish boundary and type confusion without requiring a large retrieval candidate pool.

3. **Central LLM Planner**: Scans the input text to (i) detect domain-specific concepts requiring external knowledge and generate Wikipedia retrieval queries, and (ii) identify entity mentions prone to type ambiguity and construct disambiguation prompts.

4. **Knowledge Retrieval Agent**: Executes Wikipedia searches for planner-generated queries and returns introductory passages from matched entries as factual knowledge snippets, providing contextual background for domain-specific mentions.

5. **Disambiguation Agent**: Performs context-aware reasoning over ambiguous mentions and generates natural language explanations (e.g., "Amazon here refers to the e-commerce company, not the Amazon River"), which are inserted into the prompt to assist type classification.

6. **Reflective Analysis Agent (Stage 2)**: Conducts structured self-evaluation of initial predictions, systematically analyzing four error categories (Span Error, Type Error, Spurious Detection, Omission), generating a diagnostic report with correction suggestions, and producing a final refined prediction.

### Loss & Training

KDR-Agent is entirely training-free; no LLM parameters are modified. All modules are implemented via prompt engineering and agent collaboration.

## Key Experimental Results

### Main Results: 10 Datasets × 3 LLM Backbones

| Method | Type | BC5CDR | NCBI | CoNLL-2003 | WNUT-17 | Overall Trend |
|--------|------|--------|------|------------|---------|---------------|
| ChatIE | ZS | 69.84 | 65.46 | 67.19 | 46.67 | Baseline |
| CMAS | ZS | 73.21 | 69.91 | 78.31 | 50.64 | Strong |
| Code-IE | FS | 77.61 | 71.97 | 83.01 | 69.91 | Strong baseline |
| **KDR-Agent** | FS | **82.47** | **79.41** | **83.34** | **80.78** | **Best overall** |

(GPT-4o backbone; F1 scores)

- KDR-Agent maintains comprehensive superiority on Qwen-2.5-72B and DeepSeek-V3 as well.
- The most pronounced gains are observed in the biomedical and social media domains—precisely the scenarios where external knowledge and disambiguation are most valuable.

### Ablation Study (GPT-4o, F1)

| Variant | NCBI | OntoNotes 5.0 | Twitter NER-7 |
|---------|------|---------------|---------------|
| KDR-Agent (Full) | 79.41 | 71.85 | 60.87 |
| − Reflective Error Correction | 75.91 | 70.17 | 57.81 |
| − Knowledge Retrieval Agent | 76.21 | 71.70 | 59.34 |
| − Disambiguation Agent | 75.49 | 70.73 | 55.81 |
| − Knowledge Retrieval + Disambiguation | 74.16 | 69.94 | 55.07 |
| − Negative Contrastive Demonstrations | 78.36 | 70.69 | 58.99 |

- The Disambiguation Agent contributes most to social media NER (−5.06 F1), reflecting the high ambiguity in social text.
- Knowledge Retrieval contributes most in biomedicine (−3.20 F1), where specialized terminology knowledge is critical.
- Reflective error correction yields consistent gains across all domains.

### Error Analysis (GPT-4o)

| Error Type | NCBI (w/o → w/ Reflection) | Twitter NER-7 (w/o → w/ Reflection) |
|------------|---------------------------|--------------------------------------|
| Span Error | 22.03% → 9.18% | 9.86% → 7.09% |
| Spurious Detection | 16.44% → 5.57% | 24.27% → 12.57% |
| Omission | 49.62% → 17.78% | 48.97% → 30.38% |

- The reflection stage achieves the most substantial correction of Omission errors.

## Highlights & Insights

- **Systematic multi-agent division of labor**: Each agent addresses a specific class of NER error source (insufficient knowledge, ambiguity, prediction errors), with clearly defined responsibilities.
- **Contrastive demonstrations are an elegant design choice**: Constructing negatives from four canonical error types enables the model to explicitly learn to avoid boundary and type confusion, outperforming simple positive-example retrieval.
- **Static demonstrations replace dynamic retrieval**: Eliminating the need for large annotated retrieval candidate pools fundamentally addresses the low-resource challenge.
- **Consistent cross-domain effectiveness**: Comprehensive validation across 10 datasets in 5 domains with 3 LLM backbones demonstrates strong generalizability.

## Limitations & Future Work

- **High inference cost**: Multi-agent multi-turn invocations combined with Wikipedia retrieval introduce substantially higher latency compared to single-turn prompting.
- **Limited Wikipedia coverage**: Non-English languages, emerging concepts, and highly specialized subfields may yield insufficient retrieval results.
- **English NER only**: Cross-lingual scenarios have not been evaluated.
- **Disambiguation quality depends on LLM competence**: If the LLM itself lacks domain understanding, the Disambiguation Agent's explanations may also be erroneous.
- **Demonstration selection strategy underexplored**: Efficient construction of contrastive demonstrations for new domains warrants further investigation.

## Related Work & Insights

- **vs. CMAS (ZS multi-agent NER)**: CMAS also employs multiple agents but focuses on automatic annotation and demonstration filtering without incorporating external knowledge or explicit disambiguation; KDR-Agent directly addresses these two core shortcomings of CMAS.
- **vs. GPT-NER/Code-IE (FS baselines)**: These methods rely on retrieving demonstrations from large labeled sets and degrade significantly under low-resource conditions; KDR-Agent replaces dynamic retrieval with static contrastive demonstrations and type definitions.
- **vs. C-ICL (contrastive learning for IE)**: C-ICL also employs contrastive demonstrations but operates at the sentence level; KDR-Agent constructs positive/negative pairs at the entity level, yielding more targeted supervision.

The paradigm of distributing different error types across specialized agents is generalizable to other information extraction tasks (relation extraction, event extraction). The strategy of constructing contrastive demonstrations—explicitly showing "what is wrong" alongside "what is correct"—is broadly applicable to ICL tasks. The reflect-then-correct paradigm (predict → self-evaluate → revise) represents a general design pattern for LLM agents.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Systematic multi-agent NER framework design; entity-level contrastive demonstrations are novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 10 datasets × 3 LLM backbones; complete ablation and error analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Modular design is clearly presented; problem-solution correspondence is explicit.
- **Value**: ⭐⭐⭐⭐ Directly applicable to low-resource multi-domain NER; framework is generalizable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] LLandMark: A Multi-Agent Framework for Landmark-Aware Multimodal Interactive Video Retrieval](llandmark_a_multi-agent_framework_for_landmark-aware_multimodal_interactive_vide.md)
- [\[ACL 2026\] Scaling External Knowledge Input Beyond Context Windows of LLMs via Multi-Agent Collaboration](../../ACL2026/llm_agent/scaling_external_knowledge_input_beyond_context_windows_of_llms_via_multi-agent_.md)
- [\[AAAI 2026\] ARCANE: A Multi-Agent Framework for Interpretable and Configurable Alignment](arcane_a_multi-agent_framework_for_interpretable_and_configurable_alignment.md)
- [\[AAAI 2026\] FinRpt: Dataset, Evaluation System and LLM-based Multi-agent Framework for Equity Research Report Generation](finrpt_dataset_evaluation_system_and_llm-based_multi-agent_framework_for_equity_.md)
- [\[AAAI 2026\] LieCraft: A Multi-Agent Framework for Evaluating Deceptive Capabilities in Language Models](liecraft_a_multi-agent_framework_for_evaluating_deceptive_capabilities_in_langua.md)

</div>

<!-- RELATED:END -->
