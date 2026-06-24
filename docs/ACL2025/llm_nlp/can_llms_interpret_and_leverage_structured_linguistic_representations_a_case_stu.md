---
title: >-
  [Paper Note] Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs
description: >-
  [ACL 2025][LLM (Other)] This study systematically evaluates the capability of LLMs to leverage Abstract Meaning Representation (AMR) in downstream tasks. It is found that AMR-augmented prompts significantly improve the zero-shot performance of Llama 3.1 on long-context tasks such as dialogue summarization (raising cosine similarity from 66% to 76%), whereas they typically degrade performance on short-context tasks.
tags:
  - "ACL 2025"
  - "LLM (Other)"
date: 2026-05-08
content_hash: 77c218405a2668a5
---

# Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs

**Conference**: ACL 2025  
**Code**: None  
**Area**: LLM / NLP  

## TL;DR

This study systematically evaluates the capability of LLMs to leverage Abstract Meaning Representation (AMR) in downstream tasks. It is found that AMR-augmented prompts significantly improve the zero-shot performance of Llama 3.1 on long-context tasks such as dialogue summarization (raising cosine similarity from 66% to 76%), whereas they typically degrade performance on short-context tasks.

## Background & Motivation

1. **LLMs excel in NLP tasks, but their depth of understanding remains questionable**: While LLMs showcase outstanding performance in tasks like translation and summarization, it remains unclear whether they can extract and utilize information from structured semantic representations.
2. **Effectiveness of AMR has been validated in traditional approaches**: Abstract Meaning Representation (AMR) has been proven to effectively enhance reasoning behavior in structure-aware NLP tasks, especially in long-context scenarios.
3. **Prior work relies on architectural modifications**: Most previous attempts to exploit AMR modify model architectures (e.g., text-graph attention, graph Transformers), which increases complexity and impedes generalization.
4. **Lack of direct evaluation on LLM's understanding of AMR**: No prior studies have systematically evaluated the capability of general-purpose LLMs to directly interpret linearized AMRs and how this ability varies across different task types.
5. **A new direction for prompt engineering**: Incorporating structured semantic information into prompts represents a low-cost, architecture-free enhancement strategy, yet its effectiveness boundaries have not been systematically explored.
6. **Divergent demands of long and short context tasks**: Tasks with different context lengths may benefit differently from structured representations, requiring a fine-grained experimental analysis.

## Method

### AMR Construction and Linearization

- Using IBM's transition-based neural parsers (AMR3-structbart-L and doc-sen-conll-amr-seed42 models), text is parsed into document-level AMR structures.
- The AMRs are linearized into flat text representations and fed into LLMs.

### Three Prompting Strategies

1. **Context-only (Baseline)**: Only the raw text context is provided.
2. **AMR-augmented**: Both the raw text and its corresponding linearized AMR are provided to test whether AMR can facilitate context understanding.
3. **AMR-only**: Only the linearized AMR is provided without the raw text to evaluate the LLM's capability to reason directly from AMRs.

### Task Settings

Six task types are covered: context regeneration (AMR-to-text), single-hop question answering (SQuAD 2.0), multi-hop reasoning (HotpotQA), dialogue summarization (SAMSum), sentence-level NLI (SNLI), and document-level NLI (DocNLI). For each task, zero-shot, 3-shot, and 5-shot experiments are conducted.

### Models

8-bit quantized, instruction-tuned models are used: Llama 3.1 (8B), Phi-3, and Mistral 7B. Additionally, rank-32 LoRA fine-tuning of Llama 3.1 is performed on the SAMSum dataset.

## Key Experimental Results

### AMR-to-text Regeneration Capability (LDC2020T02)

| Model | Shot | Cosine Similarity |
|------|--------|-----------|
| Llama 3.1 | 0-shot | 73% |
| Llama 3.1 | 3-shot | 80% |
| Llama 3.1 | 5-shot | **81%** |
| Phi-3 | 0-shot | 74% |
| Phi-3 | 5-shot | 76% |
| Mistral | 5-shot | 76% |

LLMs can effectively reconstruct raw text from linearized AMRs, with Llama 3.1 achieving 81% cosine similarity in the 5-shot setting.

### SAMSum Dialogue Summarization (Llama 3.1 Cosine Similarity)

| Prompting Strategy | 0-shot | 3-shot | 5-shot |
|---------------|--------|--------|--------|
| Context-only | 66% | ~74% | ~74% |
| AMR-augmented | **76%** | ~75% | ~75% |
| AMR-only | ~60% | ~70% | ~68% |

- AMR augmentation yields a significant 10-percentage-point improvement in the zero-shot scenario.
- In few-shot settings, the advantage of AMR augmentation narrows but persists.

### Short-Context Tasks (SQuAD 2.0, Llama 3.1 F1)

In contrast, AMR augmentation degrades performance in single-hop QA: 3-shot performance drops from 59% to 52%. AMR-only achieves a 48% F1 score in the 3-shot setting, but precipitously drops to 26% in the 5-shot setting, indicating that excessive AMR examples interfere with reasoning.

### NLI Tasks

Phi-3 achieves the best performance on SNLI. AMR augmentation significantly improves macro F1 in the zero-shot setting (27% to 39%), while context-only remains superior in the few-shot setting (82%).

## Highlights & Insights

- **Systematic and comprehensive evaluation framework**: A complete experimental matrix covering 6 tasks × 3 prompting strategies × 3 models × 3 shot settings.
- **Actionable key findings**: The clear and practical conclusion that AMR assists in long contexts but hinders in short contexts directly guides prompt design.
- **LLMs indeed comprehend AMR**: An 81% similarity in text reconstruction demonstrates that LLMs possess strong capabilities to interpret structured semantic representations.
- **Extensibility to other structured representations**: The methodological framework can be generalized to other structured formats, such as knowledge graphs or discourse representation structures.

## Limitations & Future Work

- **Lack of full parameter fine-tuning**: Only LoRA fine-tuning was conducted (which underperformed compared to few-shot prompting), lacking a systematic comparison with full parameter fine-tuning.
- **In-depth explanation of long-context advantages is missing**: The fundamental reasons why AMR is beneficial in long contexts (e.g., info compression, preservation of key information) are not thoroughly analyzed.
- **Limited model scale**: The evaluated models are constrained to the 7-8B scale, with no exploration of larger models (e.g., 70B).
- **No CoT prompting used in HotpotQA**: This restricts the fairness and persuasiveness of the multi-hop reasoning experiments.
- **DocNLI evaluated only on a subset of the test set**: Verification on the full test set is required to draw reliable conclusions.

## Related Work & Insights

- **AMR-to-text generation**: Zhu et al. (2019) leveraged graph structures to improve Transformer-based AMR generation quality; Koncel-Kedziorski et al. (2022) utilized graph Transformers for text generation from knowledge graphs.
- **AMR-augmented NLP tasks**: Hua et al. (2023) integrated AMR via text-graph attention to improve long dialogue summarization; Yang et al. (2024) incorporated AMR through a gating mechanism in dialogue evaluation.
- **LLM prompting techniques**: Chain-of-Thought prompting by Wei et al. (2023) and soft prompt tuning by Lester et al. (2021) point to future directions.
- **Parameter-efficient fine-tuning**: LoRA by Hu et al. (2021) and Adapters by Houlsby et al. (2019) present potential pathways for integrating structured representations with LLMs.

## Rating

- ⭐⭐⭐ Novelty: The evaluation framework is systematic, but does not introduce new models or methods, leaning more towards empirical analysis.
- ⭐⭐⭐⭐ Practicality: The findings can be directly utilized to guide prompt design and structured information utilization strategies.
- ⭐⭐⭐⭐ Experimental Thoroughness: A comprehensive experimental matrix of 6 tasks × multiple variables, featuring complete confidence intervals.
- ⭐⭐⭐⭐ Writing Quality: Standardized structure, rich visualizations, and explicit methodological descriptions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Zero-Shot Belief: A Hard Problem for LLMs](zero-shot_belief_a_hard_problem_for_llms.md)
- [\[ACL 2025\] Unintended Harms of Value-Aligned LLMs: Psychological and Empirical Insights](unintended_harms_of_value-aligned_llms_psychological_and_empirical_insights.md)
- [\[ACL 2025\] Can LLMs Understand Unvoiced Speech? Exploring EMG-to-Text Conversion with LLMs](can_llms_understand_unvoiced_speech_exploring_emg-to-text_conversion_with_llms.md)
- [\[ACL 2025\] PlanGenLLMs: A Modern Survey of LLM Planning Capabilities](plangenllms_planning_survey.md)
- [\[ACL 2025\] Culture is Not Trivia: Sociocultural Theory for Cultural NLP](culture_is_not_trivia_sociocultural_theory_for_cultural_nlp.md)

</div>

<!-- RELATED:END -->
