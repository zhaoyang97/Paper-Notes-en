---
title: >-
  [Paper Note] MultiCogEval: Evaluating LLMs Across Multi-Cognitive Levels
description: >-
  [ICML 2025][LLM Evaluation][Multi-Cognitive Levels] Inspired by Bloom's Taxonomy, this work proposes a multi-cognitive level evaluation framework, MultiCogEval, to assess the medical capabilities of LLMs across three levels: knowledge mastery, comprehensive application, and situational problem-solving. The findings reveal that the performance of all models decreases significantly as cognitive complexity increases, and model scale becomes more critical at higher levels.
tags:
  - "ICML 2025"
  - "LLM Evaluation"
  - "Multi-Cognitive Levels"
  - "Bloom's Taxonomy"
  - "Medical AI"
  - "Clinical Reasoning"
date: 2026-05-08
content_hash: 81c7f33994c8852f
---

# MultiCogEval: Evaluating LLMs Across Multi-Cognitive Levels

**Conference**: ICML 2025  
**arXiv**: [2506.08349](https://arxiv.org/abs/2506.08349)  
**Code**: [https://github.com/THUMLP/MultiCogEval](https://github.com/THUMLP/MultiCogEval)  
**Area**: LLM Evaluation  
**Keywords**: LLM Evaluation, Multi-Cognitive Levels, Bloom's Taxonomy, Medical AI, Clinical Reasoning

## TL;DR

Inspired by Bloom's Taxonomy, this work proposes a multi-cognitive level evaluation framework, MultiCogEval, to assess the medical capabilities of LLMs across three levels: knowledge mastery, comprehensive application, and situational problem-solving. The findings reveal that the performance of all models decreases significantly as cognitive complexity increases, and model scale becomes more critical at higher levels.

## Background & Motivation

### Key Challenge

**Key Challenge**: While GPT-4 achieves 90%+ on MedQA, a significant gap still exists in actual clinical diagnosis and treatment.

### 2. Limitations of Prior Work

Most benchmarks only utilize QA to test knowledge mastery, lacking a systematic evaluation framework across multiple cognitive levels.

### Limitations of Prior Work

**Limitations of Prior Work**: Medical education follows a structured path: first memory and understanding $\rightarrow$ then comprehensive application $\rightarrow$ finally actual problem-solving. LLM evaluation should also be stratified accordingly.

## Method

### Three Cognitive Levels

**Level 1: Preliminary Knowledge Mastery** (Remember/Understand)
- Multiple-choice QA, testing memory and understanding.

**Level 2: Comprehensive Knowledge Application** (Apply/Analyze)
- Clinical case analysis requiring the integration of multiple knowledge points.

**Level 3: Situational Problem-solving** (Evaluate/Create)
- Diagnosis and treatment decision-making in real-world clinical scenarios.

### Key Designs

- Cross-level knowledge coverage alignment: Ensuring different levels cover the identical scope of knowledge.
- Normalized metrics: Enabling meaningful comparisons across different cognitive levels.
- Coverage of 6 major LLM families (Llama, Qwen, Gemma, Phi, GPT, DeepSeek), ranging from 2B to 70B parameter scales.

## Key Experimental Results

### Main Results

| Model | Parameters | L1 Knowledge | L2 Application | L3 Solving | Decline |
|------|--------|--------|--------|--------|------|
| GPT-4o | - | 89.2 | 71.5 | 58.3 | -30.9 |
| Qwen2.5-72B | 72B | 85.1 | 67.3 | 53.8 | -31.3 |
| Llama-3.1-70B | 70B | 82.4 | 64.1 | 51.2 | -31.2 |
| Qwen2.5-7B | 7B | 68.3 | 48.2 | 35.1 | -33.2 |
| Gemma-2B | 2B | 45.2 | 29.8 | 18.5 | -26.7 |

### Model Scale Impact

| Scale Comparison | L1 Difference | L3 Difference | Description |
|---------|--------|--------|------|
| 7B vs 70B+ | +16.8 | +22.5 | Scale is more critical at higher levels |
| 2B vs 7B | +23.1 | +16.6 | Difference is more pronounced at lower levels |

### Key Findings

- The performance of all models decreases by approximately 30 percentage points from L1 to L3.
- Model scale plays a more substantial role at higher cognitive levels.
- Medically fine-tuned models do not necessarily outperform general large language models at L3.

## Highlights & Insights

- **Evaluation Paradigm Innovation**: For the first time, Bloom's Taxonomy is introduced into LLM medical evaluation to provide a cognitive-level perspective.
- **Counter-Intuitive Finding**: Medically fine-tuned models do not always outperform general LLMs at high cognitive levels, which might be due to overfitting to traditional QA formats.
- **Clear Capability Profiling**: Provides a cross-level capability map for each LLM family, facilitating on-demand model selection.
- **Methodological Contribution**: Cross-level knowledge coverage alignment and metric normalization make comparison meaningful.

## Limitations & Future Work

- Only English medical content is covered; multilingual evaluation remains to be expanded.
- The evaluation criteria for L3 contain subjective components, requiring more clinical expert participation.
- Multimodal medical scenarios (medical imaging + text) are not covered.
- The differentiated impact of CoT/few-shot prompts across levels can be explored further.
- Extension to multi-cognitive level evaluations in other domains (e.g., law, finance) is promising.

## Related Work & Insights

- **vs MedQA**: MedQA only tests L1, whereas this work complements it with L2 and L3.
- **vs MIMIC-IV-Ext**: MIMIC-IV-Ext only tests L3, lacking comparison with lower cognitive levels.
- **vs Application of Bloom's Taxonomy in Education**: This work migrates a mature pedagogical framework into AI evaluation.
- **vs CLIMEDBench**: CLIMEDBench covers clinical scenarios but lacks systematic classification of cognitive levels.
- **vs General LLM Benchmarks (MMLU, etc.)**: These benchmarks do not distinguish cognitive levels, conflating knowledge with reasoning.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First introduction of a cognitive-level framework.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluates 6 LLM families across 3 levels.
- Writing Quality: ⭐⭐⭐⭐⭐ Natural integration of pedagogy and AI.
- Value: ⭐⭐⭐⭐⭐ Directly guides the evaluation of medical AI.
- Replicability: ⭐⭐⭐⭐⭐ Code and datasets are open-sourced.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Modeling Multi-Dimensional Cognitive States in Large Language Models under Cognitive Crowding](../../ACL2026/llm_evaluation/modeling_multi-dimensional_cognitive_states_in_large_language_models_under_cogni.md)
- [\[ACL 2025\] HomeBench: Evaluating LLMs in Smart Homes with Valid and Invalid Instructions Across Single and Multiple Devices](../../ACL2025/llm_evaluation/homebench_evaluating_llms_in_smart_homes_with_valid_and_invalid_instructions_acr.md)
- [\[ACL 2025\] From Tools to Teammates: Evaluating LLMs in Multi-Session Coding Interactions](../../ACL2025/llm_evaluation/from_tools_to_teammates_evaluating_llms_in_multi-session_coding_interactions.md)
- [\[ACL 2025\] EducationQ: Evaluating LLMs' Teaching Capabilities Through Multi-Agent Dialogue Framework](../../ACL2025/llm_evaluation/educationq_evaluating_llms_teaching_capabilities_through_multi-agent_dialogue_fr.md)
- [\[NeurIPS 2025\] On Evaluating LLM Alignment by Evaluating LLMs as Judges](../../NeurIPS2025/llm_evaluation/on_evaluating_llm_alignment_by_evaluating_llms_as_judges.md)

</div>

<!-- RELATED:END -->
