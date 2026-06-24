---
title: >-
  [Paper Note] An Empirical Study of Many-to-Many Summarization with Large Language Models
description: >-
  [ACL 2025][Text Generation][many-to-many summarization] This work presents the first systematic study of Large Language Model (LLM) performance on the Many-to-Many Summarization (M2MS) task. By integrating 8 datasets, the authors construct a benchmark containing 47.8K samples across 5 domains and 6 languages. Evaluating 18 LLMs reveals that zero-shot LLMs perform comparably to fine-tuned traditional models, and significantly outperform them after instruction tuning. However…
tags:
  - "ACL 2025"
  - "Text Generation"
  - "many-to-many summarization"
  - "multilingual"
  - "LLM"
  - "instruction tuning"
  - "factual consistency"
date: 2026-05-08
content_hash: a1eb93cfa6a6d1d4
---

# An Empirical Study of Many-to-Many Summarization with Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2505.12983](https://arxiv.org/abs/2505.12983)  
**Code**: None  
**Area**: Multilingual NLP / Text Summarization  
**Keywords**: many-to-many summarization, multilingual, LLM, instruction tuning, factual consistency

## TL;DR

This work presents the first systematic study of Large Language Model (LLM) performance on the Many-to-Many Summarization (M2MS) task. By integrating 8 datasets, the authors construct a benchmark containing 47.8K samples across 5 domains and 6 languages. Evaluating 18 LLMs reveals that zero-shot LLMs perform comparably to fine-tuned traditional models, and significantly outperform them after instruction tuning. However, factual consistency remains a critical bottleneck.

## Background & Motivation

**Background**: Many-to-Many Summarization (M2MS) requires models to summarize source documents in any language into summaries in any target language, combining both cross-lingual translation and text summarization capabilities.

**Limitations of Prior Work**: Existing M2MS studies predominantly utilize traditional pre-trained models (e.g., mBART), lacking a systematic exploration of LLM capabilities. Moreover, existing datasets are limited to single domains, hindering comprehensive evaluation.

**Key Challenge**: LLMs are naturally endowed with multilingual capabilities and theoretically should serve as excellent M2MS solvers, but this has not been fully verified in practice.

**Goal**: To systematically evaluate the zero-shot and instruction-tuning performance of LLMs in multi-domain, multilingual M2MS scenarios.

**Key Insight**: Integrate multi-source datasets to construct a unified benchmark, covering three paradigms: zero-shot, instruction tuning, and traditional models.

**Core Idea**: Comprehensively reveal the strengths (instruction-tuned models surpassing GPT-4) and weaknesses (exacerbated factual consistency issues) of LLMs in M2MS through 47.8K multi-domain, multilingual samples.

## Method

### Overall Architecture
1) M2MS samples are integrated from 8 datasets, covering 5 domains (News, Encyclopedia, Dialogue, Guide, Technology) and 6 languages (En, Cs, De, Fr, Zh, Uk); 2) 18 LLMs are evaluated in a zero-shot setting with meticulously designed prompts containing task instructions and in-context examples; 3) Open-source LLMs undergo instruction tuning (on 19.5K training samples); 4) Fine-grained human evaluation of factuality is conducted. The focus is on a comprehensive comparison across multiple experimental paradigms.

### Key Designs
1. **Data Integration**: Samples are selected from 8 datasets such as CrossSum, XWikis, and WikiLingua, covering 5 domains (News, Encyclopedia, Dialogue, Guide, Technology) and 6 languages (English, Czech, German, French, Chinese, Ukrainian). Low-quality samples are filtered out based on three intrinsic metrics: coverage, redundancy, and coherence.
2. **Data Contamination Control**: Target-instance-level contamination is calculated on the test set to ensure the contaminated sample ratio stays below 1%, ensuring a fair evaluation.
3. **Multi-Paradigm Evaluation**: A three-track parallel comparison is conducted: zero-shot (with meticulously designed prompts + in-context examples), instruction tuning (19.5K training samples), and fine-tuned traditional models (mBART-50/PISCES).

### Loss & Training
- Traditional models are trained using standard seq2seq objectives.
- LLM instruction tuning utilizes the instruction-response format of the training set.
- Evaluation metrics: ROUGE-1/2/L, BERTScore, and GPT-4o scoring (on a 5-point scale for conciseness, coherence, and relevance).

## Key Experimental Results

### Main Results (Zero-shot LLMs vs. Fine-tuned Traditional Models, Overall R1/RL/BS)

| Model | Overall R1/RL/BS |
|------|-----------------|
| GPT-4o (zero-shot) | 26.0 / 16.6 / 66.7 |
| GPT-4 (zero-shot) | 25.7 / 16.4 / 66.4 |
| GPT-3.5-turbo | 25.2 / 16.1 / 66.7 |
| Vicuna-13B-16k | 22.9 / 13.9 / 66.0 |
| Qwen2.5-14B | 22.1 / 13.1 / 65.4 |
| LLaMa-2-7B | 18.2 / 10.8 / 63.3 |

### Ablation Study (Cross-domain Performance, R1 Metric)

| Model | News | Encyc. | Dialogue | Guide | Tech. |
|------|------|--------|----------|-------|-------|
| GPT-4o | 19.8 | 27.9 | 29.5 | 25.1 | 34.2 |
| GPT-4 | 19.5 | 26.9 | 28.9 | 24.0 | 33.8 |
| Vicuna-13B-16k | 19.0 | 27.2 | 22.6 | 20.3 | 33.0 |
| Qwen2.5-14B | 18.4 | 25.8 | 22.0 | 18.5 | 32.6 |

### Key Findings
- Data scale: 19,530 train / 14,150 val / 14,150 test samples, covering 30 language pairs.
- Flores translation capability ranking: GPT-4o (29.1) > GPT-4 (27.7) > GPT-3.5 (22.0) > Qwen2.5-14B (19.2).
- Models supporting longer contexts (e.g., Vicuna-16k) exhibit a slight advantage in M2MS.
- Zero-shot LLMs can already compete with fine-tuned traditional models (mBART-50/PISCES), with GPT-4o achieving the overall best performance.
- After instruction tuning, open-source LLMs (e.g., Qwen-14B) can outperform zero-shot GPT-4 on automatic metrics.
- Instruction tuning does not sacrifice general task capabilities (MMLU scores remain stable).
- **Factual consistency is a key bottleneck**: Human evaluation shows that open-source LLMs produce more factual errors than GPT-4, and instruction tuning might exacerbate hallucinations.
- The technology domain (Tech) achieves the highest scores, while the news domain (News) is the most challenging.
- Multilingual translation capability (Flores score) is positively correlated with M2MS performance.

## Highlights & Insights
- This is the first large-scale systematic evaluation of LLMs' M2MS capabilities (evaluating 18 open-source and closed-source LLMs across various multilingual families including Chinese and English).
- Revealed the "double-edged sword" effect of instruction tuning: it boosts automatic metrics but may exacerbate hallucinations—providing a crucial warning for alignment research.
- Data contamination control is crucial for fair evaluation and represents an essential step in benchmark design in the LLM era.
- Multi-domain analysis (across 5 domains) provides fine-grained insights: the technology domain performs the best, while the news domain is the most difficult.
- The positive correlation between multilingual translation capability (Flores score) and M2MS performance is intuitive but systematically validated for the first time.
- The data integration of 47.8K samples covering 6 languages and 30 language pairs forms a highly valuable benchmark resource.
- Validated that instruction tuning does not compromise general capabilities (no drop in MMLU), mitigating concerns about catastrophic forgetting during fine-tuning.

## Limitations & Future Work
- The root cause of instruction tuning exacerbating hallucinations may stem from factual errors in the reference summaries within the training data, necessitating cleaner training data.
- The dataset only covers 6 languages (En, Cs, De, Fr, Zh, Uk), leaving low-resource and non-Latin languages unaddressed.
- Lacks evaluation of larger LLMs (such as 70B+ models, Mixtral, and other MoE architectures).
- Has not explored M2MS capabilities for long documents (>16K tokens).
- Factuality control methods for open-source LLMs are not thoroughly investigated; only a diagnostic of the problem is provided.
- There may still be a noticeable gap between automatic metrics (ROUGE/BERTScore) and human evaluation.

## Related Work & Insights
- CrossSum (Bhattacharjee et al., 2023) and PISCES (Wang et al., 2023c) are foundational works in M2MS, proving that M2MS outperforms independent CLS (Cross-Lingual Summarization).
- The hallucination issue is closely aligned with general LLM hallucination research (Zhang et al., 2023).
- The degradation of factuality in instruction tuning deserves attention in alignment research, potentially requiring a factuality reward.
- mBART-50 (Tang et al., 2021), as a traditional baseline, remains competitive, demonstrating the advantages of the encoder-decoder architecture in summarization tasks.

## Rating
- **Novelty**: ⭐⭐⭐ The task definition and method itself are not entirely new, but the first systematic LLM evaluation is valuable.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Highly comprehensive, covering 18 LLMs, 5 domains, 6 languages, and multiple evaluation approaches.
- **Writing Quality**: ⭐⭐⭐⭐ Rigorous experimental design and in-depth analysis; the explanation of data contamination control reflects the authors' meticulousness.
- **Value**: ⭐⭐⭐⭐ The revealed factual consistency issues provide crucial warnings for practical applications.
- **Overall Rating**: A classic empirical study paradigm. The benchmark contribution is significant, and the findings hold practical value for deploying open-source LLMs.
- **Reproducibility**: The data integration pipeline is clear and can be extended to more languages and domains.
- **Scalability**: Future work can explore factuality-aware instruction tuning to alleviate hallucination issues.
- **Open Question**: How can we design M2MS training data that does not introduce hallucination signals?
- **Impact**: Provides a standardized benchmark and methodology for evaluation of LLMs in the field of multilingual summarization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Theme-Explanation Structure for Table Summarization Using Large Language Models](theme-explanation_structure_for_table_summarization_using_large_language_models_.md)
- [\[ACL 2025\] Unveiling Attractor Cycles in Large Language Models: A Dynamical Systems View of Successive Paraphrasing](unveiling_attractor_cycles_in_large_language_models_a_dynamical_systems_view_of_.md)
- [\[ACL 2025\] Enhancing Text Editing for Grammatical Error Correction: Arabic as a Case Study](enhancing_text_editing_for_grammatical_error_correction_arabic_as_a_case_study.md)
- [\[ACL 2025\] Tell, Don't Show: Leveraging Language Models' Abstractive Retellings to Model Literary Themes](tell_dont_show_leveraging_language_models_abstractive_retellings_to_model_litera.md)
- [\[ACL 2025\] DTCRS: Dynamic Tree Construction for Recursive Summarization](dtcrs_dynamic_tree_construction_for_recursive_summarization.md)

</div>

<!-- RELATED:END -->
