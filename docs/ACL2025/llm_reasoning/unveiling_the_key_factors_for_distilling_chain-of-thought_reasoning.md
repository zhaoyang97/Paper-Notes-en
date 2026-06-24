---
title: >-
  [Paper Note] Unveiling the Key Factors for Distilling Chain-of-Thought Reasoning
description: >-
  [ACL 2025][Reasoning][Chain-of-Thought] A systematic study of the three key factors influencing CoT distillation (granularity, format, and teacher models) reveals a non-monotonic relationship between SLM performance and granularity, demonstrates that format has minimal impact, and shows that stronger teachers do not always yield better students.
tags:
  - "ACL 2025"
  - "Reasoning"
  - "Chain-of-Thought"
  - "knowledge distillation"
  - "Reasoning Granularity"
  - "Small Language Models"
  - "Teacher-Student"
date: 2026-05-08
content_hash: 377ad50abfa6157a
---

# Unveiling the Key Factors for Distilling Chain-of-Thought Reasoning

**Conference**: ACL 2025  
**arXiv**: [2502.18001](https://arxiv.org/abs/2502.18001)  
**Code**: [https://github.com/EIT-NLP/Distilling-CoT-Reasoning](https://github.com/EIT-NLP/Distilling-CoT-Reasoning)  
**Area**: LLM Reasoning / CoT Distillation  
**Keywords**: Chain-of-Thought, knowledge distillation, Reasoning Granularity, Small Language Models, Teacher-Student

## TL;DR
A systematic study of the three key factors influencing CoT distillation (granularity, format, and teacher models) reveals a non-monotonic relationship between SLM performance and granularity, demonstrates that format has minimal impact, and shows that stronger teachers do not always yield better students.

## Background & Motivation
**Background**: CoT prompting significantly enhances the reasoning capabilities of LLMs, but incurs heavy computational overhead, necessitating the distillation of CoT capabilities into small language models (SLMs).

**Limitations of Prior Work**: The choices of teacher models and generation methods in CoT distillation are often arbitrary and lack systematic guidance.

**Key Challenge**: Do CoT strategies that are effective for LLMs (such as finer granularity and specific formats) apply equally well to SLMs?

**Goal**: What constitutes the most effective CoT supervision for training student models to acquire reasoning capabilities?

**Key Insight**: Drawing an analogy to human teaching, this study conducts systematic experiments across three dimensions: teacher selection, teaching granularity, and teaching format.

**Core Idea**: CoT distillation requires "tailoring education to individual needs"—customizing the granularity and teacher selection based on the capability of the student model.

## Method

### Overall Architecture
A comprehensive cross-experiment involving 4 teacher models × 7 student models × 6 granularity levels × multiple formats × 7 datasets.

### Key Designs
1. **Granularity Experiments (Granularity)**:

    - **Function**: Designing 6 granularity levels, ranging from minimal to highly detailed reasoning steps.
    - **Mechanism**: Using 1-shot prompting to control GPT-4o to generate CoT annotations with varying granularities.
    - **Design Motivation**: Testing whether SLMs benefit from finer granularity in the same way as LLMs.

2. **Format Experiments (Format)**:

    - **Function**: Testing different reasoning formats, such as natural language, Least-to-Most, and RaR.
    - **Mechanism**: Keeping the content constant while only altering the presentation structure of the reasoning chain.
    - **Design Motivation**: Investigating the sensitivity of SLMs to reasoning formats.

3. **Teacher Selection Experiments (Teacher)**:

    - **Function**: Comparing GPT-4o, Gemini-1.5-Flash, LLaMA-3-70B, and human annotations.
    - **Mechanism**: Keeping granularity and format fixed, while only varying the source of the teacher model.
    - **Design Motivation**: Verifying whether the hypothesis "the strongest teacher yields the best student" holds true.

### Loss & Training
Standard SFT: $\mathcal{L}_{distill} = \sum_i \mathcal{L}(S(x_i), \mathcal{C}_{T,g,f}(x_i) \oplus y_i)$

## Key Experimental Results

### Main Results (Impact of Granularity, GPT-4o as Teacher)

| Model | GSM8K L1 | GSM8K L3 | GSM8K L5 | GSM8K Best |
|------|---------|---------|---------|-----------|
| Gemma 2B | 49.66 | 53.37 | 53.42 | L5 (53.45) |
| LLaMA 3.2 3B | 59.59 | 62.57 | 62.29 | L4 (63.48) |
| BLOOM 3B | 18.20 | 23.81 | 22.47 | L3 (23.81) |

### Ablation Study (Granularity vs. Length)

| Setup | GSM8K Acc | Sequence Length |
|------|----------|---------|
| Level 1 | 47.61 | 100.93 |
| Level 1 + Padding | 46.62 | 143.43 |
| Level 5 | 52.92 | 138.16 |

### Key Findings
- **Finding 1**: The relationship between SLMs and granularity is non-monotonic; stronger student models benefit from fine granularity, whereas weaker student models prefer medium granularity.
- **Finding 2**: CoT formats strongly impact LLMs but have minimal effect on SLMs (due to SFT adaptation capabilities).
- **Finding 3**: Strong teachers do not always produce better students—human annotations have near-perfect accuracy but are often outperformed by LLM-generated CoTs (where diversity and complexity are more critical).

## Highlights & Insights
- The teaching analogy is natural and intuitive, with a clear breakdown across three dimensions: who teaches, what to teach, and how to teach.
- The decoupling design of the length vs. granularity experiment is ingenious.
- The finding that "human annotations are not always the best" is counter-intuitive but reasonable, as the diversity of LLM annotations compensates for accuracy.

## Limitations & Future Work
- Annotations were generated solely using 1-shot prompting; other generation strategies might yield different conclusions.
- The study does not explore hybrid granularity strategies (e.g., low granularity for simple questions and high granularity for difficult ones).
- The scale of student models is capped at 3B; models larger than 7B might exhibit different behaviors.

## Related Work & Insights
- **vs Magister et al. (2023)**: Early CoT distillation works did not investigate the influence of granularity.
- **vs Zong et al. (2023)**: They argued that stronger teachers are always better, which is disproved by this work.


## Supplementary Details
- Teacher models: GPT-4o, Gemini-1.5-Flash, LLaMA-3-70B, and human annotations.
- Student models: BLOOM (560M/1.1B/1.7B/3B), Gemma 2B, LLaMA 3.2 (1B/3B).
- Math datasets: SVAMP, GSM8K, AQuA-RAT, MATH.
- Commonsense reasoning datasets: CommonsenseQA, OpenBookQA, StrategyQA.
- Granularity levels from Level 1 to Level 6, controlled via prompt.
- CoT formats include: original CoT, Least-to-Most, RaR.
- Padding experiments demonstrate that a pure increase in sequence length is ineffective.
- Core conclusion: SLMs require customized CoT distillation strategies that align with their capabilities.
- While human annotations present high accuracy, they lack the diversity of LLM-generated ones.
- The "Only Answer" baseline can reveal the implicit pre-trained knowledge of student models.
- Weaker student models perform close to random guessing on complex datasets.
- 1-shot prompting is used to ensure consistency in granularity control.

## Rating
- Novelty: ⭐⭐⭐⭐ The first work to systematically study the three key factors of CoT distillation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ An extensive empirical study involving 4 teachers × 7 students × 7 datasets × 6 granularities.
- Writing Quality: ⭐⭐⭐⭐ The educational analogy framework is clear and the conclusions are actionable.
- Value: ⭐⭐⭐⭐ Provides a practical configuration guide for CoT distillation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Improve Vision Language Model Chain-of-thought Reasoning](improve_vlm_cot_reasoning.md)
- [\[ACL 2025\] Improving Chain-of-Thought Reasoning via Quasi-Symbolic Abstractions](improving_chain-of-thought_reasoning_via_quasi-symbolic_abstractions.md)
- [\[ACL 2025\] DRT: Deep Reasoning Translation via Long Chain-of-Thought](drt_deep_reasoning_translation_via_long_chain-of-thought.md)
- [\[ACL 2025\] Enhancing Chain-of-Thought Reasoning with Critical Representation Fine-tuning](enhancing_chain-of-thought_reasoning_with_critical_representation_fine-tuning.md)
- [\[ACL 2025\] Towards Better Chain-of-Thought: A Reflection on Effectiveness and Faithfulness](towards_better_chain-of-thought_a_reflection_on_effectiveness_and_faithfulness.md)

</div>

<!-- RELATED:END -->
