---
title: >-
  [Paper Note] Exploring the Impact of Instruction-Tuning on LLMs' Susceptibility to Misinformation
description: >-
  [ACL 2025][Social Computing][instruction-tuning] This paper presents the first systematic study on how instruction-tuning affects the susceptibility of LLMs to misinformation. The authors find that instruction-tuning shifts the model's trust from the assistant-role to the user-role, with susceptibility peaking when misinformation is presented as an independent user-turn, thereby revealing a "side effect" of instruction-tuning.
tags:
  - "ACL 2025"
  - "Social Computing"
  - "instruction-tuning"
  - "misinformation"
  - "sycophancy"
  - "knowledge conflict"
  - "user-role bias"
date: 2026-05-08
content_hash: 53d87265f92d576a
---

# Exploring the Impact of Instruction-Tuning on LLMs' Susceptibility to Misinformation

**Conference**: ACL 2025  
**arXiv**: [2507.18203](https://arxiv.org/abs/2507.18203)  
**Code**: None  
**Area**: Social Computing  
**Keywords**: instruction-tuning, misinformation, sycophancy, knowledge conflict, user-role bias

## TL;DR
This paper presents the first systematic study on how instruction-tuning affects the susceptibility of LLMs to misinformation. The authors find that instruction-tuning shifts the model's trust from the assistant-role to the user-role, with susceptibility peaking when misinformation is presented as an independent user-turn, thereby revealing a "side effect" of instruction-tuning.

## Background & Motivation

**Background**: Instruction-tuning enhances the instruction-following capabilities and safety of LLMs, serving as a standard procedure for LLM deployment.

**Limitations of Prior Work**: Instruction-tuning may excessively reinforce the model's reliance on user inputs, causing it to accept user-provided misinformation and generate hallucinations. Prior works observed this phenomenon but did not deeply analyze its causes.

**Key Challenge**: While instruction-tuning improves helpfulness, does it systematically increase susceptibility to misinformation?

**Goal**: By comparing base models and instruction-tuned models, this study aims to attribute the "increased susceptibility to misinformation" to instruction-tuning itself for the first time.

**Key Insight**: Leveraging the separation of user and assistant roles within the chat template to test the differences in impact when misinformation is presented under different roles.

**Core Idea**: Instruction-tuning shifts the model's attention from the assistant-role to the user-role, making user-provided misinformation more likely to be accepted.

## Method

### Overall Architecture
3 scenario designs (STQ / APD / UPD) $\rightarrow$ 6 instruction-tuned models + 4 corresponding base models $\rightarrow$ Farm dataset (containing misinformation) $\rightarrow$ MSR metric to measure susceptibility $\rightarrow$ Additional analyses on the impact of information length and system prompt warnings.

### Key Designs

1. **Three Scenarios (RQ1)**

    - **STQ (Single-Turn Query)**: Misinformation and the question are in the same user-turn $\rightarrow$ Baseline
    - **APD (Assistant-Provided Document)**: Misinformation is provided by the assistant in the previous turn
    - **UPD (User-Provided Document)**: Misinformation is provided by the user in the previous turn
    - **Design Motivation**: To isolate role effects—if UPD > APD, it indicates that the model trusts the user more.

2. **Base vs. Instruct Comparison (RQ2)**

    - Perform the same tests on both base and instruct versions of 4 open-source models.
    - **Design Motivation**: To attribute the effect to instruction-tuning itself.

3. **MSR Metric**

    - $\text{MSR}(\%) = |\text{correctly answered but deceived by misinformation}| / |\text{correctly answered questions}| \times 100$
    - Only calculates questions that the model originally "knows", excluding the confounding factor of knowledge deficit.

## Key Experimental Results

### Main Results -- RQ1: MSR of Instruction-Tuned Models

| Model | STQ | APD | UPD | UPD-APD Difference |
|------|-----|-----|-----|-----------|
| GPT-4o | ~25% | ~20% | **~30%** | +10% |
| Llama-3-8B-Instruct | ~30% | ~25% | **~35%** | +10% |
| Mistral-7B-Instruct | ~35% | ~30% | **~55%** | **+25%** |
| Qwen2.5-7B-Instruct | ~30% | ~35% | ~30% | -5% (Exception) |

### RQ2: Base vs. Instruct Ranking Shifts

| Model | Base Ranking | Instruct Ranking | Shift |
|------|----------|-------------|------|
| Llama-3 | APD > UPD > STQ | **UPD > STQ > APD** | Reversed |
| Llama-3.1 | APD > UPD > STQ | **UPD > STQ > APD** | Reversed |
| Mistral | APD > UPD > STQ | **UPD > APD > STQ** | Reversed |

### RQ3: Impact of Misinformation Length

| Length | UPD-APD Gap | Trend |
|------|-------------|------|
| 1 paragraph (Short) | +10-25% | UPD is significantly higher |
| 2 paragraphs (Medium) | +5-15% | Gap narrows |
| 3 paragraphs (Long) | +0-5% | **Approaching base model pattern** |

### Key Findings
- **UPD > APD**: Except for Qwen, all instruction-tuned models are more likely to be deceived by misinformation provided within the user-role.
- **Instruction-tuning is the causal driver**: The ranking for base models is APD > UPD (favoring assistant), which reverses to favoring user after instruction-tuning.
- **Increasing misinformation length mitigates the instruction-tuning effect**: Longer documents revert model behavior back to the base model pattern.
- **System prompt warnings are ineffective for open-source models**: GPT-4o's MSR drops by 69%, but models like Llama show almost no change.
- **Independent user-turn amplifies the effect**: UPD > STQ, indicating that presenting misinformation as an independent user-turn is more hazardous than embedding it within the query.

## Highlights & Insights
- **First study to attribute susceptibility to misinformation directly to instruction-tuning**: Base models favor the assistant-role, while instruction-tuned models lean towards the user-role. This provides clear causal evidence.
- **The finding of "longer is safer" offers practical implications**: Short, concise misinformation is more dangerous than lengthy texts.
- **The ineffectiveness of system prompt warnings on open-source models** exposes security vulnerabilities in real-world deployment.

## Limitations & Future Work
- Limited scale, only utilizing the Farm dataset.
- Lack of analysis on how specific training data content impacts susceptibility.
- Future directions: Mitigation strategies (e.g., role-aware training) and testing in broader scenarios.

## Related Work & Insights
- **vs. Xie et al. (2024)**: While they found that LLMs highly accept external information, this work further attributes this behavior to instruction-tuning.
- **vs. Wei et al. (2023)**: They pointed out that instruction-tuning increases sycophancy. This work validates this insight within the context of misinformation.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First systematic attribution of misinformation susceptibility to instruction-tuning; the role-separation experimental design is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 6 instruction-tuned models + 4 base models $\times$ 3 scenarios $\times$ 3 datasets.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The research questions unfold progressively with rigorous logic.
- **Value**: ⭐⭐⭐⭐⭐ Offers direct insights for secure LLM deployment and improvements in instruction-tuning methodologies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] How does Misinformation Affect Large Language Model Behaviors and Preferences?](how_does_misinformation_affect_large_language.md)
- [\[ACL 2025\] Exploring Multimodal Challenges in Toxic Chinese Detection: Taxonomy, Benchmark, and Findings](exploring_multimodal_challenges_in_toxic_chinese_detection_taxonomy_benchmark_an.md)
- [\[ACL 2025\] A Survey on Proactive Defense Strategies Against Misinformation in Large Language Models](a_survey_on_proactive_defense_strategies_against_misinformation_in_large_languag.md)
- [\[ACL 2025\] Exploring Gender Bias in Large Language Models: An In-depth Dive into the German Language](exploring_gender_bias_in_large_language_models_an_in-depth_dive_into_the_german_.md)
- [\[ACL 2025\] BanStereoSet: A Dataset to Measure Stereotypical Social Biases in LLMs for Bangla](banstereoset_a_dataset_to_measure_stereotypical_social_biases_in_llms_for_bangla.md)

</div>

<!-- RELATED:END -->
