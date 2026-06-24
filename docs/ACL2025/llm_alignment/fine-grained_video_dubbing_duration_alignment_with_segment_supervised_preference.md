---
title: >-
  [Paper Note] Fine-grained Video Dubbing Duration Alignment with Segment Supervised Preference Optimization
description: >-
  [ACL 2025][LLM Alignment][Video Dubbing] This paper proposes Segment Supervised Preference Optimization (SSPO), which models the duration alignment problem between translated text and source speech in video dubbing as segment-level preference optimization. By using sentence-by-sentence sampling and fine-grained DPO loss, it achieves duration consistency for each dialogue line while maintaining translation quality and output format.
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "Video Dubbing"
  - "Duration Alignment"
  - "Preference Optimization"
  - "DPO"
  - "Segment-level Supervision"
  - "Controllable Text Generation"
date: 2026-05-08
content_hash: 2ac8c746090589b3
---

# Fine-grained Video Dubbing Duration Alignment with Segment Supervised Preference Optimization

**Conference**: ACL 2025  
**arXiv**: [2508.08550](https://arxiv.org/abs/2508.08550)  
**Code**: [CcQunResearch/SSPO](https://github.com/CcQunResearch/SSPO)  
**Institution**: Alibaba Digital Media and Entertainment Group / Huangzhong University of Science and Technology / Beijing Jiaotong University  
**Area**: LLM Alignment  
**Keywords**: Video Dubbing, Duration Alignment, Preference Optimization, DPO, Segment-level Supervision, Controllable Text Generation

## TL;DR

This paper proposes Segment Supervised Preference Optimization (SSPO), which models the duration alignment problem between translated text and source speech in video dubbing as segment-level preference optimization. By using sentence-by-sentence sampling and fine-grained DPO loss, it achieves duration consistency for each dialogue line while maintaining translation quality and output format.

## Background & Motivation

Video dubbing systems usually consist of three cascaded subtasks: ASR $\rightarrow$ NMT $\rightarrow$ TTS. Due to differences in information density across languages (e.g., high information density in Chinese vs. low information density in English/Thai), the speech duration of translated texts often mismatches that of the original text, leading to audio-video desynchronization and severely affecting the viewing experience.

Limitations of prior work:

- **Traditional Methods** (such as AutoDubbing and VideoDubber): Designed for Seq2Seq models by modifying embeddings or introducing length prediction modules, which cannot be applied to LLMs.
- **Prompt Engineering** (e.g., GPT-4o, Claude 3.5): LLMs lack a direct perception of speech duration from text, and control via prompts alone yields limited effectiveness.
- **SFT**: Human-translated subtitles focus on narrative quality rather than speech duration; thus, SFT models cannot effectively handle duration alignment.
- **Standard DPO/RLHF**: Aligning preferences over the entire output has a granularity too coarse to achieve line-by-line duration control and is prone to disrupting the output format.

Key Challenge: The duration consistency metric $P(s_i, t_i)$ is non-differentiable, making it impossible to directly optimize LLMs using gradient descent.

## Method

### 1. Duration Consistency Metric

An asymmetric penalty function $P(s_i, t_i)$ is defined as follows:

- When the translation duration exceeds the source duration: An exponential penalty $\exp(\text{Dur}(t_i) - \text{Dur}(s_i))$ is applied, since exceeding the duration causes subtitles to overflow the timeline and is highly unacceptable.
- When the translation duration is shorter than the source duration: A linear penalty $\text{Dur}(s_i) - \text{Dur}(t_i)$ is applied, which is relatively lenient.
- Microsoft Edge TTS (edge-tts) is used to synthesize speech and obtain durations.
- Translation quality is evaluated using two reference-free translation models: KIWI-XXL and XCOMET (both with 10B parameters).

### 2. Segment-level Sampling Strategy (Algorithm 1)

For each sample in the query set (containing $n = 35$ lines of dialogue), $k$ translation candidates are sampled sentence-by-sentence:

1. Using the previously selected translations as prefixes, $k$ candidate translations are sampled from the SFT model.
2. After deduplication, the bottom 20% of candidates based on KIWI-XXL and XCOMET scores are discarded to ensure a baseline translation quality.
3. The *chosen* (minimum $P$) and *rejected* (maximum $P$) candidates are selected based on the $P$ metric.
4. Low-diversity lines are filtered: Lines with fewer than $e_1 = 4$ unique samples after deduplication or with a $P$ difference less than $e_2 = 0.08$ do not participate in optimization.

Example: For the source sentence "历史虽然会重演，但是人类是无法回到过去的。" (2.89s), among multiple English candidates, the one with the closest duration (2.93s) is selected as *chosen*, and the one with the largest duration discrepancy (3.19s) is selected as *rejected*.

### 3. Segment-level DPO Loss

- The standard DPO loss is independently calculated for each dialogue line $s_i$, where the prefix $p_i$ contains the prompt instructions and the chosen translations of all preceding lines.
- The DPO loss of each line only controls the translation duration of that specific line without affecting other lines.
- The total loss is the sum of the DPO losses across all lines.

### 4. Output Format Control

Two schemes are proposed to prevent output format collapse after training:

- **Token-level KL Divergence (TKLD)**: Adds a token-level KL divergence constraint to the loss function to prevent the policy model from deviating from the reference model.
- **LoRA Training** (Recommended): Low-Rank Adaptation naturally restricts the parameter update magnitude, requires less GPU memory, and maintains a format compliance rate close to 100%, though it converges slower and requires more iterations.

## Key Experimental Results

### Main Results: Duration Alignment Performance (Table 2)

The dataset used is the self-constructed PolySC dataset, with the test set consisting of 4 TV dramas.

| Method | zh→en P↓ | zh→en CR↑| zh→th P↓ | zh→th CR↑|
|------|------|------|------|------|
| Gold Reference (Human Translation) | 0.501 | 17.9% | 0.489 | 20.3% |
| GPT-3.5-Turbo (PE) | 0.526 | 17.7% | 0.567 | 20.7% |
| GPT-4o (PE) | 0.417 | 19.2% | 0.318 | 27.0% |
| Claude 3.5 Sonnet (PE) | 0.410 | 19.3% | 0.313 | 27.5% |
| AutoDubbing (SFT) | 0.388 | 21.0% | 0.334 | 27.8% |
| VideoDubber (SFT) | 0.344 | 22.2% | 0.314 | 29.0% |
| Qwen2.5-14B SFT | 0.423 | 20.2% | 0.362 | 26.4% |
| **Qwen2.5-14B SSPO** | **0.272** | **24.9%** | **0.198** | **36.1%** |
| Llama3.1-8B SFT | 0.389 | 20.7% | 0.370 | 26.4% |
| **Llama3.1-8B SSPO** | **0.263** | **25.9%** | **0.206** | **36.4%** |
| Alignment Bound (Theoretical Upper Bound) | 0.220 | 44.3% | 0.203 | 50.4% |

SSPO significantly reduces $P$ across all base models. For example, on Qwen2.5-14B, the $P$ value for zh→th decreases from 0.362 to 0.198, which is close to the theoretical upper bound of 0.203.

### Human Evaluation: Translation Quality (Table 4)

Four professional English/Spanish translators performed pairwise comparisons on 200 dialogue segments (each containing 20 lines).

| SSPO (Qwen2.5-14B) vs | Accuracy Win:Tie:Loss | Naturalness | Vividness | Overall |
|---|---|---|---|---|
| Gold Reference | 21:63:16 | 24:55:21 | 23:51:26 | 24:50:26 |
| Vanilla Qwen2.5-14B | 24:51:25 | 26:50:24 | 26:53:21 | 32:41:27 |
| GPT-4o | 25:51:24 | 19:58:22 | 23:48:29 | 23:49:28 |
| SFT Model | 23:48:29 | 21:54:25 | 24:49:27 | 24:50:26 |

SSPO significantly improves duration consistency without noticeably degrading translation quality. While LLM-based methods generally outperform human translation in terms of accuracy, they slightly underperform in vividness due to the lack of multimodal video/audio information.

### Ablation Study

- **Format Control (Table 5)**: Full parameter fine-tuning results in a drastic drop in format compliance rate (Qwen2.5: 81.2%/73.9%), whereas LoRA preserves it at 99.8%/99.7% and TKLD at 96.9%/97.2%.
- **beta Sensitivity (Figure 4)**: Smaller $\beta$ values yield better duration consistency but result in lower format compliance rates; a compromise value of $\beta = 0.5$ is chosen.
- **Data Scale (Figure 5)**: Approximately 10,000 dialogue lines (~600 prompt-response pairs, accounting for ~3% of PolySC) are sufficient to achieve major improvements; excess training data leads to a sharp drop in format compliance.

## Highlights & Insights

- **Novel Problem Modeling**: Formulates video dubbing duration alignment as a "local multi-segment preference optimization" task, proposing a brand-new CTG paradigm.
- **Fine-grained Control**: Employs sentence-by-sentence independent sampling and sentence-by-sentence DPO loss to achieve line-level duration alignment instead of global alignment, resolving the coarse-granularity issue in standard DPO.
- **Asymmetric Penalty Design**: Utilizes an exponential penalty for overtime translations and a linear penalty for shorter ones, which perfectly aligns with the practical requirements of dubbing.
- **Highly Data-Efficient**: Achieves alignment performance close to the theoretical upper bound with only 3% of the training data.
- **Strong Generalizability**: Demonstrates effectiveness across three base models (Llama3.1-8B, GLM-4-9B, Qwen2.5-14B) and four language pairs (zh→en/th/es, es→zh).

## Limitations & Future Work

- **Coarse TTS Duration Estimation**: Uses edge-tts synthesized durations as a proxy metric, ignoring actual dubbing factors such as character emotion, speech rate variations, etc.
- **Discrepancies in Upper Bounds across Language Pairs**: Differences in information density across various language pairs dictate the alignment ceiling; some language pairs inherently struggle with completely eliminating duration mismatches.
- **Lack of Multimodal Information**: The translation model is blind to video frames and audio emotions, resulting in a systematic gap in vividness compared to human translations.
- **High Sampling Cost**: Sentence-by-sentence sampling of $k$ candidates requires TTS synthesis and translation quality evaluation (using 10B parameter models), introducing significant overhead during training data construction.
- **Validated Only in Dubbing Scenarios**: While the concept of segment-level preference optimization holds generalizability potential (e.g., local alignment in conversation generation or code generation), this work has not yet explored them.

## Related Work & Insights

- **Length-Controllable Generation**: Kikuchi et al. (2016) utilized reward guidance during decoding; Lakew et al. (2019) injected length info into embeddings; Wu et al. (2023) developed VideoDubber with length prediction. However, these are designed for Seq2Seq models and are not applicable to LLMs.
- **Preference Optimization**: DPO (Rafailov et al., 2024) simplifies RLHF but operates at an overly coarse granularity; variants such as SimPO, KTO, and IPO suffer from similar gradient dilution issues.
- **Video Dubbing Systems**: Federico et al. (2020) proposed AutoDubbing to control translation verbosity; Wu et al. (2023) designed VideoDubber to build speech-duration-aware length-controlled NMT.

## Rating

- Novelty: ⭐⭐⭐⭐ — Formulates duration alignment as segment-level preference optimization, offering a unique perspective and defining a new "local multi-segment preference optimization" problem.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive coverage with multi-language pairs, multiple base models, human evaluations + automatic metrics + ablations + visualizations + case studies.
- Writing Quality: ⭐⭐⭐⭐ — Clear problem definition, solid mathematical notation, rigorous theoretical derivations, and an exhaustive appendix.
- Value: ⭐⭐⭐⭐ — Successfully addresses practical production issues in Alibaba Youku, with the segment-level preference optimization framework displaying potential for generalizability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] SDPO: Segment-Level Direct Preference Optimization for Social Agents](sdpo_segment-level_direct_preference_optimization_for_social_agents.md)
- [\[ACL 2025\] ASPO: Adaptive Sentence-Level Preference Optimization for Fine-Grained Multimodal Reasoning](aspo_adaptive_sentence-level_preference_optimization_for_fine-grained_multimodal.md)
- [\[ACL 2025\] Balancing the Budget: Understanding Trade-offs Between Supervised and Preference-Based Finetuning](balancing_the_budget_understanding_trade-offs_between_supervised_and_preference-.md)
- [\[ACL 2025\] Retrieval-Augmented Fine-Tuning With Preference Optimization For Visual Program Generation](retrieval-augmented_fine-tuning_with_preference_optimization_for_visual_program_.md)
- [\[ACL 2025\] Reverse Preference Optimization for Complex Instruction Following](reverse_preference_optimization_for_complex_instruction_following.md)

</div>

<!-- RELATED:END -->
