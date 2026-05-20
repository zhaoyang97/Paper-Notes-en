---
title: >-
  [Paper Note] ChartMuseum: Testing Chart Visual Reasoning in Large Vision-Language Models
description: >-
  [NeurIPS 2025][Multimodal VLM][Chart Understanding] This paper introduces ChartMuseum, a chart question-answering benchmark comprising 1…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "Chart Understanding"
  - "Visual Reasoning"
  - "Benchmark"
  - "VLM Evaluation"
  - "Chart QA"
date: 2026-05-08
content_hash: e53bafc733d19592
---

# ChartMuseum: Testing Chart Visual Reasoning in Large Vision-Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2505.13444](https://arxiv.org/abs/2505.13444)  
**Code**: [https://chartmuseum-leaderboard.github.io](https://chartmuseum-leaderboard.github.io)  
**Area**: Multimodal VLM
**Keywords**: Chart Understanding, Visual Reasoning, Benchmark, VLM Evaluation, Chart QA

## TL;DR

This paper introduces ChartMuseum, a chart question-answering benchmark comprising 1,162 expert-annotated questions and real-world charts from 184 distinct sources. It is the first benchmark to systematically distinguish visual reasoning from textual reasoning, revealing that the current strongest model, Gemini-2.5-Pro, achieves only 63.0% accuracy compared to 93% for humans, with visual reasoning performance lagging behind textual reasoning by 35%–55%.

## Background & Motivation

- **Existing chart QA benchmarks over-rely on textual reasoning**: On ChartQA, Claude-3.7-Sonnet achieves 74.1% accuracy using only extracted text information (without seeing the chart image), versus 87.4% with the image — indicating that the majority of questions do not require genuine visual reasoning.
    - On ChartMuseum, the same text-only approach yields only 15.2% (vs. 61.3% with the image), a gap of 46%, demonstrating that ChartMuseum genuinely evaluates visual reasoning.
- **Frontier models are approaching saturation on existing benchmarks**: Model accuracy on ChartQA is clustered between 85%–90%, making it difficult to differentiate model capabilities.
- **The distinction between visual and textual reasoning has been overlooked**: Chart understanding involves two types of reasoning — inferring directly from graphical relationships (visual reasoning) versus inferring from extracted text or numerical values (textual reasoning) — yet prior work does not explicitly distinguish between them.
- **A synthetic data case study exposes the problem**: The authors test models on synthetic charts containing no textual annotations; as visual complexity (number of overlays/subplots $n$) increases, model performance degrades significantly while human performance remains stable.

## Method

### Overall Architecture

ChartMuseum is a chart question-answering (Chart QA) benchmark dataset manually annotated by 13 computer science researchers. It contains 1,162 (image, question, short answer) tuples derived from 928 unique real-world charts sourced from 184 distinct websites. The dataset is split into dev/test = 162/1,000.

### Key Designs

1. **Distinguishing visual reasoning from textual reasoning**: The paper explicitly categorizes chart reasoning into two types:
    - **Visual Reasoning**: Inference drawn from graphical relationships that is difficult to express purely in natural language (e.g., judging the correlation between two variables in a scatter plot).
    - **Visual Extraction**: A subtype of visual reasoning that involves reading numerical values by interpreting visual elements (e.g., estimating a bar's value by comparing it against the y-axis scale).
    - **Textual Reasoning**: Logical, arithmetic, or comparative operations performed on already-extracted information, or directly reading textual annotations from the chart.
    - This distinction reveals a strong bias toward textual reasoning in existing benchmarks.

2. **Four-category question taxonomy**:
    - **Textual Reasoning (123 questions)**: Answerable almost entirely through textual reasoning.
    - **Visual Reasoning (510 questions)**: Primarily requires visual reasoning; the largest category.
    - **Text/Visual Reasoning (234 questions)**: Answerable via either textual or visual reasoning.
    - **Synthesis Reasoning (133 questions)**: Requires both textual and visual reasoning simultaneously.

3. **Multi-stage quality control pipeline**:
    - Stage 1: Selection of high-quality chart images.
    - Stage 2: Manual creation of question–answer pairs (no LLM assistance, no templates).
    - Stage 3: Independent reviewer verification of answer correctness.
    - Stage 4: Iterative refinement through discussion with annotators.
    - Each sample required an average of 20 minutes (10 min annotation + 5 min review + 5 min iteration), totaling approximately 400 hours.
    - Annotation guidelines: answer space of ≥4 options; answers must be objective and unambiguous; why/how/descriptive/compound questions are excluded.

### Loss & Training

This paper presents a benchmark evaluation study and does not involve model training. Evaluation employs LLM-as-a-Judge (GPT-4.1-mini) to assess answer equivalence. All questions have a single definitive answer; approximate matching with error tolerance is not used.

## Key Experimental Results

### Main Results

| Model | Visual (510) | Synthesis (133) | Visual/Text (234) | Text (123) | Overall (1000) |
|---|---|---|---|---|---|
| **Open-source small models** | | | | | |
| InternVL3-2B | 12.2 | 13.5 | 18.4 | 30.1 | 16.0 |
| Qwen2.5-VL-3B | 16.7 | 21.1 | 26.5 | 28.5 | 21.0 |
| **Open-source medium models** | | | | | |
| Qwen2.5-VL-7B | 19.4 | 24.8 | 36.3 | 41.5 | 26.8 |
| InternVL3-8B | 23.5 | 24.8 | 32.9 | 42.3 | 28.2 |
| Bespoke-MiniChart-7B | 26.3 | 32.3 | 41.0 | 54.5 | 34.0 |
| **Open-source large models** | | | | | |
| Qwen2.5-VL-32B | 29.0 | 36.1 | 46.2 | 62.6 | 38.1 |
| Pixtral-Large-124B | 31.6 | 36.1 | 40.6 | 65.9 | 38.5 |
| Qwen2.5-VL-72B | 30.4 | 35.3 | 42.3 | 68.3 | 38.5 |
| **Closed-source models** | | | | | |
| Gemini-1.5-Flash | 22.7 | 30.8 | 36.3 | 56.1 | 31.1 |
| GPT-4o | 31.8 | 45.1 | 50.9 | 65.9 | 42.2 |
| GPT-4.1 | 37.1 | 53.4 | 54.3 | 78.9 | 48.4 |
| Claude-3.5-Sonnet | 45.7 | 53.4 | 61.5 | 78.0 | 54.4 |
| Claude-3.7-Sonnet | 50.6 | 55.6 | 69.2 | 88.6 | 60.3 |
| **Reasoning models** | | | | | |
| o3 (high) | 50.4 | 63.2 | 69.7 | 85.4 | 60.9 |
| o4-mini (high) | 51.2 | 66.2 | 68.4 | 86.2 | 61.5 |
| Claude-3.7-Sonnet (think) | 52.5 | 56.4 | 71.8 | 86.2 | 61.7 |
| **Gemini-2.5-Pro** | **53.3** | **64.7** | **70.1** | **87.8** | **63.0** |
| **Human** | **98.2** | — | — | — | **93.0** |

### Ablation Study

**Text-extraction experiment comparing existing benchmarks**:

| Dataset | Text Extraction Only | With Image |
|---|---|---|
| ChartQA | 74.1% | 87.4% |
| ChartMuseum | 15.2% | 61.3% |

The gap between text-extraction-only and image-based performance on ChartMuseum reaches 46%, far exceeding ChartQA's 13%, confirming that ChartMuseum genuinely evaluates visual reasoning.

**Error analysis on visual question categories** (50 error instances sampled per model):

| Error Type | Claude-3.7-Sonnet | Gemini-2.5-Pro |
|---|---|---|
| Symbol Selection | 34% | 28% |
| Visual Comparison | 28% | 26% |
| Trajectory Tracking | 14% | 12% |
| X/Y Value Identification | 6% | 28% |
| Strategy Error | 16% | 2% |
| Textual Reasoning Error | 6% | 2% |

### Key Findings

1. **Large gap between closed-source and open-source models**: The best open-source model, Qwen2.5-VL-72B (38.5%), trails the best closed-source model, Gemini-2.5-Pro (63.0%), by 24.5 percentage points.
2. **Visual reasoning substantially lags behind textual reasoning**: All models score 35%–55% lower on the Visual category than on the Text category. For example, GPT-4.1 achieves 78.9% on Text but only 37.1% on Visual (a drop of 41.8%); Qwen2.5-VL-72B drops from 68.3% to 30.4% (a drop of 37.9%).
3. **Reasoning models yield limited gains**: Enabling extended thinking in Claude-3.7-Sonnet improves performance by only 1.4% (60.3%→61.7%), suggesting that the bottleneck lies in fundamental visual capabilities rather than reasoning chain length.
4. **Human visual reasoning is near-perfect**: Humans achieve 98.2% on the visual reasoning subset (56/57 correct), whereas the strongest model reaches only 53.3%.
5. **Specialized models still lag significantly**: Bespoke-MiniChart-7B substantially outperforms open-source models of comparable scale (34.0% vs. 26.8%/28.2%) but remains far behind closed-source models.
6. **Strategy errors**: 16% of Claude-3.7-Sonnet's errors are strategy errors — the model fails to adopt the visual reasoning "shortcut" and instead attempts to extract numerical values for computation, leading to incorrect answers.

## Highlights & Insights

- The **formal distinction between visual and textual reasoning** is the paper's most significant contribution; this framework enables quantification of the asymmetry between these two capabilities in LVLMs.
- The **"extraction-only" experiment** (Section 2.2) elegantly exposes the limitations of benchmarks such as ChartQA — 74% of questions can be answered correctly without ever viewing the chart.
- The **four-category visual task taxonomy** (Symbol Selection / Visual Comparison / Trajectory Tracking / X/Y Value Identification) provides concrete directions for future model improvement.
- The **discovery of strategy errors is particularly noteworthy**: models are over-reliant on textualized reasoning strategies and tend to extract numerical values and compute, even when a simple visual comparison would suffice — revealing a deep architectural bias in current LVLMs.
- Dataset annotation was performed entirely by humans (no LLM-generated questions), at 20 minutes per sample and approximately 400 hours in total, with rigorous quality control.

## Limitations & Future Work

- Only English-language charts and questions are included; multilingual settings are not covered.
- Only short-answer QA is evaluated; tasks such as summarization and open-ended generation are not addressed.
- Unanswerable questions are not included.
- The dataset scale (1,162 questions) is relatively modest, with limited samples in some subcategories.
- No concrete methods for improving model visual reasoning are proposed; the work is purely diagnostic.
- Future directions include designing targeted training data or architectural improvements based on the identified visual reasoning weaknesses.

## Related Work & Insights

- **Evolution of chart QA benchmarks**: FigureQA/DVQA (synthetic charts + template questions) → ChartQA (real charts + human-authored questions) → CharXiv/ChartQAPro (more complex but limited sources or model-generated questions) → ChartMuseum (multi-source + fully human-authored + reasoning-type distinction).
- **Root causes of visual reasoning difficulty**: visual encoder bottlenecks (Prismatic VLMs), misalignment in visual feature decoding, limited abstract visual reasoning capability, and difficulty recognizing features that resist textual description.
- **Limited effectiveness of CoT for visual reasoning**: Unlike the substantial gains observed in mathematics and code, extended thinking yields nearly no improvement on chart understanding, echoing findings that "thinking makes humans worse" in certain perceptual tasks.
- **Implication**: Future LVLMs must strengthen visual reasoning at the architectural level, rather than relying solely on extending reasoning chain length.

## Rating

| Dimension | Score | Comment |
|---|---|---|
| Problem Significance | ⭐⭐⭐⭐⭐ | Exposes systematic deficiencies in LVLM visual reasoning; the problem is precisely identified |
| Methodological Novelty | ⭐⭐⭐⭐ | The formal visual vs. textual reasoning distinction and four-category taxonomy are original |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ | 21 models + human baseline, multi-dimensional analysis, detailed error categorization |
| Writing Quality | ⭐⭐⭐⭐⭐ | Clear motivation chain: inadequate existing benchmarks → synthetic validation → new benchmark → comprehensive evaluation → error analysis |
| Value | ⭐⭐⭐⭐ | Provides a diagnostic tool and concrete directions for improving LVLM visual reasoning |
| Overall | 4.6/5 | A high-quality benchmark paper with precise problem formulation and complete experimental design |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Recognition through Reasoning: Reinforcing Image Geo-localization with Large Vision-Language Models](recognition_through_reasoning_reinforcing_image_geo-localization_with_large_visi.md)
- [\[NeurIPS 2025\] PhysVLM-AVR: Active Visual Reasoning for Multimodal Large Language Models in Physical Environments](physvlm-avr_active_visual_reasoning_for_multimodal_large_language_models_in_phys.md)
- [\[NeurIPS 2025\] CHOICE: Benchmarking the Remote Sensing Capabilities of Large Vision-Language Models](choice_benchmarking_the_remote_sensing_capabilities_of_large_vision-language_mod.md)
- [\[NeurIPS 2025\] MM-OPERA: Benchmarking Open-ended Association Reasoning for Large Vision-Language Models](mm-opera_benchmarking_open-ended_association_reasoning_for_large_vision-language.md)
- [\[NeurIPS 2025\] FlexAC: Towards Flexible Control of Associative Reasoning in Multimodal Large Language Models](flexac_towards_flexible_control_of_associative_reasoning_in_multimodal_large_lan.md)

</div>

<!-- RELATED:END -->
