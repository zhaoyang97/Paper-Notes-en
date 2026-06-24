---
title: >-
  [Paper Note] Judging the Judges: Can Large Vision-Language Models Fairly Evaluate Chart Comprehension and Reasoning?
description: >-
  [ACL 2025][VLM Reasoning][LVLM-as-a-Judge] This work systematically evaluates 13 open-source small LVLMs ($\le 9\text{B}$ parameters) serving as judges for chart comprehension and reasoning tasks. It finds that some open-source models (e.g., LLaVA-Critic-7B) can achieve evaluation capabilities close to GPT-4 (about 80% agreement rate), though issues like positional bias and length bias remain prevalent.
tags:
  - "ACL 2025"
  - "VLM Reasoning"
  - "LVLM-as-a-Judge"
  - "Chart Comprehension"
  - "Evaluation Benchmark"
  - "Vision-Language Model"
  - "Bias Analysis"
date: 2026-05-08
content_hash: 39fb93e7a2844794
---

# Judging the Judges: Can Large Vision-Language Models Fairly Evaluate Chart Comprehension and Reasoning?

**Conference**: ACL 2025  
**arXiv**: [2505.08468](https://arxiv.org/abs/2505.08468)  
**Code**: [https://github.com/tahmedge/chart_lvlm_judge](https://github.com/tahmedge/chart_lvlm_judge)  
**Area**: Multimodal & VLM  
**Keywords**: LVLM-as-a-Judge, Chart Comprehension, Evaluation Benchmark, Vision-Language Model, Bias Analysis

## TL;DR

This work systematically evaluates 13 open-source small LVLMs ($\le 9\text{B}$ parameters) serving as judges for chart comprehension and reasoning tasks. It finds that some open-source models (e.g., LLaVA-Critic-7B) can achieve evaluation capabilities close to GPT-4 (about 80% agreement rate), though issues like positional bias and length bias remain prevalent.

## Background & Motivation

Charts are core carriers of data visualization, and related downstream tasks (chart QA, chart captioning, etc.) have developed rapidly in recent years. Large Vision-Language Models (LVLMs) show potential in these tasks, but their qualitative evaluation faces several key bottlenecks:

**High manual evaluation costs**: Evaluating open-ended answers requires substantial labor and time, while traditional text-similarity metrics (such as BLEU) cannot capture the semantic quality of answers.

**Privacy and deployment limitations**: Enterprises are reluctant to send proprietary data to closed-source APIs like OpenAI/Google, whereas compatible open-source models with strong reasoning (70B-400B) require extremely high computational resources.

**Lack of specialized evaluation**: Prior to this, there was no systematic study on whether small open-source LVLMs can effectively evaluate chart-related tasks.

**Core Problem**: Can small ($\le 10\text{B}$ parameters) open-source LVLMs serve as low-cost alternatives to GPT-4 as automatic evaluators for chart comprehension tasks?

## Method

### Overall Architecture

A standardized "LVLM-as-a-Judge" evaluation framework is designed, spanning a combinatorial matrix of judgment types (pairwise/pointwise) $\times$ reference types (with reference/without reference) $\times$ evaluation dimensions (factual accuracy/informativeness/relevance/multi-dimensional), generating a total of approximately 100k reference evaluation data points produced by GPT-4o and LLaVA-Critic-70B.

### Key Designs

1. **Rubric Design**: Evaluation criteria across four dimensions are defined. For pairwise evaluation, the judge needs to choose the better of the two answers; for pointwise evaluation, the judge rates them on a 1-5 Likert scale. Each evaluation requires a corresponding explanation, as prior studies show that the "explanation + judgment" paradigm improves judgment quality. This design ensures multi-angle coverage and explainability of the evaluation.

2. **Evaluation Data Construction**: Three datasets are utilized: OpenCQA (1.1k open-ended QA instances), VisText (1.2k instances each of L1 structural description and L2/L3 insight description), and the newly proposed Chart-Instruct-Eval (400 instruction-following evaluation instances). For the first two datasets, outputs from Gemini-1.0-Pro and Claude-3-Haiku are collected, with reference evaluation scores computed using GPT-4o and LLaVA-Critic-70B. Chart-Instruct-Eval is motivated by the need to fill the gap in instruction-following evaluation in the chart domain. It manually prepares one good and one poor answer for each sample, where the good answer fully complies with the instructions and the bad answer ignores the instructions but shares similar content.

3. **Bias Analysis Framework**: Two indicators are defined: positional bias (whether the judgment changes after swapping the order of the two answers) and length bias (whether incorrect selections correlate with the length of more verbose answers). This is a systematic test of evaluation fairness, directly affecting the reliability of evaluation results.

### Evaluation Metrics System

- Judgment Accuracy: The agreement rate between the judge and reference answers under the pairwise scenario.
- Error Distance: The mean absolute difference between the judge's score and reference score under the pointwise scenario.
- Positional Bias/Length Bias: Percentages measuring evaluation biases.
- Format Adherence: Whether the output strictly follows the JSON format requirements.
- Instruction Following Evaluation Accuracy: Whether the model can correctly evaluate the instruction-following capabilities of other models.

## Key Experimental Results

### Main Results (Pairwise Judgment Accuracy, the higher the better)

| Model | Params | OpenCQA Avg | VisText L1 Avg | VisText L2/L3 Avg |
|------|--------|-------------|----------------|-------------------|
| LLaVA-Critic-7B | 7B | **79.5** | **79.1** | **77.1** |
| LLaVA-Next-Mistral-7B | 7B | 75.9 | 75.1 | 75.1 |
| XGen-MM-Phi3-3.8B | 3.8B | 71.6 | 75.4 | 70.7 |
| Qwen2-VL-7B | 7B | 66.9 | 57.6 | 70.0 |
| InternLM-Xcomposer-7B | 7B | 64.5 | 72.0 | 75.6 |
| PaliGemma-3B | 3B | 0.0 | 0.0 | 0.0 |
| ChartGemma-3B | 3B | 0.0 | 0.0 | 0.0 |
| Idefics-9B | 9B | 20.3 | 20.9 | 24.3 |

### Pointwise Evaluation (Error Distance, the lower the better)

| Model | OpenCQA Avg | VisText L1 Avg | VisText L2/L3 Avg |
|------|-------------|----------------|-------------------|
| LLaVA-Critic-7B | **0.5** | **0.5** | **0.6** |
| Qwen2-VL-7B | 0.7 | 0.6 | 0.7 |
| InternLM-Xcomposer-7B | 0.9 | 0.9 | 0.7 |
| PaliGemma-3B | 5.0 | 5.0 | 5.0 |

### Bias and Instruction-Following Analysis

| Model | Length Bias | Positional Bias | Instruction-Following Eval | Format Adherence |
|------|----------|----------|-------------|---------|
| Qwen2-VL-7B | **21.5** | **35.8** | **87.0** | 98.6 |
| mPLUG-Owl3-7B | 21.9 | 42.5 | **93.5** | 98.9 |
| LLaVA-Critic-7B | 76.4 | 39.6 | 45.5 | **99.7** |
| LLaVA-Next-Mistral-7B | 71.8 | 77.0 | 27.0 | 98.9 |

### Key Findings

1. **LLaVA-Critic-7B is the best judge** but heavily favors longer answers (length bias of 76.4%)—exhibiting the highest accuracy but raising concerns about fairness.
2. **Model size does not dictate evaluation ability**: The 3.8B XGen-MM outperforms the 9B Idefics. PaliGemma/ChartGemma completely fail due to their inability to follow the evaluation instruction format.
3. **Instruction-following evaluation is a blind spot**: LLaVA-Critic, which performs best in pairwise/pointwise evaluations, achieves only 45.5% on instruction-following evaluation, whereas mPLUG-Owl3 reaches 93.5%.
4. **Reference information has limited impact**: The difference in evaluation accuracy between 'with reference' and 'without reference' setups is not statistically significant ($p > 0.05$).

## Highlights & Insights

- **First systematic evaluation of LVLM-as-a-Judge in the chart domain**: Covers 13 models, 3 datasets, and various evaluation dimensions with a research design that is highly rigorous.
- **The paradox of 'accurate yet biased'**: LLaVA-Critic achieves the highest accuracy but suffers from the most severe length bias, reminding us that judgment accuracy and fairness must be considered separately.
- **New Chart-Instruct-Eval benchmark**: Fills the gap in instruction-following evaluation within the chart domain, revealing the weakness of most models in this aspect.
- **Human evaluation validation**: The correlation between two human annotators and LLaVA-Critic-70B is higher than that with GPT-4o, verifying the feasibility of using open-source models as alternative annotators.

## Limitations & Future Work

- Only GPT-4o and LLaVA-Critic-70B are used as reference evaluation standards, which themselves may introduce biases.
- The potential of fine-tuning small LVLMs specifically for chart evaluation tasks remains unexplored.
- The bias analysis is somewhat superficial, without an in-depth exploration of the root causes of biases and mitigation strategies.
- The tested chart types and complexity are limited; evaluation capabilities for more complex interactive charts or 3D charts are not covered.

## Related Work & Insights

- Aligns with the research direction of general multimodal evaluation models such as Prometheus-VL and LLaVA-Critic, but focuses on the vertical domain of charts.
- Insight: When choosing an automatic evaluation judge, one must not only focus on accuracy but also systematically examine biases; different task types may require different optimal judges.
- The dataset of 100k evaluation records itself can serve as training data to fine-tune specialized chart-evaluation models.

## Rating

- **Novelty**: ⭐⭐⭐ — Methodologically, it represents systematic evaluation research rather than proposing a new method, but it is a pioneering study in the chart domain.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 13 models, 3 datasets, and multi-dimensional analysis (accuracy/bias/instruction-following/format adherence), which is extremely comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, intuitive presentation of experimental results, and in-depth analysis.
- **Value**: ⭐⭐⭐⭐ — Provides a practical guide for automatic evaluation in the chart domain, and the bias analysis offers insightful warnings to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Benchmarking and Improving Large Vision-Language Models for Fundamental Visual Graph Understanding and Reasoning](benchmarking_and_improving_large_vision-language_models_for_fundamental_visual_g.md)
- [\[ACL 2025\] VReST: Enhancing Reasoning in Large Vision-Language Models through Tree Search and Self-Reward Mechanism](vrest_tree_search_vlm_reasoning.md)
- [\[ACL 2025\] Chart-based Reasoning: Transferring Capabilities from LLMs to VLMs](chart-based_reasoning_transferring_capabilities_from_llms_to_vlms.md)
- [\[ACL 2026\] Position: Multimodal Large Language Models Can Significantly Advance Scientific Reasoning](../../ACL2026/vlm_reasoning/position_multimodal_large_language_models_can_significantly_advance_scientific_r.md)
- [\[ICLR 2026\] SpatiaLab: Can Vision-Language Models Perform Spatial Reasoning in the Wild?](../../ICLR2026/vlm_reasoning/spatialab_can_vision-language_models_perform_spatial_reasoning_in_the_wild.md)

</div>

<!-- RELATED:END -->
