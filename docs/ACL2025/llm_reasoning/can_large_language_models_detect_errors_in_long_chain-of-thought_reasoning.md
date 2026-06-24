---
title: >-
  [Paper Note] Can Large Language Models Detect Errors in Long Chain-of-Thought Reasoning?
description: >-
  [ACL 2025][Reasoning][Long Chain-of-Thought] This paper introduces DeltaBench, the first benchmark dataset to systematically evaluate the quality of long CoT reasoning in o1-like models and the error detection capabilities of existing LLMs/PRMs. Through fine-grained human annotation of 1,236 samples, it reveals a sobering reality: o1-like models exhibit approximately 27% reasoning redundancy, 67.8% ineffective reflections, and even the strongest critic model, GPT-4-turbo-128k…
tags:
  - "ACL 2025"
  - "Reasoning"
  - "Long Chain-of-Thought"
  - "Error Detection"
  - "Process Reward Model (PRM)"
  - "Critic Model"
  - "o1-like Model"
date: 2026-05-08
content_hash: d24f71188f2b4a9f
---

# Can Large Language Models Detect Errors in Long Chain-of-Thought Reasoning?

**Conference**: ACL 2025  
**arXiv**: [2502.19361](https://arxiv.org/abs/2502.19361)  
**Code**: [https://github.com/OpenStellarTeam/DeltaBench](https://github.com/OpenStellarTeam/DeltaBench)  
**Area**: LLM Reasoning  
**Keywords**: Long Chain-of-Thought, Error Detection, Process Reward Model (PRM), Critic Model, o1-like Model

## TL;DR
This paper introduces DeltaBench, the first benchmark dataset to systematically evaluate the quality of long CoT reasoning in o1-like models and the error detection capabilities of existing LLMs/PRMs. Through fine-grained human annotation of 1,236 samples, it reveals a sobering reality: o1-like models exhibit approximately 27% reasoning redundancy, 67.8% ineffective reflections, and even the strongest critic model, GPT-4-turbo-128k, achieves only an F1 score of 40.8%.

## Background & Motivation
o1-like models (such as QwQ, DeepSeek-R1, and Gemini 2.0 Flash Thinking) have significantly enhanced the reasoning capabilities of LLMs by generating long Chain-of-Thought (CoT) steps. However, there is a lack of systematic evaluation regarding the quality and efficiency of these long CoTs: How much redundancy do they contain? Which steps contain errors? Are the reflection mechanisms effective?

At the same time, Process Reward Models (PRMs) and Critic models are increasingly important as tools for evaluating the quality of reasoning processes. However, how they perform on long CoT remains largely uninvestigated. Existing PRM benchmarks (such as ProcessBench) focus only on short CoT and evaluate only the first error or sample-level correctness, failing to meet the demand for fine-grained analysis of long CoT.

The core goals of this paper are: (1) to analyze the efficiency of long CoT generation in o1-like models; and (2) to measure the capabilities of existing PRM and Critic models in error detection on long CoT.

## Method

### Overall Architecture
The construction workflow of DeltaBench:
1. **Query Collection**: Extract problems across four domains—mathematics, coding, physics/chemistry/biology (PCB), and general reasoning—from multiple open-source datasets. Conduct deduplication using embedding clustering (NV-Embed-v2 + DBSCAN, yielding 17,510 unique queries), difficulty filtering (voting evaluation by 6 models), and uniform sampling of subcategories.
2. **Long CoT Generation**: Generate long CoT solutions using o1-like models such as QwQ-32B-Preview, DeepSeek-R1, and Gemini 2.0 Flash Thinking.
3. **Section Segmentation**: Segment long CoT by "\n\n", then use GPT-4 to identify the start/end steps of each segment and generate a summary, forming sections at the granularity of independent subtasks.
4. **Human Annotation**: Master's and Ph.D. graduates annotate each section across four dimensions: strategy transition, reasoning usefulness, reasoning correctness, and reflection efficiency.

### Key Designs

1. **Section-Level Evaluation Granularity**

    - Unlike the traditional step level (too many steps, difficult to annotate) or sample level (too coarse-grained), this work adopts a section-level approach where each section represents an independent subtask.
    - This aligns better with human cognitive patterns, reducing annotation costs while preserving fine granularity.
    - Each section is annotated across 4 dimensions: whether a strategy transition occurs, whether the reasoning is useful, whether the reasoning is correct (if errors exist, the first erroneous step plus explanation and correction must be annotated), and whether it contains a reflection and if that reflection is effective.

2. **Error Taxonomy**

    - Reasoning errors are categorized into 8 major categories and 23 specific types, including comprehension errors, reasoning errors, calculation errors, format errors, knowledge errors, reflection errors, summary errors, etc.
    - The error distribution varies significantly across different domains: mathematics is dominated by reasoning errors (25.3%); coding is dominated by reasoning and format errors; and PCB is dominated by comprehension and knowledge errors.

3. **PRM Evaluation Method**

    - Instead of using a fixed threshold (since the score distribution of long CoT differs greatly from short CoT), a Z-Score outlier detection method is adopted: $t = \mu - \sigma$.
    - Sections with scores below the threshold are predicted to be incorrect.
    - Macro-F1 is used as the evaluation metric to mitigate the imbalance between positive and negative samples.

### Evaluation Metrics
- For Critic Models: Recall, Precision, Macro-F1
- For PRMs: Z-Score-based section-level prediction + HitRate@k

## Key Experimental Results

### Main Results (Critic Model F1-Score)

| Model | Overall F1 | Math | Coding | PCB | General Reasoning |
|------|--------|------|------|-----|---------|
| GPT-4-turbo-128k | **40.76** | 37.56 | 43.06 | 45.54 | 42.17 |
| GPT-4o-mini | 37.82 | 33.26 | 37.95 | 45.98 | 46.39 |
| Doubao-1.5-Pro | 35.25 | 32.46 | 39.47 | 33.53 | 37.00 |
| DeepSeek-R1 | 28.43 | 24.17 | 29.28 | 34.78 | 35.87 |
| o1-preview | 26.97 | 22.19 | 28.09 | 33.11 | 35.94 |
| o1-mini | 19.89 | 16.71 | 21.70 | 20.37 | 26.94 |

### PRM Results

| Model | Recall | Precision | F1 |
|------|--------|-----------|-----|
| Qwen2.5-Math-PRM-7B | 30.30 | 34.96 | **29.22** |
| Qwen2.5-Math-PRM-72B | 28.16 | 29.37 | 26.38 |
| Llama3.1-8B-PRM-Deepseek | 11.70 | 15.59 | 12.02 |

### Ablation Study / Analysis Findings

| Analysis Dimension | Key Numbers | Explanation |
|---------|---------|------|
| Reasoning Redundancy Rate | ~27% | An average of 27% of sections contain useless reasoning |
| Effective Reflection Ratio | ~32.2% | 67.8% of reflections are ineffective |
| Computation Error Ratio (QwQ) | 17.9% | QwQ is noticeably weak in handling details |
| Fundamental Error Ratio | 23-25% | About a quarter of errors in QwQ and Gemini are fundamental errors |
| DeepSeek-R1 Self-Criticism Decline | 36% | Self-evaluation is 36% worse than evaluating others |

### Key Findings
- **o1-like models are poor at self-criticism**: DeepSeek-R1's self-evaluation F1 is 36% lower than its evaluation of others, with other o1-like models showing similar trends.
- **o1-like models gain no advantage as Critics**: The Critic F1 score of o1-preview is even lower than that of Qwen2.5-32B-Instruct.
- **Larger PRMs are not necessarily better**: Qwen2.5-Math-PRM-72B performs worse than the 7B version.
- **Critic performance declines significantly as CoT length increases**: From 1-3k tokens to 4-7k tokens, the F1 scores of all Critic models drop substantially.
- **PRMs are more robust to CoT length**: This is due to section-by-section evaluation, though their overall scores remain lower than those of Critic models.
- **Models are weakest at identifying strategy errors**: Compared to computation errors (which are identified relatively well), error-detecting capabilities for strategic mistakes are generally lacking.

## Highlights & Insights
- **First systematic evaluation of long CoT quality**: This fills a significant research gap, and both the data and findings are of direct value to understanding the mechanisms of o1-like models.
- **Section-level granularity design**: Achieves a proper balance between annotation cost and fine-grained evaluation.
- **Quantitative finding of "27% redundancy + 68% ineffective reflection"**: Provides strong evidence regarding the efficiency issues of o1-like models.
- **Reveals the deficiencies of PRMs in long CoT scenarios**: The strongest PRM achieves an F1 of only 29.22%, showing that existing PRMs are far from solving the long CoT evaluation problem.

## Limitations & Future Work
- The dataset scale is limited due to high-cost human annotation (1,236 samples), bottlenecking scalability.
- Human annotation inevitably introduces subjective bias, particularly in judging "reasoning usefulness."
- As a static benchmark, it cannot reflect the rapid progress of o1-like models in real time.
- The latest o1 and o3 models are not evaluated.
- **Research Idea**: One can explore training PRMs specifically for long CoT—existing PRMs are trained mainly on short CoT data and transfer poorly to long CoT; fine-tuning PRMs using the annotated data from DeltaBench could be a promising direction.

## Related Work & Insights
- ProcessBench (Zheng et al., 2024): A step-level PRM evaluation benchmark, but only for short CoT.
- CriticBench, CriticEval: Sample-level Critic evaluation, lacking support for fine-grained analysis of long CoT.
- PRM800K (Lightman et al., 2023): A classic process-supervision dataset.
- Implications of this paper's findings: (1) PRMs need to be redesigned and trained specifically for long CoT scenarios; (2) The reflection mechanisms of o1-like models are highly inefficient, leaving vast room for optimization; (3) Self-criticism capability is a critical bottleneck of these models.

## Rating
- Novelty: ⭐⭐⭐⭐ The first fine-grained quality analysis benchmark for long CoT, filling a critical gap.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers both PRM and Critic models, with multi-dimensional analysis (error types, CoT length, self- vs. cross-evaluation, reflection efficiency).
- Writing Quality: ⭐⭐⭐⭐ Clear structure, rich data visualization, and well-articulated findings.
- Value: ⭐⭐⭐⭐⭐ Offers important guiding significance for understanding and improving o1-like models; the dataset is open-sourced.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Unlocking General Long Chain-of-Thought Reasoning Capabilities of Large Language Models via Representation Engineering](glore_long_cot_representation.md)
- [\[ACL 2025\] DRT: Deep Reasoning Translation via Long Chain-of-Thought](drt_deep_reasoning_translation_via_long_chain-of-thought.md)
- [\[ACL 2025\] ProcessBench: Identifying Process Errors in Mathematical Reasoning](processbench_identifying_process_errors_in_mathematical_reasoning.md)
- [\[NeurIPS 2025\] Large Language Models Can Learn and Generalize Steganographic Chain-of-Thought under Process Supervision](../../NeurIPS2025/llm_reasoning/large_language_models_can_learn_and_generalize_steganographic_chain-of-thought_u.md)
- [\[ACL 2025\] Chain-of-Reasoning: Towards Unified Mathematical Reasoning in Large Language Models via a Multi-Paradigm Perspective](chain_of_reasoning_unified_math.md)

</div>

<!-- RELATED:END -->
