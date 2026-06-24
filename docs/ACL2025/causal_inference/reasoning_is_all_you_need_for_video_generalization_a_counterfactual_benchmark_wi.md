---
title: >-
  [Paper Note] Reasoning is All You Need for Video Generalization: A Counterfactual Benchmark with Sub-question Evaluation
description: >-
  [ACL 2025][Causal Inference][Counterfactual Reasoning] This paper proposes COVER (COunterfactual VidEo Reasoning), a multi-dimensional video counterfactual reasoning benchmark. It classifies evaluation tasks into four quadrants comprising 13 categories across two dimensions (abstract-concrete and perception-cognition). By decomposing complex questions into sub-questions (necessary conditions), the benchmark reveals that sub-question accuracy is strongly correlated with counte…
tags:
  - "ACL 2025"
  - "Causal Inference"
  - "Counterfactual Reasoning"
  - "Video QA"
  - "Sub-question Decomposition"
  - "Multimodal Evaluation"
  - "Benchmark"
date: 2026-05-08
content_hash: 138a4aadcafc6971
---

# Reasoning is All You Need for Video Generalization: A Counterfactual Benchmark with Sub-question Evaluation

**Conference**: ACL 2025  
**arXiv**: [2503.10691](https://arxiv.org/abs/2503.10691)  
**Code**: [https://github.com/gongyifan-hash/COVER-Benchmark](https://github.com/gongyifan-hash/COVER-Benchmark)  
**Area**: Causal Reasoning / Video Understanding  
**Keywords**: Counterfactual Reasoning, Video QA, Sub-question Decomposition, Multimodal Evaluation, Benchmark

## TL;DR
This paper proposes COVER (COunterfactual VidEo Reasoning), a multi-dimensional video counterfactual reasoning benchmark. It classifies evaluation tasks into four quadrants comprising 13 categories across two dimensions (abstract-concrete and perception-cognition). By decomposing complex questions into sub-questions (necessary conditions), the benchmark reveals that sub-question accuracy is strongly correlated with counterfactual reasoning ability, and enhancing reasoning capacity is the key to improving robustness in video understanding.

## Background & Motivation

**Background**: Multimodal Large Language Models (MLLMs) have achieved significant progress in video understanding, with related benchmarks (such as Video-MME and MVBench) evaluating abilities like temporal reasoning and spatio-temporal recognition.

**Limitations of Prior Work**: (1) Existing benchmarks rarely evaluate counterfactual reasoning, which refers to the ability to infer "what if a different event had occurred"; (2) pre-existing counterfactual benchmarks (e.g., CRIPP-VQA, VITATECS) focus only on the robustness of specific sub-tasks, lacking systematic evaluation from perception to cognition and from concrete to abstract; (3) there is a lack of fine-grained analysis mechanisms for the reasoning process.

**Key Challenge**: Existing evaluations only assess the correctness of the final answer, failing to diagnose which step in the reasoning chain broke down—whether it was a perception error or a cognitive reasoning error.

**Goal**: To build a systematic and multi-dimensional counterfactual video reasoning benchmark that supports fine-grained, sub-question level evaluation.

**Key Insight**: To decompose counterfactual questions into multiple sub-questions (necessary conditions); if the sub-questions are answered correctly but the summary question is wrong, it indicates insufficient reasoning integration capabilities.

**Core Idea**: Sub-question accuracy is a strong predictor of counterfactual reasoning performance—structured reasoning ability directly determines the robustness of video understanding.

## Method

### Overall Architecture
COVER is a benchmark and does not propose a new model. The core lies in its task taxonomy and the sub-question decomposition mechanism.

### Key Designs

1. **Four-Quadrant Task Taxonomy**:

    - **Abstract × Perception (A&P)**: Emotion Recognition (1 category)
    - **Concrete × Perception (C&P)**: Counting, Color, Direction, Size, Shape, Material, Position (7 categories)
    - **Concrete × Cognition (C&C)**: Action Recognition, Object Recognition (2 categories)
    - **Abstract × Cognition (A&C)**: Action Prediction, Process Understanding, Social Relation (3 categories)
    - **Design Motivation**: Cognitive science research indicates that concrete concepts rely on multimodal perceptual simulations, while abstract concepts are primarily represented through linguistic symbolic operations. This taxonomy reveals differences in model capabilities across different cognitive levels.

2. **Sub-question Decomposition Mechanism (Core Innovation)**:

    - **Function**: Each counterfactual question is decomposed into multiple sub-questions, where each sub-question serves as a necessary condition for correctly reasoning out the original question.
    - **Mechanism**: For example, "Does the boy complete the actions in order in the reversed video?" $\rightarrow$ Sub-question Q1: "What is the first action in the reversed video?" Q2: "What is the last action?" Q3: "What is the order of the actions in between?"
    - **Design Motivation**: Sub-questions break down complex reasoning into verifiable steps, allowing precise localization of model failures along the reasoning chain.

3. **Three-Dimensional Evaluation Metrics**:

    - $\text{ori}_{acc}$: Original question accuracy (fundamental video understanding capability)
    - $\text{cf}_{acc}$: Counterfactual question accuracy (reasoning robustness)
    - $\text{sub}_{acc}$: Sub-question accuracy (reasoning process quality)

### Data Construction
- Approximately 2,800 videos + around 12K-13K QA pairs, and 2,923 high-quality counterfactual question pairs.
- Seed data: 146 videos + 150 QAs (manually designed) $\rightarrow$ expanded using GPT to 720-760 pairs per quadrant.
- Validated by 8 annotators + cross-validated by 3 experts.

## Key Experimental Results

### Main Results (Overall Accuracy)

| Model | ori_acc | cf_acc | sub_acc |
|------|---------|--------|---------|
| GPT-4o | 70.26 | 45.93 | 56.94 |
| Gemini 2.0 Flash | 77.18 | 46.90 | 62.92 |
| **InternVL2.5-78B** | 76.74 | **59.46** | **67.23** |
| LLaVA-Video-72B | 64.35 | 56.04 | 61.54 |
| InternVL2.5-8B | 74.31 | 57.75 | 61.65 |
| Qwen2-VL-7B | 71.83 | 46.90 | 58.40 |
| VILA-U-7B | 60.01 | 38.42 | 47.32 |

### Four-Quadrant Analysis (Taking InternVL2.5-78B as an Example)

| Quadrant | ori_acc | cf_acc | sub_acc |
|------|---------|--------|---------|
| Abstract × Cognition (A&C) | 72.88 | 59.60 | 57.67 |
| Concrete × Cognition (C&C) | 80.95 | 63.62 | 75.62 |
| Concrete × Perception (C&P) | 75.99 | 58.25 | 63.65 |
| Abstract × Perception (A&P) | 76.86 | 56.20 | 70.07 |

### Key Findings
- **Sub-question accuracy is strongly correlated with counterfactual reasoning**: Models with higher $\text{sub}_{acc}$ consistently achieve higher $\text{cf}_{acc}$, validating that "reasoning is the foundation of robust video understanding."
- **Counterfactual reasoning is universally challenging**: The $\text{cf}_{acc}$ of all models is far lower than their $\text{ori}_{acc}$ (with a 24.3% gap for GPT-4o), indicating that hypothetical scenario reasoning is a significant shortcoming of current MLLMs.
- **Open-source models can outperform closed-source counterparts**: InternVL2.5-78B ($\text{cf}_{acc}$ 59.46%) outperforms GPT-4o (45.93%), showing the vast potential of open-source models in structured reasoning.
- **Abstract × Cognition is the most challenging**: The $\text{sub}_{acc}$ in the A&C quadrant is the lowest (such as process understanding and action prediction), requiring high-order causal reasoning.
- **CoT is not always helpful**: Automatic CoT even decreases accuracy in some models, suggesting that structured sub-questions are more effective than free-form CoT.

## Highlights & Insights
- **Sub-question decomposition evaluation paradigm**: By decomposing into necessary conditions, one can precisely diagnose the failure point in the reasoning chain—whether it is a visual perception error or a logical reasoning error. This evaluation paradigm can be generalized to other reasoning benchmarks.
- **Cognitive science foundation of the four-quadrant classification**: This is not an arbitrary taxonomy, but is grounded in the concrete/abstract and perceptual/cognitive theories from cognitive science.
- **The thesis that "reasoning capability equals generalization capability"**: Elevated sub-question accuracy ($\uparrow$) correlates with enhanced counterfactual reasoning ($\uparrow$), which in turn improves the robustness of video understanding ($\uparrow$); reasoning is the key path towards generalization.

## Limitations & Future Work
- Part of the dataset is expanded and generated by GPT, which might introduce GPT-biased tendencies.
- The scale of 2,800 videos is moderate, with some sub-tasks (e.g., emotion recognition) having smaller sample sizes.
- The benchmark only evaluates multiple-choice QA and does not cover open-ended generation tasks.
- The assumption of necessity for the sub-questions might not always hold—answering all sub-questions correctly does not guarantee the final answer is correct.

## Related Work & Insights
- **vs Video-MME**: Video-MME evaluates general video understanding without counterfactual questions or sub-question decomposition.
- **vs CRIPP-VQA**: CRIPP-VQA focuses solely on physical attribute counterfactuals, whereas COVER covers 13 task categories ranging from perception to cognition.
- **vs CoFCA**: CoFCA processes images rather than videos and lacks a four-quadrant taxonomy.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Sub-question decomposition + four-quadrant taxonomy represent a systematic new contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluates 16 models with four-quadrant analysis, CoT comparison, and frame sampling analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with compelling cognitive science motivations.
- **Value**: ⭐⭐⭐⭐ Provides a standardized utility for evaluating counterfactual reasoning in MLLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] RE-IMAGINE: Symbolic Benchmark Synthesis for Reasoning Evaluation](../../ICML2025/causal_inference/re-imagine_symbolic_benchmark_synthesis_for_reasoning_evaluation.md)
- [\[ACL 2025\] CoA-Reasoning: Explorations on Counterfactual Analysis in Physical Reasoning of LVLMs](coa-reasoning_explorations_on_counterfactual_analysis_in_physical_reasoning_of_l.md)
- [\[ACL 2025\] Causal Graph based Event Reasoning using Semantic Relation Experts](causal_graph_based_event_reasoning_using_semantic_relation_experts.md)
- [\[NeurIPS 2025\] Counterfactual Reasoning for Steerable Pluralistic Value Alignment of Large Language Models](../../NeurIPS2025/causal_inference/counterfactual_reasoning_for_steerable_pluralistic_value_alignment_of_large_lang.md)
- [\[ACL 2025\] Counterfactual Explanations for Aspect-Based Sentiment Analysis](counterfactual_explanations_for_aspect-based_sentiment_analysis.md)

</div>

<!-- RELATED:END -->
