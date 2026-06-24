---
title: >-
  [Paper Note] CoA-Reasoning: Explorations on Counterfactual Analysis in Physical Reasoning of LVLMs
description: >-
  [ACL 2025][Causal Inference][Counterfactual Reasoning] This paper proposes the CoA-Reasoning framework to systematically evaluate and enhance the causal understanding of Large Vision-Language Models (LVLMs) in physical world reasoning by constructing counterfactual scenarios, revealing significant limitations of existing models in counterfactual physical reasoning.
tags:
  - "ACL 2025"
  - "Causal Inference"
  - "Counterfactual Reasoning"
  - "Physical Reasoning"
  - "Large Vision-Language Models"
  - "Causal Analysis"
  - "Visual Commonsense"
date: 2026-05-08
content_hash: 30daf2f719586527
---

# CoA-Reasoning: Explorations on Counterfactual Analysis in Physical Reasoning of LVLMs

**Conference**: ACL 2025  
**Area**: Causal Reasoning  
**Keywords**: Counterfactual Reasoning, Physical Reasoning, Large Vision-Language Models, Causal Analysis, Visual Commonsense

## TL;DR
This paper proposes the CoA-Reasoning framework to systematically evaluate and enhance the causal understanding of Large Vision-Language Models (LVLMs) in physical world reasoning by constructing counterfactual scenarios, revealing significant limitations of existing models in counterfactual physical reasoning.

## Background & Motivation

**Background**: Large Vision-Language Models (LVLMs) such as GPT-4V, Gemini, and LLaVA perform exceptionally well on tasks like visual question answering, but their understanding of physical world laws remains limited. Physical reasoning requires understanding physical concepts such as gravity, collisions, fluids, and stability, which serves as the foundation for applying AI to embodied world tasks like robotics and autonomous driving.

**Limitations of Prior Work**: Existing evaluations of physical reasoning primarily focus on model performance in standard scenarios, failing to distinguish whether a model truly understands physical causal relationships. Models might make correct predictions based on superficial cues (e.g., "object at the edge of the table" $\rightarrow$ "will fall") rather than understanding the causal relationship between gravity and supporting surfaces. Counterfactual reasoning ("what if the conditions changed?") is the gold standard for testing causal understanding, but there is still a lack of a systematic counterfactual evaluation framework in the field of visual physical reasoning.

**Key Challenge**: The high accuracy of models in standard physical reasoning tests may be a case of "knowing the what but not the why" — they merely memorize correlations between specific visual patterns and outcomes rather than understanding the underlying physical causal mechanisms.

**Goal**: (1) Construct a counterfactual physical reasoning benchmark dataset containing "what if a certain condition changes, what will the outcome be?" style questions; (2) Evaluate the performance of existing LVLMs on counterfactual physical reasoning; (3) Propose training strategies to enhance models' counterfactual reasoning capabilities.

**Key Insight**: Counterfactual Analysis is a core tool in causal reasoning. By applying "virtual interventions" to key conditions in physical scenarios (e.g., changing the mass, shape, or position of objects) and requiring the model to predict the change in outcome, one can precisely test whether the model understands the true physical causal chain.

**Core Idea**: Construct the counterfactual physical reasoning dataset CoA-Bench, and propose the Chain-of-Analysis reasoning framework to guide LVLMs to first identify causal variables, analyze causal relationships, and finally predict counterfactual outcomes.

## Method

### Overall Architecture
The framework comprises two core components: (1) CoA-Bench counterfactual physical reasoning benchmark, which creates evaluation data by constructing counterfactual variants of real physical scenario images; (2) CoA reasoning strategy, a structured reasoning-chain method that guides the model along the path of "identifying causal variables $\rightarrow$ analyzing causal mechanisms $\rightarrow$ making counterfactual inferences."

### Key Designs

1. **Counterfactual Scenario Construction Pipeline**:

    - **Function**: Systematically generates high-quality counterfactual questions for physical reasoning scenarios.
    - **Mechanism**: Starting from real physical scenarios, identifying causal variables in the scene (such as object mass, surface friction, support point location, etc.), and then applying systematic "virtual interventions" to these variables (e.g., "if this ball becomes twice as heavy...", "if the ground becomes ice..."). Each intervention corresponds to a counterfactual question whose correct answer is derived through physical laws. Variable types cover mechanics (gravity, friction, elasticity), kinematics (velocity, trajectory), and statics (balance, stability).
    - **Design Motivation**: Testing the model's understanding of the effect of a specific variable by controlling the change of a single causal variable, enabling precise causal diagnosis.

2. **Chain-of-Analysis (CoA) Reasoning Strategy**:

    - **Function**: Guides LVLMs to perform counterfactual reasoning along the logical chain of causal analysis.
    - **Mechanism**: Decomposes counterfactual reasoning into three sub-steps: (a) causal variable identification: extracting key physical variables and their relations from images and questions; (b) causal mechanism analysis: explaining the physical laws between these variables (e.g., "mass increases $\rightarrow$ gravity increases $\rightarrow$ acceleration changes"); (c) counterfactual prediction: predicting results under counterfactual conditions based on the causal mechanism. These three steps are chained using a structured prompt template, with the output of each step serving as the input for the next.
    - **Design Motivation**: Answering counterfactual questions directly is too difficult for models; decomposing it into explicit causal analysis steps reduces reasoning difficulty while rendering the reasoning process interpretable.

3. **Physics Commonsense Distillation**:

    - **Function**: Enhances the model's counterfactual physical reasoning capability through fine-tuning.
    - **Mechanism**: Utilizing a stronger LLM (such as GPT-4o) to generate high-quality CoA reasoning trajectories (containing detailed analyses of causal variables, causal mechanisms, and counterfactual predictions) on CoA-Bench, and then fine-tuning the target LVLM with these trajectories. Fine-tuning data includes correct counterfactual reasoning exemplars and correction exemplars for common error patterns.
    - **Design Motivation**: Prompting methods alone cannot fundamentally improve a model's physical understanding; fine-tuning on counterfactual data allows physical causal knowledge to be internalized within model parameters.

### Loss & Training
Fine-tuning leverages a standard sequence-to-sequence loss, with training data structured as triplets of (image, counterfactual question, CoA reasoning trajectory + answer).

## Key Experimental Results

### Main Results

| Model | Standard Physical QA $\uparrow$ | Counterfactual Physical QA $\uparrow$ | Gap | Counterfactual post-CoA $\uparrow$ |
|------|-----------|------------|------|-----------|
| GPT-4V | 72.3 | 48.5 | -23.8 | 56.2 |
| Gemini Pro | 68.7 | 44.1 | -24.6 | 52.8 |
| LLaVA-1.5-13B | 58.2 | 35.6 | -22.6 | 43.7 |
| InternVL-Chat | 64.5 | 41.3 | -23.2 | 49.5 |
| Qwen-VL-Plus | 66.1 | 42.8 | -23.3 | 51.3 |
| LLaVA + Distilled Fine-tuning | 61.5 | 48.2 | -13.3 | - |

### Ablation Study

| Configuration | Counterfactual Acc $\uparrow$ | Note |
|------|----------|------|
| Direct Answer (w/o CoA) | 35.6 | Baseline: LLaVA direct answer |
| Complete CoA | 43.7 | +8.1; significant improvement |
| Causal Variable Identification Only | 38.2 | Improvement even with just the first step |
| Causal Mechanism Analysis Only | 40.5 | Causal mechanism analysis contributes the most |
| w/o Multi-step Decomposition (Single-step CoA) | 39.8 | Effectiveness decreases without decomposition |
| Distilled Fine-tuning + CoA | **52.3** | Dual improvement from fine-tuning + reasoning strategy |

### Key Findings
- All models show a 20-25 percentage point performance decline in counterfactual physical reasoning compared to standard physical reasoning, revealing severe deficiencies in LVLMs' causal understanding.
- The CoA reasoning strategy yields an average improvement of 8 percentage points without requiring additional training, demonstrating that explicit causal analysis indeed aids reasoning.
- The causal mechanism analysis step contributes the most — simply prompting the model to "state the physical laws" significantly boosts accuracy.
- The combination of distilled fine-tuning and CoA yields the best performance, boosting LLaVA's counterfactual accuracy from 35.6 to 52.3 (+16.7).
- Mechanics-based counterfactual questions are the most challenging (lowest accuracy), while statics-based ones are relatively the easiest.

## Highlights & Insights
- The construction methodology of the counterfactual physical reasoning benchmark can be transferred to other domains (e.g., social reasoning, economic reasoning), with the core idea being "changing one condition to observe how the outcome changes."
- The three-step decomposition strategy of CoA makes physical reasoning interpretable, allowing precise localization of which reasoning step the model fails at.
- The substantial performance gap between standard and counterfactual scenarios represents strong evidence that the "physical understanding" of current LVLMs remains far from profound.

## Limitations & Future Work
- Counterfactual scenarios are currently limited to static images, without involving dynamic physical processes in videos.
- The complexity of physical laws far exceeds the scope covered in this paper (which only touches upon basic mechanics); more complex scenarios like fluid mechanics and thermodynamics remain to be explored.
- The physical correctness of distilled training data relies on the teacher model; GPT-4V itself makes errors in physical reasoning.
- Future iterations could consider introducing physical simulators (such as PyBullet) to generate precise counterfactual outcomes and improve data quality.

## Related Work & Insights
- **vs CLEVR (Johnson et al., 2017)**: Visual reasoning in CLEVR focuses on geometric attributes, whereas this work extends to physical causal reasoning.
- **vs PTR (Hong et al., 2021)**: PTR evaluates physical reasoning but does not involve counterfactuals, whereas this work adds a causal diagnosis dimension.
- **vs CogBench (Wang et al., 2024)**: A multimodal cognitive evaluation benchmark, whereas this work focuses specifically on the counterfactual subproblem of physical reasoning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Counterfactual physical reasoning is a novel entry point in LVLM research.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensively evaluates multiple models, but the distillation fine-tuning experiment is only conducted on a single model.
- Writing Quality: ⭐⭐⭐⭐ The framework design is logically clear, but some details could be more elaborate.
- Value: ⭐⭐⭐⭐ Holds significant reference value for understanding the causal reasoning capabilities of LVLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Reasoning is All You Need for Video Generalization: A Counterfactual Benchmark with Sub-question Evaluation](reasoning_is_all_you_need_for_video_generalization_a_counterfactual_benchmark_wi.md)
- [\[ACL 2025\] Counterfactual Explanations for Aspect-Based Sentiment Analysis](counterfactual_explanations_for_aspect-based_sentiment_analysis.md)
- [\[ECCV 2024\] Learning Chain of Counterfactual Thought for Bias-Robust Vision-Language Reasoning](../../ECCV2024/causal_inference/learning_chain_of_counterfactual_thought_for_bias-robust_vision-language_reasoni.md)
- [\[ICLR 2026\] On the Eligibility of LLMs for Counterfactual Reasoning: A Decompositional Study](../../ICLR2026/causal_inference/on_the_eligibility_of_llms_for_counterfactual_reasoning_a_decompositional_study.md)
- [\[ACL 2025\] Causal Graph based Event Reasoning using Semantic Relation Experts](causal_graph_based_event_reasoning_using_semantic_relation_experts.md)

</div>

<!-- RELATED:END -->
