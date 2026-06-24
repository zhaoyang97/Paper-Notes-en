---
title: >-
  [Paper Note] Debate, Reflect, and Distill: Multi-Agent Feedback with Tree-Structured Preference Optimization for Efficient Language Model Enhancement
description: >-
  [ACL 2025][LLM Alignment][Multi-Agent Debate] This paper proposes the D&R framework, which allows a small model (student) to engage in multi-round debates with multiple large models (teachers) to collect self-reflections and teacher feedback. The debate logs are organized into preference trees for distillation using Tree-structured DPO (T-DPO). This achieves an average improvement of 14.18 points on MMLU Pro and MATH, with better inference efficiency than baselines.
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "Multi-Agent Debate"
  - "Knowledge Distillation"
  - "Preference Optimization"
  - "Self-Reflection"
  - "Tree-Structured DPO"
date: 2026-05-08
content_hash: ff48ab511390e2b4
---

# Debate, Reflect, and Distill: Multi-Agent Feedback with Tree-Structured Preference Optimization for Efficient Language Model Enhancement

**Conference**: ACL 2025  
**arXiv**: [2506.03541](https://arxiv.org/abs/2506.03541)  
**Code**: [https://github.com/zhouxiaofengshelf/D-R](https://github.com/zhouxiaofengshelf/D-R)  
**Area**: LLM Alignment  
**Keywords**: Multi-Agent Debate, Knowledge Distillation, Preference Optimization, Self-Reflection, Tree-Structured DPO

## TL;DR
This paper proposes the D&R framework, which allows a small model (student) to engage in multi-round debates with multiple large models (teachers) to collect self-reflections and teacher feedback. The debate logs are organized into preference trees for distillation using Tree-structured DPO (T-DPO). This achieves an average improvement of 14.18 points on MMLU Pro and MATH, with better inference efficiency than baselines.

## Background & Motivation

**Background**: While large LLMs achieve outstanding performance, their deployment cost is high. Knowledge distillation is a vital technique to bridge the gap between large and small models. Although multi-agent debate has been proven to enhance reasoning quality, there lacks a systematic approach to distilling the knowledge generated during debates into small models.

**Limitations of Prior Work**: (a) Traditional distillation only utilizes the final outputs of teacher models, discarding error-correction information during the reasoning process; (b) Methods like MAGDi use debate data but fail to fully exploit the structured feedback (self-reflection, teacher correction) within the debates; (c) Standard DPO is not suitable for handling hierarchical preference data generated in multi-round debates.

**Key Challenge**: Multi-round debates generate rich reasoning trajectories and error-correction signals, but how can these unstructured dialogue logs be converted into effective training signals?

**Goal**: To design a complete pipeline that efficiently extracts structured learning signals from multi-agent debates and distills them into a small model using a tailored preference optimization method.

**Key Insight**: Each round of a debate contains a hierarchical structure of "prior feedback + current response," which is naturally suited for representation as a tree structure—where correct and incorrect responses serve as children of the same root node, and the root node contains the structured information from the previous round.

**Core Idea**: Collect self-reflections and teacher feedback through multi-round debates, organize the debate logs into preference trees, and perform T-DPO to distill this knowledge into the small model.

## Method

### Overall Architecture
A three-stage pipeline: (1) **Debate**—the student engages in a debate of up to 4 rounds with 3 teachers, collecting responses, self-reflections, and teacher feedback in each round to construct a Multi-Agent Interaction Graph (MAG); (2) **Reflect**—extracts preference trees from the MAG, where the root of each tree is "question + structured information from the prior round" and leaf nodes are correct and incorrect responses; (3) **Distill**—a two-stage training consisting of Supervised Fine-Tuning (SFT) on correct responses, followed by T-DPO on preference trees.

### Key Designs

1. **Multi-Agent Debate (Debate)**:

    - **Function**: Enables the student (Mistral-7B) to engage in multi-round discussions with three teachers (GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro).
    - **Mechanism**: In each round, all agents receive structured information from the previous round (responses, self-reflections, teacher feedback) to generate new answers. Agents with incorrect answers perform self-reflection, while agents with correct answers provide constructive feedback to those who erred. The debate stops after up to 4 rounds or when a consensus is reached.
    - **Design Motivation**: Error-correction dynamics during debates offer richer learning signals than simple imitation learning. The student not only sees correct answers but also learns "where it went wrong, why, and how to correct it."

2. **Preference Tree Construction (Reflect)**:

    - **Function**: Extracts structured preference pairs from the Multi-Agent Interaction Graph (MAG).
    - **Mechanism**: For each debate round, with "question + prior-round structured information (prior responses, self-reflections, teacher feedback)" as the root node, transitions from incorrect to correct answers are treated as chosen, and transitions from correct to incorrect, or persistently incorrect answers as rejected. A single debate cascade can yield multiple preference trees (leading to a 2.5x to 7.7x data amplification).
    - **Design Motivation**: The tree structure encodes the hierarchical relations of the debate into preference pairs, enabling DPO to learn "how to correct errors given specific feedback."

3. **T-DPO Distillation (Distill)**:

    - **Function**: Trains the small model in a two-stage process.
    - **Mechanism**: Stage 1 - SFT: Train on correct responses using the standard language modeling loss: $\mathcal{L}_{SFT} = -\mathbb{E} \log \pi_\theta(y|I,x)$. Stage 2 - T-DPO: Perform DPO on preference trees, using the structured information at the root node as extra context: $\mathcal{L}_{T-DPO} = -\mathbb{E}[\log \sigma(\beta \log \frac{\pi_\theta(y_w|I,x,SI)}{\pi_{ref}(y_w|I,x,SI)} - \beta \log \frac{\pi_\theta(y_l|I,x,SI)}{\pi_{ref}(y_l|I,x,SI)})]$. $\pi_{ref}$ is initialized from the SFT model.
    - **Design Motivation**: SFT teaches the model to generate correct answers, while T-DPO teaches the model how to correct its course when receiving feedback—the two processes are complementary.

### Loss & Training
- Uses LoRA fine-tuning (rank=16, alpha=32).
- Teacher models: GPT-4o + Claude 3.5 Sonnet + Gemini 1.5 Pro.
- Student models: Mistral-7B-Instruct-v0.2 (main experiments), Llama-3.1-8B (extended experiments).

## Key Experimental Results

### Main Results (Mistral-7B $\to$ D&R)

| Method | CS | Physics | Biology | MATH | Average |
|------|-----|---------|---------|------|------|
| No Distillation | 20.49 | 18.46 | 48.95 | 8.02 | 23.98 |
| Single Teacher (Claude 3.5) | 30.73 | 30.77 | 62.76 | 16.56 | 35.21 |
| MAGDi (Multi-Teacher) | 22.93 | 20.00 | 54.39 | 11.26 | 27.15 |
| D&R (SFT only) | 32.68 | 29.23 | 64.44 | 17.64 | 36.00 |
| **D&R (Full)** | **33.17** | **34.77** | **67.36** | **17.32** | **38.16** |

An average gain of 14.18 points (23.98 $\to$ 38.16), with the largest improvement in Biology (+18.41). Its inference token consumption is 528.83 vs 627.10 for the baseline, demonstrating superior efficiency.

### Ablation Study

| Configuration | MATH Accuracy |
|------|--------------|
| D&R (Full) | 34.77 |
| w/o Self-reflection data | 33.85 (-0.92) |
| w/o Teacher feedback data | 32.31 (-2.46) |
| w/o Both | 31.38 (-3.39) |
| + Inference-time self-reflection | 36.92 (+2.15) |

### Key Findings
- **Teacher feedback contributes more than self-reflection**: Removing teacher feedback drops performance by -2.46 compared to -0.92 for removing self-reflection, indicating that explicit error correction from strong models is more instructional.
- **T-DPO has limited impact on weaker models but is effective for intermediate ones**: On Llama-3.1-8B, T-DPO surpasses SFT-only by 0.96 points (47.06 $\to$ 48.02), whereas on Mistral-7B, SFT slightly outperforms T-DPO on MATH.
- **High data efficiency**: Significant improvements can be achieved with only 100 debates. D&R consistently outperforms MAGDi across all data scales.
- **Improved inference efficiency**: D&R not only achieves higher accuracy but also consumes fewer tokens during inference (528 vs 627).

## Highlights & Insights
- **A complete pipeline of Debate $\to$ Preference Tree $\to$ DPO**: Translates unstructured multi-round dialogues into structured preference training data, which is a methodological novelty. This paradigm of "extracting training signals from interactions" is highly transferable to other multi-agent learning scenarios.
- **Data amplification effect**: Multiple preference trees can be extracted from a single debate (2.5x to 7.7x data amplification), yielding richer data than conventional single-round distillation.
- **Inference-time self-reflection capability**: The fine-tuned model internalizes self-correction capabilities (further pushing performance by +2.15 points when prompting for self-reflection at inference time), showing that the error-correction patterns from debates are successfully internalized.

## Limitations & Future Work
- **Evaluated only on knowledge/reasoning tasks**: Generalizability to other task types, such as generation or classification, remains unexplored.
- **Evaluation relies solely on final answer correctness**: The validity of reasoning paths is not verified, leaving a risk of false positives where the answer is correct but the reasoning is flawed.
- **High debate costs**: Requires calling three commercial APIs (GPT-4o, Claude, Gemini), although this is a one-time cost during the data generation phase.
- **Future Improvements**: (a) Open-source models could substitute some teachers to cut costs; (b) Process-supervised Reward Models (PRMs) could be incorporated to assess the quality of reasoning paths.

## Related Work & Insights
- **vs MAGDi**: MAGDi also leverages multi-agent debate for distillation, but directly applies SFT on debate outputs without utilizing preference structures. D&R's preference trees + T-DPO significantly outperform MAGDi (38.16 vs 27.15).
- **vs Standard DPO**: Standard DPO operates on flat preference pairs, whereas T-DPO leverages the hierarchical structure of debates to provide richer context.
- **vs Self-Play**: Self-Play methods (e.g., SPIN) let models play against themselves, which lacks explicit error-correction signals from external strong models.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The construction of preference trees and the introduction of T-DPO are highly innovative contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Includes multiple benchmarks, extensive ablation studies, different student models, and efficiency analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Methods are clearly described, and the logic across different pipeline stages is coherent.
- **Value**: ⭐⭐⭐⭐ Provides a systematic approach to multi-agent distillation; T-DPO can be generalized to other scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Expectation Confirmation Preference Optimization for Multi-Turn Conversational Recommendation Agent](expectation_confirmation_preference_optimization_for_multi-turn_conversational_r.md)
- [\[ICML 2025\] M³HF: Multi-agent Reinforcement Learning from Multi-phase Human Feedback of Mixed Quality](../../ICML2025/llm_alignment/m3hf_multi-agent_reinforcement_learning_from_multi-phase_human_feedback_of_mixed.md)
- [\[ACL 2025\] QueryAttack: Jailbreaking Aligned Large Language Models Using Structured Non-natural Query Language](queryattack_jailbreaking_aligned_large_language_models_using_structured_non-natu.md)
- [\[ACL 2025\] Tempest: Autonomous Multi-Turn Jailbreaking of Large Language Models with Tree Search](tempest_autonomous_multi-turn_jailbreaking_of_large_language_models_with_tree_se.md)
- [\[ACL 2025\] AutoMixAlign: Adaptive Data Mixing for Multi-Task Preference Optimization in LLMs](automixalign_adaptive_data_mixing.md)

</div>

<!-- RELATED:END -->
