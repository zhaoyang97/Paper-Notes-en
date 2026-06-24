---
title: >-
  [Paper Note] Adapting Vision-Language Models for Evaluating World Models
description: >-
  [NeurIPS 2025][Multimodal VLM][world model evaluation] This paper proposes UNIVERSE (UNIfied Vision-language Evaluator for Rollouts in Simulated Environments), a unified semantic evaluator for world model rollouts constructed by fine-tuning only the projection head of PaliGemma 2 (0.07% of total parameters). UNIVERSE achieves performance comparable to task-specific models on action recognition and character recognition, while exhibiting strong alignment with human judgments.
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "world model evaluation"
  - "VLM adaptation"
  - "action recognition"
  - "character recognition"
  - "lightweight fine-tuning"
date: 2026-05-08
content_hash: a5f12f494ba7a93c
---

# Adapting Vision-Language Models for Evaluating World Models

**Conference**: NeurIPS 2025
**arXiv**: [2506.17967](https://arxiv.org/abs/2506.17967)  
**Code**: None  
**Area**: Multimodal VLM / World Model Evaluation
**Keywords**: world model evaluation, VLM adaptation, action recognition, character recognition, lightweight fine-tuning

## TL;DR

This paper proposes UNIVERSE (UNIfied Vision-language Evaluator for Rollouts in Simulated Environments), a unified semantic evaluator for world model rollouts constructed by fine-tuning only the projection head of PaliGemma 2 (0.07% of total parameters). UNIVERSE achieves performance comparable to task-specific models on action recognition and character recognition, while exhibiting strong alignment with human judgments.

## Background & Motivation

World models are conditional generative models that simulate environment dynamics by predicting future observations, and are increasingly important in planning, simulation, and embodied AI. However, evaluating rollout quality poses fundamental challenges:

1. **Distributional metrics** (FID/FVD) lack semantic grounding and cannot capture action alignment or entity consistency.
2. **Text-video metrics** ignore timestamp-level action conditioning.
3. **Human evaluation** is costly and difficult to scale.
4. **General-purpose VLMs** perform poorly when applied directly — VideoLLaMA3 achieves only 12.7% action recognition accuracy in the zero-shot setting.

These challenges motivate the study of how to adapt VLMs into automatic evaluators for world model rollouts.

## Method

### Overall Architecture

The UNIVERSE pipeline proceeds as follows:
- **Input**: a rollout video frame sequence $V = (o_{t_1}, \dots, o_{t_k})$ and a natural language question $Q$
- **Output**: a predicted answer $\hat{A}$, evaluated against a reference answer $A$

The evaluation protocol defines two recognition tasks:
- **Action Recognition (AR)**: assesses whether the generated frame sequence accurately reflects the effect of the conditioned action.
- **Character Recognition (CR)**: assesses whether entities maintain consistent identity and appearance over time.

Each task supports three question-answering formats: binary (yes/no), multiple-choice, and open-ended.

### Key Designs

1. **Partial Fine-Tuning Strategy (Projection Head Only)**

    **Function**: Only the projection head $\theta_P$ (2.66M parameters, 0.07% of total) between the visual encoder and language decoder of PaliGemma 2 3B is trained.

    **Mechanism**: A systematic comparison across five fine-tuning configurations (zero-shot / full fine-tuning / dual-component / single-component / LoRA) reveals that projection head fine-tuning offers the best cost-performance trade-off — it achieves performance second only to visual encoder fine-tuning (~11% parameters) at substantially lower computational cost. The training objective is the causal language modeling loss:
    $$\mathcal{L}(S) = -\sum_{t=1}^{T_{\text{SUFF}}} \log P(s_t^{\text{SUFF}} \mid S_{<t'})$$

    **Design Motivation**: Limited data and computational resources in the world model evaluation setting necessitate minimizing trainable parameters while preserving the model's pretrained knowledge.

2. **Uniform Frame Sampling + Mixed Supervision**

    **Function**: $k=8$ frames are uniformly sampled from a 14-frame rollout rather than taking the first $k$ frames; training data mixing ratios are set to $\alpha_{AR}=0.8$ and $\beta_{OE}=0.8$.

    **Mechanism**: Uniform sampling preserves long-range temporal structure — at only 2 frames, uniform sampling improves multiple-choice accuracy from 65.53% to 83.93% (+18.4 percentage points). The data mixing ratios are optimized via hierarchical ablation: task ratios are tuned first (AR requires more data due to slower convergence), followed by format ratios (open-ended QA contributes most to generalization).

    **Design Motivation**: AR requires temporal causal reasoning; taking only the initial frames discards critical dynamic information. CR converges rapidly (surpassing 97% within 12.5% of training epochs), whereas AR requires more supervision and longer temporal context.

### Loss & Training

- Backbone: PaliGemma 2 3B (SigLIP-400M visual encoder + Gemma 2 2B decoder)
- Training format: visual tokens + text prefix (question) + text suffix (answer, used only during training)
- Frame resolution: $224 \times 224$, 256 patches per frame
- Training dataset: 32,453 training clips, 194,718 QA pairs

## Key Experimental Results

### Main Results

| Model | AR Accuracy | CR Accuracy | Trainable Parameters |
|-------|-------------|-------------|----------------------|
| VideoLLaMA3 7B (zero-shot) | 12.7% | 6.4% | 0% |
| PaliGemma 2 3B (zero-shot) | 29.7% | 17.2% | 0% |
| $\mathcal{F}_V$ (visual encoder fine-tuning) | 2nd | 1st | ~11% |
| $\mathcal{F}_L$ (language head fine-tuning) | mid | mid | ~72% |
| **UNIVERSE ($\mathcal{F}_P$)** | **1st** | **3rd** | **0.07%** |

UNIVERSE surpasses all baselines on AR (including full model fine-tuning) and ranks third on CR, behind only visual encoder fine-tuning and task-specific models.

### Ablation Study

**Effect of Frame Sampling Strategy on AR Performance (2 frames, Exact Match)**:

| Sampling Strategy | Binary | Multiple-Choice | Open-Ended |
|-------------------|--------|-----------------|------------|
| First-2 | 84.42% | 65.53% | 65.38% |
| **Uniform-2** | **90.47%** | **83.93%** | **82.68%** |

Uniform sampling substantially outperforms sequential sampling across all formats, with the largest gains observed at low frame counts.

### Key Findings

- **Zero-shot VLMs are entirely insufficient**: even the 7B VideoLLaMA3 achieves only 12.7% AR accuracy, demonstrating the necessity of domain adaptation.
- **CR is substantially easier than AR**: CR converges to 97%+ within 12.5% of training epochs (~4K samples), whereas AR requires both more frames and longer training.
- **Alignment with human evaluation**: across 8 diverse environment configurations (varying model scale, resolution, and domain), UNIVERSE exhibits high agreement with human judgments, reaching a substantial level of Cohen's κ.
- **Optimized vs. default data mixing**: hierarchically tuned data ratios yield significant gains on AR multiple-choice and open-ended formats.

## Highlights & Insights

- A unified multi-task evaluator is constructed with only 0.07% of parameters, avoiding the overhead of training separate models per task and format.
- The systematic study of adaptation strategies (5,154 GPU-days) provides practical guidance for VLM adaptation under resource-constrained settings.
- The evaluation protocol is elegantly designed: binary, multiple-choice, and open-ended formats present progressively increasing difficulty, systematically exposing capability differences.

## Limitations & Future Work

- Validation is limited to a game simulation environment (Bleeding Edge) and has not been extended to real-world scenarios.
- The evaluation protocol covers only basic semantic tasks (action and character recognition), with no treatment of higher-level causal reasoning.
- Scalability to long rollouts is limited — longer videos would require intelligent sampling or hierarchical summarization.
- The model may inherit data biases from pretrained VLMs.

## Related Work & Insights

- The approach extends the LLM-as-a-Judge paradigm (Zheng et al., 2023) to video world model evaluation.
- UNIVERSE is complementary to the structured evaluation protocol of Cosmos (Agarwal et al., 2025), being more lightweight and not reliant on simulator-specific infrastructure.
- A promising future direction is applying UNIVERSE's adaptation strategy to other temporally sensitive video understanding tasks.

## Rating

⭐⭐⭐ High engineering utility with comprehensive systematic experiments; however, the novelty is concentrated in engineering design choices rather than methodological breakthroughs, and the evaluation setting is confined to gaming environments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Towards Evaluating Proactive Risk Awareness of Multimodal Language Models](towards_evaluating_proactive_risk_awareness_of_multimodal_language_models.md)
- [\[NeurIPS 2025\] Evaluating Multimodal Large Language Models on Core Music Perception Tasks](evaluating_multimodal_large_language_models_on_core_music_perception_tasks.md)
- [\[CVPR 2025\] Evaluating Vision-Language Models as Evaluators in Path Planning](../../CVPR2025/multimodal_vlm/evaluating_vision-language_models_as_evaluators_in_path_planning.md)
- [\[ACL 2025\] Do Vision-Language Models Have Internal World Models? Towards an Atomic Evaluation](../../ACL2025/multimodal_vlm/do_vision-language_models_have_internal_world_models_towards_an_atomic_evaluatio.md)
- [\[ACL 2025\] AlignMMBench: Evaluating Chinese Multimodal Alignment in Large Vision-Language Models](../../ACL2025/multimodal_vlm/alignmmbench_evaluating_chinese_multimodal_alignment_in_large_vision-language_mo.md)

</div>

<!-- RELATED:END -->
