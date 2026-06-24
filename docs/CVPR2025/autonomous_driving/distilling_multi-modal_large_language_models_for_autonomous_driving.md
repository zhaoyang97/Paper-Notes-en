---
title: >-
  [Paper Note] Distilling Multi-modal Large Language Models for Autonomous Driving
description: >-
  [CVPR 2025][Autonomous Driving][Multi-modal Large Language Models] This paper proposes the DiMA framework, which performs knowledge distillation between a Multi-modal Large Language Model (MLLM) and a visual end-to-end planner through joint training. It designs three surrogate tasks—masked reconstruction, future prediction, and scene editing—to enrich scene representations. During inference, the LLM can be discarded, utilizing only the visual planner. This achieves a 37% redu…
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Multi-modal Large Language Models"
  - "Knowledge Distillation"
  - "End-to-End Autonomous Driving"
  - "Long-tail Scenarios"
  - "Visual Planner"
date: 2026-05-08
content_hash: 211bafd0a280a1d0
---

# Distilling Multi-modal Large Language Models for Autonomous Driving

**Conference**: CVPR 2025  
**arXiv**: [2501.09757](https://arxiv.org/abs/2501.09757)  
**Code**: None  
**Area**: Autonomous Driving  
**Keywords**: Multi-modal Large Language Models, Knowledge Distillation, End-to-End Autonomous Driving, Long-tail Scenarios, Visual Planner

## TL;DR
This paper proposes the DiMA framework, which performs knowledge distillation between a Multi-modal Large Language Model (MLLM) and a visual end-to-end planner through joint training. It designs three surrogate tasks—masked reconstruction, future prediction, and scene editing—to enrich scene representations. During inference, the LLM can be discarded, utilizing only the visual planner. This achieves a 37% reduction in L2 trajectory error and an 80% reduction in collision rate on nuScenes.

## Background & Motivation

End-to-end autonomous driving systems perform well in general navigation scenarios but struggle in long-tail scenarios (e.g., rare operations like three-point turns or overtaking), primarily due to the limited scale and lack of diversity in training datasets. Large Language Models (LLMs) are pre-trained on massive internet data, possessing rich world knowledge and chain-of-thought reasoning capabilities, and have recently been used as driving planners to improve generalization in long-tail scenarios.

However, using LLMs as planners faces two major issues: (1) the computational overhead during inference is too huge to be practical; (2) standard image tokenization strategies use frozen pre-trained image encoders to generate dense, unstructured token embeddings, which are inefficient for structured scene understanding in autonomous driving.

The core idea of this paper is to utilize the MLLM's world knowledge during training and discard the LLM during inference, retaining only the highly efficient visual planner—achieving the optimal balance of "LLM during training, no LLM during inference." The key innovation lies in treating the scene encoder of the visual planner as a trainable tokenizer for the MLLM, enabling it to learn language-grounded structured representations.

## Method

### Overall Architecture
DiMA consists of two major components: (1) a visual end-to-end planner (scene encoder + planning Transformer); (2) a multi-modal large language model (adapter layer + LLM + task-specific decoding heads). The scene encoder is shared between both components, functioning as both a feature extractor for the planning Transformer and a tokenizer for the MLLM. Training is split into two stages: first, pre-training the visual planner for 60 epochs, followed by joint training of the visual planner and the MLLM for 30 epochs.

### Key Designs

1. **BEAM Scene Token Embeddings**:

    - The scene encoder encodes the input multi-view image sequences into four types of structured token embeddings:
        - **B**EV tokens: Bird's-Eye-View features
        - **E**go tokens: Ego-vehicle interaction features, initialized by learnable embeddings
        - **A**gent tokens: Surrounding agent features
        - **M**ap tokens: Map element features
    - These BEAM tokens are projected into the LLM's embedding space via their respective Q-former adapter layers.
    - Compared to dense, unstructured image tokens, BEAM provides structured inputs with explicit physical meanings.
    - Key difference: The scene encoder is jointly trained with the MLLM (unlike methods like TOKEN that freeze the scene encoder).

2. **Surrogate Tasks**:

    - **Masked BEV Reconstruction**: Randomly masks BEV tokens and requires the MLLM to reconstruct the masked BEV features using the context from the rest of the multi-modal sequence. Supervised using L2 loss, pushing the MLLM to learn global scene understanding.
    - **Future BEV Prediction**: Given current BEV tokens, predicts the BEV token embeddings for the next two time steps. Supervised using L2 loss, encouraging the LLM to learn spatial-temporal cues.
    - **Scene Editing**: Augments the scene by adding or removing surrounding agents while constructing corresponding Q&A pairs. When adding, map constraints and predicted trajectories are considered to create new agent tokens. This task forces the model to learn how surrounding agents influence the ego-vehicle's trajectory.

3. **Knowledge Distillation and VQA**:

    - **Feature Distillation**: Minimizes the KL divergence between the LLM's second-to-last layer features and the planning Transformer's features, forcing both branches to learn consistent representations.
    - **Visual Question Answering (VQA)**: Trained on the DriveLM dataset, covering four QA categories: perception, prediction, planning, and behavior. For nuScenes samples not in DriveLM, Llama3-70B is used to generate similar QA pairs.
    - **LLM Planning**: The MLLM also predicts trajectories for the ego-vehicle and surrounding agents.

### Loss & Training
- Total loss = $L_{planning} + L_{LLM} + L_{recon} + L_{future} + L_{distill}$
- Stage 1 (60 epochs): Trains only the visual planner (perception + prediction + planning)
- Stage 2 (30 epochs): Shared joint training, with the LLM fine-tuned using LoRA
- Uses LLaVA-v1.5-7B as the base LLM
- Surrogate task decoding heads use a 3-layer Linear + ReLU

## Key Experimental Results

### Main Results (Standard Evaluation - Full Validation Set)

| Dataset | Metric | Ours (DiMA VAD-Base) | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| nuScenes | Avg L2 (m) ↓ | 0.47 | 0.56 (PARA-Drive) | -16.1% |
| nuScenes | Avg Collision (%) ↓ | 0.06 | 0.17 (PARA-Drive) | -64.7% |
| nuScenes (targeted) | Avg L2 (m) ↓ | 0.71 | 0.91 (PARA-Drive) | -22.0% |
| nuScenes (targeted) | Avg Collision (%) ↓ | 0.05 | 0.14 (PARA-Drive) | -64.3% |

### Results under VAD Evaluation Settings

| Method | Avg L2 ↓ | Avg Collision ↓ | FPS |
|------|----------|-----------------|-----|
| VAD-Base | 0.72 | 0.22 | 4.5 |
| DiMA (VAD-Tiny) | 0.38 | 0.15 | 16.8 |
| DiMA (VAD-Base) | 0.29 | 0.10 | 4.5 |
| DriveVLM-Dual | 0.31 | 0.10 | - |

### Long-tail Scenario Performance (Zero-shot Three-point Turn)

| Method | Avg L2 ↓ | Collision ↓ |
|------|----------|-------------|
| VAD-Base | 1.57 | 0.00 |
| PARA-Drive | 1.29 | 5.33 |
| TOKEN | 1.18 | 4.00 |
| DiMA (VAD-Base) | 1.05 | 0.00 |

### Ablation Study

| Configuration | Avg L2 ↓ | Avg Collision ↓ | Description |
|------|---------|--------|------|
| VAD-Tiny baseline | 0.60 | 0.29 | No MLLM |
| + VQA + BEV tokens | 0.62 | 0.26 | Using only BEV tokens causes slight performance degradation |
| + All BEAM tokens | 0.52 | 0.21 | Structured tokens bring significant improvement |
| + Distillation | 0.48 | 0.19 | Feature distillation is effective |
| + Masked recon | 0.42 | 0.18 | Masked reconstruction brings further improvement |
| + Future pred | 0.39 | 0.16 | Future prediction task is effective |
| + Scene editing (full) | 0.38 | 0.15 | Scene editing brings final improvement |

### Key Findings
- Training the MLLM solely with BEV tokens yields unstable results; all BEAM structured tokens must be used.
- Each surrogate task incrementally contributes to performance gains, with the scene editing task providing a significant final boost.
- DiMA (VAD-Tiny) outperforms VAD-Base in performance while being 4 times faster.
- The three-point turn is a zero-shot scenario (only appearing in the validation set), yet DiMA still achieves the best performance, validating the effectiveness of world knowledge transfer.
- No LLM is required during inference, maintaining the same FPS as the baseline visual planner.

## Highlights & Insights
- The "LLM during training, no LLM during inference" paradigm is a highly practical design philosophy, perfectly balancing knowledge utilization and execution efficiency.
- The design of BEAM structured tokens as MLLM inputs is far superior to dense image tokens and possesses clear physical meanings.
- The scene editing task is highly creative: virtual addition/deletion of agents forces the model to learn causal relationships.
- Joint training rather than freezing the scene encoder is critical, enabling true integration of visual representations and linguistic knowledge.
- The significant improvement in long-tail scenarios (44% L2 error reduction) demonstrates the transfer value of the LLM's world knowledge.

## Limitations & Future Work
- Limitations of open-loop evaluation: showing a good predicted trajectory does not guarantee superior closed-loop driving performance.
- Complex training pipeline: requires two-stage training, the DriveLM dataset, and additionally generated QA pairs.
- LLaVA-v1.5-7B as the base LLM is not the latest or strongest; utilizing a stronger LLM may further improve performance.
- Scene editing currently only supports adding/deleting vehicles, without involving more complex edits such as pedestrians or weather changes.
- Lack of computational cost analysis: the training overhead of jointly training with the MLLM is not explicitly detailed.
- VQA results are only presented qualitatively, lacking quantitative comparisons with other VQA methods.

## Related Work & Insights
- Comparison with TOKEN: TOKEN freezes the scene encoder as a tokenizer, whereas DiMA trains it jointly, achieving better results.
- Comparison with DriveVLM: DriveVLM uses dense visual tokens, while DiMA uses structured BEAM tokens and does not require the LLM during inference.
- Comparison with OmniDrive: OmniDrive has low inference efficiency and high collision rates; DiMA outperforms it in both aspects.
- The design concept of surrogate tasks originates from self-supervised learning (e.g., masked reconstruction in MAE, temporal prediction), but is innovatively combined with LLM distillation.
- This distillation framework can be generalized to other scenarios that require "large models during training, but high speed during inference."

## Rating
- Novelty: ⭐⭐⭐⭐ BEAM structured tokens and the design of surrogate tasks are novel, though the macro philosophy of distillation is not pioneering.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive with multiple evaluation protocols, long-tail scenario analysis, and detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with intuitive diagrams, though some paragraphs have a high density of information requiring repeated reading.
- Value: ⭐⭐⭐⭐⭐ Addresses the core conflict of LLMs in autonomous driving being "good but slow," achieving nuScenes SOTA with strong practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Embracing Large Language Models in Traffic Flow Forecasting](../../ACL2025/autonomous_driving/embracing_large_language_models_in_traffic_flow_forecasting.md)
- [\[CVPR 2025\] Distilling Monocular Foundation Model for Fine-grained Depth Completion](distilling_monocular_foundation_model_for_fine-grained_depth_completion.md)
- [\[CVPR 2025\] SOLVE: Synergy of Language-Vision and End-to-End Networks for Autonomous Driving](solve_synergy_of_language-vision_and_end-to-end_networks_for_autonomous_driving.md)
- [\[AAAI 2026\] TimeBill: Time-Budgeted Inference for Large Language Models](../../AAAI2026/autonomous_driving/timebill_time-budgeted_inference_for_large_language_models.md)
- [\[ICCV 2025\] ETA: Efficiency through Thinking Ahead, A Dual Approach to Self-Driving with Large Models](../../ICCV2025/autonomous_driving/eta_efficiency_through_thinking_ahead_a_dual_approach_to_self-driving_with_large.md)

</div>

<!-- RELATED:END -->
