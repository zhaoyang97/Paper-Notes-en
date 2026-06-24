---
title: >-
  [Paper Note] Learning Procedural-aware Video Representations through State-Grounded Hierarchy Unfolding
description: >-
  [AAAI 2026][LLM Pretraining][Procedural video understanding] This paper proposes a Task-Step-State (TSS) three-level semantic framework that introduces "state" as a visual grounding layer within the conventional task-step hierarchy, and designs a progressive pretraining strategy following a U-shaped path (Task→Step→State→Step→Task) to unfold the TSS hierarchy stage by stage. The approach achieves comprehensive state-of-the-art performance on task recognition, step recognition…
tags:
  - "AAAI 2026"
  - "LLM Pretraining"
  - "Procedural video understanding"
  - "state grounding"
  - "hierarchical learning"
  - "progressive pretraining"
  - "video representation"
date: 2026-05-08
content_hash: d69dbf8a66e5fb78
---

# Learning Procedural-aware Video Representations through State-Grounded Hierarchy Unfolding

**Conference**: AAAI 2026
**arXiv**: [2511.20073](https://arxiv.org/abs/2511.20073)  
**Code**: [https://github.com/zhao-jinghan/TSS-unfolding](https://github.com/zhao-jinghan/TSS-unfolding)  
**Area**: LLM Pretraining
**Keywords**: Procedural video understanding, state grounding, hierarchical learning, progressive pretraining, video representation

## TL;DR
This paper proposes a Task-Step-State (TSS) three-level semantic framework that introduces "state" as a visual grounding layer within the conventional task-step hierarchy, and designs a progressive pretraining strategy following a U-shaped path (Task→Step→State→Step→Task) to unfold the TSS hierarchy stage by stage. The approach achieves comprehensive state-of-the-art performance on task recognition, step recognition, and step forecasting tasks on the COIN and CrossTask datasets.

## Background & Motivation

Understanding and executing goal-directed procedural activities (e.g., operational steps in instructional videos) is a core capability for intelligent agents. Existing methods learn procedural video representations by aligning visual content with textual descriptions at the task/step level—for instance, using task names from wikiHow (e.g., "Make Orange Juice") and step descriptions (e.g., "Cut the orange") as supervision signals.

However, this two-level approach has **critical limitations**:
- Task and step descriptions are highly abstract, making it difficult to form robust alignments with specific, observable details in visual data.
- There exists a **substantial semantic gap** between abstract instructions such as "Cut the orange" and the raw pixels that actually depict the action.
- During visual-text alignment training, this gap prevents the model from grounding abstract procedures in actually visible content.

**Key Challenge**: The semantic gap between abstract text and concrete visual observations. The authors' key insight is to introduce **"state"**—textual snapshots of object configurations (e.g., "the orange is no longer whole; the flesh is exposed")—as a semantic grounding layer that anchors abstract procedures to what the model can actually observe.

From a logical perspective, states constitute the skeleton of any procedural task: a task is a macro-level transition from an initial state to a final state, and steps are the actions that drive intermediate state transitions. This forms the three-level Task-Step-State framework.

## Method

### Overall Architecture
The method comprises two core contributions:
1. **TSS Framework Construction**: An LLM is used to generate pre-, mid-, and post-state descriptions for each step, forming a three-level semantic structure.
2. **Progressive Pretraining Strategy**: Stage-by-stage training along a U-shaped path: Task→Step→State→Step→Task.

### Key Designs

1. **Task-Step-State Framework Construction**:

    - **Function**: Augments the conventional two-level task-step structure with a "state" layer, associating three types of states with each step $s_{i,j}$:
        - **Pre-state $c_{i,j}^b$**: Object configuration before the action begins.
        - **Mid-state $c_{i,j}^m$**: Object configuration during the action.
        - **Post-state $c_{i,j}^a$**: Object configuration after the action completes.
    - **Mechanism**: GPT-4o-mini with chain-of-thought (CoT) prompting is employed to automatically generate state texts from wikiHow task-plus-step descriptions, constructing a rich three-level knowledge base.
    - **Scale**: 1,053 tasks, 10,588 steps, and 31,764 state descriptions.
    - **Design Motivation**: State descriptions capture observable object configurations (e.g., shape, attributes, spatial relations), whose semantic distance to visual content is far smaller than that of abstract step descriptions.

2. **Pseudo-Label Generation**:

    - **Function**: Generates training supervision signals through video-text feature alignment.
    - **Mechanism**:
        - Frozen S3D visual encoder and Sentence-BERT text encoder are used to extract features respectively.
        - Text features are aggregated and clustered (10,038 step nodes; approximately 9,000–10,000 state nodes).
        - Video clips are matched to text nodes via cosine similarity, with the top-3 nodes selected as multi-class pseudo-labels.
    - **Five pseudo-label types**: TaskVNM, StepVNM, StateVNM, StepNRL (Node Relation Learning), StepTCL (Task Context Learning).
    - **Design Motivation**: Leverages large-scale weak supervision from pretrained encoders, eliminating the need for manual temporal annotations.

3. **Progressive Pretraining Strategy**:

    - **Function**: Trains the model in stages along a specific path, with each stage focusing on one semantic level.
    - **Optimal path**: **Task→Step→State→Step→Task** (Path-5/6).
    - **Model architecture**: Frozen S3D visual encoder + trainable bottleneck adapter (512→128→512) + randomly initialized task head.
    - **Knowledge transfer mechanism**: Adapter weights are retained and passed to the next stage upon completion of each stage; the task head is discarded and re-initialized.
    - **Design Motivation**: The strategy first performs top-down analysis (Task→Step→State) and then bottom-up synthesis (State→Step→Task), forming a complete analysis-synthesis cycle. Key findings include:
        - Directly jumping from State→Task (Path-4) yields poor results, confirming that the step level serves as a necessary intermediate bridge.
        - Joint training (Mix_Train) underperforms progressive training, as it fails to capture the causal relationships between hierarchy levels.
        - The final Step→Task back-traversal (Path-6 vs. Path-5) yields only marginal gains.

### Loss & Training
- BCEWithLogitsLoss multi-class loss.
- Adam optimizer, lr=1e-4, batch_size=256.
- Training data: 4.1 million video clips (from an 85K-video subset of HowTo100M).
- Approximately 90 seconds per epoch (Step/State stages) or 30 seconds (Task stage); 1,500 epochs total.
- Trained on 8×H200 GPUs.

## Key Experimental Results

### Main Results

**Comparison with SOTA (Path-5, COIN dataset)**:

| Method | TR(MLP) | SR(MLP) | SF(MLP) | TR(Trans) | SR(Trans) | SF(Trans) |
|--------|---------|---------|---------|-----------|-----------|-----------|
| No pretrain | 2.09 | 1.37 | 0.84 | 78.31 | 39.23 | 35.43 |
| Paprika (SOTA) | 81.54 | 42.39 | 34.10 | 82.83 | 41.19 | 38.93 |
| **Ours (Path-5)** | **83.78** | **44.54** | **38.07** | **83.11** | **42.42** | **40.40** |
| Gain vs. Paprika | +2.24 | +2.15 | +3.97 | +0.28 | +1.23 | +1.47 |

**CrossTask dataset**:

| Method | TR(MLP) | SR(MLP) | SF(MLP) | TR(Trans) | SR(Trans) | SF(Trans) |
|--------|---------|---------|---------|-----------|-----------|-----------|
| Paprika | 89.65 | 56.21 | 55.77 | 90.27 | 55.57 | 55.67 |
| **Ours (Path-5)** | 89.44 | **57.92** | **57.13** | 89.44 | **57.08** | **56.50** |

### Ablation Study

| Configuration | COIN TR(MLP) | COIN SR(MLP) | COIN SF(MLP) | Notes |
|---------------|-------------|-------------|-------------|-------|
| Path-1 (Task only) | 73.31 | 34.18 | 23.67 | Task-level pretraining only |
| Path-2 (Task→Step) | 82.45 | 43.06 | 36.04 | Adding step level improves results |
| Path-3 (→State) | 80.73 | 41.28 | 34.35 | Direct transition to State degrades performance |
| Path-4 (→State→Task) | 77.74 | 37.51 | 24.84 | State→Task jump is too large |
| **Path-5 (→State→Step)** | **83.78** | **44.54** | **38.07** | **U-shaped back-traversal is optimal** |
| Path-6 (→Step→Task) | 83.30 | 44.04 | 36.94 | Additional Task back-traversal yields marginal gains |
| Mix_Train (joint) | 77.74 | 38.43 | 29.79 | Joint training underperforms progressive training |
| Fusion-AvgPool | 83.11 | 44.35 | 36.88 | Feature fusion is effective but weaker than progressive training |

### Key Findings
- **The state level is the key driver**: The best result without states (Path-2) achieves 43.06 SR; the optimal result with states reaches **44.54** SR.
- **Progressive training outperforms joint training**: Mix_Train SR (38.43) is substantially lower than Path-5 (44.54), highlighting the importance of causal relationships between hierarchy levels.
- **The step level is a necessary intermediate bridge**: Path-4 (direct State→Task jump) suffers a sharp performance drop, while Path-5 (State→Step intermediate transition) achieves the best results.
- **The U-shaped path simulates analysis-synthesis**: Top-down analysis proceeds to the most concrete state level, followed by bottom-up synthesis back to abstract steps.
- Gains are more pronounced with a simple MLP downstream head (+3.97 SF), indicating genuinely higher-quality pretrained representations.
- Feature fusion (AvgPool/Concat) is also effective, validating the complementary nature of state information.

## Highlights & Insights
- **Conceptual innovation of "state" as a visual grounding layer**: Abstract procedural knowledge is grounded in visual data through observable object configurations.
- **Systematic exploration of progressive pretraining paths**: Rather than arbitrarily choosing a training order, the paper systematically validates six distinct paths through ablation experiments.
- **Efficient low-cost design**: The visual encoder is frozen, and only a bottleneck adapter (512→128→512) is trained, achieving high parameter efficiency.
- The LLM-based state description generation approach is practical and scalable.
- The Path-4 vs. Path-5 comparison precisely reveals the existence of the semantic gap and the bridging role of the step level.

## Limitations & Future Work
- Reliance on the S3D encoder and wikiHow corpus may limit applicability to broader video types.
- LLM-generated state descriptions may lack accuracy, particularly for abstract or complex operations.
- Evaluation is conducted only on COIN and CrossTask, both of which are relatively limited in scale.
- The fixed 9.6-second video segmentation may be inappropriate for all steps, as some steps are much shorter or longer.
- Direct visual prediction from video to state is not explored; state information is utilized only indirectly through text-visual alignment.

## Related Work & Insights
- The concept of "state" bridges procedural learning and object-centric video understanding.
- Progressive/curriculum learning ideas are particularly effective for hierarchically structured knowledge.
- The paradigm of using LLMs as knowledge augmentation tools (generating state descriptions) is generalizable to other tasks requiring intermediate semantic layers.
- The analysis-synthesis (U-shaped) learning path offers new insights for designing pretraining strategies over multi-level knowledge structures.
- The adapter fine-tuning strategy keeps the computational cost of large-scale pretraining manageable.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Both the TSS three-level framework and the U-shaped progressive training path are original contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Systematic ablation over 6 paths + fusion strategy comparisons + SOTA benchmarking.
- Writing Quality: ⭐⭐⭐⭐⭐ — Tight logical chain; the ablation analysis is highly instructive.
- Value: ⭐⭐⭐⭐ — The state-grounding idea has broad applicability, though dataset scale limits overall impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] GranAlign: Granularity-Aware Alignment Framework for Zero-Shot Video Moment Retrieval](granalign_granularity-aware_alignment_framework_for_zero-shot_video_moment_retri.md)
- [\[NeurIPS 2025\] Mouse-Guided Gaze: Semi-Supervised Learning of Intention-Aware Representations for Reading Detection](../../NeurIPS2025/llm_pretraining/mouse-guided_gaze_semi-supervised_learning_of_intention-aware_representations_fo.md)
- [\[AAAI 2026\] Beyond Cosine Similarity: Magnitude-Aware CLIP for No-Reference Image Quality Assessment](beyond_cosine_similarity_magnitude-aware_clip_for_no-reference_image_quality_ass.md)
- [\[ECCV 2024\] Cross-Domain Learning for Video Anomaly Detection with Limited Supervision](../../ECCV2024/llm_pretraining/cross-domain_learning_for_video_anomaly_detection_with_limited_supervision.md)
- [\[ICCV 2025\] FlowMo: Flow to the Mode — Mode-Seeking Diffusion Autoencoders for State-of-the-Art Image Tokenization](../../ICCV2025/llm_pretraining/flow_to_the_mode_mode-seeking_diffusion_autoencoders_for_state-of-the-art_image_.md)

</div>

<!-- RELATED:END -->
