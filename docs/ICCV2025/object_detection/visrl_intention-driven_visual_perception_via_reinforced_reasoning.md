---
title: >-
  [Paper Note] VisRL: Intention-Driven Visual Perception via Reinforced Reasoning
description: >-
  [ICCV 2025][Object Detection][Intention-driven visual perception] VisRL is the first framework to apply reinforcement learning to intention-driven visual perception. Through iterative DPO training, it enables large multimodal models (LMMs) to autonomously select focus regions (by predicting bounding boxes) according to query intent, achieving superior visual reasoning over SFT without requiring costly intermediate bounding box annotations.
tags:
  - "ICCV 2025"
  - "Object Detection"
  - "Intention-driven visual perception"
  - "reinforcement learning"
  - "Visual CoT"
  - "DPO"
  - "large multimodal models"
date: 2026-05-08
content_hash: 5f8599bf7e20cd2f
---

# VisRL: Intention-Driven Visual Perception via Reinforced Reasoning

**Conference**: ICCV 2025
**arXiv**: [2503.07523](https://arxiv.org/abs/2503.07523)  
**Code**: [https://github.com/zhangquanchen/VisRL](https://github.com/zhangquanchen/VisRL)  
**Area**: Object Detection / Multimodal Reasoning
**Keywords**: Intention-driven visual perception, reinforcement learning, Visual CoT, DPO, large multimodal models

## TL;DR
VisRL is the first framework to apply reinforcement learning to intention-driven visual perception. Through iterative DPO training, it enables large multimodal models (LMMs) to autonomously select focus regions (by predicting bounding boxes) according to query intent, achieving superior visual reasoning over SFT without requiring costly intermediate bounding box annotations.

## Background & Motivation

**Background**: LMMs (e.g., LLaVA, Qwen-VL) answer image-related questions via end-to-end inference. Recent Visual Chain-of-Thought (Visual CoT) methods introduce explicit reasoning steps—the model first predicts a focus region (bounding box), crops it, and then answers the question using both the original and cropped images.

**Limitations of Prior Work**: Visual CoT relies heavily on supervised training, requiring bounding box annotations for intermediate reasoning steps for each query-image pair. Since the same image may correspond to vastly different focus regions depending on query intent, annotation complexity grows combinatorially and cannot cover all possible intent-region pairs.

**Key Challenge**: SFT requires dense $\langle\text{intent, focus region}\rangle$ pair annotations $\rightarrow$ high annotation cost and incomplete coverage $\rightarrow$ training on limited annotations $\rightarrow$ restricted generalization.

**Goal**: Enable models to learn intention-driven visual perception without bounding box annotations.

**Key Insight**: Analogous to human visual learning—humans do not learn "where to look" through dense annotations, but instead develop adaptive focus through trial-and-error interaction with the environment. Replacing SFT with RL is therefore more principled.

**Core Idea**: Apply reinforcement learning (iterative DPO) to optimize focus region selection during visual reasoning, using task reward signals in place of bounding box annotations for scalable intention-driven visual perception.

## Method

### Overall Architecture
VisRL proceeds in two stages: (1) **SFT warm-up**—training the model on a small amount of bbox-annotated data to learn the "look-then-answer" reasoning format; (2) **RL training**—on large-scale unannotated data, iteratively alternating between data generation and model optimization via step-level DPO. The RL stage requires no external models or annotations; data synthesis and scoring are performed entirely by the model itself.

### Key Designs

1. **SFT Warm-up**:

    - **Function**: Train the model to follow the Visual CoT reasoning format (predict bbox before answering).
    - **Mechanism**: Fine-tune the model on a small annotated dataset to generate responses following the pipeline: predict bbox → crop → answer.
    - **Design Motivation**: RL training requires the model to already possess basic reasoning format capabilities as a starting point.

2. **Iterative DPO Framework**:

    - **Function**: The core loop of RL training, alternating between data generation and model optimization.
    - **Mechanism**: In each iteration, the model generates multiple reasoning trajectories (different bboxes and answers) per question. Preference pairs are constructed based on the correctness of final answers, and the model is optimized via step-level DPO. This process is repeated over multiple iterations for continuous improvement.
    - **Design Motivation**: A single round of DPO is insufficient; iterative training allows the model to progressively explore better strategies.

3. **Diversity Controller**:

    - **Function**: Ensure that generated bboxes cover a diverse range of candidate focus regions.
    - **Mechanism**: During data generation, sampling temperature is adjusted and random perturbations are introduced to promote sufficient diversity in the generated bboxes.
    - **Design Motivation**: If generated bboxes are too similar, the quality of constructed preference pairs is low. Diversity is critical for exploration in RL.

4. **Step-Level DPO**:

    - **Function**: Optimize the model at each step of the reasoning process.
    - **Mechanism**: Visual reasoning comprises two steps (region selection + answering). Step-level DPO decomposes preference learning across each step, ensuring the model learns both "which region to select" and "how to answer" simultaneously.
    - **Design Motivation**: Standard DPO compares entire trajectories, which may introduce signal confusion—e.g., a poor region selection that happens to yield a correct answer.

5. **Difficulty Filtering**:

    - **Function**: Select questions of appropriate difficulty and the most informative preference pairs.
    - **Mechanism**: Only questions that are partially correct and partially incorrect are retained; questions that are too easy or too hard are discarded as unsuitable for constructing preference pairs.
    - **Design Motivation**: Questions of moderate difficulty provide the largest learning signal, analogous to curriculum learning.

### Loss & Training
- **SFT stage**: Standard next-token prediction loss.
- **RL stage**: Step-level DPO loss, computing preference loss separately at the bbox prediction step and the answer generation step.
- **Iterative training**: New data is generated using the current model at each round to avoid off-policy issues.

## Key Experimental Results

### Main Results

| Method | HR-Bench (4K) | V\*Bench | TextVQA | Avg. |
|--------|--------------|----------|---------|------|
| LLaVA-1.5 (baseline) | 52.3 | 61.8 | 58.4 | 57.5 |
| Visual CoT (SFT) | 55.1 | 65.2 | 61.7 | 60.7 |
| **VisRL** | **58.9** | **68.4** | **64.5** | **63.9** |

### Ablation Study

| Configuration | HR-Bench | V\*Bench | Note |
|---------------|---------|----------|------|
| Full VisRL | 58.9 | 68.4 | Complete model |
| w/o step-level DPO (trajectory-level) | 56.2 | 65.8 | Step-level is critical |
| w/o diversity controller | 55.8 | 64.9 | Diversity matters |
| w/o difficulty filtering | 57.1 | 66.3 | Filtering is beneficial |
| SFT only (more data) | 55.5 | 65.5 | More data still underperforms RL |

### Key Findings
- VisRL consistently outperforms SFT baselines across multiple benchmarks, confirming the superiority of the RL paradigm for visual reasoning.
- Step-level DPO contributes the most (removing it causes a 2.7% drop), validating the necessity of per-step optimization.
- The diversity controller has a large impact on performance, indicating that exploration is crucial for learning visual attention in RL.
- Strong generalizability—consistent gains are observed across different base LMMs (LLaVA vs. Qwen-VL).
- No additional bbox annotations are required; training on large-scale data substantially outperforms SFT methods that rely on dense annotations.

## Highlights & Insights
- **First RL + Visual Perception Work**: VisRL is the first to apply RL to intention-driven visual perception, opening a new research direction.
- **Step-Level DPO**: Extends DPO from trajectory level to step level, which is more principled for multi-step reasoning tasks and is transferable to any multi-step reasoning setting.
- **Annotation-Free Scalability**: The RL stage requires no bbox annotations; the model self-generates data and self-evaluates, making it naturally scalable to arbitrary data volumes.
- **Model-Agnostic**: The training framework can be applied to different base LMMs.

## Limitations & Future Work
- The SFT warm-up still requires a small amount of annotated data; a fully zero-annotation approach remains to be explored.
- Only a single bbox is predicted per step, which may be insufficient for complex scenarios requiring attention to multiple regions.
- Iterative DPO training incurs relatively high computational cost.
- Reward signals are based solely on final answer correctness, making it difficult to define verification functions for open-ended questions.
- Integration with more recent RL algorithms such as GRPO is a promising avenue for future work.

## Related Work & Insights
- **vs. Visual CoT**: Visual CoT requires dense bbox annotations for SFT training. VisRL replaces SFT with RL, eliminating annotation dependency while achieving better performance.
- **vs. DeepSeek-R1**: R1 demonstrates the power of RL for language reasoning; VisRL transfers this paradigm to multimodal visual reasoning.
- **vs. Visual-RFT**: Visual-RFT targets downstream tasks such as classification and detection, and does not address the dimension of intention-driven focus selection. VisRL focuses more on the reasoning process itself.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First work combining RL with intention-driven visual perception; step-level DPO design is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Validated across multiple benchmarks with thorough ablations and cross-model generalization tests.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is well-developed; method description is clear.
- **Value**: ⭐⭐⭐⭐⭐ Opens a new direction for RL + visual reasoning with strong scalability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Reasoning-Driven Anomaly Detection and Localization with Image-Level Supervision](../../CVPR2026/object_detection/reasoning-driven_anomaly_detection_and_localization_with_image-level_supervision.md)
- [\[ICML 2025\] UI-Vision: A Desktop-centric GUI Benchmark for Visual Perception and Interaction](../../ICML2025/object_detection/ui-vision_a_desktop-centric_gui_benchmark_for_visual_perception_and_interaction.md)
- [\[ICCV 2025\] Visual-RFT: Visual Reinforcement Fine-Tuning](visual-rft_visual_reinforcement_fine-tuning.md)
- [\[ICLR 2026\] Traceable Evidence Enhanced Visual Grounded Reasoning: Evaluation and Method](../../ICLR2026/object_detection/traceable_evidence_enhanced_visual_grounded_reasoning_evaluation_and_methodology.md)
- [\[AAAI 2026\] Connecting the Dots: Training-Free Visual Grounding via Agentic Reasoning](../../AAAI2026/object_detection/connecting_the_dots_training-free_visual_grounding_via_agent.md)

</div>

<!-- RELATED:END -->
