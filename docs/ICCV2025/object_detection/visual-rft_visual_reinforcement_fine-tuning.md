---
title: >-
  [Paper Note] Visual-RFT: Visual Reinforcement Fine-Tuning
description: >-
  [ICCV 2025][Object Detection][Reinforcement Fine-Tuning] Visual-RFT extends the Reinforcement Learning with Verifiable Rewards (RLVR) paradigm from DeepSeek R1—originally applied to mathematics and code—to visual percept…
tags:
  - "ICCV 2025"
  - "Object Detection"
  - "Reinforcement Fine-Tuning"
  - "GRPO"
  - "Verifiable Reward"
  - "Visual Perception"
  - "Few-Shot Learning"
date: 2026-05-08
content_hash: 0e108acaced23420
---

# Visual-RFT: Visual Reinforcement Fine-Tuning

**Conference**: ICCV 2025
**arXiv**: [2503.01785](https://arxiv.org/abs/2503.01785)
**Code**: [https://github.com/Liuziyu77/Visual-RFT](https://github.com/Liuziyu77/Visual-RFT)
**Area**: Object Detection / Multimodal Reasoning
**Keywords**: Reinforcement Fine-Tuning, GRPO, Verifiable Reward, Visual Perception, Few-Shot Learning

## TL;DR
Visual-RFT extends the Reinforcement Learning with Verifiable Rewards (RLVR) paradigm from DeepSeek R1—originally applied to mathematics and code—to visual perception tasks. It introduces task-specific verifiable reward functions, including an IoU reward for object detection and a CLS reward for classification, achieving substantial improvements over SFT on fine-grained classification, few-shot detection, and grounded reasoning with only a fraction of the training data.

## Background & Motivation

**Background**: OpenAI o1 and DeepSeek R1 have demonstrated the powerful capabilities of large reasoning models fine-tuned via Reinforcement Learning with Verifiable Rewards (RLVR). A key advantage of RFT is data efficiency—unlike SFT, which requires large amounts of high-quality annotated data, RFT learns through trial and error and can achieve strong domain-specific performance with minimal samples.

**Limitations of Prior Work**: The prevailing assumption is that RFT is applicable only to tasks with clearly verifiable answers (e.g., correctness of math solutions or code test cases). For visual perception tasks such as detection and classification, whose outputs are bounding box coordinates or category labels, designing verifiable rewards is non-trivial. Consequently, RL in LVLM post-training has primarily been used to reduce hallucinations rather than to improve visual perception.

**Key Challenge**: The SFT paradigm is data-hungry—it requires large amounts of high-quality labeled data to "imitate" correct answers. In data-scarce scenarios (e.g., medical imaging, rare species), SFT under few-shot conditions may even degrade performance.

**Goal**: To demonstrate that RFT is extensible to visual perception tasks; to design verifiable reward functions for various visual tasks; and to surpass SFT under limited data conditions.

**Key Insight**: Although visual perception tasks differ from mathematics in output format, they share objective evaluation criteria—detection IoU and classification matching—which can serve as the basis for verifiable rewards.

**Core Idea**: Design task-specific verifiable reward functions for visual perception tasks, transfer the R1-style RLVR paradigm to the visual domain, and enable data-efficient visual fine-tuning.

## Method

### Overall Architecture
Given an image and a question as input, the policy model (LVLM) generates multiple responses, each containing chain-of-thought reasoning and a final answer. Designed verifiable reward functions evaluate the quality of each response, and the model is updated via the GRPO policy gradient algorithm. No additional reward model is required, as rewards are computed directly by rule.

### Key Designs

1. **IoU Verifiable Reward (Detection)**:

    - **Function**: Evaluates the quality of predicted bounding boxes against ground-truth annotations.
    - **Mechanism**: For each predicted box, the maximum IoU against all GT boxes is computed. The overall reward integrates both precision and recall, with an additional format reward to ensure structured output.
    - **Design Motivation**: Unlike the binary 0/1 reward in mathematics, detection tasks require a continuous reward signal. IoU naturally provides a continuous quality measure for predictions at negligible computational cost.

2. **CLS Verifiable Reward (Classification)**:

    - **Function**: Evaluates the correctness of classification predictions.
    - **Mechanism**: Exact matching—a reward of 1 is assigned if the predicted category matches the GT, and 0 otherwise.
    - **Design Motivation**: Classification answers have objective ground truth and can be verified directly by rule.

3. **Chain-of-Thought Reasoning Format**:

    - **Function**: Encourages the LVLM to produce reasoning before outputting the final answer.
    - **Mechanism**: The prompt instructs the model to output its reasoning within `<think>...</think>` tags and the final answer within `<answer>...</answer>` tags. For detection tasks, structured location and confidence outputs are required.
    - **Design Motivation**: Chain-of-thought reasoning substantially improves model performance—enabling the model to analyze fine-grained visual features for classification and to reason about spatial positions for detection.

4. **GRPO Policy Optimization**:

    - **Function**: Updates the LVLM using Group Relative Policy Optimization.
    - **Mechanism**: For each question, $G$ responses are sampled from the current policy and their rewards are computed. GRPO eliminates the need for an additional critic model by computing the advantage function from the relative quality of responses within the group, and updates the policy using a PPO-style clipped objective with KL divergence regularization.
    - **Design Motivation**: GRPO is more lightweight than PPO (no critic model required), and its effectiveness has been validated by DeepSeek R1.

### Loss & Training
- GRPO objective: maximize $\mathbb{E}_{o \sim \pi_\theta(q)}[R(q,o) - \beta \text{KL}[\pi_\theta \| \pi_{ref}]]$
- For each question, $G$ responses form a group; advantages are computed from within-group relative rewards.

## Key Experimental Results

### Main Results

| Task | Data Size | Visual-RFT | SFT | Gain |
|------|-----------|-----------|-----|------|
| Fine-grained Classification (1-shot) | ~100 | +24.3% acc | −4.3% acc | RFT >> SFT |
| Few-shot Detection COCO 2-shot | Very few | +21.9 mAP | baseline | Significant |
| Few-shot Detection LVIS | Very few | +15.4 mAP | baseline | Significant |
| Open-vocab Detection COCO new (2B) | — | 31.3 mAP | 9.8 mAP | +21.5 |
| Open-vocab Detection LVIS rare (2B) | — | 20.7 mAP | 2.7 mAP | +18.0 |

### Ablation Study

| Configuration | Classification Acc | Detection mAP | Notes |
|--------------|-------------------|---------------|-------|
| Visual-RFT (full) | Highest | Highest | Full model |
| w/o Chain-of-Thought | Notable drop | Drop | Reasoning critical for fine-grained tasks |
| SFT (same data) | Significantly lower | Significantly lower | SFT underperforms RFT in few-shot regime |
| SFT (more data) | Still below RFT | Still below RFT | Scaling data alone cannot close the gap |

### Key Findings
- **Dramatic data efficiency gap**: In the one-shot setting, RFT improves accuracy by 24.3% while SFT degrades by 4.3%—a gap of 28.6%. The imitation-based SFT paradigm fails completely under extremely limited data.
- **Chain-of-thought reasoning plays a critical role**: The model reasons about spatial positions during detection and analyzes key visual features during classification.
- **Strong generalization**: Visual-RFT rapidly transfers to novel categories in open-vocabulary detection, including LVIS rare classes.
- **Verifiable rewards are simple yet effective**: Both IoU and CLS rewards are computed by lightweight rules, requiring no trained reward model.

## Highlights & Insights
- **Paradigm shift**: Transitioning from SFT's "data scaling" to RFT's "reward function design" represents a significant shift in the visual model training paradigm.
- **Few-shot killer application**: Visual-RFT demonstrates substantial promise in severely data-scarce scenarios such as medical imaging and rare species recognition.
- **IoU reward design**: Directly repurposing a standard detection evaluation metric as an RL reward elegantly bridges evaluation and training. Any computable evaluation metric is a candidate for a verifiable reward.
- **Fully open-source**: Training code, data, and evaluation scripts are all publicly released.

## Limitations & Future Work
- GRPO requires sampling multiple responses per question, making training less efficient than SFT.
- The framework has been validated only for detection and classification; designing reward functions for tasks with continuous outputs such as segmentation is more complex.
- Chain-of-thought reasoning improves accuracy but increases inference latency.
- The advantage of RFT over SFT may diminish when sufficient training data is available.
- Reward function design currently requires manual effort for each task; automated reward design is an important future direction.

## Related Work & Insights
- **vs. DeepSeek R1**: R1 applies RLVR in the purely linguistic domain. Visual-RFT successfully extends this to multimodal visual tasks, demonstrating the generality of RLVR.
- **vs. VisRL**: VisRL focuses on the decision process of "where to look," whereas Visual-RFT targets final visual perception outcomes. The two approaches are complementary.
- **vs. SFT (conventional)**: SFT is data-hungry imitation learning; Visual-RFT is data-efficient reward-driven learning.

## Rating
- Novelty: ⭐⭐⭐⭐ Transferring the R1 paradigm to the visual domain is a relatively direct yet pioneering contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Broad task coverage (classification + detection + grounding + open-vocabulary) with rich experimental settings.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clear; results are presented concisely.
- Value: ⭐⭐⭐⭐⭐ A paradigm-shifting work with full open-source release and significant community impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Dynamic-DINO: Fine-Grained Mixture of Experts Tuning for Real-time Open-Vocabulary Object Detection](dynamicdino_finegrained_mixture_of_experts_tuning_for_realti.md)
- [\[ICCV 2025\] VisRL: Intention-Driven Visual Perception via Reinforced Reasoning](visrl_intention-driven_visual_perception_via_reinforced_reasoning.md)
- [\[ICCV 2025\] Visual Modality Prompt for Adapting Vision-Language Object Detectors](visual_modality_prompt_for_adapting_vision-language_object_detectors.md)
- [\[NeurIPS 2025\] Automated Detection of Visual Attribute Reliance with a Self-Reflective Agent](../../NeurIPS2025/object_detection/automated_detection_of_visual_attribute_reliance_with_a_self-reflective_agent.md)
- [\[CVPR 2026\] A Closer Look at Cross-Domain Few-Shot Object Detection: Fine-Tuning Matters and Parallel Decoder Helps](../../CVPR2026/object_detection/a_closer_look_at_cross-domain_few-shot_object_detection_fine-tuning_matters_and_.md)

</div>

<!-- RELATED:END -->
