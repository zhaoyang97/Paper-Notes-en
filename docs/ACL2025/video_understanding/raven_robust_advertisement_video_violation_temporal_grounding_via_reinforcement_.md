---
title: >-
  [Paper Note] RAVEN: Robust Advertisement Video Violation Temporal Grounding via Reinforcement Reasoning
description: >-
  [ACL 2025][Video Understanding][Advertisement violation detection] This paper proposes the RAVEN framework, which integrates curriculum reinforcement learning with multimodal LLMs. Through hierarchical reward mechanisms and progressive training strategies, RAVEN achieves precise temporal grounding and category prediction of advertisement video violations, unlocking emergent reasoning capabilities without requiring explicit reasoning annotation data.
tags:
  - "ACL 2025"
  - "Video Understanding"
  - "Advertisement violation detection"
  - "temporal grounding"
  - "curriculum reinforcement learning"
  - "GRPO"
  - "multimodal large language models"
date: 2026-05-08
content_hash: bb3172ea8b30b6a1
---

# RAVEN: Robust Advertisement Video Violation Temporal Grounding via Reinforcement Reasoning

**Conference**: ACL 2025  
**arXiv**: [2510.16455](https://arxiv.org/abs/2510.16455)  
**Code**: None  
**Area**: Video Understanding  
**Keywords**: Advertisement violation detection, temporal grounding, curriculum reinforcement learning, GRPO, multimodal large language models

## TL;DR

This paper proposes the RAVEN framework, which integrates curriculum reinforcement learning with multimodal LLMs. Through hierarchical reward mechanisms and progressive training strategies, RAVEN achieves precise temporal grounding and category prediction of advertisement video violations, unlocking emergent reasoning capabilities without requiring explicit reasoning annotation data.

## Background & Motivation

**Background**: Advertisement video violation detection is a key mechanism for platform compliance. Early methods relied on small models (ViT, ResNet, etc.) with limited generalization capability. With the advancement of LLMs, multimodal large language models (MLLMs) are increasingly applied in content moderation.

**Limitations of Prior Work**: Existing approaches face three primary challenges: (1) videos require not only predicting violation categories but also precisely locating corresponding temporal segments, and a single video may contain multiple violations; (2) target video segment boundaries in annotated data are often imprecise (noisy annotations), wherein direct fitting via SFT results in incorrect learning; (3) SFT generalizes poorly to out-of-domain (OOD) data and triggers catastrophic forgetting.

**Key Challenge**: Precise temporal boundary annotations are extremely expensive, while coarse-grained annotations can mislead SFT training; meanwhile, models must possess reasoning capabilities to process complex scenarios, but annotating the reasoning process for every sample is impractical.

**Goal**: Design a framework that leverages large-scale coarse-grained annotated data while achieving precise temporal grounding and maintaining the generalization capabilities of MLLMs.

**Key Insight**: Inspired by DeepSeek-R1, reinforcement learning (rather than SFT) can run purely to unlock reasoning capabilities, combined with curriculum learning strategies to handle annotation data of varying quality.

**Core Idea**: Through curriculum reinforcement learning and hierarchical reward mechanisms, MLLMs spontaneously generate structured reasoning capabilities without reasoning annotations, facilitating robust learning from noisy annotated data.

## Method

### Overall Architecture

RAVEN is based on Qwen2.5-VL as the policy model. The overall workflow is as follows: given input video $V$, a list of violation labels $T$, and a prompt $P$, the model outputs a set of violation categories $\mathcal{C} = \{c_1, c_2, \dots, c_n\}$ and corresponding temporal intervals $\mathcal{X}_c = (t_c^l, t_c^r)$. The model generates the reasoning process within the `<think>` tag and outputs structured results within the `<answer>` tag.

### Key Designs

#### 1. Data Construction and Curriculum Strategy

- **Function**: Categorize data into a precisely annotated subset and a large-scale coarse-grained annotated dataset according to annotation quality.
- **Mechanism**: Precise annotations are used in the early stages of curriculum learning to build a baseline, while coarse-grained annotations are utilized in late-stage large-scale training. No offline reasoning data is required.
- **Design Motivation**: To circumvent the issue of SFT directly fitting to noisy annotations, leveraging the natural robustness of RL against noise.

#### 2. Hierarchical Reward Function Design

RAVEN designs five categories of reward functions:

**（a）Thinking Format Reward**: Ensures the model outputs reasoning within `<think></think>` tags and the final result within `<answer></answer>` tags.

**（b）Grounding Format Reward**: Categorized into soft and strict levels—soft only requires the inclusion of temporal coordinates, while strict requires specific keywords such as "temporal start" and "temporal end".

**（c）Temporal IoU Reward (Main Reward)**: Evaluates the overlap between the predicted and ground-truth intervals, utilizing a binarized threshold to maintain robustness:

$$R_{\text{IoU}} = \begin{cases} 1 & \text{if IoU}(\mathcal{X}_c, \mathcal{Y}_c) > 0.5 \\ 0 & \text{otherwise} \end{cases}$$

**（d）Temporal Boundary Alignment Reward (Auxiliary Reward)**: Encourages precise boundary alignment using continuous values:

$$R_{\text{Boundary}} = \exp\left(-\sigma^2\left[(t_c^l - y_c^l)^2 + (t_c^r - y_c^r)^2\right]\right)$$

**（e）Violation Category Consistency Reward**: Ensures the predicted category matches the annotated category, binarized.

#### 3. Three-Stage Curriculum Reinforcement Training

- **Stage 1 (Precisely Annotated Data)**: $R_{\text{Total}} = R_{\text{IoU}} + \alpha_1 \cdot R_{\text{Boundary}} + R_{\text{Category}}$ to establish basic capabilities in interval prediction and category identification.
- **Stage 2 (Coarsely Annotated Data)**: $R_{\text{Total}} = R_{\text{IoU}} + \alpha_2 \cdot R_{\text{Boundary}}$ is used by dropping the category consistency reward (as coarse annotations are unreliable) to learn roughly correct intervals.
- **Stage 3 (Full Dataset Fine-Tuning)**: $R_{\text{Total}} = \alpha_3 \cdot R_{\text{IoU}} + \alpha_4 \cdot R_{\text{Boundary}} + \alpha_5 \cdot R_{\text{Category}}$ to balance all objectives.

### Loss & Training

The Group Relative Policy Optimization (GRPO) algorithm is employed for reinforcement learning. This eliminates the need for cold-start reasoning training, allowing direct training from the pre-trained Qwen2.5-VL model.

## Key Experimental Results

### Main Results

**Violation Detection Performance on Industrial Dataset** (38K training videos, 5K testing videos, 6 violation categories):

| Method | Avg Category Precision/Recall | Avg Temporal Grounding (mIoU) |
|------|----------------------|-------------------|
| Small Models | 0.697/0.657 | - |
| LLaVA-v1.5-SFT | 0.782/0.758 | 0.436 |
| Qwen2.5-VL-7B-SFT | 0.805/0.774 | 0.456 |
| **Ours** | **0.826/0.788** | **0.555** |

**Performance on Public Dataset MultiHateClip**:

| Method | Category Precision/Recall | Temporal Grounding (mIoU) |
|------|-----------------|---------------|
| LLaVA-v1.5-SFT | 0.509/0.501 | 0.370 |
| Qwen2.5-VL-7B-SFT | 0.537/0.517 | 0.384 |
| **Ours** | **0.551/0.530** | **0.435** |

**Online A/B Test** (20% traffic, one full day):

| Method | Category Precision/Recall | Temporal Grounding (mIoU) |
|------|-----------------|---------------|
| Small Models | 0.711/0.668 | - |
| Qwen2.5-VL-7B-SFT | 0.800/0.787 | 0.478 |
| **Ours** | **0.821/0.803** | **0.563** |

### Ablation Study

**Effect of Structured Thinking**:

| Method | Category Precision/Recall | Temporal Grounding (mIoU) |
|------|-----------------|---------------|
| Qwen2.5-VL-7B-SFT | 0.805/0.774 | 0.456 |
| RAVEN (w/o Structured Thinking) | 0.810/0.779 | 0.537 |
| RAVEN (w/ Structured Thinking) | 0.826/0.788 | 0.555 |

**Generalization Capability (3 In-Domain Categories Trained $\to$ 2 Out-of-Domain Categories Tested)**:

| Method | In-Domain mIoU | Out-of-Domain mIoU |
|------|---------|---------|
| Qwen2.5-VL-7B-SFT | 0.433 | 0.246 |
| RAVEN | 0.546 | 0.408 |

**Reward Function Ablation**:

| Boundary Reward | Format Reward | Curriculum Learning | mIoU |
|----------------|---------------|---------|------|
| ✓ | strict | ✓ | **0.555** |
| ✓ | soft | ✓ | 0.547 |
| ✓ | strict | ✗ | 0.508 |
| ✗ | strict | ✓ | 0.540 |

### Key Findings

1. RAVEN outperforms SFT baselines in temporal grounding by approximately 10 percentage points (mIoU 0.555 vs. 0.456).
2. Online A/B tests validate practical efficiency, with an 8.5% increase in temporal accuracy relative to SFT models.
3. Generalization capabilities of RL training far exceed SFT: out-of-domain mIoU improves from 0.246 to 0.408, representing a 65.9% gain.
4. Structured thinking contributes significantly to temporal grounding: mIoU increases from 0.537 to 0.555.
5. The curriculum learning strategy is critical: removing it drops mIoU by 4.7 percentage points.

## Highlights & Insights

1. **Cold-Start-Free Inference**: No reasoned process data annotation is required; RL spontaneously induces reasoning capability, drastically reducing data costs.
2. **Robustness to Noisy Annotations**: Sophisticated curriculum learning is utilized to separate precise/coarse data, working in tandem with binarized IoU thresholds to handle annotation noise.
3. **Industrial-Grade Validation**: Validated via offline experiments as well as online A/B testing, proving the practical deployability of the mechanism.
4. **Mitigating Catastrophic Forgetting**: Compared to SFT, RL significantly enhances out-of-domain generalization, which is crucial for practical deployments.
5. **Exquisite Hierarchical Reward Design**: The hierarchy of Format $\to$ Grounding $\to$ Boundary $\to$ Category guarantees clear optimization targets for each element.

## Limitations & Future Work

1. RAVEN is built on a 7B model; larger models may yield further improvements.
2. The industrial dataset remains unreleased, limiting reproducibility.
3. The violation categories are relatively coarse (6 major categories); fine-grained sub-category detection is yet to be explored.
4. The impact of video sampling strategies (frame rates, keyframe selection) on performance was not fully discussed.
5. Future work can explore extending the RAVEN framework to other temporal grounding tasks (e.g., video summarization, event detection).

## Related Work & Insights

- **DeepSeek-R1 (Guo et al., 2025)**: The paradigm of activating reasoning capabilities through RL directly inspired RAVEN.
- **GRPO (Shao et al., 2024)**: Group Relative Policy Optimization, which serves as the core training algorithm for RAVEN.
- **VSLNet / 2D-TAN**: Traditional temporal grounding methods, which lack reasoning capabilities.
- **Insight**: The integrated paradigm of RL + Curriculum Learning + Hierarchical Rewards can be generalized to other scenarios requiring robust training under noisy annotations.

## Rating

| Dimension | Rating |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] When Thinking Drifts: Evidential Grounding for Robust Video Reasoning](../../NeurIPS2025/video_understanding/when_thinking_drifts_evidential_grounding_for_robust_video_reasoning.md)
- [\[ICCV 2025\] VTimeCoT: Thinking by Drawing for Video Temporal Grounding and Reasoning](../../ICCV2025/video_understanding/vtimecot_thinking_by_drawing_for_video_temporal_grounding_and_reasoning.md)
- [\[NeurIPS 2025\] TempSamp-R1: Effective Temporal Sampling with Reinforcement Fine-Tuning for Video LLMs](../../NeurIPS2025/video_understanding/tempsampr1_effective_temporal_sampling_with_reinforcement_fi.md)
- [\[CVPR 2026\] SARL-STG: A Spatially Aware Reinforcement Learning Framework for Refining MLLMs in Spatio-Temporal Video Grounding](../../CVPR2026/video_understanding/sarl-stg_a_spatially_aware_reinforcement_learning_framework_for_refining_mllms_i.md)
- [\[CVPR 2026\] Learning to Refuse: Refusal-Aware Reinforcement Fine-Tuning for Hard-Irrelevant Queries in Video Temporal Grounding](../../CVPR2026/video_understanding/learning_to_refuse_refusal-aware_reinforcement_fine-tuning_for_hard-irrelevant_q.md)

</div>

<!-- RELATED:END -->
