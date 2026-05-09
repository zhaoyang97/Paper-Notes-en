---
title: >-
  [Paper Note] TempSamp-R1: Effective Temporal Sampling with Reinforcement Fine-Tuning for Video LLMs
description: >-
  [NeurIPS 2025][Video Understanding][Video Temporal Grounding] This paper proposes TempSamp-R1, a mixed-policy reinforcement fine-tuning framework that integrates high-quality off-policy (ground truth) guidance into GRPO's on-policy sampling and introduces nonlinear soft advantage estimation to stabilize training, achieving state-of-the-art performance on video temporal grounding (Charades-STA R1@0.7: 52.9%, ActivityNet R1@0.5: 56.0%).
tags:
  - NeurIPS 2025
  - Video Understanding
  - Video Temporal Grounding
  - Reinforcement Learning Fine-Tuning
  - GRPO
  - Mixed-Policy Sampling
  - Chain-of-Thought
date: 2026-05-08
content_hash: 562d1d3b9f1613d6
---

# TempSamp-R1: Effective Temporal Sampling with Reinforcement Fine-Tuning for Video LLMs

**Conference**: NeurIPS 2025
**arXiv**: [2509.18056](https://arxiv.org/abs/2509.18056)
**Code**: [GitHub](https://github.com/HVision-NKU/TempSamp-R1)
**Area**: Video Understanding
**Keywords**: Video Temporal Grounding, Reinforcement Learning Fine-Tuning, GRPO, Mixed-Policy Sampling, Chain-of-Thought

## TL;DR
This paper proposes TempSamp-R1, a mixed-policy reinforcement fine-tuning framework that integrates high-quality off-policy (ground truth) guidance into GRPO's on-policy sampling and introduces nonlinear soft advantage estimation to stabilize training, achieving state-of-the-art performance on video temporal grounding (Charades-STA R1@0.7: 52.9%, ActivityNet R1@0.5: 56.0%).

## Background & Motivation

1. **State of the Field**: MLLMs excel at general video understanding but still struggle with temporal grounding tasks, which require precise comprehension of spatiotemporal relationships in long videos.

2. **Limitations of Prior Work**: SFT-based methods overfit to static timestamp annotations and lack flexible temporal reasoning. GRPO's on-policy sampling is inefficient in large temporal search spaces, making it difficult to find temporally precise solutions.

3. **Root Cause**: GRPO uses ground truth only for computing IoU rewards, rather than treating it as a dynamic learning resource; additionally, high rewards from off-policy solutions introduce bias in advantage estimation.

4. **Paper Goals**: Design a more stable and efficient RL fine-tuning framework that fully leverages annotation information to guide policy learning.

5. **Starting Point**: Incorporate ground truth directly as an off-policy solution in policy optimization, while addressing the reward distribution bias this introduces.

6. **Core Idea**: Mixed-policy sampling + nonlinear soft advantage estimation + hybrid CoT training.

## Method

### Overall Architecture
Built on Qwen2.5-VL-7B, the framework generates $G-1$ on-policy samples and 1 off-policy solution (ground truth) per query. A soft advantage estimation module computes advantage values for stable policy optimization.

### Key Designs

1. **Mixed-Policy Sampling**:
    - Function: Provides temporally precise guidance solutions.
    - Mechanism: For each query, $G-1$ solutions are sampled from the current policy, and 1 ground truth is added as an off-policy solution; advantages are computed via joint normalization.
    - Design Motivation: Pure on-policy sampling rarely produces high-IoU solutions in large temporal search spaces; ground truth provides precise temporal anchoring.

2. **Nonlinear Soft Advantage Estimation**:
    - Function: Mitigates advantage bias caused by high rewards from off-policy solutions.
    - Mechanism: An asymmetric transformation is applied to rewards: $\tilde{r}_i = \tau + \alpha_1 \cdot \ln((r_i - \tau) + 1)$ when $r_i \geq \tau$, with exponential expansion in the low-reward region. This compresses advantage gaps near optimal solutions while amplifying differences among suboptimal ones.
    - Design Motivation: Directly using off-policy rewards causes all on-policy solutions to receive negative advantages, suppressing effective exploration.

3. **Hybrid CoT Training**:
    - Function: Enables a single model to support both CoT and non-CoT inference.
    - Mechanism: Two-stage training — the initialization stage trains direct answering without reasoning; subsequently, format rewards encourage \<Think\> reasoning steps.
    - Design Motivation: Charades-STA and ActivityNet benefit from CoT, while QVHighlights benefits from direct prediction; the hybrid mode is complementary.

### Loss & Training
- IoU Reward: $R_{IoU}$ based on the intersection-over-union between predicted and ground truth temporal intervals.
- Timestamp Matching Reward: $R_{ts} = \lambda_{rec} \cdot F2 + \lambda_{score} \cdot \frac{1}{1+WMSE}$
- Format Reward: Regex-based verification of the \<Think\>...\</Think\>\<Answer\>...\</Answer\> structure.
- Training: 4 × A100, batch size = 1 per GPU, $G=4$ (3 on-policy + 1 off-policy).
- Two-stage training: the initialization stage trains direct answering; the subsequent stage adds format rewards to encourage reasoning steps. Video frame rate is fixed at 2 FPS.

## Key Experimental Results

### Main Results

| Dataset | Metric | TempSamp-R1 | TimeZero | VideoChat-R1 | Gain |
|--------|------|-------------|----------|-------------|------|
| Charades-STA | R1@0.7 | **52.9%** | 47.9% | 50.2% | +2.7% |
| ActivityNet | R1@0.5 | **56.0%** | 47.3% | - | +8.7% |
| QVHighlights | mAP | **30.0%** | - | - | +3.0% |

### Ablation Study

| Configuration | mIoU | Note |
|------|------|------|
| SFT only | 20.6 | Baseline |
| GRPO only | 30.7 | Pure on-policy |
| TempSamp-R1 | **34.7** | Mixed-policy + soft advantage |

### Key Findings
- Selecting the optimal inference mode (CoT vs. non-CoT) in hybrid CoT training yields an additional 4%+ improvement in mIoU.
- Performance remains robust in low-data regimes.
- Nonlinear reward shaping is particularly effective on datasets with large search spaces, such as ActivityNet.
- Few-shot experiments: with only 50 training videos, TempSamp-R1 achieves mIoU 44.7% (vs. SFT 41.9%); with 500 videos, it reaches R1@0.5 64.0% (vs. SFT 51.4%, GRPO 55.3%), with a training time of 218 minutes (vs. GRPO 338 minutes).
- Advantage shaping ablation: directly injecting GT rewards (Mixed-policy only) drops R1@0.5 to 63.0% (due to distribution shift and reduced diversity); reward downscaling achieves 70.3%; advantage anchoring achieves 70.7%; nonlinear shaping yields the best result at 72.1%.
- Skewness analysis: GRPO exhibits consistently negative skew (dominated by low-reward solutions); Mixed-policy alone shows high positive skew (over-reliance on high-reward solutions); nonlinear shaping maintains near-zero skewness, ensuring stable optimization.
- Cross-domain generalization: training on Charades-STA and testing on ActivityNet, TempSamp-R1 outperforms GRPO by +4.0% mIoU and +4.7% R1@0.5.

## Highlights & Insights
- Ground truth is cleverly repurposed from an "evaluation tool" to a "learning resource."
- The asymmetric design of nonlinear advantage estimation is applicable to any RL scenario involving high-quality external solutions.
- Hybrid CoT training enables a single model to adapt to queries of varying complexity.
- The method is simple, requiring only minor modifications on top of GRPO.
- Ablation on sample count: with only 2 on-policy samples + 1 off-policy sample, R1@0.7 already improves from GRPO's 34.4% to 44.6% (+10.2%), representing the largest gain. As the sample count increases to 4/6/8, the gap narrows but TempSamp-R1 consistently leads.

## Limitations & Future Work
- Relies on ground truth as off-policy guidance, making it inapplicable in unannotated settings.
- The threshold $\tau$ and coefficients in the nonlinear transformation require manual tuning.
- Combination with larger models (e.g., 72B) has not been explored.
- The video frame rate is fixed at 2 FPS; higher frame rates may improve performance.
- Reward distribution analysis: GRPO exhibits low median and high variance on ActivityNet; TempSamp-R1 shows a significantly higher median with reduced variance, indicating that mixed-policy sampling consistently finds higher-quality solutions.

## Related Work & Insights
- **vs. TimeZero**: TimeZero relies on pure GRPO on-policy sampling, whereas TempSamp-R1 incorporates off-policy guidance. ActivityNet R1@0.5: TempSamp-R1 56.0% vs. TimeZero 47.3% (+8.7%).
- **vs. VideoChat-R1**: VideoChat-R1 focuses on reward function design, while TempSamp-R1 focuses on sampling strategy optimization. Charades-STA R1@0.7: TempSamp-R1 52.9% vs. VideoChat-R1 50.2% (+2.7%).
- **vs. iMOVE (SFT)**: SFT methods overfit to timestamps, whereas TempSamp-R1 learns flexible reasoning through RL.

## Rating

### Implementation Details
Built on Qwen2.5-VL-7B, trained on 4 × A100, batch size = 1 per GPU.
$G=4$ (3 on-policy + 1 off-policy), video frame rate 2 FPS.
- Novelty: ⭐⭐⭐⭐ Mixed-policy sampling combined with soft advantage estimation constitutes an effective improvement to RL fine-tuning.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across 3 datasets.
- Writing Quality: ⭐⭐⭐⭐ Method motivation and design rationale are clearly articulated.
- Value: ⭐⭐⭐⭐ Offers practical reference for both video temporal grounding and RL fine-tuning research.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Seeing the Arrow of Time in Large Multimodal Models](seeing_the_arrow_of_time_in_large_multimodal_models.md)
- [\[NeurIPS 2025\] DualGround: Structured Phrase and Sentence-Level Temporal Grounding](dualground_phrase_temporal.md)
- [\[NeurIPS 2025\] Empower Words: DualGround for Structured Phrase and Sentence-Level Temporal Grounding](empower_words_dualground_for_structured_phrase_and_sentencel.md)
- [\[NeurIPS 2025\] When Thinking Drifts: Evidential Grounding for Robust Video Reasoning](when_thinking_drifts_evidential_grounding_for_robust_video_reasoning.md)
- [\[NeurIPS 2025\] Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models](structured_sparse_transition_matrices_to_enable_state_tracking_in_state-space_mo.md)

<!-- RELATED:END -->
