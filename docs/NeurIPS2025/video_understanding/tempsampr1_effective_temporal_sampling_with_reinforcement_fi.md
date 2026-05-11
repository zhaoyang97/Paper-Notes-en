---
title: >-
  [Paper Note] TempSamp-R1: Effective Temporal Sampling with Reinforcement Fine-Tuning for Video LLMs
description: >-
  [NeurIPS 2025][Video Understanding][temporal grounding] TempSamp-R1 is a reinforcement fine-tuning framework that addresses the inefficiency of on-policy sampling in GRPO for video temporal grounding—caused by the vast s…
tags:
  - "NeurIPS 2025"
  - "Video Understanding"
  - "temporal grounding"
  - "GRPO"
  - "off-policy"
  - "soft advantage"
  - "hybrid CoT"
  - "video LLM"
date: 2026-05-08
content_hash: ea3aac6d0c9f6215
---

# TempSamp-R1: Effective Temporal Sampling with Reinforcement Fine-Tuning for Video LLMs

**Conference**: NeurIPS 2025
**arXiv**: [2509.18056](https://arxiv.org/abs/2509.18056)
**Code**: [github.com/HVision-NKU/TempSamp-R1](https://github.com/HVision-NKU/TempSamp-R1)
**Area**: Video Temporal Understanding / Reinforcement Fine-Tuning
**Keywords**: temporal grounding, GRPO, off-policy, soft advantage, hybrid CoT, video LLM

## TL;DR
TempSamp-R1 is a reinforcement fine-tuning framework that addresses the inefficiency of on-policy sampling in GRPO for video temporal grounding—caused by the vast search space—by introducing ground-truth annotations as off-policy supervision signals, non-linear soft advantage estimation, and a hybrid CoT training paradigm, achieving new state-of-the-art results on Charades-STA, ActivityNet, and QVHighlights.

## Background & Motivation
**Background**: MLLMs perform well on general video question answering but struggle with tasks requiring precise temporal understanding, such as temporal grounding and highlight detection. SFT-based approaches tend to overfit deterministic timestamp annotations and lack temporal reasoning capacity. GRPO (DeepSeek-R1-style) is effective for mathematical reasoning but yields limited gains in video temporal grounding.

**Limitations of Prior Work**: (1) The search space for video temporal grounding is enormous—locating (start, end) pairs on a continuous time axis is substantially harder than selecting from discrete mathematical answers; (2) pure on-policy GRPO sampling rarely produces high-IoU solutions in such a large search space, resulting in sparse and unstable rewards (top-1 IoU reward on ActivityNet remains persistently low and oscillates); (3) introducing high-reward off-policy solutions (e.g., ground truth) biases advantage estimation—the high reward from GT inflates the group mean, causing all on-policy advantages to become negative.

**Key Challenge**: How can a policy be effectively guided to learn precise temporal grounding in a large search space while avoiding the distributional shift introduced by off-policy samples?

**Key Insight**: GT annotations are mixed into the GRPO sampling group as off-policy solutions, while non-linear reward shaping is applied to eliminate the negative impact of distributional shift on advantage estimation.

## Method

### Overall Architecture
TempSamp-R1 is built on the GRPO framework. For each query, $G$ solutions are sampled ($G-1$ on-policy and 1 off-policy GT). IoU rewards are computed and converted into normalized advantage values via a soft advantage estimation module for policy optimization. Training proceeds in two stages: the model first learns direct output generation, then a format reward is introduced to encourage chain-of-thought reasoning. At inference time, a single model supports both CoT and non-CoT modes.

### Key Designs
1. **Mix-Policy Sampling**:
    - Function: GT annotations are mixed into the GRPO sampling group as off-policy solutions, providing precise positive signals for temporal grounding.
    - Mechanism: For each query $q$, $G-1$ solutions $\{o_1,...,o_{G-1}\}$ are sampled from the current policy $\pi_\theta$, and one external off-policy solution $o_G$ (from GT annotations) is appended. Normalized advantages are computed over the joint distribution: $A_i = \frac{r_i - \text{mean}(\{r_1,...,r_{G-1}\} \cup \{r_G\})}{\text{std}(\{r_1,...,r_{G-1}\} \cup \{r_G\})}$. An advantage anchoring strategy is also proposed: $A_G = \lambda_{\text{off}} \cdot \max\{A_i | i \in \{1,...,G-1\}\}$ (with $\lambda_{\text{off}}=1.2$) to decouple the off-policy and on-policy advantages.
    - Design Motivation: Pure on-policy GRPO in a large search space can almost never sample high-IoU solutions, yielding sparse rewards and weak learning signals. GT provides precise temporal anchors to compensate for insufficient on-policy exploration; however, the high reward from GT inflates the group mean, necessitating soft advantage estimation to remove the bias.

2. **Non-Linear Soft Advantage Estimation**:
    - Function: An asymmetric non-linear transformation is applied to rewards, compressing differences in the high-reward region while amplifying differences in the low-reward region.
    - Mechanism: A piecewise function is defined as $\tilde{r}_i = \begin{cases}\tau + \alpha_1 \cdot \ln((r_i - \tau) + 1), & r_i \geq \tau \\ \tau - \frac{e^{\alpha_2(\tau - r_i)} - 1}{e^{\alpha_2} - 1}, & r_i < \tau\end{cases}$, where $\tau=0.8$ is the threshold, $\alpha_1=0.01$ controls logarithmic compression, and $\alpha_2=1$ controls exponential amplification. The logarithmic branch suppresses gradient spikes from optimal solutions such as GT; the exponential branch amplifies the discriminability among suboptimal solutions.
    - Design Motivation: In standard GRPO, the high reward of the off-policy solution causes all on-policy advantages to become negative, incorrectly penalizing high-quality on-policy solutions. After non-linear shaping, the high-reward region is compressed and the low-reward region is amplified, yielding more informative gradients and more stable optimization.

3. **Hybrid Chain-of-Thought Training**:
    - Function: A single model is trained to support both CoT and non-CoT inference, with the mode selected at inference time according to query complexity.
    - Mechanism: Two-stage training—the initialization stage optimizes the model to generate accurate final answers (non-CoT mode), after which a format reward is introduced to encourage generating reasoning steps within `<Think>...</Think>` and final answers within `<Answer>...</Answer>`. The format reward is 1 for correct formatting and 0 otherwise. At inference, Mixed CoT takes the best result from both modes.
    - Design Motivation: Different queries have different complexity—simple queries can be answered directly, while complex queries require reasoning. CoT and non-CoT are complementary, and the Mixed mode outperforms either mode alone across all metrics.

### Loss & Training
The standard GRPO objective is adopted: $\mathcal{J}(\theta) = \frac{1}{G}\sum_{i=1}^{G}[\min(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}A_i, \text{clip}(\cdot, 1-\epsilon, 1+\epsilon)A_i) - \beta\text{KL}(\pi_\theta||\pi_{ref})]$, with $\pi_{\theta_{old}} = \pi_\theta$ for computational simplicity. Task rewards: IoU reward $R_{\text{IoU}}$ for temporal grounding; timestamp matching reward $R_{\text{ts}} = \lambda_{\text{rec}} \cdot F2 + \lambda_{\text{score}} \cdot \frac{1}{1+\text{WMSE}}$ for highlight detection. The base model is Qwen2.5-VL-7B-Instruct, trained on 4×A100 GPUs with video sampled at 2 FPS.

## Key Experimental Results

### Main Results: SOTA Comparison on Temporal Understanding Benchmarks

| Method | Type | Charades R1@0.7 | ActivityNet R1@0.5 | QVHighlights mAP |
|---|---|---|---|---|
| TimeChat | SFT | 23.7 | — | 21.7 |
| iMOVE | SFT | 45.3 | 50.7 | — |
| VideoChat-R1 | RL | 50.2 | — | — |
| TimeZero | RL | 47.9 | 47.3 | — |
| **TempSamp-R1 (no-CoT)** | RL | 52.2 | 55.4 | **30.0** |
| **TempSamp-R1 (CoT)** | RL | **52.9** | **56.0** | 28.3 |
| **TempSamp-R1 Mixed CoT** | RL | **56.3** | **58.7** | 29.3 |

### Ablation Study: Component Contributions (Charades-STA)

| Configuration | R1@0.5 | R1@0.7 | mIoU |
|---|---|---|---|
| GRPO baseline | 71.7 | 50.2 | 60.8 |
| + off-policy (reward scaling) | 72.5 | 51.1 | 61.0 |
| + off-policy (advantage anchor) | 73.0 | 51.7 | 61.3 |
| + off-policy (non-linear shaping) | 73.6 | 52.2 | 61.7 |
| + hybrid CoT (Mixed) | **76.0** | **56.3** | **64.2** |

### Key Findings
- Pure on-policy GRPO on ActivityNet yields a top-1 IoU reward persistently below 0.3 with high variance; off-policy guidance rapidly stabilizes the reward above 0.6.
- Among the three off-policy integration strategies, non-linear reward shaping > advantage anchoring > reward scaling.
- Mixed CoT outperforms standalone CoT and non-CoT on all metrics, improving mIoU by 2.1–2.5 points.
- Few-shot capability: using only 10% of training data still achieves over 90% of the performance of full-data GRPO training.

## Highlights & Insights
- The paper precisely diagnoses the root cause of GRPO's failure in temporal grounding—the large search space leads to sparse rewards under on-policy sampling.
- The piecewise design of the non-linear soft advantage is elegant: logarithmic compression suppresses gradient spikes in the high-reward region while exponential amplification enhances discriminability in the low-reward region.
- Mixed CoT is a simple yet effective design that enables the same model to adaptively select its reasoning depth.
- The work extends RL fine-tuning from mathematical reasoning to video temporal understanding, validating the cross-domain potential of the R1 paradigm.

## Limitations & Future Work
- Off-policy sampling relies on GT annotations, which are unavailable at inference time, creating an inconsistency between training exploration and inference.
- Validation is primarily on temporal grounding tasks; effectiveness on general video QA remains unexplored.
- The hyperparameters of the non-linear transformation ($\tau, \alpha_1, \alpha_2$) may require task-specific tuning.
- Experiments are conducted only on a 7B model; it is unclear whether off-policy guidance remains necessary for larger models.

## Related Work & Insights
- **vs. TimeZero/VideoChat-R1**: These GRPO-based methods rely solely on on-policy sampling. TempSamp-R1 introduces off-policy signals to address sparse rewards, improving R1@0.5 on ActivityNet by 8.7 points.
- **vs. SFT methods (iMOVE, etc.)**: SFT overfits to deterministic timestamps, whereas RL fine-tuning learns more flexible temporal reasoning. TempSamp-R1 Mixed CoT surpasses iMOVE by 11 points on Charades R1@0.7.
- **Insight**: In RL tasks with large search spaces, judiciously incorporating off-policy expert signals may be a broadly effective strategy; the non-linear reward shaping approach is generalizable to other RL fine-tuning scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ Extending R1-style RL to video temporal grounding is valuable; the combination of off-policy sampling and soft advantage estimation is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ State-of-the-art results on 3 benchmarks, detailed ablation studies, and few-shot evaluation.
- Writing Quality: ⭐⭐⭐⭐ Problem analysis is thorough; the motivation-to-solution logical chain is clear.
- Value: ⭐⭐⭐⭐ Provides a practical RL fine-tuning framework for video temporal understanding; the Mixed CoT design is reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models](structured_sparse_transition_matrices_to_enable_state_tracking_in_state-space_mo.md)
- [\[NeurIPS 2025\] Seeing the Arrow of Time in Large Multimodal Models](seeing_the_arrow_of_time_in_large_multimodal_models.md)
- [\[NeurIPS 2025\] Enhancing Temporal Understanding in Video-LLMs through Stacked Temporal Attention in Vision Encoders](enhancing_temporal_understanding_in_videollms_through_stacke.md)
- [\[NeurIPS 2025\] INST-IT: Boosting Instance Understanding via Explicit Visual Prompt Instruction Tuning](inst-it_boosting_instance_understanding_via_explicit_visual_prompt_instruction_t.md)
- [\[NeurIPS 2025\] PixFoundation 2.0: Do Video Multi-Modal LLMs Use Motion in Visual Grounding?](pixfoundation_20_do_video_multi-modal_llms_use_motion_in_visual_grounding.md)

</div>

<!-- RELATED:END -->
