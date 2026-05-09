---
title: >-
  [Paper Note] AdaDrive: Self-Adaptive Slow-Fast System for Language-Grounded Autonomous Driving
description: >-
  [ICCV 2025][Autonomous Driving][Large Language Models] AdaDrive presents the first LLM-augmented autonomous driving framework with an adaptive slow-fast architecture. Two adaptive connectors dynamically determine *when to activate the LLM* (Connector-W) and *how much the LLM contributes* (Connector-H), achieving SOTA performance on language-grounded driving benchmarks (driving score 80.9%) while reducing inference latency to 189ms and GPU memory to 6.79GB.
tags:
  - ICCV 2025
  - Autonomous Driving
  - Large Language Models
  - Adaptive Slow-Fast System
  - Language-Grounded Driving
  - Efficient Inference
date: 2026-05-08
content_hash: 125e45ef7bfcb5dd
---

# AdaDrive: Self-Adaptive Slow-Fast System for Language-Grounded Autonomous Driving

**Conference**: ICCV 2025
**arXiv**: [2511.06253](https://arxiv.org/abs/2511.06253)
**Code**: [https://github.com/ReaFly/AdaDrive](https://github.com/ReaFly/AdaDrive)
**Area**: Autonomous Driving
**Keywords**: Large Language Models, Autonomous Driving, Adaptive Slow-Fast System, Language-Grounded Driving, Efficient Inference

## TL;DR
AdaDrive presents the first LLM-augmented autonomous driving framework with an adaptive slow-fast architecture. Two adaptive connectors dynamically determine *when to activate the LLM* (Connector-W) and *how much the LLM contributes* (Connector-H), achieving SOTA performance on language-grounded driving benchmarks (driving score 80.9%) while reducing inference latency to 189ms and GPU memory to 6.79GB.

## Background & Motivation
- **State of the Field**: LLMs can provide high-level reasoning and decision-making capabilities for autonomous driving, yet efficient integration remains an open problem.
- **First-generation methods** (LMDrive, AD-H): Synchronous architectures where the LLM participates at every step—precise in reasoning but with high latency and memory overhead, precluding real-time deployment.
- **Second-generation methods** (AsyncDriver, DriveVLM): Asynchronous architectures with fixed-frequency LLM activation—reduced overhead but unable to adapt to dynamic driving scenarios. The LLM may not be invoked during emergencies, yet wastes computation in simple scenarios.
- **Root Cause**: High-frequency LLM activation ensures performance but incurs unacceptable latency; low-frequency fixed activation misses critical scenarios and lacks flexibility.
- **Key Insights**: (1) LLM activation should be scene-complexity-driven rather than fixed-frequency; (2) the LLM's contribution should not be a binary all-or-nothing decision—continuous weighted fusion (e.g., weight 0.7) outperforms full-weight fusion (1.0), as validated experimentally (Table 4: ID#3 vs. ID#4).

## Method

### Overall Architecture
AdaDrive adopts a parallel slow-fast dual-path architecture: the fast path (lightweight planner) processes every frame at high frequency for trajectory prediction; the slow path (LLM) is activated at low frequency as a cognitive unit providing decision assistance. The two paths are adaptively connected via Connector-W and Connector-H. An LS-Qformer handles temporal feature extraction, and a streaming memory buffer manages historical context.

### Key Designs

1. **Connector-W: Adaptive LLM Activation**

   - **Function**: Dynamically determines whether the LLM should be activated for the current frame.
   - **Mechanism**: An MLP predicts a confidence score $\theta_T$ from the current driving context feature $f_T'$, which is converted to a binary decision $\pi_T \in \{0, 1\}$ via Gumbel-Softmax.
   - **Adaptive Activation Loss (Core Innovation)**:
     $$\mathcal{L}_{ada} = \pi_T \cdot (\mathcal{L}_T^{LLM} + \gamma) + (1-\pi_T) \cdot \mathcal{L}_T$$
     where $\gamma = \max(d - (L_T - L_T^{LLM}), 0)$.
   - **Training Mechanism**: Two forward passes are performed at each step—one with LLM assistance ($W_T^{LLM}$) and one without ($W_T$)—comparing their respective trajectory losses. The model automatically learns to activate the LLM when it provides significant benefit ($\mathcal{L}_T^{LLM} \ll \mathcal{L}_T$).
   - **Role of Penalty Term $\gamma$**: Controls activation frequency via a preset margin $d=0.3$, ensuring LLM activation only when its contribution is sufficiently significant.
   - **Design Motivation**: No manual annotation of ground truth for "when to use the LLM" is required; comparative learning automatically discovers the optimal activation timing.

2. **Connector-H: Dynamic LLM Contribution Scaling**

   - **Function**: Controls the degree to which LLM features contribute to trajectory prediction when the LLM is activated.
   - **Mechanism**: The confidence score $\theta_T$ predicted by Connector-W serves as a continuous weighting coefficient rather than a simple full-weight fusion.
   - **Fusion Formula**: $W_T^{Fuse} = \mathcal{P}(f_T' + \theta_T \cdot f_T'')$
   - **Unified Inference Formula**:
     $$W_T = \begin{cases} \mathcal{P}(f_T'), & \text{LLM not activated} \\ \mathcal{P}(f_T' + \theta_T \cdot f_T''), & \text{LLM activated} \end{cases}$$
   - **Design Motivation**: Experiments demonstrate that continuous weighting (e.g., $\theta_T=0.7$) outperforms binary full-weight fusion ($\theta_T=1.0$), and adaptive scaling yields finer-grained feature integration.

3. **Long-Short Q-former (LS-Qformer)**

   - **Function**: Enhances temporal modeling of visual features, balancing current-frame precision with long-range context retention.
   - **Mechanism**: Learnable tokens are divided into two groups—memory tokens $\mathbf{Q}^m$ propagate across frames to aggregate long-range information, while local tokens $\mathbf{Q}^l$ focus on the current frame.
   - **Formula**: $f_T' = [\mathbf{Q}^l; \mathbf{Q}_T^m] = \text{Q-former}(\mathbf{Q}^l, \mathbf{Q}_{T-1}^m, f_T, \mathbf{I}_T)$
   - **Hyperparameters**: 20 local tokens + 20 memory tokens.
   - **Design Motivation**: Standard Q-formers process each frame independently, neglecting temporal dependencies. LS-Qformer simultaneously extracts salient current-frame features and models temporal evolution through its grouped design.

4. **Propagative Memory Fusion (PMF)**

   - **Function**: Manages a fixed-size memory buffer for streaming data, preventing unbounded GPU memory growth.
   - **Mechanism**: When the buffer is full, features of the frame to be evicted are fused into adjacent frames: $\hat{f}_{T-k+1}' = (f_{T-k}' + f_{T-k+1}')/2$
   - **Comparison with FIFO**: FIFO directly discards information from the oldest frame; PMF preserves historical context through fusion.
   - **Buffer Capacity**: $k=10$.

### Loss & Training
- AdamW optimizer with cosine learning rate scheduling; initial learning rate $1 \times 10^{-5}$.
- Training for 15 epochs; margin $d=0.3$ in the adaptive activation loss.
- The visual encoder is initialized from LMDrive pretraining and frozen; the LLM is TinyLLaMA (1.1B).
- The planner is a 4-layer Transformer with only 3M parameters.
- A warmup phase allows the LLM-assisted and LLM-free trajectory losses to converge to stable values before adaptive training begins.

## Key Experimental Results

### Main Results

| Method | LLM Params | DS ↑ | RC ↑ | IS ↑ | Memory ↓ | Latency ↓ |
|--------|-----------|------|------|------|----------|-----------|
| LMDrive (LLaMA2-7B) | 7B | 32.8 | 40.1 | 0.81 | 26.91G | 526ms |
| LMDrive (TinyLLaMA) | 1.1B | 25.2 | 38.6 | 0.71 | 16.29G | 445ms |
| AD-H (Mipha-3B) | 3.35B | 41.1 | 48.5 | 0.86 | — | — |
| **AdaDrive** | **1.1B+3M** | **42.9** | **53.4** | **0.82** | **6.79G** | **189ms** |

### Ablation Study

| ID | Connector-W | Connector-H | LS-Qformer | DS ↑ | RC ↑ | IS ↑ |
|----|-------------|-------------|------------|------|------|------|
| 1 | ✗ | ✗ | ✗ | 67.4 | 75.3 | 0.86 |
| 2 | ✗ | ✗ | ✓ | 71.9 | 82.6 | 0.84 |
| 3 | ✓ | ✗ | ✓ | 77.9 | 84.8 | 0.89 |
| 4 | ✓ | ✓ | ✓ | **80.9** | **87.6** | **0.90** |

### Key Findings
- The average activation frequency of the adaptive mechanism is only 0.28 (short routes) and 0.33 (full routes), yet performance approaches that of full activation (frequency = 1.0), with GFLOPs reduced by 62%.
- Challenging routes (dense urban streets, nighttime, mountain roads) exhibit higher activation frequencies, validating the rationale of the adaptive mechanism.
- Temporal distribution analysis shows the LLM is primarily activated at critical moments such as turns and intersections, remaining silent during cruising.
- LS-Qformer improves DS from 75.8 to 80.9 (+5.1) compared to a standard Q-former.
- PMF outperforms hard FIFO replacement; notably, a smaller memory buffer ($k=10$) yields the best results.

## Highlights & Insights
- **Elegant Design of the Adaptive Activation Loss**: No ground-truth annotation of when to activate the LLM is required; comparative learning during training automatically discovers the optimal activation timing. This design is generalizable to other systems requiring on-demand activation of expensive modules.
- **Continuous Fusion Outperforms Binary Fusion**: The ablation results for Connector-H (ID#3 vs. ID#4) reveal a valuable insight—LLM outputs should not be used at full weight; adaptive weighting is superior.
- **Extreme Efficiency**: Using a 1.1B small model with a 3M planner, AdaDrive substantially outperforms the 7B model baseline in both memory (6.79G vs. 26.91G) and speed (189ms vs. 526ms), while achieving stronger performance.
- **Interpretability of LLM Activation Patterns**: Activations are concentrated at turns and intersections, which aligns with intuitive expectations.

## Limitations & Future Work
- Training Connector-W requires two forward passes per step (with/without LLM), increasing training cost.
- The margin $d=0.3$ in the penalty term is a manually preset hyperparameter that may require tuning for different scenarios.
- The simple averaging in PMF may not be optimal; attention-weighted fusion could be more effective.
- The IS score (0.82) on long-distance tasks remains below AD-H (0.86), indicating room for improvement in safety.
- Validation is limited to CARLA simulation; real-world deployment performance remains unknown.

## Related Work & Insights
- **vs. LMDrive**: Synchronous per-step LLM invocation yields unacceptable 526ms latency; AdaDrive's adaptive invocation achieves only 189ms.
- **vs. AsyncDriver/DriveVLM**: Fixed-frequency activation cannot adapt to dynamic scenarios; AdaDrive activates on demand.
- **vs. AD-H**: AD-H employs additional intermediate language command supervision to train a hierarchical multi-agent system with a larger total parameter count (3.35B); AdaDrive is more lightweight yet achieves stronger performance.
- **vs. Flash-VStream**: Video understanding focuses on high-level semantic dialogue, whereas autonomous driving targets low-level high-frequency trajectory prediction—the two have fundamentally different objectives.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The adaptive activation loss and continuous fusion design are highly innovative; the dual-connector architecture is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Ablations are comprehensive and activation pattern analyses are convincing, though evaluation is limited to CARLA simulation.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated and figures are informative.
- Value: ⭐⭐⭐⭐⭐ Provides a practical paradigm for efficient LLM deployment in autonomous driving.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] AD-GS: Object-Aware B-Spline Gaussian Splatting for Self-Supervised Autonomous Driving](ad-gs_object-aware_b-spline_gaussian_splatting_for_self-supervised_autonomous_dr.md)
- [\[NeurIPS 2025\] AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning](../../NeurIPS2025/autonomous_driving/autovla_a_vision-language-action_model_for_end-to-end_autonomous_driving_with_ad.md)
- [\[CVPR 2026\] MeanFuser: Fast One-Step Multi-Modal Trajectory Generation and Adaptive Reconstruction via MeanFlow for End-to-End Autonomous Driving](../../CVPR2026/autonomous_driving/meanfuser_fast_one-step_multi-modal_trajectory_generation_and_adaptive_reconstru.md)
- [\[NeurIPS 2025\] HoloLLM: Multisensory Foundation Model for Language-Grounded Human Sensing and Reasoning](../../NeurIPS2025/autonomous_driving/holollm_multisensory_foundation_model_for_language-grounded_human_sensing_and_re.md)
- [\[CVPR 2026\] KnowVal: A Knowledge-Augmented and Value-Guided Autonomous Driving System](../../CVPR2026/autonomous_driving/knowval_a_knowledge-augmented_and_value-guided_autonomous_driving_system.md)

<!-- RELATED:END -->
