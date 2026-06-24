---
title: >-
  [Paper Note] RL-RC-DoT: A Block-level RL Agent for Task-Aware Video Compression
description: >-
  [CVPR 2025][LLM Agent][Video compression] Proposes RL-RC-DoT, a reinforcement learning-based macroblock-level quantization parameter (QP) control agent for task-aware video compression. By modeling QP selection as a sequential decision-making problem in RL, the agent learns to allocate more bitrate to task-relevant regions under given bitrate constraints, significantly improving performance on vehicle detection and ROI saliency coding tasks. A key advantage is that it does no…
tags:
  - "CVPR 2025"
  - "LLM Agent"
  - "Video compression"
  - "Reinforcement learning"
  - "Task-aware coding"
  - "Quantization parameter"
  - "Macroblock-level control"
date: 2026-05-08
content_hash: 9679933f1d18121a
---

# RL-RC-DoT: A Block-level RL Agent for Task-Aware Video Compression

**Conference**: CVPR 2025  
**arXiv**: [2501.12216](https://arxiv.org/abs/2501.12216)  
**Code**: To be confirmed  
**Area**: LLM Agent  
**Keywords**: Video compression, Reinforcement learning, Task-aware coding, Quantization parameter, Macroblock-level control

## TL;DR
Proposes RL-RC-DoT, a reinforcement learning-based macroblock-level quantization parameter (QP) control agent for task-aware video compression. By modeling QP selection as a sequential decision-making problem in RL, the agent learns to allocate more bitrate to task-relevant regions under given bitrate constraints, significantly improving performance on vehicle detection and ROI saliency coding tasks. A key advantage is that it does not require running downstream task models during inference, making it suitable for edge device deployment.

## Background & Motivation
**Background**: Traditional video encoders (H.264/H.265/VVC) are optimized for human perception, minimizing reconstruction error ($PSNR$/$SSIM$). However, in modern applications, the vast majority of videos are processed by AI systems—such as object detection in autonomous driving, action recognition in security, and industrial quality inspection.

**Limitations of Prior Work**: Encoding optimized for human perception is sub-optimal for downstream AI tasks at equivalent bitrates—the regions AI focuses on differ from those of human attention. For example, small distant vehicles are crucial for detection, but $PSNR$ optimization might allocate bitrate to large areas of the sky.

**Key Challenge**: The need to achieve task-aware bitrate allocation while maintaining compatibility with standard encoders (requiring no modifications at the decoder side). While entirely new neural encoders are flexible, they are incompatible with existing infrastructure.

**Goal**: To control quantization parameters (QP) at a macroblock granularity within standard encoder frameworks, prioritizing bitrate allocation for task-relevant regions.

**Key Insight**: Modeling QP optimization as a sequential decision-making task in reinforcement learning, where the agent learns the long-term impact of QP selection on task performance and bitrate constraints.

**Core Idea**: An RL agent selects QPs at the macroblock level to optimize downstream task performance rather than reconstruction quality, without requiring downstream models during inference.

### Loss & Training
Trained using standard policy gradient methods in RL. The reward consists of two components: (1) improvements in downstream task performance metrics (e.g., $mAP$); and (2) a penalty term for bitrate constraint satisfaction. Training requires the complete encoding-decoding-detection pipeline, but only a lightweight policy network is retained during inference.

## Method

### Overall Architecture
In standard video encoders, the QP of each macroblock is determined by an RL agent, rather than using fixed or rate-distortion optimized QPs. The agent learns to maximize downstream task performance under bitrate constraints.

### Key Designs

1. **Macroblock-level QP Control**:

    - **Function**: To select QP values (e.g., QP range 0-51 in H.264/H.265) independently for each macroblock.
    - **Mechanism**: Task-relevant regions use low QPs (high quality, high bitrate), while irrelevant regions use high QPs (high compression, low bitrate). For example, in vehicle detection tasks, macroblocks containing vehicles are allocated more bitrate.
    - **Design Motivation**: Frame-level QP control is too coarse—the task importance of different regions within a single frame can vary drastically.

2. **RL Optimization Framework**:

    - **Function**: The agent learns to balance the long-term impact of QP selection.
    - **Mechanism**: State = current frame content and encoding states (used bitrate, buffer status); Action = macroblock QP selection; Reward = downstream task performance (e.g., $mAP$) + bitrate constraint satisfaction. Temporal difference learning is used to handle cross-frame dependencies.
    - **Design Motivation**: QP selection has cross-frame dependencies—consuming too much bitrate for the current frame squeezes the budget for subsequent frames. The credit assignment mechanism of RL is naturally suited for handling such delayed effects.

3. **Zero Downstream Models During Inference**:

    - **Function**: The trained policy directly predicts QPs from video content features without needing to run detection/segmentation models.
    - **Mechanism**: During the training phase, the agent learns "which visual features are important for the task" by interacting with the downstream task model, and encodes this knowledge into the weights of the policy network during inference.
    - **Design Motivation**: Suitable for streaming media and edge devices (e.g., in-vehicle cameras) to reduce computational overhead, as the encoding side typically has limited computing power.

## Key Experimental Results

### Vehicle Detection Task (BDD100K, YOLO-v5)

| Metric | RL-RC-DoT BD-rate | Description |
|------|:---:|------|
| Detection Precision | **-24.7% ± 1.38%** | Saves 24.7% bitrate at the same performance level |
| PSNR | +1.19% ± 0.46% | Slight degradation in human perception quality |

Cross-model validation: SSD detector achieves BD-rate $\approx$ -23%, and DeepLab segmentation also shows significant improvement.

### ROI Saliency Coding

| Metric | BD-rate | Description |
|------|:---:|------|
| Saliency-weighted PSNR | **-25.64% ± 0.99%** | Substantial quality improvement in important regions |
| Standard PSNR | -5.26% ± 0.36% | Overall quality also sees improvement |

### Other Downstream Tasks
- Video segmentation (DAVIS): BD-rate = **-8%**
- Multi-object tracking (ByteTracker): BD-rate = **-3.2%**

### Ablation Study

| Configuration | Precision BD-rate | ROI PSNR BD-rate | Description |
|------|:---:|:---:|------|
| Full RL-RC-DoT | **-24.7%** | **-25.64%** | Full model |
| w/o Reward Info | -21.3% | -20.51% | Without extra reward info |
| $\gamma=0$ (Myopic Policy) | -15.2% | -12.8% | Disregards long-term cross-frame impact |

### Key Findings
- KL divergence between QP map and Eigen-CAM: RL-RC-DoT 2.6 vs x264 4.4—the agent indeed learns to focus on task-relevant regions.
- For $\gamma=0$ (myopic policy), BD-rate drops from -24.7% to -15.2%, proving that considering cross-frame long-term impacts is crucial.
- Inference speed: 0.004 seconds/frame = 250 FPS, yielding negligible encoding overhead.
- Training: 8 parallel environments + V100 32GB, ~4 days for 20 million frames.

## Highlights & Insights
- **Zero dependence on downstream models during inference** is a key practical advantage—enabling the method to be deployed on resource-constrained edge devices (e.g., in-vehicle cameras). This resolves a long-standing pain point in task-aware coding, where many methods require running detection models on the encoder side, which is unrealistic in computationally restricted scenarios.
- **Rationality of RL modeling**: The impact of QP selection is not limited to the current macroblock but affects the encoding quality of subsequent frames through the bitrate budget, which is a typical sequential decision-making problem. RL is better suited than myopic optimization to handle such trade-offs across time steps.
- **Compatibility with standard encoders**: Only the QP parameters are controlled without modifying the encoder itself, ensuring that compressed videos can be processed by any standard decoder.

## Limitations & Future Work
- RL training requires downstream task models to participate in reward computation, resulting in high training costs—reward model distillation could be explored to reduce costs.
- Different RL policies need to be trained for different downstream tasks—multi-task RL or task-conditioned policies could be considered.
- Lack of comparison with state-of-the-art neural video encoders (such as DCVC-DC, etc.).
- Macroblock granularity might not be fine enough for high-resolution video—pixel-level or CTU-level control could be explored.
- Only two tasks (detection and ROI) were validated; this could be extended to more tasks like segmentation, tracking, etc.

## Related Work & Insights
- **vs Rate-Distortion Optimization (RDO)**: Traditional RDO optimizes reconstruction distortion, whereas RL-RC-DoT optimizes task performance—representing a fundamental shift in coding objectives.
- **vs Scalable Coding / ROI Coding**: ROI coding requires manual annotation of important regions, whereas RL-RC-DoT automatically learns regional importance.
- **Insights for Edge AI**: Deploying lightweight RL policies on the sensor side to optimize data transmission is a general paradigm, which is not limited to video coding.

## Rating
- Novelty: ⭐⭐⭐ Running RL for video coding parameter optimization is not entirely new, but the combination of macroblock-level + task-awareness + no downstream task model at inference time is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validation on two tasks + rate-performance curve analysis.
- Writing Quality: ⭐⭐⭐⭐ Clearly defined problems.
- Value: ⭐⭐⭐⭐ Directly valuable for task-oriented video compression and edge AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] GTR: Guided Thought Reinforcement Prevents Thought Collapse in RL-based VLM Agent Training](../../ICCV2025/llm_agent/gtr_guided_thought_reinforcement_prevents_thought_collapse_i.md)
- [\[AAAI 2026\] COVR: Collaborative Optimization of VLMs and RL Agent for Visual-Based Control](../../AAAI2026/llm_agent/covrcollaborative_optimization_of_vlms_and_rl_agent_for_visu.md)
- [\[ICLR 2026\] AgentGym-RL: An Open-Source Framework to Train LLM Agents for Long-Horizon Decision Making via Multi-Turn RL](../../ICLR2026/llm_agent/agentgym-rl_an_open-source_framework_to_train_llm_agents_for_long-horizon_decisi.md)
- [\[ICLR 2026\] Meta-RL Induces Exploration in Language Agents](../../ICLR2026/llm_agent/meta-rl_induces_exploration_in_language_agents.md)
- [\[ACL 2026\] SOLAR-RL: Semi-Online Long-horizon Assignment Reinforcement Learning](../../ACL2026/llm_agent/solar-rl_semi-online_long-horizon_assignment_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
