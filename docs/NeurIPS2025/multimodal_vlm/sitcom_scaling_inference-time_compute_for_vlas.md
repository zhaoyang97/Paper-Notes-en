---
title: >-
  [Paper Note] SITCOM: Scaling Inference-Time COMpute for VLAs
description: >-
  [NeurIPS 2025][Multimodal VLM][Inference-time compute scaling] SITCOM proposes an inference-time compute scaling framework inspired by Model Predictive Control (MPC). It performs multi-step rollout simulation of a pretra…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "Inference-time compute scaling"
  - "VLA"
  - "world model"
  - "model predictive control"
  - "robotic manipulation"
date: 2026-05-08
content_hash: 7f2a480a18c1126a
---

# SITCOM: Scaling Inference-Time COMpute for VLAs

**Conference**: NeurIPS 2025
**arXiv**: [2510.04041](https://arxiv.org/abs/2510.04041)
**Code**: N/A
**Area**: Multimodal VLM
**Keywords**: Inference-time compute scaling, VLA, world model, model predictive control, robotic manipulation

## TL;DR

SITCOM proposes an inference-time compute scaling framework inspired by Model Predictive Control (MPC). It performs multi-step rollout simulation of a pretrained VLA using a learned dynamics model and selects optimal trajectories via a reward model, transforming a single-step VLA into a robust long-horizon planner. On the SIMPLER benchmark, it improves task success rate from 48% to 72%.

## Background & Motivation

Robot learning has long been constrained by three core challenges: the high cost of acquiring annotated data, limited generalization, and difficulty in long-horizon planning. Vision-Language-Action (VLA) models have made notable progress by grounding natural language instructions into control commands, yet they face critical limitations in real-world deployment:

**Lack of lookahead**: VLAs are inherently single-step predictors and cannot evaluate the long-term consequences of actions.

**Accumulated errors**: In open-loop execution, small errors compound across steps, causing failures in multi-step tasks.

**Poor adaptability to dynamic environments**: Plans cannot be adjusted in response to environmental changes during execution.

Existing solutions either train explicit reasoning via chain-of-thought (CoT) data—which requires expensive annotation—or employ world models that are often computationally expensive and task-specific.

**Key Insight**: SITCOM transfers the concept of inference-time compute scaling from language models to robot control. Rather than modifying the training paradigm, it "thinks more" at inference time through parallel rollouts and reward-based ranking before acting—analogous to MPC: lookahead simulation, evaluation, and selection at each decision step.

## Method

### Overall Architecture

SITCOM's inference pipeline:
1. **Candidate generation**: At each decision step, $n$ candidate actions are sampled from the VLA policy with a high sampling temperature.
2. **Rollout simulation**: For each candidate action, a dynamics model predicts the next-state image; the VLA then samples a subsequent action from that predicted image, iterating for $l$ steps to generate complete trajectories.
3. **Reward evaluation**: A reward is computed for the final state of each trajectory, incorporating gripper-object distance, object-target distance, and a grasp success indicator.
4. **Trajectory selection and execution**: The action sequence from the highest-reward trajectory is executed in the real environment.
5. **Repetition**: The loop continues at the environment's replanning frequency until task completion.

Two rollout modes are provided:
- **SITCOM (EnvSim)**: Oracle rollouts using environment instances (serves as an upper bound).
- **SITCOM (Dynamics)**: Rollouts using the learned dynamics model (the practically deployable variant).

### Key Designs

1. **Transformer dynamics model**: An encoder-decoder architecture is adopted. The encoder processes image patches concatenated with action information; the decoder predicts patches of the next frame. Joint training uses an $L_1$ pixel loss and LPIPS perceptual loss to balance low-level accuracy with high-level visual coherence. A two-stage training strategy is employed: (1) pretraining on approximately 25,000 BridgeV2 trajectories to learn general dynamics; (2) fine-tuning on SIMPLER environment trajectories to adapt to target environment visuals and physics, bridging the Real2Sim gap.

2. **DAgger-style adaptation strategy**: A model trained only on single-step prediction performs well for one step but suffers severe object reconstruction degradation during multi-step rollouts due to compounding errors. Inspired by DAgger, the model is trained to predict from its own previous predictions rather than always from ground-truth observations, aligning the training distribution with the autoregressive inference distribution and substantially reducing prediction drift in long-horizon rollouts.

3. **VLA fine-tuning**: As no public expert data exists for the SIMPLER environment, approximately 100 expert trajectories are curated: the pretrained model generates trajectories, heuristic rules filter successful executions, and human review ensures high quality. Standard cross-entropy loss is used to fine-tune the discretized action tokens.

### Loss & Training

The dynamics model employs a composite $L_1$ + LPIPS loss: $L_1$ ensures pixel-level accuracy, while LPIPS ensures perceptual visual fidelity. The reward comprises three signals: gripper-object clearance (guiding approach), object-target distance (guiding placement), and a grasp success indicator.

Default configuration: rollout length of 10 steps, 5 candidate trajectories. Planning time scales linearly with the number of candidates (approximately 35 seconds for 5 candidates, approximately 160 seconds for 25 candidates).

## Key Experimental Results

### Main Results — Task Success Rate on SIMPLER

| Task | OpenVLA | OpenVLA-SFT | SITCOM (EnvSim) | SITCOM (World Model) |
|------|---------|-------------|-----------------|---------------------|
| Put carrot on plate | 0.0 | 0.50 | 0.71 | 0.66 |
| Put spoon on tablecloth | 0.0 | 0.63 | 0.83 | 0.83 |
| Stack green on yellow block | 0.042 | 0.17 | 0.58 | 0.62 |
| Put eggplant in basket | 0.0 | 0.63 | 0.92 | 0.79 |
| **Average** | **0.01** | **0.48** | **0.76** | **0.72** |

### Dynamics Model Quality

| Model | FID↓ | OFL↓ |
|------|------|------|
| Base model (BridgeV2 only) | 17.0 | 1.665 |
| **Fine-tuned model** | **11.2** | **0.992** |

### Ablation Study — Effect of Number of Candidates

| # Candidates | 1 | 5 | 10 | 15 | 20 | 25 |
|--------|---|---|----|----|----|----|
| Planning time (s) | 21 | 35 | 75 | 100 | 130 | 160 |

### Key Findings

- Inference-time compute scaling is effective in robot control: success rate improves from 48% (OpenVLA-SFT) to 72–76% (SITCOM).
- The learned dynamics model (72%) closes to within 4% of the oracle simulator (76%), validating the approach's feasibility.
- VLA fine-tuning alone improves performance from 1% to 48%, resolving approximately 40% of the Real2Sim gap.
- Increasing the number of candidates yields consistent gains up to 25 candidates, with earlier saturation on some tasks.
- Complex tasks (e.g., placing eggplant in basket) benefit more from longer rollouts.
- DAgger-style adaptation substantially improves object reconstruction quality in long-horizon rollouts, though prediction drift is not fully eliminated.
- Primary failure modes: VLA Real2Sim gap and insufficient dexterous manipulation capability (e.g., slight mistiming of gripper closure causing objects to slip).

## Highlights & Insights

- The core idea is concise and powerful: transferring "inference-time compute scaling" from LLMs to robot control, trading more computation for better decisions.
- The MPC-style rollout-and-ranking framework is general and compatible with any VLA policy.
- The DAgger-style training strategy elegantly addresses the distributional shift inherent in autoregressive prediction.
- The qualitative analysis of failure modes is thorough and candid, clearly identifying dexterous manipulation and the Real2Sim gap as primary bottlenecks.
- The two-stage dynamics model training (large-scale pretraining + in-domain fine-tuning) is a practical and reproducible recipe.

## Limitations & Future Work

- The reward signal relies on oracle simulator states (assuming perfect environment knowledge); real-world deployment requires replacing this with a learned reward model.
- A deterministic dynamics model struggles with stochastic environments; future work could explore probabilistic action-conditioned video diffusion models.
- Inference time is a significant bottleneck (35 seconds for 5 candidates), limiting real-time control frequency.
- Evaluation is conducted only in simulation (SIMPLER); real-robot experiments are absent.
- The dynamics model is trained solely on successful trajectories, yielding unreliable predictions for failure states.
- Rollout length must be set manually, with no adaptive mechanism despite task-dependent optimal values.

## Related Work & Insights

- Unlike explicit decomposition via CoT reasoning (ECoT, zero-shot annotation), SITCOM conducts long-horizon reasoning through implicit simulation-based evaluation.
- Compared to world model approaches such as Dreamer and TD-MPC2, SITCOM predicts in pixel space using a lightweight Transformer, balancing visual fidelity with computational efficiency.
- Compared to generative video models such as GAIA-1 and UniSim, SITCOM avoids expensive diffusion architectures.
- Promising future directions include combining action chunking to reduce the number of inference calls, and developing general vision-based reward models.

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Scaling Inference-Time Search with Vision Value Model for Improved Visual Comprehension](../../ICCV2025/multimodal_vlm/scaling_inferencetime_search_with_vision_value_model_for_imp.md)
- [\[CVPR 2026\] Scaling Test-Time Robustness of Vision-Language Models via Self-Critical Inference Framework](../../CVPR2026/multimodal_vlm/scaling_test-time_robustness_of_vision-language_models_via_self-critical_inferen.md)
- [\[NeurIPS 2025\] ADMN: A Layer-Wise Adaptive Multimodal Network for Dynamic Input Noise and Compute Resources](admn_a_layerwise_adaptive_multimodal_network_for_dynamic_inp.md)
- [\[NeurIPS 2025\] BioCLIP 2: Emergent Properties from Scaling Hierarchical Contrastive Learning](bioclip_2_emergent_properties_from_scaling_hierarchical_contrastive_learning.md)
- [\[NeurIPS 2025\] NaViL: Rethinking Scaling Properties of Native Multimodal Large Language Models under Data Constraints](navil_rethinking_scaling_properties_of_native_multimodal_large_language_models_u.md)

</div>

<!-- RELATED:END -->
