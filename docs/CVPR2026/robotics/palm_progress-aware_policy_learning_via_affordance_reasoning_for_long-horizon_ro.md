---
title: >-
  [Paper Note] PALM: Progress-Aware Policy Learning via Affordance Reasoning for Long-Horizon Robotic Manipulation
description: >-
  [CVPR 2026][Robotics][Long-horizon manipulation] This paper proposes PALM, a unified VLA framework that employs structured fine-grained affordance prediction across four categories (global, local, spatial, and dynamic) as implicit reasoning anchors, and incorporates continuous sub-task progress estimation to enable seamless task transitions. PALM achieves an average completion length of 4.48 on CALVIN ABCD (surpassing the previous SOTA by 12.5%), a success rate of 91.8% on LIBERO-LONG, and more than twice the baseline performance in real-world long-horizon generalization evaluations.
tags:
  - CVPR 2026
  - Robotics
  - Long-horizon manipulation
  - affordance reasoning
  - progress awareness
  - vision-language-action model
  - diffusion Transformer
date: 2026-05-08
content_hash: 8db28c3e8df199d4
---

# PALM: Progress-Aware Policy Learning via Affordance Reasoning for Long-Horizon Robotic Manipulation

**Conference**: CVPR 2026
**arXiv**: [2601.07060](https://arxiv.org/abs/2601.07060)
**Code**: [Project Page](https://plan-lab.github.io/palm)
**Area**: Object Detection / Robotic Manipulation / VLA Models
**Keywords**: Long-horizon manipulation, affordance reasoning, progress awareness, vision-language-action model, diffusion Transformer

## TL;DR

This paper proposes PALM, a unified VLA framework that employs structured fine-grained affordance prediction across four categories (global, local, spatial, and dynamic) as implicit reasoning anchors, and incorporates continuous sub-task progress estimation to enable seamless task transitions. PALM achieves an average completion length of 4.48 on CALVIN ABCD (surpassing the previous SOTA by 12.5%), a success rate of 91.8% on LIBERO-LONG, and more than twice the baseline performance in real-world long-horizon generalization evaluations.

## Background & Motivation

1. **State of the Field**: VLA models have achieved notable progress in short-horizon manipulation tasks, with representative methods spanning autoregressive approaches (OpenVLA, RT series), diffusion-based approaches (Diffusion Policy, π0), and predictive approaches (Seer). However, these models continue to struggle with long-horizon, multi-step tasks.

2. **Limitations of Prior Work**: (1) Lack of structured affordance cues — models have no explicit representation of which object to manipulate next, which part to contact, where to place it, or what motion to apply. (2) Lack of intra-subtask progress tracking — visually similar states may correspond to different action phases, leading to failure modes characteristic of long-horizon execution, including repeated actions, skipped steps, and premature termination.

3. **Root Cause**: Standard behavior cloning trains on demonstrations from different task phases without distinguishing phase differences. States that are visually similar but belong to different task phases become indistinguishable, causing policy instability during long-horizon execution.

4. **Paper Goals**: (1) Provide the policy with explicit, structured affordance representations as reasoning anchors. (2) Introduce continuous sub-task progress signals to resolve phase ambiguity and stabilize long-horizon execution.

5. **Starting Point**: Construct a closed loop of perception–action–progress. Affordance prediction serves as an intermediate implicit reasoning step, while the progress signal functions as a temporal regularizer.

6. **Core Idea**: Perform structured prediction of future interaction scenes via four categories of affordance, then jointly generate action and progress sequences using a progress-aware inverse dynamics model.

## Method

### Overall Architecture

Given a language instruction $l$, observation $o_t$, and robot state $s_t$, a multimodal encoder (CLIP text + MAE vision + MLP state) feeds into a GPT-2 Transformer backbone for feature fusion. Two sets of learnable queries are then applied: (1) fine-grained affordance queries that predict future interaction cues $\hat{\mathbf{F}}_{t+n}$; and (2) action-progress queries conditioned on the affordance representations. A diffusion Transformer (DiT) jointly decodes the action sequence $\hat{a}_{t:t+n-1}$ and progress sequence $\hat{p}_{t:t+n-1}$.

### Key Designs

1. **Fine-Grained Affordance Prediction**:

    - **Function**: Predict future interaction cues at multiple levels of granularity as a structured prior for the control policy.
    - **Mechanism**: Implemented via four dedicated sub-queries and corresponding decoding heads:
        - **Global**: Predicts "which object" (instance-level semantics), supervised by object segmentation masks obtained via Grounding DINO + SAM.
        - **Local**: Predicts "which part of the object" (contact-level geometry), supervised by Gaussian heatmaps generated from annotated contact points.
        - **Spatial**: Predicts "where to place" (candidate placement regions), supervised via a set-matching loss using candidate points sampled from SpatialVLM + RoboPoint.
        - **Dynamic**: Predicts "how to move" (motion region evolution), supervised by a VAE reconstruction loss over dynamic regions extracted by CoTracker.
    - **Design Motivation**: The four affordance types are complementary, providing task-relevant scene understanding progressively from coarse to fine.

2. **Progress-Aware Policy**:

    - **Function**: Estimate the continuous completion degree $p_t \in [0,1]$ within each sub-task to enable smooth sub-task transitions.
    - **Mechanism**: The scalar progress $p_t$ is appended to the action output, so the policy jointly predicts $(a_t, p_t)$. The progress signal resolves ambiguity between visually similar but phase-distinct states, acting as a temporal regularizer that encourages monotonic, phase-consistent evolution of latent states. Training supervision is derived from manually annotated continuous progress labels and semantic phase segmentation of long-horizon video data (EPIC-KITCHENS, RoboCerebra).
    - **Design Motivation**: Enables sub-task transitions without requiring a separate high-level planner or hierarchical controller.

3. **Inverse Dynamics-Based Diffusion Transformer Decoding**:

    - **Function**: Conditioned on the current observation and predicted affordance latents, generate the $n$-step action-progress trajectory.
    - **Mechanism**: Extends the classical inverse dynamics formulation (inferring actions from two frames) to infer an $n$-step action-progress sequence from the current input and a single-step affordance latent. A DiT performs conditional denoising: $(\hat{a}_{t:t+n-1}, \hat{p}_{t:t+n-1}) = \text{DiT}(l, o_t, s_t, \hat{\mathbf{F}}_{t+n})$, optimized with a standard diffusion noise-prediction objective.
    - **Design Motivation**: Diffusion models can capture complex multimodal action distributions and produce temporally smoother trajectories.

### Loss & Training

- Affordance losses: $\mathcal{L}_{global}$ (Focal + Dice), $\mathcal{L}_{local}$ (Focal + KL), $\mathcal{L}_{spatial}$ (set-matching L2), $\mathcal{L}_{dynamic}$ (VAE ELBO).
- Action-progress loss: standard diffusion denoising loss $\mathcal{L}_{DiT}$.
- Two-stage training: large-scale pretraining (DROID + BridgeV2 + EPIC-KITCHENS + RoboCerebra) → fine-tuning (942 manually annotated trajectories).
- Backbone: GPT-2 Transformer, 384-dim, 24 layers, 12 heads.
- Vision: MAE ViT-B + Perceiver Resampler.

## Key Experimental Results

### Main Results

CALVIN ABCD (1,000 rollouts per task):

| Method | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Avg. Len.↑ |
|--------|--------|--------|--------|--------|--------|-----------|
| OpenVLA | 91.3% | 77.8% | 62.0% | 52.1% | 43.5% | 3.27 |
| π0 | 93.8% | 85.0% | 76.7% | 68.1% | 59.9% | 3.92 |
| Seer | 94.4% | 87.2% | 79.9% | 72.2% | 64.3% | 3.98 |
| PALM (✗ progress) | 95.3% | 85.6% | 79.5% | 74.3% | 67.0% | 4.02 |
| **PALM** | **96.9%** | **93.8%** | **89.3%** | **85.9%** | **82.0%** | **4.48** |

LIBERO full suite (3 seeds × 500 episodes):

| Method | Average | Spatial | Object | Goal | Long |
|--------|---------|---------|--------|------|------|
| CoT-VLA | 81.1% | 87.5% | 91.6% | 87.6% | 69.0% |
| CoA-VLA | 79.8% | 85.3% | 93.1% | 85.8% | 55.0% |
| **PALM** | **94.5%** | **95.2%** | **96.7%** | **94.3%** | **91.8%** |

### Ablation Study

Component ablation on CALVIN ABCD:

| Ablation | Pretrain Avg. Len. | Finetune Avg. Len. |
|----------|-------------------|-------------------|
| PALM (full) | 4.48 | 4.48 |
| ✗ Affordance Foresight | 3.90 | 3.58 |
| ✗ Inverse Dynamic | 4.17 | 3.92 |
| ✗ Progress Prediction | 3.73 | 4.02 |

Real-world long-horizon generalization (6-step sequential tasks):

| Generalization Setting | OpenVLA Avg. Len. | Octo Avg. Len. | PALM Avg. Len. |
|-----------------------|------------------|----------------|----------------|
| Random positioning | 0.95 | 0.65 | **3.05** |
| Visual distraction | 1.60 | 0.95 | **3.80** |
| Unseen lighting | 1.25 | 1.05 | **3.55** |

### Key Findings

- **Progress prediction is central to long-horizon generalization**: Removing progress prediction drops the CALVIN 5-task success rate from 82.0% to 67.0% (−15%), with an even larger drop in the pretraining setting (4.48→3.73), indicating that large-scale long-horizon video data is particularly beneficial for learning progress priors.
- **Affordance prediction is most critical at fine-tuning**: Removing affordance causes the largest drop in fine-tuned Avg. Len. (4.48→3.58), demonstrating that structured affordance is indispensable when adapting to downstream robot data.
- **Cumulative contribution of four affordance types**: Performance improves incrementally from Global → +Local → +Spatial → +Dynamic, with dynamic affordance (motion region prediction) providing the final marginal gain.
- **+22.8% on LIBERO-LONG**: Success rate improves from 69.0% (CoT-VLA) to 91.8%, demonstrating the greatest advantage of affordance + progress in the most challenging long-horizon setting.
- **2–3× real-world generalization over baselines**: Across all three generalization settings, PALM consistently achieves 2–3× the Avg. Len. of OpenVLA.

## Highlights & Insights

- **Elegant closed-loop design**: Perception (affordance prediction) → action (diffusion policy) → progress (sub-task tracking) form a coherent closed loop, with each module serving a clear functional role while being tightly coupled through a shared Transformer backbone and attention mechanism.
- **Progress signal resolves phase ambiguity**: A single scalar $p_t \in [0,1]$ yields substantial gains in long-horizon performance — a simple yet highly effective regularization strategy that is broadly transferable to any policy learning setting requiring long-horizon execution.
- **Structured attention design is well-motivated**: Affordance sub-queries attend only to context tokens (maintaining decoupling), action queries attend to both context and affordance tokens (obtaining conditional information), and causal attention preserves temporal consistency.

## Limitations & Future Work

- Affordance supervision labels depend on multiple off-the-shelf tools (Grounding DINO, SAM, SpatialVLM, CoTracker), resulting in a complex annotation pipeline.
- The four affordance categories are manually defined and may omit other important interaction cues (e.g., force/tactile feedback, audio).
- Fine-tuning uses only 942 annotated trajectories and 200 real-world demonstrations; while data efficiency is favorable, generalization to entirely new task categories remains to be validated.
- Real-world evaluation is limited to single-arm tabletop manipulation; bimanual and mobile manipulation settings have not been tested.

## Related Work & Insights

- **vs. Seer**: Seer predicts future images as foresight; PALM predicts structured affordances (more compact and task-relevant). CALVIN Avg. Len. improves from 3.98 to 4.48.
- **vs. π0**: Also a diffusion policy but lacks explicit affordance and progress signals; CALVIN 5-task success rate improves from 59.9% to 82.0%.
- **vs. CoT-VLA / CoA-VLA / TraceVLA**: Various methods that enhance VLA reasoning (chain-of-thought, affordance chains, behavioral traces) but none incorporate progress tracking; the gap is pronounced on LIBERO.
- The combined paradigm of progress signals and affordance prediction is transferable to other domains requiring long-horizon planning, such as autonomous driving.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of affordance prediction and progress estimation constitutes a novel closed-loop design, though individual components (MAE, CLIP, DiT) are well-established.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Two major simulation benchmarks, three real-world generalization settings, comprehensive ablations including data ablation.
- **Writing Quality**: ⭐⭐⭐⭐ — Overall structure is clear with rich figures, though the density of methodological detail requires careful re-reading.
- **Value**: ⭐⭐⭐⭐⭐ — Achieves substantial improvements on the core challenge of long-horizon manipulation; the progress signal idea is broadly transferable.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] RoboCerebra: A Large-scale Benchmark for Long-horizon Robotic Manipulation Evaluation](../../NeurIPS2025/robotics/robocerebra_a_large-scale_benchmark_for_long-horizon_robotic_manipulation_evalua.md)
- [\[CVPR 2026\] Learning to See and Act: Task-Aware Virtual View Exploration for Robotic Manipulation](learning_to_see_and_act_task-aware_virtual_view_exploration_for_robotic_manipula.md)
- [\[CVPR 2026\] BiPreManip: Learning Affordance-Based Bimanual Preparatory Manipulation through Anticipatory Collaboration](bipremanip_learning_affordance-based_bimanual_preparatory_manipulation_through_a.md)
- [\[CVPR 2026\] GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer](geco-srt_geometry-aware_continual_adaptation_for_robotic_cross-task_sim-to-real_.md)
- [\[CVPR 2026\] Language-Grounded Decoupled Action Representation for Robotic Manipulation](language-grounded_decoupled_action_representation_for_robotic_manipulation.md)

<!-- RELATED:END -->
