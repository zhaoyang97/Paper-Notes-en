---
title: >-
  [Paper Note] FlowMotion: Training-Free Flow Guidance for Video Motion Transfer
description: >-
  [CVPR 2026][LLM Pretraining][Video Motion Transfer] FlowMotion is a training-free video motion transfer framework that directly leverages the latent prediction output of flow-based T2V models to construct motion guidance…
tags:
  - "CVPR 2026"
  - "LLM Pretraining"
  - "Video Motion Transfer"
  - "Flow Matching"
  - "Training-Free"
  - "Latent Prediction"
  - "Velocity Regularization"
date: 2026-05-08
content_hash: f1c191ae39aad357
---

# FlowMotion: Training-Free Flow Guidance for Video Motion Transfer

**Conference**: CVPR 2026  
**arXiv**: [2603.06289](https://arxiv.org/abs/2603.06289)  
**Code**: [HKUST-LongGroup/FlowMotion](https://github.com/HKUST-LongGroup/FlowMotion)  
**Area**: LLM Pretraining  
**Keywords**: Video Motion Transfer, Flow Matching, Training-Free, Latent Prediction, Velocity Regularization

## TL;DR

FlowMotion is a training-free video motion transfer framework that directly leverages the latent prediction output of flow-based T2V models to construct motion guidance signals, avoiding gradient backpropagation through internal model layers while maintaining motion fidelity and significantly reducing inference time and memory overhead.

## Background & Motivation

1. **Video motion transfer demand**: Given a source video and a text prompt, the goal is to generate a target video that preserves the motion patterns (object movement, camera trajectories, etc.) of the source while rendering a new scene—widely applicable in virtual reality, filmmaking, and related fields.
2. **High cost of training-based methods**: MotionDirector, MotionInversion, and similar methods require fine-tuning temporal attention or LoRA parameters for each reference video, taking 20 minutes to 2+ hours, making them unsuitable for real-time or large-scale scenarios.
3. **Inefficiency of existing training-free methods**: MotionClone, SMM, DiTFlow, and others depend on intermediate layer outputs (attention maps / diffusion features), requiring gradient backpropagation through deep internal layers, consuming 51–89 GB of GPU memory and 350–1800+ seconds of inference time.
4. **Internal layer dependency limits flexibility**: Existing training-free methods are tied to specific architectures (U-Net / DiT) and are difficult to generalize to new models; some also require additional inversion processes, further increasing time overhead.
5. **Rise of flow-based T2V models**: Models such as Wan and HunyuanVideo based on flow matching + DiT have become SOTA, but existing motion transfer methods have not fully exploited the properties of flow-based models.
6. **Key observation—early latent predictions encode rich temporal information**: The authors find that in the first few denoising steps of flow-based T2V models, the latent prediction (single-step estimate of the clean latent) already contains coarse motion trajectories and temporal dynamics, with appearance details accumulating subsequently—providing the theoretical basis for constructing motion guidance directly on prediction outputs.

## Method

### Overall Architecture

FlowMotion is built on top of flow-based T2V models (e.g., Wan2.1/2.2), with the core workflow:

1. **Source video motion representation extraction** (no inversion needed): The source video is encoded to clean latent $z_0^{src}$, forward-noised to $z_t^{src}$, fed to the T2V model to predict velocity $v_t^{src}$, and then the latent prediction $\hat{z}_0^{src}(t) = z_t^{src} - t \cdot v_t^{src}$ is computed as the motion representation.
2. **Flow guidance during target video generation**: In the first 10 denoising steps, the latent prediction $\hat{z}_0(t)$ of the target latent $z_t$ is computed and aligned with the source video's motion representation via a flow guidance loss. Gradients are backpropagated only to the latent itself, not through internal model layers.
3. **Velocity regularization**: The velocity at each step is regularized to suppress over-alignment and directional abrupt changes, ensuring smooth and stable motion evolution.

### Flow Guidance Design (Two Objectives)

- **Latent Alignment (LA)**: Directly aligns source and target latent predictions to maintain global motion consistency: $\mathcal{L}_{LA} = \|\hat{z}_0^{src}(t) - \hat{z}_0(t)\|_2^2$
- **Difference Alignment (DA)**: Computes frame-to-frame differences $\triangle(\hat{z}_0^{src}(t))$ and $\triangle(\hat{z}_0(t))$ and aligns them, emphasizing temporal changes and suppressing static appearance information: $\mathcal{L}_{DA} = \|\triangle(\hat{z}_0^{src}(t)) - \triangle(\hat{z}_0(t))\|_2^2$
- Total loss: $\mathcal{L}_{FG} = \alpha \cdot \mathcal{L}_{LA} + \beta \cdot \mathcal{L}_{DA}$, where $\alpha:\beta = 4:1$

### Velocity Regularization

To prevent over-fitting to appearance details and temporal instability from directly optimizing the latent prediction:

1. Compute the cumulative average velocity $v_t^{avg} = (z_t - z_1) / (t-1)$
2. Decompose the current velocity into a projection component $v_t^{proj}$ along $v_t^{avg}$ and an orthogonal component $v_t^{orth}$
3. Suppress the orthogonal component with a decay factor $\gamma=0.1$: $v_t^{reg} = v_t^{proj} + \gamma \cdot v_t^{orth}$
4. Compute the latent prediction using the regularized velocity: $\hat{z}_0(t) = z_t - t \cdot v_t^{reg}$

### Loss Function & Optimization

- Guidance is applied only during the first 10 / 50 denoising steps; at each step, Adam optimizer performs 3 iterations to optimize the target latent
- Learning rate 0.003, CFG scale = 6
- Gradients backpropagate only to the latent, not through internal model layers → extremely low memory overhead

## Key Experimental Results

### Main Results (Table 1)

| Method | Type | Backbone | Text Sim.↑ | Motion Fid.↑ | Temp. Cons.↑ | Train Time(s) | Infer Time(s) | Memory(GB) |
|--------|------|----------|-----------|-------------|-------------|-----------|-----------|---------|
| LoRA Tuning | train | Wan2.1-1.3B | 0.327 | 0.782 | 0.977 | 8100 | 135 | 25.0 |
| MotionDirector | train | ZeroScope-0.7B | 0.335 | 0.801 | 0.969 | 1662 | 140 | 28.0 |
| MotionInversion | train | ZeroScope-0.7B | 0.328 | 0.839 | 0.970 | 1170 | 115 | 24.0 |
| DeT | train | CogVideoX-2B | 0.340 | 0.812 | 0.980 | 2760 | 133 | 20.0 |
| MotionClone | free | AnimateDiff-1.3B | 0.332 | 0.786 | 0.940 | - | 804 | 51.5 |
| MOFT | free | AnimateDiff-1.3B | 0.338 | 0.582 | 0.973 | - | 576 | 75.0 |
| SMM | free | ZeroScope-0.7B | 0.322 | 0.762 | 0.958 | - | 1839 | 89.4 |
| DiTFlow | free | CogVideoX-2B | 0.350 | 0.691 | 0.983 | - | 349 | 63.5 |
| **FlowMotion** | **free** | **Wan2.1-1.3B** | **0.347** | **0.850** | **0.986** | **-** | **213** | **19.3** |

FlowMotion achieves the **best Motion Fidelity (0.850) and Temporal Consistency (0.986)**, with Text Similarity second only to DiTFlow; inference time is only 213s (fastest among training-free methods) and memory is only 19.3 GB (lowest among all methods).

### Ablation Study (Table 3)

| Variant | Text Sim.↑ | Motion Fid.↑ | Temp. Cons.↑ |
|---------|-----------|-------------|-------------|
| w/o DA (remove difference alignment) | 0.341 | 0.842 | 0.981 |
| w/o VR (remove velocity regularization) | 0.313 | 0.809 | 0.968 |
| **Full FlowMotion** | **0.347** | **0.850** | **0.986** |

Removing VR causes significant drops across all metrics (especially Text Sim. from 0.347→0.313), demonstrating that velocity regularization is crucial for stable optimization.

### Memory Efficiency Analysis (Table 4, same backbone Wan2.1-1.3B)

| Guidance Source | Memory (GB) |
|----------------|------------|
| Pure inference (no guidance) | 17.7 |
| **Latent Prediction (this method)** | **19.3** |
| Velocity output | 93.1 |
| Attention Map & Feature | OOM |

Latent prediction guidance adds only 1.6 GB over pure inference, while velocity-based guidance requires 93 GB and attention-based guidance results in OOM.

### User Study (Table 2, 20 volunteers, 1–5 scale)

| Method | Motion↑ | Temp.↑ | Text↑ | Overall↑ |
|--------|---------|--------|-------|----------|
| MotionInversion | 3.41 | 3.34 | 2.69 | 2.83 |
| DiTFlow | 2.48 | 3.18 | 3.16 | 2.63 |
| DeT | 3.87 | 3.83 | 3.38 | 3.47 |
| **FlowMotion** | **4.51** | **4.52** | **4.51** | **4.45** |

## Highlights & Insights

- **Extremely simple and efficient**: Guidance signals are based directly on model prediction outputs; gradients do not pass through internal model layers, requiring only 19.3 GB memory and 213s inference—the most efficient training-free method
- **No inversion needed**: Source video motion representation is extracted via forward noising + empty prompt, skipping the time-consuming inversion process
- **Architecture-agnostic**: Does not depend on specific attention structures or U-Net/DiT internal modules; validated on both Wan2.1-1.3B and Wan2.2-5B
- **Elegant velocity regularization design**: Decomposing velocity into projection along the cumulative direction and an orthogonal component, then decaying the orthogonal component to suppress over-alignment—a concise and effective approach

## Limitations & Future Work

- The motion representation is still global latent-level alignment, lacking fine-grained control over local/regional motion (e.g., transferring only foreground motion while keeping background free)
- Using latent prediction as the motion representation couples appearance information to some extent; the authors note that using clean latent $z_0^{src}$ instead improves accuracy but reduces text alignment and background diversity—adaptive balancing remains an open question
- Evaluation is conducted only at 480×720, 49 frames; scalability to higher resolutions and longer videos is not verified
- Baseline methods use different backbones (due to architecture incompatibility), somewhat limiting fairness

## Rating

- Novelty: ⭐⭐⭐⭐ — Approaching motion transfer from the latent prediction perspective of flow matching is novel and the design is concise
- Experimental Rigor: ⭐⭐⭐⭐ — Covers quantitative/qualitative/ablation/user study/memory analysis with comprehensive baselines
- Writing Quality: ⭐⭐⭐⭐ — Clear figures and tables, convincing motivation analysis, well-structured
- Significance: ⭐⭐⭐⭐ — Achieves significant improvements in both efficiency and performance for training-free motion transfer, with practical value

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SynCity: Training-Free Generation of 3D Worlds](../../ICCV2025/llm_pretraining/syncity_training-free_generation_of_3d_worlds.md)
- [\[ICCV 2025\] Make Your Training Flexible: Towards Deployment-Efficient Video Models](../../ICCV2025/llm_pretraining/make_your_training_flexible_towards_deployment-efficient_video_models.md)
- [\[NeurIPS 2025\] Through the River: Understanding the Benefit of Schedule-Free Methods for Language Model Training](../../NeurIPS2025/llm_pretraining/through_the_river_understanding_the_benefit_of_schedule-free_methods_for_languag.md)
- [\[ICML 2026\] XTransfer: Modality-Agnostic Few-Shot Model Transfer for Human Sensing at the Edge](../../ICML2026/llm_pretraining/xtransfer_modality-agnostic_few-shot_model_transfer_for_human_sensing_at_the_edg.md)
- [\[ICML 2026\] Compute as Teacher: Turning Inference Compute Into Reference-Free Supervision](../../ICML2026/llm_pretraining/compute_as_teacher_turning_inference_compute_into_reference-free_supervision.md)

</div>

<!-- RELATED:END -->
