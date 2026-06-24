---
title: >-
  [Paper Note] Diffusion-VLA: Generalizable and Interpretable Robot Foundation Model via Self-Generated Reasoning
description: >-
  [ICML 2025][VLM Reasoning][Vision-Language-Action] DiVLA (Diffusion-VLA) is proposed to unify the reasoning capabilities of autoregressive VLMs and the action generation capabilities of diffusion models into an end-to-end framework. By directly embedding self-generated language reasoning into policy learning via a Reasoning Injection Module, DiVLA achieves generalization to unseen objects, interpretable action decision-making, and high-speed inference (82Hz for the 2B model).
tags:
  - "ICML 2025"
  - "VLM Reasoning"
  - "Vision-Language-Action"
  - "Diffusion Policy"
  - "Autoregressive Reasoning"
  - "Robot Manipulation"
  - "Multimodal Foundation Models"
date: 2026-05-08
content_hash: 40e30bcff1be70ae
---

# Diffusion-VLA: Generalizable and Interpretable Robot Foundation Model via Self-Generated Reasoning

**Conference**: ICML 2025  
**arXiv**: [2412.03293](https://arxiv.org/abs/2412.03293)  
**Code**: None  
**Area**: Multimodal VLM  
**Keywords**: Vision-Language-Action, Diffusion Policy, Autoregressive Reasoning, Robot Manipulation, Multimodal Foundation Models

## TL;DR

DiVLA (Diffusion-VLA) is proposed to unify the reasoning capabilities of autoregressive VLMs and the action generation capabilities of diffusion models into an end-to-end framework. By directly embedding self-generated language reasoning into policy learning via a Reasoning Injection Module, DiVLA achieves generalization to unseen objects, interpretable action decision-making, and high-speed inference (82Hz for the 2B model).

## Background & Motivation

Existing robot foundation models suffer from a fundamental trade-off between two paradigms:

**Autoregressive VLA Models** (e.g., RT-2, OpenVLA): By modeling action prediction as next-token prediction, they inherit the reasoning capability of LLMs. However, discretizing continuous actions into fixed-size tokens compromises action coherence and precision. Moreover, step-by-step token generation leads to low inference efficiency in real-time control scenarios.

**Diffusion Policy Models** (e.g., Diffusion Policy): By modeling action sequences via a noise-denoising process, they better capture the multimodal distribution of robot actions and offer faster generation speeds. However, they naturally lack reasoning capabilities, making them struggle with complex tasks requiring semantic understanding.

**Core Problem**: Can the reasoning capabilities of autoregressive models be combined with the robust, high-frequency action generation benefits of diffusion models? Simple concatenation fails to fully exploit the reasoning potential, as there remains an implicit gap between logical reasoning and executable robotic policies.

## Method

### Overall Architecture

DiVLA is built upon a pretrained Vision-Language Model (VLM) and comprises three core components:

1. **Vision Encoder**: Uses SigLIP to encode multi-view images (wrist camera + external camera) into dense visual features, which are then mapped to a fixed number of $N$ visual embeddings via a Transformer. For multi-view inputs, features from each view are encoded by a shared SigLIP backbone and then concatenated.

2. **VLM Backbone**: Adopts Qwen2-VL (available in 2B, 8B, and 72B variants), retaining its autoregressive text generation capabilities for reasoning. It is initialized with open-source pretrained weights. The framework design decouples vision-language understanding from action generation, allowing for flexible replacement with other VLMs.

3. **Diffusion Action Head**: The final layer embedding of the VLM generates a fixed number of action tokens. These tokens are transformed by a projection module (a two-layer MLP + LayerNorm, similar to LLaVA's projector design) and fed into a standard Diffusion Policy decoder, which is initialized with random weights. A bottom-level MLP is appended to predict robot joint-space actions. For multi-embodiment robots, quick adaptation is achieved by simply initializing new MLP layers.

### Key Designs

#### Reasoning Injection Module

This is the core contribution of the paper. Unlike most autoregressive VLAs that recursively recurrentize the reasoning output as input for the next step, DiVLA proposes a more efficient direct embedding strategy:

- **Reasoning Generation**: The VLM autoregressively generates task decomposition and explanatory text (e.g., "grasp the blue toy car", "place it in the toy bin").
- **Injection Mechanism**: The final tokenized embedding of the reasoning output is extracted and directly injected into each layer of the policy network via **Feature-wise Linear Modulation (FiLM)**. FiLM modulates the policy network features by learning affine transformations $\gamma$ and $\beta$:

$$h_{out} = \gamma(r) \odot h_{in} + \beta(r)$$

where $r$ is the reasoning embedding and $h_{in}$ denotes the intermediate features of the policy network.

- **Design Philosophy**: The policy network primarily processes action-related tokens. The reasoning module acts as an auxiliary enhancement, providing semantic context without dominating the main decision stream. This "injection" rather than "concatenation" design avoids the computational and operational complexity of iterative input-output loops.

#### Reasoning Data Augmentation

The original Droid pretraining data only contains robot actions and partial observations/language instructions, lacking reasoning trajectories. The authors leverage GPT-4o to automatically convert the raw data into a format containing reasoning annotations, maintaining consistency in the network architecture across pretraining and fine-tuning.

#### Multi-Embodiment Adaptation

When deploying to different robot embodiments, unlike Octo which duplicates independent action decoders, DiVLA only needs to initialize and train a new MLP layer for evaluation, preserving the knowledge from the pretraining data for rapid adaptation.

### Loss & Training

**Joint Training Objective**:

$$L = L_{diff} + \alpha \cdot L_{ntp}$$

- $L_{diff}$: Diffusion loss for action generation.
- $L_{ntp}$: Next-token prediction loss for reasoning text generation.
- In experiments, the scale of $L_{ntp}$ is consistently about one-tenth of $L_{diff}$; thus, $\alpha = 10$ is set to balance the two losses.

**Training Details**:
- Pretraining Data: Droid (39K trajectories) for DiVLA-2B/7B; OXE + Droid for DiVLA-72B.
- Fine-tuning Method: The vision encoder and VLM are frozen, and LoRA is used to fine-tune the VLM.
- Learning Rate: Fixed at 2e-5.
- Fine-tuning Data: Multi-task data for the same embodiment is trained jointly.

## Key Experimental Results

### Main Results

**Multi-task learning (5 tasks, Franka robot)**:

| Model | Pretrained Trajectories | In-Distribution Avg Success Rate | Visual Generalization Avg Success Rate |
|------|------------|----------------|-----------------|
| Diffusion Policy | - | 27.9% | 8.9% |
| TinyVLA | - | 45.5% | 28.9% |
| Octo | 970K | 24.3% | 17.8% |
| OpenVLA-7B | 970K | 39.4% | 26.7% |
| **DiVLA-2B** | **39K** | **83.6%** | **57.8%** |

**Zero-shot Bin Picking (102 unseen objects)**:

| Model | Success Rate |
|------|-------|
| Diffusion Policy | 8.9% |
| Octo | 19.6% |
| TinyVLA | 23.5% |
| OpenVLA | 28.4% |
| **DiVLA** | **63.7%** |

**Dual-arm Desktop Clearing (AgileX Dual-arm Robot)**:

| Scenario | Diffusion Policy | OpenVLA | DiVLA-2B |
|------|-----------------|---------|----------|
| Seen Objects | 45.8% | 0% | **72.9%** |
| Mixed Objects | 31.2% | 0% | **70.8%** |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| Diffusion Policy only (no reasoning) | 27.9% (Multi-task) | Lack of semantic understanding leads to poor generalization |
| Autoregressive VLA only | 39.4% (OpenVLA) | Action precision and speed are limited |
| DiVLA without reasoning injection | Not reported | Implicit gap between reasoning and action |
| DiVLA full scheme | 83.6% (Multi-task) | Reasoning injection significantly improves generalization |
| DiVLA-2B inference speed | 82 Hz | Single A6000 GPU |
| DiVLA-7B inference speed | 42 Hz | Single A6000 GPU |

### Key Findings

1. **High Data Efficiency**: DiVLA-2B outperforms Octo and OpenVLA (retrained on 970K trajectories) using only 39K pretraining trajectories, improving the in-distribution success rate by 44.2% (absolute percentage).

2. **Visual Robustness**: Under three types of visual variations (added distractors, changed background, altered lighting), all methods degrade, but DiVLA consistently maintains the highest success rate without requiring any data augmentation.

3. **Cluttered Bin Picking Performance**: In complex scenarios (6-11 objects randomly stacked), other methods degrade dramatically (DP drops to 9.2%), while DiVLA maintains an approximate 60% success rate (overall average 66.2%), outperforming the runner-up OpenVLA by 20.9%.

4. **Explainable Reasoning**: The model adaptively adjusts its reasoning—when the target object is replaced, the reasoning text immediately switches from "grasp the blue toy car" to "grasp the hex wrench", indicating that the model makes context-aware decisions rather than executing blind pre-programmed actions.

5. **Semantic Generalization by Analogy**: The model generalizes by classifying a screwdriver as a hex wrench (based on visual similarity) or identifying a green glove as a "green glove", demonstrating semantic generalization capability beyond simple template matching.

6. **Model Scalability**: The DiVLA family scales from 2B to 72B, with generalization and performance increasing with model capacity, conforming to the scaling laws.

## Highlights & Insights

1. **Elegant Architectural Design**: Fusing reasoning and action via FiLM injection rather than recursive concatenation retains the policy enhancement from reasoning without extra computational overhead—reasoning injection does not increase computation at inference time.

2. **Unified "Say-Can" Model**: DiVLA retains both conversational question-answering capabilities and robot control capabilities, which has historically been difficult to achieve in a single model.

3. **Practical Value**: An inference speed of 82Hz satisfies real-time control requirements. Observing the self-generated reasoning text allows direct failure diagnosis, which is critical for real-world deployment.

4. **Lightweight Cross-Embodiment Transfer**: Swapping only the final MLP layer adapts the model to new robot configurations (e.g., from Franka to AgileX dual-arm), yielding high knowledge reuse efficiency.

## Limitations & Future Work

1. **Lack of Simulation Benchmarks**: Evaluation is conducted primarily on real robots, lacking comparison with more baselines on standard simulation benchmarks (e.g., CALVIN, RLBench).
2. **Dependency of Reasoning Data on GPT-4o**: Preparing reasoning annotations during pretraining relies on GPT-4o generation, which is costly and potentially noisy. Future work could investigate automated or cheaper ways to collect reasoning data.
3. **Insufficient Ablations**: The exact quantitative contribution of the reasoning injection module (compared to no injection) is not presented as an independent ablation.
4. **Coarse-Grained Reasoning**: Current reasoning only yields phrase-level descriptions without exploring finer-grained or multi-step reasoning chains for complex, long-horizon tasks.
5. **Practicality of the 72B Model**: While showing scaling effects, the paper does not report the inference speed of the 72B model, leaving its feasibility for real-time control unverified.

## Related Work & Insights

- **RT-2 / OpenVLA**: Pioneered NTP for robot learning, but action discretization and slow inference speeds remain bottlenecks. DiVLA addresses this by restricting NTP to reasoning tasks and delegating actions to a diffusion model.
- **Diffusion Policy**: Demonstrated the strength of diffusion models in capturing multimodal action distributions but completely lacks linguistic reasoning. DiVLA bridges this gap.
- **π₀ (Physical Intelligence)**: Similarly integrates language models with action generation but uses flow matching instead of diffusion. DiVLA's reasoning injection module provides more explicit interpretability.
- **Transfusion / Show-O**: Explored unifying NTP and diffusion in image generation; DiVLA transfers this concept to the robotics domain.
- **FiLM Conditioning**: Inspired by RT-1 and YAY, using language conditioning to modulate policy networks has precedent. DiVLA's innovation lies in using *self-generated* reasoning (rather than human instructions) as the modulation signal.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The reasoning injection design is creative, though unifying autoregressive and diffusion models has already been explored in visual generation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive multi-task evaluation on real robots, but lacks simulation benchmarks and thorough ablation experiments.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured and clearly motivated, though some experimental details are deferred to the appendix.
- **Value**: ⭐⭐⭐⭐⭐ — High practical value; 82Hz real-time inference, interpretable decision-making, and strong generalization capability are critical for robot deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Thinking Diffusion: Penalize and Guide Visual-Grounded Reasoning in Diffusion Multimodal Language Models](../../CVPR2026/vlm_reasoning/thinking_diffusion_penalize_and_guide_visual-grounded_reasoning_in_diffusion_mul.md)
- [\[CVPR 2026\] dMLLM-TTS: Self-Verified and Efficient Test-Time Scaling for Diffusion Multi-Modal Large Language Models](../../CVPR2026/vlm_reasoning/dmllm-tts_self-verified_and_efficient_test-time_scaling_for_diffusion_multi-moda.md)
- [\[ICLR 2026\] Vision-SR1: Self-Rewarding Vision-Language Model via Reasoning Decomposition and Multi-Reward Policy Optimization](../../ICLR2026/vlm_reasoning/vision-sr1_self-rewarding_vision-language_model_via_reasoning_decomposition_and_.md)
- [\[CVPR 2026\] SegCompass: Exploring Interpretable Alignment with Sparse Autoencoders for Enhanced Reasoning Segmentation](../../CVPR2026/vlm_reasoning/segcompass_exploring_interpretable_alignment_with_sparse_autoencoders_for_enhanc.md)
- [\[CVPR 2026\] ReAlign: Generalizable Image Forgery Detection via Reasoning-Aligned Representation](../../CVPR2026/vlm_reasoning/realign_generalizable_image_forgery_detection_via_reasoning-aligned_representati.md)

</div>

<!-- RELATED:END -->
