---
title: >-
  [Paper Note] C-NAV: Towards Self-Evolving Continual Object Navigation in Open World
description: >-
  [NeurIPS 2025][Robotics][Continual Learning] This paper proposes C-Nav, a continual object navigation framework that employs a **dual-path anti-forgetting mechanism** (feature distillation + feature replay) and **LOF-based adaptive experience selection** to enable navigation agents to incrementally learn new object categories while effectively mitigating catastrophic forgetting. C-Nav surpasses full data replay baselines across 4 mainstream architectures and 2 datasets.
tags:
  - NeurIPS 2025
  - Robotics
  - Continual Learning
  - Object Navigation
  - Catastrophic Forgetting
  - Feature Distillation
  - Feature Replay
  - LOF
  - Embodied Agents
date: 2026-05-08
content_hash: 3496654f31b3dcd2
---

# C-NAV: Towards Self-Evolving Continual Object Navigation in Open World

**Conference**: NeurIPS 2025
**arXiv**: [2510.20685](https://arxiv.org/abs/2510.20685)
**Code**: [https://bigtree765.github.io/C-Nav-project](https://bigtree765.github.io/C-Nav-project)
**Area**: Robotics
**Keywords**: Continual Learning, Object Navigation, Catastrophic Forgetting, Feature Distillation, Feature Replay, LOF, Embodied Agents

## TL;DR

This paper proposes C-Nav, a continual object navigation framework that employs a **dual-path anti-forgetting mechanism** (feature distillation + feature replay) and **LOF-based adaptive experience selection** to enable navigation agents to incrementally learn new object categories while effectively mitigating catastrophic forgetting. C-Nav surpasses full data replay baselines across 4 mainstream architectures and 2 datasets.

## Background & Motivation

**Background**: Object navigation (ObjectNav) is a fundamental capability for embodied intelligence. Current SOTA methods (OVRL-V2, PIRLNav, NavID, etc.) rely on pretrained visual encoders and large-scale demonstration trajectories, assuming all target categories are provided at once during training.

**Open-World Requirements**: In real-world deployments, robots must continuously encounter new object categories and changing environments, demanding incremental learning capabilities. However, existing methods suffer an average success rate drop of approximately **40%** when sequentially learning new categories, indicating severe catastrophic forgetting.

**Cost of Data Replay**: Naive data replay (storing full trajectories) can alleviate forgetting, but navigation trajectories are extremely long (a single trajectory can span hundreds of frames), incurring **high storage overhead**, high inter-frame redundancy, and **privacy risks** (indoor scenes may expose sensitive spatial information).

**Root Cause Analysis**: Catastrophic forgetting stems from two independent sources—**representation drift** in the multimodal encoder (feature space drift caused by input distribution shift) and **policy degradation** in the action decoder (failure of the historical feature-to-action mapping)—each requiring separate constraints.

**Research Gap**: While continual learning has been extensively studied in single-modal tasks such as image classification, it has **not been systematically investigated** in object navigation, which requires long-horizon sequential decision-making and multimodal fusion. A standardized evaluation benchmark is also lacking.

**Key Insight**: The paper models keyframe selection as an **outlier detection problem** in feature space, storing only the deep features of semantically salient frames (rather than raw images), while using a dual-path mechanism to independently stabilize the encoder and decoder, achieving continual navigation at minimal storage cost.

## Method

### Overall Architecture

C-Nav consists of two core modules: (1) a **dual-path anti-forgetting mechanism**—the feature distillation path constrains representational consistency in the multimodal encoder, while the feature replay path replays keyframe features from previous tasks to the action decoder to preserve policy consistency; (2) **adaptive experience selection**—visual observations are encoded by CLIP, and the Local Outlier Factor (LOF) algorithm detects semantically anomalous frames as keyframes, storing only their encoder output features and action labels into a feature buffer.

### Key Design 1: Feature Distillation — Preserving Representational Consistency

- **Objective**: Freeze the previous-stage encoder $f_{k-1}$ and constrain the current encoder $f_k$ to produce outputs consistent with the old encoder on new data.
- **Implementation**: Minimize the $\ell_2$ distance between the outputs of the old and new encoders on the same observation: $\mathcal{L}_{\text{KD}} = \sum_{t=1}^{L} \|f_{k-1}(o_t) - f_k(o_t)\|_2^2$.
- **Design Motivation**: The multimodal encoder processes heterogeneous inputs (RGB, depth, pose, text). Learning new tasks causes input distribution shifts that induce feature space drift, causing failures on old categories even when the decoder is unchanged. Feature distillation softly constrains the continuity of the representation space.

### Key Design 2: Feature Replay — Preserving Policy Consistency

- **Objective**: During new-task training, mix keyframe encoded features and corresponding action labels from previous tasks into the action decoder's training to prevent policy forgetting.
- **Implementation**: Cross-entropy loss with inflection weighting: $\mathcal{L}_{\text{FR}} = \frac{1}{L}\sum_{t} -w_t \log \pi_k(a_t | h_{1:t})$, where action transition points receive higher weights $w_t = 1 + \gamma \cdot \mathbb{1}_{a_t \neq a_{t-1}}$ ($\gamma=3.48$).
- **Design Motivation**: Storing deep features instead of raw images substantially reduces storage and avoids privacy leakage. Inflection weighting emphasizes critical decision points such as turns and stops, which are the most prone to policy degradation.

### Key Design 3: Adaptive Experience Selection via LOF

- **Objective**: Automatically identify semantically salient frames (e.g., entering a new space, spotting the target, spatial transitions) from lengthy navigation trajectories to compress storage.
- **Implementation**: A pretrained CLIP extracts features $\mathbf{v}_i$ from RGB observations; inter-frame cosine distances are computed, and LOF measures the local outlier degree of each frame. Frames with LOF > 1 are designated as keyframes and stored in the feature buffer.
- **Design Motivation**: Adjacent frames in navigation trajectories are highly redundant (dozens of nearly identical frames when moving straight), making uniform sampling wasteful. Recasting keyframe selection as a density estimation problem retains only the few semantically distinctive frames needed to capture the trajectory's core information.

### Key Design 4: Continual-ObjectNav Benchmark Construction

- Four-stage incremental learning benchmarks are constructed based on HM3D (6 categories, 75,488 trajectories) and MP3D (21 categories, 59,604 trajectories).
- Target category sets across stages are strictly disjoint ($\mathcal{C}_i \cap \mathcal{C}_j = \emptyset$); evaluation measures SR and SPL over all seen categories.
- Four mainstream architectures are systematically evaluated: RNN-Based, Bev-Based, Transformer-Based, and LLM-Based.

## Loss & Training

The total loss is a weighted combination of three terms:

$$\mathcal{L} = \mathcal{L}_{\text{Curr}} + \lambda_{\text{KD}} \cdot \mathcal{L}_{\text{KD}} + \lambda_{\text{FR}} \cdot \mathcal{L}_{\text{FR}}$$

where $\mathcal{L}_{\text{Curr}}$ is the behavior cloning loss (with inflection weighting) on the current task, and $\lambda_{\text{KD}} = \lambda_{\text{FR}} = 5$. Training uses the AdamW optimizer with a learning rate of $3 \times 10^{-4}$, a 1,000-step linear warmup followed by linear decay, 25 epochs per stage, and a batch size of 32. Visual encoders (CLIP-ResNet50 + PointNav-ResNet50) are frozen during training.

## Key Experimental Results

### Main Results: HM3D Benchmark

| Method | HM3D-RNN Avg SR | HM3D-Trans Avg SR | HM3D-Bev Avg SR | HM3D-LLM Avg SR |
|------|:-:|:-:|:-:|:-:|
| Fine-tuning | 32.8 | 31.4 | 32.0 | 28.4 |
| LoRA | — | 34.0 | 36.3 | 39.9 |
| LwF | 34.4 | 31.7 | 32.6 | 26.2 |
| Model Merge | 40.8 | 45.1 | 45.0 | 42.5 |
| Data Replay | 44.1 | 52.7 | 53.2 | 52.2 |
| **C-Nav** | **50.0** | **55.8** | **56.3** | **52.2** |

C-Nav improves average SR over data replay by **2.75%** on HM3D (averaged across 4 architectures), with the largest gain on the RNN architecture (+5.9%). The Bev-Based architecture achieves the best overall performance.

### Main Results: MP3D Benchmark

| Method | MP3D-RNN Avg SR | MP3D-Trans Avg SR | MP3D-Bev Avg SR | MP3D-LLM Avg SR |
|------|:-:|:-:|:-:|:-:|
| Fine-tuning | 19.4 | 20.1 | 27.4 | 14.7 |
| LoRA | — | 24.2 | 29.5 | 31.1 |
| LwF | 22.0 | 20.7 | 27.6 | 14.6 |
| Model Merge | 30.3 | 33.8 | 41.7 | 26.5 |
| Data Replay | 33.8 | 37.7 | 41.8 | 26.3 |
| **C-Nav** | **36.4** | **38.1** | **42.4** | **36.1** |

C-Nav improves average SR over data replay by **3.35%** on MP3D. The LLM-Based architecture shows the largest gain (+9.8%), suggesting that feature distillation and feature replay are particularly effective for large language model decoders.

### Ablation Study: Contribution of Each Dual-Path Component

| Component | HM3D-Bev Avg SR | HM3D-Bev Last SR |
|------|:-:|:-:|
| w/o Feature Distillation (KD) | 32.5 | 21.9 |
| w/o Feature Replay (FP) | 38.9 | 26.7 |
| C-Nav (All) | **56.3** | **46.5** |

Removing feature distillation causes an approximately **22%** drop in average SR on HM3D; removing feature replay causes an approximately **12%** drop, demonstrating that the two components are complementary, with feature distillation contributing more to anti-forgetting.

### Ablation Study: Adaptive Experience Selection

Under the constraint of retaining 50% of trajectory length, adaptive sampling outperforms uniform sampling by **3.65%** SR on HM3D and **3.2%** on MP3D, and falls only 1.9%/1.3% short of full-length feature replay, validating that LOF effectively identifies high-information keyframes.

## Highlights & Insights

- **First Continual ObjectNav Benchmark**: Systematically defines the Continual-ObjectNav task, covering 4 architectures (RNN/Transformer/Bev/LLM) and 6 baseline methods, filling a critical evaluation gap in the field.
- **Dual-Path Design Targets the Root Cause of Forgetting**: Forgetting is decomposed into two independent sources—representation drift and policy degradation—with feature distillation and feature replay applied separately, yielding a logically coherent design.
- **Novel and Practical LOF Keyframe Selection**: Keyframe selection is formulated as outlier detection within density estimation, eliminating the need for manual threshold tuning and automatically adapting to diverse trajectory characteristics.
- **Storage- and Privacy-Friendly**: Storing encoder output features rather than raw RGB-D images substantially reduces storage requirements while avoiding privacy risks associated with indoor scenes.
- **Consistently Effective Across Architectures**: C-Nav outperforms full data replay across 4 architecturally diverse models, demonstrating strong generalizability.

## Limitations & Future Work

- Evaluation is conducted exclusively in simulation (Habitat); sim-to-real transfer on physical robots has not been validated.
- Each stage introduces only 1–3 new categories; scalability to large-scale category increments (e.g., 50+ categories) remains unclear.
- The LLM-Based architecture (Qwen2-0.5B) shows limited advantage with restricted training data; the impact of larger LLMs or more training data is not explored.
- Feature distillation requires retaining a full copy of the previous-stage encoder, which may incur additional memory overhead as the number of stages grows (even though encoders are frozen, they must still be loaded at inference).
- The sensitivity of LOF's neighborhood parameter $k$ is not thoroughly analyzed; different environments (e.g., densely furnished rooms vs. open corridors) may require different configurations.
- Only discrete action spaces (6 atomic actions) are evaluated; extension to continuous control or finer-grained navigation policies is not addressed.

## Related Work & Insights

- **Object Navigation**: Methods are divided into zero-shot approaches (CoW, ESC, VoroNav, VLFM, etc., leveraging VLM/LLM reasoning with frontier exploration—training-free but less efficient) and learning-based methods (OVRL-V2, PIRLNav, NavID, SPOC, etc.—end-to-end trained with high accuracy but lacking incremental learning capability). C-Nav extends the latter with incremental learning.
- **Continual Learning**: Regularization (LwF, EWC), data replay (iCaRL), architectural expansion (DER), and parameter-efficient fine-tuning (LoRA, Prompt Tuning) methods are primarily designed for image classification and transfer poorly to long-horizon multimodal navigation.
- **Key Distinction**: C-Nav is the first to introduce continual learning into ObjectNav, proposing feature-level distillation and replay (rather than logit-level or data-level), and constructs the first systematic benchmark for this setting.

## Rating

- Novelty: ⭐⭐⭐⭐ — First Continual-ObjectNav benchmark; concise and effective dual-path anti-forgetting design; novel perspective of LOF-based keyframe selection.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers 4 architectures, 2 datasets, and 6 baselines with comprehensive ablations, but lacks real-robot validation.
- Writing Quality: ⭐⭐⭐⭐ — Problem formulation is clear, framework diagrams are intuitive, and mathematical notation is rigorous; overall writing quality is high.
- Value: ⭐⭐⭐⭐ — The benchmark and method provide an important contribution to continual embodied intelligence research, though the simulation-to-real gap remains to be addressed in future work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] COOPERA: Continual Open-Ended Human-Robot Assistance](coopera_continual_open_ended_human_robot_assistance.md)
- [\[NeurIPS 2025\] EfficientNav: Towards On-Device Object-Goal Navigation with Navigation Map Caching and Retrieval](efficientnav_towards_on-device_object-goal_navigation_with_navigation_map_cachin.md)
- [\[ICCV 2025\] NavMorph: A Self-Evolving World Model for Vision-and-Language Navigation in Continuous Environments](../../ICCV2025/robotics/navmorph_a_self-evolving_world_model_for_vision-and-language_navigation_in_conti.md)
- [\[NeurIPS 2025\] DexFlyWheel: A Scalable Self-Improving Data Generation Framework for Dexterous Manipulation](dexflywheel_a_scalable_and_self-improving_data_generation_framework_for_dexterou.md)
- [\[NeurIPS 2025\] LLM World Models Are Mental: Output Layer Evidence of Brittle World Model Use in LLM Mechanical Reasoning](llm_world_models_are_mental_output_layer_evidence_of_brittle_world_model_use_in_.md)

</div>

<!-- RELATED:END -->
