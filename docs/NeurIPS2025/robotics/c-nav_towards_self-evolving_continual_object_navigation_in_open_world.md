---
title: >-
  [Paper Note] C-NAV: Towards Self-Evolving Continual Object Navigation in Open World
description: >-
  [NeurIPS 2025][Robotics][continual learning] The paper proposes C-Nav, a framework that employs **dual-path anti-forgetting** (feature distillation + feature replay) and **adaptive experience selection** (LOF-based anomaly detection for keyframe selection) to prevent catastrophic forgetting as a navigation agent incrementally learns new object categories, surpassing full data replay baselines across 4 different architectures.
tags:
  - NeurIPS 2025
  - Robotics
  - continual learning
  - object navigation
  - catastrophic forgetting
  - feature distillation
  - feature replay
  - LOF
date: 2026-05-08
content_hash: 7a8d82639db444f9
---

# C-NAV: Towards Self-Evolving Continual Object Navigation in Open World

**Conference**: NeurIPS 2025
**arXiv**: [2510.20685](https://arxiv.org/abs/2510.20685)
**Code**: [https://bigtree765.github.io/C-Nav-project](https://bigtree765.github.io/C-Nav-project)
**Area**: Robotics
**Keywords**: continual learning, object navigation, catastrophic forgetting, feature distillation, feature replay, LOF

## TL;DR

The paper proposes C-Nav, a framework that employs **dual-path anti-forgetting** (feature distillation + feature replay) and **adaptive experience selection** (LOF-based anomaly detection for keyframe selection) to prevent catastrophic forgetting as a navigation agent incrementally learns new object categories, surpassing full data replay baselines across 4 different architectures.

## Background & Motivation

**Background**: Object navigation (ObjectNav) is a core task in embodied AI. Current SOTA methods (OVRL-V2, PIRLNav, NavID, etc.) rely on pretrained visual encoders and large-scale demonstration trajectories, achieving strong performance on fixed category sets.

**Limitations of Prior Work**: These methods assume all categories and data are available at once during training. In open-world settings that continuously incorporate new objects, model parameter updates cause **catastrophic forgetting**—performance on old categories degrades sharply (~40% drop in SR) after learning new ones.

**Key Challenge**: Direct data replay (storing full trajectories) can mitigate forgetting, but navigation trajectories are extremely long (up to hundreds of frames per episode), highly redundant, and privacy-sensitive (indoor spatial layouts are exposed), making storage and privacy costs prohibitive.

**Goal**: Enable a navigation agent to **incrementally learn new categories** while retaining navigation skills for old ones, **without storing raw trajectories**.

**Key Insight**: The paper decomposes forgetting into two independent sources—**representation drift** in the encoder and **policy degradation** in the decoder—and applies separate constraints to each. Keyframe selection is formulated as an outlier detection problem in feature space to compress storage.

**Core Idea**: Dual-path anti-forgetting (feature distillation to stabilize encoder representations + feature replay to stabilize action decoder policy) combined with LOF-based adaptive experience selection (storing only features of semantically salient frames rather than raw images).

## Method

### Overall Architecture

C-Nav consists of two major modules: (1) **dual-path anti-forgetting mechanism**—a feature distillation path constrains the consistency of multimodal encoder outputs, while a feature replay path replays keyframe features from previous tasks to the action decoder; (2) **adaptive experience selection**—CLIP-extracted visual features are processed by the LOF algorithm to detect semantically anomalous frames as keyframes, storing only their deep features and action labels.

### Key Design 1: Feature Distillation for Representation Consistency

- **Function**: The previous-stage encoder $f_{k-1}$ is frozen; during training on new data, the current encoder $f_k$ is constrained to produce outputs close to those of $f_{k-1}$.
- **Mechanism**: Minimize the $\ell_2$ distance between old and new encoder outputs on the same observations: $\mathcal{L}_{\text{KD}} = \sum_{t=1}^{L} \|f_{k-1}(o_t) - f_k(o_t)\|_2^2$.
- **Design Motivation**: The multimodal encoder processes RGB, depth, pose, and text inputs; distributional shift causes feature space drift, causing the downstream decoder to fail even if its parameters are unchanged.

### Key Design 2: Feature Replay for Policy Consistency

- **Function**: Stores encoded features $h_t \in \mathbb{R}^d$ and corresponding action labels from keyframes of previous tasks, mixing them with current data during decoder training.
- **Mechanism**: Uses cross-entropy loss with inflection weighting: $\mathcal{L}_{\text{FR}} = \frac{1}{L}\sum_{t=1}^{L} -w_t \log \pi_k(a_t | h_{1:t})$, where action transition points receive higher weight ($w_t = 1 + \gamma \cdot \mathbb{1}_{a_t \neq a_{t-1}}$).
- **Design Motivation**: Storing features instead of raw images avoids privacy leakage and dramatically reduces storage; inflection weighting emphasizes critical decision points such as turns and stops.

### Key Design 3: Adaptive Experience Selection via LOF

- **Function**: Automatically selects semantically salient keyframes from each trajectory rather than using uniform sampling or storing all frames.
- **Mechanism**: CLIP encodes RGB observations into features $\mathbf{v}_i$; the Local Outlier Factor (LOF) is computed for each frame, and frames with $\text{LOF}(\mathbf{v}_i) > 1$ are selected as keyframes. A high LOF indicates that a frame deviates from its neighborhood density in feature space—typically corresponding to semantically salient moments such as entering a new room, discovering the target object, or navigating a path transition.
- **Design Motivation**: Adjacent frames in navigation trajectories are highly redundant (visual change is minimal during slow translational motion); uniform sampling cannot distinguish informative frames, while LOF naturally identifies "different" frames within a continuous feature stream.

### Key Design 4: Overall Training Objective

$$\mathcal{L} = \mathcal{L}_{\text{Curr}} + \lambda_{\text{KD}} \cdot \mathcal{L}_{\text{KD}} + \lambda_{\text{FR}} \cdot \mathcal{L}_{\text{FR}}$$

where $\mathcal{L}_{\text{Curr}}$ is the behavior cloning loss on the current task (also with inflection weighting), and $\lambda_{\text{KD}} = \lambda_{\text{FR}} = 5$.

## Loss & Training

- **Behavior cloning loss**: Cross-entropy with inflection weighting ($\gamma = 3.48$), emphasizing supervision at action transition frames
- **Feature distillation loss**: $\ell_2$ distance constraining old and new encoder outputs to be consistent
- **Feature replay loss**: Replays previous-task features from the feature buffer to train the decoder
- **Optimizer**: AdamW with linear warmup over 1000 steps to $3 \times 10^{-4}$, followed by linear decay
- **Training scale**: 25 epochs per stage, batch size 32, 2× A6000 GPUs
- **Encoders**: CLIP-ResNet50 (RGB) + PointNav pretrained ResNet50 (depth), both frozen

## Key Experimental Results

### Main Results: Performance on HM3D Across Different Architectures (SR%)

| Method | RNN-Avg | RNN-Last | Trans-Avg | Trans-Last | Bev-Avg | Bev-Last | LLM-Avg | LLM-Last |
|--------|---------|----------|-----------|------------|---------|----------|---------|----------|
| Finetuning | 32.8 | 21.3 | 31.4 | 19.5 | 32.0 | 20.8 | 28.4 | 16.4 |
| LoRA | - | - | 34.0 | 22.5 | 36.3 | 24.1 | 39.9 | 24.1 |
| LwF | 34.4 | 25.1 | 31.7 | 19.2 | 32.6 | 21.7 | 26.2 | 11.7 |
| Model Merge | 40.8 | 20.4 | 45.1 | 24.5 | 45.0 | 18.5 | 42.5 | 19.9 |
| Data Replay | 44.1 | 33.6 | 52.7 | 39.6 | 53.2 | 44.2 | 52.2 | 40.9 |
| **C-Nav** | **50.0** | **40.3** | **55.8** | **46.5** | **56.3** | **46.5** | **52.2** | **42.2** |

C-Nav outperforms Data Replay by an average of **2.75% SR** across all four architectures, without storing raw trajectories.

### Ablation Study: Contribution of Dual-Path Components (HM3D, SR%)

| Ablation | RNN-Avg | RNN-Last | Trans-Avg | Trans-Last | Bev-Avg | Bev-Last | LLM-Avg | LLM-Last |
|----------|---------|----------|-----------|------------|---------|----------|---------|----------|
| w/o KD (no feature distillation) | 28.2 | 16.9 | 31.4 | 20.2 | 32.5 | 21.9 | 33.6 | 19.9 |
| w/o FP (no feature replay) | 37.9 | 27.9 | 45.9 | 32.6 | 38.9 | 26.7 | 42.7 | 30.2 |
| **All (full C-Nav)** | **50.0** | **40.3** | **55.8** | **46.5** | **56.3** | **46.5** | **52.2** | **42.2** |

Removing feature distillation reduces average SR by approximately **22%** on HM3D; removing feature replay reduces it by approximately **12%**, indicating that encoder representation drift is the primary source of forgetting.

### Adaptive Sampling Ablation (HM3D, 50% trajectory length, SR%)

| Sampling Strategy | RNN-Avg | Trans-Avg | Bev-Avg | LLM-Avg |
|-------------------|---------|-----------|---------|---------|
| Uniform (50%) | 43.2 | 50.5 | 49.4 | 47.7 |
| Data Replay (Full) | 44.1 | 52.7 | 53.2 | 52.2 |
| Adaptive (50%) | 47.3 | 53.7 | 52.8 | 51.6 |
| C-Nav Full | 50.0 | 54.7 | 56.3 | 52.2 |

Adaptive sampling using only 50% of frames outperforms uniform sampling by **3.65%** and falls only **1.9%** short of full data replay.

## Highlights & Insights

1. **Valuable problem formulation**: The paper is the first to systematically define the Continual-ObjectNav benchmark, covering 4 mainstream architectures (RNN/Transformer/BEV/LLM-based) × multiple continual learning methods, filling a research gap in continual learning for embodied navigation.
2. **Elegant dual-path decoupled design**: Forgetting is attributed to two independent sources—encoder representation drift and decoder policy degradation—addressed separately via distillation and replay, with experiments confirming their complementarity.
3. **Clever use of LOF for keyframe selection**: The problem of identifying important frames is formulated as anomaly detection; LOF automatically locates semantic change points within a continuous feature stream, outperforming uniform sampling at half the data volume.
4. **Storing features instead of raw images**: This simultaneously addresses privacy concerns (no indoor RGB stored) and dramatically compresses storage (feature vectors vs. high-resolution images), offering strong practical engineering advantages.
5. **Cross-architecture generalizability**: The method consistently performs well across RNN, Transformer, BEV, and LLM-based architectures, demonstrating the generality of the design.

## Limitations & Future Work

1. **Limited benchmark scale**: HM3D covers only 6 categories in 4 stages, and MP3D has 21 categories across 4 stages—far from a truly open world with hundreds of continuously growing categories; scalability to larger settings remains unexplored.
2. **Simulator-only evaluation**: All experiments are conducted in the Habitat simulator without sim-to-real validation; issues such as sensor noise and dynamic obstacles in real robot deployment are not addressed.
3. **Frozen encoders limit representation learning**: Both CLIP-ResNet50 and PointNav ResNet50 are frozen, with distillation applied only at the fusion layer, limiting the encoder's ability to adapt to new scenes and potentially becoming a bottleneck on more challenging tasks.
4. **LOF hyperparameter sensitivity not fully discussed**: The neighborhood size $k$ in LOF significantly affects keyframe selection, yet no sensitivity analysis is provided.
5. **Fixed replay buffer size**: Each category stores features from $p=80$ trajectories; the performance–storage trade-off under varying buffer sizes is not discussed.

## Related Work & Insights

| Method | Type | Requires Raw Trajectories | Forgetting Mitigation | Storage Cost |
|--------|------|--------------------------|----------------------|--------------|
| **Data Replay** | Data replay | ✅ Yes | Good | High (scales linearly with categories) |
| **LwF** | Regularization (logit distillation) | ❌ No | Poor (near Finetuning on HM3D) | Low |
| **C-Nav** | Feature distillation + feature replay | ❌ Feature vectors only | **Best** | Medium (feature-level, far smaller than image-level) |

- vs. **Data Replay**: C-Nav achieves +2.75% / +3.35% SR on HM3D / MP3D respectively, without storing raw images, offering clear storage and privacy advantages.
- vs. **LwF**: LwF applies KL divergence distillation on logits, but the navigation action space is small (6 actions), providing insufficient information at the logit level; C-Nav's feature-level distillation preserves richer information.
- vs. **LoRA / Model Merge**: Both approaches partially mitigate forgetting, but final-stage (Last) performance remains noticeably below C-Nav, indicating that parameter constraints or merging alone cannot substitute for explicit representation and policy consistency preservation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First Continual-ObjectNav benchmark + dual-path anti-forgetting + LOF keyframe selection; both problem formulation and method design are original, though individual components (knowledge distillation, experience replay, LOF) are not novel in themselves
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 4 architectures × 2 datasets × multiple baselines × detailed ablations; comprehensive and systematic
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, rigorous problem formulation, well-organized figures and tables; some mathematical notation conflicts (LOF neighborhood $k$ clashes with task stage index $k$)
- **Value**: ⭐⭐⭐⭐ — The benchmark itself is a significant contribution; the method is broadly applicable and offers tangible advancement for the embodied AI community

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] COOPERA: Continual Open-Ended Human-Robot Assistance](coopera_continual_open_ended_human_robot_assistance.md)
- [\[NeurIPS 2025\] EfficientNav: Towards On-Device Object-Goal Navigation with Navigation Map Caching and Retrieval](efficientnav_towards_on-device_object-goal_navigation_with_navigation_map_cachin.md)
- [\[ICCV 2025\] NavMorph: A Self-Evolving World Model for Vision-and-Language Navigation in Continuous Environments](../../ICCV2025/robotics/navmorph_a_self-evolving_world_model_for_vision-and-language_navigation_in_conti.md)
- [\[NeurIPS 2025\] Policy Compatible Skill Incremental Learning via Lazy Learning Interface](policy_compatible_skill_incremental_learning_via_lazy_learning_interface.md)
- [\[NeurIPS 2025\] DexFlyWheel: A Scalable Self-Improving Data Generation Framework for Dexterous Manipulation](dexflywheel_a_scalable_and_self-improving_data_generation_framework_for_dexterou.md)

</div>

<!-- RELATED:END -->
