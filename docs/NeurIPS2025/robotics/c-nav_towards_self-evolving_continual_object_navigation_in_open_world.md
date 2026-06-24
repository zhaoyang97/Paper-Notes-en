---
title: >-
  [Paper Note] C-NAV: Towards Self-Evolving Continual Object Navigation in Open World
description: >-
  [NeurIPS 2025][Robotics][continual learning] The C-Nav framework is proposed, which avoids catastrophic forgetting for navigation agents when continuously learning new object categories through **dual-pathway anti-forgetting** (feature distillation + feature replay) and **adaptive experience selection** (LOF anomaly detection for keyframe selection), outperforming full data replay baselines across four distinct architectures.
tags:
  - "NeurIPS 2025"
  - "Robotics"
  - "continual learning"
  - "object navigation"
  - "catastrophic forgetting"
  - "feature distillation"
  - "feature replay"
  - "LOF"
date: 2026-05-08
content_hash: 41778a5c92171648
---

# C-NAV: Towards Self-Evolving Continual Object Navigation in Open World

**Conference**: NeurIPS 2025  
**arXiv**: [2510.20685](https://arxiv.org/abs/2510.20685)  
**Code**: [https://bigtree765.github.io/C-Nav-project](https://bigtree765.github.io/C-Nav-project)  
**Area**: Robotics  
**Keywords**: continual learning, object navigation, catastrophic forgetting, feature distillation, feature replay, LOF

## TL;DR

The C-Nav framework is proposed, which avoids catastrophic forgetting for navigation agents when continuously learning new object categories through **dual-pathway anti-forgetting** (feature distillation + feature replay) and **adaptive experience selection** (LOF anomaly detection for keyframe selection), outperforming full data replay baselines across four distinct architectures.

## Background & Motivation

**Background**: Object Navigation (ObjectNav) is a core task in embodied AI. Current SOTA methods (e.g., OVRL-V2, PIRLNav, NavID) rely on pre-trained visual encoders and large-scale demonstration trajectories, demonstrating excellent performance on fixed category sets.

**Limitations of Prior Work**: These methods assume that all categories and data are available at once during training. When continuously incorporating new objects in an open-world setting, model parameter updates lead to **catastrophic forgetting**—while new categories are learned, the navigation ability for old categories drops drastically (by approximately 40% Success Rate (SR)).

**Key Challenge**: Direct data replay (storing entire trajectories) can mitigate forgetting, but navigation trajectories are incredibly long (up to hundreds of frames per trajectory), highly redundant, and raise privacy concerns (leakage of spatial information of indoor scenes), making storage and privacy costs unacceptable.

**Goal**: To enable the navigation agent to **incrementally learn new categories** while retaining navigation skills for old categories, **without the need to store raw trajectories**.

**Key Insight**: Decompose forgetting into two independent sources—**representation drift** of the encoder and **policy degradation** of the decoder—and constrain them separately. Concurrently, formulate keyframe selection as an outlier detection problem in the feature space to compress storage.

**Core Idea**: Dual-pathway anti-forgetting (feature distillation to stabilize encoder representations + feature replay to stabilize action decoder policies) coupled with LOF-based adaptive experience selection (storing only features of semantic mutation frames instead of raw images).

## Method

### Overall Architecture

C-Nav consists of two main modules: (1) **Dual-pathway anti-forgetting mechanism**—the feature distillation pathway constrains the output consistency of the multimodal encoder, while the feature replay pathway replays keyframe features of previous tasks to the action decoder; (2) **Adaptive experience selection**—visual features are extracted using CLIP, and the Local Outlier Factor (LOF) algorithm is applied to detect semantic outlier frames as keyframes, storing only their deep features and action labels.

### Key Designs

#### Key Design 1: Feature Distillation for Representation Consistency

- **Function**: Freeze the encoder of the previous stage $f_{k-1}$, and constrain the output of the current encoder $f_k$ to remain close to $f_{k-1}$ during training on new data.
- **Mechanism**: Minimize the $\ell_2$ distance between the new and old encoders on the same observation: $\mathcal{L}_{\text{KD}} = \sum_{t=1}^{L} \|f_{k-1}(o_t) - f_k(o_t)\|_2^2$.
- **Design Motivation**: Multimodal encoders process RGB, depth, pose, and text inputs. Distribution shifts lead to feature space drift; consequently, downstream decoders fail due to altered input features even if they remain unchanged.

#### Key Design 2: Feature Replay for Policy Consistency

- **Function**: Store the encoded features $h_t \in \mathbb{R}^d$ of keyframes from previous tasks along with their corresponding action labels, mixing them with the current data to train the action decoder.
- **Mechanism**: Employ a cross-entropy loss with inflection weighting: $\mathcal{L}_{\text{FR}} = \frac{1}{L}\sum_{t=1}^{L} -w_t \log \pi_k(a_t | h_{1:t})$, where action transition points are heavily weighted ($w_t = 1 + \gamma \cdot \mathbb{1}_{a_t \neq a_{t-1}}$).
- **Design Motivation**: Storing features instead of raw images prevents privacy leaks and substantially compresses storage; inflection weighting emphasizes critical decision points such as turning or stopping.

#### Key Design 3: Adaptive Experience Selection via LOF

- **Function**: Automatically filter keyframes with significant semantic changes from each trajectory rather than uniformly sampling or storing all frames.
- **Mechanism**: Encode RGB observations using CLIP to obtain features $\mathbf{v}_i$, calculate the Local Outlier Factor (LOF) for each frame, and select frames with $\text{LOF}(\mathbf{v}_i) > 1$ as keyframes. A high LOF indicates that the frame deviates from its neighborhood density in the feature space—typically corresponding to semantic mutations such as entering a new room, discovering a target object, or change in trajectory direction.
- **Design Motivation**: Adjacent frames in a navigation trajectory are highly redundant (visual changes are minimal during slow robot translation). Uniform sampling cannot distinguish information density, whereas LOF is naturally suited for identifying "differing" frames within continuous feature streams.

#### Key Design 4: Overall Training Objective

$$\mathcal{L} = \mathcal{L}_{\text{Curr}} + \lambda_{\text{KD}} \cdot \mathcal{L}_{\text{KD}} + \lambda_{\text{FR}} \cdot \mathcal{L}_{\text{FR}}$$

where $\mathcal{L}_{\text{Curr}}$ is the behavior cloning loss of the current task (also with inflection weighting), and $\lambda_{\text{KD}} = \lambda_{\text{FR}} = 5$.

## Loss & Training

- **Behavior Cloning Loss**: Cross-entropy with inflection weighting ($\gamma = 3.48$) to focus supervision on action transition frames.
- **Feature Distillation Loss**: $\ell_2$ distance constraint to maintain consistency between the old and new outputs of the encoder.
- **Feature Replay Loss**: Replays features of old tasks from the feature buffer to train the decoder.
- **Optimizer**: AdamW, linear warmup for 1000 steps to $3 \times 10^{-4}$, followed by linear decay.
- **Training Scale**: 25 epochs per stage, batch size of 32, on 2×A6000 GPUs.
- **Encoders**: CLIP-ResNet50 (RGB) + PointNav pre-trained ResNet50 (Depth), both frozen.

## Key Experimental Results

### Main Results: Performance of Different Architectures on the HM3D Dataset (SR%)

| Method | RNN-Avg | RNN-Last | Trans-Avg | Trans-Last | Bev-Avg | Bev-Last | LLM-Avg | LLM-Last |
|------|---------|----------|-----------|------------|---------|----------|---------|----------|
| Finetuning | 32.8 | 21.3 | 31.4 | 19.5 | 32.0 | 20.8 | 28.4 | 16.4 |
| LoRA | - | - | 34.0 | 22.5 | 36.3 | 24.1 | 39.9 | 24.1 |
| LwF | 34.4 | 25.1 | 31.7 | 19.2 | 32.6 | 21.7 | 26.2 | 11.7 |
| Model Merge | 40.8 | 20.4 | 45.1 | 24.5 | 45.0 | 18.5 | 42.5 | 19.9 |
| Data Replay | 44.1 | 33.6 | 52.7 | 39.6 | 53.2 | 44.2 | 52.2 | 40.9 |
| **C-Nav** | **50.0** | **40.3** | **55.8** | **46.5** | **56.3** | **46.5** | **52.2** | **42.2** |

C-Nav outperforms Data Replay by **2.75%** in average SR across the four architectures without requiring the storage of raw trajectories.

### Ablation Study: Contribution of Dual-pathway Components (HM3D, SR%)

| Ablation Setting | RNN-Avg | RNN-Last | Trans-Avg | Trans-Last | Bev-Avg | Bev-Last | LLM-Avg | LLM-Last |
|----------|---------|----------|-----------|------------|---------|----------|---------|----------|
| w/o KD (w/o feature distillation) | 28.2 | 16.9 | 31.4 | 20.2 | 32.5 | 21.9 | 33.6 | 19.9 |
| w/o FP (w/o feature replay) | 37.9 | 27.9 | 45.9 | 32.6 | 38.9 | 26.7 | 42.7 | 30.2 |
| **All (Full C-Nav)** | **50.0** | **40.3** | **55.8** | **46.5** | **56.3** | **46.5** | **52.2** | **42.2** |

Removing feature distillation leads to an average SR decrease of approximately **22%** (HM3D), while removing feature replay results in a **12%** drop, indicating that representation drift of the encoder is the primary source of forgetting.

### Adaptive Sampling Ablation (HM3D, 50% Length, SR%)

| Sampling Method | RNN-Avg | Trans-Avg | Bev-Avg | LLM-Avg |
|----------|---------|-----------|---------|---------|
| Uniform (50%) | 43.2 | 50.5 | 49.4 | 47.7 |
| Data Replay (Full) | 44.1 | 52.7 | 53.2 | 52.2 |
| Adaptive (50%) | 47.3 | 53.7 | 52.8 | 51.6 |
| C-Nav Full | 50.0 | 54.7 | 56.3 | 52.2 |

Adaptive sampling surpasses uniform sampling by **3.65%** while only utilizing 50% of the frames, and performs only **1.9%** below full replay.

## Highlights & Insights

1. **Valuable Problem Definition**: A Continual-ObjectNav benchmark is systematically defined for the first time, covering 4 mainstream architectures (RNN/Transformer/BEV/LLM-based) $\times$ multiple continual learning methods, filling the research gap of continual learning in embodied navigation.
2. **Elegant Dual-pathway Decoupled Design**: Forgetting is attributed to two independent sources—representation drift in the encoder and policy degradation in the decoder—which are resolved through distillation and replay, respectively. This logical formulation is validated by experiments demonstrating their complementarity.
3. **Ingenious Keyframe Selection via LOF**: The task of identifying "which frames are important" is transformed into an anomaly detection problem. LOF is leveraged to automatically pinpoint semantic mutations in continuous feature streams, outperforming uniform sampling with only half the volume of data.
4. **Storage of Features over Raw Images**: This design both addresses privacy concerns (by avoiding storing indoor RGB images) and substantially compresses storage (feature vectors vs. high-resolution images), rendering it highly practical in engineering contexts.
5. **Cross-architecture Generalizability**: The proposed method is consistently effective across RNN, Transformer, BEV, and LLM architectures, proving the general applicability of the design.

## Limitations & Future Work

1. **Limited Benchmark Scale**: HM3D only contains 6 categories across 4 stages, and MP3D contains 21 categories but still only has 4 stages. This significantly lags behind a real-world open world (hundreds of categories, continuously growing), leaving it unknown whether the method scales up.
2. **Simulator-Only Environments**: All experiments are conducted within the Habitat simulator, lacking sim-to-real validation. Challenges in real robot deployment, such as sensor noise and dynamic obstacles, are unaddressed.
3. **Frozen Encoders Restrict Representation Learning**: Both CLIP-ResNet50 and PointNav ResNet50 are frozen, and distillation only constrains fusion layers. This limits the encoder's adaptation to new scenes, which may pose a bottleneck in more challenging tasks.
4. **Underdiscussed Sensitivity to LOF Hyperparameters**: The neighborhood size $k$ of LOF heavily impacts keyframe selection, yet a sensitivity analysis is absent in the paper.
5. **Fixed Replay Buffer Size**: The method stores $p=80$ trajectory features per category, without discussing the impact of different buffer sizes on the performance-storage trade-off.

## Related Work & Insights

| Method | Type | Requires Raw Trajectories | Forgetting Mitigation | Storage Overhead |
|------|------|----------------|-------------|---------|
| **Data Replay** | Data Replay | ✅ Yes (raw trajectories) | Good | High (scales linearly with categories) |
| **LwF** | Regularization (logit distillation) | ❌ No | Poor (close to Finetuning on HM3D) | Low |
| **C-Nav** | Feature Distillation + Feature Replay | ❌ No (Only features) | **Best** | Medium (feature-level, far smaller than image-level) |

- vs. **Data Replay**: C-Nav achieves 2.75%/3.35% higher SR on HM3D/MP3D, respectively, without storing raw images, showing clear storage and privacy advantages.
- vs. **LwF**: LwF performs KL-divergence distillation on logits, but the action space of navigation tasks is small (6 actions) making the logit layer insufficiently informative. C-Nav performs distillation at the feature level, preserving information more adequately.
- vs. **LoRA / Model Merge**: Both mitigate forgetting to some extent, but their performance at the final stage (Last) is significantly lower than C-Nav, demonstrating that parameter constraints/merging cannot substitute explicit representation and policy consistency maintenance.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First Continual-ObjectNav benchmark + dual-pathway anti-forgetting + LOF-based keyframe selection. The problem definition and method design are novel, though the individual components (knowledge distillation, experience replay, LOF) themselves are not entirely new.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 4 architectures $\times$ 2 datasets $\times$ multiple baselines $\times$ detailed ablations, overall highly comprehensive and systematic.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured, rigorous problem definitions, and standardized figures/tables; however, there is some overlap in mathematical notation (conflict between the neighborhood size $k$ in LOF and the task stage index $k$).
- **Value**: ⭐⭐⭐⭐ — The benchmark itself is a major contribution, and the method is highly generalizable, providing tangible value to move the embodied AI community forward.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] NavMorph: A Self-Evolving World Model for Vision-and-Language Navigation in Continuous Environments](../../ICCV2025/robotics/navmorph_a_self-evolving_world_model_for_vision-and-language_navigation_in_conti.md)
- [\[NeurIPS 2025\] COOPERA: Continual Open-Ended Human-Robot Assistance](coopera_continual_open_ended_human_robot_assistance.md)
- [\[NeurIPS 2025\] EfficientNav: Towards On-Device Object-Goal Navigation with Navigation Map Caching and Retrieval](efficientnav_towards_on-device_object-goal_navigation_with_navigation_map_cachin.md)
- [\[NeurIPS 2025\] Learning Interactive World Model for Object-Centric Reinforcement Learning](learning_interactive_world_model_for_object-centric_reinforcement_learning.md)
- [\[CVPR 2026\] Memory-Augmented Scene Understanding and Exploration for Open-World Aerial Object-Goal Navigation](../../CVPR2026/robotics/memory-augmented_scene_understanding_and_exploration_for_open-world_aerial_objec.md)

</div>

<!-- RELATED:END -->
