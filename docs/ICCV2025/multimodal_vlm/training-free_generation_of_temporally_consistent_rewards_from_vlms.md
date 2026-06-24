---
title: >-
  [Paper Note] Training-free Generation of Temporally Consistent Rewards from VLMs
description: >-
  [ICCV 2025][Multimodal VLM][Vision-Language Models] T²-VLM proposes a training-free, temporally consistent reward generation framework. By querying the VLM only once at the beginning of each episode to generate spatially aware subgoals, and subsequently tracking the completion status of these subgoals using a Bayesian particle filter, it generates structured RL rewards. This approach achieves state-of-the-art performance on robotic manipulation benchmarks with significantly r…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "Vision-Language Models"
  - "Reward Generation"
  - "Reinforcement Learning"
  - "Robotic Manipulation"
  - "Bayesian Tracking"
date: 2026-05-08
content_hash: 505814483fee7284
---

# Training-free Generation of Temporally Consistent Rewards from VLMs

**Conference**: ICCV 2025  
**arXiv**: [2507.04789](https://arxiv.org/abs/2507.04789)  
**Code**: [https://github.com/nuomizai/T2VLM](https://github.com/nuomizai/T2VLM)  
**Area**: Multimodal VLM  
**Keywords**: Vision-Language Models, Reward Generation, Reinforcement Learning, Robotic Manipulation, Bayesian Tracking

## TL;DR

T²-VLM proposes a training-free, temporally consistent reward generation framework. By querying the VLM only once at the beginning of each episode to generate spatially aware subgoals, and subsequently tracking the completion status of these subgoals using a Bayesian particle filter, it generates structured RL rewards. This approach achieves state-of-the-art performance on robotic manipulation benchmarks with significantly reduced computational cost.

## Background & Motivation

**Background**: Utilizing Vision-Language Models (VLMs) for embodied AI tasks has been a research hotspot in recent years. Due to their strong capabilities in goal decomposition and visual understanding, VLMs have naturally become candidates for designing reward functions in robotic manipulation tasks. Existing methods, such as VLM-RM and CLIPScore, attempt to directly use the VLM output as RL reward signals.

**Limitations of Prior Work**: Directly using VLMs to score every frame as a reward poses three key problems: (1) **Lack of robotic domain knowledge in pre-training data**—VLMs are trained on internet data and have limited understanding of robotic manipulation scenarios, leading to inaccurate rewards; (2) **Prohibitively high computational cost of frame-by-frame queries**—large VLMs suffer from slow inference speed and poor real-time performance; (3) **Lack of temporal consistency in frame-wise rewards**—VLMs evaluate each frame independently and may produce contradictory rewards for adjacent frames (e.g., estimating 50% completion in the previous frame but suddenly 30% in the next), resulting in unstable RL training.

**Key Challenge**: A fundamental contradiction exists between the VLM's strong semantic understanding and its lack of precise perception in the robotics domain, as well as between the computational cost of frame-by-frame querying and the real-time requirements of RL training.

**Goal**: To design a method that provides accurate, temporally consistent reward signals for an entire episode using only a single VLM query, without requiring fine-tuning of the VLM, while maintaining extremely low computational overhead.

**Key Insight**: The authors observe that while VLMs cannot precisely evaluate the completion degree of each frame, they excel at high-level goal decomposition—breaking down complex tasks into multiple subgoals. If the VLM can provide a list of subgoals and initial completion estimates just once at the start of an episode, and a lightweight tracking algorithm can then continuously monitor state changes of these subgoals, temporally consistent rewards can be generated at a very low cost.

**Core Idea**: Decompose the reward generation problem into two steps—(1) a one-time high-level goal decomposition and initialization by the VLM, and (2) tracking subgoal state changes and generating continuous rewards using a Bayesian particle filter. This leverages the VLM's semantic understanding for "planning" and classic state estimation algorithms for "execution."

## Method

### Overall Architecture

The input to T²-VLM consists of a sequence of visual observations and a task description from the robotic manipulation environment, and the output is a scalar reward at each timestep. The overall pipeline is divided into two phases: **Initialization Phase**—at the start of the episode, the VLM analyzes the initial scene image and task description to generate a list of spatially aware subgoals along with their initial completion estimates; **Tracking Phase**—in subsequent timesteps, trackers such as SAM2 track the objects involved in the subgoals, while a Bayesian particle filter updates the completion status of each subgoal, mapping state changes to reward signals.

### Key Designs

1. **VLM Spatially Aware Subgoal Generation**:

    - **Function**: Decompose complex manipulation tasks into trackable subgoals.
    - **Mechanism**: Send the initial frame image and task description (e.g., "put the red block on the blue plate") to the VLM (e.g., GPT-4V), guiding the VLM via carefully designed prompts to output: (a) a list of subgoals (e.g., "pick up red block", "move above blue plate", "place block down"); (b) the key objects and spatial relations involved in each subgoal; (c) the initial completion estimates (values between 0-1) for each subgoal in the current scene. The key is that the prompt requires the VLM to provide spatialized descriptions (including object positions and spatial relations) rather than purely semantic descriptions.
    - **Design Motivation**: VLMs excel at semantic understanding and goal decomposition, but perform poorly at precise spatial perception and temporal reasoning. Therefore, the VLM is tasked only with what it does best—one-time high-level planning—leaving precise state tracking to dedicated algorithms.

2. **SAM2 Object State Tracking**:

    - **Function**: Continuously monitor spatial state changes of objects involved in the subgoals.
    - **Mechanism**: Based on the subgoal descriptions provided by the VLM, the objects that need to be tracked are automatically identified. SAM2 (Segment Anything Model 2) is then utilized to track the position, size, and relative spatial relations of these objects in subsequent frames. Subgoal hidden state vectors—which encode spatial relations between objects (such as distance and contact states)—are extracted from the tracking results. These hidden state vectors serve as observations for the Bayesian filter.
    - **Design Motivation**: SAM2 is a zero-shot visual tracker that can track almost any object without training, which aligns with T²-VLM's training-free design philosophy. Measuring subgoal completion progress by tracking changes in object spatial relations is more accurate and faster than direct evaluation using a VLM.

3. **Bayesian Particle Filter Reward Generation**:

    - **Function**: Translate object state changes into temporally consistent reward signals.
    - **Mechanism**: Use a particle filter to maintain the completion estimates of each subgoal. The initial completion degree provided by the VLM is used to initialize the particle distribution. At each timestep, particle weights are updated based on the hidden states tracked by SAM2—if the spatial relations of objects evolve towards subgoal completion (e.g., the distance between objects decreases), the weight of particles representing high completion increases. The final reward is given by $r_t = \Delta s_t = s_t - s_{t-1}$, which is the increment of completion degree at the current step. The smoothing characteristic of the particle filter naturally guarantees the temporal consistency of the rewards.
    - **Design Motivation**: The particle filter is a classic Bayesian state estimation method with inherent temporal smoothness, preventing sudden jumps in rewards caused by single-frame observation noise. This addresses the core issue of reward inconsistency associated with frame-by-frame VLM evaluation.

### Loss & Training

T²-VLM itself does not involve any training. The generated rewards are directly used to train an RL agent (such as SAC), and the RL loss details follow the standard actor-critic loss. The only "learning" in the entire system occurs on the RL agent side.

## Key Experimental Results

### Main Results

Contrast of task success rates on two robotic manipulation benchmarks (MetaWorld and RLBench):

| Method | MetaWorld Avg. Success Rate↑ | RLBench Avg. Success Rate↑ | Reward Calculation Time per Frame↓ | Needs Training |
|------|---------------------|-------------------|-----------------|------------|
| Sparse Reward | 12.3% | 8.7% | - | No |
| VLM-RM | 45.6% | 34.2% | 2.1s | Fine-tuning required |
| VLM-Score | 52.3% | 38.8% | 1.8s | No |
| LIV | 48.9% | 36.5% | 0.9s | Training required |
| T²-VLM | **61.7%** | **47.3%** | **0.05s** | **No** |

Performance comparison under different VLM backbones:

| VLM Backbone | MetaWorld Success Rate↑ | Reward Accuracy↑ |
|---------|-----------------|-----------|
| GPT-4V | **61.7%** | **0.82** |
| LLaVA-1.5 | 55.2% | 0.73 |
| InternVL | 57.8% | 0.76 |

### Ablation Study

| Configuration | MetaWorld Success Rate↑ | Description |
|------|-----------------|------|
| Full T²-VLM | **61.7%** | Full model |
| w/o Bayesian tracking (frame-by-frame VLM query) | 52.3% | Degrades to VLM-Score |
| w/o VLM initialization (random particle initialization) | 48.5% | VLM's prior estimation is crucial |
| w/o SAM2 tracking (using simple template matching) | 53.1% | SAM2's precise tracking contributes significantly |
| w/o Spatially aware prompt | 55.4% | Spatial information helps subgoal decomposition |

### Key Findings

- The Bayesian particle filter is the core contribution; removing it drops the success rate from 61.7% to 52.3% (degrading to frame-wise VLM evaluation), indicating that temporal consistency is crucial for RL training.
- Although the initial estimation from the VLM does not need to be precise for every frame, it provides critical prior distribution information for the particle filter, and the performance drops significantly without it.
- Computational efficiency increases by approximately 36x (0.05s vs. 1.8s) because the VLM is queried only once at the beginning of each episode, with all subsequent steps relying on lightweight tracking.
- The advantage is more pronounced in long-horizon tasks (tasks requiring 5+ subgoals), as temporal consistency is more critical for long tasks.
- Robustness across different VLM backbones—decent results are achieved even when using a weaker LLaVA, indicating the method does not overly depend on the VLM's raw power.

## Highlights & Insights

- **Divide-and-Conquer Design Philosophy**: It elegantly combines the semantic planning capabilities of VLMs with the temporal reasoning capabilities of classic state estimation. The VLM performs one-time high-level planning, while the particle filter handles continuous low-level state estimation, playing to their respective strengths. This "one-time planning + continuous tracking" paradigm can be transferred to other real-time systems that require VLM involvement.
- **Importance of Temporal Consistency**: The experiments clearly demonstrate the importance of temporal consistency in rewards for RL training. This is an overlooked but critical issue—noise and inconsistency in reward signals can severely disrupt RL credit assignment.
- **Practical Value of Computational Efficiency**: Reducing VLM calls from once-per-frame to once-per-episode makes it feasible to employ VLM-based rewards in real-world robotic systems.

## Limitations & Future Work

- Dependency on the VLM's ability to correctly decompose subgoals—if the VLM misunderstands the task, the entire system will suffer from cascading failures.
- The state space of the particle filter is manually designed (e.g., distances between objects), which might not cover completion conditions for all types of subgoals.
- Currently only validated in simulated environments; in the real world, the tracking robustness of SAM2 and scene understanding of the VLM may be insufficient.
- The VLM query at the start of each episode still introduces a latency of a few seconds, which might be inadequate for scenarios demanding rapid startup.
- Mutual dependencies among subgoals (such as sequential constraints) are not explicitly modeled.

## Related Work & Insights

- **vs VLM-RM (Rocamonde et al., 2024)**: VLM-RM requires fine-tuning the VLM to provide rewards, leading to high training costs and low generalization. T²-VLM is completely training-free and ensures temporal consistency via a particle filter.
- **vs VLM-Score**: VLM-Score queries the VLM frame-by-frame, resulting in high computational costs and inconsistent rewards. T²-VLM queries the VLM only once, relying on tracking algorithms to provide consistent rewards.
- **vs Eureka (Ma et al., 2024)**: Eureka uses LLMs to generate reward code but requires structured environmental state information. T²-VLM operates directly from visual observations, making it more suitable for visual RL scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of combining VLM planning with Bayesian tracking is novel, but the individual components are combinations of existing technologies.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two benchmarks, multiple baselines, various VLM backbones, and detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear method description and well-elaborated motivation.
- Value: ⭐⭐⭐⭐ Significantly reduces the computational cost of VLM reward generation, offering practical value for the embodied AI field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Is Less More? Exploring Token Condensation as Training-free Test-time Adaptation](is_less_more_exploring_token_condensation_as_training-free_test-time_adaptation.md)
- [\[ICCV 2025\] Exploiting Vision Language Model for Training-Free 3D Point Cloud OOD Detection](exploiting_vision_language_model_for_training-free_3d_point_cloud_ood_detection_.md)
- [\[NeurIPS 2025\] Training-free Online Video Step Grounding](../../NeurIPS2025/multimodal_vlm/training-free_online_video_step_grounding.md)
- [\[ICCV 2025\] BabyVLM: Data-Efficient Pretraining of VLMs Inspired by Infant Learning](babyvlm_data-efficient_pretraining_of_vlms_inspired_by_infant_learning.md)
- [\[ICML 2026\] FreeRet: MLLMs as Training-Free Retrievers](../../ICML2026/multimodal_vlm/freeret_mllms_as_training-free_retrievers.md)

</div>

<!-- RELATED:END -->
