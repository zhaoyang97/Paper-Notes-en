---
title: >-
  [Paper Note] ERL-VLM: Enhancing Rating-Based RL to Leverage Feedback from Large VLMs
description: >-
  [ICML2025][Multimodal VLM][RLHF] ERL-VLM is proposed to leverage Large Vision-Language Models (VLMs) to provide absolute ratings for single trajectories instead of pairwise preferences. By combining stratified sampling and MAE loss to address data imbalance and noisy labels, it significantly improves VLM feedback-driven reward learning.
tags:
  - "ICML2025"
  - "Multimodal VLM"
  - "RLHF"
  - "Rating-based RL"
  - "VLM Feedback"
  - "Reward Learning"
  - "AI Feedback"
date: 2026-05-08
content_hash: ebe8710ecda04603
---

# ERL-VLM: Enhancing Rating-Based RL to Leverage Feedback from Large VLMs

**Conference**: ICML2025  
**arXiv**: [2506.12822](https://arxiv.org/abs/2506.12822)  
**Code**: [tunglm2203/erlvlm](https://github.com/tunglm2203/erlvlm)  
**Area**: Multimodal RL / VLM Feedback-Driven Reward Learning  
**Keywords**: RLHF, Rating-based RL, VLM Feedback, Reward Learning, AI Feedback

## TL;DR

ERL-VLM is proposed to leverage Large Vision-Language Models (VLMs) to provide absolute ratings for single trajectories instead of pairwise preferences. By combining stratified sampling and MAE loss to address data imbalance and noisy labels, it significantly improves VLM feedback-driven reward learning.

## Background & Motivation

### Core Problem
Manual design of reward functions in reinforcement learning is both time-consuming and error-prone. Although RLHF learns rewards from human feedback, large-scale collection of human feedback is highly expensive and inefficient.

### Limitations of Prior Work
- **Preference-based methods**: Such as RL-VLM-F, which require the VLM to compare which of two trajectories is better. Limitations: ① Single preference feedback provides limited information, requiring a large number of queries; ② Processing two trajectories simultaneously doubles the token count, leading to high computational costs; ③ Trajectories with similar quality can cause the VLM to "hallucinate" incorrect preferences.
- **Similarity-based scoring methods**: Such as CLIP Score, which computes the cosine similarity of image-text embeddings as the reward. Limitations: The signal is noisy, and it heavily relies on the quality of task descriptions and alignment with the pre-trained data distribution.
- None of the above methods fully utilize the reasoning capabilities of large VLMs.

### Design Motivation
Utilizing VLMs (such as Gemini) to provide absolute Likert-scale ratings (**very bad** $\rightarrow$ **very good**) for a **single trajectory** is more expressive, token-efficient, and ensures that all queried samples can be used for training compared to pairwise comparisons.

## Method

### Overall Architecture

ERL-VLM alternates among three phases:

1. **Data Collection**: The agent interacts with the environment using policy $\pi_\theta$, storing transitions $(s_t, I_t, a_t, s_{t+1}, I_{t+1}, \hat{r}_t)$ into a replay buffer $\mathcal{B}$.
2. **VLM Rating & Reward Learning**: Every $K$ steps, $N$ segments are sampled from $\mathcal{B}$ and sent to the VLM to obtain rating labels $\tilde{y}$, which are then used to update the reward model $\hat{r}_\psi$.
3. **Policy Learning**: Re-label the entire replay buffer using the updated $\hat{r}_\psi$, and then train the policy using SAC/IQL.

### Rating Generation

- Given a trajectory segment $\sigma = \{(s_1, a_1), \ldots, (s_H, a_H)\}$, the VLM outputs a discrete rating $\tilde{y} \in \mathcal{C} = \{0, 1, \ldots, n-1\}$.
- A two-stage prompt is employed: first analyzing the agent's behavior, and then outputting the rating based on the analysis.
- Multi-frame image sequences and action sequences are used for complex tasks to provide richer context.

### Rating-Based Reward Learning

Given a segment $\sigma$, the normalized return predicted by the reward model is $\tilde{R}(\sigma) = \sum_{t=1}^k \hat{r}_\psi(s_t, a_t)$ (min-max normalized into $[0,1]$). The probability of it belonging to the $i$-th class rating is defined as:

$$P_\sigma(i) = \frac{\exp\bigl(-(\tilde{R}(\sigma) - \bar{R}_i)(\tilde{R}(\sigma) - \bar{R}_{i+1})\bigr)}{\sum_{j=0}^{n-1} \exp\bigl(-(\tilde{R}(\sigma) - \bar{R}_j)(\tilde{R}(\sigma) - \bar{R}_{j+1})\bigr)}$$

where $\bar{R}_i$ represents the rating class boundaries, satisfying $0 = \bar{R}_0 \le \bar{R}_1 \le \cdots \le \bar{R}_n = 1$.

### Key Designs

**Challenge 1: Imbalanced Data Classes**  
In the early stages of training, "bad" ratings dominate, which causes the reward model to degenerate into always predicting the dominant class.

- **Solution**: Employ **stratified sampling** to ensure that each minibatch contains samples from all rating classes; concurrently, incorporate a **weighted loss** that scales with class frequency.

**Challenge 2: Noisy Labels from VLM Hallucinations**  

- **Solution**: Use **MAE loss** instead of cross-entropy loss, as MAE offers theoretical robustness guarantees against label noise:

$$\mathcal{L}_{MAE}(\psi, \mathcal{D}) = \mathbb{E}_{(\sigma, \tilde{y}) \sim \mathcal{U}_S(\mathcal{D})} \left[\sum_{i=0}^{n-1} |\mu_\sigma(i) - P_\sigma(i)|\right]$$

where $\mu_\sigma(i)$ is the indicator function (equals 1 if $\tilde{y}=i$, and 0 otherwise), and $\mathcal{U}_S$ denotes the stratified sampling strategy.

- Label smoothing performs poorly in multi-class rating scenarios (verified empirically), because estimating the ratio of noisy labels generated by the VLM is extremely difficult.

## Key Experimental Results

### Main Results

#### MetaWorld Low-Level Control Tasks (3 Tasks)

| Method | Sweep Into | Drawer Open | Soccer |
|------|-----------|------------|--------|
| CLIP Score | Large fluctuations, unstable | Large fluctuations | Large fluctuations |
| RoboCLIP | Unstable | Moderate | Unstable |
| RL-VLM-F | Poor | Comparable to ERL-VLM | Poor |
| **ERL-VLM** | **Best** | **Best (tied)** | **Best** |

#### ALFRED High-Level Vision-Language Navigation Tasks (20 Tasks, 4 Categories)

| Task Category | RL-VLM-F | Sparse Reward | **ERL-VLM** |
|---------|----------|--------------|-------------|
| PickupObject | Near failure | Moderate | **Outperforms sparse env reward** |
| PutObject | Near failure | Moderate | **Outperforms sparse env reward** |
| CoolObject | Near failure | Low | **Best** |
| CleanObject | Near failure | Low | **Best** |

Key Findings: ERL-VLM **outperforms the baseline sparse environmental rewards** on PickupObject and PutObject, indicating that rating feedback not only provides task completion signals but also introduces beneficial reward shaping for critical states.

#### Real-Robot Experiments (Sawyer 7-DOF)

| Method | Sweep Bowl | Drawer Open | Pickup Banana |
|------|-----------|------------|--------------|
| BC | 0.50±0.10 | 0.23±0.06 | 0.17±0.06 |
| Sparse Rewards | 0.57±0.06 | 0.37±0.06 | 0.30±0.10 |
| **ERL-VLM** | **0.73±0.06** | **0.60±0.10** | **0.47±0.12** |

### Ablation Study

- **Vanilla RbRL (Original Framework)** performs the worst.
- **MAE Loss Only** $\rightarrow$ yields the most significant improvement (robustness to noisy labels).
- **Stratified Sampling Only** $\rightarrow$ shows obvious improvement on Sweep Into and Drawer Open.
- **Full ERL-VLM** $\rightarrow$ achieves the best performance.
- **Number of Rating Categories $n$**: $n=4$ suffers from performance degradation (due to increased VLM ambiguity); $n=2$ or $n=3$ performs best depending on the task (binary judgment vs. degree assessment).

## Highlights & Insights

1. **Absolute Ratings vs. Pairwise Preferences**: Rating contains higher information density than preference—learning better reward functions under the same query budget while cutting token costs by approximately half.
2. **Stratified Sampling + MAE Loss**: Simple and effective solution to the two core bottlenecks in VLM feedback scenarios (class imbalance and noisy labels) without requiring complex denoising or filtering mechanisms.
3. **Outperforming Environmental Sparse Rewards**: On certain ALFRED tasks, ERL-VLM even outperforms manually designed sparse rewards, demonstrating that VLM ratings can provide implicit reward shaping.
4. **Real-Robot Transfer**: An effective policy can be trained with only 50 demonstrations and offline ratings, proving the practical deployment potential of the method.
5. **Importance of Prompt Design**: The two-stage prompt (first analyze, then rate) significantly improves the quality of VLM feedback.

## Limitations & Future Work

1. **VLM Dependency**: Relies heavily on commercial large VLMs like Gemini-1.5-Pro, which entails high inference costs and API instability risks.
2. **Sensitivity to Number of Rating Classes**: The choice of $n$ requires task-specific tuning, and an adaptive selection mechanism is currently lacking.
3. **VLM Bias Propagation**: Inherent biases from large foundation models may propagate to the RL agent, raising safety-critical concerns.
4. **Visual-Only Observations**: Currently, only image and action descriptions are utilized, without exploiting other modalities such as tactile or force feedback.
5. **Prompt Engineering Overhead**: Different environments (MetaWorld / ALFRED / real robot) require tailored prompt templates, leaving generality to be verified.
6. **Limited Validation in Offline Scenarios**: Real-robot experiments only validate offline training with 50 demonstrations, leaving online continual learning scenarios insufficiently explored.

## Related Work & Insights

- **Rating-based RL** (White et al., 2024): This serves as the theoretical foundation of this work. ERL-VLM introduces critical enhancements tailored to the characteristics of VLM feedback.
- **RL-VLM-F** (Wang et al., 2024): The closest baseline, which uses VLM preferences rather than ratings, but suffers from low query efficiency.
- **CLIP/RoboCLIP Score**: Typical methods that use embedding space similarity as rewards, which are highly noisy and unstable.
- **Insights**: Combining **AI feedback with evaluative feedback** is a highly promising direction that can be extended to the alignment of multimodal LLM agents.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Combines rating-based RL with VLM feedback. The enhancements of stratified sampling and MAE loss are simple yet highly effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers low-level control, high-level navigation, and real-world robotics with comprehensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear, fluent, and highly informative tables/graphs.
- **Value**: ⭐⭐⭐⭐ — Provides a solid baseline and practical enhancements for VLM-driven RL reward learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] GTR: Guided Thought Reinforcement Prevents Thought Collapse in RL-Based VLM Agent](../../ICCV2025/multimodal_vlm/gtr_guided_thought_reinforcement_prevents_thought_collapse_in_rl-based_vlm_agent.md)
- [\[ACL 2025\] Improving Medical Large Vision-Language Models with Abnormal-Aware Feedback](../../ACL2025/multimodal_vlm/improving_medical_large_vision-language_models_with_abnormal-aware_feedback.md)
- [\[ICML 2025\] From Black Boxes to Transparent Minds: Evaluating and Enhancing the Theory of Mind in Multimodal Large Language Models](from_black_boxes_to_transparent_minds_evaluating_and_enhancing_the_theory_of_min.md)
- [\[ICML 2026\] Benchmarking and Enhancing VLM for Compressed Image Understanding](../../ICML2026/multimodal_vlm/benchmarking_and_enhancing_vlm_for_compressed_image_understanding.md)
- [\[ICLR 2026\] Enhancing Geometric Perception in VLMs via Translator-Guided Reinforcement Learning](../../ICLR2026/multimodal_vlm/enhancing_geometric_perception_in_vlms_via_translator-guided_reinforcement_learn.md)

</div>

<!-- RELATED:END -->
