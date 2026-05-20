---
title: >-
  [Paper Note] GoalLadder: Incremental Goal Discovery with Vision-Language Models
description: >-
  [NeurIPS 2025][Multimodal VLM][Vision-Language Models] This paper proposes GoalLadder, a framework that leverages VLMs to incrementally discover and rank candidate goal states…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "Vision-Language Models"
  - "Reinforcement Learning"
  - "Goal Discovery"
  - "ELO Rating"
  - "Reward Function"
date: 2026-05-08
content_hash: 7fb990a5ee0f8266
---

# GoalLadder: Incremental Goal Discovery with Vision-Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2506.16396](https://arxiv.org/abs/2506.16396)  
**Code**: None  
**Area**: Multimodal VLM
**Keywords**: Vision-Language Models, Reinforcement Learning, Goal Discovery, ELO Rating, Reward Function

## TL;DR

This paper proposes GoalLadder, a framework that leverages VLMs to incrementally discover and rank candidate goal states, employs an ELO rating system to handle noisy feedback, and defines distance-based rewards in a learned embedding space. Using only a single language instruction, the method trains RL agents to achieve approximately 95% success rate.

## Background & Motivation

Natural language instructions (e.g., "open the drawer") provide a concise specification for RL tasks, yet extracting effective reward functions from such instructions remains a central challenge. Prior approaches suffer from two categories of limitations:

**Embedding-based methods** (e.g., VLM-RM using CLIP): embed task descriptions and observations into a shared space and use cosine similarity as the reward. However, the mismatch between CLIP's training data and the target environment leads to noisy reward functions.

**Preference-based methods** (e.g., RL-VLM-F): use VLMs to compare trajectory segments and generate preference labels for training a reward function. While more accurate than embedding-based approaches, erroneous VLM judgments corrupt the preference dataset, and the approach requires a large number of VLM queries.

The authors argue that any practical VLM-feedback method must simultaneously address two critical issues: **(a) robustness to noisy feedback** and **(b) VLM query efficiency**.

## Method

### Overall Architecture

The core idea of GoalLadder is *incremental goal discovery*: during RL training, VLMs are progressively queried to identify environment states that more closely approximate the task goal. The pipeline consists of four cyclically repeated phases:

1. **Collection**: The RL agent interacts with the environment under the current SAC policy to collect new episodes.
2. **Discovery**: The VLM is queried to determine whether a newly observed state is superior to the current best candidate goal.
3. **Ranking**: Pairs of candidate goals are sampled from the buffer, pairwise-compared via VLM, and their ELO scores are updated accordingly.
4. **Training**: The agent is trained with rewards defined as the negative distance to the top candidate goal in the embedding space.

### Key Designs

#### 1. Candidate Goal Discovery

A candidate goal buffer $\mathcal{B}_g$ is maintained, where each candidate $g_i = (o_i, e_i)$ stores an image and an ELO score. Observations $o_j$ are randomly sampled from newly collected trajectories and compared against the current highest-ranked goal $g^*$:

- VLM query: $y = \text{VLM}(o^*, o_j, l)$, $y \in \{-1, 0, 1\}$
- If $y=1$ (new observation is superior), $o_j$ is added to the buffer.
- If $y=0$ or $y=-1$, the observation is discarded, keeping the buffer focused on high-quality candidates.

This filtering mechanism avoids spending VLM query budget on irrelevant states.

#### 2. ELO Rating System

The ELO rating mechanism, borrowed from chess, is adopted to handle noisy VLM feedback. For a pair of candidate goals $(g_i, g_j)$:

- Expected score: $E_i = \frac{1}{1 + 10^{(e_j - e_i)/C}}$, $C=400$
- Score update: $e_i \leftarrow e_i + T(S_i - E_i)$, $T=32$

The ELO system incrementally absorbs noisy comparison results and adaptively adjusts scores, preventing a single erroneous VLM judgment from causing severe score distortion.

#### 3. Embedding Space Reward Definition

A VAE is trained as a visual feature extractor $\psi(\cdot)$ to map observations to compact latent representations:

$$\mathcal{L} = -\mathbb{E}_{\psi(z_t|o_t)} \log p_\theta(o_t|z_t) + D_{KL}(\psi(z_t|o_t) \| p(z_t))$$

The reward is defined as the Euclidean distance to the best candidate goal: $R(s_{t-1}, a_{t-1}) = -d(z_t, z^*)$

In practice, rewards are max-min normalized to $[0,1]$ and a nonlinear transformation $\hat{r} = r^{20}$ is applied, ensuring the agent receives a disproportionately larger reward as it approaches the goal.

### Loss & Training

- **RL backbone**: Soft Actor-Critic (SAC), with one gradient update per environment step.
- **Reward updates**: The target state and reward function are updated every $L=5000$ steps, with all stored transitions relabeled accordingly.
- **VLM feedback**: In OpenAI Gym environments, $M=5$ queries are issued every $K=2000$ steps; in Metaworld, every $K=500$ steps.
- **Buffer management**: The goal buffer is capped at $|\mathcal{B}_g|=10$; the lowest-ranked candidate is removed every $L$ steps.
- **VLM backbone**: Gemini 2.0 Flash.
- **VAE architecture**: 6-layer convolutional encoder + 6-layer deconvolutional decoder, latent dimension $|z|=16$.

## Key Experimental Results

### Main Results

Evaluation is conducted on 2 classic control tasks and 5 Metaworld robotic manipulation tasks:

| Method | Avg. Final Success Rate | Notes |
|--------|------------------------|-------|
| Oracle (ground-truth reward) | ~97% | Upper bound |
| **GoalLadder** | **~95%** | Language instruction only |
| RL-VLM-F | ~45% | Best competitor |
| VLM-RM | ~15% | CLIP embedding reward |
| RoboCLIP | ~10% | Video-language similarity |

GoalLadder approaches Oracle performance across all 7 tasks and even surpasses Oracle on the Drawer Open task.

### Ablation Study

- **VLM query efficiency**: GoalLadder requires on average only ~4,500 VLM queries to solve Metaworld tasks, compared to ~15,000 for PEBBLE (which uses ground-truth reward preferences).
- **Goal discovery process**: Top candidate goals steadily improve throughout training, and the buffer naturally organizes itself according to task progress.
- **ELO convergence**: Once a clearly superior goal is discovered (~50K steps), the ELO system rapidly promotes it to the top position.

### Key Findings

1. GoalLadder outperforms the best competitor (RL-VLM-F) by approximately 50 percentage points.
2. Surpassing Oracle on Drawer Open suggests that hand-crafted reward functions can themselves be suboptimal.
3. RL-VLM-F performs adequately on simple tasks but collapses on complex Metaworld tasks, highlighting the difficulty of learning reward functions from noisy preferences.
4. GoalLadder requires no environment modifications (e.g., robot removal), whereas RL-VLM-F requires robot removal to facilitate VLM judgment.

## Highlights & Insights

1. **The combination of ELO and VLM is particularly elegant**: a well-established rating system is repurposed to handle the inherent noise in VLM feedback, without assuming feedback accuracy.
2. **Paradigm shift**: the approach moves from "learning a reward function" to "discovering goal states + distance rewards," circumventing the difficulty of training a reward model from noisy labels.
3. **High query efficiency**: the embedding space is trained on unlabeled data, with only a small number of VLM queries needed to identify goal states; rewards then generalize automatically to unseen states.
4. **Surprising experimental results**: near-Oracle performance, with one task even exceeding Oracle, strongly validates the effectiveness of the method.

## Limitations & Future Work

1. The approach assumes task goals can be represented by a single image (static goals), and cannot handle dynamic tasks requiring sequential judgment.
2. Visual feature similarity serves as a proxy for state distance, which may be insufficient in certain environments.
3. Validation is limited to simulated environments; real-robot scenarios are not explored.
4. VLM inference costs remain non-trivial (single-agent training takes approximately 45 hours on a V100 GPU).
5. The framework could be extended to video-understanding VLMs to accommodate dynamic goals.

## Related Work & Insights

- **Fundamental distinction from RL-VLM-F**: RL-VLM-F trains a parameterized reward function, whereas GoalLadder directly maintains a ranked set of goal states.
- **Comparison with PEBBLE**: PEBBLE requires 3× more queries (with ground-truth reward preferences), yet GoalLadder achieves comparable performance using noisy VLM feedback.
- **Broader insight**: The ELO rating idea generalizes to other settings requiring signal extraction from noisy feedback, such as reward model training in RLHF.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (The combination of ELO rating, VLM-based goal discovery, and embedding-space distance rewards is highly original)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Comprehensive validation across 7 tasks, but lacks real-robot experiments)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear motivation, natural method presentation, intuitive figures)
- Value: ⭐⭐⭐⭐⭐ (Substantially outperforms competing methods; high practical value)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Unified Reinforcement and Imitation Learning for Vision-Language Models](unified_reinforcement_and_imitation_learning_for_vision-language_models.md)
- [\[NeurIPS 2025\] RoboRefer: Towards Spatial Referring with Reasoning in Vision-Language Models for Robotics](roborefer_towards_spatial_referring_with_reasoning_in_vision-language_models_for.md)
- [\[NeurIPS 2025\] HoPE: Hybrid of Position Embedding for Long Context Vision-Language Models](hope_hybrid_of_position_embedding_for_long_context_visionlan.md)
- [\[NeurIPS 2025\] Adapting Vision-Language Models for Evaluating World Models](adapting_visionlanguage_models_for_evaluating_world_models.md)
- [\[CVPR 2026\] Diagnosing and Repairing Unsafe Channels in Vision-Language Models via Causal Discovery and Dual-Modal Safety Subspace Projection](../../CVPR2026/multimodal_vlm/diagnosing_and_repairing_unsafe_channels_in_vision-language_models_via_causal_di.md)

</div>

<!-- RELATED:END -->
