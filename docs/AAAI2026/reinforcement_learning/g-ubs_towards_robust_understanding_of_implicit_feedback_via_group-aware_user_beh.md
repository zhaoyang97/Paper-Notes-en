---
title: >-
  [Paper Note] G-UBS: Towards Robust Understanding of Implicit Feedback via Group-Aware User Behavior Simulation
description: >-
  [AAAI 2026][Reinforcement Learning][implicit feedback] This paper proposes G-UBS (Group-aware User Behavior Simulation), a paradigm that employs a User Group Manager (UGM) based on a "Summarize–Cluster–Reflect" LLM workflow to generate group profiles, combined with group-aware reinforcement learning in a User Feedback Modeler (UFM), achieving robust user behavior understanding under implicit feedback noise. The paper also introduces IF-VR, the first multimodal implicit feedback benchmark for video recommendation.
tags:
  - AAAI 2026
  - Reinforcement Learning
  - implicit feedback
  - user behavior simulation
  - group-awareness
  - recommender systems
date: 2026-05-08
content_hash: 2bf5df7b614138d7
---

# G-UBS: Towards Robust Understanding of Implicit Feedback via Group-Aware User Behavior Simulation

**Conference**: AAAI 2026
**arXiv**: [2508.05709](https://arxiv.org/abs/2508.05709)
**Code**: None
**Area**: Reinforcement Learning
**Keywords**: implicit feedback, user behavior simulation, group-awareness, reinforcement learning, recommender systems

## TL;DR

This paper proposes G-UBS (Group-aware User Behavior Simulation), a paradigm that employs a User Group Manager (UGM) based on a "Summarize–Cluster–Reflect" LLM workflow to generate group profiles, combined with group-aware reinforcement learning in a User Feedback Modeler (UFM), achieving robust user behavior understanding under implicit feedback noise. The paper also introduces IF-VR, the first multimodal implicit feedback benchmark for video recommendation.

## Background & Motivation

**Importance and challenges of implicit feedback**: On multimodal content platforms such as TikTok, Kuaishou, and Tencent Video, users rarely provide explicit feedback (likes, ratings). Platforms rely primarily on implicit behavioral signals (quick swipes, non-clicks, low completion rates) to infer user preferences.

**The fundamental problem with implicit feedback — noise**:

- **Quick swipe ≠ dislike**: May result from accidental operations (one-handed use), user habits, or environmental distractions rather than genuine disinterest in the content.
- **Noise leads to misjudgment**: Incorrectly inferring user interests degrades recommendation accuracy and ultimately causes user churn.

**Limitations of existing approaches**:
- **Embedding-based methods** (DeepFM, CDR, etc.): Map implicit feedback to feature embeddings but cannot truly understand why a user dislikes content; poor interpretability.
- **LLM-based methods** (RecCoT, KuaiSim, etc.): Handle only text modalities and lack multimodal perception; more critically, **they do not address noise in individual implicit feedback**.
- **User simulation methods** (USImAgent, OASIS, etc.): LLM-driven large-scale simulation relies on model capability; LLMs without fine-tuning understand implicit feedback inaccurately.

**Core innovation direction**: Leveraging **contextual guidance from relevant user groups** to denoise individual implicit feedback. The intuition is: if a user belonging to a middle-aged/elderly group quickly swipes past an extreme sports video, the group profile "prone to physical discomfort from aerial footage" enables more accurate inference of the swipe reason.

## Method

### Overall Architecture

G-UBS consists of two collaborative agents:
1. **UGM (User Group Manager)**: An LLM-based "Summarize–Cluster–Reflect" workflow that generates up to 50 group profiles for 1,000+ users.
2. **UFM (User Feedback Modeler)**: Integrates UGM group profiles and multimodal information, optimizing individual user simulators via group-aware reinforcement learning.

### Key Designs

#### 1. **UGM: Summarize–Cluster–Reflect Workflow**

**Phase 1: Summarize (generate initial group profiles)**

Given a user profile set $\mathcal{U}$ of 1,000+ users (containing ID, occupation, age, gender, and interest tags), DeepSeek-R1 performs initial classification. A desired number of groups $k$ and grouping mode $M$ (by interest/demographics) are specified, yielding $k$ user groups $\mathcal{G}$ and representative users $U_g$:

$$U_g, \mathcal{G} = \mathcal{S}(\mathcal{U}, k, M) \quad |\mathcal{G}| = k$$

**Phase 2: Cluster (assign users to groups)**

Based on similarity to representative users $u_g$ (TF-IDF), the top-60 most similar users in each group are selected to form the initial user cluster $C_g$:

$$\mathcal{C}_g = \{u \in \mathcal{U}, u_g \in U_g \mid \text{Sim}(u, u_g) \geq \tau_g\}$$

GPT-4o then generates an initial group profile $\hat{P_g}$ for each group.

**Phase 3: Reflect (refine group profiles)**

Key insight: the first two phases are constrained by LLM context length and do not incorporate user watch histories, potentially causing mismatches between interest tags and actual behavior. The reflection phase introduces watch history for secondary verification:

$$C_g' = \{u \in C_g \mid \text{Match}(u, \hat{P_g}, h) = \text{'Yes'}\}$$

Groups with fewer than 10 matching users do not yield a group profile. GPT-4o then integrates the profiles and watch histories of all retained users to produce the final group profile $P_g$.

**Design Motivation**: The three-step workflow sequentially addresses three distinct problems — summarization establishes a macro-level framework, clustering performs micro-level matching, and reflection eliminates inconsistencies between tags and behavior.

#### 2. **UFM: Group-Aware Reinforcement Learning Training**

**SFT warm-up**: Using 50K explicit dislike feedback instances (e.g., "dislike this content," "dislike this creator"), GPT-4o generates chain-of-thought annotations for supervised fine-tuning, enabling the model to quickly grasp the core task logic.

**Profile Sampling**: At each training step, three types of profiles are sampled:
- Training user profile $u_T$
- Group profile $P_g$ (the group to which $u_T$ belongs)
- Similar user profile $u_S$ (another user in the same group)

Three responses are generated respectively: $O = \{o_S, o_T, o_G\}$.

**Reward mechanism**: Three reward types:
- Format reward $r_{format}$: ensures correct output format (`<think>`/`<answer>` tags)
- Skip reward $r_{skip}$: predicts whether the user quickly swiped past
- Choice reward $r_{choice}$: selects the correct reason option for the quick swipe

Total reward: $R(o) = r_{format} + r_{skip} + r_{choice}$

**GA-GRPO (Group-Aware GRPO)**:

Rewards for the three profile types are weighted separately: $R_T = R(o_T) \times W_T$, $R_G = R(o_G) \times W_G$, $R_S = R(o_S) \times W_S$

Normalized advantage function:

$$A_R = \frac{R - \text{mean}(\{R_T, R_S, R_G\})}{\text{std}(\{R_T, R_S, R_G\})}$$

The training objective incorporates a KL divergence constraint to prevent excessive policy deviation:

$$\max_{\pi_\theta} \mathbb{E}_{o \sim \pi_{\theta_\text{old}}} \left[\sum_{o \in O} \frac{\pi_\theta(o)}{\pi_{\theta_\text{old}}(o)} \cdot A_R - \beta \text{D}_\text{KL}(\pi_\theta \| \pi_\text{ref})\right]$$

The optimal weight combination is $W_T = 0.7, W_G = 0.15, W_S = 0.15$.

**Design Motivation**: Incorporating group-level and similar-user information into RL training enables the model to leverage group commonalities to denoise individual noisy feedback signals.

#### 3. **IF-VR Dataset Construction**

The first multimodal implicit feedback benchmark for video recommendation, sourced from the Tencent Video APP:
- 15K user profiles (including age, gender, occupation, and interest tags)
- 25K videos with titles
- 933K interaction records
- 50K explicit dislike feedback + 72K implicit feedback annotations (GPT-4o annotation + manual review)
- Covers two recommendation modes: sequential video recommendation (swipe-skip) and click simulation

**Implicit feedback taxonomy**:
- Content-driven: vulgar content, clickbait, disturbing visuals
- Algorithm-driven: inaccurate user profiling, repeated recommendations, insufficient diversity
- User-driven: accidental operations, temporary lack of viewing intent

### Loss & Training

Training pipeline: SFT (1 epoch) → RL (200 steps). Base model: Qwen2.5-VL-7B, full-parameter fine-tuning, learning rate 1e-5, 4 × A100 80G GPUs.

## Key Experimental Results

### Main Results

Comparison with SOTA LLMs/MLLMs on IF-VR:

| Model | Person Play Rate | Total Play Rate | Play Rate>30% | Click Rate | Reason F1 | Reason Acc |
|-------|-----------------|-----------------|---------------|------------|-----------|------------|
| Original Rec. | 46.5% | 48.3% | 76.3% | 21.4% | - | - |
| Qwen3-235b | 48.3% | 51.6% | 83.8% | 21.9% | 38.6% | 42.3% |
| DeepSeek-R1 | 49.6% | 53.0% | 83.8% | 22.7% | 41.3% | 48.0% |
| GPT-4o | 51.3% | 52.8% | 84.7% | 23.0% | 37.4% | 40.5% |
| **G-UBS** | **52.3%** | **55.3%** | **88.7%** | **25.7%** | **55.6%** | **62.9%** |

G-UBS improvements: Person Play Rate +5.8pp (vs. original recommendation), Reason Acc +14.9pp (vs. GPT-4o).

User simulation on public datasets (MovieLens & Amazon Books):

| Method | MovieLens Acc | MovieLens F1 | Amazon Acc | Amazon F1 |
|--------|-------------|-------------|-----------|-----------|
| Agent4Rec | 69.1% | 69.8% | 68.9% | 67.9% |
| GPT-4o | 72.2% | 73.6% | 73.4% | 73.6% |
| SimUser | 79.1% | 77.7% | 79.1% | 79.4% |
| **G-UBS** | **79.9%** | **78.2%** | **80.1%** | **80.2%** |

### Ablation Study

Training pipeline ablation (Table 7, the most critical ablation):

| SFT | RL | Group Profile | Person Play Rate | Play Rate>30% | Judge F1 | Reason F1 |
|-----|----|---------------|-----------------|---------------|---------|---------|
| ✗ | ✗ | ✗ | 47.2% | 79.3% | 35.8% | 30.6% |
| ✓ | ✗ | ✗ | 48.6% | 80.8% | 38.0% | 36.4% |
| ✓ | ✓ | ✗ | 51.2% | 87.4% | 51.4% | 46.5% |
| ✗ | ✓ | ✓ | 50.8% | 87.9% | 52.6% | 50.0% |
| ✓ | ✓ | ✓ | **52.3%** | **88.7%** | **54.9%** | **55.6%** |

UGM grouping strategy ablation (Table 5):

| Grouping Basis | Person Play Rate | Reason F1 |
|----------------|-----------------|-----------|
| Interest only | **52.3%** | **55.6%** |
| Demographics only | 52.0% | 55.1% |
| Interest + Demographics | 52.2% | 55.3% |

### Key Findings

1. **Group profiles are the primary driver of improvement**: SFT+RL without groups (51.2%) → SFT+RL+Group (52.3%), Reason F1 jumps from 46.5% to 55.6% (+9.1pp).
2. **Interest-tag grouping outperforms demographic grouping**: Users of the same age/gender may have vastly different interests; interest tags more directly reflect preferences.
3. **TF-IDF outperforms BERT and K-Means**: Because user profiles are structured concatenated word sequences rather than natural language.
4. **Optimal number of groups is 20**: Too few → large intra-group interest divergence; too many → insufficient users per group, profiles lack representativeness.
5. **Visual information is effective**: Removing video frames causes a 1.4pp drop in Person Play Rate (50.9% vs. 52.3%).
6. **Optimal weights are $W_T=0.7, W_G=0.15, W_S=0.15$**: Moderate weighting of group and similar-user signals is beneficial; excessive weighting causes individual differences to be ignored.

## Highlights & Insights

1. **The core idea of group-based denoising is concise and powerful**: Individual behavior is noisy, but group behavior is more stable — group commonalities guide individual understanding.
2. **The three-stage UGM design is pragmatic**: Summarize → Cluster → Reflect, with each step addressing a concrete constraint (LLM capacity → vector matching → history verification).
3. **Practical value of IF-VR**: The first multimodal implicit negative feedback benchmark, sourced from a real-world APP (Tencent Video), covering 933K interaction records.
4. **Deployability**: FP16 quantization deployment; QPS = 5.3 on 4×A100, supporting 458K videos processed daily.
5. **Compelling case analysis**: The skydiving video example vividly demonstrates the denoising capability of group profiles ("middle-aged/elderly users prone to discomfort from aerial footage").

## Limitations & Future Work

1. **UGM relies on closed-source LLMs** (GPT-4o for profile generation and annotation), incurring high costs and limited reproducibility.
2. **IF-VR annotation depends on GPT-4o + manual review**; systematic evaluation of annotation quality is insufficient.
3. **Validation limited to video recommendation**: Implicit feedback patterns in e-commerce, news, and other domains may differ substantially.
4. **SFT requires explicit dislike feedback**: The method requires adaptation for platforms lacking explicit feedback.
5. **Group profiles are static**: User interests evolve over time, necessitating a periodic update mechanism.
6. **RL training uses only 200 steps**: Whether longer training yields further improvements remains unexplored.

## Related Work & Insights

- **Implicit feedback mining**: DFN (xie2021) + CDR (chen2021) → the ceiling of embedding-based methods lies in their lack of interpretability.
- **User simulation**: SimUser → this work extends it with a group-aware dimension.
- **RL for Recommendation**: DeepSeek-R1's GRPO → extended in this work to GA-GRPO.
- May inspire group-based recommendation strategies for cold-start users.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of group-awareness and RL-based denoising is novel; the IF-VR dataset has unique value.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive coverage of SOTA comparison, ablation, hyperparameters, case studies, and public datasets.
- Writing Quality: ⭐⭐⭐⭐ — Architecture diagrams are clear and experiments are detailed, though notation in the methods section is somewhat cluttered.
- Value: ⭐⭐⭐⭐ — High practical value for industrial deployment (validated on Tencent Video), though theoretical depth is limited.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Robust Deep Reinforcement Learning against Adversarial Behavior Manipulation](../../ICLR2026/reinforcement_learning/robust_deep_reinforcement_learning_against_adversarial_behavior_manipulation.md)
- [\[AAAI 2026\] In-Token Rationality Optimization: Towards Accurate and Concise LLM Reasoning via Self-Feedback](in-token_rationality_optimization_towards_accurate_and_concise_llm_reasoning_via.md)
- [\[NeurIPS 2025\] Deep RL Needs Deep Behavior Analysis: Exploring Implicit Planning by Model-Free Agents](../../NeurIPS2025/reinforcement_learning/deep_rl_needs_deep_behavior_analysis_exploring_implicit_planning_by_model-free_a.md)
- [\[AAAI 2026\] Bi-Level Contextual Bandits for Individualized Resource Allocation under Delayed Feedback](bi-level_contextual_bandits_for_individualized_resource_allocation_under_delayed.md)
- [\[AAAI 2026\] Aligning Machiavellian Agents: Behavior Steering via Test-Time Policy Shaping](aligning_machiavellian_agents_behavior_steering_via_test-tim.md)

<!-- RELATED:END -->
