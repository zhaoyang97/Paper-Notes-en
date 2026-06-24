---
title: >-
  [Paper Note] Preference Adaptive and Sequential Text-to-Image Generation
description: >-
  [ICML 2025][Image Generation][Text-to-Image] PASTA models personalized T2I generation as a multi-turn sequential decision-making problem. By generating candidate prompts via VLM, training a user preference model via EM, and learning a value function using offline RL (IQL), it significantly outperforms baseline LMMs in human evaluations.
tags:
  - "ICML 2025"
  - "Image Generation"
  - "Text-to-Image"
  - "Personalization"
  - "Sequential Interaction"
  - "Reinforcement Learning"
  - "User Preference"
date: 2026-05-08
content_hash: 480fef762c5a6312
---

# Preference Adaptive and Sequential Text-to-Image Generation

**Conference**: ICML 2025  
**arXiv**: [2412.10419](https://arxiv.org/abs/2412.10419)  
**Code**: [https://www.kaggle.com/datasets/googleai/pasta-data](https://www.kaggle.com/datasets/googleai/pasta-data) (dataset)  
**Area**: Image Generation  
**Keywords**: Text-to-Image, Personalization, Sequential Interaction, Reinforcement Learning, User Preference

## TL;DR
PASTA models personalized T2I generation as a multi-turn sequential decision-making problem. By generating candidate prompts via VLM, training a user preference model via EM, and learning a value function using offline RL (IQL), it significantly outperforms baseline LMMs in human evaluations.

## Background & Motivation
**Background**: T2I diffusion models (such as Stable Diffusion XL) can generate high-quality images, but single-turn generation struggles to precisely match user intent—especially for complex or abstract concepts.

**Limitations of Prior Work**: (a) Users' initial prompts are often incomplete or ambiguous; (b) one-shot generation cannot be iteratively refined; (c) different users have different implicit preferences, making generic solutions hard to adapt.

**Key Challenge**: The uncertainty of user intent requires interactive exploration to resolve, but existing T2I systems lack the capability for personalized multi-turn interaction.

**Key Insight**: Formulating the problem as a Latent Contextual MDP—where the user type is a latent variable inferred progressively through interaction.

**Core Idea**: LMM (Gemini Flash) generates a large pool of candidate prompts $\rightarrow$ the value function selects the optimal slate $\rightarrow$ user provides selection feedback $\rightarrow$ iterate for $H$ turns.

## Method

### Overall Architecture
Initial prompt $\rightarrow$ LMM generates $L_C=25$ candidate prompts (divided into 5 categories) $\rightarrow$ value model selects $L=4$ prompts $\rightarrow$ each prompt generates $M=4$ images $\rightarrow$ user selects their favorite column $\rightarrow$ repeat for $H=5$ turns.

### Key Designs

1. **User Preference Model (EM Training)**:

    - Assume $K$ discrete user types, each with a different scoring function
    - Scoring model $s_\theta(k, p, I)$: based on CLIP encoder + user-specific encoder heads
    - Utility model: $R_\theta = \text{Agg}(s_\theta(k,p,I_1),...,s_\theta(k,p,I_M))$
    - Choice model: $C_\theta = \text{Softmax}(\tau_\theta \cdot R_{1,t}^k,..., \tau_\theta \cdot R_{L,t}^k)$
    - E-step: Compute the posterior $\gamma_i(k)$ of each sample belonging to each user type
    - M-step: Maximize the weighted log-likelihood
    - Design Motivation: Use EM to discover the clustering structure of user types, personalizing the preference model

2. **Candidate Generation and Selection (LMM + Value Function)**:

    - LMM (Gemini 1.5 Flash) serves as the candidate generator to provide $L_C$ candidates
    - Candidates are divided into 5 categories (rephrase, expansion, style, subject change, creative)
    - At most 1 candidate is selected from each category to ensure diversity
    - Value function decomposition: $q_\phi(h, P) = \frac{1}{L} \sum_{p \in P} f_\phi(h, p)$
    - Design Motivation: Candidate generation introduces diversity/exploration, while the value function enables exploitation; decomposition reduces selection complexity from exponential to $O(L_C \log L_C)$

3. **Offline RL Training (IQL)**:

    - Uses Implicit Q-Learning to avoid evaluating out-of-distribution state-actions
    - $\alpha$-expectile value estimation approximates the optimal Q-value
    - Training data: Real human evaluation data + simulated user data (30,000+ trajectories)
    - Design Motivation: Offline RL is efficient, and IQL is robust to out-of-distribution actions

### Loss & Training
- User model: EM alternating optimization (BT preference loss + score regression loss + choice cross-entropy)
- Value function: IQL loss = TD error + expectile value loss
- Two-stage data: Pre-training on large-scale single-turn data (HPS v2, Pick-a-Pic, SAC) + fine-tuning on human multi-turn data

## Key Experimental Results

### Main Results (Human Evaluation, "Better/Same/Worse" ratio)

| Method | Turn 2 Better | Turn 3 Better | Turn 4 Better | Turn 5 Better |
|------|--------------|--------------|--------------|--------------|
| PASTA (full) | ~50% | ~55% | ~50% | ~48% |
| Gemini Flash Baseline | ~40% | ~38% | ~35% | ~32% |

### Ablation Study

| Configuration | Performance | Explanation |
|------|------|------|
| PASTA (Real + Sim data) | **Best** | The two data sources complement each other |
| PASTA (Real data only) | Second best | Insufficient data limits generalization |
| PASTA (Sim data only) | Slightly worse than baseline | Discrepancy between simulated data and real distribution |
| Gemini Flash (No RL) | Baseline | Lacks sequential optimization capability |

### User Model Evaluation

| Number of User Types $K$ | Pick-a-Pic Accuracy | HPS Rank Correlation |
|---------------|------------------|-------------|
| 1 | ~60% | ~0.28 |
| 8 | ~64% | ~0.33 |
| 32 | ~65% | ~0.34 |
| 64 | ~65% (saturated) | ~0.35 |

### Key Findings
- A user type count of $K=8$ is sufficient to capture major preference variations (animals, landscapes, food, portraits, etc.)
- Combined training on both real and simulated data is the most effective—simulated data provides quantity, while real data provides quality
- For abstract prompts (e.g., "image of happiness"), different user types lead to noticeably different visual styles
- Slate selection mechanism and diversity constraints are crucial for exploration

## Highlights & Insights
- **Problem Formalization**: Modeling personalized T2I as a Latent Contextual MDP is highly natural and elegant.
- **EM User Modeling**: Automatically discovers preference clusters without requiring explicit user type annotations.
- **Slate Value Decomposition**: The selection strategy with $O(L_C \log L_C)$ complexity makes large candidate sets feasible.
- **First Open-source Multi-turn T2I Dataset**: Holds long-term value for the research community.

## Limitations & Future Work
- Human evaluation might be affected by assessor demographic bias (lacking A/B testing infrastructure).
- Simulated users do not fully reflect real human behaviors.
- The prompt expansion of generated images is invisible to users, limiting interaction transparency.
- The value function is based on Gemma 2B (text-only) and cannot leverage visual information.

## Related Work & Insights
- EUREKA/L2R series use LLMs for reward design.
- Diffusion model alignment (DPO for diffusion).
- Preference elicitation theory in recommender systems.
- Insight: T2I should not be a one-shot interaction, but an iterative co-creation process.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to introduce personalized sequential decision-making into T2I; EM-based user modeling is highly ingenious.
- Experimental Thoroughness: ⭐⭐⭐⭐ Human evaluation + simulated users + ablations + model analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Comprehensive and systematic problem definition, method design, and experimental evaluation.
- Value: ⭐⭐⭐⭐⭐ Open-source dataset + practical framework, providing a significant boost to personalized generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Diffusion Adaptive Text Embedding for Text-to-Image Diffusion Models](../../NeurIPS2025/image_generation/diffusion_adaptive_text_embedding_for_texttoimage_diffusion.md)
- [\[CVPR 2026\] OSPO: Object-Centric Self-Improving Preference Optimization for Text-to-Image Generation](../../CVPR2026/image_generation/ospo_object-centric_self-improving_preference_optimization_for_text-to-image_gen.md)
- [\[ECCV 2024\] AdaNAT: Exploring Adaptive Policy for Token-Based Image Generation](../../ECCV2024/image_generation/adanat_exploring_adaptive_policy_for_token-based_image_generation.md)
- [\[CVPR 2026\] Towards Robust Sequential Decomposition for Complex Image Editing](../../CVPR2026/image_generation/towards_robust_sequential_decomposition_for_complex_image_editing.md)
- [\[ICCV 2025\] Adaptive Routing of Text-to-Image Generation Requests Between Large Cloud Models and Small Edge Models](../../ICCV2025/image_generation/adaptive_routing_of_text-to-image_generation_requests_between_large_cloud_model_.md)

</div>

<!-- RELATED:END -->
