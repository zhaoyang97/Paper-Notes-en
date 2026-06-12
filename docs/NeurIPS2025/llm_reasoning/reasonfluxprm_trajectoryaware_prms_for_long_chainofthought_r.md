---
title: >-
  [Paper Note] ReasonFlux-PRM: Trajectory-Aware PRMs for Long Chain-of-Thought Reasoning in LLMs
description: >-
  [NeurIPS 2025][LLM Reasoning][Process Reward Model] ReasonFlux-PRM identifies that existing PRMs fail to effectively evaluate the intermediate thinking trajectories of reasoning models…
tags:
  - "NeurIPS 2025"
  - "LLM Reasoning"
  - "Process Reward Model"
  - "Trajectory-Aware"
  - "Data Selection"
  - "Reinforcement Learning"
  - "Test-Time Scaling"
date: 2026-05-08
content_hash: 7a7ce8a3646abec5
---

# ReasonFlux-PRM: Trajectory-Aware PRMs for Long Chain-of-Thought Reasoning in LLMs

**Conference**: NeurIPS 2025  
**arXiv**: [2506.18896](https://arxiv.org/abs/2506.18896)  
**Code**: [github.com/Gen-Verse/ReasonFlux](https://github.com/Gen-Verse/ReasonFlux)  
**Area**: LLM Reasoning  
**Keywords**: Process Reward Model, Trajectory-Aware, Data Selection, Reinforcement Learning, Test-Time Scaling  

## TL;DR
ReasonFlux-PRM identifies that existing PRMs fail to effectively evaluate the intermediate thinking trajectories of reasoning models, and proposes a trajectory-aware PRM that fuses step-level alignment, quality, and coherence scores with a trajectory-level template-guided reward. The approach consistently outperforms strong baselines including Qwen2.5-Math-PRM-72B across three settings: offline data selection (SFT +12.1%), online RL reward (+4.5%), and test-time Best-of-N scaling (+6.3%).

## Background & Motivation

**Background**: Process Reward Models (PRMs) provide reward signals for each intermediate step in LLM reasoning and have been widely adopted in RL training and test-time search. Existing PRMs (e.g., Math-Shepherd, Qwen-Math-PRM) are primarily trained on models' **final output responses**—structured, linear, and well-organized step-by-step CoT.

**Limitations of Prior Work**: With the rise of reasoning models such as DeepSeek-R1 and OpenAI-o1, model outputs have evolved into a two-part **trajectory–response** format: a lengthy, less-organized intermediate thinking trajectory (containing branching, backtracking, and self-correction), followed by a concise final response. Existing PRMs perform poorly when evaluating intermediate thinking trajectories—their score distributions heavily overlap and fail to distinguish trajectories of different quality, sometimes even selecting training data inferior to human-curated sets.

**Key Challenge**: Thinking trajectories differ fundamentally from final responses: (1) trajectories contain branching and backtracking (non-linear), whereas responses are linear; (2) trajectories exhibit weaker global coherence, whereas responses are carefully organized. PRMs trained on responses naturally fail to generalize to trajectories.

**Goal**: How to design a general-purpose PRM that can effectively evaluate both intermediate thinking trajectories and final responses?

**Key Insight**: Introduce reward signals at two levels simultaneously—step-level rewards via softmax-weighted fusion of alignment, quality, and coherence scores; trajectory-level rewards via a template-guided approach that assesses the transferability of the overall reasoning strategy.

**Core Idea**: Jointly train a PRM with multi-dimensional step-level rewards (alignment + quality + coherence) and template-guided trajectory-level rewards, enabling evaluation of the complete reasoning process of reasoning models rather than only their final outputs.

## Method

### Overall Architecture
Given trajectory–response data $(s, a)$ (where $s$ is the thinking trajectory and $a$ is the final response), ReasonFlux-PRM computes a step-level reward $r_t^{\text{step}}$ for each thinking step $s_t$ (via softmax-weighted fusion of alignment, quality, and coherence scores), and simultaneously computes a trajectory-level reward $r^{\text{final}}$ (via template-guided verification) for the entire trajectory. The PRM is trained jointly with both levels of reward. The trained PRM can then be applied in three settings: offline data selection, GRPO online reward, and Best-of-N test-time scaling.

### Key Designs

1. **Three-Dimensional Step-Level Reward Design**:

    - **Alignment Score** $r_t^{\text{align}} = \text{sim}(\Phi(s_t), \Phi(a_t))$: computes the cosine similarity between each thinking step and the corresponding final response step using a pretrained encoder, encouraging thinking steps that are relevant to the final answer.
    - **Quality Score** $r_t^{\text{qual}} = J(s_t | x, s_{<t}, a)$: uses GPT-4o as a judge to assess each step's logical correctness, internal coherence, and progression toward the final answer.
    - **Coherence Score** $r_t^{\text{coh}}$: measures semantic coherence between adjacent steps using a contrastive mutual information formulation, penalizing abrupt topic shifts.
    - The three scores are fused into $r_t^{\text{step}}$ via softmax-weighted aggregation.
    - **Design Motivation**: relying solely on alignment penalizes complex but effective exploratory steps; relying solely on quality fails to capture inter-step relationships; the three dimensions are complementary.

2. **Template-Guided Trajectory-Level Reward**:

    - **Function**: evaluates whether the high-level reasoning strategy embedded in the full trajectory is reproducible.
    - **Mechanism**: GPT-4o first extracts a reasoning template $\mathcal{T}$ (a high-level step sequence) from the trajectory–response pair; the policy model $\pi_\theta$ then generates $N$ responses to the problem guided by the template; the average correctness serves as the trajectory-level reward: $r^{\text{final}} = \frac{1}{N}\sum_j \mathbb{I}(y^{(j)} \text{ is correct})$.
    - **Design Motivation**: Step-level rewards focus on local quality, whereas the template-guided reward assesses whether the reasoning strategy can actually solve the problem—a trajectory that is globally effective but locally inelegant still receives a high score.

3. **Joint Training Objective**:

    - $\mathcal{L}_{\text{total}} = \lambda_{\text{step}} \cdot \frac{1}{T}\sum_t \mathcal{L}(R_\phi(s_t), r_t^{\text{step}}) + \lambda_{\text{final}} \cdot \mathcal{L}(R_\phi(x,y), r^{\text{final}})$
    - MSE loss is used; training is conducted on the OpenThoughts-114K dataset.

### Loss & Training
- **Backbone models**: Qwen2.5-1.5B/7B-Instruct
- **Training data**: OpenThoughts-114K (thinking trajectories and responses generated by DeepSeek-R1); 1K trajectories are sampled to construct template-guided rewards.
- **Online RL integration**: The PRM reward is blended with GRPO's rule-based reward via $r^{\text{new}} = (1-\beta) r^{\text{out}} + \beta \hat{r}$.

## Key Experimental Results

### Main Results

**Offline Data Selection (SFT on Qwen2.5-14B-Instruct; 1K samples selected from 59K s1 data)**

| Data Source | AIME24 | AIME25 | MATH500 | GPQA-Diamond |
|---------|--------|--------|---------|-------------|
| Human-curated (s1k) | 33.3 | 33.3 | 78.8 | 41.4 |
| Qwen2.5-Math-PRM-72B | 33.3 | 26.7 | 77.0 | 39.4 |
| **ReasonFlux-PRM-7B** | **40.0** | **33.3** | **84.8** | **47.5** |

**Online RL (GRPO policy optimization on DeepSeek-R1-Distill-Qwen-7B)**

| Reward Signal | AIME24 | AIME25 | MATH500 | GPQA-Diamond |
|---------|--------|--------|---------|-------------|
| Rule-based | 50.2 | 38.3 | 89.6 | 47.1 |
| Qwen-Math-PRM-7B | 51.2 | 40.8 | 92.8 | 49.1 |
| **ReasonFlux-PRM-7B** | **54.6** | **44.2** | **94.8** | **51.6** |

### Ablation Study

| Configuration | AIME25 | MATH500 | Note |
|------|--------|---------|------|
| $\alpha=0.1$ (weak trajectory-level) | 6.7 | 81.2 | Insufficient local signal |
| $\alpha=0.8$ | 33.3 | 83.6 | Good |
| $\alpha=1.0$ | 33.3 | **84.8** | Best on MATH500 |
| $\alpha=1.5$ | **40.0** | 83.2 | Best on AIME25 |

### Key Findings
- **7B PRM outperforms 72B PRM**: ReasonFlux-PRM-7B surpasses Qwen2.5-Math-PRM-72B by approximately 6–8% in data selection, even exceeding human-curated s1k data.
- **1K curated samples > 59K raw data**: Models trained on 1K samples selected by ReasonFlux-PRM outperform those trained on all 59K raw samples (MATH500: 84.8 vs. ~68).
- **Stacking SFT and RL yields significant gains**: SFT with ReasonFlux-PRM-selected data followed by RL with ReasonFlux-PRM reward achieves MATH500 89.8% (a 12.8% improvement over the backbone at 77.0%).
- **Larger PRM scales better**: ReasonFlux-PRM improves MATH500 by 3.8% when scaling from 1.5B to 7B.
- **Existing PRMs exhibit heavily overlapping score distributions on thinking trajectories**, making them nearly incapable of distinguishing trajectory quality between R1 and Gemini.

## Highlights & Insights
- **The paper formally identifies and validates the important problem of "PRM failure on thinking trajectories"**: this finding has significant implications for all work that uses PRMs to evaluate reasoning model outputs—PRMs trained on final responses cannot be directly applied to internal thinking processes.
- **The template-guided trajectory-level reward is elegantly designed**: rather than directly judging trajectory correctness, it extracts the high-level reasoning strategy and verifies its reproducibility—thereby avoiding excessive penalization of exploratory steps.
- **A single PRM covers three application scenarios**: offline data selection, online RL reward, and test-time scaling—demonstrating strong generality.

## Limitations & Future Work
- Relies on GPT-4o as a judge to generate quality scores and reasoning templates, increasing construction cost.
- Validation is primarily limited to mathematical and scientific reasoning; generalization to open-ended tasks (dialogue, code) remains unexplored.
- The step segmentation strategy is simple (splitting on `"\n\n"`), which may be insufficient for trajectories with more complex structure.
- The $\alpha$ parameter is currently tuned manually; adaptive learning could be explored in future work.

## Related Work & Insights
- **vs. Qwen2.5-Math-PRM**: Trained on final responses, it cannot differentiate trajectory quality. ReasonFlux-PRM-7B outperforms its 72B counterpart across all tasks.
- **vs. Math-Shepherd / Skywork-PRM**: These PRMs perform even worse on thinking trajectories, sometimes selecting data no better than random sampling.
- **vs. s1k (human-curated)**: Data selected by ReasonFlux-PRM surpasses human-curated data by 6% on MATH500, demonstrating that automated data selection can exceed human expert curation.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic study of PRM failure on thinking trajectories; the three-dimensional step-level reward design is comprehensive.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers offline/online/test-time settings + ablation + efficiency analysis + case studies + end-to-end validation.
- Writing Quality: ⭐⭐⭐⭐ Problem analysis is clear; three key takeaways are convincing.
- Value: ⭐⭐⭐⭐⭐ Directly addresses the core problem of PRMs in the era of reasoning models; the 7B model is open-sourced and ready to use.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Let Me Think! A Long Chain-of-Thought Can Be Worth Exponentially Many Short Ones](let_me_think_a_long_chainofthought_can_be_worth_exponentiall.md)
- [\[NeurIPS 2025\] Latent Chain-of-Thought for Visual Reasoning](latent_chain-of-thought_for_visual_reasoning.md)
- [\[NeurIPS 2025\] On Learning Verifiers and Implications to Chain-of-Thought Reasoning](on_learning_verifiers_and_implications_to_chain-of-thought_reasoning.md)
- [\[NeurIPS 2025\] Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought](reasoning_by_superposition_a_theoretical_perspective_on_chain_of_continuous_thou.md)
- [\[ACL 2026\] Long-Context Reasoning Through Proxy-Based Chain-of-Thought Tuning](../../ACL2026/llm_reasoning/long-context_reasoning_through_proxy-based_chain-of-thought_tuning.md)

</div>

<!-- RELATED:END -->
