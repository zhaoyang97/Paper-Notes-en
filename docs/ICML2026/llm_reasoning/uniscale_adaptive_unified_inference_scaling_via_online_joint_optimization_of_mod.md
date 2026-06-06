---
title: >-
  [Paper Note] UniScale: Adaptive Unified Inference Scaling via Online Joint Optimization of Model Routing and Test-time Scaling
description: >-
  [ICML 2026][LLM Reasoning][Model routing] The UniScale framework unifies model routing and test-time scaling (TTS) into a single decision space…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "Model routing"
  - "Test-time scaling"
  - "Joint optimization"
  - "LinUCB"
  - "Contextual multi-armed bandit"
date: 2026-05-08
content_hash: 8037facf3952c878
---

# UniScale: Adaptive Unified Inference Scaling via Online Joint Optimization of Model Routing and Test-time Scaling

**Conference**: ICML 2026  
**arXiv**: [2605.30898](https://arxiv.org/abs/2605.30898)  
**Code**: To be confirmed  
**Area**: LLM Inference  
**Keywords**: Model routing, Test-time scaling, Joint optimization, LinUCB, Contextual multi-armed bandit

## TL;DR
The UniScale framework unifies model routing and test-time scaling (TTS) into a single decision space, using a LinUCB contextual multi-armed bandit to perform online adaptive inference policy learning, addressing the fine-grained quality-cost trade-off in LLM deployment.

## Background & Motivation

**Background**: LLM deployment requires a trade-off between inference quality and computational cost. Existing methods operate along two independent dimensions: model routing (switching between models of different scales) and test-time scaling (TTS, increasing inference-time computation within a fixed model).

**Limitations of Prior Work**: Model routing only supports discrete switching, resulting in coarse-grained performance changes. Single-model TTS is limited by the model's inherent capacity, showing diminishing returns as computation increases. These two mechanisms are designed independently and cannot collaborate in dynamic environments.

**Key Challenge**: How to provide a continuous and fine-grained quality-cost trade-off from model scale to inference depth while maintaining computational efficiency?

**Goal**: Construct a Unified Inference Scaling (UIS) paradigm that treats model selection and TTS parameters as configurations in a single optimization space, supporting online adaptation.

**Key Insight**: Modeling UIS configuration selection as a contextual multi-armed bandit problem to capture the alignment between query complexity and system capability, supporting continuous policy adjustment under environmental drift.

**Core Idea**: Jointly optimize model routing and TTS so they compensate for each other—TTS narrows the performance gap between discrete models, while routing provides larger models when TTS reaches diminishing returns.

## Method

### Overall Architecture
UniScale is an online closed-loop system: (1) updates the reward estimator based on historical feedback; (2) uses the LinUCB acquisition function to select the optimal UIS configuration; (3) executes inference and applies path-aware early stopping; (4) generates a composite reward for policy optimization through dense verification feedback and a cost model.

### Key Designs

1. **Unified Inference Scaling Space**:

    - **Function**: Parameterizes UIS as a joint configuration of $(M, QP, CP, BS)$—where $M$ is the base model, and $QP$ (Question Parallelism), $CP$ (Candidate Parallelism), and $BS$ (Beam Size) parameterize TTS.
    - **Mechanism**: This parameterization constructs an expressive quality-cost frontier, using search intensity to fill discrete model gaps and model routing to move beyond search plateaus.
    - **Design Motivation**: Addresses the fundamental issues of routing discreteness and TTS capacity limits, providing a continuous and fine-grained control space.

2. **Online Adaptive Learning based on LinUCB**:

    - **Function**: Models UIS configuration selection using a contextual multi-armed bandit, estimating rewards with a linear reward predictor $\hat{r}_t = \langle \mathbf{x}_{t,a}, \boldsymbol{\theta} \rangle$ via joint semantic representations $\mathbf{x}_{t,a}$ of queries and actions.
    - **Mechanism**: Selects the configuration that maximizes the LinUCB acquisition function $a_t = \arg\max_{a \in \mathcal{A}}(\hat{\boldsymbol{\theta}}_t^\top \mathbf{x}_{t,a} + \alpha\sqrt{\mathbf{x}_{t,a}^\top \mathbf{A}_t^{-1} \mathbf{x}_{t,a}})$, balancing exploitation and exploration.
    - **Design Motivation**: Captures alignment between queries and configurations, supporting rapid convergence in non-stationary environments; uses the Sherman-Morrison formula to reduce complexity from $\mathcal{O}(d^3)$ to $\mathcal{O}(d^2)$.

3. **Path-aware Early Stopping + Dense Verification + Cost Modeling**:

    - **Function**: Path-aware early stopping dynamically eliminates low-potential inference paths; dense verification feedback combines binary correctness with continuous verifier scores; the cost model uses equivalent FLOPs (eFLOPs) to unify computation and memory overhead.
    - **Mechanism**: Early stopping condition is $\frac{j \cdot V(p_{i,j}) + (H_{\max}-j) \cdot v_{\sup}}{H_{\max}} < V_{\max}$; dense feedback is $r_t = w_1 \cdot \text{Correct}(a_t) + w_2 \cdot \text{Score}(a_t) + w_3 \cdot (1-\tilde{C}_{\text{UIS}}(a_t))$.
    - **Design Motivation**: Significantly reduces inference costs for all configurations while maintaining quality, providing richer training signals for LinUCB.

## Key Experimental Results

### Main Results

| Method | Cost-Sensitive Mode Reward | Cost-Sensitive Mode Accuracy | Quality-First Mode Reward | Quality-First Mode Accuracy |
|------|------------|------------|------------|------------|
| Random | 0.5731 | 55.00% | 0.6175 | 52.88% |
| Greedy | 0.5589 | 45.00% | 0.5780 | 52.88% |
| k-NN | 0.6590 | 41.38% | 0.5807 | 46.75% |
| **UniScale (TTS)** | **0.7184** | **43.75%** | **0.6184** | **46.50%** |
| **UniScale (Routing)** | **0.5873** | **34.12%** | **0.5459** | **50.00%** |
| **UniScale (UIS)** | **0.5589** | **45.00%** | **0.5780** | **52.88%** |

### Ablation Study

| Configuration | Full Model Reward | w/o Path-aware Early Stopping | w/o Dense Verification | w/o Cost Model |
|------|---------|----------|---------|---------|
| UIS Full | 0.6590 | -8.2% | -5.7% | -4.3% |
| TTS Only | 0.7184 | -6.1% | -3.9% | -2.8% |

### Key Findings
- UIS improves rewards by 8.2% and 12.1% compared to TTS and routing alone, respectively, validating the effectiveness of joint optimization.
- Path-aware early stopping contributes the most to performance in ablation (-8.2%), followed by dense verification feedback (-5.7%).
- Ours performs optimally in both cost-sensitive and quality-first modes, demonstrating broad adaptability.

## Highlights & Insights
- **Innovative Problem Reformulation**: Merges independent model routing and TTS into a unified decision space.
- **Efficient Online Learning**: The combination of LinUCB and Transformer encoders ensures fast convergence while minimizing computational overhead.
- **Engineering Completeness**: The three-layer mechanism (early stopping, verification, cost model) works in tight coordination to solve specific pain points in real-world deployment.

## Limitations & Future Work
- Cost modeling assumptions—eFLOPs mapping may still exhibit bias on heterogeneous hardware.
- Verifier dependency—the performance upper bound is limited by the quality of the verifier.
- Future Directions: Incorporating quantitative model tuning and dynamic verifier updates; extending to streaming/multi-turn dialogue scenarios.

## Related Work & Insights
- **vs. Model Ensemble Routing**: Traditional routing uses discrete switching; ours uses continuous scaling.
- **vs. Single-model TTS**: TTS is limited by capacity under a fixed model; ours breaks through via routing.
- **vs. Reinforcement Learning**: LinUCB provides theoretical guarantees (regret bounds) and high online efficiency.

## Rating
- Novelty: ⭐⭐⭐⭐⭐  High originality in the unified framework design.
- Experimental Thoroughness: ⭐⭐⭐⭐  Covers TTS, routing, and UIS dimensions; baselines could be further enriched.
- Writing Quality: ⭐⭐⭐⭐⭐  Clear logic and natural problem introduction.
- Value: ⭐⭐⭐⭐⭐  Directly addresses practical LLM deployment needs; the framework is easily generalizable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Lookahead Sample Reward Guidance for Test-Time Scaling of Diffusion Models](lookahead_sample_reward_guidance_for_test-time_scaling_of_diffusion_models.md)
- [\[ICML 2026\] Prism: Efficient Test-Time Scaling via Hierarchical Search and Self-Verification for Discrete Diffusion Language Models](prism_efficient_test-time_scaling_via_hierarchical_search_and_self-verification_.md)
- [\[ICML 2026\] Inference Time Optimization with Confidence Dynamics](inference_time_optimization_with_confidence_dynamics.md)
- [\[ICML 2026\] ETS: Energy-Guided Test-Time Scaling for Training-Free RL Alignment](ets_energy-guided_test-time_scaling_for_training-free_rl_alignment.md)
- [\[ACL 2026\] Parallel Test-Time Scaling for Latent Reasoning Models](../../ACL2026/llm_reasoning/parallel_test-time_scaling_for_latent_reasoning_models.md)

</div>

<!-- RELATED:END -->
