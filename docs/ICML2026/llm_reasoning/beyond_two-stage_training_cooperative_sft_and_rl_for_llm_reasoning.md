---
title: >-
  [Paper Note] Beyond Two-Stage Training: Cooperative SFT and RL for LLM Reasoning
description: >-
  [ICML 2026][LLM Reasoning][Reinforcement Learning] The BRIDGE framework is proposed to model the integration of SFT and RL as a bilevel optimization problem—where SFT, acting as an upper-level teacher…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "Reinforcement Learning"
  - "Supervised Fine-Tuning"
  - "Bilevel Optimization"
  - "Meta-Learning"
date: 2026-05-08
content_hash: 6eae9b4328acc06a
---

# Beyond Two-Stage Training: Cooperative SFT and RL for LLM Reasoning

**Conference**: ICML 2026  
**arXiv**: [2509.06948](https://arxiv.org/abs/2509.06948)  
**Code**: https://github.com/ChanLiang/BRIDGE  
**Area**: Reinforcement Learning  
**Keywords**: LLM Reasoning, Reinforcement Learning, Supervised Fine-Tuning, Bilevel Optimization, Meta-Learning  

## TL;DR

The BRIDGE framework is proposed to model the integration of SFT and RL as a bilevel optimization problem—where SFT, acting as an upper-level teacher, learns to selectively transfer beneficial supervision signals to the RL student via a lightweight LoRA module, achieving an average absolute improvement exceeding 3 percentage points across five mathematical reasoning benchmarks.

## Background & Motivation

**Background**: SFT (Supervised Fine-Tuning) and RLVR (Reinforcement Learning via Verifiable Rewards) are the two primary paradigms for LLM reasoning post-training. SFT efficiently mimics expert trajectories but is prone to overfitting, while RLVR discovers high-reward trajectories through exploration but suffers from low sample efficiency. The current mainstream approach is a "two-stage pipeline"—performing SFT followed by RL.

**Limitations of Prior Work**: Two-stage methods do not always outperform pure RL (performing even worse on Llama-3.2-3B). The advantage of SFT is primarily a static initialization effect; once the RL stage begins, supervision signals are discarded, and the model must rely on unguided exploration. Meanwhile, single-stage hybrid methods (direct weighting $J_{\text{hyb}}(\theta) = J_{\text{RL}}(\theta) + \mu J_{\text{SFT}}(\theta)$) perform worse—experiments show that the rewards of naive mixing are even lower than those of pure RL.

**Key Challenge**: Not all supervision updates are beneficial for reward optimization. Directly mixing SFT and RL gradients can be counterproductive. The core problem is: how to dynamically extract supervision signals that truly facilitate RL reward maximization?

**Key Insight**: The authors observe a hierarchical teacher-student relationship between SFT and RL—SFT possesses expert reasoning trajectories (teacher), and RL seeks high-reward policies through exploration (student). Modeling the two as a bilevel optimization (Stackelberg Game) allows SFT to adaptively adjust the supervision method based on its helpfulness to RL.

**Core Idea**: Meta-learning is employed to let SFT learn how to "teach" RL—by maximizing a cooperative gain signal of joint training relative to pure RL, ensuring that supervision updates are adopted only when they contribute to reward optimization.

## Method

### Overall Architecture

BRIDGE models the SFT-RL integration as a bilevel optimization problem: the upper level (leader) is the SFT objective, controlling a lightweight LoRA teacher module $w$; the lower level (follower) is the RL objective, optimizing the LLM backbone parameters $\theta$. Each training step alternates between two updates: (1) Student update—fusing SFT and RL gradients to update $\theta$; (2) Teacher update—updating the LoRA parameters $w$ by maximizing the cooperative gain signal. At inference time, $w$ is directly merged into $\theta$ with zero additional overhead.

### Key Designs

1. **Bilevel Formulation**:

    - **Function**: Models the hierarchical collaborative relationship between SFT and RL.
    - **Mechanism**: The upper objective is $\max_w J_{\text{SFT}}(w, \theta^*(w))$, and the lower constraint is $\theta^*(w) = \arg\max_\theta J_{\text{RL}}(\theta, w)$. The optimization of SFT is conditioned on the optimal solution of RL, ensuring supervision signals do not interfere with reward optimization. This is transformed into a single-level problem via penalty relaxation: $\max_{\theta,w} (1-\lambda) J_{\text{SFT}}(\theta,w) - \lambda \, p(w,\theta)$, where $p(w,\theta)$ measures the deviation of $\theta$ from the RL optimal solution.
    - **Design Motivation**: Solving bilevel problems directly requires second-order derivatives, which is infeasible at LLM scale; penalty relaxation requires only first-order gradients, and the approximation error is $O(1-\lambda)$.

2. **LoRA Teacher Module and Cooperative Gain Signal**:

    - **Function**: Uses an independent low-rank parameter space to coordinate the two objectives.
    - **Mechanism**: The policy is defined as $\pi_{\theta+w}$, where the teacher reshapes the loss landscape of $\theta$ optimization by modifying $w$. The teacher's meta-objective is $J_{\text{meta}} = (1-\lambda) J_{\text{SFT}}(\theta,w) + \lambda [J_{\text{RL}}(\theta,w) - J_{\text{RL}}(\hat{\theta},w)]$, where $\hat{\theta}$ represents auxiliary parameters for pure RL updates. The second term is the **Cooperative Gain**—the reward difference between joint training and pure RL.
    - **Design Motivation**: The independent low-rank subspace isolates meta-updates from RL optimization to prevent mutual interference; it results in zero inference cost after merging; the cooperative gain signal naturally eliminates additive bias noise in rewards.

3. **Alternating Three-Way Update Algorithm**:

    - **Function**: Efficiently executes the approximate solution of the bilevel optimization.
    - **Mechanism**: Each step samples two mini-batches (SFT and RL) and performs three updates: (a) Student $\theta^{k+1} = \theta^k + \alpha [(1-\lambda)\nabla_\theta J_{\text{SFT}} + \lambda \nabla_\theta J_{\text{RL}}]$; (b) Auxiliary parameters $\hat{\theta}^{k+1} = \hat{\theta}^k + \alpha \nabla_{\hat\theta} J_{\text{RL}}$ (pure RL baseline); (c) Teacher $w^{k+1} = w^k + \beta \nabla_w J_{\text{meta}}$. Both policies share the same batch of rollouts to evaluate cooperative gain, reducing variance.
    - **Design Motivation**: Auxiliary parameters $\hat{\theta}$ provide a pure RL baseline to quantify the cooperative gain; shared rollouts eliminate confounding sampling noise.

## Key Experimental Results

### Main Results

Evaluations were conducted on three LLMs of different scales and series. Training data consisted of the MATH dataset hard split (8.5K problems), and five mathematical reasoning benchmarks were assessed:

| Method | MATH500 | Minerva Math | OlympiadBench | AIME24 | AMC23 | Average |
|------|---------|-------------|---------------|--------|-------|------|
| RL | 64.4 | 26.5 | 27.0 | 3.3 | 40.0 | 32.2 |
| SFT→RL | 66.0 | 24.3 | 26.8 | 9.0 | 35.0 | 32.2 |
| SFT+RL | 55.6 | 20.6 | 25.0 | 3.3 | 42.5 | 29.4 |
| CHORD | 66.0 | 23.2 | 25.9 | 6.7 | 40.5 | 32.5 |
| **BRIDGE** | **66.2** | **23.9** | **28.9** | **13.3** | **47.5** | **36.0** |

> Qwen2.5-3B results. BRIDGE averages 36.0%, exceeding the strongest baseline CHORD by 3.5 percentage points.

| Model | Highest Baseline | BRIDGE | Gain (abs) |
|------|---------|--------|---------|
| Qwen2.5-3B | 32.5 (CHORD) | 36.0 | +3.5 |
| Llama-3.2-3B | 21.9 (LUFFY) | 24.7 | +2.8 |
| Qwen3-8B | 45.9 (CHORD) | 49.9 | +4.0 |

### Ablation Study

| Configuration | Avg. Accuracy | Description |
|------|-----------|------|
| BRIDGE | 49.9 | Full method (Qwen3-8B) |
| w/o $J_{\text{meta}}$ | 40.3 | Without meta-objective, degenerates to naive multi-task, down 9.6 pts |

| Metric | RL | SFT→RL | BRIDGE | Description |
|------|-----|---------|--------|------|
| Training Time (hr, 3B) | 6.1 | 12.3 | 6.9 | BRIDGE saves 44% time vs. two-stage |
| Training Time (hr, 8B) | 38.5 | 39.1 | 33.5 | BRIDGE saves 14% time vs. two-stage |
| GPU Memory (GB, 3B) | 52.2 | 45.9 | 59.3 | Approx. 11% memory increase |
| Accuracy (%, 3B) | 32.2 | 32.2 | 36.0 | 11.8% performance gain |

**Reward Noise Robustness**: When rewards are flipped with probability $p=0.2$, BRIDGE still reaches 32.9%, while pure RL drops to 13.9% (the gap widens from +3.8 to +19.0), because the additive bias $p$ in the cooperative gain is precisely canceled out.

## Highlights & Insights

- **SFT-RL Integration from a Meta-Learning Perspective**: This is the first work to model reasoning LLM training as a bilevel optimization, providing a more principled framework than heuristic mixing.
- **Natural Noise Immunity of Cooperative Gain Signals**: It is mathematically proven that bias terms in reward noise are canceled in the difference, making the method robust even under imperfect verifiers.
- **Solution Diversity Increases**: Pass@32 on AIME24 is 10 percentage points higher than pure RL, indicating that SFT trajectories enrich the exploration space.
- **Cross-Domain Zero-Shot Generalization**: Models trained on math outperform pure RL zero-shot on code (LiveCodeBench) and science (GPQA), whereas SFT→RL remains below the base model.

## Limitations & Future Work

- Testing was limited to automatically verifiable tasks (math/logic/code); the effects on subjective rewards (e.g., open-ended generation) were not verified.
- Auxiliary parameters $\hat{\theta}$ require maintaining a full LLM copy, leading to significant memory pressure at 70B+ scales.
- Although experiments show the mixing coefficient $\lambda$ is robust within the 0.3-0.7 range, the optimal value still requires adjustment by model/task.

## Related Work & Insights

- **CHORD** (Zhang et al., 2025): Dynamic weighting of SFT and RL at global and token levels; it is the strongest heuristic baseline.
- **LUFFY** (Yan et al., 2025): Injects SFT demonstrations as off-policy trajectories into RL, though limited by distribution mismatch.
- **SRFT** (Fu et al., 2025): Entropy-aware weighting and clipping to reduce objective interference, but remains a heuristic hybrid.
- **SimpleRL** (Zeng et al., 2025): A pure RL baseline framework which finds that short CoT fine-tuning may harm reasoning.
- **Insight**: Bilevel optimization provides a general framework for "learning how to teach," which can be generalized to other scenarios requiring the integration of imitation and exploration.

## Rating

- Novelty: ⭐⭐⭐⭐ (First to model SFT-RL integration as bilevel optimization)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Three models, five benchmarks, multidimensional analysis, robustness/generalization/diversity)
- Writing Quality: ⭐⭐⭐⭐ (Clear motivation derivation, complete mathematical formulas)
- Value: ⭐⭐⭐⭐ (Provides a principled alternative for SFT-RL integration)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] ETS: Energy-Guided Test-Time Scaling for Training-Free RL Alignment](ets_energy-guided_test-time_scaling_for_training-free_rl_alignment.md)
- [\[NeurIPS 2025\] First SFT, Second RL, Third UPT: Continual Improving Multi-Modal LLM Reasoning via Unsupervised Post-Training](../../NeurIPS2025/llm_reasoning/first_sft_second_rl_third_upt_continual_improving_multi-modal_llm_reasoning_via_.md)
- [\[ICML 2026\] Beyond Test-Time Memory: State-Space Optimal Control for LLM Reasoning](beyond_test-time_memory_state-space_optimal_control_for_llm_reasoning.md)
- [\[ICML 2026\] On Robustness and Chain-of-Thought Consistency of RL-Finetuned VLMs](on_robustness_and_chain-of-thought_consistency_of_rl-finetuned_vlms.md)
- [\[ACL 2026\] SHAPE: Stage-aware Hierarchical Advantage via Potential Estimation for LLM Reasoning](../../ACL2026/llm_reasoning/shape_stage-aware_hierarchical_advantage_via_potential_estimation_for_llm_reason.md)

</div>

<!-- RELATED:END -->
