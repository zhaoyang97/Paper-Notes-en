---
title: >-
  [Paper Note] EAPO: Enhancing Policy Optimization with On-Demand Expert Assistance
description: >-
  [ICML 2026][Reinforcement Learning][Expert-assisted RL] EAPO embeds "consulting an external expert" as a learnable discrete action within the policy space. This allows the LLM to call a stronger model on-demand during th…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Expert-assisted RL"
  - "On-demand consultation"
  - "Sparse rewards"
  - "Knowledge internalization"
  - "Verifiable rewards"
date: 2026-05-08
content_hash: 4c67ab9205bfa5b0
---

# EAPO: Enhancing Policy Optimization with On-Demand Expert Assistance

**Conference**: ICML 2026  
**arXiv**: [2509.23730](https://arxiv.org/abs/2509.23730)  
**Code**: None  
**Area**: Reinforcement Learning / LLM Reasoning  
**Keywords**: Expert-assisted RL, On-demand consultation, Sparse rewards, Knowledge internalization, Verifiable rewards

## TL;DR
EAPO embeds "consulting an external expert" as a learnable discrete action within the policy space. This allows the LLM to call a stronger model on-demand during the RL training phase to obtain intermediate hints. By gradually decaying the acceptance rate, expert knowledge is internalized into the policy itself. During evaluation, the model performs independent reasoning and consistently outperforms pure self-exploratory RL on mathematical reasoning benchmarks such as AIME and AIMO.

## Background & Motivation

**Background**: The current mainstream paradigm for enhancing LLM reasoning via RL is RLVR (Reinforcement Learning with Verifiable Rewards), represented by outcome-supervised algorithms like GRPO and DAPO. In these frameworks, the model explores long-chain reasoning paths entirely on its own, receiving a 0/1 reward from a verifier at the end.

**Limitations of Prior Work**: For tasks with vast search spaces like complex long-range mathematics, pure self-exploration means most rollouts fail to receive a positive reward. This leads to extremely sparse positive samples, high gradient estimation variance, and slow, unstable training. While test-time scaling solutions like Tree-of-Thoughts, Mixture-of-Agents, and LeaP compensate with additional compute during inference, they do not improve the policy's intrinsic capability and introduce significant communication and implementation overhead.

**Key Challenge**: There is a structural tension between "requiring external guidance during training" and "the necessity of independence during evaluation." Previous approaches either focused on complete independence (self-exploratory RL) or complete dependence (expert workflows, distillation), lacking a mechanism that dynamically leverages experts during training and detaches them for evaluation.

**Goal**: Design an RL framework that enables the policy to seek help from external experts on-demand during training to densify reward signals, while remaining capable of independent reasoning after training.

**Key Insight**: Model "seeking expert help" as a learnable discrete action $\alpha_t$ within the policy's action space, allowing the policy to decide if, when, and how to consult. Use an acceptance rate annealing $\rho_s = s^{-1}$ to gradually close this channel, forcing the model to internalize the strategies provided by expert hints into its own parameters.

**Core Idea**: Treat "consulting an expert" as a learnable action and use a step-wise decaying acceptance rate to gradually distill external information into the policy itself.

## Method

### Overall Architecture

EAPO unfolds a problem-solving session as a multi-turn trajectory $H_T = \{(\tau_t, \alpha_t, o_t)\}_{t=1}^T$. At each step, the policy first generates a private reasoning segment $\tau_t$, then decides on an action $\alpha_t$ (continue independent reasoning, consult an expert, or output the answer). If consultation is chosen, the environment returns an expert response $o_t$ to be appended to the context; otherwise, $o_t = \varnothing$. The trajectory probability is factorized as $\pi_\theta(\tau_t, \alpha_t \mid H_{t-1}) = \pi_\theta^\tau(\tau_t \mid H_{t-1}) \cdot \pi_\theta^\alpha(\alpha_t \mid H_{t-1}, \tau_t)$. Finally, end-to-end verifiable reward optimization is performed using $R$ (a combination of F1 and format rewards), with the objective $\max_\theta \mathcal{J}_{\text{EAPO}} = \mathbb{E}[R(\mathcal{E}(H_T), g)]$. During evaluation, $\alpha$ is strictly locked to "no consultation," forcing independent rollout.

For implementation, the policy backbone uses DeepSeek-R1-Distill-Qwen-7B, the expert pool uses QwQ-32B (heterogeneous and stronger), and the training data is DAPO-MATH.

### Key Designs

1.  **Learnable "Consult Expert" Action**:
    - **Function**: Elevates external assistance to a first-class citizen in the policy action space, where the policy jointly optimizes whether to call and what query to send.
    - **Mechanism**: The action space is expanded to include discrete choices like $\{\text{think}, \text{consult-expert}, \text{answer}\}$. Upon triggering a consultation, the policy generates a structured query $q_{t,i}$ for the expert; the expert returns $r_{t,i}$, which is integrated into the history. The entire trajectory receives gradients under a single RL objective, so "when to ask" and "what to ask" are shaped by reward signals. Three emergent rollout patterns were observed: self-resolution for simple problems, direct consultation (querying 3 experts for cross-reference) for hard problems, and decomposition (splitting sub-problems) for complex problems.
    - **Design Motivation**: Previous expert-assisted workflows hard-coded consultations in external pipelines, meaning the model never learned "when it should ask." Elevating this to an action allows the policy to **discriminate problem difficulty and allocate external aid rationally**, avoiding both blind dependence and isolated struggle.

2.  **Parallel Multi-Expert Querying and Alignment**:
    - **Function**: Queries up to $K$ experts simultaneously within a single turn to broaden information coverage and reduce fragile dependence on a single reasoning path.
    - **Mechanism**: At turn $t$, the policy selects a concurrency level $C_t \in [0, K]$, constructs $\mathcal{Q}_t = \{q_{t,i}\}_{i=1}^{C_t}$ for synchronous dispatch, and collects $o_t = \{r_{t,i}\}_{i=1}^{C_t}$ to perform "comparison + coordination" for the next context step. In experiments where $K=3$, the parallel mode surfaced more evidence than sequential consultation, solving instances that sequential modes failed due to turn budget exhaustion.
    - **Design Motivation**: Since turn budgets are finite in long-range reasoning, decoupling "information coverage" from "interaction rounds" allows for multi-perspective evidence in a single turn, improving robustness and success rates. Results show Parallel EAPO outperforms Sequential EAPO by 4 points on AIMO 2025.

3.  **Internalization via Acceptance Rate Annealing + Turn Budget Contraction**:
    - **Function**: Ensures the consultation channel is frequently open early in training and gradually closed later, distilling expert-acquired strategies into the model.
    - **Mechanism**: Each time an expert returns a response, it is accepted into the history with a probability $\rho_s = s^{-1}$ based on the global training step $s$. A sample $u \sim U(0,1)$ is drawn; if $u \le \rho_s$, it is accepted; otherwise, it is treated as unavailable, forcing the policy to continue independently. Simultaneously, the turn budget per episode is reduced from the initial training value toward the evaluation budget. This creates an implicit curriculum: early stages leverage dense expert hints to cross the "zero-reward" zone, mid-stages force independent trials when responses are occasionally withheld, and late stages see the expert channel closed, leaving the policy to rely on learned patterns.
    - **Design Motivation**: Simply adding experts creates path dependency where the model fails when experts are removed at deployment. Annealing ensures a smooth transition between "training with experts" and "testing without experts," which is key to EAPO's success in independent evaluation. It is essentially online knowledge distillation implemented via RL, targetting the teacher's "suggestions" at critical trajectory steps rather than just token distributions.

### Loss & Training

The reward function is piecewise: $R = \text{F1}(\hat{y}, g)$ if F1 is non-zero; $R = 0.1$ if F1 is 0 but format is correct; otherwise $R = 0$. The trajectory generation $p(o_{t+1} \mid \alpha_{t+1})$ is determined by the expert service and acceptance rate annealing. The optimization algorithm is based on online RL (a GRPO variant like DAPO), with implicit penalties on consultation actions (as decaying acceptance rates reduce their expected utility).

## Key Experimental Results

### Main Results

Policy 7B + Expert 32B, trained on DAPO-MATH. Evaluation on AIME 2024/2025 + AIMO 2025 using Pass@32 and Variance (Var):

| Method Category | Method | Avg Pass@32 ↑ | Avg Var ↓ | Remarks |
|---|---|---|---|---|
| Base | DeepSeek-R1-Distill-7B | 42.53 | 0.0947 | Baseline |
| Offline Workflow | Expert-Assisted Workflow | 49.39 | 0.2039 | 7B Policy + 3×32B Expert (hard-coded) |
| Offline Workflow | LeaP | 47.08 | 0.1652 | Parallel paths with mutual summarization |
| Distillation | LoRA Distill | 43.24 | 0.0969 | Distilled from 32B |
| Online RL | Self-Exploratory RL | 59.16 | 0.0727 | Pure outcome-driven RL |
| Online RL | **EAPO (Ours)** | **64.07** | **0.0643** | +4.91 vs Self-Exploratory RL |

EAPO achieved a Pass@32 of 64.17 on AIMO 2025, approximately a 9-point gain over self-exploratory RL, with lower variance across the board, indicating improved stability.

### Ablation Study

| Configuration | Avg Pass@32 | Avg Var | Description |
|---|---|---|---|
| Sequential EAPO + 32B Expert | 61.79 | 0.0721 | Single query per turn |
| Parallel EAPO + 14B Expert | 61.55 | 0.0692 | Weaker expert |
| **Parallel EAPO + 32B Expert** | **64.07** | **0.0643** | Full version |
| Homogeneous EAPO (7B Expert = Policy) | 58.85 | 0.0756 | Degenerates to self-exploration level |
| Heterogeneous EAPO (Llama-8B Expert) | 60.66 | 0.0727 | Requires complementary abilities |

### Key Findings

-   **Expert Capability Floor**: EAPO loses its gain when the expert is homogeneous with the policy (same 7B model) (58.85 vs 59.16), suggesting the gain stems from information injection of things the policy hasn't mastered. Complementarity is a prerequisite.
-   **Parallelism + Large Expert Synergy**: Parallel querying primarily improves exploration efficiency and robustness, while expert capacity determines the quality of injected information. Both are orthogonal and necessary.
-   **Cross-Domain Generalization**: Although trained on DAPO-MATH, EAPO outperforms the base and self-exploratory RL on HumanEval, HLE, GPQA, MMLU, EvalPlus, HotpotQA, and SimpleQA, suggesting it learns a general "consultation-internalization" mechanism.
-   **Diminishing Returns with Scale**: Moving from a 7B to a 14B policy yields overall gains but marginal returns decrease, as larger models can independently handle more long-tail cases in the dataset.

## Highlights & Insights

-   Upgrading "external model calls" from a pipeline engineering task to a policy action and optimizing it jointly is a natural but under-explored path. It allows RL algorithms to natively express metacognitive behaviors like "I don't know this, please advise," resulting in the spontaneous emergence of complex rollout strategies.
-   The use of **explicit temporal decay** in the acceptance rate $\rho_s = s^{-1}$ is simple yet effective. It functions as an annealing curriculum for online knowledge distillation, avoiding the loss of reasoning structure found in offline distillation and the fragility of hard KL constraints.
-   The causal chain of "Assistance reduces reward sparsity → Policy learns stronger patterns → Internalization allows independent execution" can be transferred to any agent task with long-range sparse rewards, such as learnable "request code review" actions in code generation or "query demonstration library" actions in robotics.

## Limitations & Future Work

-   The paper does not provide a compute budget for the training phase. Since each rollout can trigger up to three 32B model inferences (especially early in training), the H100 hour requirement likely far exceeds the self-exploration baseline, which affects fair comparison.
-   The "static expert" assumption: The authors do not discuss whether a co-drifting or collapsing scenario would occur if the expert were also an updatable LLM (self-teacher or synchronous training).
-   The annealing curve $\rho_s = s^{-1}$ is manually set; its optimality across different task difficulty distributions is unknown. A more principled approach would be to adaptively adjust based on the policy's own confidence.
-   Requirement of explicit heterogeneity: Homogeneous EAPO performs similarly to self-exploration, indicating the framework cannot create something from nothing via "self-talk." External information sources are a prerequisite, limiting applicability when no stronger expert is available.

## Related Work & Insights

-   **vs Self-Exploratory RL (DAPO / GRPO)**: Pure self-exploration has no external aid and sparse rewards. EAPO introduces an expert channel during training to densify rewards and reverts to self-exploration for testing, serving as a case of "training-evaluation asymmetry."
-   **vs Distillation (Full / LoRA)**: Distillation matches teacher token distributions offline and misses the meta-decision of "when to ask." EAPO distills teacher suggestions at critical steps and lets the policy decide the distillation granularity via RL.
-   **vs Test-Time Scaling (LeaP / ToT / MoA)**: These methods require multi-model collaboration at deployment, repeating costs for every query. EAPO shifts costs to the training phase as a one-time payment for a single-model deployment.
-   **vs Expert-Assisted Workflow**: Hard-coded multi-agent pipelines place "consultation" outside the policy; EAPO enables the model to learn adaptive resource allocation for external help.

## Rating

-   Novelty: ⭐⭐⭐⭐ The combination of "consult expert" as an action and annealing for internalization is rare and consistent in RL literature, though individual components are not entirely revolutionary.
-   Experimental Thoroughness: ⭐⭐⭐⭐ Covered math plus 7 cross-domain benchmarks, along with ablations on parallelism, expert scale, and homogeneity. Missing training compute comparison.
-   Writing Quality: ⭐⭐⭐⭐ Clean equations, clear motivation hierarchy, and insightful induction of the three emergent rollout modes.
-   Value: ⭐⭐⭐⭐ Provides a general template for leveraging stronger models during RL training while maintaining deployment independence, highly practical for industry scenarios with asymmetric resources.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Making Expert Reasoning Learnable with Self-Distillation](making_expert_reasoning_learnable_with_self-distillation.md)
- [\[ICML 2026\] Learning to Route Languages for Multilingual Policy Optimization](learning_to_route_languages_for_multilingual_policy_optimization.md)
- [\[ICML 2026\] Metis: Learning to Jailbreak LLMs via Self-Evolving Metacognitive Policy Optimization](metis_learning_to_jailbreak_llms_via_self-evolving_metacognitive_policy_optimiza.md)
- [\[ICML 2026\] Revisiting Regularized Policy Optimization for Stable and Efficient Reinforcement Learning in Two-Player Games](revisiting_regularized_policy_optimization_for_stable_and_efficient_reinforcemen.md)
- [\[ACL 2026\] Visually-Guided Policy Optimization for Multimodal Reasoning](../../ACL2026/reinforcement_learning/visually-guided_policy_optimization_for_multimodal_reasoning.md)

</div>

<!-- RELATED:END -->
