---
title: >-
  [Paper Note] Skill-Pro: Learning Reusable Skills from Experience via Non-Parametric PPO for LLM Agents
description: >-
  [ICML 2026][LLM Agent][Reusable skills] Skill-Pro explicitly extracts interaction experiences of LLM agents into "Activation + Execution + Termination" skill triplets. It employs semantic gradients to generate candidate…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "Reusable skills"
  - "Skill-MDP"
  - "Non-parametric PPO"
  - "Semantic gradient"
  - "Procedural memory"
date: 2026-05-08
content_hash: bfc184d416df4c88
---

# Skill-Pro: Learning Reusable Skills from Experience via Non-Parametric PPO for LLM Agents

**Conference**: ICML 2026  
**arXiv**: [2602.01869](https://arxiv.org/abs/2602.01869)  
**Code**: https://github.com/Miracle1207/Skill-Pro (Available)  
**Area**: LLM Agent / Procedural Memory / Non-Parametric Optimization  
**Keywords**: Reusable skills, Skill-MDP, Non-parametric PPO, Semantic gradient, Procedural memory

## TL;DR
Skill-Pro explicitly extracts interaction experiences of LLM agents into "Activation + Execution + Termination" skill triplets. It employs semantic gradients to generate candidate skills and utilizes a PPO-style trust region verification (PPO Gate) to determine their inclusion. Ultimately, it achieves a reuse rate of 0.85+ and significant performance improvements on ALFWorld / Mastermind using an extremely compact memory library of approximately 800 tokens.

## Background & Motivation

**Background**: Current LLM agents primarily rely on "on-the-fly reasoning" for sequential decision-making—re-executing prompt parsing, CoT, and ReAct for every task. Even in recurring similar scenarios, solutions are derived from scratch. To incorporate past experience, mainstream approaches follow two paths: parametric (fine-tuning via RL/RLHF/DPO) and non-parametric (external memory + retrieval augmentation).

**Limitations of Prior Work**: Parametric methods are expensive to train, prone to catastrophic forgetting, and risk losing general capabilities. While non-parametric methods are cost-effective, they currently focus almost entirely on **episodic memory**—storing past trajectories, reflections, graphs, or workflows as "historical records." During decision-making, agents must retrieve and re-reason through these records, remaining trapped in an inference-heavy cycle with high token consumption and low reliability.

**Key Challenge**: The human brain utilizes **procedural memory** in addition to episodic memory—once learned, it enables unconscious "situation $\rightarrow$ action" execution. Existing LLM agent memory systems **only cover the episodic layer and lack the procedural layer**. Storing experience is not equivalent to direct reuse; passive narratives must be transformed into executable programs while ensuring updates do not degrade the foundational LLM's general capabilities.

**Goal**: (C1) Ensure stored experiences are **executable** program units rather than narratives; (C2) Guarantee that new units are **reliably reusable** in future tasks with stable gains; (C3) Perform the entire optimization **without updating LLM parameters**.

**Key Insight**: Adapt the Reinforcement Learning paradigm of "small updates + trust regions" from PPO to the natural language level—replacing parameter updates with skill text updates, gradients with "semantic gradients" (natural language modification suggestions), and clipping with probability ratio cropping of the frozen LLM over historical trajectories.

**Core Idea**: Formalize agent procedural memory as a **Skill-MDP** (where a skill $\omega$ is defined by the triplet of activation condition $\mathcal{I}_\omega$, execution flow $\pi_\omega$, and termination condition $\beta_\omega$). Use **Non-Parametric PPO** (candidate generation via semantic gradients + verification via PPO Gate + online score-based pruning) to evolve the skill pool continuously without modifying LLM weights.

## Method

### Overall Architecture

Input: A fixed LLM policy $\pi_\text{LLM}$, an initial skill pool $\Omega_0$ (can be empty), and a batch of trajectories $\mathcal{T}^{(B)}$ from the environment. Output: An evolved skill pool $\Omega^*$. When making decisions with $\Omega^*$, the agent directly follows the sequence: "Select Skill $\rightarrow$ Execute atomic actions according to the Skill."

The pipeline consists of three phases:

1.  **Decision-Making Phase (Skill-MDP)**: At each timestep $t$, the skill-selection policy $\mu$ selects a skill $\omega_t$ from $\Omega$ based on the current state $s_t$ (via similarity matching or top-k value ranking). The frozen LLM then generates atomic actions $a_t \sim \pi_\text{LLM}(a \mid s_t, \omega_t)$ guided by the execution flow of $\omega_t$ until $\beta_{\omega_t}(s_t)=1$. The episode trajectory is collected into a buffer.

2.  **Evolution Phase (Non-Parametric PPO)**: Periodically, a batch is sampled from the buffer. For each invoked skill, a semantic gradient $g_i = \nabla_\text{sem}(\tau_i, \omega)$ is computed and aggregated across trajectories into $\bar{g}_\omega$. $N_c$ candidate skills are generated as $\omega' = \omega \oplus \bar{g}_\omega$. The PPO Gate calculates a clipped surrogate score for each candidate on historical trajectories; only the best candidate from $N_c$ with a score $>0$ replaces the old skill.

3.  **Maintenance Phase (Score-Based Maintenance)**: Each skill maintains an online score based on "cumulative advantage / invocation count." Skills with scores $\le 0$ or semantic redundancies are pruned. When the capacity is exceeded, skills are eliminated in ascending order of their scores.

The entire system **does not modify a single LLM parameter**; all "learning" is completed through the addition, deletion, and modification of skill text.

### Key Designs

1.  **Skill-MDP and Skill Triplet Structure**:
    *   **Function**: Reshapes "experience" from passive narrative into program units directly callable by the agent, addressing C1 (executability).
    *   **Mechanism**: Extends the classic MDP to $\mathcal{M}_\Omega = (\mathcal{S}, \mathcal{A}, \Omega, P, R, \gamma)$, where each skill is formalized as $\omega = \langle \mathcal{I}_\omega, \pi_\omega, \beta_\omega \rangle$. $\mathcal{I}_\omega$ represents natural language activation conditions (e.g., "At the start of the task, when no feedback is available"), $\pi_\omega$ represents ordered execution steps (e.g., "Step 1: Establish hypothesis space..."), and $\beta_\omega$ represents natural language termination conditions. The hierarchical policy decomposes as $\pi_\Omega(\omega_t, a_t \mid s_t) = \mu(\omega_t \mid s_t, \Omega)\,\pi_\text{LLM}(a_t \mid s_t, \omega_t)$, allowing the agent to perform "macro-action" scheduling over the state space while delegating atomic actions to the LLM.
    *   **Design Motivation**: Existing memory systems store trajectories/insights that require the LLM to "re-derive" actions during invocation. Explicit triplets provide skills with inherent trigger and exit logic, allowing direct routing by $\mu$ and amortizing the "re-derivation" cost into the training phase.

2.  **Non-Parametric PPO: Semantic Gradient + PPO Gate Trust Region Verification**:
    *   **Function**: Provides a skill update mechanism that is both directional (gradient-like) and safe (trust-region) without updating LLM weights, addressing C2 and C3.
    *   **Mechanism**: (a) **Semantic Gradient**—For each trajectory $\tau_i$ using $\omega$, the LLM performs hindsight attribution to output structured natural language modification suggestions $g_i = (g_i^{(\mathcal{I})}, g_i^{(\pi)}, g_i^{(\beta)})$. Common failure modes are extracted via aggregation to obtain $\bar{g}_\omega$. An LLM operator $\oplus$ applies $\bar{g}_\omega$ to the original skill to generate candidates $\omega' = \omega \oplus \bar{g}_\omega$. (b) **PPO Gate**—Treating the frozen LLM as a stochastic policy, the importance ratio $\rho_t(\omega') = \pi_\text{LLM}(a_t \mid s_t, \omega') / \pi_\text{LLM}(a_t \mid s_t, \omega)$ is calculated for each step. The advantage $\hat{A}_t = G_t - \bar{R}$ uses return-to-go minus a running baseline. Finally, the PPO-style clipped surrogate is computed: $L^\text{CLIP}(\omega') = \hat{\mathbb{E}}_\tau [\frac{1}{|\tau|}\sum_t \min(\rho_t \hat{A}_t,\ \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\hat{A}_t)]$. The candidate with the maximum $L^\text{CLIP} > 0$ is selected.
    *   **Design Motivation**: Relying solely on LLM "skill rewriting" often produces hallucinated or unstable variants. Unlike TextGrad, which optimizes static variables, sequential decision-making requires counterfactual evaluation of actual rewards. PPO's clipped objective naturally provides "step-wise, verifiable" update semantics—here mapped from parameter space to skill text space.

3.  **Online Advantage-Based Skill Pool Maintenance**:
    *   **Function**: Maintains a fixed capacity $K$, allowing "good" skills to persist while "poor" skills are automatically eliminated, preventing monotonic expansion.
    *   **Mechanism**: Advantages are defined as $\tilde{r}_t = r_t - \bar{r}$. The gain of a skill is defined as its average advantage during activation: $G(\omega; \tau) = \frac{1}{|\mathcal{T}_\omega(\tau)|}\sum_{t \in \mathcal{T}_\omega(\tau)} \tilde{r}_t$. Online cumulative gain $G_b$ and invocation counts $N_b$ are maintained to compute $\text{Score} = G_b / \max(1, N_b)$.
    *   **Design Motivation**: As the baseline $\bar{R}$ rises, "previously useful" skills naturally drop to zero and are eliminated, creating evolutionary pressure. The advantage form is more robust than "success rate" for handling sparse rewards.

### Loss & Training

The global optimization objective is $\max_\mathcal{E} \mathbb{E}_{\tau \sim \pi_{\Omega^*}}[\sum_t \gamma^t r_t]$, where $\Omega^* = \mathcal{E}^{(N)}(\Omega_0)$ is the steady-state skill pool after $N$ iterations of the evolution operator $\mathcal{E}$. Ours does not modify $\pi_\text{LLM}$ or $\mu$, only $\mathcal{E}$. The surrogate in the PPO Gate is consistent with standard PPO, with hyperparameters including clip threshold $\epsilon$, candidate count $N_c$, pool capacity $K$, and batch size $B$.

## Key Experimental Results

### Main Results

Benchmarks: ALFWorld (housekeeping tasks, Train/OOD splits) + Mastermind (code-breaking game, v0/Hard/Extreme difficulties). Backbone: Major proprietary LLM; Cross-agent evaluation: Gemma-3 4B / Qwen3 32B / Llama-3.3 70B. Baselines: RAG, Expel, A-MEM, AWM, G-Memory.

**Reuse Rate + Storage/Execution Cost**:

| Method | Mastermind-v0 Reuse ↑ | Cross-agent (Qwen3-32B) ↑ | Total Storage Tokens ↓ | Per-step Incremental Prompt Tokens ↓ |
|------|--------|--------|--------|--------|
| RAG | 0.349 | 0.146 | 116,527 | 2,698 |
| Expel | 0.285 | 0.270 | 294,447 | 5,210 |
| A-MEM | 0.020 | 0.018 | 200,129 | 1,214 |
| AWM | 0.080 | 0.060 | 391,706 | 3,658 |
| G-Memory | 0.091 | 0.264 | 40,510 | 434 |
| **Skill-Pro** | **0.925** | **0.875** | **816** | **273** |

Storage is **~50×** smaller than the next best baseline (G-Memory), with a reuse rate **~10×** higher.

**Performance (Success Rate / Average Reward)**:

| Task | State | CoT | ReAct | AWM | G-Memory | **Skill-Pro** |
|------|------|------|------|------|------|------|
| ALFWorld Train | 0.312 | 0.600 | 0.580 | 0.700 | 0.681 | **0.900** |
| ALFWorld OOD | 0.262 | 0.620 | 0.640 | 0.900 | 0.812 | **0.909** |
| Mastermind-v0 | 0.388 | 0.531 | 0.557 | 0.546 | 0.577 | **0.606** |
| Mastermind-Hard | 0.336 | 0.381 | 0.405 | 0.299 | 0.406 | **0.463** |
| Cross-agent Llama-3.3-70B | 0.613 | 0.542 | 0.604 | 0.550 | 0.535 | **0.647** |

### Ablation Study

| Configuration | Key Observation |
|------|------|
| Full Skill-Pro | Reuse rate 0.925 / Mastermind-v0 0.606 |
| w/o Semantic Gradient | Candidate skill quality drops; continuous improvement fails |
| w/o PPO Gate | Hallucinated skills enter the pool; unstable performance |
| w/o Online Score Pruning | Pool bloats, retrieval noise increases, long-term gains vanish |

### Key Findings

*   **Extreme Compression**: The counter-intuitive finding of "0.925 reuse with only 816 tokens" indicates that converting experience into **programs rather than narratives** increases information density by orders of magnitude.
*   **PPO Gate is Irreplaceable**: Removing trust region verification allows "better-looking but performance-degrading" candidates into the pool, confirming that semantic gradients carry hallucination risks.
*   **Cross-agent Transferability**: Skills learned via Gemma-3 4B transfer directly to Qwen3 32B / Llama-3.3 70B, suggesting that natural language skills achieve "protocol-level" readability across LLMs.

## Highlights & Insights

*   **Mapping PPO to Prompt/Skill Space**: Parameter update $\rightarrow$ Text update, Gradient $\rightarrow$ Semantic gradient, Importance ratio $\rightarrow$ Ratio of LLM probabilities for the same action, Clip $\rightarrow$ Trust-region text verification. This mapping is exceptionally clean, bridging RL and in-context learning.
*   **Executable vs. Narrative**: Episodic memory essentially treats the LLM as a "secondary reasoning engine," incurring unavoidable token costs. Procedural memory "solidifies" reasoning into skill text, shifting the cost model to simple matching and execution.
*   **General Framework**: This can be viewed as a general evolution framework for any prompt-as-program scenario by replacing skill triplets with tool-call schemes or task sub-flows.

## Limitations & Future Work

*   Skills are currently **explicitly readable**, which occupies the context window; future work could explore implicit/compressed procedural representations.
*   Both semantic gradients and the PPO Gate rely on the LLM's own attribution; the **base LLM's capability ceiling limits skill evolution**, as seen in the reduced gains on Mastermind-Extreme.
*   The router $\mu$ is relatively simplistic; as the skill pool grows, sophisticated RL-based retrievers may be necessary.

## Related Work & Insights

*   **vs. RAG / Expel / A-MEM**: These store episodic experiences requiring re-reasoning; Skill-Pro transforms them into executable programs, reducing storage by two orders of magnitude.
*   **vs. AWM (Agent Workflow Memory)**: AWM maintains explicit workflows but lacks an optimization mechanism; Skill-Pro's Non-Parametric PPO fills this gap.
*   **vs. TextGrad**: TextGrad optimizes static variables for single-step response quality; Skill-Pro targets sequential decision-making with counterfactual reward evaluation.

## Rating
*   Novelty: ⭐⭐⭐⭐⭐ 
*   Experimental Thoroughness: ⭐⭐⭐⭐ 
*   Writing Quality: ⭐⭐⭐⭐⭐ 
*   Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Lifting Traces to Logic: Programmatic Skill Induction with Neuro-Symbolic Learning for Long-Horizon Agentic Tasks](lifting_traces_to_logic_programmatic_skill_induction_with_neuro-symbolic_learnin.md)
- [\[ICML 2026\] EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle](evolver_self-evolving_llm_agents_through_an_experience-driven_lifecycle.md)
- [\[ICML 2026\] Internalizing Agency from Reflective Experience](internalizing_agency_from_reflective_experience.md)
- [\[ICML 2026\] On Information Self-Locking in Reinforcement Learning for Active Reasoning of LLM Agents](on_information_self-locking_in_reinforcement_learning_for_active_reasoning_of_ll.md)
- [\[ICML 2026\] PragLocker: Protecting Agent Intellectual Property in Untrusted Deployments via Non-Portable Prompts](praglocker_protecting_agent_intellectual_property_in_untrusted_deployments_via_n.md)

</div>

<!-- RELATED:END -->
