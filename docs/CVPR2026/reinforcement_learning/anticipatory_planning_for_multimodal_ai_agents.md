---
title: >-
  [Paper Note] Anticipatory Planning for Multimodal AI Agents
description: >-
  [CVPR 2026][Reinforcement Learning][Multimodal agents] This paper proposes TraceR1, a two-stage RL framework in which the first stage employs trajectory-level reward optimization to train agents to perform multi-step loo…
tags:
  - "CVPR 2026"
  - "Reinforcement Learning"
  - "Multimodal agents"
  - "anticipatory planning"
  - "trajectory-level reinforcement learning"
  - "GUI interaction"
  - "tool use"
  - "GRPO"
date: 2026-05-08
content_hash: c4e945121ce1a9f2
---

# Anticipatory Planning for Multimodal AI Agents

**Conference**: CVPR 2026
**arXiv**: [2603.16777](https://arxiv.org/abs/2603.16777)
**Code**: Not released
**Area**: Reinforcement Learning
**Keywords**: Multimodal agents, anticipatory planning, trajectory-level reinforcement learning, GUI interaction, tool use, GRPO

## TL;DR

This paper proposes TraceR1, a two-stage RL framework in which the first stage employs trajectory-level reward optimization to train agents to perform multi-step look-ahead planning, while the second stage applies grounded fine-tuning via tool execution feedback to improve single-step precision. The approach achieves open-source state-of-the-art results across 7 GUI and tool-use benchmarks.

## Background & Motivation

**Background**: Multimodal agents have made significant progress in GUI interaction and tool invocation, yet the vast majority of existing systems remain fundamentally **reactive** — selecting the next action based solely on the current observation without considering long-term consequences.

**Limitations of Prior Work**: In multi-step tasks, the effects of actions are often delayed and cumulative. Reactive agents cannot anticipate downstream consequences, leading to progressive goal deviation and poor planning coherence in long-horizon tasks.

**Key Challenge**: Both mainstream technical approaches face fundamental obstacles — model-free RL relies on sparse terminal rewards and struggles to capture long-range dependencies, while model-based planning requires constructing a world model, which is exceedingly difficult in visually rich interactive environments.

**Goal**: To efficiently train multimodal agents capable of adaptive anticipatory reasoning, enabling consistent planning in complex long-horizon tasks.

**Key Insight**: Rather than building an explicit world model, the paper directly applies RL at the trajectory level, training the model to predict a sequence of future actions while executing only the first step — analogous to the human strategy of "thinking several steps ahead, acting one step at a time."

**Core Idea**: A two-stage training procedure — trajectory-level alignment for global consistency followed by grounded RL for single-step executability — unifies anticipatory planning and precise execution.

## Method

### Overall Architecture

TraceR1 adopts a plan-act loop: given the current observation, the model predicts a multi-step future trajectory $\hat{\tau}_{t:T}$ but executes only the first action, then replans upon receiving environmental feedback. Training proceeds in two stages:

- **Stage 1 (Anticipatory Trajectory Optimization)**: Trajectory-level RL with a global alignment reward encouraging coherent multi-step planning.
- **Stage 2 (Grounded Reinforcement Fine-tuning)**: Step-level RL using execution feedback from a frozen tool agent to improve single-step precision.

The backbone model is Qwen3-VL-8B-Thinking, trained with the EasyR1 framework.

### Key Design 1: Trajectory-Level Alignment Reward

- **Function**: Given the user instruction, current observation, and interaction history, the model predicts an action sequence over $T$ future steps, which is aligned against a reference trajectory.
- **Mechanism**: A discounted trajectory reward is defined as $R(\hat{\tau}, \tau^*) = \sum_{t=1}^{T} \gamma^{t-1} r_t$, where $r_t = \lambda_{\text{align}} \cdot \text{sim}(\hat{a}_t, a_t^*) - \lambda_{\text{rep}} \cdot \text{rep}(\hat{a}_{1:t})$. The term $\text{sim}$ measures action alignment and $\text{rep}$ penalizes repetitive or cyclic actions.
- **Design Motivation**: SFT under teacher forcing optimizes token-by-token and neglects global coherence. Trajectory-level RL enables the model to learn cross-step dependencies and avoid redundant or unstable rollouts.

### Key Design 2: Repetition Penalty and Temporal Discount

- **Function**: $\lambda_{\text{rep}}$ penalizes repeated actions within a trajectory; $\gamma$ serves as a temporal discount factor that biases the model toward near-term correctness.
- **Mechanism**: These components jointly prevent reward hacking — without repetition penalty, the planner may click the same UI element or invoke the same tool repeatedly to inflate rewards; $\gamma < 1$ prevents overfitting to highly uncertain long-range predictions.
- **Design Motivation**: Ablation studies confirm that removing either component leads to significant performance degradation (see Ablation Study section).

### Key Design 3: Grounded RL Fine-tuning

- **Function**: The model outputs $(\hat{a}_t, \hat{g}_t)$, which are passed to a frozen tool agent (e.g., UI-TARS-7B) for execution; the execution result is compared against the ground truth to derive a step-level reward.
- **Mechanism**: For GUI tasks, a coordinate matching reward is used; for tool-use tasks, an answer matching reward is applied: $r_t^G = \mathbb{1}[\text{coord match}]$ or $\mathbb{1}[\text{answer match}]$.
- **Design Motivation**: The trajectory reward in Stage 1 is abstract and provides no signal regarding whether predicted actions are actually executable. Stage 2 supplies concrete execution outcomes as corrective signals, compensating for the "idealized planning" problem.

### Key Design 4: Plan-Act Loop at Inference

- **Function**: At inference time, the model predicts a multi-step trajectory but executes only the first action, then replans upon observing the updated state.
- **Mechanism**: This is analogous to Model Predictive Control (MPC): rolling prediction, single-step execution, and continuous correction.
- **Design Motivation**: Multi-step prediction provides anticipatory context, yet since the environment changes continuously, executing one step and replanning balances look-ahead capability with robustness.

## Loss & Training

Both stages employ **GRPO (Group-Relative Policy Optimization)** as the optimization objective:

- Stage 1: $\nabla_\theta J(\theta) = \mathbb{E}_{\hat{\tau}}[\hat{A}(\hat{\tau}, \tau^*) \nabla_\theta \log \pi_\theta(\hat{\tau} | u, s_t, \tau_{1:t-1})]$, where $\hat{A}$ is the normalized group-relative advantage derived from trajectory rewards.
- Stage 2: The trajectory reward is replaced by the grounded step reward $r_t^G$; GRPO updates are applied identically.

Training data for GUI tasks includes trajectory datasets from AgentNet, AndroidControl, GUI-Odyssey, Multimodal-Mind2Web, and AgentTrek; tool-use tasks use trajectory data from T3-Agent together with an executable toolbox.

## Key Experimental Results

### Main Results: Online GUI Benchmarks (Table 1 — Success Rate %)

| Model | Params | AndroidWorld | OSWorld-Verified |
|-------|--------|:---:|:---:|
| OpenAI CUA-o3 | — | 52.5 | 38.1 |
| UI-TARS-2 | — | 73.3 | 53.1 |
| Claude 4.5 Sonnet | — | — | 62.9 |
| Agent S2.5 w/ o3 | 7B w/ — | — | 56.0 |
| Qwen3-VL-32B-Thinking | 32B | 61.4 | 35.6 |
| **TraceR1 (Qwen3-VL-32B w/ Ours)** | **32B w/ 8B** | **64.8** | **41.2** |

**Key Takeaway**: TraceR1 improves the OSWorld success rate of Qwen3-VL-32B-Thinking from 35.6% to 41.2% (relative gain of 15.7%) and AndroidWorld from 61.4% to 64.8%, establishing open-source state-of-the-art performance.

### Tool-Use Benchmarks (Table 3 — GAIA & GTA)

| Model | Params | GAIA AnsAcc | GTA AnsAcc | GTA ToolAcc | GTA CodeExec |
|-------|--------|:---:|:---:|:---:|:---:|
| GPT-4o | — | 33.4 | 57.1 | 63.4 | 95.1 |
| GPT-5 | — | 59.3 | 60.9 | 68.3 | 98.7 |
| Qwen3-VL-8B | 8B | 31.5 | 49.2 | 56.8 | 74.2 |
| T3-Agent | 7B | 16.9 | 53.8 | 64.6 | 84.3 |
| **TraceR1** | **8B** | **40.2** | **56.7** | **65.7** | **87.4** |

**Key Takeaway**: At the 8B scale, TraceR1 surpasses GPT-4o on GAIA (40.2 vs. 33.4) and outperforms Qwen3-VL-8B of the same scale by +8.7 AnsAcc.

### Ablation Study

| Configuration | AndroidWorld | OSWorld-Verified | GTA |
|---------------|:---:|:---:|:---:|
| Full TraceR1 (w/ Stage 2) | 64.8 | 41.2 | 56.7 |
| w/o Stage 2 | 57.2 | 36.3 | 50.2 |

Removing Stage 2 causes an average drop of approximately 6%, demonstrating that grounded execution feedback is critical for planning stability.

Additional ablation findings:
- **Prediction horizon $T$**: Performance peaks at $T \approx 10$; larger values cause accumulated uncertainty and performance degradation.
- **$\lambda_{\text{rep}} = 0$**: Removing the repetition penalty induces reward hacking (e.g., repeatedly clicking the same element).
- **$\gamma = 1$**: Removing the temporal discount causes the model to overfit to highly uncertain long-range predictions.

## Highlights & Insights

1. **The "think several steps, act one step" paradigm is elegant and concise**: No explicit world model is required; trajectory-level RL directly instills anticipatory reasoning, making the approach far more tractable than model-based planning.
2. **The two-stage decoupled design is well-motivated**: Stage 1 addresses "seeing far ahead" (global consistency), while Stage 2 addresses "acting accurately" (execution feasibility), with a clear division of responsibilities.
3. **Strong generality**: The same framework applies to both GUI interaction (desktop and mobile) and general tool invocation, with comprehensive validation across 7 benchmarks.
4. **Open-source 8B model surpasses GPT-4o**: TraceR1 at 8B outperforms GPT-4o on GAIA, offering exceptional cost-effectiveness.
5. **Thorough ablations on repetition penalty and temporal discount**: The experiments clearly expose the reward hacking problem and its resolution.

## Limitations & Future Work

1. **Limitations of short-horizon updates**: The current method provides only local corrections and cannot reshape the agent's understanding of long-term feasibility and task structure. Future work may explore multi-round or hierarchical planning mechanisms.
2. **Stage 2 depends on a frozen tool agent**: The quality of the tool agent directly affects the reliability of grounded rewards; errors in the tool agent introduce noise into the corrective signal.
3. **Offline training vs. online interaction**: The current grounded setup is offline and does not involve real online environment interaction, potentially limiting adaptability to dynamic environment changes.
4. **Sensitivity to prediction horizon**: Performance degrades when $T > 10$, indicating that the method still has bottlenecks on very long-horizon tasks.
5. **Absence of memory and state update mechanisms**: The current framework lacks cross-episode memory integration and cannot learn from historical failures.

## Related Work & Insights

| Compared Method | Key Differences |
|-----------------|-----------------|
| **GUI-R1 / InfiGUI-R1** | Both adopt R1-style RL training but apply only step-level rewards, lacking trajectory-level global optimization. TraceR1 surpasses them by over 40% on AndroidControl-High, validating the necessity of trajectory-level reasoning. |
| **Agent S2 / GTA1** | Rely on closed-source models (o3/GPT-5) as planners with open-source small models for execution. TraceR1 does not depend on closed-source planners and instead directly trains the intrinsic planning capabilities of open-source models, yielding greater autonomy. |
| **UI-TARS-1.5/2** | Commercial closed-source systems with strong performance but no reproducibility. TraceR1 with an 8B open-source model paired with a 32B executor approaches the performance level of UI-TARS-1.5. |

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The two-stage design combining trajectory-level RL and grounded fine-tuning represents a meaningful advance over existing R1-style methods; the MPC-inspired "predict multiple steps, execute one" strategy is relatively novel in the context of multimodal agent training.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Seven benchmarks spanning online/offline GUI and tool-use settings; comprehensive ablations covering Stage 2, prediction horizon, repetition penalty, and temporal discount; results reported as the mean of three independent runs.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, well-articulated motivation, rigorous mathematical notation, and rich figures and tables; related work is thoroughly categorized.
- **Value**: ⭐⭐⭐⭐ — Provides a general and practical training paradigm for anticipatory planning in multimodal agents; the result of an 8B model surpassing GPT-4o carries strong practical significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Controlling Multimodal Conversational Agents with Coverage-Enhanced Latent Actions](../../ACL2026/reinforcement_learning/controlling_multimodal_conversational_agents_with_coverage-enhanced_latent_actio.md)
- [\[CVPR 2026\] RoboAgent: Chaining Basic Capabilities for Embodied Task Planning](roboagent_chaining_basic_capabilities_for_embodied_task_planning.md)
- [\[NeurIPS 2025\] Deep RL Needs Deep Behavior Analysis: Exploring Implicit Planning by Model-Free Agents](../../NeurIPS2025/reinforcement_learning/deep_rl_needs_deep_behavior_analysis_exploring_implicit_planning_by_model-free_a.md)
- [\[CVPR 2026\] BRIDGE: Multimodal-to-Text Retrieval via Reinforcement-Learned Query Alignment](bridge_multimodal-to-text_retrieval_via_reinforcement-learned_query_alignment.md)
- [\[CVPR 2026\] Lifelong Imitation Learning with Multimodal Latent Replay and Incremental Adjustment](lifelong_imitation_learning_multimodal_latent_rep.md)

</div>

<!-- RELATED:END -->
