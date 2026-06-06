---
title: >-
  [Paper Note] On Information Self-Locking in Reinforcement Learning for Active Reasoning of LLM Agents
description: >-
  [ICML 2026][LLM Agent][Active Reasoning] Focusing on the failure mode in multi-round active reasoning where "Action Selection (AS)" and "Belief Tracking (BT)" hinder each other—causing outcome-only RL to fall into low-in…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "Active Reasoning"
  - "Agentic RL"
  - "Credit Assignment"
  - "Self-Locking Failure"
  - "advantage reweighting"
date: 2026-05-08
content_hash: d678f5b628638dad
---

# On Information Self-Locking in Reinforcement Learning for Active Reasoning of LLM Agents

**Conference**: ICML 2026  
**arXiv**: [2603.12109](https://arxiv.org/abs/2603.12109)  
**Code**: https://github.com/unimpor/T3 (Available)  
**Area**: LLM Reasoning / Agent / Reinforcement Learning  
**Keywords**: Active Reasoning, Agentic RL, Credit Assignment, Self-Locking Failure, advantage reweighting

## TL;DR
Focusing on the failure mode in multi-round active reasoning where "Action Selection (AS)" and "Belief Tracking (BT)" hinder each other—causing outcome-only RL to fall into low-information Information Self-Locking (SeL)—this paper provides a formal definition of the "Self-Locking region" and a coupled gradient analysis from a POMDP perspective. It proposes AReW: using directional critiques obtained from the environment or readout layers to perform additive reweighting on stepwise advantages, achieving up to a 60-point performance gain across 9 active reasoning tasks.

## Background & Motivation

**Background**: Outcome-based RL (PPO / GRPO / GSPO) has become the de facto standard for training LLM agents in tasks requiring iterative "interaction and reasoning," such as deep research, coding, tool usage, and multi-round medical consultations. These tasks share a common trait: initial information is incomplete, necessitating that the agent actively query the environment or user to collect evidence and update internal judgments of the task state.

**Limitations of Prior Work**: The authors observed a counter-intuitive phenomenon on benchmarks like PE (Preference Estimation), MediQ (Medical Diagnosis), and FloDial (Troubleshooting): while reward curves increase, the "information per round (AS proxy)" and "belief gain per round (BT proxy)" remain stagnant or even regress. Instead of becoming better at questioning or evidence integration, agents learn shortcuts—using heuristics or priors to guess answers while bypassing interaction. On MediQ, after replacing all patient feedback with "Unknown," the performance of un-RL-trained models dropped by 10.75 (41.25 → 30.50), while RL-trained models only dropped by 5.50 (61.00 → 55.50), with belief consistency rising from 78.7 to 92.8. This indicates that RL makes agents "more stubborn and less dependent on interaction."

**Key Challenge**: Agent capabilities can be decoupled into AS (deciding what observations to obtain) and BT (deciding if observations can be assimilated). However, these are bi-directionally coupled under outcome rewards: weak BT prevents informative actions from receiving credit (hindering AS learning), while weak AS denies BT useful evidence (hindering BT learning). When these two channels lock, they form an attractor of low information and low BT, defined by the authors as Information Self-Locking (SeL).

**Goal**: (i) Formalize what SeL is and why it exists stably under outcome RL; (ii) Provide a lightweight solution that breaks SeL without introducing calibrated intermediate rewards or training dense reward models.

**Key Insight**: In many active reasoning scenarios, determining "whether this action elicited new information" and "whether this belief update moved toward the ground truth" is much easier than determining "the exact reward of this step." The former only requires checking if environment feedback contains new evidence or if self-reported confidence increased—qualitative directional critiques that can be provided by rules or cheap LLM annotations without calibration.

**Core Idea**: Translate these $\pm 1/0$ directional critiques into a likelihood-margin auxiliary objective. Its gradient is equivalent to adding a constant bias to the standard policy gradient advantage: $\hat{A}_t \leftarrow A_t + \lambda u_t$. This moves credit from negatively critiqued steps to positively critiqued ones internally within a trajectory, keeping the outcome reward and RL backbone unchanged.

## Method

### Overall Architecture
The authors model active reasoning as a POMDP $(\mathcal{S},\mathcal{A},\mathcal{O},T,O,R,\gamma)$, where the latent state $s^\star$ represents hidden user preferences, diagnoses, or solutions. Agent behavior is decomposed into two interleaved kernels:

-   **AS kernel** $\pi_\omega^{\mathrm{as}}(a_t\mid b_t)$: Selects environment actions based on current internal belief $b_t$, determining the next observation $o_t \sim O(\cdot\mid s^\star, a_t)$;
-   **BT kernel** $\pi_\omega^{\mathrm{bt}}(b_{t+1}\mid b_t, a_t, o_t)$: Integrates new observations into the belief.

Trajectories are alternating sequences of (AS round, Update round). Sections 2 and 3 formalize SeL using oracle-belief trajectories and Bayesian updates. They prove that within the SeL region, outcome-gradient signals for improving AS/BT decay at a first-order scale of $\eta \cdot (\text{current capability})$ (Thm. 3.4), making it difficult to escape SeL spontaneously. Section 4 introduces AReW for advantage reweighting using directional critiques to boost this first-order term.

### Key Designs

1.  **SeL Formalization and Coupled Gradient Decomposition**:
    -   **Function**: Provides a falsifiable explanation for why outcome RL fails in active reasoning and defines the SeL region.
    -   **Mechanism**: Defines belief potential $\Psi(b) := b(s^\star)$. AS-specific information capability $I_{\mathrm{AS}}(\omega) := \mathbb{E}[\Psi(\bar b_H) - \Psi(\bar b_0)]$ is isolated using oracle-belief trajectories $\bar\tau \sim (\pi_\omega^{\mathrm{as}}, \mathsf{BayesUpd})$. BT capacity $C_{\mathrm{BT}}(\omega)$ is defined by the sum of "absorbed positive components" $\Delta\Psi_t^+ = \max(0, \Psi(b_{t+1})-\Psi(b_t))$ from on-policy trajectories. The SeL region is $\mathcal{R}_{\delta,\varepsilon} := \{\omega: I_{\mathrm{AS}}\le\delta,\, C_{\mathrm{BT}}\le\varepsilon\}$. Decomposing $\nabla_\omega \log p_\omega(\tau)$ into AS-channel and BT-channel gradients $g_{\mathrm{as}}, g_{\mathrm{bt}}$, Thm. 3.4 proves that in the SeL region, one-sided drifts $( \Delta_{\mathrm{as}}^+ I_{\mathrm{AS}},\, \Delta_{\mathrm{bt}}^+ C_{\mathrm{BT}} )^\top \preceq \eta M (I_{\mathrm{AS}}, C_{\mathrm{BT}})^\top + o(\eta)$. Since non-zero terms in $M$ are multiplied by the other channel's capability, AS cannot learn when BT is weak, and vice versa.
    -   **Design Motivation**: Prior explanations for LLM agent failure often stopped at "sparse rewards." This work pinpoints the cause to "AS-BT bi-directional credit masking," explaining why rewards rise while capabilities do not, and provides a direct intervention interface.

2.  **AS / BT Directional Critique**:
    -   **Function**: Provides cheap directional signals $z_t \in \{-1, 0, +1\}$ for each step without constructing calibrated dense rewards.
    -   **Mechanism**: **AS critique** $z_t^{\mathrm{as}}$ is derived from "whether the action obtained new evidence," determined by rules or an LLM-judge (e.g., non-dominance of attributes in PE, non-empty new facts in MediQ). **BT critique** $z_t^{\mathrm{bt}} := \mathrm{Sign}(\hat\Psi_{t+1} - \hat\Psi_t)$, where $\hat\Psi_t$ is the confidence toward the ground-truth candidate reported by the agent itself. Prop. 4.1 proves the gain from AReW is $I_{\mathrm{AS}}(\hat{\mathcal{T}}_{\mathrm{as}}) - I_{\mathrm{AS}}(\mathcal{T}_{\mathrm{as}}) = \eta W(\omega) (2\,\mathrm{Acc}_{\mathrm{as}} - 1)$, requiring only weighted accuracy $> 1/2$ for positive gains.
    -   **Design Motivation**: Calibrated step rewards are nearly unobtainable in long-horizon agentic tasks (requiring biased reward models or intensive annotation). Directional critiques only require "good/bad/unknown" judgments, making them robust to noise and aligned with SeL channels.

3.  **Likelihood-margin Objective → Advantage Reweighting**:
    -   **Function**: Injects directional critiques into policy gradients while ensuring (i) local action on critiqued steps and (ii) no changes to outcome rewards or RL optimization mechanisms.
    -   **Mechanism**: For a trajectory $\tau$, let $\mathcal{P}_\tau := \{t: z_t=+1\}$ and $\mathcal{N}_\tau := \{t: z_t=-1\}$. An intra-trajectory likelihood-margin objective is defined as $\hat{\mathcal{L}}(\omega;\tau) := \frac{1}{|\mathcal{P}_\tau|}\sum_{t\in\mathcal{P}_\tau}\log\pi_{\omega,t} - \frac{1}{|\mathcal{N}_\tau|}\sum_{t\in\mathcal{N}_\tau}\log\pi_{\omega,t}$. Its gradient reflects $\sum_t u_t \nabla_\omega \log\pi_{\omega,t}$, where $u_t = +1/|\mathcal{P}_\tau|, -1/|\mathcal{N}_\tau|, 0$. Due to the centering property ($\sum_t u_t = 0$), it introduces no mean drift. The final augmented surrogate $\hat{\mathcal{L}}_{\mathrm{aug}} := \mathcal{J}_{\mathrm{RL}} + \lambda \mathbb{E}_\tau[\hat{\mathcal{L}}]$ is equivalent to modifying the advantage to $\hat A_t := A_t + \lambda u_t$.
    -   **Design Motivation**: Using additive advantage shaping preserves unbiased outcome reward estimation while shifting credit from negative to positive steps, breaking the drift suppression chain described in Thm. 3.4.

### Loss & Training
The final training objective is $\hat{\mathcal{L}}_{\mathrm{aug}}(\omega) = \mathcal{J}_{\mathrm{RL}}(\omega) + \lambda\,\mathbb{E}_\tau[\hat{\mathcal{L}}(\omega;\tau)]$. In practice, this involves replacing the advantage per step with $A_t + \lambda u_t$. AS critiques are generated by rules or feedback parsers. BT critiques use agent-reported confidence $\hat\Psi_t$ for $s^\star$ candidates in each Update Round, taking the sign of the difference between consecutive rounds. $\lambda$ is scaled to match the sparse reward magnitude.

## Key Experimental Results

### Main Results
9 active reasoning tasks across 4 domains (Preference Estimation, Medical Diagnosis, Troubleshooting, $\tau^2$-bench) using Qwen-2.5-7B-Instruct and LLaMA-3.1-8B-Instruct. Below are core results for Qwen-2.5-7B-Instruct + PPO (average outcome reward on test set):

| Task | o4-mini (Direct) | Vanilla PPO | AReW–AS only | AReW–AS+BT | Gain (AS+BT vs Vanilla) |
|------|----------------|-------------|--------------|------------|-------------------|
| PE-G S=2 | 17.11 | 24.00 | 46.00 | 49.33 | +25.3 |
| PE-G S=3 | 21.15 | 18.33 | 32.00 | **80.33** | **+62.0** |
| PE-F D=8 | 8.42 | 30.52 | 39.62 | 47.89 | +17.4 |
| PE-F D=6 | 12.47 | 32.03 | 42.10 | 44.47 | +12.4 |
| MediQ | 74.67 | 50.50 | 57.25 | 61.25 | +10.8 |
| FloDial-Easy | 35.00 | 37.33 | 43.67 | 41.00 | +3.7 |
| FloDial-Hard | 26.33 | 21.33 | 36.00 | 42.33 | +21.0 |

LLaMA-3.1-8B-Instruct showed identical trends (PE-G S=3 jumped from 11.00 to 77.67). AReW outperformed in 27 out of 28 (task, RL algo) configurations. AS+BT significantly outperformed AS-only in 11 out of 14 cases, validating concurrent intervention.

### Ablation Study

| Config | PE-G S=3 | FloDial-Hard | Description |
|------|----------|--------------|------|
| Vanilla PPO | 18.3 | 21.3 | Outcome-only baseline |
| AReW (α=0, Clean critique) | 80.3 | 36.0 | Upper bound |
| AReW (α=0.1, 10% Flip) | 40.0 | 30.3 | Still beats vanilla |
| AReW (α=0.2) | 65.0 | 29.0 | Still beats vanilla |
| AReW (α=0.3) | 31.3 | 27.6 | Still beats vanilla |
| AReW (α=0.4) | 22.3 | 30.6 | Near threshold |
| AReW (α=0.5) | 30.3 | 23.3 | Random critique decays to baseline |

This aligns with Prop. 4.1's $2\,\mathrm{Acc} - 1$ scaling: performance only collapses when weighted accuracy drops to 0.5.

### Key Findings
- **Coupling is Real**: In PE-G/MediQ, fixing action sequences and only replacing BT (using rules or frontier models for belief updates) significantly increases the correlation between AS and reward, proving weak BT masks AS contributions.
- **AS-only Boosts BT**: Pure AS critique not only raises AS proxies but also BT proxies, indicating that AReW indirectly helps BT learning by increasing information flow.
- **RL Algorithm Agnostic**: AReW remains effective with GRPO/GSPO; group-based rollouts alone do not solve SeL, as the coupling is structural rather than a sampling variance issue.
- **Insights from $\tau^2$-bench**: Vanilla PPO training led to spikes in tool execution errors and token usage. AReW restored interaction rhythm and significantly reduced error rates.

## Highlights & Insights
-   **Precise Failure Naming**: Positioning "reward gain without capability gain" as AS-BT bi-directional credit masking (SeL) rather than just "sparse rewards" provides a clear diagnostic framework.
-   **Directional Critique over Calibrated Reward**: Prop. 4.1 shows that as long as critique accuracy $>1/2$, gains are positive. This suggests that for long-horizon agentic RL, seeking binary "good/bad" signals is more practical than pursuit of calibrated process reward models.
-   **Zero-Intrusion Implementation**: Implementation involves a simple constant bias in advantages. The centering property ensures no need to modify critics, KL terms, or clipping—making it a highly portable plug-in for PPO/GRPO/GSPO.

## Limitations & Future Work
-   **Dependency on Feedback-Rich Environments**: Relies on signals like "new evidence elicited" or "confidence change," which may not exist in environments without feedback (e.g., pure text math proofs).
-   **BT Critique Reliance on Self-Reporting**: If models are severely mis-calibrated, the sign of $z_t^{\mathrm{bt}}$ may be distorted. Robustness was shown, but behavior under extreme mis-calibration requires more study.
-   **Theoretical Assumptions**: Thm. 3.4 relies on several regularity conditions (Lipschitz coupling, action-invariant drift) that might not be strictly met in real LLMs.
-   **Scalability**: Future directions include extending SeL to multi-agent collaboration, generating directional critiques online via LLM-judges, and complementing process reward models.

## Related Work & Insights
-   **vs Process Reward Model (PRM)**: While PRMs seek step-level calibration, AReW requires only directional signals. They are complementary; AReW can serve as a PRM-free alternative or addition.
-   **vs Vanilla GRPO/GSPO**: Group sampling reduces variance but does not solve structural SeL coupling, explaining why GRPO gains are less pronounced in agentic tasks than in math.
-   **vs Information-Seeking RL**: Unlike prior work adding auxiliary rewards, AReW explains why such rewards are often unstable from the perspective of gradient suppression between channels.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Precisely names and formalizes a key failure mode; advantage shaping is highly transferable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive coverage across 9 tasks and 3 algorithms, though verification on larger scales (30B+) is missing.
- Writing Quality: ⭐⭐⭐⭐ Clear narrative flow; theorems align well with experiments, though assumptions are numerous.
- Value: ⭐⭐⭐⭐⭐ Directional critiques + advantage reweighting has immediate engineering significance for training research and coding agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Reducing Belief Deviation in Reinforcement Learning for Active Reasoning of LLM Agents](../../ICLR2026/llm_agent/reducing_belief_deviation_in_reinforcement_learning_for_active_reasoning.md)
- [\[AAAI 2026\] MoralReason: Generalizable Moral Decision Alignment For LLM Agents Using Reasoning-Level Reinforcement Learning](../../AAAI2026/llm_agent/moralreason_generalizable_moral_decision_alignment_for_llm_agents_using_reasonin.md)
- [\[ICML 2026\] EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle](evolver_self-evolving_llm_agents_through_an_experience-driven_lifecycle.md)
- [\[ICML 2026\] Towards Feedback-to-Plan Decisions for Self-Evolving LLM Agents in CUDA Kernel Generation](towards_feedback-to-plan_decisions_for_self-evolving_llm_agents_in_cuda_kernel_g.md)
- [\[ICML 2026\] Talk, Judge, Cooperate: Gossip-Driven Indirect Reciprocity in Self-Interested LLM Agents](talk_judge_cooperate_gossip-driven_indirect_reciprocity_in_self-interested_llm_a.md)

</div>

<!-- RELATED:END -->
