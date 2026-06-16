---
title: >-
  [Paper Note] On Information Self-Locking in Reinforcement Learning for Active Reasoning of LLM Agents
description: >-
  [ICML 2026][LLM Agent][Agentic RL] Addressing the failure mode where "Action Selection (AS)" and "Belief Tracking (BT)" hinder each other in multi-turn active reasoning for LLM agents—causing outcome-only RL to fall into Low Information Self-Locking (SeL)—this paper provides a coupled gradient analysis and formal definition of the "self-locking region"
tags:
  - ICML 2026
  - LLM Agent
  - Agentic RL
  - advantage reweighting
date: 2026-05-08
content_hash: dd4c31055c849812
---
# On Information Self-Locking in Reinforcement Learning for Active Reasoning of LLM Agents

**Conference**: ICML 2026  
**arXiv**: [2603.12109](https://arxiv.org/abs/2603.12109)  
**Code**: https://github.com/unimpor/T3 (Available)  
**Area**: LLM Reasoning / Agent / Reinforcement Learning  
**Keywords**: Active Reasoning, Agentic RL, Credit Assignment, Self-Locking Failure, Advantage Reweighting

## TL;DR
Addressing the failure mode where "Action Selection (AS)" and "Belief Tracking (BT)" hinder each other in multi-turn active reasoning for LLM agents—causing outcome-only RL to fall into Low Information Self-Locking (SeL)—this paper provides a coupled gradient analysis and formal definition of the "self-locking region" from a POMDP perspective. It proposes AReW: using directional critiques (obtainable from environments or readout layers) to perform additive reweighting on stepwise advantages, yielding up to a 60-point performance gain across 9 active reasoning tasks.

## Background & Motivation

**Background**: Outcome-based RL (PPO / GRPO / GSPO) has become the de facto standard for training LLM agents, utilized in tasks requiring "interleaved interaction and reasoning" such as deep research, coding, tool usage, and multi-turn medical consultation. A common feature of these tasks is incomplete initial information; the agent must actively query the environment or user to collect evidence and update internal judgments of the task state.

**Limitations of Prior Work**: The authors observed an anti-intuitive phenomenon on benchmarks such as PE (Preference Estimation), MediQ (Medical Diagnosis), and FloDial (Troubleshooting): while reward curves indeed rise, the "information volume per round (AS proxy)" and "belief gain per round (BT proxy)" remain stagnant or even regress. Instead of learning to "ask better questions and integrate evidence," agents learn shortcuts like using heuristics or priors to output answers directly while bypassing interaction. On MediQ, after replacing all patient feedback with "Unknown," the performance of the non-RL model dropped by 10.75 (41.25→30.50), whereas the RL-trained model dropped by only 5.50 (61.00→55.50), while belief consistency rose from 78.7 to 92.8—indicating that RL actually made the agent "more stubborn and less dependent on interaction."

**Key Challenge**: Agent capabilities can be decoupled into AS (deciding what observations to get) and BT (deciding whether observations can be absorbed). However, under outcome rewards, these two are bidirectionally coupled: weak BT prevents informative actions from receiving credit (AS cannot learn), and weak AS deprives BT of useful evidence to learn from (BT cannot learn). Once these two channels lock, they form an attractor of low information and low BT—which the authors term Information Self-Locking (SeL).

**Goal**: (i) Formalize what SeL is and why it exists stably under outcome RL; (ii) provide a lightweight solution to break SeL **without introducing calibrated intermediate rewards or training dense reward models**.

**Key Insight**: In many active reasoning scenarios, determining "whether this action elicited new information" and "whether this belief update moved toward the ground truth" is much easier than determining "how much reward this step deserves." The former only requires checking if environment feedback contains new evidence or if self-report confidence increases—qualitative directional critiques that can be provided by rules or cheap LLM labeling without calibration.

**Core Idea**: Translate these $\pm1/0$ directional critiques into a likelihood-margin auxiliary objective. Its gradient is equivalent to adding a constant bias to the advantage of standard policy gradients: $\hat{A}_t \leftarrow A_t + \lambda u_t$. This moves credit within the trajectory from steps with negative critiques to those with positive critiques while keeping the outcome reward and RL backbone unchanged.

## Method

### Overall Architecture
The authors model active reasoning as a POMDP $(\mathcal{S},\mathcal{A},\mathcal{O},T,O,R,\gamma)$, where the latent state $s^\star$ is the hidden user preference/diagnosis/patch. Agent behavior is split into two interleaved kernels: the AS kernel $\pi_\omega^{\mathrm{as}}(a_t\mid b_t)$ selects environment actions based on the current belief $b_t$ to determine the next observation $o_t \sim O(\cdot\mid s^\star, a_t)$, and the BT kernel $\pi_\omega^{\mathrm{bt}}(b_{t+1}\mid b_t, a_t, o_t)$ integrates new observations into the belief. A trajectory is an alternating sequence of (AS round, Update round). The paper first formalizes SeL using oracle-belief trajectories and Bayesian updates, proving that the outcome-gradient is first-order decayed in the self-locking region, then uses AReW to pull this first-order term back up by adding directional critiques to the advantage.

### Key Designs

**1. SeL Formalization and Coupled Gradient Decomposition: Attributing "Reward Growth without Capability Growth" to Bidirectional AS-BT Credit Masking**

Previous explanations for LLM agent RL failures mostly relied on "sparse rewards," but this cannot explain stagnant or regressive AS/BT proxies despite rising reward curves. Seeking a falsifiable mechanism, the authors define belief potential $\Psi(b) := b(s^\star)$. AS informational capability $I_{\mathrm{AS}}(\omega) := \mathbb{E}[\Psi(\bar b_H) - \Psi(\bar b_0)]$ is isolated using oracle-belief trajectories $\bar\tau \sim (\pi_\omega^{\mathrm{as}}, \mathsf{BayesUpd})$, which replaces BT with perfect Bayesian updates. BT capacity $C_{\mathrm{BT}}(\omega)$ is defined by the sum of "absorbed positive components" $\Delta\Psi_t^+ = \max(0, \Psi(b_{t+1})-\Psi(b_t))$ of on-policy trajectories. The self-locking region is defined as $\mathcal{R}_{\delta,\varepsilon} := \{\omega: I_{\mathrm{AS}}\le\delta,\, C_{\mathrm{BT}}\le\varepsilon\}$.

By decomposing $\nabla_\omega \log p_\omega(\tau)$ into AS-channel and BT-channel gradients $g_{\mathrm{as}}, g_{\mathrm{bt}}$, Theorem 3.4 shows that in the self-locking region, the one-sided drift of both channels satisfies $(\Delta_{\mathrm{as}}^+ I_{\mathrm{AS}},\, \Delta_{\mathrm{bt}}^+ C_{\mathrm{BT}})^\top \preceq \eta M (I_{\mathrm{AS}}, C_{\mathrm{BT}})^\top + o(\eta)$. Crucially, the non-zero terms of matrix $M$ multiply the capability of the **other** channel—BT cannot learn when AS is weak, and AS cannot learn when BT is weak. Thus, the escape time has an explicit lower bound, making it difficult for the agent to spontaneously escape. This analysis provides a direct intervention interface: since the channels stifle each other's credit, SeL can be unlocked by providing separate directional signals for each.

**2. AS / BT Directional Critique: Replacing Calibrated Dense Rewards with Cheap $\pm1/0$ Directional Signals**

Calibrated step rewards are nearly impossible to obtain in long-horizon agentic tasks—they either require training reward models that introduce new biases or massive labeling. SeL lacks directional signals of "whether this step improved or worsened." Thus, the authors assign $z_t \in \{-1, 0, +1\}$ per step: the AS critique $z_t^{\mathrm{as}}$ checks if the action elicited new evidence from the environment/user (e.g., in PE, whether an attribute pair is non-dominated; in MediQ, whether the new fact set is non-empty). The BT critique $z_t^{\mathrm{bt}} := \mathrm{Sign}(\hat\Psi_{t+1} - \hat\Psi_t)$, where $\hat\Psi_t$ is the ground-truth candidate confidence reported by the prompted agent—a scalar for instrumentation that need not equal the analytical belief $b_t$.

The threshold for "good/bad/unknown" judgments is low and robust to noise, perfectly matching the two channels of SeL. Proposition 4.1 quantifies the benefit: the gain from AReW is $I_{\mathrm{AS}}(\hat{\mathcal{T}}_{\mathrm{as}}) - I_{\mathrm{AS}}(\mathcal{T}_{\mathrm{as}}) = \eta W(\omega) (2\,\mathrm{Acc}_{\mathrm{as}} - 1)$. As long as the weighted accuracy $\mathrm{Acc}_{\mathrm{as}} > 1/2$, the gain is positive—this is why it avoids calibration.

**3. Likelihood-Margin Objective → Advantage Reweighting: Injecting Critiques into Policy Gradient without Invading the RL Backbone**

To inject directional signals without biasing the outcome reward estimate, the authors define $\mathcal{P}_\tau := \{t: z_t=+1\}$ and $\mathcal{N}_\tau := \{t: z_t=-1\}$ for trajectory $\tau$. They construct an intra-trajectory likelihood-margin auxiliary objective $\hat{\mathcal{L}}(\omega;\tau) := \frac{1}{|\mathcal{P}_\tau|}\sum_{t\in\mathcal{P}_\tau}\log\pi_{\omega,t} - \frac{1}{|\mathcal{N}_\tau|}\sum_{t\in\mathcal{N}_\tau}\log\pi_{\omega,t}$. The gradient is $\sum_t u_t \nabla_\omega \log\pi_{\omega,t}$, where $u_t = +1/|\mathcal{P}_\tau|,\,-1/|\mathcal{N}_\tau|,\,0$ based on $z_t$. The centering property $\sum_t u_t = 0$ ensures it is a pure margin without mean shift.

The final augmented surrogate $\hat{\mathcal{L}}_{\mathrm{aug}} := \mathcal{J}_{\mathrm{RL}} + \lambda \mathbb{E}_\tau[\hat{\mathcal{L}}]$ is equivalent to modifying the advantage to $\hat A_t := A_t + \lambda u_t$, while the critic, ratio, and KL terms remain unchanged. Credit is transferred from steps with negative critiques to those with positive critiques, breaking the link in Theorem 3.4. Since the modification is a constant bias on the advantage, AReW is RL-algorithm agnostic and can be applied to PPO, GRPO, or GSPO.

### Loss & Training
The final training objective is Eq. (2): $\hat{\mathcal{L}}_{\mathrm{aug}}(\omega) = \mathcal{J}_{\mathrm{RL}}(\omega) + \lambda\,\mathbb{E}_\tau[\hat{\mathcal{L}}(\omega;\tau)]$. In implementation, the advantage for each step is replaced by $A_t + \lambda u_t$. AS critiques are provided by rules/feedback parsers. BT critiques are generated by making the agent output confidence $\hat\Psi_t$ for $s^\star$ candidates during each Update Round, using the sign of the difference between consecutive rounds. $\lambda$ is scaled to match the sparse rewards; empirical results show low sensitivity to this hyperparameter.

## Key Experimental Results

### Main Results
9 active reasoning tasks across 4 domains (Preference Estimation, Medical Diagnosis, Troubleshooting, $\tau^2$-bench Customer Service), using Qwen-2.5-7B-Instruct and LLaMA-3.1-8B-Instruct. The following table shows core results using Qwen-2.5-7B-Instruct + PPO (numbers represent mean outcome reward on test set):

| Task | o4-mini (Direct) | Vanilla PPO | AReW–AS only | AReW–AS+BT | AS+BT Gain vs. Vanilla |
|------|------------------|-------------|--------------|------------|-----------------------|
| PE-G S=2 | 17.11 | 24.00 | 46.00 | 49.33 | +25.3 |
| PE-G S=3 | 21.15 | 18.33 | 32.00 | **80.33** | **+62.0** |
| PE-F D=8 | 8.42 | 30.52 | 39.62 | 47.89 | +17.4 |
| PE-F D=6 | 12.47 | 32.03 | 42.10 | 44.47 | +12.4 |
| MediQ | 74.67 | 50.50 | 57.25 | 61.25 | +10.8 |
| FloDial-Easy | 35.00 | 37.33 | 43.67 | 41.00 | +3.7 |
| FloDial-Hard | 26.33 | 21.33 | 36.00 | 42.33 | +21.0 |

Pattern consistency was maintained on LLaMA-3.1-8B-Instruct, with PE-G S=3 jumping from 11.00 to 77.67 (+66.7) and PE-F D=6 from 6.00 to 54.65 (+48.7). AReW outperformed the baseline in 27 out of 28 (task, RL algo) configurations; AS+BT significantly outperformed AS-only in 11 of 14 comparisons, validating the necessity of dual-channel intervention.

### Ablation Study

| Configuration | PE-G S=3 | FloDial-Hard | Description |
|------|----------|--------------|------|
| Vanilla PPO | 18.3 | 21.3 | Outcome-only baseline |
| AReW (α=0, Clean Critique) | 80.3 | 36.0 | Upper bound |
| AReW (α=0.1, 10% Flipping) | 40.0 | 30.3 | Still outperforms vanilla |
| AReW (α=0.2) | 65.0 | 29.0 | Still outperforms vanilla |
| AReW (α=0.3) | 31.3 | 27.6 | Still outperforms vanilla |
| AReW (α=0.4) | 22.3 | 30.6 | Near threshold |
| AReW (α=0.5) | 30.3 | 23.3 | Random critique degrades to baseline |

This matches the $2\,\mathrm{Acc} - 1$ scaling in Proposition 4.1: performance only collapses when weighted accuracy falls below 0.5.

### Key Findings
- **Coupling is Real**: On PE-G/MediQ, fixing the action sequence while replacing BT (using rules or frontier models for belief updates) caused the correlation between AS and reward to rise significantly. This indicates weak BT masks AS contributions—directly validating Obs. 2 and Theorem 3.4.
- **AS-only Boosts BT**: Pure AS critiques not only raised the AS proxy but also the BT proxy. This shows AReW indirectly assists BT learning by increasing information flow from the AS side, further confirming intrinsic coupling.
- **RL-Algorithm Agnostic**: Applying AReW to GRPO/GSPO yielded results consistent with PPO. Group-based multiple rollouts do not solve SeL, as the coupling is structural rather than a sampling variance issue.
- **Discovery on $\tau^2$-bench**: After vanilla PPO training, tool execution errors and token usage spiked as turn counts were compressed. AReW restored the interaction pace and significantly reduced error rates, showing SeL also occurs in real tool-calling scenarios.

## Highlights & Insights
- **Precise Naming of Failure Mode**: Locating the "reward growth without capability growth" phenomenon—previously vaguely attributed to "sparse rewards" or "shortcut learning"—to bidirectional AS-BT credit masking, and providing a formal definition with an escape-time lower bound, is a rare instance of "diagnosing the disease before prescribing the medicine."
- **Directional Critique vs. Calibrated Reward**: Proposition 4.1 characterizes critique quality as weighted accuracy $>1/2$, suggesting to the community: stop obsessing over process reward models in long-horizon agentic RL; a "good/bad" judgment is sufficient. The engineering value of this conclusion extends beyond this paper to RAG, coding, and research agents where environment feedback carries directional signals.
- **Zero-Intrusion Implementation**: The actual change in AReW is merely adding a constant bias to the advantage. The centering property ensures no changes are needed for the critic, KL, or clipping—the true reason it improves performance across PPO, GRPO, and GSPO.

## Limitations & Future Work
- **Dependence on Directional Feedback**: Directional critiques require signals like "information elicitation" or "self-reported confidence changes" to be cheaply obtainable. This is not directly applicable to tasks with extremely sparse rewards and no environment feedback (e.g., pure text mathematical proofs).
- **BT Critique Depends on Self-Reported Confidence**: $\hat\Psi_t$ is derived from prompted probabilities, which are subject to LLM calibration. Severe over/under-confidence might distort the sign of $z_t^{\mathrm{bt}}$. While the paper proves robustness to noise, degradation under extreme mis-calibration is not fully explored.
- **Theoretical Assumptions**: Theorem 3.4 relies on several regularity conditions (Lipschitz reward-belief coupling, action-invariant harmful belief drift, etc.). Whether real LLMs strictly satisfy these can only be inferred from empirical evidence—a "correct direction but limited precision" analysis.
- **Extensible Directions**: Extending the SeL framework to multi-agent collaboration, generating directional critiques online using LLM-judges, and complementing process reward models (using PRM for high confidence and directional critiques for low confidence) are natural extensions.

## Related Work & Insights
- **vs. Process Reward Model (PRM)**: PRM approaches (e.g., Math-Shepherd) seek step-level calibrated rewards. AReW seeks only directionality, additivity, and zero calibration, proving that weighted accuracy $>1/2$ is enough. These are not conflicting; PRM provides absolute values while AReW provides direction.
- **vs. Vanilla GRPO/GSPO**: Group sampling reduces variance through multi-rollout, but SeL is structural. This explains why GRPO gains are less significant in agentic tasks compared to math, suggesting future agentic RL should include built-in channel-level credit assignment.
- **vs. Information-Seeking RL**: Previous works used auxiliary rewards for query diversity. This work explains why such rewards are often unstable from the perspective of gradient signals being suppressed by the other channel.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Precisely named and formalized the agentic RL failure mode. The minimalist advantage shaping has high engineering potential.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covered 4 domains, 9 tasks, 2 backbones, and 3 RL algorithms, including noise robustness and $\tau^2$-bench; lacks validation on larger scales (30B+).
- **Writing Quality**: ⭐⭐⭐⭐ Clear storyline; theorems and experiments are tightly coupled. Multiple assumptions in theorems make it slightly dense for non-RL readers.
- **Value**: ⭐⭐⭐⭐⭐ "Directional critique + advantage reweighting" could become a default trick for agentic RL, with immediate engineering implications for training research and coding agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

1. **Active Retrieval Augmented Generation** (2024), arXiv ID: 2401.xxxxx
2. **Let's Verify Step by Step** (2023), arXiv ID: 2305.20050
3. **Training Language Models to Follow Instructions with Human Feedback** (2022), NeurIPS

</div>

<!-- RELATED:END -->

## Related Papers

- [\[ICLR 2026\] Reducing Belief Deviation in Reinforcement Learning for Active Reasoning of LLM Agents](../../ICLR2026/llm_agent/reducing_belief_deviation_in_reinforcement_learning_for_active_reasoning.md)
- [\[AAAI 2026\] MoralReason: Generalizable Moral Decision Alignment For LLM Agents Using Reasoning-Level Reinforcement Learning](../../AAAI2026/llm_agent/moralreason_generalizable_moral_decision_alignment_for_llm_agents_using_reasonin.md)
- [\[ICML 2026\] EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle](evolver_self-evolving_llm_agents_through_an_experience-driven_lifecycle.md)
- [\[ICML 2026\] Towards Feedback-to-Plan Decisions for Self-Evolving LLM Agents in CUDA Kernel Generation](towards_feedback-to-plan_decisions_for_self-evolving_llm_agents_in_cuda_kernel_g.md)
- [\[ICML 2026\] Talk, Judge, Cooperate: Gossip-Driven Indirect Reciprocity in Self-Interested LLM Agents](talk_judge_cooperate_gossip-driven_indirect_reciprocity_in_self-interested_llm_a.md)

</div>

<!-- RELATED:END -->
