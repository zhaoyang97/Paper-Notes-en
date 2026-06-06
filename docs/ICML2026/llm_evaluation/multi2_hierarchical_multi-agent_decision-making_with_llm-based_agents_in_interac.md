---
title: >-
  [Paper Note] Multi$^2$: Hierarchical Multi-Agent Decision-Making with LLM-Based Agents in Interactive Environments
description: >-
  [ICML 2026][LLM Evaluation][Hierarchical Agent] This paper proposes the Multi$^2$ framework, which explicitly decouples the "planning" and "execution" of LLM agents into System 1 (an SFT-trained sub-goal planner) and Sys…
tags:
  - "ICML 2026"
  - "LLM Evaluation"
  - "Hierarchical Agent"
  - "Objective Drift"
  - "Sub-goal Planning"
  - "Offline-to-Online RL"
  - "Long-horizon Interaction"
date: 2026-05-08
content_hash: 28535a8cb9eda66f
---

# Multi$^2$: Hierarchical Multi-Agent Decision-Making with LLM-Based Agents in Interactive Environments

**Conference**: ICML 2026  
**arXiv**: [2606.03698](https://arxiv.org/abs/2606.03698)  
**Code**: https://park-sangeun.github.io/Multi-Square/ (Project Page)  
**Area**: LLM Agent / Multi-Agent / Offline-to-Online RL  
**Keywords**: Hierarchical Agent, Objective Drift, Sub-goal Planning, Offline-to-Online RL, Long-horizon Interaction

## TL;DR
This paper proposes the Multi$^2$ framework, which explicitly decouples the "planning" and "execution" of LLM agents into System 1 (an SFT-trained sub-goal planner) and System 2 (an atomic action executor trained via offline-to-online RL). By utilizing role-specific LoRA adapters and training objectives with policy-anchoring and KL regularization, the framework significantly mitigates objective drift and improves token efficiency across three long-horizon interactive environments: ScienceWorld, ALFWorld, and TextCraft.

## Background & Motivation

**Background**: Enabling LLM agents to complete long-horizon tasks involving "multi-turn interaction-observation-decision" (e.g., embodied tasks, text games, tool use) is a core objective of agentic AI. Mainstream approaches include prompt-based methods (ReAct, Reflexion, ADaPT) and fine-tuning-based methods (Single-agent RL like GRPO, Hierarchical RL like Glider).

**Limitations of Prior Work**: Long-horizon interactions are extremely fragile. The authors observe two specific issues in ScienceWorld: (1) **objective drift**—as a task unfolds, the initial intent is "drifted" away due to cumulative context and overlapping execution errors, causing the agent to deviate further from the goal; (2) **token waste**—agents rely on increasingly long interaction histories to maintain intent, leading to a surge in inference tokens. ReAct performance drops fastest over long horizons; even with hierarchy, Glider performance continues to decline as the horizon increases.

**Key Challenge**: Existing hierarchical methods mainly solve issues on the "planning side" (reducing planning failure by decomposing tasks). However, the **execution side is not explicitly trained to correct compound errors**. Methods like Glider primarily update the high-level planner during online adaptation while leaving the low-level executor mostly static. Furthermore, the planner and executor often share the same LoRA adapter, distinguishing roles only via prompts, which leads to blurred role boundaries as context accumulates. This causes the executor to fall into invalid action loops when facing constraint-dense action spaces.

**Goal**: To build reliable and efficient interactive LLM agents that simultaneously satisfy (1) explicit task decomposition to preserve intent, (2) robust action execution improved through environment interaction, and (3) token-efficient invocation.

**Key Insight**: The authors view "planning" and "execution" as two fundamentally different optimization problems—planning requires stable semantic supervision signals (suitable for SFT), while execution requires error correction through dynamic environment feedback (suitable for RL). The two should utilize **different training paradigms + different parameters (LoRA adapters)**, rather than sharing weights and switching via prompts.

**Core Idea**: Replace shared-adapter hierarchical prompting with a role-specific hierarchical architecture: "System 1 (SFT sub-goal planning) + System 2 (offline-to-online RL atomic action execution)." System 2 is designed with an offline loss containing a policy-anchoring term and an online loss with KL regularization, ensuring stable initialization from offline data followed by continuous self-improvement without mode collapse.

## Method

### Overall Architecture
Multi$^2$ models the decision process as a partially observable MDP $\langle\mathcal{S},\mathcal{A},\mathcal{O},\mathcal{T},\Omega,\mathcal{R},\gamma\rangle$. Given a task description $K_n$ and observation $\mathbf{o}_t$, **System 1** (parameters $\phi$) outputs a sub-goal $g_h \sim \pi_\phi(\cdot \mid \mathbf{o}_t; K_n)$. Conditioned on $\mathbf{o}_t$ and $g_h$, **System 2** (parameters $\theta$) selects atomic actions $\mathbf{a}_t \sim \pi_\theta(\cdot \mid \mathbf{o}_t; g_h)$. Once $g_h$ is completed (or terminated), System 2 calls System 1 again to generate $g_{h+1}$, forming a hierarchical closed-loop. Both systems share the same pretrained backbone (Qwen-2.5 3B / Mistral 7B / Llama-3.1 8B) but use **independent LoRA adapters**—only the adapter corresponding to the current role is activated, decoupling role expertise at the parameter level.

Training data is split into two parts: $\mathcal{D}_{sys1} = \{(K_n, \{(\mathbf{o}_h, g_h)\}_{h=1}^H)\}$, a sequence of task-observation-subgoal triplets, and $\mathcal{D}_{sys2} = \{(g^{(i)}, (\xi_t^{(i)})_{t=1}^{M^{(i)}})\}$, where $\xi_t = (\mathbf{o}_t, \mathbf{a}_t, r_t, \mathbf{o}_{t+1})$ is a sequence of low-level transitions conditioned on sub-goals. The dataset construction pipeline improves upon Glider by adding code-rule-based sub-goal extraction and prompt notarization to improve reproducibility.

### Key Designs

1.  **Role-Specific Hierarchy + LoRA Decoupling (System 1 / System 2)**:
    - **Function**: Assigns "semantic planning" and "atomic execution" to two independent LoRA adapters, preventing role boundaries from being diluted by prompt context when sharing parameters.
    - **Mechanism**: Two LoRAs are attached to the same backbone. System 1 is solely responsible for inputting "task + observation" and outputting "sub-goals"; System 2 is solely responsible for inputting "observation + sub-goal" and outputting "actions." During inference, the activated adapter is switched based on the current stage. System 1 is invoked "on-demand" (only when a sub-goal is reached), naturally providing token efficiency.
    - **Design Motivation**: Ablations (Figure 6c) demonstrate that the median and IQR of shared adapters are significantly lower than role-specific adapters—parameter-level decoupling maintains role expertise more stably than prompt-level decoupling and serves as the structural foundation for solving objective drift.

2.  **Offline RL Objective with Policy-Anchoring Term (System 2 Phase 1)**:
    - **Function**: Uses IQL-style offline RL to stably initialize System 2 from static datasets while mitigating the "over-imitation of templated trajectories" issue.
    - **Mechanism**: For the Critic, the Q-function minimizes TD-error $\mathcal{L}_\omega = \mathbb{E}[(r_t + \gamma V_\psi(\mathbf{o}_{t+1}; g_h) - Q_\omega(\mathbf{o}_t, \mathbf{a}_t; g_h))^2]$, and the V-function uses expectile regression $\mathcal{L}_\psi = \mathbb{E}[L^\tau(A(\mathbf{o}_t, \mathbf{a}_t))]$ where $L^\tau(u) = |\tau - \mathbb{1}\{u<0\}|\,u^2$. For the Actor, in addition to the standard advantage-weighted imitation term $\exp(\beta A(\mathbf{o}_t, \mathbf{a}_t; g_h)) \log \pi_\theta^{off}(\mathbf{a}_t \mid \mathbf{o}_t; g_h)$, a **policy-anchored term** $\lambda A(\mathbf{o}_t, \pi_\theta^{off}(\mathbf{a}_t \mid \mathbf{o}_t; g_h); g_h)$ is added—allowing the critic to directly score actions sampled by the current policy as an additional signal.
    - **Design Motivation**: Pure IQL only performs weighted imitation of dataset actions, which can trap the policy within the offline distribution and lead to over-templated behavior. The anchoring term allows high-advantage actions preferred by the critic to participate in policy updates, enhancing generalization beyond offline trajectories. Figure 7a verifies that removing this term decreases the median and lengthens the low-score tail.

3.  **Online Refinement with KL Regularization (System 2 Phase 2)**:
    - **Function**: Continues from the offline policy with online interaction for self-improvement while preventing mode collapse.
    - **Mechanism**: Uses a mixed replay buffer $\mathcal{B}_{sys2}$ (offline data + newly sampled online transitions). The loss is $\mathcal{L}_\theta^{online} = -\mathbb{E}[w_t \log \pi_\theta^{on}(\mathbf{a}_t \mid \mathbf{o}_t; g_h)] + \eta\, \mathbb{E}[D_{KL}(\pi_\theta^{on} \,\|\, \pi_\theta^{off})(\mathbf{o}_t; g_h)]$, where $w_t = \exp(\frac{1}{\alpha} A(\mathbf{o}_t, \mathbf{a}_t; g_h))$ uses advantage reweighting to reinforce high-reward actions; the KL term pulls the online policy toward the offline policy to stabilize the distribution.
    - **Design Motivation**: Pure online AWAC exhibits high variance on LLM agents and easily collapses into a single behavioral pattern. KL anchoring to the offline policy acts as a "trust region," keeping exploration-improvement within safe bounds. Figure 7b shows that Vanilla-AWAC occasionally fails at the low end; adding KL regularization results in a tighter, more reliable distribution.

### Loss & Training
The overall training consists of three phases (Algorithm 1): (1) System 1 is trained for $EP_1$ epochs using SFT loss $\mathcal{L}_\phi = -\mathbb{E}[\log \pi_\phi(g_h \mid \mathbf{o}_t; K_n)]$; (2) System 2 is jointly trained for $EP_2$ epochs using the aforementioned $\mathcal{L}_\omega, \mathcal{L}_\psi, \mathcal{L}_\theta^{offline}$ to obtain $\pi_\theta^{off}$; (3) Setting $\pi_\theta^{on} \leftarrow \pi_\theta^{off}$, new transitions are collected via environment rollouts and added to the buffer, followed by $EP_3$ epochs of training with $\mathcal{L}_\theta^{online}$. System 1 is no longer updated in Phase 3 and is only responsible for sampling sub-goals. All training uses LoRA.

## Key Experimental Results

### Main Results
Evaluated on ScienceWorld (curriculum long-horizon), ALFWorld (household operations, sparse reward), and TextCraft (symbolic recipe synthesis) across Qwen-2.5 3B, Mistral 7B, and Llama-3.1 8B backbones using strict pass@1:

| Setup (Llama-3.1 8B) | ScienceWorld ID | ScienceWorld OOD | ALFWorld ID | ALFWorld OOD | TextCraft |
|---|---|---|---|---|---|
| ReAct | 23.23 | 10.30 | 6.72 | 7.46 | 9.00 |
| Reflexion (pass@6) | 23.20 | 6.97 | 35.71 | 29.29 | 11.00 |
| ADaPT | 26.20 | 12.27 | 11.54 | 6.41 | 5.00 |
| GRPO | 25.79 | 4.61 | 8.57 | 7.50 | 5.50 |
| Glider | 60.48 | 34.36 | 43.57 | 37.86 | 9.50 |
| **Multi$^2$** | **67.61** | 30.68 | **57.86** | **56.43** | **35.60** |

On Mistral 7B, Multi$^2$ achieved 69.97 on ScienceWorld ID (vs. Glider 58.33) and 44.50 on TextCraft (vs. Glider 28.50). Multi$^2$ is the best performer across most backbone × split combinations.

### Ablation Study
On ScienceWorld ID + Llama-3.1 8B:

| Configuration | Key Observation |
|------|------|
| Full Multi$^2$ | Highest median + narrowest IQR |
| (1) RL-SFT swapped roles | Worst performance, clear role mismatch |
| (2) Only RL (both systems RL) | Low median + high variance, RL unstable for high-level planning |
| (3) Only SFT (both systems SFT) | Fair median but frequent failures, lack of execution robustness |
| Single model (no hierarchy) | Median significantly lower than hierarchical version |
| Shared adapter (hierarchical but shared LoRA) | Both median and IQR lower than role-specific |
| Vanilla-IQL (no policy-anchored) | Median decrease + longer low-score tail |
| Vanilla-AWAC (no KL regularization) | High variance, obvious low-end failures |

### Key Findings
- **Roles and training paradigms must match**: SFT for System 1 + RL for System 2 is the only stable combination. Swapping or homogenizing them leads to significant performance drops, indicating that semantic planning and atomic execution are indeed fundamentally different optimization problems.
- **Objective drift is most severe on difficult tasks**: Figure 5 shows that prompt-based methods drop sharply with difficulty (Easy/Medium/Hard). Fine-tuning-based methods also show significant drops from Medium to Hard. Only Multi$^2$ remains high on Hard tasks, showing that the synergistic effect of hierarchy + RL executor is most significant on long horizons.
- **Token efficiency comes from structure, not prompt compression**: The high token efficiency of Multi$^2$ stems from hierarchical on-demand invocation (System 1 is only called when sub-goals are completed) + compact action formats through RL fine-tuning, eliminating the need for long context examples.
- **Online RL provides larger gains on OOD**: Figure 9 shows that online RL brings more pronounced improvements on OOD splits. When distribution shift occurs, mismatches between sub-goals and environment dynamics are more frequent, and online interaction corrects these execution-level errors.

## Highlights & Insights
- **Reversing Kahneman's System 1/2 Metaphor**: In psychology, System 1 is fast/intuitive while System 2 is slow/deliberative. In this paper, System 1 is "slow planning" and System 2 is "fast execution." While the names are borrowed, the decomposition logic is clear—it is a rare work in LLM agent design that carries "training paradigm = role definition" to its logical conclusion.
- **The policy-anchored advantage term is a reusable trick**: This can be applied in all scenarios involving "offline RL imitation + concerns about templating," allowing the critic to score actions sampled by the current policy as a regularizer, providing more signal than simple noise or dropout.
- **Role-specific LoRA adapters** are more robust than prompt-based role separation—this finding has universal implications for multi-agent LLM system design, suggesting that "prompt-based role switching" fails under long context and that parameter-level isolation should be used.

## Limitations & Future Work
- **Ours**: Extensions to larger models (>8B) and real physical embodied environments remain unvalidated. The hierarchical dataset construction relies on Glider’s code-rule pipeline, requiring redesigning sub-goal extraction rules for new environments.
- **Additional Insights**: The experiments only cover three text-based interactive environments; more complex scenarios like function calling or multi-agent collaboration are not included. System 1 is frozen after training, meaning it cannot adapt online to new task distributions, so OOD sub-goal quality is limited by offline data. The KL regularization hyperparameter $\eta$ likely impacts stability, but sensitivity analysis is not provided.
- **Future Directions**: Exploring lightweight online updates for System 1 (e.g., prompt-based meta-updates); investigating latent communication channels between System 1/2 instead of explicit natural language sub-goals to further save tokens; extending to 3+ tiers (task-subtask-subgoal-action) for even longer horizons.

## Related Work & Insights
- **vs. Glider (Hu et al., ICML 2025)**: Glider is also hierarchical but uses only prompts to distinguish roles, shares a LoRA, and only updates the planner online. Multi$^2$ uses independent adapters, updates only the executor online, and adds policy-anchored/KL regularization to improve execution robustness. Multi$^2$ is consistently superior to Glider across backbones, with notable gains in TextCraft (9.5 → 35.6 on Llama-3.1 8B).
- **vs. ADaPT (Prasad et al., NeurIPS 2024)**: ADaPT uses prompt-based planner-executor hierarchy without fine-tuning. Multi$^2$ proves that prompt-only hierarchy is significantly outperformed in the fine-tuning era; parameter updates + role specificity are essential.
- **vs. ReAct / Reflexion**: Pure prompt-based methods suffer from severe objective drift on long horizons. Multi$^2$ suppresses failure modes (invalid action loops, unproductive loops) via explicit sub-goal anchoring and RL executor error correction.
- **vs. Standard Offline-to-Online RL (DigiRL, Bai et al. 2024)**: These focus on general policy improvement without customization for hierarchical executor roles. Multi$^2$’s policy-anchored + KL regularized loss is specifically designed for stably refining the executor within a hierarchical structure.

## Rating
- Novelty: ⭐⭐⭐⭐ The triadic alignment of "Role = Training Paradigm = LoRA Adapter" is new, and the combined loss of policy-anchored advantage and KL-regularized refinement is original. However, the overall framework (hierarchy + offline-to-online) is a solid improvement over Glider rather than a paradigm shift.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three environments × three backbones × ID/OOD splits, plus 5 types of ablations (config/structure/adapter/loss/scale) + difficulty stratification + token efficiency + online adaptation analysis.
- Writing Quality: ⭐⭐⭐⭐ Logical and clear conceptualization of "objective drift." The dual perspective of horizon robustness + token efficiency in Figure 1 is very persuasive.
- Value: ⭐⭐⭐⭐ Highly practical for researchers in long-horizon LLM agent tasks—the hierarchical dataset is open-sourced, and the policy-anchored and KL regularization tricks can be transferred to other offline-to-online LLM RL scenarios. It also provides clear empirical evidence for the "prompt vs. parameter-level role separation" debate.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Which LLM Multi-Agent Protocol to Choose?](../../ICLR2026/llm_evaluation/which_llm_multi-agent_protocol_to_choose.md)
- [\[ICML 2026\] HiPER: Hierarchical Reinforcement Learning with Explicit Credit Assignment for Large Language Model Agents](hiper_hierarchical_reinforcement_learning_with_explicit_credit_assignment_for_la.md)
- [\[ICML 2026\] Agent World Model: Infinity Synthetic Environments for Agentic Reinforcement Learning](agent_world_model_infinity_synthetic_environments_for_agentic_reinforcement_lear.md)
- [\[ACL 2026\] Fin-Bias: Comprehensive Evaluation for LLM Decision-Making under human bias in Finance Domain](../../ACL2026/llm_evaluation/fin-bias_comprehensive_evaluation_for_llm_decision-making_under_human_bias_in_fi.md)
- [\[NeurIPS 2025\] Heterogeneous Adversarial Play in Interactive Environments](../../NeurIPS2025/llm_evaluation/heterogeneous_adversarial_play_in_interactive_environments.md)

</div>

<!-- RELATED:END -->
