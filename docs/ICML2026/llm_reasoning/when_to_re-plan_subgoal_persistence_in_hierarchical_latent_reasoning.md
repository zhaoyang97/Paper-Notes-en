---
title: >-
  [Paper Note] When to Re-Plan: Subgoal Persistence in Hierarchical Latent Reasoning
description: >-
  [ICML 2026][LLM Reasoning][Hierarchical Reasoning] This paper introduces persistent manager-worker subgoals into the Hierarchical Reasoning Model (HRM). It discovers that the critical factor in latent reasoning is not si…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "Hierarchical Reasoning"
  - "Latent Computation"
  - "Subgoal Persistence"
  - "HRM"
  - "Planning Stability"
date: 2026-05-08
content_hash: 3f5e85bdf6cf543f
---

# When to Re-Plan: Subgoal Persistence in Hierarchical Latent Reasoning

**Conference**: ICML 2026  
**arXiv**: [2606.03741](https://arxiv.org/abs/2606.03741)  
**Code**: No public code  
**Area**: LLM Reasoning / Latent Reasoning Architecture  
**Keywords**: Hierarchical Reasoning, Latent Computation, Subgoal Persistence, HRM, Planning Stability  

## TL;DR
This paper introduces persistent manager-worker subgoals into the Hierarchical Reasoning Model (HRM). It discovers that the critical factor in latent reasoning is not simply the injection of subgoals, but that they must persist for $P=3$ to $6$ low-level update steps; excessively rapid re-planning disrupts compositional structures, while overly aggressive alignment interferes with task learning.

## Background & Motivation
**Background**: Long-range reasoning systems typically follow two trajectories. One is explicit Chain-of-Thought (CoT), where reasoning is written as a token sequence. The other is latent reasoning, which compresses multi-step computation into hidden states via iterative internal updates. Hierarchical latent architectures like HRM implement deeper internal computation using slow high-level states and fast low-level states.

**Limitations of Prior Work**: Explicit token-based reasoning naturally possesses a temporal structure where "each token constrains subsequent tokens." Conversely, hidden state updates in latent reasoning lack such external commitment. High-level states can change their "mind" at every step or remain static for long periods; the architecture itself does not specify how long a mid-term intention should persist.

**Key Challenge**: if subgoals are re-issued at every step, the worker might be overwritten before forming multi-step computations around them. If subgoals persist too long, the worker's hidden state drifts, rendering old goals rigid. Thus, a latent planner must find a suitable re-planning cycle to balance stability and adaptability.

**Goal**: The authors aim to empirically characterize the role of subgoal persistence by explicitly adding feudal-style subgoals to HRM and scanning the manager period $P$ and alignment weight $\lambda$ to determine when to re-plan.

**Key Insight**: The concept of "commitment time" from options or feudal hierarchies in reinforcement learning is ported to latent reasoning. Here, the "action" is not an environmental move but a low-level hidden state update; the goal output by the manager is a direction vector in the latent space.

**Core Idea**: A high-level module issues a normalized directional subgoal $g$ every $P$ micro-steps. This $g$ persistently biases low-level updates throughout these $P$ steps, while a cosine alignment loss encourages the low-level net displacement to move in this direction.

## Method
The method is termed Subgoal-Augmented HRM. It retains the original HRM backbone with slow high-level states $z^H$ and fast low-level states $z^L$ while adding an explicit subgoal interface. High-level states no longer influence the low-level only through implicit recurrent coupling but instead periodically project a direction vector that steers the worker's internal computation over multiple steps.

### Overall Architecture
The HRM backbone contains two latent states: the low-level state updates at every micro-step, while the high-level state updates once every $T$ low-level steps. Subgoal-Augmented HRM introduces a manager period $P$. At time $t_k=kP$, the high-level state outputs $\tilde g_k=W_g z^H_{t_k}$ via $W_g$, which is normalized to $g_k=\tilde g_k/(\|\tilde g_k\|_2+\epsilon)$. This $g_k$ remains constant for the following $P$ low-level updates.

During low-level updates, the subgoal is added to the input or the update via projection $V_L$, creating a continuous steering term. To prevent the subgoal from being an unconstrained bias, the model computes a cosine alignment loss between the net low-level displacement $\Delta z^L_k=z^L_{t_k+P}-z^L_{t_k}$ and $g_k$ over each commitment window.

### Key Designs
1.  **Directional Subgoals vs. Target States**:
    - **Function**: Provides the latent reasoner with a mid-term intentional signal while avoiding the prescription of an absolute hidden state.
    - **Mechanism**: The manager outputs a unit direction $g_k$ expressing "where to move next" rather than a specific destination. An optional commitment gate $\alpha_k=\sigma(w_\alpha^\top z^H_{t_k})$ adjusts the subgoal intensity, allowing the high-level to soften commitments under uncertainty.
    - **Design Motivation**: Latent hidden dynamics are non-stationary. Forcing the model to track absolute target states is fragile. Directional signals act as planning priors that provide structure without fully locking the worker's state.

2.  **Persistence Period $P$ Controlling Stability-Adaptability Trade-off**:
    - **Function**: Explicitly controls the duration for which a subgoal remains unchanged across low-level micro-steps.
    - **Mechanism**: For $t\in[t_k,t_k+P)$, the active goal remains $g_k$. Low-level updates take the form $z^L_{t+1}=f_L(z^L_t,z^H_t,\tilde x_t+\alpha(t)V_Lg(t);\theta_L)$. $P=1$ indicates per-step re-planning, while larger $P$ denotes longer commitment.
    - **Design Motivation**: The core hypothesis is that compositional latent computation requires several consecutive update steps structured around the same intent; without persistence, the manager-worker architecture exists in form but lacks function.

3.  **Window-level Cosine Alignment Loss**:
    - **Function**: Aligns the worker’s net displacement with the subgoal direction, moving beyond passive bias reception.
    - **Mechanism**: Each window uses $\mathcal{L}_{align}^{(k)}=1-\cos(\Delta z^L_k,g_k)$, with total loss $\mathcal{L}=\mathcal{L}_{HRM}+\lambda\sum_k\mathcal{L}_{align}^{(k)}$. Since HRM detaches states between segments, the alignment loss backpropagates only within segments.
    - **Design Motivation**: This transforms the subgoal from an "auxiliary input feature" into an "internal geometric prior." However, $\lambda$ must be small to prevent the directional constraint from competing with task gradients.

### Loss & Training
The training objective combines the original HRM task loss, ACT halting loss, and the alignment loss from subgoal windows. The HRM backbone is fixed: hidden size 512, 4-layer high-level transformer, 4-layer low-level transformer, 8 heads, and a maximum of 16 internal steps. Models are trained using AdamATan2 with a base learning rate of $10^{-4}$ and weight decay of 0.1.

The paper reports two experimental regimes: a main study on the arc-aug-1000 dataset (global batch size 768) scanning $P$ and $\lambda$, and an interference ablation on a single NVIDIA L4 (batch size 64) comparing full, baseline, and random directions at $\lambda=0.10, P=4$. The authors emphasize that absolute losses between different regimes are not comparable; only intra-regime differences are analyzed.

## Key Experimental Results

### Main Results
The main experiments evaluate the model on arc-aug-1000 (derived from ARC-AGI and ConceptARC). Primary metrics include training LM loss, token-level accuracy, exact accuracy, alignment loss, and ACT halting depth. The critical independent variables are persistence period $P$ and alignment weight $\lambda$.

| Setting | Metric | Result | Control | Gain |
|:---|:---|:---|:---|:---|
| No subgoal baseline | LM loss ↓ | 1.640 | Original HRM | Reference |
| $P=1, \lambda=0.05$ | LM loss ↓ | 1.674 | Baseline 1.640 | Worse; proves per-step re-planning is harmful |
| $P=2, \lambda=0.05$ | LM loss ↓ | 1.638 | Baseline 1.640 | Marginal improvement |
| $P=3, \lambda=0.05$ | LM loss ↓ | 1.544 | Baseline 1.640 | -0.096; optimal single point |
| $P=3$ to $P=8$ | LM loss range | [1.544, 1.590] | Baseline 1.640 | Moderate over-commitment remains acceptable |
| ConceptARC-mini | LM loss ↓ | 2.308 | Baseline 2.316 | Consistent direction but small magnitude |

### Ablation Study
The key ablation addresses whether performance drops at $\lambda=0.10$ (past the optimal point) stem from architectural capacity, auxiliary loss, or the learned direction itself.

| Configuration | Key Metric | Description |
|:---|:---|:---|
| A_full | Train LM loss 1.327 (+0.100 vs baseline) | Learned direction + injection + alignment loss; over-strong direction causes interference. |
| B_baseline | Train LM loss 1.227 | Injection and alignment disabled; reverts to vanilla HRM. |
| E_random | Train LM loss 1.230 (+0.003 vs baseline) | Random unit directions are nearly equivalent to the baseline. |
| A_full vs E_random | Gap 0.097 | Interference is caused by learned directional content, not extra modules or the loss itself. |
| $\lambda$ sweep | $\lambda\approx0.05$ optimal | Alignment acts as a soft prior; $\lambda \ge 0.20$ is significantly harmful. |

### Key Findings
- Persistence is a necessary condition. $P=1$ is worse than having no subgoals, suggesting that the presence of a manager or goal vector is insufficient without enough time to organize multi-step latent computation.
- The optimal interval is $P \in [3, 6]$. Performance decays slowly from $P=3$ to $P=8$, indicating that the penalty for a stale subgoal is lower than the penalty for no commitment.
- The optimal weight for alignment loss is narrow. $\lambda=0.05$ acts as a lightweight planning prior, while $\lambda=0.10$ begins to compete with task objectives.
- The random direction ablation is crucial: E_random is nearly identical to the baseline, which proves that learned subgoals are not merely decorative but actively influence representation capacity.

## Highlights & Insights
- The paper asks a precise question: latent reasoning requires more than just "extra steps"; it requires a decision on how frequently internal intentions should update. This timescale issue is masked by token sequences in CoT but must be explicitly designed in latent computation.
- The failure of $P=1$ is a powerful negative result. It precludes the explanation that "merely adding a subgoal module helps" and attributes the gains specifically to the persistence mechanism.
- The random direction ablation elegantly shows that learned subgoals can both help (at the right weight) and hurt (when too strong), rather than being neutral additions.

## Limitations & Future Work
- Evaluation is primarily on ARC/ConceptARC style tasks, and the main metric is training LM loss. While this demonstrates mechanistic behavior, it does not yet prove generalization across all reasoning tasks.
- The strongest conclusions are empirical and behavioral; there is a lack of representation-level analysis regarding whether subgoals correspond to interpretable sub-problems or intermediate program structures.
- The past-sweet-spot analysis in the ablation study used a single seed. While the gap is much larger than typical seed variance, multi-seed ablations would be more robust.
- The method introduces additional hyperparameters $P$ and $\lambda$. Future deployments might require adaptive or learned re-planning mechanisms if different tasks require different persistence periods.

## Related Work & Insights
- **vs. Chain-of-Thought**: CoT uses explicit tokens for trajectories and commitment. This work investigates internal commitment timescales within hidden states, which is more relevant for low-latency latent reasoning.
- **vs. Original HRM**: HRM features high/low-level loops and halting but lacks explicit mid-term intentional signals; this work transforms high-level states into persistent directional goals.
- **vs. Feudal RL / Options**: This work borrows the manager-worker and temporal abstraction concepts but replaces environmental actions with hidden-state updates, tailored for latent variable reasoning.
- **Transferable Insights**: Future latent planners could learn *when* to re-plan rather than using a fixed $P$; subgoal directions could also be aligned with interpretable intermediate tasks, program snippets, or retrieval targets.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Investigating subgoal persistence as a core variable in latent reasoning is a fresh perspective with a simple mechanism.
- Experimental Thoroughness: ⭐⭐⭐☆☆ Scans and ablations support the mechanistic conclusions, but task scope and metrics could be broader.
- Writing Quality: ⭐⭐⭐⭐☆ Well-structured around the stability-adaptability trade-off; negative results and ablations are clearly explained.
- Value: ⭐⭐⭐⭐☆ Insightful for building internal planning-based reasoning models, especially regarding re-planning timescales.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought Reasoning](how_far_ahead_do_llms_plan_uncovering_the_latent_horizon_in_chain-of-thought_rea.md)
- [\[ICLR 2026\] When Shallow Wins: Silent Failures and the Depth-Accuracy Paradox in Latent Reasoning](../../ICLR2026/llm_reasoning/when_shallow_wins_silent_failures_and_the_depth-accuracy_paradox_in_latent_reaso.md)
- [\[ICML 2026\] Modeling Hierarchical Thinking in Large Reasoning Models](modeling_hierarchical_thinking_in_large_reasoning_models.md)
- [\[ICML 2026\] The Deterministic Horizon: When Extended Reasoning Fails and Tool Delegation Becomes Necessary](the_deterministic_horizon_when_extended_reasoning_fails_and_tool_delegation_beco.md)
- [\[ICML 2026\] LatentChem: From Textual CoT to Latent Thinking in Chemical Reasoning](latentchem_from_textual_cot_to_latent_thinking_in_chemical_reasoning.md)

</div>

<!-- RELATED:END -->
