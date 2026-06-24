---
title: >-
  [Paper Note] Dual-Scale World Memory for LLM Agents towards Hard-Exploration Problems
description: >-
  [ICLR 2026][LLM Agent][Hard-exploration] GLoW is proposed to equip LLM agents with a dual-scale textual world memory—combining a "global trajectory frontier" and "local multi-path advantage reflection." It achieves new SOTA among LLM-based methods on sparse-reward hard-exploration tasks in Jericho text games, approaching the performance of the strongest RL methods with $100\times–800\times$ fewer environmental interactions.
tags:
  - "ICLR 2026"
  - "LLM Agent"
  - "Hard-exploration"
  - "World Memory"
  - "Go-Explore"
  - "Advantage Learning"
  - "Jericho"
date: 2026-05-08
content_hash: d9c70b288d6c2f58
---

# Dual-Scale World Memory for LLM Agents towards Hard-Exploration Problems

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=bH5uHIVtTe](https://openreview.net/forum?id=bH5uHIVtTe)  
**Code**: [https://github.com/mnskim/glow](https://github.com/mnskim/glow)  
**Area**: LLM Agent / Exploration & Decision Making  
**Keywords**: Hard-exploration, LLM Agent, World Memory, Go-Explore, Advantage Learning, Jericho  

## TL;DR
GLoW is proposed to equip LLM agents with a dual-scale textual world memory—combining a "global trajectory frontier" and "local multi-path advantage reflection." It achieves new SOTA among LLM-based methods on sparse-reward hard-exploration tasks in Jericho text games, approaching the performance of the strongest RL methods with $100\times–800\times$ fewer environmental interactions.

## Background & Motivation
**Background**: LLM agents (e.g., ReAct, Reflexion) leverage pre-trained knowledge to excel in robot planning, software engineering, and web automation. However, they still significantly underperform compared to humans in "hard-exploration" problems. These problems are characterized by **huge state-action spaces, deceptive local optima, and sparse rewards**. For instance, *Zork1* in Jericho has a vocabulary of 697 words and instructions up to five words long, resulting in theoretical actions per step as high as $O(697^5)\approx1.64\times10^{14}$, with only a tiny fraction being grammatically and contextually valid.

**Limitations of Prior Work**: Hard exploration poses two core challenges for LLM agents. First, **global learning**: the need to accumulate long-term knowledge of "which discoveries are valuable." ReAct/Reflexion only performs local trial-and-error without a long-term knowledge precipitation mechanism, causing exploration to get trapped in local optima. Second, **local trial-and-error**: the need to quickly refine exploration strategies from sparse feedback. Existing self-reflection essentially estimates Q-values from single trajectories, leading to high variance and frequent errors in causal attribution under sparse rewards. Conversely, RL/MCTS methods (e.g., XTX, MC-DML) are powerful but require hundreds of thousands to millions of interactions, suffering from extremely poor sample efficiency.

**Key Challenge**: Deciding "which state to return to for further exploration" (select) and "how to explore from that state" (explore) both require structured learning from historical experience. **However, these two tasks require different scales**—the former needs a global perspective to judge value peaks and bottlenecks, while the latter requires fine-grained local action progress signals. Conflating the two leads to either blind exploration or knowledge gaps.

**Goal**: To enable LLM agents to possess the exploration capability to continuously break through sparse reward bottlenecks without relying on massive interactions.

**Core Idea**: **Dual-scale textual world memory**. Based on the "select-explore" iterative framework of Go-Explore, the authors split world memory into two layers. At the global scale, a **value-sorted trajectory frontier** is maintained (preserving full temporal contexts of "how it was reached and why it got stuck" for high-value discoveries), where the LLM performs value decomposition for principled state selection. At the local scale, **Multi-path Advantage Reflection (MAR)** is used to infer advantages across multiple trajectories starting from the same point, "densifying" sparse rewards into progress signals to guide exploration. Instead of modeling transition dynamics, world memory encodes exploration experience into structured **textual** representations.

## Method

### Overall Architecture
GLoW follows the select $\leftrightarrow$ explore iterative loop of the Go-Explore family but upgrades the "state archive + heuristics" to dual-scale world memory. In each iteration: ① The global world memory $g_{\text{LLM}}$ analyzes the trajectory frontier $F$, producing a set of key states with value annotations $W_{\text{global}}$, and then $\text{align}_{\text{LLM}}$ selects the most promising state $s_{\text{next}}$ from the archive $A$; ② After returning to that state via action playback, it enters the local exploration phase, sampling $n$ trajectories sequentially, using MAR after each to refine the local world memory $W_{\text{local}}$ to guide the next; ③ New trajectories are fed back into the frontier $F$ and archive $A$.

```mermaid
flowchart TD
    F[Trajectory Frontier F<br/>Value-sorted top-k high-value trajectories] -->|gLLM Value Decomposition| WG[Global World Memory Wglobal<br/>Key states (s, v, v')]
    A[State Archive A] -->|alignLLM Alignment & Selection| SN[Selected State s_next]
    WG --> SN
    SN --> EXP[Local Exploration Phase<br/>Sample n trajectories]
    EXP -->|MAR Multi-path Advantage Reflection| WL[Local World Memory Wlocal<br/>Key state advantage As*]
    WL -->|Guide πexplore| EXP
    EXP -->|Backfill new trajectories| F
    EXP -->|Backfill new states| A
```

### Key Designs

**1. Value-sorted Trajectory Frontier: Replacing isolated state archives with full trajectories.** Traditional Go-Explore stores only discrete states, losing the "action-observation sequence" context, which prevents accurate credit assignment under sparse rewards. GLoW maintains a trajectory frontier $F=\{\tau_1,...,\tau_k\}$, preserving $k$ full trajectories sorted by a value function $v:T\to\mathbb{R}$. The value is defined as the maximum cumulative return within an episode: $v(\tau_i)=\max_{t\in[1,T]}\sum_{j=1}^{t}r_j^i$—this is particularly suitable for the sparse structure of Jericho where negative rewards or deaths may occur midway. New trajectories update the frontier via a sliding window: $F_{t+1}=\text{top-k}(F_t\cup\{\tau_{\text{new}}\},v)$, replacing old trajectories with superior ones. Thus, any state's reachable value can be inferred as $v(s_i)=\max_{\tau\in F,s_i\in\tau}v(\tau)$. By preserving full sequences, the frontier can infer cross-trajectory sequence dependencies obscured by sparse feedback, such as "taking the lamp and sword before entering the cellar" in *Zork1*.

**2. Global Value Decomposition: Bringing UCB exploration-exploitation decomposition into LLM reasoning.** Inspired by the UCB value decomposition $\bar V(s)+c\sqrt{\log(N)/n_s}$, GLoW task $g_{\text{LLM}}$ with analyzing frontier trajectories and annotating each key state with a pair of values $W_{\text{global}}=g_{\text{LLM}}(F)=\{(s_i,v_i,v_i')\}$: $v_i$ is the achieved value (exploitation term), and $v_i'$ is the potential future value inferred by the LLM (exploration reward term). Crucially, $v_i'$ **cannot be derived directly from trajectory scores**—it requires the LLM to reason about "why the trajectory failed and what progress solving the current bottleneck might bring." For example, if multiple high-value trajectories converge but get stuck at a certain state, it indicates an unexplored high-value region behind it, assigning that bottleneck state a high $v_i'$. This effectively implements "optimism in the face of uncertainty" using the LLM's semantic analysis of bottlenecks, replacing statistical rewards with semantic inference. Subsequently, $\text{align}_{\text{LLM}}(s,W_{\text{global}})$ scores each state in the archive, $s_{\text{next}}=\arg\max_s \text{score}[s]$, naturally balancing "proximity to verified high-reward areas" (exploitation) and "proximity to high-potential bottlenecks" (exploration).

**3. Multi-path Advantage Reflection (MAR): From Q-value reflection to advantage reflection.** The authors point out that existing self-reflection is equivalent to estimating Q-values from a single trajectory, which suffers from high variance and misattribution in sparse reward settings. Drawing on the advantage function $A(s,a)=Q(s,a)-V(s)$ and findings from PRM research that "advantages capture progress signals better than Q-values," MAR mimics the TRPO approach of performing multiple rollouts from the same starting point to perform comparative inference across $n$ trajectories. It is an LLM operator that takes the local trajectory set $T_s$ and the frontier $F$ as input, outputting structured text $W_{\text{local}}=\{(s_1^*,A_{s_1^*}),...,(s_k^*,A_{s_k^*})\}$ (typically 2–4 key states). Accurate semantic advantage inference is ensured by two design principles: **multi-trajectory comparison** allows the LLM to aggregate divergent outcomes, identify good/bad actions, and focus on signal-rich key states; using **frontier trajectories as stable references** serves as a "value baseline," using context-based reasoning instead of numerical subtraction to judge if a new trajectory constitutes true progress. This "pseudo-densifies" sparse rewards into progress signals like "one should take the sword first even if there's no immediate reward."

**4. Advantage-driven exploration strategy**. Local world memory enhances exploration through a policy defined by the LLM agent: $\pi_{\text{explore}}(a|s_t,h_t)=\text{Agent}_{\text{LLM}}(s_t,h_t,W_{\text{local}},T_s,F)$, where $h_t$ is the current trajectory history and $T_s$ is the set of trajectories sampled in this phase. The policy leverages both the advantages learned in $W_{\text{local}}$ and the successful strategies from frontier $F$. To handle the exponential action space of Jericho, a hybrid action generation scheme of "free generation + soft constraints on valid Jericho actions" is adopted—which is key to why the authors' reproduced LLM baseline significantly outperforms the near-zero scores reported in previous literature.

## Key Experimental Results

### Main Results
Evaluated on 10 Jericho games, average of 3 runs ± standard deviation. GLoW sets new SOTA among LLM methods on 7/10 games, being globally optimal on 3/10 and runner-up on 5/10.

| Game | XTX (RL, 800k steps) | MC-DML (MCTS, ~400k steps) | ICRL (LLM, 1k steps) | IGE (LLM, 1k steps) | **GLoW (1k steps)** |
|------|------|------|------|------|------|
| Zork1 | 103.4 | 48.66 | 51.7 | 44.3 | **73.0** |
| Deephome | 77.7 | 67.0 | 24.0 | 71.3 | **75.0** |
| Ludicorp | 78.8 | 19.67 | 32.0 | 28.3 | **73.7** |
| Enchanter | 52.0 | 20.0 | 43.3 | 50.0 | **61.7** |
| Ztuu | – | 23.67 | 16.7 | 15.0 | **29.3** |
| Temple | – | 8.0 | 8.0 | 8.0 | **13.0** |
| Balances | 24 | 10.0 | 11.7 | 10.0 | **16.7** |

With only **1,000 interactions**, GLoW approaches the performance of XTX (which uses 800× more interactions) on *Deephome/Ludicorp* and surpasses it on *Enchanter*. Compared to IGE, which is also an LLM-Go-Explore variant, GLoW performs better on 8/10 games.

### Ablation Study
Removing components one by one (examples from *Zork1/Deephome/Ludicorp*):

| Variant | Zork1 | Deephome | Ludicorp | Balances |
|------|------|------|------|------|
| GLoW (Full) | **73.0** | **75.0** | **73.7** | **16.7** |
| ✗ MAR (Local World Memory) | 70.0 | 56.7 | 54.7 | 11.7 |
| ✗ Wglobal (Global Value Decomposition) | 62.0 | 61.3 | 63.3 | 13.3 |
| ✗ Trajectory Frontier F | 61.7 | 57.7 | 63.3 | 11.7 |
| ✗ All (≈IGE + Multi-path Reflection) | 51.3 | 56.0 | 22.0 | 10.0 |
| Standard IGE | 44.3 | 71.3 | 28.3 | 10.0 |

### Key Findings
- **Synergy of three components**: Adding only multi-path reflection to IGE (the "✗ All" variant) shows no clear improvement over IGE, indicating that GLoW's performance stems from the mutual synergy of global value decomposition, trajectory frontiers, and MAR.
- **Advantage learning vs. Self-reflection**: Replacing MAR with Reflexion (also multi-path but performing only single-trajectory reflection) results in significant score drops in most games, verifying that the advantage formulation better utilizes multi-trajectory information.
- **Exploration-Exploitation trade-off of $n$**: A larger $n$ (more exploration per state) leads to deeper local learning, while a smaller $n$ leads to more frequent state selection, making it easier to escape local optima. Experiments show $n=3$ (frontier size $k=5$) achieves the best balance.
- **Scaling to stronger LLMs**: When switching to a more powerful LLM, GLoW surpasses all previous methods on 4 out of 6 difficult/extreme games.

## Highlights & Insights
- **"Textual World Memory" vs. Dynamical World Models**: Treating the LLM world model as an implicit representation of task-related knowledge and using structured text to store exploration experience avoids the difficulty of explicit transition function modeling and naturally adapts to LLM reasoning interfaces.
- **"Semanticizing" classic RL theories**: UCB exploration rewards $\to$ LLM's potential value $v'$ of bottlenecks; numerical baselines for advantage functions $\to$ reference baselines using frontier trajectories. This mapping of "implementing RL principles via LLM reasoning" is highly insightful, with theoretical motivations provided for variance reduction via multi-trajectory comparison.
- **Massive breakthrough in sample efficiency**: Approaching or even surpassing RL methods with millions of interactions using only 1,000 interactions highlights that the potential of LLM agents in hard exploration is significantly underestimated when equipped with the right memory structures.
- **Honest baseline reproduction**: The authors' use of hybrid action generation improved ReAct/Reflexion/ICRL from "near-zero" to performance levels comparable to RL baselines, clarifying prior misconceptions regarding LLM agents' capabilities in Jericho.

## Limitations & Future Work
- **Deterministic environment assumption**: Returning to states relies on action sequence playback, assuming environmental determinism. Extensions for stochastic environments are discussed in the appendix but not empirically tested.
- **LLM calling cost**: Both the select and explore phases rely heavily on LLM reasoning ($g_{\text{LLM}}$, $\text{align}_{\text{LLM}}$, MAR). While environmental interaction is low, API call frequency and token costs are significant.
- **Dependency on valid action constraints**: All methods assume access to the valid action set provided by Jericho. Exploration difficulty would increase in real-world open environments lacking such priors.
- **Domain-specific novelty criteria**: Archive de-duplication utilizes domain-specific novelty, and its cross-task generalizability remains to be verified.
- **Evaluation scope**: Limited to the text-based game Jericho; generalization to continuous or multimodal hard-exploration domains like robotics or web navigation has not yet been explored.

## Related Work & Insights
- **Go-Explore Family**: Original Go-Explore (heuristic selection + random exploration) $\to$ XTX (imitation learning selection + DQN curiosity exploration) $\to$ IGE (LLM used for both stages). GLoW's innovations are principled selection via trajectory frontiers + LLM value decomposition, and local advantage learning via MAR.
- **LLM Agents & Self-reflection**: ReAct, Reflexion, and ICRL provide local trial-and-error and multi-episode memory but lack long-term value structures; GLoW upgrades "self-reflection" to "multi-path advantage reflection."
- **RL Exploration Theory**: UCB/Optimism, advantage functions, multi-rollout TRPO, and progress signals from PRM—these classic ideas are systematically "semanticized" into LLM reasoning, forming the core methodological inspiration of this paper.
- **Insight**: For researchers working on long-horizon LLM agent tasks, "splitting memory into global/local scales and mapping them respectively to state selection and advantage estimation from RL" is a reusable design paradigm, especially in sparse-reward scenarios requiring continuous exploration.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of dual-scale textual world memory and the systematic mapping of UCB value decomposition and advantage functions into LLM reasoning (MAR) is highly novel. While built on the Go-Explore framework, the innovations are clear and self-consistent.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 11 baselines across RL/MCTS/LLM categories, 10 games, 3 repetitions. Ablations decompose components and verify synergies, with additional analysis on $n$ trade-offs and scaling with stronger LLMs. Slightly lacks validation on stochastic or broader domains.
- **Writing Quality**: ⭐⭐⭐⭐ The logic from motivation to theoretical mapping to method and experiments is smooth. Fig 1/2 and the *Zork1* troll example make the abstract mechanisms very intuitive.
- **Value**: ⭐⭐⭐⭐ Approaching the strongest RL with $100\times–800\times$ fewer interactions and refreshing SOTA for LLMs on Jericho hard exploration provides a convincing paradigm and reproducible code for "memory structure-driven LLM agent exploration."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Meta-RL Induces Exploration in Language Agents](meta-rl_induces_exploration_in_language_agents.md)
- [\[ICLR 2026\] Scaling Synthetic Task Generation for Agents via Exploration](scaling_synthetic_task_generation_for_agents_via_exploration.md)
- [\[AAAI 2026\] DEPO: Dual-Efficiency Preference Optimization for LLM Agents](../../AAAI2026/llm_agent/depo_dual-efficiency_preference_optimization_for_llm_agents.md)
- [\[ICLR 2026\] Go-Browse: Training Web Agents with Structured Exploration](go-browse_training_web_agents_with_structured_exploration.md)
- [\[ICLR 2026\] VitaBench: Benchmarking LLM Agents with Versatile Interactive Tasks in Real-world Applications](vitabench_benchmarking_llm_agents_with_versatile_interactive_tasks_in_real-world.md)

</div>

<!-- RELATED:END -->
