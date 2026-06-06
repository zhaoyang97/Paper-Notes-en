---
title: >-
  [Paper Note] Scaling Small Agents Through Strategy Auctions
description: >-
  [ICML 2026][LLM Agent][strategy auctions] The paper proposes SALE (Strategy Auctions for Workload Efficiency): heterogeneous Qwen3 agents submit "short strategy plans" as bids for each task. Winners are selected based on…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "strategy auctions"
  - "heterogeneous agent routing"
  - "freelance market"
  - "test-time self-improvement"
  - "deep search"
date: 2026-05-08
content_hash: d971f696122ea767
---

# Scaling Small Agents Through Strategy Auctions

**Conference**: ICML 2026  
**arXiv**: [2602.02751](https://arxiv.org/abs/2602.02751)  
**Code**: TBD  
**Area**: LLM Agent / Multi-agent Routing  
**Keywords**: strategy auctions, heterogeneous agent routing, freelance market, test-time self-improvement, deep search

## TL;DR
The paper proposes SALE (Strategy Auctions for Workload Efficiency): heterogeneous Qwen3 agents submit "short strategy plans" as bids for each task. Winners are selected based on cost-minus-value, while historical auction memory allows cheaper agents to continuously refine their bids. On deep search and coding tasks, SALE exceeds the pass@1 of the largest model while reducing dependence on the largest agent by 52% and total overhead by 35%.

## Background & Motivation
**Background**: There is general industry optimism that "small models + tools" can replace large models in agentic workflows, assuming small LLM agents are sufficient once they outsource reasoning to environments and tools.

**Limitations of Prior Work**: The authors conducted fine-grained evaluations using Qwen3 (4B/8B/14B/32B) on deep search and coding tasks along the "human solving time" axis $\tau(t)$. They found that on simple tasks, the smallest agent achieves approximately 87–92% of the largest agent's pass@1, but this drops to only 17–25% for the most complex tasks ($\tau \leq 60$ minutes). Small agent performance does not scale with task complexity, yet relying solely on "large model fallback" results in significant compute waste for simple tasks.

**Key Challenge**: Existing routing strategies face a dilemma. Non-predictive routing (running all models to completion before selection) incurs explosive costs in agentic scenarios (trajectories can reach millions of tokens). Predictive routing (learning an additional small router) requires specialized training, is tied to specific model sets, degrades on difficult tasks, and lacks test-time self-improvement capabilities.

**Goal**: Design a routing mechanism that simultaneously satisfies: ① almost negligible inference overhead; ② plug-and-play compatibility with any off-the-shelf agent; ③ sustained accuracy on long-range tasks; ④ ability for small agents to grow stronger with use, gradually assuming more workload.

**Key Insight**: Borrow from the freelance market—hirers post tasks, freelancers bid with short "how I plan to do it" proposals, and the platform selects winners based on price/quality scores; losers upgrade their own proposals by observing past cases. Sun et al. (2024) and others have proven that plan quality is strongly correlated with execution quality, making plan-as-bid both informative and low-cost.

**Core Idea**: Arrange heterogeneous agents into a test-time auction market where bids are short strategy plans rather than full solutions. The winner is selected via cost-minus-value. Memory of "past winning/losing bids" drives the self-iteration of small agents, merging task routing and self-improvement.

## Method
### Overall Architecture
For each task $t$ and set of heterogeneous agents $\mathcal{A} = \{a_i\}_{i=1}^{|\mathcal{A}|}$ (four Qwen3 sizes), the SALE pipeline is: ① Each agent $a_i$ generates a short strategy $s_{t,i}$ as a bid based on $(t, \text{environment } E)$; ② Cost $C_{t,i}$ is estimated using price signal $\pi(a_i)$ and bid length, while value $V_{t,i}$ is estimated using entropy + self-evaluation + cross-evaluation jury; ③ The agent with the minimum cost-minus-value is chosen as the provisional winner; ④ Any agent cheaper than the provisional winner retrieves "losing/winning strategy pairs" for similar tasks from shared memory $\mathcal{M}$, performs contrastive refinement, and re-bids. If any refined bid yields a lower $C-V$, the winner is replaced; ⑤ The final winner executes its strategy to generate the full trajectory. The entire auction phase accounts for < 1% of total inference tokens and latency.

### Key Designs

1.  **Strategy Bidding**:
    - **Function**: Each agent outputs a short plan detailing "how I intend to decompose the task, which tools to use, and potential pitfalls" as input for the routing decision.
    - **Mechanism**: Leveraging the positive correlation between plan-quality and execution-quality, expensive non-predictive routing is replaced by partially predictive routing where everyone outputs a few hundred tokens. The bid acts as a quality signal and a roadmap; the winner executes without re-planning.
    - **Design Motivation**: Agentic trajectories often span hundreds of thousands of tokens; running all agents to completion is neither feasible nor green. Pure task-description routing fails on long-range tasks. Short strategies are the lowest-cost carriers of "price-quality" signals.

2.  **Cost-Value Scoring & Min-Max Weight Learning**:
    - **Function**: Maps "whether $a_i$ should do task $t$" to a scalar score $C_{t,i} - V_{t,i}$, where smaller is better.
    - **Mechanism**: Cost is defined as $C_{t,i} = w_c \cdot \pi(a_i) \cdot |s_{t,i}|$, multiplying price per million tokens by bid length—long strategies indicate both longer trajectories (Goebel & Zips 2025) and higher failure rates (Xiong et al. 2025a), acting as a dual cost-risk proxy. Value is defined as $V_{t,i} = w_h \cdot H(s_{t,i}) + \sum_{a_j \in \mathcal{A}} w_j \cdot \gamma_j(s_{t,i})$, where $H$ is token-level average entropy (high entropy corresponds to high information/low redundancy) and $\gamma_j \in \{0, \dots, 5\}$ are jury scores (including self-eval). Weights are learned via min-max optimization on a training set: $\min_{w,x,Q} Q\ \text{s.t.}\ z_t \leq Q\ \forall t$, minimizing the $C-V$ of the worst-case task.
    - **Design Motivation**: Cost signals must be "cheaply obtainable," while value must balance intrinsic quality (entropy) and extrinsic recognition (jury). Ablations show removing self-eval or shrinking the jury size drops performance; min-max is more robust for tail tasks than mean loss.

3.  **Strategy Refinement from Auction Memory**:
    - **Function**: Allows losing cheap agents to rewrite their bids after seeing historical "losing vs winning" cases to compete for tasks against large models.
    - **Mechanism**: After auctions, $\mathcal{M}(t') = (t', \{s_{t',i}\}, y_{t'})$ (with winning/losing labels) is stored. For a new task $t$, refinement is triggered only for agents cheaper than the provisional winner. They retrieve top-$\tilde{k}$ (lose, win) strategy pairs based on cosine similarity, use a contrastive prompt to learn "why the rival won," and output a refined bid $s^r_{t,i}$.
    - **Design Motivation**: Forcing memory-informed bids for all agents would multiply token costs. Opportunistic refinement keeps a shortcut if the small agent already won, focusing extra cost on cheap agents with "upset potential." This upgrades routing into a continuous self-improvement mechanism.

### Loss & Training
No routing or refinement networks were trained; off-the-shelf Qwen3 models were used throughout. The only "learning" is the min-max scalar optimization of $w = (w_c, w_h, \{w_j\})$ (a MIP with big-M constraints, see Appendix D), fitted on a training subset.

## Key Experimental Results

### Main Results
Evaluated on HST-Bench (753 tasks, 5 complexity bins based on human solving time $\tau(t)$); agent pool: Qwen3 4B/8B/14B/32B; price: $\$0.05 / \$0.09 / \$0.16 / \$0.36$ per Mt. SALE figures are averages of 5 random task orders.

| Domain | Setting | 32B agent pass@1 | SALE pass@1 | SALE Gain | SALE $/Mt | 32B $/Mt | Cost Reduction |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Deep search | All | 63.8 | 67.3 | +3.5 | 0.21 | 0.36 | -42% |
| Deep search | $\tau \leq 0.1$ (Simplest) | 87.5 | 91.3 | +3.8 | 0.22 | 0.36 | -39% |
| Deep search | $\tau \leq 60$ (Hardest) | 12.5 | 16.3 | +3.8 | 0.23 | 0.36 | -36% |
| Coding | $\tau \leq 0.1$ | 95.0 | 98.3 | +3.3 | 0.18 | 0.36 | -50% |
| Coding | $\tau \leq 0.5$ | 79.7 | 82.0 | +2.3 | — | 0.36 | — |

Overall, SALE reduces dependence on the largest agent by 65% in deep search and 40% in coding (-52% cross-domain); total costs were reduced by 42% in deep search and 25% in coding (-35% combined).

### Ablation Study
| Configuration | Performance/Phenomenon | Note |
| :--- | :--- | :--- |
| Any single Qwen3 agent | Dominated by SALE in both pass@1 and $/Mt | SALE pushes the Pareto frontier outward. |
| Off-the-shelf predictive router | Either pass@1 lower than 32B or saves no money | Task descriptions are wrong signals in agentic scenarios. |
| Removing jury self-eval / jury size | pass@1 decreases (Appendix I) | Mixed self/cross-eval jury is indispensable. |
| Removing entropy as value | pass@1 decreases | High-entropy plan $\leftrightarrow$ superior planning. |
| No memory refinement | Small agent win rate fails to grow over time | Refinement is key to "scaling up small agents." |
| Overhead of auction | < 1% of total inference cost | Negligible overhead. |

### Key Findings
- pass@1 decreases monotonically with $\tau(t)$ across all sizes, proving HST-Bench human-time scales align with LLM difficulty.
- Large agents do not offset their price with "shorter trajectories"—traces are only slightly shorter for simple tasks and often longer for complex tasks.
- As auction memory grows, the win rate of small agents rises significantly, replicating "freelancer momentum" dynamics.

## Highlights & Insights
- Shifting routing signals from "training a router model" to "agent-generated plans" is lightweight and low-resistance; this is "partially predictive routing."
- Using plan length as a proxy for both cost and risk is a practical dual-purpose variable.
- The asymmetric design (only cheap agents refine) couples "small model improvement" with "token control," a great example of system-level design for test-time self-improvement.
- Min-max weight learning is friendlier to tail tasks, suggesting agent routing should avoid simple mean loss.

## Limitations & Future Work
- The agent pool was limited to 4 Qwen3 sizes; jury robustness across families (e.g., Qwen × Llama) is untested.
- The $O(|\mathcal{A}|^2)$ jury evaluation creates a bottleneck for large agent pools, requiring sparse or hierarchical juries.
- Evaluation relies on LLM-as-judge; domains like coding could benefit from stricter unit-test execution-level evaluation.
- Memory stores strategy pairs but not trace errors; harder tasks might require trace-level memory.

## Related Work & Insights
- **vs Predictive Router (Hu et al. 2024)**: These require independent networks and fail on hard tasks; SALE uses "self-submitted plans" and plug-and-play capability.
- **vs Non-predictive Router (Chen et al. 2024)**: All runners finish before selection; SALE truncates this via plans, keeping overhead < 1%.
- **vs Agent scaling (Kwa et al. 2025 / Sinha et al. 2026)**: SALE moves from "single agent scaling" to "system-level scaling," breaking single-model Pareto limits through market structure.
- **vs Agent virtual economy (Tomasev et al. 2025)**: SALE concretizes abstract "agent economies" into quantifiable "workload efficiency" with an end-to-end implementation.
- **vs Memory-driven agent**: Traditional memory enhances single agent reasoning; SALE uses memory as a "market feedback signal" to reallocate labor across models.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Combining "strategy bits + auction memory" for self-improvement is a unique perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive tasks and ablations, though multi-family evidence is lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, formulas, and cross-references.
- Value: ⭐⭐⭐⭐⭐ Provides a replicable system-level answer for industrial multi-model agent deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Scaling, Benchmarking, and Reasoning of Vision-Language Agents for Mobile GUI Navigation](scaling_benchmarking_and_reasoning_of_vision-language_agents_for_mobile_gui_navi.md)
- [\[NeurIPS 2025\] AgentTTS: Large Language Model Agent for Test-time Compute-optimal Scaling Strategy in Complex Tasks](../../NeurIPS2025/llm_agent/agenttts_large_language_model_agent_for_testtime_computeopti.md)
- [\[ICML 2026\] EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle](evolver_self-evolving_llm_agents_through_an_experience-driven_lifecycle.md)
- [\[ICML 2026\] Weasel: Domain-Generalizable Web Agents through Importance-Diversity Data Selection](weasel_out-of-domain_generalization_for_web_agents_via_importance-diversity_data.md)
- [\[ACL 2026\] Polaris: A Gödel Agent Framework for Small Language Models through Experience-Abstracted Policy Repair](../../ACL2026/llm_agent/polaris_a_gödel_agent_framework_for_small_language_models_through_experience-abs.md)

</div>

<!-- RELATED:END -->
