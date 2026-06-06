---
title: >-
  [Paper Note] Game of Thought: Robust Information Seeking with Large Language Models Using Game Theory
description: >-
  [ICML 2026][Reinforcement Learning][20 Questions] This paper models the proactive LLM questioning scenario (20 Questions / medical diagnosis / troubleshooting) as a two-player zero-sum extensive-form game (EFG) and propo…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "20 Questions"
  - "Nash equilibrium"
  - "EFG"
  - "subgame search"
  - "worst-case optimization"
date: 2026-05-08
content_hash: 8f3b7e5d28c6adc5
---

# Game of Thought: Robust Information Seeking with Large Language Models Using Game Theory

**Conference**: ICML 2026  
**arXiv**: [2602.01708](https://arxiv.org/abs/2602.01708)  
**Code**: None  
**Area**: LLM Reasoning / Game Theory / Information Seeking  
**Keywords**: 20 Questions, Nash equilibrium, EFG, subgame search, worst-case optimization

## TL;DR
This paper models the proactive LLM questioning scenario (20 Questions / medical diagnosis / troubleshooting) as a two-player zero-sum extensive-form game (EFG) and proposes Game of Thought (GoT). By using depth-limited subgame construction and CFR to compute Nash equilibria, GoT generates "randomized questioning strategies" that significantly reduce worst-case interaction rounds across all datasets, achieving a 15–40% improvement over UoT in weighted variants.

## Background & Motivation

**Background**: The ability of LLMs to proactively clarify and ask questions to complete information is core to agents, medical diagnosis, and troubleshooting. Current mainstream approaches include prompt-based searches like Self-Consistency and Tree-of-Thought, as well as Uncertainty of Thought (UoT) proposed by Hu et al. 2024, which performs limited-depth search to maximize expected information gain in 20 Questions.

**Limitations of Prior Work**: UoT explicitly assumes that target items are drawn from a uniform distribution. However, in real-world scenarios, the target distribution is often unknown and non-uniform; if an opponent deliberately selects the "hardest to guess" item, UoT's worst-case performance becomes poor. In high-risk scenarios like medical diagnosis or troubleshooting, losses in the worst-case are the metrics that truly determine usability.

**Key Challenge**: To ensure usability under "unknown distribution + high risk" scenarios, optimization for the worst-case is required. Heuristics that maximize information gain only guarantee average performance and are not robust against an adversary. Simultaneously, modeling the opponent as an adversary makes the problem an imperfect-information EFG, where the cost of constructing a complete game tree is prohibitively high (e.g., constructing 3,763 infosets for 25 candidates takes 5–6 hours).

**Goal**: (1) Define a clean adversarial mathematical model for LLM information seeking; (2) Design an algorithm to compute the Nash Equilibrium (NE) under this model; (3) Implement the algorithm using subgame search for large state spaces.

**Key Insight**: The item chooser is viewed as a malicious opponent who selects $s^*$ from $\mathcal{S}$. The questioner sequentially asks binary questions $q_t$, answered by an oracle $f(q_t, s^*)$. The game ends when $|S(H)|=1$, with the questioner's cost being $|H|$. This constitutes a two-player zero-sum EFG. According to Von Neumann’s minimax theorem $\min_x \max_y u(x,y)=\max_y \min_x u(x,y)$, a randomized NE must exist, and finding the NE is equivalent to optimizing for the worst-case distribution.

**Core Idea**: Formalize the "Strategic Language Search (SLS)" problem as an EFG. Inspired by poker bots, use depth-limited subgame search + CFR via LLM interfaces to approximate the NE.

## Method

### Overall Architecture
The problem is formalized as a quadruplet $(\mathcal{S}, \mathcal{Q}, f, g)$: where $\mathcal{S}$ is the set of candidate items, $\mathcal{Q}$ is the set of natural language questions the LLM can generate, $f:\mathcal{Q}\times\mathcal{S}\to\{0,1\}$ is the answer function (where the LLM acts as the oracle), and $g$ is the "proposal of $m$ candidate questions given the remaining item set." The GoT 4-step loop per round consists of: (1) Using the LLM for depth-limited simulation to construct a tree along candidate questions; (2) Translating the tree into an EFG subgame; (3) Computing the approximate NE of the subgame using CFR (LiteEFG); (4) Sampling the next question from the questioner's NE distribution. This repeats until $|S(H)|=1$.

### Key Designs

1.  **SLS / SLSR / WSLS Formalism (Game-theoretic framework for LLM info search)**:
    -   **Function**: Transforms the vague problem of "what question should the LLM ask" into a rigorously analyzable two-player zero-sum EFG, making NE solutions and worst-case upper bounds definable.
    -   **Mechanism**: Defines $SLS = (S, Q, f)$ where the item chooser privately selects $s^*$, and the questioner sequentially asks $q_t$ and observes $a_t=f(q_t,s^*)$. Given history $H_t=(Q_t,A_t)$, the consistent set $S(H)=\{s:f(Q(\tau),s)=A(\tau)\forall \tau\}$. The game ends when $|S(H)|=1$, and the questioner cost = $|H|$. $SLSR$ limits candidate questions to $m$ outputs from $g(S(H))$ (more realistic). $WSLS$ assigns a weight $w(s)$ to each item, with cost = $w(s^*)|H|$, characterizing the "higher cost of missing critical targets." Theorem 3.7 proves that even-split is the NE when $\mathcal{Q}=\mathcal{Q}_\infty$, verifying that UoT's heuristic is only optimal under uniform distribution constraints.
    -   **Design Motivation**: Heuristics like UoT lack an "optimum" benchmark, making it impossible to evaluate how far current methods are from the theoretical best. The EFG formalization provides NE as the gold standard for worst-case optimization and separates solvability proofs (e.g., Theorem 3.6 proving NP-completeness of best-response) from approximation algorithms.

2.  **Subgame Search: On-demand construction + CFR for local NE**:
    -   **Function**: Avoids explicit construction of the full game tree by expanding a fixed-depth subgame only upon reaching the current infoset.
    -   **Mechanism**: At the current infoset $I(H_t)$, the LLM generates $g(S(H_t))$ candidate questions. For each candidate $c_i$, the LLM acting as $f$ splits $S(H_t)$ into $Y(S(H_t),c_i)=\{s:f(c_i,s)=1\}$ and its complement $\bar Y$. A simulation tree is built by recursing $d$ steps. When translated to an EFG, the item chooser is allowed to pick a new distribution over $S(H_t)$ at the subgame root (standard safe subgame search). The payoff for leaf node $l$ is $d(l)-1+h(l)$, with a heuristic $h(l):=\log_2(|S(l)|)$ providing an optimistic lower bound. LiteEFG's CFR is used to find the approximate NE, and the next question is sampled. Theorem 5.1 proves this subgame search is "safe" (value estimation depends only on $S(H_t)$).
    -   **Design Motivation**: The cost of full game tree construction is exponential to $|\mathcal{S}|$, but humans only look a few steps ahead in 20 Questions. Subgame search has been proven effective in poker bots. The key to safe subgame search is allowing the opponent to "re-select" at the root, granting more power to the adversary, which ensures that the resulting strategy maintains a worst-case bound when applied back to the full game.

3.  **Weighted variant + Weighted Heuristic (Handling non-uniform item importance)**:
    -   **Function**: Enables GoT to prioritize identifying "high-risk targets" in WSLSR, avoiding disasters where total rounds are low but critical items are missed.
    -   **Mechanism**: The EFG payoff is replaced by $w(s^*)\cdot|H|$, and the heuristic becomes $h(l)=\max_{s\in S(l)} w(s)\cdot(d(l)+\log_2(|S(l)|))$. Weight information is included in the questioning prompt, biasing the LLM towards excluding high-weight items first. NE solving automatically shifts probability mass toward strategies that identify high-weight items early.
    -   **Design Motivation**: UoT's information gain is weight-agnostic and thus degrades to uniform behavior in weighted scenarios. GoT explicitly incorporates weights via the payoff function; combined with the minimax structure of the NE, it naturally yields "high-stakes" strategies (e.g., if weight=100 vs 1, GoT will immediately ask if it is the high-weight item).

### Loss & Training
The study does not train the LLM. It directly uses GPT-4.1 / Qwen-2.5-72B as $f, g$ for questions and answers. All "training" occurs in the EFG solving layer (CFR iterations). CFR converges to an approximate NE within a few hundred iterations per subgame, with negligible cost compared to LLM calls. The primary computational bottleneck remains the latency of LLM API calls.

## Key Experimental Results

### Main Results
The authors evaluate 5 datasets: 20Q-Common (136 items), 20Q-S128, 20Q-Breeds (25), Medical diagnosis DX (100 diseases), and Troubleshooting FloDial (59 faults). Metric: worst-case interaction length $L_{worst}=\max_{s\in\mathcal{S}}|H^s|$ (lower is better):

| Method | Common (4.1) | S128 (4.1) | Breeds (4.1) | DX (4.1) | FloDial (4.1) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GoT** | **10.2** | **11.8** | **7.4** | **12.2** | **7.9** |
| UoT | 11 | 13 | 9 | 13 | 9 |
| DP (Direct Prompting) | 13.8 | 16.2 | 7.8 | 16.8 | 12.7 |
| DC (Direct Choice) | 12.9 | 14.6 | 9.3 | 16.2 | 11.6 |

Results on Qwen-2.5-72B are consistent (GoT optimal across all, e.g., 10.5 vs UoT's 12 on DX).

**Weighted variant** (worst-case $\max_s w(s)|H^s|$, lower is better):

| Method | Common | Breeds | DX | FloDial |
| :--- | :--- | :--- | :--- | :--- |
| **GoT** | **152.1** | **23.2** | **78.3** | **61.4** |
| UoT | 227.4 | 32.1 | 110.0 | 81.0 |
| DP | 224.0 | 36.9 | 116.0 | 90.1 |

GoT shows 15–40% improvement over UoT; UoT degrades significantly in weighted scenarios because its information gain is weight-agnostic.

### Ablation Study

| Config | DX worst-case | Description |
| :--- | :--- | :--- |
| GoT, $d$=3, $m$=3 | 12.2 | Full model |
| GoT, increase $d$ | Monotonic decrease → Plateau | Deeper depth nears optimal strategy for given candidate set |
| UoT, increase $d$ | Nearly static | Info gain does not assist in worst-case optimization |
| WSLSR Breeds, inject correct qu | GoT reaches optimal | Verifies optimality of NE solver |
| WSLSR Breeds, no injection | GoT performance drops | NE is limited by the quality of LLM candidate questions |

### Key Findings
- All methods are 2–3 rounds away from the theoretical lower bound $\log_2(|\mathcal{S}|)$, indicating that actual LLMs struggle to generate "perfectly bisecting" natural language questions—this is an inherent upper bound for current LLM search.
- GoT's worst-case advantage grows with weight skewness, proving the robustness of NE solvers is structural, not a trick.
- On average-case performance, GoT is comparable to UoT (DX experiment), but the magnitude of GoT's worst-case improvement matches UoT's average-case improvement over DP—meaning GoT "significantly improves the tail without sacrificing the mean."
- Average performance under different priors: Under unfavorable priors $X_{UoT}$, UoT's average performance drops significantly while GoT remains stable. This shows GoT is robust to distribution uncertainty as well as worst-case scenarios.

## Highlights & Insights
- This is the first work to cleanly formalize the LLM agent "questioning strategy" as an extensive-form game and Nash Equilibrium, providing a verifiable framework for worst-case optimization beyond average-case heuristics.
- Applying "Safe subgame search to LLM interfaces" is a natural yet novel migration. Poker AI solved approximate solving for large game trees; this work provides the first clean template for applying these tools to LLM agents in negotiation, dialogue policy, etc.
- The presence of Theorem 3.7 / Theorem 5.1 elevates the method from engineering implementation to a framework with optimality guarantees, which is rare in LLM reasoning works.
- Using the LLM as $g$ to limit the candidate set, then searching for the NE within the "LLM-expressible" range, is a clever simplification that compresses an infinite action space into a searchable size.

## Limitations & Future Work
- Authors admit it only supports strictly binary questions; extensions to open-ended answers remain future work.
- Scaling: Each step requires $m^d$ LLM simulation calls (default $d=m=3 \to 27$ calls), which is costly. Latency-sensitive agents may need aggressive pruning or batch calls.
- Subgame search is safe but not strictly optimal—deep values outside the subgame rely on the $\log_2|S(l)|$ heuristic. Heuristic bias propagates to global strategy, which the paper did not quantify.
- WSLSR performance is heavily dependent on the quality of candidate questions from the LLM. GoT's "ceiling" is effectively the LLM's own questioning capability; it does not solve the root problem of "how to make LLMs generate better bisecting questions."
- There is no comparison with MCTS / AlphaZero style searches; the trade-off between subgame search and MCTS in LLM agents remains an open question.

## Related Work & Insights
- **vs UoT (Hu et al. 2024)**: UoT assumes uniform distributions to maximize expected info gain. This paper proves that is equivalent to the NE when $\mathcal{Q}=\mathcal{Q}_\infty$ (Theorem 3.7), but UoT's worst-case is naturally worse given finite sets and unknown distributions.
- **vs Tree-of-Thought / Self-Consistency**: Those methods search the "answer space"; GoT searches the "questioning strategy space" while explicitly considering an adversary.
- **vs Poker bots (Libratus / Pluribus)**: Technically inherits subgame search + CFR; the novelty lies in adapting it to LLM interfaces and compressing the state space using $S(H)$.
- **Transferable Insights**: (1) Any "dialogue / inquiry / negotiation" task can be rewritten as an EFG to find an NE. (2) Using LLMs as action set generators is a general recipe for compressing infinite decision trees. (3) "Worst-case optimization doesn't have to lose average performance" is a valuable conclusion for LLM reasoning research.

## Rating
- Novelty: ⭐⭐⭐⭐ The formalization of EFG/NE for LLM information seeking is a first; the method is a brilliant reuse of classical game theory.
- Experimental Thoroughness: ⭐⭐⭐⭐ 5 datasets × 2 LLMs × Weighted/Unweighted × Mean/Worst-case; lacks comparison with MCTS or RL-finetuned LLMs.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear formalization, seamless flow from definition to algorithm to experiment. Examples make concepts easy to grasp.
- Value: ⭐⭐⭐⭐ Establishes a benchmark for worst-case optimization in LLM info seeking, significant for high-risk domains like medical diagnosis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Pruning as a Cooperative Game: Surrogate-Assisted Layer Contribution Estimation for Large Language Models](../../ICLR2026/reinforcement_learning/remix_reinforcement_routing_for_mixtures_of_loras_in_llm_finetuning.md)
- [\[ACL 2026\] Understanding Generalization in Role-Playing Models via Information Theory](../../ACL2026/reinforcement_learning/understanding_generalization_in_role-playing_models_via_information_theory.md)
- [\[ICML 2026\] The Shape of Reasoning: Topological Analysis of Reasoning Traces in Large Language Models](the_shape_of_reasoning_topological_analysis_of_reasoning_traces_in_large_languag.md)
- [\[AAAI 2026\] Distributionally Robust Online Markov Game with Linear Function Approximation](../../AAAI2026/reinforcement_learning/distributionally_robust_online_markov_game_with_linear_function_approximation.md)
- [\[ICLR 2026\] Robust Multi-Objective Controlled Decoding of Large Language Models](../../ICLR2026/reinforcement_learning/robust_multi-objective_controlled_decoding_of_large_language_models.md)

</div>

<!-- RELATED:END -->
