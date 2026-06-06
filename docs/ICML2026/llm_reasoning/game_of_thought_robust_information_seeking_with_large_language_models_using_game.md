---
title: >-
  [Paper Note] Game of Thought: Robust Information Seeking with Large Language Models Using Game Theory
description: >-
  [ICML 2026][LLM Reasoning][20 Questions] This paper models LLM active questioning scenarios (20 Questions / medical diagnosis / troubleshooting) as a two-player zero-sum extensive-form game (EFG)…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "20 Questions"
  - "Nash equilibrium"
  - "EFG"
  - "subgame search"
  - "worst-case optimization"
date: 2026-05-08
content_hash: a9e126b1af4828d5
---

# Game of Thought: Robust Information Seeking with Large Language Models Using Game Theory

**Conference**: ICML 2026  
**arXiv**: [2602.01708](https://arxiv.org/abs/2602.01708)  
**Code**: None  
**Area**: LLM Reasoning / Game Theory / Information Search  
**Keywords**: 20 Questions, Nash equilibrium, EFG, subgame search, worst-case optimization

## TL;DR
This paper models LLM active questioning scenarios (20 Questions / medical diagnosis / troubleshooting) as a two-player zero-sum extensive-form game (EFG), and proposes Game of Thought (GoT): using depth-limited subgame construction + CFR to compute Nash equilibria, thereby generating "randomized questioning strategies" that significantly reduce worst-case interaction rounds across all datasets, with a 15–40% improvement over UoT under the weighted variant.

## Background & Motivation

**Background**: Enabling LLMs to proactively clarify and ask questions to fill in missing information is a core capability in agent, medical, and troubleshooting domains. Mainstream approaches include Self-Consistency, Tree-of-Thought, and Uncertainty of Thought (UoT, Hu et al. 2024)—the latter performs depth-limited search in 20 Questions, maximizing expected information gain.

**Limitations of Prior Work**: UoT explicitly assumes "the target item is uniformly sampled," but in reality, the target distribution is often unknown and non-uniform; if an adversary deliberately picks "the hardest-to-guess item," UoT performs poorly in the worst case. In high-risk scenarios like medical diagnosis and troubleshooting, worst-case loss is the true determinant of usability.

**Key Challenge**: To ensure usability in "unknown distribution + high risk" scenarios, worst-case optimization is required; heuristics that maximize information gain only guarantee good mean performance and are not robust to adversaries. Modeling the opponent as an adversary turns the problem into an imperfect-information EFG, but constructing the full game tree is prohibitively expensive (with 25 candidates, building 3763 infosets already takes 5–6 hours).

**Goal**: (1) Define a clean adversarial mathematical model for LLM information search; (2) Design an algorithm under this model that can compute Nash equilibria (NE); (3) Use subgame search to make the algorithm practical in large state spaces.

**Key Insight**: Treat the item selector as a malicious adversary—selecting $s^*$ from $\mathcal{S}$, while the questioner sequentially asks binary questions $q_t$ and receives oracle answers $f(q_t,s^*)$; when $|S(H)|=1$, the game ends and the questioner's cost is $|H|$. This forms a two-player zero-sum EFG, and by the Von Neumann minimax theorem $\min_x \max_y u(x,y)=\max_y \min_x u(x,y)$, a randomized NE always exists; computing the NE is equivalent to optimizing for the worst-case distribution.

**Core Idea**: Formalize the "Strategic Language Search (SLS)" problem as an EFG; inspired by poker bots, use depth-limited subgame search + CFR to approximate NE via the LLM interface.

## Method

### Overall Architecture
The problem is formalized as a quadruple $(\mathcal{S},\mathcal{Q},f,g)$: $\mathcal{S}$ is the set of candidate items, $\mathcal{Q}$ is the set of natural language questions the LLM can generate, $f:\mathcal{Q}\times\mathcal{S}\to\{0,1\}$ is the answer function with the LLM as oracle, and $g$ proposes $m$ candidate questions given the remaining item set. Each GoT round follows four steps: (1) Use the LLM for depth-limited simulation to build a tree along candidate questions; (2) Translate the tree into an EFG subgame; (3) Use CFR (LiteEFG) to compute an approximate NE for the subgame; (4) Sample the next question from the questioner's NE distribution. Repeat until $|S(H)|=1$.

### Key Designs

1. **SLS / SLSR / WSLS Formalization (Game-theoretic framework for LLM information search)**:

    - **Function**: Converts the vague "which question should the LLM ask" problem into a rigorously analyzable two-player zero-sum EFG, making NE computation and worst-case bounds well-defined.
    - **Mechanism**: Define SLS = (S, Q, f): the item chooser privately selects $s^*$, the questioner sequentially asks $q_t$ and observes $a_t=f(q_t,s^*)$, with history $H_t=(Q_t,A_t)$, consistent set $S(H)=\{s:f(Q(\tau),s)=A(\tau)\forall \tau\}$; the game ends when $|S(H)|=1$, and the questioner's cost is $|H|$. SLSR restricts candidate questions to the $m$ outputs of $g(S(H))$ (more realistic, as LLMs can only list a finite set at a time); WSLS assigns weights $w(s)$ to items, with cost $w(s^*)|H|$, capturing "missing a dangerous target is costlier." Theoretically: Theorem 3.7 proves that when $\mathcal{Q}=\mathcal{Q}_\infty$, even-split is NE, verifying that UoT's approximate strategy is only optimal under uniform distribution.
    - **Design Motivation**: Heuristics like UoT lack a reference "optimum," making it unclear how far current methods are from optimal; EFG formalization provides NE as the gold standard for worst-case optimization, and separates solvability proofs (e.g., Theorem 3.6 shows best-response is NP-complete) from approximate algorithms.

2. **Subgame Search: On-demand construction + CFR for local NE**:

    - **Function**: Avoids explicit construction of the full game tree (25 items already require 5–6 hours), instead expanding a fixed-depth subgame only at the current infoset as needed.
    - **Mechanism**: At the current infoset $I(H_t)$, use the LLM to generate $g(S(H_t))$ candidate questions; for each candidate $c_i$, use the LLM as $f$ to partition $S(H_t)$ into $Y(S(H_t),c_i)=\{s:f(c_i,s)=1\}$ and its complement $\bar Y$; recursively simulate $d$ steps to build the simulation tree; when translating to EFG, allow the item chooser to reselect a distribution over $S(H_t)$ at the subgame root (the standard practice for safe subgame search); the payoff at leaf node $l$ is $d(l)-1+h(l)$, with heuristic $h(l):=\log_2(|S(l)|)$ providing an optimistic lower bound; use LiteEFG's CFR to compute an approximate NE, and sample the next question from the questioner's strategy. Theorem 5.1 proves this subgame search is safe (value estimates depend only on $S(H_t)$).
    - **Design Motivation**: Full game tree construction cost grows exponentially with $|\mathcal{S}|$, but humans playing 20Q only look a few steps ahead—subgame search has proven feasible in poker bots; the key to safe subgames is letting the opponent "reselect" at the subgame root, effectively granting more power to the adversary, ensuring that the resulting strategy maintains worst-case guarantees when returned to the original game.

3. **Weighted Variant + Weighted Heuristic (Handling real-world item importance imbalance)**:

    - **Function**: In WSLSR, GoT prioritizes identifying "dangerous targets," avoiding disasters where "few rounds but critical items are missed."
    - **Mechanism**: In the EFG, replace the payoff with $w(s^*)\cdot|H|$, and correspondingly adjust the heuristic to $h(l)=\max_{s\in S(l)} w(s)\cdot(d(l)+\log_2(|S(l)|))$; the questioner's prompt also includes weight information, biasing the LLM to eliminate high-weight items first. Thus, NE computation automatically allocates probability mass to strategies that "identify high-weight items early."
    - **Design Motivation**: UoT's information gain is weight-agnostic, so it degenerates under weighted scenarios; GoT explicitly incorporates weights into the payoff function, and the minimax structure of NE naturally yields "big bets for high weights"—for example, with weights 100 vs 1, GoT always directly asks "is it the high-weight item" first.

### Loss & Training
No LLM is trained; GPT-4.1 / Qwen-2.5-72B are directly used as $f,g$ to provide questions and answers. All "training" occurs at the EFG solving layer (CFR iterations). CFR converges to approximate NE within a few hundred iterations per subgame, negligible compared to LLM call costs; GoT's main computational bottleneck is the multi-second LLM latency.

## Key Experimental Results

### Main Results
Five datasets: 20Q-Common (136 items), 20Q-S128, 20Q-Breeds (25), medical diagnosis DX (100 diseases), troubleshooting FloDial (59 faults). Worst-case interaction length $L_{worst}=\max_{s\in\mathcal{S}}|H^s|$ (lower is better):

| Method | Common (4.1) | S128 (4.1) | Breeds (4.1) | DX (4.1) | FloDial (4.1) |
|--------|-------------|------------|--------------|---------|--------------|
| **GoT** | **10.2** | **11.8** | **7.4** | **12.2** | **7.9** |
| UoT | 11 | 13 | 9 | 13 | 9 |
| DP (Direct Prompting) | 13.8 | 16.2 | 7.8 | 16.8 | 12.7 |
| DC (Direct Choice) | 12.9 | 14.6 | 9.3 | 16.2 | 11.6 |

Results are consistent on Qwen-2.5-72B (GoT always optimal, DX: 10.5 vs UoT 12).

**Weighted variant** (worst-case $\max_s w(s)|H^s|$, lower is better):

| Method | Common | Breeds | DX | FloDial |
|--------|--------|--------|-----|---------|
| **GoT** | **152.1** | **23.2** | **78.3** | **61.4** |
| UoT | 227.4 | 32.1 | 110.0 | 81.0 |
| DP | 224.0 | 36.9 | 116.0 | 90.1 |

GoT improves over UoT by 15–40%; UoT degrades significantly under weighting, as its information gain is weight-agnostic.

### Ablation Study

| Configuration | DX worst-case | Notes |
|---------------|--------------|-------|
| GoT, $d$=3, $m$=3 | 12.2 | Full model |
| GoT, increasing $d$ | Monotonically decreases → plateau | Greater depth approaches "optimal strategy for current candidate set" |
| UoT, increasing $d$ | Almost unchanged | Information gain does not help worst-case optimization |
| WSLSR Breeds, weight skew=100, correct question injected | GoT achieves optimum | Verifies NE optimality |
| WSLSR Breeds, weight skew=100, not injected | GoT performance drops | NE is optimal for candidate set, limited by LLM candidate quality |

### Key Findings
- All methods are still 2–3 rounds away from the theoretical lower bound $\log_2(|\mathcal{S}|)$, indicating that current LLMs struggle to generate "perfectly bisecting" natural language questions—this is the fundamental limit of LLM information search.
- GoT's worst-case advantage increases with weight skew, showing that NE computation's robustness to "unfriendly distributions" is structural, not a trick.
- In average-case, GoT is comparable to UoT (DX experiment), but GoT's worst-case improvement is as large as UoT's average-case improvement over DP—i.e., GoT "significantly improves the tail without sacrificing the mean."
- Under different priors: with a prior $X_{UoT}$ unfavorable to UoT, UoT's average performance drops significantly, while GoT remains almost unchanged; thus, GoT is robust not only in the worst case but also to distributional uncertainty.

## Highlights & Insights
- The "questioning strategy" problem for LLM agents is, for the first time, cleanly formulated as an extensive-form game + Nash equilibrium, providing an analyzable and verifiable worst-case optimization framework, whereas prior work mostly tuned heuristics for the mean.
- "Safe subgame search on the LLM interface" is a natural but rarely explored transfer: poker AI has long solved large game tree approximation, and this is the first clean application to LLM agents; the technical route can be directly transferred to negotiation, dialogue policy, and other LLM decision scenarios.
- The existence of Theorem 3.7 / Theorem 5.1 means the method is not just an engineering implementation but comes with optimality guarantees—rare in LLM reasoning improvement work.
- Using the LLM itself as $g$ to restrict the candidate question set, and then searching for NE within "LLM-expressible questions," is a clever simplification: it reduces the infinite action space to an enumerable set, making CFR solvable.

## Limitations & Future Work
- The method currently only supports strictly binary questions; extension to open-ended answers is not addressed.
- Self-assessment: Each step requires $m^d$ LLM simulation calls (default $d=m=3$ → 27 calls), which is costly; more aggressive pruning or batch calls are needed for latency-sensitive real-time agents.
- Subgame search is safe but not strictly optimal—the deep value outside the subgame is still given by the heuristic $\log_2|S(l)|$, and heuristic bias can propagate to the global strategy; the paper does not quantify this bias.
- WSLSR performance strongly depends on the quality of candidate questions generated by the LLM (as shown in ablation), so GoT's "ceiling" is actually the LLM's own questioning ability; it does not solve the core problem of "how to get LLMs to generate better bisecting questions."
- No comparison of GoT under MCTS / AlphaZero-style search; the trade-off between subgame search and MCTS in LLM agents remains an open question.

## Related Work & Insights
- **vs UoT (Hu et al. 2024)**: UoT assumes uniform distribution to maximize expected info gain; this is shown to be equivalent to Theorem 3.7's NE when $\mathcal{Q}=\mathcal{Q}_\infty$, but in reality, candidate question sets are finite and distributions are unknown, so UoT naturally performs worse in the worst case.
- **vs Tree-of-Thought / Self-Consistency**: Those methods search the "answer space"; GoT searches the "questioning strategy space" and explicitly considers the adversary.
- **vs Poker bot (Libratus / Pluribus)**: Technically inherits subgame search + CFR; the novelty is adapting it to the LLM interface and compressing the state space using $S(H)$.
- **Transferable Insights**: (1) Any "dialogue / inquiry / negotiation" task can be reformulated as EFG NE computation—this paper provides a template; (2) Using the LLM as $g$ to restrict the action space is a general recipe for compressing infinite decision trees to a searchable scale; (3) "Worst-case optimization can be achieved without sacrificing average performance" is a valuable reflection for LLM reasoning research.

## Rating
- Novelty: ⭐⭐⭐⭐ The formalization and implementation of the EFG/NE framework for LLM information search is a first; the method itself is a brilliant reuse of classic game theory rather than a brand-new algorithm.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five datasets × two LLMs × weighted/unweighted × average/worst-case, with solid coverage; lacks comparison with MCTS and RL-finetuned LLMs.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear formalization, seamless flow from definitions to theorems to algorithms to experiments, with examples 1/2/3 making concepts easy to grasp; very smooth to read.
- Value: ⭐⭐⭐⭐ Establishes a benchmark method for worst-case optimization in "LLM information search," with real-world significance for high-risk scenarios like medical diagnosis.

## Related Papers

- [\[ICML 2026\] WZ-LLM：用 Wilf–Zeilberger 符号引导 + LLM 自动证明组合恒等式](automated_formal_proofs_of_combinatorial_identities_via_wilf-zeilberger_guidance.md)
- [\[AAAI 2026\] LLMs for Game Theory: Entropy-Guided In-Context Learning and Adaptive CoT Reasoning](../../AAAI2026/llm_reasoning/llms_for_game_theory_entropy-guided_in-context_learning_and_adaptive_cot_reasoni.md)
- [\[ICML 2025\] Improving Rationality in the Reasoning Process of Language Models through Self-playing Game](../../ICML2025/llm_reasoning/improving_rationality_in_the_reasoning_process_of_language_models_through_self-p.md)
- [\[ICML 2026\] A Formal Comparison Between Chain of Thought and Latent Thought](a_formal_comparison_between_chain_of_thought_and_latent_thought.md)
- [\[ICLR 2026\] Nudging the Boundaries of LLM Reasoning](../../ICLR2026/llm_reasoning/nudging_the_boundaries_of_llm_reasoning.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] LLMs for Game Theory: Entropy-Guided In-Context Learning and Adaptive CoT Reasoning](../../AAAI2026/llm_reasoning/llms_for_game_theory_entropy-guided_in-context_learning_and_adaptive_cot_reasoni.md)
- [\[ICML 2026\] DecepChain: Inducing Deceptive Reasoning in Large Language Models](decepchain_inducing_deceptive_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Stratagem: Learning Transferable Reasoning via Trajectory-Modulated Game Self-Play](../../ACL2026/llm_reasoning/stratagem_learning_transferable_reasoning_via_trajectory-modulated_game_self-pla.md)
- [\[AAAI 2026\] Chain-of-Thought Driven Adversarial Scenario Extrapolation for Robust Language Models](../../AAAI2026/llm_reasoning/chain-of-thought_driven_adversarial_scenario_extrapolation_for_robust_language_m.md)
- [\[ICML 2026\] SmartThinker: Progressive Chain-of-Thought Length Calibration for Efficient Large Language Model Reasoning](smartthinker_progressive_chain-of-thought_length_calibration_for_efficient_large.md)

</div>

<!-- RELATED:END -->
