---
title: >-
  [Paper Note] RL4RLA: Teaching ML to Discover Randomized Linear Algebra Algorithms Through Curriculum Design and Graph-Based Search
description: >-
  [ICML 2026][Reinforcement Learning][Curriculum Learning] RL4RLA utilizes a "numerical curriculum of increasing difficulty + Monte Carlo Graph Search (MCGS)" to drive an RL agent in composing interpretable randomized nume…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Curriculum Learning"
  - "Monte Carlo Graph Search"
  - "Symbolic Program Synthesis"
  - "Sketching Algorithms"
  - "Preconditioners"
date: 2026-05-08
content_hash: 1d68362b0bec8e92
---

# RL4RLA: Teaching ML to Discover Randomized Linear Algebra Algorithms Through Curriculum Design and Graph-Based Search

**Conference**: ICML 2026  
**arXiv**: [2605.18004](https://arxiv.org/abs/2605.18004)  
**Code**: https://github.com/Tim-Xiong/RL4RLA  
**Area**: Reinforcement Learning / Algorithm Discovery / Randomized Numerical Linear Algebra  
**Keywords**: Curriculum Learning, Monte Carlo Graph Search, Symbolic Program Synthesis, Sketching Algorithms, Preconditioners  

## TL;DR
RL4RLA utilizes a "numerical curriculum of increasing difficulty + Monte Carlo Graph Search (MCGS)" to drive an RL agent in composing interpretable randomized numerical linear algebra (RLA) algorithms from primitives. It successfully rediscovering classic methods such as Sketch-and-Precondition, Randomized Kaczmarz, and Newton Sketch.

## Background & Motivation
**Background**: Works like AlphaTensor, AlphaDev, FunSearch, and AlphaEvolve have advanced "algorithm discovery through search" in domains like matrix multiplication, sorting, and mathematical theorems. However, RLA algorithms (e.g., sketching, leverage-score sampling, stochastic Krylov), which underpin large-scale scientific computing, have long relied on manual design by numerical analysts, lacking a general automated discovery framework.

**Limitations of Prior Work**: LLM-driven methods (FunSearch / AlphaEvolve / AlgoTune) depend heavily on pre-training distributions and excel at "local optimization of existing implementations" (library replacement, JIT) rather than assembling new structures from scratch. Additionally, RLA algorithms are inherently multi-step with sparse rewards: a solution like Blendenpik requires sequentially combining 5–7 primitives (sketch → QR → preconditioner construction → iterative refinement). Using vanilla RL to find signals in an exponential program space is nearly impossible.

**Key Challenge**: The "compositional depth" of high-performance RLA algorithms is positively correlated with the "reward sparsity" of RL search—the algorithms most worth discovering are those lacking intermediate rewards to guide search convergence.

**Goal**: (i) Ensure search results are interpretable symbolic programs rather than black boxes; (ii) decompose the discovery of multi-step compositional algorithms into steps with sufficient local signals; (iii) create a reusable search engine for RLA primitives like sketching, preconditioning, and importance sampling.

**Key Insight**: The authors identify "compositional patterns" in RLA—most high-performance RLA algorithms can be written in a two-part structure: setup (sketch, factorize) + iteration (preconditioned update). Furthermore, a natural "difficulty ladder" exists among classical algorithms: Landweber → GD → Preconditioned GD → Sketched Preconditioned GD → Subsampling → Leverage-Score Sampling, where each step adds a single primitive to resolve a numerical failure mode exposed by the previous step.

**Core Idea**: Algorithm discovery is modeled as sequential decision-making via MCGS on a Directed Acyclic Graph (DAG) of symbolic programs. A curriculum based on "numerical failure modes" teaches the agent to add primitives incrementally, compressing the exponential search space into several shallow local problems.

## Method

### Overall Architecture
RL4RLA represents each candidate algorithm as an explicit symbolic program $\mathcal{A}=(\mathcal{P}_{\text{setup}},\mathcal{P}_{\text{iteration}})$, where the setup segment performs one-time preprocessing and the iteration segment defines updates. Programs are constructed by inserting primitives of the form "`target ← operator(operand_1, operand_2)`". The primitive library includes SKETCH, HHQR, MATVEC, INV, etc. A type system ensures executable programs, followed by automatic dead-code elimination. The curriculum splits the global search into $S$ stages $(\mathcal{C}_s)_{s=1}^{S}$, where each stage $\mathcal{C}_s=(A_s,b_s,\mathbf{w}_s)$ specifies a family of linear systems and reward weights. Search at stage $s$ hot-starts from the best algorithm found at $s-1$, learning only one new primitive. Each candidate undergoes MCGS Selection → Expansion → Simulation → Backpropagation. During simulation, the program is executed on random problem instances; rewards are weighted based on residual, convergence monotonicity, complexity, and condition number. LUCB adaptive stopping determines stage completion.

### Key Designs

1.  **Numerical Curriculum**:
    - **Function**: Decomposes the sparse reward problem of finding a 6-step algorithm into 5 local searches, each inserting one primitive into the previous algorithm.
    - **Mechanism**: A hand-designed ladder of problem instances is used. Each step introduces **one** numerical failure mode: starting from a $5\times 5$ well-conditioned system (solvable by Landweber), expanding to $m\times n$ rectangular matrices (requiring normal equations), increasing to $10000\times 50$ ill-conditioned systems (forcing preconditioning $M=R$ such that $\kappa(AR^{-1})\approx 1$), increasing complexity penalties (forcing sketched QR: $SA=QR^{-1}$ for Blendenpik), requiring subsampling, and finally replacing $U$ in $A=U\Lambda V^\top$ with heavy-tailed distributions to force leverage-score sampling. Reward weights $\mathbf{w}_s$ amplify the failure signals of the previous stage's algorithm.
    - **Design Motivation**: This injects the inductive bias of numerical analysts into the RL environment, transforming a globally sparse program space into steps with local signals without fixing the algorithm structure.

2.  **Monte Carlo Graph Search (MCGS) + UCD**:
    - **Function**: Merges semantically equivalent intermediate states in the program space, reducing the $O(b^d)$ state explosion of tree search to $O(|\mathcal{S}|)$ unique states.
    - **Mechanism**: The search structure is upgraded from a tree to a DAG $\mathcal{G}=(\mathcal{S},\mathcal{E})$. When expanding, if a normalized program (after dead-code elimination) already exists, a parent edge is linked to the existing node. During backpropagation, rollout rewards $R$ update $N(s,a)$ and $\hat{Q}(s,a)$ along all paths to that state. Action selection uses UCD calibrated for DAGs: $a'=\arg\max_a[\hat{Q}(s,a)+c\sqrt{\log N(s)/N(s')}]$, normalizing by child node visits $N(s')$ rather than edge visits to avoid over-exploration in multi-parent scenarios.
    - **Design Motivation**: Identical RLA algorithms can often be constructed via different action sequences. MCGS allows experience from one evaluation to be shared across all paths, focusing the compute budget on truly unique programs.

3.  **LUCB Adaptive Stopping + Multi-objective Weighted Reward**:
    - **Function**: Eliminates manual playout budgets and unifies multiple quality objectives into a single scalar reward.
    - **Mechanism**: Each decision point uses Lower/Upper Confidence Bounds to identify the leader $a_{\text{leader}}$ and challenger $a_{\text{challenger}}$. Search transitions when $\hat{Q}(a_{\text{leader}})-U(a_{\text{leader}})>\hat{Q}(a_{\text{challenger}})+U(a_{\text{challenger}})$. Rewards are calculated as $R(\mathcal{A})=\sum_{k\in\{\text{acc},\text{decay},\text{comp},\text{cond}\}} w_k R_k$, considering relative residual, worst-step contraction ratio $\rho_{\max}$, computational cost, and condition number.
    - **Design Motivation**: Adaptive stopping ensures fair comparisons under fixed budgets. Multi-objective weighting allows the curriculum to explicitly "demand" solutions to specific failure modes by adjusting $\mathbf{w}_s$.

### Loss & Training
There is no neural network to train. "Learning" occurs entirely within the $(\hat{Q},N)$ statistics of MCGS. Each curriculum run serves as a "training run". Each target algorithm is evaluated over 20 runs for success rate and playouts-to-success; the primitive library is slightly adjusted across stages (typically 17 to 25 primitives).

## Key Experimental Results

### Main Results
Evaluations were conducted on 5 curricula: 4 for linear systems (Preconditioned Weighted SGD, Block Randomized Kaczmarz, Subsampled Least Square GD, Sketched Preconditioned GD) and 1 for Newton Sketch on logistic regression.

| Target Algorithm | Method | Playouts ↓ | Time (s) / Success Rate |
| :--- | :--- | :--- | :--- |
| Preconditioned Weighted SGD | MCTS | 34902 | 380.7 / 75% |
| | MCGS+UCT | 13037 | 193.2 / 80% |
| | **MCGS+UCD** | **10721** | **191.1 / 80%** |
| Block Randomized Kaczmarz | MCTS | 66468 | 468.0 / 75% |
| | **MCGS+UCD** | **25158** | **205.0 / 75%** |
| Subsampled Least Square GD | MCTS | 15847 | 10.4 / 75% |
| | **MCGS+UCD** | **5061** | 8.9 / 80% |
| Sketched Preconditioned GD | MCTS | 17655 | 142.9 / 75% |
| | **MCGS+UCD** | **6034** | 58.4 / 75% |
| Newton Sketch | MCTS | 2557 | 5949.6 / 100% |
| | **MCGS+UCD** | **1416** | **4480.9 / 100%** |

Across 5 curricula, MCGS reduces playouts by $2\text{--}3\times$ relative to MCTS. As targets become more compositional (deeper/sparser rewards), the advantage of UCD over UCT increases—on Block Randomized Kaczmarz, UCD requires 35% fewer playouts than UCT.

### Ablation Study
| Configuration | Key Phenomenon | Note |
| :--- | :--- | :--- |
| Full Curriculum + MCGS+UCD | Newton Sketch 100% success | Complete proposal |
| Skipping any curriculum stage | Newton Sketch **0%** success | Curriculum is a prerequisite for reachability, not just acceleration |
| MCGS Revisit Ratio (len=8/10/12) | 0.578 / 0.530 / 0.520 | Gains from merging decay slowly as programs lengthen |
| MCGS Primitive Library 17→25 | 0.578→0.533 | Merging remains effective as library size grows |
| Generalization to PSD Eigenvalues | 3-stage curriculum 100% success | Requires only a `VEC_NORMALIZE` primitive and Rayleigh quotient reward |

### Key Findings
- The 0% vs 100% success rate on Newton Sketch highlights that the curriculum is a "reachability tool"—certain algorithms are never found by RL within reasonable budgets without stage-wise guidance.
- The advantage of MCGS increases with compositional depth. The gap between UCD and UCT is most pronounced in the sparsest reward tasks, confirming theoretical analyses regarding over-exploration in UCT for DAGs.
- Domain adaptation requires only replacing a thin interface (primitives + constraints + rewards); core MCGS and curriculum logic remain unchanged.

## Highlights & Insights
- "Scaling numerical failure modes into a ladder" is an elegant piece of inductive bias engineering. It translates domain knowledge into an RL curriculum, providing direction while allowing the agent to discover novel combinations.
- MCGS + UCD is an overlooked gem in the "program synthesis + MCTS" lineage (e.g., AlphaZero). When search targets can be deduplicated (e.g., algebraic equivalence in linear algebra, query normalization in SQL), DAG search provides $2\text{--}3\times$ speedup with minimal changes.
- LUCB stopping and multi-objective rewards form a reusable template for any discovery task where "algorithm quality" is a weighted sum of numerical metrics, eliminating manual budget tuning.

## Limitations & Future Work
- The experiments focus on "rediscovering" classical algorithms rather than "discovering entirely new" RLA algorithms; discovering new algorithms would require larger budgets and systemic formal analysis.
- The curriculum is still manually designed—the agent lacks the ability to automatically identify the "next failure mode."
- Evaluations are performed on synthetic systems; it remains to be seen if discovered algorithms are Pareto optimal on real scientific workloads.
- The primitive library is currently at the level of basic linear algebra; extending it to Krylov subspaces or PDE operators would require redesigning the type system and dead-code elimination.

## Related Work & Insights
- **vs FunSearch / AlphaEvolve**: The latter uses LLMs as mutation operators, biasing search toward the training distribution. RL4RLA relies on no pre-trained priors and searches on explicitly typed symbolic programs for better controllability.
- **vs AlgoTune**: AlgoTune performs edit-compile-test optimization on existing implementations; RL4RLA performs algorithm-level synthesis.
- **vs AlphaTensor / AlphaDev**: While sharing the MCTS-style discovery, RL4RLA operates in the "symbolic linear algebra program" space and uses curricula specifically to address reward sparsity.
- **vs Learned Sketching / Learned Preconditioner**: Those works neuralize single components into parameters; RL4RLA outputs readable symbolic programs for formal analysis.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐ 

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Provable Benefit of Curriculum in Transformer Tree-Reasoning Post-Training](provable_benefit_of_curriculum_in_transformer_tree-reasoning_post-training.md)
- [\[ICML 2026\] Adaptive Bandit Algorithms for Contextual Matching Markets](adaptive_bandit_algorithms_for_contextual_matching_markets.md)
- [\[ICML 2026\] The Surprising Difficulty of Search in Model-Based Reinforcement Learning](the_surprising_difficulty_of_search_in_model-based_reinforcement_learning.md)
- [\[ICML 2026\] Learning to Search and Searching to Learn for Generalization in Planning](learning_to_search_and_searching_to_learn_for_generalization_in_planning.md)
- [\[ICML 2026\] Learning to Approximate Uniform Facility Location via Graph Neural Networks](learning_to_approximate_uniform_facility_location_via_graph_neural_networks.md)

</div>

<!-- RELATED:END -->
