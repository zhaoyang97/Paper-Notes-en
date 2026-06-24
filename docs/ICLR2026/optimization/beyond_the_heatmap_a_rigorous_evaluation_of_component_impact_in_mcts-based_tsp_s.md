---
title: >-
  [Paper Note] Beyond the Heatmap: A Rigorous Evaluation of Component Impact in MCTS-Based TSP Solvers
description: >-
  [ICLR2026][Optimization][Traveling Salesman Problem] This is an evaluation paper that deconstructs the mainstream "Heatmap + MCTS" paradigm for solving TSP. Using extensive experiments, the authors demonstrate that the "heatmap complexity," which the community has focused on, is not the most critical factor. Instead, neglected MCTS search hyperparameters dominate performance. A zero-learning, zero-parameter k-nearest neighbor prior heatmap (GT-Prior)…
tags:
  - "ICLR2026"
  - "Optimization"
  - "Traveling Salesman Problem"
  - "Heatmap+MCTS"
  - "Monte Carlo Tree Search"
  - "Hyperparameter Tuning"
  - "Fair Evaluation"
date: 2026-05-08
content_hash: 4687a4738ee4c1df
---

# Beyond the Heatmap: A Rigorous Evaluation of Component Impact in MCTS-Based TSP Solvers

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=H6PLJnnK6e](https://openreview.net/forum?id=H6PLJnnK6e)  
**Code**: https://github.com/LOGO-CUHKSZ/beyond-heatmap-mcts-tsp  
**Area**: Neural Combinatorial Optimization / TSP Solving / Evaluation Methodology  
**Keywords**: Traveling Salesman Problem, Heatmap+MCTS, Monte Carlo Tree Search, Hyperparameter Tuning, Fair Evaluation

## TL;DR
This is an evaluation paper that deconstructs the mainstream "Heatmap + MCTS" paradigm for solving TSP. Using extensive experiments, the authors demonstrate that the "heatmap complexity," which the community has focused on, is not the most critical factor. Instead, neglected MCTS search hyperparameters dominate performance. A zero-learning, zero-parameter k-nearest neighbor prior heatmap (GT-Prior), when paired with tuned MCTS, can match or even exceed complex learning models like DIFUSCO.

## Background & Motivation

**Background**: The Traveling Salesman Problem (TSP) is a classic NP-hard problem in combinatorial optimization. Recent mainstream paradigms for solving large-scale TSP with machine learning follow the "Heatmap + Monte Carlo Tree Search (MCTS)" framework proposed by Fu et al. (2021). First, a neural network predicts a heatmap $P^N \in [0,1]^{N\times N}$ for each edge $(i,j)$, where $P^N_{ij}$ represents the probability of the edge appearing in the optimal tour. This heatmap is then used as a prior for an MCTS based on k-opt moves to search for high-quality solutions. Following this line, the community has continuously increased heatmap model complexity, moving from supervised GCNs (Att-GCN) to meta-learning GNNs (DIMES), diffusion models (DIFUSCO), and unsupervised learning (UTSP).

**Limitations of Prior Work**: Almost all subsequent work has focused on how to learn more accurate heatmaps, while the other half of the paradigm—the MCTS search component—has been treated as a fixed, pre-tuned black box. Default hyperparameters are commonly reused with minimal tuning, and the true impact of auxiliary steps like sparsification or extra supervision is rarely reported. This creates a significant flaw in evaluation: if MCTS configurations are not aligned across different methods, the conclusion that "Heatmap A is better than Heatmap B" might result from the MCTS accidentally being better tuned rather than the quality of the heatmap itself.

**Key Challenge**: MCTS configuration is a systematically ignored confounder. A strong heatmap paired with poorly tuned MCTS may perform poorly, while a weak heatmap with expertly tuned MCTS might excel. In such cases, horizontal comparisons with fixed MCTS settings are unfair and may seriously mislead the research direction.

**Goal**: The authors do not intend to propose yet another SOTA solver. Instead, they aim to answer three questions: Q1: To what extent does MCTS configuration determine final solution quality? Q2: Can a simple, parameter-free heatmap with tuned MCTS match or exceed complex learned heatmaps? Q3: Which MCTS hyperparameters are most critical, and how does their impact change with heatmap type and problem scale?

**Key Insight**: The authors make two core assertions: (1) Strategic calibration of MCTS has a massive impact on solution quality and must be addressed seriously; (2) Their proposed GT-Prior (a parameter-free heatmap based on the inherent k-nearest neighbor structure of TSP) can compete with complex learned heatmaps and offers stronger generalization.

**Core Idea**: Rather than continuing to escalate heatmap complexity, one should fairly tune MCTS and introduce a strong yet simple baseline to re-examine the default assumption that "complex heatmaps are the primary source of performance."

## Method

### Overall Architecture
The "method" in this paper is not a new model, but an **evaluation methodology**. The goal is to decouple the contributions of the heatmap and the search components within the "Heatmap + MCTS" paradigm and quantify them separately. The logic is as follows: first, "standardize, tune, and explain" the neglected MCTS variable—performing independent MCTS hyperparameter grid searches for each heatmap type to ensure fairness, then using SHAP analysis to identify critical hyperparameters (answering Q1/Q3). Finally, a zero-parameter GT-Prior heatmap is introduced as a "floor baseline" to compete against complex learned heatmaps under tuned MCTS conditions (answering Q2).

The evaluated MCTS models TSP as an MDP: each state is a valid tour, and actions are k-opt moves that modify the current solution. The search starts from an initial tour sampled with probability $\propto e^{P^N_{ij}}$. Edge weights are initialized as $W_{ij} = 100 \cdot P^N_{ij}$. Each simulation step uses a potential function to select edges for modification:

$$Z_{ij} = \frac{W_{ij}}{\Omega_i} + \alpha \sqrt{\frac{\ln(M+1)}{Q_{ij}+1}}$$

where $\Omega_i = \sum_{j\neq i} W_{ij}$ is the edge weight normalization term, $M$ is the global move count, $Q_{ij}$ is the edge visit frequency, and $\alpha$ is the exploration coefficient—the first term represents exploitation (preferring higher weights), and the second is UCB-style exploration. If a k-opt move shortens the tour ($\Delta L < 0$), it is accepted; otherwise, the search restarts from a newly sampled initial tour. After each move, edge weights are updated based on the improvement:

$$W_{ij} \leftarrow W_{ij} + \beta\left(\exp\left(\frac{L(\pi)-L(\pi')}{L(\pi)}\right)-1\right)$$

where $\beta$ is the learning rate. The evaluation framework revolves around several key MCTS knobs: $\alpha$, $\beta$, maximum k-opt depth, etc.

### Key Designs

**1. MCTS Hyperparameter Tuning Pipeline: Transforming "Search" from a Black Box to an Aligned Baseline**

Prior work often fixed MCTS default configurations for horizontal comparisons, leading to "true heatmap value" being contaminated by search settings. Ours mandates an **independent** MCTS hyperparameter grid search for **every heatmap and every problem scale (TSP-500/1000/10000)**. These configurations are evaluated on a synthetic tuning set (128 instances for TSP-500/1000; 16 for TSP-10000) to select the set with the lowest average optimality gap. Six key hyperparameters are tuned: Exploration coefficient Alpha ($\alpha$), edge weight update aggressiveness Beta ($\beta$), maximum k-opt depth Max\_Depth, candidate edge set size per node Max\_Candidate\_Num, simulation steps per move Param\_H, and a boolean Use\_Heatmap (whether to use the heatmap for initial candidate set construction). This step is effective because it aligns the "search strength" confounder by ensuring every heatmap performs at its best, allowing subsequent comparisons to reflect intrinsic heatmap quality. Furthermore, this grid search is an offline pre-computation with costs comparable to training learned heatmaps, without affecting inference time.

**2. SHAP-Based Hyperparameter Importance Attribution: Quantifying Each Knob's Utility**

To understand "why" and "which knob is most critical," the authors use SHAP (SHapley Additive exPlanations) from game theory to attribute changes in tour length to each MCTS hyperparameter. Positive SHAP values indicate tour lengthening (worse), while negative values indicate shortening (better). This analysis is model-agnostic and quantifies the marginal contribution and non-linear interactions of each hyperparameter. The findings are specific: Max\_Candidate\_Num consistently has a strong (often positive) impact, suggesting that reducing the candidate set from large default values improves both speed and quality. Max\_Depth generally shows positive SHAP, meaning excessive k-opt depth hinders finding good solutions quickly. Alpha and Use\_Heatmap exhibit mixed effects, as their optimal values depend on the specific heatmap (strong non-linear interaction). Beta has a significantly positive impact on SoftDist, implying its default update strategy is suboptimal. Param\_H consistently shows minimal impact within the tested range. This design answers Q3 and moves the selection of hyperparameters from empirical heuristics to quantifiable conclusions.

**3. GT-Prior: A Zero-Learning, Zero-Parameter k-Nearest Neighbor Prior Heatmap as a "Litmus Test"**

Prior evaluations lacked a strong, simple baseline, making it difficult to measure the "added value" of complex heatmaps. Ours observes that edges in optimal TSP tours overwhelmingly connect to a city's nearest neighbors: empirically, the probability of selecting an edge within the top 5 nearest neighbors exceeds 94%, and top 10 exceeds 99%. This distribution is highly consistent across various scales and distributions. This "k-nearest neighbor prior" was previously used only implicitly for constructing sparse graph inputs; GT-Prior makes it explicit. First, the empirical distribution of selecting the $k$-th nearest neighbor is calculated from (near-)optimal solutions:

$$\hat{P}_N(k) = \frac{1}{|\mathcal{I}|}\sum_{I\in\mathcal{I}} P^I_N(k), \quad P^I_N(k) = \frac{n^I_k}{N}$$

where $n^I_k$ is the frequency of the "connection to the $k$-th nearest neighbor" in an instance's optimal solution. Then, the heatmap is defined directly as $P^N_{ij} = \hat{P}_N(k_{ij})$, where $k_{ij}$ is the proximity rank of city $j$ among neighbor of city $i$. This heatmap is **parameter-free, scale-invariant, and requires zero training or inference time**. Its effectiveness stems from encoding the essential structural prior of TSP solutions (locality). Since this prior itself is extremely strong, the marginal utility of adding an expensive neural network layer is naturally limited—GT-Prior serves as a benchmark to quantify whether this marginal gain is worth the cost.

## Key Experimental Results

Experiments cover three scales (TSP-500/1000/10000), various synthetic distributions (uniform, cluster, explosion, implosion), and real-world TSPLIB benchmarks. Ground truth was obtained via Concorde (500/1000) or LKH-3 (10000). The metric is the optimality gap (relative gap to the optimal solution; lower is better).

### Main Results: GT-Prior Matches/Outperforms Complex Learned Heatmaps

All methods used the MCTS tuning pipeline proposed in this work. GT-Prior achieved performance on par with heavy models like DIFUSCO with zero heatmap generation time, even performing best on TSP-10000:

| Heatmap | Type | TSP-500 Gap | TSP-1000 Gap | TSP-10000 Gap |
| :--- | :--- | :--- | :--- | :--- |
| Zero (Tuned MCTS only) | MCTS | 0.66% | 1.16% | 3.80% |
| Att-GCN | SL+MCTS | 0.69% | 1.09% | 3.03% |
| DIMES | RL+MCTS | 0.43% | 1.11% | 3.06% |
| UTSP | UL+MCTS | 0.90% | 1.53% | — |
| SoftDist | dist+MCTS | 0.43% | 0.80% | 2.95% |
| DIFUSCO | SL+MCTS | 0.33% | 0.53% | 2.37% |
| Fast-T2T | SL+MCTS | 0.12% | 0.65% | 4.22% |
| **GT-Prior** | prior+MCTS | **0.50%** | **0.85%** | **2.14%** |

GT-Prior's 2.14% on TSP-10000 is superior to all learning methods (DIFUSCO 2.37%, Fast-T2T 4.22%), with zero heatmap generation overhead. Even the "Zero" heatmap, which provides no edge guidance and relies solely on tuned MCTS (with Use\_Heatmap set to False and candidates chosen by distance), achieved a respectable 0.66% on TSP-500—demonstrating that the search component itself contributes significantly to solution quality.

### Impact of MCTS Configuration

| Phenomenon | Data | Explanation |
| :--- | :--- | :--- |
| Range of gaps for one heatmap | DIMES@TSP-10000: 4.86% → 91.31% | MCTS settings alone can shift results from usable to disastrous. |
| Default vs. Optimal MCTS | Default settings are often far from optimal | Fixed-MCTS horizontal comparisons are invalid. |

This comparison answers Q1: MCTS configuration is the dominant performance factor, and proper tuning significantly improves all heatmaps.

### Generalization Experiments: Transferring TSP-500 Configs to Other Scales

| Method | TSP-1000 Degradation | TSP-10000 Degradation |
| :--- | :--- | :--- |
| DIFUSCO | +0.33% | +2.91% |
| Fast-T2T | +0.75% | -0.06% |
| **GT-Prior** | **+0.03%** | **-0.01%** |

GT-Prior's degradation is nearly zero (even slightly improving), showing significantly better generalization than learning models like DIFUSCO because it encodes scale-invariant structural priors.

### Key Findings
- **Search > Heatmap**: The gains from MCTS tuning often match or exceed those from upgrading the heatmap model. Default MCTS configurations are usually sub-optimal, which was the source of unfair comparisons in the past.
- **Critical Knobs**: Max\_Candidate\_Num (limiting candidates improves speed and quality) and Max\_Depth (avoiding excessive depth) have the greatest impact. Param\_H (simulation count) is surprisingly ineffective, overturning the intuition that "more simulations are always better."
- **Questionable Marginal Gains from Complexity**: With well-tuned search, the zero-parameter GT-Prior approaches complex models, suggesting that many reported "SOTA improvements" might stem from better-tuned search rather than smarter heatmaps.

## Highlights & Insights
- **Exposing the "Confounder"**: The greatest value of this paper lies in its methodology. It clearly identifies MCTS configuration as a systematically ignored confounder and quantifies it using SHAP and grid search. This "clean the evaluation before concluding" approach can be transferred to any "learning module + search/decoding module" paradigm (e.g., LLM inference with model + decoding strategy).
- **The Power of Strong Simple Baselines**: GT-Prior uses an extremely strong structural statistical prior (top 5 neighbors >94%) to challenge complex diffusion and meta-learning models. It reminds researchers to ask, "How much does my complex model actually improve over a zero-parameter prior?"
- **Free Generalization**: GT-Prior is scale-invariant with near-zero transfer degradation. This is appealing for practical deployment, as it eliminates the need to retrain heatmap models for different scales.
- **Reproducible Tuning Pipeline**: By open-sourcing the standardized MCTS tuning pipeline, the authors provide a "fair yardstick" for future research, which is a valuable engineering contribution in itself.

## Limitations & Future Work
- **Scope within Heatmap + MCTS**: All conclusions are based on this specific paradigm and MCTS implementation. They do not imply that learning components are unimportant in all neural combinatorial optimization methods—changing the search backend (e.g., LKH, other local searches) might yield different results.
- **GT-Prior Dependence on Statistical Priors**: The k-nearest neighbor prior is stable on uniform and several synthetic distributions but requires (near-)optimal solutions for statistics. Whether it holds for distributions with extreme structures where nearest-neighbor assumptions fail is an open question.
- **Cost of Grid Search**: While it is a one-time offline cost, independent grid search for every heatmap and scale is still significant. The authors mention more efficient Bayesian optimization (e.g., SMAC3) as potential alternatives.
- **Lack of a "New Optimal Solver"**: This is an evaluation/deconstruction paper, not a solver paper. It provides the methodology for "how to evaluate and what baseline to use" rather than a new SOTA for direct leaderboard competition.

## Related Work & Insights
- **vs. Att-GCN / DIMES / DIFUSCO / UTSP**: These works escalate heatmap complexity (supervised → meta → diffusion → unsupervised), assuming "more complex is better." Ours demonstrates that with aligned MCTS, the marginal advantage of these complex models over GT-Prior is limited and can even be reversed at larger scales.
- **vs. SoftDist (Xia et al., 2024)**: SoftDist already expressed skepticism toward model complexity by using a distance-based simplified heatmap. Ours goes further—GT-Prior is entirely parameter-free and non-learning. With systematic MCTS tuning and SHAP analysis, it upgrades the observation that "simple works" into a reproducible evaluation methodology.
- **vs. Original Fu et al. (2021) MCTS Framework**: Ours reuses the MCTS itself but redefines it from a "fixed black box" to a "critical variable that must be tuned independently for each heatmap," completing the missing half of this research line.

## Rating
- Novelty: ⭐⭐⭐⭐ New perspective rather than a new model; "deconstructing mainstream assumptions + zero-parameter baseline reversal" is high-impact.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 3 scales, 4 distributions, TSPLIB, 7+ heatmaps, SHAP attribution, and generalization analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem-driven (Q1/Q2/Q3), clear argumentation, and measured conclusions.
- Value: ⭐⭐⭐⭐⭐ Provides a fair evaluation yardstick for the Heatmap+MCTS line; methodology is transferable to other "learning+search" paradigms.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] FrontierCO: Real-World and Large-Scale Evaluation of Machine Learning Solvers for Combinatorial Optimization](frontierco_real-world_and_large-scale_evaluation_of_machine_learning_solvers_for.md)
- [\[ICML 2026\] LoRe: Adaptive Interaction-Evaluation Routing with Per-Step Interaction Budgets for Iterative Graph Solvers](../../ICML2026/optimization/lore_adaptive_interaction-evaluation_routing_with_per-step_interaction_budgets_f.md)
- [\[ICLR 2026\] Beyond Short Steps in Frank-Wolfe Algorithms](beyond_short_steps_in_frank-wolfe_algorithms.md)
- [\[ICLR 2026\] Beyond Aggregation: Guiding Clients in Heterogeneous Federated Learning](beyond_aggregation_guiding_clients_in_heterogeneous_federated_learning.md)
- [\[ICLR 2026\] Elastic Optimal Transport: Theory, Application, and Empirical Evaluation](elastic_optimal_transport_theory_application_and_empirical_evaluation.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICLR 2026\] FrontierCO: Real-World and Large-Scale Evaluation of Machine Learning Solvers for Combinatorial Optimization](frontierco_real-world_and_large-scale_evaluation_of_machine_learning_solvers_for.md)
- [\[ICML 2026\] LoRe: Adaptive Interaction-Evaluation Routing with Per-Step Interaction Budgets for Iterative Graph Solvers](../../ICML2026/optimization/lore_adaptive_interaction-evaluation_routing_with_per-step_interaction_budgets_f.md)
- [\[ICLR 2026\] Beyond Short Steps in Frank-Wolfe Algorithms](beyond_short_steps_in_frank-wolfe_algorithms.md)
- [\[ICLR 2026\] Beyond Aggregation: Guiding Clients in Heterogeneous Federated Learning](beyond_aggregation_guiding_clients_in_heterogeneous_federated_learning.md)
- [\[ICLR 2026\] Elastic Optimal Transport: Theory, Application, and Empirical Evaluation](elastic_optimal_transport_theory_application_and_empirical_evaluation.md)

</div>

<!-- RELATED:END -->
