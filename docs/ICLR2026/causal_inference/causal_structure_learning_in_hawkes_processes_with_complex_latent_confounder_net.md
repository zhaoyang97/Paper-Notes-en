---
title: >-
  [Paper Note] Causal Structure Learning in Hawkes Processes with Complex Latent Confounder Networks
description: >-
  [ICLR 2026][Causal Inference][Paper Note] When a multivariate Hawkes process contains an unknown number of latent subprocesses at unknown locations, this paper first proves that the Hawkes process is equivalent to a discrete-time linear autoregressive causal model after discretizing continuous-time event sequences using minimal windows. It then utilizes the **
tags:
  - ICLR 2026
  - Causal Inference
date: 2026-05-08
content_hash: 9a5e8cbd181edfb0
---
# Causal Structure Learning in Hawkes Processes with Complex Latent Confounder Networks

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=mA78uXqcnl](https://openreview.net/forum?id=mA78uXqcnl)  
**Code**: To be confirmed  
**Area**: Causal Inference / Causal Discovery  
**Keywords**: Hawkes processes, causal structure learning, latent confounding, rank constraints, identifiability

## TL;DR
When a multivariate Hawkes process contains an unknown number of latent subprocesses at unknown locations, this paper first proves that the Hawkes process is equivalent to a discrete-time linear autoregressive causal model after discretizing continuous-time event sequences using minimal windows. It then utilizes the **rank constraints of the cross-covariance matrix** of observed counts to provide necessary and sufficient conditions for identifying latent confounding subprocesses and all causal edges. Based on this, a two-stage iterative algorithm is designed that does not require prior knowledge of the existence, number, or connections of latent variables.

## Background & Motivation

**Background**: Multivariate Hawkes processes are standard tools for modeling time-dependent "event-triggered event" scenarios—such as social network retweets, neural spike trains, and financial transactions. Mainstream approaches for learning Hawkes causal structures follow two paths: (1) Granger causality combined with maximum likelihood to fit excitation kernels directly on continuous-time event sequences; or (2) binning events into counts and performing likelihood-based estimation on bin counts to reduce dependence on high-resolution timestamps.

**Limitations of Prior Work**: Almost all methods implicitly assume **causal sufficiency**—that all task-related subprocesses are fully observed, with the goal being to find causal edges between these observed processes. However, real-world systems are often only partially observable: in neuroscience, many neurons are never recorded due to limited measurement techniques, yet they significantly influence recorded neurons. If these unobserved subprocesses are **confounders**, ignoring them creates spurious causal edges between observed processes, leading to incorrect conclusions. Existing work like Shelton et al. (2018) can only complete missing event timestamps within an observed subprocess via posterior sampling but cannot handle "entirely unobserved subprocesses" unless the existence and number of latent processes are specified in advance—which is an unrealistic requirement.

**Key Challenge**: The continuous-time Hawkes intensity $\lambda_i(t)$ is an instantaneous rate dependent on history. Performing causal discovery with latent variables directly on it is extremely difficult—there are no readily available conditional independence tests, nor can interventions be performed on latent processes. Meanwhile, classical causal discovery with latent variables (e.g., rank-based linear latent variable methods or temporal LPCMCI) relies on assumptions violated in Hawkes scenarios: they either require acyclicity, weak autocorrelation, or exogenous latent variables, whereas Hawkes processes naturally involve dense cross-lag dependencies, endogenous latent processes, and allow for cycles and self-loops.

**Goal**: To recover the **summary causal graph** between observed and latent subprocesses under a fully partially-observed setting, without pre-specifying the existence, number, or locations of latent processes.

**Key Insight**: The authors' key observation is that by discretizing continuous time with a window width $\Delta$, the Hawkes process converges to a **discrete-time linear autoregressive model** as $\Delta \to 0$. Once a linear representation is obtained, latent confounding leaves **characteristic low-rank fingerprints** in the second-order statistics (cross-covariance) of observed variables, allowing latent processes to be identified using only observed counts via rank tests.

**Core Idea**: Replace "continuous-time maximum likelihood fitting" with "discretized linear causal representation + cross-covariance rank constraints," transforming Hawkes causal discovery with unknown latent confounders into a set of provable and computable rank tests.

## Method

### Overall Architecture

The logical chain of the paper is: **Continuous Hawkes → Discrete Linear Autoregression (equivalent representation) → Rank constraints characterizing d-separation (translating graph structure into observable statistics) → Replacing latent variables with observed surrogates (enabling rank tests for latent confounders) → Two-stage iterative algorithm (alternating between causal edge identification and new latent process discovery) → Identifiability theorems (providing necessary and sufficient conditions).**

Two graphs must be distinguished: the **summary causal graph** is the actual target for recovery, where nodes are subprocesses and directed edges correspond to non-zero excitation functions, allowing cycles, self-loops, and latent nodes. The **window causal graph** treats counts of each subprocess within each window of length $\Delta$ as a variable; when expanded over time lags, it is a Directed Acyclic Graph (DAG) since the future cannot influence the past. There is a one-to-one correspondence: an edge $N_2 \to N_1$ in the summary graph is equivalent to all lagged variables of $N_2$ pointing to the current variable $N_1^{(n)}$ in the window graph. All "identification" in this paper is first performed via rank tests on observed variables in the window graph and then translated back into causal conclusions for the summary graph.

The model is denoted as **PO-MHP (Partially Observed Multivariate Hawkes Process-based Causal Model)**: the node set $N_G$ contains $p$ observed nodes $O_G$ and $q$ latent nodes $L_G$. The core concept is the **parent-cause set** $P_G$: for a subprocess $N_i$, it is the minimal set such that $N_i$ is locally independent of all other subprocesses given $P_G$ (Proposition 3.4, equivalent to local Markov properties for cyclic graphs). The algorithm identifies the parent-cause set for each subprocess (including discovered latent ones) one by one.

### Key Designs

**1. Discretizing Hawkes into Linear Autoregressive Models: Converting continuous-time problems into linear representations for statistical testing**

Performing causal discovery directly on continuous intensity $\lambda_i(t)=\mu_i+\sum_j\int_0^t \phi_{ij}(t-s)\,dN_j(s)$ lacks a handle. Theorem 4.1 provides a critical equivalent representation: defining the count of the $n$-th window as $N_i^{(n)}:=N_i(n\Delta)-N_i((n-1)\Delta)$, then as $\Delta\to 0$:

$$N_i^{(n)}=\sum_{j=1}^{l}\sum_{k=1}^{n}\theta_{ij}^{(k)}N_j^{(n-k)}+\varepsilon_i^{(n)}+\theta_i^{(0)},$$

where the background term $\theta_i^{(0)}=\Delta\cdot\mu_i$, the excitation coefficients $\theta_{ij}^{(k)}=\int_{(k-1)\Delta}^{k\Delta}\phi_{ij}(s)\,ds$, and $\varepsilon_i^{(n)}$ is serially uncorrelated white noise. The value of this step is that each current count becomes a "linear combination of past lagged counts + noise," thus the discrete variables **encode the causal structure of the continuous subprocesses**. In practice, not all lags are needed—since excitation functions usually have finite support as decay kernels, distal lag coefficients approach zero, allowing truncation to at most $m$ effective lags ($m\ge K$, where $K$ is the number of effective lags). $\Delta$ is chosen via grid search within a stable interval where structure recovery is robust to $\Delta$ perturbations. This distinguishes the approach from binned-likelihood methods: the latter only reconstruct bin count likelihood without modeling structural relationships between bins, whereas this work establishes a **linear causal representation** of discretized Hawkes processes.

**2. Rank Constraints Characterizing d-separation: Translating "graphical conditional independence" into "cross-covariance matrix rank"**

With the linear white noise representation, the causal structure induces characteristic low-rank patterns in the cross-covariance matrices of observed discrete variables. Lemma 4.2 provides the foundational equivalence: for any disjoint variable sets $A_v,B_v,C_v$, "$C_v$ d-separates $A_v$ and $B_v$" if and only if $\mathrm{rank}(\Sigma_{A_v\cup C_v,\,B_v\cup C_v})=|C_v|$. Based on this, Proposition 4.3 gives four equivalent statements for "identifying the observed parent-cause set": $P_G\subseteq O_G$ is a parent-cause set of $O_1$ in the summary graph, equivalent to the lagged variable set $P_v$ containing all parent variables of $O_1^{(n)}$ in the window graph, equivalent to $P_v$ d-separating $O_1^{(n)}$ from other observed variables, equivalent to $\mathrm{rank}(\Sigma_{O_1^{(n)}\cup P_v,\,O_v\setminus O_1^{(n)}})=|P_v|$. Crucially, these four statements **involve only observed variables**—if an observed subprocess satisfies this rank criterion, its parent-cause set is uniquely determined. This design replaces "testing conditional independence" with "calculating the rank of a cross-covariance matrix," bypassing the lack of CI tests for Hawkes processes while naturally accommodating cycles and self-loops.

**3. Symmetric Path Condition + Low-rank Signatures: Inferring latent confounding subprocesses from observed effects**

After identifying observed parent-cause sets, only latent confounding remains. The intuition is: when the same latent process drives multiple observed subprocesses via excitation kernels, its contribution to the current values of these observations is **aligned** across lags. Consequently, the corresponding rows of the cross-covariance matrix fall into a low-dimensional subspace—manifesting as a "rank deficiency that no pure observed parent-cause set can explain." To map this deficiency exactly to graph structure, the paper constrains the form of the excitation function (Assumption 1): $\phi_{ij}(s)=a_{ij}w(s)$, meaning the "node-pair strength $a_{ij}$" is multiplied by a "shared decay function $w(s)$," which includes common exponential decays $\alpha_{ij}e^{-\beta s}$. In the simplest two-lag case (Fig 2a/2b), $[O_1^{(n)},O_2^{(n)}]^\top$ is driven by latent lags $[L_1^{(n-1)},L_1^{(n-2)}]^\top$ via a **rank-1 coefficient matrix** $E$, resulting in $\mathrm{rank}(\Sigma_{\{O_1^{(n)},O_2^{(n)}\},\{O_i^{(j)}\}_{i\in\{3,4\}}})=1$. This precisely indicates a single latent confounder $L_1$ acting as a parent-cause for both $O_1$ and $O_2$ (if $O_1,O_2$ also have self-loops, the rank becomes $2m+1$ because their own lagged variables must be included). To cover complex cases where latent confounders connect to observations through multiple intermediate latent processes, the paper proposes the **Symmetric Acyclic Path Situation (Definition 4.4)**: directed paths from $L_1$ to each observed process in the effect set consist only of intermediate latent processes, have equal length, and are acyclic. When this condition is met, Proposition 4.5 provides the necessary and sufficient criterion $\mathrm{rank}(\Sigma_{\{O_i^{(j)}\}_{i\in\{1,2\}},\,O_v\setminus\{O_1^{(n)},O_2^{(n)}\}})=2m+1$ if and only if such a latent confounder $L_1$ exists—this is the core step for "inferring latent existence from observed effects."

**4. Surrogate Observables Replacing Latent Variables: Making latent processes rank-testable**

Latent variables are not observed and cannot be placed directly into a cross-covariance matrix. Definition 4.6 introduces a clever substitute mechanism: for each inferred latent process $L_1$, designate one of its observed effects $De(L_1):=O_1$ as an **observed surrogate** (requiring a directed path from $L_1$ to $O_1$ that does not pass through other observed processes), and define its sibling set $Sib(De(L_1))$ as all other observed subprocesses influenced by $L_1$ via paths not crossing through other observed processes. With surrogates, Theorem 4.7 generalizes Proposition 4.3 to cases where the test subject or candidate parents contain latent variables: when $N_1$ is latent or the candidate set $P_G'$ includes latent processes, the identification condition becomes $\mathrm{rank}(\Sigma_{A_v,B_v})=|A_v|-1$ (where $A_v$ and $B_v$ are constructed using surrogates and siblings). Theorem 4.8 further generalizes Proposition 4.5 to "identifying latent processes from latent processes"—i.e., when both test subjects might be latent, their respective surrogates are used to determine if they share a common latent parent. The essence of this design is: whenever evaluating rank, **always replace latent processes with observed surrogates**, thereby reducing "rank tests involving latent variables" to "rank tests involving only observed variables."

**5. Two-stage Iterative Algorithm + Identifiability Theorem: Alternating "edge identification" and "latent discovery" until convergence**

With four identification theorems, Algorithm 1 is driven by an **active process set** $A_G$ (subprocesses whose parents are not yet identified, initially $O_G$). **Phase I (Causal Relationship Identification)**: Iterate through each subprocess in $A_G$, testing its parents using the current $A_G\cup O_G$. Once a subprocess's parent-cause set is completely covered by Proposition 4.3 or Theorem 4.7, it is identified and removed from $A_G$ until no further updates occur. **Phase II (New Latent Process Discovery)**: When Phase I can no longer resolve any processes, exhaustive pairings in $A_G$ are checked using Proposition 4.5 and Theorem 4.8 to search for new latent confounders. If two pairs overlap in subprocesses, they are merged (implying a shared latent parent). New latent processes are added to $A_G$, their effects are removed, and the algorithm returns to Phase I. Note that because summary graphs have cycles, observed processes identified as "effects" may still be "causes" for others and thus remain in the scope. The two phases alternate until $A_G$ is empty or stable. Theorem 5.1 provides the global identifiability guarantee: **as long as each latent confounder and its observed surrogates ($\ge 2$) satisfy the symmetric path condition of Definition 4.4, the causal graph over observed and latent confounders is identifiable; if no latent processes exist, Phase I alone ensures full identification.**

### Loss & Training
This work is a theoretical and algorithmic contribution without a loss function to optimize. The entire pipeline consists of deterministic statistical tests: after estimating the cross-covariance matrix of observed discrete variables, a series of **rank tests** are performed (in practice, lags statistically significant to the current variable are retained to estimate effective lag $K$, and $\Delta$ is grid-searched for a stable interval). Progression follows the two-stage iteration of Algorithm 1. All guarantees rely on two mild assumptions: the excitation function takes the form $\phi_{ij}(s)=a_{ij}w(s)$, and standard **rank-faithfulness**—precluding pathological parameterizations where causal relations are unidentifiable, which holds almost surely under infinite data (the failure set has Lebesgue measure zero).

## Key Experimental Results

### Main Results

Compared against six strong baselines on synthetic data: SHP, THP (likelihood methods for discretized Hawkes), NPHC (cumulant-based method); plus Hier. Rank, RLCD (rank methods for i.i.d. linear latent models), and LPCMCI (time-series method for exogenous latent confounders)—the latter three rely on strong assumptions violated in this setting. Six synthetic graph families were tested: one fully observed graph (Fig 1b) and five structures with latent processes (Fig 2a and Figs 3a–3d). The metric is average F1 over ten runs.

| Setting | Structure | Results |
|------|------|------|
| Fully Observed | Fig 1b | Ours consistently outperforms all baselines in F1. |
| Single Latent Confounder | Fig 2a | Ours leads significantly; latent cases require larger sample sizes. |
| Complex Latent Paths (Cases 3–4) | Fig 3a/3b | Ours recovers structures stably; baselines drop significantly due to assumption violations. |

> A notable observation: latent variable scenarios typically require larger sample sizes. Because stationary Hawkes processes have spectral radii $<1$, causal influence decays along latent paths, making reliable detection more data-intensive.

### Ablation Study

| Configuration | Description |
|------|------|
| Input Type Comparison | Tested on event sequences from continuous Hawkes (Eq 1) and data from the discrete model (Eq 2); both are effective. |
| Large Graphs / $\Delta$ Sensitivity | Appendix Q reports $\Delta$ sensitivity—$\Delta$ smaller than the excitation function support is sufficient; structure is robust in the stable interval. |
| Robustness to Rank-Faithfulness Violation | Appendix Q tests robustness when rank-faithfulness is violated. |

### Real Data
Public cellular network alarm dataset (with expert-verified ground truth): 18 alarm types, 55 devices, approximately 8 months of 35k events. Focusing on device id=8, a five-alarm subgraph (Alarm id=0–3 and 7) was selected, and **Alarm id=7 was intentionally treated as a latent process by removing it**. Since Alarm id=1 and id=3 are observed effects of this latent process, the proposed method successfully re-identified id=7 as a latent subprocess from observed data and recovered its main influences (Fig 5), quantitatively outperforming representative baselines (Appendix Q.4).

### Key Findings
- **Discretized representation is the foundation**: Theorem 4.1's conversion of continuous Hawkes to linear autoregression is the prerequisite for all subsequent rank tests; without it, the framework does not hold.
- **Longer latent paths increase difficulty**: Causal influence decays along latent paths, naturally requiring more data for latent variable scenarios; this is an inherent difficulty rather than a flaw.
- **Symmetric path condition is the boundary**: Once intermediate latent paths are asymmetric in length or contain cycles (e.g., adding $L_5\to L_3$ in Fig 2d), Definition 4.4 is violated, the corresponding rank signatures fail, and latent confounders cannot be identified.

## Highlights & Insights
- **The "Hawkes ≈ Linear Autoregression as $\Delta\to 0$" bridge is elegant**: It moves causal discovery for a continuous-time stochastic process entirely into the mature toolbox of linear latent variable rank constraints—the most critical "aha" moment and the source of all subsequent provability.
- **Using "rank-deficit = latent confounder fingerprint" to detect the invisible**: When a single latent process drives multiple observations, its aligned contribution across lags leads to low-rank cross-covariance. This logic of "inferring invisible structure from second-order statistics" is transferable to other linear temporal systems with shared hidden factors.
- **The surrogate observable mechanism is key to making latent variables "computable"**: Replacing latent nodes with their observed effects for rank testing bridges "theoretically existing latent variables" with "practically computable statistics," a strategy worth emulating in other latent causal discovery tasks.
- **The first Hawkes causal discovery framework that doesn't pre-specify latent counts/locations**: Compared to legacy methods requiring fixed latent counts, this is far more practical for real-world scenarios where the existence of hidden variables is unknown.

## Limitations & Future Work
- **Excitation functions are constrained to separable forms** $\phi_{ij}(s)=a_{ij}w(s)$ (shared decay kernel). The authors noted future work could relax this to node-specific decay rates to expand applicability.
- **Dependence on Symmetric Acyclic Path Conditions (Def 4.4) and Rank-faithfulness**: Identifiability guarantees fail when latent confounder paths to multiple effects are asymmetric in length or contain cycles, defining the structural boundaries of the method.
- **Computational complexity grows with the number of subprocesses and graph density**: Phase II uses exhaustive pair searches for latent confounders; larger scales lead to more iterations. The authors listed designing lower-complexity discovery algorithms as a future direction.
- **Real-world evaluation is relatively thin**: Evaluated only on one cellular network dataset where the latent process was manually removed. Verification across more diverse real-world scenarios is needed.
- **Each latent confounder requires at least 2 observed surrogates** for identification (Theorem 5.1), leaving latent processes with only one observed effect unidentifiable.

## Related Work & Insights
- **vs Likelihood-based Hawkes (SHP / THP / NPHC / Granger-MLE)**: These perform MLE/cumulant fitting on continuous or binned data and assume causal sufficiency (fully observed). This paper avoids likelihood fitting, instead searching for low-rank patterns in linear representations of discretized data to identify latent confounders—a paradigm shift from "fitting details" to "test paradigm."
- **vs Binned-likelihood (Shlomovich et al. 2022)**: They reconstruct bin count likelihood but do not model structural relationships between bins; this work establishes a linear causal representation of discretized Hawkes, which is a fundamental structural difference.
- **vs i.i.d. Linear Latent Rank Methods (Hier. Rank, RLCD)**: These allow latents but usually only identify up to equivalence classes and rely on structural/cardinality assumptions incompatible with Hawkes dynamics. This paper targets subprocesses rather than static variables and provides identifiability via Hawkes-specific time-aware rank constraints.
- **vs Time-series Causality (LPCMCI)**: It assumes weak autocorrelation and exogenous latent confounding, whereas Hawkes processes naturally feature dense cross-lag dependencies and endogenous latent processes, violating LPCMCI's premises. This work accommodates both endogenous and exogenous latent processes.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The first Hawkes causal discovery framework that identifies complex latent confounder networks without pre-specifying their number or location. The combination of the discretization bridge and rank signatures is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers six graph families in synthetic data and real-world cellular network validation, though real data is limited and the latent process was artificially created.
- Writing Quality: ⭐⭐⭐⭐ Rigorous theoretical derivation with a clear progression from definitions to theorems, though the density of rank constraint notation may be challenging for readers without a causal discovery background.
- Value: ⭐⭐⭐⭐⭐ Solves the real-world pain point of partial observability with a provable method, holding direct value for scenarios where hidden variables are ubiquitous, such as neuroscience and network telemetry.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Conditional Independent Component Analysis for Estimating Causal Structure with Latent Variables](conditional_independent_component_analysis_for_estimating_causal_structure_with_.md)
- [\[ICLR 2026\] CARL: Preserving Causal Structure in Representation Learning](carl_preserving_causal_structure_in_representation_learning.md)
- [\[ICLR 2026\] An Orthogonal Learner for Individualized Outcomes in Markov Decision Processes](an_orthogonal_learner_for_individualized_outcomes_in_markov_decision_processes.md)
- [\[ICLR 2026\] Distributional Equivalence in Linear Non-Gaussian Latent-Variable Cyclic Causal Models](distributional_equivalence_in_linear_non-gaussian_latent-variable_cyclic_causal_.md)
- [\[ICLR 2026\] Score-based Greedy Search for Structure Identification of Partially Observed Causal Models](score-based_greedy_search_for_structure_identification_of_partially_observed_cau.md)

</div>

<!-- RELATED:END -->
