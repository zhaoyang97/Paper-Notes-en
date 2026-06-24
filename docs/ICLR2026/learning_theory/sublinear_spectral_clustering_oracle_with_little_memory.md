---
title: >-
  [Paper Note] Sublinear Spectral Clustering Oracle with Little Memory
description: >-
  [ICLR 2026][Learning Theory][Spectral Clustering Oracle] This paper designs the first "small-memory" sublinear spectral clustering oracle for clusterable graphs. By using a new subroutine that estimates random walk collision probabilities in batches, it reduces the space required to construct the data structure from the previous $\Omega(\sqrt{n})$ to potentially below $n^{0.01}$. This achieves a space-time trade-off curve of $S \cdot T = \tilde{O}(n)$ and proves that this cur…
tags:
  - "ICLR 2026"
  - "Learning Theory"
  - "Sublinear Algorithms"
  - "Spectral Clustering Oracle"
  - "Space-Time Trade-off"
  - "Random Walk"
  - "Collision Probability"
date: 2026-05-08
content_hash: 8a91c46d8e537367
---

# Sublinear Spectral Clustering Oracle with Little Memory

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=0GpolO2auw](https://openreview.net/forum?id=0GpolO2auw)  
**Area**: Learning Theory / Sublinear Algorithms  
**Keywords**: Spectral Clustering Oracle, Sublinear Algorithms, Space-Time Trade-off, Random Walk, Collision Probability

## TL;DR
This paper designs the first "small-memory" sublinear spectral clustering oracle for clusterable graphs. By using a new subroutine that estimates random walk collision probabilities in batches, it reduces the space required to construct the data structure from the previous $\Omega(\sqrt{n})$ to potentially below $n^{0.01}$. This achieves a space-time trade-off curve of $S \cdot T = \tilde{O}(n)$ and proves that this curve is nearly optimal for a natural class of methods.

## Background & Motivation

**Background**: Graph clustering involves partitioning vertices into several communities that are "dense internally and sparse externally." For large graphs, it is undesirable to read the entire graph before clustering. Thus, researchers study **sublinear spectral clustering oracles** (Peng 2020; Gluch et al. 2021; Shen & Peng 2023). Given query access to an adjacency list, these first construct a compact data structure $D$ in sublinear time. Subsequently, for any vertex $x$, $\textsc{WhichCluster}(G,x)$ can be answered in sublinear time without incurring a global $\Omega(n)$ cost.

**Limitations of Prior Work**: All existing sublinear spectral clustering oracles require **at least $\Omega(\sqrt{n})$ space** to construct $D$. Peng (2020) utilizes $\tilde{\Theta}(\sqrt{n})$ space; Gluch et al. (2021) and Shen & Peng (2023) utilize $\Omega(n^{1-\delta})$ ($\delta \le \tfrac12$), which remains $\ge \sqrt{n}$. For scenarios involving trillion-edge graphs, streaming models (single-pass sublinear space), cloud platforms (expensive storage and transmission), and GPUs/TPUs (high compute but limited on-chip memory), $\sqrt{n}$ working memory is the bottleneck.

**Key Challenge**: This $\sqrt{n}$ space lower bound **does not arise from the clustering process itself, but from the dot product estimation step within it**. A key primitive in spectral clustering is estimating the dot product of spectral embeddings $\langle f_x,f_y\rangle$, which can be reduced to estimating the collision probability of two random walk distributions $\langle M^t\mathbf{1}_x,\,M^t\mathbf{1}_y\rangle$ ($M$ is the lazy random walk transition matrix). Prior methods run $R \approx \sqrt{n}$ independent walks from each vertex and store all endpoints to construct an empirical distribution to compute the dot product. Here, space $O(R)$ and time $O(Rt)$ are strictly tied; to maintain accuracy, $R$ must be $\ge \sqrt{n}$, preventing the reduction of space.

**Goal**: (1) Break the $\sqrt{n}$ space barrier to create an oracle with space far less than $\sqrt{n}$ (e.g., $n^{0.01}$) while maintaining sublinear query time; (2) Characterize the trade-off frontier between space $S$ and query time $T$, and prove its optimality.

**Key Insight**: Since the bottleneck lies in "space being tied to the number of walks $R$," can they be **decoupled**? Specifically, can the total number of walks $R$ be maintained (to preserve accuracy) while using memory far less than $R$? Batching techniques in the field of distribution testing (e.g., uniformity testing by Canonne & Yang 2024) provide inspiration.

**Core Idea**: Replace "storing all walk endpoints at once" with **batch** estimation of collision probabilities. Divide $R$ walks into $B = R/M$ batches. For each batch, run only $M$ walks, temporarily store these $M$ endpoints to calculate the batch dot product, and then average over all batches. Space is reduced to $O(M)$, leading to a trade-off $M \cdot R \approx n$, or $S \cdot T = \tilde{O}(n)$.

## Method

### Overall Architecture

The input is a $d$-regular, $(k,\varphi,\varepsilon)$-clusterable graph $G$ (internal conductance $\phi_{in}(C_i) \ge \varphi$, external conductance $\phi_{out}(C_i,V) \le \varepsilon$, and clusters of similar size), accessible only via adjacency list queries. The output is a data structure $D$ supporting $\textsc{WhichCluster}$ queries with a small overall misclassification ratio.

The pipeline is a four-layer bottom-up hierarchy. The key is to **only replace the bottom-most "collision probability estimation" primitive**, while the upper clustering framework largely follows Shen & Peng (2023):

-   **Bottom Layer**: `EstRWDot` (Alg. 1) uses the batching trick to estimate the collision probability $\langle M^t\mathbf{1}_x,M^t\mathbf{1}_y\rangle$ of a vertex pair in $\tilde{O}(M)$ space; `EstColliProb` (Alg. 2) generalizes this to a set of sampled vertices to output a Gram matrix.
-   **Middle Layer**: Dot product oracle `InitOracle` (Alg. 3) + `QueryDot` (Alg. 4). These perform eigendecomposition on the Gram matrix to obtain a compact matrix $\Psi$, "translating" collision probabilities into estimates of spectral embedding dot products $\langle f_x,f_y\rangle$, following the same space-time trade-off.
-   **Upper Layer**: The clustering oracle uses the dot product oracle to build a similarity graph $H$ among sampled landmark vertices; the connected components of $H$ represent the clusters. For queries, $x$ is compared against landmarks to determine its cluster.
-   **By-product**: Directly applying this collision probability estimation to detect the spectral gap of the random walk operator yields a sublinear algorithm for distinguishing "1 cluster vs. 2 clusters," accompanied by a matching space-time lower bound.

### Key Designs

**1. Batch Collision Probability Estimation: Leveraging $O(M)$ memory for $R$ walks**

This is the engine for breaking the $\sqrt{n}$ barrier. Prior work stores $R \approx \sqrt{n}$ walk endpoints simultaneously, forcing space $\ge \sqrt{n}$. `EstRWDot` (Alg. 1) switches to batching: select batch count $B = R/M$. For each $b = 1, \dots, B$, run $M$ lazy random walks of length $t$ from both $x$ and $y$. Use the endpoint frequencies to construct empirical distributions $\hat p_x, \hat p_y$, calculate the batch dot product $Z_b = \langle\hat p_x, \hat p_y\rangle$, and finally take the average $Z = \frac{1}{B} \sum_b Z_b$. At any moment, only $M$ endpoints from one batch need to be stored, reducing space to $O(M \log n)$ bits while time remains $O(Rt)$.

Formal Guarantee (Lemma 3.1): Set $t \ge \tfrac{20 \log n}{\varphi^2}$ and $1 \le M \le O\!\big(\tfrac{n^{1/2-20\varepsilon/\varphi^2}}{k}\big)$. If 

$$R\ \ge\ \frac{c\,k^2\,n^{-1+40\varepsilon/\varphi^2}}{\sigma_{err}^2\,M},$$

then with probability $\ge 0.99$, $|Z - \langle M^t \mathbf{1}_x, M^t \mathbf{1}_y \rangle| \le \sigma_{err}$. The efficiency lies in the fact that accuracy is determined by the total count $R$, while memory is determined only by the batch size $M$; they are **decoupled**. By choosing $M$ far smaller than $\sqrt{n}$, the trade-off $M \cdot R \approx n$ is achieved. `EstColliProb` (Alg. 2) extends this to a set of sampled vertices $I_S$ ($|I_S|=s$), calling `EstRWDot` for each pair $(s_i, s_j)$ and using the median trick over $O(\log n)$ repetitions to obtain a robust Gram matrix $G \approx (M^tS)^T(M^tS)$ in $\tilde{O}(M \cdot s^2)$ space. 

**2. Low-memory Dot Product Oracle: Decoding spectral embedding dot products**

With collision probabilities, one must convert them into spectral embedding dot products $\langle f_x,f_y\rangle$ ($f_x=U_{[k]}^T\mathbf{1}_x$). `InitOracle` (Alg. 3) first calculates the Gram matrix $G$ on $s$ uniformly sampled landmark vertices, then performs eigendecomposition on $\tfrac{n}{s}G = \hat W \hat\Sigma \hat W^T$ to construct the compact matrix:

$$\Psi\ =\ \tfrac{n}{s}\,\hat W_{[k]}\,\hat\Sigma_{[k]}^{-2}\,\hat W_{[k]}^T\ \in\ \mathbb{R}^{s\times s}.$$

This $\Psi$ is the only object stored long-term (size $n^{O(\varepsilon/\varphi^2)} \cdot \mathrm{polylog}$). During queries, `QueryDot` (Alg. 4) computes collision probability vectors $\alpha_x, \alpha_y$ for $x, y$ to the $s$ landmarks and outputs $\langle f_x, f_y\rangle_{apx} = \alpha_x^T \Psi \alpha_y$. Theorem 3.2 guarantees $|\langle f_x, f_y \rangle_{apx} - \langle f_x, f_y \rangle| \le \xi/n$. The initialization and query space-time complexity are:

$$S\ =\ \big(\tfrac{k}{\xi}\big)^{O(1)} n^{O(\varepsilon/\varphi^2)} M\,\mathrm{polylog}(n),\qquad T\ =\ \big(\tfrac{k}{\xi}\big)^{O(1)}\frac{n^{1+O(\varepsilon/\varphi^2)}}{M}\cdot\frac{1}{\varphi^2}\,\mathrm{polylog}(n).$$

The product $S \cdot T = \tilde{O}(n^{1 + O(\varepsilon/\varphi^2)})$ is independent of $M$. Unlike Gluch et al. (2021), space here can be far below $\sqrt{n}$. To ensure sublinear query time, $M \ge n^{c\varepsilon/\varphi^2}$ is required.

**3. Similarity Graph Clustering Oracle: Plug-and-play**

The top-level clustering logic largely reuses Shen & Peng (2023). It is based on a structural property: in a $(k, \varphi, \varepsilon)$-clusterable graph, for most vertices, if $x, y$ are in the same cluster, then $\langle f_x, f_y \rangle \approx \tfrac{k}{n}$; otherwise, it is $\approx 0$. During preprocessing (`ConstructOracle`), $s = \tfrac{k \log k}{\gamma}$ vertices are sampled to form set $S$. For every pair $u, v \in S$, `QueryDot` estimates $\langle f_u, f_v \rangle_{apx}$ and an edge is added to similarity graph $H$ if the value is large. The connected components of $H$ correspond to clusters. For a query, `Search` compares $x$ against the landmarks. The space constraint **only affects space and query time, not other guarantees** like misclassification rates.

**4. 1-cluster vs. 2-cluster Distinction & Matching Lower Bound**

Ours provides a sublinear algorithm for distinguishing a "single-cluster $\varphi$-expander" from the "union of two disjoint $\varphi$-expanders" (Alg. 5). The task is reduced to detecting the spectral gap of $M^t$. For $t = O(\log n/\varphi^2)$, the second eigenvalue $v_2(M^t)$ is $\le n^{-10}$ for 1-cluster graphs and exactly $1$ for 2-cluster graphs. The algorithm builds a small proxy matrix $G \in \mathbb{R}^{O(\log n) \times O(\log n)}$ using the batching trick and checks if $(v_2(\tfrac{n}{s}G))^2 < 0.6$. This requires $\tilde{O}(M)$ space and $\tilde{O}(n/M)$ time (Theorem 1.2).

Theorem 1.3 proves that any algorithm solving this problem via a random walk oracle with error $\le 1/3$ must satisfy $S \cdot T \ge \Omega(n)$. The proof reduces graph random walks to **space-bounded distribution testing**. By constructing pairs of hard instances and using an **inductive coupling argument**, it is shown that each observation provides at most $O(S/n)$ information under space constraint $S$, leading to $T \cdot O(S/n) = \Omega(1)$. This confirms the trade-off is nearly tight within $\mathrm{polylog}$ factors.

### Loss & Training
Ours is a purely theoretical/algorithmic work and does not involve training or loss functions. The core "hyperparameter" is $M$ (batch size), which adjusts the balance between space and query time, along with walk length $t$ and median repetitions.

## Key Experimental Results

Experiments on Stochastic Block Model (SBM) synthetic graphs validate the space-time trade-off. Parameters: $n=3000, k=3$.

### Main Results (Space Efficiency)

Using 10,400 words as the $1\times$ baseline for space $S$:

| Oracle | Space (words) | × Baseline | Const. Success Rate | Accuracy |
|--------|---------------|------------|-----------------------|----------|
| Ours | 9900 | 0.95× | 1 | 0.9833 |
| Ours | 10100 | 0.97× | 1 | 0.9900 |
| Ours | 10400 | 1× | 1 | 0.9907 |
| Prev. | 34840 | 3.35× | 0 | 0 (Failed) |
| Prev. | 43888 | 4.22× | 0.6 | 0.9860 |
| Prev. | 44383 | 4.27× | 1 | 0.9997 |

Ours achieves 0.98+ accuracy at approximately $1\times$ space. Prior methods **failed to construct** the oracle even at $3.35\times$ space, needing $4.27\times$ to reach comparable accuracy. Ours reduces space requirements by about 4 times.

### Ablation Study (Space-Time Trade-off)

- **$S$ vs $T$ (Figure 1)**: Using "total random walk steps per query" as a proxy for $T$, space $S$ and time $T$ are inversely proportional, fitting the theoretical $S \cdot T = \tilde{O}(n^{1 + O(\varepsilon)})$.
- **Accuracy**: Remained stable across different $M$ settings (0.9833 to 1.0).

### Key Findings
- **The space-time knob is real**: Memory decreases as query time increases, and vice-versa, following the predicted curve while maintaining high accuracy.
- **Verification of the "hard lower bound" in prior work**: Previous oracles do not see a graceful degradation in accuracy when space is insufficient; they simply fail to construct (e.g., producing too many or too few components in $H$), which corresponds to the $\Omega(\sqrt{n})$ barrier.

## Highlights & Insights
- **Locating the bottleneck is more critical than the solution**: The authors identified that the $\sqrt{n}$ space bound is rooted in the dot product primitive rather than the clustering logic. By replacing a single subroutine, they achieved comprehensive improvement.
- **Decoupling "Accuracy $\leftrightarrow$ Memory" via batching is a portable technique**: Splitting $R$ samples into batches of $M$, storing only one batch, and averaging results ensures memory depends on $M$ while accuracy depends on $R$. This method, originating in distribution testing, is strictly analyzed here for random walks and can be applied to many sublinear algorithms.
- **Matching Upper and Lower Bounds**: By reducing random walks to space-bounded distribution testing, the $S \cdot T = \Omega(n)$ proof ensures that the trade-off is not just an empirical observation but a theoretical necessity.

## Limitations & Future Work
- **Reliance on strong clusterability assumptions**: Requires $(k,\varphi,\varepsilon)$-clusterable graphs with balanced sizes and very small external conductance $\varepsilon$ (e.g., $10^{-5}$ relative to $\varphi^2$).
- **Large constants and polylog factors**: Space includes factors like $(k/\xi)^{O(1)}$ and $n^{O(\varepsilon/\varphi^2)}$, which might offset practical gains for moderate $n$.
- **Lower bound specific to random walk methods**: $S \cdot T = \Omega(n)$ is proven for the class of "natural" methods using collision probability. Whether non-random walk frameworks can break this remains open.
- **Future Directions**: Relaxing conductance gaps, extending batching to directed/weighted graphs, and engineering verification on trillion-edge real-world graphs.

## Related Work & Insights
- **vs. Peng (2020)**: The first robust sublinear spectral clustering oracle; space and time are both $\tilde{O}_\varphi(\sqrt{n} \cdot \mathrm{poly}(k/\varepsilon))$. Ours compresses space far below $\sqrt{n}$.
- **vs. Gluch et al. (2021)**: Dot product oracle requires $\Omega(n^{1-\delta})$ space. Ours (Item 1) reuses this clustering framework but replaces the dot product primitive to satisfy the $S \cdot T = \tilde{O}(n^{1+O(\varepsilon)})$ trade-off.
- **vs. Shen & Peng (2023)**: Space $\Omega(n^{1-\delta})$. Ours (Item 2) uses this clustering oracle as a backbone and is the main experimental baseline.
- **vs. Canonne & Yang (2024)**: Used batching for uniformity testing. Ours is the first to rigorously analyze this for graph random walks and embed it into a spectral clustering pipeline.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to break the $\Omega(\sqrt{n})$ space barrier for spectral clustering oracles with a matching lower bound.
- Experimental Thoroughness: ⭐⭐⭐ limited to small SBM graphs; sufficient as theoretical proof but not exhaustive.
- Writing Quality: ⭐⭐⭐⭐ Clear technical overview; the identification of the bottleneck and reduction logic is well-explained.
- Value: ⭐⭐⭐⭐ Provides a novel tool for memory-constrained large graph clustering; the batching technique is highly reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Oracle-Efficient Hybrid Online Learning with Constrained Adversaries](oracle-efficient_hybrid_learning_with_constrained_adversaries.md)
- [\[ICLR 2026\] Dynamical properties of dense associative memory](dynamical_properties_of_dense_associative_memory.md)
- [\[ICLR 2026\] On the Spectral Differences Between NTK and CNTK and Their Implications for Point Cloud Recognition](on_the_spectral_differences_between_ntk_and_cntk_and_their_implications_for_poin.md)
- [\[ICLR 2026\] Efficient Testing for Correlation Clustering: Improved Algorithms and Optimal Bounds](efficient_testing_for_correlation_clustering_improved_algorithms_and_optimal_bou.md)
- [\[ICLR 2026\] Memory-Statistics Tradeoff in Continual Learning with Structural Regularization](memory-statistics_tradeoff_in_continual_learning_with_structural_regularization.md)

</div>

<!-- RELATED:END -->
