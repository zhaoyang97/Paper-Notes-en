---
title: >-
  [Paper Note] Characterizing the Discrete Geometry of ReLU Networks
description: >-
  [ICLR 2026][learning_theory][Paper Note] This paper abstracts the polyhedral complex partitioned by fully connected ReLU networks in the input space into a "connectivity graph" (where regions are nodes and adjacent regions share edges). It proves that the **average degree of this graph is universally upper-bounded by $2d$ (twice the input dimension), independ
tags:
  - ICLR 2026
  - learning_theory
date: 2026-05-08
content_hash: 765f8eb46bd334e8
---
# Characterizing the Discrete Geometry of ReLU Networks

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=TgLW2DiRDG](https://openreview.net/forum?id=TgLW2DiRDG)  
**Code**: https://github.com/bl-ake/ICLR-2026  
**Area**: Learning Theory / Neural Network Geometry  
**Keywords**: ReLU Networks, Polyhedral Complex, Connectivity Graph, Average Degree, Graph Diameter

## TL;DR
This paper abstracts the polyhedral complex partitioned by fully connected ReLU networks in the input space into a "connectivity graph" (where regions are nodes and adjacent regions share edges). It proves that the **average degree of this graph is universally upper-bounded by $2d$ (twice the input dimension), independent of network width and depth**. It also provides an upper bound for the graph diameter $O(m^\ell)$ that is independent of the input dimension. Theoretical bounds are verified using synthetic data and datasets like MNIST/CIFAR10, revealing that "training data tends to reside in regions with higher connectivity."

## Background & Motivation
**Background**: ReLU networks implement continuous piecewise linear (CPWL) functions. Each "piece" is defined on a polyhedron in the input space, and these polyhedra together form a **polyhedral complex** that partitions the entire input space. Since the non-linearity of the network only occurs at the boundaries between regions, how these regions are "stitched together" directly determines the network's behavior. Over the past decade, the community has focused extensively on "how many regions a given architecture can partition" (starting from Montúfar 2014), but these works only count the **number** of regions without describing their **arrangement**.

**Limitations of Prior Work**: Accurately calculating the entire complex is infeasible for most networks, as the number of regions grows **exponentially** with input dimension and network size. Consequently, beyond "bounds on the total number of regions," almost nothing is known about the geometry of the complex (how regions are adjacent, how many faces one must cross to travel from one region to another). The few works describing arrangement (e.g., Fan et al. 2024 providing an upper bound on the average number of faces) rely on strong assumptions (no bias terms in the first hidden layer or low-rank weights) and provide **asymptotic** results regarding network scale, which are not applicable to general architectures.

**Key Challenge**: The goal is to obtain universal geometric properties that are **independent of specific weight values and architecture size**. However, existing tools either only count quantities, require harsh assumptions, or provide only asymptotic bounds. The exponential explosion of regions makes the "region-by-region analysis" approach entirely unviable.

**Goal**: Establish geometric properties that hold for **all fully connected ReLU networks**, specifically focusing on the **connectivity graph** of the complex—where nodes are polyhedral regions and an edge connects two regions sharing a $(d-1)$-dimensional face. The research answers three questions: What is the average number of neighbors for each region (average node degree)? How does this number change as the network grows? What is the maximum number of faces one must cross to travel between any two regions (graph diameter)?

**Key Insight**: Following the topological perspective of Masden (2025), the authors use **sign sequences** to encode each cell. This transforms "continuous geometric objects" into "discrete combinatorial objects." Consequently, counting faces, counting neighbors, and determining adjacency are converted into combinatorial counting of sign sequences, bypassing direct geometric coordinate calculations.

**Core Idea**: Translate the polyhedral complex into a connectivity graph using sign sequences, and then leverage "recursive removal of bent hyperplanes of neurons + induction" to prove: the average degree is always $\le 2d$ and monotonically approaches $2d$ as scale increases, and the graph diameter bound is independent of the input dimension.

## Method

### Overall Architecture
This is a **pure theory paper**. The main thread is "discretizing geometric problems → proving main conclusions → providing computable algorithms for verification." The framework is divided into three layers: **(1) Discretization Language**—unifying the representation of polyhedral complexes and connectivity graphs using sign sequences, allowing statements about regions, faces, and adjacency to be converted into combinatorial counting of $\{-1,0,1\}^n$ vectors; **(2) Theoretical Results**—proving the average degree upper bound $2d$, asymptotic convergence, the lower bound on degrees, and upper/lower bounds on the connectivity graph diameter within this language; **(3) Computation and Empirics**—providing an algorithm based on Breadth-First Search (BFS) + Linear Programming (LP) to accurately enumerate polyhedra and construct the connectivity graph, then using it to verify theoretical bounds on synthetic and real datasets, while observing distribution patterns of training data.

Let the network input dimension be $d$, the total number of hidden neurons be $n$, the maximum layer width be $m$, and the depth (number of hidden layers) be $\ell$. The core object is the connectivity graph $G=(V,E)$: $V$ consists of the sign sequences of all $d$-cells (maximal regions where the network mapping is affine). An edge exists between two sign sequences if they **differ at exactly one position**. The **degree** of a node equals the number of faces of its region, which is its number of neighbors. The heavy lifting of the proofs lies in "how to accurately count these cells," where sign sequences serve as the key to making counting inductive.

### Key Designs

**1. Sign Sequences and Bent Hyperplanes: Translating Continuous Geometry to Discrete Combinatorics**

To reason about region adjacency without specific coordinates, the authors adopt the encoding from Masden (2025). For a point $x$ in the input space, each hidden neuron $i$ performs an affine transformation $w_i^T x_{p_i}+b_i$ before ReLU: if the value is $>0$, the sign $S(x)_i=1$; if $=0$, then $0$; if $<0$, then $-1$. Thus, $x$ receives a **sign sequence** $S(x)\in\{-1,0,1\}^n$ of length $n$. In the first hidden layer, points where $S(x)_i=0$ lie on a hyperplane. In deeper layers, since $x_{p_i}$ is calculated from the CPWL functions of previous layers, the set of points where $S(x)_i=0$ becomes a **Bent Hyperplane (BH)** that can self-intersect or even disconnect. Each BH divides the space into $S(x)_i=1$ and $S(x)_i=-1$ sides. All BHs together partition the input space into a polyhedral complex.

This encoding is critical because it establishes a **one-to-one correspondence between cells and sign sequences** (under mild assumptions excluding measure-zero degenerate weights, the mapping $S:\mathcal C\to\{-1,0,1\}^n$ is well-defined and injective): a $k$-cell sign sequence has exactly $d-k$ zeros (corresponding to the intersection of $d-k$ BHs containing it) and $k$ non-zero elements. $d$-cells (no zeros) are the nodes of the connectivity graph; two $d$-cells are adjacent if and only if their sign sequences differ by exactly one sign flip. Thus, "counting faces of a region = counting its neighbors = counting positions in the sign sequence that can be flipped into a valid sequence," turning a geometric problem into a combinatorial one.

**2. Neuron Peeling + Induction: Proving the $2d$ Average Degree Upper Bound**

The main conclusion is Theorem 3.4 (a special case of Theorem 3.1): **The average degree of the connectivity graph (average number of faces of $d$-cells) is at most $2d$**, independent of network width, depth, or specific weight values. The core of the proof is a **peeling-counting recursion**. The authors define $\mathcal C-h_i$ as the complex after "removing all cells contained in the bent hyperplane $h_i$ of neuron $i$ and merging cells that shared the removed faces." Lemma 3.2 proves that for any neuron $i$, every $k$-cell belongs to exactly one of three categories: **Type 1** (cells on $h_i$, $i$-th sign bit is 0), **Type 2** (cells in $\mathcal C-h_i$ unaffected by $h_i$), and **Type 3** (cells that were one in $\mathcal C-h_i$ but split in two by $h_i$). Classification depends solely on the sign sequence: whether the $i$-th bit is zero, and whether setting it to zero results in a sequence that remains a valid cell in the complex.

Based on this, Lemma 3.3 provides the recurrence relation:
$$N_k(\mathcal C)=N_k(h_i)+N_k(\mathcal C-h_i)+N_{k-1}(h_i),$$
where $N_k$ denotes the number of $k$-cells. Since each $(d-1)$-cell is a common face of exactly two $d$-cells, the average number of faces is $\frac{2N_{d-1}(\mathcal C)}{N_d(\mathcal C)}$. The authors perform **double mathematical induction on the number of BHs $n$ and dimension $d$**, deriving $(n, d)$ from $(n-1, d-1)$ and $(n-1, d)$ to satisfy the $\le 2d$ bound. While a similar conclusion for single-layer networks (hyperplane arrangements) was established by Fukuda et al. (1991), that proof cannot be extended to BH arrangements in deep networks; this paper bridges the gap using sign sequence correspondence.

**3. Asymptotic Convergence and Degree Lower Bound: The Upper Bound is Tight**

To ensure the upper bound is not vacuous, the authors add tightness and lower bounds. Theorem 3.5 provides a **lower bound**: if the first hidden layer has $n_1$ neurons, each $d$-cell has at least $\min(n_1,d)$ neighbors, and thus the average degree is at least $\min(n_1,d)$. Theorem 3.6 proves that when constructing a network sequence $\mathcal C_n$ by adding neurons to the last layer, the average number of faces of $d$-cells **increases monotonically**. Theorem 3.7 further proves that for shallow single-hidden-layer networks, as the number of neurons $n\to\infty$, the average number of faces **converges exactly to $2d$**:
$$\lim_{n\to\infty}\frac{2N_{d-1}(\mathcal C_n)}{N_d(\mathcal C_n)}=2d.$$
These three theorems indicate that $2d$ is a tight bound that is approached, rather than an unreachable loose limit. Experiments also show that the average number of faces tends toward $2d$ as depth increases.

**4. Decoupling Diameter Upper Bound from Input Dimension**

The final metric is the **diameter** of the connectivity graph—the maximum of the shortest paths between any two regions, geometrically representing the "minimum number of faces to pass through from one polyhedron to any other." Theorem 3.8 gives:
$$\mathrm{diam}(G)=\Omega\!\left(\frac{\ln N_d(\mathcal C)}{\ln n}\right) \text{ and } O(m^\ell).$$
The lower bound $\Omega$ is intuitive: more regions lead to a larger diameter. The most interesting result is the upper bound $O(m^\ell)$—it **depends only on width $m$ and depth $\ell$, and is completely independent of input dimension $d$**, even though the number of regions explodes exponentially with $d$. This implies that regardless of the input space dimensionality, the "hop cost" to traverse the complex remains largely invariant for a fixed architecture. Its practical value is noted in the Discussion: Ji et al. (2022) used the Hamming distance of sign sequences to bound empirical training error. However, Hamming distance fails when a BH is crossed multiple times; the connectivity graph shortest path is a more appropriate metric. Combined with Theorem 3.8, it allows for **empirical error bounds independent of input dimension**.

### Loss & Training
The paper does not introduce new training objectives. For empirical verification, the authors provide **Algorithm 1**: starting from a valid sign sequence $s$ obtained from a point in the input space, it uses BFS to enumerate polyhedra and incrementally build the connectivity graph. For a current sequence $s$, the system of half-space inequalities $\Phi x+\beta\le 0$ for that polyhedron is derived layer-by-layer; an LP (`SOLVELP`) is solved for each neighbor direction to determine if the corresponding inequality is **redundant**. Non-redundant inequalities represent real neighbors, where edges are added and new sequences are queued. This geometric traversal is similar to BFS in Xu et al. (2022) but records shared faces to construct the graph during search.

## Key Experimental Results

### Main Results
The authors first verify theoretical bounds on **synthetic data**: using three isotropic, unit-variance Gaussians to generate clustering tasks, they sweep input dimension $d$, depth, and width. Exhaustive enumeration is performed for small networks.

| Config ($d{=}4$ / $d{=}5$) | Polyhedra Count | Avg Degree | Diameter |
|--------|------|------|------|
| Depth 1, Width 16 | 2517 / 6885 | 7.32 / 9.02 | 22.5 / 23.2 |
| Depth 2, Width 16 | 4.2e4 / 2.7e5 | 7.72 / 9.61 | 41.1 / 42.6 |
| Depth 4, Width 16 | 6.2e5 / 5.0e6 | 7.85 / 9.80 | 76.4 / 70.9 |

The average degree is consistently **below the upper bound $2d$** ($<8$ for $d{=}4$, $<10$ for $d{=}5$) and approaches it as the network grows. The neighbor distribution is **unimodal and right-skewed**, matching Theorem 3.1/3.7.

### Diameter & Distribution Analysis

| Observation | Result | Corresponding Theory |
|------|------|---------|
| Fixed architecture, varying $d$ | Diameter estimates are almost identical across $d$ | Theorem 3.8 (independent of $d$) |
| Fixed width, varying depth | Diameter grows approximately **logarithmically** | $O(m^\ell)$ bound is rarely saturated |
| Regions with data vs. empty regions | Regions containing data have **higher** avg neighbors | Section 5.2 Key Finding |

### Key Findings
- **Robustness of Average Degree**: Across all synthetic and real experiments (California Housing, MNIST, CIFAR10), the average degree strictly falls below $2d$.
- **Training Data Prefers High Connectivity**: Polyhedra containing training data generally have higher neighbor counts than the overall average. Since the number of faces for any polyhedron is upper-bounded by $n$, this actually reduces the right-skewness of the distribution.
- **Bounded/Unbounded vs. Task Type**: Polyhedra with more neighbors are more likely to be **unbounded**. In classification (MNIST/CIFAR10), the proportion of unbounded regions containing data is higher than the overall average, while the opposite is true for regression (CA Housing). The authors explain that classification concentrates complexity on decision boundaries (data in outer unbounded regions), while regression fits data values (data in inner bounded regions).

## Highlights & Insights
- **Discretization as Leverage**: Translating "region adjacency" to "one-bit sign sequence flips" transforms geometric analysis into inductive combinatorial counting. This is the fundamental reason the upper bound proofs succeed.
- **The "Counter-Intuitive" Beauty of the $2d$ Bound**: The average degree depends only on input dimension and is independent of width/depth. This invariant is rare in deep learning theory.
- **Direct Practical Application**: Theorem 3.8 suggests that using shortest paths in the connectivity graph instead of Hamming distance provides empirical error bounds independent of input dimension, linking geometry to generalization, interpretability, and robustness.
- **Algorithmic Verification**: Algorithm 1 allows the "average degree approaches $2d$" claim to be observed directly on datasets like MNIST, making the theory falsifiable.

## Limitations & Future Work
- **Unexplained Phenomena**: Why training pushes data toward high-face-count regions remains unexplained. The impact of specific structures like Conv or Skip-connections is not yet characterized.
- **Activation Function Constraints**: Conclusions apply only to ReLU (and generalizable CPWL activations). There are no direct implications for sigmoid or tanh.
- **Computational Bottlenecks**: Exact enumeration is still infeasible for large networks; empirical work is limited to small networks or low-dimensional latent representations.
- **Future Directions**: Quantitatively linking the data's preference for high-connectivity regions with generalization; extending the peeling-induction framework to Skip/Conv structures; and designing dimension-independent robustness certificates based on diameter bounds.

## Related Work & Insights
- **vs. Region Counting (Montúfar 2014; Goujon 2024)**: Previous works only bounded the **number** of regions; this work characterizes **how they are adjacent**.
- **vs. Fan et al. (2024)**: Fan also bounds average faces but relies on strong assumptions like zero bias or low-rank weights. This work provides an upper bound for **any architecture**.
- **vs. Fukuda et al. (1991)**: This work generalizes the $2k$ average face conclusion from hyperplane arrangements (single layer) to BH arrangements (deep ReLU networks).
- **vs. Ji et al. (2022)**: Improves methodological empirical error bounds by replacing Hamming distance with shortest path distance, which is independent of input dimension.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First time providing width/depth-invariant average degree bounds for any fully connected ReLU network.
- Experimental Thoroughness: ⭐⭐⭐⭐ Synthetic exhaustive enumeration + three real datasets, though limited by computational scale.
- Writing Quality: ⭐⭐⭐⭐ Clear progression from sign sequences to peeling-induction; great visualization (Fig 2/3).
- Value: ⭐⭐⭐⭐ Provides rare "architecture-independent" geometric invariants with clear downstream application potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">
</div>

## Related Papers

- [\[ICLR 2026\] Information Estimation with Discrete Diffusion](information_estimation_with_discrete_diffusion.md)
- [\[ICLR 2026\] Characterizing Pattern Matching and Its Limits on Compositional Task Structures](characterizing_pattern_matching_and_its_limits_on_compositional_task_structures.md)
- [\[ICLR 2026\] A Unification of Discrete, Gaussian, and Simplicial Diffusion](a_unification_of_discrete_gaussian_and_simplicial_diffusion.md)
- [\[ICLR 2026\] Beyond Spectra: Eigenvector Overlaps in Loss Geometry](beyond_spectra_eigenvector_overlaps_in_loss_geometry.md)
- [\[ICLR 2026\] On Universality of Deep Equivariant Networks](on_universality_of_deep_equivariant_networks.md)

</div>

<!-- RELATED:END -->
