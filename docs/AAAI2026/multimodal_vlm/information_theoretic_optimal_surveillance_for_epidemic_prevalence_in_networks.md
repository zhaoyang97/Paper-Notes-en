---
title: >-
  [Paper Note] Information Theoretic Optimal Surveillance for Epidemic Prevalence in Networks
description: >-
  [AAAI 2026][Multimodal VLM][epidemic surveillance] This paper introduces TestPrev, the first epidemic surveillance framework that employs mutual information as an optimization criterion. It selects an optimal subset of n…
tags:
  - "AAAI 2026"
  - "Multimodal VLM"
  - "epidemic surveillance"
  - "information theory"
  - "mutual information"
  - "network spreading"
  - "sensor selection"
date: 2026-05-08
content_hash: d822c4dea280e257
---

# Information Theoretic Optimal Surveillance for Epidemic Prevalence in Networks

**Conference**: AAAI 2026
**arXiv**: [2601.04267](https://arxiv.org/abs/2601.04267)
**Code**: N/A
**Area**: Multimodal VLM
**Keywords**: epidemic surveillance, information theory, mutual information, network spreading, sensor selection

## TL;DR

This paper introduces TestPrev, the first epidemic surveillance framework that employs mutual information as an optimization criterion. It selects an optimal subset of nodes in a network to maximize mutual information with the disease prevalence distribution, thereby providing distribution-level insights into outbreak size that traditional methods cannot offer. The paper proves the NP-hardness of this problem, designs a greedy algorithm GreedyMI, and demonstrates its superiority over baselines on both synthetic and real-world networks.

## Background & Motivation

1. **Existing surveillance methods lack distribution-level insights**: Prior epidemic surveillance research has focused on optimizing individual metrics such as detection probability, detection delay, and peak timing, and is unable to provide complete distributional information about outbreak size. Distribution-level understanding is critical for public health planning (e.g., assessing the risk of large-scale outbreaks).

2. **Surveillance node selection under limited resources**: Large-scale testing is costly and resource-constrained, requiring the selection of the most informative subset of nodes within a limited budget—a core optimization problem in public health.

3. **Classical sensor placement methods are ill-suited for epidemic scenarios**: Information-theoretic sensor placement criteria such as Caselton–Zidek–Krause (CZK) have been successfully applied in environmental monitoring, but their objective $I(X_A; X_{V\setminus A})$ maximizes information about unmonitored node states rather than about the prevalence distribution—a fundamental distinction.

4. **Threshold effects in epidemic spreading require distributional understanding**: Disease outbreaks often exhibit pronounced threshold effects (bimodal distributions between small and large outbreaks). Methods that optimize a single metric (e.g., detection probability) may select node sets that are arbitrarily suboptimal with respect to prevalence mutual information.

5. **Structural differences from existing mutual information criteria**: The CZK objective is submodular and admits a $(1-1/e)$ greedy approximation, whereas the TestPrev objective proposed here can be supermodular in general and cannot be approximated within a $\Theta(\log n)$ factor (unless P=NP), necessitating new algorithmic strategies.

6. **Lack of efficient exact algorithms for special network structures**: While mutual information can be computed efficiently on trees, paths, and under 1-hop propagation, no prior work has systematically exploited these structural properties to accelerate the optimization.

## Method

### Problem Formulation: TestPrev

**Disease model**: The Independent Cascade model $\text{IC}(\lambda, d)$ on a network $G=(V,E)$, where an infected node $u$ activates neighbor $v$ with probability $\lambda_{(u,v)}$, and the process continues until no new infections occur. Weighted prevalence is defined as $Z = \sum w_i X_i$, where $X_v$ denotes the infection state of node $v$.

**Optimization objective**: Given budget $k$, find a node subset $A^* \in \arg\max\, M(A) = I(X_A; Z)$, i.e., maximize the mutual information between the states of monitored nodes and the prevalence. This is equivalent to minimizing the conditional entropy $H(Z \mid X_A)$.

### Theoretical Analysis

- **NP-hardness** (Theorem 1): TestPrev is NP-hard even under 1-hop propagation, and cannot be approximated within a factor of $(1-\varepsilon)\log n$ (unless P=NP), proven by reduction from the minimum set cover problem.
- **Distinction from CZK**: (1) The CZK optimal solution can be arbitrarily poor with respect to prevalence mutual information; (2) The TestPrev objective is supermodular under independent variables (Obs. 3) but also admits submodular instances (Obs. 4)—exhibiting higher structural complexity than CZK.
- **Distinction from detection probability** (Obs. 5): Node selection that maximizes detection likelihood can be $\Theta(n)$ worse than optimal under the TestPrev objective.

### Efficient Algorithms for Special Networks

**1-hop propagation**: Node states are independent on a bipartite graph; the conditional entropy $H(Z \mid X_A) = H(Z_A^-)$ follows a Poisson binomial distribution and can be computed exactly in $O(|W|^2)$. By supermodularity, the greedy heuristic has complexity $O(k|W|^3)$.

**Rooted tree networks**: The EntropyOnTree algorithm proceeds by filtering feasible infection vectors, contracting active-edge paths, removing uninfected nodes, and performing message passing to compute the unconditional prevalence distribution, yielding exact computation of $H(Z \mid X_A)$. The greedy heuristic has complexity $O(k \cdot 2^k \cdot n^3)$, which is polynomial for fixed $k$.

**Path networks**: A closed-form solution for the optimal node spacing is derived (Theorem 6): $g_j = \log\!\left(\tfrac{k+1-j}{k+2-j}\right) / \log \lambda$, solved via the chain rule and binary entropy function.

### GreedyMI: General Greedy Strategy

For general networks, mutual information is estimated via sampling:
1. Sample $T$ cascade realizations from $\text{IC}(\lambda, d)$ to construct a data matrix $D$.
2. Greedily add nodes one at a time, selecting at each step the node $v$ that minimizes the empirical conditional entropy $H_D(Z \mid X_{A \cup \{v\}})$.
3. Empirical entropy is estimated via hash-based grouping, with total complexity $O(T(n+m) + k^2 n^2)$.
4. Sample complexity is $O(n \cdot 2^k / \varepsilon^2)$; in practice, the number of feasible state configurations is far smaller than the theoretical upper bound.

## Key Experimental Results

### Network Datasets and Experimental Setup (Table 1)

| Network Type | Nodes | Edges | Clustering Coefficient | Avg. Shortest Path | Propagation Parameters |
|---|---|---|---|---|---|
| PowLaw (Power-law) | 675.3 | 1118.8 | 0.052 | 4.1 | $\lambda \in \{0.1, 0.2\}$, $d \in \{2, 4\}$ |
| ER (Random graph) | 1000 | 24912.0 | 0.049 | 2.03 | $\lambda \in \{0.05, 0.07\}$, $d \in \{2, 4\}$ |
| HospICU (Real hospital) | 879 | 3575 | 0.599 | 4.31 | $\lambda \in \{0.1, 0.2\}$, $d \in \{2, 4\}$ |

Synthetic networks use 10 replicates each; 30,000 cascades are sampled per scenario. Both known-source and random-source seed scenarios are evaluated.

### GreedyMI vs. Baseline Performance

| Network | Scenario | GreedyMI Advantage | Key Finding |
|---|---|---|---|
| PowLaw | Known-source | Consistently outperforms baselines; gap grows with budget | Degree approximates GreedyMI due to high-degree hub nodes in power-law networks |
| ER | Known-source | Outperforms baselines; expected std. dev. reduced by ~5% | Diminishing returns less pronounced due to uniform degree distribution |
| HospICU | Known-source | Substantially outperforms baselines; expected std. dev. reduced by up to 80% | Vulnerable outperforms Degree, indicating dynamics-based selection can surpass structural selection |
| All networks | Random-source | Still outperforms baselines but with smaller margins | Random sources mitigate the problem of low-degree neighbors being highly vulnerable but low-information |

**A budget of only 2% of nodes achieves 60%+ reduction in prevalence variance.** GreedyMI achieves a better balance between relevance and redundancy compared to top-$k$ methods.

### Sampling Convergence

30,000 cascade samples (approximately 1/34 of the maximum joint alphabet size) are sufficient for convergence. Parameter regimes with large cascades require more samples.

## Highlights & Insights

- **First prevalence mutual information surveillance criterion**: This is the first work to bring the mutual information framework from environmental sensor placement into epidemic surveillance, targeting the outbreak size distribution rather than a single metric, and filling an important theoretical gap.
- **In-depth complexity analysis**: The NP-hardness and inapproximability of TestPrev are rigorously established, with systematic characterization of its fundamental differences from CZK and detection-probability criteria.
- **Algorithm design from special to general cases**: Exact or closed-form solutions are provided for 1-hop, tree, and path networks, while a sampling-based greedy strategy addresses general networks, balancing theoretical rigor and practical applicability.
- **Validation on a real hospital network**: The method is validated on an ICU contact network, where the 80% variance reduction demonstrates substantial value for healthcare-associated infection surveillance.

## Limitations & Future Work

- GreedyMI lacks provable approximation guarantees; the supermodular/non-submodular nature of TestPrev complicates algorithm analysis.
- Closed-form solutions beyond path networks have not been derived, limiting the generalizability of theoretical insights to other graph families.
- The sampling approach converges slowly for large-cascade scenarios, with sample complexity growing exponentially in the budget $k$.
- Only the Independent Cascade (IC) model is considered; extensions to more complex epidemic models such as SIS and SEIR remain unexplored.
- Homogeneous transmission probability $\lambda$ is assumed; in practice, edge-weight heterogeneity may affect node selection strategies.

## Related Work & Insights

| Comparison | Advantage of This Work |
|---|---|
| **Leskovec et al. (2007) sensor placement** | That work optimizes detection delay; this paper proves such approaches can be $\Theta(n)$ worse than optimal under prevalence mutual information. TestPrev provides distribution-level information rather than a single detection metric. |
| **CZK criterion (Caselton & Zidek 1984; Krause et al. 2008)** | CZK maximizes information about unobserved node states but may be ineffective for the prevalence distribution. This paper's objective directly targets prevalence and reveals fundamental structural differences (submodular vs. supermodular). |
| **Christakis & Fowler (2010) social network sensors** | Leverages the "sensor" property of friend nodes for early detection but provides no distributional insight. This paper quantifies the information gain of surveillance via a mutual information framework. |
| **Tsui et al. (2024) active learning selection** | Employs an active learning framework with iterative test feedback for node selection. This paper addresses one-shot (non-adaptive) selection but provides stronger theoretical foundations and distribution-level optimization. |

## Rating

- ⭐⭐⭐⭐⭐ **Theoretical Depth**: Solid contributions including NP-hardness proofs, inapproximability results, supermodularity analysis, and closed-form derivations.
- ⭐⭐⭐⭐ **Experimental Thoroughness**: Synthetic and real-world networks, multiple propagation parameters, two seed scenarios, solution structure analysis, and convergence analysis.
- ⭐⭐⭐⭐ **Value**: Clear application prospects in public health surveillance, particularly for healthcare-associated infections, with high effectiveness at low budget.
- ⭐⭐⭐⭐ **Writing Quality**: Rigorous theoretical derivations and clear problem motivation, though the heavy notation requires background in information theory and graph theory.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Conditional Information Bottleneck for Multimodal Fusion: Overcoming Shortcut Learning in Sarcasm Detection](conditional_information_bottleneck_for_multimodal_fusion_overcoming_shortcut_lea.md)
- [\[AAAI 2026\] Exploring LLMs for Scientific Information Extraction using the SciEx Framework](exploring_llms_for_scientific_information_extraction_using_the_sciex_framework.md)
- [\[ICCV 2025\] Evading Data Provenance in Deep Neural Networks](../../ICCV2025/multimodal_vlm/evading_data_provenance_in_deep_neural_networks.md)
- [\[NeurIPS 2025\] Multimodal Bandits: Regret Lower Bounds and Optimal Algorithms](../../NeurIPS2025/multimodal_vlm/multimodal_bandits_regret_lower_bounds_and_optimal_algorithms.md)
- [\[ICLR 2026\] LiveWeb-IE: A Benchmark For Online Web Information Extraction](../../ICLR2026/multimodal_vlm/liveweb-ie_a_benchmark_for_online_web_information_extraction.md)

</div>

<!-- RELATED:END -->
