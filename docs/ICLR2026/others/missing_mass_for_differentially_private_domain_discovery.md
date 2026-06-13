---
title: >-
  [Paper Note] Missing Mass for Differentially Private Domain Discovery
description: >-
  [ICLR 2026][differential privacy] This paper revisits the differentially private domain discovery problem through the lens of missing mass…
tags:
  - "ICLR 2026"
  - "differential privacy"
  - "domain discovery"
  - "missing mass"
  - "Weighted Gaussian Mechanism"
  - "Zipfian data"
  - "top-k selection"
date: 2026-05-08
content_hash: e5e07976c147a337
---

# Missing Mass for Differentially Private Domain Discovery

**Conference**: ICLR 2026
**arXiv**: [2603.14016](https://arxiv.org/abs/2603.14016)  
**Code**: Available (provided in appendix)  
**Area**: Other
**Keywords**: differential privacy, domain discovery, missing mass, Weighted Gaussian Mechanism, Zipfian data, top-k selection

## TL;DR
This paper revisits the differentially private domain discovery problem through the lens of missing mass, providing the first near-optimal $\ell_1$ missing mass upper bounds for the simple and scalable Weighted Gaussian Mechanism (WGM) on Zipfian data, as well as distribution-free $\ell_\infty$ missing mass guarantees. WGM is further applied as a domain discovery preprocessing step for private top-$k$ and $k$-hitting set problems over unknown domains, with theoretical results validated on six real-world datasets.

## Background & Motivation

**Background**: In modern data analysis, many applications involve data domains that are either unknown a priori or impractically large (e.g., query logs, user reviews, purchase histories), making domain discovery a critical preprocessing step for efficient downstream analysis. Differential privacy (DP) provides formal guarantees for privacy-preserving analysis of sensitive data, while simultaneously complicating the domain discovery task. DP set union (also known as key selection or partition selection) has become a core component of several industrial DP frameworks, including Google Plume, Apple, and OpenDP.

**Limitations of Prior Work**: Despite a substantial body of work on DP set union algorithms, provable utility guarantees are nearly absent. Existing results (e.g., Desfontaines et al. 2022, Chen et al. 2025) express utility relative to other algorithms rather than as absolute performance guarantees, making it difficult to assess the actual performance of existing algorithms and the extent to which they can be improved. Furthermore, algorithms for top-$k$ selection and $k$-hitting set under unknown domain settings remain scarce.

**Key Challenge**: Traditionally, the utility of DP set union is measured by cardinality (i.e., how many unique items are recovered), which ignores frequency information. An algorithm that recovers many low-frequency items while missing critical high-frequency ones may perform well by cardinality metrics yet offer poor practical utility. Missing mass—the fraction of total frequency accounted for by unrecovered items—is a more meaningful metric, yet no theoretical missing mass guarantees for DP domain discovery had been established prior to this work.

**Goal**: (1) Establish provable missing mass upper bounds for WGM, addressing the open question of the absolute performance of DP domain discovery algorithms; (2) extend WGM's missing mass guarantees to downstream problems (top-$k$, $k$-hitting set); (3) empirically validate the theoretical results.

**Key Insight**: By redefining the utility measure from cardinality to missing mass (i.e., the fraction of total frequency covered by recovered items), the authors find that elegant theoretical guarantees can be established for the simple WGM. The Zipfian (power-law) distribution assumption provides a natural measure of data complexity, enabling meaningful guarantees in settings commonly encountered in practice.

**Core Idea**: Replace cardinality with missing mass as the utility measure for DP domain discovery; establish near-optimal guarantees for WGM on Zipfian data; and extend WGM as a domain discovery preprocessing step to top-$k$ and $k$-hitting set problems.

## Method

### Overall Architecture
The overall framework centers on the Weighted Gaussian Mechanism (WGM). For the DP set union problem, WGM directly outputs a subset of high-frequency items. For downstream problems such as top-$k$ and $k$-hitting set, a two-phase approach is adopted (Algorithm 2): (1) run WGM with half the privacy budget to obtain a candidate domain $D$; (2) run a standard known-domain algorithm on $D$ with the remaining privacy budget. Overall privacy is guaranteed via the basic composition theorem.

### Key Designs

1. **Weighted Gaussian Mechanism (WGM)**:

    - **Function**: Discovers high-frequency items in a dataset under differential privacy constraints.
    - **Mechanism**: WGM operates in three stages. **Subsampling**: sample without replacement at most $\Delta_0$ items from each user's item set to bound user contribution. **Weighted histogram**: construct a weighted histogram over the subsampled data, where each user's contribution to item $x$ is weighted by $1/\sqrt{|\tilde{W}_i|}$ (normalized so that each user's $\ell_2$ norm equals 1). **Noise thresholding**: add zero-mean Gaussian noise with standard deviation $\sigma$ to each weighted count and output items whose noisy counts exceed threshold $T$. Privacy parameters satisfy $\sigma = \Theta(\frac{1}{\epsilon}\sqrt{\log(1/\delta)})$ and $T = \tilde{\Theta}(\max\{\sigma, 1\})$.
    - **Design Motivation**: WGM's weighting scheme is its core advantage—by $\ell_2$-normalizing user contributions rather than applying simple truncation, it retains more useful information. Compared to more complex sequential or policy mechanisms, WGM is simple, parallelizable, and scalable.

2. **$\ell_1$ Missing Mass Upper Bound on Zipfian Data (Theorem 3.3)**:

    - **Function**: Establishes a high-probability upper bound on the missing mass of WGM for data following a Zipfian distribution.
    - **Mechanism**: Assuming the data satisfies the $(C,s)$-Zipfian condition (i.e., the $r$-th largest frequency satisfies $N_{(r)}/N \leq C/r^s$ with $s>1$), the paper proves that WGM achieves missing mass
$$\text{MM}(W,S) = \tilde{O}\left(\frac{C^{1/s}}{s-1} \left(\frac{\max_i|W_i|}{\epsilon N \sqrt{q^*}}\right)^{(s-1)/s}\right),$$
where $q^* = \min\{|\max_i W_i|, \Delta_0\}$. The proof relies on three auxiliary lemmas: an upper bound on missing mass after subsampling, a guarantee that high-frequency items remain high-frequency after subsampling, and an upper bound on the frequency of items missed by the thresholding step.
    - **Design Motivation**: Missing mass decreases as $N$ grows and decreases as $s$ increases (i.e., distributions more concentrated at the head)—both are intuitively correct. The lower bound theorem (Theorem 3.5) proves that the dependence on $\epsilon$ and $N$ is near-optimal.

3. **Distribution-Free $\ell_\infty$ Missing Mass Guarantee and Downstream Applications**:

    - **Function**: Establishes missing mass guarantees without requiring the Zipfian assumption, and extends results to top-$k$ and $k$-hitting set.
    - **Mechanism**: Theorem 3.6 proves the $\ell_\infty$ missing mass upper bound for WGM:
$$\text{MM}_\infty(W,S) \leq \tilde{O}\left(\frac{\max_i|W_i|}{\epsilon N \sqrt{q^*}}\right),$$
without requiring the Zipfian assumption. For top-$k$ (Theorem 4.3), combining WGM with the Peeling Exponential Mechanism yields
$$\text{MM}^k(W,S) \leq \tilde{O}\left(\frac{k}{N}\left(\frac{\max_i|W_i|}{\epsilon\sqrt{q^*}} + \frac{\sqrt{k}\log M}{\epsilon}\right)\right).$$
For $k$-hitting set (Theorem 4.5), combining WGM with the User Peeling Mechanism yields a $(1-1/e)$ approximation ratio with controllable additive error.
    - **Design Motivation**: The $\ell_\infty$ guarantee, while weaker than $\ell_1$, requires no distributional assumptions and suffices to derive guarantees for downstream problems. The key insight of the two-phase approach is that the candidate domain produced by WGM, although smaller than $\bigcup_i W_i$, still contains high-quality items, thereby reducing the search space and simplifying downstream problems.

### Loss & Training
This is a purely theoretical work and involves no machine learning training. The core contributions are rigorous mathematical guarantees in the form of upper bounds and nearly matching lower bounds.

## Key Experimental Results

### Main Results

Missing mass of DP set union is evaluated on six real-world datasets under $(1, 10^{-5})$-DP:

| Dataset | Type | WGM Performance |
|--------|------|---------|
| Reddit | Large | Gap from policy mechanisms <5% |
| Amazon Games | Large | Gap from policy mechanisms <5% |
| Movie Reviews | Large | Gap from policy mechanisms <5% |
| Steam Games | Small | Gap from policy mechanisms <5% |
| Amazon Magazine | Small | Gap from policy mechanisms <5% |
| Amazon Pantry | Small | Gap from policy mechanisms <5% |

Note: WGM's missing mass is within 5% of the computationally intensive Policy Gaussian/Policy Greedy mechanisms across all datasets. By contrast, under cardinality-based metrics used in prior experiments, sequential methods typically output more than twice as many unique items.

### Downstream Task Experiments

Top-$k$ missing mass (small datasets, $\Delta_0 = 100$):

| Method | Steam Games | Amazon Magazine | Amazon Pantry |
|------|------------|-----------------|---------------|
| Limited-Domain ($\bar{k}=k$) | High MM | High MM | High MM |
| Limited-Domain ($\bar{k}=5k$) | Medium MM | Medium MM | Medium MM |
| Limited-Domain ($\bar{k}=\infty$) | Medium MM | Medium MM | Medium MM |
| **WGM-then-top-$k$ (Ours)** | **Lowest MM** | **Lowest MM** | **Lowest MM** |

WGM-then-top-$k$ consistently outperforms the Limited-Domain baseline across all datasets and all values of $k$, with the advantage growing as $k$ increases.

$k$-Hitting Set (small datasets): WGM-then-hitting-set performs comparably to the non-private greedy algorithm and the private greedy algorithm assuming a public domain, and even surpasses the latter on Steam Games and Amazon Magazine—because the smaller candidate domain produced by WGM reduces the difficulty of the downstream search problem.

### Key Findings
- **The shift from cardinality to missing mass is critical**: Under cardinality metrics, WGM is substantially inferior to sequential methods (differing by roughly a factor of 2 in the number of output items); under missing mass, the gap shrinks to under 5%. This is because cardinality ignores item frequency, and WGM, despite outputting fewer items, better covers high-frequency ones.
- **Simplicity does not imply weakness for WGM**: As a simple, parallelizable, and scalable method, WGM nearly matches computationally intensive policy and sequential methods under the missing mass metric.
- **Counterintuitive advantage of the two-phase approach**: In the $k$-hitting set setting, using WGM as a domain discovery step actually helps downstream algorithms achieve better results by narrowing the candidate domain.
- **Practical value of theoretical guarantees**: Theorem 3.5 provides a nearly matching lower bound, demonstrating that WGM's missing mass performance is indeed close to the optimal achievable by any DP algorithm satisfying the soundness assumption.

## Highlights & Insights
- **The profound impact of metric choice**: Switching from cardinality to missing mass not only changes the perceived performance of WGM (from "substantially inferior" to "near-optimal") but also enables, for the first time, the derivation of absolute utility guarantees. This underscores the critical importance of selecting appropriate metrics in theoretical research.
- **Theoretical vindication of a simple method**: Given that WGM is already widely deployed in industry, this paper provides rigorous theoretical support for its use—an important research paradigm of establishing formal guarantees for methods already validated in practice.
- **Generality of the two-phase composition**: The paradigm of "first apply WGM for domain discovery, then apply a known-domain algorithm" can be extended to any DP problem requiring unknown-domain handling.

## Limitations & Future Work
- **$\ell_1$ guarantees require the Zipfian assumption**: Although Zipfian distributions are common in practice, this assumption limits the generality of the theoretical guarantees. No meaningful guarantees are provided for the case $s \leq 1$.
- **Upper and lower bounds for top-$k$ and $k$-hitting set are not matched**: Gaps remain, and closing them is a natural direction for future work.
- **Subsampling strategy can be improved**: All methods use uniform sampling without replacement; the adaptive weighted subsampling of Chen et al. (2025) may further reduce missing mass.
- **Experimental scale is limited**: On large datasets, top-$k$ and $k$-hitting set nearly achieve zero missing mass, providing insufficient stress-testing of algorithm limits.
- Future work may consider extending the missing mass framework to other DP problems, such as frequency moment estimation and histogram release.

## Related Work & Insights
- **vs. Policy Gaussian/Greedy (Gopi et al. 2020, Carvalho et al. 2022)**: These sequential/policy methods perform better under cardinality metrics but incur high computational cost and are not parallelizable. This paper shows that under the more practically meaningful missing mass metric, the simple WGM is nearly equivalent.
- **vs. Limited-Domain Top-$k$ (Durfee & Rogers 2019)**: This baseline requires an additional hyperparameter $\bar{k}$ and provides weaker guarantees ($k$-relative error); WGM-then-top-$k$ consistently outperforms it under the stricter missing mass metric.
- **vs. Desfontaines et al. 2022, Chen et al. 2025**: Their utility results are relative (compared to other algorithms); this paper establishes the first absolute utility guarantees.

## Rating
- Novelty: ⭐⭐⭐⭐ The shift to the missing mass perspective is conceptually straightforward yet yields deep theoretical insights and practical value.
- Experimental Thoroughness: ⭐⭐⭐⭐ Six real-world datasets covering diverse scales and domains, with experimental validation across all three problems.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical exposition, clear motivation, and well-structured theorem statements.
- Value: ⭐⭐⭐⭐ Provides the first absolute theoretical guarantees for WGM, which is already widely adopted in industry, with significant practical implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Improved Differentially Private Algorithms for Rank Aggregation](../../AAAI2026/others/improved_differentially_private_algorithms_for_rank_aggregation.md)
- [\[ICLR 2026\] Noise-Aware Generalization: Robustness to In-Domain Noise and Out-of-Domain Generalization](noise-aware_generalization_robustness_to_in-domain_noise_and_out-of-domain_gener.md)
- [\[ICLR 2026\] Learning Structure-Semantic Evolution Trajectories for Graph Domain Adaptation](learning_structure-semantic_evolution_trajectories_for_graph_domain_adaptation.md)
- [\[ICLR 2026\] Distributionally Robust Classification for Multi-Source Unsupervised Domain Adaptation](distributionally_robust_classification_for_multi-source_unsupervised_domain_adap.md)
- [\[ICLR 2026\] Learning Adaptive Distribution Alignment with Neural Characteristic Function for Graph Domain Adaptation](learning_adaptive_distribution_alignment_with_neural_characteristic_function_for.md)

</div>

<!-- RELATED:END -->
