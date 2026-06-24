---
title: >-
  [Paper Note] Bridging ML and Algorithms: Comparison of Hyperbolic Embeddings
description: >-
  [ICLR 2026][Graph Learning][Hyperbolic Embeddings] This is an empirical benchmark paper that "bridges academic silos." The authors bring together 14 hyperbolic embedding methods from the Machine Learning (ML), Network Theory (NT), and Algorithms communities—which have long ignored each other—to compete on 38 real-world networks + 600 simulated networks. They find that the near-linear BFKL algorithm from the 2016 Algorithms community is approximately 100x faster than the popul…
tags:
  - "ICLR 2026"
  - "Graph Learning"
  - "Hyperbolic Embeddings"
  - "Scale-free Networks"
  - "HRG Model"
  - "Benchmark Comparison"
  - "Information Criterion"
  - "Greedy Routing"
date: 2026-05-08
content_hash: f3e79e107ae32788
---

# Bridging ML and Algorithms: Comparison of Hyperbolic Embeddings

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=vxiyM9HJw4](https://openreview.net/forum?id=vxiyM9HJw4)  
**Code**: TBD  
**Area**: Graph Learning / Hyperbolic Geometric Embeddings  
**Keywords**: Hyperbolic Embeddings, Scale-free Networks, HRG Model, Benchmark Comparison, Information Criterion, Greedy Routing  

## TL;DR
This is an empirical benchmark paper that "bridges academic silos." The authors bring together 14 hyperbolic embedding methods from the Machine Learning (ML), Network Theory (NT), and Algorithms communities—which have long ignored each other—to compete on 38 real-world networks + 600 simulated networks. They find that the near-linear BFKL algorithm from the 2016 Algorithms community is approximately 100x faster than the popular Poincaré/Lorentz embeddings in the ML community while maintaining comparable or superior quality. They also propose a new quality metric, ICV, which penalizes high-dimensional and high-radius embeddings.

## Background & Motivation
**Background**: Embedding a network $(V,E)$ into hyperbolic geometry $G$ (mapping $m: V \to G$) is a long-standing problem because the exponential growth property of the hyperbolic plane is naturally suited for representing hierarchical structures and scale-free networks. However, research is scattered across three almost isolated communities: the ML community, represented by Nickel & Kiela’s Poincaré embeddings (NIPS 2017) and Lorentz embeddings (ICML 2018), uses Riemannian Stochastic Gradient Descent (RSGD) to learn embeddings; the Network Theory (NT) community, originating from the Hyperbolic Random Graph (HRG, 2010) model by Krioukov et al., treats hyperbolic geometry as a tool for understanding natural networks; and the Algorithms community focuses on the complexity of embedding algorithms, producing near-linear $\tilde{O}(n)$ algorithms such as BFKL (ESA 2016).

**Limitations of Prior Work**: After reviewing the literature, the authors discovered an awkward reality—these communities **hardly cite each other and are often unaware of each other's existence**. ML papers rarely cite algorithmic work, and vice versa; many ML reviews explicitly claim that Nickel & Kiela (2017) is the "pioneering work in hyperbolic embeddings," completely ignoring the rich accumulation in the NT/Algorithms communities over the preceding decade (e.g., Muscoloni et al. posted hyperbolic embeddings using ML methods on arXiv in February 2016). More importantly, **there is no cross-community horizontal comparative study**, leading to a high likelihood that the ML community is reinventing the wheel and using methods that are several orders of magnitude slower.

**Key Challenge**: The ML community seeks to integrate hyperbolic geometry into differentiable analytical pipelines, thus prioritizing numerical precision. In contrast, the Algorithms/NT communities have long possessed fast and effective unsupervised embedders, but because they are published in different venues and use different quality metrics (MAP/MR vs. GSR/GSF/LL), their results cannot be directly compared.

**Goal**: To conduct the first empirical competition of hyperbolic embedders across the ML, NT, and Algorithms communities with unified metrics and datasets, providing a practical guide for the ML community to "switch to faster algorithms." **Core Idea**: Embeddings are static products that can be used plug-and-play. Since all embedders generate static embeddings, they can be freely replaced within an ML pipeline. Therefore, one only needs to select the one that offers "sufficient quality at higher speed" based on a fair horizontal benchmark. Simultaneously, the **Information Control Value (ICV)** is introduced to correct the optimization illusion that metrics always improve with higher dimensionality.

## Method

### Overall Architecture
This paper does not propose a new embedding algorithm but builds a **unified evaluation framework**: 14 embedders from three communities (Poincaré PE, Lorentz LE, BFKL, Coalescent, KVK, Mercator, LTiling, TreeRep, Anneal, LPCS, HMCS, CLOVE, etc., plus DHRG discrete optimization as post-processing) are run on two types of data—38 real-world networks (7 hierarchies, 21 connectomes, 10 others) and 600 simulated 2D networks generated using the HRG model. They are then scored using six quality metrics from the literature (MAP, MR, GSF, GSR, GRE, LL) plus the self-developed ICV, with a focus on comparing BFKL against 2D Lorentz.

### Key Designs

**1. The "Theoretical Foundation" for Unified Evaluation: HRG Model connects the three communities to the same coordinate system**. The entire comparison is made possible by the common language provided by the Hyperbolic Random Graph (HRG) model. HRG uses parameters $N, R, \alpha, T$ to scatter $N$ points in a hyperbolic disk of radius $R$. Angular coordinates $\phi$ are uniformly distributed, and radial coordinates follow the density $f(r) = \alpha\sinh(\alpha r)/(\cosh(\alpha R)-1)$. Any two points are connected with probability
$$p(a,b) = \left(1 + \exp\frac{\delta(m(a),m(b)) - R}{2T}\right)^{-1}$$
where $\delta$ is the hyperbolic distance. Radial coordinates correspond to "popularity" (smaller is more popular), and angular coordinates correspond to "similarity" (closer is more similar). The resulting graphs naturally possess a power-law degree distribution ($\beta = 2\alpha+1$) and high clustering coefficients—the two hallmarks of real-world scale-free networks. With this generative model, "embedding quality" can be unified as the goodness-of-fit in the maximum likelihood sense, and simulated data can be generated in controllable batches, allowing methods from the three communities to be measured by the same yardstick.

**2. Quantifying "Who is Faster and Better" into a Comparable Metric Matrix**. The authors deliberately synthesized metrics from various communities and explained their orientations: MeanRank (MR, the average number of non-neighbors closer to $u$ than $v$ on edge $u \to v$, $\ge 1$) and Mean Average Precision (MAP) from the ML community favor hierarchical structures where "similar nodes are closer to a common ancestor"; the Greedy Success Rate (GSR), Stretch Factor (GSF), and Greedy Routing Efficiency (GRE) from the NT/Routing community measure whether one can reach a destination by following the nearest neighbor; Log-likelihood (LL) is optimized when connected nodes have distance $<R$ and unconnected nodes have distance $>R$. Unifying these metrics revealed a key conclusion: BFKL is **approximately 100x faster** than 2D LE across all experiments, while MAP/MR quality is comparable. The difference density plots in Figure 4 show a significant proportion of negative values (indicating BFKL is better). Furthermore, logit regression proves that there is **no significant monotonic relationship** between "extra time spent" and "quality improvement," implying that the extra computational power the ML community pays for slow algorithms often does not buy better embeddings.

**3. ICV: Debunking the "High Dimension = Better" Optimization Illusion using Minimum Description Length**. All standard metrics indicate that higher dimensions lead to better embeddings, but the authors point out that this is merely an optimization illusion—reducing dimensionality is equivalent to adding constraints to optimization; without constraints, scores are naturally higher. For a fair comparison, they designed the **Information Control Value (ICV)** based on the Minimum Description Length (MDL) principle, balancing "edge prediction quality" against the "description length of the embedding": embeddings with higher dimensions and larger radii have longer description lengths and are penalized. This penalty has dual rationality: high-radius embeddings are difficult to visualize and more prone to numerical precision issues. Under ICV, the conclusion reverses: **2D embeddings are actually better for most real-world networks**, and the advantage of high-dimensional LE disappears. The authors also implemented this idea as "Penalty"—an improved version of BFKL based on DHRG that actively compresses the embedding radius, further increasing the ICV.

**4. DHRG Discrete Post-processing: Avoiding Numerical Precision Mines via Hyperbolic Tesselation**. The outputs of several embedders (BFKL, LE, CLOVE) can be improved using DHRG (Discrete HRG) as post-processing. instead of mapping nodes to continuous points on the hyperbolic plane, DHRG maps them to the **tiles** of a hyperbolic tessellation (such as a bitruncated order-3 heptagonal tiling), where distances are calculated discretely based on "how many tiles apart." This approach avoids the floating-point precision disasters of continuous coordinates at large radii (typically $R \approx 30$ tiles) and provides algorithmic benefits: given a tile $t_1$ and a set of tiles $T$, the array of "number of tiles at distance $i$ from $t_1$" can be calculated in $O(R^2)$ time, allowing for efficient LL calculation and embedding optimization via local search. In experiments, DHRG generally improved the MAP of various embedders, with significant gains for 2D Lorentz.

## Key Experimental Results

### Main Results (Real Networks vs. Simulated Networks)

| Dimension | Key Findings |
|-----------|--------------|
| Speed | BFKL is **approx. 100x faster** than 2D LE / Poincaré (near-linear $\tilde{O}(n)$ vs. RSGD iterations). |
| Quality (MAP/MR) | BFKL quality is "comparable or better" than 2D LE; Figure 4 shows a substantial proportion of cases where BFKL outperforms LE. |
| Mercator | The full version quality is between BFKL and 2D LE, but it is slower than LE on large graphs; the fast version is usually inferior to BFKL. |
| TreeRep | Good quality on hierarchical data, but significantly worse on FACEBOOK, connectomes, etc. (the cost of learning a tree instead of an embedding). |
| CLOVE (2025 SOTA) | Excellent quality and speed on real networks; however, performance on simulated networks is mediocre, rarely entering the top 10%. |
| KVK | Often achieves the best quality but is extremely slow, even slower than LE. |
| Anneal | Excellent on connectomes (its original domain), but mediocre on other data because it can only produce small-radius embeddings. |
| LPCS / Coalescent | Noticeably weaker on simulated HRG networks (a point for future research). |

### Comparison of Information Criteria (ICV)

| Setting | Standard Metric Conclusion | ICV Conclusion |
|---------|----------------------------|----------------|
| Dim vs. Quality | Higher dimensions are better (including 3D > 2D) | **2D is actually optimal** for most real networks; high-dimensional advantages disappear. |
| Radius | No penalty | Penalizes large radii (numerical risk + visualization difficulty), leading to the Penalty improved version. |

### Key Findings
- **Cross-community Non-reproducibility**: Contrary to Nickel & Kiela (2017), 200D Euclidean SGD embeddings in this paper consistently outperform low-dimensional Poincaré (this non-reproducibility phenomenon was previously studied by Bansal & Benton 2021). Contrary to (2018), Lorentz and Poincaré have comparable quality here, with Poincaré sometimes performing better.
- **Time Investment $\neq$ Quality Return**: Kendall-tau tests show no significant monotonic relationship between extra time spent and MAP improvement (p-values are non-significant at most temperatures).
- **Conclusion**: Embedder components are plug-and-play in ML pipelines; they should be selected based on quality metrics vs. resource consumption—where the Algorithms community's long-ignored BFKL is often the most cost-effective choice.

## Highlights & Insights
- **Highly Valuable Topic Selection**: It reveals and quantifies a long-ignored systemic issue where "three communities do not cite each other and ML uses methods several orders of magnitude slower." It is a classic "Emperor's New Clothes" contribution—relying not on a new model, but on an honest horizontal comparison.
- **ICV Corrects the Core**: Using the MDL principle to identify "high dimension = better" as an optimization illusion and penalizing high radii has corrective significance for the entire hyperbolic embedding evaluation paradigm, rather than being a single-paper trick.
- **Honest Reporting of Non-reproducibility**: It directly points out conflicts between its results and classic ML papers (Nickel & Kiela) and traces them back to known reproducibility studies, demonstrating solid academic integrity.
- **Plug-and-play Practical Positioning**: By emphasizing that all embedders generate static embeddings that are freely replaceable, the conclusions can be immediately translated into actionable advice for ML engineers.

## Limitations & Future Work
- **Single Network Regime**: Simulation experiments only cover the "ultra-small world" regime generated by HRG, excluding other generative models like PSO, nPSO, and other degree distribution regimes (acknowledged by the authors, though not considered a severe threat).
- **ICV is still a Preliminary Proposal**: The authors state that a more systematic study of ICV is left for future work; currently, it has only been verified to flip dimensionality conclusions, and its properties and boundaries of applicability are not yet fully characterized.
- **Unable to Evaluate Angular Coordinate Quality**: Existing quality metrics focus on distance preservation and lack means to compare angular coordinate (similarity space) quality on real embeddings, leaving a gap for evaluating community detection tasks.
- **Some Methods May Be Affected by Hyperparameters**: For instance, LTiling performed poorly; the authors admit this might stem from improper hyperparameter settings or the test graphs being too shallow, leaving room for further robustness testing.
- **Extensible Framework**: New hyperbolic embedders emerge every year; the framework in this paper can easily incorporate them, which is its strength as a "living benchmark."

## Related Work & Insights
- **ML Hyperbolic Embedding Lineage**: Poincaré (Nickel & Kiela 2017) $\to$ Lorentz (2018) $\to$ Product Manifolds (Gu et al. 2019) $\to$ Numerical Precision (Sala et al. 2018, LTiling Yu & De Sa 2019) $\to$ TreeRep (learning trees instead of embeddings).
- **NT/Algorithms Lineage**: HRG (Krioukov 2010) $\to$ Internet Embedding and Greedy Routing (Boguñá 2010) $\to$ HyperMap $O(n^3)$ $\to$ HyperMapCN $O(n^2)$ $\to$ **BFKL Near-linear** (Bläsius 2016) $\to$ DHRG Discrete Optimization (Celińska-Kopczyńska & Kopczyński 2022) $\to$ CLOVE using TSP for communities (Balogh 2025).
- **Insights**: This paper is a model for the "cross-community benchmark" paradigm. When a direction evolves in parallel across different communities, conducting an honest unified comparison is often as valuable as a new model. A direct reminder for GNN/Geometric Deep Learning researchers: before choosing a hyperbolic embedding component, check the near-linear solutions from the Algorithms community; do not assume RSGD is the only option. The ICV concept also suggests that reports of "larger dimension/capacity is better" require calibration using information criteria.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Not in a new algorithm, but in the first systematic horizontal benchmark bridging three communities and the corrective ICV metric. The research perspective is rare and sharp.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 14 methods $\times$ (38 real networks + 600 simulated networks) $\times$ 7 metrics. Large coverage and honest reporting of contradictions with classic papers; points deducted for the single simulation regime and preliminary ICV validation.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear explanation of hyperbolic geometry and community backgrounds; narrative of motivation is compelling. Information intensity is slightly high in the metrics section.
- **Value**: ⭐⭐⭐⭐⭐ — Immediate practical significance for the ML community (switching to 100x faster BFKL) and prompts a rethink of evaluation paradigms. Broad impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Graph Tokenization for Bridging Graphs and Transformers](graph_tokenization_for_bridging_graphs_and_transformers.md)
- [\[ICLR 2026\] Bridging Input Feature Spaces Towards Graph Foundation Models](bridging_input_feature_spaces_towards_graph_foundation_models.md)
- [\[ICLR 2026\] Structurally Human, Semantically Biased: Detecting LLM-Generated References with Embeddings and GNNs](structurally_human_semantically_biased_detecting_llm-generated_references_with_e.md)
- [\[AAAI 2026\] Hyperbolic Continuous Structural Entropy for Hierarchical Clustering](../../AAAI2026/graph_learning/hyperbolic_continuous_structural_entropy_for_hierarchical_clustering.md)
- [\[ICML 2025\] Hyperbolic-PDE GNN: Spectral Graph Neural Networks in the Perspective of A System of Hyperbolic Partial Differential Equations](../../ICML2025/graph_learning/hyperbolic-pde_gnn_spectral_graph_neural_networks_in_the_perspective_of_a_system.md)

</div>

<!-- RELATED:END -->
