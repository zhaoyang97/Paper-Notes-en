---
title: >-
  [Paper Note] On the Trade-off Between Expressivity and Privacy in Graph Representation Learning
description: >-
  [ICLR 2026][Graph Learning][Paper Note] This paper theoretically characterizes the fundamental tension between "expressivity of graph representations" and "edge-level differential privacy" for the first time. It proposes using **noisy homomorphism density vectors** as graph embeddings: these maintain full discriminative power in expectation while injecting c
tags:
  - ICLR 2026
  - Graph Learning
date: 2026-05-08
content_hash: fcb0db1d076e8929
---
# On the Trade-off Between Expressivity and Privacy in Graph Representation Learning

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=XXLDvwMwbe](https://openreview.net/forum?id=XXLDvwMwbe)  
**Code**: [https://github.com/tamaramagdr/private-homcounts](https://github.com/tamaramagdr/private-homcounts)  
**Area**: Graph Representation Learning / Differential Privacy / Expressivity Theory  
**Keywords**: Homomorphism Density, Differential Privacy, Graph Expressivity, Smooth Sensitivity, Privacy-Expressivity Trade-off

## TL;DR
This paper theoretically characterizes the fundamental tension between "expressivity of graph representations" and "edge-level differential privacy" for the first time. It proposes using **noisy homomorphism density vectors** as graph embeddings: these maintain full discriminative power in expectation while injecting calibrated noise based on the smooth sensitivity of each density to satisfy formal differential privacy guarantees. It further proves an explicit trade-off where more expressive pattern classes require higher noise levels.

## Background & Motivation
**Background**: There are two nearly orthogonal research lines in graph representation learning. One is **expressivity analysis**, which focuses on whether an algorithm can distinguish two non-isomorphic graphs—typically characterized by $k$-WL hierarchies or homomorphism counts. The other is **graph privacy**, which concerns whether an algorithm leaks sensitive structural information (e.g., edge existence), typically addressed by adding noise to satisfy Differential Privacy (DP).

**Limitations of Prior Work**: These two objectives are naturally in conflict. Consider two non-isomorphic graphs $G_1, G_2$ differing by only one edge: an expressive algorithm **must** produce different embeddings $\varphi(G_1)\ne\varphi(G_2)$ to capture structural differences; however, an edge-level DP algorithm **must** make these embeddings indistinguishable $\varphi(G_1)\approx\varphi(G_2)$, otherwise an attacker could confirm the existence of that edge. Existing DP graph learning methods focus only on optimal utility under privacy constraints and **provide no expressivity guarantees**; conversely, expressivity analysis work never considers privacy.

**Key Challenge**: Expressivity requires "distinguishing neighboring graphs," while privacy requires "blunting differences between neighboring graphs"—they are directly opposed at the scale of neighboring graphs (minimal perturbations). Prior to this, almost no work formally studied the trade-off between privacy, expressivity, and utility.

**Goal**: To investigate the extent to which embeddings can achieve "provable expressivity and provable privacy" simultaneously for graph-level tasks under edge-level privacy guarantees, and to quantify this trade-off.

**Key Insight**: The authors leverage a simple but crucial observation—**DP noise is zero-mean**. If the embedding itself is an "expectation-complete" representation, adding zero-mean noise does not destroy its expectation, thus "preserving expressivity in expectation." Homomorphism counts are ideal tools for this: Lovász's classical results show that homomorphism counts can distinguish any pair of non-isomorphic graphs, and normalized homomorphism densities can construct expectation-complete embeddings.

**Core Idea**: Use "noisy homomorphism density vectors" as graph embeddings—calibrating Gaussian noise according to the sensitivity of each density to single-edge perturbations. This ensures overlapping distributions for neighboring graphs (satisfying DP) while maintaining an expectation equal to the noise-free densities (preserving expressivity) due to the zero-mean property of the noise.

## Method

### Overall Architecture
The method is a one-time preprocessing pipeline: "Sample patterns → Calculate homomorphism densities → Calibrate noise via smooth sensitivity → Release private embeddings." The input is an edge-sensitive graph $G$, and the output is a real-valued vector embedding $\tilde t(\mathcal F, G)$ satisfying tCDP. This embedding can be fed into **any** downstream learner (k-NN, SVM, Random Forest, etc.), incurring no additional privacy cost due to the post-processing property of DP.

Specifically: A set of patterns $\mathbf F=(F_1,\dots,F_d)$ is sampled from a distribution $\mathcal D$ with full support over the target graph class $\mathcal F$; the choice of pattern class determines the achievable expressivity level. For each pattern, the normalized homomorphism density $t(F_i,G)$ is calculated. Smooth sensitivity bounds are then used to calculate the required noise scale for each graph locally, and calibrated zero-mean Gaussian noise is injected. Finally, the node count $|V(G)|$ is concatenated to the embedding to resolve the degeneracy where homomorphism densities cannot distinguish a graph from its blow-up.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Input: Edge-sensitive graph G"] --> B["Sample pattern vector from distribution D<br/>F = (F1...Fd), pattern class determines expressivity"]
    B --> C["Noisy Homomorphism Density Embedding<br/>Calculate t(F,G) and add zero-mean noise<br/>Preserves discriminative power in expectation"]
    C --> D["Smooth Sensitivity Calibration<br/>Determine noise scale via local sensitivity<br/>Provides tCDP guarantee"]
    D -->|Concatenate V(G) to break blow-up degeneracy| E["Private Embedding t̃(F,G)"]
    E -->|DP post-processing, zero extra downstream cost| F["Downstream: k-NN / SVM / RF<br/>Graph Classification or Regression"]
    B -.->|Stronger pattern classes yield higher expressivity<br/>but require more noise injection| D
```

### Key Designs

**1. Noisy Homomorphism Density Embedding: Trading realizations for privacy, guarding expressivity via expectation**

To solve the core contradiction where expressivity and privacy oppose each other at the neighboring graph scale, the authors shift the discriminative focus from "individual realizations" to "expectation." Homomorphism density is defined as the normalized count $t(F,G)=\dfrac{\hom(F,G)}{|V(G)|^{|V(F)|}}$. The noisy embedding is $\tilde t(F,G)=t(F,G)+N$, where $N$ is zero-mean noise. The key is: although the embedding is no longer permutation-invariant after adding noise (which DP requires via randomization), since $\mathbb E[N]=0$, it holds that $\mathbb E[\tilde t(F,G)]=t(F,G)$. Thus, the embedding is equal to the noise-free density in expectation.

Ours proves (Theorem 4.1) that as long as patterns $F\sim\mathcal D$ and $\mathcal D$ has full support over $\mathcal F$, a single noisy density is $\mathcal F$-expectation-expressive. When $\mathcal F=\mathcal G$ (all graphs), it is even expectation-complete. Furthermore (Theorem 4.2), when the number of sampled patterns $d$ is sufficiently large, the noisy density vector is truly $\mathcal F$-expressive with probability at least $1-\theta$. Essentially, noise not only preserves expressivity in expectation but also maintains full discrimination with high probability. A technical detail is that homomorphism densities cannot distinguish $G$ from its $p$-blowup; concatenating the node count $|V(G)|$ breaks this degeneracy at "zero privacy cost" since neighboring graphs have the same number of nodes.

**2. Smooth Sensitivity Calibration: Compressing noise to usable levels via local sensitivity**

Addressing the practical issue where noise based on worst-case scenarios (global sensitivity) makes embeddings unusable, the authors refine the noise scale using the smooth sensitivity framework. A counting lemma provides the global sensitivity upper bound: for neighboring graphs $G\sim_1 G'$, $GS_{t,F}=|t(F,G)-t(F,G')|\le e(F)\,d_\square(G,G')=2e(F)/n^2$. However, global sensitivity characterizes the worst behavior across all graphs, leading to poor performance. Local sensitivity $LS_{t,F}(G)=\max_{G':d_{\text{edge}}(G,G')\le 1}|t(F,G)-t(F,G')|$ is usually much smaller, but adding noise based on it directly does not guarantee DP. Thus, the authors use **smooth sensitivity**—a smooth upper bound on local sensitivity:

$$S_{t,F}(G)=\max_{G'\in\mathcal G}\Big(e^{-\beta\,d_{\text{edge}}(G,G')}\cdot LS_{t,F}(G')\Big).$$

For density vectors, the overall smooth sensitivity is bounded by the $\ell_2$ norm $S_{t,*}(G)=\|S_{t,F_1}(G),\dots,S_{t,F_d}(G)\|_2$ (Proposition 4.4). When the maximum degree $\Delta_{\max}$ is bounded (a common prior), Ours provides a tighter bound (Theorem 4.5): for a pattern $F$ with $m>1$ nodes,

$$|t(F,G)-t(F,G')|\le\frac{2e(F)}{n^2}\Big(\frac{\Delta_{\max}}{n}\Big)^{m-2}.$$

For large graphs and patterns, $(\Delta_{\max}/n)^{m-2}\ll 1$, making this bound much tighter than the counting lemma. The final main result (Theorem 4.6) defines the private mechanism: adding Gaussian noise $\tilde t(\mathbf F,G)=t(\mathbf F,G)+N\!\big(0,\frac{[S_{t,*}(G)]^2}{2\rho'}I_d\big)$, which satisfies $\big(2\rho'+d\cdot 4\beta^2,\ \frac{1}{4\beta}\big)$-tCDP. Because smooth sensitivity is bounded by global sensitivity, this mechanism's privacy-utility trade-off is significantly better than the standard Gaussian mechanism.

**3. Pattern Class as Trade-off Knob: Stronger expressivity requires more noise**

This is the central theoretical insight: quantifying the abstract "expressivity vs. privacy" into an operational knob. Ours proves (Proposition 4.9) that for a given privacy parameter $\rho'$, the variance of Gaussian noise required to achieve the guarantees of Theorem 4.8 is:

$$\sigma^2=O\!\Big(\big(\max_{F\in\mathcal F}e(F)\big)^2/n^4\Big).$$

The noise level is determined by the maximum number of edges $\max_{F}e(F)$ in the pattern class. Stronger expressivity typically implies patterns with higher edge counts. By linking "homomorphism-distinguishing closed graph classes" to specific GNN architectures: MPNN (1-WL) corresponds to trees ($e(F)\le m-1$), $k$-FGNN ($k$-WL) corresponds to graphs with treewidth $\le k$ ($e(F)\le km-\binom{k+1}{2}$), etc. The conclusion is clear: to match the expressivity of a certain GNN in expectation, sample patterns from the corresponding class; however, **stronger classes with more edges require more noise, potentially lowering utility**.

### Loss & Training
Ours does not train the embeddings—homomorphism densities are deterministic calculations with added noise, serving as one-time preprocessing. While homomorphism counting is generally NP-hard, polynomial-time algorithms $O(|V(F)|\,|V(G)|^{\text{tw}(F)+1})$ exist for patterns with bounded treewidth. Downstream, basic learners (k-NN/SVM/RF) are used without end-to-end training.

## Key Experimental Results

Experiments aim to verify: Q1 whether stronger pattern classes improve performance under fixed privacy budgets; Q2 how utility degrades as privacy requirements increase; Q3 if private embeddings are practically useful. Datasets include OGBG molecular data, social networks, and synthetic SBMs. $d=50$, $\rho'\in[10^{-8},1]$, converted to $(\epsilon,\delta)$-DP for interpretation.

### Main Results (OGBG Datasets, Utility and Attack Success Rate)

| Setting | MOLHIV (AUC↑) | MOLBBBP (AUC↑) | MOLBACE (AUC↑) | MOLLIPO (RMSE↓) |
|------|------|------|------|------|
| Ours Private ($\epsilon=1$) utility | **0.692** | **0.602** | **0.652** | **1.086** |
| Ours Private ($\epsilon=1$) Attack Rate | 0.003 | 0.025 | 0.027 | 0.011 |
| Ours Non-private ($\epsilon=\infty$) utility | 0.745 | 0.644 | 0.752 | 1.055 |
| Ours Non-private ($\epsilon=\infty$) Attack Rate | 0.955 | 1.000 | 0.990 | 0.992 |
| RR baseline Private ($\epsilon=1$) | 0.488 | 0.440 | 0.457 | 1.568 |
| DPRR baseline Private ($\epsilon=1$) | 0.595 | 0.539 | 0.648 | 1.499 |

At $\epsilon=1$, the private embeddings outperform RR/DPRR GNN baselines across all OGBG datasets. In MOLHIV/MOLBBBP/MOLLIPO, utility remains above 90% of the non-private version, while attack success rates drop from near 1.0 to below 0.03.

### Ablation Study (Expressivity-Privacy Trade-off Q1/Q2)

| Experiment | Setting | Observation | Conclusion |
|------|------|------|------|
| SBM (Q1) | Cycle vs. Tree patterns | Cycle densities perform significantly better | Stronger patterns improve performance for struct-heavy tasks |
| MOLHIV (Q2) | Max Treewidth tw $\in\{1,2,3\}$ | AUC decreases as tw increases | In this task, tw=1 is sufficient; stronger patterns only add noise |

### Key Findings
- **Zero-mean noise is the pivot**: By shifting discrimination to expectation, noise and expressivity are no longer in hard conflict.
- **Smooth/Bounded-degree sensitivity is essential**: Without refined sensitivity bounds, the noise levels would render embeddings unusable.
- **Stronger expressivity is not always better**: The choice depends on the task; on MOLHIV, higher treewidth patterns only lead to utility drops due to increased noise.
- **Private embeddings are highly informative**: Even with simple k-NN/SVM classifiers, results approach state-of-the-art non-private baselines on network data.

## Highlights & Insights
- **Repurposing DP noise as a "talisman" for expressivity**: Most view DP noise as pure loss; Ours argues that for "expectation-complete" representations, zero-mean noise preserves expressivity in expectation.
- **Quantifying the trade-off via $e(F)$**: The abstract tension is compressed into a calculable scaling $\sigma^2=O((\max e(F))^2/n^4)$, linking GNN-homomorphism correspondences directly to privacy costs.
- **Transferable Logic**: The "Expectation-complete representation + Zero-mean DP noise" recipe can be generalized to other scenarios requiring both discrimination and privacy.

## Limitations & Future Work
- **Inherent Homomorphism Costs**: Counting can be expensive; more importantly, relying solely on structural densities ignores node/edge features, which may limit performance in feature-dominant tasks.
- **Scope**: Designed specifically for edge-level privacy in graph-level tasks; node-level privacy or node-level tasks are not covered.
- **Degree Bound Dependence**: Tighter bounds rely on a known maximum degree $\Delta_{\max}$; without it, one must use private estimation or looser global bounds.
- **Future Directions**: Finer sensitivity analysis for specific graph classes and private encoding of edge features.

## Related Work & Insights
- **vs. DP Graph Learning (e.g., DPRR)**: Previous works pursue utility under privacy but lack expressivity guarantees; Ours provides both and outperforms them with simpler classifiers.
- **vs. Homomorphism Expressivity (e.g., Welke et al.)**: Prior works use densities for high-expressivity embeddings but ignore privacy; Ours builds upon their completeness to prove expressivity remains in expectation after noise.
- **vs. k-WL Analysis**: While traditional analysis identifies abstract upper bounds, Ours maps expressivity levels to pattern edge counts $e(F)$, enabling the first quantified trade-off with privacy.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Robustness in Text-Attributed Graph Learning: Insights, Trade-offs, and New Defenses](robustness_in_text-attributed_graph_learning_insights_trade-offs_and_new_defense.md)
- [\[ICLR 2026\] Graph Representational Learning: When Does More Expressivity Hurt Generalization?](graph_representational_learning_when_does_more_expressivity_hurt_generalization.md)
- [\[ICLR 2026\] Topology Matters in RTL Circuit Representation Learning](topology_matters_in_rtl_circuit_representation_learning.md)
- [\[ICLR 2026\] Temporal Graph Thumbnail: Robust Representation Learning with Global Evolutionary Skeleton](temporal_graph_thumbnail_robust_representation_learning_with_global_evolutionary.md)
- [\[ICML 2026\] T-GINEE: A Tensor-Based Multilayer Graph Representation Learning](../../ICML2026/graph_learning/t-ginee_a_tensor-based_multilayer_graph_representation_learning.md)

</div>

<!-- RELATED:END -->
