---
title: >-
  [Paper Note] How hard is learning to cut? Trade-offs and sample complexity
description: >-
  [ICLR 2026][learning_theory][Chvátal-Gomory cut] This paper provides the first **lower bound** on sample complexity for "learning-to-cut." It proves that for both branch-and-cut (B&C) tree size and gap closed scoring functions, the number of samples required to learn a mapping to cuts is at least equivalent (up to a constant factor) to fitting a general target functi
tags:
  - ICLR 2026
  - learning_theory
  - Chvátal-Gomory cut
date: 2026-05-08
content_hash: 82f465bbcf1809ae
---
# How hard is learning to cut? Trade-offs and sample complexity

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=pc3ZW7lmaV](https://openreview.net/forum?id=pc3ZW7lmaV)  
**Code**: Not released  
**Area**: learning_theory  
**Keywords**: Learning theory, sample complexity, learning-to-cut, branch-and-bound, integer programming, Chvátal-Gomory cut, VC dimension, fat-shattering dimension  

## TL;DR
This paper provides the first **lower bound** on sample complexity for "learning-to-cut." It proves that for both branch-and-cut (B&C) tree size and gap closed scoring functions, the number of samples required to learn a mapping to cuts is at least equivalent (up to a constant factor) to fitting a general target function using the same function class. The lower bounds closely match known upper bounds, suggesting both scores are comparable in learnability. Experiments using GNNs further corroborate that gap closed serves as a robust proxy for tree size.

## Background & Motivation

**Background**: Branch-and-cut (B&C) is the cornerstone of integer linear programming (ILP/MILP) solvers, where "cut selection" (choosing which cut to add to the LP relaxation) is a pivotal decision. Recent works use machine learning (imitation learning, reinforcement learning, neural scoring) to replace manual heuristics. Two mainstream scoring functions evaluate cuts: (i) **B&C tree size**—the relative reduction in tree size after adding a cut, which aligns with runtime but is computationally expensive; (ii) **gap closed**—the improvement in the LP relaxation value, which is significantly cheaper and widely used as a proxy for tree size.

**Limitations of Prior Work**: A fundamental question in learning algorithms is the **sample complexity**: how many training samples are needed for generalization over an unknown distribution. Existing theories (Balcan et al. 2021, Cheng et al. 2024) only provide **upper bounds** for specific algorithms (e.g., fixed or neural-generated CG weights) based on tree size. No prior work has established a **lower bound** or provided a principled comparison between the two scoring criteria.

**Key Challenge**: While gap closed is a "proxy" for tree size, the two are difficult to compare directly—a cut might significantly reduce tree size without closing the LP gap, and vice-versa (the paper provides a counter-example with $\max 5x_1+8x_2$). How effective is this proxy? Which score is harder to learn? If one is inherently more complex, it would dictate algorithm design and evaluation standards.

**Goal**: (1) Provide theoretical lower bounds on sample complexity applicable to broad function classes (neural networks, GNNs) and compare them with upper bounds for tightness; (2) Empirically verify whether gap closed is a viable proxy for tree size.

**Core Idea**: **Reduce "learning cuts" to "learning general target functions."** By imposing translation and scale-invariance assumptions on the concept class, the study proves that the fat-shattering/VC dimension of the cut-learning problem is not lower than the difficulty of fitting arbitrary targets with the underlying function class. Thus, the lower bound is determined by the complexity of the concept class itself (e.g., $WL\log(W/L)$ for neural networks) and holds for both scoring functions.

## Method

### Overall Architecture
The paper combines a theoretical lower-bound derivation with experimental verification. The theoretical part formalizes learning-to-cut as a supervised regression problem: learning a function $f \in F$ that maps an ILP instance $I$ to CG cut weights, minimizing $\mathbb{E}_{(I,s)\sim D}[(h(I,f(I))-s)^2]$, where $s$ is the score label of the "best cut." Utilizing fat-shattering theorems from Anthony & Bartlett, the capability to shatter instances is linked to the underlying function class through structural assumptions. The experimental part trains a GNN on Simplex tableau cuts to evaluate whether models trained on gap closed are effective for tree size reduction.

### Key Designs

**1. Formalizing cut learning as supervised regression distinct from unsupervised settings**: The objective is $\min_{f\in F}\mathbb{E}_{(I,s)\sim D}[(h(I,f(I))-s)^2]$, where labels $s$ and instances $I$ are sampled jointly. The paper justifies this supervised setting (Remark 2.1) by noting the stochasticity of B&C tree size (performance variability from tie-breaking) and the fact that fat-shattering arguments hold even for deterministic but complex label families. This differs from the **unsupervised** upper-bound setting of Cheng et al. (2024), enabling a more direct comparison of bounds.

**2. Reducing cut-learning difficulty to underlying function class difficulty**: This is the technical core. **Assumption 1 (Closed under translation and scaling)** requires $F$ to be closed under input translation and coordinate-wise output scaling—a property satisfied by neural networks with non-zero activations. **Assumption 2 (Shattering under row restriction)** requires that when $f$ is restricted to $n$ input variables, the VC dimension of each coordinate $F_i[n]$ remains a constant, denoted $\mathrm{VCdim}(F[n])$. By defining a squashing function $\sigma'$ to map outputs to CG weights in $[0,1]^m$, the final concept class is $F_{s,\sigma'}:=\{I\mapsto s(I,h(I)):h\in F_{\sigma'}\}$. **Theorem 3.2** derives the lower bound $m_L(\epsilon,\delta)=\Omega\!\left(\frac{\mathrm{VCdim}(F[n])}{\epsilon}\right)$ for both scores. **Corollary 3.3** shows this matches the complexity of learning a general objective function with $F_n$.

**3. Instantiating bounds for neural networks**: Applying the abstract bounds to ReLU networks ($\le L$ layers, $\le W$ weights). **Proposition 3.4** yields $m_L(\epsilon,\delta)\ge \frac{1}{\epsilon C}\,\overline{W}L\log(\overline{W}/L)$, where $\overline{W}$ excludes weights from ignored inputs. Comparing this against Cheng et al. (2024)'s upper bound $O\!\left(\frac{1}{\epsilon^2}(LW\log(U+m)+W\log M)\right)$, the gap is roughly $1/\epsilon$ (ignoring $\log$ factors), which is generally unavoidable. Notably, the lower bound **does not depend** on the data magnitude $M$.

**4. Extension to Simplex tableau cuts**: Practical solvers extract CG cuts from the optimal simplex tableau. The paper reformulates the concept class $\tilde G$ where $g \in G$ scores $m$ tableau cuts. **Theorem 3.6** proves that even with a restricted cut pool, the sample complexity is still driven by the underlying VC dimension: $m_L(\epsilon,\delta)=\Omega(\mathrm{VCdim}(G[n])/\epsilon)$. **Proposition 3.7** provides the corresponding neural network bound $\frac{1}{\epsilon C}\overline{W}L\log(\overline{W}/L)$, which remains nearly tight with the upper bound $O(WL\log(Um)/\epsilon^2)$, ensuring the theory applies to real-world solver settings.

## Key Experimental Results

GNNs were used to encode (ILP instance, cut) pairs as weighted bipartite graphs with 3D point features. Scorer models were trained on Simplex tableau cuts to guide cut selection, comparing "trained on tree size" vs. "trained on gap closed." Datasets included Set Cover, Facility Location, Knapsack, and Vertex Cover (1000 instances each). Models were evaluated on 250 test instances using Gurobi 12.0.1 (heuristics/presolve disabled).

### Main Results (Average B&C Tree Size, Lower is Better)

**Set Cover**

| Strategy | Trained on B&C tree | Trained on gap closed |
|------|:---:|:---:|
| Optimal (Perfect Predictor) | – | 4.95 |
| Parallelism | – | 8.29 |
| Efficacy | – | 9.90 |
| Mix | – | 9.30 |
| Random | – | 9.71 |
| **GNN** | **8.27** | **8.65** |

**Facility Location**

| Strategy | Trained on B&C tree | Trained on gap closed |
|------|:---:|:---:|
| Optimal (Perfect Predictor) | – | 86.31 |
| Parallelism | – | 144.09 |
| Efficacy | – | 123.63 |
| Mix | – | 133.72 |
| Random | – | 152.46 |
| **GNN** | **128.85** | **134.61** |

### Ablation Study (GNN Comparison)

| Problem | GNN (tree size) | GNN (gap closed) | Gap to Optimal |
|------|:---:|:---:|:---:|
| Set Cover | 8.27 | 8.65 | Room for improvement relative to 4.95 |
| Facility Location | 128.85 | 134.61 | Room for improvement relative to 86.31 |

### Key Findings
- **GNNs demonstrate learning capability**: In Facility Location and Knapsack, GNNs outperformed classic heuristics (Parallelism/Efficacy/Mix); in Set Cover and Vertex Cover, they matched SOTA heuristics.
- **Gap closed is a strong proxy**: GNNs trained on the cheaper gap closed metric performed only slightly worse in terms of tree size compared to direct training on tree size, supporting the theory that both scores share similar learning difficulty.
- **Room for growth**: Both GNN versions remain significantly behind the perfect predictor (Optimal), indicating learning-to-cut is not yet saturated.

## Highlights & Insights
- **First lower bound for learning-to-cut**: Fills a theoretical gap, establishing that cut learning is fundamentally no easier than general function approximation.
- **Elegant reduction strategy**: Uses structural assumptions to link "learning cuts" to "learning-anything," narrowing the gap between upper and lower bounds to $1/\epsilon$.
- **Theory-Practice loop**: Theoretical parity between scores $\to$ Experimental validation of gap closed as a proxy $\to$ Principled justification for long-standing solver habits.
- **Data-independent lower bound**: Unlike upper bounds containing $\log M$ (coefficient magnitude), the lower bound is purely structural ($W, L$), providing a cleaner complexity landscape.

## Limitations & Future Work
- **Unexploited scoring structure**: The authors hypothesize that since gap closed only requires cuts that actually separate the LP solution, it might eventually allow for **better upper bounds**, breaking the current symmetry.
- **$W>CL$ assumption**: The neural network lower bound inherits structural constraints from bit-extraction techniques; removing this requires new proof paths specific to ILP shattering.
- **Unknown distribution dependency**: In cases with very simple, deterministic distributions, the actual complexity could be lower than this bound.
- **Single cut limitation**: The study focuses on "one cut at a time," whereas modern solvers add cuts in rounds (batches).
- **Experimental scale**: Instances used are relatively small (e.g., 16-item knapsack), and code is not yet public.

## Related Work & Insights
- **Learning-to-cut upper bounds**: Balcan et al. (2021) and Cheng et al. (2024) provide the reference points for this work's lower-bound comparison.
- **Scoring and Selection**: Techniques such as Imitation Learning (Paulus et al. 2022), scoring functions (Huang et al. 2022), and RL (Tang et al. 2020) are the primary targets for this sample complexity analysis.
- **Algorithmic theory**: The usage of fat-shattering and VC dimensions follows the lineage of Anthony, Bartlett, and Shalev-Shwartz.
- **Insight**: The reduction technique of mapping "task-specific learning" to "base function class difficulty" can be generalized to other learning-to-optimize domains (e.g., branching or heuristic selection).

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Establishes the first lower bound for the field and a unified framework for score comparison.
- **Experimental Thoroughness**: ⭐⭐⭐ — Experiments support the theory but focus on smaller instances.
- **Writing Quality**: ⭐⭐⭐⭐ — Rigorous theoretical development with clear motivations and counter-examples.
- **Value**: ⭐⭐⭐⭐ — Provides the first principled justification for the gap closed metric and sets a trajectory for refining learning-to-cut theory.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Mitigating the Curse of Detail: Scaling Arguments for Feature Learning and Sample Complexity](mitigating_the_curse_of_detail_scaling_arguments_for_feature_learning_and_sample.md)
- [\[ICLR 2026\] T-Tamer: Provably Taming Trade-offs in ML Serving](t-tamer_provably_taming_trade-offs_in_ml_serving.md)
- [\[ICLR 2026\] Near-Optimal Sample Complexity Bounds for Constrained Average-Reward MDPs](near-optimal_sample_complexity_bounds_for_constrained_average-reward_mdps.md)
- [\[ICLR 2026\] Minimax Sample Complexity of Graph Neural Networks: Lower Bounds and Structural Effects](minimax_sample_complexity_of_graph_neural_networks_lower_bounds_and_structural_e.md)
- [\[ICLR 2026\] Sampling Complexity of TD and PPO in RKHS](sampling_complexity_of_td_and_ppo_in_rkhs.md)

</div>

<!-- RELATED:END -->
