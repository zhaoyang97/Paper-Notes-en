---
title: >-
  [Paper Note] Exchangeability of GNN Representations with Applications to Graph Retrieval
description: >-
  [ICLR 2026][Graph Learning][Paper Note] This paper discovers and proves a novel probabilistic symmetry: node embeddings of standardly trained GNNs are **exchangeable random variables** along the dimensional axis. Leveraging this property, the authors approximate high-dimensional transportation similarity as one-dimensional sorted Euclidean similarity, design
tags:
  - ICLR 2026
  - Graph Learning
date: 2026-05-08
content_hash: a1e486adb4062264
---
# Exchangeability of GNN Representations with Applications to Graph Retrieval

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=HQcCd0laFq](https://openreview.net/forum?id=HQcCd0laFq)  
**Code**: [https://rebrand.ly/graphhash](https://rebrand.ly/graphhash)  
**Area**: Graph Learning / Graph Retrieval / Probabilistic Symmetry  
**Keywords**: Exchangeability, GNN Representations, Locality Sensitive Hashing (LSH), Optimal Transport Similarity, Graph Retrieval  

## TL;DR
This paper discovers and proves a novel probabilistic symmetry: node embeddings of standardly trained GNNs are **exchangeable random variables** along the dimensional axis. Leveraging this property, the authors approximate high-dimensional transportation similarity as one-dimensional sorted Euclidean similarity, designing GraphHash—the first unified LSH retrieval framework for asymmetric graph similarities.

## Background & Motivation
- **Background**: Neural graph retrieval aims to find the top-b most relevant graphs from a large-scale corpus $\mathcal{C}=\{G_c\}$ for a query graph $G_q$. Recent work has confirmed that node embedding similarity based on **transportation** (such as hinge distance for subgraph matching or Graph Edit Distance (GED)) significantly outperforms single-vector pooling and graph kernel methods in accuracy.
- **Limitations of Prior Work**: Although transportation similarity is accurate, it is **highly unsuitable for indexing and retrieval**. Calculating exact transportation distance in high dimensions $D>1$ is $O(n^3)$, and even approximate Sinkhorn iterations require $O(n^2)$. More critically, graph similarities like subgraph matching and GED are inherently **asymmetric**, whereas existing LSH methods (e.g., L1 grid projection, L2 line projection) only support symmetric Euclidean distances, preventing sub-linear time retrieval for massive graph databases.
- **Key Challenge**: There is a long-standing trade-off: retrieval "speed" requires symmetric Euclidean metrics and hashable structures, while "accuracy" requires asymmetric high-dimensional transportation metrics.
- **Goal**: To find a bridge that reduces expensive high-dimensional asymmetric transportation similarity to hashable one-dimensional Euclidean similarity without loss (with high probability), achieving both high accuracy and sub-linear retrieval speed.
- **Core Idea**: **"Exchangeability"**. The authors prove that under standard iid initialization, permutation-invariant loss, and common optimizers (SGD/Adam), the columns $X[:,1],\dots,X[:,D]$ of the trained GNN node embedding matrix $X\in\mathbb{R}^{n\times D}$ are exchangeable random variables, meaning $p(X)=p(X\pi)$. This implies each embedding dimension is identically distributed, allowing **any single dimension to provide a concentration estimate of the overall transportation similarity**, effectively reducing the $D$-dimensional problem to 1D sorted matching.

## Method

### Overall Architecture
The paper follows a two-stage structure: first proving a property and then applying it. The first stage theoretically establishes the exchangeability of GNN embeddings (§3), and the second stage translates this into GraphHash (§4), a practical LSH algorithm for graph retrieval. The proof follows a four-step chain: "Parameter transformation $\Gamma_\pi$ → Gradient equivariance → Parameter density invariance → Embedding exchangeability." The application follows a three-step implementation: "Dimensional reduction via identical distribution → Fourier features for sorted similarity as inner product → Random hyperplane hashing."

```mermaid
flowchart TD
    A[iid Initialized Parameters θ₀<br/>p(θ₀)=p(Γπ θ₀)] --> B[Lemma 2: Construct Permutation-Induced Transformation Γπ<br/>Xπ = GNN_Γπ(θ)(G)]
    B --> C[Lemma 3: Permutation-Invariant Loss ⇒ Gradient Equivariance]
    C --> D[Lemma 4: Invariant Parameter Density throughout Training Trajectory<br/>p(θt)=p(Γπ θt)]
    D --> E[Theorem 5: Exchangeable Embeddings<br/>p(X)=p(Xπ)]
    E --> F[Prop 7: Single-dimension Similarity Concentration<br/>sim/D ≈ simd]
    F --> G[Prop 9: Fourier Features<br/>simd ∝ cos(T̂q,d, T̂c,d)]
    G --> H[Random Hyperplane Hashing<br/>GraphHash Indexing/Query]
```

### Key Designs

**1. Reducing "embedding exchangeability" to "parameter symmetry" via permutation-induced transformation $\Gamma_\pi$.** Analyzing non-linear training dynamics directly to prove dimensional exchangeability is difficult. The authors' ingenuity lies in "moving" the dimensional permutation $\pi$ to the parameter space: for a two-layer MLP $x=\sigma(\sigma(\text{feat}\,\Psi)\Theta)$, defining $\Gamma_\pi(\theta):=\theta\,\mathrm{Diag}(I,\pi)$ transforms the output from $x$ to $x\pi$. For GNNs, where message passing entangles permutations across layers, the construction is more complex, but Lemma 2 proves that for gated GNNs, GINs, GATs, GCNs, and even Graph Transformers, such a bijection $\Gamma_\pi$ exists. Furthermore, its Jacobian determinant $|\det(\partial\Gamma_\pi(\theta)/\partial\theta)|=1$ (measure-preserving), which equates "exchangeable embedding dimensions" to "parameter density invariance under $\Gamma_\pi$."

**2. Propagating symmetry from initialization to any training step via "gradient equivariance + initial iid."** At the start of training $t=0$, parameters are iid, naturally satisfying $p(\theta_0)=p(\Gamma_\pi(\theta_0))$. The key question is whether training destroys this. Since the loss is invariant to embedding dimension permutations (which holds automatically in graph retrieval as similarity depends on the transportation plan), the loss landscape is symmetric under $\Gamma_\pi$. This leads to gradient equivariance (Lemma 3): permuting initial parameters is equivalent to permuting the entire training trajectory, $\theta_0\mapsto\Gamma_\pi(\theta_0)\Rightarrow\theta_t\mapsto\Gamma_\pi(\theta_t)$. Combined with the measure-preserving property, Lemma 4 yields $p(\theta_t)=p(\Gamma_\pi(\theta_t))$ for all $t\ge0$, and Theorem 5 finally derives $p(X)=p(X\pi)$. A direct corollary: the expected embedding matrix $\mathbb{E}[X]$ averaged across random seeds collapses to a **rank-one matrix**, echoing phenomena like GNN over-smoothing and Transformer rank collapse.

**3. Reducing high-dimensional transportation similarity to 1D sorted Euclidean similarity via identical distribution.** The authors unify various graph similarities into a convex, decomposable, and potentially asymmetric cost function $\rho(x)=\sum_d\rho(x[d])$: $\rho(\cdot)=[\cdot]_+$ for subgraph matching hinge distance, and $\rho(\cdot)=e_\ominus[\cdot]_++e_\oplus[-\cdot]_+$ for GED with asymmetric edit costs. The overall transportation similarity is:
$$\mathrm{sim}(G_c,G_q)=\max_{P\in\mathcal{P}_n}\sum_{u,u'}\sum_{d\in[D]} s\big(x^{(q)}(u)[d]-x^{(c)}(u')[d]\big)\cdot P[u,u'].$$
By Proposition 6 (joint exchangeability), dimensions are identically distributed, so the similarity $\mathrm{sim}_d$ of any single dimension $d$ is an unbiased scaled estimate of the total. The 1D transportation problem degenerates into scalar matching—combined with the concavity of $s$, this simplifies to the similarity of **sorted vectors**: $\mathrm{sim}_d(G_c,G_q)=s\big(\mathrm{SORT}(X^{(q)}[:,d])-\mathrm{SORT}(X^{(c)}[:,d])\big)$. Proposition 7 provides concentration guarantees: for $D>\frac{1}{\epsilon^2\delta}$, then $\Pr(|\mathrm{sim}/D-\mathrm{sim}_d|\le\epsilon)\ge1-\beta_0\delta$.

**4. GraphHash: Transforming asymmetric similarity into hashable inner products via Fourier features + Random Hyperplanes.** The 1D sorted similarity $\mathrm{sim}_d=s(\cdot)$ resembles a shift-invariant kernel $\kappa(x-x')$ but is not a true kernel due to asymmetry, preventing the direct use of Rahimi-Recht Random Fourier Features. Extending Roy et al. (2023), Proposition 9 proves the existence of an $\mathbb{R}^4$ Fourier feature vector such that $\mathrm{sim}_d=\sum_u\int F_{q,d}(\iota\omega_u)^\top F_{c,d}(\iota\omega_u)\,d\omega_u$. Sampling $M$ frequencies with $p(\omega)\propto|S(\iota\omega)|$ (amplitude of the Fourier transform of the scoring function) yields a finite-dimensional vector $\hat T_{\bullet,d}\in\mathbb{R}^{4nM}$ satisfying $\mathrm{sim}_d\propto\cos(\hat T_{q,d},\hat T_{c,d})$. Finally, applying random hyperplanes $W[r,t]\sim\mathcal{N}(0,1)$ produces hash codes $h^{(d)}(G_c)=\mathrm{sign}(W\hat T_{c,d})$, forming an **effective asymmetric LSH** for the original transportation similarity—the first unified LSH framework supporting subgraph matching, general-cost GED, and more.

## Key Experimental Results

### Main Results (Retrieval Performance, MAP vs. Number of Graphs Retrieved)
- Datasets: ptc-fr / ptc-fm / cox2 / ptc-mr from TUDatasets, each with 500 query graphs and 100,000 corpus graphs.
- Two types of asymmetric supervision: Subgraph Matching (SM, using VF2) and GED (asymmetric costs $e_\oplus=1$ insertion, $e_\ominus=2$ deletion).
- Comparison with 5 ANN baselines: FourierHashNet, Random Hyperplanes (RH), IVF (FAISS/ColBERT style), DiskANN (HNSW), Random.

| Method | SM Task | GED Task | Remarks |
|------|---------|----------|------|
| **Ours (GraphHash)** | **Globally Optimal** | **Globally Optimal** | Only method covering the full selectivity spectrum |
| FourierHashNet | Second best, but SM MAP on ptc-fr/ptc-mr below 50% exhaustive upper bound | Second best | Single-vector pooling, limited selectivity |
| RH | Worse than random (symmetric cosine fails for containment) | Occasionally matches GraphHash but high variance | Hard to tune |
| IVF / DiskANN | Poor | Poor | Multi-vector but relies on L2/cosine symmetric metrics |
| Random | Significantly lowest | Significantly lowest | Baseline lower bound |

### Key Findings
1. Exchangeability holds not only at initialization but is **maintained after backpropagation with non-convex losses**—this is the most counter-intuitive and crucial empirical conclusion.
2. Optimal MAP-selectivity trade-off is achieved at hash bits $\dim_h\approx10$, with diminishing returns for higher dimensions.
3. Multi-vector indices (IVF/DiskANN) fail under asymmetric supervision, confirming the core pain point that "symmetric metrics $\neq$ graph similarity."
4. RH performs worse than random on SM tasks, indicating that cosine hashing on pooled vectors is completely unsuitable for asymmetric queries like subgraph containment.
5. GraphHash is the only method capable of covering the entire selectivity spectrum (from high recall to high precision), whereas FourierHashNet's MAP plateaus early on ptc-fr/ptc-mr.

## Highlights & Insights
- **Probabilistic Symmetry vs. Algebraic Symmetry**: Previous works on weight space symmetry (permutation invariance in MLPs, model merging, etc.) were algebraic properties independent of training. This paper characterizes **naturally emerging** probabilistic symmetry during training, identifying "random initialization + equivariant gradients" as the source.
- **Universal Applicability**: Exchangeability holds across a wide range of architectures (gated GNN, GIN, GAT, GCN, Graph Transformer), optimizers (SGD/Adam), and standard initializations (Kaiming/Xavier).
- **Theory-to-System Bridge**: Exchangeability explains rank collapse in GNNs/Transformers (theoretical value) and directly enables sub-linear retrieval (practical value).
- **First Asymmetric Graph LSH**: It unifies heterogeneous metrics like subgraph matching and general-cost GED into a single LSH framework, filling a gap where asymmetric transportation similarities were previously unhashable.
- **Dimensionality Reduction for Complexity**: Reducing from $O(n^3)$/$O(n^2)$ high-dimensional transportation to 1D sorting ($O(n\log n)$) + hash lookups substantially improves scalability.
- **Elegance of Measure-Preserving Construction**: The property that the Jacobian determinant of $\Gamma_\pi$ is constant 1 means the permutation does not change the probability measure in parameter space, which is key to transferring symmetry from initialization to post-training.

## Limitations & Future Work
- **Dependency on Permutation-Invariant Loss**: The core proof relies on the loss being invariant to embedding dimension permutations. While the appendix suggests this can be relaxed to "joint transformation invariance," universal applicability across arbitrary losses requires further validation.
- **Double-Edged Sword of Rank-One Collapse**: While rank-one expected embeddings prove exchangeability, they also suggest that models from a single training run might be constrained by probabilistic symmetry, a relationship to representation capacity/over-smoothing not fully explored here.
- **Data Scale**: Exchangeability validation used a cox2 subset with 5,000 retrains, and retrieval experiments focused on molecular graphs (TUDatasets). Performance on large-scale heterogeneous graphs or social networks remains to be tested.
- **Trade-off between Approximation Error and $D$**: Proposition 7 requires $D>\frac{1}{\epsilon^2\delta}$ for concentration. The impact of low-dimensional embeddings and the number of frequency samples $M$ on hashing accuracy could be further characterized.

## Related Work & Insights
- **Weight Space Symmetry**: Building on MLP permutation invariance (Hecht-Nielsen, 1990), symmetry across activations (Godfrey et al., 2022), and model merging (Ainsworth et al., 2022), this work pushes the line from "algebraic/static" to "probabilistic/emergent."
- **Rank Collapse**: Corroborates rank collapse in Transformers (Dong et al., 2023), State Space Models (Joseph et al., 2025), and GNN over-smoothing (Roth et al., 2024) through a unified probabilistic lens.
- **Graph Retrieval and Transportation Similarity**: Extends the hinge distance for SM (Roy et al., 2022) and GED (Jain et al., 2024) by generalizing FourierHashNet (Roy et al., 2023) to arbitrary asymmetric similarities.
- **Mechanism**: The use of limited-dimensional Fourier features to approximate shift-invariant kernels (Rahimi & Recht, 2007) is generalized from symmetric kernels to asymmetric scoring functions $s(\cdot)$, forming the technical foundation for GraphHash.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — First to prove emergent exchangeability in GNN training and translate it into a retrieval algorithm.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Solid validation of exchangeability with 5000 retrains and retrieval comparisons; however, datasets are limited to molecular graphs.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear four-step proof chain and natural transition from theory to application, though Fourier feature derivations are dense.
- **Value**: ⭐⭐⭐⭐⭐ — Provides both a unified explanation for rank collapse and the first scalable LSH framework for asymmetric graph similarities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] GNN-as-Judge: Unleashing the Power of LLMs for Graph Learning with GNN Feedback](gnn-as-judge_unleashing_the_power_of_llms_for_graph_learning_with_gnn_feedback.md)
- [\[ICLR 2026\] On The Expressive Power of GNN Derivatives](on_the_expressive_power_of_gnn_derivatives.md)
- [\[ICLR 2026\] Towards Improved Sentence Representations using Token Graphs](towards_improved_sentence_representations_using_token_graphs.md)
- [\[ICLR 2026\] Directed Semi-Simplicial Learning with Applications to Brain Activity Decoding](directed_semi-simplicial_learning_with_applications_to_brain_activity_decoding.md)
- [\[ICLR 2026\] CORDS - Continuous Representations of Discrete Structures](cords_-_continuous_representations_of_discrete_structures.md)

</div>

<!-- RELATED:END -->
