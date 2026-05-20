---
title: >-
  [Paper Note] Exchangeability of GNN Representations with Applications to Graph Retrieval
description: >-
  [ICLR 2026 Oral][GNN] This paper identifies that trained GNN node embeddings are exchangeable random variables along the feature dimension (i.e.…
tags:
  - "ICLR 2026 Oral"
  - "GNN"
  - "exchangeability"
  - "graph retrieval"
  - "LSH"
  - "GraphHash"
  - "transportation distance"
  - "Wasserstein distance"
date: 2026-05-08
content_hash: 7f42e62e94f41a43
---

# Exchangeability of GNN Representations with Applications to Graph Retrieval

**Conference**: ICLR 2026 Oral
**OpenReview**: [HQcCd0laFq](https://openreview.net/forum?id=HQcCd0laFq)
**Code**: Available  
**Area**: Other
**Keywords**: GNN, exchangeability, graph retrieval, LSH, GraphHash, transportation distance, Wasserstein distance

## TL;DR
This paper identifies that trained GNN node embeddings are exchangeable random variables along the feature dimension (i.e., $p(\mathbf{X}) = p(\mathbf{X}\pi)$ holds for any dimensional permutation $\pi$), and exploits this property to approximate transportation-distance-based (EMD/Wasserstein) graph similarity as Euclidean distance via dimension-wise sorting. A unified locality-sensitive hashing (LSH) framework, GraphHash, is constructed upon this foundation, consistently outperforming baselines including FourierHashNet, DiskANN, IVF, CORGII, and SWWL in AUC on subgraph matching and graph edit distance (GED) retrieval tasks, scaling to corpus sizes of one million graphs.

## Background & Motivation
**Background**: Graph retrieval requires measuring inter-graph similarity (e.g., subgraph matching, graph edit distance), which involves optimal transport problems with prohibitive computational cost ($O(n^3)$ Hungarian algorithm or $O(n^2)$ Sinkhorn). Exhaustive pairwise scoring is infeasible for large-scale corpora ($|C| \gg 10^5$).

**Limitations of Prior Work**: Standard approximate nearest neighbor (ANN) methods such as DiskANN and IVF assume Euclidean or cosine similarity. However, GNN-produced graph representations are sets of node embeddings $\mathbf{X} \in \mathbb{R}^{n \times D}$, and inter-graph similarity is a transportation distance defined over embedding sets — one that cannot be directly accelerated by vector-based ANN methods.

**Key Challenge**: Transportation distance computation is expensive and does not satisfy the metric conditions required by standard LSH, whereas Euclidean distance is efficient but fails to capture the alignment relationships between sets.

**Goal**: To identify a theoretically grounded approximation that reduces transportation-distance-based graph similarity to standard Euclidean distance, thereby enabling efficient graph retrieval via LSH.

**Key Insight**: The paper identifies a previously unnoticed probabilistic symmetry of GNN embeddings — exchangeability along the feature dimension — and leverages it to simplify transportation distance computation.

**Core Idea**: The exchangeability of GNN embeddings along the feature dimension allows sorted embeddings to approximate transportation distances via Euclidean distance, enabling efficient LSH-based graph retrieval.

## Method

### Overall Architecture
Pipeline: GNN encoder → node embedding matrix $\mathbf{X} \in \mathbb{R}^{n \times D}$ → per-dimension independent sorting to obtain $\mathbf{v}_d = \text{sort}(\mathbf{X}_{:,d})$ → random Fourier feature mapping $\mathbf{t}_d = F(\mathbf{v}_d)$ → random hyperplane LSH hashing $\mathbf{h}_d = \text{sign}(\mathbf{W}\mathbf{t}_d)$ → independent per-dimension bucket lookup with aggregated voting.

### Key Designs
1. **Discovery and Proof of Exchangeability**:
    - Function: Proves that under standard training conditions (i.i.d. initialization + permutation-invariant loss + equivariant optimizer), the feature dimensions of GNN node embedding matrices $\mathbf{X}$ are exchangeable random variables.
    - Mechanism: If the initial parameters $\theta_0$ have i.i.d. dimensions, then by Lemma 2 (permutation-induced transformation $\Gamma_\pi$), SGD updates preserve equivariance $\theta_t(\pi) = \Gamma_\pi(\theta_t)$, yielding $p(\mathbf{X}) = p(\mathbf{X}\pi)$. The key lemma chain is: exchangeable initialization → equivariant gradients → equivariant updates → exchangeable embeddings (Theorem 5).
    - Design Motivation: This is not a new constraint imposed on GNNs, but rather a symmetry that existing GNNs naturally possess under standard training. The property is shown to hold under modern components including BatchNorm, LayerNorm, Dropout, Adam, and Adagrad (with detailed proofs added in the rebuttal).

2. **Dimension-Wise Sorting Approximation of Transportation Distance**:
    - Function: Using exchangeability, the $D$-dimensional transportation distance $\text{sim}(\mathbf{G}_c, \mathbf{G}_q)$ is decomposed into a sum of $D$ independent one-dimensional sorting problems: $\text{sim}(\mathbf{G}_c, \mathbf{G}_q) \approx \frac{1}{D}\sum_{d=1}^D \text{sim}_d(\mathbf{G}_c, \mathbf{G}_q)$.
    - Mechanism: By exchangeability, the marginal distribution of each dimension $d$ is identical. Computing Euclidean distance after independent per-dimension sorting is equivalent to solving one-dimensional optimal transport (one-dimensional Wasserstein distance = sorted $L_1$ distance). The approximation error is given in Proposition 7, where $\Pr(|\frac{\text{sim}}{D} - \text{sim}_d| \geq \epsilon)$ converges at rate $O(1/D)$. Even with inter-dimensional dependence, the $O(1/D)$ convergence is maintained under $L_2$-bounded embeddings.
    - Design Motivation: The $D$-dimensional combinatorial optimization problem is decomposed into $D$ parallelizable one-dimensional sorting problems, reducing computational complexity from $O(n^3 D)$ to $O(nD \log n)$.

3. **GraphHash: LSH Framework**:
    - Function: Applies Fourier feature mapping and random hyperplane LSH to sorted embeddings to enable sublinear-time queries.
    - Mechanism: For the sorted embedding $\mathbf{v}_d$ of each dimension $d$, random Fourier features are computed as $\mathbf{t}_d = [\cos(\omega_j^T \mathbf{v}_d + b_j)]_{j=1}^M$ ($M=10$), with hash code $\mathbf{h}_d = \text{sign}(\mathbf{W}\mathbf{t}_d)$. Query time is $O(|C|^\gamma)$, where $\gamma = \frac{\log(1/p)}{\log(1/p')} < 1$.
    - Design Motivation: Fourier features map sorted embeddings into a Hilbert space where Euclidean distance is meaningful, while random hyperplane LSH theoretically guarantees higher collision probability in high-similarity regions. Space complexity is only $O(D|C|)$, requiring just 3.5 MB for 100K graphs.

### Theoretical Guarantees
- Theorem 5: GNN embedding exchangeability holds under standard training conditions.
- Proposition 7: Approximation error converges at rate $O(1/D)$, independent of inter-dimensional dependence.
- Theorem 18: LSH collision probability satisfies $(p, p', r_1, r_2)$-sensitivity.

## Key Experimental Results

### Main Results: AUC of MAP-Efficiency Tradeoff ($|C|=1$M)

| Dataset (Task) | GraphHash | FourierHashNet | DiskANN | IVF | CORGII |
|----------------|-----------|----------------|---------|-----|--------|
| COX2 (SM) | **0.361** | 0.332 | 0.213 | - | 0.274 |
| COX2 (GED) | **0.230** | 0.222 | 0.190 | - | 0.154 |
| PTC-FM (SM) | **0.347** | 0.322 | 0.161 | - | 0.216 |
| PTC-FM (GED) | **0.284** | 0.270 | 0.231 | - | 0.186 |
| PTC-FR (SM) | **0.333** | 0.317 | 0.157 | - | 0.217 |
| PTC-MR (SM) | **0.337** | 0.288 | 0.122 | - | 0.205 |

### Ablation Study

| Configuration | Key Metric | Description |
|---------------|-----------|-------------|
| Approximation error (MAP*-MAP)/MAP* | 6.73%–11.89% | Bounded accuracy loss in approximating transportation distance (<12%) |
| Cross-seed stability (10 runs) | std < 0.009 AUC | Performance is highly stable across random seeds |
| $D=2\to30$ | AUC improves monotonically | Consistent with Prop. 7: larger $D$ yields more accurate approximation |
| dim_h=10 | Optimal AUC | Hash length of 10 bits is the sweet spot |
| $M$ (Fourier dim)=10 | Optimal AUC | Performance degrades sharply for $M<10$ |
| MMD test $p_X$ vs $p_{X\pi}$ | MMD²≈−3.89e-5 ± 2.69e-5 | Statistically confirms exchangeability (MMD near 0) |
| vs SWWL (SM) | GraphHash 0.354 vs SWWL 0.023 | SWWL's symmetric similarity fails to capture asymmetric subgraph matching |

### Key Findings
- Exchangeability holds under BatchNorm, LayerNorm, Dropout, and Adam (verified theoretically and empirically).
- Approximation error is <12%; scales to $|C|=10^6$ with only 3.5 MB per 100K graphs.
- GraphHash achieves the best AUC on all 4 datasets for the subgraph matching task.
- GraphHash achieves the best AUC on 3 out of 4 datasets for the GED task.

## Highlights & Insights
- **A New Theoretical Perspective on GNN Representations**: Exchangeability is a previously unnoticed probabilistic symmetry of GNN embeddings, with theoretical significance extending beyond the specific application. Prior work on GNN symmetry focused on node permutation invariance/equivariance; this paper reveals a complementary symmetry in the "dimension direction."
- **Reducing Graph Retrieval to Standard Vector Retrieval**: Through exchangeability and dimension-wise sorting, the NP-hard transportation distance is approximated as a Euclidean distance amenable to LSH acceleration.
- **Clear Positioning in Retriever–Reranker Architecture**: GraphHash functions as a retriever (fast coarse filtering), orthogonally complementing rerankers such as IsoNet++ and SWWL (precise scoring).
- **Proof Extended to Non-Permutation-Invariant Losses**: By decomposing the network into an embedding layer and a classification head, the intermediate-layer embeddings are shown to remain exchangeable.

## Limitations & Future Work
- Validation is limited to small molecular graph datasets (PTC, COX2, with fewer than 50 nodes); large-scale social networks and knowledge graphs have not been tested.
- Although bounded, the approximation error of ~12% may necessitate reranker compensation for precision-critical applications.
- The theoretical assumption of i.i.d. initialization leaves open the question of whether GNNs that are fine-tuned from pretrained weights still satisfy exchangeability.
- The framework currently supports only subgraph matching and GED similarity; extensions to other tasks such as graph alignment have not been explored.

## Related Work & Insights
- **vs FourierHashNet**: Also an LSH-based method but does not exploit exchangeability; achieves lower AUC than GraphHash across all settings.
- **vs DiskANN/IVF**: Standard ANN methods flatten graph embeddings into a single vector for retrieval, losing set-structural information.
- **vs IsoNet++**: An early interaction model in which query and corpus embeddings cannot be computed independently, precluding indexing. Single-query latency is 32.52s vs. 3.54s for GraphHash.
- **vs CORGII**: Also a graph retrieval retriever; GraphHash consistently achieves higher AUC (gap of 0.05–0.12).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The discovery of exchangeability, the dimension-wise sorting approximation of transportation distance, and the GraphHash construction are all original contributions with strong theoretical merit.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 4 datasets, million-graph scaling, MMD statistical tests, and extensive hyperparameter analysis, though dataset scale remains limited.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear and well-organized; reviewers noted the work is "well-structured and clearly presented."
- Value: ⭐⭐⭐⭐ Offers a novel theoretical perspective and provides practical theoretical tools for graph retrieval, though the current experimental datasets are small, limiting demonstrated practical impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] LeanRAG: Knowledge-Graph-Based Generation with Semantic Aggregation and Hierarchical Retrieval](../../AAAI2026/others/leanrag_knowledge-graph-based_generation_with_semantic_aggregation_and_hierarchi.md)
- [\[ICLR 2026\] Addressing Divergent Representations from Causal Interventions on Neural Networks](addressing_divergent_representations_causal.md)
- [\[AAAI 2026\] Learning Fair Representations with Kolmogorov-Arnold Networks](../../AAAI2026/others/learning_fair_representations_with_kolmogorov-arnold_networks.md)
- [\[ACL 2026\] Reliable Evaluation Protocol for Low-Precision Retrieval](../../ACL2026/others/reliable_evaluation_protocol_for_low-precision_retrieval.md)
- [\[ICLR 2026\] Learning Structure-Semantic Evolution Trajectories for Graph Domain Adaptation](learning_structure-semantic_evolution_trajectories_for_graph_domain_adaptation.md)

</div>

<!-- RELATED:END -->
