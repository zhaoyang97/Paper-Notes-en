---
title: >-
  [Paper Note] LotusFilter: Fast Diverse Nearest Neighbor Search via a Learned Cutoff Table
description: >-
  [CVPR 2025][Information Retrieval & RAG][Diverse Nearest Neighbor Search] This paper proposes LotusFilter, which constructs a cutoff table by precomputes neighbor relationships for each vector offline and performs diversity filtering using greedy set deletion during the online stage. This reduces the complexity of traditional diverse search from $O(DS^2)$ to $O(T+S+KL)$. The filtering process requires only 0.02 ms/query, utilizing only 1/40 of the memory compared to tradition…
tags:
  - "CVPR 2025"
  - "Information Retrieval & RAG"
  - "Diverse Nearest Neighbor Search"
  - "Cutoff Table"
  - "OrderedSet Data Structure"
  - "Hyperparameter Learning"
  - "RAG Retrieval Deduplication"
date: 2026-05-08
content_hash: 343c1d0aef8e9962
---

# LotusFilter: Fast Diverse Nearest Neighbor Search via a Learned Cutoff Table

**Conference**: CVPR 2025  
**arXiv**: [2506.04790](https://arxiv.org/abs/2506.04790)  
**Code**: [https://github.com/matsui528/lotf](https://github.com/matsui528/lotf)  
**Area**: Vector Search / Diverse Search  
**Keywords**: Diverse Nearest Neighbor Search, Cutoff Table, OrderedSet Data Structure, Hyperparameter Learning, RAG Retrieval Deduplication

## TL;DR
This paper proposes LotusFilter, which constructs a cutoff table by precomputes neighbor relationships for each vector offline and performs diversity filtering using greedy set deletion during the online stage. This reduces the complexity of traditional diverse search from $O(DS^2)$ to $O(T+S+KL)$. The filtering process requires only 0.02 ms/query, utilizing only 1/40 of the memory compared to traditional methods.

## Background & Motivation

**Background**: Approximate Nearest Neighbor Search (ANNS) is a core component of applications like RAG, recommendation systems, and image retrieval. Modern ANNS methods such as HNSW and IVF can achieve sub-millisecond search times on million-scale datasets, but their top-K results are often highly redundant. For example, searching with a cat image might return multiple near-identical photos of the same cat; retrieving RAG documents for medical queries might return almost identical paragraphs.

**Limitations of Prior Work**: Diverse Nearest Neighbor Search (DNNS) is a classic research direction to address this issue, but existing methods face three core bottlenecks: (1) Selecting a subset of $K$ items from $S$ candidates is an NP-hard problem, where naive enumeration incurs a cost of $\binom{S}{K}$; (2) Computing pairwise distances among candidates requires a complexity of at least $O(DS^2)$, which is extremely slow in high-dimensional scenarios; (3) The filtering step requires accessing the original vectors, which leads to slow disk I/O if the vectors are not fully memory-resident.

**Key Challenge**: Modern ANNS methods (e.g., HNSW) typically store only compressed vector representations, making it impossible for traditional DNNS methods to be directly integrated as post-processing modules. This results in a "fast search, slow diversification" bottleneck. For instance, the GMM method requires $O(DKS)$ complexity and accesses high-dimensional raw vectors. For OpenAI embeddings with $D=1536$, filtering a single query takes 13.4 ms, which is over 15 times the search time itself. Users require a pure post-processing solution that can be integrated as a black box into any ANNS engine.

**Goal**: To design an extremely lightweight post-processing filter that achieves search result diversification rapidly without accessing raw vectors, while being plug-and-play with any ANNS method.

**Key Insight**: Shift run-time distance computation entirely to the offline stage by pre-recording a list of "too-close neighbors" for each database vector, thereby requiring only simple set deletion operations during run-time.

**Core Idea**: Precompute a cutoff table that records the set of near-neighbor IDs for each vector. At query time, greedily pop the nearest candidate and delete its neighbors, replacing $O(D)$ distance computations with $O(1)$ set operations.

## Method

### Overall Architecture
LotusFilter consists of two stages: offline pre-processing and online filtering.

**Offline Stage**: (1) Build a search index $\mathcal{I}$ using any ANNS method; (2) Perform a range search for each database vector $\mathbf{x}_n$ to collect the set of neighbor IDs $\mathcal{L}_n$ whose squared distance is smaller than a threshold $\varepsilon$, forming the cutoff table; (3) Learn the optimal $\varepsilon^*$ using bracketing optimization.

**Online Stage**: (1) Retrieve $S$ candidates $\mathcal{S}$ using ANNS; (2) Load $\mathcal{S}$ into an OrderedSet data structure; (3) Greedy loop: pop the candidate $k$ closest to the query, add it to the result set $\mathcal{K}$, and remove all IDs in $\mathcal{L}_k$ from $\mathcal{S}$; (4) Repeat until $|\mathcal{K}|=K$. The entire filtering process involves no floating-point operations, operating solely on integer ID sets.

### Key Designs
1. **Cutoff Table Precomputation**:
    - **Function**: Store the neighbor relations of each vector offline to avoid any run-time distance computation.
    - **Mechanism**: For each $\mathbf{x}_n$, a range search is performed to find $\mathcal{L}_n = \{i \mid \|\mathbf{x}_n - \mathbf{x}_i\|_2^2 < \varepsilon, n \neq i\}$. The cutoff table is an "array of integer arrays", where each entry stores near-neighbor IDs with a distance smaller than $\varepsilon$. The average length is $L = \frac{1}{N}\sum_{n=1}^N |\mathcal{L}_n|$, consuming $64LN$ bits of memory (using 64-bit integers).
    - **Design Motivation**: Traditional DNNS methods require real-time computation of $O(S^2)$ pairwise distances during the filtering stage, incurring a cost of $O(DS^2)$. The cutoff table finishes all distance comparisons beforehand, requiring only $O(1)$ hash table lookups at run-time to determine if two vectors are "too close." For the OpenAI dataset with $N=9\times10^5$, the cutoff table occupies only 136 MiB and takes approximately 54 seconds to construct.

2. **OrderedSet Data Structure**:
    - **Function**: Efficiently support Pop (extract minimum element) and Remove (delete specified element) operations within the filtering loop.
    - **Mechanism**: Simultaneously maintain the raw array $\mathbf{v}$, its corresponding hash set $\mathcal{V}$, and a head pointer $c$. The Remove operation only deletes elements in $\mathcal{V}$ (lazy deletion) with a cost of $O(1)$. The Pop operation scans the array starting from position $c$ until an element still present in $\mathcal{V}$ is found, with a cost of $O(\Delta)$, where $\Delta \leq L$ (since at most $L$ elements are deleted between two Pop operations). `boost::unordered_flat_set` is utilized to achieve $O(1)$ hash operations.
    - **Design Motivation**: Implementing Remove on a naive array requires a linear search of $O(V)$; a naive hash set cannot support Pop (unordered); and a priority queue incurs a deletion cost of $O(\log V)$. OrderedSet exchanges extra memory to obtain both $O(1)$ Remove and $O(L)$ Pop, driving the complexity of the entire filtering loop down to $O(KL)$.

3. **Bracketing-Based Threshold $\varepsilon$ Learning**:
    - **Function**: Automatically select the optimal cutoff threshold $\varepsilon^*$ to balance search relevance and diversity.
    - **Mechanism**: Treat the objective function $f(\mathcal{K}) = \frac{1-\lambda}{K}\sum_{k \in \mathcal{K}}\|\mathbf{q}-\mathbf{x}_k\|_2^2 - \lambda\min_{i,j \in \mathcal{K},i\neq j}\|\mathbf{x}_i-\mathbf{x}_j\|_2^2$ as a univariate function of $\varepsilon$, and solve $\varepsilon^* = \arg\min_\varepsilon \mathbb{E}_{\mathbf{q} \in \mathcal{Q}_{\text{train}}}[f^*(\varepsilon, \mathbf{q})]$ using a training query set $\mathcal{Q}_{\text{train}}$. A bracketing method is employed to recursively shrink the search interval.
    - **Design Motivation**: If $\varepsilon$ is too small, diversity is insufficient; if it is too large, the cutoff table expands, and over-pruning leads to insufficient candidates. This parameter is sensitive to the data distribution, making manual tuning impractical. By extracting the first 1,000 vectors from the database as training queries, the optimal value can be learned automatically at a very low cost.

### Loss & Training
The objective function consists of two terms: the relevance term $\frac{1-\lambda}{K}\sum_{k \in \mathcal{K}}\|\mathbf{q}-\mathbf{x}_k\|_2^2$ (the smaller, the better) and the diversity term $-\lambda\min_{i,j \in \mathcal{K},i\neq j}\|\mathbf{x}_i-\mathbf{x}_j\|_2^2$ (smaller values denote more distant nearest pairs, yielding better diversity). The parameter $\lambda \in [0,1]$ controls the trade-off: $\lambda=0$ degenerates into standard NNS, while $\lambda=1$ degenerates into the MAX-MIN diversification problem. LotusFilter provides a theoretical guarantee: the squared distance between any two vectors in the post-filtered result set satisfies $\|\mathbf{x}_i - \mathbf{x}_j\|_2^2 \geq \varepsilon$, meaning the diversity term is bounded. Additionally, a safeguard mode is introduced: if over-pruning exhausts the candidate set, the filtering is terminated immediately, and the remaining candidates are directly appended to the result set to ensure $|\mathcal{K}|=K$.

## Key Experimental Results

### Main Results
**OpenAI Dataset** ($N=900\text{K}$, $D=1536$, $\lambda=0.3$, $K=100$, $S=500$):

| Method | Overall Score $f$ ↓ | Search Time (ms) | Filtering Time (ms) | Total Time (ms) | Memory Overhead (bit) |
|------|-------------|------------|------------|----------|-------------|
| Pure ANNS (No Filtering) | 0.200 | 0.855 | - | 0.855 | - |
| K-means Clustering | 0.223 | 0.941 | 6.94 | 7.88 | $4.42\times10^{10}$ |
| GMM | **0.177** | 0.977 | 13.4 | 14.4 | $4.42\times10^{10}$ |
| **LotusFilter** | **0.171** | 1.00 | **0.02** | **1.03** | $1.14\times10^9$ |

**Preprocessing Time and Cutoff Table Scale**:

| $N$ | $\lambda$ | Learned $\varepsilon^*$ | Average Length $L$ | Training Time (s) | Construction Time (s) |
|-----|---------|-----------------|------------|-----------|-----------|
| $9\times10^3$ | 0.3 | 0.39 | 8.7 | 96 | 0.16 |
| $9\times10^4$ | 0.3 | 0.33 | 10.1 | 176 | 3.8 |
| $9\times10^5$ | 0.3 | 0.27 | 18.4 | 1020 | 54 |
| $9\times10^5$ | 0.5 | 0.29 | 29.3 | 1087 | 54 |

### Ablation Study
**Effect of Initial Candidate Size $S$** ($K=100$, $\lambda=0.3$, OpenAI Dataset):

Increasing $S$ means ANNS yields more candidates for filtering, which theoretically leads to better solutions but scales the running time linearly. Experiments show that $f$ decreases significantly when $S$ increases from 200 to 500, after which returns diminish:

| $S$ | Overall Score $f$ ↓ | Total Time (ms) | Trend |
|-----|-------------|----------|------|
| 200 | ~0.185 | ~0.9 | Few candidates, limited diversity |
| 500 | 0.171 | 1.03 | Optimal balance point |
| 1000 | ~0.168 | ~1.2 | Continued improvement with diminishing returns |
| 2000 | ~0.166 | ~1.5 | Significant increase in running time |

### Key Findings
- LotusFilter achieves the optimal overall score (0.171) with a filtering time of only 0.02 ms/query, rendering the total time only 1/14 of GMM and 1/8 of clustering.
- The memory overhead is only $1.14\times10^9$ bits (136 MiB), which is **1/40** of traditional methods (which require storing raw vectors of $4.42\times10^{10}$ bits).
- The average length $L$ of the cutoff table is at most 30 within the experimental range, guaranteeing the practical efficiency of $O(KL)$ filtering.
- Qualitative experiment (MS MARCO): For the query "tonsillitis is a throat infection that occurs on the tonsils", pure NNS returns 3 top-5 results that are almost word-for-word identical to "strep throat is a bacterial infection of the throat and tonsils". LotusFilter successfully deduplicates these and returns diverse results including tonsillitis, mastoiditis, spongy dermatitis, etc.
- Qualitative experiment (Revisited Paris): For a query image of the Centre Pompidou in Paris, NNS returns 5 images with nearly identical compositions, whereas LotusFilter returns diverse photos taken from various angles and distances.
- The recall of Top-1 is unaffected—the first loop iteration inevitably preserves the nearest neighbor from the initial search.
- As $N$ increases, $\varepsilon^*$ gradually decreases while $L$ increases, indicating that a finer cutoff threshold is required as the data density grows.
- When $\lambda$ increases from 0.3 to 0.5, $L$ grows from 18.4 to 29.3, indicating that while the cutoff table expands when users demand higher diversity, it remains highly controllable.

## Highlights & Insights
- **Precomputation Over Real-Time Computation**: Shifts the $O(DS^2)$ online pairwise distance computation entirely to offline cutoff table construction, leaving only integer set operations at run-time. This "space-for-time + offline-for-online" paradigm yields massive benefits in high-dimensional scenarios, as the cost of traditional methods scales with $D$, whereas LotusFilter is independent of $D$.
- **OrderedSet Data Structure Design**: By simultaneously maintaining an array (preserving order) and a hash set (fast deletion), it trades extra memory for a Pop $O(L)$ + Remove $O(1)$ combination. This design precisely matches the greedy filtering pattern, proving simpler and more efficient than priority queues or skip lists.
- **Architectural Advantages of Pure Post-Processing**: LotusFilter does not modify ANNS indices or access raw vectors, allowing it to be integrated as a black-box module into any ANNS pipeline. This means that if diversification yields poor utility, the filter can be switched off to seamlessly fallback, boasting exceptionally low engineering costs due to its zero-intrusive design.
- **Practicality of the Safeguard Mechanism**: When $\varepsilon$ is large, it may over-prune the candidate set. The safeguard mode terminates filtering prematurely and fills the remaining slots with raw candidates, ensuring the result count is always $K$. This graceful degradation design is crucial for production systems.

## Limitations & Future Work
- A global single threshold $\varepsilon$ might lack flexibility in scenarios with uneven data distribution, leading to potential over-pruning in dense regions and insufficient filtering in sparse regions. Future studies could explore region-adaptive thresholds.
- Learning $\varepsilon$ requires predetermining $K$ beforehand. If $K$ varies at run-time, $\varepsilon^*$ may no longer be optimal, limiting the flexibility in dynamic query scenarios.
- There is only a lower-bound guarantee for the diversity term of the objective function $f$. A theoretical proof of optimality for the entire objective is lacking, leaving the gap to the optimal solution unquantifiable.
- The cutoff table memory of $64LN$ bits might become non-negligible for ultra-large datasets (e.g., $N>10^8$), necessitating investigation into cutoff table compression schemes.
- In low-dimensional vector spaces, simpler methods (e.g., GMM) may perform better; LotusFilter's advantages are primarily prominent in high-dimensional scenarios (e.g., $D>768$).
- End-to-end RAG evaluations (such as LLM-as-a-judge) have not yet been conducted, leaving the actual improvement of diverse search on downstream LLM generation quality uncertain.
- The name "LotusFilter" stems from the process of deleting neighboring points inside circles centered around each vector during filtering, resembling the visual imagery of lotus leaves spreading across water.

## Related Work & Insights
- **vs GMM (Greedy Max-Min)**: GMM selects points furthest from the selected set via greedy iterations. This yields the strongest diversity but requires real-time $O(DKS)$ computation. While approximating diversity, LotusFilter reduces the filtering time to only 1/670 of GMM (0.02 ms vs 13.4 ms).
- **vs MMR (Maximal Marginal Relevance)**: MMR cannot directly leverage modern ANNS methods and requires full database computation, making online deployment extremely slow. LotusFilter, acting as a pure post-processing module, is perfectly compatible with ANNS engines such as HNSW or IVF.
- **vs Learned Index**: LotusFilter inherits the paradigm of learned data structures (such as Learned B-trees optimizing indices by learning data distributions), but innovatively applies learning to the diversification threshold rather than the index structure itself.
- **vs Hirata et al. (Modern ANNS + DNNS)**: This is the only work applying modern ANNS to diverse inner product search, but it still relies on online distance computations. LotusFilter achieves a qualitative leap by completely eliminating online distance computations.

## Rating
- Novelty: ⭐⭐⭐⭐ The approach of precomputing the cutoff table + greedy filtering is simple and elegant, though the core idea is not entirely brand new (resembling the precomputation concept of Bloom Filters).
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three datasets cover text and image scenarios, with complete quantitative and qualitative evaluations, including detailed reports on complexity analysis, ablation studies, and preprocessing durations.
- Writing Quality: ⭐⭐⭐⭐ The algorithm description is clear, theoretical derivations are rigorous, and the logical chain from problem definition to data structure design is complete.
- Value: ⭐⭐⭐⭐⭐ Directly targets the core bottleneck of search duplication in RAG systems. The 0.02 ms filtering and plug-and-play design give it immense engineering value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Welfarist Formulations for Diverse Similarity Search](../../ICLR2026/information_retrieval/welfarist_formulations_for_diverse_similarity_search.md)
- [\[ACL 2025\] Drama: Diverse Augmentation from Large Language Models to Smaller Dense Retrievers](../../ACL2025/information_retrieval/drama_diverse_augmentation_from_large_language_models_to_smaller_dense_retriever.md)
- [\[ACL 2025\] FlashBack: Efficient Retrieval-Augmented Language Modeling for Fast Inference](../../ACL2025/information_retrieval/flashbackefficient_retrieval-augmented_language_modeling_for_long_context_infere.md)
- [\[ICML 2026\] LEMUR: Learned Multi-Vector Retrieval](../../ICML2026/information_retrieval/lemur_learned_multi-vector_retrieval.md)
- [\[ICLR 2026\] Graph-based Nearest Neighbors with Dynamic Updates via Random Walks](../../ICLR2026/information_retrieval/graph-based_nearest_neighbors_with_dynamic_updates_via_random_walks.md)

</div>

<!-- RELATED:END -->
