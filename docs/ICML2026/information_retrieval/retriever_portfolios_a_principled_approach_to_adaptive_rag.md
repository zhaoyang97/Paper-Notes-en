---
title: >-
  [Paper Note] Retriever Portfolios: A Principled Approach to Adaptive RAG
description: >-
  [ICML 2026][Information Retrieval & RAG][Retrieval-Augmented Generation] Ours reformulates the "retriever selection" problem in RAG as a best-of-$k$ combinatorial optimization problem. A complementary size-$k$ portfolio…
tags:
  - "ICML 2026"
  - "Information Retrieval & RAG"
  - "Retrieval-Augmented Generation"
  - "Retriever Selection"
  - "Submodular Optimization"
  - "Query Routing"
  - "best-of-k"
date: 2026-05-08
content_hash: ff177c59829ad542
---

# Retriever Portfolios: A Principled Approach to Adaptive RAG

**Conference**: ICML 2026  
**arXiv**: [2605.31176](https://arxiv.org/abs/2605.31176)  
**Code**: None  
**Area**: Information Retrieval / Adaptive RAG  
**Keywords**: Retrieval-Augmented Generation, Retriever Selection, Submodular Optimization, Query Routing, best-of-k

## TL;DR
Ours reformulates the "retriever selection" problem in RAG as a best-of-$k$ combinatorial optimization problem. A complementary size-$k$ portfolio is selected from 360 candidates offline using a greedy approach, and a lightweight contrastive router is trained to distribute each query to the top-$\ell$ members in the portfolio online. This approach outperforms single retrievers and inference-time tuning methods like Vendi-RAG across 4 QA benchmarks while significantly reducing token and latency costs.

## Background & Motivation

**Background**: Mainstream RAG systems almost exclusively adopt a "one-size-fits-all" paradigm, selecting a single retriever and a fixed set of hyperparameters (Lewis 2020; Karpukhin 2020; Shuster 2021), ranging from early DPR and FiD to Self-RAG.

**Limitations of Prior Work**: The distribution of QA queries is highly heterogeneous—some are single-hop factoid questions (lexical match suffices), some are multi-hop reasoning tasks (requiring diverse multi-document aggregation), and others involve domain-specific terminology. Ample evidence suggests no single fixed retriever is optimal across all queries (Karpukhin 2020; Jeong 2024; Kalra 2025a). Even for a specific parameterized retriever, the optimal hyperparameters drift per query (Rezaei & Dieng 2025).

**Key Challenge**: Existing adaptive solutions either follow a "few preset strategies + classifier" route (e.g., Adaptive-RAG choosing among "no retrieval / single-step / multi-step") or an "online per-query hyperparameter search" route (e.g., Vendi-RAG iteratively executing retrieve-generate-judge to tune diversity parameter $s$). The former lacks expressive power, while the latter suffers from explosive inference costs and cannot be parallelized.

**Goal**: To select a small, behaviorally complementary subset from a **large** pool of candidate retrievers that covers different regions of the query distribution in expectation, without performing online search per query.

**Key Insight**: Reframe the retriever choice as an "algorithm selection" problem where each retriever is a "candidate algorithm" and each query is a "problem instance." Thus, the RAG retriever selection problem = a data-driven solution portfolio problem (Drygala 2025) / catalogue problem (Kleinberg 2004). Once mapped to this framework, the objective function $F(S)=\mathbb{E}_{q}[\max_{r\in S} s(q,r)]$ naturally satisfies non-negativity, monotonicity, and submodularity, allowing the classic greedy algorithm to achieve a $(1-1/e)$ approximation.

**Core Idea**: Replace the average objective with a best-of-$k$ portfolio objective + offline greedy portfolio selection + online contrastive routing, amortizing the cost of "adaptation" to the offline phase.

## Method

### Overall Architecture

The pipeline consists of two stages:

**Offline Stage**: Evaluate and score all candidate retrievers on a training query set to obtain a score matrix $s(q,r)\in[0,1]$ (using Recall@$k$ as $s$). Then, use Algorithm 1 to greedily select $k$ retrievers to form a portfolio $S$. Simultaneously, train a contrastive router based on retrieval supervision.

**Online Stage**: For a given query $\mathbf{q}$, the router encodes it and calculates similarity with $k$ "retriever embeddings." The top-$\ell$ members of the portfolio are selected to **parallelly** perform retrieval + LLM generation, and finally, a selector aggregates the candidate answers. Crucially, $\ell\le k$ are **fixed** small constants, ensuring predictable latency and parallelizable calls.

### Key Designs

1.  **Best-of-$k$ Portfolio Objective + Submodular Approximation Guarantee**:
    - **Function**: Rewrites the "retriever selection" as a provable combinatorial optimization problem.
    - **Mechanism**: Defines the score of a portfolio $S\subseteq\mathcal{R}$ as $\mathrm{score}(q,S)=\max_{r\in S} s(q,r)$, with the overall objective $F(S)=\mathbb{E}_{q\sim\mathcal{D}}[\max_{r\in S} s(q,r)]$. This $\max$ form inherently rewards members that "cover different query subgroups" (redundant retrievers with similar behavior contribute little to the maximum). Algorithm 1 runs greedy selection after sampling $N$ queries: each step adds $r$ that maximizes marginal gain $\frac{1}{N}\sum_{q\in Q}\max(0,s(q,r)-V[q])$, where $V[q]$ is the current best score for $q$ in $S$. The complexity is $\mathcal{O}(|\mathcal{R}|N)$ per step. Theorem 3.1 provides a theoretical guarantee: with $N=\mathcal{O}((k\log|\mathcal{R}|+\log(1/\delta))/\epsilon^2)$ samples, $F(S)\ge (1-1/e)\mathrm{OPT}-\epsilon$ holds with probability $1-\delta$.
    - **Design Motivation**: Previous methods like Adaptive-RAG / Vendi-RAG relied on heuristics. By formalizing "covering heterogeneous query distributions" via a best-of-$k$ submodular objective with Hoeffding bounds, greedy selection is **theoretically guaranteed** to be near-optimal. The sample complexity scales with $\log|\mathcal{R}|$ rather than query distribution support size, enabling the use of a large pool of 360 candidates.

2.  **360-Dimensional Candidate Pool (Cross-family + Dual Backbone)**:
    - **Function**: Provides a **truly heterogeneous** candidate set for the portfolio selection algorithm to choose from.
    - **Mechanism**: The pool combines three retriever families with two embedding backbones (MPNet and E5). ① **DiscountedSimilarity (DS)**: Greedily selects $n=4$ chunks from FAISS top-$M=1000$ candidates; hyperparameters $(\gamma, r)$ control similarity penalties, resulting in 140 configurations + 1 dense baseline per backbone (141 total). ② **Vendi**: Balances relevance and intra-set diversity using Vendi-score; diversity parameter $s\in[0,1]$ with step 0.05 yields 21 configurations. ③ **GraphDense**: performs BFS expansion on an "entity-chunk bipartite graph" using query entities, followed by re-ranking with MPNet/E5; vary hop counts and max document frequency for 36 configurations. The total pool $|\mathcal{R}|=360$. Scores are pre-cached for efficiency.
    - **Design Motivation**: The gains of a portfolio depend on whether members are complementary. If the pool only contained variations of DS, best-of-$k$ would collapse to a single retriever. Empirical results show size-5 portfolios contain diverse members like GraphDense/E5, Vendi/E5, and GraphDense/MPNet (Table 2), confirming that cross-family and cross-backbone diversity allows the greedy algorithm to outperform "top-$k$ by average score" baselines.

3.  **Two-stage Pipeline (Offline Portfolio + Online Contrastive Router)**:
    - **Function**: Avoids high latency of online per-query hyperparameter search by compressing routing into a lightweight forward pass.
    - **Mechanism**: The router takes raw query text + cached MPNet/E5 query embeddings, passes them through a frozen Flan-T5-Large encoder, and fuses them with backbone-specific embeddings to output similarity scores for each portfolio member. The training objective is a multi-positive contrastive loss (Chen et al. 2024), where "retrievers achieving the highest Recall@$k$ for the query" are treated as positives. At inference, the top-$\ell$ members ($\ell\in\{2,3\}$) are executed in **parallel**, followed by LLM-based answer selection.
    - **Design Motivation**: Unlike Vendi-RAG's serial "retrieve $\to$ generate $\to$ LLM judge $\to$ tune $s \to$ retrieve" loop, the router makes a decision in one forward pass. All $\ell$ branches run in parallel. Token cost and wall-clock time scale linearly with $\ell$ and are independent of $k$. A configuration like $(k=4,\ell=2)$ maintains high accuracy with predictable costs, outperforming Vendi-RAG on the cost-accuracy curve (Figure 4).

### Loss & Training

- **Offline Portfolio**: Uses Recall@$k$ as $s(q,r)$ (calculated using ground-truth documents). Algorithm 1 is run on a union training set pooled from 4 benchmarks. Main results use $k=4$ or $5$.
- **Router**: Multi-positive contrastive loss; positives = retrievers with max Recall@$k$ for the query. Details in Appendix B.9.
- **Answer Model**: Gemma-3-27B-It and Llama-3.1-70B-Instruct; fixed prompt templates ensure fairness.

## Key Experimental Results

### Main Results

Evaluated on four QA benchmarks (HotpotQA / MusiQue / TriviaQA / 2WikiMultiHopQA) with two LLMs. end-to-end Exact Match (Subset of Table 3, best in bold):

| Method | HotpotQA (Gemma) | MusiQue (Gemma) | 2Wiki (Gemma) | HotpotQA (Llama) | MusiQue (Llama) | 2Wiki (Llama) |
|------|------------------|------------------|----------------|-------------------|------------------|----------------|
| No retrieval | 0.326 | 0.061 | 0.226 | 0.348 | 0.059 | 0.192 |
| NN retrieval (MPNet) | 0.395 | 0.129 | 0.241 | 0.476 | 0.139 | 0.292 |
| Best DS retriever | 0.513 | 0.139 | 0.354 | 0.435 | 0.109 | 0.244 |
| Best Vendi retriever | 0.511 | 0.143 | 0.356 | 0.433 | 0.112 | 0.245 |
| Vendi-RAG ($T=20$) | 0.285 | 0.131 | 0.256 | 0.483 | 0.206 | 0.290 |
| **All-pool portfolio** $(k{=}4,\ell{=}2)$ | 0.552 | 0.173 | 0.405 | **0.590** | 0.182 | 0.414 |
| **All-pool portfolio** $(k{=}4,\ell{=}3)$ | **0.558** | **0.195** | **0.414** | 0.583 | **0.209** | **0.419** |

Retrieval-only performance (Figure 3, average across 4 datasets): Size-5 learned portfolio achieves Support Recall 0.594 / F1 0.500, while the "top-5 by average score" baseline only reaches 0.492 / 0.432, a ~10% gap.

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Top-$k$ by avg score | Recall 0.492 @ $k=5$ | Selecting by avg score leads to repetitive GraphDense/E5 dominance, lacking complementarity. |
| Single retriever × 4$k$ docs | F1 drops from 0.32 $\to$ 0.11 ($k{=}1\to 5$) | Increasing chunks for a single retriever hurts F1 by introducing excessive noise. |
| Portfolio $k=2$ | Better than single retriever with 20 docs | Gain comes from retriever complementarity, not just more context. |
| Vendi-only portfolio $(k{=}5,\ell{=}2)$ | Avg EM significantly lower than all-pool | Portfolio advantage shrinks when limited to a single family/backbone. |
| Routing budget $\ell=2 \to 3$ | Visible EM gain on MusiQue / 2Wiki | $\ell$ acts as a cost-accuracy knob; $\ell{=}2$ is sufficient for easy tasks like TriviaQA. |

### Key Findings

- **Complementarity > Average Score**: The 2nd and 3rd members of the portfolio are often not the ones with the highest average scores, but those that "cover failed queries" for previous members. This is the fundamental difference between best-of-$k$ and average-best selection.
- **More Chunks $\neq$ Complementarity**: Doubling chunks for a single retriever cannot replace a portfolio; Recall may increase, but F1 plummets, indicating extra chunks are noise rather than signal.
- **Offline Portfolio crushes Vendi-RAG**: In controlled comparisons within the same Vendi/MPNet search space, fixed portfolios achieve or exceed Vendi-RAG's EM with fewer tokens and lower wall-clock time.
- **Cross-dataset Generalization**: A union-trained portfolio outperforms the best specific retriever family on all 4 datasets, proving that the retriever complementarity learned via best-of-$k$ is transferable across tasks.

## Highlights & Insights

- **Mapping RAG selection to the solution portfolio framework**: Recognizing $\max_{r\in S}s(q,r)$ as a coverage function on the query dimension allows the use of combinatorial optimization tools (submodularity, tight approximations).
- **Offline Amortization = Parallel Inference**: Unlike inference-time tuning (Vendi-RAG / Self-RAG), portfolios use online routing + parallel execution. There are **no serial dependencies on previous outputs**, which is a major selling point for latency-sensitive production systems.
- **Contrastive Router Design**: Using "best Recall@$k$ retriever" as a multi-positive contrastive target transforms the selection problem into a standard contrastive learning task. This approach translates well to expert LLM routing or tool-use agents.
- **RETRIEVER Agnostic**: Sparse, graph, and generative retrievers can all be included in $\mathcal{R}$. Future extensions to BM25 or LLM-as-retriever require no algorithmic changes.

## Limitations & Future Work

- **Dependence on Ground-Truth Support Documents**: Scoring $s(q,r)$ requires Recall@$k$, which depends on annotated supporting documents. Proxies for open-domain or unlabeled production queries (e.g., LLM judges, click-through data) are not explored.
- **Objective limited to $|S|\le k$ rather than cost-weighted**: Real-world retrievers have varying costs (dense vs. graph BFS). Currently, only quantity is controlled, not unit cost, which might lead to expensive portfolios.
- **Robustness under Distribution Shift**: The router's performance in real-world domains (medical/legal) with distribution shift relative to the academic QA training pool is not empirically tested.
- **Lack of comparison with joint retriever-LLM training**: Whether joint fine-tuning of retrievers and LLMs reduces the portfolio advantage remains an open question.
- **Future Directions**: Changing $|S|\le k$ to a knapsack constraint ($\sum_{r\in S} c(r)\le B$) for heterogeneous costs or end-to-end joint training of portfolios and routers.

## Related Work & Insights

- **vs. Adaptive-RAG (Jeong 2024)**: Adaptive-RAG uses query complexity classifiers for 3 manual strategies; ours performs combinatorial optimization over 360 fine-grained retrievers with theoretical guarantees.
- **vs. Vendi-RAG (Rezaei & Dieng 2025)**: Vendi-RAG tunes diversity $s$ per-query online via serial loops; ours compresses this search space offline, lowering token usage and enabling parallel execution.
- **vs. MoR (Kalra 2025b)**: MoR focuses on score-level fusion for a **given** set; ours solves **how to select the small set** from a large pool. They are orthogonal and can be combined.
- **vs. RouterDC (Chen 2024)**: borrows the multi-positive contrastive routing paradigm but shifts the target from expert LLMs to retrievers, with a focus on portfolio selection theory.
- **vs. Drygala 2025 / Kleinberg 2004**: Theoretical ancestors. Ours is a precise instantiation in the RAG domain, leveraging a $(1-1/e)$ approximation for submodular segmentation problems.

## Rating
- Novelty: ⭐⭐⭐⭐ Reformulates RAG retriever selection as a provable best-of-$k$ portfolio problem, a rare formalization in the field.
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 QA benchmarks × 2 LLMs × 360 retrievers, covering main results, ablations, and cost-accuracy; lacks real-world production data.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, theoretical guarantees, and pipeline structure; includes full pseudocode and formulas.
- Value: ⭐⭐⭐⭐ Provides an immediately applicable "offline portfolio + online routing" paradigm for existing RAG systems, highly industrial-friendly.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] BlitzRank: Principled Zero-shot Ranking Agents with Tournament Graphs](blitzrank_principled_zero-shot_ranking_agents_with_tournament_graphs.md)
- [\[AAAI 2026\] REAP: Enhancing RAG with Recursive Evaluation and Adaptive Planning for Multi-Hop Question Answering](../../AAAI2026/information_retrieval/reap_enhancing_rag_with_recursive_evaluation_and_adaptive_planning_for_multi-hop.md)
- [\[ACL 2026\] CORAL: Adaptive Retrieval Loop for Culturally-Aligned Multilingual RAG](../../ACL2026/information_retrieval/coral_adaptive_retrieval_loop_for_culturally-aligned_multilingual_rag.md)
- [\[ICLR 2026\] Revela: Dense Retriever Learning via Language Modeling](../../ICLR2026/information_retrieval/revela_dense_retriever_learning_via_language_modeling.md)
- [\[AAAI 2026\] RRRA: Resampling and Reranking through a Retriever Adapter](../../AAAI2026/information_retrieval/rrra_resampling_and_reranking_through_a_retriever_adapter.md)

</div>

<!-- RELATED:END -->
