---
title: >-
  [Paper Note] Differentially Private Synthetic Text Generation for Retrieval-Augmented Generation (RAG)
description: >-
  [ACL 2026][LLM Safety][Differential Privacy] DP-SynRAG utilizes LLMs to distill private RAG databases into differentially private synthetic text libraries in a **one-time** process. Subsequent queries do not consume any…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Differential Privacy"
  - "Synthetic Data"
  - "RAG"
  - "private prediction"
  - "subsample-and-aggregate"
date: 2026-05-08
content_hash: 39103d705e80eda3
---

# Differentially Private Synthetic Text Generation for Retrieval-Augmented Generation (RAG)

**Conference**: ACL 2026  
**arXiv**: [2510.06719](https://arxiv.org/abs/2510.06719)  
**Code**: Extended based on https://github.com/sarus-tech/dp-rag  
**Area**: LLM Security / Differential Privacy / RAG  
**Keywords**: Differential Privacy, Synthetic Data, RAG, private prediction, subsample-and-aggregate

## TL;DR
DP-SynRAG utilizes LLMs to distill private RAG databases into differentially private synthetic text libraries in a **one-time** process. Subsequent queries do not consume any privacy budget. On Medical Synth, MovieLens, and SearchQA datasets, its accuracy significantly outperforms query-time DP-RAG (where DP-RAG performance collapses in multi-query scenarios).

## Background & Motivation
**Background**: RAG grounds LLMs by connecting them to external private knowledge bases. however, in medical, customer, and recommendation scenarios, databases contain highly sensitive content like PII and medical records. Research demonstrates that (a) even benign queries can cause LLMs to regurgitate private segments; (b) targeted extraction and membership inference attacks can efficiently retrieve raw records.

**Limitations of Prior Work**: Existing private RAG solutions (DP-RAG / Koga 2025 / Wang 2025) rely on **query-time DP**, adding noise to the output layer of every query response. This causes the privacy budget to **accumulate linearly with the number of queries**: to maintain $\varepsilon_{\text{query}} = 10$ over 1000 queries, the total budget $\varepsilon_{\text{total}} \approx 10000$. This leads to either rapid budget depletion or noise levels that render the system unusable. Figure 3 shows that DP-RAG fails completely after 20 queries when $\varepsilon_{\text{total}}=20$.

**Key Challenge**: A knowledge base is a resource that is "read repeatedly," but query-time DP assumes "every read must pay for privacy." This assumption is fundamentally mismatched for the multi-query nature of RAG. Privacy costs should be paid once during "database construction," after which all queries benefit from post-processing immunity.

**Goal**: Construct a solution that is (1) independent of query count with a fixed privacy budget, (2) requires no DP-SGD fine-tuning of LLMs, and (3) **preserves task-critical details** (disease names, user preferences, etc.) within synthetic text rather than merely learning dataset-average styles.

**Key Insight**: Build upon the private prediction framework (subsample-and-aggregate + multi-document logit aggregation + clipping) but introduce a strategy of **keyword-based clustering followed by intra-cluster synthesis**. Prior works (Amin/Tang/Hong) utilized random subsampling of the entire database, which only captures global average features. Ours uses DP clustering to ensure each subset is thematic, thereby generating synthetic text that preserves "locality."

**Core Idea**: Transform query-time DP into data-time DP through a five-step process: "DP keyword histogram $\rightarrow$ DP soft clustering $\rightarrow$ DP embedding reranking $\rightarrow$ intra-cluster private prediction rewriting $\rightarrow$ LLM self-filtering." This pays a one-time budget for infinite queries.

## Method

### Overall Architecture
The pipeline consists of two stages (Algorithm 1, 5 sub-steps):

**Stage 1: DP Soft Clustering**
(a) **Keyword Histogram**: The LLM extracts $K$ representative keywords from each document (limiting to $K$ yields a sensitivity of $\sqrt{K}$). A histogram is summed across the database and Gaussian noise is added: $h' = h + \mathcal{N}(0, \sigma_h^2 I)$.
(b) **Keyword Soft Clustering**: The top-$R$ keywords $W = \{w_1, \ldots, w_R\}$ are taken from $h'$ (in descending order of frequency). The process iterates through keywords in **reverse frequency order** (to prevent high-frequency, uninformative words from dominating) to form clusters $C_r = \{d_i \mid w_r \in d_i, \text{and } d_i \text{ belongs to } <L \text{ clusters}\}$. Each document is assigned to a maximum of $L$ clusters.
(c) **Embedding Reranking**: For each $C_r$, a noisy mean embedding is calculated via the Gaussian mechanism: $\mu(C_r) = \sum_{d_i \in C_r} \mathcal{E}(d_i) + \mathcal{N}(0, \sigma_\mu^2 I)$. The exponential mechanism is then used to select a similarity threshold $\theta_s$, retaining $S_r = \{d_i \in C_r \mid \text{sim}(\mathcal{E}(d_i), \mu(C_r)) > \theta_s\}$.

**Stage 2: Synthetic Text Generation**
(d) **Private Prediction**: Logit aggregation is performed in parallel for each $S_r$. Each document is paired with a rephrase prompt $p_i = (\text{"Rephrase the following text:"}, d_i)$. The logit for the $n$-th token is $z_n(S_r) = \sum_{d_i \in S_r} \text{clip}_c(\mathcal{L}(p_i, y_{r, <n}))$. Softmax sampling is equivalent to the exponential mechanism with sensitivity $c$. This is repeated for $T$ steps to obtain synthetic text $y_r$ of length $T$.
(e) **Self-filtering**: Each $y_r$ plus a downstream task description is fed back to the LLM to verify task utility. Only "YES" entries enter the synthetic library (post-processing does not consume privacy budget).

**Privacy Accounting** (Theorem 1): The entire pipeline satisfies $(\varepsilon, \delta)$-DP, where $\rho = \frac{K}{2\sigma_h^2} + L \left( \frac{1}{8}\varepsilon_{\theta_s}^2 + \frac{1}{2\sigma_\mu^2} + \frac{T}{2}\left(\frac{c}{\tau}\right)^2 \right)$, converted to $\varepsilon = \rho + \sqrt{4\rho\log(1/\delta)}$. A key technique is **overlapping parallel composition**; since each document is in at most $L$ clusters, the privacy cost of parallel processing is $L \cdot \rho_{\text{cluster}}$ rather than $R \cdot \rho_{\text{cluster}}$.

### Key Designs

1.  **DP Keyword Soft Clustering (vs. Random Subsets in Prior Private Prediction)**:
    *   **Function**: Segregates the original database into thematic clusters, allowing private prediction to retain locality (specific details like disease names) instead of just global averages.
    *   **Mechanism**: Uses LLM-extracted keywords + noisy histograms to select top-$R$ themes. Clusters are assigned in reverse frequency order (distinguishing low-frequency terms first; $L$ overlaps increase relevant theme probability). Finally, DP embedding means + thresholding remove outliers. This removal occurs within clusters and adds no budget cost.
    *   **Design Motivation**: Amin et al.'s random subsampling only captures average characteristics, which is useless for RAG tasks requiring specific facts. Table 1 shows DP-Synth/Aug-PE achieves 0% accuracy on Medical Synth. Hard clustering ($L=1$) misallocates polysemous documents (dropping accuracy by 31.88% on Llama-3.1). Soft clustering + reranking represents a fine trade-off between locality and noise.

2.  **Token-level Private Prediction via Softmax (Implicit Noise via Exponential Mechanism)**:
    *   **Function**: Leverages the natural randomness of LLM token sampling to provide DP guarantees during document rewriting, avoiding explicit noise addition.
    *   **Mechanism**: For each $d_i \in S_r$, logits for the $n$-th token are clipped to $[-c, c]$ and summed to form $z_n(S_r)$. Softmax sampling is then equivalent to an exponential mechanism with utility equal to clipped logits and sensitivity $c$. To improve quality, a centered clipping variant is used to preserve relative differences between high-logit tokens.
    *   **Design Motivation**: Adding Gaussian noise directly to logits distorts distributions (small logits are overwhelmed). Using softmax randomness as the DP noise source is more "natural" and less damaging to generation quality.

3.  **Data-time DP + Self-filtering Post-processing**:
    *   **Function**: Spends the entire DP budget on building the synthetic library once, making subsequent queries budget-free. LLM-based filtering then improves RAG accuracy by removing low-quality synthesis.
    *   **Mechanism**: The construction process satisfies $(\varepsilon, \delta)$-DP. Due to post-processing immunity, all subsequent operations (indexing, retrieval, inference) are "free." Self-filtering uses public task information and synthetic text, never touching the raw database.
    *   **Design Motivation**: Query-time DP is a fundamental mismatch in RAG. Moving the budget to construction results in a flat "query count vs. accuracy" curve (Figure 3).

### Loss & Training
**Training-free**. All LLMs are frozen and used for inference (extraction, rewriting, and filtering use the same model). Key parameters: $K=10$ keywords/doc, $R=500$ or $1000$, $L=5$ overlap, $k=80-100$ docs/cluster, $T=70$ tokens, $\tau=1.0$, $\varepsilon_{\text{total}}=10$.

## Key Experimental Results

### Main Results

Three datasets × three LLMs × five methods ($\varepsilon_{\text{total}}=10$ for DP-SynRAG, $\varepsilon_{\text{query}}=10$ for DP-RAG):

| Dataset | Method | Phi-4-mini (3.8B) | Gemma-2-2B | Llama-3.1-8B |
| :--- | :--- | :--- | :--- | :--- |
| **Medical Synth** | Non-RAG | 0.00 | 0.00 | 0.00 |
| | RAG (no DP) | 87.00 | 85.20 | 86.20 |
| | DP-Synth (Amin'24) | 0.00 | 0.00 | 0.00 |
| | Aug-PE | 0.00 | 0.00 | 0.00 |
| | **Ours** | **67.26** | **67.06** | **61.26** |
| | DP-RAG ($\varepsilon_{\text{total}}{\approx}10\text{k}$) | 59.92 | 67.06 | 48.94 |
| **MovieLens** | RAG (no DP) | 67.80 | 54.60 | 70.80 |
| | DP-Synth | 37.60 | 16.64 | 46.12 |
| | Aug-PE | 36.16 | 26.04 | 44.96 |
| | **Ours** | **42.56** | **41.08** | 54.12 |
| | DP-RAG ($\varepsilon_{\text{total}}{\approx}5\text{k}$) | 34.72 | 40.48 | **56.80** |
| **SearchQA** | RAG (no DP) | 92.16 | 94.12 | 95.10 |
| | DP-Synth | 60.20 | 20.20 | 40.00 |
| | **Ours** | **89.61** | **85.10** | **91.18** |
| | DP-RAG ($\varepsilon_{\text{total}}{\approx}1\text{k}$) | 85.10 | 83.14 | 84.90 |

**Conclusion**: Ours significantly leads all baseline methods with the same budget (DP-Synth/Aug-PE) across all datasets and outperforms DP-RAG in most cases—despite DP-RAG using over 1000x the actual budget.

### Ablation Study

Impact of DP-SynRAG components (accuracy %):

| Dataset / Model | Full | w/o Retrieval | w/o Self-filter | Hard cluster ($L=1$) |
| :--- | :--- | :--- | :--- | :--- |
| Medical Synth / Phi-4 | 67.26 | 65.92 (-1.34) | 66.78 (-0.48) | 42.52 (**-24.74**) |
| Medical Synth / Llama-3.1 | 61.26 | 57.74 (-3.52) | 52.20 (**-9.06**) | 29.38 (**-31.88**) |
| MovieLens / Llama-3.1 | 54.12 | 53.76 (-0.36) | 45.12 (**-9.00**) | 46.56 (-7.56) |
| SearchQA / Gemma-2 | 85.10 | 83.73 (-1.37) | (N/A) | 67.06 (**-18.04**) |

**Conclusion**: (1) Soft clustering ($L>1$) is the most critical component; hard clustering drops accuracy by 5–32%. (2) Self-filtering is vital for Medical/MovieLens. (3) Embedding reranking provides consistent small gains.

### Key Findings
*   **Fundamental Difference in Utility Curves**: Ours exhibits a horizontal line (fixed budget), while DP-RAG shows a sharp decline. At 20+ queries, DP-RAG is unusable even at high total budgets.
*   **Failures of Global Baselines**: DP-Synth and Aug-PE fail (0% accuracy) on Medical Synth, proving that learning only global average features is a dead end for fact-retrieval RAG tasks.
*   **DP Ceiling on Rare Themes**: Accuracy approaches 0% when the ground-truth disease appears in <30 documents. This is an inherent DP trade-off: a system that consistently answers for rare diseases is by definition leaking the presence of rare patients.
*   **The Overlap Sweet Spot**: $L=5$ is stable across datasets, balancing polysemy handling and noise dispersion.

## Highlights & Insights
*   **Timing Shift as Contribution**: Shifting privacy costs from "per-query" to "one-time construction" releases massive utility in RAG's "read repeatedly" scenario.
*   **Natural Randomness as DP Source**: Using the softmax/exponential mechanism equivalence avoids the distortion inherent in manual logit noise addition.
*   **Overlapping Parallel Composition**: Applying zCDP to cluster overlaps ($L\rho$ budget) is a sophisticated application of DP theory with broad applicability to multi-branch DP workflows.
*   **Honesty regarding DP Limits**: Acknowledgement that rare themes must have near-zero utility to maintain privacy provides clear boundaries for practical use.

## Limitations & Future Work
*   **Inapplicability to Rare Themes**: Accuracy for themes with <30 documents is almost zero due to DP constraints, limiting long-tail knowledge applications.
*   **Static Database Constraint**: Full reconstruction is required for significant updates (e.g., ~40 minutes for 8000 docs). Incremental refresh mechanisms are needed.
*   **Tight Budget Collapse**: Utility drops sharply at $\varepsilon < 5$. Future exploration into non-token-level locality-preserving generation is required.
*   **Surface-form Keyword Dependency**: Clustering may fail if synonyms represent the same theme differently; synonym expansion could be a future improvement.

## Related Work & Insights
*   **vs. DP-RAG (Grislain 2025)**: Query-time DP with per-query budget vs. Ours (data-time DP with one-shot budget).
*   **vs. DP-Synth (Amin 2024)**: Both use private prediction, but Amin's random subsampling misses locality; ours adds DP clustering to solve this.
*   **vs. DP-OPT / Fine-tuning**: Ours is training-free and significantly more computationally efficient for database updates compared to DP-SGD.
*   **Insight**: The timing of the DP budget spend (build-time vs. use-time) should be a primary design consideration for any DP system.

## Rating
*   Novelty: ⭐⭐⭐⭐
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐
*   Writing Quality: ⭐⭐⭐⭐
*   Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Beyond Explicit Refusals: Soft-Failure Attacks on Retrieval-Augmented Generation](beyond_explicit_refusals_soft-failure_attacks_on_retrieval-augmented_generation.md)
- [\[ACL 2026\] Knowledge Poisoning Attacks on Medical Multi-Modal Retrieval-Augmented Generation](knowledge_poisoning_attacks_on_medical_multi-modal_retrieval-augmented_generatio.md)
- [\[ICML 2026\] ACTG-ARL: Differentially Private Conditional Text Generation with RL-Boosted Control](../../ICML2026/llm_safety/actg-arl_differentially_private_conditional_text_generation_with_rl-boosted_cont.md)
- [\[ACL 2026\] AGSC: Adaptive Granularity and Semantic Clustering for Uncertainty Quantification in Long-text Generation](agsc_adaptive_granularity_and_semantic_clustering_for_uncertainty_quantification.md)
- [\[AAAI 2026\] Privacy-protected Retrieval-Augmented Generation for Knowledge Graph Question Answering](../../AAAI2026/llm_safety/privacy-protected_retrieval-augmented_generation_for_knowledge_graph_question_an.md)

</div>

<!-- RELATED:END -->
