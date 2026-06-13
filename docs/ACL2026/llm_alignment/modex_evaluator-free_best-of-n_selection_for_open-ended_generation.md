---
title: >-
  [Paper Note] ModeX: Evaluator-Free Best-of-N Selection for Open-Ended Generation
description: >-
  [ACL 2026][LLM Alignment][Best-of-N] The authors model Best-of-N selection for open-ended text generation as "finding modal clusters on a similarity graph of generated texts." By utilizing n-gram Jaccard graph constructi…
tags:
  - "ACL 2026"
  - "LLM Alignment"
  - "Best-of-N"
  - "Self-consistency"
  - "Spectral Clustering"
  - "Mode Extraction"
  - "Evaluator-free"
date: 2026-05-08
content_hash: 4e1efb16380bad7e
---

# ModeX: Evaluator-Free Best-of-N Selection for Open-Ended Generation

**Conference**: ACL 2026  
**arXiv**: [2601.02535](https://arxiv.org/abs/2601.02535)  
**Code**: https://github.com/deeplearning-wisc/ModeX  
**Area**: LLM Inference / Test-time Compute / Best-of-N Sampling  
**Keywords**: Best-of-N, Self-consistency, Spectral Clustering, Mode Extraction, Evaluator-free  

## TL;DR
The authors model Best-of-N selection for open-ended text generation as "finding modal clusters on a similarity graph of generated texts." By utilizing n-gram Jaccard graph construction, recursive Fiedler vector spectral clustering, and centrality-based centroid selection, they generalize self-consistency to tasks without standard answers (summarization, code, math) without requiring any reward models or LLM-judges.

## Background & Motivation

**Background**: Single-path LLM generation is sensitive to sampling noise, where a single unfavorable token can trigger hallucination propagation. Best-of-N (BoN) and self-consistency mitigate this by sampling multiple paths and selecting the optimal one, which has proven effective for mathematical and multiple-choice tasks.

**Limitations of Prior Work**: Existing BoN/self-consistency methods rely heavily on two types of external components: (1) external reward models or process reward models (expensive and requiring specialized training data), or (2) exact string matching for voting (only applicable to finite answer spaces). For tasks with **infinite output spaces** like summarization, code generation, and open-ended QA, both fail: reward models are often unavailable, and exact-match voting fails because the same semantics can be expressed through countless surface forms.

**Key Challenge**: In open-ended generation, it is impossible to enumerate answers for majority voting, and obtaining cheap, reliable external evaluators is difficult. The problem of "which sample is best" remains a dilemma.

**Goal**: Design an **evaluator-free** BoN selection method that naturally generalizes majority voting to open-ended generation without pre-defined answer sets.

**Key Insight**: The authors observe that high-quality generations tend to **cluster** in semantic space, while hallucinations or anomalous outputs appear as sparse outliers. Thus, identifying the most likely correct sample translates into finding the sample at the center of the densest semantic cluster—a classic **mode estimation** problem in statistics.

**Core Idea**: Treat N generations as a similarity graph (edge weights = n-gram Jaccard). Use spectral clustering to recursively extract the largest "semantic modal cluster," then select the node with the highest degree as the centroid. Essentially, this replaces exact-match voting with **Kernel Density Estimation (KDE)**, generalizing majority voting to continuous semantic manifolds.

## Method

### Overall Architecture
ModeX is a three-step pipeline (Figure 2): (1) **Adjacency Matrix Construction**: For N sampled outputs, use the sum of unigram, bigram, and trigram Jaccard similarities as edge weights to construct a symmetric matrix $A \in \mathbb{R}^{N \times N}$. (2) **Recursive Spectral Clustering**: Calculate the Fiedler vector (second smallest eigenvector) of the graph Laplacian $L = D - A$ for bisection. Use conductance $\phi$ to determine split quality; if $\phi < \tau$ (where $\tau=0.8$), the larger sub-cluster is retained for further recursion; otherwise, the process stops. (3) **Centrality-based Centroid Selection**: Select the node $v_c = \arg\max_i \sum_j \tilde{A}_{ij}$ with the highest weighted degree in the final cluster as the output. The process requires no neural network re-evaluation, and its complexity $\mathcal{O}(N^2)$ is significantly lower than the generation cost $\mathcal{O}(NL)$. A lightweight variant, ModeX-Lite, is also introduced for **mid-generation pruning**.

### Key Designs

1.  **n-gram Jaccard Similarity Graph (Replacing exact match/embeddings)**:
    - **Function**: Quantifies text similarities as graph edge weights, mathematically defining majority voting in infinite output spaces.
    - **Mechanism**: $A_{i,j} = s_1(v_i, v_j) + s_2(v_i, v_j) + s_3(v_i, v_j)$, where $s_k$ is the Jaccard similarity of $k$-gram sets. These capture lexical coverage (unigram), phrase fluency (bigram), and structural information (trigram). Ablations in Appendix F show that removing trigrams causes the largest performance drop, confirming higher-order n-grams carry significant information.
    - **Design Motivation**: The authors compared LastTokenEmb and SentenceBERT embedding similarities (Table 3). n-gram Jaccard performed better across all three tasks because embeddings struggle with critical tokens in highly structured tasks like code or math, whereas Jaccard aligns directly with specific tokens and syntactic fragments.

2.  **Recursive Fiedler Vector Spectral Clustering (Adaptive clusters + Mode separation)**:
    - **Function**: Automatically identifies the "main mode" from N samples without a predefined $K$.
    - **Mechanism**: Solve $f = \arg\min_{u^\top \mathbf{1}=0, \|u\|=1} u^\top L u$, bisect based on $f_i \ge 0$, and use conductance $\phi(\mathcal{G}_1, \mathcal{G}_2) = \mathrm{cut} / \min(\mathrm{vol}_1, \mathrm{vol}_2)$ to judge if a "meaningful split" exists. If $\phi < 0.8$, recursion continues. Theoretically, this relates to Cheeger’s inequality $\lambda_2 / 2 \le \phi^* \le \sqrt{2\lambda_2}$; in the large $N$ limit, the Fiedler cut is equivalent to slicing through "low-density valleys" between modes.
    - **Design Motivation**: Fixed-K K-means is unsuitable for scenarios where the number of modes varies by task or input. Recursive bisection with a conductance threshold is a parameter-efficient "adaptive clustering" method, and conductance is more stable than normalized cut (Figure 5 ablation).

3.  **Degree Centrality for Centroid = Empirical KDE Peak**:
    - **Function**: Identifies the "most representative" output within the primary cluster as the final answer.
    - **Mechanism**: Select the node with the maximum degree $v_c = \arg\max_i \sum_j \tilde{A}_{ij}$ in the sub-cluster adjacency matrix $\tilde{A}$. Theoretically, if Jaccard is viewed as a kernel $K(\cdot, \cdot)$, the weighted degree $d(v_i) = \sum_j S(v_i, v_j) \propto \hat{p}(v_i)$ represents the kernel density estimate at that point—selecting the maximum degree is equivalent to finding the **mode within a unimodal cluster**.
    - **Design Motivation**: Rigorously formalizes "open-ended majority voting" by shifting from discrete counts to continuous KDE peak estimation. Theorem 2 provides a formal proof of this correspondence.

### Loss & Training
The method is entirely training-free and operates only during inference: N paths are generated in parallel → Graph construction → Recursive clustering → Centroid selection. Hyperparameters are limited to the conductance threshold $\tau = 0.8$ and pruning interval $T = 100$ tokens for ModeX-Lite. Sensitivity experiments show stable performance within reasonable ranges ($\tau \in [0.5, 0.8]$, $T \in [100, 500]$).

## Key Experimental Results

### Main Results
Evaluated using Qwen2.5-7B-Instruct and LLaMA3.1-8B-Instruct across three open-ended tasks (CNN/DailyMail summarization, HumanEval code, Math-500):

| Model / Method | CNN/DM ROUGE-L | HumanEval Pass@1 | Math-500 Acc |
|------------|---------------|------------------|--------------|
| Qwen Single Path (mean±std) | 20.17 ± 0.28 | 69.89 ± 3.59 | 70.98 ± 1.74 |
| Qwen + Self-Refine (k=4) | 18.22 | 26.22 | 68.67 |
| Qwen + LLM Judge (N=16) | 19.72 | 65.24 | 74.67 |
| Qwen + Perplexity BoN (N=16) | 21.06 | 73.17 | 78.00 |
| Qwen + Self-Certainty BoN (N=16) | 19.32 | 55.49 | 67.00 |
| **Qwen + ModeX (N=16)** | 21.06 | **75.61** | **78.00** |
| **Qwen + ModeX-Lite (N=16)** | **21.89** | **78.66** | 75.33 |
| Qwen + Gold-Standard BoN (RM, N=16) | 20.49 | – | 82.00 |
| LLaMA Single Path | 21.30 ± 0.34 | 18.29 ± 15.22 | 38.75 ± 1.98 |
| **LLaMA + ModeX (N=16)** | 22.70 | **32.32** | **49.33** |
| **LLaMA + ModeX-Lite (N=16)** | **22.80** | 29.88 | 45.33 |

Qwen's code performance improved from 69.89% to 78.66% Pass@1 (+8.8 points), approaching or exceeding gold-standard BoN using reward models. Summarization performance consistently outperformed LLM-Judge and Self-Refine.

### Ablation Study
Ablations on similarity functions, n-gram combinations, and LLM scalability:

| Configuration | CNN/DM ROUGE-L | HumanEval Pass@1 | Math-500 Acc |
|------|---------------|------------------|--------------|
| Single Path | 20.17 | 69.89 | 70.98 |
| ModeX-LastTokenEmb (N=16) | 20.26 | 75.00 | 71.33 |
| ModeX-SentenceBERT (N=16) | 20.92 | 72.56 | 70.67 |
| **ModeX-n-gram (N=16)** | **21.06** | **75.61** | **78.00** |
| ModeX (N=8) full n-gram | 21.08 | – | – |
| (-) Unigram | 20.91 | – | – |
| (-) Bigram | 20.78 | – | – |
| (-) Trigram | 20.78 | – | – |
| Qwen2.5-14B Single | 19.90 | 30.41 | 72.62 |
| Qwen2.5-14B + ModeX (N=8) | 20.66 | **39.02** | **77.67** |
| Qwen2.5-32B + ModeX (N=8) | 20.87 | **35.37** | **78.67** |
| GPT-4 + ModeX (N=16, AIME2025) | – | – | 30.00 (vs 20.42 baseline) |

Complexity comparison (Single CNN/DM sample, Qwen-7B): Single Path 5.5s / Self-Refine 31.7s / LLM Judge 10.7s / **ModeX-Lite N=16 only 9.1s**, which is 3.5× faster than Self-Refine.

### Key Findings
- **Structure-Aware Selection > Brute-Force Sampling**: Increasing N from 4 to 16 yielded only +1.34 points for LLM-Judge (LLaMA Math), while ModeX-Lite gained +7.33 points, indicating that "more samples without principled selection" is ineffective.
- **Early Pruning is Viable**: On Math tasks, Figure 4 shows that < 50% of the trajectory is sufficient to identify high-quality paths, supporting the rationale for mid-generation pruning in ModeX-Lite.
- **Surpassing Reward Models**: Case in point: Qwen Math ModeX-Lite 75.33% vs gold-standard RM-BoN 82% still shows a gap; however, in code tasks, ModeX achieves state-of-the-art results compared to having no available reward models.
- **Self-Refine Can Degrade Performance**: Iterative self-correction dropped LLaMA summarization from 21.30 to 15.28, confirming that more compute without a selection mechanism can amplify error propagation.

## Highlights & Insights
- **Paradigm Shift from "Counting" to "Density Estimation"**: Translating majority voting into KDE via mode estimation on a continuous semantic manifold provides both theoretical interpretability and generalizability to any task defining a similarity kernel.
- **Spectral Clustering for "Adaptive Modal Slicing"**: The combination of Fiedler vectors, Cheeger's inequality, and recursive conductance thresholds is equivalent to slicing through low-density valleys in multi-modal distributions—a perfect application of graph theory to NLP selection problems.
- **n-gram Jaccard Outperforms Embeddings**: Robust performance in code/math suggests that embeddings are not universal; token/syntactic-level overlap is more accurate for highly structured tasks. This observation may inspire other ensemble tasks.
- **Early Pruning in ModeX-Lite**: Discovering that high-quality paths are distinguishable at 50% length allows spectral clustering to be integrated into the token stream, transforming BoN from "select after sampling" to "prune while sampling," offering a new path for test-time compute optimization.

## Limitations & Future Work
- **Authors' Admissions**: (1) n-gram Jaccard misses deep semantic paraphrasing; valid but surface-different paraphrases may be misjudged as outliers. (2) Reliance on the "majority = correct" assumption; if the model **mode-collapses into a hallucination**, ModeX will reinforce the error.
- **Self-Observations**: (1) The method is essentially "collective voting" and may harm tasks requiring long-tail or low-probability correct answers (e.g., creative writing, reverse engineering). (2) A fixed conductance threshold $\tau=0.8$ might not be optimal across all tasks and models, as only Math was swept. (3) N=16 is relatively small; ModeX's behavior with N > 100 (e.g., for GPT-4) is unknown. (4) Performance of trigrams under different tokenization (Chinese/Code) may require language-specific tuning.
- **Improvement Ideas**: Replace Jaccard with a hybrid kernel of token-level edit distance and embedding similarity; for creative tasks, identify high-degree outliers away from the main cluster; utilize "learn-to-tune" for adaptive conductance thresholds.

## Related Work & Insights
- **vs Self-Consistency (Wang et al. 2023)**: SC is limited to exact-match voting for arithmetic/choice; ModeX generalizes to open text without answer extraction post-processing.
- **vs LLM-as-Judge (Zheng et al. 2023)**: LLM Judge requires secondary inference and introduces evaluator bias; ModeX is >10 points higher in Pass@1 on Qwen code and faster (9.1s vs 10.7s).
- **vs Reward-Model BoN**: Gold-standard but requires specific training; ModeX is a universal solution when RMs are missing (e.g., code tasks).
- **vs Self-Certainty / Perplexity BoN (Kang et al. 2025)**: These rely on internal signals; perplexity is unfaithful for long-form answers (tending toward shortest responses). ModeX uses external consensus structures, making it more robust.
- **vs Self-Refine (Madaan et al. 2023)**: Serial refinement propagates errors and is slow (31.7s/sample); ModeX-Lite is faster (9.1s) and more accurate.

## Rating
- Novelty: ⭐⭐⭐⭐ Uses spectral clustering to generalize majority voting to open-ended generation; a clean conceptual leap with strong algorithmic mapping.
- Experimental Thoroughness: ⭐⭐⭐⭐ 3 tasks × 2 models × 5 baselines, plus large model scaling (14B/32B/GPT-4) and extensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Logical progression from motivation to theory to experiments; Theorem 1/2 and Cheeger’s inequality provide a clear probabilistic justification.
- Value: ⭐⭐⭐⭐⭐ Training-free, evaluator-free, and $\mathcal{O}(N^2)$ complexity. Directly applicable to any LLM system needing inference-time reliability, especially for code and writing where RMs are absent.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond)](../../NeurIPS2025/llm_alignment/artificial_hivemind_the_open-ended_homogeneity_of_language_models_and_beyond.md)
- [\[ACL 2026\] Alignment Data Map for Efficient Preference Data Selection and Diagnosis](alignment_data_map_for_efficient_preference_data_selection_and_diagnosis.md)
- [\[ACL 2026\] Towards Bridging the Reward-Generation Gap in Direct Alignment Algorithms](towards_bridging_the_reward-generation_gap_in_direct_alignment_algorithms.md)
- [\[ACL 2026\] RbtAct: Rebuttal as Supervision for Actionable Review Feedback Generation](rbtact_rebuttal_as_supervision_for_actionable_review_feedback_generation.md)
- [\[ACL 2026\] Mitigating Selection Bias in Large Language Models via Permutation-Aware GRPO](mitigating_selection_bias_in_large_language_models_via_permutation-aware_grpo.md)

</div>

<!-- RELATED:END -->
