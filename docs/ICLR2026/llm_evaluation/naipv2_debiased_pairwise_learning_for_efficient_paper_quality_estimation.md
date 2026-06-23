---
title: >-
  [Paper Note] NAIPv2: Debiased Pairwise Learning for Efficient Paper Quality Estimation
description: >-
  [ICLR 2026][LLM Evaluation][Bradley-Terry] NAIPv2 reformulates "paper quality scoring" as **pairwise ranking learning within the same field and year**, augmented by an RTS signal that probabilistically fuses review scores with confidence. It learns relative superiority during training and degrades to a linear-time pointwise regressor during deployment, achievin
tags:
  - ICLR 2026
  - LLM Evaluation
  - Bradley-Terry
date: 2026-05-08
content_hash: 96f814454f0523c8
---
# NAIPv2: Debiased Pairwise Learning for Efficient Paper Quality Estimation

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=rNl8XiSHiJ](https://openreview.net/forum?id=rNl8XiSHiJ)  
**Code**: [https://sway.cloud.microsoft/Pr42npP80MfPhvj8](https://sway.cloud.microsoft/Pr42npP80MfPhvj8)  
**Area**: Paper Quality Estimation / Automated Peer Review / LLM Evaluation  
**Keywords**: Paper Quality Estimation, Pairwise Learning, Debiasing, Bradley-Terry, Review Confidence, Pointwise Inference  

## TL;DR
NAIPv2 reformulates "paper quality scoring" as **pairwise ranking learning within the same field and year**, augmented by an RTS signal that probabilistically fuses review scores with confidence. It learns relative superiority during training and degrades to a linear-time pointwise regressor during deployment, achieving SOTA results on ICLR review prediction (78.2% AUC / 0.432 Spearman) while being thousands of times faster than autoregressive LLM reviewers.

## Background & Motivation
**Background**: AI-for-Science systems (automated surveys, research agents, literature intelligence tools) require rapid identification of high-quality work from massive volumes of "new papers." However, new papers lack citation histories or journal metrics. Furthermore, the DORA and Leiden Manifesto warn against using prestige-based metrics as proxies for quality, shifting research toward quality estimation based on the **paper content itself**.

**Limitations of Prior Work**: Current solutions fall into two categories, each with significant drawbacks. First, autoregressive LLMs generate reviews or predict scores (e.g., DeepReview, CycleReviewer); these provide good explainability but suffer from **extremely slow inference** (approx. 3 minutes per paper on an RTX 3090) and fail without PDFs. Second, direct regression of review scores (e.g., NAIPv1) is fast (ten papers per second) but performs only slightly better than random.

**Key Challenge**: Why does direct regression fail? The authors attribute this to two factors: (1) **systematic neglect of review confidence**, where existing methods treat all scores as equally credible despite vast differences in reviewer expertise; and (2) **lack of unified scoring standards**, where score scales vary across fields, years, and even individual reviewers, making absolute values impossible to calibrate (i.e., "translational inconsistency").

**Goal**: To achieve the accuracy of autoregressive methods with the speed of regression methods by building a fast, debiased framework for paper quality estimation.

**Key Insight**: **Bypass scale inconsistency using pairwise learning.** By learning the relative order of "which is better" only between papers in the same field and year, the model inherently avoids cross-domain and cross-temporal scale bias. Simultaneously, **reviews are treated as noisy observations of latent true quality**, with confidence modulating noise variance to produce a probabilistic supervision signal (RTS). Crucially, the model uses **pairwise learning during training but pointwise inference during deployment**—the shared backbone allows the pairwise loss to implicitly induce a globally consistent pointwise scoring function, maintaining $O(C)$ linear complexity for deployment.

## Method

### Overall Architecture
NAIPv2 integrates three components: first, **RTS** probabilistically fuses multiple review scores and confidence values into a single quality label $\in [0, 1]$. Second, papers are partitioned into "intra-field, intra-year" groups using clustering and metadata to construct the **NAIDv2** dataset. Finally, **Bradley-Terry pairwise training** is performed within groups using a shared single-branch backbone, which reverts to pointwise scoring during inference.

```mermaid
flowchart LR
    A[Title + Abstract] --> B[Qwen3-Embedding<br/>Hierarchical Clustering → Field Tag]
    C[Review Scores + Confidence] --> D[RTS Gaussian Fusion<br/>→ Quality Label ∈ [0,1]]
    B --> E[Intra-field/year Grouping]
    D --> E
    E --> F[Bradley-Terry Pairwise Training<br/>Shared LLaMA-3 + MLP]
    F -. Shared Params θ .-> G[Pointwise Inference ŷ = f(x; θ)<br/>O(C) Linear Complexity]
```

### Key Designs

**1. RTS (Review Tendency Signal): Probabilistic Fusion of Confidence as Noise Variance**　Instead of averaging scores with fixed weights, RTS treats each score $s_i$ as a noisy observation of the latent true quality $x \in [0, 1]$, where review confidence $c_i$ determines the reliability. This is modeled as a Gaussian likelihood $p(s_i \mid x, c_i) \propto \mathcal{N}(s_i \mid x, \sigma(c_i)^2)$, where variance is linearly modulated by confidence: $\sigma(c_i) = 0.2(1 - c_i) + 0.05$. Higher confidence leads to lower variance and a sharper impact on the fusion. Multiplying the likelihoods of $n$ reviews yields a Gaussian aggregation for $x$, whose closed-form mean is a **score weighted by precision (inverse square of confidence)**: $\mathbb{E}[x] = \frac{\sum_i s_i \sigma(c_i)^{-2}}{\sum_i \sigma(c_i)^{-2}}$. The result is truncated and normalized to $[0, 1]$. This ensures high-confidence reviews carry more weight while low-confidence reviews are included without dominating.

**2. Intra-field/year Pairwise Training + Bradley-Terry: Root-level Debiasing**　For a pair of papers $(a, b)$, a binary preference label is constructed from RTS values: $\text{RTS}_{ab} = \mathbb{I}[\text{RTS}_a > \text{RTS}_b]$. Each paper is processed through a shared LLaMA-3 backbone + MLP head to obtain scalar scores $\hat{y}_a, \hat{y}_b$. The Bradley-Terry preference probability is $\hat{z}_{ab} = \text{sigmoid}(\hat{y}_a - \hat{y}_b)$, optimized via Standard BCE loss. Crucially, **pairs are only constructed within the same cluster and publication year**, preventing spurious comparisons across varying scales. Domain tags are generated via hierarchical clustering of Qwen3-Embedding-4B encodings (with max distance 1.0) rather than GPT keywords, which were found to result in sparse valid pairs.

**3. Pairwise Training to Pointwise Inference: Linear Complexity via Shared Backbone**　While the pairwise loss constrains the relative difference $\hat{y}_a - \hat{y}_b$, since both branches share the same parameters $\theta$, all papers are projected into the same representation space. Consequently, single-paper inference $\hat{y} = f(x; \theta)$ naturally yields a globally comparable pointwise score. Unlike "concatenated pairwise" methods (which require $O(C \log C)$ comparisons), NAIPv2 maintains $O(C)$ linear complexity while achieving higher accuracy.

**4. Difficulty-Aware Curriculum Learning**　Pairs are binned by the RTS gap $\Delta_{ab} = |\text{RTS}_a - \text{RTS}_b|$. Large gaps represent "easy" pairs (clear quality difference), while small gaps represent "hard" pairs. Early training prioritizes easy pairs to capture coarse-grained quality features, gradually upsampling hard pairs to refine fine-grained differentiation.

## Key Experimental Results

The NAIDv2 dataset contains **24,276 ICLR submissions (2021–2025)**, including parsed PDF content and clustered domain labels. The test set is restricted to 1,029 papers from 2025. Models were trained on 4×A40 using 8-bit quantization + LoRA; 10k pairs take approximately 1 hour.

### Main Results (ICLR Review Prediction)

| Category | Method | Acc↑ | F1↑ | AUC↑ | NDCG↑ | ρ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Lower Bound | Random | 0.514 | 0.410 | 0.527 | 0.525 | 0.002 |
| Upper Bound | Info. Leak | 0.819 | 0.757 | 0.894 | 0.995 | 0.984 |
| API | ChatGPT-pointwise | 0.644 | 0.427 | 0.654 | 0.702 | 0.315 |
| API | ChatGPT-pairwise | 0.658 | 0.448 | 0.655 | 0.686 | 0.297 |
| Autoregressive | CycleReviewer (70B) | 0.678 | 0.574 | - | - | 0.267 |
| Autoregressive | DeepReview (14B) | 0.689 | **0.623** | - | - | 0.405 |
| Regression | NAIPv1 (8B) | 0.545 | 0.472 | 0.605 | 0.629 | 0.183 |
| Regression | **NAIPv2 (ours, 8B)** | **0.706** | 0.609 | **0.782** | **0.771** | **0.432** |

NAIPv2 leads across Acc, AUC, NDCG, and $\rho$. While F1 is slightly lower than the 14B DeepReview, inference speed is orders of magnitude faster (comparable to NAIPv1 at >10 papers/sec vs. 3 mins/paper for autoregressive models).

### Ablation Study
**Learning Paradigm & Complexity**:

| Paradigm | AUC | ρ | Theory Complexity |
| :--- | :--- | :--- | :--- |
| Pointwise (RTS Label) | 0.633 | 0.237 | O(C) |
| Pairwise Concat | 0.720 | 0.351 | O(C log C) |
| **NAIPv2** | **0.782** | **0.432** | **O(C)** |

**Grouping Strategy** (Time+Hier is optimal) and **RTS Signal Superiority**:

| Grouping | AUC | ρ | | Signal | AUC | ρ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| None | 0.739 | 0.400 | | Cites | 0.583 | 0.358 |
| Pub. Time | 0.736 | 0.392 | | Mean | 0.753 | 0.385 |
| Keyword | 0.556 | 0.007 | | Weighted | 0.754 | 0.402 |
| Hier. Cluster | 0.753 | 0.401 | | Median | 0.757 | 0.398 |
| **Time+Hier** | **0.782** | **0.432** | | **RTS** | **0.782** | **0.432** |

### Key Findings
- **Keyword-based grouping degrades to random** ($\rho=0.007$): Too many candidate keywords cause sparse valid pairs, confirming that grouping quality determines the ceiling of pairwise learning.
- **Citations are the poorest signal** (AUC 0.583): Empirically supports the weak correlation between review quality and citation impact (quality $\neq$ future impact).
- **Cross-conference generalization**: On 13,223 unseen NeurIPS submissions, NAIPv2 prediction scores increase monotonically with decision categories (Rejected → Oral), indicating the model learns transferable quality signals rather than ICLR-specific patterns.

## Highlights & Insights
- **"Pairwise training, pointwise inference" is the core innovation**: By using a shared backbone, the debiasing capability of pairwise supervision is "distilled" into a pointwise function. This allows the model to enjoy the robustness of relative ranking while retaining linear inference complexity.
- **RTS transforms "review confidence" from ignored metadata into statistically meaningful variance**, which is more principled than simple weighted averaging.
- **Debiasing is implemented through concrete constraints** (intra-field/year comparisons) rather than abstract objectives, supported by robust hierarchical clustering for domain labeling.

## Limitations & Future Work
- **Dependency on OpenReview data**: NeurIPS data is skewed as rejected reviews are only public if authors volunteer them. This limits the reliability of certain tasks and the model's generalizability to closed-review venues.
- **Rigidity of the RTS Gaussian assumption**: For papers with very few reviews or unreliable confidence scores, the fused signal may be unstable.
- **Performance saturation**: Gains level off after approximately 10k pairs, suggesting that more redundant constraints offer limited help. Selection of high-utility pairs remains an open problem.

## Related Work & Insights
- **Automated Peer Review** (OpenReviewer, DeepReview, CycleReviewer) focuses on LLM autoregressive generation, which is strong in explainability but weak in speed. NAIPv2 is complementary, producing fast numerical scores for retrieval and ranking.
- **Numerical Quality Estimation** (NAIPv1, de Winter) previously either aligned with citation impact (weakly correlated with quality) or ignored confidence and scale inconsistency. NAIPv2 addresses these via RTS and pairwise debiasing.
- **Pairwise Ranking** (Bradley-Terry) is mature in IR/recommendation but underutilized in peer review. NAIPv2 decouples training and inference to achieve the best of both worlds.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Decoupling pairwise training/pointwise inference and RTS probabilistic fusion provides a fresh and effective perspective on translational inconsistency in paper evaluation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Compares against three major classes of methods with extensive ablations. Cross-conference validation on NeurIPS adds significant evidence.
- **Writing Quality**: ⭐⭐⭐⭐ — Logical progression from pain points to solutions. Figures and experiments are well-aligned.
- **Value**: ⭐⭐⭐⭐ — Provides a practical, fast, and debiased scorer for AI-for-Science, alongside a valuable 24k scale dataset (NAIDv2).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] THEMIS: Towards Holistic Evaluation of MLLMs for Scientific Paper Fraud Forensics](themis_towards_holistic_evaluation_of_mllms_for_scientific_paper_fraud_forensics.md)
- [\[ICML 2025\] Sample Efficient Demonstration Selection for In-Context Learning](../../ICML2025/llm_evaluation/sample_efficient_demonstration_selection_for_in-context_learning.md)
- [\[ICLR 2026\] Measuring LLM Novelty as the Frontier of Original and High-Quality Output](measuring_llm_novelty_as_the_frontier_of_original_and_high-quality_output.md)
- [\[ICLR 2026\] DISCO: Diversifying Sample Condensation for Efficient Model Evaluation](disco_diversifying_sample_condensation_for_efficient_model_evaluation.md)
- [\[ICLR 2026\] SparseEval: Efficient Evaluation of Large Language Models by Sparse Optimization](sparseeval_efficient_evaluation_of_large_language_models_by_sparse_optimization.md)

</div>

<!-- RELATED:END -->
