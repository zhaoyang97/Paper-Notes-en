---
title: >-
  [Paper Note] Active Learning with Low-Rank Structure for Data Selection
description: >-
  [ICML 2026][Learning Theory][low-rank approximation] Addressing the mismatch where existing coreset methods assume geometric clustering while modern datasets exhibit global algebraic (low-rank) structures, this paper proposes a data selection framework based on low-rank approximation and residual sensitivity sampling. Using a weighted subset of size $\tilde{O}(k+1/\varepsilon^2)$, the method approximates the full average loss to a $(1\pm\varepsilon)$ relative error (with an a…
tags:
  - "ICML 2026"
  - "Learning Theory"
  - "Data Selection"
  - "Coreset"
  - "low-rank approximation"
  - "sensitivity sampling"
  - "active learning"
date: 2026-05-08
content_hash: 329d93c24f4f0082
---

# Active Learning with Low-Rank Structure for Data Selection

**Conference**: ICML 2026  
**arXiv**: [2606.16045](https://arxiv.org/abs/2606.16045)  
**Code**: Not released  
**Area**: Learning Theory / Data Selection / Coreset  
**Keywords**: Data selection, coreset, low-rank approximation, sensitivity sampling, active learning

## TL;DR
Addressing the mismatch where existing coreset methods assume geometric clustering while modern datasets exhibit global algebraic (low-rank) structures, this paper proposes a data selection framework based on low-rank approximation and residual sensitivity sampling. Using a weighted subset of size $\tilde{O}(k+1/\varepsilon^2)$, the method approximates the full average loss to a $(1\pm\varepsilon)$ relative error (with an additive term proportional to the optimal rank-$k$ approximation cost $\Phi_k$). It outperforms uniform and cluster-based sampling on tabular data and Llama3-8B / Qwen2.5-3B fine-tuning.

## Background & Motivation
**Background**: In the era of foundation models, training costs are prohibitive, leading to a consensus that training on a carefully selected high-quality subset is sufficient. Sener & Savarese (ICLR 2018, [SS18]) reformulated active learning as a **coreset selection** problem, using $k$-center clustering heuristics to pick representative points. Axiotis et al. (ICML 2024, [ACH+24]) further utilized $k$-means clustering with sensitivity sampling to construct coresets. The core principle is that if the average loss (or gradient) on the coreset approximates the full dataset, training on the subset is approximately equivalent to full-set training.

**Limitations of Prior Work**: These methods assume the dataset possesses an **intrinsic geometric structure effectively captured by clustering**. However, the authors point out that many modern datasets are not "locally clustered" but instead have a **global algebraic structure** better characterized by low-rank approximation or Principal Component Analysis (PCA). In high-dimensional data, clustering focuses on **local groupings** and misses the **principal variance directions**, causing selected subsets to discard highly informative components.

**Key Challenge**: "Clustering measures **distance** between points" vs. "Information is concentrated in a few **principal (spectral) directions**." Clustering implicitly assumes point-to-point distance is more critical than direction. Work like LoRA demonstrates that low-rank updates capture the most important components in parameter space; naively clustering embeddings may ignore these critical directions.

**Goal**: Construct a **small weighted subset with theoretical guarantees** under the assumption of an approximately low-rank dataset, while minimizing the number of queries to the expensive model loss $\ell$.

**Key Insight / Core Idea**: Combine accurate but expensive loss evaluations on a small subset with fast-computable sketches to capture global dataset directions. Specifically, **construct a low-rank sketch of the dataset to estimate leverage scores and sample rows (data points) proportionally**, ensuring the subset reflects principal variance directions rather than just geometric diversity.

## Method

### Overall Architecture
The approach formalizes data selection as a **row subset selection + loss-preserving coreset construction** problem. Given a data matrix $D\in\mathbb{R}^{n\times m}$, the goal is to choose a weighted subset $S\subseteq D$ with weights $w(x)$ to minimize $\Delta(S)=\big|\sum_{x\in D}\ell(x)-\sum_{x\in S}w(x)\ell(x)\big|$, while minimizing queries to $\ell$. The pipeline is: identify a $k$-dimensional subspace $V$ (e.g., spanned by top-$k$ singular vectors) $\rightarrow$ decompose each point $x$ into a projection $v(x)$ in $V$ and an orthogonal residual $r(x)$ $\rightarrow$ compute a **sensitivity score** $\sigma(x)$ under a "low-rank loss regularity assumption" $\rightarrow$ sample proportionally to $\sigma(x)$ $\rightarrow$ apply weights $1/(s\,p(x))$ for an unbiased estimate $\rightarrow$ provide high-probability approximation guarantees via concentration inequalities.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Dataset D + Pre-trained Embeddings"] --> B["Low-Rank Loss Regularity Assumption<br/>Loss decomposable along principal directions + residual penalty"]
    B --> C["Low-Rank Subspace V<br/>SVD / Leverage-score row selection"]
    C --> D["Residual Sensitivity Score σ(x)<br/>Projection loss + Residual norm"]
    D -->|Sample p(x)=σ(x)/Σσ| E["Weighted Subset S<br/>Weights 1/(s·p(x))"]
    E --> F["Train/Fine-tune on S<br/>≈ Full training"]
```

### Key Designs

**1. Low-Rank Loss Regularity Assumption: Formalizing "Approximately Low-Rank Loss"**

While clustering assumes data clusters, this work requires an assumption (Assumption 2.1) characterizing that loss is decomposable along principal directions with controllable orthogonal contributions. Decomposing any point $y$ over subspace $V=\mathrm{span}\{v_1,\dots,v_k\}$ as $y=\sum_i\alpha_i v_1 + r(y)$, where $\alpha_i=\langle y,v_i\rangle$ and $r(y)=\mathrm{Proj}_{V^\perp}(y)$, it is assumed constants $\lambda,\gamma>0$ exist such that:

$$|\ell(y)-\ell(v(y))|\le\lambda\|r(y)\|_2^2,\qquad |\ell(v(y))-\textstyle\sum_i\alpha_i^2\ell(v_i)|\le\gamma\sum_i|\alpha_i^2+1|\,\ell(v_i).$$

The first is a smoothness condition: the loss deviation from $V$ grows at most **quadratically** with the distance to $V$ (satisfied by Lipschitz gradients/bounded curvature). The second suggests loss in the subspace is approximated by a weighted combination of losses along basis directions; errors are absorbed into $\gamma$ without requiring the loss to be additive or strictly quadratic. The crucial regime $\lambda\gg\gamma$ targets scenarios where orthogonal information is sparse, common in LoRA-based LLM fine-tuning.

**2. Residual Sensitivity Score: Probability Proportional to "True Loss Contribution"**

Cluster-based sensitivity sampling scores points by geometric distance to centroids, which can bias toward outliers. This method replaces this with a **sensitivity score** based on low-rank structure (Algorithm 1, Line 7):

$$\sigma(x)=(\gamma+1)\big(\alpha_1^2\ell(v_1)+\dots+\alpha_t^2\ell(v_t)\big)+\gamma k\xi+\lambda\|r(x)\|_2^2,$$

consisting of a **projection loss term** (contribution along principal directions weighted by landmark losses $\ell(v_i)$), a **basis loss term** $\gamma k\xi$, and a **residual loss term** $\lambda\|r(x)\|_2^2$ (deviation from the principal subspace). Points with high contributions—either high loss on principal directions or large residuals—have a higher selection probability. The resulting subset reflects **spectral axes + real loss** rather than purely geometric diversity.

**3. Two Coreset Guarantees: Sketches vs. Real Row Selection**

Theorem 2.2 provides a general guarantee: a weighted subset $S$ of size $s=\mathcal{O}(1/\varepsilon^2)$ satisfies with probability $\ge0.9$:

$$\Big|\sum_{x\in D}\ell(x)-\sum_{x\in S}w(x)\ell(x)\Big|\le\varepsilon\cdot\Big(\sum_{x\in D}\ell(x)+\gamma\|D\|_F^2+\gamma k|D|\max\ell+2\lambda\,\Phi_k(D)\Big),$$

where $\Phi_k(D)=\min_{\mathrm{rank}(D_k)\le k}\|D-D_k\|_F^2=\sum_{i>k}\sigma_i^2$ is the optimal rank-$k$ approximation cost. As data becomes more low-rank, $\Phi_k$ decreases, and the coreset becomes more accurate. Theorem 2.3 extends this to **row subset selection**, ensuring $S \subseteq D$ consists of **actual data points** with similar approximation bounds using bicriteria algorithms.

### Loss & Training
The method does not introduce a new training loss but performs selection based on existing model loss $\ell(x,y;\mathcal{A})$. The expected loss of batch selection is decomposed into generalization error, training error, and coreset loss. Low-rank sensitivity sampling targets the **coreset loss** using current model parameters. In practice, the subspace dimension/landmark count defaults to 20% of the dataset. LLM experiments utilize BERT embeddings and leverage-score selected landmarks, fitting landmark loss combinations in embedding space via Kernel Ridge Regression to estimate $\alpha$.

## Key Experimental Results

### Main Results
**Tabular Data (Default of Credit Card Clients)**: Comparison of random, clustering, and low-rank sensitivity sampling across coreset sizes.

| Task / Subset Size | Metric | Low-rank Sampling (Ours) | Clustering | Random |
|--------|------|------|------|------|
| Coreset Error @ $s{=}1000$ | $\|\sum w\|x\|^2-L_{\text{true}}\|$ | Lowest (10x < Random, 50% < Cluster) | Mid | Highest |
| Downstream Accuracy @ $s{=}5000$ | Test Accuracy | ~74% | ~70% | ~67% |

**LLM Fine-tuning (Llama3-8B-Instruct, BERT embeddings, $k$ fixed at 25%)**: Average validation accuracy across GSM8k, ViGGO, and SQL tasks.

| Method | 25% Avg | 12.5% Avg | 6.25% Avg |
|------|---------|-----------|-----------|
| Uniform Sampling | 76.5 | 69.2 | 52.0 |
| K-Center | 74.9 | 63.9 | 52.3 |
| Graph Cut | 74.0 | 65.8 | 45.7 |
| Clustering-based SS [ACH+24] | 77.5 | **71.0** | 54.6 |
| Low-rank SS (Ours) | **77.6** | 70.4 | **54.9** |
| Full (100%) | 81.1 | — | — |

Low-rank SS consistently outperforms uniform sampling and is generally superior to clustering-based sampling. On ViGGO @ 25%, it achieves 88.3 (Clustering 86.6, Full 94.0). Selection requires forward passes on only 20% of data, representing ~6.67% of full training runtime.

### Ablation Study

| Configuration / Analysis | Key Findings |
|------|---------|
| Structure Diagnosis (GSM8k) | Low-rank error < Clustering error; GSM8k matches low-rank assumptions better. |
| Heatmap of $\Delta(S)$ ($k,\lambda$) | Low-rank preserves lower error across $k, \lambda$ grids. |
| Objective / Embedding Swap | Consistent leads using gradient norms or GTR-base embeddings; more robust to $\lambda$. |
| Model Swap (Qwen2.5-3B) | Ours vs. Clustering SS: +4.34 / +3.08 points at 25%/12.5% rates. |

## Highlights & Insights
- **Theorizing the Clustering vs. Low-Rank Choice**: Instead of general claims, the author provides Assumption 2.1 as a verifiable condition, allowing users to diagnose dataset structure before selecting the method.
- **Spectral Sensitivity over Geometric Distance**: Shifting from "covering space" to "preserving principal variance" is a directional correction to the [SS18]/[ACH+24] paradigm, naturally supporting non-uniform weighted sampling.
- **Theory to Engineering**: Size $\mathcal{O}(1/\varepsilon^2)$ is decoupled from data dimension and total sample count. The row subset selection variant ensures the use of real data points, making it highly practical for labeling tasks.

## Limitations & Future Work
- **Dependency on Assumption 2.1**: In datasets with high-rank noise or significant outliers, the additive term $2\lambda\Phi_k(D)$ grows, degrading guarantees.
- **Moderate Gains**: In LLM fine-tuning, the lead over clustering-based methods is often within a few percentage points rather than a total sweep.
- **Hyperparameter $\lambda, \gamma$**: Sensitivity to $\lambda$ requires tuning, and while setting $\gamma=0$ worked for LLMs, it introduces overhead for other tasks.
- **Future Directions**: Automating the diagnosis of $\lambda$ and $k$, hybridizing low-rank and geometric methods, and extending guarantees to post-retraining loss.

## Related Work & Insights
- **vs [SS18] (k-center coreset)**: [SS18] pursues geometric coverage; this work uses low-rank sketches and spectral sensitivity, which is more robust for high-dimensional low-rank data.
- **vs [ACH+24] (Clustering SS)**: Both use sensitivity sampling, but [ACH+24] relies on $k$-means distances. This work is superior when data matches spectral (low-rank) rather than local (cluster) patterns.
- **vs LoRA**: Borrowing the insight that parameter updates are low-rank, this work maps that observation to the **data space**, arguing that spectral selection is more logical for structured data.

## Rating
- Novelty: ⭐⭐⭐⭐ Shifting from geometric to spectral perspective with verifiable assumptions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Good coverage across tabular and LLM tasks, though gains are sometimes moderate.
- Writing Quality: ⭐⭐⭐⭐ Complete chain from assumption to theorem to algorithm.
- Value: ⭐⭐⭐⭐ Low-overhead, theoretically grounded data selection for foundation model fine-tuning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Catastrophic Forgetting is Low-Rank: A Function-Space Theory for Continual Adaptation](catastrophic_forgetting_is_low-rank_a_function-space_theory_for_continual_adapta.md)
- [\[ICLR 2026\] High-dimensional Analysis of Synthetic Data Selection](../../ICLR2026/learning_theory/high-dimensional_analysis_of_synthetic_data_selection.md)
- [\[ICLR 2026\] Optimizing Data Augmentation through Bayesian Model Selection](../../ICLR2026/learning_theory/optimizing_data_augmentation_through_bayesian_model_selection.md)
- [\[ICLR 2026\] Saddle-To-Saddle Dynamics in Deep ReLU Networks: Low-Rank Bias in the First Saddle Escape](../../ICLR2026/learning_theory/saddle-to-saddle_dynamics_in_deep_relu_networks_low-rank_bias_in_the_first_saddl.md)
- [\[ICML 2026\] When Sample Selection Bias Precipitates Model Collapse](when_sample_selection_bias_precipitates_model_collapse.md)

</div>

<!-- RELATED:END -->
