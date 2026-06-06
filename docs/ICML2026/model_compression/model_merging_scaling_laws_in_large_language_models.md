---
title: >-
  [Paper Note] Model Merging Scaling Laws in Large Language Models
description: >-
  [ICML 2026][Model Compression][Model Merging] The authors empirically identify a dual-axis power law of the form $L=L_*+BN^{-\beta}+A_0 N^{-\gamma}/(k+b)$ using 10…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Model Merging"
  - "Scaling Law"
  - "Power Law"
  - "Task Arithmetic"
  - "TIES/DARE"
date: 2026-05-08
content_hash: 9947e33c530bd6ae
---

# Model Merging Scaling Laws in Large Language Models

**Conference**: ICML 2026  
**arXiv**: [2509.24244](https://arxiv.org/abs/2509.24244)  
**Code**: https://github.com/InfiXAI/Merging-Scaling-Law (Available)  
**Area**: LLM Pre-training / Model Merging / Scaling Law  
**Keywords**: Model Merging, Scaling Law, Power Law, Task Arithmetic, TIES/DARE

## TL;DR
The authors empirically identify a dual-axis power law of the form $L=L_*+BN^{-\beta}+A_0 N^{-\gamma}/(k+b)$ using 10,866 merged models. The base scale $N$ determines the performance floor, while the number of experts $k$ determines the tail. Four mainstream merging methods (Average, TA, TIES, DARE) share the same curve, transforming the decision of "how many experts to merge" and "when to stop" into a predictable and budget-aware engineering problem.

## Background & Motivation
**Background**: Model merging has emerged as a low-cost "expert integration" paradigm following multi-task SFT. Linear weighting (Model Soups, Task Arithmetic) and versions with preprocessing (TIES, DARE) are widely used in scenarios involving LLMs and LoRA adapters.

**Limitations of Prior Work**: Merging remains largely heuristic-based—testing different subsets, orders, and normalization coefficients. This is computationally expensive and lacks guidance from scaling laws similar to those in pre-training. Given a target loss, it is currently impossible to pre-determine the required number of experts or whether scaling the base model versus adding another expert is more cost-effective.

**Key Challenge**: The gain curve of merging is clearly non-linear but exhibits a certain regularity (steep early gains followed by saturation). Without an analytical form to describe this curve, engineering practice relies on exhaustive search, wasting GPU resources.

**Goal**: (1) Find a compact formula that characterizes the impact of both $N$ (base parameter count) and $k$ (number of merged experts); (2) Prove its validity across different merging algorithms, backbones, in-domain, and cross-domain evaluations; (3) Provide a practical method to extrapolate the entire curve by measuring only three points.

**Key Insight**: Merging is viewed as "calculating the equal-weight average of multiple task vectors." Under a second-order Taylor expansion, the variance of equal-weight averaging shrinks at a rate of $1/k$. As variance enters the loss via the Hessian, it manifests as the $A(N)/k$ term. Consequently, the authors expect a "floor + 1/k tail" structure, which is validated through large-scale empirical tests.

**Core Idea**: A unified power law for floor $+ 1/(k+b)$ tail is used to describe the CE curves of all merging methods. This formula integrates base scale and expert count, making merging a budget-aware and predictable process.

## Method

### Overall Architecture
The authors fine-tune nine domain experts (algebra, analysis, geometry, discrete, number_theory, code, chemistry, physics, biology) on the Qwen2.5 series (0.5B to 72B), covering both in-domain and cross-domain evaluations. For each $(N,k)$ combination, they traverse or uniformly sample all $\binom{9}{k}$ expert subsets. Four merging algorithms (Average, TA, TIES, DARE) are applied to synthesize models, and token-level CE is measured, resulting in a grid of data from 10,866 merged models. A weighted non-linear least squares fit is applied to the curve $\mathbb{E}[L\mid N,k]=L_\infty(N)+A(N)/(k+b)$, where $L_\infty(N)=L_*+BN^{-\beta}$ and $A(N)=A_0 N^{-\gamma}$. The model is validated using $R^2$ and residual analysis.

### Key Designs

1.  **Unified floor+tail Scaling Law**:
    - **Function**: Characterizes the concurrent impact of base scale and expert count on merging loss.
    - **Mechanism**: $\mathbb{E}[L\mid N,k]=L_*+BN^{-\beta}+\frac{A_0 N^{-\gamma}}{k+b}$, where the floor term $L_*+BN^{-\beta}$ decreases monotonically with $N$, and the tail term $A_0 N^{-\gamma}/(k+b)$ decays as a reciprocal of $k$. Fitting uses weights $\propto k$ to stabilize early $k$ noise. All methods achieve $R^2 > 0.98$ across all slices.
    - **Design Motivation**: Combines the observations that "larger bases merge better" and "diminishing returns with more experts" into a single expression, allowing direct ROI comparison between scaling the base and adding experts.

2.  **Theory Deriving 1/k Tail from Second-order Taylor Expansion**:
    - **Function**: Explains why all merging algorithms exhibit a $1/k$ tail under equal-weight normalization.
    - **Mechanism**: Each task vector is denoted as $v_i$. After equal-weight merging, the perturbation mean is $c\mu$ and the covariance is $\Sigma/k$. The second-order Taylor expansion gives $\mathbb{E}[L]=L(\theta_0)+cg^\top\mu+\frac{1}{2}c^2\mu^\top H\mu+\frac{c^2}{2k}\mathrm{Tr}(H\Sigma)+\mathcal{O}(k^{-3/2})$. The first three terms form $L_\infty(N)$, and the last term represents $A(N)/k$. A corollary shows that standard deviation between subsets shrinks by $1/\sqrt{k}$. Preprocessing algorithms like TIES/DARE are seen as modifications to $\Psi(v)$, which do not alter the leading-order form.
    - **Design Motivation**: Provides a theoretical bridge rather than just empirical fitting, explaining why diverse implementations like TIES and DARE converge to the same curve.

3.  **Three-point Fitting + Budget Algorithm for $k^*$**:
    - **Function**: Extrapolates the full $k$-curve using only three points $k \in \{1, 2, 4\}$ and recommends the "most cost-effective expert count" $k^*$.
    - **Mechanism**: Since the formula has three degrees of freedom ($L_\infty, A, b$), three points are theoretically sufficient. Empirical results show three-point fitting can recover the full 9-point curve, consistently estimating $k^*$ at $5 \sim 6$, corresponding to the elbow position where $\Delta_k \approx A/[(k+b)(k+1+b)] \sim k^{-2}$.
    - **Design Motivation**: Merging the full $k$-grid is expensive in real-world scenarios. The three-point method makes "measure a small batch, then decide budget" a feasible workflow.

### Loss & Training
Ours does not introduce new training losses. All data points are derived from frozen bases and independently fine-tuned experts. Evaluation uses token-level cross-entropy on 30M held-out tokens. Merging coefficients use equal-weight normalization $\alpha_{i,k}=c/k$. Fitting employs weighted non-linear least squares with weights $\propto k$ to suppress high variance at small $k$.

## Key Experimental Results

### Main Results

| Setting | Model Scale $N$ | Avg. Domain CE at $k=9$ | Reduction vs. 0.5B |
| :--- | :--- | :--- | :--- |
| In-domain | 0.5B | 0.739 | — |
| In-domain | 7B | ~0.52 | ~30% |
| In-domain | 32B | 0.430 | 41.9% |
| Cross-domain | 0.5B $\to$ 32B | Synchronous Shift | Floor and tail both shrink |
| Fitting Quality | All points | $R^2 > 0.98$ | Uniform residuals for floor/tail |

### Ablation Study

| Configuration | Key Observation | Description |
| :--- | :--- | :--- |
| Average / TA / TIES / DARE | $R^2 > 0.98$ for the same formula | Methodological differences are absorbed into $L_\infty, A, b$ constants. |
| Candidate Pool $M=9 \to 8 \to 7$ | Floor remains stable, tail reduction slows | Diversity primarily lowers the tail rather than the floor. |
| Three-point $k \in \{1, 2, 4\}$ fitting | Extrapolation error is minimal | Sufficient for budget decision support. |
| Different donor orders (DARE) | Whisker length shrinks by ~83% at $k=8$ | Order sensitivity shrinks by $1/(k+b)$. |
| Cross-backbone (LLaMA-3.2 3B / LLaMA-3 8B) | Same 1/k tail | Formula form is transferable. |

### Key Findings
- "Larger bases merge better" is quantified: At $k=9$, the 32B model reduces CE by 41.9% compared to 0.5B, with both floor and tail shrinking simultaneously. This implies larger bases offer lower asymptotic performance and require fewer experts.
- The elbow typically appears at $k \approx 5 \sim 6$: Achieving 85% of gains requires 5 experts; 90% requires 6. Beyond this, additional experts provide marginal utility.
- Methodological differences diminish at scale: For $N=32B$ and $k \approx 8$, the gap in mean CE between Avg/TA/TIES/DARE is $\lesssim 2\%$. Merge-to-merge variance shrinks toward a common floor at a rate of $\sim 1/k$.
- Order sensitivity also decays following $1/(k+b)$; optimizing the merge order is practically meaningless for $k \geq 6$.

## Highlights & Insights
- Validates "folk wisdom" with rigorous curves ($R^2 > 0.98$) using 10,866 merged models. The scale and systematic nature of this study exceed previous merging papers, serving as the most authoritative empirical evidence in the field.
- The decoupling of floor and tail is highly practical: The relative magnitude of $A/L$ allows one to instantly judge the ROI of "adding another expert" versus "upgrading the base model scale."
- The three-point fitting method upgrades scaling laws from "post-hoc summaries" to "predictive tools," allowing the elbow to be identified without running all $k$. This "measurement-extrapolation" logic can be transferred to other compositionality studies (e.g., number of RAG retrieval sources, ensemble sizes).

## Limitations & Future Work
- The formula only covers equal-weighted merging. For non-equal or learned weights (e.g., routing-based or optimized merges), it only explains the leading order, with differences absorbed as finite-$k$ deviations.
- Expert capacity is treated as a latent variable inside $A(N)$. Dimensions such as LoRA rank or fine-tuning token counts are not explicitly modeled, though the paper acknowledges this as a natural extension.
- Evaluations are limited to cross-entropy; there is still a gap between CE and downstream task accuracy. Whether the elbow is consistent for long-tail tasks like "coding/math" requires further validation.
- While the 9 domains are diverse, they are all within the Mixture-of-Thoughts/OpenScience datasets. Generality for truly heterogeneous scenarios (e.g., multilingual, multimodal, safety alignment) remains to be tested.

## Related Work & Insights
- **vs Kaplan/Chinchilla Scaling Laws**: While those characterize the relationship between $(N, D, C)$ and loss, this work adds the compositional dimension "number of experts $k$" and shows $N$ and $k$ are decouplable axes.
- **vs Yadav et al. (2024) Empirical Studies**: The latter noted empirically that differences between methods decrease as experts increase; this work explains that observation through a unified formula where common $L_\infty(N)$ dominates large $k$ and $A(N)/(k+b)$ dominates small $k$.
- **vs TIES/DARE Specific Merging Algorithms**: Ours does not compete with them but rather places them in the same framework, demonstrating that these preprocessing steps merely adjust task vector means/covariances without changing the power-law skeletal structure.

## Rating
- Novelty: ⭐⭐⭐⭐ First to provide $(N,k)$ dual-axis merging scaling laws with a theoretically provable first-order derivation. The formula is simple, but the approach is a natural extension of the scaling law lineage.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 10,866 merged models, 9 domains, 7 scales, 4 methods, and cross-backbone validation. This is likely the largest-scale study in existing merging literature.
- Writing Quality: ⭐⭐⭐⭐ Formulas and figures are clear, explaining the physical significance of floor/tail effectively; however, the in-domain/cross-domain sections are slightly repetitive.
- Value: ⭐⭐⭐⭐⭐ Provides a practical "three-point fitting $\to$ budget decision" workflow with immediate engineering significance for industry-scale merging, LoRA repository management, and expert routing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training](decouple_searching_from_training_scaling_data_mixing_via_model_merging_for_large.md)
- [\[ICML 2026\] LLMs as Noisy Channels: A Shannon Perspective on Model Capacity and Scaling Laws](llms_as_noisy_channels_a_shannon_perspective_on_model_capacity_and_scaling_laws.md)
- [\[ICML 2026\] FRISM: Fine-Grained Reasoning Injection via Subspace-Level Model Merging for Vision–Language Models](frism_fine-grained_reasoning_injection_via_subspace-level_model_merging_for_visi.md)
- [\[ICML 2026\] Saliency-Aware Model Merging](saliency-aware_model_merging.md)
- [\[ICML 2026\] NanoQuant: Efficient Sub-1-Bit Quantization of Large Language Models](nanoquant_efficient_sub-1-bit_quantization_of_large_language_models.md)

</div>

<!-- RELATED:END -->
