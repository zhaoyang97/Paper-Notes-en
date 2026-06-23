---
title: >-
  [Paper Note] Larger Datasets Can Be Repeated More: A Theoretical Analysis of Multi-Epoch Scaling in Linear Regression
description: >-
  [ICLR 2026][learning_theory][SGD] Under the analytically tractable setting of linear regression with multi-epoch SGD, this paper defines and characterizes the "effective reuse rate" $E(K,N)$—the ratio of equivalent one-pass data size to the actual $N$ samples trained over $K$ epochs. It proves that $E(K,N)$ depends not only on the number of epochs $K$
tags:
  - ICLR 2026
  - learning_theory
  - SGD
date: 2026-05-08
content_hash: 68774e892edfe4c4
---
# Larger Datasets Can Be Repeated More: A Theoretical Analysis of Multi-Epoch Scaling in Linear Regression

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=0CXjpAxHUE](https://openreview.net/forum?id=0CXjpAxHUE)  
**Code**: To be confirmed  
**Area**: Learning Theory / Data Scaling Laws  
**Keywords**: Scaling laws, Multi-epoch training, Data reuse, SGD, Linear regression, Strongly convex, Zipf distribution  

## TL;DR
Under the analytically tractable setting of linear regression with multi-epoch SGD, this paper defines and characterizes the "effective reuse rate" $E(K,N)$—the ratio of equivalent one-pass data size to the actual $N$ samples trained over $K$ epochs. It proves that $E(K,N)$ depends not only on the number of epochs $K$ but also increases with the dataset size $N$ (saturation point at $\Theta(\log N)$ for strongly convex data and powers of $N$ for Zipf data). This refines the implicit assumption in Muennighoff et al. (2023) that $E(K,N)\approx K$ is independent of $N$, revealing that "**larger datasets can be repeated more.**"

## Background & Motivation

**Background**: Scaling laws (e.g., Kaplan, Chinchilla) serve as the core framework for characterizing LLM pre-training performance, yet they are almost entirely established on the "one-pass" paradigm—where each data point is used at most once. As data demand grows from under 10B tokens in GPT-2 to 36T tokens in Qwen3, public high-quality data is projected to be exhausted by 2028. Consequently, multi-epoch training on the same dataset has become an unavoidable practical strategy.

**Limitations of Prior Work**: Theoretical understanding of multi-epoch effects on scaling laws is sparse. Muennighoff et al. (2023) provided an empirical approximation $N'(K,N)=\big[1+R^*(1-e^{-(K-1)/R^*})\big]\cdot N$ (with fit constant $R^*\approx15.39$), suggesting that "repeated data in the first 4 epochs is nearly as good as fresh data." However, this formula has two flaws: a visible gap remains between it and true curves, and it implies $E(K,N)=N'/N$ **only depends on $K$ and is independent of $N$**, meaning the benefit of $K$ repetitions is identical regardless of dataset size.

**Key Challenge**: The question of "when the marginal benefit of repeated data diminishes" has been simplified by empirical formulas into a constant curve regarding $K$. Intuitively, larger datasets should sustain more repetitions before saturating. Rigorous analysis of whether $N$ enters the characterization of effective reuse rate is lacking.

**Goal**: In a simple yet precisely analyzable setting (linear regression with non-asymptotic SGD), the paper aims to analytically compute the ratio $E(K,N)$ ("training $K$ epochs on $N$ samples $\approx$ training once on how many fresh samples") and reveal its dual dependency on both $K$ and $N$.

**Key Insight**: **(Data scale enters the reuse rate)** The paper defines the effective reuse rate as $E(K,N):=\frac1N\min\{N'\ge0:\bar R^*(1,N')\le \bar R^*(K,N)\}$ (the relative multiplier of one-pass samples needed to match $K$-epoch performance under the optimal learning rate). It proves a phase transition from "linear gain" to "saturation," where the **transition point shifts later as $N$ increases**.

## Method

### Overall Architecture
The paper is purely theoretical. The main sequence involves: establishing a standard linear regression setting with multi-shuffle SGD; defining $E(K,N)$ under the "optimal learning rate" criterion; deriving asymptotic expansions for $\bar R^*(K,N)$ (optimal excess risk) for both small-$K$ and large-$K$ regimes under two typical data spectra (strongly convex and Zipf power-law); solving for $E(K,N)$; and finally validating the theoretical phase transitions using synthetic data and LLM pre-training experiments.

```mermaid
flowchart LR
    A[Linear Regression + K-epoch Shuffled SGD<br/>Initial w0=0, Learning Rate η] --> B[Define Optimal Excess Risk<br/>R*(K,N)=min_η E[R(w_KN)]]
    B --> C[Define Effective Reuse Rate<br/>E(K,N)=N'/N]
    C --> D1[Strongly Convex Spectrum<br/>λd≥μ]
    C --> D2[Zipf Power-law Spectrum<br/>pi∝i^-α]
    D1 --> E1["R* Small K: ~log(KN)/KN<br/>Large K: ~1/N"]
    D2 --> E2["R* Small K: (KN)^-(a-1)/a<br/>Large K: N^-(a-1)/(a-b)"]
    E1 --> F1["E(K,N): K or Θ(log N)"]
    E2 --> F2["E(K,N): K or Θ(N^b/(a-b))"]
    F1 --> G[Saturation point shifts later as N grows<br/>→ Larger data allows more repeats]
    F2 --> G
```

### Key Designs

**1. Effective Reuse Rate $E(K,N)$**: This translates "repeated vs. fresh" into a comparable multiplier. The authors define excess risk $R(w):=L(w)-\frac12\sigma^2$ (subtracting irreducible noise) and denote the optimal excess risk for $K$-epoch $N$-sample SGD as $\bar R^*(K,N):=\min_{\eta\in(0,1/D^2]}\mathbb E_{w\sim W_{K,N,\eta}}[R(w)]$. By allowing both one-pass and multi-epoch training to use their respective optimal learning rates, confounding factors from learning rate selection are eliminated. Thus, $E(K,N)=\frac1N\min\{N':\bar R^*(1,N')\le\bar R^*(K,N)\}$ becomes a clean, analytically quantifiable measure.

**2. Dual-Interval Scaling and Phase Transitions under Strong Convexity**: Under the assumption $\lambda_d\ge\mu$, parameter prior $w^*_i\ne0$, and a feasible epoch range $K=O(N^{0.1})$, the paper characterizes the optimal excess risk (Theorem 4.1):

$$\bar R^*(K,N)=\begin{cases}\dfrac{\sigma^2\mathrm{tr}(H)}{8\lambda_d}\,(1+o_N(1))\cdot\dfrac{\log(KN)}{KN}, & K=o(\log N)\\[2mm]\dfrac{\sigma^2 d}{2}\,(1+o_N(1))\cdot\dfrac{1}{N}, & K=\omega(\log N).\end{cases}$$

When $K\ll\log N$, risk decays as $\Theta(\log T/T)$ (where $T=KN$), implying each extra epoch is roughly as good as an extra pass over fresh data. When $K\gg\log N$, risk degrades to $\Theta(1/N)$, becoming independent of $K$. This lead to $E(K,N)$ (Theorem 4.2): $E(K,N)=K(1+o(1))$ for small $K$ (**Effective Reuse Zone**) and $E(K,N)=\frac{\mathrm{tr}(H)}{4\lambda_d d}(1+o_N(1))\cdot\log N$ for large $K$ (**Limited Reuse Zone**). The phase transition occurs as $\lim_{N\to\infty}K/\log N$ moves from $0$ to $\infty$, with saturation capped at $\Theta(\log N)$.

**3. "Larger Datasets Can Be Repeated More"**: This is the core insight. For a fixed data distribution, the maximum number of epochs that stay within the "Effective Reuse Zone" increases with $N$ because the transition occurs at $K\sim\log N$. Consequently, multi-epoch training on $N$ samples can approach the performance of one-pass training on $\Theta(N\log N)$ samples, which is **super-linear in $N$**. This contradicts the Muennighoff assumption that the "effective number of epochs" is a constant scale-invariant property.

**4. Zipf Power-law Spectrum**: Natural data often exhibits long tails. Under power-law assumptions ($p_i=ci^{-(a-b)}$, $\Lambda_i=i^{-b}$, $a-b>1$), $\bar R^*(K,N)$ decays as $(KN)^{-(a-1)/a}$ in the small-$K$ zone and as $N^{-(a-1)/(a-b)}$ in the large-$K$ zone (Theorem 5.1). Consequently, the saturation point for $E(K,N)$ becomes $\Theta(N^{b/(a-b)})$ (Theorem 5.2). The "linear gain $\to$ saturation" structure persists, but the **saturation point is a power of $N$ rather than logarithmic**, determined by the decay rates of the Hessian eigenvalues and parameter norms.

**5. Technical Byproduct: Optimal Learning Rate for Multi-epoch SGD**: To calculate risk to $o(1)$ relative error, the authors derive the optimal learning rate for multi-epoch SGD in linear regression (Lemma 4.4) and corresponding approximation formulas for expected excess risk (Lemma G.1). These results hold independent value. Experiments use $\eta\propto\log(KN)/KN$ with a grid-searched coefficient $c^*$ to match the theory.

## Key Experimental Results

### Main Results (LLM Pre-training Validation, Section 6.3)

| Setting | Configuration |
|------|------|
| Model | 0.3B parameters, adapted from Qwen2.5-0.5B architecture |
| Data | DCLM subset, 200B tokens; fresh data sizes 0.2B/0.5B/0.8B/1.0B/2B, each trained for 100 epochs |
| Control | 200B fresh tokens in one pass |
| Learning Rate | Constant schedule (to align with theory and avoid scheduler confounding) |
| Saturation Criterion | $E(K,N)=\lambda K$, setting $\lambda=0.75$ |

| Phenomenon | Observation | Theoretical Correspondence |
|------|----------|----------|
| Small $K$ ($\lesssim5$) | $E(K,N)$ grows approximately linearly with $K$ | Reproduces Muennighoff's $K\le4$ result |
| Large $K$ | $E(K,N)$ increases and saturates at higher levels as fresh data size $N$ increases | Refutes scale-invariant epoch assumption |
| Saturation $K(\lambda,N)$ vs $N$ | Linear fit $y=0.80\log x+5.21$, $r=0.97$ | Confirms $\Theta(\log N)$ saturation for strongly convex data |

### Ablation Study (Section 6.1–6.2)

| Setting | Key Results |
|------|----------|
| Strongly Convex ($d=100$, $\sigma=0.1$, $\eta\propto\log(KN)/KN$) | $E(K,N)$ grows linearly before saturating at $E\approx K$ relative to $\log N$, matching Theorem 4.2 |
| Zipf Power-law ($d=10^5$, $a=4.5$, $b=1$) | Power-law fit exponent in large-$K$ zone is $0.279\approx b/(a-b)=2/7$, matching Theorem 5.2 |

### Key Findings
- The effective reuse rate exhibits a clear phase transition from $E\approx K$ to saturation, with the transition point shifting later as $N$ increases.
- Saturation is $\Theta(\log N)$ for strongly convex data and $\Theta(N^{b/(a-b)})$ for Zipf data; these theoretical scalings are validated across synthetic and LLM experiments.
- Practical Implication: Multi-epoch training on $N$ data points can approximate the performance of $\Theta(N\log N)$ fresh data points, providing direct guidance for budget allocation in the data-scarcity era.

## Highlights & Insights
- **Quantification of Empirical Questions**: $E(K,N)$ is defined cleanly (using optimal learning rates for both sides), providing the first rigorous non-asymptotic characterization of "repeated vs. fresh" data.
- **Identification of $N$-dependency**: The core insight that "larger datasets can be repeated more" is both counter-intuitive and practically significant, correcting widely cited assumptions in data-constrained scaling laws.
- **Theory-Experiment Alignment**: The fit of $0.80\log N$ ($r=0.97$) shows high consistency between the $\Theta(\log N)$ theory and LLM pre-training, bridging the gap between 100D linear regression and 0.3B LLMs.

## Limitations & Future Work
- **Model Simplification**: The core conclusions rely on linear regression and MSE. While LLM experiments provide support, the form of $E(K,N)$ for deep networks and cross-entropy remains an open question.
- **Constant Learning Rate Assumption**: The $\log N$ saturation factor is tied to a constant learning rate; the paper suggests that decaying rates might reduce reuse to $O(\kappa_H)$, but a full characterization is missing.
- **Epoch Upper Bound**: The strongly convex analysis requires $K=O(N^{0.1})$ to control error, leaving extremely large epoch regimes theoretically uncovered.
- **Distribution Dependency**: The order of the saturation point ($\log N$, power of $N$, etc.) depends on the data spectrum, meaning critical epoch counts for real data must still be estimated per distribution.

## Related Work & Insights
- **Empirical Data Reuse**: Muennighoff et al. (2023) and Xue et al. (2023) debated the value of early epochs vs. multi-epoch degradation; this work provides the theoretical boundaries for these phenomena.
- **Scaling Laws**: Kaplan and Chinchilla set the one-pass paradigm; this paper fills the gap for the multi-epoch paradigm.
- **Lin et al. (2025)**: Also analyzed data reuse in linear regression but only showed $E(K,N)=\Theta(K)$ for small $K$. This paper provides explicit loss characterizations with $o(1)$ error and complete scaling across various data spectra.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First to analytically characterize $E(K,N)$ to $o(1)$ precision and reveal its $N$-dependency.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Combines synthetic data with 0.3B LLM pre-training; though constant LR and epoch bounds limit full coverage.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation, well-structured theorems, and direct correspondence between theory and experiments.
- **Value**: ⭐⭐⭐⭐⭐ Directly informs training budget allocation and scaling law modeling in the data-scarcity era.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] A Generalized Geometric Theoretical Framework of Centroid Discriminant Analysis for Linear Classification of Multi-dimensional Data](a_generalized_geometric_theoretical_framework_of_centroid_discriminant_analysis_.md)
- [\[ICLR 2026\] Learning under Quantization for High-Dimensional Linear Regression](learning_under_quantization_for_high-dimensional_linear_regression.md)
- [\[ICLR 2026\] Closed-form $\ell_r$ norm scaling with data for overparameterized linear regression and diagonal linear networks under $\ell_p$ bias](closed-form_ell_r_norm_scaling_with_data_for_overparameterized_linear_regression.md)
- [\[ICLR 2026\] Theory of Scaling Laws for In-Context Regression: Depth, Width, Context and Time](theory_of_scaling_laws_for_in-context_regression_depth_width_context_and_time.md)
- [\[ICLR 2026\] Robustness of Probabilistic Models to Low-Quality Data: A Multi-Perspective Analysis](robustness_of_probabilistic_models_to_low-quality_data_a_multi-perspective_analy.md)

</div>

<!-- RELATED:END -->
