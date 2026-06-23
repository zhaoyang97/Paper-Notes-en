---
title: >-
  [Paper Note] Learning Correlated Reward Models: Statistical Barriers and Opportunities
description: >-
  [ICLR 2026][learning_theory][Random Utility Model] This paper demonstrates that mainstream pairwise preference data in RLHF **cannot** learn the correlation between user utilities, while best-of-three (ranking one out of three) data is both necessary and sufficient. Based on this, it provides the first identifiability results for the correlated probit model and a near-
tags:
  - ICLR 2026
  - learning_theory
  - Random Utility Model
date: 2026-05-08
content_hash: 54137070677b8a2c
---
# Learning Correlated Reward Models: Statistical Barriers and Opportunities

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=TbEyl6krsY](https://openreview.net/forum?id=TbEyl6krsY)  
**Code**: To be confirmed  
**Area**: Learning Theory / Preference Modeling / RLHF Reward Models  
**Keywords**: Random Utility Model, Correlated Probit Model, IIA Hypothesis, best-of-three preferences, Identifiability, Sample Complexity  

## TL;DR
This paper demonstrates that mainstream pairwise preference data in RLHF **cannot** learn the correlation between user utilities, while best-of-three (ranking one out of three) data is both necessary and sufficient. Based on this, it provides the first identifiability results for the correlated probit model and a near-optimal polynomial-time estimator.

## Background & Motivation
**Background**: The Random Utility Model (RUM) is a classic framework for modeling human preferences, utilized from mathematical psychology and econometrics in the 1960s to modern RLHF reward modeling. The most common is the logit/Bradley-Terry model—the only RUM satisfying the "Independence of Irrelevant Alternatives" (IIA), which enables learning from pairwise comparisons and efficient computation via softmax.

**Limitations of Prior Work**: The IIA assumption in the LLM context implies that all users share a single latent utility function, collapsing rich human preferences into a coarse global ranking. The classic "Red Bus/Blue Bus" paradox illustrates that when correlations exist between utilities, IIA yields contradictory counterfactual predictions. To capture personalized preferences, one must abandon IIA and explicitly model utility correlations (e.g., the correlated probit model $X\sim\mathcal{N}(\mu,\Sigma)$).

**Key Challenge**: Although the correlated probit model is highly expressive, statistical and computational guarantees for learning it are virtually non-existent. Prior to this work, even the question of "identifiability" (whether it can be uniquely determined regardless of sample size) remained unanswered. Two fundamental questions persisted: ① What kind of data is required? ② How many samples are needed?

**Goal**: To answer these questions and establish the theoretical boundaries for learning correlated probit models from preference data.

**Key Insight**: **The order of the data determines whether correlations can be learned.** Pairwise data loses correlation information (a statistical barrier), while best-of-three data successfully restores it (an opportunity). The authors reduce the task of "learning correlated covariance" to a geometric problem of "inferring Gaussian parameters from ternary ranking probabilities" and then extend ternary conclusions to arbitrary $n$ alternatives via inductive aggregation.

## Method

### Overall Architecture
The paper centers on the thesis "pairwise is insufficient, best-of-three is sufficient," progressing along three lines: first, an impossibility theorem establishes the ceiling for pairwise data; second, identifiability is geometrically proven for the minimal case of $n=3$; finally, the ternary conclusions are aggregated into an estimator for arbitrary $n$ with matching lower and upper bounds. A normalization convention is used throughout: since global translation/scaling of utilities does not change the choice distribution, the authors fix $\langle\mu,\mathbf{1}\rangle=0$, $\Sigma\mathbf{1}=0$, and $\mathrm{Tr}(\Sigma)=n$, ensuring $X$ lies on the hyperplane $\mathbf{1}^\top X=0$.

```mermaid
flowchart TD
    A[Correlated Probit Model X~N(mu, Sigma)<br/>Normalization: 1·mu=0, Sigma·1=0, Tr=n] --> B[Thm 3.2 Pairwise Impossibility<br/>Infinite sets of mu, Sigma yield identical pairwise probabilities]
    A --> C[n=3 Ternary Case Thm 4.1<br/>Projection to 2D plane + Whitening]
    C --> D[Thm 4.4 Aggregation to arbitrary n<br/>Reconstruct global matrix from O(n^3) 3x3 sub-matrices]
    D --> E[Thm 5.2 Estimator<br/>O(n^2) sub-matrices, N=O(n^2/eps^2) samples]
    E --> F[Thm 5.3 Lower Bound<br/>Each pair must appear Omega(eps^-2) times, proving near-optimality]
```

### Key Designs

**1. Impossibility Theorem for Pairwise Data: Nailing the Statistical Barrier.** This is the starting point and most impactful conclusion. The authors prove (Theorem 3.2) that for any $n\geq 3$ and normalized $(\mu,\Sigma)$, there exists an **infinite set** $S$ where all $(\mu',\Sigma')$ produce identical pairwise probabilities $\mathbb{P}\{X_i\geq X_j\}$. Intuitively, if $\mu=0$, then $\mathbb{P}\{X_i\geq X_j\}=1/2$ for any $i,j$ regardless of $\Sigma$—the symmetry of pairwise comparisons erases all correlation information. This explains why the correlated probit model lacked statistical guarantees: it was not a computational failure, but a lack of information in the data itself.

**2. Geometric Identification for the Ternary Case: Inverting Probabilities to Half-spaces.** In the minimal learnable case of $n=3$ (Theorem 4.1), the authors project the degenerate Gaussian on the plane $\mathbf{1}^\top X=0$ to a 2D space $V\sim\mathcal{N}(\dot\mu,\dot\Sigma)$. The six total ranking events $\{X_i\geq X_j\geq X_k\}$ correspond to probability masses in six sectors on the plane. A whitening transform $\dot\Sigma^{-1/2}$ is applied to make the distribution isotropic, turning the three event boundaries into half-spaces spanned by unit vectors $\tilde c_i$. Consequently, the univariate "crossing probabilities" $\mathbb{P}\{c_i^\top V\geq 0\}=\Phi(\langle\tilde c_i,\tilde\mu\rangle)$ directly yield $\alpha_i=\langle\tilde c_i,\tilde\mu\rangle$ due to the strict monotonicity of $\Phi$ (Lemma 4.2). The angles between vectors $\alpha_{ij}=\langle\tilde c_i,\tilde c_j\rangle$ are solved via bivariate event probabilities $\gamma_{12}=\mathbb{P}\{\langle X,v_1\rangle,\langle X,v_2\rangle\leq 0\}$ (Lemma 4.3). With all $\alpha_i,\alpha_{ij}$, an invertible linear system using $c_3=c_1+c_2$ reconstructs $\dot\Sigma^{1/2}$ and $\dot\mu$, thereby recovering $\mu, \Sigma$. **This is why best-of-three is both necessary and sufficient**: ternary ranking probabilities provide angle information that pairwise data lacks.

**3. Small-to-Large Aggregation: Stitching Global from Local.** For three alternatives, Theorem 4.1 identifies $(\bar\mu_{ijk},\bar\Sigma_{ijk})$ up to an unknown scale $t_{ijk}>0$. Aggregation relies on **shared pairwise quadratic forms** to align scales: since $c_{ij}^\top\bar\Sigma_{ijk}c_{ij}=c_{ij}^\top\Sigma c_{ij}=\sigma_{ii}+\sigma_{jj}-2\sigma_{ij}$ is independent of $k$, one can set $t_{123}=1$ and propagate to determine all $t_{ijk}$. Then, $\sum_j c_{ij}^\top\Sigma c_{ij}=n\sigma_{ii}+\mathrm{Tr}(\Sigma)$ combined with a summation over $i$ recovers $\mathrm{Tr}(\Sigma)$, solving for each $\sigma_{ij}$. Means are recovered via $c_{ij}^\top(t_{ijk}\tilde\mu_{ijk})=\mu_i-\mu_j$ paired with $\sum_j(\mu_1-\mu_j)=n\mu_1$ (Theorem 4.4). This upgrades "local identifiability" to "global identifiability."

**4. Finite Sample Estimator and Matching Lower Bound.** Naive aggregation requires estimating all $O(n^3)$ sub-matrices. The authors instead use $\tilde O(n^2)$ sub-matrices to provide a polynomial-time estimator (Theorem 5.2) under an observability assumption $\mathbb{P}\{i\succ j\succ k\}\geq\gamma$. With $N\geq C n^2\varepsilon^{-2}\gamma^{-24}\log(n/\delta)\log^6(n/(\gamma\varepsilon))$ ternary ranking observations, it guarantees $\|\mu-\mu^*\|_\infty\leq\varepsilon$ and $\|\Sigma-\Sigma^*\|_\infty\leq\varepsilon$. A matching lower bound (Theorem 5.3) shows any successful estimator must observe **each pair** $(i,j)$ at least $\Omega(\varepsilon^{-2})$ times, leading to a total sample size of $\Omega(n^2\varepsilon^{-2}\log(1/\delta))$, which aligns with the upper bound in $n$, $\varepsilon$, and $\delta$.

## Key Experimental Results

### Main Results (Real Data, Prediction Accuracy, Median Quantile)
Using rankings of 4 out of 6 alternatives as context to predict preferences for the remaining two. Comparison between logit / matrix completion / direct / probit (pairwise) / probit (best-of-three).

| Dataset (Variant, Feature) | logit | Matrix Comp. | probit (pairwise) | probit (best-of-three) |
|---|---|---|---|---|
| sushi (B, default) | 0.65 | 0.66 | 0.64 | **0.65** |
| sushi (A, onehot) | 0.66 | 0.67 | 0.65 | **0.68** |
| sushi (B, onehot) | 0.67 | 0.68 | 0.66 | **0.67** |
| MovieLens 1k (onehot) | 0.62 | 0.60 | 0.60 | **0.61** |
| Netflix 100k (onehot) | 0.62 | 0.61 | 0.59 | **0.62** |
| Netflix 150k (onehot) | 0.61 | 0.60 | 0.56 | **0.61** |
| jokes (onehot) | 0.61 | 0.61 | 0.59 | **0.61** |

Key trend: best-of-three probit **consistently outperforms** pairwise probit (by 3–5 percentage points on several datasets) and generally matches or slightly exceeds logit; pairwise probit is often the worst performer, confirming the theoretical "pairwise information deficiency."

### Ablation Study (Synthetic Data, vs. Ground Truth)
Mean $\mu\in\{0,r\}$, covariance $\Sigma\in\{I,\,\text{bin}(\pm1),\,rI,\,r\}$. "direct" is the simulated ideal upper bound.

| $\mu$ | $\Sigma$ | logit | probit (pairwise) | probit (best-of-three) |
|---|---|---|---|---|
| 0 | $I$ | 0.50 | 0.50 | 0.50 |
| 0 | bin | 0.50 | 0.45–0.54 | **0.79** |
| 0 | $r$ | 0.50 | 0.47–0.55 | **0.67** |
| $r$ | bin | 0.79 | 0.95 | **0.95** |
| $r$ | $r$ | 0.63 | 0.67–0.69 | **0.71** |

Key findings: In **pure correlation** scenarios ($\mu=0$, $\Sigma=$ bin), both logit and pairwise probit fail (accuracy ~0.50), while best-of-three probit reaches 0.79, **directly replicating the "pairwise cannot learn correlation" theory**. Figure 2 further shows pairwise probit hallucinates false correlations, whereas best-of-three recovers a covariance matrix near identical to ground truth.

### Key Findings
- **Interpretable Correlations**: Covariances learned on Netflix/MovieLens have clear semantics—sequels (Spider-Man 1 & 2, Kill Bill Vol.1 & 2) show strong positive correlation (0.56–0.65); art-house films (Memento, Eternal Sunshine) show strong negative correlation with Hollywood blockbusters (Pearl Harbor, Cheaper by the Dozen) (approx -0.40). In the sushi data, sea urchin is polarising, and cucumber rolls exhibit negative correlation with toro (fatty tuna).
- **Welfare Maximization**: Evaluating the expected maximum utility of a fixed-size subset $\max_{R:|R|=N}\mathbb{E}[\max_{i\in R}X_i]$, the best-of-three model is more rational in selecting diverse recommendation sets.
- The gain of best-of-three over logit is more moderate in datasets where mean effects dominate—correlation is a "bonus" in those cases but decisive in correlation-driven scenarios.

## Highlights & Insights
- **Elevation of Data Collection Paradigm**: While previous RLHF discussions focused on models/algorithms, this paper points out that **data order is its own information bottleneck**—no algorithm can recover correlation information missing from pairwise data. This provides direct engineering guidance: to achieve personalization, one should collect best-of-three rankings rather than pairwise comparisons.
- **First Complete Theoretical Loop**: From impossibility (lower bound) to identifiability (existence) to a polynomial-time estimator (construction) to sample lower bounds (optimality), the four pieces of the puzzle are complete.
- **Elegant Geometric Proof**: Projecting high-dimensional degenerate Gaussians to 2D and solving for angles via half-space probabilities provides a clear geometric intuition (Figure 1).
- **Theory Grounded in Real Data**: The transition from theory to producing interpretable structures on recommendation datasets avoids the common pitfall of "theory without validation."

## Limitations & Future Work
- **Strong Observability Assumption**: Theorem 5.2 relies on $\mathbb{P}\{i\succ j\succ k\}\geq\gamma$, and the sample complexity's $\gamma^{-24}$ term is quite pessimistic—actual sample needs might explode if certain ternary rankings are rare. Whether this exponent can be tightened is undiscussed.
- **Gaussian Assumption in Probit**: Correlations are locked into a Gaussian structure; it may mismatch real-world preferences that are heavy-tailed, multimodal, or non-Gaussian.
- **Cost of Best-of-three Data**: While necessary, obtaining full rankings of three items is more taxing and noise-prone for human annotators than pairwise comparisons. The paper does not quantify the trade-off between labeling cost and utility gain.
- **Modest Gains over Logit**: Outside of synthetic strong-correlation scenarios, improvements on real data are often within 1–3 percentage points. Whether this justifies re-engineering RLHF data pipelines requires larger-scale validation.
- **Scaling to LLMs**: Experiments stayed within traditional recommendation datasets; end-to-end validation on true LLM RLHF (prompt-response pairs) is mentioned as motivation but not yet executed.

## Related Work & Insights
- **RUM and IIA Lineage**: From Luce-Shephard-McFadden (logit) and Bradley-Terry to the Red/Blue Bus paradox and empirical IIA violations (Benson et al.), this paper sits at the end of the "critique of IIA" chain but provides the first quantitative characterization of **learnability**.
- **RLHF Preference Modeling**: Relates to Christiano et al.'s preference alignment pipeline and critiques by Bai/Casper/Sorensen/Conitzer on "single global reward" limitations. This paper provides a theoretical basis for why "diverse/personalized preferences are unlearnable" under current paradigms and offers a remedy.
- **Methodological Inspiration**: ① The paradigm "higher-order data unlocks higher-order structure" is transferable to other latent variable models requiring higher-order moments; ② For practitioners, the takeaway is to **redesign preference labeling protocols** to include ternary rankings for critical tasks; ③ Identifiability analysis should precede loss-reduction as a validity check for new preference models.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to answer if/how/how much data is needed for correlated probit. "Pairwise impossible + best-of-three optimal" is a clean, impactful conclusion.
- Experimental Thoroughness: ⭐⭐⭐ Synthetic data supports theory well, and real data is interpretable, but scale is small and lacks true LLM RLHF validation.
- Writing Quality: ⭐⭐⭐⭐ Theoretical narrative is step-by-step with clear geometric intuition; high formula density but well-organized.
- Value: ⭐⭐⭐⭐ Provides principled guidance for preference data collection; a solid theoretical advancement in preference learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Barriers for Learning in an Evolving World: Mathematical Understanding of Loss of Plasticity](barriers_for_learning_in_an_evolving_world_mathematical_understanding_of_loss_of.md)
- [\[ICLR 2026\] Statistical and Structural Identifiability in Representation Learning](statistical_and_structural_identifiability_in_representation_learning.md)
- [\[ICLR 2026\] Near-Optimal Sample Complexity Bounds for Constrained Average-Reward MDPs](near-optimal_sample_complexity_bounds_for_constrained_average-reward_mdps.md)
- [\[ICLR 2026\] Automata Learning and Identification of the Support of Language Models](automata_learning_and_identification_of_the_support_of_language_models.md)
- [\[ICLR 2026\] Interactive Learning of Single-Index Models via Stochastic Gradient Descent](interactive_learning_of_single-index_models_via_stochastic_gradient_descent.md)

</div>

<!-- RELATED:END -->
