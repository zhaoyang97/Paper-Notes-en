---
title: >-
  [Paper Note] Contrastive Predictive Coding Done Right for Mutual Information Estimation
description: >-
  [ICLR 2026][Self-Supervised Learning][InfoNCE] This paper theoretically debunks the long-standing misconception that "InfoNCE is a mutual information estimator"—it is actually a variational lower bound of another divergence (K-way JSD) and can never converge to the KL divergence. The authors introduce a simple modification by adding an "anchor class" (InfoNCE-ancho
tags:
  - ICLR 2026
  - Self-Supervised Learning
  - InfoNCE
  - proper scoring rule
date: 2026-05-08
content_hash: 057d90fa4f3608df
---
# Contrastive Predictive Coding Done Right for Mutual Information Estimation

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=JodkBXWgbA](https://openreview.net/forum?id=JodkBXWgbA)  
**Code**: To be confirmed  
**Area**: Self-supervised / Representation Learning  
**Keywords**: Contrastive learning, Mutual information estimation, InfoNCE, Density ratio estimation, proper scoring rule  

## TL;DR
This paper theoretically debunks the long-standing misconception that "InfoNCE is a mutual information estimator"—it is actually a variational lower bound of another divergence (K-way JSD) and can never converge to the KL divergence. The authors introduce a simple modification by adding an "anchor class" (InfoNCE-anchor), allowing the critic to directly learn unambiguous density ratios. This results in a low-bias, low-variance, plug-and-play MI estimator. Furthermore, they use proper scoring rules to unify the NCE / InfoNCE / f-divergence family of contrastive objectives into a single framework.

## Background & Motivation
**Background**: InfoNCE (van den Oord et al., 2018) was originally proposed for contrastive representation learning, but the original appendix casually interpreted it as a variational lower bound for mutual information (MI). Consequently, the community adopted it as the default tool for MI estimation, followed by a vast number of subsequent works such as Poole, Song & Ermon, and Gowri.

**Limitations of Prior Work**: It has long been known that the MI lower bound provided by InfoNCE is loose—it is generally described as a "low-variance but high-bias" estimator: when the true MI is large, the estimated value is systematically underestimated due to the $\log K$ ceiling ($K$ is the number of negative samples / contrastive elements in a batch). However, the reasons for "why it is effective (low variance)" and "why it is limited (high bias)" have never been clearly explained; the community simply accepted its flaws empirically.

**Key Challenge**: The fundamental issue is not that "$K$ is not large enough," but that the objective optimized by InfoNCE is not a variational representation of the KL divergence at all. The authors prove that even if $\log K \ge D(q_1\|q_0)$, the InfoNCE objective value remains strictly smaller than the KL divergence for any finite $K$, so increasing $K$ further cannot close this gap. More critically, the critic learned by InfoNCE can only estimate the density ratio $\frac{p(x,y)}{p(x)p(y)}$ up to an "arbitrary function $C(y)$," preventing it from being used in plug-in estimators.

**Goal**: (1) Clarify exactly what divergence InfoNCE measures and how far it is from true MI; (2) Provide a minimal modification that allows the critic to learn the density ratio itself **unambiguously**, supporting low-bias plug-in MI estimation; (3) Incorporate this modification into a more general theoretical framework to unify existing contrastive objectives.

**Key Insight**: The authors reformulate density ratio estimation as a **tensorized classification problem** and insert an "anchor class" as a fixed reference distribution. Once this immovable reference is established, the multiplicative ambiguity of the density ratio is pinned down.

**Core Idea**: Add an "anchor class (class 0)" consisting entirely of samples from the noise distribution $q_0$ to the InfoNCE classification setting. It serves as a fixed reference to eliminate arbitrary scaling, enabling the critic to directly and consistently estimate $\frac{q_1(x)}{q_0(x)}$. The resulting plug-in estimator retains the low variance of InfoNCE while significantly reducing its bias.

## Method

### Overall Architecture
The logical chain of the paper follows three steps: "Deconstruct, Reconstruct, and Unify." **Step 1** categorizes existing variational MI estimators into three types to locate the pathology of InfoNCE. **Step 2** provides a precise characterization of the InfoNCE objective (it is a tight lower bound of K-way JSD rather than KL), explaining why it cannot perform plug-in estimation. **Step 3** proposes InfoNCE-anchor to fix it and uses proper scoring rules to consolidate the family of contrastive objectives into a unified framework.

First, consider the **taxonomy of estimators** provided by the authors (Table 1), which serves as the map for understanding the paper:

- **Type 1 (Same variational lower bound used for both training and evaluation)**: DV, NWJ, InfoNCE. Simple and natural, but McAllester & Stratos (2020) proved that any distribution-free high-probability MI lower bound is stuck by a $\log N$ sample size ceiling.
- **Type 2 (Train with one lower bound, evaluate by substituting into another lower bound)**: MINE, JS, SMILE. Training is more stable, but a mismatch is introduced between the optimization and evaluation objectives; the $\log N$ criticism still applies.
- **Type 3 (Train to learn the density ratio, evaluate using a plug-in)**: PCC/D-RFC, f-DIME, JSD-LB, and the proposed InfoNCE-anchor. These directly learn the density ratio $\frac{p(x,y)}{p(x)p(y)}$ and substitute it into the definition of MI. This decouples "density ratio learning" from the "specific lower bound," thereby bypassing the inherent limitations of variational lower bounds to achieve lower bias.

The reason InfoNCE is trapped in Type 1 is precisely because it cannot learn unambiguous density ratios to upgrade to Type 3; the full value of InfoNCE-anchor is to "elevate" it into Type 3.

### Key Designs

**1. Precise Characterization of InfoNCE: A Tight Lower Bound of K-way JSD, Not a Variational Representation of KL**

To fix something, one must first explain how it is broken. The authors abstract InfoNCE as a comparison between two distributions $q_1$ (positive samples, specialized as $p(x|y)$) and $q_0$ (noise distribution, $p(x)$): the critic $r_\theta$ assigns high scores to positive samples $x_1\sim q_1$ and low scores to $K-1$ negative samples, with the objective:
$$D_{\text{InfoNCE}}(\theta)=\mathbb{E}\Big[\log\frac{r_\theta(x_1)}{\tfrac{1}{K}\sum_{z=1}^{K}r_\theta(x_z)}\Big].$$.
Classic conclusions (Proposition 1) only provide a loose $D_{\text{InfoNCE}}(\theta)\le\min\{\log K,\,D(q_1\|q_0)\}$. Theorem 2 of this paper provides a **tighter** upper bound: $D_{\text{InfoNCE}}(\theta)\le D_{K\text{-JS}}(q_1,q_0)$, where $D_{K\text{-JS}}$ is a $K$-way generalization of the Jensen–Shannon divergence, with equality **if and only if** $r_\theta(x)\propto \frac{q_1(x)}{q_0(x)}$.

This strikes at two critical points. First, since equality is achieved only when $r_\theta\propto q_1/q_0$ (up to a multiplicative constant), the critic learned by InfoNCE inherently possesses multiplicative ambiguity and cannot be directly substituted into a plug-in estimator. Second, the authors use concrete numbers to debunk the illusion that "$K$ just needs to be large enough": if $D(q_1\|q_0)=2$ bits, then $D_{\text{InfoNCE}}\le 1.19$ for $K=4$, and only reaches $\le 1.93$ for $K=64$; it remains strictly smaller than the KL value of $2$ for any finite $K$. Thus, InfoNCE is **not a variational representation of the KL divergence at all**—this is in stark contrast to DV and NWJ, where the lower bound becomes tight when the critic equals the true log-density ratio.

**2. InfoNCE-anchor: Adding an Anchor Class to Pin Down Multiplicative Ambiguity**

The root cause is that the "density ratio is only learned up to a constant / function," so a way must be found to remove this degree of freedom. The authors rewrite density ratio estimation as a $(K+1)$-class classification problem over $X^K$ (tensorized), with the key addition of an **anchor class (class 0)**:
$$\text{class }0:\ q_0(x_1)q_0(x_2)\cdots q_0(x_K),\qquad \text{class }z\ (z\in[K]):\ q_1(x_z)\prod_{i\ne z}q_0(x_i).$$
Here, class $z$ denotes that "the $z$-th sample comes from $q_1$ and the rest from $q_0$," while class 0 denotes that "all samples come from $q_0$." The class priors are set as $p(z{=}0)=\frac{\nu}{K+\nu}$ and $p(z)=\frac{1}{K+\nu}\,(z\in[K])$, where $\nu\ge 0$ controls the weight of the anchor class. The significance of the anchor class is that it acts as a **fixed reference distribution**; when calculating the posterior via Bayes' theorem, the additional constant term $\nu$ in the denominator locks the scaling degree of freedom. The posterior is:
$$p(z|x_{1:K})=\begin{cases}\dfrac{\nu}{\nu+\sum_i \frac{q_1(x_i)}{q_0(x_i)}}, & z=0\\[2mm]\dfrac{q_1(x_z)/q_0(x_z)}{\nu+\sum_i \frac{q_1(x_i)}{q_0(x_i)}}, & z\in[K].\end{cases}$$
By parameterizing the classifier as $p_\theta(z|x_{1:K})$ (replacing $q_1(x_i)/q_0(x_i)$ with $r_\theta(x_i)$) and performing maximum likelihood estimation, the **InfoNCE-anchor objective** is obtained:
$$L_{K;\nu}(\theta)=-\tfrac{K}{K+\nu}\mathbb{E}\Big[\log\tfrac{r_\theta(x_1)}{\nu+\sum_i r_\theta(x_i)}\Big]-\tfrac{\nu}{K+\nu}\mathbb{E}\Big[\log\tfrac{\nu}{\nu+\sum_i r_\theta(x_i)}\Big].$$
Compared to the original InfoNCE, it only includes an additional second term (log-likelihood of the anchor class), and the implementation overhead is negligible. **Fisher consistency (Theorem 3)** provides the guarantee for this design: when $\nu>0$, the global optimum is $r_{\theta^*}(x)=\frac{q_1(x)}{q_0(x)}$ (almost everywhere under $q_0$, **exact and without constants**); whereas if $\nu=0$ and $K\ge2$, it reduces back to InfoNCE, which can only learn $C\cdot\frac{q_1(x)}{q_0(x)}$. The toggle of $\nu$ is the watershed for "whether plug-in estimation is possible." It also unifies several old objectives: $K{=}2,\nu{=}0$ corresponds to ranking NCE, and $K{=}1,\nu{=}1$ corresponds to standard NCE / JS variational bounds.

**3. Proper Scoring Rule Generalization: Unifying the Contrastive Objective Family**

The cross-entropy (log score) above is just one loss function for estimating class posteriors. The authors point out that once density ratio estimation is reformulated as "class probability estimation (CPE)," **any strictly proper scoring rule** can generate a consistent density ratio estimation objective. For a strictly convex, differentiable generating function $\Psi$, one can define its induced scoring rule $\lambda^\Psi$ and write the overall objective as $L^\Psi_{K;\nu}(\eta_\theta)=\mathbb{E}_{p(x_{1:K},z)}[\lambda_z(\eta_\theta(x_{1:K}))]$. **Theorem 6** characterizes the gap from the optimal solution using Bregman divergence: for $\nu>0$,
$$L^\Psi_{K;\nu}(\eta_\theta)-L^\Psi_{K;\nu}(\eta^*)=\tfrac{\nu}{K+\nu}\,\mathbb{E}_{q_0}\!\Big[B_\Psi\big(\tfrac{r^*(x_{1:K})}{\nu},\tfrac{r_\theta(x_{1:K})}{\nu}\big)\Big],$$
with equality if and only if $r_\theta(x)=\frac{q_1(x)}{q_0(x)}$. The log score is a canonical instance giving InfoNCE-anchor, while the $K{=}1,\nu{=}1$ case reduces to standard variational bounds for $f$-divergences, thereby incorporating f-DIME, f-MICL, etc. The value of this step is "unification": NCE, InfoNCE, and $f$-divergence variants are proven to be special cases of the same density ratio estimation framework under different scoring rules and $(K,\nu)$ settings. The authors also provide asymptotic properties for $K\to\infty$ (Theorem 10): as $\nu/K\to\beta$, InfoNCE-anchor monotonically interpolates between the DV bound ($\beta{=}0$, tightest) and the NWJ bound ($\beta{\to}\infty$, loosest).

### Loss & Training
In practice, the critic is parameterized as $r_\theta(x,y)=e^{c_\theta(x,y)}$ (where $c_\theta$ is a neural network), or in the exponential/cosine form commonly used in representation learning: $r_\theta(x,y)=e^{\frac{1}{\tau}\frac{f_\theta(x)^\top g_\theta(y)}{\|f_\theta(x)\|\|g_\theta(y)\|}}$ with temperature $\tau$. Given a batch size $B$, the authors set $K=B-1$ and default $\nu=1$. The loss is the summation of the two terms above within the batch, where the second term (anchor class) adds almost no cost over InfoNCE. No hyperparameter tuning is required; default settings are consistently more stable than baselines.

## Key Experimental Results

### Main Results
The authors validate the method in three areas: synthetic/real MI estimation benchmarks, protein-protein interaction prediction, and self-supervised representation learning (where the third area is presented as an intentional **negative result**).

Protein-protein interaction prediction (using the PMI of pre-trained pLMs as a threshold to determine interaction, AUROC, average of 20 runs):

| Objective | Kinase (AUROC) | Ligand (AUROC) |
|------|------|------|
| LMI | 0.74 ±0.08 | 0.87 ±0.04 |
| χ² (DRF) | 0.76 ±0.07 | 0.92 ±0.03 |
| JS (=anchor $K{=}1,\nu{=}1$ special case) | 0.77 ±0.08 | 0.95 ±0.02 |
| **InfoNCE-anchor** | **0.80 ±0.06** | **0.97 ±0.01** |
| Spherical | 0.73 ±0.07 | 0.87 ±0.05 |

InfoNCE-anchor is optimal in both tasks, with JS ranking second, reaffirming the "value of large $K$ for accurate density ratio estimation." Standard InfoNCE is directly inapplicable here due to the unknown $C(y)$. On MI estimation benchmarks (Gaussian / MNIST / IMDB-BERT text, with MI ranging from 2 to 10 bits), InfoNCE-anchor (log score) consistently exhibits low bias and low variance, being the only estimator across several plots to stick closely to the ground truth.

### Ablation Study
Self-supervised representation learning (ResNet-18 pre-trained on CIFAR-100, $B{=}256$, linear probing)—this is the key negative result:

| Objective | Top-1 (%) | Top-5 (%) | Description |
|------|------|------|------|
| InfoNCE | **65.98** | **89.69** | Standard contrastive objective, strongest representation |
| InfoNCE-anchor | 65.74 | 89.24 | Accurate MI estimation, but **no improvement** downstream |
| χ² | 65.59 | 88.4 | Comparable to InfoNCE |
| JS (Small $K$) | 61.69 | 87.33 | Clearly weaker, highlighting the importance of large $K$ |
| Spherical | 4.33 | 17.91 | Nearly collapsed, unfavorable optimization geometry |

### Key Findings
- **Accurate MI estimation ≠ Better representation**: While the anchor class makes the density ratio identifiable and reaches SOTA in MI estimation, the SSL linear probing performance remains nearly unchanged (65.74 vs 65.98). This suggests that the unknown multiplicative factor $C(y)$ in InfoNCE is either approximately constant or irrelevant for representation learning.
- **Support from representation structure analysis**: The CKA between features learned by InfoNCE and anchor is $\approx 0.88$ (both backbone and projector), indicating nearly identical global geometry. Uniformity increased slightly (0.350 vs 0.357) while Alignment decreased slightly (-3.77 vs -3.81), cancelling each other out.
- **Scoring rule differences stem from optimization, not consistency**: All strictly proper scoring rules are Fisher consistent in the population/non-parametric limit, but Spherical collapsed to 4.33% in SSL, with the training objective plateauing almost immediately—the difference arises from optimization landscapes rather than statistical inconsistency; log score is the only reliably working instance.
- **Mechanism**: The gains of contrastive learning come from "learning structured density ratios / factorizing PMI + large $K$ + the favorable optimization properties of the log score," **not** from the precise estimation of MI.

## Highlights & Insights
- **Debunking the "misconception" with data**: Instead of just saying InfoNCE is a loose bound qualitatively, the authors provide the tight upper bound of Theorem 2 and calculate that "if $D{=}2$, it only reaches 1.93 even with $K{=}64$," proving that large $K$ alone cannot fix the gap. This clarifies years of misreading "representation learning = maximizing MI."
- **A $\nu>0$ switch for identifiability**: Using an anchor class as a fixed reference to eliminate multiplicative ambiguity is a minimalist but elegant design that corresponds exactly to the watershed of Fisher consistency—it is a beautiful "less is more" design with near-zero implementation cost.
- **Transferable unified perspective**: Reformulating density ratio estimation as "class probability estimation" using proper scoring rules and tensorized classification unifies NCE, InfoNCE, and $f$-divergence variants. This language is reusable for any downstream task requiring unambiguous density ratios (e.g., PPI determination, statistical testing, causal/independence measures).
- **Honest negative results**: The authors proactively report that the anchor class does not improve SSL and use CKA / Uniformity-Alignment to explain why "accurate MI estimation" is decoupled from "representation quality"—this level of self-falsification is more informative than simply reporting improvements.

## Limitations & Future Work
- **No gain in SSL**: The core modification does not bring downstream gains in the main battlefield of representation learning; its value lies mostly in scenarios requiring accurate MI/density ratio values (estimation, testing, scientific computing), rather than beating SSL benchmarks.
- **Optimization landscapes of scoring rules remain unmapped**: Why log score works while Spherical collapses was only observed (immediate plateauing) without complete characterization. The authors admit that "characterizing the landscape properties of successful vs. failed instances is beyond the scope of this paper."
- **Generalization of conclusions**: Negative results are based on the ResNet-18 / CIFAR-100 / linear probe setting. Whether "accurate MI estimation is decoupled from representation quality" holds on larger models, larger batches, or harder tasks requires broader validation (though the authors used different backbones in the appendix).
- **Potential improvements**: Making "good optimization properties" explicit—since differences stem from the optimization landscape, one could design scoring rules with friendlier landscapes for large batches and low noise to re-align "accuracy" with "representation quality."

## Related Work & Insights
- **vs InfoNCE (van den Oord et al., 2018)**: This paper proves InfoNCE is a tight lower bound for K-way JSD, not a variational representation for KL, and the critic can only learn the density ratio up to $C(y)$; InfoNCE-anchor is the "correct way" to do InfoNCE by removing this ambiguity.
- **vs α-InfoNCE (Poole et al., 2019) / MLInfoNCE (Song & Ermon，2020)**: The authors point out a flaw in the prior claim that α-InfoNCE is a tight lower bound for α-skew KL, as it can actually exceed that divergence. While MLInfoNCE satisfies $\le D(q_1\|q_0)$, it lacks an interpretation as a loss from a proper classification setting—whereas InfoNCE-anchor naturally stems from a valid MLE classification problem.
- **vs Type 3 plug-in methods (PCC/D-RFC, f-DIME, JSD-LB)**: These directly learn the density ratio and then plug it in. This paper unifies them (along with $f$-divergence variants for $K{=}1,\nu{=}1$) as special cases and points out that GAN-DIME / HD-DIME are essentially equivalent to the Tsai et al. estimator.
- **vs Uniformity-Alignment perspective (Wang & Isola, 2020)**: The paper uses these metrics to analyze the effect of the anchor class on feature geometry and provides measurable evidence that contrastive learning benefits from structured density ratios rather than precise MI.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Debunks widely misused InfoNCE-as-MI theoretically and provides a minimal, self-consistent fix + unified framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ Sufficiently validated across three scenarios with honest negative results, but the SSL scale is small and the optimization landscape of scoring rules wasn't deeply explored.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear theoretical derivations; the taxonomy and theorems build up logically; arguments are well-supported by counter-examples.
- Value: ⭐⭐⭐⭐ An important clarification and new SOTA tool for the "MI estimation" community; direct gains for SSL practice are limited but the conceptual correction value is high.

<!-- RELATED:START -->
<div class="related-papers" markdown="1">
</div>

## Related Papers

- [\[ICLR 2026\] Bidirectional Predictive Coding](bidirectional_predictive_coding.md)
- [\[ICLR 2026\] Maximizing Incremental Information Entropy for Contrastive Learning](maximizing_incremental_information_entropy_for_contrastive_learning.md)
- [\[CVPR 2026\] Robust Spiking Neural Networks by Temporal Mutual Information](../../CVPR2026/self_supervised/robust_spiking_neural_networks_by_temporal_mutual_information.md)
- [\[ICLR 2026\] Self-Predictive Representations for Combinatorial Generalization in Behavioral Cloning](self-predictive_representations_for_combinatorial_generalization_in_behavioral_c.md)
- [\[ICLR 2026\] On the Alignment Between Supervised and Self-Supervised Contrastive Learning](on_the_alignment_between_supervised_and_self-supervised_contrastive_learning.md)

</div>

<!-- RELATED:END -->
