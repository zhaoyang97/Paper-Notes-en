---
title: >-
  [Paper Note] What Exactly Does Guidance Do in Masked Discrete Diffusion Models
description: >-
  [ICLR 2026][Image Generation][Masked Discrete Diffusion] This paper provides the first rigorous characterization of classifier-free guidance (CFG) in **masked discrete diffusion models** under low-dimensional ($1D/2D$) analytical settings. It demonstrates that CFG moves probability mass from "inter-class overlapping regions" to "class-exclusive regions," and the convergence speed of reverse sampling dynamics toward the target distribution accelerates **doubly exponentially**…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "Masked Discrete Diffusion"
  - "classifier-free guidance"
  - "tilted distribution"
  - "convergence rate"
  - "double exponential"
date: 2026-05-08
content_hash: d15f14833bb244d2
---

# What Exactly Does Guidance Do in Masked Discrete Diffusion Models

**Conference**: ICLR 2026  
**Code**: None  
**Area**: Diffusion Models / Discrete Diffusion / Theoretical Analysis  
**Keywords**: Masked Discrete Diffusion, classifier-free guidance, tilted distribution, convergence rate, double exponential

## TL;DR
This paper provides the first rigorous characterization of classifier-free guidance (CFG) in **masked discrete diffusion models** under low-dimensional ($1D/2D$) analytical settings. It demonstrates that CFG moves probability mass from "inter-class overlapping regions" to "class-exclusive regions," and the convergence speed of reverse sampling dynamics toward the target distribution accelerates **doubly exponentially** with respect to the guidance strength $w$.

## Background & Motivation

**Background**: Early diffusion models were established in continuous state spaces (adding Gaussian noise and denoising). Recently, **discrete diffusion**—using masks or categorical jumps instead of Gaussian corruption—has emerged, particularly suitable for discrete data such as language, molecules, and proteins. The "absorbing state/masking" forward process (where each dimension is independently replaced by a mask symbol [M]) is one of the most common types. To achieve controllable conditional generation, **classifier-free guidance**, highly successful in continuous diffusion, has been adapted to discrete diffusion (Nisonoff et al. 2024, etc.) by modifying the reverse transition rates, significantly improving sample quality and controllability in practice.

**Limitations of Prior Work**: While the mechanism of CFG in continuous diffusion has theoretical explanations (under simplified $1D$ Gaussian settings), **there is almost no theoretical characterization of what CFG actually modifies in discrete diffusion**. It is only known that "moderate $w>0$ works best," but this optimal point neither corresponds to sampling from a simple explicit distribution nor explains how it affects the dynamics of the sampling trajectory.

**Key Challenge**: When $w>0$, CFG **does not** sample from any explicit distribution; instead, it modifies the reverse dynamics nonlinearly. Therefore, two fundamental questions remain unanswered: "what distribution does it actually generate" and "what determines the convergence speed." Existing conclusions from continuous diffusion rely heavily on Gaussian/compact support assumptions and are restricted to $1D$, making them difficult to generalize directly.

**Goal**: Within the analytical subclass of masked discrete diffusion, this paper aims to precisely answer two questions for **any finite mixture** data distribution: (Q1) How does guidance change the distribution of generated samples? (Q2) How does guidance change the convergence rate of the reverse dynamics?

**Key Insight**: The authors reduce the dimensionality to $D=1$ and $D=2$. Under low dimensions, the reverse rate matrix of masked diffusion has an explicit solution, allowing the generated distribution and sampling trajectory to be calculated and explained **analytically and precisely** rather than approximately.

**Core Idea**: By using low-dimensional analyticity, this paper transforms CFG from a "black-box heuristic" into a white-box. It proves that guidance essentially "utilizes the geometric structure of data support overlap," amplifying class-exclusive regions, suppressing inter-class shared regions, and causing the convergence rate to exhibit a doubly exponential increase with $w$.

## Method

### Overall Architecture

This is a **pure theoretical analysis** paper. It does not propose new models or algorithms but provides a rigorous characterization of an existing discrete CFG construction. The overall logic is: first, assume the data distribution is a mixture of multiple classes (Assumption 1.1, $p(\cdot)=\sum_{k=1}^{M} a_k\, p(\cdot|z_k)$), with the goal of sampling only class $z_1$; second, derive the reverse dynamics with CFG under a masking (absorbing state) forward process; finally, solve for the "final generated distribution" and "convergence rate along the reverse trajectory" in $D=1$ and $D=2$ settings, and discuss generalizations to higher dimensions.

The forward process of masked diffusion is a continuous-time Markov process $\frac{dp_t}{dt}=Q_t p_t$, where each dimension independently jumps to the mask state [M]. The corresponding exact reverse process $\frac{dq_t}{dt}=\bar Q_{T-t} q_t$ has rates determined by the **concrete score** $\frac{p_t(y)}{p_t(x)}$ (the discrete analogue of $\nabla\log p_t$ in continuous diffusion). The paper assumes exact scores and exact simulation, focusing on the effect of CFG itself on the dynamics, leaving score estimation errors and numerical discretization errors for future work.

The target distribution of CFG is the **tilted distribution**:

$$p_{z,w}(\cdot)\ \propto\ p(\cdot)\,p(z|\cdot)^{1+w}\ \propto\ p(\cdot)^{-w}\,p(\cdot|z)^{1+w},$$

where $w\ge -1$ is the guidance strength: $w=-1$ returns the full data distribution $p$, $w=0$ returns the class-conditional distribution $p(\cdot|z)$, and larger $w>0$ biases more toward states "more like class $z$." The key issue is: because diffusion follows dynamics, it is **impossible to sample directly from $p_{z,w}$**; one can only modify the reverse rate matrix. This is the source of the subsequent analysis on "how much the actual generated distribution deviates from the tilted distribution."

The following four key designs are: Discrete CFG construction $\to$ $1D$ exact results $\to$ $2D$ deviation results $\to$ Doubly exponential convergence rate.

### Key Designs

**1. Discrete CFG Construction: Achieving "Tilting" via Geometric Interpolation of the Rate Matrix**

In continuous diffusion, guidance is a linear extrapolation of the score $\nabla\log p_t$. Since there is no continuous score in discrete space, the authors follow the construction of Nisonoff et al. (2024), performing the operation on the **reverse transition rates**. First, the class-conditional distribution $p(\cdot|z)$ is evolved under the same forward process $Q_t$ to obtain the class-conditional reverse rate $\bar Q^z_t(y,x)=\frac{p_t(y|z)}{p_t(x|z)}Q_t(x,y)$. Then, the CFG rate is obtained by **geometrically interpolating** the unguided rate $\bar Q_t$ and the class-conditional rate $\bar Q^z_t$:

$$\hat Q^{z,w}_t(y,x)=\bar Q_t(y,x)^{-w}\,\bar Q^z_t(y,x)^{1+w},\qquad y\ne x.$$

When $w=-1$, $\hat Q^{z,-1}_t=\bar Q_t$ (generating the entire mixture $p$ without control). When $w=0$, $\hat Q^{z,0}_t=\bar Q^z_t$ (exactly generating $p(\cdot|z)$). In practice, the best results appear at intermediate $w>0$, but at this point, $\hat Q^{z,w}_t$ does not correspond to the exact reverse of any explicit distribution—this is the object of the paper's analysis.

**2. 1D Precise Characterization: The generated distribution exactly equals the tilted distribution, with mass removed from overlapping regions**

In $D=1$ (single token), the CFG reverse rate matrix simplifies dramatically: it is **exactly equal to the unguided reverse rate targeting the tilted distribution $p_{z,w}$** (up to a normalization constant $Z_{z,w}=\sum_{x=1}^{N-1}p(x)^{-w}p(x|z)^{1+w}$). Theorem 3.1 provides an explicit formula for the reverse trajectory and yields a clean conclusion: **the final generated distribution $q^{z,w}_T$ is exactly equal to the tilted distribution $p_{z,w}$**. This differs from continuous diffusion (where CFG deviates from the tilted distribution in $1D$).

Furthermore (Proposition 3.1): if the support $X_1$ of class $z_1$ does not intersect with other classes, then $q^{z_1,w}_T=p(\cdot|z_1)$ and guidance has no effect. If an overlapping region $S_1=X_1\cap(\cup_{k\ge2}X_k)$ exists, $p(x|z_1)$ is maintained on the exclusive region $X_1\setminus S_1$, while the overlapping region $S_1$ is reweighted by $\big(\tfrac{a_1 p(x|z_1)}{\sum_{k\in I_1}a_k p(x|z_k)}\big)^{w}p(x|z_1)$. As $w\to\infty$, the mass in the overlapping region is completely cleared, and the distribution converges to the restriction of $p(\cdot|z_1)$ on the exclusive region. Intuitively: CFG moves mass from the fuzzy region $S_1$ to the exclusive region while **keeping the local mean and variance within the exclusive region unchanged** (Remark 3.1).

**3. 2D Deviation Characterization: Generated distribution no longer equals the tilted distribution, marginal projection overlaps introduce new coefficients**

In $D=2$ (multiple tokens), the story changes: the CFG reverse rate matrix is **no longer** equal to the reverse rate of the tilted distribution ($\hat Q^{z,w}_t\ne C\,\bar Q_t[p_{z,w}]$). Theorem 3.2 provides an explicit expression for the $2D$ reverse dynamics, which, besides $p_{z,w}$ and $Z$, includes a set of coefficients $\{c_x,d_x\}$ that specifically encode the manipulation of **marginal distributions** by guidance, such as $c_{x_1}=\frac{\sum_l p(x_1,l)^{-w}p(x_1,l|z)^{1+w}}{p(x_1)^{-w}p(x_1|z)^{1+w}}$ (and $d_{x_2}$ similarly along the other coordinate). The final generated distribution is a "weighted version" of the tilted distribution:

$$q^{z,w}_T(x)=\frac{1/c_{x_1}+1/d_{x_2}}{1/c_N+1/d_N}\,p_{z,w}(x).$$

More importantly, the geometric picture (Proposition 3.3, Remark 3.4): the exclusive support of class $z_1$ is partitioned into five regions $R_1, R_{2,1}, R_{2,2}, R_3, R_4$, reflecting different degrees of "privacy"—$R_1$ is the most private (even marginal projections do not intersect with other classes), $R_4=S_1$ is the complete overlap region, $R_3$ has both projections overlapping, and $R_{2,i}$ has only the $i$-th dimensional projection overlapping. Their (unnormalized) weights satisfy $A_1^{z_1,w}\ge A_{2,i}^{z_1,w}\ge A_3^{z_1,w}\ge A_4^{z_1,w}$, meaning **more private regions are assigned larger weights by guidance**. This indicates that even if there is no overlap in the full space, as long as a coordinate projection hits another class, CFG will suppress it hierarchically based on "how many dimensions overlap"—a new phenomenon not seen in $1D$.

**4. Doubly Exponential Convergence Rate: TV distance becomes doubly exponentially faster with guidance strength $w$**

The second core conclusion concerns the "speed" of the dynamics. In $1D$ (Proposition 3.2), the Total Variation (TV) distance from the reverse trajectory to the generated distribution $\mathrm{TV}(q^{z,w}_t,p_{z,w})=\big(\tfrac{1-e^{-(T-t)}}{1-e^{-T}}\big)Z$ decays exponentially over time with rate $Z$. Rewriting $Z$ using $\alpha$-divergence as $Z=\exp\!\big(w D_{1+w}(p(\cdot|z)\,\|\,p)\big)$, we get $\log Z_{z,w}\sim w\sup_x\frac{p(x|z)}{p(x)}$ for $w\gg1$. Consequently, the overall decay rate of TV distance has a **doubly exponential dependence** on $w$ (Remark 3.2)—this explains the abrupt shift in sampling behavior observed in practice when $w$ is large. In $2D$ (Proposition 3.4), a similar relationship holds: $-\ln\mathrm{TV}(q^{z,w}_t,q^{z,w}_T)=\exp(\Theta(w))\ln\!\big(\tfrac{1-e^{-T}}{1-e^{-(T-t)}}\big)$, preserving the doubly exponential dependence. The conclusion is that guidance not only reshapes the output distribution but also **controls the dynamical speed of the sampling trajectory**.

### Loss & Training
Ours does not introduce a new training objective, following the standard denoising score entropy (DSE) to learn the concrete score:
$$L_{\text{DSE}}=\mathbb{E}_{x_0\sim p}\,\mathbb{E}_{x\sim p_{t|0}(\cdot|x_0)}\Big[\sum_{y\ne x}\big( s^\theta_t(x,y)-\tfrac{p_{t|0}(y|x_0)}{p_{t|0}(x|x_0)}\log s^\theta_t(x,y)\big)\Big].$$
The analysis assumes both the score and simulation are exact, studying only the effect of CFG.

## Key Experimental Results

Numerical experiments use a small Transformer to train the score, using Tau-leaping with 50 steps, a log-linear schedule, and 10K samples per group on a single RTX 4070 Laptop GPU. The goal is **theoretical validation** rather than performance benchmarking.

### Main Results (Theory vs. Empirical Comparison)

| Setting | Key Phenomenon | Theoretical Correspondence |
| :--- | :--- | :--- |
| 1D Non-overlapping Support | Guidance is completely ineffective; generated/tilted/class distributions coincide. | Prop. 3.1-(1) |
| 1D Overlapping Support | Mass is removed from overlapping regions; generated distribution approximates tilted distribution (despite score/discretization error). | Prop. 3.1-(2) |
| 1D, Fixed $t=0.5$, sweep $w$ | Empirical TV curve matches theory at small $w$; plateau/upturn appears at large $w$. | Prop. 3.2 (discrepancy at large $w$ attributed to numerical instability of Tau-leaping during abrupt transitions) |
| 2D Full Disjoint Support but Overlapping Projections | Guidance suppresses the top and right corners of the central diamond. | Prop. 3.3 / Thm. 3.2 |
| 2D Overlapping Support | Strong suppression of top-right overlap region; light suppression of top-left/bottom-right due to marginal projection overlap. | Prop. 3.3 |

### Ablation Study (5D Mixed Hypercube)

5D Setting: Two classes supported on $\{0,1,2\}^5$ and $\{2,3,4\}^5$, overlapping only at the single point $(2,2,2,2,2)$; target class is $\{2,3,4\}^5$. The space is partitioned by "how many dimensions equal 2" ($k=\#\{d:x_d=2\}$), where $k=0$ is the exclusive region and $k=5$ is the complete overlap point.

| $\#\{d:x_d=2\}$ | 0 | 1 | 2 | 3 | 4 | 5 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| State Count | 32 | 80 | 80 | 40 | 10 | 1 |
| Mean Density per State $w=1$ (1e-3) | 4.984 | 4.541 | 4.020 | 3.386 | 2.208 | 0.280 |
| Mean Density per State $w=2$ (1e-3) | 5.573 | 4.783 | 3.867 | 2.844 | 1.588 | 0.200 |

### Key Findings
- **Single-state mass decreases monotonically with degree of overlap**: Each state in the exclusive region ($k=0$) receives the most mass, the complete overlap point ($k=5$) receives the least, and partial overlaps are in between—perfectly consistent with the "privacy-level hierarchical weighting" in $2D$.
- **Higher $w$ leads to more drastic redistribution**: From $w=1$ to $w=2$, mass moves further from high-overlap regions (large $k$) to exclusive/low-$k$ regions ($k=0$ mean $4.984 \to 5.573$, $k=5$ mean $0.280 \to 0.200$).
- **Partial overlap regions are not suppressed uniformly**: Depending on $k$, partial overlap regions may gain or lose mass relative to $p(\cdot|z)$, confirming that the mechanism of $\{c_x,d_x\}$ based on "projection dimensions" generalizes to high dimensions.
- **Numerical cost of large $w$**: The theoretically predicted abrupt transitions cause Tau-leaping to become unstable at large $w$, leading to empirical TV curves deviating from theory—reminding practitioners that large guidance strength requires more refined numerical schemes.

## Highlights & Insights
- **Turning "Heuristic CFG" into a White Box**: Provides **exact** (not approximate) characterizations of the generated distribution and convergence rate in low-dimensional analytical settings, marking the first rigorous theory for discrete diffusion CFG.
- **Qualitative differences between 1D and 2D** are counter-intuitive: $1D$ generated distribution exactly equals the tilted distribution, while $2D$ deviates, with the deviation explicitly given by coefficients $\{c_x,d_x\}$ of marginal projection overlaps. This shows that the dimension itself changes the geometric effect of guidance.
- **"Privacy Hierarchy" is a transferable intuition**: CFG = utilizing the data support overlap geometry, where more private class-exclusive regions are amplified and fuzzier shared regions are suppressed. This intuition can guide expectations for guidance behavior in high dimensions and explains the trade-off where strong guidance sacrifices diversity.
- **Doubly Exponential Convergence** quantifies why sampling behavior changes so violently when $w$ increases, which is useful for understanding and tuning guidance strength.

## Limitations & Future Work
- **Dimensionality limits**: Exact results are limited to $1D/2D$; for $D\ge3$, there are only **conjectured** behaviors based on marginal projection overlaps (unique/partial-overlap/full-overlap regions), lacking formal proof.
- **Idealized Assumptions**: The analysis assumes exact scores and exact reverse simulation, avoiding score estimation errors and numerical discretization errors. Experiments show Tau-leaping becomes unstable at large $w$, leaving a gap between theory and practice for future study. ⚠️ For complex expressions like $2D$ coefficients $\{c_x,d_x\}$ and $\alpha_t(x)$, refer to Theorem 3.2 / Appendix E of the original paper.
- **Data Assumptions**: Relies on the structure of "multi-class mixture + classes supported on subsets $X_k\subsetneq S$" (Assumption 1.1); whether the support geometry of real language/molecular data holds this remains unverified.
- **Improvement Ideas**: Generalizing $1D/2D$ blueprints to formal results for general $D$; analyzing perturbations to the exact conclusions caused by score approximation and Tau-leaping discretization; designing more stable samplers for large $w$ based on the doubly exponential phenomenon.

## Related Work & Insights
- **vs. Continuous Diffusion CFG Theory (Bradley & Nakkiran 2024; Chidambaram et al. 2024)**: They analyze continuous CFG under $1D$ Gaussian/compact support assumptions and find deviations from the tilted distribution. Ours targets **discrete** diffusion, allows for **any finite mixture**, and finds that the $1D$ discrete generated distribution exactly equals the tilted distribution (opposite to the continuous case), with deviation only occurring in $2D$—conditions are more general, and conclusions are more precise.
- **vs. Discrete CFG Constructions (Nisonoff et al. 2024; Schiff et al. 2024)**: They propose/improve discrete guidance via rate modification and report empirical gains. Ours does not change the method but clarifies its **mechanism**, filling the theoretical gap.
- **Insights**: Understanding "controllable generation" as "mass redistribution on the overlapping geometry of data supports" may be transferable to other controllable generation tasks, such as quantitative analysis of the diversity-fidelity trade-off.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First rigorous, analytical theory for discrete diffusion CFG; both the $1D/2D$ difference and doubly exponential convergence are new conclusions.
- Experimental Thoroughness: ⭐⭐⭐⭐ $1D/2D/5D$ numerical experiments fully support the theory, though there are no large-scale experiments on real data as this is an analytical paper.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and well-distilled conclusions; the $2D$ explicit expressions are dense and require careful reading against the original text.
- Value: ⭐⭐⭐⭐ Provides a solid theoretical anchor and transferable intuition for understanding and tuning discrete diffusion guidance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Discrete Guidance Matching: Exact Guidance for Discrete Flow Matching](discrete_guidance_matching_exact_guidance_for_discrete_flow_matching.md)
- [\[ICLR 2026\] Improving Classifier-Free Guidance in Masked Diffusion: Low-Dim Theoretical Insights with High-Dim Impact](improving_classifier-free_guidance_in_masked_diffusion_low-dim_theoretical_insig.md)
- [\[ICLR 2026\] Stage-wise Dynamics of Classifier-Free Guidance in Diffusion Models](stage-wise_dynamics_of_classifier-free_guidance_in_diffusion_models.md)
- [\[ICLR 2026\] Guidance Watermarking for Diffusion Models](guidance_watermarking_for_diffusion_models.md)
- [\[ICLR 2026\] Bringing Stability to Diffusion: Decomposing and Reducing Variance of Training Masked Diffusion Models](bringing_stability_to_diffusion_decomposing_and_reducing_variance_of_training_ma.md)

</div>

<!-- RELATED:END -->
