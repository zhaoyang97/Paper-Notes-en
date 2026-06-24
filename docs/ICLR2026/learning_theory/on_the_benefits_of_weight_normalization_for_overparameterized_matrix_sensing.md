---
title: >-
  [Paper Note] On the Benefits of Weight Normalization for Overparameterized Matrix Sensing
description: >-
  [ICLR 2026][Learning Theory][Weight Normalization] This paper provides the first theoretical characterization of Weight Normalization (WN) in overparameterized matrix sensing. By decoupling matrix variables into "direction (Stiefel manifold) + magnitude (symmetric matrix)" and applying Riemannian Gradient Descent, the method achieves **linear convergence** under finite samples (an exponential acceleration compared to the sublinear lower bound of standard GD). Notably…
tags:
  - "ICLR 2026"
  - "Learning Theory"
  - "Optimization"
  - "Weight Normalization"
  - "Matrix Sensing"
  - "Overparameterization"
  - "Riemannian Optimization"
  - "Saddle Point Escape"
date: 2026-05-08
content_hash: b44ec2480bf21d08
---

# On the Benefits of Weight Normalization for Overparameterized Matrix Sensing

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=JIMM5YLShy](https://openreview.net/forum?id=JIMM5YLShy)  
**Code**: To be confirmed  
**Area**: Learning Theory / Optimization  
**Keywords**: Weight Normalization, Matrix Sensing, Overparameterization, Riemannian Optimization, Saddle Point Escape

## TL;DR
This paper provides the first theoretical characterization of Weight Normalization (WN) in overparameterized matrix sensing. By decoupling matrix variables into "direction (Stiefel manifold) + magnitude (symmetric matrix)" and applying Riemannian Gradient Descent, the method achieves **linear convergence** under finite samples (an exponential acceleration compared to the sublinear lower bound of standard GD). Notably, as the degree of overparameterization increases, both iteration complexity and sample complexity **decrease polynomially**.

## Background & Motivation
**Background**: Normalization techniques (BatchNorm / LayerNorm / WeightNorm) are standard in modern deep networks, stabilizing training and improving generalization in practice. Weight Normalization (WN), which decouples parameters into "direction" and "magnitude" for separate optimization, has recently regained attention due to its seamless integration into LoRA (e.g., DoRA and other PEFT methods). However, the theoretical reasons for WN's effectiveness remain unclear—existing results only cover simpler scenarios like implicit regularization in overparameterized least squares or diagonal linear networks.

**Limitations of Prior Work**: To characterize the benefits of WN, a non-convex "testbed" with a rich loss landscape is required. Overparameterized matrix sensing is a classic example: the goal is to recover a low-rank PSD matrix $A \in \mathbb{S}^m_+$ from linear measurements $y_i = \mathrm{Tr}(M_i^\top A)$ using Burer-Monteiro factorization $Y Y^\top \approx A$ ($Y \in \mathbb{R}^{m\times r}$). Since the true rank $r_A$ is unknown beforehand, practice often adopts an overparameterized setting where $r > r_A$ to ensure exact recovery.

**Key Challenge**: Overparameterization here is a double-edged sword. Previous work (Xiong et al., 2024) proved that even in the population (infinite sample) setting, **the convergence rate of standard Gradient Descent (GD) has a sublinear lower bound** $\Omega(1/t)$, which is exponentially slower than the linear rate achieved when $r=r_A$. In other words, overparameterization hinders GD instead of helping it; moreover, without WN, the magnitude of random initialization must be meticulously controlled (often proportional to $1/\kappa$), otherwise convergence is difficult to guarantee.

**Goal**: Prove that WN can bypass this sublinear lower bound and quantify the counter-intuitive phenomenon that "more overparameterization leads to faster convergence and fewer samples."

**Key Insight**: The authors observe that WN direction variables are naturally constrained on smooth manifolds (spheres / Stiefel manifolds), which aligns with the framework of Riemannian optimization. By extending scalar WN to matrix variables and using Riemannian Gradient Descent (RGD) to optimize the direction, they investigate whether the decoupled loss landscape becomes "benign."

**Core Idea**: Use polar decomposition to split $Y = X\tilde\Theta$ into a direction $X$ on the Stiefel manifold and a symmetric matrix magnitude $\Theta$. By running RGD for $X$ and GD for $\Theta$, the **sublinear convergence of standard GD is upgraded to linear convergence**, turning overparameterization from an "enemy" into a "friend."

## Method

### Overall Architecture
The paper does not propose a new algorithm for benchmarking but establishes a convergence theory for "WN + Riemannian Optimization" in overparameterized matrix sensing. The logic follows three steps: **(1) Reparameterization**—rewriting the original Burer-Monteiro objective $\min_Y \frac14\|\mathcal{M}(YY^\top)-y\|^2$ into a direction-magnitude decoupled WN form via polar decomposition; **(2) Optimizer**—applying RGD to the direction $X$ on the Stiefel manifold (projection to tangent space + polar retraction) and standard GD to the magnitude $\Theta$ in the space of symmetric matrices, using alternating updates; **(3) Convergence Analysis**—proving that the iterations undergo two phases ("saddle point phase" $\to$ "linear convergence phase") and quantifying the polynomial improvement of iteration/sample complexity as a function of the overparameterization degree $r$.

Specifically, $Y\in\mathbb{R}^{m\times r}$ is written as $Y = X\tilde\Theta$ via polar decomposition, where $X \in \mathrm{St}(m,r) = \{X\in\mathbb{R}^{m\times r}\mid X^\top X = I_r\}$ is the direction (orthogonal basis of an $r$-dimensional subspace) and $\tilde\Theta \in \mathbb{S}^r_+$ is the magnitude. Two simplifications are made: merging $\tilde\Theta\tilde\Theta^\top$ into a single matrix $\Theta\in\mathbb{S}^r_+$ and **relaxing the PSD constraint to simple symmetry** $\Theta\in\mathbb{S}^r$. This relaxation does not change the global objective in the overparameterized regime but avoids the SVD / matrix exponents required for PSD cone optimization, significantly saving computation. The final objective is:

$$\min_{X,\Theta}\; f(X,\Theta) := \tfrac14\big\|\mathcal{M}(X\Theta X^\top)-y\big\|^2,\quad \text{s.t. } X\in\mathrm{St}(m,r),\ \Theta\in\mathbb{S}^r.$$

### Key Designs

**1. Matrix-version Weight Normalization: Decoupling "Direction" and "Magnitude" via Polar Decomposition**

Scalar WN splits a vector into a unit direction plus a scalar length; this work extends it to matrix variables. The challenge is that running GD directly on $Y$ results in a landscape filled with saddle points and lacks global smoothness, further slowed down by overparameterization. The authors separate these roles using $Y = X\tilde\Theta$, where $X$ is an orthogonal basis on the Stiefel manifold (geometrically the "orientation" of an $r$-dimensional subspace) and $\tilde\Theta$ is the "scaling" within that subspace. An essential engineering choice is merging $\tilde\Theta\tilde\Theta^\top$ into a symmetric matrix $\Theta$ and relaxing the PSD constraint. This places the magnitude variable in the linear space $\mathbb{S}^r$, allowing for simple GD updates and avoiding SVD overhead for PSD projection, without losing global optimality when $r>r_A$. After decoupling, the direction variable is constrained on a smooth manifold, bringing the problem into the mature toolkit of Riemannian optimization—the geometric root of all subsequent benefits.

**2. Alternating Updates with RGD for Direction and GD for Magnitude**

After decoupling, the problem shifts to "how to iterate on the manifold." For the direction $X$, the Euclidean gradient is first projected onto the tangent space of the Stiefel manifold at $X_t$ to obtain the Riemannian gradient $G_t$ (formally $G_t = (I-X_tX_t^\top)\nabla_X G_t + \tfrac{X_t}{2}(X_t^\top \nabla G_t - \nabla G_t^\top X_t)$, see Eq (3) in the paper for the exact projection), followed by a **polar retraction** to pull the updated point back to the manifold:

$$X_{t+1} = (X_t - \eta G_t)\,(I_r + \eta^2 G_t^\top G_t)^{-1/2},$$

where $\eta$ is the step size. While polar retraction is used for theoretical simplicity, the authors verify that QR and Cayley retractions are numerically equivalent. For magnitude $\Theta$, standard GD is used: $\Theta_{t+1} = \Theta_t - \mu K_t$, where $K_t := \nabla_\Theta f(X_{t+1},\Theta_t)$. This update also ensures the symmetry constraint—if $\Theta_0\in\mathbb{S}^r$, all subsequent $\Theta_t$ automatically fall in $\mathbb{S}^r$. This process is denoted as Algorithm 1 (RGD). Unlike standard GD, the direction is "held steady" by the manifold constraint and the magnitude can scale independently; their coordination enables the linear convergence rate.

**3. Two-stage Convergence Proof: Saddle Escape and Linear Convergence**

This is the primary theoretical result (Theorem 3.2). Under random initialization $X_0\sim\mathrm{St}(m,r)$ and the condition that the measurement operator satisfies $(r+r_A+1,\delta)$-RIP, the trajectory of the reconstruction error $\|X_t\Theta_t X_t^\top - A\|_F$ is divided by a burn-in time $t_0$ into two phases:

- **Saddle Point Phase** ($1\le t\le t_0$): The error decreases monotonically but seemingly slowly as the iteration traverses/escapes several saddle points; $t_0$ is upper-bounded by $O\!\big(\frac{\kappa^4 m^4 r^4 r_A^2}{(r-r_A)^8}\big)$.
- **Linear Convergence Phase** ($t\ge t_0+1$): The error converges to zero at a linear rate, $\|X_t\Theta_t X_t^\top - A\|_F \le 3\big(1 - \tfrac{c_3(r-r_A)^4}{\kappa^4 m^2 r^2 r_A}\big)^{t-t_0}$.

The geometric intuition comes from the direction-magnitude decoupling: using the alignment matrix $\Phi_t := U^\top X_t$ (where $U$ is the left singular vector matrix of $A$), its singular values are the cosines of the principal angles between $\mathrm{span}(U)$ and $\mathrm{span}(X_t)$. The core of the proof is $\mathrm{Tr}(\Phi_t\Phi_t^\top)\to r_A$, meaning the subspaces eventually align. In the first phase, $\mathrm{Tr}(\Phi_t\Phi_t^\top)$ climbs monotonically from near 0 to $r_A-0.5$. After passing this threshold, the alignment error $r_A - \mathrm{Tr}(\Phi_t\Phi_t^\top)$ drops linearly to 0. This linear rate represents an exponential acceleration over the $\Omega(1/t)$ sublinear lower bound of standard GD.

**4. Turning Overparameterization from an "Enemy" into a "Friend"**

The most counter-intuitive conclusion: adding parameters does not slow down the process but rather makes it faster and more sample-efficient. Let $r = p\,r_A$ ($p>1$). The burn-in time can be rewritten as $O\!\big(\frac{\kappa^4 m^4 p^4}{(p-1)^8 r_A^2}\big)$, which decreases polynomially with $p$; the linear phase rate $\exp\!\big(-O(\frac{(p-1)^4 r_A}{\kappa^4 m^2 p^2}t)\big)$ also accelerates as $p$ increases. For light overparameterization where $r=r_A+c$ ($c=O(1)$), the rate is roughly $\exp(-O(\frac{t}{\kappa^4 m^2 r_A^3}))$. Increasing to $r=c\,r_A$ improves this to $\exp(-O(\frac{r_A t}{\kappa^4 m^2}))$—an improvement of up to $O(r_A^4)$ in the exponent. Similarly, for statistical complexity, RIP holding with high probability under Gaussian design requires a sample size $n = O\!\big(\frac{\kappa^4 m^7 r^9 r_A^2}{(r-r_A)^{12}}\big)$, **which also decreases polynomially as $r$ increases**, potentially saving a factor of $O(r_A^{12})$. This sharply contrasts with standard GD, where overparameterization is a burden; WN turns it into a tunable acceleration knob.

### Loss & Training
The optimization objective is the non-convex least squares from Eq (2): $f(X,\Theta)=\frac14\|\mathcal{M}(X\Theta X^\top)-y\|^2$. The key hyperparameters are determined by Theorem 3.2: direction step size $\eta = O\!\big(\frac{(r-r_A)^4}{\kappa^2 m^2 r^2 r_A}\big)$, magnitude step size $\mu=2$, and RIP constant $\delta = O\!\big(\frac{(r-r_A)^6}{\kappa^2 m^3 r^4 r_A}\big)$. Initialization is truly random: $X_0 = Z_0(Z_0^\top Z_0)^{-1/2}$ ($Z_0$ has i.i.d. entries $\mathcal{N}(0,1)$) and $\|\Theta_0\|\le 2$—significantly more relaxed than the $1/\kappa$-proportional "small magnitude" initialization required by standard GD.

## Key Experimental Results

Experiments focus on verifying the theory using synthetic data ($A=U\Sigma U^\top$, Gaussian measurements for RIP) and image reconstruction. The core comparison is between WN+RGD and standard GD regarding the squared reconstruction error vs. iteration curves.

### Main Results: Linear Convergence of WN vs. Sublinear GD

| Setting | Key Parameter | Phenomenon | Conclusion |
|----------|----------|------|------|
| Different Condition Numbers $\kappa$ (Fig 2a) | $m=10, r=5, r_A=3, n=60000,\ \kappa\in\{50,75,100\}$ | After the saddle phase, WN+RGD converges linearly to 0, **independent of $\kappa$**; GD becomes stuck in a slower sublinear phase with much larger error. | Supports linear rate in Theorem 3.2. |
| Different Overparameterization $r$ (Fig 2b) | $m=300, r_A=5, \kappa=10,\ r\in\{50,75,100\}, n=50000$ | Under WN, larger $r$ leads to faster convergence and shorter saddle plateaus; standard GD slightly slows down with larger $r$. | Supports "overparameterization as a friend." |
| Full Rank $r=m$ (Fig 2c) | — | WN converges extremely fast in the full-rank case. | WN is robust to extreme overparameterization. |

### Saddle-to-Saddle Dynamics (Fig 1)

| Metric | Phenomenon | Theoretical Correspondence |
|--------|------|----------|
| Squared Error (1a) | Multi-level plateaus appear; each plateau = a saddle point. | Lemma 4.1 |
| Gradient Norm (1b) | Drops by several orders of magnitude at saddle points. | Small gradient characterization of saddles. |
| $\|X_t\Theta_t X_t^\top - A_\ell\|_F^2$ (1c) | Iteratively approaches the best rank-$\ell$ approximations $A_\ell$. | Incremental/Sequential Learning. |
| $\sigma_i(\Phi_t\Phi_t^\top)$, $\sigma_i(\Theta_t)$ (1d–1f) | Both direction and magnitude exhibit sequential learning patterns. | $\mathrm{Tr}(\Phi_t\Phi_t^\top)\to r_A$ |

### Key Findings
- **Saddle Essence = Incremental Learning**: Lemma 4.1 proves that saddle points of the population loss correspond exactly to the best rank-$\ell$ approximations $A_\ell$ of $A$. The algorithm learns $A$ "rank-by-rank." Each escape from a saddle point corresponds to leaving the neighborhood of some $A_\ell$.
- **Overparameterization Accelerates Saddle Escape**: The burn-in time upper bound decreases as $r$ increases, manifested in experiments as shorter plateaus and earlier entry into the linear convergence phase.
- **Insensitivity to Retraction Type**: Polar retraction is chosen for proof convenience, but QR and Cayley retractions show nearly identical numerical performance (Fig 3a); the method remains robust under noisy measurements (Fig 3b).

## Highlights & Insights
- **Translating WN into Riemannian Optimization**: The observation that direction variables naturally reside on the Stiefel manifold transforms an empirical question ("why does normalization work?") into a convergence analysis problem with mature manifold tools.
- **Quantifying "More is Faster"**: While the intuition of overparameterization usually involves computational costs, this work provides explicit formulas showing burn-in time and sample complexity **decrease polynomially** with $r$ (up to $O(r_A^4)$ for rate and $O(r_A^{12})$ for samples), turning WN into a calculable acceleration knob.
- **Transferable PSD-to-Symmetry Relaxation**: Merging $\tilde\Theta\tilde\Theta^\top$ and relaxing the PSD constraint maintains global optimality in the overparameterized regime while saving SVD costs. This idea of "using overparameterization to exchange constraints for relaxation" could be applied to other manifold/cone-constrained problems.

## Limitations & Future Work
- **Limited to Matrix Sensing**: Conclusions rely on symmetric low-rank PSD matrix sensing and the RIP assumption. Generalization to asymmetric, non-PSD, or general deep networks remains open. The practical benefits of WN in LoRA are used as motivation but not end-to-end verified.
- **Large Complexity Constants**: High-order factors like $m^7 r^9$ and $\kappa^4$ appear in the complexity bounds. While sufficient for theoretical characterization, they are far from practical constants, and the tightness of these bounds is unknown.
- **Dependence on Random Initialization and RIP**: Analysis provides high-probability conclusions and does not cover pathological cases (extremely large $\kappa$) or measurement operators where RIP fails. Experimental scale is also relatively small ($m$ up to 300).
- **Future Directions**: Extending the direction-magnitude decoupling + RGD framework to non-convex tensor decomposition or robust matrix completion to verify if overparameterization remains a "friend" across broader problems.

## Related Work & Insights
- **Vs. Standard GD (Stöger & Soltanolkotabi 2021; Xiong et al. 2024)**: Previous works showed that randomly initialized GD either only achieves constant error via early stopping or has a sublinear lower bound $\Omega(1/t)$. They also showed overparameterization slows down convergence and requires precise initialization. Oursachieves linear convergence with random initialization, where overparameterization actually accelerates the process—a direct exponential improvement.
- **Vs. Existing WN Theory (Wu et al. 2020; Chou et al. 2024; Cisneros-Velarde et al. 2025)**: Prior work analyzed implicit regularization in overparameterized least squares or implicit bias in diagonal networks. This work is the first to characterize **how WN utilizes overparameterization to gain faster convergence** in non-convex matrix sensing.
- **Vs. Riemannian Optimization (Absil et al. 2008; Boumal 2023)**: This work uses standard RGD notations and tools (tangent space projection + retraction). The contribution lies not in the optimizer itself, but in interfacing it with the WN decoupling structure and completing the two-stage convergence proof for matrix sensing.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First proof that normalization benefits from overparameterization; both perspective and conclusions are novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Synthetic and image experiments fully support the theory, though scale is small and primarily for verification.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical narrative, good geometric intuition for the two phases, high formula density but well-structured.
- Value: ⭐⭐⭐⭐ Provides rare theoretical support for WN / DoRA-style PEFT methods; testbed conclusions are highly insightful.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Testing Fourier Sparsity via Implicit Sensing](testing_fourier_sparsity_via_implicit_sensing.md)
- [\[ICLR 2026\] Gradient Descent Dynamics of Rank-One Matrix Denoising](gradient_descent_dynamics_of_rank-one_matrix_denoising.md)
- [\[ICLR 2026\] In-Context Algorithm Emulation in Fixed-Weight Transformers](in-context_algorithm_emulation_in_fixed-weight_transformers.md)
- [\[ICLR 2026\] Improved High-Dimensional Estimation with Langevin Dynamics and Stochastic Weight Averaging](improved_high-dimensional_estimation_with_langevin_dynamics_and_stochastic_weigh.md)
- [\[ICLR 2026\] Enabling Fine-Tuning of Direct Feedback Alignment via Feedback-Weight Matching](enabling_fine-tuning_of_direct_feedback_alignment_via_feedback-weight_matching.md)

</div>

<!-- RELATED:END -->
