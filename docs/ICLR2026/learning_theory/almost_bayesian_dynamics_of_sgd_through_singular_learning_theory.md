---
title: >-
  [Paper Note] Almost Bayesian: Dynamics of SGD Through Singular Learning Theory
description: >-
  [ICLR 2026][learning_theory][Paper Note] This paper describes long-run SGD as diffusion in porous media on a singular loss landscape, characterizes the geometric complexity of reachable low-loss regions using the local learning coefficient, and derives that the steady-state distribution of SGD is approximately equal to a Bayesian posterior corrected by reacha
tags:
  - ICLR 2026
  - learning_theory
date: 2026-05-08
content_hash: b5d0d2a518169d8e
---
# Almost Bayesian: Dynamics of SGD Through Singular Learning Theory

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=5ebDXlue3d](https://openreview.net/forum?id=5ebDXlue3d)  
**Code**: No open-source code seen  
**Area**: Learning Theory / SGD Dynamics  
**Keywords**: SGD dynamics, Singular Learning Theory, local learning coefficient, fractional Fokker-Planck equation, Bayesian posterior  

## TL;DR
This paper describes long-run SGD as diffusion in porous media on a singular loss landscape, characterizes the geometric complexity of reachable low-loss regions using the local learning coefficient, and derives that the steady-state distribution of SGD is approximately equal to a Bayesian posterior corrected by reachability temperature.

## Background & Motivation
**Background**: There are two long-standing parallel lines in deep learning theory. One starts from optimization dynamics, viewing SGD as a noisy gradient flow, often described by Langevin or Fokker-Planck equations to show how the parameter distribution moves over time; the other starts from Bayesian statistics, using Singular Learning Theory (SLT) to explain why non-identifiable models with degenerate Hessians like neural networks can still generalize.

**Limitations of Prior Work**: Traditional SGD-Bayes connections typically rely on approximately quadratic local minima, such as approximating SGD as an Ornstein-Uhlenbeck process to derive a Gaussian posterior interpretation. However, loss basins in neural networks are highly degenerate: many parameterizations correspond to the same function, the Hessian has many directions near zero, and local shapes are not regular paraboloids. Consequently, BIC, quadratic approximations, and ordinary Brownian diffusion in regular models struggle to explain the late-stage trajectories of real SGD.

**Key Challenge**: Empirically, solutions found by SGD seem related to the Bayesian posterior, yet SGD does not sample freely across the entire parameter space. It is constrained by initialization, noise scale, low-loss connectivity structures, and degenerate directions, allowing it to visit only specific local regions. The problem becomes: if neural networks are singular models, is SGD sampling the Bayesian posterior, or a version distorted by dynamical reachability?

**Goal**: The authors aim to provide a theoretical framework that answers three questions simultaneously: first, why late-stage SGD often exhibits sub-diffusion rather than ordinary Brownian diffusion; second, how the local learning coefficient from SLT enters the SGD trajectory; and third, under what conditions the steady-state distribution of SGD can establish an explicit relationship with the Bayesian posterior.

**Key Insight**: The key observation is that a degenerate loss landscape can be likened to a porous medium. Low-loss parameters are not a regular Euclidean ball but rather reachable regions connected by narrow channels, flat directions, and bottlenecks. The local learning coefficient $\lambda(w)$ characterizes how the "good parameter volume" shrinks with respect to the error threshold, while the spectral dimension $d_s$ describes how fast the SGD diffusion process actually explores these reachable states.

**Core Idea**: Use a fractional Fokker-Planck equation to characterize anomalous diffusion in late-stage SGD, then translate the local learning coefficient from SLT into an effective diffusion coefficient, thereby demonstrating that the SGD steady state is a reachability-weighted version of the Bayesian posterior.

## Method

### Overall Architecture
This paper does not propose a new optimizer but rather a theoretical model to explain the long-term behavior of SGD. The overall approach can be summarized as: starting from the noise dynamics of standard SGD and noting that ordinary Langevin/Brownian models cannot explain late-stage sub-diffusion; using a fractional Fokker-Planck equation to describe anomalous diffusion with memory; subsequently using the local learning coefficient $\lambda(w)$ from SLT to describe the porous geometry of low-loss regions; and finally synthesizing the geometric and spectral dimension $d_s$ into an effective diffusion coefficient $D_\xi(w)$ to solve for the local steady-state distribution and link it to the Bayesian posterior.

Put more simply, the paper deconstructs "where SGD goes" into two questions. The geometric question asks: how many approximately equivalent low-loss parameters are in a local region? The dynamical question asks: how many of these parameters can SGD noise and gradients visit within a finite time? The former is controlled by the local learning coefficient, while the latter is controlled by the spectral and walk dimensions; together, they determine the steady-state weights of SGD across different loss basins.

This framework focuses on the late stages of training rather than the violent drifts near initialization. The authors acknowledge that super-diffusion may occur early in SGD, but as long as the probability mass does not vanish during training, the steady-state solution is primarily determined by the long-duration sub-diffusion phase. Thus, the model focuses on the regime of small learning rates and large batches near critical points: here, gradient noise no longer dominates, and the degenerate low-loss structure begins to dictate the parameter distribution.

### Key Designs
**1. Fractional Fokker-Planck: Rewriting Late-stage SGD as Sub-diffusion with Memory**

The standard Langevin perspective writes SGD as a stochastic differential equation similar to $dw/dt=-\gamma \nabla L(w)+\Sigma_w$, where the displacement scale is typically Brownian $R(t)\propto t^{1/2}$. However, this paper and existing experiments observe that while early training may be super-diffusive, the later stages often turn into sub-diffusion where $R(t)\propto t^{1/d_{walk}}$ with $d_{walk}>2$, or even ultra-slow diffusion approaching $R(t)\propto \log t$. Ordinary Fokker-Planck equations cannot naturally express this memory effect.

The authors therefore replace the time derivative with the Caputo fractional derivative $D_t^\alpha$, obtaining the fractional Fokker-Planck equation for SGD:

$$
D_t^\alpha p(w,t)=\nabla\cdot\left(D(w,t)\nabla p(w,t)-\gamma p(w,t)\nabla L_m[w]\right).
$$

Here $p(w,t)$ is the parameter distribution, $D(w,t)$ is the diffusion coefficient, $\gamma$ is a coefficient similar to friction or learning rate scale, and $L_m$ is the empirical loss. The role of the fractional derivative is not decorative; it makes the current change dependent on the power-law memory of the past trajectory, which exactly corresponds to the phenomenon of SGD being slowed down by bottlenecks, plateaus, and local phase transitions in degenerate basins.

**2. Local Learning Coefficient: Replacing Parameter Dimension with SLT Effective Dimension**

In regular statistical models, the complexity term in BIC comes from the quadratic volume near a non-degenerate Hessian minimum, with complexity primarily controlled by the parameter dimension $d/2$. However, neural networks are singular models where the Hessian is often degenerate and there are many equivalent parameterizations; the volume of the low-loss set cannot be described by ordinary ellipsoid volume. The local learning coefficient (LLC) from SLT serves this exact problem.

The paper uses a local singular integral to characterize the volume of the low-loss region near a parameter $w^*$:

$$
V(\epsilon)=\int_{B_r(w^*,\epsilon)}\rho(w)dw,
$$

and defines the local learning coefficient through the approximate scaling relationship $V(\epsilon)\propto \epsilon^{\lambda(w^*)}$. Intuitively, a smaller $\lambda(w)$ indicates that the volume of low-loss parameters in that region is "thicker," more degenerate, and the local effective complexity of the model is lower. The key translation in this paper is viewing $\lambda(w)$ as the local mass dimension in a porous medium—how the low-loss pore volume shrinks with the error height.

This interpretation bridges SLT from static generalization theory to SGD dynamics. LLC is no longer just an indicator of "how complex this solution is," but a geometric constraint on whether SGD can move, stay, or visit other low-loss states locally.

**3. Spectral and Walk Dimensions: Distinguishing "Number of Good Parameters" from "Reachability by SGD"**

$\lambda(w)$ alone is insufficient because a large low-loss volume does not equate to easy exploration by SGD. A basin can be wide but contain narrow channels, many bottlenecks, and complex paths, leading to slow diffusion. The paper thus introduces the spectral dimension $d_s$ to describe the growth of the state volume actually occupied by the diffusion process within time $t$:

$$
V_s(t)\sim t^{d_s/2}.
$$

The walk dimension $d_{walk}$ describes the displacement scale:

$$
R(t)\sim t^{1/d_{walk}}.
$$

Near critical points where the local porous structure is stable, the authors borrow the Alexander-Orbach type relationship to link these to the LLC:

$$
d_{walk}(t)=\frac{2\lambda(w_t)}{d_s}.
$$

The meaning of this relationship is significant: LLC accounts for the "geometric capacity of the low-loss region," while the spectral dimension accounts for the "reachable capacity seen by SGD dynamics." If $d_s$ is small, SGD might only crawl slowly even in a large flat region; a higher relative spectral dimension indicates more thorough exploration within the same low-loss region.

**4. Reachability Tempering: Writing SGD Steady State as a Dynamically Corrected Bayesian Posterior**

To find the steady state, the paper simplifies the position-dependent, anisotropic diffusion tensor into a tractable scalar diffusion coefficient. The authors argue that in the regime of large batches, small learning rates, and late-stage training, most eigenvalues of the Hessian and diffusion tensor are near 0, allowing the effective diffusion tensor to be approximated as a low-rank or even scalar function. Choosing a coarse-graining scale $\xi$ yields the effective diffusion coefficient:

$$
D_\xi(w)=\xi^{2-2\lambda(w)/d_s}.
$$

When $D_\xi$ is approximately constant within a local region $W$, the fractional Fokker-Planck equation reduces to the ordinary steady-state Fokker-Planck equation, with the solution:

$$
p_s(w)\propto \exp\left(-\frac{\gamma L_m[w]}{D_\xi}\right).
$$

If $L$ is the log loss and $\gamma=1$ for simplicity, the authors further obtain:

$$
p_s(w)^{mD_\xi}\propto p(X_m|w),
$$

leading to

$$
p(w|X_m)=\frac{\rho(w)p_s(w)^{mD_\xi}}{Z_{mD_\xi}}.
$$

This is the source of the "Almost Bayesian" title. SGD does not naively sample the Bayesian posterior; it generates a steady-state distribution restricted by local reachability. Only after correcting this steady-state distribution by temperature $mD_\xi$ does it align with the Bayesian likelihood/posterior. Low LLC regions attract SGD solutions more easily, but the final probability must be corrected for reachability determined by the spectral dimension and coarse-graining scale.

### Mechanism Example
One can think of the model as many identical networks on a 2D moons classification task. Each network starts from a different initialization and trains to a low-loss region using SGD. After training, these solutions fall into several parameter space clusters: some are frequently visited by SGD, while others have non-negligible probability in the Bayesian posterior but are difficult for SGD to reach from common initializations and noise scales.

While traditional accounts might ask: "Are SGD samples and the SGLD-approximated Bayesian posterior the same?", this paper provides a more granular answer: first estimate the LLC near each SGD solution to observe if SGD favors low LLC regions; then sample the local Bayesian posterior near low-loss, low-LLC solutions using SGLD; finally, temper the SGD steady-state probability according to $D_\xi$. In experiments, with $\xi=0.5$, the tempered SGD distribution almost overlaps with the SGLD-approximated posterior in cluster concentration, suggesting the difference stems primarily from dynamical reachability rather than lack of relationship.

This example also explains why "Is SGD a Bayesian sampler?" cannot be answered with a simple yes/no. Raw SGD samples favor certain basins that are easier to access dynamically; however, if one knows the local geometry and reachability scale of these basins, this bias can be corrected back to a distribution close to the Bayesian posterior.

### Loss & Training
The paper does not propose a new training loss. Theoretical analysis defaults to empirical loss $L_m[w]$. When linking to the Bayesian posterior, log loss or equivalent KL divergence is primarily considered, as $e^{-mL_m[w]}$ can then be interpreted as the likelihood $p(X_m|w)$.

Training and estimation strategies serve to validate theoretical hypotheses. LLC is estimated using the estimator from Lau et al. and the devinterp toolchain, with the core form:

$$
\hat{\lambda}(w^*)=\frac{n}{\log n}\left(E_{w\mid B_r(w^*)}[L_n(w)]-L_n(w^*)\right).
$$

The spectral dimension is estimated from a power-law fit of weight displacement. The authors record total displacement $R(t)$ and use linear regression on:

$$
\log R(t)=\frac{d_s}{2\lambda(w)}\log t+c
$$

to obtain $d_s$ and the fit quality $r^2$. Thus, the "training strategy" in experiments is essentially: allowing models to enter a sufficiently long late-stage training phase, periodically estimating LLC and displacement, and checking if $d_s \leq \bar{\lambda}$, displacement predictions, and the tempered posterior hold true.

## Key Experimental Results

### Main Results
The authors validate the theory in three settings: fully connected ReLU networks on MNIST, small language models on TinyStories, and fine-tuning vision models on Tiny ImageNet. Main results focus on two questions: whether sub-diffusion predictions fit weight displacement, and whether the spectral dimension is bounded by the average LLC.

| Model / Setting | $\lambda$ | $d_s$ | $\alpha$ | $r^2$ | Description |
|--------|------|------|------|------|------|
| TinyStories-1M | 32 | 21.422 | 0.33 | 0.98 | Small LM continued training; sub-diffusion fits well |
| TinyLlama-15M | 76.1 | 48.3 | 0.32 | 0.98 | Larger LM; $d_s < \lambda$ still holds |
| TinyStories-33M | 39.3 | 38.7 | 0.49 | 0.98 | $d_s$ close to but slightly below LLC; stable fit |
| ResNet18 | 72.05 | 0.57 | 0.004 | ~1 | Adam then low-LR SGD fine-tuning; near-exact fit in late SGD |
| ResNet34 | 73.5 | 0.62 | 0.004 | ~1 | Vision fine-tuning results consistent with theory |
| VGG16 | 159.7 | 0.14 | 0.001 | ~1 | High LLC but very low spectral dimension, indicating limited reachability |

The second set of experiments examines the relationship between "SGD steady state vs Bayesian posterior." The authors train 500 identical fully connected networks on the moons dataset, obtain solution clusters via SGD, approximate local Bayesian posteriors with SGLD, and compare tempered SGD distributions with Bayesian distributions.

| Metric | Bayes vs Tempered SGD | Meaning |
|------|---------|------|
| $K(\mathrm{Bayes}\Vert\mathrm{Tempered\ SGD})$ | 0.009 | Very small KL divergence; cluster probabilities are close |
| Wasserstein distance | 0.002 | Very small mass transport distance |
| Jensen-Shannon divergence | 0.003 | Very small symmetric distribution difference |
| Optimal coarse-graining scale | $\xi=0.5$ | Tempered SGD most closely matches SGLD posterior at this scale |

### Ablation Study
Ablations mainly examine how optimizers and hyperparameters affect the spectral dimension, LLC, and performance. On MNIST fully connected networks, SGD and Adam exhibit different dynamical structures: SGD aligns better with the LLC-displacement relationship established using original parameter metrics, while Adam behaves more complexly due to adaptive preconditioning changing the geometric metric.

| Configuration | $d_s$ Mean | $d_s$ Std Dev | Final $\lambda$ Mean | Final $\lambda$ Std Dev | Mean Test Acc |
|------|---------|------|------|------|------|
| Adam | 0.4061 | 0.9068 | 3.0957 | 5.7533 | 90.4297 |
| SGD | 7.8165 | 10.2494 | 12.5270 | 11.8393 | 94.0592 |

The authors also report several hyperparameter trends. For SGD, final displacement and average LLC are strongly correlated in the large-batch, low-learning-rate regime; the correlation of learning rate with the spectral dimension $d_s$ is more pronounced than with $\lambda$, consistent with the explanation that "learning rate changes dynamical reachability, while LLC characterizes local geometry." For Adam, the spectral dimension sometimes predicts performance better than LLC because Adam's adaptive preconditioning is equivalent to changing the Riemannian metric of the parameter space.

### Key Findings
- Late-stage SGD weight displacement is typically not ordinary Brownian diffusion but sub-diffusion described by $R(t)\propto t^{1/d_{walk}}$; $r^2$ values on LMs and vision models are mostly between 0.98 and ~1, suggesting strong explanatory power for the power-law model.
- Experiments support $d_s \leq \bar{\lambda}$: the spectral dimension is bounded by the average local learning coefficient, fitting the theoretical picture where "states reachable by SGD cannot exceed the geometric capacity of the low-loss region."
- SGD solutions favor regions with lower LLC, which is consistent with "low LLC corresponding to simpler, flatter, and more likely-to-generalize local structures."
- The raw SGD distribution is not identical to the Bayesian posterior, but after $D_\xi$ tempering, metrics like KL, Wasserstein, and JS in moons experiments are significantly reduced.
- Results for Adam are more complex and suggest the theory primarily applies to vanilla SGD or late-stage low-LR SGD; adaptive optimizers may require redefining LLC to match their metrics.

## Highlights & Insights
- Pushes the LLC of SLT from a "generalization complexity indicator" to an "SGD reachability metric." This step is valuable because it makes the LLC curves common in developmental interpretability more than just observables—they can now enter dynamical equations.
- The title "Almost Bayesian" is accurate. The paper does not exaggeratedly claim SGD is Bayesian sampling, but rather points out a dynamical temperature gap determined by local geometry, spectral dimension, and coarse-graining scale.
- The porous media analogy captures the essence of neural network loss landscapes: low-loss regions are not regular basins but reachable channels within high-dimensional degenerate structures. This analogy is finer than "flat minima" as it considers both volume and connectivity/access speed.
- The spectral dimension $d_s$ is a training dynamics indicator worth watching. it may reveal more about whether an optimizer is exploring, localizing, or stuck at a bottleneck than loss, Hessian, or LLC alone.
- Insights for transfer learning and LR scheduling: if an initialization area has low LLC but also low $d_s$, the model may not move enough within a wide basin; increasing LR or decreasing batch size might not be "generalization magic" but rather changing the reachable spectral dimension.

## Limitations & Future Work
- Theoretical reliance on late-stage steady-state approximation. Real SGD may produce non-equilibrium probability flows due to label noise, non-stationary data, or LR schedules, potentially violating the $D_t^\alpha p=0$ local steady-state assumption.
- Scalar diffusion coefficient approximation requires specific regimes (large batch, low LR, late stage). If gradient noise is strongly anisotropic or the model is in early super-diffusion/phase transition stages, the diffusion tensor might not be representable by a simple $D_\xi(w)$.
- Adaptive optimizers like Adam are not yet fully covered theoretically. The explanation is that Adam changes the parameter space metric, weakening the correspondence between original LLC and dynamics; future work needs to re-establish SLT metrics under the optimizer-induced metric.
- Selection of the coarse-graining scale $\xi$ remains empirical. While $\xi=0.5$ worked well in moons experiments, how to automatically select $\xi$ for different models, datasets, and training stages remains an open question.
- While experiments cover MNIST, TinyStories, Tiny ImageNet, and additional CIFAR results, they are still relatively small-scale or controlled experiments. Proving this theory explains the long-term training of large foundation models requires cheaper, more stable LLC and spectral dimension estimation methods.

## Related Work & Insights
- **vs Mandt et al. (SGD as Bayesian inference)**: Traditional approximations place SGD near non-degenerate quadratic minima to get a Gaussian posterior; this work handles the singular, degenerate, non-quadratic landscapes common in NNs, rewriting the posterior relationship with reachability tempering.
- **vs Watanabe’s Singular Learning Theory**: SLT originally explained Bayesian generalization error and WBIC; this paper embeds LLC into the SGD diffusion equation, allowing static complexity metrics to participate in training dynamics modeling.
- **vs Chen et al. (Anomalous Diffusion Observations)**: Prior work observed transitions from super-diffusion to sub-diffusion in deep training; this work provides a fractional Fokker-Planck and fractal dimension explanation, with quantitative experimental verification.
- **vs Flat Minima / Hessian Spectral Analysis**: Flat minima focus on curvature magnitude; this work distinguishes low-loss volume, reachable channels, and diffusion speed. LLC can be seen as a local complexity metric better suited for singular models than pure Hessian eigenvalues.
- **Training Diagnostic Insights**: In the future, $\lambda(t)$, $d_s(t)$, and displacement curves could be used as training diagnostic signals to identify emergence, grokking, bottlenecks in transfer learning, or to design more structured learning-rate schedules.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Linking fractional diffusion, porous media, spectral dimension, and SLT posterior is a highly distinct theoretical combination.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers toy posterior, MNIST, LMs, and vision fine-tuning, enough to support major claims, though large-scale validation remains limited.
- Writing Quality: ⭐⭐⭐⭐ Clear main line, with intuition and proofs in the appendix; the symbol and assumption transitions are fast, presenting a high bar for readers unfamiliar with SLT.
- Value: ⭐⭐⭐⭐⭐ This paper provides a better explanation for "why SGD is almost Bayesian" suited to the singularity of neural networks and opens new directions for using LLC in training dynamics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Optimizing Data Augmentation through Bayesian Model Selection](optimizing_data_augmentation_through_bayesian_model_selection.md)
- [\[ICLR 2026\] Learning Admissible Heuristics for A*: Theory and Practice](learning_admissible_heuristics_for_a_theory_and_practice.md)
- [\[ICLR 2026\] Reshaping Reasoning in LLMs: A Theoretical Analysis of RL Training Dynamics through Pattern Selection](reshaping_reasoning_in_llms_a_theoretical_analysis_of_rl_training_dynamics_throu.md)
- [\[ICLR 2026\] "Noisier" Noise Contrastive Estimation is (Almost) Maximum Likelihood](noisier_noise_contrastive_estimation_is_almost_maximum_likelihood.md)
- [\[ICLR 2026\] Sharp Asymptotic Theory for Q-Learning with LD2Z Learning Rate and Its Generalization](sharp_asymptotic_theory_for_q-learning_with_textttld2z_learning_rate_and_its_gen.md)

</div>

<!-- RELATED:END -->
