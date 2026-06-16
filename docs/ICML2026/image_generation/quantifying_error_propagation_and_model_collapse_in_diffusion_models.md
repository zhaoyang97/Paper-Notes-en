---
title: >-
  [Paper Note] Quantifying Error Propagation and Model Collapse in Diffusion Models
description: >-
  [ICML 2026][Image Generation][Diffusion Model] This work provides the first set of paired upper and lower bounds for the phenomenon of "model collapse induced by recursive training with synthetic data" in score-based diffusion models. Specifically, it establishes that the single-generation divergence satisfies $\chi^2(\hat p^{i+1}\|q_i)\asymp \varepsilon_{\star,i}^
tags:
  - ICML 2026
  - Image Generation
  - Diffusion Model
  - score matching
  - Error Accumulation
date: 2026-05-08
content_hash: 535210698b4ee004
---
# Quantifying Error Propagation and Model Collapse in Diffusion Models

**Conference**: ICML 2026  
**arXiv**: [2602.16601](https://arxiv.org/abs/2602.16601)  
**Code**: None  
**Area**: Diffusion Models / Generative Model Theory  
**Keywords**: Diffusion models, model collapse, recursive training, score matching, error accumulation

## TL;DR
This work provides the first set of paired upper and lower bounds for the phenomenon of "model collapse induced by recursive training with synthetic data" in score-based diffusion models. Specifically, it establishes that the single-generation divergence satisfies $\chi^2(\hat p^{i+1}\|q_i)\asymp \varepsilon_{\star,i}^2$, and the multi-generation cumulative divergence $D_N$ is a weighted sum of previous score error energies decaying geometrically by $(1-\alpha)^{2m}$. This effectively transforms the empirical observation that "adding fresh data alleviates collapse" into a precise decay law.

## Background & Motivation
**Background**: Current generative AI increasingly relies on synthetic data self-training (self-training, self-improving diffusion, etc.). However, it has been repeatedly observed that if the proportion of synthetic data in the training distribution is too high, the model significantly degrades after multiple recursive rounds—exhibiting tail loss, diversity collapse, and overall distribution drift. This phenomenon is collectively termed model collapse. Theoretical works primarily focus on regression models or parametric MLE estimation; for diffusion models, only a few architecture-specific upper bounds for two-layer score networks exist (Fu et al. 2024; Cui et al. 2026), and these provide **only upper bounds**.

**Limitations of Prior Work**: Pure upper bounds cannot answer the question of "what is the minimum error"—an upper bound of zero does not imply the model does not collapse. Furthermore, existing upper bounds rely on the path energy $\hat\varepsilon_i^2$ of the learned model, which cannot be directly linked to the training objective (the score-matching loss $\varepsilon_{\star,i}^2$ on the ideal path), leading to a disconnect between theory and experimental observations. Additionally, existing theories rarely involve the most practical knob: how the fresh data ratio $\alpha$ precisely suppresses error.

**Key Challenge**: Each generation of training involves two opposing forces—"error dilution" brought by fresh real data and "error injection" brought by imperfect score learning. Quantifying collapse requires simultaneously characterizing both forces and decoupling them into an interpretable recurrence.

**Goal**: In a population-level, architecture-agnostic scenario, this work aims to answer two sub-questions for the recursive training pipeline $\hat p^i \to q_i = \alpha p_{\text{data}} + (1-\alpha)\hat p^i \to \hat p^{i+1}$: (a) How is the single-generation divergence $I_i = \chi^2(\hat p^{i+1}\|q_i)$ characterized by score error? (b) How does the multi-generation cumulative divergence $D_i = \chi^2(\hat p^i\|p_{\text{data}})$ evolve over generations?

**Key Insight**: The authors use Girsanov measure transformation to map the drift error of the reverse SDE onto the Radon-Nikodym derivative of the path measure, which is then projected to the marginal likelihood ratio $R_i(\mathbf x) = \mathbb E_{\mathbb P^\star_i}[e^{Z_T^i}\mid \mathbf Y_{t_0}=\mathbf x]$ at the terminal time $t_0$. The problem then becomes "how much of the path error is preserved during marginalization"—which is precisely an observability problem.

**Core Idea**: An "error observability coefficient" $\eta_i\in[0,1]$ is introduced to measure how much of the path score error falls onto the terminal state. Consequently, the single-generation lower bound becomes $I_i\gtrsim \eta_i\cdot\varepsilon_{\star,i}^2$. By accumulating this single-generation estimate along the exact contraction formula for $\chi^2$ divergence for the refresh step $q_i=\alpha p_{\text{data}}+(1-\alpha)\hat p^i$, i.e., $\chi^2(q_i\|p_{\text{data}})=(1-\alpha)^2 \chi^2(\hat p^i\|p_{\text{data}})$, the $(1-\alpha)^{2m}$ geometric discount decay law naturally emerges.

## Method

### Overall Architecture
This paper addresses whether and at what speed the recursive training pipeline $\hat p^i \to q_i = \alpha p_{\text{data}} + (1-\alpha)\hat p^i \to \hat p^{i+1}$ leads to model collapse under the tension between fresh data dilution and imperfect score error injection. The authors decompose the problem into two levels: First, on a single-generation scale, Girsanov measure transformation is used to map drift errors of the reverse SDE to marginal likelihood ratios. A new "observability" coefficient is introduced to sandwich the single-generation divergence $I_i=\chi^2(\hat p^{i+1}\|q_i)$. Second, on a multi-generation scale, this single-generation estimate is substituted into the exact $\chi^2$ contraction of the refresh step to derive the geometric discount decomposition of the cumulative divergence $D_N$. Both sets of conclusions hold strictly within the "small score error perturbation" region where $\varepsilon_{\star,i}^2\le 1$.

### Key Designs

**1. Observability Coefficient $\eta_i$: Translating Path Error to Marginal Divergence**

The fundamental difficulty in diffusion model analysis is that non-zero score errors along the path do not necessarily cause a shift in the marginal distribution at the terminal time $t_0$ of the reverse diffusion—errors might be averaged out during marginalization. Previously, researchers could only bypass this via upper bounds. The authors introduce a scalar $\eta_i\in[0,1]$ to directly quantify "how much path error manifests in the terminal state." Specifically, the stochastic integral $M_T^i = -\int_{t_0}^T \mathbf e_{i,s}\cdot \mathrm d\bar{\mathbf B}_s$ (coupling the path score error $\mathbf e_{i,s}$ with the reverse Brownian motion) is defined. By Itô isometry, $\mathrm{Var}_{\mathbb P^\star_i}(M_T^i)=\varepsilon_{\star,i}^2$. The "proportion of variance retained by the terminal state" is defined as $\eta_i = \mathrm{Var}_{\mathbb P^\star_i}(\mathbb E[M_T^i\mid \mathbf Y_{t_0}]) / \varepsilon_{\star,i}^2$. The intuition is clear: perturbations coupled with sample states (e.g., $\mathbf e_{i,t}(\mathbf x)=\mathbf w\mathbf x+\xi(t)$) leave an imprint on the terminal state, $\eta_i>0$; purely time-dependent or path-orthogonal perturbations are averaged out by the conditional expectation, $\eta_i=0$. With this, a lower bound is established for the first time as $I_i\ge \tfrac14\eta_i\varepsilon_{\star,i}^2 - C\varepsilon_{\star,i}^4$, directly linking "path error" to "marginal divergence." Furthermore, $\eta_i$ can be numerically estimated on real data like CIFAR-10 and is almost always greater than 0.

**2. Girsanov Sandwich for Single-Generation Equivalence $I_i\asymp \varepsilon_{\star,i}^2$ (Theorem 3.5): Providing Directly Monitorable Proxies**

Previous results either only provided KL upper bounds or used learned-path energy $\hat\varepsilon_i^2$ (which is inaccessible in practice), failing to align with training objectives. This work sandwiches the single-generation divergence using the score matching loss $\varepsilon_{\star,i}^2$ on the ideal path within the small-error region. The upper bound follows from Girsanov + data processing to obtain $\mathrm{KL}(\hat p^{i+1}\|q_i)\le \tfrac12\hat\varepsilon_i^2$, while the lower bound comes from the observability argument $\chi^2(\hat p^{i+1}\|q_i)\ge \tfrac14\eta_i\varepsilon_{\star,i}^2-C\varepsilon_{\star,i}^4$. To unify both sides under ideal-path energy, a key technique proves that $\varepsilon_{\star,i}^2$ and $\hat\varepsilon_i^2$ are equivalent under assumptions of Girsanov density $L^{1+\delta}$-integrability (A3) and quadratic variation moment conditions (A4), while $\chi^2$ and KL differ only by a constant in the perturbation region. Combined, this yields $\tfrac14\eta_i\varepsilon_{\star,i}^2 - C\varepsilon_{\star,i}^4 \le \chi^2(\hat p^{i+1}\|q_i)\le 4\varepsilon_{\star,i}^2 + c\varepsilon_{\star,i}^4$. Since both sides are expressed via ideal-path energy corresponding to training objectives, the theory can be directly verified by experiments—Figure 4 shows both $\chi^2$ and KL divergences being sandwiched by $\varepsilon_{\star,i}^2$ on a 10D GMM.

**3. Multi-Generation Geometric Discount Decomposition $D_N \asymp \sum (1-\alpha)^{2(N-i)}\varepsilon_{\star,i}^2$ (Theorem 4.2): Explaining Why Fresh Data Prevents Collapse**

Quantifying the empirical rule "larger $\alpha$ leads to stronger suppression" has been lacking. This work accumulates single-generation equivalence along generations to obtain a precise decay law. The core algebraic advantage is that the $\chi^2$ divergence of the refresh step satisfies an exact equality $\chi^2(q_i\|p_{\text{data}})=(1-\alpha)^2\chi^2(\hat p^i\|p_{\text{data}})$ (Lemma F.1)—this is precisely why $\chi^2$ was chosen over KL, as KL lacks such clean quadratic contraction. By recursing this with the single-gen equivalence $\chi^2(\hat p^{i+1}\|q_i)\asymp \varepsilon_{\star,i}^2$ and adding a tail assumption (A5) on an adaptive "good set" $\mathcal G_i$ (to prevent synthetic models from piling mass in regions where $p_{\text{data}}$ is extremely small), one obtains $D_{N+1}+C_{\text{bias}}\asymp \sum_{i=i_0}^N (1-\alpha)^{2(N-i)}\varepsilon_{\star,i}^2 + (1-\alpha)^{2(N+1-i_0)}D_{i_0}$. Proposition 4.1 further provides a converse: if $\sum_i \varepsilon_{\star,i}^2=\infty$ or a score-error lower bound exists, $\limsup D_i$ will not vanish, and the model must collapse. This conclusion transforms empirical facts into an explicit $(1-\alpha)^{2m}$ decay—errors from $m$ generations ago are compressed by $(1-\alpha)^{2m}$, equivalent to an effective memory of $\sim 1/\alpha$, while providing engineering guidance: as long as $\sum \varepsilon_{\star,i}^2<\infty$ and $\alpha>0$, $D_N$ remains stable without requiring errors to converge to zero every generation.

### Loss & Training
No new training loss is proposed. The theory is built upon standard variance-preserving OU forward SDE $\mathrm d\mathbf X_t = -\tfrac12\mathbf X_t\mathrm dt + \mathrm d\mathbf B_t$ and its reverse SDE, using the score matching loss $\varepsilon_{\star,i}^2 = \mathbb E_{\mathbb P^\star_i}[\int_{t_0}^T \|\mathbf e_{i,s}(\mathbf Y_s)\|_2^2 \mathrm ds]$. Minimax-optimal score estimation errors satisfy $\varepsilon_{\star,i}^2 \lesssim \mathrm{polylog}(n_i)\,n_i^{-1}(1/t_0)^{d/2}$, meaning sample size must grow exponentially with environmental dimension $d$ to ensure the perturbation region holds; however, under low-dimensional manifold assumptions, it only depends on intrinsic dimension $d^\star\ll d$.

## Key Experimental Results

### Main Results
The authors validate the theory using three datasets: 10D Gaussian Mixture (5 components, $\sigma^2\mathbf I_{10}$), Fashion-MNIST, and CIFAR-10. All experiments use PCA projection to 2D for visualization and directly estimate $\eta_i$, $\varepsilon_{\star,i}^2$, and $\chi^2$/KL divergences.

| Dataset | Objective | Result |
|--------|---------|------|
| 10D Gaussian Mixture ($\alpha\in\{0.1,0.5,0.9\}$, 20 gen) | Impact of $\alpha$ on collapse speed (Fig.1) | Distribution continues to drift at $\alpha=0.1$; retains structure but widens at $\alpha=0.5$; remains stable throughout at $\alpha=0.9$ |
| 10D GMM (20 gen) | Single-gen bounds (Prop 3.1 + 3.3) | $\mathrm{KL}(\hat p^{i+1}\|q_i)\le \tfrac12\hat\varepsilon_i^2$ fits tightly; $\chi^2\ge \tfrac18\hat\eta_i\varepsilon_{\star,i}^2$ is validated (Fig.3) |
| 10D GMM (20 gen, $\alpha\in\{0.1,0.5\}$) | Two-sided equivalence Thm 3.5 | Both $\chi^2$ and KL are sandwiched between $\tfrac14\hat\eta_i\varepsilon_{\star,i}^2$ and $4\varepsilon_{\star,i}^2$ (Fig.4) |
| 10D GMM (20 gen) | Geometric discount decomposition Thm 4.2 | Broad contribution at $\alpha=0.1$; only recent generations contribute at $\alpha=0.9$; diagonal structure is clear (Fig.5) |
| Fashion-MNIST | $\alpha$ effect in real images (Fig.8/10) | Consistent with GMM: high $\alpha$ is stable, low $\alpha$ leads to collapse after multiple generations |
| CIFAR-10 | Existence of observability $\eta_i$ in real data (Fig.2/9) | State-dependent perturbations (aligned / random) yield clear $\hat\eta_i>0$; time-only perturbations yield $\hat\eta_i\approx 0$ |

### Ablation Study
The paper lacks traditional component ablation (being purely theoretical and validation-based), but Fig. 2 "ablates" score error types on CIFAR-10 to reveal the source of $\eta_i$:

| Perturbation Type | Form of $\mathbf e_{i,t}(\mathbf x)$ | Estimated $\hat\eta_i$ | Description |
|----------|-----------------------------------|-------------------|------|
| Aligned (with drift) | $\mathbf w_i \mathbf x$, $\mathbf w_i$ along drift | Highest | Error is amplified by reverse trajectory, leaving a strong imprint on terminal state |
| Random (stochastic direction) | $\mathbf w\mathbf x$, $\mathbf w$ random | Medium | Still retains state dependency, $\eta_i$ significantly greater than 0 |
| Time-only | $\xi(t)$, state-independent | Near 0 | Conditional expectation averages it out; no impact on marginals |

### Key Findings
- **The $(1-\alpha)^{2m}$ decay law is the core actionable conclusion**: To stabilize cumulative divergence, engineers do not need each generation's score error to reach zero; they only need $\sum \varepsilon_{\star,i}^2<\infty$ and a fresh data ratio $\alpha>0$. The effective memory window is approximately $1/\alpha$ generations.
- **State-dependency determines collapse visibility**: Purely time-dependent perturbations do not lead to observed divergence growth ($\eta_i\approx 0$). However, since initializations and optimization noise in actual neural score models almost inevitably introduce state-dependent perturbations, model collapse is a "universal phenomenon" in practice.
- **$\chi^2$ divergence is a critical technical choice**: The exact equality $\chi^2(q_i\|p_{\text{data}})=(1-\alpha)^2\chi^2(\hat p^i\|p_{\text{data}})$ for the refresh step is the algebraic root that allows geometric discount recurrence to hold (which KL lacks). Within the perturbation region, $\chi^2$ and KL are equivalent (verified in Fig. 4), so lower bound conclusions also transfer to KL.
- **The first stage ($i<i_0$) allows for deviation**: The theory assumes stability after the first $i_0$ generations. The deviation at $i=1$ in Fig. 3 does not violate the theory but corresponds to a "transient state"—in practice, stability is usually reached after one or two generations.

## Highlights & Insights
- **The proposal of observability coefficient $\eta_i$ is truly novel**: It quantifies the long-standing "path error $\to$ marginal divergence" gap in diffusion model analysis with a single scalar that can be estimated from data—this is the most striking design of the paper.
- **Sophisticated choice of $\chi^2$ divergence**: The authors deliberately chose $\chi^2$ over KL because the refresh step $q_i=\alpha p_{\text{data}}+(1-\alpha)\hat p^i$ satisfies clean quadratic contraction under $\chi^2$. This observation is directly transferable to other "mix-and-retrain" frameworks (e.g., mixing with a reference policy in RLHF).
- **Architecture-agnostic population-level perspective**: Unlike existing theories that typically assume two-layer score networks, this work starts from path measures. This strategy of "reducing to SDE measure ratios and using Girsanov projection" can be applied to other score-flavor generative models like flow matching or consistency models.
- **Experimental validation of lower bounds**: Unlike many purely theoretical works, all theorems provide estimable constants and quantities. The authors substitute estimated $\hat\eta_i$ and $\hat\varepsilon_{\star,i}^2$ into the bounds on GMM/Fashion-MNIST/CIFAR-10 to compare with actual divergences—this "theory-experimental loop" is rare in model collapse literature.

## Limitations & Future Work
- **Only applicable to the small score error perturbation region**: All results assume $\varepsilon_{\star,i}^2\le 1$. Eq. (12) suggests high-dimensional data requires sample sizes $n_i\sim (1/t_0)^{d/2}$ to reach this region; actual large models might operate outside this. The authors explicitly list "lower bounds for the large error region" as an open problem.
- **Neglects discretization error and initialization bias**: The theory uses continuous-time reverse SDEs and assumes starting from $\mathcal N(0,\mathbf I_d)$ (using OU exponential convergence to absorb errors). Real diffusion models use discrete-step samplers, and these errors remain unquantified.
- **Lack of a priori guarantees for the lower bound of $\eta_i$**: The paper only notes that it is almost always $>0$ in practice (validated on CIFAR-10). There is no theoretical criterion for when $\eta_i$ might collapse to zero for more complex architectures (e.g., transformer-based DiT), which is a key gap for applying the theory to SOTA models.
- **Does not answer if a limit distribution exists**: The authors state that whether recursive training converges to an $\alpha$-dependent limit distribution remains an open question. Current conclusions only guarantee whether $D_i$ is bounded or unbounded without characterizing its limit form.
- **Potential improvements**: Linking $\eta_i$ explicitly to score network architecture, activation functions, and initialization distributions could provide engineering guidance on which architectures are more resistant to collapse. Additionally, extending analysis to multi-stage training (different mixing ratios $\alpha_k$ for different generation generations) might directly serve real self-improving pipelines.

## Related Work & Insights
- **vs Fu et al. (2024) / Cui et al. (2026)**: They provide architecture-specific upper bounds for two-layer score networks. This work provides architecture-agnostic population-level upper and lower bounds—filling the lower bound gap and covering a wider range, though its finite-sample rates are less specific.
- **vs Bertrand et al. (2024)**: Also studying generative model stability, they take an MLE route (targeting parametric distributions). This paper focuses on score-based diffusion, using completely different technical tools (Girsanov rather than fixed points on parameter space).
- **vs Gerstgrasser et al. (2024)**: They empirically found that accumulating real and synthetic data can block collapse. This paper provides a quantitative explanation for this phenomenon via the $(1-\alpha)^{2m}$ decay law, turning empirical observations into calculable decay rates.
- **vs Chen et al. (2023c) / Benton et al. (2024)**: These are classic KL upper bounds for diffusion sampling convergence. This paper uses the same Girsanov tools but conversely provides $\chi^2$ lower bounds—a dual application of the toolkit that can be migrated to any "single-step SDE error $\to$ marginal divergence" analysis.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First distribution divergence lower bound for diffusion models + observability coefficient + geometric discount decomposition.
- Experimental Thoroughness: ⭐⭐⭐ Pure theory + validation experiments on GMM/Fashion-MNIST/CIFAR-10. Sufficient for a theory paper but lacks alignment with SOTA self-consuming experimental settings.
- Writing Quality: ⭐⭐⭐⭐ Clear structure. Reasonable explanations for assumptions A1-A5. Decomposing the Girsanov argument into observability and ratio control is pedagogically valuable.
- Value: ⭐⭐⭐⭐ Provides actionable engineering conclusions ($\alpha>0$ + summable error energy ensures stability) and clear open problems (large error region bounds, discretization errors, limit distributions). Significant for both diffusion model theory and self-training design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Test-Time Iterative Error Correction for Efficient Diffusion Models](../../ICLR2026/image_generation/test-time_iterative_error_correction_for_efficient_diffusion_models.md)
- [\[ICML 2026\] Escaping Mode Collapse in LLM Generation via Geometric Regulation](escaping_mode_collapse_in_llm_generation_via_geometric_regulation.md)
- [\[ICML 2026\] Alignment-Guided Score Matching for Text-to-Image Alignment in Diffusion Models](alignment-guided_score_matching_for_text-to-image_alignment_in_diffusion_models.md)
- [\[ICML 2026\] A Unified Framework for Diffusion Model Unlearning with f-Divergence](a_unified_framework_for_diffusion_model_unlearning_with_f-divergence.md)
- [\[ICLR 2026\] Error as Signal: Stiffness-Aware Diffusion Sampling via Embedded Runge-Kutta Guidance](../../ICLR2026/image_generation/error_as_signal_stiffness-aware_diffusion_sampling_via_embedded_runge-kutta_guid.md)

</div>

<!-- RELATED:END -->
