---
title: >-
  [Paper Note] Quantifying Error Propagation and Model Collapse in Diffusion Models
description: >-
  [ICML 2026][Image Generation][Diffusion models] This paper provides the first set of matching upper and lower bounds for the phenomenon of "model collapse caused by recursive training with synthetic data" in score-based…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Diffusion models"
  - "model collapse"
  - "recursive training"
  - "score matching"
  - "error accumulation"
date: 2026-05-08
content_hash: c4cc26a7268cf275
---

# Quantifying Error Propagation and Model Collapse in Diffusion Models

**Conference**: ICML 2026  
**arXiv**: [2602.16601](https://arxiv.org/abs/2602.16601)  
**Code**: None  
**Area**: Diffusion Models / Generative Modeling Theory  
**Keywords**: Diffusion models, model collapse, recursive training, score matching, error accumulation

## TL;DR
This paper provides the first set of matching upper and lower bounds for the phenomenon of "model collapse caused by recursive training with synthetic data" in score-based diffusion models: single-generation divergence $\chi^2(\hat p^{i+1}\|q_i)\asymp \varepsilon_{\star,i}^2$, and multi-generation cumulative divergence $D_N$ is a weighted sum of past score error energies geometrically discounted by $(1-\alpha)^{2m}$. This formalizes the empirical fact that "adding fresh data mitigates collapse" into a precise decay law.

## Background & Motivation
**Background**: Current generative AI increasingly relies on synthetic data for self-training (self-training, self-improving diffusion, etc.), but it has been repeatedly observed that once the proportion of synthetic data in the training distribution is too high, the model significantly degrades after multiple recursive rounds—losing tails, collapsing diversity, and shifting the overall distribution, collectively known as model collapse. Theoretical work has mostly focused on regression models or parametric MLE estimation, with only a few architecture-specific upper bounds for diffusion models using two-layer score networks (Fu et al. 2024; Cui et al. 2026), and these provide **only upper bounds**.

**Limitations of Prior Work**: Pure upper bounds cannot answer the question of "what is the minimum error"—a zero upper bound does not imply no collapse. Moreover, existing upper bounds depend on the learned-path energy $\hat\varepsilon_i^2$, which cannot be directly linked to the training objective (score-matching loss on the ideal path $\varepsilon_{\star,i}^2$), leading to a disconnect between theory and experimental observations. Additionally, existing theories rarely involve the most practical knob: "how the fresh data ratio $\alpha$ precisely suppresses error."

**Key Challenge**: Each generation of training involves two opposing forces: "error dilution" brought by fresh real data and "error injection" brought by imperfect score learning. Quantifying collapse requires simultaneously characterizing both and decoupling them into an interpretable recurrence.

**Goal**: In a population-level setting independent of specific network architectures, for the recursive training pipeline $\hat p^i \to q_i = \alpha p_{\text{data}} + (1-\alpha)\hat p^i \to \hat p^{i+1}$, answer two sub-questions: (a) How is the single-generation divergence $I_i = \chi^2(\hat p^{i+1}\|q_i)$ characterized by score error? (b) How does the multi-generation cumulative divergence $D_i = \chi^2(\hat p^i\|p_{\text{data}})$ evolve over generations?

**Key Insight**: The authors use Girsanov measure transformation to map the drift error of the reverse SDE onto the Radon-Nikodym derivative of the path measure, then project it to the marginal likelihood ratio $R_i(\mathbf x) = \mathbb E_{\mathbb P^\star_i}[e^{Z_T^i}\mid \mathbf Y_{t_0}=\mathbf x]$ at the terminal time $t_0$. The problem then becomes "how much path error is preserved during marginalization"—which is precisely an observability problem.

**Core Idea**: An "error observability coefficient" $\eta_i\in[0,1]$ is introduced to measure how much path score error reflects onto the terminal state. Thus, the single-generation lower bound becomes $I_i\gtrsim \eta_i\cdot\varepsilon_{\star,i}^2$. By accumulating this single-generation estimate along the exact contraction formula for $\chi^2$ divergence during the refreshening step $q_i=\alpha p_{\text{data}}+(1-\alpha)\hat p^i$, where $\chi^2(q_i\|p_{\text{data}})=(1-\alpha)^2 \chi^2(\hat p^i\|p_{\text{data}})$, the geometrically discounted $(1-\alpha)^{2m}$ decay law naturally emerges.

## Method

### Overall Architecture
The theoretical framework revolves around the recursive training $\hat p^i \xrightarrow{\text{mix}} q_i \xrightarrow{\text{train}} \hat p^{i+1}$, split into two levels: (1) Single-generation analysis—given the current score error energy $\varepsilon_{\star,i}^2$, the single-generation divergence $I_i$ is sandwiched using Girsanov + a new "observability" coefficient; (2) Multi-generation accumulation—the single-generation results are substituted into the $\chi^2$ contraction of the refreshening step to obtain a geometrically discounted decomposition of $D_N$. Both hold strictly in the "small score error perturbation" regime $\varepsilon_{\star,i}^2\le 1$.

### Key Designs

1.  **Observability Coefficient $\eta_i$**:
    - **Function**: Measures how much score error along the path "appears" in the marginal distribution at the reverse diffusion terminal time $t_0$, serving as a bridge to translate path energy into terminal divergence.
    - **Mechanism**: Define the random variable $M_T^i = -\int_{t_0}^T \mathbf e_{i,s}\cdot \mathrm d\bar{\mathbf B}_s$ (the stochastic integral of score error and reverse Brownian motion along the path). From Itô isometry, $\mathrm{Var}_{\mathbb P^\star_i}(M_T^i)=\varepsilon_{\star,i}^2$. Let $\eta_i = \mathrm{Var}_{\mathbb P^\star_i}(\mathbb E[M_T^i\mid \mathbf Y_{t_0}]) / \varepsilon_{\star,i}^2 \in [0,1]$. Intuition: Perturbations coupled with sample states (e.g., $\mathbf e_{i,t}(\mathbf x)=\mathbf w\mathbf x+\xi(t)$) leave a mark on the terminal state, so $\eta_i>0$; purely time-dependent or path-orthogonal perturbations are averaged out by the conditional expectation, so $\eta_i=0$.
    - **Design Motivation**: Non-zero path score error does not necessarily lead to non-zero marginal divergence (a fundamental difficulty in diffusion model analysis), which was previously bypassed using only upper bounds. By introducing $\eta_i$, the lower bound $I_i\ge \tfrac14\eta_i\varepsilon_{\star,i}^2 - C\varepsilon_{\star,i}^4$ directly connects "path error" and "marginal divergence" for the first time, and it can be numerically verified that $\eta_i>0$ almost always holds on real data like CIFAR-10.

2.  **Single-generation Equivalence $I_i\asymp \varepsilon_{\star,i}^2$ via Girsanov Sandwiched Bounds (Theorem 3.5)**:
    - **Function**: Sandwiches $I_i = \chi^2(\hat p^{i+1}\|q_i)$ using the score matching loss $\varepsilon_{\star,i}^2$ on the ideal path within the small score error regime, providing a proxy that can be directly monitored in engineering.
    - **Mechanism**: Upper bounds follow Girsanov + data processing to get $\mathrm{KL}(\hat p^{i+1}\|q_i)\le \tfrac12\hat\varepsilon_i^2$; lower bounds come from the observability argument $\chi^2(\hat p^{i+1}\|q_i)\ge \tfrac14\eta_i\varepsilon_{\star,i}^2-C\varepsilon_{\star,i}^4$. The key technique is proving that the ideal-path energy $\varepsilon_{\star,i}^2$ and learned-path energy $\hat\varepsilon_i^2$ are equivalent under the Girsanov density $L^{1+\delta}$-integrability assumption A3 and quadratic variation moment condition A4, while $\chi^2$ and KL differ only by a constant in the perturbation region. Combined, this yields $\tfrac14\eta_i\varepsilon_{\star,i}^2 - C\varepsilon_{\star,i}^4 \le \chi^2(\hat p^{i+1}\|q_i)\le 4\varepsilon_{\star,i}^2 + c\varepsilon_{\star,i}^4$.
    - **Design Motivation**: Previous results either only had KL upper bounds or used learned-path energy (not available in practice). Now, both directions are expressed using ideal-path energy, which corresponds exactly to the training objective, allowing the theory to be directly verified by experiments (in Figure 4, both $\chi^2$ and KL divergences on a 10D GMM are sandwiched by $\varepsilon_{\star,i}^2$).

3.  **Multi-generation Geometric Discounting Decomposition $D_N \asymp \sum (1-\alpha)^{2(N-i)}\varepsilon_{\star,i}^2$ (Theorem 4.2)**:
    - **Function**: Precisely decomposes the cumulative divergence after $N$ generations into a weighted sum of score error energies from each generation, discounted by geometric coefficients, providing a quantitative answer to "why adding fresh data prevents collapse."
    - **Mechanism**: The $\chi^2$ divergence of the refreshening step satisfies the exact equality $\chi^2(q_i\|p_{\text{data}})=(1-\alpha)^2\chi^2(\hat p^i\|p_{\text{data}})$ (Lemma F.1)—a key algebraic advantage of $\chi^2$ over KL. Using this alongside the single-gen equivalence $\chi^2(\hat p^{i+1}\|q_i)\asymp \varepsilon_{\star,i}^2$ for recursion, and adding a tail assumption A5 on an adaptive "good set" $\mathcal G_i$ (to prevent synthetic models from placing massive mass in regions where $p_{\text{data}}$ is extremely small), leads to $D_{N+1}+C_{\text{bias}}\asymp \sum_{i=i_0}^N (1-\alpha)^{2(N-i)}\varepsilon_{\star,i}^2 + (1-\alpha)^{2(N+1-i_0)}D_{i_0}$. Proposition 4.1 also provides a converse: if $\sum_i \varepsilon_{\star,i}^2=\infty$ or a score-error lower bound exists, $\limsup D_i$ will not vanish, and the model must collapse.
    - **Design Motivation**: Formalizes the empirical fact "larger $\alpha \to$ stronger suppression" into an explicit $(1-\alpha)^{2m}$ decay law—errors from $m$ generations ago are compressed by a factor of $(1-\alpha)^{2m}$, equivalent to an effective memory $\sim 1/\alpha$. This also provides engineering guidance: to stabilize $D_N$, one only needs $\sum \varepsilon_{\star,i}^2<\infty$ and $\alpha>0$, without requiring the error of each generation to converge to 0.

### Loss & Training
No new training loss is proposed. The theory is built upon standard variance-preserving OU forward SDE $\mathrm d\mathbf X_t = -\tfrac12\mathbf X_t\mathrm dt + \mathrm d\mathbf B_t$ and its reverse SDE, using the score matching loss $\varepsilon_{\star,i}^2 = \mathbb E_{\mathbb P^\star_i}[\int_{t_0}^T \|\mathbf e_{i,s}(\mathbf Y_s)\|_2^2 \mathrm ds]$. Minimax-optimal score estimation error satisfies $\varepsilon_{\star,i}^2 \lesssim \mathrm{polylog}(n_i)\,n_i^{-1}(1/t_0)^{d/2}$, meaning sample size must grow exponentially with the environment dimension $d$ to ensure the perturbation regime holds; however, under low-dimensional manifold assumptions, it only depends on the intrinsic dimension $d^\star\ll d$.

## Key Experimental Results

### Main Results
The authors verify the theory using three datasets: 10D Gaussian Mixture (5 components, $\sigma^2\mathbf I_{10}$), Fashion-MNIST, and CIFAR-10. All experiments use PCA projection to 2D for visualization and directly estimate $\eta_i$, $\varepsilon_{\star,i}^2$, and $\chi^2$/KL divergences.

| Dataset | Verification Goal | Results |
| :--- | :--- | :--- |
| 10D GMM ($\alpha\in\{0.1,0.5,0.9\}$, 20 gens) | Impact of $\alpha$ on collapse speed (Fig.1) | Distribution continues to diffuse at $\alpha=0.1$; retains structure but widens at $\alpha=0.5$; fully stable at $\alpha=0.9$ |
| 10D GMM (20 gens) | Single-gen upper/lower bounds (Prop 3.1 + 3.3) | $\mathrm{KL}(\hat p^{i+1}\|q_i)\le \tfrac12\hat\varepsilon_i^2$ fits tightly; $\chi^2\ge \tfrac18\hat\eta_i\varepsilon_{\star,i}^2$ verified (Fig.3) |
| 10D GMM (20 gens, $\alpha\in\{0.1,0.5\}$) | Two-sided equivalence Thm 3.5 | Both $\chi^2$ and KL are sandwiched by $\tfrac14\hat\eta_i\varepsilon_{\star,i}^2$ and $4\varepsilon_{\star,i}^2$ (Fig.4) |
| 10D GMM (20 gens) | Geometric decay decomposition Thm 4.2 | Broad contribution for $\alpha=0.1$, only recent gens contribute for $\alpha=0.9$, clear diagonalized structure (Fig.5) |
| Fashion-MNIST | Effect of $\alpha$ on real images (Fig.8/10) | Consistent with GMM: high $\alpha$ is stable, low $\alpha$ leads to collapse after multiple generations |
| CIFAR-10 | Existence of observability $\eta_i$ in real data (Fig.2/9) | State-dependent perturbations (aligned / random) give clear $\hat\eta_i>0$; time-only perturbations give $\hat\eta_i\approx 0$ |

### Ablation Study
The paper lacks traditional component ablation (being pure theory + verification), but Fig. 2 "ablates" score error types on CIFAR-10, revealing the source of $\eta_i$:

| Perturbation Type | $\mathbf e_{i,t}(\mathbf x)$ Form | Estimated $\hat\eta_i$ | Explanation |
| :--- | :--- | :--- | :--- |
| Aligned (with drift) | $\mathbf w_i \mathbf x$, $\mathbf w_i$ along drift | Highest | Error is amplified by the reverse trajectory, leaving a strong mark on the terminal state |
| Random (random direction) | $\mathbf w\mathbf x$, $\mathbf w$ random | Medium | Still state-dependent, $\eta_i$ is significantly greater than 0 |
| Time-only | $\xi(t)$, state-independent | Near 0 | Conditional expectation averages it out, no effect on marginals |

### Key Findings
- **The $(1-\alpha)^{2m}$ decay law is the core actionable conclusion**: To stabilize cumulative divergence, engineers do not need each generation's score error to reach zero, but only $\sum \varepsilon_{\star,i}^2<\infty$ and a fresh data ratio $\alpha>0$, with an effective memory window of approximately $1/\alpha$ generations.
- **State-dependence determines collapse visibility**: Purely temporal perturbations do not cause observed divergence growth ($\eta_i\approx 0$), but the random initialization and optimization noise of actual neural score models almost certainly introduce state-dependent perturbations, making collapse a "universal phenomenon" in practice.
- **$\chi^2$ divergence is a critical technical choice**: The exact equality $\chi^2(q_i\|p_{\text{data}})=(1-\alpha)^2\chi^2(\hat p^i\|p_{\text{data}})$ for the refreshening step (which KL lacks) is the algebraic root that allows for the geometric decay recurrence; since $\chi^2$ and KL are equivalent in the perturbation region (verified in Fig. 4), the lower bound conclusions also translate back to KL.
- **Transience in the First Phase ($i<i_0$)**: Theoretical assumptions hold after $i_0$ generations; the deviation at $i=1$ in Fig. 3 does not violate the theory but corresponds to "transience"—in practice, stability is usually reached after one or two generations.

## Highlights & Insights
- **The proposal of the observability coefficient $\eta_i$ is truly novel**: It quantifies the long-standing gap in diffusion analysis between "path error" and "marginal divergence" with a single scalar that can be estimated from data—this is the most "aha" design of the paper.
- **Subtle choice of $\chi^2$ divergence**: Ours deliberately chooses $\chi^2$ over KL because the refreshening step $q_i=\alpha p_{\text{data}}+(1-\alpha)\hat p^i$ satisfies a clean quadratic contraction under $\chi^2$; this observation is directly transferable to other "mix-and-retrain" frameworks (e.g., mixing with a reference policy in RLHF).
- **Architecture-agnostic population-level perspective**: Most existing theories assume two-layer score networks, but Ours skips architecture-specific arguments entirely, starting from the path measure level—this strategy of "simplifying to SDE measure ratios, then using Girsanov projection" can be directly applied to other score-flavored generative models like flow matching and consistency models.
- **Lower bounds that can be directly verified by experiment**: Unlike most pure theory papers, all theorems provide estimable constants and quantities. The authors substitute estimated $\hat\eta_i$ and $\hat\varepsilon_{\star,i}^2$ into the bounds on GMM/Fashion-MNIST/CIFAR-10 and compare them with the actual divergence—this "theory-experiment loop" is rare in model collapse literature.

## Limitations & Future Work
- **Only applicable to small score error perturbation regimes**: All results assume $\varepsilon_{\star,i}^2\le 1$, but Equation (12) indicates that high-dimensional data requires sample sizes $n_i\sim (1/t_0)^{d/2}$ to reach this regime, which large models in practice might exceed. The authors explicitly list "lower bounds for large error regimes" as an open problem.
- **Neglects discretization and initialization errors**: The theory uses continuous-time reverse SDEs and assumes starting from $\mathcal N(0,\mathbf I_d)$ (using exponential OU convergence), whereas actual diffusion models use discrete-step samplers, and this part of the error is not quantified.
- **Lack of a-priori guarantees for the lower bound of $\eta_i$**: The paper can only state that "it is almost always $>0$ in practice" (verified on CIFAR-10), but there are no theoretical criteria for when $\eta_i$ might collapse to 0 for more complex architectures (e.g., transformer-based DiT, flow matching), which is a key gap for applying the theory to SOTA models.
- **Does not answer "if there is a limit distribution"**: The authors point out that whether recursive training converges to some $\alpha$-dependent limit distribution remains an open question; current conclusions only guarantee whether $D_i$ is bounded or unbounded.
- **Improvable directions**: Explicitly linking $\eta_i$ with score network architecture, activation functions, and initialization distributions might provide engineering guidance on "which architectures are more resistant to collapse"; additionally, extending the analysis to multi-stage training (using different mixing ratios $\alpha_k$ for different generations) could directly serve real self-improving pipelines.

## Related Work & Insights
- **vs Fu et al. (2024) / Cui et al. (2026)**: They provide architecture-specific upper bounds for two-layer score networks; Ours provides architecture-agnostic population-level upper and lower bounds—filling the lower bound gap with broader coverage, though the finite-sample rates are less specific.
- **vs Bertrand et al. (2024)**: Also studies recursive generative model stability but follows an MLE route (for parameterized distributions); Ours focuses on score-based diffusion, using completely different techniques (Girsanov instead of fixed point on parameter space).
- **vs Gerstgrasser et al. (2024)**: They empirically found that "accumulating real + synthetic data can block collapse"; Ours provides a quantitative explanation for this phenomenon via the $(1-\alpha)^{2m}$ decay law—turning an empirical observation into a calculable decay rate.
- **vs Chen et al. (2023c) / Benton et al. (2024)**: These are classic KL upper bounds for diffusion sampling convergence; Ours uses the same Girsanov tools but in reverse to provide a $\chi^2$ lower bound—a dual application of the same toolchain.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First lower bound for distribution divergence in diffusion models + observability coefficient + geometric discounting decomposition; all three are new to model collapse literature.
- Experimental Thoroughness: ⭐⭐⭐ Pure theory + verification experiments; 10D GMM/Fashion-MNIST/CIFAR-10 are used, which is sufficient for a theory paper, but lacks alignment with SOTA self-consuming experimental setups.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, explanations for the reasonableness of assumptions A1-A5, and splitting the Girsanov argument into two independent challenges (observability and ratio control) has great educational value.
- Value: ⭐⭐⭐⭐ Provides actionable engineering conclusions ($\alpha>0$ + summable error energy equals stability) and clear open problems (lower bounds in large error regions, discretization error, limit distributions), with significance for both diffusion theory and self-training design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Test-Time Iterative Error Correction for Efficient Diffusion Models](../../ICLR2026/image_generation/test-time_iterative_error_correction_for_efficient_diffusion_models.md)
- [\[NeurIPS 2025\] A Closer Look at Model Collapse: From a Generalization-to-Memorization Perspective](../../NeurIPS2025/image_generation/a_closer_look_at_model_collapse_from_a_generalization-to-memorization_perspectiv.md)
- [\[ICML 2026\] Escaping Mode Collapse in LLM Generation via Geometric Regulation](escaping_mode_collapse_in_llm_generation_via_geometric_regulation.md)
- [\[ICML 2026\] A Unified Framework for Diffusion Model Unlearning with f-Divergence](a_unified_framework_for_diffusion_model_unlearning_with_f-divergence.md)
- [\[ICML 2026\] Alignment-Guided Score Matching for Text-to-Image Alignment in Diffusion Models](alignment-guided_score_matching_for_text-to-image_alignment_in_diffusion_models.md)

</div>

<!-- RELATED:END -->
