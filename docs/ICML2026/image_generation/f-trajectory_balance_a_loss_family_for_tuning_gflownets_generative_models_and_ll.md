---
title: >-
  [Paper Note] $f$-Trajectory Balance: A Loss Family for Tuning GFlowNets, Generative Models, and LLMs with Off- and On-Policy Data
description: >-
  [ICML2026][Image Generation][f-divergence] This work generalizes the $\mathbb{KL}_{sq}$ proxy loss—which computes the "squared difference of log-probabilities" as seen in GFlowNet and Kimi—to the entire family of $f$-divergences. This results in a tunable family of mode-seeking $\leftrightarrow$ mode-covering losses where on-policy gradients equal the true $f$-divergence gradients and off-policy optimality remains consistent. Validations are conducted on synthetic grids…
tags:
  - "ICML2026"
  - "Image Generation"
  - "f-divergence"
  - "Trajectory Balance"
  - "GFlowNets"
  - "off-policy"
  - "LLM RLHF"
date: 2026-05-08
content_hash: 4456b2b5e208fc77
---

# $f$-Trajectory Balance: A Loss Family for Tuning GFlowNets, Generative Models, and LLMs with Off- and On-Policy Data

**Conference**: ICML2026  
**arXiv**: [2605.15417](https://arxiv.org/abs/2605.15417)  
**Code**: To be confirmed  
**Area**: Reinforcement Learning / GFlowNets / Generative Model Fine-tuning  
**Keywords**: f-divergence, Trajectory Balance, GFlowNets, off-policy, LLM RLHF

## TL;DR
This work generalizes the $\mathbb{KL}_{sq}$ proxy loss—which computes the "squared difference of log-probabilities" as seen in GFlowNet and Kimi—to the entire family of $f$-divergences. This results in a tunable family of mode-seeking $\leftrightarrow$ mode-covering losses where on-policy gradients equal the true $f$-divergence gradients and off-policy optimality remains consistent. Validations are conducted on synthetic grids, SynFlowNet molecule generation, diffusion model conditional sampling, and asynchronous LLM RL (GSM8k / MATH).

## Background & Motivation
**Background**: Recent advances in GFlowNet training (Trajectory Balance), variational inference (VarGrad), and LLM RL fine-tuning (e.g., Kimi K1.5 / K2) have converged on a specific loss form: the squared difference between the model log-prob and the target log-prob, denoted as $\mathbb{KL}_{sq}=\tfrac12 (\log\pi_\theta/\pi_\star)^2$. Its appeal lies in two properties: the auto-diff gradient of the on-policy expectation exactly equals the true $\nabla_\theta \mathbb{KL}(\pi_\theta\|\pi_\star)$, and it remains a valid loss with the same global optimum when sampled from an arbitrary off-policy distribution $\mu$.

**Limitations of Prior Work**: However, $\mathbb{KL}_{sq}$ inherits the geometric properties of reverse KL and is inherently **mode-seeking**, causing models to collapse into a few high-reward modes. Many scenarios, such as drug discovery or exploratory agents, require **mode-covering** behavior (forward KL, Hellinger, etc.). Existing methods either use REINFORCE/PPO gradients (high variance) or construct Pearson $\chi^2$ variants via log-variance divergence (which breaks the critical "on-policy gradient matching" property), lacking a unified framework.

**Key Challenge**: No prior work has simultaneously achieved low-variance stable training, off-policy validity, and tunable mode behavior.

**Goal**: (i) Construct a proxy loss $\mathcal{L}_f$ for any $f$-divergence such that its on-policy gradient matches $\nabla_\theta D_f$ and its off-policy optimum remains invariant; (ii) Prove that any "shift-invariant loss acting on log-prob differences" corresponds to a unique $f$-divergence; (iii) Generalize VarGrad's in-batch normalization to any $f$; (iv) Resolve numerical explosions associated with $\exp(\beta^{-1}r)$ under small $\beta$.

**Key Insight**: The "magic" of $\mathbb{KL}_{sq}$ stems from the fact that for $f(u)=u\log u$, the derivative $f'(u)=\log u + 1$ makes the score function coefficient exactly $\log\pi_\theta - \log\pi_\star$. For a general $f$, one can replicate this gradient alignment by replacing this coefficient with $f'(\exp t) - f'(1)$ and integrating over $t$.

**Core Idea**: Replace the squared loss with a family of **shift-invariant integral losses** $\mathcal{L}_f(\Delta) = \int_0^{\Delta}\bigl(f'(\exp t)-f'(1)\bigr)\,dt$, acting on the log-prob difference $\Delta_\theta(\mathbf{y})=\log\pi_\theta(\mathbf{y})-\log\pi_\star(\mathbf{y})$.

## Method

### Overall Architecture
The proposed method addresses the limitation of $\mathbb{KL}_{sq}$ (mode-seeking behavior) in training GFlowNets, diffusion models, and LLM RL. By replacing the squared loss with an integral loss $\mathcal{L}_f$ determined by a convex function $f$, the model still operates on the log-prob difference $\Delta_\theta(\mathbf{y})=\log\pi_\theta(\mathbf{y})-\log\pi_\star(\mathbf{y})$. This allows continuous sliding between mode-seeking and mode-covering behaviors by changing $f$ (or tuning a knob $\alpha$) while preserving "on-policy gradient alignment" and "off-policy validity." The framework uses DevGrad for in-batch estimation of unknown partition functions and Tempered Loss to stabilize energy scales when the regularization coefficient $\beta$ is small. In implementation, one simply replaces `(log_pi_theta - log_pi_star)**2` with `L_f(log_pi_theta - log_pi_star)`.

### Key Designs

**1. $f$-Divergence Proxy Loss and Bi-directional Equivalence Theorem: Generalizing Squared Loss via Integration**

The essence of the squared loss is that the derivative $f'(u)=\log u+1$ of $f(u)=u\log u$ aligns the score function coefficient with $\log\pi_\theta-\log\pi_\star$. Following this, for any convex function $f$ normalized such that $f(1)=0$ and $f'(1)=f''(1)=1$, the authors define the proxy loss as $\mathcal{L}_f(\Delta)=\int_0^{\Delta}\bigl(f'(\exp t)-f'(1)\bigr)\,dt$ (Prop 4.2). Substituting $f(u)=u\log u$ yields $\tfrac12\Delta^2$, proving it as a generalization of $\mathbb{KL}_{sq}$. For $\alpha$-divergences, it yields the closed form $\mathcal{L}_\alpha(\Delta)=\frac{1}{(\alpha-1)^2}e^{(\alpha-1)\Delta}-\frac{\Delta}{\alpha-1}-\frac{1}{(\alpha-1)^2}$, which reduces to the Trajectory Balance loss as $\alpha\to 1$. This construction ensures: (1) the expectation of the auto-diff gradient equals the true $\nabla_\theta D_f(\pi_\theta\|\pi_\star)$, and (2) the loss is shift-invariant with respect to $\Delta$, ensuring off-policy validity as the missing constant $\log Z$ does not affect the minimizer.

Furthermore, the authors prove a bijection (Prop 4.4): any shift-invariant differentiable loss $\ell$ uniquely corresponds to an $f$-divergence $f_\ell(u)=\lambda_1\int_1^u \ell'(\log t)\,dt+\lambda_2(u-1)+c$. This implies that training GFlowNets/LLMs with any convex shift-invariant loss implicitly optimizes some $f$-divergence; the bijection provides a "lookup table" for selecting mode behaviors.

**2. DevGrad: Generalizing VarGrad for Any $f$ to Handle Unknown Partition Functions**

In LLM RL, the partition function $Z$ of $\pi_\star(\mathbf{y})=\frac{1}{Z}\exp(\mathcal{R}(\mathbf{y}))$ is usually intractable. The authors propose solving a 1D optimization within each batch $\mathcal{B}=\{\mathbf{y}_1,\dots,\mathbf{y}_B\}$ to estimate $\widehat{\log Z}=\arg\min_C\frac1B\sum_i\mathcal{L}_f(\Delta(\mathbf{y}_i)+C)$. This is then substituted back into the loss with a stop-gradient: $\mathcal{L}_f^{\text{DG}}(\mathcal{B},\theta)=\frac1B\sum_i\mathcal{L}_f\bigl(\Delta(\mathbf{y}_i)+\text{SG}[\widehat{\log Z}]\bigr)$. This step simultaneously reduces gradient variance through in-batch normalization (centering the score function coefficients), a mechanism valid for any generalized deviation (Rockafellar 2006). When $\mathcal{L}_f(y)=y^2$, $\widehat{\log Z}$ is the mean and the loss becomes the variance (classic VarGrad); when $\mathcal{L}_f(y)=|y|$, it becomes the mean absolute deviation around the median (Total Variation divergence).

**3. Tempered Loss: Rescaling Energy for Numerical Stability under Small $\beta$**

When the KL regularization coefficient $\beta$ is very small (e.g., $0.005$), the term $\exp(\beta^{-1}r)$ can lead to overflow (e.g., $e^{200}$), while hard clipping loses gradient signals. The authors introduce a tempered distribution $\tilde p_\beta\propto p^\beta$, where the energy $\beta\mathcal{R}_\star=\beta\log\pi_{\text{ref}}+r$ is independent of $1/\beta$. The resulting tempered loss $\tilde{\mathcal{L}}_{f,\beta}(\Delta)=\frac1\beta\mathcal{L}_f(\beta\Delta)$ preserves the optimum (as tempered log-prob equality implies original log-prob equality) and ensures gradient scales do not drift with $\beta$. Using $f(u)=u\log u$ recovers the Kimi loss $\frac{1}{2\beta}\bigl(\beta\log\frac{\pi_\theta}{\mathcal{R}}-\log \tilde Z\bigr)^2$.

### Loss & Training
In GFlowNets, replacing the Trajectory Balance squared loss $(\Delta(\tau, \theta, \phi))^2$ with $\mathcal{L}_f(\Delta(\tau, \theta, \phi))$ yields **$f$-Trajectory Balance**. Prop 5.1 proves $\nabla_\theta D_f(\pi_F\|\pi_B)=\mathbb{E}_{\tau\sim\pi_{F,\theta}}[\nabla_\theta \mathcal{L}_f]$. For LLM asynchronous RL, tempered DevGrad is used without needing clipping, importance weighting, or masking. For GFlowNets, $\alpha \in \{0.5, 0.75, 1.2, 2\}$ is common, where $\alpha < 1$ is mode-covering and $\alpha > 1$ is mode-seeking.

## Key Experimental Results

### Main Results

| Task | Configuration | Key Metrics | Phenomena |
| :--- | :--- | :--- | :--- |
| Hypergrid (4 modes) | Forward KL ($\alpha=0$) / Hellinger ($\alpha=0.5$) | Mode discovery + Convergence speed | Discovers all 4 modes; faster convergence than TB ($\alpha=1$). |
| Hypergrid | Pearson $\chi^2$ ($\alpha=2$) | Mode discovery | Stuck in the first discovered mode (extreme mode-seeking). |
| SynFlowNet (3 tasks) | $\alpha=0.75$ vs TB | Diversity of high-reward molecules | Diversity significantly higher than TB with similar rewards. |
| SynFlowNet | $\alpha=1.2$ | Diversity | Mode collapse observed. |
| SynFlowNet | Annealing $\alpha: 0.75\to 1.2$ | Diversity + High-reward count | Achieved the best trade-off. |
| MNIST Diffusion | TB | Digit frequency | Over-samples 0 and 6; poor coverage. |
| MNIST Diffusion | $f$-TB (Annealed $\alpha$) | Digit frequency | More uniform mode distribution. |
| GSM8k+MATH Async RL | Qwen2.5 3B–14B / OLMo-2-7B | Entropy-Reward trade-off | Validated Forward KL/Reverse KL/Pearson/JS; curves match theory; PPO unstable in off-policy. |

### Ablation Study

| Configuration | Behavior | Description |
| :--- | :--- | :--- |
| $\mathcal{L}_f$ + DevGrad (Full) | On-policy gradient matches $D_f$ + Off-policy invariant | All claims hold. |
| Without DevGrad, using $\widehat{\log Z}=\bar r$ | Only holds for small $\beta$ or near on-policy | Validates the Kimi approximation as a special case. |
| Without Tempered Loss (small $\beta$) | Overflows at $e^{200}$ scale / Clipping required | Confirms the necessity of tempered scaling. |
| $\mathcal{L}_f$ for backward policy | Corresponds to divergence $h$, not $f$ | Consistent with TB; validity comes from loss form, not divergence matching. |

### Key Findings
- **Tunable knob**: Adjusting $\alpha$ in the $\alpha$-divergence family allows smooth transitions between mode-covering and mode-seeking behaviors. 
- **Stability in Async RL**: In LLM asynchronous RL with 50-step latency, tempered DevGrad remains stable without clipping or importance ratios, whereas optimized PPO variants (CISPO, DAPO) struggle.
- **Kimi Interpretation**: The Kimi loss is explained as reverse KL tempered loss with $\widehat{\log Z} \approx \bar r$, placing industrial heuristics into a clear theoretical framework.

## Highlights & Insights
- The **Shift-invariant loss $\leftrightarrow$ $f$-divergence** bijection provides a "recipe book" for designing custom losses with specific mode behaviors.
- The universality of $\mathcal{L}_f$ across GFlowNets, LLMs, and Diffusion models suggests that trajectory-level $f$-divergence minimization might be a unified framework for generative model alignment.
- DevGrad elegantly combines partition function estimation and variance reduction into a single mechanism, which is highly valuable for RLHF/GRPO pipelines.

## Limitations & Future Work
- The on-policy gradient of the backward policy $\pi_B$ corresponds to a derived divergence $h$ rather than $f$ itself; the impact on learning dynamics in deep DAGs requires further analysis.
- The stable $\alpha$ range in complex tasks like SynFlowNet is relatively narrow ($[0.75, 1.2]$), suggesting a gap between theoretical possibility and engineering stability.
- Evaluation was limited to verifiable rewards (GSM8k/MATH). Future work should test performance on noisy RLHF preference data.

## Related Work & Insights
- **vs Trajectory Balance (Malkin 2022a)**: TB is a special case of $f$-TB for reverse KL ($\alpha=1$). This work extends it to the full $f$-family.
- **vs VarGrad / Log-variance Divergence (Richter 2020)**: VarGrad is reverse KL DevGrad. This work corrects variants like Pearson $\chi^2$ that previously broke on-policy gradient matching.
- **vs Kimi K1.5/K2 (Team 2025a,b)**: Provides a theoretical coordinate system for Kimi's empirical losses and enables mode-covering variants.
- **vs Silva 2024**: While Silva et al. discussed other divergences for GFlowNets, they focused on the on-policy setting. The core value-add here is **off-policy validity**.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Unifies scattered "squared loss variants" into the $f$-divergence family with a bi-directional equivalence theorem.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers various domains, though LLM experiments lack RLHF preference data.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation links the dense equations; excellent intuition provided for the $\alpha$ knob.
- **Value**: ⭐⭐⭐⭐⭐ Provides a plug-and-play loss dictionary for RL fine-tuning and GFlowNets, grounding empirical practices in theory.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Esoteric Language Models: A Family of Any-Order Diffusion LLMs](esoteric_language_models_a_family_of_any-order_diffusion_llms.md)
- [\[ICML 2026\] Discrete Diffusion Samplers and Bridges: Off-Policy Algorithms and Applications in Latent Spaces](discrete_diffusion_samplers_and_bridges_off-policy_algorithms_and_applications_i.md)
- [\[ICML 2026\] A Diffusive Classification Loss for Learning Energy-based Generative Models](a_diffusive_classification_loss_for_learning_energy-based_generative_models.md)
- [\[ICML 2026\] EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization](evogm_learning_to_merge_llms_via_evolutionary_generative_optimization.md)
- [\[ICML 2026\] Hölder++: Improving the Quality-Coherence Trade-off in Multimodal VAEs](hölder_improving_the_quality-coherence_trade-off_in_multimodal_vaes.md)

</div>

<!-- RELATED:END -->
