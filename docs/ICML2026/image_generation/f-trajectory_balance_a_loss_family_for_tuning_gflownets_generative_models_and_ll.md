---
title: >-
  [Paper Note] $f$-Trajectory Balance: A Loss Family for Tuning GFlowNets, Generative Models, and LLMs with Off- and On-Policy Data
description: >-
  [ICML2026][Image Generation][f-divergence] The $\mathbb{KL}_{sq}$ surrogate loss (squared log-prob difference) used in GFlowNet and Kimi is generalized to the entire family of $f$-divergences. This results in a tunable f…
tags:
  - "ICML2026"
  - "Image Generation"
  - "f-divergence"
  - "Trajectory Balance"
  - "GFlowNets"
  - "off-policy"
  - "LLM RLHF"
date: 2026-05-08
content_hash: 23fcb4af769671a7
---

# $f$-Trajectory Balance: A Loss Family for Tuning GFlowNets, Generative Models, and LLMs with Off- and On-Policy Data

**Conference**: ICML2026  
**arXiv**: [2605.15417](https://arxiv.org/abs/2605.15417)  
**Code**: TBD  
**Area**: Reinforcement Learning / GFlowNets / Generative Model Fine-tuning  
**Keywords**: f-divergence, Trajectory Balance, GFlowNets, off-policy, LLM RLHF

## TL;DR
The $\mathbb{KL}_{sq}$ surrogate loss (squared log-prob difference) used in GFlowNet and Kimi is generalized to the entire family of $f$-divergences. This results in a tunable family of mode-seeking $\leftrightarrow$ mode-covering losses where the on-policy gradient matches the true $f$-divergence gradient and off-policy data remains valid with a consistent global optimum. The method is validated on synthetic grids, molecule generation via SynFlowNet, conditional diffusion sampling, and asynchronous LLM RL (GSM8k / MATH).

## Background & Motivation
**Background**: Recent advances in GFlowNet training (Trajectory Balance), variational inference (VarGrad), and LLM RL fine-tuning (e.g., Kimi K1.5 / K2) have converged on a specific loss form: the squared difference between the model log-probability and the target log-probability, $\mathbb{KL}_{sq}=\tfrac12 (\log\pi_\theta/\pi_\star)^2$. Its appeal lies in two properties: its on-policy auto-diff gradient in expectation exactly equals the true $\nabla_\theta \mathbb{KL}(\pi_\theta\|\pi_\star)$, and it remains a valid loss for off-policy data sampled from any distribution $\mu$ without shifting the optimal point.

**Limitations of Prior Work**: $\mathbb{KL}_{sq}$ inherits the geometric properties of reverse KL and is inherently **mode-seeking**, causing models to collapse onto a few high-reward modes. Applications like drug discovery, exploratory agents, or scenarios requiring diverse high-reward candidates prefer **mode-covering** behavior (e.g., forward KL, Hellinger). Existing methods either rely on high-variance REINFORCE/PPO gradients or utilize log-variance variants like Pearson $\chi^2$ that break the critical "on-policy gradient matching" property, lacking a unified framework.

**Key Challenge**: Simultaneously achieving low-variance stable training, off-policy validity, and tunable mode behavior.

**Goal**: (i) Construct a surrogate loss $\mathcal{L}_f$ for any $f$-divergence such that its on-policy gradient equals $\nabla_\theta D_f$ and its off-policy optimum remains consistent; (ii) prove that any shift-invariant loss acting on log-prob differences corresponds to a unique $f$-divergence; (iii) generalize the batch-normalization of VarGrad to any $f$; (iv) resolve numerical instabilities at small values of $\beta$ where $\exp(\beta^{-1}r)$ explodes.

**Key Insight**: The "magic" of $\mathbb{KL}_{sq}$ stems from the fact that for $f(u)=u\log u$, the derivative $f'(u)=\log u + 1$ results in a score function coefficient proportional to $\log\pi_\theta - \log\pi_\star$. For a general $f$, the same gradient alignment can be replicated by replacing this coefficient with $f'(\exp t) - f'(1)$ and integrating over $t$.

**Core Idea**: A family of **shift-invariant integral losses** $\mathcal{L}_f(\Delta) = \int_0^{\Delta}\bigl(f'(\exp t)-f'(1)\bigr)\,dt$ is proposed to replace the squared loss, where $\Delta_\theta(\mathbf{y})=\log\pi_\theta(\mathbf{y})-\log\pi_\star(\mathbf{y})$.

## Method

### Overall Architecture
Input consists of sampled trajectories/sequences $\mathbf{y}\sim\mu$ (on-policy or off-policy) and a target distribution $\pi_\star(\mathbf{y})\propto \pi_{\text{ref}}(\mathbf{y})\exp(\beta^{-1}r(\mathbf{y}))$. A specific $f$-divergence (such as an $\alpha$-divergence) is selected to derive the corresponding loss function $\mathcal{L}_f$. If the partition function $Z$ is unknown, it is estimated via in-batch **DevGrad**. For small $\beta$, the **Tempered Loss** scales the energy function back to a stable range. Standard auto-differentiation is used to obtain $\nabla_\theta \mathcal{L}_f$, which is plug-and-play across GFlowNet (replacing the squared TB loss), diffusion tuning, and asynchronous LLM RL. Implementation requires only replacing `(log_pi_theta - log_pi_star)**2` with `L_f(log_pi_theta - log_pi_star)`.

### Key Designs

1.  **$f$-Divergence Surrogate Loss $\mathcal{L}_f$ and Bidirectional Equivalence (Prop 4.2 + Prop 4.4)**:
    *   **Function**: Maps any convex function $f$ (normalized such that $f(1)=0$ and $f''(1)=1$) to a scalar loss acting on $\Delta_\theta = \log\pi_\theta - \log\pi_\star$. This ensures on-policy gradient matching with $\nabla_\theta D_f$ and off-policy optimality at $\pi_\theta = \pi_\star$.
    *   **Mechanism**: Defined as $\mathcal{L}_f(\Delta) = \int_0^{\Delta}\bigl(f'(\exp t)-f'(1)\bigr)\,dt$. Setting $f(u)=u\log u$ recovers $\tfrac12 \Delta^2$, demonstrating it generalizes $\mathbb{KL}_{sq}$. Conversely, for any shift-invariant differentiable loss $\ell$, the corresponding $f$-divergence is given by $f_\ell(u)=\lambda_1 \int_1^u \ell'(\log t)\,dt + \lambda_2(u-1)+c$. Thus, shift-invariant losses and $f$-divergences are two sides of the same coin. For $\alpha$-divergences, the closed-form loss is $\mathcal{L}_\alpha(\Delta)=\frac{1}{(\alpha-1)^2}e^{(\alpha-1)\Delta}-\frac{\Delta}{\alpha-1}-\frac{1}{(\alpha-1)^2}$, which converges to the squared loss/Trajectory Balance as $\alpha \to 1$.
    *   **Design Motivation**: On-policy gradient alignment ensures the theoretical target is the divergence being optimized. Shift-invariance ensures that adding a constant (like $\log Z$) does not affect minimization, enabling off-policy validity. Establishing a bijection proves that any convex loss used for tuning is implicitly optimizing an $f$-divergence.

2.  **DevGrad: Generalizing VarGrad to Any $f$ (Handling Unknown $Z$)**:
    *   **Function**: When $Z$ in $\pi_\star(\mathbf{y})=\frac{1}{Z}\exp(\mathcal{R}(\mathbf{y}))$ is uncomputable (standard in LLM RL), it estimates $\log Z$ using in-batch statistics while preserving on-policy gradient matching.
    *   **Mechanism**: For a batch $\mathcal{B}=\{\mathbf{y}_1,\dots,\mathbf{y}_B\}$, it solves the 1D optimization $\widehat{\log Z}=\arg\min_C \tfrac{1}{B}\sum_i \mathcal{L}_f(\Delta(\mathbf{y}_i)+C)$ and applies a stop-gradient: $\mathcal{L}_f^{\text{DG}}(\mathcal{B},\theta)=\tfrac{1}{B}\sum_i \mathcal{L}_f\bigl(\Delta(\mathbf{y}_i)+\text{SG}[\widehat{\log Z}]\bigr)$. For $\mathcal{L}_f(y)=y^2$, $\widehat{\log Z}$ is the mean, recovering VarGrad. For $\mathcal{L}_f(y)=|y|$, it uses the median, corresponding to Total Variation divergence.
    *   **Design Motivation**: In-batch normalization centralizes score function coefficients, which is the source of variance reduction in VarGrad. This property holds for any generalized deviation (Rockafellar 2006). Kimi K1.5/K2's use of $\bar r$ instead of $\log Z$ is a specific approximation of this framework under reverse KL.

3.  **Tempered Loss: Resolving Energy Explosion at Small $\beta$**:
    *   **Function**: At small KL regularization coefficients $\beta$ (e.g., $0.005$), $\exp(\beta^{-1}r)$ can reach $e^{200}$, causing numerical overflow. Tempered Loss scales the objective to a stable range.
    *   **Mechanism**: Defines a tempered distribution $\tilde p_\beta \propto p^\beta$ with energy $\beta\mathcal{R}_\star=\beta\log\pi_{\text{ref}}+r$. The tempered loss is $\tilde{\mathcal{L}}_{f,\beta}(\Delta)=\frac{1}{\beta}\mathcal{L}_f(\beta\Delta)$. Since equality of tempered log-probs implies equality of original log-probs, the optimum is unchanged. The $1/\beta$ factor stabilizes the gradient scale. Substituting $f(u)=u\log u$ yields the Kimi loss $\frac{1}{2\beta}(\beta\log\frac{\pi_\theta}{\mathcal{R}}-\log\tilde Z)^2$.
    *   **Design Motivation**: Transforms an engineering trick into a theoretically grounded stabilization method suitable for various $f$-divergences.

### Loss & Training
In GFlowNets, the Trajectory Balance squared loss $(\Delta(\tau,\theta,\phi))^2$ is replaced by $\mathcal{L}_f(\Delta(\tau,\theta,\phi))$, resulting in **$f$-Trajectory Balance**. Prop 5.1 proves $\nabla_\theta D_f(\pi_F\|\pi_B)=\mathbb{E}_{\tau\sim\pi_{F,\theta}}[\nabla_\theta \mathcal{L}_f]$. For LLM asynchronous RL, tempered DevGrad is used without needing clipping, importance weighting, or masking. Typical settings use $\alpha\in\{0.5, 0.75, 1.2, 2\}$, where $\alpha<1$ favors mode-covering and $\alpha>1$ favors mode-seeking.

## Key Experimental Results

### Main Results

| Task | Configuration | Key Metrics | Observations |
| :--- | :--- | :--- | :--- |
| Hypergrid (4 modes) | Forward KL ($\alpha=0$) / Hellinger ($\alpha=0.5$) | Mode Discovery + Convergence | Finds all 4 modes; faster convergence than TB ($\alpha=1$). |
| Hypergrid | Pearson $\chi^2$ ($\alpha=2$) | Mode Discovery | Stuck on the first discovered mode (extreme mode-seeking). |
| SynFlowNet (3 tasks) | $\alpha=0.75$ vs TB | High-reward diversity | Significantly higher diversity than TB with comparable rewards. |
| SynFlowNet | $\alpha=1.2$ | Diversity | Mode collapse observed. |
| SynFlowNet | Annealing $\alpha: 0.75\to 1.2$ | Diversity + Reward count | Achieved best tradeoff. |
| MNIST Diffusion | TB | Digit frequency | Over-sampled 0 and 6, poor coverage. |
| MNIST Diffusion | $f$-TB (annealed $\alpha$) | Digit frequency | Modes are distributed more uniformly. |
| GSM8k+MATH Async RL | Qwen2.5 3B–14B / OLMo-2-7B | Entropy–Reward tradeoff | Forward KL, Reverse KL, Pearson, and JS all successful; optimized PPO unstable under off-policy lag. |

### Ablation Study

| Configuration | Behavior | Description |
| :--- | :--- | :--- |
| $\mathcal{L}_f$ + DevGrad (Full) | Gradient matching $D_f$ + Off-policy validity | All theoretical claims hold. |
| Without DevGrad (using $\bar r$) | Effective only at small $\beta$ or near on-policy | Validates alignment with Kimi's approximation. |
| Without Tempered Loss (small $\beta$) | $e^{200}$ overflow / Gradient loss from clipping | Confirms necessity of tempered scaling. |
| $\mathcal{L}_f$ for backward policy | Corresponds to derived divergence $h$ | Consistency with TB: validity comes from loss properties, not divergence matching. |

### Key Findings
*   Adjusting $\alpha$ within the $\alpha$-divergence family provides a **smooth control knob** for mode-covering $\leftrightarrow$ mode-seeking behavior.
*   In asynchronous LLM RL with 50-step lag, tempered DevGrad remains stable without clipping or importance ratios, whereas optimized PPO variants (CISPO + DAPO) struggle. This demonstrates the engineering dividends of off-policy validity.
*   The Kimi K1.5/K2 loss is formally situated as a tempered DevGrad approximation of reverse KL, providing a theoretical anchor for industrial practices and suggesting easy transitions to mode-covering variants.

## Highlights & Insights
*   The bijective mapping between **shift-invariant losses and $f$-divergences** provides a unified framework for loss design, showing that previous disparate loss functions were simply different $f$-divergences.
*   The common framework across GFlowNets, LLM RL, and Diffusion models suggests that trajectory-level $f$-divergence minimization could be a unified approach for generative model alignment.
*   DevGrad unifies partition function estimation and variance reduction (score centering), which is valuable for any RL pipeline requiring reward baselines.

## Limitations & Future Work
*   The effect of backward policy gradients matching a derived divergence $h$ rather than $f$ on learning dynamics in deep DAGs (non-trees) requires further analysis.
*   SynFlowNet experiments suggest a narrow range of usable $\alpha$ (outside $[0.75, 1.2]$ is unstable), indicating a gap between theoretical flexibility and practical stability.
*   LLM experiments were limited to verifiable rewards (GSM8k/MATH). Future work is needed to validate mode-covering benefits in noisier scenarios like RLHF preference data or long CoT reasoning.

## Related Work & Insights
*   **vs Trajectory Balance (Malkin 2022a)**: TB is a special case of $f(u)=u\log u$ (reverse KL); this work extends it to the whole family with tunable mode behavior.
*   **vs VarGrad (Richter 2020)**: VarGrad is reverse KL DevGrad; this work generalizes it using generalized deviations.
*   **vs Kimi K1.5/K2 (Team 2025a,b)**: Situates Kimi's empirical loss within a theoretical coordinate system.
*   **vs Silva 2024**: Extends beyond Silva's on-policy divergence work by ensuring off-policy validity, which is critical for GFlowNet exploration.

## Rating
*   Novelty: ⭐⭐⭐⭐⭐ Unifies scattered loss variants into a single family with a bidirectional equivalence theorem.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Covers various benchmarks, though lacks RLHF preference data.
*   Writing Quality: ⭐⭐⭐⭐ Dense math but well-motivated with intuitive visualizations.
*   Value: ⭐⭐⭐⭐⭐ Provides a plug-and-play loss dictionary for RL fine-tuning and GFlowNets.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Discrete Diffusion Samplers and Bridges: Off-Policy Algorithms and Applications in Latent Spaces](discrete_diffusion_samplers_and_bridges_off-policy_algorithms_and_applications_i.md)
- [\[ICML 2026\] Esoteric Language Models: A Family of Any-Order Diffusion LLMs](esoteric_language_models_a_family_of_any-order_diffusion_llms.md)
- [\[ICML 2026\] A Diffusive Classification Loss for Learning Energy-based Generative Models](a_diffusive_classification_loss_for_learning_energy-based_generative_models.md)
- [\[ICML 2026\] EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization](evogm_learning_to_merge_llms_via_evolutionary_generative_optimization.md)
- [\[ICML 2026\] AtelierEval: Agentic Evaluation of Humans & LLMs as Text-to-Image Prompters](ateliereval_agentic_evaluation_of_humans_llms_as_text-to-image_prompters.md)

</div>

<!-- RELATED:END -->
