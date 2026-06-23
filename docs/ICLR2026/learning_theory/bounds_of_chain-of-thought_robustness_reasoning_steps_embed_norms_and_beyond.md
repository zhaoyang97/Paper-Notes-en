---
title: >-
  [Paper Note] Bounds of Chain-of-Thought Robustness: Reasoning Steps, Embed Norms, and Beyond
description: >-
  [ICLR 2026][learning_theory][Paper Note] This paper establishes the first set of theoretical upper bounds for the robustness of Chain-of-Thought (CoT) against input perturbations. Under the Lipschitz continuity assumption, it is proven that "more reasoning steps result in a smaller upper bound for output fluctuation, yet perturbations cannot be eliminated eve
tags:
  - ICLR 2026
  - learning_theory
date: 2026-05-08
content_hash: 9fc883c083be6600
---
# Bounds of Chain-of-Thought Robustness: Reasoning Steps, Embed Norms, and Beyond

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=cusZbViSLd](https://openreview.net/forum?id=cusZbViSLd)  
**Code**: To be confirmed  
**Area**: Learning Theory / LLM Reasoning  
**Keywords**: Chain-of-Thought Robustness, Input Perturbation, Lipschitz Continuity, Linear Self-Attention, Embedding Norms

## TL;DR
This paper establishes the first set of theoretical upper bounds for the robustness of Chain-of-Thought (CoT) against input perturbations. Under the Lipschitz continuity assumption, it is proven that "more reasoning steps result in a smaller upper bound for output fluctuation, yet perturbations cannot be eliminated even with infinite steps." Taking Linear Self-Attention (LSA) as a case study, the work demonstrates that the "tolerable input perturbation radius is negatively correlated with the norms of input embeddings and hidden state vectors," with experimental curves across 4 mainstream LLMs and 3 reasoning datasets aligning with the theory.

## Background & Motivation
**Background**: Chain-of-Thought (CoT) enables large models to generate reasoning processes step-by-step, significantly improving performance on complex tasks. However, extensive empirical studies have found CoT to be extremely sensitive to inputs—slight modifications in the prompt can lead to massive fluctuations in the final answer. To mitigate this, the community has developed numerous prompt optimization methods (e.g., TextGrad using "textual gradients" for rewriting, OPRO letting models iteratively generate better prompts).

**Limitations of Prior Work**: Existing works almost exclusively treat "CoT robustness" as a **purely empirical phenomenon**—observing that perturbations amplify and then employing various tricks to suppress them. None have clarified exactly **how** perturbations propagate through the reasoning process or **why** they amplify into output fluctuations. Without this mechanistic understanding, prompt optimization remains limited to ad-hoc parameter tuning.

**Key Challenge**: A natural yet unanswered question persists: What determines CoT's robustness to input perturbations? Is it the length of the reasoning chain? Specific model properties? Or the training data? The individual contributions and directions of these factors remain entirely unexplored.

**Goal**: This work decomposes the problem into two layers: (1) Under general architecture-agnostic assumptions, how does the number of reasoning steps $K$ affect the upper bound of output fluctuations? (2) Within specific attention models, which internal quantities (vector norms, training data distribution, residual coefficients) determine robustness?

**Key Insight**: The authors adopt the perspective of viewing CoT as a **multi-step iterative process**—where the output of each step serves as the input for the next, $h_{k,x}=f(h_{k-1,x},x)$. By applying a mild and widely adopted **Lipschitz continuity** assumption (constraining the rate of output growth to prevent explosion) to the mapping function $f$, the propagation of perturbations can be precisely characterized via recurrence relations.

**Core Idea**: Instead of inventing new methods, the authors **derive provable upper bounds for CoT robustness**. They first prove a general upper bound for output fluctuations (revealing the role and limits of reasoning steps) and then use Linear Self-Attention as a case study to map these bounds onto observable vector norms, theoretically deriving actionable robustness levers.

## Method

### Overall Architecture
The paper follows a chain from "general to specific" and "theory to verification." There are no trainable modules; the core consists of two theorems.

Let the embedding vectors of the user query and output be $x,y\in\mathbb{R}^d$, with input perturbation $\delta$ such that the perturbed input is $\tilde{x}=x+\delta$. CoT is modeled as a multi-step iteration: the $k$-th hidden state is $h_{k,x}=f(h_{k-1,x},x)$ (with $h_{1,x}=f(0,x)$), and the output fluctuation caused by perturbation at step $k$ is denoted as $\varepsilon_k=h_{k,\tilde{x}}-h_{k,x}$. The derivation is split into four parts:

1.  **General Upper Bound**: Under the Lipschitz assumption for $f$, the upper bound of the final step output fluctuation $\|\varepsilon_K\|$ is expanded (Theorem 1) to analyze the impact of "reasoning steps $K$."
2.  **Tolerable Perturbation Radius**: Conversely, if the output fluctuation must remain within an acceptable range $\|\varepsilon\|\le R$, what is the maximum allowable input perturbation $\|\delta\|$? Taking $K\to\infty$ yields a **non-zero lower bound**, proving that "infinite reasoning cannot eliminate perturbations."
3.  **LSA Implementation**: The abstract Lipschitz constants $C,\gamma$ are mapped to a Linear Self-Attention model (Theorem 2). It is proven that the tolerable perturbation radius is negatively correlated with input embedding norm $R_x$ and hidden state norm $R_h$. The effects of training data covariance $\Gamma$ and residual coefficient $\eta$ are also discussed.
4.  **Operational Levers**: Theorem 2 suggests a prompt selection criterion—choosing prompts that maximize the perturbation upper bound to enhance robustness without modifying the model.

### Key Designs

**1. Modeling CoT as Lipschitz Iterations to Derive "More Steps, More Robust" Upper Bound**

To address the lack of clarity on perturbation propagation, the mapping function $f$ is constrained by Lipschitz continuity: constants $C,\gamma\in\mathbb{R}$ exist such that 

$$\|f(h_1,x_1)-f(h_2,x_2)\|\le \gamma\|h_1-h_2\|+C\|x_1-x_2\|.$$

By recursively expanding the input $h$ with the previous step's output, **Theorem 1** is derived:

$$\|\varepsilon_K\|\le\Big(A\gamma^K+\tfrac{C}{1-\gamma}(1-\gamma^K)\Big)\|\delta\|,\quad A=\max\tfrac{\|\varepsilon_1\|}{\|\delta\|}.$$

This bound decomposes perturbation propagation into **two physically distinct paths**: (i) Perturbations hidden in the hidden states, which are multiplied by $\gamma$ at each step, resulting in the coefficient $A\gamma^K$—if $\gamma<1$ (a reasonable assumption for well-trained models), this decays exponentially with steps; (ii) Perturbations hidden in the input vector, which accumulate over steps because $x$ remains constant, resulting in the coefficient $\sum_{k=1}^K C\gamma^k=\tfrac{C}{1-\gamma}(1-\gamma^K)$. With fixed $C, \gamma$, the bound is determined by $K$ and $\|\delta\|$. Larger $K$ yields a tighter bound, theoretically explaining why "longer, more structured reasoning chains suppress perturbations."

**2. Solving for Input Perturbation Radius: "Infinite Steps Cannot Eliminate Noise"**

Practical tasks tolerate some output fluctuation without changing the final answer (e.g., classification requires only that the top-probability option does not flip). Setting an acceptable boundary $\|\varepsilon\|\le R$ and requiring the right side of Theorem 1 $\le R$ leads to (Eq. 3):

$$\|\delta\|\le\frac{R}{A\gamma^K+\tfrac{C}{1-\gamma}(1-\gamma^K)}.$$

The tolerable radius shrinks as $R$ increases or as $C,\gamma$ increase (indicating the model's inability to suppress fluctuations). The real insight comes from taking $\gamma<1$ and $K\to\infty$ (Eq. 4):

$$\|\delta\|\le\frac{R(1-\gamma)}{C}.$$

This is a **non-zero constant**—even with infinite reasoning, if input perturbation exceeds this threshold, the model cannot eliminate the resulting output fluctuation. The paper provides an intuitive example: if a numerical reasoning problem is "perturbed" into a coding problem, no amount of reasoning will recover the original answer. This conclusion corrects the optimistic expectation that "models can stabilize simply by thinking more": CoT can **attenuate** but not **neutralize** perturbations.

**3. LSA Case Study: Mapping Bounds to Observable Vector Norms**

Since $C,\gamma$ in Theorem 1 are abstract, the third part maps them to a specific model. Linear Self-Attention (LSA) is chosen—a simplified Transformer layer where non-linear softmax is replaced by linear mapping for analytical tractability. Let $E=[h,x]$,

$$f_{\mathrm{LSA}}(h,x;\theta)=E+\frac{W^{PV}E\,E^\top W^{KQ}E}{\rho}.$$

Substituting optimal parameters $\theta^*$ and introducing a residual coefficient $\eta\in(0,1)$ to prevent gradient explosion, **Lemma 1** provides upper bounds for $C,\gamma$ (where $\alpha=(\mathrm{Tr}(\Gamma^{-2}))^{-1/4}$, given $\|x\|\le R_x,\|h\|\le R_h$):

$$C\le\eta+\alpha^{-1}\|\Gamma^{-1}\|R_h^2,\qquad \gamma\le\sqrt{\eta^2+4R_x^2\alpha^{-2}\|\Gamma^{-1}\|^2R_h^2}.$$

Substituting these into Eq. 3 yields **Theorem 2**—the certified tolerable perturbation radius for LSA at step $K$ ($\beta=\alpha^{-1}sR_h^2$，$s=\|\Gamma^{-1}\|$):

$$\|\delta\|\le\frac{(1-\gamma)R}{(\eta+\beta)+A(1-\gamma)(1+\beta)\gamma^K},\qquad K\to\infty:\ \|\delta\|\le\frac{(1-\gamma)R}{\eta+\beta}.$$

Five factors and their directions are identified: $R$ (acceptable range, positive correlation); $R_x$ (input embedding norm, **negative correlation**—larger input vectors weaken robustness); $R_h$ (hidden state norm, **negative correlation**—larger internal states are more easily swayed); $\Gamma$ (training data covariance, inconsistency increases sensitivity); $\eta$ (residual coefficient, larger values preserve more input information and perturbations). This translates an abstract problem into two concrete levers: **minimizing vector norms during inference and increasing data consistency during training.**

**4. Prompt Selection Criterion: Maximizing $A^{-1}$**

Finally, the theory is converted into an operational tool. Let $\tau=\alpha^{-1}s$, $F$ be the right side of Theorem 2, and $A=(R_xR_h)^2$. Differentiation yields (Eq. 9):

$$\frac{\partial F}{\partial A}=-\frac{R\tau^2}{2(\eta+\tau R_h^2)\sqrt{\eta^2+\tau^2 A}}<0.$$

Since $F$ is strictly negatively correlated with $A$, a larger $A^{-1}$ implies a larger tolerable perturbation radius. For any problem: use all candidate prompts to construct inputs, calculate $A$ from the embedding and final hidden state norms, and **select the prompt with the largest $A^{-1}$**. This zero-training, forward-only strategy serves to validate the utility of the theory.

## Key Experimental Results

Experiments were conducted on 4 mainstream LLMs (Llama2-7b, Llama3.1-8b, Deepseek-R1-Distilled-Llama3.1-8b denoted as Llama-R1-8b, Qwen3-8b) across 3 reasoning datasets (MATH, MMLU-Pro, GPQA). Two metrics: **EM** (Exact Match, higher is better) and **OF** (Output Fluctuation, normalized entropy of answers across multiple prompts, lower is better). Input perturbations were constructed from prompts generated during TextGrad / OPRO / CFPO optimization.

### Main Results: Stronger Models are More Robust

| Model | MATH EM | MATH OF | MMLU-Pro EM | MMLU-Pro OF | GPQA EM | GPQA OF |
|------|---------|---------|-------------|-------------|---------|---------|
| Llama2-7b | 14.2 | 0.475 | 11.2 | 0.622 | 17.5 | 0.509 |
| Llama3.1-8b | 45.8 | 0.366 | 41.0 | 0.350 | 26.6 | 0.467 |
| Llama-R1-8b | 64.8 | 0.158 | 44.8 | 0.292 | 28.5 | 0.371 |
| Qwen3-8b | **77.2** | **0.097** | **46.9** | **0.162** | **37.3** | **0.214** |

As capability increases, EM rises while OF falls. Theoretical explanation: Stronger models often have (i) more consistent training data (better cleaning/synthesis) → $\Gamma$ raises the perturbation bound, (ii) longer/more structured reasoning chains → $K$ increases in Theorem 1, tightening the fluctuation bound. Llama-R1 and Qwen3, which support Long-CoT, exemplify these effects.

### Key Findings
- **Reasoning Steps (Figs 2, 3)**: OF generally decreases as CoT steps $K$ increase, consistent with Theorem 1. However, OF **converges to a stable non-zero level** after approximately 16 steps, empirically confirming Eq. 4 "infinite steps cannot eliminate perturbations." EM does not necessarily rise with $K$—problems requiring more steps are often harder, leading to potential accuracy drops.
- **Embedding Norm Threshold**: A **sudden jump** in OF occurs when input embedding norms rise from 60 to 70, suggesting a threshold models can handle. Beyond this, most perturbations exceed Theorem 2's bound, causing drastic fluctuations.
- **Correlation of Hidden State Norms**: Most data points for hidden state norms cluster in a narrow (140, 150) range—well-trained models tend to encode data into small, fixed norm intervals for anti-perturbation. Because $\gamma$ depends on the norm's upper bound rather than its specific value, and LayerNorm provides buffering, OF changes less obviously with hidden state norms (Pearson 0.229).
- **Effectiveness of Prompt Selection (Table 3)**: Selecting prompts by maximizing $A^{-1}$ outperformed TextGrad / OPRO / CFPO across all settings. For example, Llama3.1-8b on GPQA improved from a base of 23.7 to **32.3** (compared to CFPO's 27.6); Qwen3-8b reached **49.2** on MMLU-Pro (top baseline 45.9).

## Highlights & Insights
- **From Empirical Phenomena to Provable Bounds**: CoT robustness has long been treated as a tuning problem. This work provides the first set of bounds connecting input perturbation to output fluctuation, where every term corresponds to an observable or controllable variable (steps, norms, consistency, residuals).
- **"Attenuation but not Neutralization" is a Crucial Insight**: The non-zero lower bound in Eq. 4 theoretically refutes the naive expectation that "thinking more" always stabilizes a model. It highlights that the robustness ceiling is determined by the model and its data.
- **Physical Decomposition of Two Perturbation Paths**: Theorem 1 splits perturbations into a "hidden state path" (exponential decay via $\gamma$) and an "input path" (cumulative sum converging to $\tfrac{C}{1-\gamma}$), explaining why long chains suppress some but not all noise.
- **Direct Application to Prompt Selection**: Theorem 2 translates abstract constants into embedding/hidden state norms. The $A^{-1}$ criterion is a zero-training tool that can be migrated to any scenario requiring the selection of stable prompts.

## Limitations & Future Work
- **Reliance on Lipschitz Continuity and LSA Simplification**: Theorem 1 requires Lipschitz $f$ and $\gamma<1$. Theorem 2 simplifies the analysis to Linear Self-Attention (linearized softmax, $\rho=1$). Whether non-linear attention, multi-head structures, and deep stacking in real Transformers maintain these conclusions end-to-end remains partially discussed in the appendix but not fully verified in the main text.
- **Theoretical Factors ($\Gamma, \eta$) remain Unverified**: The authors note that verifying data consistency $\Gamma$ and residual coefficients $\eta$ requires modifying training data and architectures. This work focuses on theoretical derivation to inspire future work; empirical tests cover only $R, R_x, R_h$.
- **Saturation of the OF Metric**: Output fluctuation is measured by normalized entropy, which has a maximum value constrained by the number of prompts. This causes the OF to "plateau" when perturbations exceed 0.2 or norms are too large, potentially masking trends.

## Related Work & Insights
- **vs. Empirical CoT Robustness Methods**: Unlike prior works that use denoising or defensive structures, this work explains the **mechanism** of perturbation propagation.
- **vs. Prompt Optimization (TextGrad/OPRO)**: While others search the prompt space, this work derives a **closed-form criterion** ($\max A^{-1}$) from Theorem 2, which outperforms search-based methods in experiments by being more principled.
- **vs. CoT Theoretical Analysis**: While following the multi-step iteration model of previous authors, this work is the first to provide **explicit upper bounds** and link them to quantifiable vector norms.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First provable input-output fluctuation bounds for CoT.
- Experimental Thoroughness: ⭐⭐⭐⭐ Systematic verification across models/datasets, though some theoretical factors ($\Gamma/\eta$) lack empirical tests.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logical flow mapping theory to experiments.
- Value: ⭐⭐⭐⭐ Defines a theoretical ceiling for CoT and provides actionable levers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

1.  **TextGrad: Automatic "Gradient" Optimization of Text via Large Language Models** (ArXiv 2024)
2.  **Large Language Models as Optimizers** (ICLR 2024)
3.  **Measuring and Improving Chain-of-Thought Robustness in Large Language Models** (ACL 2023)

</div>

<!-- RELATED:END -->

## Related Papers

- [\[ICLR 2026\] Poisson Midpoint Method for Log-Concave Sampling: Beyond the Strong Error Lower Bounds](poisson_midpoint_method_for_log_concave_sampling_beyond_the_strong_error_lower_b.md)
- [\[ICLR 2026\] Computing Equilibrium beyond Unilateral Deviation](computing_equilibrium_beyond_unilateral_deviation.md)
- [\[ICLR 2026\] Reshaping Reasoning in LLMs: A Theoretical Analysis of RL Training Dynamics through Pattern Selection](reshaping_reasoning_in_llms_a_theoretical_analysis_of_rl_training_dynamics_throu.md)
- [\[ICLR 2026\] Better Bounds for the Distributed Experts Problem](better_bounds_for_the_distributed_experts_problem.md)
- [\[ICML 2026\] Multi-task Linear Regression without Eigenvalue Lower Bounds: Adaptivity, Robustness and Safety](../../ICML2026/learning_theory/multi-task_linear_regression_without_eigenvalue_lower_bounds_adaptivity_robustne.md)

</div>

<!-- RELATED:END -->
