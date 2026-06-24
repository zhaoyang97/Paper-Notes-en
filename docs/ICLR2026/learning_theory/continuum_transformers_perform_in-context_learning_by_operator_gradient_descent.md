---
title: >-
  [Paper Note] Continuum Transformers Perform In-Context Learning by Operator Gradient Descent
description: >-
  [ICLR2026][Learning Theory][Continuum Transformer] This paper provides the first theoretical characterization of the In-Context Learning (ICL) phenomenon in "Continuum Transformers" (Transformer variants that handle infinite-dimensional function inputs for PDE surrogate modeling). It proves that forward propagation is equivalent to performing gradient descent in an **operator RKHS**. In the limit of infinite depth, the model recovers the Bayesian optimal predictor…
tags:
  - "ICLR2026"
  - "Learning Theory"
  - "In-Context Learning"
  - "Neural Operators"
  - "Continuum Transformer"
  - "Operator RKHS"
  - "Operator Gradient Descent"
  - "Bayesian Optimal Predictor"
date: 2026-05-08
content_hash: 3c4443abb06e7a51
---

# Continuum Transformers Perform In-Context Learning by Operator Gradient Descent

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=X63V2CWjj3](https://openreview.net/forum?id=X63V2CWjj3)  
**Code**: https://github.com/yashpatel5400/opicl  
**Area**: Learning Theory / In-Context Learning / Neural Operators  
**Keywords**: In-Context Learning, Continuum Transformer, Operator RKHS, Operator Gradient Descent, Bayesian Optimal Predictor

## TL;DR
This paper provides the first theoretical characterization of the In-Context Learning (ICL) phenomenon in "Continuum Transformers" (Transformer variants that handle infinite-dimensional function inputs for PDE surrogate modeling). It proves that forward propagation is equivalent to performing gradient descent in an **operator RKHS**. In the limit of infinite depth, the model recovers the Bayesian optimal predictor, and the parameters implementing this gradient descent are shown to be stationary points of the pre-training objective.

## Background & Motivation
**Background**: Standard Transformers have long been observed to possess ICL capabilities—improving prediction accuracy on new tasks without updating parameters by simply placing training samples $(x^{(i)}, y^{(i)})$ in the context window. A series of theoretical works (Akyürek 2022, Garg 2022, Dai 2022, Cheng et al. 2023) identified the underlying mechanism: under specific choices of $W_k, W_q, W_v$, a single forward pass is equivalent to performing several steps of gradient descent on the context task. Cheng et al. 2023 further characterized this using kernel methods as functional gradient descent on an RKHS.

**Limitations of Prior Work**: All these results are restricted to **standard Transformers**, which process finite-dimensional vector inputs. However, there is an orthogonal subfield in machine learning—using neural operators to accelerate partial differential equation (PDE) solving. In this context, sequence elements are **infinite-dimensional functions** rather than finite-dimensional vectors. Consequently, Calvello et al. 2024 proposed the "Continuum Transformer" to generalize attention to function inputs. Surprisingly, these Transformers **also exhibit ICL**—inserting several PDE solution pairs into the context window allows for efficient solving of new related PDEs, giving rise to the "In-Context Operator Network" (ICON) research branch.

**Key Challenge**: Despite ICON's strong empirical performance, this **generalized, functional ICL has never been theoretically characterized**. The proof tools for the "forward pass = gradient descent" equivalence in finite dimensions (classical Representer Theorem, finite-dimensional Gaussian processes, optimization analysis of matrix derivatives) cannot be directly applied to infinite-dimensional function spaces. Many steps that are trivial in finite dimensions—such as explicit gradient forms, derivatives with respect to parameters, and convergence of infinite-layer compositions—must be rigorously re-established in operator spaces.

**Goal**: This work aims to elevate the theoretical line of "ICL = Kernel Gradient Descent" from finite-dimensional vector spaces to infinite-dimensional operator spaces. It addresses three questions: What is the forward pass of a Continuum Transformer doing? What does it converge to at infinite depth? And can the parameters that implement gradient descent be naturally obtained through training?

**Key Insight**: The authors discovered that by modifying the similarity nonlinearity in continuum attention from "scalar-valued" to "operator-valued," the entire ICL process can be analyzed within an **operator RKHS** framework. This allows the reuse of generalized versions of the Representer Theorem and Gaussian measures on Hilbert spaces as mathematical tools.

**Core Idea**: The paper proves that layer-wise inference in a Continuum Transformer is equivalent to step-wise **operator gradient descent** on an operator RKHS. At infinite depth, it recovers the Bayesian optimal predictor. Furthermore, the parameters implementing this behavior are shown to be stationary points of the training objective. The work delivers a mathematical framework that allows theorists to use infinite-dimensional conclusions to rigorously support finite-dimensional intuitions.

## Method

### Overall Architecture
This is a purely theoretical paper whose main objective is to shift the characterization of "ICL = Gradient Descent" from finite-dimensional Transformers to Continuum Transformers that handle infinite-dimensional functions. The logical chain is as follows: first, **remodel** the continuum attention layer by rewriting the similarity nonlinearity $H$ from a scalar value to an operator value to enable analysis in an operator RKHS; then, prove that under a specific set of parameters, **layer-wise inference is exactly equal to step-wise operator gradient descent** (Theorem 3.1); next, prove that in the **infinite depth limit, this gradient descent converges to the Bayesian optimal predictor** (Proposition 3.3); finally, prove that the parameters implementing this gradient descent are indeed **stationary points of the training objective** (Theorem 3.6), closing the loop by showing that training naturally drives the model toward the "gradient descent" configuration.

In the setup, the neural operator learns a mapping $\widehat{G}: \mathcal{A}\to\mathcal{U}$ between function spaces (often the time-evolution operator of a PDE, with inputs and outputs in the same Hilbert space $X$). Continuum attention replaces the $W_k, W_q, W_v$ matrices with **linear operators**, typically implemented via FNO-style kernel integral transforms: $W_q x_i = \mathcal{F}^{-1}(R_q \odot \mathcal{F} x_i)$, where $R_q$ is the Fourier parametrization of the query kernel. The context window follows the finite-dimensional ICL format: $n$ input-output function pairs $(f^{(i)}, u^{(i)})$ are sequenced, with the $(n{+}1)$-th output position filled with 0, and the model predicts $u^{(n+1)}$ from the $(n{+1})$-th input $f$.

### Key Designs

**1. Operator-Valued Nonlinearity: Rewriting Continuum Attention for RKHS Analysis**

In original continuum attention $\mathrm{ContAttn}(X) = (W_v X)\,M\,\mathrm{softmax}((W_q X),(W_k X))$, the attention weight matrix remains in $\mathbb{R}^{n\times n}$, meaning similarities are **scalar-valued**. The authors point out that if this scalar similarity $H: Q^{n+1}\times K^{n+1}\to \mathbb{R}^{(n+1)\times(n+1)}$ is maintained, the problem **cannot be characterized as gradient descent on an RKHS**. Thus, they introduce a non-trivial generalization—making the nonlinearity **operator-valued**: $H: Q^{n+1}\times K^{n+1}\to (\mathcal{L}(V))^{(n+1)\times(n+1)}$, where $\mathcal{L}(V)$ is the set of bounded linear operators from $V \to V$. Under this, the layer-wise update of an $m$-layer Continuum Transformer is written as:

$$Z_{\ell+1} = Z_\ell + \Big(\widetilde{H}(W_{q,\ell} X_\ell, W_{k,\ell} X_\ell)\, M\, (W_{v,\ell} Z_\ell)^{\top}\Big)^{\top},$$

where Key/Query operators only act on the input function rows $X_\ell$, and the Value operator acts on the full $Z_\ell$. The mask $M$ is the block diagonal $\big[\begin{smallmatrix}I_{n\times n}&0\\0&0\end{smallmatrix}\big]$. This step is the key to the entire paper.

**2. Operator Gradient Descent Equivalence: Layer-wise Inference = One Step of Gradient Descent**

The context task is to find $O^* = \arg\min_O \sum_i \|u^{(i)} - O f^{(i)}\|_X^2$ from samples $\{(f^{(i)},u^{(i)})\}$, which can be solved via iterations $O_{\ell+1}=O_\ell - \eta_\ell \nabla L(O_\ell)$. Theorem 3.1 gives the core equivalence: Let $\kappa: X\times X\to\mathcal{L}(X)$ be any **operator-valued kernel** and $\mathcal{O}$ be its induced operator RKHS. Let $O_\ell$ be the result of the $\ell$-th step of operator gradient descent ($O_0=0$). There exist scalar step sizes $r'_0,\dots,r'_m$ such that when the Continuum Transformer takes parameters:

$$[\widetilde{H}(U,W)]_{i,j}=\kappa(u^{(i)},w^{(j)}),\quad W_{v,\ell}=\begin{bmatrix}0&0\\0&-r'_\ell I\end{bmatrix},\quad W_{q,\ell}=I,\quad W_{k,\ell}=I$$

then for any test function $f$, $T_\ell(f)= -O_\ell f$. In other words, **the context prediction of the $\ell$-th layer is exactly the result of $\ell$ steps of operator gradient descent**. The derivation requires a **generalized Representer Theorem** for operator RKHS to obtain the explicit gradient form.

**3. Recovering Bayesian Optimal Predictor at Infinite Depth: Gaussian Measures on Hilbert Spaces**

In finite dimensions, it has been shown that if outputs come from a GP marginal with kernel $\kappa$, a Transformer of depth $m \to \infty$ recovers the Bayesian optimal predictor. To reproduce this in the operator setting, the authors define "Gaussian measures for sampling the true operator $O$." Using Jorgensen & Tian 2024, they generalize GPs to Hilbert spaces: $U|F \sim \mathcal{N}(0, K(F))$ is a $\kappa$-Gaussian random variable when $[K(F)]_{i,j}=\kappa(f^{(i)}, f^{(j)})$ and any projection $\langle v, u^{(i)}\rangle_X$ follows a Gaussian distribution with corresponding variance (Definition 3.2). Proposition 3.3 then proves that when depth $m \to \infty$, the Continuum Transformer's prediction converges to the **Best Linear Unbiased Predictor (BLUP)**, which is the **Bayesian optimal predictor** in the MSE sense according to Hilbert space kriging theory.

**4. Pre-training Convergence to Gradient Descent Parameters: Gradient Flow on Hilbert Functionals**

Theorem 3.6 proves that the parameters implementing operator gradient descent are **stationary points of the training objective**. The analysis uses **Fréchet differentiability** for functionals on Hilbert spaces. Under a **rotational symmetry** assumption (Assumption 3.4: there exists a self-adjoint invertible operator $\Sigma$ such that the distribution is invariant under $\Sigma^{1/2} M \Sigma^{-1/2}$), the authors show that the fixed points take the form $W_{q,\ell}=b_\ell \Sigma^{-1/2}$, $W_{k,\ell}=c_\ell \Sigma^{-1/2}$, and $W_v = \big[\begin{smallmatrix}0&0\\0&r_\ell I\end{smallmatrix}\big]$. When $\Sigma=I$, this reduces to the configuration in Theorem 3.1.

### Loss & Training
The context loss is defined as $L(W_v,W_q,W_k)=\mathbb{E}\big[\,\|[Z_{m+1}]_{2,n+1}+u^{(n+1)}\|_X^2\,\big]$. Theorem 3.6 characterizes training as minimizing the sum of the gradient norms of this loss with respect to $(r, W_q, W_k)$ in the Hilbert–Schmidt norm $\|\cdot\|_{HS}$, proving this sum is zero at the aforementioned fixed points.

## Key Experimental Results
Experiments verify three theoretical claims on $X=L^2(\mathbb{T}^2)$ using Hilbert–Schmidt integral operator kernels.

### Main Results

| Experiment | Theory Verified | Setup | Findings |
| :--- | :--- | :--- | :--- |
| BLUP Convergence | Prop. 3.3 | Fixed parameters from Theorem 3.1, 4 sets of $(k_x, k_y)$ kernels | When nonlinearity matches the data kernel, loss decreases monotonically and converges to the BLUP error level. |
| Poisson Equation | Prop. 3.3 (Robustness) | 2D Poisson $\Delta u=f$, unknown true kernel | Parameters exhibit expected optimization even without exact kernel matching; linear kernels perform best. |
| Parameter Convergence | Theorem 3.6 | 250-layer Transformer, 5 training runs | The pairwise HS cosine similarity of Key/Query operators converges to 1 during training, verifying the fixed-point characterization. |

### Key Findings
- **Kernel matching is a prerequisite for optimality**: If the nonlinearity $\widetilde{H}$ matches the data kernel $\kappa$, each layer corresponds to one step of operator gradient descent, eventually reaching the BLUP error.
- **Robustness exceeds theoretical guarantees**: Parameters converge even when some kernels violate specific technical assumptions, suggesting the existence of a stronger, layer-independent version of Theorem 3.6.
- **Reproducibility**: Results are stable across various operator samplings and training initializations.

## Highlights & Insights
- **Operator-valued nonlinearity is a decisive modeling choice**: This shift allows the infinite-dimensional problem to fall into the operator RKHS framework, enabling the use of the Representer Theorem.
- **Portable "Finite to Infinite" Toolkit**: The use of generalized Representer Theorems, Hilbert space Gaussian measures, and Fréchet calculus allows analysts to apply finite-dimensional intuitions to function spaces rigorously.
- **Theory directs practical improvement**: Since the RKHS is induced by the PDE parameter distribution, one can estimate $\kappa$ for specific PDE meta-learning tasks and use it to parameterize $\widetilde{H}$ directly.

## Limitations & Future Work
- **Dependence on structural assumptions**: Assumptions like rotational symmetry are common in optimization analysis but may not hold for real-world PDE data.
- **Difficulty in kernel selection**: Optimality requires matching the data kernel, which is often unknown in practice.
- **Weakness of Theorem 3.6**: Experiments suggest a stronger convergence result that is independent of layer depth $\ell$, which remains to be proven.
- **Limited scale**: Validation is restricted to synthetic Gaussian fields and Poisson equations on $L^2(\mathbb{T}^2)$, lacking testing on complex, high-dimensional, or non-periodic real PDE benchmarks.

## Related Work & Insights
- **vs. Cheng et al. 2023**: This work elevates the "ICL = Kernel Gradient Descent" characterization from finite vector spaces to infinite operator spaces, which is non-trivial and requires different mathematical machinery.
- **vs. Cole et al. 2024**: While both study Continuum Transformer ICL, Cole et al. focus on sample complexity and generalization for linear elliptic PDEs, whereas this work focuses on the mechanism of the forward pass.
- **vs. ICON Series**: These empirical works demonstrated the power of in-context operator networks; this paper provides the missing theoretical grounding for why they work.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<div class="related-papers" markdown="1">

- **Akyürek, E., et al. (2022).** What learning algorithm is in-context learning? Investigations with linear models.
- **Cheng, S., et al. (2023).** Transformers as algorithms: Generalization and stability in in-context learning.
- **Calvello, A., et al. (2024).** Continuum attention: A continuous-time view of transformers.

</div>

## Related Papers

- [\[ICLR 2026\] Transformers Learn Latent Mixture Models In-Context via Mirror Descent](transformers_learn_latent_mixture_models_in-context_via_mirror_descent.md)
- [\[ICLR 2026\] Interactive Learning of Single-Index Models via Stochastic Gradient Descent](interactive_learning_of_single-index_models_via_stochastic_gradient_descent.md)
- [\[ICLR 2026\] Transformers with Endogenous In-Context Learning: Bias Characterization and Mitigation](transformers_with_endogenous_in-context_learning_bias_characterization_and_mitig.md)
- [\[ICLR 2026\] Adversarially Pretrained Transformers May Be Universally Robust In-Context Learners](adversarially_pretrained_transformers_may_be_universally_robust_in-context_learn.md)
- [\[ICLR 2026\] In-Context Algorithm Emulation in Fixed-Weight Transformers](in-context_algorithm_emulation_in_fixed-weight_transformers.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Transformers Learn Latent Mixture Models In-Context via Mirror Descent](transformers_learn_latent_mixture_models_in-context_via_mirror_descent.md)
- [\[ICLR 2026\] Transformers Trained via Gradient Descent Can Provably Learn a Class of Teacher Models](transformers_trained_via_gradient_descent_can_provably_learn_a_class_of_teacher_.md)
- [\[ICLR 2026\] Transformers with Endogenous In-Context Learning: Bias Characterization and Mitigation](transformers_with_endogenous_in-context_learning_bias_characterization_and_mitig.md)
- [\[ICLR 2026\] Interactive Learning of Single-Index Models via Stochastic Gradient Descent](interactive_learning_of_single-index_models_via_stochastic_gradient_descent.md)
- [\[ICLR 2026\] In-Context Algorithm Emulation in Fixed-Weight Transformers](in-context_algorithm_emulation_in_fixed-weight_transformers.md)

</div>

<!-- RELATED:END -->
