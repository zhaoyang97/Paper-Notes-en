---
title: >-
  [Paper Note] SVRG and Beyond via Posterior Correction
description: >-
  [ICML2026][Optimization][SVRG] This paper demonstrates that the classic variance reduction algorithm SVRG is essentially a special case of Bayesian "Posterior Correction" (PoCo) under an isotropic Gaussian posterior. Based on this, it automatically derives two new extensions previously difficult to obtain: a Newton-type variant that simultaneously corrects the Hessian, and an Adam-type variant (IVON-PoCo) scalable to deep learning.
tags:
  - "ICML2026"
  - "Optimization"
  - "SVRG"
  - "Variance Reduction"
  - "Posterior Correction"
  - "Bayesian Learning Rule"
  - "Natural Gradient"
date: 2026-05-08
content_hash: 3f14c7c2d2eb7986
---

# SVRG and Beyond via Posterior Correction

**Conference**: ICML2026  
**arXiv**: [2512.01930](https://arxiv.org/abs/2512.01930)  
**Code**: Yes (The paper claims it is public on GitHub, though no specific URL is provided in the text. ⚠️ Refer to the original paper)  
**Area**: Optimization Theory / Variational Bayes / Variance Reduction  
**Keywords**: SVRG, Variance Reduction, Posterior Correction, Bayesian Learning Rule, Natural Gradient

## TL;DR
This paper demonstrates that the classic variance reduction algorithm SVRG is essentially a special case of Bayesian "Posterior Correction" (PoCo) under an isotropic Gaussian posterior. Based on this, it automatically derives two new extensions previously difficult to obtain: a Newton-type variant that simultaneously corrects the Hessian, and an Adam-type variant (IVON-PoCo) scalable to deep learning.

## Background & Motivation

**Background**: Variance reduction is a powerful tool for accelerating stochastic optimization. SVRG uses occasional full-batch gradients to stabilize subsequent mini-batch updates. For over a decade, it has inspired a large family of variants including SAGA, SARAH, SPIDER, and $\alpha$-SVRG.

**Limitations of Prior Work**: Despite over ten years of research on the SVRG family, it has never been fundamentally linked to Bayesian methods. Existing work at most treats SVRG as a "tool to accelerate Bayesian inference" (e.g., accelerating Stochastic Variational Inference), which merely wraps variance reduction outside the Bayesian process and fails to reveal the deeper equivalence between the two.

**Key Challenge**: The absence of this connection means one cannot systematically "go beyond SVRG" using Bayesian principles. For example, deriving a Newton-type SVRG that corrects the Hessian simultaneously is difficult to achieve naturally with existing variance reduction techniques (most Newton-SVRGs only correct the gradient and never the Hessian).

**Goal**: To bridge this missing gap between SVRG and Bayesian methods and utilize it to derive new algorithms that reach beyond the capabilities of traditional SVRG.

**Key Insight**: The authors noticed a recently proposed Bayesian method called Posterior Correction (PoCo). Originally designed for knowledge transfer tasks like continual learning and model merging, it seemingly has nothing to do with variance reduction. However, comparing SVRG’s double-loop structure with PoCo’s mechanism of "using an old posterior to correct a new update" reveals a high degree of formal consistency.

**Core Idea**: Prove that SVRG is a special case of PoCo under an isotropic Gaussian posterior. By switching to more flexible exponential family posteriors, new SVRG extensions can be automatically derived. Thus, SVRG’s "gradient correction" receives a new interpretation: it is a knowledge transfer mechanism between new and old gradients.

## Method

### Overall Architecture

The backbone of the paper is a derivation chain of "Probabilistic → Generalized → Specialized → Re-generalized" rather than a data pipeline. Formulas are used here to clarify the logic.

The starting point is reformulating Empirical Risk Minimization (ERM) $\bm{\theta}_*=\arg\min_{\bm\theta}\sum_{i=1}^N \ell_i(\bm\theta)$ via Variational Bayes (VB) as an optimization over a distribution $q(\bm\theta)$: $q_*=\arg\min_{q}\sum_i \mathbb{E}_q[\ell_i]+\mathbb{D}_{\rm KL}[q\|p_0]$. This is solved using the Bayesian Learning Rule (BLR)—performing natural gradient descent on the natural parameters $\bm\lambda$. BLR can also be written in a "Bayesian update" form $q\leftarrow q^{1-\eta}\prod_i\exp(-\eta\hat\ell_i)$, where $\hat\ell_i(\bm\theta)=\widetilde\nabla\mathcal{L}_i(\bm\lambda)^\top \mathbf{T}(\bm\theta)$ is the linearized "site function" of the loss.

Building on this, PoCo uses an old parameter $\bm\lambda_{\rm out}$ to construct an old posterior $\hat q_{\rm out}$ and multiplies/divides it into the BLR update (effectively multiplying by 1, which does not change the result), obtaining an update with a correction term:

$$q\leftarrow q^{1-\eta}\,\hat q_{\rm out}^{\eta}\prod_{i=0}^{N}\exp\!\big(-\eta[\hat\ell_i-\hat\ell_{i|\rm out}]\big).$$

By converting this into a mini-batch, double-loop version and specializing it for different posterior families $q$, the framework yields three algorithms: SVRG, a Newton-type, and an Adam-type. The derivation chain is summarized in the table below:

| Posterior Family $q$ | Specialized Algorithm from PoCo | Corrected Object |
|:---|:---|:---|
| Isotropic Gaussian $\mathcal{N}(\bm m,\mathbf{I})$ | SVRG / VSGD-PoCo (Alg. 3) | Gradient |
| Full-covariance Gaussian $\mathcal{N}(\bm m,\mathbf{S}^{-1})$ | VON-PoCo (Newton-type, Alg. 5) | Gradient + Hessian (SVRH) |
| Diagonal Gaussian $\mathcal{N}(\bm m,\mathrm{diag}(\bm s)^{-1})$ | IVON-PoCo / IVON-PoCoMo (Alg. 4) | Gradient + Diag Hessian, Scalable |
| Bernoulli | SVRG-style update for STE (Mentioned) | Gradient |

### Key Designs

**1. Rewriting SVRG as "Posterior Correction": Variance reduction as knowledge transfer**

To address the gap where SVRG was never linked to Bayesian methods, the core contribution is proving that the two are different ways of writing the same update. By formulating PoCo's corrected update into an unbiased, single-sample mini-batch double-loop version:

$$q_{\rm in}\leftarrow q_{\rm in}^{1-\eta}\,\hat q_{\rm out}^{\eta}\exp\!\big(-\eta N[\hat\ell_{i|\rm in}-\hat\ell_{i|\rm out}]\big),$$

and expanding it via natural parameters $\bm\lambda_{\rm in}$, one obtains a formula that maps one-to-one to the SVRG inner loop update ($\mathbf{g}_{\rm in}=\nabla\ell_i(\bm\theta_{\rm in})-\nabla\ell_i(\bm\theta_{\rm out})+\frac1N\mathbf{g}_{\rm out}$)—where $\bm\theta$ is replaced by $\bm\lambda$ and standard gradients by natural gradients (Theorem 1). When the posterior is set to an isotropic Gaussian $q=\mathcal{N}(\bm\theta\mid\bm m,\mathbf{I})$ and the delta method is applied (i.e., sampling noise $\bm\epsilon\leftarrow 0$, equivalent to $\mathbb{E}_q[\ell_i]\approx\ell_i(\bm m)$), the update reduces exactly to SVRG (Theorem 2). The resulting stochastic algorithm is called VSGD-PoCo; its only difference from SVRG is the addition of Gaussian weight perturbations $\bm\theta = \bm m + \bm\epsilon$ at two points. This connection provides a new interpretation for SVRG: the full-batch gradient represents the aggregation of old knowledge, and gradient correction is the use of old knowledge to stabilize mini-batch steps—effectively a knowledge transfer between new and old gradients.

**2. Newton-type Extension: Correcting the Hessian simultaneously (SVRH)**

Standard SVRG only corrects the gradient. The paper points out that once the posterior is changed to a full-covariance Gaussian $q=\mathcal{N}(\bm m,\mathbf{S}^{-1})$, the PoCo framework "automatically" requires the precision matrix (i.e., Hessian) to be corrected as well—this is not a manual addition but an inevitable result of formula 17 under a full Gaussian (Theorem 3). The mean update becomes Newton-style (with a preconditioner $\mathbf{S}_{\rm in}^{-1}$ and a proximal term $\mathbf{H}_{\rm out\backslash i}(\bm m_{\rm in}-\bm m_{\rm out})$), and the precision matrix is updated using a "Stochastic Variance Reduced Hessian" (SVRH) estimate:

$$\mathbf{S}_{\rm in}\leftarrow(1-\eta)\mathbf{S}_{\rm in}+\eta N\big[\mathbb{E}_{q_{\rm in}}[\nabla^2\ell_i]+\bar{\mathbf{H}}_{\rm out\backslash i}\big].$$

This results in VON-PoCo. The authors emphasize that such Hessian correction would not appear if SVRG were naively applied to Bayesian algorithms—it is the natural gradient in PoCo that allows it to emerge. To their knowledge, no previous Newton-type SVRG corrects the Hessian in this way.

**3. Adam-type Scalable Extension: IVON-PoCo / IVON-PoCoMo**

Full covariance is impractical for large models, so a diagonal Gaussian $q=\mathcal{N}(\bm m,\mathrm{diag}(\bm s)^{-1})$ is used instead, making the storage overhead comparable to AdamW. Applying PoCo to the IVON optimizer yields IVON-PoCo (or IVON-PoCoMo with momentum). It avoids full-batch computations—which are expensive and unrealistic in online scenarios like LLM pre-training—by using a "mega-batch" (potentially dozens of times the size of an inner mini-batch) to gradually estimate the full-batch gradient/Hessian. Since the mega-batch deviates from original SVRG, the paper uses a coefficient $\alpha < 1$ to downweight the correction term:

$$q_{\rm in}\leftarrow q_{\rm in}^{1-\eta}\,\hat q_{\rm out}^{\eta\alpha}\exp\!\big(-\eta N[\hat\ell_{i|\rm in}-\alpha\hat\ell_{i|\rm out}]\big).$$

When $\alpha=0$, it reverts to standard BLR; when $\alpha=1$, it represents perfect correction under a full batch. Interestingly, applying this to an isotropic Gaussian exactly recovers $\alpha$-SVRG (Yin et al., 2025), though this paper arrives there via "mega-batches" rather than "early variance reduction scheduling." Computation and memory costs are comparable to implementing $\alpha$-SVRG with Adam, and Hessian correction adds almost no extra cost (as the Hessian must be calculated anyway). The primary overhead remains the mega-batch calculation and dual gradients common to all SVRG methods.

### Loss & Training
The paper is a theoretical unification of optimization algorithms and introduces no new loss functions. Training follows the SVRG-style double loop: the outer loop computes a large (full/mega) batch gradient (and Hessian) using old parameters, while the inner loop performs corrected updates using mini-batches, periodically refreshing the outer batch. VSGD-PoCo only adds two Gaussian sampling steps compared to SVRG. IVON-PoCoMo stores additional $\mathbf{h}_{\rm out}$ and $\bm\sigma_{\rm out}$ (each $\Theta(d)$) and utilizes practical techniques like $\alpha$, warmup, debiasing, and momentum to stabilize training.

## Key Experimental Results

### Main Results

| Scenario | Comparison | Results |
|:---|:---|:---|
| Logistic Regression (MNIST / Covertype / CIFAR-10, convex) | VSGD vs VSGD-PoCo; IVON vs IVON-PoCo | Adding PoCo significantly accelerates both, approaching full-batch minima (L-BFGS level), with performance jumps after each mega-batch refresh. |
| GPT-2 (125M) Pre-training (OpenWebText, 50B tokens) | AdamW / IVON / IVON-PoCoMo | Validation perplexity: 18.4 / 18.0 / **17.4** (IVON-PoCoMo is lowest); perplexity drops immediately whenever correction is applied. |
| ImageNet ResNet-50 | SGD / IVON / AdamW / IVON-PoCo | IVON-PoCo is significantly better when measured by "optimization steps"; however, it is comparable to baselines when measured by "data samples seen / gradient computations". |

### Key Findings / Analysis of Limitations

| Dimension | Phenomenon | Explanation |
|:---|:---|:---|
| Convex Problems | PoCo brings strong acceleration | Consistent with Johnson & Zhang (2013); performance surges after the first outer loop. |
| GPT-2 | Superior perplexity but no real speedup | IVON-PoCoMo requires more gradient calculations; wall-clock time is not saved, equivalent to roughly tripling the batch size. |
| Deep Learning | Hard to beat baselines by data volume | Matches the conclusion of Defazio & Bottou (2019) that "variance reduction fails in deep learning" (observed here with larger models). |

### Key Findings
- In convex Logistic Regression, PoCo correction almost always pushes VSGD/IVON to the level of full-batch minima, with performance jumps precisely corresponding to "mega-batch refresh" moments.
- In GPT-2 pre-training, IVON-PoCoMo achieves a lower final validation perplexity (17.4 vs. 18.0/18.4), but this does not translate into training speedup—aligning with the known limitations of SVRG in deep learning.
- Natural gradients are the key to "going beyond SVRG": Newton-type Hessian correction appears automatically only within the natural-gradient-based PoCo framework and cannot be obtained by naively porting SVRG to Bayesian algorithms.

## Highlights & Insights
- The most significant "Aha!" moment is the reinterpretation of variance reduction as knowledge transfer: SVRG’s full-batch gradient = aggregation of old knowledge, and gradient correction = knowledge transfer between old and new gradients. This unifies SVRG with seemingly unrelated methods like continual learning and model merging under PoCo.
- The paradigm of "changing the posterior family yields a new algorithm" is highly productive: Isotropic Gaussian → SVRG, Full Gaussian → Newton-type, Diagonal Gaussian → Adam-type, Bernoulli → SVRG-style STE. A single framework produces a suite of variants.
- The Newton-type variant "forces" the emergence of Hessian correction (SVRH), something existing Newton-SVRGs missed. The key is natural gradients rather than naive migration—an insight that could guide the design of stronger second-order variance reduction methods.

## Limitations & Future Work
- The lack of "speedup" in deep learning is an honestly acknowledged flaw: while GPT-2 and ImageNet show step-by-step improvements, they are not superior to AdamW/IVON when calculated by data volume or wall-clock time, inheriting SVRG's long-standing issues in deep learning.
- Overhead from mega-batches and dual gradient computations is significant; the cost of $2nN+\lfloor nN/m\rfloor|\mathcal{M}|$ is quite realistic for large models. It is only more efficient than SVRG when $|\mathcal{M}|<m$.
- Full-covariance VON-PoCo is infeasible in high dimensions, forcing a fallback to diagonal approximations in practice, which compresses second-order information.
- The positioning of the paper is primarily theoretical groundwork. The authors explicitly state their hope that "variance reduction can truly become effective for deep learning in the future"—currently providing a perspective and algorithm template rather than a ready-to-use accelerator.

## Related Work & Insights
- **vs. Classic SVRG / SAGA / SARAH / SPIDER**: These perform variance reduction at the gradient level and are unrelated to Bayes; this paper absorbs SVRG as a special case of PoCo, providing a Bayesian interpretation and an extrapolation path.
- **vs. Newton-type SVRG (Derezinski 2025 / Sadiev 2024, etc.)**: Others only correct the gradient and leave the Hessian untouched; VON-PoCo corrects the Hessian via SVRH, being the first to do so.
- **vs. $\alpha$-SVRG (Yin et al., 2025)**: The downweighted update derived from mega-batches in this paper exactly recovers $\alpha$-SVRG under isotropic Gaussians, but they come from different motivations (mega-batches vs. early variance reduction scheduling); this paper provides a Bayesian origin.
- **vs. BLR / IVON (Khan & Rue 2023; Shen et al. 2024)**: This work stands on the shoulders of BLR, embedding PoCo into a double-loop to upgrade IVON to the variance-reduced IVON-PoCo.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to fundamentally link SVRG with Bayes and automatically derive Newton/Adam-type variants.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid validation on convex problems; covers GPT-2 and ImageNet for deep learning, though speedup conclusions are somewhat negative.
- Writing Quality: ⭐⭐⭐⭐ The derivation chain is clear, and specialization layers are well-defined. High theoretical density requires some background knowledge.
- Value: ⭐⭐⭐⭐ Provides a unified framework where "changing the posterior yields new algorithms," laying the foundation for second-order/Bayesian variance reduction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Beyond Short Steps in Frank-Wolfe Algorithms](../../ICLR2026/optimization/beyond_short_steps_in_frank-wolfe_algorithms.md)
- [\[ICLR 2026\] Learning to Recall with Transformers Beyond Orthogonal Embeddings](../../ICLR2026/optimization/learning_to_recall_with_transformers_beyond_orthogonal_embeddings.md)
- [\[NeurIPS 2025\] Stable Coresets via Posterior Sampling: Aligning Induced and Full Loss Landscapes](../../NeurIPS2025/optimization/stable_coresets_via_posterior_sampling_aligning_induced_and_full_loss_landscapes.md)
- [\[ICLR 2026\] Beyond Aggregation: Guiding Clients in Heterogeneous Federated Learning](../../ICLR2026/optimization/beyond_aggregation_guiding_clients_in_heterogeneous_federated_learning.md)
- [\[ICLR 2026\] Beyond the Heatmap: A Rigorous Evaluation of Component Impact in MCTS-Based TSP Solvers](../../ICLR2026/optimization/beyond_the_heatmap_a_rigorous_evaluation_of_component_impact_in_mcts-based_tsp_s.md)

</div>

<!-- RELATED:END -->
