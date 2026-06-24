---
title: >-
  [Paper Note] Singular Bayesian Neural Networks
description: >-
  [ICML 2026][AI Safety][Low-rank decomposition] This paper parameterizes the weight matrix directly as $W=AB^\top$ instead of applying mean-field distributions to $W$ itself, thereby inducing a **low-rank posterior singular with respect to the Lebesgue measure**. This reduces parameter complexity from $O(mn)$ to $O(r(m+n))$ and PAC-Bayes complexity from $\sqrt{mn}$ to $\sqrt{r(m+n)}$. Across MLP, LSTM, and Transformer architectures, it achieves OOD detection performance surpas…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "Low-rank decomposition"
  - "singular posterior"
  - "PAC-Bayes"
  - "OOD detection"
  - "mean-field variational inference"
date: 2026-05-08
content_hash: 7b3451a171e80572
---

# Singular Bayesian Neural Networks

**Conference**: ICML 2026  
**arXiv**: [2602.00387](https://arxiv.org/abs/2602.00387)  
**Code**: None  
**Area**: Bayesian Neural Networks / Variational Inference / Model Compression / Uncertainty Quantification  
**Keywords**: Low-rank decomposition, singular posterior, PAC-Bayes, OOD detection, mean-field variational inference

## TL;DR
This paper parameterizes the weight matrix directly as $W=AB^\top$ instead of applying mean-field distributions to $W$ itself, thereby inducing a **low-rank posterior singular with respect to the Lebesgue measure**. This reduces parameter complexity from $O(mn)$ to $O(r(m+n))$ and PAC-Bayes complexity from $\sqrt{mn}$ to $\sqrt{r(m+n)}$. Across MLP, LSTM, and Transformer architectures, it achieves OOD detection performance surpassing a 5-member Deep Ensemble while using $33\times$ fewer parameters.

## Background & Motivation

**Background**: Bayesian Neural Networks (BNNs) provide principled uncertainty quantification by maintaining weight distributions rather than point estimates, which is critical for high-stakes scenarios like healthcare and autonomous driving. The dominant approximation method is Mean-Field Variational Inference (MFVI), where each weight $w_{ij}$ is modeled as an independent Gaussian $\mathcal{N}(\mu_{ij}, \sigma_{ij}^2)$, requiring **twice** the parameters (mean + variance) of a deterministic model.

**Limitations of Prior Work**: (1) **Parameter explosion**—MFVI requires $O(mn)$ variational parameters, restricting BNNs to small models; (2) **Excessively strong independence assumptions**—fully factorized posteriors erase structural correlations between weights, damaging expressivity; (3) Cinquin et al. (2021) identified fundamental pathologies in weight-space inference for Transformers (e.g., difficult prior settings and complex mappings between weight and function spaces); (4) Existing low-rank work has flaws: post-hoc low-rank perturbations (Rank-1 Mult.) rely on pre-trained backbones and lose end-to-end uncertainty, low-rank covariance approximations still parameterize full-rank $W$ means, and LoRA-style Bayesian variants are limited to fine-tuning.

**Key Challenge**: Modern neural networks empirically exhibit **low intrinsic dimensionality** (Aghajanyan et al., 2021; rapid decay of weight matrix singular values), yet the full-rank and independent parameterization of BNNs structurally ignores this fact, wasting parameters and losing correlations.

**Goal**: (1) Directly parameterize the weight matrix as a low-rank product so the posterior naturally resides on a low-rank manifold; (2) Establish theoretical guarantees for PAC-Bayes tightness, reducing generalization complexity from $\sqrt{mn}$ to $\sqrt{r(m+n)}$; (3) Enable end-to-end training across MLP, LSTM, and Transformer architectures; (4) Learn uncertainty from scratch without relying on pre-trained backbones.

**Key Insight**: The authors observe that by applying mean-field inference to the **factors** $A$ and $B$ instead of $W$, the induced posterior $q_W$ is automatically supported on the rank-$r$ manifold $\mathcal{R}_r$, which has **zero volume** under the Lebesgue measure. Consequently, the result is not just "approximately low-rank" but a posterior that is **strictly singular** to the Lebesgue measure. This geometric property acts as a strong inductive bias: all $W_{ij}$ are coupled through shared factors $A_{ik}$ and $B_{jk}$, automatically generating structured correlations.

**Core Idea**: Place the Bayesian treatment on the low-rank factors rather than the weights, turning "singularity" into a quantifiable inductive bias. Use the Eckart-Young-Mirsky theorem to strictly characterize approximation errors via tail singular values $\sum_{i>r} \sigma_i^2$.

## Method

### Overall Architecture
This paper addresses the issues of parameter doubling and forced weight independence in BNNs under MFVI. Instead of applying mean-field distributions to the weight matrix $W \in \mathbb{R}^{m \times n}$, it decomposes $W$ into two low-rank factors $W = AB^\top$ ($A \in \mathbb{R}^{m \times r}, B \in \mathbb{R}^{n \times r}$) and places Bayesian uncertainty on $A$ and $B$. This reduces variational parameters from $O(mn)$ to $O(r(m+n))$, while the distribution of $W$ is pushed forward via $A$ and $B$ to reside on the rank-$r$ manifold. Specifically, heavy-tailed scale-mixture Gaussian priors $p_A(A) = \prod_j [\pi \mathcal{N}(0, \sigma_1^2) + (1-\pi)\mathcal{N}(0, \sigma_2^2)]$ are used for factors to promote sparsity. The variational posteriors $q_A, q_B$ remain mean-field Gaussians, using reparameterization $A = \mu_A + \log(1+\exp(\rho_A)) \circ \epsilon_A$ to ensure differentiability. The training objective decomposes the ELBO into a data-fitting term $\mathbb{E}_{q_A q_B}[\log p(\mathcal{D}|AB^\top)]$ and a regularization term $\beta(\text{KL}(q_A \| p_A) + \text{KL}(q_B \| p_B))$. This low-rank variational layer is a drop-in replacement across three architectures: MLPs factorize fully connected layers; Transformers factorize Q/K/V projections and FFNs, with embeddings using batch sparsity to sample only rows corresponding to current tokens; LSTMs factorize $W_{ih}$ and $W_{hh}$, sampling $A$ and $B$ once per batch and caching $W$ for reuse across time steps.

### Key Designs

**1. Induced Singular Posterior: Constraining Uncertainty to Low-Rank Manifolds**

Typical MFVI posteriors have positive density everywhere in the weight space, allowing weights to move freely, which wastes parameters and contradicts the "low intrinsic dimensionality" of modern networks. By performing inference on factors $(A, B)$ and pushing forward to $W = AB^\top$, the distribution is strictly pinned to a low-rank manifold. The paper formalizes this via three steps: **Lemma 3.2** proves $q_W(\mathcal{R}_r) = 1$, meaning the posterior mass lies entirely on the set of rank-$r$ matrices $\mathcal{R}_r$; **Lemma 3.3** proves that for $r < \min(m, n)$, $\mathcal{R}_r$ has zero Lebesgue measure; **Theorem 3.4** concludes that $q_W$ is singular to the Lebesgue measure. Unlike "positive density everywhere" in MFVI, this geometric constraint provides a powerful inductive bias. Wilson & Izmailov (2020) noted that Bayesian generalization depends on the **support** and **inductive bias** of the posterior: restricting support to a low-rank manifold acts as a strong prior belief with implicit regularization—any update to $W_{ij} = \sum_k A_{ik} B_{jk}$ requires modifying shared factors that affect entire rows/columns, preventing the model from over-fitting individual samples with local weights.

**2. Structured Weight Correlation: Recovering MFVI's Missing Dependencies**

The cost of full factorization in MFVI is the loss of structured correlations. Low-rank parameterization recovers these correlations even with a minimal parameter budget. Although $A$ and $B$ are independently mean-field, the elements of $W$ are not. **Lemma 3.5** provides $\text{Cov}(W_{ij}, W_{i'j'}) = \sum_k \text{Cov}(A_{ik}B_{jk}, A_{i'k}B_{j'k})$, showing that correlation arises whenever two weights share a latent factor $k$. The rank $r$ controls the complexity of this correlation structure; higher $r$ allows complex block-wise correlations while keeping parameters at $O(r(m+n))$. Figure 1 in the paper illustrates that full-rank BBB correlation matrices are mostly diagonal, whereas the low-rank version exhibits block structures. This filters out high-frequency noise inconsistent with the dominant low-rank structure and allows uncertainty to propagate along the "shared subspace," capturing epistemic uncertainty that MFVI misses.

**3. Theoretical Guarantees: Quantifying Gains via EYM and PAC-Bayes**

One might fear that low-rank constraints sacrifice expressivity. The paper uses several theorems to quantify the trade-off. **Theorem 3.6** (EYM Loss Bound) shows that under an $L$-Lipschitz loss, the difference between the optimal rank-$r$ truncation and the full-rank optimum is bounded by the tail singular values:

$$|\mathbb{E}\ell(W^*x,y) - \mathbb{E}\ell(W^*_r x, y)| \le LR \sqrt{\sum_{i>r} \sigma_i^2(W^*)}.$$

**Theorem 3.7** decomposes the error of the learned $W = AB^\top$ into an optimizable **learning error** $\|W - W^*_r\|_F$ and an unavoidable **rank bias** $\sigma_{>r}$. **Theorem 3.8** shows the PAC-Bayes complexity ratio is $\sqrt{r(m+n)/mn} \ll 1$, yielding significantly tighter generalization bounds when $r \ll \min(m, n)$. **Theorem 3.9** provides a non-vacuous generalization bound based on low-rank Gaussian complexity (Pinto et al., 2025). This framework makes selecting $r$ principled through singular value decay analysis or ablation studies.

### Loss & Training
All three components of the ELBO are estimated via Monte Carlo (as scale-mixture priors lack closed-form KL). The Adam optimizer is used, with $\sigma = \log(1+\exp(\rho))$ ensuring positive variance. $\beta$ acts as a KL temperature to adjust regularization strength. The rank $r_\ell$ for each layer can be tuned independently. During prediction, Monte Carlo averaging is performed over multiple weight samples.

## Key Experimental Results

### Main Results
The authors compared Deterministic, Deep Ensemble (5), Full-Rank BBB, Low-Rank (Ours), LR-SVD init, and Rank-1 Mult. across MIMIC-III (MLP), Beijing Air Quality (LSTM), and SST-2 (Transformer).

| Dataset (Arch) | Metric | Ours Low-Rank | Full-Rank BBB | Deep Ens. (5) | Params |
|--------------|------|--------------|--------------|---------------|------|
| MIMIC-III (MLP) | AUC-OOD↑ | **0.802** | 0.770 | 0.738 | 13.6k vs 44.8k / 112k |
| MIMIC-III (MLP) | AUPR-In↑ | **0.824** | 0.807 | 0.721 | — |
| Beijing AQ (LSTM) | PICP↑ | **0.790** | 0.788 | 0.310 | 47k vs 132k / 330k |
| Beijing AQ (LSTM) | AUROC-OOD↑ | 0.710 | 0.492 | **0.730** | — |
| SST-2 (Transformer) | Acc↑ | 0.806 | 0.752 | **0.825** | **1.5M** vs 19.8M / 49.6M |
| SST-2 (Transformer) | AUROC-OOD↑ | 0.640 (2nd) | 0.622 | **0.657** | — |
| SST-2 Training Time | min | **8.2** | 23.1 | 64.7 | — |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Low-Rank (random init, r=15) | Best OOD AUC=0.802 | Full model |
| LR-SVD init | OOD AUC=0.713 | Initializing with SVD degrades performance (locks rank too early) |
| Rank-1 Mult. (post-hoc) | OOD AUC=0.705 | Confirms end-to-end low-rank > post-hoc perturbations |
| Full-Rank BBB | OOD AUC=0.770 | Confirms contribution of singular posterior |
| $r$ Scan (PAC-Bayes Fig 3) | $r^* \approx 11$ threshold | PAC-Bayes bounds become vacuous beyond threshold |

### Key Findings
- **OOD Detection vs Likelihood Calibration Trade-off**: The low-rank model outperforms Deep Ensembles in OOD detection and uncertainty metrics (PICP/AUPR-Err) but slightly trails in-distribution NLL/ECE. Structured correlations appear more focused on epistemic uncertainty, while ensembles favor likelihood calibration.
- Weight matrices in modern architectures exhibit rapid singular value decay (especially embeddings), providing strong empirical support for low-rank parameterization.
- Full-Rank BBB performed worst on Transformers (0.752 acc), corroborating Cinquin et al.'s findings on weight-space inference pathologies in Transformers; low-rank constraints instead stabilized training.
- A single rank-$r$ BNN can match the predictive performance of a 5-member Deep Ensemble while saving $33\times$ parameters.

## Highlights & Insights
- **"Singularity" as a Feature**: Traditional Bayesian methods avoid singular posteriors; this work actively constructs them and quantifies their inductive bias—an elegant paradigm for geometricizing prior beliefs.
- **EYM Theorem + Pushforward**: Introducing classical matrix analysis tools into Bayesian Deep Learning allows for a clear loss upper bound to guide the selection of $r$.
- **Architecture-Agnostic Drop-in Replacement**: Low-rank variational layers can replace standard Keras/PyTorch layers, making BNNs significantly easier to deploy in industry.
- The "Bayesian on low-rank factors" approach is transferable to LoRA fine-tuning, diffusion model weights, and neural field parameters.

## Limitations & Future Work
- The rank $r$ still requires manual selection or ablation search; while singular value decay analysis helps, it typically requires a pre-trained backbone.
- Deep Ensembles still maintain an advantage in in-distribution likelihood (NLL=0.300 vs Ours 0.433 on MIMIC-III), suggesting structured correlations do not improve all metrics.
- Experimental scale is relatively small (max 4-layer BERT-mini); verification on billion-scale models is pending.
- Scale-mixture priors + Monte Carlo KL introduce additional sampling costs and hyperparameter tuning ($\pi, \sigma_1, \sigma_2$).
- Future directions: Combining with function-space methods (SNGP/Laplace), extending to SSM/Mamba, and applying to weight uncertainty in generative models for "safe generation."

## Related Work & Insights
- **vs Rank-1 Multiplicative (Dusenberry 2020a)**: Their approach adds rank-1 perturbations to a deterministic backbone post-hoc; Ours learns low-rank end-to-end and outperforms in OOD detection.
- **vs Low-Rank Covariance (Tomczak 2020)**: They use low-rank + diagonal covariance, but keep full-rank means; Ours uses low-rank for the weights themselves.
- **vs LoRA Bayesian (Yang 2024)**: LoRA requires a pre-trained backbone; Ours trains from scratch.
- **vs Deep Ensemble**: Ensembles are "poor man's Bayes" with $5\times$ parameters; Ours is a single model with $5$–$33\times$ parameter savings and better OOD detection.
- **vs Watanabe’s Singular Learning Theory**: The "singularity" here refers to the induced posterior being singular w.r.t. the Lebesgue measure (geometric), which differs from Watanabe’s concept of asymptotic model singularity.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The geometric perspective of "singular posteriors" and the EYM loss bound framework are original paradigms for BDL.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers MLP/LSTM/Transformer with multiple OOD metrics; lacks large-scale LLM validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical derivations with clear structure (Definition-Lemma-Theorem).
- Value: ⭐⭐⭐⭐ Enables BNN scalability to modern architectures with easy drop-in implementation; however, in-distribution calibration still lags behind Ensembles.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Active Continual Learning with Metaplastic Binary Bayesian Neural Networks](active_continual_learning_with_metaplastic_binary_bayesian_neural_networks.md)
- [\[ICLR 2026\] Robustify Spiking Neural Networks via Dominant Singular Deflation under Heterogeneous Training Vulnerability](../../ICLR2026/ai_safety/robustify_spiking_neural_networks_via_dominant_singular_deflation_under_heteroge.md)
- [\[ICML 2026\] Frequency Matching in Spiking Neural Networks for mmWave Sensing](frequency_matching_in_spiking_neural_networks_for_mmwave_sensing.md)
- [\[ICML 2026\] How Does Bayesian Sampling Help Membership Inference Attacks?](how_does_bayesian_sampling_help_membership_inference_attacks.md)
- [\[ICLR 2026\] Robust Spiking Neural Networks Against Adversarial Attacks](../../ICLR2026/ai_safety/robust_spiking_neural_networks_against_adversarial_attacks.md)

</div>

<!-- RELATED:END -->
