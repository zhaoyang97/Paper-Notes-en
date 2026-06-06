---
title: >-
  [Paper Note] Singular Bayesian Neural Networks
description: >-
  [ICML 2026][Low-Rank Decomposition] This work directly parameterizes the weight matrix as $W=AB^\top$ instead of applying a mean-field distribution to $W$ itself…
tags:
  - "ICML 2026"
  - "Low-Rank Decomposition"
  - "Singular Posterior"
  - "PAC-Bayes"
  - "OOD Detection"
  - "Mean-Field Variational Inference"
date: 2026-05-08
content_hash: 56e250b3b8653c4b
---

# Singular Bayesian Neural Networks

**Conference**: ICML 2026  
**arXiv**: [2602.00387](https://arxiv.org/abs/2602.00387)  
**Code**: None  
**Area**: Bayesian Neural Networks / Variational Inference / Model Compression / Uncertainty Quantification  
**Keywords**: Low-Rank Decomposition, Singular Posterior, PAC-Bayes, OOD Detection, Mean-Field Variational Inference

## TL;DR
This work directly parameterizes the weight matrix as $W=AB^\top$ instead of applying a mean-field distribution to $W$ itself, thereby inducing a **low-rank posterior that is singular with respect to the Lebesgue measure**. The number of parameters is reduced from $O(mn)$ to $O(r(m+n))$, and the PAC-Bayes complexity is tightened from $\sqrt{mn}$ to $\sqrt{r(m+n)}$. On MLP/LSTM/Transformer architectures, the method achieves OOD detection performance surpassing 5-member Deep Ensembles with $33\times$ fewer parameters.

## Background & Motivation

**Background**: Bayesian Neural Networks (BNNs) provide principled uncertainty quantification by maintaining distributions over weights rather than point estimates, which is crucial for high-stakes applications such as healthcare and autonomous driving. The mainstream approximation is Mean-Field Variational Inference (MFVI): each weight $w_{ij}$ is modeled as an independent Gaussian $\mathcal{N}(\mu_{ij}, \sigma_{ij}^2)$, requiring **twice** as many parameters as deterministic models (mean + variance).

**Limitations of Prior Work**: (1) **Parameter explosion**—MFVI requires $O(mn)$ variational parameters, restricting BNNs to small models; (2) **Overly strong independence assumption**—fully factorized posteriors erase structural correlations among weights, harming expressiveness; (3) Cinquin et al. (2021) further point out fundamental pathologies in weight-space inference for Transformers (difficult prior specification, challenging mapping from weight to function space); (4) Existing low-rank approaches fall into three camps, each with flaws: post-hoc low-rank perturbations (Rank-1 Mult.) rely on pretrained backbones and lose end-to-end uncertainty; low-rank covariance approximations still parameterize full-rank $W$ means; LoRA-style Bayesian variants can only fine-tune pretrained models.

**Key Challenge**: Modern neural networks empirically exhibit **low intrinsic dimensionality** (Aghajanyan et al. 2021; rapid singular value decay in weight matrices), yet BNNs' full-rank + independent parameterization structurally ignores this, wasting parameters and losing correlations.

**Goal**: (1) Directly parameterize weight matrices as low-rank products so the posterior naturally lies on a low-rank manifold; (2) Establish PAC-Bayes-based theoretical guarantees, tightening generalization complexity from $\sqrt{mn}$ to $\sqrt{r(m+n)}$; (3) Enable end-to-end training, covering MLP / LSTM / Transformer architectures; (4) Avoid reliance on pretrained backbones, learning uncertainty from scratch.

**Key Insight**: The authors observe that applying mean-field to the **factors** $A, B$ rather than $W$ induces a posterior $q_W$ that is automatically supported on the rank-$r$ manifold $\mathcal{R}_r$—which has **zero volume** under the Lebesgue measure. In other words, the result is not "approximately low-rank," but a posterior **strictly singular** with respect to Lebesgue measure. This geometric property itself is a strong inductive bias: all $W_{ij}$ are coupled via shared factors $A_{ik}, B_{jk}$, automatically inducing structured correlations.

**Core Idea**: Place the Bayesian distribution on low-rank factors rather than weights, turning "singularity" into a quantifiable inductive bias, and use the Eckart-Young-Mirsky theorem to rigorously characterize approximation error via the tail singular values $\sum_{i>r} \sigma_i^2$.

## Method

### Overall Architecture
Each weight matrix $W \in \mathbb{R}^{m \times n}$ is parameterized as $W = AB^\top$, where $A \in \mathbb{R}^{m \times r}, B \in \mathbb{R}^{n \times r}$. A scale-mixture Gaussian prior is placed on the factors: $p_A(A) = \prod_j [\pi \mathcal{N}(0, \sigma_1^2) + (1-\pi)\mathcal{N}(0, \sigma_2^2)]$ (heavy tails promote sparsity). The variational posteriors $q_A, q_B$ are mean-field Gaussians, with reparameterization $A = \mu_A + \log(1+\exp(\rho_A)) \circ \epsilon_A$ for differentiable sampling. The ELBO decomposes into a data fit term $\mathbb{E}_{q_A q_B}[\log p(\mathcal{D}|AB^\top)]$ and a regularization term $\beta(\text{KL}(q_A \| p_A) + \text{KL}(q_B \| p_B))$. Instantiations for three architectures: MLP directly factorizes fully connected layers; Transformer factorizes Q/K/V projections and FFN, with embeddings using batch sparsity to sample only rows corresponding to current tokens; LSTM factorizes $W_{ih}, W_{hh}$, sampling $A, B$ once per batch and caching $W$ across timesteps.

### Key Designs

1. **Inducing Singular Posterior and Geometric Inductive Bias**:

    - Function: Places Bayesian uncertainty directly on the low-rank manifold, avoiding MFVI's full-space diffusion.
    - Mechanism: Variational inference is performed on factors $(A, B)$, and the distribution of $W = AB^\top$ is obtained via pushforward. **Lemma 3.2** proves $q_W(\mathcal{R}_r) = 1$ (supported on the set of rank-$r$ matrices); **Lemma 3.3** shows that for $r < \min(m, n)$, $\mathcal{R}_r$ has zero Lebesgue measure; **Theorem 3.4** directly concludes $q_W$ is singular with respect to Lebesgue measure. This means $q_W$ has no Lebesgue density—fundamentally contrasting with MFVI's "everywhere positive density."
    - Design Motivation: Wilson & Izmailov (2020) argue Bayesian generalization depends on posterior **support** and **inductive bias**. MFVI favors "independently tunable weights," while this work favors "weights coupled via shared factors," aligning better with the low-rank nature of modern deep networks and providing implicit regularization—updating $W_{ij} = \sum_k A_{ik} B_{jk}$ must modify shared factors affecting entire rows/columns, preventing local memorization.

2. **Structured Weight Correlations (Lemma 3.5)**:

    - Function: Captures global correlations among weights under a low parameter budget, compensating for MFVI's independence loss.
    - Mechanism: Although $A, B$ are mean-field, elements of $W$ are **not independent**—$\text{Cov}(W_{ij}, W_{i'j'}) = \sum_k \text{Cov}(A_{ik}B_{jk}, A_{i'k}B_{j'k})$; any two weights sharing latent factor $k$ are correlated. Rank $r$ controls the richness of correlation structure: higher rank allows more complex block correlations, yet parameters remain $O(r(m+n))$. Figure 1 in the paper shows full-rank BBB yields diagonal correlations, while low-rank yields block structures.
    - Design Motivation: Filters out high-frequency noise inconsistent with dominant low-rank structure, capturing "shared subspace" uncertainty propagation invisible to MFVI.

3. **Theoretical Guarantees: EYM Loss Decomposition + PAC-Bayes Tightening**:

    - Function: Formalizes "low-rank ≠ degenerate" and quantifies complexity gains.
    - Mechanism: **Theorem 3.6** (EYM loss bound): Under $L$-Lipschitz loss, the loss gap between optimal rank-$r$ truncated SVD and full-rank optimum is controlled by tail singular values $|\mathbb{E}\ell(W^*x,y) - \mathbb{E}\ell(W^*_r x, y)| \le LR \sqrt{\sum_{i>r} \sigma_i^2(W^*)}$. **Theorem 3.7** decomposes the error between learned $W = AB^\top$ and full-rank optimum into **learning error** $\|W - W^*_r\|_F$ + **rank bias** $\sigma_{>r}$. **Theorem 3.8** gives PAC-Bayes complexity ratio $\sqrt{r(m+n)/mn} \ll 1$; when $r \ll \min(m, n)$, the bound is significantly tightened. **Theorem 3.9** uses Pinto et al. (2025)'s low-rank Gaussian complexity to provide a complementary non-vacuous generalization bound.
    - Design Motivation: Provides theoretical guidance for choosing $r$—via singular value decay analysis or ablation, and enables prediction of loss upper bounds.

### Loss & Training
All three ELBO terms are estimated via Monte Carlo (scale-mixture prior has no closed-form KL); Adam optimizer is used; $\sigma = \log(1+\exp(\rho))$ ensures positivity; $\beta$ is a KL temperature. Each layer's rank $r_\ell$ is independently tunable. At prediction, Monte Carlo averages over multiple weight samples.

## Key Experimental Results

### Main Results
The authors compare Deterministic / Deep Ensemble (5) / Full-Rank BBB / Low-Rank (Ours) / LR-SVD init / Rank-1 Mult. on three datasets: MIMIC-III (ICU mortality, MLP), Beijing Air Quality (PM2.5 prediction, LSTM), SST-2 (sentiment classification, Transformer).

| Dataset (Arch) | Metric | Ours Low-Rank | Full-Rank BBB | Deep Ens. (5) | Params |
|---------------|--------|---------------|--------------|---------------|--------|
| MIMIC-III (MLP) | AUC-OOD↑ | **0.802** | 0.770 | 0.738 | 13.6k vs 44.8k / 112k |
| MIMIC-III (MLP) | AUPR-In↑ | **0.824** | 0.807 | 0.721 | — |
| Beijing AQ (LSTM) | PICP↑ | **0.790** | 0.788 | 0.310 | 47k vs 132k / 330k |
| Beijing AQ (LSTM) | AUROC-OOD↑ | 0.710 | 0.492 | **0.730** | — |
| SST-2 (Transformer) | Acc↑ | 0.806 | 0.752 | **0.825** | **1.5M** vs 19.8M / 49.6M |
| SST-2 (Transformer) | AUROC-OOD↑ | 0.640 (2nd) | 0.622 | **0.657** | — |
| SST-2 Training Time | min | **8.2** | 23.1 | 64.7 | — |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| Low-Rank (random init, r=15) | Best OOD AUC=0.802 | Full model |
| LR-SVD init | OOD AUC=0.713 | SVD initialization degrades performance (prematurely locks rank) |
| Rank-1 Mult. (post-hoc) | OOD AUC=0.705 | End-to-end low-rank > post-hoc low-rank perturbation |
| Full-Rank BBB | OOD AUC=0.770 | Validates singular posterior's contribution |
| Different $r$ sweep (see PAC-Bayes Fig 3) | $r^* \approx 11$ threshold | Exceeding threshold makes PAC-Bayes bound vacuous |

### Key Findings
- **OOD detection vs likelihood calibration trade-off**: Low-rank models outperform Deep Ensembles on OOD detection and uncertainty metrics (PICP/AUPR-Err), but are slightly inferior to Ensembles on in-distribution NLL/ECE—structured correlations focus more on epistemic uncertainty, while ensembles focus on likelihood calibration.
- Modern architectures' weight matrices indeed exhibit rapid singular value decay (especially embeddings), strongly supporting low-rank parameterization.
- On Transformers, Full-Rank BBB performs worst (0.752 acc), confirming Cinquin et al.'s findings on pathologies in Transformer weight-space inference; low-rank constraints stabilize training.
- A single rank-$r$ BNN can match the predictive performance of a 5-member Deep Ensemble, saving $33\times$ parameters.

## Highlights & Insights
- **"Singularity" is a feature, not a bug**: Traditional Bayesian methods avoid singular posteriors; this work actively constructs and quantifies their inductive bias—an elegant paradigm for geometrizing "prior beliefs."
- **EYM theorem + Pushforward**: Brings classic matrix analysis tools into Bayesian deep learning complexity analysis, providing explicit loss upper bounds for "choosing $r$"—highly practical.
- **Architecture-agnostic drop-in replacement**: The low-rank variational layer can directly replace standard Keras layers, making it highly deployable in engineering—important for BNN adoption in industry.
- The "Bayesian on low-rank factors" idea can transfer to LoRA fine-tuning (partially mentioned by the authors), diffusion model weight uncertainty, neural field parameters, etc.

## Limitations & Future Work
- Rank $r$ still requires manual selection or ablation search; while singular value decay analysis helps, SVD requires a pretrained backbone, so end-to-end training must rely on ablation.
- Deep Ensembles still have an advantage in in-distribution likelihood (NLL=0.300 vs Ours 0.433 on MIMIC-III), indicating structured correlations do not universally outperform.
- Experimental scale remains small (largest is 4-layer BERT-mini), not yet validated on true billion-scale models; the paper acknowledges this as "foundational" work.
- Scale-mixture prior + Monte Carlo KL introduces extra sampling cost, and hyperparameters $\pi, \sigma_1, \sigma_2$ require tuning.
- Future directions: combine with SNGP/Laplace function-space methods, extend to SSM/Mamba architectures, integrate with generative model weight uncertainty for "safe generation."

## Related Work & Insights
- **vs Rank-1 Multiplicative (Dusenberry 2020a)**: Adds rank-1 multiplicative perturbations to deterministic backbones post-hoc; this work learns low-rank end-to-end from initialization, with clear OOD advantages.
- **vs Low-Rank Covariance (Tomczak 2020)**: Applies low-rank + diagonal to covariance, but weight means remain full-rank; this work directly applies low-rank to $W$ itself.
- **vs LoRA Bayesian (Yang 2024)**: LoRA requires fine-tuning pretrained backbones; this work trains from scratch.
- **vs Deep Ensemble**: Ensembles are "poor man's Bayesian" via multiple point estimate samples, with $5\times$ parameters; this work uses a single model, saving $5$–$33\times$ parameters, with better OOD detection but slightly worse in-distribution likelihood.
- **vs SNGP / Linearized Laplace**: These operate in function space or only on the last layer; this work performs end-to-end weight-space inference, offering complementarity.
- **vs Watanabe's Singular Learning Theory**: Here, "singular" refers to posteriors singular with respect to Lebesgue measure (geometric), distinct from Watanabe's notion of asymptotic model singularity.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The geometric perspective of "singular posterior" and EYM loss bound framework are truly original Bayesian deep learning paradigms
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers MLP/LSTM/Transformer, multiple OOD metrics; lacks large-scale LLM validation
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical derivations are rigorous and self-consistent, with clear definition–lemma–theorem structure
- Value: ⭐⭐⭐⭐ Makes BNNs truly scalable to modern architectures, easily deployable as drop-in layers; but still lags Ensembles in in-distribution calibration

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] On the Epistemic Uncertainty of Overparametrized Neural Networks](on_the_epistemic_uncertainty_of_overparametrized_neural_networks.md)
- [\[ICML 2026\] The Implicit Bias of Adam and Muon on Smooth Homogeneous Neural Networks](the_implicit_bias_of_adam_and_muon_on_smooth_homogeneous_neural_networks.md)
- [\[ICML 2026\] Bullet Trains: Parallelizing Training of Temporally Precise Spiking Neural Networks](bullet_trains_parallelizing_training_of_temporally_precise_spiking_neural_networ.md)
- [\[ICML 2026\] How the Optimizer Shapes Learned Solutions in Equivariant Neural Networks](how_the_optimizer_shapes_learned_solutions_in_equivariant_neural_networks.md)
- [\[ICML 2026\] How Does Bayesian Sampling Help Membership Inference Attacks?](how_does_bayesian_sampling_help_membership_inference_attacks.md)

</div>

<!-- RELATED:END -->
