---
title: >-
  [Paper Note] Statistical and Structural Identifiability in Representation Learning
description: >-
  [ICLR 2026][learning_theory][ICA] This paper decomposes "representation stability" into two independent concepts: statistical identifiability (consistent representations across multiple retraining sessions) and structural identifiability (representations aligned to true generative factors). It proposes a "near-identifiability" definition with error tol
tags:
  - ICLR 2026
  - learning_theory
  - ICA
date: 2026-05-08
content_hash: e4213a5acf6ffd3b
---
# Statistical and Structural Identifiability in Representation Learning

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=Wa3cfE3Iay](https://openreview.net/forum?id=Wa3cfE3Iay)  
**Area**: Representation Learning Theory / Identifiability  
**Keywords**: Identifiability, Representation Learning, ICA, Disentanglement, Near-Isometry

## TL;DR
This paper decomposes "representation stability" into two independent concepts: statistical identifiability (consistent representations across multiple retraining sessions) and structural identifiability (representations aligned to true generative factors). It proposes a "near-identifiability" definition with error tolerance $\epsilon$ and proves statistical $\epsilon$-near-identifiability for a class of models with non-linear decoders (e.g., MAE, supervised learners, GPT intermediate layers). It indicates that using linear ICA to post-process the latent space eliminates residual linear uncertainty, yielding a minimalist "recipe" for disentanglement. This approach achieves SOTA on synthetic disentanglement benchmarks using a vanilla autoencoder and separates biological variation from batch effects in biological foundation models.

## Background & Motivation

**Background**: Despite differences in modalities, tasks, and data, various self-supervised models exhibit surprising stability in their internal representations—different models seem to converge to a shared set of world representations. The classic tool for studying this phenomenon is "identifiability": in likelihood inference, it refers to data being sufficient to uniquely determine model parameters; in neural networks, it is relaxed to "infinite data being sufficient to determine the representation of a trained model up to some equivalence class (e.g., linear transformation)."

**Limitations of Prior Work**: Existing identifiability results have two types of limitations. First, they either impose strong assumptions on the data generation process (e.g., contrastive learning requiring the augmentation distribution to be isotropic in the latent space, which is unverifiable without ground-truth factors) or assume a linear relationship between representations and loss (Roeder et al. can only handle the penultimate layer where representations are linearly mapped to the loss). Second, the literature generally **fails to distinguish** between two essentially different types of "stability": consistency of representations across multiple retraining runs and alignment of representations to true generative factors are distinct properties but are often conflated.

**Key Challenge**: In modern models (e.g., MAE), the truly useful components are often **intermediate layer** representations, which are mapped to the loss by **non-linear decoders/heads**. These fall outside the coverage of existing theories. Simultaneously, "perfect pointwise identifiability" is unrealistic for modern large models; theory requires a relaxed version that tolerates small errors.

**Goal**: (1) Provide clear, model-agnostic definitions of statistical and structural near-identifiability; (2) prove that intermediate representations of a broad class of models with non-linear decoders are statistically $\epsilon$-near-identifiable; (3) provide the minimal assumptions and practical algorithms to bridge statistical identifiability and structural identifiability (i.e., disentanglement).

**Key Insight**: The authors introduce a "slack variable" $\epsilon$, relaxing identifiability from "exact equality" to "differing by a simple transformation group $H$ plus a small perturbation $\epsilon$." The magnitude of $\epsilon$ is directly linked to the **local bi-Lipschitz constant** of the decoder—a quantity that can be controlled by common regularization techniques (tending toward "dynamic isometry").

**Core Idea**: Replace "strong assumptions on the data generation process" with a **mild model-side assumption** of "near-isometric/bi-Lipschitz decoders." This proves that intermediate representations are near-identifiable under rigid transformations. Then, linear ICA is used to reduce the remaining linear uncertainty to signed permutations, resulting in a minimalist disentanglement recipe: "autoencoder + latent space ICA."

## Method

### Overall Architecture

This is a theoretical paper that connects the phenomenon of "why representations are stable" through three progressive theorems, validated by four sets of experiments. The logic is: first, redefine "identifiability" (introducing $\epsilon$ relaxation and transformation group $H$), distinguishing between **statistical** and **structural** levels; second, prove the statistical level (Theorem 1: intermediate representations are near-identifiable under rigid transformations $H_{\text{rigid}}$, with $\epsilon$ controlled by the decoder's bi-Lipschitz constant); third, use ICA to tighten linear uncertainty (Theorem 2: reducing from $H_{\text{linear}}$ to $H_{\text{rigid}}$ via whitening, then to signed permutations $H_\sigma$ via ICA); finally, upgrade statistical identifiability to structural identifiability when the data generation process also satisfies bi-Lipschitz conditions (Theorem 3: autoencoder + ICA can approximately recover true latent factors $g^{-1}$).

Let the data distribution be $P(x)$, model $M=\{L_\theta:\theta\in\Theta\}$, and $F:\theta\mapsto f_\theta$ map parameters to the representation function $f_\theta:X\to\mathbb{R}^D$, where $S\subset\Theta$ is the set of minima for the expected loss $\mathbb{E}_{x\sim P}[L_\theta(x)]$. Three core transformation groups are used: $H_{\text{linear}}$ (invertible linear transformations), $H_{\text{rigid}}$ (rigid transformations consisting of rotations/reflections/translations, where the primary uncertainty is rotation in $SO(D)$), and $H_\sigma$ (signed permutations, as latent variables lack natural ordering and signs, making this layer generally impossible and unnecessary to eliminate).

### Key Designs

**1. Definitions of $\epsilon$-Near-Identifiability for Statistical and Structural Levels: Splitting "Stability" into "Consistency" and "Correctness"**

Prior works treat representation stability as a single property. This paper points out it consists of two levels. Statistical identifiability characterizes "consistency": representations obtained by optimizing the same model multiple times differ only by a simple transformation. Formally (Definition 1), if for any $\theta,\theta' \in S$ there exists $h\in H$ such that $\lVert f_\theta - h\circ f_{\theta'}\rVert \le \epsilon$, the model is said to be statistically $\epsilon$-near-identifiable under group $H$; the norm is $L^\infty$ (essential supremum w.r.t. $P$). When $\epsilon=0$, it degrades to exact identifiability, generalizing classic mathematical statistics. Structural identifiability characterizes "correctness": it assumes the existence of an unobservable generative factor $u$ ($P(u)$, $P(x\mid u)$, and $u(x)=\arg\sup_u P(x\mid u)$ is well-defined almost everywhere). If for all $\theta\in S$ there exists $\lVert h\circ f_\theta - u\rVert\le\epsilon$, the model is said to $\epsilon$-near-identify the structure $u$ (Definition 2). Intuitively, statistical identifiability means "same every time," while structural identifiability means "right every time," the latter being strictly stronger—disentanglement is a special case where components of $P(u)$ are independent. The significance of introducing $\epsilon$ is that while modern large models cannot be exactly identifiable pointwise, "near-identifiability" is achievable and measurable.

**2. Theorem 1—Rigid Near-Identifiability of Intermediate Representations under Non-linear Decoders: Extending Identifiability from the Final Layer to Any Intermediate Layer**

Previous results, such as Roeder et al., only cover the penultimate layer where representations are **linearly** mapped to the loss (negative log-likelihood of exponential families $L_\theta(x,y)=-\eta_\theta(x)^\top t_\theta(y)+A_\theta(x)$). However, many models prioritize earlier representations mapped by **non-linear** decoders/heads. This paper decomposes the end-to-end network as $H:\theta\mapsto g_\theta\circ f_\theta$ (encoder $f_\theta$ followed by decoder $g_\theta$). Assuming the overall output $g_\theta\circ f_\theta$ is statistically identifiable (e.g., the network learns the optimal conditional mean under MSE loss), the degree of near-identifiability of the encoder representation $f_\theta$ is determined by the **local bi-Lipschitz constant** of the decoder $g_\theta$. Specifically, if $1+L$ is a local bi-Lipschitz upper bound for $g_\theta$, then $(P,\Theta,L_\theta,F)$ is statistically $\epsilon$-near-identifiable under $H_{\text{rigid}}$, where

$$\epsilon = c_D\sqrt{2L+L^2}\,\Delta$$

where $c_D$ and $\Delta$ are model-independent constants. The intuition is: bi-Lipschitz constraints control how much the decoder "distorts distances"; when small changes in latents cause only small changes in output, the constant is near 1 and $\epsilon$ is small. This is the most general quantification of intermediate layer identifiability known to date, covering MAE, next-token predictors, and supervised learners. It shifts strong assumptions from the "data generation process" to the "model class"—where bi-Lipschitz/dynamic isometry is exactly what techniques like spectral normalization, ReZero, and zero-initialized residuals aim for in practice (Jacobian singular values concentrated around 1).

**3. Theorem 2—ICA Eliminating Residual Linear Uncertainty: Tightening Rigid Ambiguity to Signed Permutations**

Theorem 1 leaves rigid (or linear) uncertainty. To truly utilize representations, this ambiguity must be removed. This paper does not propose a new ICA theory but proves that the relaxation of $\epsilon$-near-identifiability does not "interfere" with downstream ICA. If $(P,\Theta,L_\theta,F)$ is statistically $\epsilon$-near-identifiable under $H_{\text{linear}}$, applying whitening followed by contrast-function-based ICA yields a new model $(P,\Theta',L'_\theta,F')$ that is statistically $\epsilon'$-near-identifiable under signed permutations $H_\sigma$, where

$$\epsilon' = K\epsilon + K'\epsilon^2$$

where $K,K'$ are constants independent of $\epsilon$, determined by the representation covariance spectrum and ICA contrast functions. The mechanism tightens uncertainty in two steps: whitening reduces linear uncertainty to rigid, and ICA (with sufficient convergence) reduces rigid to signed permutations, preserving "nearness" via new constants. This step is crucial for bridging theory and practice—it shows that as long as representations are linearly identifiable, a **purely unsupervised** ICA post-processing can approximate "standardized" latent coordinates.

**4. Theorem 3—From Statistical to Structural Identifiability: AE + ICA Recovering True Factors under bi-Lipschitz Data Generation**

To move from "same every time" to "right every time," assumptions must be placed on the data generation process. This paper assumes the true factor distribution $P(u)$ is a multivariate distribution with independent non-Gaussian components, and data $P(x)$ is generated via a push-forward by a $(1+\delta)$-bi-Lipschitz smooth diffeomorphism $g$. For an encoder-decoder model achieving perfect reconstruction (i.e., $g_\theta\circ f_\theta$ structurally identifies the identity function in the limit), $f_\theta$ $\epsilon$-near-identifies the structure $g^{-1}$ under $H_{\text{rigid}}$; adding whitening + ICA leads to $\epsilon'$-near-identification under $H_\sigma$. The key is that the assumption on the data generation process (bi-Lipschitz) is **homologous** to the assumption on the model, making structural identifiability a direct corollary of Theorem 2. The cost is that Theorem 3 requires **perfect reconstruction** (autoencoder type), whereas Theorem 2 only requires identifiable outputs. The authors also argue using continuously relaxed dSprites-style images: translating a white square is locally isometric (1-bi-Lipschitz), where $\lVert f'(p)\rVert_2=2r$ is constant, preserving geometric distance $\propto|p_1-p_0|$—demonstrating that isometric approximation of real image manifolds is plausible, thus reducing disentanglement to the minimalist "vanilla autoencoder + latent linear ICA" recipe.

### Loss & Training

This paper introduces no new loss functions. Experiments use standard objectives: reconstruction loss for autoencoders and original losses for contrastive/masked self-supervised models. The only factors deliberately adjusted are those controlling the decoder's bi-Lipschitz constant—such as the LeakyReLU leak parameter $\alpha$ (the bi-Lipschitz upper bound for a 3-layer decoder is roughly $1/\alpha^K$ with $K=3$; $\alpha=1$ is linear, $\alpha=0$ is ReLU) and weight decay (known to sufficiently regularize the Lipschitz constant of decoders in vanilla autoencoders). ICA is a **post-training** latent space process (whitening + contrast function ICA) and does not participate in model training.

## Key Experimental Results

### Main Results

Four sets of experiments correspond to four claims. Table 1 verifies statistical near-identifiability and ICA's disambiguation capability: between model pairs with identical architectures/loss/data but different initializations, it reports mean $\ell_2$ errors normalized by latent space diameter under various optimal transformations. ICA efficiency is defined as the percentage reduction in $\ell_2$ error relative to rigid transformation.

| Model Pair | Permutation | Supervised Rigid | Supervised Linear | ICA (% eff.) |
|------------|-------------|------------------|-------------------|--------------|
| Pythia-160M-0 → Pythia-160M-1 | 0.219 | 0.150 | 0.131 | 0.202 (25%) |
| MAE-timm → MAE-original | 0.197 | 0.109 | 0.036 | 0.145 (59%) |
| CheXpert-small → CheXpert-base | 0.218 | 0.104 | 0.048 | 0.175 (38%) |
| ResNet-18-fc-1 → ResNet-18-fc-2 | 0.382 | 0.206 | 0.175 | 0.312 (40%) |

GPT-like models (Pythia) show excellent linear alignment (consistent with Roeder's theory), while MAE shows rigid alignment as predicted by this paper's theory (including across model sizes); in all cases, unsupervised ICA eliminates substantial linear uncertainty, with ICA efficiency on MAE reaching nearly 60% of the optimal supervised rigid transformation.

Table 2 verifies structural identifiability (disentanglement): on four synthetic datasets, vanilla autoencoder + latent ICA is compared against specialized disentanglement networks using InfoMEC metrics (Modularity InfoM, Explicitness InfoE, Compactness InfoC; the first two are primary).

| Model | aggregated (InfoM InfoE InfoC) | Shapes3D | MPI3D | Falcor3D | Isaac3D |
|-------|-------------------------------|----------|-------|----------|---------|
| AE | (0.39 0.76 0.25) | (0.34 0.99 0.16) | (0.42 0.40 0.31) | (0.37 0.83 0.20) | (0.41 0.80 0.34) |
| β-VAE* | (0.59 0.81 0.55) | (0.59 0.99 0.49) | (0.45 0.71 0.51) | (0.71 0.73 0.70) | (0.60 0.80 0.51) |
| β-TCVAE* | (0.58 0.72 0.59) | (0.61 0.82 0.62) | (0.51 0.60 0.57) | (0.66 0.74 0.71) | (0.54 0.70 0.46) |
| BioAE* | (0.54 0.75 0.36) | (0.56 0.98 0.44) | (0.45 0.66 0.36) | (0.54 0.73 0.31) | (0.63 0.65 0.33) |
| AE + ICA (ours) | (0.65 0.83 0.40) | (0.79 0.99 0.52) | (0.44 0.66 0.31) | (0.71 0.83 0.33) | (0.64 0.82 0.43) |

AE + ICA, tuned only via weight decay, **outperforms all specialized models** on average across aggregated metrics (results with * are from Hsu et al. 2023, not reproduced here).

### Ablation Study

Table 3/4 shows de-confounding results on the real-world biological foundation model OpenPhenom (a large MAE): perturbations are classified (control vs perturbed) using original embeddings (Base), whitening (PCA), whitening + ICA (PCA + ICA), and whitening + random rotation (PCA + Rand), evaluated across batches.

| Configuration | Mean AUROC (↑) | Hoyer Sparsity (↑) | Bio-variation Concentration (Top 25%) (↑) | Note |
|---------------|---------------|-----------------|-----------------------------------|------|
| Base | ~0.66–0.80 | Lower | 0.163 | Raw embeddings |
| PCA | Gain | Gain | 0.332 | Whitening only |
| PCA + ICA | **Highest** | **Highest** | **0.386** | Whitening + ICA rotation |
| PCA + Rand | Intermediate | Medium | 0.287 | Whitening + random rotation |

For example, on the EIF3H gene, AUROC improved from Base 0.682 → PCA 0.724 → PCA + ICA 0.749; PCA + Rand at 0.725 was significantly worse than ICA, showing gains stem from ICA's specific rotation rather than just whitening or arbitrary rotation.

### Key Findings
- **Bi-Lipschitz constants predict identifiability**: Warmup experiments (MNIST, adjusting LeakyReLU $\alpha$) show empirical $\sqrt{L+L^2}$ terms predict $\ell_2$ error of rigid alignment, matching Theorem 1—a rare case of identifiability theory being directly verifiable by experiment.
- **ICA gains come from "the right rotation"**: The PCA + Rand control indicates that random rotation cannot replicate ICA's uplift; ICA finds specific rotations that concentrate biological signals into few features (sparsity and concentration rise together), effectively separating biological variation from technical batch effects.
- **Architectural differences match theoretical typing**: GPT-like models align linearly, while MAE-like models align rigidly, consistent with how their respective losses map representations to loss (linear vs non-linear decoders).

## Highlights & Insights
- Clearly separates "representation stability" into statistical identifiability (consistency) and structural identifiability (correctness), and uses $\epsilon$ relaxation to make "near-identifiability" a measurable and verifiable concept for the first time—whereas previous pointwise identifiability was impossible for modern large models.
- The most ingenious step is "shifting the assumption": moving strong assumptions from verifiable data generation processes to model classes controlled by common regularization (dynamic isometry/bi-Lipschitz), allowing the theory to cover MAE, supervised learners, and GPT intermediate layers.
- Provides a near "tuning-free" disentanglement recipe—vanilla autoencoder + latent linear ICA—which beats specialized β-VAE/β-TCVAE/BioAE on synthetic benchmarks and is directly transferable as a post-processor for any pre-trained model.
- Real biological application (cell painting de-batching) demonstrates purely unsupervised separation of technical and biological variation, substantially improving cross-batch OOD generalization, grounding abstract theory in the practical pain points of drug discovery.

## Limitations & Future Work
- Local bi-Lipschitz conditions are difficult to verify directly in practice; the authors rely on "dynamic isometry" regularization as an indirect argument, and the absolute magnitude of $\epsilon$ cannot be precisely measured.
- Theorem 3's structural identifiability relies on **perfect reconstruction** and a set of strong assumptions—data generation being a bi-Lipschitz diffeomorphism with independent non-Gaussian true factors—which complex real data might not satisfy; image isometry arguments are largely based on simplified synthetic manifolds like dSprites.
- All theorems hold in the "infinite data limit"; how finite sample statistical estimation errors and ICA convergence affect conclusions is not given finite-sample guarantees.
- Structural identifiability is entirely unsupervised, relying on inductive bias rather than intervention, meaning there is no guarantee that recovered factors are "correct" in a causal sense.

## Related Work & Insights
- **vs Roeder et al. (2021)**: They only cover the penultimate layer linearly mapped to loss, with $H_{\text{linear}}$ identifiability; this paper extends the conclusion to any intermediate layer and non-linear mappings via decoder bi-Lipschitz assumptions, unifying them under $\epsilon$-near-identifiability.
- **vs Zimmermann et al. (2021) (InfoNCE)**: They require the augmentation distribution to be isotropic in latent space (a strong, unverifiable data assumption) to get $H_{\text{rigid}}$ identifiability; this paper uses mild model-side assumptions, making them fewer and more controllable.
- **vs Buchholz & Schölkopf (2024) (Non-linear ICA)**: Theorem 2 is similar in form to their Theorem 3.1, but this paper uses $L^\infty$ bi-Lipschitz for pointwise constraints, while they use $L^2$ mean-near-isometry for weaker Jacobian constraints.
- **vs Horan et al. (2021) (Isometry Learning + ICA)**: This paper generalizes their perfect identifiability case, proving $\epsilon$-nearness does not add extra complexity to downstream ICA and pushes it from synthetic settings to foundation-model-scale real biological data.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First clear distinction between statistical and structural identifiability, providing a universal $\epsilon$-near-identifiability definition and most general intermediate layer theorems.
- Experimental Thoroughness: ⭐⭐⭐⭐ Ranges from MNIST controlled validation and pre-trained model measurement to synthetic disentanglement and real biological foundation models; however, finite sample/absolute $\epsilon$ quantification is weak.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical layering with intuitive explanations; however, theorems are provided in informal versions with details relegated to appendices, creating a high entry barrier.
- Value: ⭐⭐⭐⭐⭐ Simultaneously unifies identifiability theory and provides a directly reusable unsupervised disentanglement recipe with real biological applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Memory-Statistics Tradeoff in Continual Learning with Structural Regularization](memory-statistics_tradeoff_in_continual_learning_with_structural_regularization.md)
- [\[ICLR 2026\] Learning Correlated Reward Models: Statistical Barriers and Opportunities](learning_correlated_reward_models_statistical_barriers_and_opportunities.md)
- [\[ICLR 2026\] SEINT: An Efficient SE(p)-Invariant Transport Metric Driven by Polar Transport Discrepancy-based Representation](an_efficient_sep-invariant_transport_metric_driven_by_polar_transport_discrepanc.md)
- [\[ICLR 2026\] Minimax Sample Complexity of Graph Neural Networks: Lower Bounds and Structural Effects](minimax_sample_complexity_of_graph_neural_networks_lower_bounds_and_structural_e.md)
- [\[ICLR 2026\] A Statistical Learning Perspective on Semi-dual Adversarial Neural Optimal Transport Solvers](a_statistical_learning_perspective_on_semi-dual_adversarial_neural_optimal_trans.md)

</div>

<!-- RELATED:END -->
