---
title: >-
  [Paper Note] SCRAPL: Scattering Transform with Random Paths for Machine Learning
description: >-
  [ICLR 2026][Optimization][scattering transform] To address the prohibitive computational cost of using the multivariate scattering transform (ST) as a differentiable loss function due to its large number of paths $P$…
tags:
  - "ICLR 2026"
  - "Optimization"
  - "scattering transform"
  - "random path sampling"
  - "DDSP"
  - "importance sampling"
  - "variance reduction"
date: 2026-05-08
content_hash: 6d00665fd7f925d5
---

# SCRAPL: Scattering Transform with Random Paths for Machine Learning

**Conference**: ICLR 2026
**arXiv**: [2602.11145](https://arxiv.org/abs/2602.11145)
**Code**: Available (Python package, [project website](https://christhetree.github.io/scrapl/))
**Area**: Signal Processing / Time Series
**Keywords**: scattering transform, random path sampling, DDSP, importance sampling, variance reduction

## TL;DR

To address the prohibitive computational cost of using the multivariate scattering transform (ST) as a differentiable loss function due to its large number of paths $P$, this paper proposes SCRAPL—a framework that samples a single random path per training step and stabilizes gradient updates via three variance-reduction techniques: P-Adam (path-adaptive momentum), P-SAGA (path stochastic average gradient), and $\theta$-importance sampling. On unsupervised sound matching tasks, SCRAPL achieves Pareto optimality by attaining accuracy close to the full-path ST at computational costs comparable to multi-scale spectral (MSS) loss.

## Background & Motivation

**Background** The scattering transform (ST) is a wavelet-based nonlinear operator that decomposes high-resolution inputs into low-resolution coefficients along multiple paths. ST distances have been shown through perceptual studies to reliably predict auditory dissimilarity, and joint time–frequency scattering (JTFS) provides an idealized model of spectro-temporal receptive fields in the human auditory cortex. This makes ST distances theoretically optimal perceptual loss functions for audio generation and deep inverse problems.

**Limitations of Prior Work** Despite their theoretical appeal, ST-based losses are computationally infeasible in practice: JTFS contains hundreds of paths, each involving a multivariate wavelet convolution, making a full forward–backward pass over all $P$ paths approximately $P$ times more expensive than a single-path evaluation. In granular synthesizer matching experiments, full-JTFS training is 25× slower than multi-scale spectral (MSS) loss, rendering ST losses practically unusable for neural network training. Meanwhile, MSS, although computationally efficient, yields uninformative gradients when there is input–output temporal misalignment or when the synthesizer involves spectro-temporal modulations—making it an inadequate substitute for ST.

**Key Challenge** There is a fundamental tension between the quality of ST loss and its computational cost. Naïve random path sampling—computing only one path per step—reduces computation by a factor of $P$, but the resulting gradient variance is too large for stable training.

**Key Insight** The ST loss is structurally a finite sum over $P$ path losses, which maps directly onto the classical finite-sum setting in stochastic optimization. Variance-reduction techniques such as SAGA can therefore be applied. However, paths are not independently and identically distributed—different paths correspond to distinct time–frequency modulation patterns—so standard algorithms cannot be applied directly.

**Core Idea** Transform the tree-structured computation of the scattering transform into a stochastic optimization problem, and enable stable convergence with a single path per step through architecture-aware variance-reduction techniques.

## Method

### Overall Architecture

SCRAPL replaces the full-path ST loss within a standard neural network training loop. At each step, one path (or a small subset) is randomly sampled from the $P$ available paths, and only that path's loss and gradient are computed. Three complementary techniques compensate for the high variance of single-path gradients: P-Adam addresses gradient scale heterogeneity across paths, P-SAGA exploits historical gradient information to reduce variance, and $\theta$-importance sampling biases sampling toward more informative paths.

### Key Designs

1. **P-Adam: Path-Adaptive Momentum Estimation**

   - **Function**: Maintains independent first- and second-moment estimates for each path, replacing the shared moments used across all paths in standard Adam.
   - **Mechanism**: Standard Adam's moment estimates $(m, v)$ use exponential moving averages to smooth gradients across consecutive iterations. Because SCRAPL selects one path at random per step and gradient distributions differ substantially across paths, pooling moments conflates heterogeneous signals. P-Adam maintains per-path moments $(m_p, v_p)$ and adapts the exponential decay coefficient based on the time elapsed since path $p$ was last sampled, $(k - \tau_p)/P$—decaying more aggressively for stale estimates to prevent outdated history from dominating updates. The bias-correction exponent is adjusted from Adam's $\beta^k$ to $\beta^{k/P}$ to account for the path count.
   - **Design Motivation**: Path gradients are heterogeneous, corresponding to different spectro-temporal modulation scales; sharing moments across paths destabilizes the update direction.

2. **P-SAGA: Path Stochastic Average Gradient Acceleration**

   - **Function**: Maintains a table of historical gradients for all visited paths and applies a variance-reduction correction to the current gradient.
   - **Mechanism**: Classical SAGA variance reduction is applied along the path dimension rather than the data dimension. P-SAGA stores the most recent P-Adam update $\hat{g}_p$ for each path and tracks the set of visited paths $\Gamma$. The update at each step equals the current path's P-Adam gradient minus its stored historical gradient plus the mean gradient over all visited paths. Crucially, the additional memory overhead of P-SAGA scales with $P$ rather than with dataset size $N$, preserving practical feasibility.
   - **Design Motivation**: The dominant source of variance in single-path sampling is the difference in gradients across paths. P-SAGA explicitly cancels this variance by contrasting current and historical estimates, yielding smoother convergence.

3. **$\theta$-Importance Sampling: Architecture-Aware Path Sampling Bias**

   - **Function**: Constructs a non-uniform path sampling distribution $\pi$ prior to training by analyzing the curvature of the loss landscape, so that more informative paths are sampled more frequently.
   - **Mechanism**: The DDSP autoencoder consists of a learnable encoder $E_x$ (a neural network) and a fixed but differentiable synthesizer decoder $D$. For each parameter dimension $u$ and path $p$, the sensitivity of the ST loss to that parameter is computed via its partial derivative, and Hessian–vector products approximated by power iteration yield the largest eigenvalue of the loss landscape as an importance score $C_{u,p}$. Path sampling probabilities $\pi_p$ are then aggregated over parameter dimensions. The computation parallelizes over both paths and parameter dimensions and is performed only once before training.
   - **Design Motivation**: Different paths correspond to different time–frequency modulation patterns; for a given synthesizer configuration, some paths are inherently more informative than others (e.g., a slow-AM synthesizer requires only low-rate modulation paths). Uniform sampling wastes the computational budget on irrelevant paths.

### Loss & Training

The SCRAPL loss is an unbiased estimator of the full-path ST loss (proven in Proposition 3.1 via the chain rule and linearity of expectation). Under the DDSP paradigm, the encoder CNN operates on a constant-Q transform (CQT) and the decoder is a non-learnable synthesizer. JTFS configuration: $J{=}12$, $Q_1{=}8$, $Q_2{=}2$, $J_\text{fr}{=}3$, $Q_\text{fr}{=}2$, yielding approximately 315–483 paths. Training uses AdamW with no additional hyperparameters.

## Key Experimental Results

### Main Results (Granular Synthesizer Sound Matching)

| Method | Synthesizer Param. L1‰ ↓ | Compute Cost (ms) | Notes |
|---|---|---|---|
| Supervised P-loss | 20.5±0.2 | 0.5 | Theoretical upper bound |
| Full JTFS | 42.4 | 1731 | Best unsupervised, but extremely slow |
| **SCRAPL (+$\theta$-IS)** | **65.7±4.2** | **89.8** | Accuracy near JTFS, speed near MSS |
| MSS Log+Linear | 259.1±1.7 | 19.1 | Completely fails to match slope parameter |
| PANNs Wavegram | 158.9±4.4 | 29.3 | Matches density only |
| MS-CLAP | 165.9±8.2 | 75.6 | Matches density only |

### Ablation Study

| Configuration | Param. L1‰ ↓ | Convergence Steps ↓ | Validation Curve Variance ↓ |
|---|---|---|---|
| SCRAPL (sampling only) | 99.7±8.2 | 10906±1170 | 5.30±0.25 |
| +P-Adam | 87.4±14.5 | 8006±697 | 6.98±0.25 |
| +P-SAGA | 73.8±13.4 | 7296±683 | **3.46±0.15** |
| **+$\theta$-IS** | **65.7±4.2** | **6014±642** | **3.27±0.12** |
| Full JTFS | 42.4 | 1442 | 5.66 |

### Key Findings

- SCRAPL without any additional optimization techniques already outperforms all non-JTFS baselines, demonstrating that random path sampling of ST paths is itself a viable strategy.
- P-SAGA is the critical variance-reduction component (statistically significant, $p < 0.01$); $\theta$-IS yields statistically significant improvements in total variance and convergence speed.
- In chirplet synthesizer experiments, $\theta$-IS reduces $\theta_\text{AM}$ parameter error by 25–55% and $\theta_\text{FM}$ error by 14–80%, with convergence time reduced by 23–50%.
- In Roland TR-808 real drum machine experiments, SCRAPL performs consistently under both time-aligned and misaligned (meso) conditions, whereas MSS degrades severely under misalignment—validating the time-invariance advantage of ST distances.
- Visualization of $\theta$-IS sampling probabilities confirms that distinct path distributions are learned for different synthesizer configurations, with high-probability paths aligning with the corresponding synthesizer parameter ranges.

## Highlights & Insights

- SCRAPL transforms scattering transform losses—long regarded as "too expensive to be practical"—into a viable training objective, representing a contribution to the audio and signal processing community analogous to the shift from full-batch gradient descent to SGD.
- The mathematical rigor is notable: Proposition 3.1 establishes unbiasedness, and the derivations of P-Adam and P-SAGA are clean and introduce no additional hyperparameters.
- The design of $\theta$-importance sampling embodies the principle of domain-knowledge-guided sampling: rather than sampling all paths blindly, it allocates the sampling budget according to the curvature of the loss landscape with respect to synthesizer parameters, integrating signal-processing domain understanding with stochastic optimization.
- The experimental design deliberately employs a non-deterministic synthesizer (granular synthesis involves stochastic micro-timing), which is precisely the regime where MSS gradients become uninformative while ST distances remain valid—ensuring tight alignment between motivation and evaluation.

## Limitations & Future Work

- Validation is currently limited to audio/DDSP settings. Although SCRAPL is theoretically applicable to computer vision (2D rotation–translation scattering) and other domains employing the ST, cross-domain validation is absent.
- The initialization of $\theta$-IS requires power iteration for Hessian–vector products, which may incur non-negligible overhead for large-scale models.
- In TR-808 experiments, SCRAPL fails to recover the decay portion of drum sounds, possibly due to underrepresentation of low-frequency paths in the sampling distribution—suggesting the need for adaptive importance sampling that dynamically updates $\pi$ during training rather than fixing it at initialization.
- The current theoretical analysis establishes only unbiasedness; rigorous convergence rate analysis, particularly in the non-convex setting, is left for future work.

## Related Work & Insights

- **vs. pGST (pruned graph scattering)**: pGST performs fixed feature selection (retaining ~10% of paths), whereas SCRAPL more aggressively uses one path per step with variance reduction. The fundamental distinction is that pGST discards path information, while SCRAPL preserves all information in expectation.
- **vs. MSS (multi-scale spectral loss)**: MSS is the standard loss in DDSP but yields uninformative gradients under non-deterministic or misaligned conditions. SCRAPL makes JTFS—a theoretically grounded perceptual loss—practical within the computational budget of MSS.
- **Transferability**: The combination of finite-sum stochastic optimization and architecture-aware importance sampling generalizes to any loss function with a tree-structured decomposition.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First systematic study of stochastic optimization for scattering transforms; the path-adaptive designs of P-Adam and P-SAGA represent non-trivial algorithmic contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three DDSP tasks (granular / chirplet / TR-808), detailed ablations, comparison against full-path baselines, and statistical significance testing.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Mathematically rigorous with complete proofs, clear algorithmic pseudocode, and information-dense figures.
- **Value**: ⭐⭐⭐⭐ — Makes a class of high-quality but impractical perceptual loss functions operationally usable, with direct impact on the differentiable digital signal processing field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] FrontierCO: Real-World and Large-Scale Evaluation of Machine Learning Solvers for Combinatorial Optimization](frontierco_real-world_and_large-scale_evaluation_of_machine_learning_solvers_for.md)
- [\[ICLR 2026\] Exploring Diverse Generation Paths via Inference-time Stiefel Activation Steering](exploring_diverse_generation_paths_via_inference-time_stiefel_activation_steerin.md)
- [\[ACL 2026\] CLewR: Curriculum Learning with Restarts for Machine Translation Preference Learning](../../ACL2026/optimization/clewr_curriculum_learning_with_restarts_for_machine_translation_preference_learn.md)
- [\[ICLR 2026\] Rapid Training of Hamiltonian Graph Networks using Random Features](rapid_training_of_hamiltonian_graph_networks_using_random_features.md)
- [\[AAAI 2026\] pFed1BS: Personalized Federated Learning with Bidirectional Communication Compression via One-Bit Random Sketching](../../AAAI2026/optimization/personalized_federated_learning_with_bidirectional_communication_compression_via.md)

</div>

<!-- RELATED:END -->
