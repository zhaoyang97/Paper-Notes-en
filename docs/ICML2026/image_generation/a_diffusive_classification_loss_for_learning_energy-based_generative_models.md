---
title: >-
  [Paper Note] A Diffusive Classification Loss for Learning Energy-based Generative Models
description: >-
  [ICML2026][Image Generation][Diffusive Classification Loss] This paper proposes DiffCLF, which reformulates energy estimation across temporal noise levels as a classification problem. By training jointly with DSM…
tags:
  - "ICML2026"
  - "Image Generation"
  - "Diffusive Classification Loss"
  - "Energy-based Models"
  - "Denoising Score Matching"
  - "Mode Blindness"
  - "Boltzmann Generator"
date: 2026-05-08
content_hash: 95e74ecb0e96ab1c
---

# A Diffusive Classification Loss for Learning Energy-based Generative Models

**Conference**: ICML2026  
**arXiv**: [2601.21025](https://arxiv.org/abs/2601.21025)  
**Code**: https://github.com/h2o64/diffclf  
**Area**: Image Generation / Diffusion Models / Energy-based Models  
**Keywords**: Diffusive Classification Loss, Energy-based Models, Denoising Score Matching, Mode Blindness, Boltzmann Generator  

## TL;DR
This paper proposes DiffCLF, which reformulates energy estimation across temporal noise levels as a classification problem. By training jointly with DSM, it learns more reliable energy functions without expensive maximum likelihood sampling, specifically addressing the mode blindness of score matching regarding multi-modal weights.

## Background & Motivation
**Background**: Diffusion models and stochastic interpolants typically learn the score, which is the gradient of the log-density with respect to the input. This training objective is efficient as it avoids computing normalization constants or performing inner-loop MCMC sampling; the sampling phase only requires plugging the score into a reverse SDE/ODE to generate data.

**Limitations of Prior Work**: Many downstream tasks require the energy itself rather than just the score. For example, model composition involves taking the product or mixture of multiple model densities; Boltzmann Generators use learned energy for reweighting and SMC; and free energy difference estimation relies on the log-density of intermediate distributions. Energy learned solely through DSM is reliable within each connected mode but remains insensitive to the relative weights between different modes.

**Key Challenge**: Maximum likelihood can constrain global density ratios but requires sampling from $p^\theta_t$; DSM is computationally cheap but only constrains local gradients, leading to the "mode blindness" where distributions with identical shapes but different mixing weights appear nearly identical to the model. The authors aim to learn global energy suitable for downstream tasks without reverting to expensive EBM maximum likelihood training.

**Goal**: First, design an energy supervision signal computable using only existing noising/interpolation samples. Second, ensure compatibility with DSM without sacrificing generation quality. Third, provide theoretical justification for recovering the true marginal distribution and empirically demonstrate improvements in composition, BG, and molecular energy learning.

**Key Insight**: The diffusion process naturally provides a sequence of temporal marginal distributions $p_{t_1},\ldots,p_{t_N}$. Given a noisy sample, if the model predicts which temporal level it originated from, the softmax logit of the classifier must compare energy values across different times. This comparison directly involves the relative height of the log-density rather than just the gradient.

**Core Idea**: Use a classification loss identifying "which diffusion time the sample comes from" to supervise the EBM energy scale, while using DSM to fix local scores. This allows the model to capture both global density ratios and local generative dynamics.

## Method
The starting point is a unified noising framework: given a samplable stochastic process $X_t$ and independent Gaussian noise $Z$, the observed variable is $Y_t=X_t+\gamma(t)Z$. The authors aim to learn the marginal density $p_t(y)$ at each time $t$, but only permit learning an unnormalized energy $U^\theta_t(y)$ alongside a learnable free-energy/bias term $F^\theta_t$. To avoid expectations under the model distribution required by maximum likelihood, DiffCLF bypasses inner-loop sampling.

### Overall Architecture
During training, a set of times $t_{1:N}$ is sampled, and noisy samples $Y_{t_i}$ are generated from the corresponding marginal distributions. The model computes energy logits $-U^\theta_{t_j}(y)+F^\theta_{t_j}$ for all temporal levels for the same sample $y$, followed by an $N$-class softmax to predict the true source time $t_i$. This treats each $p_t$ as a class-conditional density for multi-class logistic regression.

DiffCLF provides log-density comparisons across time, while DSM continues to provide score supervision. The joint objective $L_{DSM}+L_{clf}$ aligns energy slopes within each time slice via DSM and aligns energy heights across modes and times via DiffCLF. The paper proves that the true marginal distribution is an optimal solution for DiffCLF, and joint training with DSM eliminates non-uniqueness caused by common positive scaling functions.

### Key Designs
1. **Energy Learning via Temporal Classification**:
	- **Function**: Transforms the estimation of $p_t(y)$ into a supervised classification problem of identifying the temporal marginal of $y$.
	- **Mechanism**: For a sample $Y_{t_i}$, a cross-entropy loss is applied using $p^\theta(c=i|y)=\exp(-U^\theta_{t_i}(y)+F^\theta_{t_i})/\sum_j\exp(-U^\theta_{t_j}(y)+F^\theta_{t_j})$. Since softmax compares energy values, the model learns relative density magnitudes at the same point across different times.
	- **Design Motivation**: Score matching only considers $\nabla_y\log p_t(y)$. If two multi-modal distributions have the same mode locations but different weights, their scores are nearly identical. The classification posterior changes with mixing weights, thus providing the missing global ratio information.

2. **Complementing rather than Replacing DSM**:
	- **Function**: Maintains original diffusion/interpolation generation quality while giving the energy numerical meaning for downstream use.
	- **Mechanism**: DSM constrains $-\nabla_y U^\theta_t(y)$ to match the true score, while DiffCLF constrains the unnormalized density ratios across $t$. Together, they fix both the shape and relative height of the energy.
	- **Design Motivation**: DiffCLF alone has non-unique solutions (e.g., multiplying all temporal densities by a common positive function preserves the posterior). Adding DSM provides local constraints, theoretically making the true $p_t$ the unique optimal solution.

3. **Binary and Multi-class Computational Paths**:
	- **Function**: Offers adjustable versions balancing accuracy and computation.
	- **Mechanism**: The multi-class version compares $N$ temporal levels simultaneously for richer supervision. The binary version compares pairs $(t,t')$ with a loss involving two softplus terms, adding roughly 50% overhead to DSM. The paper also links the binary limit to time-score matching.
	- **Design Motivation**: In training, the number of energy network forward passes is the primary cost. The binary version stays close to the DSM budget, while the multi-class version reduces estimation variance and enhances self-consistency.

### Loss & Training
The total objective is $L_{DSM}+L_{clf}$. The DSM component uses standard denoising regression from diffusion models or stochastic interpolants. The DiffCLF component samples multiple temporal levels per batch, calculates energy logits for all candidates, and applies cross-entropy. The authors emphasize in the pseudocode that the batch size is adjusted compared to DSM-only baselines to ensure the number of DSM updates remains consistent. For diffusion models, Thornton/Karras-style energy preconditioning is adopted to ensure a simple Gaussian prior energy at $t=T$.

## Key Experimental Results

### Main Results
Experiments cover synthetic high-dimensional Gaussian mixtures (MOG), stochastic interpolants, molecular systems, model composition, Boltzmann Generators, and free energy estimation.

| Task / Dataset | Metric | DiffCLF / Ours | Main Comparison | Conclusion |
|--------|------|------|----------|------|
| MOG-40, DM, 128D | Classification Loss $L_{clf}$ | 4.40±1.00 | DSM 383.53±35.99; CtSM 20.86±4.93 | DiffCLF significantly corrects energy ratios across modes/time |
| MOG-40, DM, 128D | MMD ×100 | 3.54±1.34 | DSM 1.99±0.35; CtSM 5.20±0.34 | Generation quality remains comparable; energy is improved without sacrificing sampling |
| ALDP Molecular System | Langevin PMF | 0.094±0.001 | DSM 1.047±0.924; FPE 0.104±0.004 | Significantly outperforms DSM and slightly beats FPE for Langevin energy use |
| Chignolin System | Train Time | 18.9 GPU h | FPE 49.6 GPU h | More lightweight than FPE regularization, approx. 2.6x faster |
| ALDP Solvation Free Energy | Estimated Value | 29.02±0.41 | Lbase 27.30±0.45; Ref 29.43±0.01 | TI estimates with DiffCLF are closer to reference values |

### Ablation Study
The core comparisons focus on DSM-only, DSM+CtSM, and DSM+DiffCLF.

| Configuration | Key Metric | Description |
|------|---------|------|
| DSM only | MOG-40 128D $L_{clf}=383.53±35.99$ | Good scores/generation but worst energy self-consistency, showing mode weight blindness |
| DSM + CtSM | MOG-40 128D $L_{clf}=20.86±4.93$ | Time-score constraints help but still rely on local derivatives, failing global ratios |
| DSM + DiffCLF | MOG-40 128D $L_{clf}=4.40±1.00$ | Classification directly compares energy heights, best recovering true log-density |
| FPE regularization | ALDP train time 8.1 GPU h | Learns molecular energy well but requires backprop through time derivatives, scores, and Laplacian |
| DiffCLF | ALDP train time 5.6 GPU h | Maintains Langevin JS/PMF close to FPE with faster training |

### Key Findings
- DiffCLF is more sensitive to "energy correctness" than to "sample appearance." In MOG experiments, DSM's MMD is not always poor, but classification loss and log-density scatter show it fails to learn density heights.
- Advantages of DiffCLF are amplified in mode composition and BG tasks which directly utilize learned marginal energy. DSM’s mode weight errors lead to significantly biased composition ratios.
- Results in molecular systems show DiffCLF is not just for toy problems. Statistics of Langevin dynamics using $U^\theta_{t=0}$ improve significantly on ALDP and Chignolin compared to DSM while being more efficient than FPE.

## Highlights & Insights
- The clever mapping of "diffusion time" to a classification label allows a model to answer "which noise level is this," indirectly providing energy ratio supervision.
- It clearly addresses mode blindness: scores are local slopes that cannot reliably encode mass ratios of disconnected modes; the classification posterior depends on the ratio of density values, thus "seeing" mixing weights.
- The method is minimally invasive to score-based training. It doesn't require discarding DSM or inner-loop EBM sampling; it merely involves additional time-conditioned forward passes.
- Downstream tasks like model composition and free energy difference are well-chosen as they rely on numerical energy values rather than just visual quality.

## Limitations & Future Work
- Experimental scale is currently small to medium. Large-scale SMC composition for images remains a future direction.
- While cheaper than MLE, the multi-class version still requires $N+1$ network evaluations per sample. Trade-offs between the number of levels, batch size, and throughput need engineering for large models.
- The method relies on sufficient "classifiability" between marginals. If time sampling is too dense, the binary limit might degrade into local constraints like time-score matching, potentially re-introducing mode blindness.
- The learned energy remains unnormalized. While acceptable for many tasks, applications requiring the exact normalizing constant still need additional estimation.

## Related Work & Insights
- **vs DSM / score matching**: DSM learns scores and is cheap but insensitive to weights of disconnected modes. DiffCLF supplements this with global density ratios via classification.
- **vs Conditional Time Score Matching**: CtSM uses $\partial_t\log p_t$ constraints, acting as a derivative-level fix. DiffCLF directly uses density ratios across discrete levels, making it more effective for mode weights.
- **vs Fokker-Planck regularization**: FPE regularization uses the density PDE and is computationally heavy. DiffCLF uses standard cross-entropy, achieving similar or better quality in molecular systems faster.
- **Inspiration**: Generative model training often has "natural labels" (time, noise level, temperature). Converting these into density ratio classification is a general path to recover energy scales for score models.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Uses temporal classification for EBM learning to address the root cause of score matching mode blindness.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers synthetic, molecular, composition, and BG tasks, though large-scale image experiments are limited.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical motivation and connections, though the dense experimental results require careful attention.
- Value: ⭐⭐⭐⭐⭐ Highly valuable for diffusion applications requiring energy values (e.g., physical sampling and compositional generation).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] $f$-Trajectory Balance: A Loss Family for Tuning GFlowNets, Generative Models, and LLMs with Off- and On-Policy Data](f-trajectory_balance_a_loss_family_for_tuning_gflownets_generative_models_and_ll.md)
- [\[NeurIPS 2025\] Latent Zoning Network: A Unified Principle for Generative Modeling, Representation Learning, and Classification](../../NeurIPS2025/image_generation/latent_zoning_network_a_unified_principle_for_generative_modeling_representation.md)
- [\[AAAI 2026\] Symmetrical Flow Matching: Unified Image Generation, Segmentation, and Classification with Score-Based Generative Models](../../AAAI2026/image_generation/symmetrical_flow_matching_unified_image_generation_segmentation_and_classificati.md)
- [\[ICML 2026\] A Kinetic Energy Perspective of Flow Matching](a_kinetic_energy_perspective_of_flow_matching.md)
- [\[ICML 2026\] EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization](evogm_learning_to_merge_llms_via_evolutionary_generative_optimization.md)

</div>

<!-- RELATED:END -->
