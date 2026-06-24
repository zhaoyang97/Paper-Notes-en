---
title: >-
  [Paper Note] A Diffusive Classification Loss for Learning Energy-based Generative Models
description: >-
  [ICML2026][Image Generation][Diffusive classification loss] This paper proposes DiffCLF, which reformulates energy estimation across temporal noise levels as a classification problem. By training jointly with DSM, it learns more reliable energy functions without requiring expensive maximum likelihood sampling, specifically alleviating the "mode blindness" of score matching regarding multi-modal weights.
tags:
  - "ICML2026"
  - "Image Generation"
  - "Diffusive classification loss"
  - "energy-based models"
  - "denoising score matching"
  - "mode blindness"
  - "Boltzmann Generator"
date: 2026-05-08
content_hash: 5ee365ef1e3effb0
---

# A Diffusive Classification Loss for Learning Energy-based Generative Models

**Conference**: ICML2026  
**arXiv**: [2601.21025](https://arxiv.org/abs/2601.21025)  
**Code**: https://github.com/h2o64/diffclf  
**Area**: Image Generation / Diffusion Models / Energy-based Models  
**Keywords**: Diffusive classification loss, energy-based models, denoising score matching, mode blindness, Boltzmann Generator  

## TL;DR
This paper proposes DiffCLF, which reformulates energy estimation across temporal noise levels as a classification problem. By training jointly with DSM, it learns more reliable energy functions without requiring expensive maximum likelihood sampling, specifically alleviating the "mode blindness" of score matching regarding multi-modal weights.

## Background & Motivation
**Background**: Diffusion models and stochastic interpolants typically learn only the score, i.e., the gradient of the log-density with respect to the input. This training objective is highly efficient because it bypasses the normalization constant and avoids inner-loop MCMC sampling. During sampling, the score is simply plugged into a reverse SDE/ODE to generate data.

**Limitations of Prior Work**: Many downstream tasks require the energy itself rather than just the score. For example, model composition involves taking the product or mixture of densities from multiple models; Boltzmann Generators use learned energy for reweighting and SMC; and free energy difference estimation relies on the log-density of intermediate distributions. Energy learned solely through DSM is only reliable within each connected mode and is insensitive to the relative weights between different modes.

**Key Challenge**: Maximum likelihood estimation (MLE) can constrain global density ratios but requires sampling from $p^\theta_t$. DSM is computationally cheap but only constrains local gradients, making it prone to treating distributions with "same shape but different mixture weights" as nearly identical. The authors aim to solve the problem of "learning global energy useful for downstream tasks without reverting to expensive EBM MLE training."

**Goal**: First, design an energy supervision signal computable using only existing noising/interpolation samples. Second, ensure compatibility with DSM without sacrificing original generation quality. Third, theoretically explain why this signal recovers the true marginal distribution and empirically demonstrate improvements in composition, BG, and molecular energy learning.

**Key Insight**: The diffusion process naturally yields a sequence of temporal marginal distributions $p_{t_1},\ldots,p_{t_N}$. Given a noisy sample, if a model is tasked with identifying its originating time level, the classifier's softmax logit must compare energy values across different times. This comparison directly involves the relative height of the log-density rather than just the gradient.

**Core Idea**: Use a classification loss—predicting "which diffusion time the sample came from"—to supervise the EBM energy scale, while using DSM to fix local scores. This allows the model to simultaneously acquire global density ratios and local generative dynamics.

## Method
The starting point is a unified noising framework: given a process $X_t$ and independent Gaussian noise $Z$, the observed variable is $Y_t=X_t+\gamma(t)Z$. The objective is to learn the marginal density $p_t(y)$ at each time $t$ by modeling an unnormalized energy $U^\theta_t(y)$ and a learnable free-energy/bias term $F^\theta_t$. To avoid inner-loop sampling required by MLE gradients, DiffCLF circumvents the sampling process.

### Overall Architecture
During training, a set of time steps $t_{1:N}$ is sampled, and noisy samples $Y_{t_i}$ are generated from corresponding marginals. The model calculates the logit $-U^\theta_{t_j}(y)+F^\theta_{t_j}$ for all time levels for the same sample $y$, followed by an $N$-class softmax to predict the true originating time $t_i$. This treats each $p_t$ as a class-conditional density for multi-class logistic regression.

DiffCLF provides log-density comparisons across time, while DSM provides score supervision. The joint objective $L_{DSM}+L_{clf}$ functions as follows: DSM aligns the energy slopes within each time slice, and DiffCLF aligns the energy heights across different modes and time steps. The paper proves that the true marginal distribution is an optimal solution for DiffCLF, and joint DSM eliminates non-uniqueness caused by a common positive multiplicative function.

### Key Designs
**1. Energy Learning via Temporal Classification: Replacing "density estimation" with "time level identification."** Score matching only considers $\nabla_y\log p_t(y)$. For multimodal distributions with identical mode locations but different mixture weights, the scores are nearly identical. This is the root of mode blindness. DiffCLF reformulates this as a supervised classification: given $Y_{t_i}$, the model predicts the time marginal using $p^\theta(c=i\mid y)=\exp(-U^\theta_{t_i}(y)+F^\theta_{t_i})/\sum_j\exp(-U^\theta_{t_j}(y)+F^\theta_{t_j})$ with cross-entropy. Since the softmax compares energy values at the same point $y$ across different times, the model is forced to learn relative heights; the classification posterior changes with mixture weights, recovering the global density ratios invisible to the score.

**2. Joint with DSM rather than replacement: Local slopes by DSM, global heights by classification.** Using DiffCLF alone allows non-unique solutions—multiplying all temporal densities by a common positive function preserves the classification posterior, "unbinding" the energy shape. The authors thus combine it with Denoising Score Matching (DSM). DSM constrains $-\nabla_y U^\theta_t(y)$ to approximate the true score, fixing the energy slope within each time slice. DiffCLF constrains unnormalized density ratios between different $t$, fixing energy heights across time and modes. Together, they pin down both shape and relative height, theoretically returning the unique optimal solution to the true $p_t$ without sacrificing generation quality.

**3. Binary and Multi-class Computational Paths: Dialing between supervision richness and compute budget.** The primary cost is the number of forward passes through the energy network. The multi-class version compares $N$ time levels, providing richer supervision and lower variance through self-consistency. The binary version compares a pair $(t,t')$, simplifying the loss to two softplus terms with an overhead of $\sim 50\%$ relative to DSM, keeping the budget close to pure DSM. The paper also links the binary limit to time-score matching: as time sampling becomes infinitely dense, it degenerates into a local derivative constraint.

### Loss & Training
The total objective is $L_{DSM}+L_{clf}$. The DSM component uses standard denoising regression from diffusion models or stochastic interpolants. For DiffCLF, multiple time levels are sampled per batch to compute energy logits and cross-entropy. The authors emphasize adjusting batch sizes for fair comparison with DSM-only baseline updates. For diffusion models, Thornton/Karras-style energy preconditioning is adopted to ensure the zero-network state corresponds to a simple Gaussian prior energy.

## Key Experimental Results

### Main Results
Experiments cover high-dimensional Gaussian mixtures, stochastic interpolants, molecular systems, model composition, Boltzmann Generators, and free energy estimation. Key findings highlighting DiffCLF’s value: on MOG-40, it significantly reduces classification self-consistency loss without degrading FD/MMD; on molecular systems, it matches FPE regularization quality with faster training; and on ALDP free energy estimation, TI results are closer to references than the Lbase baseline.

| Task / Dataset | Metric | DiffCLF / Ours | Comparison | Conclusion |
|--------|------|------|----------|------|
| MOG-40, DM, 128D | Classification Loss $L_{clf}$ | 4.40±1.00 | DSM: 383.53±35.99; CtSM: 20.86±4.93 | DiffCLF significantly corrects cross-mode/time energy ratios |
| MOG-40, DM, 128D | MMD ×100 | 3.54±1.34 | DSM: 1.99±0.35; CtSM: 5.20±0.34 | Generation quality remains comparable |
| ALDP Molecule | Langevin PMF | 0.094±0.001 | DSM: 1.047±0.924; FPE: 0.104±0.004 | Significantly outperforms DSM in Langevin dynamics from energy |
| Chignolin Molecule | Training Time | 18.9 GPU h | FPE: 49.6 GPU h | More lightweight than FPE, ~2.6x faster |
| ALDP Solvation Free Energy | Estimated Value | 29.02±0.41 | Lbase: 27.30±0.45; Ref: 29.43±0.01 | TI estimation closer to reference with DiffCLF |

### Ablation Study
The paper analyzes alternative objectives and level counts rather than standard module removals. The core comparison is between DSM-only, DSM+CtSM, and DSM+DiffCLF.

| Configuration | Key Metric | Description |
|------|---------|------|
| DSM only | MOG-40 128D $L_{clf}=383.53±35.99$ | Good scores/generation, but worst energy consistency (mode blindness) |
| DSM + CtSM | MOG-40 128D $L_{clf}=20.86±4.93$ | Time-score constraints help but rely on local derivatives; global ratios remain imperfect |
| DSM + DiffCLF | MOG-40 128D $L_{clf}=4.40±1.00$ | Classification directly compares heights; best at recovering log-density |
| FPE regularization | ALDP train time 8.1 GPU h | Good energy but high cost (backpropping time derivs, scores, Laplacian) |
| DiffCLF | ALDP train time 5.6 GPU h | Maintains Langevin JS/PMF quality comparable to FPE with faster training |

### Key Findings
- DiffCLF is more sensitive to "energy correctness" than "sample visual correctness." In MOG experiments, DSM's MMD isn't always poor, but $L_{clf}$ shows it fails to learn density heights.
- Advantages are amplified in mode composition and BG tasks which directly use learned marginal energy; DSM results in skewed proportions due to mode weight errors.
- Molecular results prove DiffCLF isn't a toy loss: Langevin dynamics statistics on ALDP and Chignolin using $U^\theta_{t=0}$ improve significantly, while being more efficient than FPE regularization.

## Highlights & Insights
- The cleverest part is treating "diffusion time" as a classification label. Since diffusion training already samples noise levels, asking the model "which level is this?" indirectly provides energy ratio supervision.
- It provides a clear explanation for mode blindness: the score is a local slope and cannot encode mass ratios between disconnected modes. The classification posterior depends on density ratios, naturally "seeing" mixture weights.
- The method is minimally intrusive. It doesn't replace DSM or require EBM inner-loop sampling; it just adds extra time-conditioned forward passes, making it easy to embed in diffusion, interpolants, or discrete CTMCs.
- The downstream tasks are well-chosen. Composition, BG, and free energy are tasks where numerical energy values matter more than human-perceived image quality.

## Limitations & Future Work
- Experimental scale is relatively small/medium. The authors acknowledge that large-scale SMC composition for images remains a future direction.
- While cheaper than MLE, the multi-class version requires $N+1$ network evaluations. Engineering trade-offs between level counts, batch size, and throughput are needed for large models.
- The method relies on sufficient "classifiability" between marginals. If time sampling is too dense, the binary limit approaches time-score matching, potentially re-introducing mode blindness.
- The learned energy remains unnormalized. While acceptable for many tasks, applications requiring precise normalizing constants still need additional estimation.

## Related Work & Insights
- **vs DSM / score matching**: DSM learns scores—cheap but insensitive to relative weights of disconnected modes. DiffCLF supplements this with global ratio supervision via classification.
- **vs Conditional Time Score Matching**: CtSM uses $\partial_t\log p_t$ as a derivative-level fix. DiffCLF compares density ratios directly across finite levels, handling global ratios more directly.
- **vs Fokker-Planck regularization**: FPE regularization is theoretically strong but computationally heavy. DiffCLF reaches similar statistical quality in molecular systems while training significantly faster.
- **Inspiration**: Many generative models have natural "auxiliary labels" (time, noise level, temperature). Converting these labels into density ratio classification is a general path to supplementing score models with an energy scale.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Using temporal classification for EBM energy learning addresses the root cause of mode blindness.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers various domains, though large-scale image experiments are limited.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical motivation; dense but well-detailed appendices.
- Value: ⭐⭐⭐⭐⭐ High value for diffusion applications requiring energy values, especially in physics sampling and composition.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] $f$-Trajectory Balance: A Loss Family for Tuning GFlowNets, Generative Models, and LLMs with Off- and On-Policy Data](f-trajectory_balance_a_loss_family_for_tuning_gflownets_generative_models_and_ll.md)
- [\[NeurIPS 2025\] Latent Zoning Network: A Unified Principle for Generative Modeling, Representation Learning, and Classification](../../NeurIPS2025/image_generation/latent_zoning_network_a_unified_principle_for_generative_modeling_representation.md)
- [\[AAAI 2026\] Symmetrical Flow Matching: Unified Image Generation, Segmentation, and Classification with Score-Based Generative Models](../../AAAI2026/image_generation/symmetrical_flow_matching_unified_image_generation_segmentation_and_classificati.md)
- [\[ICML 2026\] A Kinetic Energy Perspective of Flow Matching](a_kinetic_energy_perspective_of_flow_matching.md)
- [\[CVPR 2026\] Advancing Image Classification with Discrete Diffusion Classification Modeling](../../CVPR2026/image_generation/advancing_image_classification_with_discrete_diffusion_classification_modeling.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICML 2026\] $f$-Trajectory Balance: A Loss Family for Tuning GFlowNets, Generative Models, and LLMs with Off- and On-Policy Data](f-trajectory_balance_a_loss_family_for_tuning_gflownets_generative_models_and_ll.md)
- [\[AAAI 2026\] Symmetrical Flow Matching: Unified Image Generation, Segmentation, and Classification with Score-Based Generative Models](../../AAAI2026/image_generation/symmetrical_flow_matching_unified_image_generation_segmentation_and_classificati.md)
- [\[NeurIPS 2025\] Latent Zoning Network: A Unified Principle for Generative Modeling, Representation Learning, and Classification](../../NeurIPS2025/image_generation/latent_zoning_network_a_unified_principle_for_generative_modeling_representation.md)
- [\[ICML 2026\] A Kinetic Energy Perspective of Flow Matching](a_kinetic_energy_perspective_of_flow_matching.md)
- [\[CVPR 2026\] Transition Models: Rethinking the Generative Learning Objective](../../CVPR2026/image_generation/transition_models_rethinking_the_generative_learning_objective.md)

</div>

<!-- RELATED:END -->
