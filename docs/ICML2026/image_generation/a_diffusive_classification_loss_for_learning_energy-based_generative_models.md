---
title: >-
  [Paper Note] A Diffusive Classification Loss for Learning Energy-based Generative Models
description: >-
  [ICML 2026][Image Generation][Boltzmann Generator] This paper proposes DiffCLF, which reformulates the energy estimation between noise levels of a diffusion process as a classification problem. By joint training with DSM, it learns more reliable energy functions without introducing expensive maximum likelihood sampling, specifically alleviating the "mode blindness" of
tags:
  - ICML 2026
  - Image Generation
  - Boltzmann Generator
date: 2026-05-08
content_hash: f54718af1c6deaf7
---
# A Diffusive Classification Loss for Learning Energy-based Generative Models

**Conference**: ICML2026  
**arXiv**: [2601.21025](https://arxiv.org/abs/2601.21025)  
**Code**: https://github.com/h2o64/diffclf  
**Area**: Image Generation / Diffusion Models / Energy-based Models  
**Keywords**: diffusive classification loss, energy-based models, denoising score matching, mode blindness, Boltzmann Generator  

## TL;DR
This paper proposes DiffCLF, which reformulates the energy estimation between noise levels of a diffusion process as a classification problem. By joint training with DSM, it learns more reliable energy functions without introducing expensive maximum likelihood sampling, specifically alleviating the "mode blindness" of score matching regarding multimodal mixture weights.

## Background & Motivation
**Background**: Diffusion models and stochastic interpolants typically learn the score, i.e., the gradient of the log-density with respect to the input. This training objective is highly efficient as it avoids computing normalization constants or inner-loop MCMC sampling. During sampling, the score is simply plugged into a reverse SDE/ODE to generate data.

**Limitations of Prior Work**: Many downstream tasks require the energy itself rather than just the score. For example, model composition involves the product or mixture of densities from multiple models; Boltzmann Generators use learned energy for reweighting and Sequential Monte Carlo (SMC); and free energy difference estimation relies on the log-density of intermediate distributions. The energy learned solely via DSM is only reliable within each connected mode and remains insensitive to the relative weights between different modes.

**Key Challenge**: Maximum likelihood estimation (MLE) can constrain global density proportions but requires sampling from $p^\theta_t$; DSM is computationally cheap but only constrains local gradients, often treating distributions with "identical shapes but different mixture weights" as nearly the same. The authors aim to solve the problem of "learning global energy applicable to downstream tasks without reverting to expensive EBM maximum likelihood training."

**Goal**: First, design an energy supervision signal computable using only existing noising/interpolation samples. Second, ensure compatibility with DSM without sacrificing original generative quality. Third, theoretically explain why this signal recovers the true marginal distribution and experimentally demonstrate improvements in composition, Boltzmann Generators (BG), and molecular energy learning.

**Key Insight**: The diffusion process naturally provides a sequence of temporal marginal distributions $p_{t_1}, \dots, p_{t_N}$. Given a noisy sample, if the model is tasked with identifying which time level it originated from, the softmax logit of the classifier must compare energy values across different time steps. This comparison directly involves the relative heights of log-densities rather than just their gradients.

**Core Idea**: Use a classification loss—predicting "which diffusion time a sample comes from"—to supervise the scale of the EBM energy, while using DSM to fix local scores. This allows the model to simultaneously acquire global density proportions and local generative dynamics.

## Method
The starting point of the paper is a unified noising framework: given a samplable random process $X_t$ and independent Gaussian noise $Z$, the observed variable is $Y_t = X_t + \gamma(t)Z$. The authors aim to learn the marginal density $p_t(y)$ at each time $t$, but only permit learning the unnormalized energy $U^\theta_t(y)$ and a learnable free-energy/bias term $F^\theta_t$. Direct MLE would introduce expectations under the model distribution in the gradient, requiring sampling from the current EBM. DiffCLF's core mechanism bypasses this inner sampling.

### Overall Architecture
During training, a set of time points $t_{1:N}$ is sampled, and noisy samples $Y_{t_i}$ are generated from the corresponding marginals. The model computes logits $-U^\theta_{t_j}(y) + F^\theta_{t_j}$ for the same sample $y$ across all time levels and performs an $N$-class softmax to predict the true source time $t_i$. This is equivalent to treating each $p_t$ as a class-conditional density and performing multiclass logistic regression.

DiffCLF provides log-density comparisons across time, while DSM continues to provide score supervision. The joint objective $L_{DSM} + L_{clf}$ functions such that DSM aligns energy slopes within each time slice, and DiffCLF aligns energy heights between different modes and time steps. The paper proves that the true marginal distribution is one of the optimal solutions for DiffCLF, and joint DSM training eliminates non-uniqueness caused by common positive scaling functions.

### Key Designs
**1. Energy learning via time-level classification: Replacing "density estimation" with "identifying source time."** Score matching only examines $\nabla_y \log p_t(y)$. For multimodal distributions with identical mode locations but different mixture weights, the scores are almost identical—local gradients cannot "see" global proportions, which is the root of mode blindness. DiffCLF reformulates estimating $p_t(y)$ as a supervised classification task: given a noisy sample $Y_{t_i}$, the model predicts its source time marginal using $p^\theta(c=i \mid y) = \exp(-U^\theta_{t_i}(y) + F^\theta_{t_i}) / \sum_j \exp(-U^\theta_{t_j}(y) + F^\theta_{t_j})$ with cross-entropy. Since the softmax compares energy values at the same point $y$ across levels, the model is forced to learn the relative heights of densities at different times. The classification posterior varies with mixture weights, thereby recovering the global density proportions that the score misses.

**2. Joint with DSM rather than replacing it: Local slopes to DSM, global heights to classification.** Using only DiffCLF still leads to non-unique solutions—if all time densities are multiplied by a shared positive function, the classification posterior remains unchanged, leaving the energy shape unconstrained. Therefore, the authors join it with Denoising Score Matching (DSM). DSM constrains $-\nabla_y U^\theta_t(y)$ to approximate the true score, fixing the energy slope within each time slice. DiffCLF constrains the unnormalized density ratios between different $t$, fixing the energy height across time and modes. Together, they pin down both the shape and relative height of the energy, theoretically recovering the true $p_t$ as the unique optimal solution without sacrificing the generative quality of diffusion/interpolation models.

**3. Binary and multi-class calculation paths: Knobs between supervision richness and computational budget.** The number of forward passes through the energy network is the primary cost. The authors provide two versions: the multiclass version compares $N$ time levels simultaneously, providing richer supervision, lower estimation variance through more levels, and enhanced self-consistency. The binary version only compares a pair $(t, t')$, reducing the loss to two softplus terms with an overhead of approximately 50% of DSM, keeping the budget close to pure DSM. The paper also notes a connection between the binary limit and time-score matching—when time sampling is infinitely dense, it degenerates into a local derivative-style constraint.

### Loss & Training
The total training objective is $L_{DSM} + L_{clf}$. The DSM component uses standard denoising regression from diffusion models or stochastic interpolants. For the DiffCLF component, multiple time levels are sampled per batch, and energy logits across all candidate times are computed for cross-entropy. The authors emphasize in the pseudocode that for fair comparison with DSM-only, the batch size in DiffCLF training is adjusted to ensure the number of DSM updates remains consistent. For diffusion models, Thornton/Karras-style energy preconditioning is adopted, ensuring the zero-network state corresponds to a simple Gaussian prior energy.

## Key Experimental Results

### Main Results
The main experiments cover high-dimensional Gaussian mixtures, stochastic interpolants, molecular systems, model composition, Boltzmann Generators, and free energy estimation. Results demonstrating DiffCLF's value are highlighted below: On MOG-40, it significantly reduces classification consistency loss without significantly degrading FD/MMD. On molecular systems, it approaches the quality of FPE regularization but with faster training. In ALDP free energy estimation, it is closer to the reference value than the original Lbase.

| Task / Dataset | Metric | DiffCLF / Ours | Main Comparison | Conclusion |
|----------------|--------|----------------|-----------------|------------|
| MOG-40, DM, 128D | Classification Loss $L_{clf}$ | 4.40±1.00 | DSM 383.53±35.99; CtSM 20.86±4.93 | DiffCLF significantly corrects energy proportions across modes/time |
| MOG-40, DM, 128D | MMD ×100 | 3.54±1.34 | DSM 1.99±0.35; CtSM 5.20±0.34 | Generative quality remains comparable; objective is not to sacrifice sampling for energy |
| ALDP Molecular System | Langevin PMF | 0.094±0.001 | DSM 1.047±0.924; FPE 0.104±0.004 | Significantly outperforms DSM and slightly better than FPE using learned energy for Langevin |
| Chignolin Molecular | Training Time | 18.9 GPU h | FPE 49.6 GPU h | More lightweight than FPE regularization, approx. 2.6x faster |
| ALDP Solvation Free Energy | Estimated | 29.02±0.41 | Lbase 27.30±0.45; Ref 29.43±0.01 | TI estimates are closer to reference with DiffCLF |

### Ablation Study
The paper lacks traditional "remove module A/B" style ablations common in vision models but provides multiple alternative training objectives and analyses of the number of levels. The core comparison is between DSM-only, DSM+CtSM, and DSM+DiffCLF. DSM handles the score, CtSM attempts to supplement information using time-scores, while DiffCLF directly constrains energy using classification posteriors.

| Configuration | Key Metric | Description |
|---------------|------------|-------------|
| DSM only | MOG-40 128D $L_{clf}=383.53±35.99$ | Scores and generative metrics can be decent, but energy self-consistency is worst, reflecting mode weight blindness. |
| DSM + CtSM | MOG-40 128D $L_{clf}=20.86±4.93$ | Time-score constraints help, but still rely on local derivatives and fail to fully solve global proportions. |
| DSM + DiffCLF | MOG-40 128D $L_{clf}=4.40±1.00$ | Classification objective directly compares energy heights, best at recovering correct log-density. |
| FPE regularization | ALDP train time 8.1 GPU h | Learns molecular energy well but requires backpropping time derivatives, scores, and Laplacians; high cost. |
| DiffCLF | ALDP train time 5.6 GPU h | Maintains Langevin JS/PMF close to FPE while training faster. |

### Key Findings
- DiffCLF is more sensitive to "energy correctness" than "sample appearance." In MOG experiments, DSM's MMD isn't always poor, but the classification loss and log-density scatter show it fails to learn density heights correctly.
- In mode composition and BG tasks, DiffCLF's advantage is magnified because these tasks directly utilize learned marginal energy; DSM's errors in mode weights lead to significantly skewed proportions in composed distributions.
- Molecular system results show DiffCLF is more than a toy loss: on ALDP and Chignolin, statistics from Langevin dynamics run with $U^\theta_{t=0}$ improve significantly while training faster than FPE regularization.

## Highlights & Insights
- The most ingenious aspect is treating "diffusion time" as a classification label. Diffusion training already samples different noise levels; DiffCLF simply asks the model to identify these levels, indirectly obtaining energy proportion supervision.
- It clearly elucidates mode blindness: scores are local slopes and cannot reliably encode the mass ratio of disconnected modes. Classification posteriors depend on ratios of density values, thus naturally "seeing" mixture weights.
- The method has low intrusion on existing score-based pipelines. it doesn't require abandoning DSM or sampling from the EBM inner loop—it just adds several time-conditional forward evaluations, making it easy to embed in diffusion models, stochastic interpolants, or discrete CTMCs.
- Downstream experiments are well-chosen. Model composition, Boltzmann Generators, and free energy differences are tasks that rely more on energy values than just "how good the images look," proving that this work isn't just a minor metric patch.

## Limitations & Future Work
- The scale of experiments is currently small-to-medium. The authors admit that large-scale SMC composition for image modeling remains a future direction; current evidence mainly comes from MOG, molecular systems, and toy compositions.
- While cheaper than MLE, the multiclass version still requires $N+1$ network evaluations. For large models, a more systematic engineering trade-off between the number of time levels, batch size, and training throughput is needed.
- The method relies on sufficient differentiability/classifiability between different temporal marginals. If time sampling is too dense, the binary limit degenerates into local constraints (like time-score matching), which might re-introduce mode blindness.
- The learned energy is still unnormalized. While many downstream tasks accept this, tasks requiring an exact normalizing constant will need additional estimation or calibration.

## Related Work & Insights
- **vs DSM / score matching**: DSM learns scores—cheap but blind to the relative weights of disconnected modes. DiffCLF supplements this by comparing energy values through classification posteriors, fixing global density proportions alongside DSM's local score supervision.
- **vs Conditional Time Score Matching**: CtSM uses computable conditional time-scores to constrain $\partial_t \log p_t$, acting more as a derivative-level patch. DiffCLF directly performs density-ratio classification across finite time levels, making it more direct for mode weighting.
- **vs Fokker-Planck regularization**: FPE regularization is based on marginal density PDEs, which is theoretically strong but computationally heavy. DiffCLF's supervision comes from standard cross-entropy and achieves similar or better statistical quality in molecular systems while being faster.
- **Insights for Future Work**: Many generative model training processes have natural "auxiliary labels" like time, noise level, temperature, or annealing stages. Converting these labels into density ratio classification could be a universal route for completing the energy scale of score models.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Uses diffusion time-level classification for EBM energy learning, capturing the root cause of score matching's mode blindness.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers synthetic data, molecules, composition, BG, and free energy, though large-scale image experiments are still lacking.
- Writing Quality: ⭐⭐⭐⭐ Theoretical motivation and connection to existing work are clear; experimental sections are dense and require back-and-forth comparison.
- Value: ⭐⭐⭐⭐⭐ Highly valuable for diffusion applications requiring energy values rather than just samples, particularly for composite generation and physical sampling scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] $f$-Trajectory Balance: A Loss Family for Tuning GFlowNets, Generative Models, and LLMs with Off- and On-Policy Data](f-trajectory_balance_a_loss_family_for_tuning_gflownets_generative_models_and_ll.md)
- [\[NeurIPS 2025\] Latent Zoning Network: A Unified Principle for Generative Modeling, Representation Learning, and Classification](../../NeurIPS2025/image_generation/latent_zoning_network_a_unified_principle_for_generative_modeling_representation.md)
- [\[AAAI 2026\] Symmetrical Flow Matching: Unified Image Generation, Segmentation, and Classification with Score-Based Generative Models](../../AAAI2026/image_generation/symmetrical_flow_matching_unified_image_generation_segmentation_and_classificati.md)
- [\[ICML 2026\] A Kinetic Energy Perspective of Flow Matching](a_kinetic_energy_perspective_of_flow_matching.md)
- [\[CVPR 2026\] Transition Models: Rethinking the Generative Learning Objective](../../CVPR2026/image_generation/transition_models_rethinking_the_generative_learning_objective.md)

</div>

<!-- RELATED:END -->
