---
title: >-
  [Paper Note] TAVAE: A VAE with Adaptable Priors Explains Contextual Modulation in the Visual Cortex
description: >-
  [ICLR 2026][Image Generation][Variational Autoencoder] This paper extends the VAE formalism to propose the Task-Amortized VAE (TAVAE), which explains contextual modulation in the primary visual cortex (V1) by learning task-specific priors over an already-learned representation. The framework accounts for bimodal population responses observed when test stimuli deviate from training stimuli in an orientation discrimination task.
tags:
  - ICLR 2026
  - Image Generation
  - Variational Autoencoder
  - Task Priors
  - V1
  - Contextual Modulation
  - Probabilistic Inference
date: 2026-05-08
content_hash: 20fc16ff7038c6dc
---

# TAVAE: A VAE with Adaptable Priors Explains Contextual Modulation in the Visual Cortex

**Conference**: ICLR 2026
**arXiv**: [2602.11956](https://arxiv.org/abs/2602.11956)
**Code**: [https://github.com/CSNLWigner/mouse-V1-task-priors](https://github.com/CSNLWigner/mouse-V1-task-priors)
**Area**: Computational Neuroscience / Visual Cortex Modeling
**Keywords**: Variational Autoencoder, Task Priors, V1, Contextual Modulation, Probabilistic Inference

## TL;DR
This paper extends the VAE formalism to propose the Task-Amortized VAE (TAVAE), which explains contextual modulation in the primary visual cortex (V1) by learning task-specific priors over an already-learned representation. The framework accounts for bimodal population responses observed when test stimuli deviate from training stimuli in an orientation discrimination task.

## Background & Motivation

**State of the Field**: Deep learning models—both discriminative and generative—have been successfully applied to model neural responses in the visual system. Population activity in V1 has been interpreted as a representation of the posterior over latent variables in a generative model, and priors learned from natural image statistics have been shown to explain several V1 properties.

**Limitations of Prior Work**: V1 responses are strongly influenced not only by the stimulus itself, but also by non-stimulus factors such as task context. Recent studies have identified systematic task-specific biases in V1 activity. However, existing models cannot account for these biases—adapting a standard VAE to a new task requires retraining from scratch, which is both data-inefficient and biologically implausible.

**Root Cause**: Task learning should reuse previously acquired visual representations without retraining the entire network, yet must flexibly incorporate task-specific priors to modulate inference.

**Paper Goals**: To develop a VAE framework that flexibly acquires task priors while reusing learned representations, and to use this framework to explain task-related contextual modulation in mouse V1.

**Starting Point**: The likelihood (generative model) and recognition model of the VAE are held fixed; the variational posterior is reweighted via Bayes' rule using a new task prior: $q_T(\mathbf{z}|\mathbf{x}) \propto q(\mathbf{z}|\mathbf{x}) \cdot p_T(\mathbf{z}) / p_0(\mathbf{z})$.

**Core Idea**: Task-dependent response biases in V1—including bimodal population responses under uncertainty—are explained solely by learning task-specific priors, without retraining the VAE.

## Method

### Overall Architecture
A standard VAE (linear generative model + Laplace prior + GSM scaling) is first trained on natural images to establish a V1 model. The TAVAE formalism then learns only the variance parameters of the task prior $p_T(\mathbf{z})$ (maintaining a Laplace family distribution), applying Bayesian reweighting to the existing posterior. The self-consistent equations for task learning converge in as few as 5 iterations.

### Key Designs

1. **Bayesian Reweighting of Task Priors**:

    - Function: Produce task-adapted posteriors through prior modification, without retraining the VAE.
    - Mechanism: $q_T(\mathbf{z}|\mathbf{x}) \propto q(\mathbf{z}|\mathbf{x}) \cdot p_T(\mathbf{z}) / p_0(\mathbf{z})$. The prior follows a Laplace distribution; only the per-latent variance $\sigma_{T,i}$ is learned.
    - Design Motivation: Biologically, this corresponds to higher cortical areas transmitting prior information to V1 via top-down connections.

2. **Self-Consistent Equation Solving**:

    - Function: Efficient learning of task prior parameters.
    - Mechanism: $\sigma_{T,i} = \frac{1}{n} \sum_{\mathbf{x}} \mathbb{E}_{q_T}[|z_i|]$ is a self-consistent equation solved iteratively (converges in 5 steps).
    - Design Motivation: Avoids full ELBO optimization, enabling data-efficient task learning.

3. **Linear Generative Model + GSM Scaling**:

    - Function: Construct a VAE whose properties align with those of V1.
    - Mechanism: Linear generative model $p(\mathbf{x}|\mathbf{z},s) = \mathcal{N}(\mathbf{x}; e^s \mathbf{A}\mathbf{z}, \sigma^2 \mathbf{I})$; Laplace prior encourages learning of Gabor filters; GSM scaling improves inference at low contrast.
    - Design Motivation: Ensures a meaningful correspondence between latent variables and V1 neurons.

### Loss & Training

During the VAE stage, the standard ELBO (negative log-likelihood + KL divergence) is optimized. During the TAVAE stage, only the log-likelihood of the task prior (Eq. 4) is optimized, reduced to iterative self-consistent equation solving.

## Key Experimental Results

### Main Results

Calcium imaging recordings from 10 mice (15,027 neurons in V1) compared against TAVAE model predictions:

| Model Configuration | Population Response Correlation $r$ | Contextual Modulation Correlation |
|---|---|---|
| **TAVAE (45°, 90°)** | **0.78±0.02** | **0.58±0.09** |
| VAE (no task prior) | 0.53±0.12 | — |
| TAVAE (45°, 135°) | 0.54±0.10 | 0.32±0.17 |
| TAVAE (eager adapter) | 0.53±0.11 | -0.10±0.23 |

### Ablation Study

| Analysis | Key Finding |
|---|---|
| Task learning effect | Population response sharpening + reduced baseline activity—qualitatively consistent with differences between trained and naive mice |
| OOD stimulus bimodality | When test stimuli deviate from the training distribution, population responses become bimodal (peaks displaced from the stimulus orientation)—consistent between model and experiment |
| Within-session prior update trajectory (Day 2) | Asymmetry of bimodality changes between early and late trials within a session, consistent with model predictions of gradual prior updating |
| GSM scaling ablation | Model remains functional without scaling, but fit quality is slightly reduced |
| Contrast dependence | Lower contrast increases uncertainty → more pronounced bimodality → better match to experimental data |

### Key Findings
- Task priors in TAVAE enhance responses of neurons whose preferred orientation matches the task and suppress responses of non-matching neurons, producing population response sharpening.
- The most striking finding: when the test stimulus orientation is mismatched with the prior, population responses exhibit a counterintuitive bimodal pattern—a trough at the stimulus orientation flanked by two peaks—a direct signature of uncertainty in probabilistic inference.
- Evidence for prior updating speed: trial-grouping analysis within Day 2 sessions shows the prior gradually shifts from (45°, 135°) toward (45°, 90°).
- The likelihood function and prior can be recovered from the mode locations of the population response; the inferred likelihood closely matches the population response of untrained animals.

## Highlights & Insights
- **Bimodality as a signature of uncertainty**: The posterior is unimodal in high-dimensional latent space, but projects to a bimodal distribution in the one-dimensional orientation space—revealing how multi-hypothesis uncertainty manifests at the population level, with profound implications for understanding neural coding.
- **Highly efficient task adaptation**: Only the prior variance parameters are learned (self-consistent equations converge in 5 iterations), with neither the recognition model nor the generative model modified—computationally efficient and biologically plausible as a mechanism of top-down prior modulation.
- **Closed-loop model–experiment interaction**: Rather than fitting neural data post hoc, the paper derives predictions from normative principles and validates them on 15,027 neurons—this prediction-first paradigm is more convincing than fit-then-interpret approaches.
- **Recovering generative model parameters from population responses**: Mode location shifts are used to infer likelihood width and prior strength, providing a methodology for reverse-engineering probabilistic inference parameters from neural data.

## Limitations & Future Work
- The linear generative model is a simplification—V1 computations likely involve nonlinear components.
- The task prior assumes residual independence and zero-mean Laplace distributions, limiting expressiveness (cross-feature correlations in the prior cannot be modeled).
- The temporal resolution of calcium imaging is insufficient to distinguish between sampling-based and MAP-estimate representations of the posterior.
- Validation is restricted to a simple orientation discrimination task; generalization to more complex visual tasks remains untested.
- The assumption that the prior ceases updating after Day 2 rests primarily on behavioral evidence (trends in false alarm rates).

## Related Work & Insights
- **vs. standard VAE models**: Standard VAEs incorporate only natural image priors and cannot explain task-induced response biases. TAVAE resolves this by introducing task priors that modulate inference without altering the learned representation.
- **vs. Csikor et al. (2025)**: That work uses a hierarchical VAE to learn contextual priors from natural images. TAVAE extends the framework to task-specific priors; the two approaches are complementary.
- **Implications for VLM/multimodal research**: The TAVAE principle of "preserving representations while flexibly adapting priors" is analogous to parameter-efficient fine-tuning in multimodal models—modulating outputs through priors or prompts without modifying the backbone.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The theoretical formalism is elegant and original; the prediction and validation of bimodal responses are particularly striking.
- Experimental Thoroughness: ⭐⭐⭐⭐ Large-scale recording data from 15K neurons combined with comparisons across multiple model variants.
- Writing Quality: ⭐⭐⭐⭐ Theory and experiments are tightly integrated, though the paper presupposes familiarity with neuroscience background.
- Value: ⭐⭐⭐⭐⭐ A significant advance for computational neuroscience, establishing a new paradigm for applying VAEs to neural systems.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] ViHOI: Human-Object Interaction Synthesis with Visual Priors](../../CVPR2026/image_generation/vihoi_human-object_interaction_synthesis_with_visual_priors.md)
- [\[ICLR 2026\] Eliminating VAE for Fast and High-Resolution Generative Detail Restoration](eliminating_vae_for_fast_and_high-resolution_generative_detail_restoration.md)
- [\[ICLR 2026\] COSMO-INR: Complex Sinusoidal Modulation for Implicit Neural Representations](cosmo-inr_complex_sinusoidal_modulation_for_implicit_neural_representations.md)
- [\[CVPR 2026\] FRAMER: Frequency-Aligned Self-Distillation with Adaptive Modulation Leveraging Diffusion Priors for Real-World Image Super-Resolution](../../CVPR2026/image_generation/framer_frequency-aligned_self-distillation_with_adaptive_modulation_leveraging_d.md)
- [\[ICLR 2026\] Mod-Adapter: Tuning-Free and Versatile Multi-concept Personalization via Modulation Adapter](mod-adapter_tuning-free_and_versatile_multi-concept_personalization_via_modulati.md)

<!-- RELATED:END -->
