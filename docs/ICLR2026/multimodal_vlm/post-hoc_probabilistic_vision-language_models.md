---
title: >-
  [Paper Note] Post-hoc Probabilistic Vision-Language Models
description: >-
  [ICLR 2026][Multimodal VLM][vision-language models] A training-free post-hoc uncertainty estimation method is proposed that applies Laplace approximation to the last few layers of VLMs such as CLIP and SigLIP…
tags:
  - "ICLR 2026"
  - "Multimodal VLM"
  - "vision-language models"
  - "uncertainty quantification"
  - "Bayesian inference"
  - "Laplace approximation"
  - "active learning"
date: 2026-05-08
content_hash: 89b05f953fbeb6db
---

# Post-hoc Probabilistic Vision-Language Models

**Conference**: ICLR 2026
**arXiv**: [2412.06014](https://arxiv.org/abs/2412.06014)  
**Code**: Available (Project page)  
**Area**: Multimodal VLM / Uncertainty Quantification
**Keywords**: vision-language models, uncertainty quantification, Bayesian inference, Laplace approximation, active learning

## TL;DR

A training-free post-hoc uncertainty estimation method is proposed that applies Laplace approximation to the last few layers of VLMs such as CLIP and SigLIP, and analytically derives uncertainty over cosine similarity. The method achieves substantial improvements over baselines in both uncertainty quantification and active learning.

## Background & Motivation

### State of the Field

Vision-language models (VLMs) such as CLIP and SigLIP have achieved remarkable success across classification, retrieval, and generation tasks. The core operation of these models is to map images and text into a shared latent space and evaluate their correspondence via **cosine similarity**.

However, this deterministic mapping has a fundamental limitation: it **cannot capture uncertainty over concepts**. Specifically:

**Domain shift**: When a VLM is applied to a downstream task, discrepancies between the training domain and the target domain render predictions unreliable, yet the model cannot express its degree of uncertainty.

**Out-of-distribution (OOD) samples**: For unseen images or concepts, VLMs still produce a single deterministic embedding, making it impossible to distinguish between confident correct predictions and speculative ones.

**Safety-critical applications**: In domains such as medical diagnosis and autonomous driving, uncertainty estimation is essential for reliable decision-making.

Existing uncertainty estimation methods typically require:

### Limitations of Prior Work

Retraining the entire model (e.g., Monte Carlo Dropout, ensemble methods).

### Root Cause

Modifying the model architecture (e.g., probabilistic embedding methods).

### Starting Point

Substantial additional computational resources.

These approaches are impractical for large-scale VLMs—models such as CLIP are trained on billions of image-text pairs, making retraining prohibitively expensive. Consequently, a **training-free** post-hoc uncertainty estimation method offers considerable practical value.

## Method

### Overall Architecture

BayesVLM (the name given to the proposed method) operates as follows:

1. All VLM parameters are kept fixed.
2. A Bayesian posterior approximation is constructed for only the last few layers.
3. Laplace approximation is used to obtain a Gaussian posterior over the parameters.
4. Uncertainty over cosine similarity under the posterior distribution is derived analytically.

### Key Designs

1. **Laplace Posterior Approximation**:

    - The weights of the last few VLM layers are treated as random variables.
    - Laplace approximation is used to construct a Gaussian posterior over the weights: $p(\theta | D) \approx \mathcal{N}(\theta^*, \Sigma)$
    - Here $\theta^*$ denotes the pretrained weights (MAP estimate), and $\Sigma$ is approximated by the inverse of the Fisher information matrix.
    - **Design Motivation**: Laplace approximation expands around the MAP estimate, naturally leveraging the already-trained model parameters without requiring retraining.

2. **Analytic Derivation of Uncertainty over Cosine Similarity**:

    - In conventional VLMs, cosine similarity $s = \frac{f_I \cdot f_T}{\|f_I\| \|f_T\|}$ is deterministic.
    - When embeddings become random variables, cosine similarity also becomes a random variable.
    - The paper analytically derives the distributional properties (mean and variance) of cosine similarity, avoiding the computational overhead of Monte Carlo sampling.
    - **Design Motivation**: A closed-form solution is not only computationally efficient but also eliminates approximation errors introduced by sampling.

3. **Restricting to the Last Few Layers**:

    - Bayesian treatment is applied only to the last few layers rather than the entire VLM.
    - This substantially reduces computational cost—the Fisher information matrix covers only the parameters of the last few layers rather than the full model.
    - In practice, the last few layers have the greatest influence on downstream task adaptation.
    - **Design Motivation**: To balance computational cost against the effectiveness of uncertainty estimation; uncertainty in the feature extractor is approximated as zero in the posterior.

4. **Training-Free Property**:

    - The method requires only a lightweight calibration step and no fine-tuning of model parameters.
    - Calibration involves computing the Fisher information matrix (or its approximation) and can be performed on a small dataset.
    - Model parameters remain entirely unchanged, preserving original performance.
    - **Design Motivation**: To enable plug-and-play application to any pretrained VLM, maximizing practical usability.

### Loss & Training

- **No additional training required**: The method is entirely post-hoc.
- **Calibration**: A diagonal or Kronecker-factored approximation of the Fisher information matrix is computed using a small dataset.
- **Inference pipeline**: Forward pass to obtain embeddings → uncertainty computation via Laplace approximation → output of mean prediction and uncertainty estimate.

## Key Experimental Results

### Main Results

The method is validated in two primary application settings:

**Uncertainty Quantification**

| Setting | Metric | BayesVLM | Deterministic Baseline | Advantage |
|---------|--------|----------|----------------------|-----------|
| In-distribution | ECE | Significant improvement | Overconfident | Better calibration |
| OOD Detection | AUROC | Notable gain | No uncertainty | OOD identification |
| Domain Shift | Prediction reliability | More robust | Performance degrades | Reliable uncertainty signal |

**Active Learning**

| Dataset | Metric | BayesVLM | Random Sampling | Other Baselines |
|---------|--------|----------|----------------|-----------------|
| Multiple downstream tasks | Sample efficiency | **Highest** | Baseline | Moderate |
| Limited annotation budget | Accuracy | **Best** | Poor | Second best |

### Ablation Study

| Configuration | Key Metric | Remarks |
|---------------|-----------|---------|
| Number of layers processed | Uncertainty quality | Good results with only the last 1–2 layers |
| Fisher matrix approximation | Calibration quality | Diagonal approximation is sufficient; Kronecker factorization yields better results |
| Different VLM backbones | Generality | Effective on both CLIP and SigLIP |

### Key Findings

- **Well-calibrated**: The uncertainty estimates provided by BayesVLM are well-calibrated—when the model expresses low confidence, errors are indeed more frequent.
- **Interpretability**: Uncertainty estimates are intuitively interpretable—ambiguous or out-of-distribution samples receive higher uncertainty.
- **Active learning efficiency**: Uncertainty-based sample selection substantially outperforms random sampling, with particularly pronounced benefits under limited annotation budgets.
- **Original performance preserved**: As a post-hoc method, model parameters are unmodified and original classification/retrieval performance is unaffected.
- **Computational efficiency**: Analytic derivation avoids Monte Carlo sampling, incurring minimal inference overhead.

## Highlights & Insights

- **Well-motivated problem selection**: Uncertainty estimation for VLMs is an underexplored yet critically important problem, particularly in safety-critical applications.
- **Elegant method design**: No retraining, no architectural modification, and no significant additional computation—a true plug-and-play solution.
- **Theory–practicality balance**: Laplace approximation rests on solid theoretical foundations, while analytic derivation ensures computational efficiency.
- **Probabilistic treatment of cosine similarity**: Converting deterministic cosine similarity into an uncertainty-aware random variable constitutes an elegant theoretical contribution.
- **Diverse downstream applications**: The method demonstrates value in both uncertainty quantification and active learning, two practical scenarios.

## Limitations & Future Work

- **Approximation quality**: Laplace approximation assumes a Gaussian posterior, which may be insufficiently accurate in high-dimensional spaces.
- **Restriction to the last few layers**: Uncertainty propagation from deeper VLM layers is neglected, potentially underestimating total uncertainty.
- **Fisher matrix computation**: For very large models, even a diagonal approximation may incur non-trivial computational cost.
- **Limited evaluation benchmarks**: Uncertainty estimation lacks standardized evaluation protocols, and performance may vary considerably across datasets.
- **Scope limited to classification/retrieval**: Applicability to generative VLMs (e.g., LLaVA, GPT-4V) has not been verified.
- **Autoregressive generation**: The method is suited to dual-encoder architectures such as CLIP; extension to autoregressive VLM architectures requires further work.

## Related Work & Insights

- **CLIP** (Radford et al., 2021): The most representative deterministic VLM and the primary target of the proposed method.
- **SigLIP** (Zhai et al., 2023): An improved variant of CLIP using sigmoid loss, equally compatible with the proposed method.
- **Laplace Approximation**: A classical Bayesian approximation technique that has recently regained attention in deep learning (Laplace Redux, Daxberger et al., 2021).
- **Monte Carlo Dropout** (Gal & Ghahramani, 2016): Approximates Bayesian inference via dropout but requires multiple forward passes.
- **Probabilistic Embeddings** (Kirchhof et al., 2023): Models embeddings as distributions rather than point estimates, but requires retraining.
- **Active Learning** (Settles, 2009): Uncertainty-based sample selection is a classical strategy in active learning.

**Insights**: Post-hoc methods represent a pragmatic path for introducing Bayesian uncertainty into large-scale pretrained models. This paradigm can be extended to uncertainty estimation in other pretrained models (e.g., LLMs, audio models). The probabilistic treatment of cosine similarity may inspire new uncertainty-aware retrieval and matching algorithms.

## Rating

- Novelty: ⭐⭐⭐⭐ — Post-hoc Laplace approximation is not new, but the analytic derivation over cosine similarity in VLMs is a novel contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Validation across two settings (uncertainty quantification and active learning) and multiple VLM backbones.
- Writing Quality: ⭐⭐⭐⭐ — Theoretical derivations are clear and the method description is concise and accessible.
- Value: ⭐⭐⭐⭐ — Addresses a practical need in VLM deployment with broad prospects for safety-critical applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Fine-Grained Post-Training Quantization for Large Vision Language Models with Quantization-Aware Integrated Gradients](../../CVPR2026/multimodal_vlm/fine-grained_post-training_quantization_for_large_vision_language_models_with_qu.md)
- [\[ICLR 2026\] GTR-Bench: Evaluating Geo-Temporal Reasoning in Vision-Language Models](gtr-bench_evaluating_geo-temporal_reasoning_in_vision-language_mod.md)
- [\[ICLR 2026\] Mixing Importance with Diversity: Joint Optimization for KV Cache Compression in Large Vision-Language Models](mixing_importance_with_diversity_joint_optimization_for_kv_cache_compression_in_.md)
- [\[ICLR 2026\] Enhanced Continual Learning of Vision-Language Models with Model Fusion](enhanced_continual_learning_of_vision-language_models_with_model_fusion.md)
- [\[ICLR 2026\] OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models](omnispatial_towards_comprehensive_spatial_reasoning_benchmark_for_vision_languag.md)

</div>

<!-- RELATED:END -->
