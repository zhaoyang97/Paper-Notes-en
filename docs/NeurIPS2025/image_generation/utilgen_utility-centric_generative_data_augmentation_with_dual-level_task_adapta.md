---
title: >-
  [Paper Note] UtilGen: Utility-Centric Generative Data Augmentation with Dual-Level Task Adaptation
description: >-
  [NeurIPS 2025][Image Generation][Data Augmentation] This paper proposes UtilGen, a utility-centric generative data augmentation framework that evaluates the downstream task utility of synthetic data via a meta-learning weight network, and employs a dual-level optimization strategy—model-level DPO and instance-level (prompt + noise) optimization—to adaptively generate high-utility synthetic training data, achieving an average improvement of 3.87% across 8 benchmarks.
tags:
  - NeurIPS 2025
  - Image Generation
  - Data Augmentation
  - Task Utility
  - Diffusion Models
  - Bi-level Optimization
  - DPO
date: 2026-05-08
content_hash: c1745636ab575e3d
---

# UtilGen: Utility-Centric Generative Data Augmentation with Dual-Level Task Adaptation

**Conference**: NeurIPS 2025
**arXiv**: [2510.24262](https://arxiv.org/abs/2510.24262)
**Code**: Not available
**Area**: Diffusion Models / Image Generation / Data Augmentation
**Keywords**: Data Augmentation, Task Utility, Diffusion Models, Bi-level Optimization, DPO

## TL;DR

This paper proposes UtilGen, a utility-centric generative data augmentation framework that evaluates the downstream task utility of synthetic data via a meta-learning weight network, and employs a dual-level optimization strategy—model-level DPO and instance-level (prompt + noise) optimization—to adaptively generate high-utility synthetic training data, achieving an average improvement of 3.87% across 8 benchmarks.

## Background & Motivation

Existing generative data augmentation methods primarily focus on optimizing the **intrinsic attributes** of synthetic data—fidelity and diversity—for example, by aligning with the real data distribution through LoRA fine-tuning or enhancing data variation through diversified prompts. However, these methods overlook a critical issue: different downstream tasks and model architectures impose fundamentally different requirements on training data. Within the same category, a sample beneficial for one task may be entirely uninformative for another.

Prior methods lack mechanisms to adjust the data generation process based on downstream task feedback, resulting in synthetic data of high visual quality yet limited task-specific contribution. This motivates the authors to ask: can the paradigm shift from **"visual quality-centric"** to **"task utility-centric"** data augmentation?

The core challenges are: (1) how to efficiently evaluate the task utility of synthetic data without full training-evaluation cycles; and (2) how to systematically improve the task utility of synthetic data.

## Method

### Overall Architecture

UtilGen comprises three core modules: (1) Task-Oriented Data Valuation (TODV), which quantifies per-sample utility via a meta-learning weight network; (2) Model-Level Capability Optimization (MLCO), which fine-tunes the diffusion model using DPO to align with downstream task preferences; and (3) Instance-Level Policy Optimization (ILPO), which jointly optimizes prompt embeddings and initial noise to maximize per-sample utility.

### Key Designs

1. **Task-Oriented Data Valuation (TODV)**: A single-hidden-layer MLP weight network $\mathcal{W}_\phi$ is used to predict the utility weight of each sample: $\omega_i = \mathcal{W}_\phi(\mathcal{L}(f(x_i;\theta), y_i))$. It is trained via bi-level optimization: the inner level trains the classifier $\theta$ using weighted loss, while the outer level minimizes validation loss to update $\phi$. The trained weight network can directly predict utility scores for newly generated samples, avoiding expensive retraining. The core motivation is to leverage meta-learning to establish a fast evaluation channel from "synthetic data quality" to "downstream performance."

2. **Model-Level Capability Optimization (MLCO)**: The weight network is used to partition generated samples into high-utility and low-utility pairs, constructing a preference dataset $\mathcal{D}_{\text{preference}}$, which is then used to fine-tune the diffusion model's U-Net via Diffusion DPO. The DPO loss is: $\mathcal{L}_{\text{DPO}}(\psi) = -\mathbb{E}[\log\sigma(-\beta T\omega(\lambda_t)(\Delta\mathcal{L}_w - \Delta\mathcal{L}_l))]$, where $\Delta\mathcal{L}_w$ and $\Delta\mathcal{L}_l$ denote the noise prediction discrepancies for high-utility and low-utility samples, respectively. Iterative DPO fine-tuning progressively aligns the generative distribution of the diffusion model with downstream task requirements.

3. **Instance-Level Policy Optimization (ILPO)**: Fine-grained optimization is performed at each generation step. (a) **Prompt Embedding Optimization**: Building on class identifiers learned via textual inversion, prompt embeddings are gradient-optimized to maximize the utility score predicted by the weight network, with CLIP regularization applied to prevent semantic drift: $p^* = \arg\max_p[\mathcal{W}_\phi(\mathcal{L}(f(g(p,\epsilon_T);\theta),y)) - \lambda L_{\text{CLIP}}]$. (b) **Noise Optimization**: Leveraging the asymmetry of CFG scale in the DDIM forward and inverse processes, semantic information from high-utility samples is implicitly injected into the initial noise: $\epsilon'_t = \text{DDIM-Inv}_{\omega_w}(\text{DDIM}_{\omega_l}(\epsilon_t, p^*))$, where $\omega_l > \omega_w$ enables semantic injection.

### Loss & Training

The overall training pipeline is iterative: TODV is first trained to obtain a utility evaluator → MLCO iteratively fine-tunes the diffusion model → ILPO optimizes prompts and noise at each generation round. The data valuation stage employs a mixed real-and-synthetic dataset for bi-level optimization. The optimization signals for both MLCO and ILPO are derived from the trained weight network $\mathcal{W}_\phi$.

## Key Experimental Results

### Main Results

Classification evaluation is conducted with ResNet-50 on 8 benchmark datasets, with synthetic data volume set to 5× the real data:

| Setting | Method | IN-1k-S | IN-100-S | Cal101 | Flowers | Avg. |
|---------|--------|---------|----------|--------|---------|------|
| Synthetic only | DataDream (Prev. SOTA) | 30.35 | 35.48 | 23.61 | 65.15 | 33.30 |
| Synthetic only | **UtilGen** | **33.72** | **40.94** | **29.31** | **67.43** | **37.17** |
| Synthetic + Real | DataDream | 52.16 | 57.68 | 73.38 | 89.60 | 58.67 |
| Synthetic + Real | **UtilGen** | **54.56** | **61.54** | **75.62** | **93.62** | **62.04** |

Average gain: synthetic-only +3.87%, joint training +3.37%.

### Ablation Study

| Configuration | Accuracy (%) | Note |
|--------------|-------------|------|
| Baseline (SD v2.1) | 27.96 | No optimization |
| +MLCO | 28.68 | Model-level optimization |
| +Prompt Opt | 36.42 | Largest contribution from prompt embedding optimization |
| +Noise Opt | 37.96 | Independent gain from noise optimization |
| +MLCO+Prompt+Noise (Full) | **40.94** | Complementary; total gain +12.98% |

### Key Findings

- UtilGen is the first method in which a ResNet-50 trained on only 3× synthetic data surpasses real-data training on multiple benchmarks.
- Data influence analysis shows that UtilGen generates a significantly higher proportion of positively influential samples compared to SD v2.1.
- Synthetic data exhibits cross-architecture reusability: data generated using a weight network trained for ResNet-50 remains effective for WideResNet and CLIP.
- Cross-architecture generalization is maintained across ResNeXt-50, WideResNet-50, and MobileNetV2.

## Highlights & Insights

- The paradigm shift from optimizing visual attributes to optimizing task utility represents a significant conceptual contribution to the data augmentation literature.
- The dual-purpose design of the meta-learning weight network is elegant: it simultaneously reweights samples during classifier training and provides utility evaluation signals for data generation.
- The noise optimization in ILPO draws on the observation of CFG scale asymmetry in the DDIM inverse process, achieving semantic injection at zero additional training cost.

## Limitations & Future Work

- The utility evaluation of the weight network depends on the current state of the classifier; estimates may be inaccurate when the classifier is weak.
- TODV requires pre-training the weight network, increasing overall pipeline complexity and the burden of multi-stage hyperparameter tuning.
- The framework has only been validated on classification tasks; utility evaluation mechanisms for more complex tasks such as detection and segmentation may require redesign.
- Cost analysis indicates that generating 50,000 images requires only 4.7 hours / $100, but the computational overhead of iterative DPO fine-tuning is not discussed in detail.
- Textual inversion uses 16-shot real images (consistent with DataDream); performance under extremely low-resource settings (e.g., 1–2 shots) remains to be verified.
- Noise optimization relies on the CFG scale asymmetry specific to DDIM; applicability to non-DDIM samplers is uncertain.

## Related Work & Insights

- Unlike Data Shapley and related data valuation methods, this work avoids expensive retraining by using a lightweight weight network for online valuation.
- The GAP method also employs downstream model feedback but relies solely on adversarial loss (maximizing classifier loss); the utility weights in UtilGen provide a more fine-grained, instance-level usefulness signal.
- The core distinction from DataDream lies in the optimization objective: DataDream fine-tunes via LoRA to improve fidelity and align with the real data distribution, whereas UtilGen fine-tunes via DPO to align with downstream task preferences—fundamentally different optimization targets.
- This work can inspire future extensions of utility-oriented thinking to other generative tasks, such as text augmentation and 3D data augmentation.
- Cost-benefit analysis shows that the cost of generating synthetic data is substantially lower than manual annotation ($100 vs. $800 for comparable scale) while achieving superior quality.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The utility-centric paradigm shift is highly significant; first work to drive augmentation with DPO + utility feedback.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 8 datasets, multiple architectures, complete ablations, and in-depth influence analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with thorough method descriptions.
- **Value**: ⭐⭐⭐⭐⭐ Establishes a new paradigm for data augmentation with substantial empirical gains and practical applicability.

<!-- NeurIPS 2025 | image_generation | 2510.24262 -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Non-Asymptotic Analysis of Data Augmentation for Precision Matrix Estimation](non-asymptotic_analysis_of_data_augmentation_for_precision_matrix_estimation.md)
- [\[NeurIPS 2025\] Large-Scale Training Data Attribution for Music Generative Models via Unlearning](large-scale_training_data_attribution_for_music_generative_models_via_unlearning.md)
- [\[NeurIPS 2025\] Dual Data Alignment Makes AI-Generated Image Detector Easier Generalizable](dual_data_alignment_makes_ai-generated_image_detector_easier_generalizable.md)
- [\[NeurIPS 2025\] Increasing the Utility of Synthetic Images through Chamfer Guidance](increasing_the_utility_of_synthetic_images_through_chamfer_guidance.md)
- [\[ICLR 2026\] Pseudo-Nonlinear Data Augmentation: A Constrained Energy Minimization Viewpoint](../../ICLR2026/image_generation/pseudo-nonlinear_data_augmentation_a_constrained_energy_minimization_viewpoint.md)

</div>

<!-- RELATED:END -->
