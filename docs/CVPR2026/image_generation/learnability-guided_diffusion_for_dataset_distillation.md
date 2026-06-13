---
title: >-
  [Paper Note] Learnability-Guided Diffusion for Dataset Distillation
description: >-
  [CVPR 2026][Image Generation][Dataset Distillation] This paper proposes LGD, a learnability-driven incremental dataset distillation framework that constructs the distilled dataset in stages…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Dataset Distillation"
  - "Learnability Guidance"
  - "Diffusion Models"
  - "Incremental Synthesis"
  - "Redundancy Analysis"
date: 2026-05-08
content_hash: 5d8f64776d4c0587
---

# Learnability-Guided Diffusion for Dataset Distillation

**Conference**: CVPR 2026
**arXiv**: [2604.00519](https://arxiv.org/abs/2604.00519)  
**Code**: [https://jachansantiago.github.io/learnability-guided-distillation/](https://jachansantiago.github.io/learnability-guided-distillation/)  
**Area**: Image Generation / Dataset Distillation
**Keywords**: Dataset Distillation, Learnability Guidance, Diffusion Models, Incremental Synthesis, Redundancy Analysis

## TL;DR

This paper proposes LGD, a learnability-driven incremental dataset distillation framework that constructs the distilled dataset in stages, conditioning each stage on the current model state to generate complementary rather than redundant training samples. By injecting learnability-score gradients into diffusion sampling, LGD reduces the 80–90% inter-sample information redundancy observed in existing methods by 39.1%, achieving 60.1% accuracy at 50 IPC on ImageNet-1K and 87.2% at 100 IPC on ImageNette.

## Background & Motivation

**Background**: Dataset distillation aims to synthesize a compact surrogate dataset from a large collection such that models trained on it achieve performance comparable to training on the full data. Early methods optimize synthetic data at the pixel level by matching training trajectories (gradient matching, trajectory matching), but these approaches are computationally infeasible for high-resolution datasets. Recent methods (DiT, Minimax Diffusion, IGD, MGD3) leverage pretrained diffusion models to generate distilled data, substantially reducing computational cost.

**Limitations of Prior Work**: The central issue with existing methods is **severe sample redundancy**. When a 50-IPC distilled dataset is partitioned into five disjoint 10-IPC subsets, any single subset captures 80–90% of the training signal present in the remaining subsets. This redundancy stems from two sources: (1) methods that optimize for visual diversity (DiT) ignore training-signal similarity; (2) methods that match average training trajectories (IGD) drive all samples toward similar gradient profiles, producing "medium-intensity" gradients that are suboptimal at every training stage.

**Key Challenge**: Model training is inherently stage-dependent — early stages require strong gradients to establish coarse features, while later stages require fine-grained gradients for refinement. No single sample can simultaneously satisfy both requirements, yet optimizing for an "average" trajectory produces exactly such compromise samples rather than stage-specialized ones.

**Key Insight**: Distillation is reformulated as a **sequential learning problem** — given a distilled dataset and a model trained on it, the goal is to generate new samples that maximize marginal learning gain.

**Core Idea**: Learnability guidance — at each stage, the framework assesses what kinds of samples the current model can learn from (high loss under the current model but low loss under a reference model = learnable, not noise), and injects gradients of the learnability score into diffusion sampling to steer generation toward such samples.

## Method

### Overall Architecture

Incremental distillation loop: $\mathcal{D}$ is divided into $K$ increments $\mathcal{I}_1, \dots, \mathcal{I}_K$. Starting from seed data $\mathcal{D}_1$ (default: IGD's 10 IPC) → train model $\theta_0$ to convergence → generate candidates via learnability-guided diffusion → rank by learnability score and select the best samples to augment the dataset → train a new model on the expanded dataset → repeat. Each increment is conditioned on the current model state, naturally forming a curriculum from easy to hard.

### Key Designs

1. **Learnability Score and Regularized Objective**:
    - Function: Quantifies the learning value of a sample for the current model.
    - Mechanism: $\mathcal{S}(x,y) = \mathcal{L}(\theta_{i-1}, x, y) - \omega \cdot \mathcal{L}(\theta^*, x, y)$, where $\theta_{i-1}$ is the current model, $\theta^*$ is a reference model trained on the full dataset, and $\omega$ controls regularization strength.
    - Design Motivation: Maximizing the current model's loss alone produces degenerate samples (noise or OOD). The reference model term penalizes samples that the reference model also finds difficult — a high $\mathcal{S}$ indicates that the current model cannot handle a sample while the reference model can, representing a learnable knowledge gap.

2. **Learnability-Guided Diffusion Sampling (LGD)**:
    - Function: Steers the generation trajectory toward high-learnability regions during diffusion denoising.
    - Mechanism: The noise prediction is modified as $\tilde{\epsilon}_\phi(x_t, t, y) = \epsilon_\phi(x_t, t, y) + \lambda \cdot \rho_t \cdot \nabla_{x_t} \mathcal{S}(x_t, y)$, where $\rho_t = \sqrt{1-\bar{\alpha}_t} \frac{\|\epsilon_\phi\|}{\|\nabla_{x_t}\mathcal{S}\|}$ is a timestep-dependent scaling factor. Guidance is applied only at timesteps $t \in [10, 45]$ out of 50 total steps.
    - Design Motivation: Analogous to classifier guidance, but with the learnability score replacing class probability. The scaling factor normalizes guidance magnitude relative to the noise level. Applying guidance only at intermediate steps avoids disrupting global structure in early steps and over-constraining fine details in later steps.

3. **Learnability Sample Selection**:
    - Function: Selects the most informative samples from a pool of candidates.
    - Mechanism: For each sample slot to be filled, $\kappa = 3$ candidates are generated and ranked by learnability score; the highest-scoring candidate is retained. Selected samples are added to a memory buffer $\mathcal{M}^c$, and subsequent selections are performed in the context of the already-constructed dataset.
    - Design Motivation: Stochastic diffusion sampling may still produce low-learnability samples despite guidance. Sample selection complements guided sampling, and their combination ensures high-quality increments. Sequentially populating the memory buffer allows diversity guidance (cosine distance repulsion from existing samples) to operate naturally.

### Loss & Training

Incremental optimization objective: $\mathcal{I}_i^* = \arg\max_{\mathcal{I}} [\mathcal{L}(\theta_{i-1}, \mathcal{I}) - \mathcal{L}(\theta^*, \mathcal{I})]$. Key hyperparameters: guidance strength $\lambda = 15$, reference model weight $\omega = 0.5$, candidate multiplier $\kappa = 3$, deviation guidance $\gamma = 50$. Seed data uses IGD's 10 IPC, with 10 IPC added per stage. A soft-label protocol is adopted for ImageNet-1K and a hard-label protocol for ImageNette/Woof.

## Key Experimental Results

### Main Results

| Dataset | IPC | Model | Ours (LGD) | IGD | MGD3 | DiT | Gain (vs. IGD) |
|---------|-----|-------|-----------|-----|------|-----|----------------|
| ImageNette | 50 | ResNet-18 | 85.0% | 81.0% | 81.5% | 75.2% | +4.0% |
| ImageNette | 100 | ResNet-18 | 86.9% | 84.4% | 85.6% | 77.8% | +2.5% |
| ImageNette | 100 | ConvNet-6 | 87.2% | 84.5% | 86.5% | 78.2% | +2.7% |
| ImageWoof | 100 | ResNet-18 | 72.9% | 70.6% | 71.3% | 62.3% | +2.3% |
| ImageNet-1K | 50 | ResNet-18 | 60.1% | 59.8% | 60.2% | 52.9% | +0.3% |

### Ablation Study (Redundancy Analysis)

| Method | Cross-Increment Avg. Accuracy | Redundancy Level | Notes |
|--------|------------------------------|-----------------|-------|
| DiT | 94.7% | Highest | Any subset nearly fully substitutes for the others |
| IGD | 87.1% | High | Marginal improvement, still heavily overlapping |
| LGD (Ours) | 57.65% | Significantly reduced | Complementary increments; redundancy reduced by 39.1% |

### Key Findings
- Existing distillation methods exhibit severe redundancy: cross-accuracy among disjoint subsets of DiT reaches 91–98%.
- LGD incremental training exhibits larger per-stage loss spikes (average $\Delta = 0.20$ vs. DiT's $0.06$), indicating that newly added samples contain genuinely novel information.
- Accuracy under incremental training improves steadily: LGD rises from 64.1% at IPC 10 to 89.1% at IPC 100, whereas DiT plateaus after IPC 50.
- Learning dynamics visualization shows that LGD-generated samples are more "informative" (16.2%) and "harder" (2.6%), and exhibit the lowest JS divergence from the original training data distribution.
- Cross-architecture generalization is strong: 50-IPC data distilled with ResNet-AP-10 achieves 85.0% on ResNet-18.

## Highlights & Insights
- **Redundancy Diagnostic Framework**: The incremental partitioning and cross-evaluation protocol is not merely a synthesis tool but a general diagnostic instrument for analyzing information distribution in any distillation method.
- **Distillation as Curriculum Learning**: Each stage automatically induces a curriculum from easy to hard — difficulty is not predefined but determined by the evolving state of the model.
- **Elegance of the "Learnable Knowledge Gap" Concept**: High loss for the current model combined with low loss for the reference model indicates samples that are neither noise nor excessively difficult, but precisely at the learning boundary.
- **Empirical Findings of Broad Impact**: The 80–90% redundancy rate is striking and reveals a fundamental bottleneck in existing methods.

## Limitations & Future Work
- On ImageNet-1K, LGD (60.1%) and MGD3 (60.2%) are essentially on par; the advantage of the incremental strategy is less pronounced at large scale.
- A pretrained reference model $\theta^*$ (trained on the full dataset) is required, introducing additional preparation overhead.
- Computing learnability scores and gradient guidance at each stage requires model inference, making the approach slower than one-shot generation.
- The framework depends on IGD's 10-IPC seed data, so seed quality affects subsequent increments.
- The optimal configuration of increment count $K$ and per-increment IPC $N_i$ is dataset-dependent, and no adaptive determination method is provided.

## Related Work & Insights
- **vs. IGD (Influence-Guided Diffusion)**: IGD matches the gradients of average training trajectories, driving all samples toward similar gradient profiles. LGD conditions generation on the current model state, producing complementary samples at each stage.
- **vs. Minimax Diffusion**: Minimax balances diversity and representativeness to control generation but does not address training-signal redundancy. LGD directly measures and optimizes for informational complementarity.
- **vs. MGD3**: MGD3 conditions generation on feature-space modes and marginally outperforms LGD on ImageNet-1K (60.2% vs. 60.1%), but LGD is superior on smaller datasets.
- **Insight**: The essence of data efficiency lies not in the quality of individual samples but in the informational complementarity among samples — dataset quality should be evaluated in terms of marginal gain rather than average gain.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Reformulating distillation as incremental curriculum learning is novel and natural; learnability-guided diffusion sampling is a well-motivated contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluation on three datasets with in-depth redundancy analysis and convincing training dynamics visualization; the advantage on ImageNet-1K is not substantial.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is clearly articulated, problem formulation is precise, and redundancy analysis visualization is excellent.
- Value: ⭐⭐⭐⭐ The redundancy diagnostic framework offers broader inspiration to the field, though further validation on large-scale datasets is needed.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] CaO2: Rectifying Inconsistencies in Diffusion-Based Dataset Distillation](../../ICCV2025/image_generation/cao2_rectifying_inconsistencies_in_diffusion-based_dataset_distillation.md)
- [\[CVPR 2026\] GrOCE: Graph-Guided Online Concept Erasure for Text-to-Image Diffusion Models](groce_graph-guided_online_concept_erasure_for_text-to-image_diffusion_models.md)
- [\[CVPR 2026\] Pluggable Pruning with Contiguous Layer Distillation for Diffusion Transformers](pluggable_pruning_with_contiguous_layer_distillation_for_diffusion_transformers.md)
- [\[CVPR 2026\] Imagine Before Concentration: Diffusion-Guided Registers Enhance Partially Relevant Video Retrieval](imagine_before_concentration_diffusion-guided_registers_enhance_partially_releva.md)
- [\[CVPR 2026\] Quantization with Unified Adaptive Distillation to enable multi-LoRA based one-for-all Generative Vision Models on edge](quantization_with_unified_adaptive_distillation_to_enable_multi-lora_based_one-f.md)

</div>

<!-- RELATED:END -->
