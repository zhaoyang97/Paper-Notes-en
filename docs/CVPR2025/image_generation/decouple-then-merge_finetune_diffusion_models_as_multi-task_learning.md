---
title: >-
  [Paper Note] Decouple-Then-Merge: Finetune Diffusion Models as Multi-Task Learning
description: >-
  [CVPR 2025][Image Generation][Diffusion model finetuning] This paper views diffusion model training as a multi-task learning problem and proposes the Decouple-then-Merge (DeMe) framework. By first group-finetuning multiple specialized models across different timesteps to eliminate gradient conflicts, and then merging them back into a single model in the parameter space, DeMe significantly improves generation quality without introducing extra inference overhead.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Diffusion model finetuning"
  - "multi-task learning"
  - "model merging"
  - "gradient conflict"
  - "timestep decoupling"
date: 2026-05-08
content_hash: 6d09d2d30b11b9fe
---

# Decouple-Then-Merge: Finetune Diffusion Models as Multi-Task Learning

**Conference**: CVPR 2025  
**arXiv**: [2410.06664](https://arxiv.org/abs/2410.06664)  
**Code**: [GitHub](https://github.com/qianli-ma/deme)  
**Area**: Diffusion Models / Image Generation  
**Keywords**: Diffusion model finetuning, multi-task learning, model merging, gradient conflict, timestep decoupling

## TL;DR
This paper views diffusion model training as a multi-task learning problem and proposes the Decouple-then-Merge (DeMe) framework. By first group-finetuning multiple specialized models across different timesteps to eliminate gradient conflicts, and then merging them back into a single model in the parameter space, DeMe significantly improves generation quality without introducing extra inference overhead.

## Background & Motivation

**Background**: Diffusion models generate images by learning a multi-step denoising process, where the standard practice is to share model parameters across all timesteps. Although this facilitates knowledge sharing and training efficiency, the denoising tasks at different timesteps actually differ significantly—large timesteps generate low-frequency basic content, while small timesteps generate high-frequency details.

**Limitations of Prior Work**: Gradient conflicts exist between different timesteps. The authors experimentally find that the cosine similarity of gradients from non-adjacent timesteps is very low, indicating that different denoising tasks interfere with each other during training (negative transfer), which degrades the overall generation quality. Existing loss re-weighting methods can only mitigate but not fundamentally resolve this issue.

**Key Challenge**: On one hand, cross-timestep knowledge sharing is required to maintain efficiency; on the other hand, gradient conflicts lead to negative transfer. Timestep-level model ensembles can avoid conflicts but introduce N-fold storage and GPU memory overhead (e.g., 6 independent models), making them impractical.

**Goal**: To eliminate gradient conflicts while maintaining knowledge sharing, without introducing any extra overhead during inference.

**Key Insight**: To tackle the problem from the perspectives of multi-task learning and model merging—first decoupling to eliminate conflicts, and then merging to preserve knowledge.

**Core Idea**: Divide the timesteps into N non-overlapping intervals to finetune N separate models, employ specially designed training techniques to prevent overfitting, and finally merge them back into a single model via weighted task vectors.

## Method

### Overall Architecture
Starting from a pre-trained diffusion model, the total timesteps $[0,T)$ are partitioned into $N$ non-overlapping intervals. A replica is initialized from the pre-trained model for each interval and finetuned with three proposed training techniques. After finetuning, the parameter differences (task vectors) between each finetuned model and the pre-trained model are computed, and then merged back into a single model via weighted sum for inference.

### Key Designs

1. **Decoupled Finetuning + Three Training Techniques**:

    - **Function**: Eliminate gradient conflicts while maintaining cross-timestep knowledge
    - **Mechanism**: (a) Channel-wise Projection—apply a learnable channel projection matrix $W \in \mathbb{R}^{C \times C}$ (initialized as an identity matrix) on intermediate features, as the discrepancies before and after finetuning predominantly reside in the channel dimension rather than the spatial dimension; (b) Consistency Loss—constrain the discrepancy between the outputs of the finetuned model and the original model to prevent excessive deviation; (c) Probabilistic Sampling—sample timesteps from the corresponding interval with probability $1-p$ and globally with probability $p$ to preserve memory of other intervals
    - **Design Motivation**: Pure decoupling causes each model to only handle denoising in its designated interval while forgetting others. The design of Channel-wise Projection originates from the experimental observation that activation differences before and after finetuning are concentrated in the channel dimension

2. **Model Merging via Task Vectors**:

    - **Function**: Seamlessly merge the N finetuned models into a single model for inference
    - **Mechanism**: Compute the task vector $\tau_i = \theta_i - \theta$ for each model, and then obtain $\theta_{merged} = \theta + \sum_{i=1}^N w_i \tau_i$. The merging weights $w_i$ are optimized via grid search
    - **Design Motivation**: Model merging techniques have been shown to consolidate knowledge from models finetuned on different tasks/datasets. This work is the first to apply them to cross-timestep merging in diffusion models. The merged model retains the original size with zero extra inference overhead

3. **Loss Landscape Analysis**:

    - **Function**: Explain why decoupled finetuning can improve already converged pre-trained models
    - **Mechanism**: Visual analysis shows that while the pre-trained model is at a critical point (zero gradient, sparse contours) over the full range of timesteps $[0,1000)$ on the loss landscape, it lies in dense contour areas (with clear optimization directions) for each individual sub-interval. Decoupling allows the model to escape the global critical point
    - **Design Motivation**: Provide theoretical intuition on why further finetuning can continuously improve an already converged model

### Loss & Training
The overall loss is the standard denoising loss plus the Consistency Loss: $\mathcal{L} = \|\epsilon - \epsilon_{\theta_i}(x_t, t)\|^2 + \|\epsilon_\theta(x_t, t) - \epsilon_{\theta_i}(x_t, t)\|^2$. Each model is finetuned for 20K iterations (totaling 80K equivalent iterations when N=4).

## Key Experimental Results

### Main Results (Unconditional Generation, DDPM)

| Dataset | Metric (FID↓) | DeMe (Merged) | Pre-trained | Min-SNR-γ | ANT-UW | Gain |
|--------|-----------|-------------|--------|-----------|--------|------|
| CIFAR10 | FID | **3.51** | 4.42 | 5.77 | 4.21 | -0.91 |
| LSUN-Church | FID | **7.27** | 10.69 | 10.82 | 10.43 | -3.42 |
| LSUN-Bedroom | FID | **5.84** | 6.46 | 6.41 | 6.48 | -0.62 |

### Ablation Study (CIFAR10, DDIM 100 steps)

| Configuration | FID↓ | Description |
|------|------|------|
| N=1, No techniques (Traditional) | 4.40 | Baseline |
| N=1 + Channel Projection | 4.45 | CP is harmful without decoupling |
| N=8 + Prob. Sampling | 4.32 | Begins to improve after decoupling |
| N=8 + PS + CL | 4.27 | Further improvement with Consistency Loss |
| N=8 + PS + CL + CP | **3.87** | Optimal combination of all techniques, FID decreases by 0.53 |

### Key Findings
- The model merging scheme even outperforms model ensembles (e.g., on LSUN-Church, merging achieves FID=7.27 vs. ensemble FID=9.57), indicating that merging yields effects beyond simple ensembling.
- Channel-wise Projection is harmful without decoupling, and only becomes effective when combined with decoupling.
- All loss re-weighting baselines are nearly ineffective or even harmful under the finetuning setup, indicating they cannot truly resolve gradient conflicts.
- The method is also effective on Stable Diffusion: MS-COCO FID decreases by 0.36, and CLIP Score increases by 0.23.

## Highlights & Insights
- Re-evaluating diffusion model training from a multi-task learning perspective, with compelling findings and visualizations of gradient conflicts. This perspective is transferable to any generative model that shares parameters across multiple timesteps.
- The merged model surprisingly outperforms the ensemble model, which is counterintuitive—likely because merging finds a better equilibrium point in the parameter space than individual finetuned models.
- The Loss Landscape analysis reveals the phenomenon of "seemingly converged but still having optimization room", which offers valuable insights for understanding the training dynamics of large models.

## Limitations & Future Work
- Training requires finetuning N full models, which, although having the same total iterations, demands N-fold GPU memory (can be mitigated by sequential finetuning).
- Merging weights are obtained via grid search, with the search space growing when extending to more intervals (large N).
- Non-uniform partitioning remains unexplored, where different timestep intervals might require partitions of different sizes.
- Combining with parameter-efficient finetuning methods like LoRA could be considered to further reduce training costs.

## Related Work & Insights
- **vs Loss Reweighting (Min-SNR, P2)**: These methods attempt to balance training by adjusting loss weights for different timesteps, but experiments prove they are largely ineffective under the finetuning setup. DeMe fundamentally decouples the optimization directions.
- **vs Timestep Ensemble**: Methods like DMP use 6 independent models, leading to a 6-fold increase in storage and GPU memory. DeMe reduces this overhead to zero via merging.
- **vs ANT**: ANT introduces MTL optimization methods (such as NashMTL) into diffusion models, but underperforms compared to DeMe's decouple-then-merge paradigm.

## Rating
- Novelty: ⭐⭐⭐⭐ Viewing diffusion training as MTL and resolving it via decoupling and merging is a refreshing idea.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on 6 datasets, DDPM, and SD, with multiple baselines and comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, rich visualizations, and deep analysis.
- Value: ⭐⭐⭐⭐ A general finetuning framework widely applicable to diffusion models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] ToMA: Token Merge with Attention for Diffusion Models](../../ICML2025/image_generation/toma_token_merge_with_attention_for_diffusion_models.md)
- [\[ICML 2026\] EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization](../../ICML2026/image_generation/evogm_learning_to_merge_llms_via_evolutionary_generative_optimization.md)
- [\[CVPR 2026\] SplitFlux: Learning to Decouple Content and Style from a Single Image](../../CVPR2026/image_generation/splitflux_learning_to_decouple_content_and_style_from_a_single_image.md)
- [\[CVPR 2025\] Calibrated Multi-Preference Optimization for Aligning Diffusion Models](calibrated_multi-preference_optimization_for_aligning_diffusion_models.md)
- [\[CVPR 2025\] Generative Modeling of Class Probability for Multi-Modal Representation Learning](generative_modeling_of_class_probability_for_multi_modal_representation_learning.md)

</div>

<!-- RELATED:END -->
