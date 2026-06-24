---
title: >-
  [Paper Note] Shielded Diffusion: Generating Novel and Diverse Images using Sparse Repellency
description: >-
  [ICML2025][Image Generation][Diffusion model diversity] This paper proposes SPELL (Sparse Repellency), a training-free method that injects a **sparse repellency term** during the generation process of diffusion models. This term pushes the sampling trajectories away from a reference set of images (either protected or already generated), thereby enhancing output diversity and preventing the duplication of the training set.
tags:
  - "ICML2025"
  - "Image Generation"
  - "Diffusion model diversity"
  - "sparse repellency guidance"
  - "post-training guidance"
  - "image protection"
  - "de-duplicated generation"
date: 2026-05-08
content_hash: cfe60e549c50e14b
---

# Shielded Diffusion: Generating Novel and Diverse Images using Sparse Repellency

**Conference**: ICML2025  
**arXiv**: [2410.06025](https://arxiv.org/abs/2410.06025)  
**Code**: To be confirmed  
**Area**: Diffusion Models / Image Generation  
**Keywords**: Diffusion model diversity, sparse repellency guidance, post-training guidance, image protection, de-duplicated generation

## TL;DR

This paper proposes SPELL (Sparse Repellency), a training-free method that injects a **sparse repellency term** during the generation process of diffusion models. This term pushes the sampling trajectories away from a reference set of images (either protected or already generated), thereby enhancing output diversity and preventing the duplication of the training set.

## Background & Motivation

Text-to-image diffusion models face two major challenges when deployed:

**Lack of diversity**: Models using Classifier-Free Guidance (CFG) often produce highly similar images when repeatedly generating from the same prompt, lacking true diversity.

**Training set leakage**: Models may directly copy images from the training set, posing copyright and privacy risks.

Existing approaches either require model retraining, employ "generate-and-discard" strategies (which are computationally wasteful), or use global dense particle guidance, which perturbs all samples at every timestep and degrades image quality.

**Key Insight**: Can we design an **on-demand, sparsely intervening** post-processing guidance mechanism—one that applies correction only when the diffusion trajectory is about to fall into the "shielded zone", with corrections naturally concentrated in the early stages of generation?

## Method

### Core Framework: Sparse Repellency (SPELL)

Given a reference image set $\{z_k\}_{k=1}^K$ and a protection radius $r > 0$, the shielded region is defined as the L2 balls around each reference image:

$$S = \bigcup_{k=1}^K B_k, \quad B_k = \{x \in \mathcal{X} : \|x - z_k\|_2 \leq r\}$$

### Trajectory Correction Mechanism

At each timestep $t$ of reverse diffusion, the denoising network is used to predict the final state:

$$\hat{x}_0 = D_{\theta^*}(t, x_t)$$

If $\hat{x}_0$ falls within a certain shielded ball $B_k$, the minimum correction is applied to push it out:

$$\delta_k(\hat{x}_0) = \frac{(\hat{x}_0 - z_k) \cdot r}{\|\hat{x}_0 - z_k\|_2} - (\hat{x}_0 - z_k)$$

### Sparse Aggregation Formula

Summing up the corrections from all shielded points, natural sparsity is achieved through ReLU:

$$\Delta = \sum_{k=1}^K \sigma_{\text{relu}}\left(\frac{r}{\|\hat{x}_0 - z_k\|_2} - 1\right) \cdot (\hat{x}_0 - z_k)$$

- When $\|\hat{x}_0 - z_k\|_2 \geq r$, the ReLU output is 0, and **no intervention is applied**.
- Correction is triggered only when the predicted final state is excessively close to a reference image.
- In practice, typically only a very small number (typically 1) of shielded points are active at each timestep.

### Theoretical Derivation: DPS Perspective

SPELL can be understood as a special case of Diffusion Posterior Sampling (DPS). By Bayes' rule:

$$\nabla_{x_t} \log p_t(x_t \mid x_0 \notin S) = \nabla_{x_t} \log p_t(x_t) + \nabla_{x_t} \log p_{0|t}(x_0 \notin S \mid x_t)$$

The modified reverse SDE is:

$$d\mathbf{X}_t = \left[f(t, \mathbf{X}_t) - g(t)^2 \tilde{s}_t(\mathbf{X}_t, S)\right] dt + g(t) dB_t$$

SPELL replaces the Gaussian-based soft guidance in DPS with a hard ReLU truncation, avoiding hard-to-tune likelihood scale hyperparameters.

### Two Usage Modes

| Mode | Source of Reference Set | Application Scenario |
|------|-----------|---------|
| **Static Shielding** | Protected training set images | Preventing generation of near-duplicates of the training set |
| **Dynamic Intra-batch Repellency** | Predicted final states of current batch + historical batches | Enhancing the diversity of multiple images generated from the same prompt |

During intra-batch repellency, the shielded points are dynamically updated to the current predicted final states of each trajectory $z_{k,t} = D_{\theta^*}(t, x_t^{(k)})$.

### Over-compensation Factor

Introducing an amplification factor $\lambda$ (the paper recommends $\lambda = 1.6$) allows repellency to terminate early, enabling the trajectory to jump out of the shielded region sooner:

$$\Delta' = \lambda \cdot \Delta$$

## Key Experimental Results

### Diversity Improvement (Selected from Table 1)

| Model | Recall ↑ | Vendi Score ↑ | FID ↓ |
|------|----------|---------------|-------|
| Latent Diffusion | 0.236 | 2.527 | 9.50 |
| + SPELL | **0.289** (+22%) | **2.695** (+7%) | 9.55 |
| SD3-Medium | 0.379 | 3.749 | 20.10 |
| + SPELL | **0.483** (+27%) | **4.711** (+26%) | 35.17 |
| EDMv2 | 0.589 | 11.645 | 3.38 |
| + SPELL | **0.600** (+2%) | **11.806** (+1%) | 3.46 |
| MDTv2 | 0.623 | 12.546 | 4.88 |
| + SPELL | **0.634** (+2%) | **12.772** (+2%) | 4.38 |

- Diversity metrics improve consistently across all models.
- Precision undergoes only slight degradation or remains unchanged, while the impact on FID is marginal.
- The diversity-precision Pareto frontier of SPELL outperforms Particle Guidance, Interval Guidance, and CADS.

### Sparsity Analysis

- The magnitude of repellency correction is typically **no more than 5% of the diffusion score**, and at most 35%.
- At $t = 0.8$, only 40% of the trajectories have non-zero repellency terms, which drops to 21% at $t = 0.6$.
- Repellency is primarily concentrated in the early stage of generation, $t \in [0.6, 1.0]$, and is virtually zero in later stages.

### Large-scale Image Protection (Table 2)

| Model | Ratio of Falling into Shielded Region ↓ | Precision | Time per Image |
|------|-----------------|-----------|---------|
| EDMv2 (Without SPELL) | 7.60% | 0.792 | 2.43s |
| + SPELL-1 | 1.08% | 0.792 | 4.63s |
| + SPELL-10 | **0.16%** | 0.768 | 13.54s |

After shielding all 1.2 million ImageNet-1k training images, the rate of near-duplicate generation drops from 7.6% to 0.16%, with Precision remaining almost unchanged.

## Highlights & Insights

1. **Elegant sparse design**: ReLU gating naturally zeroes out repellency terms, activating them only when necessary, eliminating the need to compute interaction energy for every pair of particles.
2. **Training-free, plug-and-play**: Applicable to any pre-trained diffusion model (RGB/VAE latent space, with/without CFG, text/class-conditional).
3. **Single-parameter control**: Only the protection radius $r$ needs to be adjusted to smoothly control the diversity-fidelity tradeoff.
4. **Scalability to million-scale shield sets**: Combined with approximate nearest neighbor search, it can shield up to 1.2 million images.
5. **Cross-batch consistency**: By accumulating historically generated images as the reference set, diversity is guaranteed across a large number of images even with small batch sizes.

## Limitations & Future Work

1. **Shield overlapping issue**: When multiple shielded balls overlap and the trajectory happens to fall into the center of the overlap, the repelling forces may cancel each other out; guaranteeing strict separation requires quadratic programming.
2. **Limitations of L2 distance**: Currently, similarity is measured using L2 distance in the VAE latent space, which may not fully reflect semantic similarity; operating in a semantic feature space like DINOv2 could be considered.
3. **Expectation approximation**: SPELL operates on the conditional expectation $\mathbb{E}[X_0 \mid x_t]$ rather than true samples from $p_{0|t}$, which weakens theoretical guarantees when using probability flow ODE samplers.
4. **Computational overhead for large-scale shielding**: Million-scale shield sets require CPU-side approximate nearest neighbor search, increasing the generation time per image from 2.4s to 13.5s.

## Rating

- Novelty: ⭐⭐⭐⭐ — Simplifies repellency guidance from dense interactions to a sparse ReLU gating, with clear geometric intuition.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers 6 diffusion models, 2 task settings, million-scale scale experiments, and detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Well-integrated theoretical derivation and geometric interpretation, illustrated with rich diagrams.
- Value: ⭐⭐⭐⭐ — Provides a practical tool for addressing diversity and copyright protection in diffusion model deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Initialization is Half the Battle: Generating Diverse Images from a Guidance Potential Posterior](../../ICML2026/image_generation/initialization_is_half_the_battle_generating_diverse_images_from_a_guidance_pote.md)
- [\[CVPR 2026\] Refracting Reality: Generating Images with Realistic Transparent Objects](../../CVPR2026/image_generation/refracting_reality_generating_images_with_realistic_transparent_objects.md)
- [\[ICCV 2025\] LoRAverse: A Submodular Framework to Retrieve Diverse Adapters for Diffusion Models](../../ICCV2025/image_generation/loraverse_a_submodular_framework_to_retrieve_diverse_adapters_for_diffusion_mode.md)
- [\[ICCV 2025\] Less is More: Improving Motion Diffusion Models with Sparse Keyframes](../../ICCV2025/image_generation/less_is_more_improving_motion_diffusion_models_with_sparse_keyframes.md)
- [\[ACL 2025\] Synthia: Novel Concept Design with Affordance Composition](../../ACL2025/image_generation/synthia_novel_concept_design_with_affordance_composition.md)

</div>

<!-- RELATED:END -->
