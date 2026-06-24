---
title: >-
  [Paper Note] Dataset Distillation with Neural Characteristic Function: A Minmax Perspective
description: >-
  [CVPR 2025][Model Compression][Dataset Distillation] The NCFM method is proposed, which reformulates dataset distillation as a minmax adversarial optimization problem by using the Neural Characteristic Function Difference (NCFD) parameterized by a neural network on the complex plane as a distribution distance metric. By aligning both phase (authenticity) and magnitude (diversity) information, it improves performance by up to 20.5% on ImageNet subsets while reducing GPU memory…
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "Dataset Distillation"
  - "Characteristic Function"
  - "Distribution Matching"
  - "Adversarial Optimization"
  - "Complex Plane"
date: 2026-05-08
content_hash: bc54e48ded621162
---

# Dataset Distillation with Neural Characteristic Function: A Minmax Perspective

**Conference**: CVPR 2025  
**arXiv**: [2502.20653](https://arxiv.org/abs/2502.20653)  
**Code**: None  
**Area**: Model Compression  
**Keywords**: Dataset Distillation, Characteristic Function, Distribution Matching, Adversarial Optimization, Complex Plane

## TL;DR
The NCFM method is proposed, which reformulates dataset distillation as a minmax adversarial optimization problem by using the Neural Characteristic Function Difference (NCFD) parameterized by a neural network on the complex plane as a distribution distance metric. By aligning both phase (authenticity) and magnitude (diversity) information, it improves performance by up to 20.5% on ImageNet subsets while reducing GPU memory usage by over 300 times.

## Background & Motivation

**Background**: Distribution Matching (DM) based dataset distillation methods are popular due to their high computational efficiency. Existing DM methods mainly use MSE for point-wise feature matching or MMD for moment matching.

**Limitations of Prior Work**: MSE operates in the Euclidean space for point-wise comparison, which fails to capture the semantic structure of high-dimensional manifolds. Theoretically, MMD requires taking the supremum in the Reproducing Kernel Hilbert Space (RKHS), but in practice, most methods directly align the first-order moments in the feature space, which violates the theoretical requirements of MMD. Point-wise moment alignment is not equivalent to identical distributions, thereby limiting the quality of synthetic data.

**Key Challenge**: Existing distribution distance metrics are either comprehensive-deficient (point-wise MSE, moment-matching MMD) or fixed metrics that cannot adaptively adjust, thus failing to accurately capture the distribution mismatch between real and synthetic data.

**Goal**: Design a theoretically complete and adaptively optimizable distribution distance metric to achieve a balance between authenticity and diversity in synthetic data.

**Key Insight**: The Characteristic Function (CF) is the Fourier transform of the probability density function, which uniquely determines the cumulative distribution function and can fully characterize distribution information. By introducing CF into distribution matching and utilizing an adversarial framework, the metric can be learnable and adaptive.

**Core Idea**: Parameterize the frequency sampling strategy of the characteristic function using a neural network. Through minmax optimization, the model adaptively learns the metric that best distinguishes between real and synthetic distributions, while simultaneously optimizing the synthetic data to minimize this metric.

## Method

### Overall Architecture
The real and synthetic data are taken as inputs and mapped to a latent space via a feature extractor $f$, after which the Neural Characteristic Function Difference (NCFD) on the complex plane is calculated. The sampling network $\psi$ learns the optimal frequency sampling distribution by maximizing the NCFD, while the synthetic data $\tilde{\mathcal{D}}$ converges towards the real distribution by minimizing the NCFD. The entire optimization alternates between max and min steps.

### Key Designs

1. **Neural Characteristic Function Difference (NCFD)**:

    - Function: Serves as a distribution distance metric that fully captures distribution information.
    - Mechanism: CFD is defined based on the characteristic function $\Phi_{\bm{x}}(\bm{t}) = \mathbb{E}[e^{j\langle \bm{t}, \bm{x}\rangle}]$. Via Euler's formula, CFD is decomposed into two components: the magnitude difference $|\Phi_x(t) - \Phi_{\tilde{x}}(t)|^2$ (controlling distribution scale/diversity) and the phase difference $1-\cos(a_x(t)-a_{\tilde{x}}(t))$ (controlling data center/authenticity), which are balanced using a hyperparameter $\alpha$. CFD is theoretically proven to satisfy non-negativity, symmetry, and the triangle inequality as a valid distance metric.
    - Design Motivation: Uniqueness theorems guarantee a one-to-one correspondence between CFD and the underlying distribution, making it more complete than MMD's moment matching; the phase-magnitude decomposition endows the optimization with clear physical interpretations.

2. **Adversarial Minmax Framework**:

    - Function: Adaptively learns the optimal metric instead of using a fixed, static metric.
    - Mechanism: $\min_{\tilde{\mathcal{D}}} \max_{\psi} \mathcal{L}(\tilde{\mathcal{D}}, \mathcal{D}, f, \psi)$. The sampling network $\psi$ parameterizes the sampling distribution (scale mixture of normals) of the frequency parameter $t$. The max step optimizes $\psi$ to maximize the NCFD (finding the frequencies that best distinguish the two distributions), while the min step optimizes the synthetic data to minimize the NCFD, drawing inspiration from GAN adversarial training.
    - Design Motivation: Static frequency sampling cannot cover all meaningful frequency points. Adversarial learning allows the sampling strategy to automatically focus on frequency regions with the most severe distribution mismatch.

3. **Scale Mixture of Normals Sampling**:

    - Function: Flexibly parameterizes the sampling distribution of the frequency parameters.
    - Mechanism: The distribution of the frequency parameter $t$ is defined as $p(t) = \int \mathcal{N}(t|0, \Sigma) p_\Sigma(\Sigma) d\Sigma$, where the distribution of $\Sigma$ is output by the network $\psi$. This is more flexible than a single Gaussian and can represent multi-scale frequency sampling strategies. Increasing the number of samples ensures that the empirical CF converges to the true CF by Lévy's continuity theorem.
    - Design Motivation: Different frequencies exhibit varying sensitivities to distribution discrepancies; a mixture of normals can adaptively concentrate sampling around crucial frequencies.

### Loss & Training
The NCFD loss consists of a magnitude term and a phase term, balanced by $\alpha$. Training alternates between optimizing the sampling network (max step) and the synthetic data (min step). ConvNet with Instance Normalization is adopted as the feature extractor, configured with 3 layers for CIFAR, 4 layers for Tiny-ImageNet, and 5 layers for ImageNet subsets.

## Key Experimental Results

### Main Results

| Dataset | IPC | NCFM | Prev. SOTA | Gain |
|--------|-----|------|----------|------|
| CIFAR-10 | 1 | 49.5% | 46.9%(DATM) | +2.6% |
| CIFAR-10 | 10 | 71.8% | 66.5%(DSDM) | +5.3% |
| CIFAR-100 | 1 | 34.4% | 29.7%(DATM) | +4.7% |
| CIFAR-100 | 10 | 48.7% | 46.2%(DSDM) | +2.5% |
| ImageSquawk | 10 | 72.6% | 52.1%(RDED) | +20.5% |
| ImageMeow | 10 | 60.2% | 42.4%(RDED) | +17.8% |

### Ablation Study

| Configuration | CIFAR-10 IPC=10 | CIFAR-100 IPC=10 | Description |
|------|----------------|-----------------|------|
| NCFM (Full) | 71.8% | 48.7% | Full model |
| w/o minmax | 68.3% | 44.9% | Without adversarial optimization, drops by 3.5%/3.8% |
| Magnitude only ($\alpha$=1) | 69.5% | 46.2% | Lacking phase information |
| Phase only ($\alpha$=0) | 70.1% | 47.3% | Lacking magnitude information |
| Replace NCFD with MMD | 66.5% | 46.2% | NCFD significantly outperforms MMD |

### Key Findings
- Exhibits a massive advantage (+20.5%) on high-resolution ImageNet subsets, demonstrating that the ability of CF to capture distribution discrepancies in high-dimensional spaces vastly outperforms MMD/MSE.
- Reduces GPU memory usage by over 300 times (vs. DATM) due to NCFD's linear complexity, whereas trajectory matching requires storing immense intermediate states.
- Achieves lossless distillation of CIFAR-100 (synthetic training performance $\approx$ full dataset) on a single 2080Ti (2.3GB) for the first time.
- The phase and magnitude components are complementary and indispensable, with the balancing parameter $\alpha$ being optimal around 0.5.

## Highlights & Insights
- **Phase-Magnitude Decomposition of Characteristic Functions**: Decomposes distribution distance into two interpretable components: authenticity (phase) and diversity (magnitude), endowing the distillation objective with clear physical meaning. This decomposition paradigm can be transferred to any generative task requiring a balance between fidelity and diversity.
- **Adversarial Metric Learning**: Instead of designing fixed metrics, the network learns the optimal metric. This conceptually applies the GAN discriminator paradigm to distribution matching in a simple and elegant manner.
- **Extreme Efficiency**: 300$\times$ memory compression + 20$\times$ speedup, enabling dataset distillation to practically run on consumer-grade GPUs and lowering the computational barrier of this research field.

## Limitations & Future Work
- Employs ConvNet with InstanceNorm as the feature extractor, which presents a discrepancy compared to BatchNorm networks commonly used in practical applications.
- The performance improvement of the proposed method on Tiny-ImageNet (64$\times$64) is less pronounced compared to CIFAR and ImageNet subsets.
- The stability of adversarial training requires meticulous tuning; the exact ratio of max steps to min steps may affect convergence.
- Lacks evaluation on the large-scale ImageNet-1K dataset.
- Whether the scale mixture of normals assumption is the optimal choice for the sampling distribution family remains to be explored.

## Related Work & Insights
- **vs. DM/MMD methods**: DM only aligns first-order moments, whereas NCFM aligns entire distributions through CF, offering stronger theoretical guarantees and performance improvements of 5-20%.
- **vs. DATM (Trajectory Matching)**: DATM requires storing expert trajectories, resulting in massive memory overhead; NCFM bypasses trajectories, yielding over 300$\times$ higher efficiency along with superior performance.
- **vs. GAN**: Borrows adversarial training concepts from GANs, but simplifies the discriminator into a lightweight sampling network, ensuring better stability.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Introduces the characteristic function into dataset distillation for the first time, exhibiting theoretical elegance and practicality.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparisons across multiple datasets with thorough ablation studies, though lacking ImageNet-1K.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical derivations, transitioning naturally from the limitations of MMD to CF, with excellent readability.
- Value: ⭐⭐⭐⭐⭐ Implements mutual gains in both performance and efficiency, runnable on consumer-grade GPUs, yielding high practical translation value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Hyperbolic Dataset Distillation](../../NeurIPS2025/model_compression/hyperbolic_dataset_distillation.md)
- [\[ICCV 2025\] Dataset Distillation via the Wasserstein Metric](../../ICCV2025/model_compression/dataset_distillation_via_the_wasserstein_metric.md)
- [\[CVPR 2025\] Emphasizing Discriminative Features for Dataset Distillation in Complex Scenarios](emphasizing_discriminative_features_for_dataset_distillation_in_complex_scenario.md)
- [\[CVPR 2025\] Enhancing Dataset Distillation via Non-Critical Region Refinement](enhancing_dataset_distillation_via_non-critical_region_refinement.md)
- [\[CVPR 2025\] Curriculum Coarse-to-Fine Selection for High-IPC Dataset Distillation](curriculum_coarse-to-fine_selection_for_high-ipc_dataset_distillation.md)

</div>

<!-- RELATED:END -->
