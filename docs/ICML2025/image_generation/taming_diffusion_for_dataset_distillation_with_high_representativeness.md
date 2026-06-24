---
title: >-
  [Paper Note] Taming Diffusion for Dataset Distillation with High Representativeness (D³HR)
description: >-
  [ICML2025][Image Generation][dataset distillation] This work proposes the D³HR framework, which maps the complex Gaussian mixture distribution in the VAE latent space to a noise space with high normality via DDIM inversion, and then generates a highly representative distilled dataset using a group sampling strategy, comprehensively outperforming existing SOTAs on CIFAR, Tiny-ImageNet, and ImageNet-1K.
tags:
  - "ICML2025"
  - "Image Generation"
  - "dataset distillation"
  - "diffusion models"
  - "DDIM inversion"
  - "distribution matching"
  - "group sampling"
date: 2026-05-08
content_hash: 405d25e2aca0ec71
---

# Taming Diffusion for Dataset Distillation with High Representativeness (D³HR)

**Conference**: ICML2025  
**arXiv**: [2505.18399](https://arxiv.org/abs/2505.18399)  
**Code**: [lin-zhao-resoLve/D3HR](https://github.com/lin-zhao-resoLve/D3HR)  
**Area**: Image Generation  
**Keywords**: dataset distillation, diffusion models, DDIM inversion, distribution matching, group sampling

## TL;DR

This work proposes the D³HR framework, which maps the complex Gaussian mixture distribution in the VAE latent space to a noise space with high normality via DDIM inversion, and then generates a highly representative distilled dataset using a group sampling strategy, comprehensively outperforming existing SOTAs on CIFAR, Tiny-ImageNet, and ImageNet-1K.

## Background & Motivation

Dataset distillation aims to generate a small dataset to replace the original large dataset for training, which offers advantages over dataset pruning under high compression rates. Recently, diffusion-model-based methods (e.g., D4M, Minimax) have become mainstream due to their powerful generative capabilities. They extract representative latent variables in the VAE latent space, eliminating the dependence on specific teacher model architectures.

However, existing diffusion methods face three core issues in ensuring the representativeness of the distilled dataset:

**Inaccurate distribution matching**: The distribution of the VAE latent space is a multi-component Gaussian mixture distribution (Lemma 3.1: each image corresponds to an independent Gaussian component $q(z_i|x_i) \sim \mathcal{N}(\mu_i, \sigma_i^2)$), which has low normality and is difficult to describe accurately with simple distributions. D4M assumes spherical clustering with K-means, and Minimax uses cosine similarity while ignoring probability density differences, both of which fail to match accurately.

**Distribution shift caused by random noise**: Existing methods either add random noise via the DDPM forward process or generate distilled images starting from random noise. The randomness of the noise destroys the structural and representative information in the VAE space, leading to distribution shift in the denoised latent variables.

**Independent sampling**: The distilled data points match local regions of the original distribution individually, lacking constraints on the overall distribution. A limited number of $n$ samples may fail to fully represent the target distribution.

## Method

### Overall Architecture

D³HR consists of three stages: Domain Mapping $\rightarrow$ Distribution Matching $\rightarrow$ Group Sampling. First, images are encoded into latent variables via VAE, then mapped to a noise space with high normality via DDIM inversion. A Gaussian distribution is used to match the distribution in this space. Finally, the most representative subset is selected through a group sampling strategy, and distilled images are generated via DDIM sampling and VAE decoding. A pretrained DiT is used as the diffusion model backbone.

### Key Designs

#### DDIM Inversion Domain Mapping

The core idea is to map the hard-to-fit VAE latent space $\mathcal{Z}_{0,\mathcal{C}}$ to a noise space $\mathcal{Z}_{T,\mathcal{C}}$ with higher normality. For the latent variables $z_0$ in each class $\mathcal{C}$, DDIM inversion is executed:

$$z_{t+1} = \sqrt{\frac{\alpha_{t+1}}{\alpha_t}} z_t + \sqrt{\alpha_{t+1}} \left(\sqrt{\frac{1}{\alpha_{t+1}} - 1} - \sqrt{\frac{1}{\alpha_t} - 1}\right) \varepsilon_\theta(z_t, t, \mathcal{C})$$

Compared to directly adding random noise in the DDPM forward process, DDIM inversion has two key advantages:
- **Information preservation**: The mapping is a deterministic bijection, where $\mathcal{Z}_{T,\mathcal{C}}$ and $\mathcal{Z}_{0,\mathcal{C}}$ have a one-to-one correspondence, preventing the loss of key features.
- **Structural consistency**: The mapped latent variables retain the structural information of the original space, ensuring distribution alignment.

In experiments, 31-step DDIM inversion is used. The authors prove (Lemma 4.1) that when the number of steps $T$ is sufficiently large, $\mathcal{Z}_{T,\mathcal{C}}$ can be approximated as a Gaussian distribution. Ablation studies show that DDIM inversion brings an **11.5%** accuracy improvement compared to the DDPM forward process.

#### Gaussian Distribution Matching

In the noise space, since each dimension is independent and satisfies high normality, it can be accurately described by a Gaussian distribution. By calculating the mean $\mu_{T,\mathcal{C}}$ and variance $\sigma^2_{T,\mathcal{C}}$ of $\mathcal{Z}_{T,\mathcal{C}}$, $\hat{\mathcal{Z}}_{T,\mathcal{C}} \sim \mathcal{N}(\mu_{T,\mathcal{C}}, \sigma^2_{T,\mathcal{C}})$ is constructed, with the probability density:

$$f(\hat{\mathbf{z}}_{T,\mathcal{C}}) = \prod_{i=1}^{d} \frac{1}{\sqrt{2\pi (\sigma^i_{T,\mathcal{C}})^2}} \exp\left(-\frac{(\hat{z}^i_{T,\mathcal{C}} - \mu^i_{T,\mathcal{C}})^2}{2(\sigma^i_{T,\mathcal{C}})^2}\right)$$

#### Group Sampling Strategy

The Ziggurat algorithm is used to sample $n$ latent variables from $\hat{\mathcal{Z}}_{T,\mathcal{C}}$ to form a subset, but the overall distribution of $n$ samples from a single random sampling may deviate from the target. To address this, sampling is repeated $m$ times to obtain $m$ candidate subsets, and the optimal one is selected using a statistical evaluation metric:

$$\mathcal{L}_{T,\mathcal{C}} = \lambda_\mu \cdot \mathcal{L}_{\mu} + \lambda_\sigma \cdot \mathcal{L}_{\sigma} + \lambda_{\gamma_1} \cdot \mathcal{L}_{\gamma_1}$$

The three components measure the differences between the subset and the target distribution in terms of mean, standard deviation, and skewness, respectively (skewness $\gamma_1 = 0$, as the Gaussian distribution is perfectly symmetric). Finally, $j = \arg\min_{1 \leq k \leq m} \mathcal{L}^k_{T,\mathcal{C}}$ is selected.

This process can be executed in parallel on a GPU, requiring only **2.6 seconds** per class for ImageNet-1K (IPC=10, $m=10^6$) on a single RTX A6000 GPU. The hyperparameters are set to $\lambda_\mu = 1, \lambda_\sigma = 1, \lambda_{\gamma_1} = 0.5$.

## Key Experimental Results

### Main Results

#### Table 1: Comparison on ImageNet-1K with More SOTA Methods (Table A7)

| Architecture | IPC | SRe2L | DWA | TEDDY | D³HR (Ours) |
|---|---|---|---|---|---|
| ResNet-18 | 10 | 31.4±0.5 | 32.7±0.2 | 34.1±0.1 | **44.3±0.3** |
| ResNet-18 | 50 | 51.8±0.4 | 52.5±0.1 | 52.5±0.1 | **59.4±0.1** |
| ResNet-18 | 100 | 55.7±0.4 | 56.2±0.2 | 56.5±0.1 | **62.5±0.0** |
| ResNet-101 | 10 | 38.2±0.4 | 40.0±0.1 | 40.3±0.1 | **52.1±0.4** |
| ResNet-101 | 50 | 61.0±0.4 | — | — | **66.1±0.1** |
| ResNet-101 | 100 | 63.7±0.2 | — | — | **68.1±0.0** |

On ImageNet-1K with IPC=10, D³HR achieves a **12.9%** improvement over SRe2L and a **10.2%** improvement over TEDDY.

#### Table 2: Comparison on CIFAR Datasets (Table A6, 1000-epoch Validation)

| Dataset | Architecture | IPC | SRe2L | RDED | D³HR (Ours) |
|---|---|---|---|---|---|
| CIFAR-10 | ResNet-18 | 10 | 53.5±0.6 | 69.8±0.4 | **69.8±0.5** |
| CIFAR-10 | ResNet-18 | 50 | 59.2±0.4 | 75.8±0.6 | **85.2±0.4** |
| CIFAR-10 | ConvNetW128 | 10 | 46.5±0.7 | — | **55.2±0.5** |
| CIFAR-10 | ConvNetW128 | 50 | 54.3±0.3 | — | **66.8±0.4** |

On CIFAR-10 with IPC=50, D³HR improves over the best baseline RDED by **12.5%** (ResNet-18) and **17.4%** (ResNet-101).

### Ablation Study (ImageNet-1K, ResNet-18, IPC=10, Table 3)

| Configuration | Description |
|---|---|
| Base-DDPM | DDPM forward + random sampling, lowest baseline |
| Base-RS | DDIM inversion + random sampling, improves over Base-DDPM by **11.5%** |
| + $\mathcal{L}_\mu$ constraint | Further improvement after introducing the mean constraint |
| + $\mathcal{L}_\mu + \mathcal{L}_\sigma$ | Continued improvement by adding the standard deviation constraint |
| D³HR (Full) | Combined three metrics (mean + standard deviation + skewness) achieves the best performance |

## Key Findings

- **Trade-off in inversion steps**: At step $t=20$, the distribution remains a Gaussian mixture, and single Gaussian matching is inaccurate; at step $t=40$, the normality is good, but structural information loss is high, leading to degraded reconstruction quality. The optimal step is **$t=31$**.
- **Cross-architecture generalization**: A single generation of D³HR is applicable to multiple architectures including ResNet-18/101, MobileNet-V2, VGG-11, EfficientNet-B0, ShuffleNet-V2, and DeiT-Tiny. It is also effective on VGG-11 (where SRe2L fails due to the absence of BN layers).
- **Robustness**: Over 10 runs with different random seeds, D³HR outperforms D4M by **~27.5%** on average, with a smaller variance.
- **Storage efficiency**: Only the statistical parameters ($\mu, \sigma$) and the pretrained DiT weights (~0.016 GB) need to be stored to generate distilled datasets of any IPC, which is much smaller than directly storing the distilled images.
- **Multiple soft label gain**: Using soft labels from 5 teacher models (compared to D3S), D³HR consistently outperforms D3S on ImageNet-1K.

## Highlights & Insights

- **Deep analysis of the nature of VAE space**: This work theoretically proves that the VAE latent space is a multi-component Gaussian mixture distribution (one component per sample) and intuitively demonstrates the difference in normality through t-SNE visualization, providing a solid theoretical motivation for domain mapping.
- **Clever utilization of DDIM inversion**: DDIM inversion is introduced from the image editing field to dataset distillation. Utilizing its deterministic bijective property, it simultaneously solves the problems of information preservation and structural consistency, serving as an elegant implementation of the "map to a simple space and then operate" philosophy.
- **Practicality of group sampling**: By using statistical constraints on the overall distribution of the subset instead of individual matching, it significantly improves representativeness and stability with extremely low computational cost (GPU parallel sampling takes less than 3 seconds per class).
- **Complete independence from teacher models for distillation**: Unlike SRe2L/DWA/RDED which rely on specific teacher models, D³HR only relies on the pretrained diffusion model, truly achieving standard one-time distillation with multi-architecture generalizability.

## Limitations & Future Work

- **Dependency on pretrained diffusion models**: Using DiT as a backbone requires an additional 400 epochs of fine-tuning on non-ImageNet datasets, introducing training costs. Additionally, the modeling quality of the diffusion model itself on certain classes limits the distillation performance.
- **Limitations of Gaussian approximation**: Lemma 4.1 requires a sufficiently large number of inversion steps to guarantee approximation accuracy. However, there is a trade-off between the number of steps and reconstruction quality, requiring experimental search for the optimal point.
- **Dependency on soft labels**: Although the distillation process does not depend on the teacher model, the validation phase still utilizes the teacher model's soft labels for training supervision, meaning it is not completely independent of the teacher.
- **Ineffectiveness of variance adjustment**: Experiments show that increasing or decreasing the sampling variance by $\pm 50\%$ both lead to performance degradation, and the flexibility of the distribution shape is limited by the single Gaussian assumption.
- **Missing partial results**: Some baselines at IPC=100 lack results due to parameter setting issues, leading to incomplete comparison.

## Related Work & Insights

- **Evolution of dataset distillation**: From bi-level optimization (matching gradients, distributions, or trajectories) to efficient methods (SRe2L using BN statistics, RDED stitching real patches), and further to diffusion-based methods (D4M clustering VAE latents, Minimax optimizing cosine similarity), D³HR advances this line of work further through domain mapping and group sampling.
- **New role of diffusion models**: Instead of being merely generative tools, their mathematical properties of forward/backward processes (deterministic mapping, distribution transformation) can be creatively used for data compression and distribution modeling.
- **Insights**: The concept of "mapping complex distributions to a simple space and then operating" has potential application value in scenarios requiring distribution matching, such as domain adaptation, transfer learning, and few-shot data augmentation.

## Rating

- Novelty: ⭐⭐⭐⭐ — The idea of using DDIM inversion for domain mapping is novel, and the group sampling strategy is practically and cleverly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 4 datasets, 6+ architectures, comparisons with multiple baselines, detailed ablation/robustness/storage analyses, and visualizations.
- Writing Quality: ⭐⭐⭐⭐ — The motivations for the three problems are clearly argued, and the theoretical proofs correspond well with the experimental results.
- Value: ⭐⭐⭐⭐ — This work provides an efficient and practical solution for large-scale dataset distillation with prominent advantages in cross-architecture generalization and storage efficiency.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] CaO2: Rectifying Inconsistencies in Diffusion-Based Dataset Distillation](../../ICCV2025/image_generation/cao2_rectifying_inconsistencies_in_diffusion-based_dataset_distillation.md)
- [\[CVPR 2026\] Learnability-Guided Diffusion for Dataset Distillation](../../CVPR2026/image_generation/learnability-guided_diffusion_for_dataset_distillation.md)
- [\[ECCV 2024\] A High-Quality Robust Diffusion Framework for Corrupted Dataset](../../ECCV2024/image_generation/a_highquality_robust_diffusion_framework_for_corrupted_datas.md)
- [\[ICCV 2025\] DiffSim: Taming Diffusion Models for Evaluating Visual Similarity](../../ICCV2025/image_generation/diffsim_taming_diffusion_models_for_evaluating_visual_similarity.md)
- [\[ICML 2025\] Taming Rectified Flow for Inversion and Editing](taming_rectified_flow_for_inversion_and_editing.md)

</div>

<!-- RELATED:END -->
