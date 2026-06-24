---
title: >-
  [Paper Note] Diffusion Sampling Correction via Approximately 10 Parameters
description: >-
  [ICML 2025][Model Compression][Diffusion model acceleration] Proposes the PCA-based Adaptive Search (PAS) method, which leverages the geometric property where sampling trajectories reside in a low-dimensional subspace of a high-dimensional space. By extracting a small number of orthogonal basis vectors via PCA and learning only about 10 coordinate parameters, PAS corrects the truncation errors of existing fast samplers. With sub-minute training on a single A100 GPU…
tags:
  - "ICML 2025"
  - "Model Compression"
  - "Diffusion model acceleration"
  - "PCA sampling correction"
  - "low-parameter training"
  - "plug-and-play"
  - "truncation error"
date: 2026-05-08
content_hash: 04fe93ca19a20dfd
---

# Diffusion Sampling Correction via Approximately 10 Parameters

**Conference**: ICML 2025  
**arXiv**: [2411.06503](https://arxiv.org/abs/2411.06503)  
**Code**: [https://github.com/onefly123/PAS](https://github.com/onefly123/PAS)  
**Area**: Diffusion Models / Sampling Acceleration  
**Keywords**: Diffusion model acceleration, PCA sampling correction, low-parameter training, plug-and-play, truncation error

## TL;DR

Proposes the PCA-based Adaptive Search (PAS) method, which leverages the geometric property where sampling trajectories reside in a low-dimensional subspace of a high-dimensional space. By extracting a small number of orthogonal basis vectors via PCA and learning only about 10 coordinate parameters, PAS corrects the truncation errors of existing fast samplers. With sub-minute training on a single A100 GPU, it reduces the FID of DDIM on CIFAR10 from 15.69 to 4.37 (NFE=10).

## Background & Motivation

**Background**: Diffusion Probabilistic Models (DPMs) have demonstrated powerful capabilities in fields such as image generation, text-to-image, and video generation. However, their reverse denoising process typically requires hundreds to thousands of iterations, making sampling extremely slow, which poses a core bottleneck for practical applications.

**Limitations of Prior Work**: Existing training-free acceleration methods (e.g., DDIM, DPM-Solver) suffer from severely amplified cumulative truncation errors when NFE < 10, leading to a sharp drop in generation quality. Training-based methods (e.g., distillation) can achieve single-step sampling but incur extremely high training costs (requiring over 100 A100 GPU hours on CIFAR10) and destroy the interpolation capabilities of the original ODE trajectories.

**Key Challenge**: How to correct the amplified truncation errors during few-step sampling at an ultra-low cost? Although low-cost training methods have been explored (e.g., AMED, GITS), they still require training small neural networks, making their parameter scale and training overhead far from "negligible."

**Goal**: Design a plug-and-play, ultra-low-parameter method that enables existing fast solvers (such as DDIM and iPNDM) to achieve high-quality sampling when NFE < 10, while preserving the interpolation capability of the original diffusion paths.

**Key Insight**: The authors observe two key geometric properties: (1) The sampling trajectory of a single sample lies within an approximately 3-dimensional subspace of the high-dimensional space; (2) The cumulative truncation errors of different samples exhibit a uniform "S-shaped" distribution. The former implies that a very small number of basis vectors extracted via PCA can represent the sampling direction, while the latter suggests that correcting only the high-curvature regions is sufficient to bypass most steps.

**Core Idea**: Use PCA to reduce the high-dimensional sampling direction correction problem to a low-dimensional coordinate search problem, requiring only about 10 parameters and sub-minute training to significantly improve sampling quality.

## Method

### Overall Architecture

PAS is a three-step pipeline: (1) Generate ground-truth trajectories using a teacher solver with a high NFE; (2) Perform PCA decomposition on existing trajectories at each sampling step to obtain orthogonal basis vectors, and learn low-dimensional coordinates to correct the sampling direction; (3) Apply an adaptive search strategy to retain correction parameters only in high-curvature regions, compressing the stored parameters to approximately 10. The entire process does not modify the original pre-trained model at all, making it a purely plug-and-play solution.

### Key Designs

1. **PCA Basis Vector Sampling Correction**:

    - **Function**: In each sampling step $t_i \to t_{i-1}$, use PCA to extract a small number of orthogonal basis vectors from the existing sampling trajectory $\{x_{t_N}, d_{t_N}, ..., d_{t_{i+1}}\}$, and then learn the corresponding coordinates $\mathbf{C}=[c_1, c_2, c_3, c_4]$ to correct the sampling direction $d_{t_i}$.
    - **Mechanism**: Since sampling trajectories lie in an approximately 3-dimensional subspace (validated by PCA), only 3-4 basis vectors are sufficient to fully cover the trajectory space. The first basis vector is set to the normalized current direction $d_{t_i}$, and the remaining ones are obtained via SVD decomposition combined with Schmidt orthonormalization. Only 4 scalar coordinates need to be learned per sampling step, keeping the total parameter count extremely low.
    - **Design Motivation**: Traditional training methods require a neural network to directly output high-dimensional correction vectors, incurring huge parameter and computational overheads. PCA reduces the problem from "searching for the optimal direction in a D-dimensional space" to "searching in a 4-dimensional coordinate space," reducing training costs by multiple orders of magnitude.

2. **Adaptive Search Strategy**:

    - **Function**: Automatically determine which sampling steps require correction and which can be skipped based on the "S-shaped" truncation error distribution.
    - **Mechanism**: The cumulative truncation error exhibits an S-shape—slow growth at the beginning, sharp increase in the middle, and slowing down again at the end—corresponding to trajectories that are linear, then curved, and then linear again. Only the curved segments (high-curvature regions) require correction, whereas correcting linear segments introduces bias instead. By setting a tolerance $\tau$, the L2 loss before and after correction is compared to decide whether to retain the correction.
    - **Design Motivation**: Doing PCA correction without adaptive search (PAS-AS) actually performs worse than the original DDIM (FID degrades from 15.69 to 120+), because corrections on linear segments introduce biases from other basis directions. The adaptive search compresses corrections across $N$ sampling steps down to only 1–3 steps, reducing parameters from $4N$ to 4–12.

3. **SGD Coordinate Training**:

    - **Function**: Optimize coordinate parameters using SGD to move the corrected sampling points closer to the ground-truth trajectories.
    - **Mechanism**: Given a teacher trajectory $\{x_{t_i}^{gt}\}$, SGD is used to update the coordinate $\mathbf{C}$ in a single step with L2 distance as the loss function. Since there are only 4 parameters, a single-step gradient update is sufficient to find a good direction. Training is conducted on 5k ground-truth trajectories, taking less than 1 minute on CIFAR10.
    - **Design Motivation**: Coordinate parameters describe "how far to deviate from the current direction," with the initial value set to $[\|d_{t_i}\|_2, 0, 0, 0]$ (i.e., no correction). SGD searches for the optimal offset starting from this initialization, converging extremely fast.

### Loss & Training

- Ground-truth trajectory generation: Use the Heun second-order solver with 100 NFE to generate high-precision reference trajectories.
- Loss function: L1 loss is recommended (ablation studies confirm L1 outperforms L2, LPIPS, and Pseudo-Huber).
- Training strategy: Sequential correction—correct $d_{t_N}$ first, then $d_{t_{N-1}}$ (since the previous step's correction changes subsequent states)—combined with adaptive search to retain correction parameters as needed.
- Time schedule: Apply the EDM polynomial time schedule with $\rho=7$. Teacher trajectories are generated by inserting intermediate steps between the student time steps.

## Key Experimental Results

### Main Results

| Dataset | Base Sampler | Original FID (NFE=10) | FID with PAS | Params | Gain |
|--------|----------|---------------|---------|--------|------|
| CIFAR10 32×32 | DDIM | 15.69 | **4.37** | 12 | -72.1% |
| CIFAR10 32×32 | iPNDM | 3.69 | **2.84** | 8 | -23.0% |
| FFHQ 64×64 | DDIM | 18.37 | **5.61** | ~16 | -69.5% |
| FFHQ 64×64 | iPNDM | 4.95 | **4.28** | ~8 | -13.5% |
| ImageNet 64×64 | DDIM | 16.72 | **9.13** | ~20 | -45.4% |
| LSUN Bedroom 256×256 | DDIM | 11.42 | **6.23** | ~16 | -45.4% |
| Stable Diffusion 512 | DDIM | 16.56 | **14.23** | ~4 | -14.1% |

### Ablation Study

| Configuration | Key Metric (FID, NFE=10) | Description |
|------|---------|------|
| DDIM Baseline | 15.69 | No correction |
| PAS (-AS) Without Adaptive Search | 120.32 | Full-step correction degrades performance instead |
| Full PAS | **4.37** | Adaptive search is key |
| 2 Basis Vectors | ~5.5 | Already shows significant improvement |
| 4 Basis Vectors | **4.37** | Optimal |
| Trained on 500 trajectories | ~6.0 | Effective even with a small number of trajectories |
| Trained on 5k trajectories | **4.37** | Optimal balance point |

### Key Findings

- Adaptive search is crucial for the success of PAS—without it, full-step correction severely degrades quality instead.
- 4 basis vectors are sufficient to span the sampling trajectory space (consistent with the PCA cumulative variance analysis).
- As few as 500 trajectories are needed to yield significant improvements, with 5k being the optimal choice.
- PAS can be combined with Teleportation (TP): DDIM + TP + PAS achieves an FID as low as 3.16 on CIFAR10 with NFE=10.
- PAS + DDIM outperforms SOTA solvers like DPM-Solver-v3 on Stable Diffusion.
- Hyperparameters such as learning rate and tolerance are insensitive for DDIM correction, whereas they require finer tuning for iPNDM.

## Highlights & Insights

- The title "Approximately 10 Parameters" is highly impactful—12 scalar parameters can be hard-coded into inference code, incurring zero extra storage overhead.
- The application of PCA elegantly translates geometric insights into a practical algorithm. Observing and exploiting the low-dimensional subspace property is a classic example of theory-driven design.
- The discovery of the S-shaped error curve reveals the inherent patterns of diffusion sampling. The error peaks within the transition region from noise to structure, aligning with the "coarse-to-fine" generation paradigm.
- Fully preserves the original ODE trajectories—unlike distillation methods which disrupt interpolation across modes, PAS is ideal for downstream tasks that rely on trajectory continuity.

## Limitations & Future Work

- Requires generating reference trajectories beforehand to perform PCA (while low-cost, it remains a prerequisite).
- Limited improvement when NFE < 5 (on CIFAR10, DDIM + PAS still yields an FID of 41.14 with NFE=4).
- Provides limited gains for solvers like iPNDM that already have small truncation errors, and requires more careful hyperparameter tuning.
- Currently validated only under specific time schedulers; compatibility with adaptive time schedulers remains unexplored.

## Related Work & Insights

- **Trajectory Geometry Methods**: AMED utilizes the mean value theorem for dimensionality reduction, and GITS optimizes schedules leveraging trajectory consistency. PAS builds upon these by using PCA to directly correct directions.
- **Low-Cost Training Methods**: BK-SDM distills student models, and DRS trains discriminators. The parameter scale of PAS is multiple orders of magnitude lower than these approaches.
- **Complementarity to Consistency Models**: Consistency Models (CM) learn a direct mapping (single-step generation), whereas PAS corrects multi-step samplers. The former pursues ultimate speed, while the latter targets high-quality generation at moderate step counts.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Correcting diffusion sampling with ~10 parameters is a highly innovative feat of parameter-efficiency.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 5 datasets × multiple solvers × detailed ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐⭐ A highly cohesive narrative spanning theoretical motivation, geometric observation, algorithm design, and experimental validation.
- **Value**: ⭐⭐⭐⭐⭐ A highly practical, ultra-low-cost, general-purpose diffusion sampling acceleration scheme.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Sampling-Aware Quantization for Diffusion Models](../../CVPR2026/model_compression/sampling-aware_quantization_for_diffusion_models.md)
- [\[ECCV 2024\] Adaptive Compressed Sensing with Diffusion-Based Posterior Sampling](../../ECCV2024/model_compression/adaptive_compressed_sensing_with_diffusionbased_posterior_sa.md)
- [\[ICML 2025\] The Diffusion Duality](the_diffusion_duality.md)
- [\[NeurIPS 2025\] EMLoC: Emulator-based Memory-efficient Fine-tuning with LoRA Correction](../../NeurIPS2025/model_compression/emloc_emulator-based_memory-efficient_fine-tuning_with_lora_correction.md)
- [\[CVPR 2025\] Sampling Innovation-Based Adaptive Compressive Sensing](../../CVPR2025/model_compression/sampling_innovation-based_adaptive_compressive_sensing.md)

</div>

<!-- RELATED:END -->
