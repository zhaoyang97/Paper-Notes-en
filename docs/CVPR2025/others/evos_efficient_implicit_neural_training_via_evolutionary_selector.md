---
title: >-
  [Paper Note] EVOS: Efficient Implicit Neural Training via EVOlutionary Selector
description: >-
  [CVPR 2025][Implicit Neural Representation] This paper proposes EVOS, an evolutionary selection paradigm (sparse fitness evaluation + frequency-guided crossover + augmented unbiased mutation) for intelligent sparse sampling of INR training coordinates. EVOS reduces training time by 48-66% (180s $\rightarrow$ 97s) while maintaining or even improving reconstruction quality (PSNR 37.81 vs. standard 37.10).
tags:
  - "CVPR 2025"
  - "Implicit Neural Representation"
  - "Evolutionary Sampling"
  - "Frequency-Guided"
  - "Training Acceleration"
  - "Sparse Training"
date: 2026-05-08
content_hash: a27be50ad8f3c177
---

# EVOS: Efficient Implicit Neural Training via EVOlutionary Selector

**Conference**: CVPR 2025  
**arXiv**: [2412.10153](https://arxiv.org/abs/2412.10153)  
**Code**: [https://weixiang-zhang.github.io/proj-evos/](https://weixiang-zhang.github.io/proj-evos/)  
**Area**: Others  
**Keywords**: Implicit Neural Representation, Evolutionary Sampling, Frequency-Guided, Training Acceleration, Sparse Training

## TL;DR
This paper proposes EVOS, an evolutionary selection paradigm (sparse fitness evaluation + frequency-guided crossover + augmented unbiased mutation) for intelligent sparse sampling of INR training coordinates. EVOS reduces training time by 48-66% (180s $\rightarrow$ 97s) while maintaining or even improving reconstruction quality (PSNR 37.81 vs. standard 37.10).

## Background & Motivation

**Background**: Implicit Neural Representations (INRs) map coordinates to signal values (e.g., image pixels, 3D shape SDFs) using MLPs, widely used in image reconstruction and 3D representations. However, INR training requires dense iterations over all coordinate points, incurring high computational costs.

**Limitations of Prior Work**: (1) INRs exhibit a spectral bias, failing to fit high-frequency details quickly due to a preference for low frequencies. (2) Existing acceleration methods either increase architectural complexity (partitioning methods), sacrifice memory (explicit caching), or require large-scale pre-training data (meta-learning). Among sampling-based methods, static sampling ignores network training dynamics, while greedy sampling introduces heavy computational overhead.

**Key Challenge**: Most coordinate points are already thoroughly learned in the later stages of INR training, making continuous calculations on them wasteful. However, simply discarding them might miss crucial high-frequency regions.

**Goal**: Strategically perform sparse sampling of training coordinates during INR training to reduce computational overhead while enhancing the learning of high-frequency details.

**Key Insight**: Borrow from the "selection-crossover-mutation" paradigm of evolutionary algorithms—selecting valuable coordinates via fitness evaluation, balancing low- and high-frequency learning via frequency-guided crossover, and preserving sampling diversity via unbiased mutation.

**Core Idea**: Utilize an evolutionary selection strategy to sparsely sample INR training coordinates—fitness evaluation identifies high-loss points, frequency-guided crossover balances low- and high-frequency learning, and a mutation mechanism prevents sampling bias.

## Method

### Overall Architecture
In each training iteration: (1) Use sparse fitness evaluation at key iterations to compute and cache the loss for each coordinate, reusing the cache during intermediate steps. (2) Generate offspring coordinate sets via frequency-guided crossover by selecting half the coordinates from low-frequency parents (high squared-error points) and the other half from high-frequency parents (high Laplacian response points). (3) Inject unselected coordinates with probability $\alpha$ using augmented mutation to maintain diversity. (4) Train the network only on the sparse offspring set.

### Key Designs

1. **Sparse Fitness Evaluation**:

    - **Function**: Reduce the computational overhead of evaluating fitness for all coordinates at every step.
    - **Mechanism**: The full loss evaluation is performed only at key iterations (periodic intervals) to compute and cache fitness scores. Intermediate steps directly reuse these cached scores. The evaluation overhead accounts for only 1.34% of the total training time.
    - **Design Motivation**: Evaluating the loss of all coordinates at every step is equivalent to a full forward pass, rendering the sparse sampling overhead-heavy. Sparse evaluation coupled with caching is key to making evolutionary selection computationally feasible.

2. **Frequency-Guided Crossover**:

    - **Function**: Attend to the learning of both low-frequency and high-frequency regions simultaneously.
    - **Mechanism**: Dual-perspective selection—low-frequency parents are selected based on squared errors to capture high-loss coordinates (focusing on overall reconstruction), while high-frequency parents are selected using the Laplacian operator to capture coordinates in high-gradient areas (focusing on edges/textures). Half of the offspring training set is sampled from each group.
    - **Design Motivation**: Selecting strictly based on loss biases the sampling toward low-frequency regions (as spectral bias leads to larger low-frequency losses). Introducing the Laplacian high-frequency perspective balances frequency coverage.

3. **Augmented Unbiased Mutation**:

    - **Function**: Avoid selection bias where certain regions are never sampled.
    - **Mechanism**: Randomly inject unselected coordinates into the training set with a mutation ratio $\alpha$ (approximately 5%) to guarantee coverage over the entire coordinate space.
    - **Design Motivation**: Since evolutionary selection naturally favors high-fitness regions, the mutation mechanism maintains the exploration-exploitation balance.

### Loss & Training
The standard MSE reconstruction loss is adopted, calculated only over the sparsely sampled coordinate subset. The network is a standard MLP with positional encoding.

## Key Experimental Results

### Main Results (Image Reconstruction, 5K Iterations)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | Time(s) | Speedup |
|------|-------|-------|--------|---------|-------|
| Standard (Full Data) | 37.10 | 0.964 | 0.021 | 180.45 | 1.0× |
| INT (Incremental) | 34.62 | 0.923 | 0.059 | 161.89 | 1.1× |
| EVOS | **37.81** | 0.962 | **0.018** | **97.39** | **1.85×** |

EVOS is not only faster (1.85×) but also yields better quality (PSNR +0.71dB).

### Ablation Study

| Configuration | PSNR | Time(s) | Description |
|------|------|---------|------|
| EVOS w/o CFS (No Frequency Crossover) | 37.49 | 95.43 | PSNR drops by 0.32dB without high-frequency guidance |
| EVOS (Full) | 37.81 | 97.39 | Frequency crossover incurs almost no extra overhead (+2s) |
| Constant vs. Step Scheduling | - | - | Step scheduling is superior |

### Key Findings
- Sparse training surprisingly improves reconstruction quality (+0.71dB) by avoiding redundant updates on well-learned regions and focusing optimization on difficult regions.
- The selection overhead is extremely low (1.34% of training time), proving the computational efficiency of evolutionary selection in INR training.
- Frequency-guided crossover contributes +0.32dB, validating the necessity of dual-perspective frequency balancing.
- Similar acceleration performance is observed on 3D scenes (e.g., NeRF).

## Highlights & Insights
- **Counter-intuitive "Sparse is Better" result**: Rather than "achieving comparable performance with less data," it demonstrates that "using less data actually works better" by preventing ineffective updates on already converged regions.
- **Elegant Integration of Evolutionary Algorithms and Deep Learning**: The selection-crossover-mutation framework is naturally suited for "selecting the most valuable subset from a massive pool of training coordinates."
- **Practical Value of Frequency Guidance**: Using the Laplacian operator as a high-frequency detector compensates for the spectral bias of INRs at virtually zero cost.

## Limitations & Future Work
- The update frequency of the evaluation cache is a hyperparameter whose optimal value varies across different signals (images vs. 3D).
- The selection of the mutation ratio $\alpha$ lacks an adaptive mechanism.
- Evaluated predominantly on image reconstruction; evaluation on 3D reconstruction and video INRs remains limited.

## Related Work & Insights
- **vs. INT**: INT performs incremental sampling without frequency distinction; EVOS with frequency-guided crossover outperforms it by 3.19dB in PSNR.
- **vs. EGRA/Soft Mining**: These greedy sampling methods require evaluating all points at every step; EVOS drastically reduces selection overhead via sparse evaluation and caching.
- **vs. Meta-Learning INR**: Meta-learning approaches require extensive pre-training on similar datasets; EVOS is plug-and-play and requires no pre-training.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of evolutionary algorithms and INR training is novel, and the frequency-guided crossover is cleverly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensively compared with multiple baselines and ablation studies, though 3D task validation is somewhat limited.
- Writing Quality: ⭐⭐⭐⭐ The analogy to evolutionary algorithms is clear, and the motivation of the method is well-conveyed.
- Value: ⭐⭐⭐⭐ Makes a substantial contribution to INR training acceleration; the insight that sparse training can improve quality is highly inspiring.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Tuning the Frequencies: Robust Training for Sinusoidal Neural Networks](tuning_the_frequencies_robust_training_for_sinusoidal_neural_networks.md)
- [\[CVPR 2025\] ZO-SAM: Zero-Order Sharpness-Aware Minimization for Efficient Sparse Training](zo-sam_zero-order_sharpness-aware_minimization_for_efficient_sparse_training.md)
- [\[CVPR 2025\] Subnet-Aware Dynamic Supernet Training for Neural Architecture Search](subnet-aware_dynamic_supernet_training_for_neural_architecture_search.md)
- [\[AAAI 2026\] ParaRevSNN: A Parallel Reversible Spiking Neural Network for Efficient Training and Inference](../../AAAI2026/others/pararevsnn_a_parallel_reversible_spiking_neural_network_for_efficient_training_a.md)
- [\[CVPR 2025\] VKDNW: Training-free Neural Architecture Search through Variance of Knowledge of Deep Network Weights](training-free_neural_architecture_search_through_variance_of_knowledge_of_deep_n.md)

</div>

<!-- RELATED:END -->
