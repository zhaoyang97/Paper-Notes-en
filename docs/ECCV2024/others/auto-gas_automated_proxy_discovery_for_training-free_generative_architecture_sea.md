---
title: >-
  [Paper Note] Auto-GAS: Automated Proxy Discovery for Training-Free Generative Architecture Search
description: >-
  [ECCV 2024][Generative Architecture Search] This paper proposes Auto-GAS, the first training-free architecture search framework for generative adversarial networks (GANs). By automatically discovering and optimizing zero-cost proxy metrics to replace traditional training-based evaluations, it achieves a $110\times$ search speedup while maintaining comparable generation quality with training-based methods.
tags:
  - "ECCV 2024"
  - "Generative Architecture Search"
  - "Training-Free Proxies"
  - "Evolutionary Algorithms"
  - "GAN Compression"
  - "Zero-Cost Proxies"
date: 2026-05-08
content_hash: f6a595a3ea50bc38
---

# Auto-GAS: Automated Proxy Discovery for Training-Free Generative Architecture Search

**Conference**: ECCV 2024  
**Code**: [https://github.com/lliai/Auto-GAS](https://github.com/lliai/Auto-GAS)  
**Area**: Others  
**Keywords**: Generative Architecture Search, Training-Free Proxies, Evolutionary Algorithms, GAN Compression, Zero-Cost Proxies

## TL;DR

This paper proposes Auto-GAS, the first training-free architecture search framework for generative adversarial networks (GANs). By automatically discovering and optimizing zero-cost proxy metrics to replace traditional training-based evaluations, it achieves a $110\times$ search speedup while maintaining comparable generation quality with training-based methods.

## Background & Motivation

**Background**: Generative Adversarial Networks (GANs) are widely used in scenarios such as real-time image generation and image translation. However, the inference speed and memory consumption of standard GAN generators limit deployment efficiency. Generative Architecture Search (GAS) aims to automatically find the optimal GAN generator structure to achieve the best balance between speed and quality. Existing GAS methods mainly adopt differentiable search or evolutionary search strategies, both of which require training candidate architectures fully to evaluate performance.

**Limitations of Prior Work**: Training-based GAS methods (such as GAN Compression, EAGAN) are extremely computationally expensive: each candidate architecture must undergo a complete training-evaluation cycle, leading to search times typically measured in GPU days. This severely limits the practical adoption of GAS. A natural improvement is to introduce training-free search, which uses zero-cost proxies to quickly evaluate the potential of candidate architectures without training.

**Key Challenge**: However, existing zero-cost proxies (such as SynFlow, NASWOT, etc.) are designed for classification tasks, and their predictive capability drops significantly on generation tasks. The evaluation criteria for classification and generation are fundamentally different—classification focuses on discriminative ability, whereas generation focuses on distribution matching and image quality. Therefore, proxies dedicated to generation tasks must be designed, but manual design is both time-consuming and lacks theoretical guidance.

**Goal**: (1) How to automatically discover zero-cost proxies suitable for GAN generation tasks? (2) How to ensure that the discovered proxies have sufficient predictive ability to reliably guide the architecture search? (3) How to significantly accelerate the search while maintaining generation quality?

**Key Insight**: The authors observe that while a single hand-crafted proxy performs poorly on generation tasks, combining and transforming multiple feature statistics can construct composite proxies with much stronger predictive power. Thus, the design of proxy metrics can itself be viewed as a search problem, and evolutionary algorithms can be used to automatically discover the optimal proxy.

**Core Idea**: Modeling the construction of proxy metrics as a search problem, and using evolutionary algorithms to automatically discover highly predictive evaluation proxies for generative architectures within a search space composed of feature statistics, transformation operations, and encoding operations.

## Method

### Overall Architecture

The overall pipeline of Auto-GAS consists of two phases: (1) **Proxy Discovery Phase**—constructing a proxy search space and using an evolutionary algorithm to search for the optimal zero-cost proxy metric; (2) **Architecture Search Phase**—utilizing the discovered proxy metric to quickly evaluate and rank candidate GAN generator architectures, identifying the optimal architecture without training. The input consists of the definitions of the GAN search space and a small amount of reference data, and the output is the optimal generator architecture configuration.

### Key Designs

1. **Information-Aware Proxy Construction**:

    - **Function**: Extracts diverse feature statistics from candidate GAN generators to construct the basic inputs for proxy metrics.
    - **Mechanism**: Instead of relying on a single metric, this design utilizes feature map statistics (such as mean, variance, gradient information, etc.) across different network layers as raw inputs. It then defines four categories of operations—transform, encoding, reduction, and augment—to combine and process these statistics. Each candidate proxy is represented by a combinational chain of operations, similar to a "proxy program." This forms a rich proxy search space capable of expressing diverse candidate proxy metrics.
    - **Design Motivation**: A single statistic (such as using only the Jacobian norm or Fisher information) cannot comprehensively characterize generation capability, but the combination and transformation of different statistics can capture richer architectural characteristics. This design transforms proxy discovery into a combinatorial optimization problem suitable for evolutionary algorithm optimization.

2. **Evolutionary Proxy Search**:

    - **Function**: Automatically searches for the most predictive proxy metric within the proxy search space.
    - **Mechanism**: Initializes a population of random proxies and calculates the rank correlation (using Kendall's tau or Spearman correlation coefficients) between each proxy and the true architecture rankings. Individuals are selected based on correlation scores, and crossover and mutation operations are executed on elite individuals to generate the next generation of candidate proxies. Through multiple generations of evolutionary iteration, the proxies in the population gradually converge to the optimal proxy with high predictive power. To calculate the ground-truth rankings as supervision signals, only a small subset of candidate architectures needs to be trained and evaluated, keeping the cost manageable.
    - **Design Motivation**: Evolutionary algorithms are naturally suited for dealing with discrete combinatorial search spaces of this kind and do not require the proxy search space to be differentiable. By pre-calculating the true rankings on a small subset to provide evaluation signals, it delicately balances search cost and proxy quality.

3. **Training-Free Generator Search**:

    - **Function**: Utilizes the optimized proxy metric to quickly evaluate a large number of candidate generator architectures and select the optimal one.
    - **Mechanism**: Given a GAN search space (including configurations for different numbers of layers, channels, kernel sizes, etc.), each candidate generator requires only a single forward pass to generate a score through the proxy metric. The architectures are then ranked by score, and the highest-scoring architecture is selected for full training validation. The entire search process does not require any training, and the evaluation time for a single architecture is reduced from hours to seconds.
    - **Design Motivation**: The core advantage of Auto-GAS is that it reduces the computational overhead during the search phase from training-level to inference-level, making large-scale architecture search practical.

### Loss & Training

The optimization objective in the proxy discovery phase is to maximize the rank correlation coefficient (Kendall tau / Spearman correlation) between the proxy scores and the true architecture rankings. The final selected optimal architecture is still trained using the standard GAN training pipeline (adversarial loss + perceptual loss, etc.). The training strategy is kept consistent with baseline methods to ensure a fair comparison.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours (Auto-GAS) | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| CIFAR-10 | FID ↓ | Comparable to GAN Compression | GAN Compression | $110\times$ search speedup |
| Image-to-Image (pix2pix) | FID ↓ | Competitive results | EAGAN | Search efficiency substantially outperforms training-based methods |
| Image-to-Image (CycleGAN) | FID ↓ | Competitive results | Training-based GAS | Search time reduced from GPU days to GPU minutes |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Existing classification proxies (SynFlow, etc.) | Low rank correlation | Proxies designed for classification have poor predictive power on generation tasks |
| Single-statistic proxies | Medium rank correlation | A single feature statistic has limited capability |
| Composite proxies (without evolutionary optimization) | Relatively high rank correlation | Hand-crafted combinations have some effect but are insufficient |
| Auto-GAS (完整进化搜索) | Highest rank correlation | Automatically discovered proxies significantly outperform all hand-crafted proxies |

### Key Findings

- Zero-cost proxies designed for classification tasks (e.g., SynFlow, NASWOT) exhibit significantly lower rank correlation coefficients on GAN generation tasks compared to automatically discovered proxies.
- The proxies discovered by Auto-GAS exhibit a certain level of transferability across different GAN search spaces and datasets.
- The search speedup can reach over $110\times$ (compared to GAN Compression) while maintaining competitive generation quality.
- The evolutionary proxy search process typically converges within a few hundred generations, and the overall overhead of proxy discovery is far smaller than a single full training-based search.

## Highlights & Insights

- **Automating Proxy Design**: For the first time, the question of "how to evaluate the quality of an architecture" is automated, representing an important paradigm innovation in the NAS/GAS field.
- **Deep analysis of cross-task proxy discrepancy**: Clearly reveals why classification proxies are not suitable for generation tasks, providing a direction for future task-specific proxy designs.
- **Clever design of the search space**: Constructs the proxy search space through a transform-encoding-reduction-augment operational chain, which is highly expressive and efficient to search.
- **Outstanding practical value**: The $110\times$ search speedup transitions GAS from a research-grade tool to a practically usable solution.

## Limitations & Future Work

- Proxy discovery still relies on full training on a small subset to provide true ranking labels, which cannot entirely avoid training overhead.
- The interpretability of the discovered proxy metrics is relatively weak, making it difficult to extract intuitive insights for generative model design.
- Mainly validated on GANs; whether it is applicable to other generative models (such as diffusion models, VAEs) remains unexplored.
- The search space is currently limited to the generator structure; joint search of the discriminator may bring further improvements.
- While the cross-dataset transferability of the proxies exists, it is limited. Tasks with large distribution discrepancies may require searching again.

## Related Work & Insights

- **GAN Compression** (Zhu et al.): Channel pruning-based GAN acceleration, which requires full training and serves as the main baseline for Auto-GAS.
- **EAGAN** (ICLR 2024): Evolutionary GAN architecture search, which is computationally more efficient but still requires training.
- **Zero-Cost Proxy for NAS** (SynFlow, NASWOT, etc.): Training-free NAS proxies, but restricted to classification tasks.
- **TransNASBench**: A cross-task NAS benchmark, which inspired the cross-task proxy analysis in this paper.

## Rating

- Novelty: ⭐⭐⭐⭐ First to automatically discover proxy metrics in GAS, presenting a novel ideology.
- Experimental Thoroughness: ⭐⭐⭐ Evaluated across multiple tasks and datasets with somewhat complete ablation studies, though lacking more quantitative comparisons.
- Writing Quality: ⭐⭐⭐ Overall clear, though the description of the proxy search space is relatively abstract.
- Value: ⭐⭐⭐⭐ Training-free GAS with a $110\times$ speed up holds significant practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] VKDNW: Training-free Neural Architecture Search through Variance of Knowledge of Deep Network Weights](../../CVPR2025/others/training-free_neural_architecture_search_through_variance_of_knowledge_of_deep_n.md)
- [\[CVPR 2025\] Subnet-Aware Dynamic Supernet Training for Neural Architecture Search](../../CVPR2025/others/subnet-aware_dynamic_supernet_training_for_neural_architecture_search.md)
- [\[ECCV 2024\] AttnZero: Efficient Attention Discovery for Vision Transformers](attnzero_efficient_attention_discovery_for_vision_transformers.md)
- [\[CVPR 2026\] InTrain: Intrinsic Trainability for Zero-Cost Neural Architecture Search](../../CVPR2026/others/intrain_intrinsic_trainability_for_zero-cost_neural_architecture_search.md)
- [\[ICML 2026\] AutoNumerics-Zero: Automated Discovery of State-of-the-Art Mathematical Functions](../../ICML2026/others/autonumerics-zero_automated_discovery_of_state-of-the-art_mathematical_functions.md)

</div>

<!-- RELATED:END -->
