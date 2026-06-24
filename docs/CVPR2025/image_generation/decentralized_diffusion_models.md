---
title: >-
  [Paper Note] Decentralized Diffusion Models
description: >-
  [CVPR 2025][Image Generation][Decentralized Training] Decentralized Diffusion Models (DDM) proposes a method to distribute diffusion model training across completely isolated compute clusters. By independently training expert models on data partitions and integrating them using a lightweight router during inference, the authors prove that this ensemble precisely optimizes the same global Flow Matching objective as a single model, outperforming a single large model on a FLOP-f…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Decentralized Training"
  - "Flow Matching"
  - "Mixture of Experts"
  - "Diffusion Models"
  - "Large-scale Training"
date: 2026-05-08
content_hash: 8528209b8eb4b800
---

# Decentralized Diffusion Models

**Conference**: CVPR 2025  
**arXiv**: [2501.05450](https://arxiv.org/abs/2501.05450)  
**Code**: [https://decentralizeddiffusion.github.io/](https://decentralizeddiffusion.github.io/)  
**Area**: Diffusion Models / Image Generation  
**Keywords**: Decentralized Training, Flow Matching, Mixture of Experts, Diffusion Models, Large-scale Training

## TL;DR

Decentralized Diffusion Models (DDM) proposes a method to distribute diffusion model training across completely isolated compute clusters. By independently training expert models on data partitions and integrating them using a lightweight router during inference, the authors prove that this ensemble precisely optimizes the same global Flow Matching objective as a single model, outperforming a single large model on a FLOP-for-FLOP basis.

## Background & Motivation

**Background**: Diffusion models have achieved breakthrough results in fields such as image generation, video modeling, and robotic control, but the demand for training compute continues to grow. Stable Diffusion 1.5 consumed over 6,000 A100 GPU days, while Meta's Movie Gen was trained on 6,114 H100 GPUs. Large-scale training relies on high-bandwidth interconnected centralized clusters, driving up infrastructure costs.

**Limitations of Prior Work**: (1) Data-parallel training requires gradient synchronization at every step, demanding high-bandwidth network interconnection that only expensive centralized monolithic clusters can support; (2) large clusters face system-level challenges such as power supply issues, hardware failures, and network bottlenecks, resulting in a fragile, tightly-coupled system; (3) academic researchers can rarely afford such scale of computational resources, making the barrier to entry for training diffusion models increasingly high.

**Key Challenge**: Training high-quality diffusion models requires massive compute and data, yet the centralized training paradigm demands that all GPUs reside in the same high-bandwidth network, which is both expensive and fragile. Can training be decentralized across multiple independent nodes, similar to federated learning?

**Goal**: Design a scalable decentralized diffusion model training framework, allowing independent compute clusters/data centers to train individually with zero communication and subsequently combine during inference.

**Key Insight**: The key insight is that the marginal vector field of Flow Matching can be naturally decomposed into a linear combination of expert fields over multiple data subsets. This means that when diffusion models independently trained on different data partitions are ensembled, they are theoretically equivalent to a single model trained on the entire dataset.

**Core Idea**: Cluster the dataset into K partitions, train a diffusion expert model completely independently on each partition (with zero cross-model communication), and train a lightweight router to predict the relevance of each expert. During inference, expert predictions are linearly combined using the router's weights to reconstruct the global Flow Matching objective.

## Method

### Overall Architecture

DDM training consists of three steps: (1) **Data Partitioning**—extract image features using DINOv2 and partition the dataset into K semantically coherent subsets through multi-stage clustering; (2) **Expert Training**—independently train a standard Flow Matching diffusion model (e.g., DiT) on each data subset, with zero communication between experts; (3) **Router Training**—train a small DiT classifier to predict which data subset a given noisy sample $x_t$ belongs to. During inference, the router assigns weights to each expert, and the weighted combination of expert predictions yields the final output.

### Key Designs

1. **Decentralized Flow Matching (DFM) Target**:

    - Function: Theoretically prove that decentralized training is equivalent to global optimization
    - Mechanism: The marginal vector field of standard Flow Matching is $u_t(x_t) = \int_{x_0} u_t(x_t|x_0) p_t(x_t|x_0) q(x_0) / p_t(x_t) dx_0$. After partitioning the data into K disjoint subsets $\{S_1, ..., S_K\}$, it can be proven that the marginal vector field decomposes into: $u_t(x_t) = \sum_{k=1}^{K} \frac{p_{t,S_k}(x_t)}{p_t(x_t)} \cdot v_{k,t}(x_t)$, where $\frac{p_{t,S_k}(x_t)}{p_t(x_t)}$ is the router weight (the posterior probability that $x_t$ originates from subset $S_k$), and $v_{k,t}(x_t)$ is the vector field prediction of the $k$-th expert. Each expert independently optimizes the standard Flow Matching loss $\|v_{\theta,t}(x_t) - u_t(x_t|x_0)\|^2$.
    - Design Motivation: This mathematical decomposition serves as the theoretical foundation of the paper, ensuring that decentralized training does not suffer from representational capacity loss—the ensembled model theoretically optimizes the exact same objective as a globally-trained model.

2. **Classification-Based Router Training**:

    - Function: Predict the relevance of each expert to the current input during inference
    - Mechanism: The router is a small DiT-B (158M parameters) that takes a noisy sample $x_t$ and timestep $t$ as inputs and outputs a K-dimensional probability distribution. The training objective is a standard cross-entropy classification loss—given a training sample $x_0$ and its corresponding cluster label $k$, $x_t$ is obtained by adding noise under the same noise schedule, and the router is trained to predict the correct cluster label. The router is trained independently of all experts, adding only about 4% overhead in training FLOPs. It uses a DiT architecture with a learnable CLS token, decoded via a linear head into cluster logits.
    - Design Motivation: The router training is completely decoupled from the experts and does not require end-to-end gradient propagation (unlike traditional MoE), allowing the entire system to run in a fully distributed manner. The computational overhead of the small router is negligible.

3. **Top-1 Expert Selection Inference Strategy**:

    - Function: Efficiently select the most relevant expert during inference
    - Mechanism: Although theoretically a weighted combination of all experts is required to precisely match the global vector field, in practice, most experts are irrelevant to a given input (the router assigns near-zero weights). The authors systematically compared multiple inference strategies: Full (weighted sum of all experts), Top-k (selecting top k), Sample (sampling a single expert by probability), Nucleus Sampling, etc. Experiments show that **Top-1 selection** (using only the single expert with the highest router weight) performs best in terms of FID while maintaining the lowest FLOPs. This means that during inference, DDM's inference compute is nearly identical to that of a single model.
    - Design Motivation: Top-1 selection is akin to sparse activation in MoE, saving compute and improving practical performance. This is likely because each expert's prediction on its specialized data sub-distribution is more accurate than a weighted average. The Sample strategy yields massive variance and performs extremely poorly.

### Distillation

For deployment scenarios, DDM can distill K experts into a single dense model. Using the cluster label of each training sample to select the corresponding expert as the teacher, the student model learns to mimic the teacher's predictions. The distillation is trained with 1/4 batch size, matching the performance of a single model trained from scratch using only 1/3 FLOPs.

## Key Experimental Results

### Main Results (DDM vs Monolith, DiT XL/2 Architecture)

| Method | Dataset | FID↓ | CLIP-FID↓ | GFLOPs | Notes |
|------|--------|------|-----------|--------|------|
| Monolith | ImageNet | 12.81 | 5.58 | 308 | Standard single model |
| DDM 8-expert Top-1 | ImageNet | **9.84** | **5.48** | 334 | 8-expert Top-1 inference |
| Monolith | LAION | — | — | 308 | Standard single model |
| DDM 8-expert Top-1 | LAION | **Lower** | **Lower** | 334 | 200k steps = Monolith 800k steps |

### Ablation Study (Comparison of Inference Strategies, 8-expert ImageNet)

| Inference Strategy | GFLOPs | FID↓ | CLIP-FID↓ | Notes |
|---------|--------|------|-----------|------|
| Monolith | 308 | 12.81 | 5.58 | Baseline |
| Full (8) | 2490 | 10.52 | 5.83 | Weighted sum of all experts |
| Top-1 | 334 | **9.84** | **5.48** | Best strategy |
| Top-2 | 642 | 10.31 | 5.74 | More experts perform worse |
| Sample-1 | 334 | 157.05 | 51.17 | Random sampling is highly unstable |
| Oracle | 308 | 10.46 | 5.83 | Selecting expert using GT labels |

### Key Findings

- **8-expert is the optimal configuration**: In comparisons across 4/8/16 experts, 8 experts consistently performed best. 4-expert models lack sufficient capacity, whereas 16-expert models suffer from insufficient training due to the too small batch size per expert.
- **DDM outperforms the single model FLOP-for-FLOP**: After 800k steps, the ImageNet FID is 6.08 for DDM vs 8.49 for the single model (a 28% reduction). On LAION, DDM's performance at 200k steps exceeds that of the single model at 800k steps, which corresponds to a **4x training speedup**.
- **Top-1 selection unexpectedly outperforms Full ensemble**: This is likely because experts are more precise on their specialized data sub-distributions compared to the weighted average.
- **Feature-based data clustering is far superior to random partitioning**: DINO feature clustering enables experts to learn their respective sub-distributions more efficiently, whereas random partitioning results in data lacking semantic focus for each expert.
- **Distillation is effective**: After distillation, the dense model achieves equivalent performance to training from scratch with only 1/3 of the training FLOPs (FID 7.76 vs 7.82).

## Highlights & Insights

- **Perfect convergence of theory and practice**: The DFM objective is naturally derived from the mathematical formulation of Flow Matching instead of being a heuristic design. This theoretical result is elegant and significant—decentralized training loses no representational capacity.
- **Enormous practical value of decentralized training**: Training 8 experts (3B parameters each) on separate 16-GPU nodes for 6.5 days is sufficient, removing the necessity of expensive centralized supercomputers for large-scale diffusion model training. Academic labs can train high-quality models using dispersed cloud computing resources.
- **Implicit performance gains from specialization**: DDM not only achieves decentralization but also gains superior performance over a single generalist model due to expert specialization on data sub-distributions. This resembles the increased parameterization concept of MoE but bypasses joint training.
- **Extremely simple engineering integration**: The barrier to implementing DDM is remarkably low—requiring only data clustering, a standard diffusion training framework, and a lightweight router. Almost all existing diffusion training infrastructures can be reused directly.

## Limitations & Future Work

- Even with Top-1 during inference, all expert model parameters must still be loaded into memory, resulting in high memory requirements (K times that of a single model).
- Although distillation addresses deployment concerns, it increases total training costs, and the distilled quality may not match the original ensemble.
- Currently, only image generation tasks are verified; more complex diffusion applications like video generation and 3D generation remain untried.
- The quality of data clustering significantly impacts the final performance, but determining the optimal number of clusters and strategies still requires empirical tuning.
- Theoretically, each expert only sees 1/K of the data, which may lead to insufficient coverage of rare patterns in the data (e.g., uncommon classes).
- The potential value for privacy applications is substantial (experts can be trained locally on different data sources), but dedicated privacy analysis has not yet been performed.

## Related Work & Insights

- **vs DiLoCo**: DiLoCo balances local training and periodic global synchronization through inner and outer optimization loops, which still requires some communication. DDM achieves completely zero-communication independent training, making it highly suitable for extreme distributed scenarios.
- **vs Branch-Train-Merge**: BTM trains data experts and merges parameters (in NLP). Instead of merging parameters, DDM ensembles them during inference, using a router for intelligent selection. DDM has a stronger theoretical foundation.
- **vs Diffusion Soup**: Diffusion Soup merges models by averaging the weights of fine-tuned models, which is an integration in the parameter space. DDM performs ensemble in the prediction space, preserving each expert's independence and yielding better performance.
- **vs MoE (Mixtral, DeepSeek-V3)**: MoE routes tokens inside the model and requires end-to-end training. DDM's routing occurs between models, allowing each part to be trained completely independently, which drastically reduces system complexity.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The theoretical derivation of the DFM objective is the core contribution. Proving that decentralization equals global optimization is a milestone achievement.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Dual ImageNet and LAION datasets, massive ablations (inference strategies, number of experts, clustering methods, distillation), scaling up to 24B parameters.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear theoretical derivation, systematic experiments, and a helpful accompanying blog post.
- Value: ⭐⭐⭐⭐⭐ Significantly lowers the barrier to training high-quality diffusion models, offering major practical significance for both academia and industry.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Heterogeneous Decentralized Diffusion Models](../../CVPR2026/image_generation/heterogeneous_decentralized_diffusion_models.md)
- [\[CVPR 2025\] Diff2Flow: Training Flow Matching Models via Diffusion Model Alignment](diff2flow_training_flow_matching_models_via_diffusion_model_alignment.md)
- [\[CVPR 2025\] Erasing Undesirable Influence in Diffusion Models (EraseDiff)](erasing_undesirable_influence_in_diffusion_models.md)
- [\[CVPR 2025\] Diffusion-4K: Ultra-High-Resolution Image Synthesis with Latent Diffusion Models](diffusion-4k_ultra-high-resolution_image_synthesis_with_latent_diffusion_models.md)
- [\[CVPR 2025\] The Art of Deception: Color Visual Illusions and Diffusion Models](the_art_of_deception_color_visual_illusions_and_diffusion_models.md)

</div>

<!-- RELATED:END -->
