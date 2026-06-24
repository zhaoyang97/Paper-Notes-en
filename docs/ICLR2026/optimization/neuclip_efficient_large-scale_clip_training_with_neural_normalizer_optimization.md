---
title: >-
  [Paper Note] NeuCLIP: Efficient Large-Scale CLIP Training with Neural Normalizer Optimization
description: >-
  [ICLR 2026][Optimization][Contrastive Loss] NeuCLIP reformulates contrastive loss as a "minimization problem with auxiliary variables" using convex conjugation. It then employs variational analysis to collapse $n$ sample-wise auxiliary variables into a lightweight Neural Partitioning Network (NPN) to predict log-partition functions. This eliminates optimization errors that scale with the dataset/batch size ratio ($O(n/B)$) found in FastCLIP, consistently outperforming existin…
tags:
  - "ICLR 2026"
  - "Optimization"
  - "Contrastive Loss"
  - "Partition Function Estimation"
  - "Convex Conjugate"
  - "Variational Analysis"
  - "Neural Partitioning Network"
  - "Alternating Optimization"
date: 2026-05-08
content_hash: 1305c8c48fc9e8a3
---

# NeuCLIP: Efficient Large-Scale CLIP Training with Neural Normalizer Optimization

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=WoMMSVZHfP](https://openreview.net/forum?id=WoMMSVZHfP)  
**Code**: [https://github.com/Optimization-AI/NeuCLIP](https://github.com/Optimization-AI/NeuCLIP)  
**Area**: Optimization / Contrastive Learning / CLIP Training  
**Keywords**: Contrastive Loss, Partition Function Estimation, Convex Conjugate, Variational Analysis, Neural Partitioning Network, Alternating Optimization  

## TL;DR
NeuCLIP reformulates contrastive loss as a "minimization problem with auxiliary variables" using convex conjugation. It then employs variational analysis to collapse $n$ sample-wise auxiliary variables into a lightweight Neural Partitioning Network (NPN) to predict log-partition functions. This eliminates optimization errors that scale with the dataset/batch size ratio ($O(n/B)$) found in FastCLIP, consistently outperforming existing methods on CLIP training scales from millions to billions.

## Background & Motivation
**Background**: The core of CLIP training is contrastive loss, where the gradient involves a "partition function" (normalization term). This term requires comparing each positive pair against all negative samples, theoretically depending on the entire dataset. Mainstream approaches follow two paths: one is OpenCLIP/SigLIP, which uses extremely large batches to approximate the partition function within the batch (GPU-intensive); the other includes SogCLR/FastCLIP, which maintain global contrastive loss by tracking a partition function estimator for each sample, updated via moving averages in a block-coordinate fashion.

**Limitations of Prior Work**: The resource-efficient path of FastCLIP suffers from a critical flaw: its optimization error includes an amplification factor of $O(n/B)$ (where $n$ is dataset size and $B$ is batch size). Larger datasets or smaller batches worsen the error, contradicting the goal of training on billion-scale data with limited GPUs. SigLIP bypasses the partition function with a sigmoid loss but still requires large batches. AmorLIP attempts to use a lightweight network to predict the partition function, but its training objective still contains non-linear functions of the partition function, leading to a **"chicken and egg" dilemma** (requiring the partition function to train the network, which is itself used to estimate the partition function) and necessitating an extra EMA network.

**Key Challenge**: Maintaining sample-wise estimators (coordinate-based) is the root cause of scale-dependent error, but direct neural network prediction introduces biased non-linear gradient dependencies. The problem is how to amortize partition function prediction with a network while maintaining a unified objective and unbiased gradients.

**Goal**: To provide a principled framework that jointly optimizes the "encoder" and the "partition prediction network" within the **same objective**, ensuring gradients have no non-linear dependence on the partition function.

**Key Insight**: **[Convex Conjugate + Variational Analysis]** First, use convex conjugation to rewrite the contrastive loss (containing $\log$) as an equivalent minimization problem over an auxiliary variable $\alpha$, where the optimal solution is exactly the log-partition function. Second, apply the "interchangeability of infimum and integral" theorem from variational analysis to transform the minimization over $n$ variables $\alpha_i$ into a minimization over a compact function space $\alpha(\cdot)$.

## Method

### Overall Architecture
NeuCLIP reshapes robust global contrastive loss in three steps: (1) Rewriting via convex conjugation to explicitly expose the log-partition function as an "optimizable variable"; (2) Using variational analysis to replace sample-wise variables with a Neural Partitioning Network (NPN), injecting inductive bias based on the optimal solution's structure; (3) Employing alternating optimization to rotate stochastic gradient updates between the CLIP encoder block $(w,\tau)$ and the NPN block $(W_1,W_2)$, combined with acceleration techniques such as multiple NPN updates and periodic restarts.

```mermaid
flowchart TD
    A[Robust Global Contrastive Loss<br/>contains log partition function] -->|Convex Conjugate| B["min_α: exp(-α)·(ε+g) + α - 1<br/>α* = log partition function"]
    B -->|Variational Analysis Thm1| C[Sample-wise α_i minimization<br/>collapsed into function α(·) minimization]
    C -->|Inductive Bias: Feed-forward + LSE pooling| D[NPN: W1 for Image, W2 for Text]
    D --> E[Unified Objective Eq.12<br/>Gradient linear w.r.t. partition function]
    E -->|Alternating Optimization| F1[Block 1: Update Encoders w, τ]
    E -->|Alternating Optimization| F2[Block 2: Multiple updates for NPN W1, W2]
    F2 -->|Restart every Tr steps| G[Reset W1, W2 with mini-batch embeddings]
```

### Key Designs

**1. Rewriting contrastive loss via convex conjugate to make the partition function an optimizable variable**: The contrastive loss for a single image anchor is $F(w,\tau;x_i)=\log(\varepsilon+g_1(w,\tau;i,S))$, where $g_1$ is the average exponential similarity of the sample against all negative samples. Since $f(\cdot)=-\log(\cdot)$ is a convex function, using the Fenchel conjugate $f(x)=\max_y\, y\cdot x - f^*(y)$ (where $f^*(y)=-\log(-y)-1$) and substituting $\alpha=-\log(-y)$, the loss is rewritten as $\min_\alpha\,\{\exp(-\alpha)\cdot(\varepsilon+g_1)+\alpha-1\}$. The optimal solution $\alpha^*=\log(\varepsilon+g_1)$ is precisely the log-partition function. The advantage is that the partition function, previously hidden inside the $\log$ and causing non-linear reciprocal terms in gradients, is now "isolated" as an independent optimization variable $\alpha$. The objective is linear with respect to $g_1$, allowing standard SGD without estimation bias. This is a special case of Optimized Certainty Equivalent (OCE), and SogCLR’s moving average update can be derived back from this reformulation via stochastic block mirror descent.

**2. Collapsing $n$ auxiliary variables into one network via variational analysis**: Maintaining $\{\alpha_{1,i},\alpha_{2,i}\}$ per sample is the source of the $O(n/B)$ error. The authors apply a classical theorem from variational analysis (Rockafellar & Wets, Thm 14.60): $\inf_{\alpha(\cdot)\in\mathcal{F}}\int f(x,\alpha(x))\,\mu(dx)=\int \inf_{\alpha}f(x,\alpha)\,\mu(dx)$. This means "pointwise minimization before integration" is equivalent to "minimization over a function space." The revised contrastive loss fits the right-hand side, allowing the sample-wise $\min_{\alpha_{1,i}}$ to be replaced by a minimization over a function $\alpha_1(\cdot)$, parameterized by a neural network. Thus, partition function prediction shifts from "$n$ independent coordinates" to "generalization by a shared network," removing the $O(n/B)$ error scaling.

**3. Designing NPN with inductive bias based on optimal solution structure**: Instead of a generic MLP, the NPN follows the structure of the optimal solution in Eq.9: $\alpha_1^*=\log(\varepsilon+\frac{1}{|S|-1}\sum_j \exp((e_{1,i}^\top e_{2,j}-e_{1,i}^\top e_{2,i})/\tau))$. Since $e_{1,i},e_{2,i}$ are provided by encoders, the only missing information is "all $e_{2,j}$ embeddings." The NPN is defined as $\alpha_1(x_i)=\log(\varepsilon+\frac{1}{m}\sum_{j'=1}^m \exp((\cos(e_{1,i},W_{1,j'})-e_{1,i}^\top e_{2,i})/\tau))$, which passes the encoder output through a feed-forward layer $W_1\in\mathbb{R}^{d\times m}$ followed by log-sum-exp pooling. $W_{1,j}$ can be interpreted as "prototype embeddings" summarizing all text data. The text side uses a symmetric $W_2$. The encoder and NPN share the unified objective Eq.12.

**4. Alternating Optimization + Multiple Updates + Periodic Restarts**: Parameters are split into $(w,\tau)$ and $(W_1,W_2)$ for alternating updates (simultaneous updates were found unstable as NPN depends on encoder outputs). Two acceleration techniques are used: (i) **Multiple NPN updates**—the NPN is updated $T_u=10$ times per batch before updating the encoder, allowing the lightweight NPN to keep pace with the shifting encoder; (ii) **Periodic restarts**—every $T_r=500$ steps, $W_1$ is reset using random text embeddings $\{e_{2,i}\}$ and $W_2$ with image embeddings, bridging the convergence gap between the NPN and the evolving encoders. Convergence to an $\varepsilon$-stationary point is theoretically proven in $T=O(\varepsilon^{-4})$ iterations.

## Key Experimental Results

**Setup**: CLIP trained on 8×H100 with Transformer text encoders and ViT/ResNet image encoders; 5 datasets ranging from CC3M (2.7M) to DFN-1B (1B), processing 100M–3B samples; NPN hidden dimension $m=4096$, $T_r=500$, $T_u=10$. AdamW for CLIP, AdaGrad for NPN. Evaluated on Datacomp 38-task average, ImageNet variants, and Retrieval.

### Main Results (Datacomp Average, higher is better)

| Method | CC3M | CC12M | DFN-14M | DFN-192M | DFN-1B |
| :--- | :--- | :--- | :--- | :--- | :--- |
| OpenCLIP | 21.84 | 27.91 | 37.78 | 54.58 | 56.25 |
| FastCLIP | 24.74 | 31.50 | 38.45 | 54.72 | 56.68 |
| SigLIP | 22.19 | 28.60 | 37.23 | 54.26 | 56.32 |
| AmorLIP | 22.89 | 29.86 | 37.53 | 53.83 | 56.24 |
| **NeuCLIP** | **25.08** | **31.89** | **39.16** | **54.90** | **57.34** |

NeuCLIP is optimal across all 5 datasets, with training curves showing its advantage becomes more pronounced in the **later stages** of training.

### Ablation Study

| Ablation Item | Comparison | Conclusion |
| :--- | :--- | :--- |
| Training Objective | Unified vs. Separated (AmorLIP style) | Unified objective is superior |
| NPN Architecture | Inductive bias single layer vs. Random MLP | Inductive bias is more effective |
| Restart Frequency $T_r$ | 0.02k / 0.1k / 0.5k / 2.5k / ∞ | $T_r=500$ is optimal (39.16) |
| Update Steps $T_u$ | 1 / 5 / 10 / 20 / 50 | $T_u=10$ is optimal; too many cause overfitting |

Partition function estimation error (MSE, lower is better):

| Setting | OpenCLIP | FastCLIP | NeuCLIP |
| :--- | :--- | :--- | :--- |
| Error increase at smaller batch (512 vs 1024) | 8.2 | 6.2 | **0.7** |
| Error increase with larger dataset (1.37M→13.7M) | 12.8 | 9.4 | **1.9** |

### Key Findings
- NeuCLIP's estimation error of the partition function **hardly deteriorates with smaller batches or larger datasets**, validating that it eliminates the $O(n/B)$ flaw of FastCLIP.
- Both the unified objective and NPN inductive bias are essential: the former avoids the "chicken and egg" problem, while the latter is easier to train and more accurate than a generic MLP.
- $T_r$ and $T_u$ have "sweet spots": Frequent restarts make the NPN degrade into a mini-batch estimator, while too many updates cause overfitting to the current batch.

## Highlights & Insights
- **Transforming "Estimation" into "Optimization"**: The convex conjugate is the linchpin. It transforms the partition function into an explicitly optimizable variable within the objective, rendering gradients linear and unbiased.
- **Inductive Bias from Optimal Solutions**: The NPN is not an arbitrary MLP but mimics the structure of the optimal solution (log-sum-exp over prototypes). This makes it parameter-efficient and interpretable.
- **Unified Narrative**: The proof that SogCLR's moving average update is a special case of this reformulation elevates NeuCLIP as a generalized upgrade to existing methods rather than a disconnected approach.

## Limitations & Future Work
- The NPN introduces additional hyperparameters ($m, T_r, T_u$) and requirements for two optimizers (AdamW and AdaGrad). Sweet spots for $T_r$ and $T_u$ may need re-tuning for different datasets.
- Image encoders scaled up to ViT-B/16 and data to 1B, but very large models (ViT-L/H) or very large batch sizes were not extensively tested to see if the gap with OpenCLIP persists.
- On DFN-192M/1B, the performance gap between methods narrows to less than 1 point, suggesting diminishing returns of optimization accuracy when data quality is high.

## Related Work & Insights
- **Global Contrastive Loss Lineage**: SogCLR (Yuan 2022) introduced sample-wise estimators; FastCLIP (Wei 2024) added temperature optimization and distributed scheduling. NeuCLIP is the first to use variational analysis to fundamentally remove the $O(n/B)$ error factor.
- **Amortized Neural Networks**: TempNet (Qiu 2024) predicted sample-wise temperatures, and AmorLIP (Sun 2025) predicted partition functions. NeuCLIP’s unified objective provides a template for making amortized network training unbiased.
- **Inspiration**: The strategy of "rewriting difficult terms as optimizable variables via convex conjugation and collapsing sample-wise variables into networks" can be applied to other problems with full-dataset dependencies, such as large-vocabulary softmax or energy-based models.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — The combination of convex conjugation and variational analysis to transform partition function estimation into neural prediction under a unified objective is elegant and theoretically sound.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive coverage across 5 datasets (up to 1B), strong baselines, and detailed ablations. Lacks verification on extremely large models.
- **Writing Quality**: ⭐⭐⭐⭐ — Logical progression from motivation to theoretical rewriting and implementation. Dense math may be challenging for non-optimization experts.
- **Value**: ⭐⭐⭐⭐ — Provides a plug-and-play framework for CLIP training under constrained resources or large data.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] A Memory-Efficient Hierarchical Algorithm for Large-scale Optimal Transport Problems](a_memory-efficient_hierarchical_algorithm_for_large-scale_optimal_transport_prob.md)
- [\[ICLR 2026\] Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models](bi-lora_efficient_sharpness-aware_minimization_for_fine-tuning_large-scale_model.md)
- [\[ICLR 2026\] FrontierCO: Real-World and Large-Scale Evaluation of Machine Learning Solvers for Combinatorial Optimization](frontierco_real-world_and_large-scale_evaluation_of_machine_learning_solvers_for.md)
- [\[ICLR 2026\] From Sequential to Parallel: Reformulating Dynamic Programming as GPU Kernels for Large-Scale Stochastic Combinatorial Optimization](from_sequential_to_parallel_reformulating_dynamic_programming_as_gpu_kernels_for.md)
- [\[ICLR 2026\] ViTSP: Guiding Large-Scale Traveling Salesman Problem Solving with Vision-Language Models](vitsp_a_vision_language_models_guided_framework_for_solving_large-scale_travelin.md)

</div>

<!-- RELATED:END -->
