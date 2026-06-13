---
title: >-
  [Paper Note] SMoFi: Step-wise Momentum Fusion for Split Federated Learning on Heterogeneous Data
description: >-
  [AAAI 2026][Optimization][Split Federated Learning] This paper proposes SMoFi, a framework that synchronizes the momentum buffers of surrogate models on the server side at every SGD step within Split FL…
tags:
  - "AAAI 2026"
  - "Optimization"
  - "Split Federated Learning"
  - "Data Heterogeneity"
  - "Momentum Alignment"
  - "Non-IID"
  - "Convergence Acceleration"
date: 2026-05-08
content_hash: d64408535d51a059
---

# SMoFi: Step-wise Momentum Fusion for Split Federated Learning on Heterogeneous Data

**Conference**: AAAI 2026
**arXiv**: [2511.09828](https://arxiv.org/abs/2511.09828)  
**Code**: N/A  
**Area**: Federated Learning / Distributed Optimization
**Keywords**: Split Federated Learning, Data Heterogeneity, Momentum Alignment, Non-IID, Convergence Acceleration

## TL;DR
This paper proposes SMoFi, a framework that synchronizes the momentum buffers of surrogate models on the server side at every SGD step within Split FL, effectively mitigating gradient divergence caused by non-IID data. SMoFi achieves up to 7.1% accuracy improvement and up to 10.25× convergence speedup.

## Background & Motivation

### State of the Field

**Background**: Split Federated Learning (Split FL) partitions a model into a client-side and a server-side component, leveraging the server's computational power to share the training workload — making it particularly suitable for resource-constrained edge devices. In the SFLV1 framework, the server maintains multiple surrogate models trained in parallel, which are aggregated at the end of each communication round. Existing methods such as FedAvg, FedProx, FedAvgM, and SlowMo address data heterogeneity by modifying loss functions or improving aggregation strategies.

### Limitations of Prior Work

**Limitations of Prior Work**: Data heterogeneity (non-IID) is a central challenge in FL. Inconsistent local data distributions across clients cause divergence in the update directions of server-side surrogate models, degrading global model accuracy and slowing convergence after aggregation. Existing methods (FedAvgM, SlowMo) operate on momentum only between communication rounds, resulting in coarse-grained control. FedNAG's periodic aggregation can even degrade performance in certain settings.

### Root Cause

**Key Challenge**: While momentum (SGDM) improves final model accuracy, it paradoxically slows convergence under non-IID data — because momentum helps each local model converge more effectively toward its own local optimum, amplifying the divergence of update directions. Transforming momentum from a "decelerating factor" into a tool for accelerating convergence is therefore a key challenge.

### Approach

**Goal**: To exploit the natural advantage of Split FL, where the server directly controls the surrogate models, by imposing consistency constraints at every SGD step rather than between communication rounds. **Key Insight**: Synchronize the momentum buffers of all surrogate models on the server side at each step, guiding all models to update in a globally consistent direction. **Core Idea**: Step-wise momentum fusion replaces each model's local momentum with a globally aligned average momentum, requiring zero client-side modifications and incurring zero additional communication overhead.

## Method

### Overall Architecture
SMoFi builds upon the parallel update framework of SFLV1. The server maintains $|\mathcal{J}^n|$ surrogate server-side models trained in parallel. After each SGD step, SMoFi synchronizes the momentum buffers of all optimizers on the server side, replacing each local momentum with the global average momentum. Model parameters are aggregated via weighted averaging at the end of training.

### Key Designs

1. **Step-wise Momentum Alignment**:

    - **Function**: Aligns the momentum directions of all surrogate models at every SGD update step.
    - **Mechanism**: At each step $\tau$, the momentum of each surrogate optimizer is replaced with the globally aligned momentum $\bar{m}_s^{(n,\tau)}$: $m_{s,j}^{(n,\tau+1)} = \beta \bar{m}_s^{(n,\tau)} + \nabla \mathcal{L}_{\mathcal{B}_j^\tau}(\mathcal{W}_{s,j}^{(n,\tau)})$. The unified momentum constrains each model's update direction toward the global optimum rather than individual local optima.
    - **Design Motivation**: Compared to FedAvgM/SlowMo, which operate on momentum only between communication rounds (at the granularity of multiple epochs), SMoFi operates at the per-batch step level, providing tighter constraints and faster response to gradient divergence.

2. **Staleness Factor**:

    - **Function**: Handles inconsistencies in training progress across clients.
    - **Mechanism**: Since clients have varying local step counts $T_j$ due to differing data volumes, clients that finish training early no longer contribute fresh momentum. SMoFi records the final momentum of completed clients and incorporates it into the alignment computation with a polynomial decay weight $s_\alpha = (\tau - |T_j| + 1)^\alpha,\ \alpha < 0$, ensuring that the number of momentum signals participating in alignment remains $|\mathcal{J}^n|$ throughout training.
    - **Design Motivation**: Ignoring the momentum of completed clients reduces the number of alignment signals over time, weakening the constraint. The staleness factor retains these signals via decay weights while diminishing the influence of stale information.

3. **Client-Transparent Plug-in Design**:

    - **Function**: Enables deployment with zero client-side modifications.
    - **Mechanism**: SMoFi operates entirely on the server side without altering client-side code, introducing additional communication overhead, or posing privacy risks. It can be applied as a plug-in on top of any FL method, including FedAvg, FedProx, and FedNAR.
    - **Design Motivation**: A distinctive property of Split FL is that the server directly controls the training process of surrogate models, enabling fine-grained server-side constraints without requiring client cooperation.

### Theoretical Guarantees
The paper provides an $\mathcal{O}(1/N)$ convergence guarantee under strong convexity assumptions, proving that momentum alignment does not compromise convergence and that per-step alignment effectively reduces the variance of model updates.

## Key Experimental Results

### Main Results

Evaluated on CIFAR-10, CIFAR-100, and Tiny-ImageNet under non-IID settings with a Dirichlet distribution ($\alpha=0.1$).

| Method | CIFAR-10 Acc. | CIFAR-100 Acc. | Tiny-ImageNet Acc. |
|------|:---:|:---:|:---:|
| FedAvg | 77.16% | 48.10% | 33.43% |
| + FedAvgM | 79.19% | 50.28% | 33.58% |
| + SlowMo | 76.54% | 50.96% | 33.82% |
| + **SMoFi** | **81.82%** | **53.83%** | **39.73%** |

Convergence speedup (rounds to reach target accuracy vs. FedAvg): 4.61×–5.54× on CIFAR-10, up to **10.25×** on Tiny-ImageNet.

### Ablation Study

| Configuration | Tiny-ImageNet Acc. | Notes |
|------|:---:|------|
| SFLV1 (baseline) | 33.43% | No momentum alignment |
| SFLV2 (sequential) | 34.72% | Avoids divergence but incurs high latency |
| FedNAR + SMoFi | 39.73% | Best result with SMoFi plug-in |
| SMoFi (16 rounds) | Reaches target accuracy | SFLV1 requires >400 rounds |

### Key Findings
- SMoFi yields greater gains with larger models and more clients, consistent with the scaling properties required for real-world deployment.
- Effective across optimizers (SGDM/NAG/Adam/AdamW) and architectures (VGG/MobileNet/ResNet/DenseNet).
- On Tiny-ImageNet, SMoFi reaches the target accuracy in only 16 rounds, compared to over 400 rounds for SFLV1.
- Plug-in usage consistently improves baseline methods: FedAvg+SMoFi, FedProx+SMoFi, and FedNAR+SMoFi each gain 3%–7% over their respective baselines.

## Highlights & Insights
- The design is minimal yet highly effective: synchronizing only the server-side momentum buffers yields up to 10.25× convergence acceleration.
- The paper identifies the "deceleration paradox" of non-IID + momentum and reframes momentum as an acceleration tool — an insight of broader significance for distributed optimization.
- The plug-in design is highly versatile and practically applicable across diverse FL methods.
- The paper fully exploits the unique property of Split FL in which the server directly controls surrogate model training; this fine-grained optimization paradigm is generalizable to other split learning variants.

## Limitations & Future Work
- The convergence analysis relies on strong convexity assumptions; applicability to the non-convex settings of practical deep networks is limited.
- The staleness factor $\alpha$ requires manual tuning (default: $-0.1$); adaptive adjustment may be preferable.
- Validation is primarily limited to image classification; evaluation on NLP, speech, and other modalities remains limited.
- The potential benefit of aligning client-side momentum has not been explored.

## Related Work & Insights
- **vs. FedAvgM/SlowMo**: These methods operate on momentum only between communication rounds, providing coarse-grained control. SMoFi operates at every step, yielding tighter constraints and a 5.91% improvement on Tiny-ImageNet.
- **vs. SFLV2**: Sequential training avoids divergence but introduces high latency. SMoFi preserves parallel training while controlling divergence through momentum alignment.
- **vs. MergeSFL**: MergeSFL improves time efficiency via adaptive batch sizes but requires more communication rounds than SMoFi.
- The staleness factor design offers insights transferable to handling stale gradients in asynchronous distributed training.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Simple yet effective step-wise momentum alignment concept.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive validation across multiple datasets, FL methods, optimizers, and architectures.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation with well-integrated theory and experiments.
- **Value**: ⭐⭐⭐⭐ Immediately applicable to the Split FL community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Data Heterogeneity and Forgotten Labels in Split Federated Learning](data_heterogeneity_and_forgotten_labels_in_split_federated_learning.md)
- [\[ICLR 2026\] FedDAG: Clustered Federated Learning via Global Data and Gradient Integration for Heterogeneous Environments](../../ICLR2026/optimization/feddag_clustered_federated_learning_via_global_data_and_gradient_integration_for.md)
- [\[AAAI 2026\] Tackling Resource-Constrained and Data-Heterogeneity in Federated Learning with Double-Weight Sparse Pack](tackling_resource-constrained_and_data-heterogeneity_in_federated_learning_with_.md)
- [\[ICLR 2026\] Incentives in Federated Learning with Heterogeneous Agents](../../ICLR2026/optimization/incentives_in_federated_learning_with_heterogeneous_agents.md)
- [\[AAAI 2026\] FedPM: Federated Learning Using Second-order Optimization with Preconditioned Mixing of Local Parameters](fedpm_federated_learning_using_second-order_optimization_with_preconditioned_mix.md)

</div>

<!-- RELATED:END -->
