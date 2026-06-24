---
title: >-
  [Paper Note] Auto-DAS: Automated Proxy Discovery for Training-free Distillation-aware Architecture Search
description: >-
  [ECCV 2024][Model Compression][Knowledge Distillation] This paper proposes Auto-DAS, an automated proxy discovery framework based on evolutionary algorithms for training-free distillation-aware architecture search (DAS). By automatically discovering optimal proxy metrics within a search space composed of student intrinsic statistics and teacher-student interaction statistics, it bypasses the limitations of hand-crafted proxies. Auto-DAS achieves SOTA ranking correlations and…
tags:
  - "ECCV 2024"
  - "Model Compression"
  - "Knowledge Distillation"
  - "Architecture Search"
  - "Training-free Proxy"
  - "Evolutionary Algorithms"
  - "Automated Proxy Discovery"
date: 2026-05-08
content_hash: 13973406363a13b1
---

# Auto-DAS: Automated Proxy Discovery for Training-free Distillation-aware Architecture Search

**Conference**: ECCV 2024  
**Code**: [https://github.com/lliai/Auto-DAS](https://github.com/lliai/Auto-DAS)  
**Area**: Model Compression / Neural Architecture Search  
**Keywords**: Knowledge Distillation, Architecture Search, Training-free Proxy, Evolutionary Algorithms, Automated Proxy Discovery

## TL;DR

This paper proposes Auto-DAS, an automated proxy discovery framework based on evolutionary algorithms for training-free distillation-aware architecture search (DAS). By automatically discovering optimal proxy metrics within a search space composed of student intrinsic statistics and teacher-student interaction statistics, it bypasses the limitations of hand-crafted proxies. Auto-DAS achieves SOTA ranking correlations and search accuracies across various architectures and search spaces, including ResNet, ViT, and NAS-Bench-101/201.

## Background & Motivation

**Background**: Knowledge Distillation (KD) is a core technique in model compression that improves the performance of small student models by prompting them to mimic the behavior of large teacher models. Distillation-aware Architecture Search (DAS) further optimizes this process: given a teacher model, it searches for the optimal student architecture that benefits the most from distillation, which is significantly more effective than casually choosing a student architecture prior to distillation.

**Limitations of Prior Work**: Traditional DAS methods require complete distillation training for each candidate student architecture to evaluate its performance, incurring extremely high search costs (often requiring hundreds of GPU-days). The recent DisWOT method introduced a training-free proxy based on KD, which predicts the accuracy ranking after distillation with only a few forward passes, greatly accelerating the search. However, DisWOT suffers from two critical issues: (1) Its proxy metrics are hand-crafted, relying on researchers' intuitive understanding of distillation mechanisms, which limits the search space; (2) The hand-designed proxy, while effective on CNN architectures, generalizes poorly to novel architectures like Vision Transformers (ViTs).

**Key Challenge**: Training-free proxies must balance "prediction accuracy" and "design generalization." Hand-crafted designs can optimize prediction accuracy for specific architectures, but find it difficult to generalize across diverse architecture search spaces. How can we automatically discover proxy metrics that are both accurate and general?

**Goal**: (1) To automatically discover training-free distillation evaluation proxies, avoiding hand-crafted designs; (2) To ensure that the discovered proxies generalize robustly across different architecture types, such as CNNs and ViTs.

**Key Insight**: The authors observe that effective distillation proxies typically rely on two types of information: student intrinsic statistics (e.g., feature distribution, gradient information, reflecting the student's learning potential) and teacher-student interaction statistics (e.g., feature similarity, KL divergence, reflecting teacher-student compatibility). Based on this, the proxy metrics can be modeled as functions of these two types of statistics, allowing evolutionary algorithms to search for the optimal proxy in the function space.

**Core Idea**: Use evolutionary algorithms to automatically discover optimal training-free distillation evaluation proxies within a computational graph space constructed from student intrinsic statistics and teacher-student interaction statistics.

## Method

### Overall Architecture

The pipeline of Auto-DAS consists of three stages: (1) **Proxy Search Space Construction**—defining a computational graph space with student intrinsic statistics and teacher-student interaction statistics as inputs, incorporating various transformation and distance computation operations; (2) **Proxy Search**—utilizing an evolutionary algorithm (EA) to search for the optimal proxy in the search space, using the rank correlation (e.g., Kendall's Tau) between candidate proxies and ground-truth distillation accuracies as the fitness function; (3) **Distillation-aware Architecture Search**—evaluating candidate student architectures in a training-free manner using the discovered optimal proxy, and selecting the optimal student architecture for formal distillation training.

### Key Designs

1. **Proxy Search Space**:

    - **Function**: Defines all possible forms of distillation proxies.
    - **Mechanism**: The search space represents proxy functions via computational graphs. The input nodes consist of two categories: (a) Student intrinsic statistics—including feature maps and gradients from each student layer, Gram matrices of feature maps, BatchNorm statistics, singular value distributions of layer weights, etc.; (b) Teacher-student interaction statistics—including cosine similarity, KL divergence, and Centered Kernel Alignment (CKA) of teacher-student features at corresponding layers. Intermediate operation nodes include basic transformations (log, exp, abs, normalize, etc.) and network distance operators (Frobenius norm, nuclear norm, trace, etc.), inspired by prior proxy designs and KD loss functions. The final output node aggregates the intermediate computational results into a single scalar serving as the proxy score.
    - **Design Motivation**: To transform the hand-crafted design problem into a search problem. The search space design balances comprehensiveness (including sufficient statistics and operations) and feasibility (structural constraints of the computational graph guarantee that the output is a valid scalar value).

2. **Adaptive-Elite Selection Strategy**:

    - **Function**: Balance exploration and exploitation during the evolutionary search process.
    - **Mechanism**: Standard evolutionary algorithms usually retain a fixed proportion of elite individuals (e.g., top-20%). Auto-DAS proposes adaptively adjusting this elite ratio: increasing the elite pool in the early stage of the search (retaining more candidates to maintain diversity—exploration) and shrinking the elite pool in the later stage (focusing on high-fitness regions—exploitation). Specifically, the elite ratio $p$ decreases progressively following an annealing strategy with iteration $t$: $p(t) = p_{max} - (p_{max} - p_{min}) \cdot t / T$.
    - **Design Motivation**: A fixed elite ratio can easily lead to early convergence (over-exploitation) or late-stage divergence (over-exploration). The adaptive strategy explores the proxy space extensively early on and finely optimizes the most promising proxy forms later in the search.

3. **Multi-Architecture Generalization**:

    - **Function**: Ensure that the discovered proxies are effective across different architecture search spaces.
    - **Mechanism**: During the proxy search, the fitness function evaluates not only the rank correlation with distillation accuracy on a single architecture family (e.g., ResNet) but also the average correlation across multiple architecture families. This equips the searched proxies with cross-architecture predictive capabilities. During evaluation, a small set of representative architectures is distilled beforehand to obtain real accuracies (serving as the "ground truth" for the search), and then the candidate proxies compute rank correlations on these architectures.
    - **Design Motivation**: If a proxy is only effective on CNNs but fails on ViTs, its utility is severely limited. Through multi-architecture joint optimization, the discovered proxies naturally possess stronger generalization.

### Loss & Training

The "loss" during the proxy search phase is the Kendall's Tau ($\tau$) or Spearman rank correlation coefficient ($\rho$) between the candidate proxy and the ground-truth distillation accuracy, which the evolutionary algorithm aims to maximize. In the formal distillation phase, the standard KD training strategy is followed, with the loss being a weighted sum of hard-label cross-entropy and soft-label KL divergence. The search process is highly efficient: proxy evaluation requires only a single forward pass (taking around a few seconds), allowing the entire proxy search process to be completed on a single GPU within a few hours.

## Key Experimental Results

### Main Results

| Search Space | Metric | Auto-DAS | DisWOT (Prev. SOTA) | Gain |
|---|---|---|---|---|
| NAS-Bench-201 (CIFAR-10) | Kendall's τ | SOTA | Runner-up | Significant Improvement |
| NAS-Bench-201 (CIFAR-100) | Kendall's τ | SOTA | Runner-up | Significant Improvement |
| NAS-Bench-101 | Kendall's τ | SOTA | Runner-up | Significant Improvement |
| ResNet-family | Searched Student Top-1 Acc | Optimal | Runner-up | Improved |
| ViT-family | Searched Student Top-1 Acc | Optimal | Unsupported | - |

### Ablation Study

| Configuration | Key Metric | Description |
|---|---|---|
| Intrinsic Stats Only | Moderate τ | Lacks teacher-student interaction information |
| Interaction Stats Only | Moderate τ | Lacks student's self-contained information |
| Intrinsic + Interaction | Highest τ | Complementary information from both categories |
| Fixed Elite Ratio | Slightly Lower τ | Insufficient search exploration |
| Adaptive Elite | Highest τ | Better exploration-exploitation balance |

### Key Findings

- Both student intrinsic statistics and teacher-student interaction statistics are essential components of an effective proxy, providing complementary information.
- Automatically discovered proxies generalize far better than the hand-crafted DisWOT proxy on the ViT search space.
- The adaptive-elite selection strategy yields consistent search improvements.
- The proxy search cost is extremely low (a few hours on a single GPU), whereas formal distillation training requires dozens of GPU-days, yielding a search acceleration of over 100x.

## Highlights & Insights

- Treating "proxy design" itself as an optimization problem is a highly insightful meta-approach that circumvents the trial-and-error process of hand-crafting.
- The search space design leverages prior knowledge from proxy designs and KD loss functions effectively, leading to guided rather than blind searching.
- The adaptive-elite selection strategy, though simple, is highly effective and deserves to be applied to other evolutionary search scenarios.
- Auto-DAS + AttnZero (two works from the same team) form a series of contributions in "automated discovery," showcasing the power of evolutionary search across multiple AI design problems.

## Limitations & Future Work

- Proxy search still relies on the distillation training results of a small number of architectures as ground truth; for entirely new architecture search spaces, pre-distillation training is needed to acquire labels.
- The operational types in the search space are predefined, which might miss novel statistics or transformation operations.
- Validation has primarily focused on distillation for image classification, and the effectiveness of distillation-aware searches for tasks like detection and segmentation remains unverified.
- The interpretability of the searched proxies is limited; it is difficult to intuitively understand why a specific computational graph combination predicts distillation effects better.
- Proxy search under multi-teacher scenarios (where multiple teachers simultaneously distill into a student) has not yet been explored.

## Related Work & Insights

- **DisWOT** is the direct competitor and predecessor of this work, which first introduced the concept of KD-based training-free proxies.
- Training-free NAS proxy works like **ZenNAS and NASWOT** provide methodological foundations for proxy design.
- **NAS-Bench-101/201** provide standardized benchmarks for evaluating architecture search.
- Similar to AttnZero, Auto-DAS uses evolutionary algorithms to search for optimal components within structured search spaces, reflecting the methodology of "replacing design with search."
- The automated proxy discovery concept from this method can be extended to other scenarios that require proxy metrics (e.g., data quality assessment, model transferability prediction).

## Rating

- Novelty: ⭐⭐⭐⭐ Modeling proxy design as a search problem is creative; adaptive elite selection is a strong addition.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple architectures and search spaces, with comprehensive ablations.
- Writing Quality: ⭐⭐⭐ Clear motivation, but details of the search space could be more detailed.
- Value: ⭐⭐⭐⭐ Highly valuable reference for the training-free NAS and distillation-aware search domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Category Adaptation Meets Projected Distillation in Generalized Continual Category Discovery](category_adaptation_meets_projected_distillation_in_generalized_continual_catego.md)
- [\[CVPR 2026\] TAS-LoRA: Transformer Architecture Search with Mixture-of-LoRA Experts](../../CVPR2026/model_compression/tas-lora_transformer_architecture_search_with_mixture-of-lora_experts.md)
- [\[CVPR 2025\] DKDM: Data-Free Knowledge Distillation for Diffusion Models with Any Architecture](../../CVPR2025/model_compression/dkdm_data-free_knowledge_distillation_for_diffusion_models_with_any_architecture.md)
- [\[ECCV 2024\] PaPr: Training-Free One-Step Patch Pruning with Lightweight ConvNets for Faster Inference](papr_training-free_one-step_patch_pruning_with_lightweight_convnets_for_faster_i.md)
- [\[ICCV 2025\] Time-Aware Auto White Balance in Mobile Photography](../../ICCV2025/model_compression/time-aware_auto_white_balance_in_mobile_photography.md)

</div>

<!-- RELATED:END -->
