---
title: >-
  [Paper Note] Prune2Drive: A Plug-and-Play Framework for Accelerating Vision-Language Models in Autonomous Driving
description: >-
  [CVPR 2026][Multimodal VLM][Paper Note] The first plug-and-play token pruning framework dedicated to multi-view autonomous driving VLMs. By utilizing T-FPS (Token-level Farthest Point Sampling) to maintain semantic and spatial diversity, combined with view-adaptive pruning rates to optimize token budgets across cameras, it achieves 6.40× prefill acceleration
tags:
  - CVPR 2026
  - Multimodal VLM
date: 2026-05-08
content_hash: abec1ff7eaea61b6
---
# Prune2Drive: A Plug-and-Play Framework for Accelerating Vision-Language Models in Autonomous Driving

**Conference**: CVPR 2026  
**arXiv**: [2508.13305](https://arxiv.org/abs/2508.13305)  
**Code**: [https://github.com/MinhaoXiong/Prune2Drive](https://github.com/MinhaoXiong/Prune2Drive)  
**Area**: Multimodal VLM  
**Keywords**: Multi-view VLM, Visual Token Pruning, Farthest Point Sampling, View-Adaptive, Autonomous Driving Acceleration  

## TL;DR

The first plug-and-play token pruning framework dedicated to multi-view autonomous driving VLMs. By utilizing T-FPS (Token-level Farthest Point Sampling) to maintain semantic and spatial diversity, combined with view-adaptive pruning rates to optimize token budgets across cameras, it achieves 6.40× prefill acceleration on DriveLM with only 10% tokens remaining and a performance drop of just 3%.

## Background & Motivation

1. **Computational Explosion in Multi-view VLMs**: Autonomous driving VLMs (e.g., DriveMM) process 6 surround-view camera inputs. With 729 tokens per image totaling >4000 visual tokens, the $O(n^2)$ attention complexity results in unacceptable inference latency.
2. **Limitations of Prior Work**: Existing pruning methods like FastV and SparseVLM are designed for single images, ignoring the spatial and semantic diversity of multi-view setups. Direct application leads to the loss of critical view information.
3. **Incompatibility with Efficient Attention**: Methods relying on attention weights (e.g., FastV) require reading attention matrices, which is incompatible with efficient implementations like FlashAttention.
4. **Positional Bias**: Attention-score-based methods tend to systematically retain tokens at specific positions, neglecting tokens with low attention but high semantic importance (e.g., distant vehicles).
5. **Unequal View Contributions**: Front-view cameras are significantly more critical for driving decisions than rear views, yet existing methods apply uniform pruning rates across all views.
6. **Urgent Real-time Requirements**: Autonomous driving is a latency-sensitive scenario where high inference latency directly impacts safety.

## Method

### Overall Architecture

Prune2Drive aims to solve the token explosion problem in multi-view autonomous driving VLMs. It inserts an inference-time pruning stage after the visual encoder output. First, T-FPS selects a diverse subset of tokens in the token embedding space. Then, an offline search determines the specific retention budget for each camera. Finally, the pruned tokens are fed into the LLM. This training-free mechanism does not modify model weights or require attention matrices. The workflow consists of two streams: an **offline one-time** view-adaptive pruning rate optimization (producing retention rates $\alpha_i$ for each camera) and an **inference-time** per-view T-FPS execution based on the assigned budgets.

```mermaid
mermaid
flowchart TD
    A["6-view Surround Images<br/>729 tokens per image"] --> B["Visual Encoder<br/>Per-view Visual Tokens"]
    B --> C["T-FPS Diversity Sampling<br/>Iterative selection via Cosine Distance"]
    subgraph OPT["View-Adaptive Pruning Rate Optimization (Offline)"]
        direction TB
        D["Small Val Set (500 samples)"] --> E["TPE Search<br/>max M(α)=R(α)−λP(α)"]
        E --> F["Per-view Retention Rate α_i<br/>High Front, Low Rear"]
    end
    F -. Assigned token budget .-> C
    C --> G["Pruned Tokens fed to LLM"]
    G --> H["Driving QA / Inference Output"]
```

### Key Designs

**1. T-FPS: Maintaining Semantic and Spatial Diversity via Farthest Point Sampling**

Existing single-image pruning methods rely on second-layer attention scores, which introduce positional bias and conflict with FlashAttention. Furthermore, objects with low attention but high semantic importance (e.g., distant cars) are often discarded. T-FPS adapts Farthest Point Sampling from point cloud processing to the token embedding space, replacing Euclidean distance with cosine distance. It iteratively selects tokens furthest from the already selected set $\mathcal{S}$ until the target count $\mathcal{K}$ is met. This ensures the subset covers a broad semantic and spatial distribution without relying on attention scores, making it compatible with FlashAttention. The computational overhead is minimal—only 0.02s for $N=729$, accounting for less than 0.1% of total FLOPs.

**2. View-Adaptive Pruning Rate: Automatic Budget Allocation**

Different views contribute unequally to driving decisions (e.g., front vs. rear). Prune2Drive treats the retention rate $\alpha_i$ for each view as an optimizable variable, defining the objective $\mathcal{M}(\boldsymbol{\alpha}) = R(\boldsymbol{\alpha}) - \lambda P(\boldsymbol{\alpha})$. Here, $R(\boldsymbol{\alpha})$ represents semantic similarity to ground truth, while $P(\boldsymbol{\alpha}) = \sum_{i=1}^{M} \alpha_i$ penalizes the total token count to encourage sparsity. Using TPE (Tree-structured Parzen Estimator) on a small 500-sample validation set, the search converges within 3 H100 GPU hours. The results confirm the intuition: front views receive higher retention rates, while rear and side views are compressed further.

**3. Theoretical Guarantee: Tighter Error Bounds**

The paper demonstrates that the combination of T-FPS (k-center greedy, approximating Hausdorff distance minimization) and view-adaptive budgeting provides a tighter error upper bound under the View-Weighted Lipschitz continuity assumption compared to uniform random sampling:

$$\sum_{i=1}^{M} w_i \cdot d_H(V_i, S_{i,\text{Prune2Drive}}) \leq \sum_{i=1}^{M} w_i \cdot d_H(V_i, S_{i,\text{baseline}})$$

This provides theoretical grounding for the empirical success of diversity sampling and view weighting.

**4. Plug-and-Play: Training-free and Attention-agnostic**

The framework is training-free and sits directly after the visual encoder. It is compatible with various VLMs such as LLaVA-OneVision-7B (DriveMM), InternVL2.5-8B (DriveLMM-o1), and LLaVA-1.5, requiring no retraining or access to attention matrices.

## Key Experimental Results

### Main Results: DriveLM Benchmark (DriveMM Model, 10% Token Retention)

| Method | Tokens/Img | Avg Score↑ | Prefill Accel | FLOPs |
|------|:---:|:---:|:---:|:---:|
| Vanilla | 729 | 59.1 | 1× | 100% |
| FastV | 72 | 54.1 | 5.78× | 14.2% |
| SparseVLM | 72 | 55.9 | 4.06× | 14.4% |
| PACT | 72 | 56.8 | — | — |
| **Prune2Drive** | **72** | **57.4** | **6.40×** | **13.4%** |

### Main Results: DriveLMM-o1 Benchmark (10% Token Retention)

| Method | Overall Reasoning↑ | Risk Accuracy↑ | Scene Understanding↑ |
|------|:---:|:---:|:---:|
| Vanilla (100%) | 74.2 | 73.01 | 75.99 |
| FastV | 65.3 | 65.37 | 66.43 |
| DART | 67.4 | 65.32 | 68.17 |
| **Prune2Drive** | **68.3** | **68.34** | **69.86** |

### Ablation Study

| Ablation Item | DriveLMM-o1 Overall↑ | Description |
|------|:---:|:---|
| Cos Distance (Default) | 68.3 | Optimal |
| L1 Distance | 68.3 | Almost equivalent |
| L2 Distance | 67.7 | Slightly lower |
| Min Distance (Nearest) | 63.0 | Severe degradation (-5.3), validates diversity principle |
| TPE (Default) | 68.3 | Best HPO |
| Grid Search | 67.3 | -1.0 difference |
| Evolutionary | 67.6 | -0.7 difference |

**Key Finding**: At 25% token retention on DriveLM, the Match Score reaches 34.0, surpassing the original model's 33.9. Moderate pruning acts as a regularizer by removing redundant or noisy tokens.

## Highlights & Insights

1. **First Multi-view Dedicated Pruning**: Moves beyond simple transfers of single-view methods to address spatial/semantic diversity and view contribution variance.
2. **Elegant T-FPS Design**: Adapts FPS from point cloud processing to token embeddings using cosine distance to guarantee semantic diversity with only 0.02s overhead.
3. **Automatic Discovery of View Importance**: TPE search automatically allocates budgets (Front > Rear) without requiring manual priors.
4. **Significant Industrial Value**: The 6.40× acceleration is highly relevant for the deployment of real-time autonomous driving systems.

## Limitations & Future Work

1. **Under-sampling of Uniform Textures**: Objects with large areas of similar textures (e.g., a large bus) might have too few tokens retained by T-FPS, leading to information loss.
2. **Random Initialization in T-FPS**: The random selection of the initial token may introduce slight variance, which was not explicitly reported.
3. **Scale Validation**: The method was primarily validated on 7B-8B VLMs; the pruning effectiveness on larger models (70B+) remains to be explored.
4. **Static Adaptation**: Retention rates are fixed per camera across all samples, without considering dynamic scene changes (e.g., highway vs. intersection).
5. **Decoding Phase Bottleneck**: KV cache reduction only significantly accelerates the prefill stage; decoding acceleration remains limited (1.04-1.09×).

## Related Work & Insights

### vs. FastV / SparseVLM / PACT (Single-view Pruning)
FastV relies on attention scores and suffers from positional bias. SparseVLM requires cross-modal attention weights. Prune2Drive’s T-FPS is attention-agnostic and specifically designed for multi-view scenarios. In extreme compression (64 tokens), Prune2Drive (94.6%) significantly outperforms SparseVLM (86.9%) and FastV (74.3%).

### vs. DriveMM / DriveLMM-o1 (AD Dedicated VLMs)
Prune2Drive acts as an orthogonal plug-and-play module for these models without requiring weight modification.

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Seeing Clearly, Reasoning Confidently: Plug-and-Play Remedies for Vision Language Model Blindness](seeing_clearly_reasoning_confidently_plug-and-play_remedies_for_vision_language_.md)
- [\[AAAI 2026\] Plug-and-Play Clarifier: A Zero-Shot Multimodal Framework for Egocentric Intent Disambiguation](../../AAAI2026/multimodal_vlm/plug-and-play_clarifier_a_zero-shot_multimodal_framework_for_egocentric_intent_d.md)
- [\[AAAI 2026\] LLMC+: Benchmarking Vision-Language Model Compression with a Plug-and-play Toolkit](../../AAAI2026/multimodal_vlm/llmc_benchmarking_vision-language_model_compression_with_a_plug-and-play_toolkit.md)
- [\[CVPR 2026\] TreeTeaming: Autonomous Red-Teaming of Vision-Language Models via Hierarchical Strategy Exploration](treeteaming_autonomous_red-teaming_of_vision-language_models_via_hierarchical_s.md)
- [\[ICCV 2025\] Fine-Grained Evaluation of Large Vision-Language Models in Autonomous Driving](../../ICCV2025/multimodal_vlm/fine-grained_evaluation_of_large_vision-language_models_in_autonomous_driving.md)

</div>

<!-- RELATED:END -->
