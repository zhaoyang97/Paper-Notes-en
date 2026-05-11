---
title: >-
  [Paper Note] FREE-Merging: Fourier Transform for Efficient Model Merging
description: >-
  [ICCV 2025][Multimodal VLM][Model Merging] This paper is the first to identify the frequency-domain manifestation of task interference in model merging. It proposes FR-Merging…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "Model Merging"
  - "Fourier Transform"
  - "Task Interference"
  - "Frequency-Domain Analysis"
  - "Lightweight Expert"
date: 2026-05-08
content_hash: 06872d9526e05d1d
---

# FREE-Merging: Fourier Transform for Efficient Model Merging

**Conference**: ICCV 2025
**arXiv**: [2411.16815](https://arxiv.org/abs/2411.16815)
**Code**: [GitHub](https://github.com/Zhenghe123/FREE-Merging)
**Area**: Multimodal VLM
**Keywords**: Model Merging, Fourier Transform, Task Interference, Frequency-Domain Analysis, Lightweight Expert

## TL;DR

This paper is the first to identify the frequency-domain manifestation of task interference in model merging. It proposes FR-Merging, which removes low-frequency interference via high-pass filtering to construct a high-quality merged backbone, and combines it with lightweight task expert modules (FREE-Merging) to achieve an optimal performance–cost trade-off across vision, language, and multimodal tasks.

## Background & Motivation

With the proliferation of open-source fine-tuned models, model merging has emerged as an efficient approach to consolidating multiple task-specific models into a single multi-task model, avoiding the high cost of joint multi-task training and data privacy concerns. However, existing methods face two core challenges:

**Task interference degrades performance**: Conflicts exist among fine-tuned weights from different tasks. Existing methods (e.g., Task Arithmetic, Ties-Merging, DARE) operate solely in the spatial domain (pruning, sign conflict resolution, etc.) and overlook frequency-domain interference. This paper is the first to reveal that task interference is significant in the frequency domain and concentrated in low-frequency regions, and that spatial-domain methods can barely mitigate frequency-domain interference (reducing frequency-domain amplitude variance by only 1–5%, whereas FR-Merging achieves 20–24%).

**Conflict between performance and deployment cost**: Introducing task experts improves performance, but existing methods (EMR-Merging, Twin-Merging) require storing substantial task-specific knowledge (2–3% of parameters) while neglecting backbone optimization.

The core insight of this paper is that low-frequency signals capture global structural information and are more likely to contain task-specific information that causes inter-task interference, whereas high-frequency signals represent fine-grained variations with stronger generalization capacity. Directly filtering out the low-frequency component can therefore substantially reduce task interference while preserving performance.

## Method

### Overall Architecture

FREE-Merging is a two-stage approach:
- **Stage 1 — FR-Merging** (training-free): A high-pass filter is applied to each task vector $v_k = \theta_k - \theta_{\text{pre}}$ to remove low-frequency interference signals, and the filtered vectors are merged to obtain a high-quality backbone network.
- **Stage 2 — Expert Extraction** (training-free): Lightweight task experts comprising only ~1% of the parameter count are extracted from the task vectors and dynamically assigned at inference time via a router.

### Key Designs

1. **FR-Merging (Frequency-Domain High-Pass Filtering Merge)**:

    - Function: Applies a Fourier transform to each task vector, filters out low-frequency interference regions, and applies the inverse transform.
    - Mechanism: An ideal high-pass filter is applied to task vector $v(x,y)$:
    $G(x,y) = \mathcal{F}^{-1}\{H(\eta, \gamma) \cdot \mathcal{F}\{v(x,y)\}\}$
      where $H(\eta, \gamma) = \begin{cases} 1, & \sqrt{\eta^2 + \gamma^2} \geq D_0 \\ 0, & \sqrt{\eta^2 + \gamma^2} < D_0 \end{cases}$ and $D_0$ is the cutoff frequency.
      Merging coefficients are computed via mean normalization of task vectors:
    $\lambda_i = \mathbb{E}(v_i) \left(\sum_{j=1}^{K} \mathbb{E}(v_j)\right)^{-1}$
    - Design Motivation: Fine-tuned weights occupy different positions in the loss landscape, and linear interpolation can easily fall into high-loss regions. Removing low-frequency signals reduces model discrepancies, making the merged result more likely to reside within a loss basin. Experiments confirm that removing low-frequency components incurs only marginal per-task performance loss (diagonal), while substantially improving generalization (off-diagonal).

2. **Lightweight Task Expert Extraction**:

    - Function: Selects the top-d% parameters with the largest magnitude changes from task vectors as task experts, requiring only ~1% of the parameter count.
    - Mechanism: The highest-magnitude parameters are selected and rescaled:
    $e(v_i) = \mu_i M(v_i, d), \quad \mu_i = -\frac{\mathbb{E}(M(v_i, d)) \cdot \log(d)}{\lambda_i \cdot \mathbb{E}(v_i)}$
      where $M(v_i, d)$ denotes the top-d% parameters and $\mu_i$ is a scaling factor ensuring output consistency.
    - Design Motivation: Theorem 5.1 provides a theoretical guarantee that a merged model cannot simultaneously retain all the capabilities of the original models without introducing additional information (No Free Lunch). Low-frequency signals encode task-specific information; however, directly storing the low-frequency region requires an inverse FFT at every inference step, making it impractical. Parameter magnitude is therefore used as a proxy.

3. **MoE Router for Dynamic Dispatch**:

    - Function: Dynamically selects active task experts based on the input at inference time.
    - Mechanism: $\theta_* = \theta_m + \sum_{i=1}^{K} w_i e_i$, where $[w_1, \ldots, w_K] \leftarrow \arg\max(R(x))$ and $R$ is a lightweight MLP router.
    - Design Motivation: Inspired by Mixture-of-Experts, dynamic routing eliminates the overhead of loading all experts for every input.

### Loss & Training

Both FR-Merging and expert extraction are **entirely training-free**, requiring only a one-time computation. The router can be implemented as a simple MLP or other classifier. The overall pipeline executes the FFT operation only once during merging; at inference, only a lightweight router is added.

## Key Experimental Results

### Main Results

8-task vision merging (average accuracy on ViT-B/32 / ViT-L/14):

| Method | Extra Cost | ViT-B/32 Avg | ViT-L/14 Avg | Notes |
|--------|-----------|-------------|-------------|-------|
| Individual | — | 90.5 | 94.2 | Upper bound |
| Task Arithmetic | None | 70.1 | 84.5 | Baseline |
| Ties-Merging | None | 73.6 | 86.0 | |
| PCB-Merging | None | 75.8 | 86.9 | Prev. SOTA (training-free) |
| **FR-Merging** | **None** | **78.1** | **88.3** | **+2.3 / +1.4** |
| EMR-Merging | 3% storage | 87.7 | 92.8 | |
| Twin-Merging | 2% storage | 87.8 | 92.7 | |
| **FREE-Merging** | **1% storage** | **89.7** | **93.7** | **Lowest storage, highest performance** |

### Ablation Study

Frequency-domain interference quantification (ViT-B/32 amplitude variance):

| Method | Freq. Amplitude Variance | Variance Reduction | Notes |
|--------|--------------------------|--------------------|-------|
| Task Arithmetic | 0.059 | — | Baseline |
| DARE | 0.057 | ↓3% | Spatial-domain method |
| Ties-Merging | 0.058 | ↓2% | Spatial-domain method |
| PCB-Merging | 0.056 | ↓5% | Spatial-domain method |
| **FR-Merging** | **0.045** | **↓24%** | Frequency-domain method is markedly more effective |

Cross-domain validation (language models; average over RoBERTa / T0-3B / Qwen-14B):

| Method | RoBERTa | T0-3B | Qwen-14B | Average |
|--------|---------|-------|----------|---------|
| Task Arithmetic | 66.65 | 63.91 | 66.40 | 65.65 |
| FR-Merging | 70.02 | 66.88 | 68.00 | 68.30 |
| EMR-Merging | 74.20 | 67.11 | 70.98 | 70.76 |
| FREE-Merging | **80.16** | **68.68** | **72.78** | **73.87** |

### Key Findings

- In the 30-task vision merging setting, FR-Merging improves training-free performance from 48.88% (Task Arithmetic) to 53.90%, while FREE-Merging reaches 79.67%.
- High-pass filtering substantially improves generalization (off-diagonal) with only marginal per-task performance loss (diagonal), validating that low-frequency signals primarily encode task-specific information.
- FREE-Merging requires only 1% additional parameter storage yet outperforms EMR-Merging and Twin-Merging, which require 2–3%.
- The method generalizes effectively to PEFT settings (LoRA, IA³), demonstrating broad applicability.

## Highlights & Insights

1. **Breakthrough frequency-domain perspective**: This is the first work to introduce frequency-domain analysis into model merging, revealing the concentration of task interference in low-frequency regions and providing a novel theoretical lens for understanding model merging.
2. **Simplicity and effectiveness**: FR-Merging requires only a single FFT operation with a theoretical time complexity of $O(nm\log m)$, and demands no training data or gradient computation.
3. **Theoretical guarantee**: Theorem 5.1 rigorously proves a "No Free Lunch" theorem for model merging, theoretically justifying the necessity of incorporating task experts.
4. **Cross-modal generalization**: Effectiveness is validated across CV, NLP, and multimodal tasks, spanning models of varying scales from ViT to LLaMA.

## Limitations & Future Work

- The ideal high-pass filter introduces ringing artifacts; Butterworth or Gaussian high-pass filters could be explored for smoother frequency transitions.
- The cutoff frequency $D_0$ is a fixed hyperparameter; different layers or tasks may benefit from adaptive cutoff strategies.
- Router accuracy directly affects FREE-Merging performance and may degrade when task boundaries are ambiguous.
- In very large-scale merging scenarios (e.g., 30+ models), the linear growth in the number of experts remains a storage burden.

## Related Work & Insights

- The frequency-domain analysis paradigm is transferable to other parameter-space operations, such as frequency-domain pruning in knowledge distillation and model compression.
- The lightweight expert extraction approach can be combined with PEFT methods such as LoRA and Adapter for more efficient multi-task deployment.
- The insight that "low frequency encodes task-specific information while high frequency encodes generalization capacity" sheds light on the nature of parameter changes during fine-tuning.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First to introduce frequency-domain analysis into model merging; the discovery of low-frequency interference is highly pioneering.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers vision/language/multimodal tasks, full fine-tuning and PEFT, and merging settings with 8 and 30 tasks.
- Writing Quality: ⭐⭐⭐⭐ — Logically clear with fair and systematic experimental comparisons and intuitive figures.
- Value: ⭐⭐⭐⭐⭐ — Highly practical; the training-free approach lowers the barrier to model merging and has significant implications for large model deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] RobustMerge: Parameter-Efficient Model Merging for MLLMs with Direction Robustness](../../NeurIPS2025/multimodal_vlm/robustmerge_parameter-efficient_model_merging_for_mllms_with_direction_robustnes.md)
- [\[CVPR 2026\] DC-Merge: Improving Model Merging with Directional Consistency](../../CVPR2026/multimodal_vlm/dc-merge_improving_model_merging_with_directional_consistency.md)
- [\[CVPR 2026\] Label-Free Cross-Task LoRA Merging with Null-Space Compression](../../CVPR2026/multimodal_vlm/label-free_cross-task_lora_merging_with_null-space_compression.md)
- [\[ICCV 2025\] Exploiting Vision Language Model for Training-Free 3D Point Cloud OOD Detection](exploiting_vision_language_model_for_training-free_3d_point_cloud_ood_detection_.md)
- [\[ICCV 2025\] AirCache: Activating Inter-Modal Relevancy KV Cache Compression for Efficient Large Vision-Language Model Inference](aircache_activating_inter-modal_relevancy_kv_cache_compression_for_efficient_lar.md)

</div>

<!-- RELATED:END -->
