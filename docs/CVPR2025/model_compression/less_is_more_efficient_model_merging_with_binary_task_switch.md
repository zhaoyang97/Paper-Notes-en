---
title: >-
  [Paper Note] Less is More: Efficient Model Merging with Binary Task Switch
description: >-
  [CVPR 2025][Model Compression][Model merging] Controlled experiments reveal that task vectors exhibit an "impulse characteristic"—only parameters with magnitudes exceeding a threshold make positive contributions to the task. Based on this, the T-Switch method is proposed to binarize task vectors into three components: activation switches, polarity switches, and scaling knobs, achieving dynamic model merging performance significantly superior to existing baselines while requir…
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "Model merging"
  - "task vector binarization"
  - "parameter redundancy"
  - "dynamic merging"
  - "efficient storage"
date: 2026-05-08
content_hash: ea7aa3526e0f0faa
---

# Less is More: Efficient Model Merging with Binary Task Switch

**Conference**: CVPR 2025  
**arXiv**: [2412.00054](https://arxiv.org/abs/2412.00054)  
**Code**: None  
**Area**: Others  
**Keywords**: Model merging, task vector binarization, parameter redundancy, dynamic merging, efficient storage

## TL;DR
Controlled experiments reveal that task vectors exhibit an "impulse characteristic"—only parameters with magnitudes exceeding a threshold make positive contributions to the task. Based on this, the T-Switch method is proposed to binarize task vectors into three components: activation switches, polarity switches, and scaling knobs, achieving dynamic model merging performance significantly superior to existing baselines while requiring only 1-3% of the storage space.

## Background & Motivation

1. **Background**: Model merging is an efficient method that enables models to possess multi-task capabilities without additional training. By merging the parameter differences (task vectors) of multiple fine-tuned models, a multi-task model can be obtained.
2. **Limitations of Prior Work**: (a) There are massive redundant parameter conflicts among task vectors—parameter values of different tasks at the same location may conflict; (b) The overhead of storing the task vectors themselves is huge—the number of parameters in each task vector is close to the original model, requiring $K$ times the model size of storage for $K$ tasks.
3. **Key Challenge**: Dynamic merging (such as Twin-Merging) requires storing all task vectors for flexible combination, but full-precision storage cost is excessively high (e.g., 3.4GB for 8 tasks), whereas static merging (merging before use) suffers from limited performance due to conflicts.
4. **Goal**: To significantly reduce the storage overhead of task vectors while mitigating parameter conflicts.
5. **Key Insight**: Systematically controlled experiments reveal that parameters with smaller magnitudes in task vectors not only make no contribution to the task but also have negative impacts—discarding them actually improves performance. This "impulse activation" characteristic makes binarized approximation feasible.
6. **Core Idea**: Utilizing the impulse characteristics of task vectors to binarize them into three extremely lightweight components: mask + sign + scalar, achieving dynamic merging superior to full-precision using only 1-3% storage.

## Method

### Overall Architecture
Input: Pre-trained model $\boldsymbol{\theta}$ and $K$ fine-tuned models $\boldsymbol{\theta}_1,...,\boldsymbol{\theta}_K$, compute the task vectors $\boldsymbol{\tau}_i = \boldsymbol{\theta}_i - \boldsymbol{\theta}$. T-Switch compresses each task vector into a binary representation. During inference, for a target task $\mathcal{T}_i$, the approximate task vector is restored from a shared all-ones vector $\mathbf{U}$ through switch combinations and added to the pre-trained weights. The Auto-Switch extension further achieves retrieval-based automatic task switching.

### Key Designs

1. **Impulse Discard (P-Discard)**:
    - **Function**: Eliminating redundant parameters in task vectors based on parameter magnitudes.
    - **Mechanism**: Design an impulse activation function $g_m$ that retains parameters whose magnitudes exceed upper/lower thresholds (selected by ratio $\alpha$) and discards the rest. Controlled experiments show that when low-magnitude parameters are discarded (Discard Low), performance does not drop but increases (even exceeding the Individual baseline of separate fine-tuning), while discarding high-magnitude parameters (Discard High) causes a sharp performance drop. Compared to the random discard strategy of DARE, P-Discard shows a clearer advantage in merging scenarios—random discard decreases performance from the beginning, while P-Discard continuously improves as the discard rate increases up to $\alpha=0.7$.
    - **Design Motivation**: Intuitively, parameters that change significantly after fine-tuning are those that contribute to the task, whereas tiny fluctuations might be noise caused by label noise or outliers. Experiments rigorously validate this hypothesis.

2. **Binarizing Task Vectors (Bin-Discard → T-Switch)**:
    - **Function**: Further compressing the task vectors after P-Discard into binary representations.
    - **Mechanism**: After P-Discard, non-zero parameters retain only sign information (+1/-1) and are multiplied by a scaling coefficient to restore the norm of the original task vector. Specifically, it is decomposed into three components: (a) activation switch $\mathcal{S}_A^i = g_m(\boldsymbol{\tau}_i)$ as a binary mask; (b) polarity switch $\mathcal{S}_P^i = g_b(\boldsymbol{\tau}_i)$ as a binary sign; and (c) scaling knob $\lambda_i$ as a scalar. During inference, it is restored via $\hat{\boldsymbol{\theta}}_i = \boldsymbol{\theta} + \lambda_i \cdot \mathcal{S}_A^i \odot \mathcal{S}_P^i \odot \mathbf{U}$. Since the mask and sign each require only 1 bit, the storage is only 1-3% of full precision.
    - **Design Motivation**: Due to the impulse characteristic, the specific numerical values of non-zero parameters are less important than their presence and direction. Experiments validate that at a discard rate of 0.6-0.7, the binarized approximation even exceeds the performance of full-precision fine-tuned models.

3. **Auto-Switch Automatic Merging Mechanism**:
    - **Function**: Automatically determining which tasks' switches to use for test samples during inference.
    - **Mechanism**: No router training is needed. First, construct a feature query set $\mathcal{Q}_i$ for each task using a small amount of exemplar data (extracting features via the average-merged model). During inference, perform K-nearest neighbor search for the input $x$ in all query sets, and assign weights based on the proportion of each task in the nearest neighbors: $w_i(x) = |\mathcal{Q}_i \cap \mathcal{N}_x| / |\mathcal{N}_x|$. Then, simply linearly combine the task switches with weights.
    - **Design Motivation**: To avoid the training cost of parameterized routers and the retraining issue when new tasks arrive. The retrieval-based method exploits the separability of tasks in the feature space and is completely training-free.

### Loss & Training
T-Switch itself does not require training—it is directly calculated from the task vectors of existing fine-tuned models. Auto-Switch requires a small amount of exemplar data to construct the query set, but this is also a training-free process of feature extraction and nearest-neighbor indexing.

## Key Experimental Results

### Main Results (ViT-B/32 + 8 vision tasks)

| Method | Type | Storage (MB) | Average Accuracy |
|------|------|---------|-----------|
| Individual (Separate Fine-Tuning) | - | - | 91.01 |
| Task-Arithmetic | Static | - | 70.23 |
| TIES-Merging | Static | - | 72.73 |
| AdaMerging++ | Fixed | - | 81.02 |
| Twin-Merging | Dynamic | 3474.2 | 83.07 |
| EMR-Merging | Dynamic | 461.0 | 88.74 |
| **T-Switch (Ours)** | **Dynamic** | **57.0** | **90.98** |
| **Auto-Switch (Ours)** | **Dynamic** | **58.6** | **90.25** |

T-Switch achieves 90.98% accuracy with only 57MB of storage, reaching close to the Individual baseline's 91.01% and significantly exceeding all merging baselines.

### Ablation Study

| Discard Rate α | DARE-Random | P-Discard | Bin-Discard |
|---------|-------------|-----------|-------------|
| 0.1 | 69.06 | 69.31 | ~69.2 |
| 0.4 | 68.06 | 70.41 | ~70.3 |
| 0.7 | 66.56 | 72.23 | ~72.1 |
| 0.8 | 66.09 | 70.99 | ~70.8 |

### Key Findings
- **Impulse characteristic is the key insight**: Low-magnitude parameters are not only redundant but also place negative constraints on performance—discarding them improves both fine-tuning and merging performance. This stands in stark contrast to DARE's random discard.
- **Binarization is virtually lossless**: At discard rates of 0.6-0.7, Bin-Discard performs almost identically to P-Discard, and even exceeds the full-precision Individual baseline, demonstrating that the exact numerical values of task vectors are far less important than their directions.
- **Extremely high storage efficiency**: T-Switch requires only 57MB vs. 3474MB for Twin-Merging, a 60x reduction in storage while boosting performance by 8 percentage points.
- **LoRA Compatibility**: It is equally effective on low-rank task vectors from LoRA fine-tuning, indicating that the impulse characteristic is a general attribute of parameter fine-tuning.

## Highlights & Insights
- **Counter-intuitive discovery of "Less is More"**: It is commonly believed that retaining more parameter information is better, but this paper proves that the vast majority of parameters in task vectors are noise—discarding them actually improves performance. This discovery shifts the understanding of parameter conflicts in model merging.
- **Binarization as a denoising method**: Traditional binarization aims at compression; however, binarization in this work acts as a form of denoising—obtaining cleaner task representations by retaining only directional information of parameters and discarding magnitude noise.
- **Simplicity and elegance of Auto-Switch**: Replacing learnable routers with KNN retrieval not only requires no training and scales flexibly, but also leverages the ultra-low storage advantages of binarization—truly achieving highly efficient dynamic merging.

## Limitations & Future Work
- Currently validated mainly on ViT-B/32 (a relatively small model); effectiveness on larger-scale models (e.g., ViT-L/14, LLMs) needs verification.
- The choice of discard rate $\alpha$ still requires manual adjustment. Can it be determined adaptively?
- Auto-Switch's KNN retrieval requires exemplar data, and methods under a completely zero-shot scenario remain unexplored.
- Only classification tasks are considered. Do task vectors for tasks such as generation and detection also possess impulse characteristics?

## Related Work & Insights
- **vs. TIES-Merging**: TIES merges by resetting small parameters and resolving sign conflicts, but it still operates in full-precision space and relies on manual coefficients; T-Switch is more thorough by directly binarizing, which automatically maintains sign consistency.
- **vs. DARE**: DARE performs random discard + scaling, which inherently does not utilize parameter magnitude information. P-Discard selectively discards low-magnitude parameters, and the performance gap widens as the discard rate increases.
- **vs. Twin-Merging**: Twin-Merging stores full-precision task vectors and learns a router, incurring a massive storage overhead (3.4GB). T-Switch achieves equivalent or even better performance using binary representation (57MB) + retrieval.
- **vs. EMR-Merging**: EMR-Merging merges first and then prunes, which is still limited by initial merging conflicts. T-Switch directly stores each task independently after binarization, fundamentally avoiding conflicts.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The discovery of impulse characteristics and the idea of binarized merging are completely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Rigorously designed controlled experiments, thorough ablation, and validated LoRA compatibility.
- Writing Quality: ⭐⭐⭐⭐ Strong logic in the derivation chain from observation to methodology.
- Value: ⭐⭐⭐⭐⭐ 60x storage compression coupled with performance improvement, holding significant practical value for multi-task deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Task Singular Vectors: Reducing Task Interference in Model Merging](task_singular_vectors_reducing_task_interference_in_model_merging.md)
- [\[ICCV 2025\] Task Vector Quantization for Memory-Efficient Model Merging](../../ICCV2025/model_compression/task_vector_quantization_for_memory-efficient_model_merging.md)
- [\[ICCV 2025\] FREE-Merging: Fourier Transform for Efficient Model Merging](../../ICCV2025/model_compression/free-merging_fourier_transform_for_efficient_model_merging.md)
- [\[CVPR 2025\] TADFormer: Task-Adaptive Dynamic Transformer for Efficient Multi-Task Learning](tadformer_task-adaptive_dynamic_transformer_for_efficient_multi-task_learning.md)
- [\[ICLR 2026\] Cut Less, Fold More: Model Compression through the Lens of Projection Geometry](../../ICLR2026/model_compression/cut_less_fold_more_model_compression_through_the_lens_of_projection_geometry.md)

</div>

<!-- RELATED:END -->
