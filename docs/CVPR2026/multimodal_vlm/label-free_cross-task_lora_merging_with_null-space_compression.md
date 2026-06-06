---
title: >-
  [Paper Note] Label-Free Cross-Task LoRA Merging with Null-Space Compression
description: >-
  [CVPR 2026][Multimodal VLM][Model Merging] Motivated by the observation that the null-space ratio of the down-projection matrix $\mathbf{A}$ decreases during LoRA fine-tuning and is strongly correlated with task performa…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Model Merging"
  - "LoRA"
  - "Null-Space Compression"
  - "Label-Free"
  - "Cross-Task"
date: 2026-05-08
content_hash: 002c0ee4f3f8e60f
---

# Label-Free Cross-Task LoRA Merging with Null-Space Compression

**Conference**: CVPR 2026
**arXiv**: [2603.26317](https://arxiv.org/abs/2603.26317)  
**Code**: [GitHub](https://github.com/wonyoung01/nsc_merging)  
**Area**: Multimodal VLM
**Keywords**: Model Merging, LoRA, Null-Space Compression, Label-Free, Cross-Task

## TL;DR

Motivated by the observation that the null-space ratio of the down-projection matrix $\mathbf{A}$ decreases during LoRA fine-tuning and is strongly correlated with task performance, this paper proposes NSC Merging — a label-free, task-agnostic LoRA merging method that achieves state-of-the-art results across 20 heterogeneous vision tasks, 6 NLI tasks, and VLM benchmarks.

## Background & Motivation

Model merging combines independently fine-tuned checkpoints into a single multi-task model without joint training. In the era of foundation models, LoRA fine-tuning has become the de facto standard, making LoRA merging an increasingly important research direction.

Existing gradient-guided merging methods (e.g., AdaMerging) use output entropy minimization as a proxy objective to estimate merging coefficients. While effective on classification tasks, they face two fundamental limitations:
1. **Inapplicable to regression tasks**: Entropy is only meaningful for classification; tasks such as depth estimation and surface normal prediction cannot leverage this signal.
2. **Not scalable to LLMs/VLMs**: Entropy must be computed at every generated token, incurring a cost that grows linearly with sequence length.

**Key Challenge**: A merging signal is needed that applies to both classification and regression tasks without relying on output logits.

**Key Observation**: During LoRA fine-tuning, the null space of the down-projection matrix $\mathbf{A}$ is systematically compressed — i.e., an increasing proportion of input activations fall within the adapter's projection subspace. This **null-space compression** is strongly negatively correlated with task performance, making it a task-agnostic merging signal.

## Method

### Overall Architecture

1. Fine-tune the base model independently with LoRA for each task.
2. Pre-compute the Gram inverse matrix $(A_kA_k^\top)^{-1}$ for each adapter.
3. Optimize layer-wise merging coefficients $\{\lambda_k^\ell\}$ using the NSC objective.
4. Output the merged multi-task model.

### Key Designs

1. **Null-Space Ratio Definition and Compression Dynamics**:
    - **Function**: Provides a task-agnostic proxy signal for performance.
    - **Mechanism**: For a LoRA update $\Delta W = BA$, the null-space ratio is defined as $\omega_k^\ell(\mathbf{z}) = \frac{\|\text{Proj}_{\mathcal{N}(A_k^\ell)}(\mathbf{z})\|_2}{\|\mathbf{z}\|_2}$, measuring the proportion of input activations discarded by the adapter. This ratio decreases monotonically during training (null-space compression), and exhibits a strong negative correlation with task performance — for both classification and regression tasks.
    - **Design Motivation**: Since the LoRA rank is small (e.g., 16 vs. 768 dimensions), the adapter subspace covers only ~2.1% of the feature space. Null-space compression indicates that the adapter has learned to better capture task-relevant activations, making it a reliable performance proxy. Crucially, this is an input-driven signal that does not depend on output logits.

2. **NSC Merging Objective**:
    - **Function**: Learns layer-wise merging coefficients without labels.
    - **Mechanism**: Minimizes the average null-space ratio across all tasks: $\min_{\{\lambda_k^\ell\}} \frac{1}{K}\sum_{k=1}^K \mathbb{E}_{\mathbf{x} \sim \mathcal{D}_k}[\Omega_k(\mathbf{x}; \Theta_{merge})]$, where $\Omega_k$ is the mean null-space ratio across target layers. The merged model parameters are given by $W_0^\ell + \sum_k \lambda_k^\ell B_k^\ell A_k^\ell$.
    - **Design Motivation**: The null-space ratio is computed purely from adapter geometry, making it applicable to any task type. For LLMs/VLMs, only the activations of input tokens are required, so computational cost is independent of sequence length.

3. **Fast NSC: Gram Inverse Caching**:
    - **Function**: Substantially reduces computational overhead.
    - **Mechanism**: The null-space ratio admits the equivalent form $\omega_k(\mathbf{z}) = \sqrt{1 - \frac{\mathbf{z}^\top A_k^\top(A_kA_k^\top)^{-1}A_k\mathbf{z}}{\|\mathbf{z}\|_2^2}}$. Since $\mathbf{z}$ and $A_k\mathbf{z}$ are already computed during inference, only the small matrix $(A_kA_k^\top)^{-1}$ (of dimension equal to the LoRA rank) needs to be pre-cached, avoiding the construction of the full null-space projection matrix.
    - **Target Layer Selection**: The NSC objective is computed only over the last quarter of transformer blocks to balance efficiency and performance.

### Loss & Training

- **Optimizer**: AdamW; lr = 0.001 (vision) / 0.0003 (LLM/VLM)
- **Initialization**: $\lambda$ initialized to 0.4
- **Iterations**: 100 steps (vision) / 500 steps (LLM/VLM)
- **Data**: Unlabeled validation sets only

## Key Experimental Results

### Main Results — 20 Heterogeneous Vision Tasks (ViT-B)

| Method | NYUD-v2 (4 tasks) | PASCAL (5 tasks) | Taskonomy (11 tasks) | Overall Avg. |
|--------|-------------------|------------------|----------------------|--------------|
| Task Arithmetic | ~46% | ~62% | ~103% | 77.2% |
| TIES | ~47% | ~62% | ~102% | 77.3% |
| KnOTS-TIES | ~45% | ~62% | ~102% | 76.6% |
| RobustMerge | ~69% | ~85% | ~100% | 89.9% |
| **NSC (Ours)** | ~**75%** | ~**87%** | ~**100%** | **92.0%** |

*(Values are performance normalized to single-task fine-tuning.)*

### LLM Experiments (LLaMA-3-8B, 6 NLI Tasks)

| Method | MNLI | QNLI | SNLI | RTE | SICK | SciTail | Avg. |
|--------|------|------|------|-----|------|---------|------|
| TA | 92.8 | 86.8 | 93.3 | 93.6 | 83.8 | 95.0 | 90.9 |
| AdaMerging | 94.3 | 84.8 | 92.5 | 92.1 | 89.2 | 84.8 | 89.6 |
| RobustMerge | 94.3 | 88.1 | 93.7 | 93.6 | 83.0 | 94.5 | 91.2 |
| **NSC (Ours)** | 94.9 | 88.3 | 92.8 | 91.3 | **91.2** | **95.1** | **92.3** |

### Ablation Study

| Configuration | Description | Effect |
|---------------|-------------|--------|
| Full-layer NSC | Compute over all LoRA layers | Best performance, highest cost |
| Last 1/4 layers | Only the last quarter of transformer blocks | Near full-layer quality, substantially more efficient |
| Last 1 layer | Only the final block | Noticeable performance drop |
| Input IDs only | No image/text content required | Still effective on LLMs |

### Key Findings

- NSC achieves the largest gains on heterogeneous mixed classification+regression tasks (+2.1% over RobustMerge on NYUD-v2), as competing methods struggle with regression tasks.
- AdaMerging underperforms on LLMs (89.6%) due to the computational cost of entropy computation over long sequences, leaving the optimization under-resourced.
- NSC exhibits the best task balance: it does not overfit a subset of tasks at the expense of others, a known failure mode of prior methods.
- The null-space ratio remains correlated with performance in the merged model: samples with lower ratios yield higher accuracy.

## Highlights & Insights

- Null-space compression is an elegant observation: the down-projection matrix in LoRA gradually "captures" more task-relevant activations during training, and this dynamic is repurposed as a merging signal.
- The fully input-driven formulation naturally extends to regression and generative tasks, resolving the fundamental limitation of entropy-based approaches.
- The Gram inverse caching trick is practically valuable: it reduces the computational bottleneck from $O(d^2)$ to $O(r^2)$ ($r \ll d$).
- Being both label-free and output-free makes NSC the most lightweight gradient-guided merging method to date.

## Limitations & Future Work

- A small amount of unlabeled data is still required for optimization; the method is not entirely data-free.
- Normalized performance on heterogeneous vision tasks reaches only 92%, leaving a non-trivial gap relative to single-task fine-tuning.
- The choice of target layers (last 1/4) is empirically determined and may require adjustment for different model architectures.
- Experiments are limited to ViT-B-scale vision models; effectiveness on larger models (e.g., ViT-L/H) remains unexplored.

## Related Work & Insights

- **vs. AdaMerging**: Both optimize merging coefficients via gradient-based methods, but AdaMerging relies on output entropy (limited to classification, scales with sequence length), whereas NSC uses the null-space ratio (task-agnostic, input-driven).
- **vs. KnOTS**: KnOTS projects adapters onto a shared subspace and merges SVD components, focusing on adapter alignment rather than merging weight optimization.
- **vs. Task Arithmetic**: TA applies a global scaling factor and performs poorly on heterogeneous tasks (77.2%); NSC's layer-wise coefficients provide substantially finer-grained control.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The null-space compression observation is novel and compelling; deriving a merging signal from LoRA geometry is a natural and principled approach.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated on 20 vision tasks, 6 NLI tasks, and VLM benchmarks against 11 baselines, with comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation and method derivation are clear; experimental presentation is thorough.
- **Value**: ⭐⭐⭐⭐⭐ Addresses a critical bottleneck in model merging for heterogeneous tasks, with practical implications for the LoRA ecosystem.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CLIP-Free, Label-Free, Unsupervised Concept Bottleneck Models](clip-free_label_free_unsupervised_concept_bottleneck_models.md)
- [\[ICCV 2025\] FREE-Merging: Fourier Transform for Efficient Model Merging](../../ICCV2025/multimodal_vlm/free-merging_fourier_transform_for_efficient_model_merging.md)
- [\[CVPR 2026\] DC-Merge: Improving Model Merging with Directional Consistency](dc-merge_improving_model_merging_with_directional_consistency.md)
- [\[CVPR 2026\] Mind the Discriminability Trap in Source-Free Cross-domain Few-shot Learning](mind_the_discriminability_trap_in_source-free_cross-domain_few-shot_learning.md)
- [\[CVPR 2026\] GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training](gtr_turbo_merged_checkpoint_free_teacher.md)

</div>

<!-- RELATED:END -->
