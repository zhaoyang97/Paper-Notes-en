---
title: >-
  [Paper Note] Achieving More with Less: Additive Prompt Tuning for Rehearsal-Free Class-Incremental Learning
description: >-
  [ICCV 2025][Model Compression][Class-Incremental Learning] This paper proposes APT (Additive Prompt Tuning), which replaces the conventional prompt concatenation paradigm with an additive operation. By introducing only t…
tags:
  - "ICCV 2025"
  - "Model Compression"
  - "Class-Incremental Learning"
  - "Prompt Learning"
  - "Parameter-Efficient Fine-Tuning"
  - "Catastrophic Forgetting"
  - "Vision Transformer"
date: 2026-05-08
content_hash: b0522e06cfb60238
---

# Achieving More with Less: Additive Prompt Tuning for Rehearsal-Free Class-Incremental Learning

**Conference**: ICCV 2025
**arXiv**: N/A (CVF OpenAccess)  
**Code**: Unavailable  
**Area**: Model Compression
**Keywords**: Class-Incremental Learning, Prompt Learning, Parameter-Efficient Fine-Tuning, Catastrophic Forgetting, Vision Transformer

## TL;DR

This paper proposes APT (Additive Prompt Tuning), which replaces the conventional prompt concatenation paradigm with an additive operation. By introducing only two learnable vectors added to the key/value projections of the CLS token, APT achieves state-of-the-art class-incremental learning performance while substantially reducing computational overhead (41.5% reduction in GFLOPs) and trainable parameters (78.2% reduction).

## Background & Motivation

Class-Incremental Learning (CIL) requires a model to continuously recognize new classes while retaining knowledge of previously learned ones, with **catastrophic forgetting** as the central challenge. Prompt-based methods have recently emerged as the dominant paradigm, with L2P introducing three key components: (1) maintaining a learnable prompt pool, (2) selecting a subset of prompts via a query mechanism, and (3) concatenating the selected prompts to the input embeddings.

However, this framework suffers from **three overlooked fundamental problems**:

**Doubled computational overhead**: The query mechanism requires an additional complete ViT forward pass to generate the query, effectively doubling inference cost.

**Sequence length inflation**: Concatenating large numbers of prompt tokens increases input sequence length by up to 50%, while ViT computational complexity scales proportionally with token count.

**Heavy hyperparameter burden**: Multiple hyperparameters such as prompt length and prompt pool size must be tuned for each dataset.

A key motivating finding, drawn from works such as OVOR, is that using a **single prompt** can already achieve competitive performance. This leads to two core questions:

- Is the complex prompt selection framework truly necessary?
- Is prompt concatenation the most effective and efficient approach?

## Method

### Overall Architecture

The core idea of APT is to **replace concatenation with addition**, modifying only the attention computation of the CLS token. The design is grounded in two key insights:

1. **The CLS token is an information aggregator**: Prior work demonstrates that the CLS token in ViT effectively aggregates critical visual information from the input image.
2. **Addition preserves sequence length**: Unlike concatenation, additive operations do not alter the original sequence length, thereby avoiding extra computational overhead.

Concretely, each Transformer layer introduces only **two learnable vectors** $p_k^l, p_v^l \in \mathbb{R}^d$, added respectively to the key and value vectors derived from the CLS token:

$$\hat{k}_{cls}^l = k_{cls}^l + p_k^l, \quad \hat{v}_{cls}^l = v_{cls}^l + p_v^l$$

Self-attention is then computed normally:

$$\text{Attention}(Q^l, \hat{K}^l, \hat{V}^l) = \text{softmax}\left(\frac{Q^l (\hat{K}^l)^T}{\sqrt{d}}\right) \hat{V}^l$$

**Why modify only the CLS token?** Training additive prompts for every token in the sequence would result in a large parameter count ($m \times d$ per layer). Selectively modifying a subset of tokens would introduce a non-trivial decision problem. As the natural aggregation point for global representations, the CLS token is the uniquely principled choice.

**Why key and value rather than directly modifying the CLS input?** In the attention mechanism, keys govern *what to attend to* and values govern *what information to propagate*. Directly modifying key/value projections allows the prompts to more precisely influence the feature extraction process. Ablation studies confirm this design to be more effective than adding prompts at the input layer.

### Key Designs

**Minimal parameter design**: For ViT-B/16 (12 layers), APT requires training only 24 prompt vectors in total (2 per layer), amounting to $24 \times 768 = 18{,}432$ parameters (0.02M) — a 99.3% reduction compared to the 100-entry prompt pool in Coda-Prompt.

**No prompt pool or query mechanism**: APT employs a **single shared set of prompts** across all tasks, completely eliminating:
- Prompt pool maintenance costs
- Query key learning
- Additional forward pass computation
- Complex prompt selection loss functions

### Loss & Training

**Greatly simplified training objective**: APT uses only standard cross-entropy loss, requiring no auxiliary regularization or prompt selection losses:

$$\mathcal{L} = -\frac{1}{N_t} \sum_{i=1}^{N_t} \log \frac{\exp(h_t(f(x_i))_{y_i})}{\sum_j \exp(h_t(f(x_i))_j)}$$

where $f(x_i)$ denotes the final CLS token embedding after prompt modification, and $h_t$ is the task-specific classification head. Trainable parameters consist solely of the prompt vectors $P_t$ and the classification head $h_t$; all Transformer parameters remain frozen.

**Progressive Prompt Fusion (PPF) inference strategy**: To mitigate catastrophic forgetting, after training on a new task, old and new prompts are merged via a weighted average:

$$P_{t+1}^{PPF} = \alpha P_t + (1 - \alpha) P_{t+1}$$

where $\alpha \in [0, 1]$ controls the balance between retaining prior knowledge and adapting to the new task. **Crucially, PPF is applied at inference time only**; training uses only the current task's prompts to ensure sufficient task-specific learning. The scalar $\alpha$ is the **only additional hyperparameter** introduced by APT, and experiments demonstrate strong robustness across a wide range (0.2–0.8 yields minimal variation).

## Key Experimental Results

### Main Results

Results under the 10-task setting across four CIL benchmarks:

| Method | CIFAR-100 Avg Acc ↑ | ImageNet-R Avg Acc ↑ | CUB200 Avg Acc ↑ | Stanford Cars Avg Acc ↑ |
|--------|---------------------|----------------------|-------------------|------------------------|
| L2P | 83.21 | 72.64 | 71.22 | 60.39 |
| DualPrompt | 82.03 | 69.12 | 71.55 | 57.27 |
| Coda-Prompt | 86.92 | 73.21 | 73.25 | 62.24 |
| CPrompt | 87.82 | 77.14 | 77.09 | 66.77 |
| EvoPrompt | 87.97 | 76.83 | - | - |
| **APT (Ours)** | **88.88** | **79.40** | **78.50** | **71.04** |

Efficiency comparison (Split ImageNet-R):

| Method | GFLOPs (ratio) | Trainable Prompt Params (M) |
|--------|---------------|----------------------------|
| CPrompt | 37.87 (2.25×) | 0.77 |
| Coda-Prompt | 33.67 (2.00×) | 3.07 |
| EvoPrompt | 36.37 (2.16×) | 0.04 |
| OVOR-Deep | 16.81 (1.00×) | 0.11 |
| **APT** | **16.80 (1.00×)** | **0.02** |

### Ablation Study

Ablation of the PPF strategy and additive KV design:

| Variant | CIFAR-100 Acc | CIFAR-100 Forgetting | ImageNet-R Acc | ImageNet-R Forgetting |
|---------|--------------|---------------------|---------------|---------------------|
| w/o PPF | 88.17 | 7.05 | 78.56 | 8.83 |
| w/o add KV | 87.93 | 5.24 | 78.31 | 5.18 |
| **APT** | **88.88** | **3.47** | **79.40** | **4.38** |

General recognition (non-CIL setting, compared with VPT):

| Method | CUB200 | Flowers102 | Stanford Cars | Mean |
|--------|--------|------------|--------------|------|
| VPT-deep | 88.5 | 98.9 | 75.2 | 84.4 |
| Full Fine-tuning | 87.3 | 98.8 | 84.5 | 85.9 |
| **APT** | **89.1** | **99.1** | **84.2** | **86.3** |

### Key Findings

1. **2%+ lead on ImageNet-R**: APT performs most prominently on the most challenging dataset, indicating superior robustness to distribution shift.
2. **4.27% lead on Stanford Cars**: Significant advantage on fine-grained tasks, suggesting APT better captures subtle inter-class distinctions.
3. **GFLOPs essentially equal to vanilla ViT**: 16.80 vs. 16.80 — the prompts introduce no additional inference overhead.
4. **PPF primarily reduces forgetting rather than boosting accuracy**: Removing PPF increases forgetting from 3.47 to 7.05 on CIFAR-100, while accuracy changes modestly.
5. **Larger advantage in 20-task long-sequence settings**: APT leads the second-best method by 5%+ on average.

## Highlights & Insights

1. **Paradigm-level innovation**: The conceptual shift from "concatenating prompts" to "additive prompts" is elegant — it fundamentally reframes prompts from *augmenting the input* to *modifying the computation*.
2. **Less-is-more design philosophy**: Only 2 prompt vectors per layer, only cross-entropy loss, only 1 additional hyperparameter — yet the approach surpasses considerably more complex methods.
3. **General PEFT potential**: APT not only excels in CIL but also outperforms VPT-deep and full fine-tuning on general recognition tasks, demonstrating the broad applicability of the additive prompt paradigm.
4. **Sharper attention visualization**: APT's attention maps exhibit more concentrated and accurate focus on target regions compared to competing methods.

## Limitations & Future Work

1. **Manual selection of $\alpha$ remains required**: Despite strong robustness, different values are used across datasets (0.7 vs. 0.8); adaptive learning of this parameter warrants investigation.
2. **Validation limited to ViT**: Extensibility to other architectures (e.g., Swin Transformer, ConvNeXt) has not been explored.
3. **Not tested on larger pretrained models**: Performance on ViT-L/ViT-H or DINOv2 remains unknown.
4. **Task heads still require storage**: Despite minimal prompt parameters, each task still requires an independent classification head.

## Related Work & Insights

- **VPT (ECCV 2022)**: The direct predecessor of APT, which established the effectiveness of visual prompt learning but relies on the less efficient concatenation paradigm.
- **L2P (CVPR 2022)**: Introduced the prompt pool + query selection framework, which APT demonstrates to be unnecessary.
- **OVOR**: Showed that a single prompt is sufficient, motivating APT's abandonment of the prompt pool design.
- **EMA (Cai et al. 2021)**: The exponential moving average concept inspired PPF's weighted fusion strategy.
- For incremental learning systems deployed on **edge devices**, APT's extremely low overhead makes it the preferred solution.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Integrating Task-Specific and Universal Adapters for Pre-Trained Model-based Class-Incremental Learning](integrating_task-specific_and_universal_adapters_for_pre-trained_model-based_cla.md)
- [\[NeurIPS 2025\] Mixture of Noise for Pre-Trained Model-Based Class-Incremental Learning](../../NeurIPS2025/model_compression/mixture_of_noise_for_pre-trained_model-based_class-incremental_learning.md)
- [\[NeurIPS 2025\] REP: Resource-Efficient Prompting for Rehearsal-Free Continual Learning](../../NeurIPS2025/model_compression/rep_resource-efficient_prompting_for_rehearsal-free_continual_learning.md)
- [\[AAAI 2026\] Compensating Distribution Drifts in Class-incremental Learning of Pre-trained Vision Transformers](../../AAAI2026/model_compression/compensating_distribution_drifts_in_class-incremental_learning_of_pre-trained_vi.md)
- [\[ICML 2026\] HEDP: A Hybrid Energy-Distance Prompt-based Framework for Domain Incremental Learning](../../ICML2026/model_compression/hedp_a_hybrid_energy-distance_prompt-based_framework_for_domain_incremental_lear.md)

</div>

<!-- RELATED:END -->
