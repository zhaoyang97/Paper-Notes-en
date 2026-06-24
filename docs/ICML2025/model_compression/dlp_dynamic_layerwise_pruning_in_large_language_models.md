---
title: >-
  [Paper Note] DLP: Dynamic Layerwise Pruning in Large Language Models
description: >-
  [ICML 2025][Model Compression][LLM pruning] Proposes a dynamic layerwise pruning method, DLP, which adaptively calculates the relative importance of each layer using the median of weights and activation values. It performs non-uniform pruning based on the principle of "more important layers get lower sparsity ratios", reducing the perplexity of LLaMA2-7B by 7.79 and improving the average zero-shot accuracy by 2.7% at a high sparsity ratio of 70%.
tags:
  - "ICML 2025"
  - "Model Compression"
  - "LLM pruning"
  - "non-uniform layerwise sparsity"
  - "median"
  - "dynamic importance allocation"
  - "high sparsity ratio"
date: 2026-05-08
content_hash: 5411795af5ee6b3c
---

# DLP: Dynamic Layerwise Pruning in Large Language Models

**Conference**: ICML 2025  
**arXiv**: [2505.23807](https://arxiv.org/abs/2505.23807)  
**Code**: [https://github.com/ironartisan/DLP](https://github.com/ironartisan/DLP)  
**Area**: Model Compression  
**Keywords**: LLM pruning, non-uniform layerwise sparsity, median, dynamic importance allocation, high sparsity ratio

## TL;DR

Proposes a dynamic layerwise pruning method, DLP, which adaptively calculates the relative importance of each layer using the median of weights and activation values. It performs non-uniform pruning based on the principle of "more important layers get lower sparsity ratios", reducing the perplexity of LLaMA2-7B by 7.79 and improving the average zero-shot accuracy by 2.7% at a high sparsity ratio of 70%.

## Background & Motivation

Existing LLM pruning methods (such as SparseGPT, Wanda, etc.) adopt a **uniform sparsity ratio** across all layers, ignoring the varying contributions of different layers to model performance. This leads to severe performance degradation at high sparsity ratios ($\ge 70\%$). Although the OWL method attempts non-uniform layerwise pruning, it relies on a **manually set outlier threshold $M$**, which is highly sensitive to the model type and parameter scale:

- LLaMA1-7B and LLaMA1-13B (same architecture, different scales) have different optimal $M$ values.
- LLaMA1-7B and Vicuna-7B (same scale, different architectures) also have different optimal $M$ values.

This means OWL requires extensive parameter tuning in practice, and a fixed threshold may constrain the sparsity allocation to local patterns, ignoring the global importance distribution. The core motivation of DLP is to automatically determine the relative importance of each layer **without relying on any empirical values or model types**.

## Method

### Overall Architecture

The core mechanism of DLP consists of three steps:

1. **Calculate intra-layer absolute unimportance**: For each Transformer block, evaluate the "unimportance degree" of that layer based on the median of the joint weight-activation score.
2. **Convert to inter-layer relative importance**: Normalize the absolute unimportance to convert it into a cross-layer comparable Relative Importance Distribution (RID).
3. **Allocate non-uniform sparsity ratios**: Assign sparsity ratios based on the principle of "higher importance $\rightarrow$ lower sparsity ratio," incorporating a hyperparameter $\alpha$ to control the fluctuation range of sparsity ratios.

### Key Designs

#### 1. Weight-Activation Joint Importance Score

Following Wanda's scoring method, the absolute value of the weight is multiplied by the $\ell_2$ norm of the corresponding input activation to serve as the importance measure for each weight:

$$\mathbf{A}_{ij}^{l} = |\mathbf{W}_{ij}^{l}| \cdot \|\mathbf{X}_{j}^{l}\|_{2}$$

This score simultaneously accounts for the weight magnitude and the extent to which that position is activated during inference.

#### 2. Median-Based Layer Unimportance Measure

DLP selects the **median** as the function $F(\cdot)$ to measure layer unimportance:

$$S^{l} = \sum_{i=1}^{C_{\text{out}}} \sum_{j=1}^{C_{\text{in}}} \text{Median}(\mathbf{A}_{ij}^{l})$$

Three major advantages of choosing the median:

- **Insensitive to outliers**: Large language models contain numerous outlier activation values; the median is not biased by these extreme values, thus reflecting the "central tendency" more accurately.
- **Better redundancy capture**: Elements close to the median are easily represented by other elements in the same layer. Removing them has minimal impact on performance; a higher median indicates greater redundancy in that layer.
- **Low computational complexity**: Compared to the geometric median (which requires iterative approximation), the computation cost of the standard median is extremely low.

The paper empirically validates that the median outperforms other statistics such as Sum, Mean, Max, Variance, and SD (see the ablation study below).

#### 3. Relative Importance Distribution (RID)

Normalize and invert the layer unimportance to obtain the relative importance of each layer:

$$I^{l} = 1 - \frac{S^{l}}{\sum_{i=1}^{L} S^{i}}$$

The importance scores of all layers form the RID = $[I^1, I^2, \dots, I^L]$. The distribution pattern of RID is consistent with recent findings (deep layers underperform expectations): **shallow layers are more important** as they capture low-level general features, whereas deep layers tend to contain more specialized information and are more tolerant to pruning.

#### 4. Dynamic Sparsity Allocation

Introduce a hyperparameter $\alpha$ to control the sparsity ratio fluctuation range, mapping RID onto the $[0, 2\alpha]$ interval:

$$d_j = \frac{I_j - I_{\min}}{I_{\max} - I_{\min}} \times 2\alpha$$

The final sparsity ratio for each layer is:

$$R_j = p + m - d_j$$

where $p$ is the global target sparsity ratio and $m$ is the mean of $d$. This guarantees that:

- The global average sparsity ratio is exactly equal to $p$.
- The sparsity ratio of each layer fluctuates within $[p-\alpha, p+\alpha]$.
- Avoids performance collapse caused by over-pruning any single layer.

#### 5. Pruning Granularity Selection

In the appendix, the paper compares the performance of allocations made per Transformer block versus per layer, finding that **per-layer allocation** yields better performance. During actual pruning, a per-output execution rather than a per-layer execution is adopted.

### Loss & Training

DLP itself is a **training-free** pruning method, requiring only 128 calibration samples from C4 (2048 tokens each) for forward propagation to retrieve activations. The layer-wise optimization objective follows the standard $\ell_2$ reconstruction error:

$$\min_{\hat{\mathbf{W}}^l} \|\mathbf{W}^l \mathbf{X}^l - \hat{\mathbf{W}}^l \mathbf{X}^l\|_2^2$$

Post-pruning, **LoRA fine-tuning** can optionally be applied to recover performance: by fixing the pruning mask and fine-tuning with pre-trained autoregressive loss on the C4 training set, the perplexity of LLaMA1-7B can be reduced from 17.76 to 12.15.

## Key Experimental Results

### Main Results

The experiments cover 10+ models including LLaMA1 (7B/13B/30B), LLaMA2 (7B/13B), LLaMA3-8B, Mistral-7B, Vicuna-7B, Qwen-7B, etc. The perplexity results at a 70% unstructured sparsity ratio are as follows:

| Pruning Method | Layerwise Strategy | LLaMA1-7B | LLaMA1-13B | LLaMA2-7B | LLaMA2-13B | Gain over OWL |
|---------|---------|-----------|------------|-----------|------------|-----------|
| SparseGPT | Uniform | 25.38 | 18.93 | 27.84 | 19.38 | - |
| SparseGPT | OWL | 19.95 | 14.02 | 19.71 | 15.12 | - |
| SparseGPT | **DLP** | **17.76** | **12.63** | **18.58** | **13.30** | PPL↓2.19/1.39/1.13/1.82 |
| Wanda | Uniform | 86.38 | 56.26 | 76.84 | 45.76 | - |
| Wanda | OWL | 24.46 | 16.23 | 30.58 | 20.65 | - |
| Wanda | **DLP** | **20.46** | **13.65** | **22.79** | **16.19** | PPL↓4.00/2.58/7.79/4.46 |

Zero-shot average accuracy (70% sparsity ratio, 7 downstream tasks):

| Pruning Method | Layerwise Strategy | LLaMA1-7B | LLaMA1-13B | LLaMA2-7B | LLaMA2-13B |
|---------|---------|-----------|------------|-----------|------------|
| Dense | - | 64.33 | 66.78 | 64.42 | 67.04 |
| SparseGPT | Uniform | 45.32 | 48.34 | 44.72 | 47.99 |
| SparseGPT | **DLP** | **48.32** | **53.06** | **49.65** | **53.47** |
| Wanda | Uniform | 39.91 | 41.62 | 37.04 | 40.44 |
| Wanda | **DLP** | **48.62** | **52.03** | **46.25** | **51.11** |

### Ablation Study

Comparison of unimportance measurement functions (LLaMA1-7B, 70% sparsity ratio, WikiText PPL):

| Metric Method | Magnitude | SparseGPT | Wanda | Description |
|---------|-----------|-----------|-------|------|
| Sum | 3.7e3 | 18.24 | 21.03 | Summation, equivalent to Mean |
| Mean | 3.7e3 | 18.24 | 21.03 | Mean, vulnerable to outliers |
| **Median** | **3.4e3** | **17.76** | **20.40** | **Median, optimal** |
| Max | 2.9e4 | 38.57 | 931.89 | Maximum, extremely unstable |
| Variance | 4.7e5 | 21.33 | 43.31 | Variance, poor performance |
| SD | 2.5e5 | 21.42 | 38.03 | Standard deviation, poor performance |

LOD (outlier retention rate) comparison (LLaMA1-13B, 70% sparsity ratio):

| Method | LOD(%)↑ | PPL↓ | Description |
|------|---------|------|------|
| Uniform (SparseGPT) | 47.70 | 18.93 | Baseline |
| OWL (SparseGPT) | 51.97 | 14.02 | Increased outlier retention |
| **DLP (SparseGPT)** | **64.46** | **12.63** | Most outlier retention |
| Uniform (Wanda) | 55.14 | 56.26 | Baseline |
| **DLP (Wanda)** | **70.06** | **13.65** | LOD is 13.76% higher than OWL |

### Key Findings

1. **Significant inference acceleration**: On the DeepSparse engine, LLaMA2-7B achieves 2.8x speedup under 70% sparsity, 3.5x under 80% sparsity, and 3.7x under 90% sparsity.
2. **Negligible computational overhead**: The pre-computation and execution time of DLP's non-uniform sparsity are almost identical to uniform pruning (Wanda+DLP takes only 0.03 seconds more than Wanda).
3. **Further recovery via LoRA fine-tuning**: Under 70% sparsity, the PPL of LLaMA1-7B drops from 17.76 to 12.15 (a 31.6% reduction).
4. **Larger advantage at higher sparsity ratios**: Under 80% sparsity, DLP reduces PPL by up to 56% compared to OWL.
5. **Broad compatibility**: DLP seamlessly integrates with unstructured pruning, N:M sparsity, structured pruning, quantization (GPTQ), SVD, and PEFT (LoRA) methods.

## Highlights & Insights

- **Simple yet effective core idea**: Using only the median statistic replaces OWL's sensitive, hand-tuned outlier threshold $M$. The method is extremely simple but yields significant results.
- **Reversed workflow**: Measuring "unimportance" first and then inverting it to obtain importance is more intuitive than directly defining importance, as "redundancy" is easier to measure than "criticality".
- **Elegant design of the $\alpha$ parameter**: Mapping the importance to a bounded interval $[0, 2\alpha]$ prevents extreme sparsity ratios from destroying single-layer performance, while ensuring the global sparsity ratio precisely equals the target value.
- **Increasing unimportance in deeper layers**: The finding that deeper layers are less important aligns with recent work such as layer pruning/ShortGPT, providing cross-validation from a pruning perspective.

## Limitations & Future Work

1. **Magnitude baseline combined with DLP remains poor**: Although there is an improvement, Magnitude+DLP still results in a PPL as high as 3.4e3 on the 7B model, showing that DLP serves fundamentally as an "enhancer" rather than a standalone solution.
2. **Hyperparameter $\alpha$ still needs selection**: Although the dependency on $M$ is eliminated, a new hyperparameter $\alpha$ is introduced. The paper provides recommended values for different sparsity ratios in Appendix G, indicating remaining tuning space.
3. **Calibration data sensitivity is not fully explored**: Only 128 samples from C4 are used. Calibration data from different domains/distributions might affect the RID calculation.
4. **Lack of evaluation on MoE architectures**: Only dense LLMs are tested; the layer importance distribution in MoE models like Mixtral could be entirely different.
5. **Absolute performance under high sparsity remains a concern**: Even with LoRA fine-tuning, the perplexity of LLaMA1-7B under 70% sparsity (12.15) is still significantly higher than the dense model (5.68).

## Related Work & Insights

- **SparseGPT / Wanda**: Direct baselines for DLP, which acts as a plug-and-play enhancement module for their layerwise allocation.
- **OWL**: The most direct competitor, allocating sparsity based on outlier ratios but relying on empirical thresholds.
- **FPGM / LRMF**: Pioneering works utilizing medians in pruning, but they use the geometric median (computationally complex) whereas DLP simplifies it to a standard median.
- **ShortGPT / Layer Pruning**: Studies indicating that deep layers can be directly dropped, which aligns with the RID trend identified by DLP.
- **LAMP**: Another adaptive layerwise pruning method, which DLP consistently outperforms at sparsity ratios above 40%.

## Rating

- Novelty: ⭐⭐⭐⭐ — The core idea (median + inverted importance) is simple and elegant, but the overall framework is a somewhat incremental improvement.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Very comprehensive, with 10+ models, 3 pruning methods, various sparsity ratios, and 6 types of extended experiments.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, with empirical research progressing systematically, but contains redundant symbols and somewhat lengthy formulas.
- Value: ⭐⭐⭐⭐ — Highly practical, being plug-and-play, introducing zero extra overhead, and widely compatible with various compression methods.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Instruction-Following Pruning for Large Language Models](instruction-following_pruning_for_large_language_models.md)
- [\[ICML 2025\] SlimLLM: Accurate Structured Pruning for Large Language Models](slimllm_accurate_structured_pruning_for_large_language_models.md)
- [\[ACL 2025\] Wanda++: Pruning Large Language Models via Regional Gradients](../../ACL2025/model_compression/wanda_pruning_large_language_models_via_regional_gradients.md)
- [\[ICML 2025\] Weak-to-Strong Jailbreaking on Large Language Models](weak-to-strong_jailbreaking_on_large_language_models.md)
- [\[ICML 2025\] Persistent Topological Features in Large Language Models](persistent_topological_features_in_large_language_models.md)

</div>

<!-- RELATED:END -->
