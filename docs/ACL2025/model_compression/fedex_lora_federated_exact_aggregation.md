---
title: >-
  [Paper Note] FedEx-LoRA: Exact Aggregation for Federated and Efficient Fine-Tuning of Large Language Models
description: >-
  [ACL 2025][Model Compression][LoRA] FedEx-LoRA identifies that independently averaging the A and B matrices of LoRA in federated learning yields inaccurate global updates ("the mean of products $\neq$ the product of means"). By incorporating a residual error term into the frozen weight matrix to achieve exact aggregation, FedEx-LoRA consistently outperforms FedIT and FFA-LoRA across multiple reasoning and NLU tasks.
tags:
  - "ACL 2025"
  - "Model Compression"
  - "LoRA"
  - "Federated Learning"
  - "Exact Aggregation"
  - "Parameter-Efficient Fine-Tuning"
  - "Residual Correction"
date: 2026-05-08
content_hash: 57a4848c390e33fc
---

# FedEx-LoRA: Exact Aggregation for Federated and Efficient Fine-Tuning of Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2410.09432](https://arxiv.org/abs/2410.09432)  
**Code**: [https://github.com/RaghavSinghal10/fedex-lora](https://github.com/RaghavSinghal10/fedex-lora)  
**Area**: Model Compression  
**Keywords**: LoRA, Federated Learning, Exact Aggregation, Parameter-Efficient Fine-Tuning, Residual Correction

## TL;DR
FedEx-LoRA identifies that independently averaging the A and B matrices of LoRA in federated learning yields inaccurate global updates ("the mean of products $\neq$ the product of means"). By incorporating a residual error term into the frozen weight matrix to achieve exact aggregation, FedEx-LoRA consistently outperforms FedIT and FFA-LoRA across multiple reasoning and NLU tasks.

## Background & Motivation

**Background**: LoRA is a mainstream parameter-efficient fine-tuning method for LLMs, which significantly reduces trainable parameters by decomposing weight updates into low-rank matrices $\Delta W = BA$. In federated learning (FL), FedIT is the current SOTA, which applies standard FedAvg to average each client's A and B matrices separately.

**Limitations of Prior Work**: FedIT averages $A_i$ and $B_i$ individually and multiplies them to get the global update $\bar{B}\bar{A}$, whereas the ideal global update should be the average of the products of each client $\frac{1}{k}\sum B_iA_i$. Mathematically, "the mean of products $\neq$ the product of means", which introduces bias during federated aggregation.

**Key Challenge**: Directly aggregating the average of $B_iA_i$ would lead to a high-rank matrix, losing the efficiency benefits of LoRA's low-rank structure. Performing low-rank decomposition on the high-rank result would cause the rank to grow exponentially with communication rounds. FFA-LoRA circumvents this issue by freezing the A matrix, but this restricts representation capacity.

**Goal**: Achieve exact aggregation for federated LoRA while maintaining the low-rank efficiency of LoRA.

**Key Insight**: Absorb the aggregation error (the difference between the ideal update and the actual update) as a residual term into the pre-trained frozen weight matrix, which is already high-rank, requiring no additional training.

**Core Idea**: Add the residual error between the "product of means" and the "mean of products" to the frozen base weight. This preserves low-rank training of LoRA while achieving exact aggregation.

## Method

### Overall Architecture
The pipeline of FedEx-LoRA is: Server distributes the global model + LoRA modules $\to$ Clients independently train A, B $\to$ Clients upload A, B to the server $\to$ Server computes the average A, B + residual correction term $\to$ The residual is added to the frozen weights + the new A, B are issued to clients $\to$ Repeat. Inputs and outputs are identical to standard federated LoRA; the critical difference lies in the aggregation step.

### Key Designs

1. **Residual Error Term $\Delta W_{res}$**:

    - **Function**: Compensate for the bias of "product of means vs. mean of products" in federated averaging.
    - **Mechanism**:
    $$\Delta W_{res}^j = \frac{1}{k}\sum_{i=1}^{k}(B_i^j A_i^j) - \frac{1}{k}\sum_{i=1}^{k}B_i^j \times \frac{1}{k}\sum_{i=1}^{k}A_i^j$$
      Add this residual to the frozen weight: $W_0^{j+1} \leftarrow W_0^j + \Delta W_{res}^j$
    - **Design Motivation**: The residual itself is high-rank (with rank up to $k \cdot r$), which cannot fit into low-rank LoRA adapters. However, the frozen weight matrix is intrinsically high-rank, so adding the residual does not affect its structure. The residual requires no training and is computed purely dynamically.

2. **Communication Protocol Optimization**:

    - **Function**: Reduce communication overhead during residual matrix transmission.
    - **Mechanism**: The upper bound of the rank of $\Delta W_{res}$ is $k \cdot r$. It can be decomposed into two low-rank matrices for transmission using Gram-Schmidt orthonormalization, rather than transmitting the high-dimensional matrix directly.
    - **Design Motivation**: Prevent communication overhead from growing quadratically with model dimensions. Experiments show only a 2-8% increase in communication overhead compared to FedIT.

3. **Optimal Inexact Approximation (for massive client scenarios)**:

    - **Function**: Approximate the residual using truncated SVD when the number of clients is very large.
    - **Mechanism**: Quantize $\Delta W_{res}$ using truncated SVD to preserve the top $r'$ singular values, which is the optimal low-rank approximation according to the Eckart-Young theorem.
    - **Design Motivation**: Exact aggregation communication cost increases linearly with the number of clients. Utilizing approximation for massive client scenarios limits the communication volume.

4. **Analysis of Multiple Assignment Strategies**:

    - **Function**: Prove the existence of multiple assignment strategies for exact aggregation.
    - **Mechanism**: There are different combinations for aggregating $A_i$ and $B_i$ (e.g., only averaging A and retaining B, only averaging B and retaining A, etc.), all of which can achieve exact aggregation by adjusting the residual.
    - **Design Motivation**: Experimental results validate that simultaneously averaging A and B + introducing the residual correction yields the best performance.

### Loss & Training
The training strategy is completely identical to standard LoRA. The modifications of FedEx-LoRA occur solely in the aggregation step and do not introduce any additional training overhead.

## Key Experimental Results

### Main Results

Commonsense Reasoning (Llama-3.2 3B, r=32):

| Method | BoolQ | PIQA | SIQA | HellaS. | WinoG. | ARC-e | ARC-c | OBQA | Avg |
|------|-------|------|------|---------|--------|-------|-------|------|-----|
| Centralized LoRA | 73.45 | 89.65 | 82.23 | 94.41 | 87.97 | 93.88 | 82.76 | 86.60 | 86.37 |
| FedIT | 70.73 | 87.59 | 79.17 | 91.06 | 83.42 | 92.71 | 81.31 | 82.68 | 83.57 |
| FFA-LoRA | 65.78 | 84.22 | 72.41 | 82.27 | 72.53 | 90.36 | 76.28 | 75.00 | 77.35 |
| **FedEx-LoRA** | **73.21** | **89.01** | **81.98** | **94.29** | **87.29** | **93.68** | **82.33** | **86.20** | **85.99** |

Arithmetic Reasoning (r=32):

| Model | Method | GSM8K | MATH |
|------|------|-------|------|
| Mistral-7B | FedIT | 56.94 | 14.96 |
| Mistral-7B | FFA-LoRA | 56.41 | 14.88 |
| Mistral-7B | **FedEx-LoRA** | **62.62** | **16.54** |
| Gemma-2 9B | FedIT | 74.57 | 37.16 |
| Gemma-2 9B | **FedEx-LoRA** | **76.19** | **39.00** |

### Ablation Study

NLU Tasks (RoBERTa-base, GLUE, r=4):

| Method | CoLA | RTE | MRPC | SST-2 | QNLI | STS-B | Avg |
|------|------|-----|------|-------|------|-------|-----|
| Centralized LoRA | 64.31 | 75.45 | 87.99 | 94.61 | 92.75 | 90.73 | 84.31 |
| FedIT | 60.82 | 73.64 | 88.48 | 94.61 | 92.07 | 90.91 | 83.42 |
| FFA-LoRA | 59.34 | 70.04 | 87.50 | 94.27 | 91.37 | 90.26 | 82.13 |
| **FedEx-LoRA** | **62.82** | **75.09** | **89.95** | **94.84** | **92.66** | **90.95** | **84.39** |

Communication Cost (Parameter transmission ratio relative to FedEx-LoRA):

| Model | Full FT | FedEx-LoRA | FedIT | FFA-LoRA |
|------|---------|-----------|-------|----------|
| RoBERTa-base | 7.03× | 1× | 0.98× | 0.97× |
| GPT-2 | 9.48× | 1× | 0.92× | 0.89× |

### Key Findings
- FedEx-LoRA consistently outperforms FedIT and FFA-LoRA across all tasks. Its average accuracy in commonsense reasoning is 8.63% higher than FFA-LoRA and 2.42% higher than FedIT.
- On Mistral-7B/GSM8K, FedEx-LoRA (62.62) almost matches Centralized LoRA (62.77), indicating that exact aggregation virtually eliminates performance degradation caused by federated setup.
- Communication overhead is only 2-8% higher than FedIT, which is significantly smaller than the 7-10× overhead of Full Fine-Tuning.
- Analysis of aggregation bias shows: the bias increases with training rounds and exhibits different patterns across different layers, quantifying the necessity of exact aggregation.
- Performance improvements are even more pronounced under extremely low-rank settings such as $r=1$ (where the aggregation bias accounts for a larger proportion).

## Highlights & Insights
- **Simple Yet Profound Insights**: "the mean of products $\neq$ the product of means" — a single phrase reveals the fundamental issue of federated LoRA. Translating such mathematically elegant observations into practical improvements is a hallmark of excellent work.
- **Extremely Simple Solution**: No changes to the training process, no introduction of hyperparameters, and only a residual addition in the aggregation step. The plug-in nature of the method allows it to be seamlessly integrated into existing federated learning frameworks.
- **High Transferability**: This idea can be directly transferred to federated fine-tuning of other architectures such as ViTs and VLMs, or combined with privacy-preserving techniques like Differential Privacy.

## Limitations & Future Work
- In massive client scenarios (where $k$ is extremely large), the communication cost of exact aggregation grows linearly; although a truncated SVD approximation is proposed, it remains under-validated.
- Not tested under differential privacy settings, though the authors anticipate good performance.
- All experiments assume IID data distributions; performance under Non-IID federated scenarios remains to be validated.
- Only NLP tasks are validated; vision/multimodal tasks are not covered.

## Related Work & Insights
- **vs. FedIT**: FedIT directly averages the A and B matrices with FedAvg, suffering from aggregation bias. FedEx-LoRA eliminates the bias through residual correction, consistently surpassing FedIT across all tasks.
- **vs. FFA-LoRA**: FFA-LoRA freezes the A matrix to avoid bias but restricts representation capability, performing poorly in non-private settings. FedEx-LoRA preserves the flexibility of dual-matrix training.
- **vs. Centralized LoRA**: FedEx-LoRA nearly matches the performance of Centralized LoRA, demonstrating that exact aggregation largely bridges the gap between federated and centralized paradigms.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Excellent insight into the problem, simple but mathematically rigorous solution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers multiple models from RoBERTa to Gemma-2 9B, multi-task and multi-rank settings, with detailed analysis of communication overhead.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear motivation, rigorous mathematical derivation, and intuitive charts and tables.
- **Value**: ⭐⭐⭐⭐ A practical improvement for federated LoRA, highly applicable and straightforward, though the application scenario is relatively niche.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] L4Q: Parameter Efficient Quantization-Aware Fine-Tuning on Large Language Models](l4q_parameter_efficient_quantization_aware_finetuning.md)
- [\[ACL 2025\] C3A: Parameter-Efficient Fine-Tuning via Circular Convolution](parameter-efficient_fine-tuning_via_circular_convolution.md)
- [\[NeurIPS 2025\] Data Efficient Adaptation in Large Language Models via Continuous Low-Rank Fine-Tuning](../../NeurIPS2025/model_compression/data_efficient_adaptation_in_large_language_models_via_continuous_low-rank_fine-.md)
- [\[NeurIPS 2025\] Robust Federated Finetuning of LLMs via Alternating Optimization of LoRA](../../NeurIPS2025/model_compression/robust_federated_finetuning_of_llms_via_alternating_optimization_of_lora.md)
- [\[ACL 2025\] State-offset Tuning: State-based Parameter-Efficient Fine-Tuning for State Space Models](state_offset_tuning_ssm_peft.md)

</div>

<!-- RELATED:END -->
