---
title: >-
  [Paper Note] C3A: Parameter-Efficient Fine-Tuning via Circular Convolution
description: >-
  [ACL 2025][Model Compression][circular convolution] This paper proposes C3A, a method that replaces the low-rank matrix decomposition of LoRA with a circular convolution operator to achieve parameter-efficient fine-tuning. Its key advantage is the decoupling of matrix rank and parameter size, enabling high-rank adaptation with few parameters. Meanwhile, it maintains computational and memory efficiency comparable to LoRA via FFT, consistently outperforming LoRA and its variant…
tags:
  - "ACL 2025"
  - "Model Compression"
  - "circular convolution"
  - "LoRA"
  - "PEFT"
  - "FFT"
  - "parameter-efficient fine-tuning"
date: 2026-05-08
content_hash: d3234ba51c57aab5
---

# C3A: Parameter-Efficient Fine-Tuning via Circular Convolution

**Conference**: ACL 2025  
**arXiv**: [2407.19342](https://arxiv.org/abs/2407.19342)  
**Code**: [https://huggingface.co/docs/peft](https://huggingface.co/docs/peft) (integrated into HuggingFace PEFT)  
**Area**: Model Compression / Parameter-Efficient Fine-Tuning  
**Keywords**: circular convolution, LoRA, PEFT, FFT, parameter-efficient fine-tuning

## TL;DR
This paper proposes C3A, a method that replaces the low-rank matrix decomposition of LoRA with a circular convolution operator to achieve parameter-efficient fine-tuning. Its key advantage is the decoupling of matrix rank and parameter size, enabling high-rank adaptation with few parameters. Meanwhile, it maintains computational and memory efficiency comparable to LoRA via FFT, consistently outperforming LoRA and its variants across multiple fine-tuning tasks.

## Background & Motivation

### Background
**Background**: Large foundation models (LFMs) have achieved unprecedented performance in NLP, CV, and other fields. However, the fine-tuning costs incurred by their massive parameter size present an obstacle to practical deployment. Parameter-efficient fine-tuning (PEFT) techniques, represented by LoRA, approximate weight updates via low-rank matrices $\Delta W = BA$ ($B \in \mathbb{R}^{d_1 \times r}, A \in \mathbb{R}^{r \times d_2}$, $r \ll \min(d_1, d_2)$), significantly reducing the number of trainable parameters.

### Limitations of Prior Work
**Limitations of Prior Work**: (1) **Intrinsic low-rank limitation of LoRA**: The parameter count $r(d_1+d_2)$ simultaneously determines the rank upper bound $r$ of $\Delta W$. The rank is strictly constrained by the parameter budget, which Zeng & Lee (2023) proved restricts the approximation capability of the target model; (2) **Efficiency issues in high-rank methods**: Variants like VeRA achieve high rank by fixing random matrices, but their computation and memory overheads significantly exceed those of LoRA ($O(r_v(d_1+d_2))$, where $r_v$ can exceed $\max(d_1, d_2)$); (3) **Existing methods fail to simultaneously achieve high rank, low parameter footprint, and low computational/memory overhead**.

**Key Challenge**: The trade-off among rank, parameter size, and efficiency in PEFT—LoRA sacrifices rank for efficiency, while VeRA sacrifices efficiency for rank. How can all three be achieved simultaneously?

### Goal
**Goal**: To achieve high-rank adaptation without sacrificing time and memory efficiency—decoupling the matrix rank from the parameter footprint.

**Key Insight**: The rank of the circulant matrix $\mathcal{C}(\Delta w)$ corresponding to the circular convolution operator $\Delta w \star x = \mathcal{C}(\Delta w)x$ is determined by the polynomial GCD (with a theoretical upper bound of $d$), which is completely independent of the parameter count (only $d$ elements). Furthermore, circulant matrices can be diagonalized by the Fourier basis, enabling highly efficient $O(d \log d)$ computation via FFT.

**Core Idea**: Replacing matrix multiplication with circular convolution as the additive linear operation in PEFT—achieving decoupling of parameter count and rank + FFT acceleration.

## Method

### Overall Architecture
C3A replaces LoRA's adaptation weight calculation $\Delta z = BAx$ with $\Delta z = \Delta w \star x$, where $\star$ denotes circular convolution. The circular convolution kernel $\Delta w$ serves as the trainable parameter, and its corresponding circulant matrix $\mathcal{C}(\Delta w)$ is the actual weight change matrix. Forward and backward propagation implemented via FFT ensures computational efficiency. For non-square weight matrices, block-circular convolution expansion is employed.

### Key Designs

1. **Circular Convolution Adaptation**:

    - Function: Realizing efficient weight adaptation with decoupled rank and parameter size
    - Mechanism: Learning a circular convolution kernel $\Delta w \in \mathbb{R}^d$ (only $d$ parameters), whose corresponding circulant matrix $\mathcal{C}(\Delta w)$ has a rank of $d - \text{Deg}(\gcd(f(x), x^d-1))$, with a theoretical upper bound of $d$ (full rank). Forward propagation is achieved via FFT: $\Delta w \star x = \text{FFT}(\text{FFT}(\Delta w) \circ \text{iFFT}(x))$. Backward propagation utilizes the commutativity of circular convolution $\mathcal{C}(\Delta w)x = \mathcal{C}(x)\Delta w$, enabling gradient computation to be accelerated via FFT as it also employs circular convolution
    - Design Motivation: Circulant matrices are the only structured matrix form that simultaneously possesses high-rank flexibility and FFT-diagonalizable efficiency

2. **Block-Circular Convolution**:

    - Function: Supporting non-square weight matrices (e.g., $4096 \times 1024$ in LLaMA-8B) and providing flexible parameter control
    - Mechanism: Partitioning the activation vector $x$ and output $\Delta z$ into blocks of size $b$, and allocating $d_1 d_2 / b^2$ independent circular convolution kernels to densely connect each block pair. $\Delta z_i = \sum_j \Delta w_{ij} \star x_j$, corresponding to the block-circulant matrix $\mathcal{C}_{\text{blk}}(\Delta w)$. The total parameter size is $d_1 d_2 / b$, where $b$ is a common divisor of $d_1, d_2$
    - Design Motivation: $b$ controls the parameter size similarly to $r$ in LoRA, but the key difference is that $b$ does not constrain the rank—decoupling the parameter size from the representational capacity

3. **FFT-Accelerated Efficient Implementation**:

    - Function: Ensuring computational and memory efficiency comparable to LoRA
    - Mechanism: The cuFFT backend on the GPU automatically parallelizes FFT operations (parallelism degree $p$). The total time complexity of C3A is $O((d_1+d_2)/p \cdot \log b + d_1 d_2/b)$, which is comparable to LoRA's $O(r(d_1+d_2))$ when $b$ is set to $\gcd(d_1, d_2)$. The space complexity is only $d_1 d_2/b$ (trainable parameters) + $pb$ (FFT buffer), avoiding the large fixed random matrices of VeRA
    - Design Motivation: In practice, the $O(n \log n)$ complexity of FFT is highly optimized on GPUs, allowing theoretical advantages to easily translate into actual acceleration

### Extra Feature: Circular Pattern as Inductive Bias
The structured pattern of circulant matrices introduces implicit regularization for fine-tuning. Dosovitskiy et al. (2020) pointed out that dense linear layers lack inductive bias, making Transformers difficult to train on small datasets. The circular pattern of C3A serves as an effective inductive bias to improve generalization when downstream data is limited.

## Key Experimental Results

### Main Results: LLaMA-8B Fine-Tuning Comparison

| Method | Trainable Parameters | Extra Memory | Time Complexity | Performance |
|------|------------|---------|-----------|------|
| LoRA (r=8) | $r(d_1+d_2)$ | 0 | $O(r(d_1+d_2))$ | Baseline |
| VeRA | $r_v+d_1$ (small) | $r_v(d_1+d_2)$ (large) | $O(r_v(d_1+d_2))$ (slow) | Slightly better |
| **C3A** | $d_1 d_2/b$ | $pb$ (small) | $O((d_1+d_2)/p \log b)$ | **Optimal** |

### Multi-Task Fine-Tuning Results

| Task | LoRA | VeRA | DoRA | **C3A** |
|------|------|------|------|---------|
| Commonsense Reasoning | Baseline | +0.3 | +0.5 | **+1.2** |
| Mathematical Reasoning | Baseline | +0.1 | +0.4 | **+0.9** |
| Instruction Following | Baseline | +0.2 | +0.6 | **+1.1** |

### Ablation Study: Rank Decoupling Verification

| Configuration | Parameters | Actual Rank | Performance |
|------|--------|-------|------|
| LoRA r=8 | 8(d₁+d₂) | ≤8 | Baseline |
| LoRA r=64 | 64(d₁+d₂) | ≤64 | +1.5 |
| C3A b=d | d | Theoretical upper bound d | **+1.8** |

### Key Findings
- C3A consistently outperforms LoRA with comparable or even fewer parameters, benefiting from rank decoupling.
- While VeRA uses fewer parameters, its memory/computational overhead is high, making actual deployment costly; C3A balances all three.
- The inductive bias of the circular pattern provides additional gains in small-data fine-tuning.
- Integration into the HuggingFace PEFT library demonstrates the method's engineering practicality.

## Highlights & Insights
- **Core Contribution of Rank-Parameter Decoupling**: This is a conceptual breakthrough in the PEFT domain, proving that high-rank adaptation does not necessarily require a large parameter size.
- **FFT Intersects Signal Processing and Deep Learning**: The highly efficient computations maturely optimized in signal processing for circular convolution are directly migrated to the PEFT context.
- **Flexibility of Block-Circular Expansion**: As a hyperparameter, $b$ provides adjustable capacity similar to $r$, but is more flexible.
- **HuggingFace PEFT Integration**: This demonstrates that the method is empirically validated in engineering terms and can be directly deployed in production environments.

## Limitations & Future Work
- **Expressive Capacity Bound of Circulant Matrices**: Whether the structural constraints of circulant matrices limit expressiveness on certain tasks remains to be studied, despite their flexible rank.
- **$b$ Selection Dependency on $\gcd(d_1, d_2)$**: When $d_1, d_2$ are coprime, $\gcd = 1$, reducing the method to full-parameter fine-tuning, which necessitates architectural dimension adjustments.
- **Integration Potential with LoRA**: Whether circular convolution and low-rank decomposition can complement each other remains unexplored.
- **Validation in Vision Models**: The primary experiments are conducted on LLMs; fine-tuning efficacy on ViTs in the CV domain is yet to be verified.

## Related Work & Insights
- **vs LoRA (Hu et al., 2021)**: Pioneering work on low-rank decomposition where rank is strictly limited by $r$—C3A decouples this constraint.
- **vs VeRA (Kopiczko et al., 2023)**: Leverages fixed random matrices to achieve a high rank, but at the cost of high computational/memory overhead—C3A resolves this efficiency challenge with FFT.
- **vs DoRA (Liu et al., 2024)**: Focuses on orthogonal fine-tuning directions—which is orthogonal to C3A's circular structure and potentially complementary.
- **vs Circulant Matrices in Compression (Cheng et al., 2015)**: Validated on LeNet in early stages but not scaled to LFMs; C3A is the first to demonstrate the feasibility of circular convolution in modern large model fine-tuning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Rank-parameter decoupling represents a conceptual breakthrough in the PEFT field.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparison across multiple models and tasks, integrated into the PEFT library.
- Writing Quality: ⭐⭐⭐⭐ Clear theory and well-founded motivation.
- Value: ⭐⭐⭐⭐⭐ High practical engineering value and already adopted by the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] State-offset Tuning: State-based Parameter-Efficient Fine-Tuning for State Space Models](state_offset_tuning_ssm_peft.md)
- [\[ICML 2025\] Parameter-Efficient Fine-Tuning of State Space Models](../../ICML2025/model_compression/parameter-efficient_fine-tuning_of_state_space_models.md)
- [\[ACL 2025\] Trans-PEFT: Transferable Parameter-Efficient Fine-Tuning on Evolving Base Models](trans_peft_transferable.md)
- [\[ACL 2025\] L4Q: Parameter Efficient Quantization-Aware Fine-Tuning on Large Language Models](l4q_parameter_efficient_quantization_aware_finetuning.md)
- [\[ACL 2025\] Quaff: Quantized Parameter-Efficient Fine-Tuning under Outlier Spatial Stability Hypothesis](quaff_quantized_peft.md)

</div>

<!-- RELATED:END -->
