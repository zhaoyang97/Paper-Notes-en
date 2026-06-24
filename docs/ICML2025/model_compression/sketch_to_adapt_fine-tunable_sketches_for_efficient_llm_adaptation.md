---
title: >-
  [Paper Note] Sketch to Adapt: Fine-Tunable Sketches for Efficient LLM Adaptation
description: >-
  [ICML 2025][Model Compression][Parameter Sharing] SpaLLM proposes a parameter sharing method based on sketching to unify the compression and fine-tuning processes of LLMs. By compressing pre-trained weights into a lookup table (LUT) and directly fine-tuning on the table values, this approach avoids the low-rank assumption and implementation complexity of dual-tower architectures like QLoRA. It achieves superior performance compared to QLoRA/LoftQ across multiple benchmarks wi…
tags:
  - "ICML 2025"
  - "Model Compression"
  - "Parameter Sharing"
  - "Sketching"
  - "Compressive Adaptation"
  - "Lookup Table Fine-Tuning"
  - "Unified Compression and Adaptation"
date: 2026-05-08
content_hash: 98bb0e01d0c1b5f2
---

# Sketch to Adapt: Fine-Tunable Sketches for Efficient LLM Adaptation

**Conference**: ICML 2025  
**arXiv**: [2410.06364](https://arxiv.org/abs/2410.06364)  
**Code**: No public code  
**Area**: Model Compression / Efficient LLM Fine-Tuning  
**Keywords**: Parameter Sharing, Sketching, Compressive Adaptation, Lookup Table Fine-Tuning, Unified Compression and Adaptation

## TL;DR
SpaLLM proposes a parameter sharing method based on sketching to unify the compression and fine-tuning processes of LLMs. By compressing pre-trained weights into a lookup table (LUT) and directly fine-tuning on the table values, this approach avoids the low-rank assumption and implementation complexity of dual-tower architectures like QLoRA. It achieves superior performance compared to QLoRA/LoftQ across multiple benchmarks with fewer trainable parameters.

## Background & Motivation

**Background**: Fine-tuning Large Language Models (LLMs) is a critical step for downstream applications. Since the parameter scale of LLMs is massive, full-precision fine-tuning is impractical. Consequently, "compressive adaptation" methods have emerged, which first compress the model parameters and then perform parameter-efficient fine-tuning (PEFT). QLoRA is the most representative method in this field.

**Limitations of Prior Work**:
   - **Invalidity of Low-Rank Assumption**: Both QLoRA and LoftQ assume that the weight differences before and after fine-tuning are low-rank. However, research indicates that the weight differences between full fine-tuning and base models are often high-rank.
   - **Complexity of Dual-Tower Architectures**: Methods like QLoRA employ a dual-tower structure of "compressed weights + full-precision adapters". During inference, two computational paths are required (quantized weight matrix multiplication + adapter matrix multiplication). Operations with different precisions need to be processed separately, leading to high implementation complexity.
   - **Low-Bit Dilemma**: Under 3-bit or lower quantization, QLoRA often struggles to converge.

**Key Challenge**: Compression and adaptation are fragmented into two independent stages—first quantization and then adding adapters. This not only introduces unnecessary architectural complexity but also restricts the adapter's capacity to compensate for quantization loss.

**Key Insight**: Can model compression and fine-tuning be unified into a single process? That is, directly fine-tuning on the compressed parameters without requiring additional adapters?

**Core Idea**: Utilize a sketching algorithm to map each row of weights to a small lookup table (a set of centroids), fix the mapping relationship (sketching matrix $\Pi$), and directly fine-tuning on the values in the lookup table.

## Method

### Overall Architecture
The workflow of SpaLLM is divided into two phases:
1. **Compression Phase**: Perform parameter sketching on each row of weights of the pre-trained LLM, mapping it to a lookup table $w \in \mathbb{R}^k$ and a one-hot sketching matrix $\Pi \in \mathbb{R}^{d \times k}$ such that $\hat{\theta} = \Pi w$.
2. **Adaptation Phase**: Keep $\Pi$ fixed and directly perform task-specific fine-tuning on the floating-point lookup table $w$.

### Key Designs

1. **Row-wise Parameter Sketching**:

    - For each row $\theta \in \mathbb{R}^d$ of the weight matrix $\Theta \in \mathbb{R}^{n \times d}$, approximate it as $\hat{\theta} = \Pi w$.
    - $\Pi$ is a $d \times k$ one-hot matrix where each row has exactly one 1, mapping each element of $\theta$ to a specific entry in $w$.
    - Since $k < d$, multiple weight elements share the same entry value, achieving parameter sharing and compression.
    - **Design Motivation**: This parameter sharing does not rely on low-rank assumptions—even a full-rank matrix (such as an identity matrix) can be perfectly represented using a small number of shared values.

2. **Weighted Lloyd's Algorithm for Learning Sketching**:

    - Static random hash mapping is not used (LLM weights do not exhibit heavy-hitter patterns, and random mapping causes severe degradation).
    - Instead, a weighted Lloyd's clustering algorithm (a variant of k-means) is used to learn the optimal $k$ centroids.
    - Weights are inversely weighted by the Hessian diagonal $\text{diag}(H^{-1}) = \text{diag}((XX^T)^{-1})$, assigning higher clustering precision to more sensitive parameters.
    - After learning the centroids, an iterative loss error-compensation framework (similar to the column-wise greedy method in GPTQ) is used to determine the mapping for each parameter.
    - **Design Motivation**: LLM weights are sensitive to perturbations, and random sketching leads to severe degradation. Utilizing Hessian information for weighted clustering minimizes the impact on model output.

3. **Groups Per Row (GPR) Extension**:

    - To increase the number of learnable parameters (improving expressiveness), each row of weights is split into multiple continuous groups.
    - Each group independently maintains its own lookup table. The size of the mapping matrix remains unchanged, but the total number of trainable parameters increases.
    - For example, GPR=8 means each row is divided into 8 groups, with each group having its own LUT.
    - **Design Motivation**: By controlling the GPR value, the trade-off between "compression rate" and "accuracy" can be flexibly adjusted.

4. **Single-Tower Unified Architecture**:

    - During inference, only a single compressed matrix multiplication is required: lookup table indexing + accumulation, eliminating the dual-tower structure.
    - Existing efficient kernels (such as SqueezeLLM kernels) can be leveraged to accelerate inference.
    - For different tasks, only the LUT values need to be stored and switched, while the mapping relationship $\Pi$ is shared.
    - **Design Motivation**: Avoid dual-path inference in QLoRA, reducing system complexity and inference latency.

### Loss & Training
During the fine-tuning phase, the standard language modeling loss (cross-entropy for next-token prediction) is used. Since only the values in the LUT are updated, gradients of multiple virtual parameters are aggregated and averaged into the same trainable parameter, which inherently acts as a regularization effect. The learning rate is searched from $(1\times10^{-4}, 5\times10^{-5}, ..., 2\times10^{-6})$, and training is conducted for 10 epochs.

## Key Experimental Results

### Main Results

| Model/Dataset | Method | Bits | Trainable Params | WikiText-2 PPL | GSM8K Acc |
|------------|------|------|----------|---------------|-----------|
| LLaMA-2-7B | LoRA (r=64) | 16 | 160M | 5.08 | 36.9% |
| LLaMA-2-7B | QLoRA (r=64) | 4 | 160M | 5.70 | 35.1% |
| LLaMA-2-7B | LoftQ (r=64) | 4 | 160M | 5.24 | 35.0% |
| LLaMA-2-7B | SpaLLM (GPR=8) | 4 | - | 5.32 | **38.4%** |
| LLaMA-2-7B | QLoRA (r=64) | 2 | 160M | N.A. | N.A. |
| LLaMA-2-7B | LoftQ (r=64) | 2 | 160M | 7.85 | 20.9% |
| LLaMA-2-7B | SpaLLM (GPR=8) | 2 | - | 7.40 | **23.7%** |
| LLaMA-2-13B | SpaLLM (GPR=1) | 3 | 22M | **5.05** | - |
| LLaMA-3-70B | SpaLLM (GPR=4) | 4 | - | - | AVG 0.72 |

### Ablation Study

| GPR Value | Model | GSM8K Acc | Note |
|--------|------|-----------|------|
| GPR=1 | LLaMA-2-7B | ~30% | Fewest parameters, baseline performance |
| GPR=2 | LLaMA-2-7B | ~33% | Performance increases with GPR |
| GPR=4 | LLaMA-2-7B | ~35% | Close to QLoRA baseline |
| GPR=8 | LLaMA-2-7B | 38.4% | Outperforms all baselines |
| GPR=1 | LLaMA-2-13B | Outperforms baselines | Only 1/5 of the trainable parameters |

### Key Findings
- On the 13B model, SpaLLM with only GPR=1 (22M trainable parameters) outperforms all baselines; on the 7B model, GPR=8 is required to surpass the baselines.
- At 2-bit, QLoRA fails to converge completely, whereas SpaLLM still maintains reasonable performance.
- Inference efficiency: SpaLLM is approximately 3x faster than QLoRA/LoftQ and exhibits lower GPU memory usage (due to the single-tower architecture).
- On LLaMA-3-70B, the 4-bit SpaLLM fine-tuned version (39.8GB) outperforms the full-precision base model and can fit into a single L40S-48GB GPU.
- In LLM-as-a-judge evaluation, SpaLLM achieves a win-loss ratio of 0.61 (vs. LoftQ) and 91% (vs. Falcon-40B-Instruct).

## Highlights & Insights
- **Unified Perspective of "Parameter Sharing = Regularization + Compression"**: Sharing a single lookup table value across multiple weight positions naturally introduces a regularization effect, mitigating overfitting.
- **Breaking the Shackles of the Low-Rank Assumption**: Proves that parameter sharing is more suitable for compressive adaptation than low-rank decomposition, as even full-rank matrices can be perfectly represented by shared parameters.
- **Inherent Advantage of Inference Efficiency**: The single-tower architecture eliminates dual-path computation, which is critically important in multi-user concurrent serving scenarios.
- **Transferable Trick**: The combination of weighted Lloyd's clustering + Hessian weighting can be applied to any compression method requiring weight clustering.

## Limitations & Future Work
- Storing the sketching matrix $\Pi$ also requires space (indices of the one-hot matrix). Although the paper encodes this via bit count, the overhead is not discussed in detail.
- The method is only validated on the LLaMA series, lacking experiments on other architectures (e.g., Mistral, Qwen).
- The choice of GPR is currently manual; adaptive allocation of different GPRs to different layers could be considered.
- The combination with newer quantization methods such as GPTQ has not been explored.

## Related Work & Insights
- **vs QLoRA/LoftQ**: The core difference is that SpaLLM replaces the low-rank adapter with parameter sharing, eliminating the dual-tower architecture and the low-rank assumption. The advantages are prominent on low-bit and larger models.
- **vs GPTQ**: GPTQ strictly focuses on compression without fine-tuning; the compression phase of SpaLLM borrows the column-wise greedy framework of GPTQ, but the objective is to establish a lookup table rather than perform quantization.
- **vs HashedNets/ROAST**: Early parameter sharing methods used random hashing, suitable for training from scratch; SpaLLM employs learned sketching, which is tailored for post-compression fine-tuning of pre-trained models.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Introduces sketching-based parameter sharing to LLM compressive fine-tuning for the first time, unifying compression and adaptation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on multiple datasets, models, and metrics, with detailed GPR ablation and thorough efficiency comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clear method description, with highly intuitive comparison diagrams with QLoRA.
- Value: ⭐⭐⭐⭐⭐ Proposes a new compression fine-tuning paradigm with significant advantages in low-bit scenarios, demonstrating high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Sketch Down the FLOPs: Towards Efficient Networks for Human Sketch](../../CVPR2025/model_compression/sketch_down_the_flops_towards_efficient_networks_for_human_sketch.md)
- [\[ICML 2025\] BlockDialect: Block-wise Fine-grained Mixed Format Quantization for Energy-Efficient LLM Inference](blockdialect_block-wise_fine-grained_mixed_format_quantization_for_energy-effici.md)
- [\[ICML 2025\] Parameter-Efficient Fine-Tuning of State Space Models](parameter-efficient_fine-tuning_of_state_space_models.md)
- [\[NeurIPS 2025\] RefLoRA: Refactored Low-Rank Adaptation for Efficient Fine-Tuning of Large Models](../../NeurIPS2025/model_compression/reflora_refactored_low-rank_adaptation_for_efficient_fine-tuning_of_large_models.md)
- [\[ICLR 2026\] Efficient Orthogonal Fine-Tuning with Principal Subspace Adaptation](../../ICLR2026/model_compression/efficient_orthogonal_fine-tuning_with_principal_subspace_adaptation.md)

</div>

<!-- RELATED:END -->
