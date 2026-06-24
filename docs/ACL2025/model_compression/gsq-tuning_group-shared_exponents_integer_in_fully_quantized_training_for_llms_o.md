---
title: >-
  [Paper Note] GSQ-Tuning: Group-Shared Exponents Integer in Fully Quantized Training for LLMs On-Device Fine-tuning
description: >-
  [ACL 2025 (Findings)][Model Compression][Fully Quantized Training] GSQ-Tuning proposes a fully quantized fine-tuning framework based on the "Group-Shared Exponents Integer" (GSEI) format. By completely eliminating floating-point operations in both inference and training, it is combined with LoRA adapters to achieve on-device LLM fine-tuning that is close to BF16 fine-tuning in accuracy, while reducing memory by 1.85x, power consumption by 5x, and silicon area by 11x.
tags:
  - "ACL 2025 (Findings)"
  - "Model Compression"
  - "Fully Quantized Training"
  - "Integer Fine-Tuning"
  - "On-Device Deployment"
  - "Shared Exponents"
  - "LoRA Quantization"
date: 2026-05-08
content_hash: abfefe33d8961fd1
---

# GSQ-Tuning: Group-Shared Exponents Integer in Fully Quantized Training for LLMs On-Device Fine-tuning

**Conference**: ACL 2025 (Findings)  
**arXiv**: [2502.12913](https://arxiv.org/abs/2502.12913)  
**Code**: None  
**Area**: Model Compression / LLM Efficiency  
**Keywords**: Fully Quantized Training, Integer Fine-Tuning, On-Device Deployment, Shared Exponents, LoRA Quantization

## TL;DR

GSQ-Tuning proposes a fully quantized fine-tuning framework based on the "Group-Shared Exponents Integer" (GSEI) format. By completely eliminating floating-point operations in both inference and training, it is combined with LoRA adapters to achieve on-device LLM fine-tuning that is close to BF16 fine-tuning in accuracy, while reducing memory by 1.85x, power consumption by 5x, and silicon area by 11x.

## Background & Motivation

**Background**: Fine-tuning Large Language Models (LLMs) has become the standard practice for adapting to downstream tasks. Parameter-efficient fine-tuning (PEFT) methods like LoRA reduce memory requirements by training only a small number of parameters, and quantization methods (such as QLoRA) further reduce memory footprints through low-precision storage. However, the forward and backward propagation calculations during fine-tuning still rely on floating-point operations (FP16/BF16).

**Limitations of Prior Work**: Edge devices (mobile phones, IoT devices, embedded systems) typically lack efficient floating-point units—the power consumption and silicon area of floating-point multipliers are much larger than those of integer multipliers. Even if existing methods quantize model weight storage, they still need to dequantize back to floating-point representation for computation during training, making fine-tuning on edge devices impractical. Furthermore, transmitting sensitive data to the cloud for fine-tuning raises privacy concerns.

**Key Challenge**: PEFT reduces training parameters but not the requirement for computational precision; inference quantization reduces storage, but training still requires floating-point operations; existing attempts at fully quantized training (such as INT8 training) suffer from significant accuracy degradation in LLMs. The fundamental issue is how to perform the entire training workflow (forward + backward + optimizer updates) using integer operations while maintaining accuracy.

**Goal**: Design a fully integer-based LLM fine-tuning scheme that eliminates all floating-point operations during both inference and training, while maintaining accuracy comparable to BF16 fine-tuning.

**Key Insight**: The authors notice that the core issue of floating-point numbers lies in the exponent—each number carries an independent exponent, necessitating floating-point handling during multiplication. If a group of parameters shares a single exponent, then calculations within the group can be completed entirely with integers, leaving only the scaling of exponents to be handled between groups.

**Core Idea**: Propose the Group-Shared Exponents Integer (GSEI) data format, which groups parameters so that each group shares a scaling factor, and parameters within the group are represented as integers. Consequently, the vast majority of calculations in matrix multiplication become integer multiply-accumulate operations, with only a small number of inter-group scaling operations involving exponent manipulation, which can be precomputed.

## Method

### Overall Architecture

GSQ-Tuning consists of three core components: (1) the GSEI data format—representing model parameters in the form of "group-shared exponent + integer mantissa"; (2) fully integer forward propagation—performing matrix multiplication using the GSEI format to avoid floating-point operations; and (3) fully integer backward propagation—gradient computation and LoRA parameter updates are also completed in the GSEI format. The entire framework is integrated with low-rank adapters like LoRA to achieve dual optimization of parameter efficiency and full quantization.

### Key Designs

1. **GSEI Data Format (Group-Shared Exponents Integer)**:

    - **Function**: Efficiently represent floating-point parameters in integer form to eliminate element-wise floating-point operations.
    - **Mechanism**: Group parameter vectors/matrices by a fixed size (e.g., every 128 elements), extract a shared scaling factor (scale, which can be seen as a shared exponent) for each group, and round the elements in the group to integers after dividing by this scaling factor. Mathematically, for a parameter group $\mathbf{x}$, it is represented as $\mathbf{x} \approx s \cdot \mathbf{x}_{int}$, where $s$ is the group-shared scaling factor and $\mathbf{x}_{int}$ is the integer vector. In matrix multiplication $\mathbf{Y} = \mathbf{A} \cdot \mathbf{B}$, if both matrices are in GSEI format, then $\mathbf{Y}_{ij} = s_A^{(i)} \cdot s_B^{(j)} \cdot \sum_k a_{int}^{(ik)} \cdot b_{int}^{(kj)}$, where the inner summation is fully integer calculation, and the multiplication of scaling factors can be completed once at the outermost level.
    - **Design Motivation**: Unlike standard quantization (e.g., INT8), the core innovation of GSEI lies in the "group-shared exponent" design, which allows exponent processing to be moved from the inner loop to the outer loop. This converts $O(n^3)$ floating-point multiplications into $O(n^3)$ integer multiplications + $O(n^2)$ scaling operations, which is highly efficient on hardware. Compared with the MX (Microscaling) format, GSEI possesses more flexible group sizes and is specifically optimized for training scenarios.

2. **Fully Integer Forward/Backward Propagation**:

    - **Function**: Complete the entire forward and backward propagation of training under the GSEI format.
    - **Mechanism**: In forward propagation, all linear layer weights and activations are stored in GSEI format, and matrix multiplication is performed using the aforementioned integer + scaling approach. In backward propagation, gradient computation also uses the GSEI format—the gradient matrix is quantized into GSEI, and the matrix multiplication of gradients and activations (for computing parameter gradients) as well as gradients and weights (for propagating gradients) are all conducted in integer. The key challenge is that gradients typically have a larger and less stable dynamic range than weights and activations, which the authors address through adaptive group sizes and dynamic scaling factor updates.
    - **Design Motivation**: If quantization is only applied in the forward pass while the backward pass still relies on floating-point, the computational bottleneck of training remains unresolved. Full-pipeline quantization is necessary to achieve feasible fine-tuning on edge devices.

3. **GSEI-LoRA Joint Design**:

    - **Function**: Seamlessly combine GSEI quantization with LoRA adapters.
    - **Mechanism**: Pre-trained weights are frozen and stored in GSEI format, and the low-rank matrices $\mathbf{A}$ and $\mathbf{B}$ of LoRA are also initialized in GSEI format. During training, only LoRA parameters are updated, and the updated gradients and optimizer states are also represented in GSEI format. Since the parameter scale of LoRA is much smaller than that of the main model, the memory overhead of the optimizer states is further reduced. The overall memory savings come from two parts: quantized storage of weights (FP16 $\rightarrow$ GSEI, ~2x) + quantization of optimizer states (FP32 $\rightarrow$ GSEI, ~4x).
    - **Design Motivation**: The low-rank nature of LoRA makes the integer approximation error of GSEI more controllable on small matrices—low-rank decomposition itself has a denoising/regularization effect, helping offset the accuracy loss caused by quantization.

### Loss & Training

Standard LoRA fine-tuning loss (such as cross-entropy) is used, but all calculations are performed in the GSEI format. Regarding the training strategy, progressive quantization is adopted—larger group sizes are used in the initial stage (higher accuracy but slightly lower efficiency), and the group size is gradually reduced as training stabilizes. The learning rate needs to be appropriately increased based on the quantization step size of GSEI to compensate for the information loss of quantized gradients.

## Key Experimental Results

### Main Results

Fine-tuning is conducted on multiple LLMs (LLaMA-2 7B/13B, LLaMA-3 8B, etc.), comparing BF16 LoRA, INT8 LoRA, QLoRA, and GSQ-Tuning:

| Method | Precision Format | LLaMA-2 7B Avg Accuracy | Memory (GB) | Relative BF16 Memory |
|------|---------|-------------------|-----------|---------------|
| BF16 LoRA | FP16 Weights + FP32 Optimizer | Baseline (100%) | ~14 | 1.0x |
| QLoRA (4bit) | INT4 Weights + FP16 Computation | -0.5~1% | ~8 | 0.57x |
| INT8 LoRA | INT8 Full Pipeline | -2~3% | ~8 | 0.57x |
| GSQ-Tuning | GSEI Full Pipeline | -0.3~0.8% | ~7.5 | 0.54x (1.85x Savings) |

### Ablation Study

| Configuration | Accuracy Loss | Description |
|------|---------|------|
| GSEI (Group Size = 128) | Minimal | Best accuracy-efficiency trade-off |
| GSEI (Group Size = 64) | Acceptable | More aggressive but still usable |
| GSEI (Group Size = 256) | Small | Good accuracy but reduced memory savings |
| Without Dynamic Scaling | Large | Large gradient dynamic range requires adaptation |
| Without Progressive Quantization | Early Instability | Initial large group size helps |
| GSEI vs MX Format | GSEI Better | Group-shared exponent is more flexible than MX block format |

Hardware efficiency comparison (GSEI INT8 vs FP8):

| Metric | GSEI INT8 | FP8 | Advantage |
|------|-----------|-----|------|
| Power Consumption | 1x | 5x | GSEI power consumption reduced by 5x |
| Silicon Area | 1x | 11x | GSEI area reduced by 11x |
| Throughput | Comparable | Comparable | Integer multiplier is more efficient |

### Key Findings

- **GSQ-Tuning suffers minimal accuracy loss**: Compared with BF16 LoRA fine-tuning, the accuracy gap is typically within 1%, which is far superior to the 2-3% loss of direct INT8 quantized training. The core reason is that GSEI's group-shared exponent design preserves accuracy while achieving fully integer computation.
- **Memory savings mainly come from optimizer states**: Weight quantization saves around 2x, but quantizing optimizer states (Adam's first and second moments) from FP32 to GSEI contributes to larger savings.
- **Group size 128 is the sweet spot**: Under 128, accuracy starts to degrade; above 128, memory savings diminish. This is consistent with the optimal group sizes of formats like MXFP.
- **Hardware advantages are highly significant**: The power consumption and area of INT8 multipliers are much smaller than those of FP8 multipliers (5x and 11x), which is a decisive advantage for battery-powered edge devices.

## Highlights & Insights

- **The core design idea of "group-shared exponent"** is extremely elegant—extracting the exponent from the inner loop to the outer loop to convert floats to integers is a simple yet profound engineering insight. This idea can be generalized to any scenario requiring low-power matrix computation.
- **Quantization of the entire training pipeline** (not just inference) bridges the gap between QLoRA and true on-device usability. QLoRA still requires floating-point computation, whereas GSQ-Tuning truly realizes fully integer training.
- **Hardware-level analysis of power consumption and area** is very practical—most quantization papers only compare accuracy and memory, and rarely delve into energy efficiency analysis at the chip design level, which provides excellent reference value for hardware-algorithm co-design.

## Limitations & Future Work

- **Evaluation is only verified on smaller LLMs**: Experiments on 7B/13B models may not fully reflect the performance on 70B+ large models, as the quantization sensitivity of larger models might differ.
- **Lack of deployment validation on real edge devices**: The hardware analysis in the paper is based on theoretical calculations (multiplier area and power consumption formulas) without end-to-end performance testing on actual edge chips (such as NPUs, DSPs).
- **Only evaluated on LoRA fine-tuning**: The applicability of GSEI to full-parameter fine-tuning or other PEFT methods (such as Prefix Tuning, Adapter) remains unverified.
- **Progressive quantization strategy relies on heuristics**: The scheduling strategy for group size needs to be manually designed; automated quantization strategy search might yield better results.
- **Directions for improvement**: Accelerate GSEI format hardware execution in integration with NPU-specific instruction sets; explore mixed-precision schemes—using larger group sizes for quantization-sensitive layers and more aggressive quantization for insensitive layers.

## Related Work & Insights

- **vs QLoRA**: QLoRA quantizes weight storage but training calculations still use floats. GSQ-Tuning goes a step further to realize fully integer training, which is a conceptual generational upgrade.
- **vs INT8 Training (such as S2FP8)**: Direct INT8 training suffers from severe accuracy loss on LLMs. GSQ-Tuning maintains better accuracy at the same bit-width through the group-shared exponent design.
- **vs MXFP/Microscaling format**: MXFP also utilizes shared exponents but is mainly optimized for inference scenarios. GSQ-Tuning designs quantization strategies specifically for training scenarios (including gradients and optimizer states).

## Rating

- Novelty: ⭐⭐⭐⭐ The group-shared exponent design is elegant, and fully integer training is a meaningful advancement, though quantized training is not a completely new direction.
- Experimental Thoroughness: ⭐⭐⭐ Validated across multiple models and tasks with complete ablation studies, but lacks real hardware deployment and experiments on very large models.
- Writing Quality: ⭐⭐⭐⭐ Clear technical descriptions and in-depth hardware analysis, though the equations are dense and require close reading.
- Value: ⭐⭐⭐⭐ Pragmatically advances LLM deployment on edge devices; the hardware efficiency analysis is especially valuable for chip designers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] One QuantLLM for ALL: Fine-tuning Quantized LLMs Once for Efficient Deployments](one_quantllm_for_all_fine-tuning_quantized_llms_once_for_efficient_deployments.md)
- [\[ICLR 2026\] GlowQ: Group-Shared Low-Rank Approximation for Quantized LLMs](../../ICLR2026/model_compression/glowq_group-shared_low-rank_approximation_for_quantized_llms.md)
- [\[ACL 2025\] Quaff: Quantized Parameter-Efficient Fine-Tuning under Outlier Spatial Stability Hypothesis](quaff_quantized_peft.md)
- [\[ICCV 2025\] Generalized Tensor-based Parameter-Efficient Fine-Tuning via Lie Group Transformations](../../ICCV2025/model_compression/generalized_tensor-based_parameter-efficient_fine-tuning_via_lie_group_transform.md)
- [\[ACL 2025\] State-offset Tuning: State-based Parameter-Efficient Fine-Tuning for State Space Models](state_offset_tuning_ssm_peft.md)

</div>

<!-- RELATED:END -->
