---
title: >-
  [Paper Note] Untitled
description: >-
  [ACL 2025][Model Compression][Once-for-All] Academic paper note for Untitled.
tags:
  - ACL 2025
  - Model Compression
  - Once-for-All
  - LoRA
date: 2026-05-08
content_hash: f28c226ba8c6264f
---
# One QuantLLM for ALL: Fine-tuning Quantized LLMs Once for Efficient Deployments

## Basic Information

- **Conference**: ACL 2025
- **arXiv**: [2405.20202](https://arxiv.org/abs/2405.20202)
- **Code**: Not released
- **Area**: Model Compression
- **Keywords**: LLM Quantization, Once-for-All, Mixed Precision, LoRA, Quantization-Aware Fine-Tuning
- **TL;DR**: Proposes the LLM-QFA framework, which eliminates weight-sharing interference by assigning independent LoRA adapters to each quantization bit-width. It also introduces a non-parametric resource-balanced sampling scheduler, enabling one-time training to derive optimal mixed-precision quantized subnetworks for diverse deployment scenarios.

## Background & Motivation

- **Key Challenge**: LLM inference demands substantial GPU memory (e.g., LLaMA-70B requires at least 280GB). Quantization is the primary compression method, but different deployment scenarios (servers/PCs/edge devices) require quantized models with different bit-widths. Traditional methods require repeating computationally expensive quantization-aware training (QAT) for each scenario, causing deployment costs to scale linearly with the number of target scenarios.
- **Limitations of Prior Work**: **PTQ (Post-Training Quantization)** such as GPTQ/AWQ is fast but suffers from severe performance degradation at ultra-low bit-widths (2-bit, 3-bit), failing to meet deployment requirements. **QAT (Quantization-Aware Fine-Tuning)** such as QA-LoRA can mitigate performance loss but requires independent training for each bit-width, which incurs extremely high costs for multi-scenario deployment.
- **Design Motivation**: The Once-for-All (OFA) paradigm avoids repetitive training in traditional vision models by training a supernet and extracting subnetworks on demand. However, extending it to LLMs faces two core challenges: **(1) Interference issue**: The magnitude of quantization noise varies significantly across different bit-widths (2/3/4-bit), and weight sharing leads to conflicts in optimization directions. **(2) Efficiency issue**: Traditional QAT relies on full-parameter fine-tuning, which remains computationally intractable at the scale of LLMs even when combined with weight sharing.
- **Goal**: Design an OFA quantization training framework tailored for LLMs that trains once to derive the optimal quantized subnetworks for any resource-constrained scenario, reducing the marginal cost of multi-scenario deployment from "re-training" to "fast searching."

## Method

### Overall Architecture

LLM-QFA (Quantization-Aware Fine-tuning one LLM for All scenarios) decomposes multi-scenario quantized deployment into three stages: First, pre-trained weights are post-training quantized to 2/3/4-bit via GPTQ and frozen. Second, an independent LoRA adapter is assigned to each quantization configuration, and a one-time supernet training is conducted using a resource-balanced sampling strategy. Finally, the optimal mixed-precision subnetwork under given resource constraints is searched without any additional training. The entire pipeline takes only about 8 GPU hours on a single A100 to complete supernet training for LLaMA2-7B.

### Key Designs

**1. Interference-Less Fine-tuning**

The weight-sharing strategy in traditional OFA allows all bit-width configurations to share the same weights. However, the magnitude of quantization noise varies dramatically across different bit-widths, and shared updates lead to severe optimization interference. The solution proposed in this paper is to **assign independent LoRA adapters to each quantization configuration**, defining the forward pass as:

$$\mathbf{Y} = \alpha_i \cdot \hat{\mathbf{W}_i} \cdot \mathbf{X} + \beta_i \cdot \mathbf{X} + \mathbf{B_i A_i} \cdot \mathbf{X}$$

Where $\hat{\mathbf{W}_i}$ represents the weight pre-quantized to the $i$th bit-width via GPTQ and frozen, and $\mathbf{A_i}, \mathbf{B_i}$ are the LoRA weights exclusive to this bit-width. During training, only the sampled LoRA adapter is updated at each step, fundamentally eliminating gradient interference among different bit-widths. Meanwhile, this approach follows the constraints of QA-LoRA, ensuring that the LoRA weights maintain their quantization properties after merging back into the quantized weights, thereby preserving inference acceleration. The extra parameters introduced by LoRA are negligible compared to the size of the LLM itself.

**2. Resource-Balance Sampling**

On the surface, traditional uniform sampling strategies select quantization configurations for each layer with equal probability $P(Q_{l,i}) = 1/N$. However, according to the Central Limit Theorem, the variance of the average bit-width of the sampled subnetworks is $\text{Var}[Bit(s)] = \sigma^2 / L$. When the number of layers $L=32$, the variance is extremely small, making the sampling distribution closely resemble a narrow Gaussian distribution centered at 3-bit. This causes subnetworks with extreme configurations (close to all-2-bit or all-4-bit) to be severely under-sampled. In the interference-less setting, because each bit-width's LoRA is updated independently, under-sampling directly leads to under-fitting, raising a more prominent issue.

This paper proposes using a **non-parametric triangular wave scheduler** to dynamically adjust the mean of the sampling distribution, thereby forcing the mixed Gaussian distribution to approximate a uniform distribution:

$$E[Bit(s,t)] = (b_N - b_1) \cdot |2 \cdot \frac{t}{SL} - 1|$$

Within a scheduling period $SL$, the mean of the sampling distribution smoothly shifts from the highest bit-width to the lowest and back, thereby ensuring that subnetworks of various average bit-widths receive sufficient training resources.

**3. Search Optimized Subnet**

The search process is completely decoupled from training and requires no additional retraining, which is the core mechanism of LLM-QFA to achieve "train once, deploy many times." The detailed process consists of three steps:

- **Step 1 — Coarse-grained Exploration**: Randomly sample 100 mixed-precision subnetworks from the supernet, evaluate their performance on the validation set, and establish an initial mapping between performance and configurations.
- **Step 2 — Sensitivity Analysis**: Conduct correlation analysis between subnetwork performance and the quantization bit-width of each layer, identify critical layers sensitive to quantization, and shrink the search space based on the sensitivity ranking.
- **Step 3 — Fine-grained Search**: Sample another 50 subnetworks from the narrowed space and select the optimal configuration meeting the given resource constraints.

The entire search process only requires a few forward passes, with costs far lower than a single complete QAT training process.

## Experiments

### Experimental Setup

| Item | Configuration |
|------|------|
| **Models** | LLaMA2-7B, LLaMA2-13B |
| **Quantization Method** | GPTQ (2/3/4-bit), configuration consistent with QA-LoRA |
| **Training Data** | Alpaca (52K instruction data) |
| **Optimizer** | Paged AdamW, batch size 16, lr = 2×10⁻⁵ |
| **Scheduling Period** | $SL$ = 8K steps |
| **Training Cost** | Single A100, LLaMA2-7B takes approx. **8 GPU hours** (10K steps) |
| **Evaluation Benchmarks** | MMLU (0-shot/5-shot), Common Sense QA (7 datasets) |
| **Subnet Search** | [100, 50] two-stage search, searching on evaluation set for MMLU and ARC-C for CSQA |

### Main Results

**Table 1: MMLU 5-shot average accuracy (%) — average across all bit-widths**

| Method | LLaMA2-7B (4/3/2-bit Avg) | LLaMA2-13B (4/3/2-bit Avg) |
|------|:-:|:-:|
| GPTQ | 33.3 | 40.0 |
| QA-LoRA | 37.6 | 45.8 |
| **LLM-QFA** | **37.9** | **46.1** |

**Table 2: Common Sense QA 5-shot accuracy (%) — average across 7 tasks**

| Method | Bit | LLaMA2-7B Avg | LLaMA2-13B Avg |
|------|:---:|:-:|:-:|
| GPTQ | 4 | 71.3 | 75.0 |
| QA-LoRA | 4 | 72.4 | 75.2 |
| **LLM-QFA** | **4** | **72.4** | **75.4** |
| GPTQ | 3 | 52.1 | 67.7 |
| QA-LoRA | 3 | 67.7 | 68.5 |
| **LLM-QFA** | **3** | **68.0** | **72.0** |
| GPTQ | 2 | 36.8 | 35.7 |
| QA-LoRA | 2 | 61.7 | 68.1 |
| **LLM-QFA** | **2** | **61.7** | **68.4** |

### Efficiency Analysis & Ablation Study

- **Multi-Scenario Deployment Efficiency**: QA-LoRA requires $N$ independent training runs, causing costs to grow linearly. LLM-QFA requires only 1 training run and $N$ quick searches, where the search overhead is significantly lower than retraining. GPTQ is faster, but its performance degradation at 2/3-bit is unacceptable.
- **Validation of Mixed-Precision Superiority**: By sampling 100 mixed-precision configurations from GPTQ and QA-LoRA to compare with LLM-QFA, it is shown that LLM-QFA performs more robustly across all resource constraints, proving that the advantage stems not only from the mixed-precision scheme itself but also from the superior supernet training quality.
- **Performance Under Various Resource Constraints**: LLM-QFA achieves 45.0% ARC-C accuracy at an average bit-width of 2.1, outperforming QA-LoRA by approximately 5% under the same resource constraint. The resources required to achieve QA-LoRA's 3-bit performance level on ARC-C are reduced by 1.2 times.
- **Ablation — Interference-Less Fine-tuning**: The shared-LoRA variant (where all bit-widths share a single LoRA adapter) consistently scores lower than the independent LoRA version across all resource requirements, validating the necessity of the decoupled design.
- **Ablation — Resource-Balanced Sampling**: The uniform sampling version underperforms the balanced sampling strategy even within its theoretical "advantageous range" (around 3-bit), indicating that the under-fitting issue has a global impact.
- **Ablation — Scheduler Configurations**: A short period ($SL$=1K steps) harms robustness, particularly affecting the convergence of low-bit configurations. A long period (16K steps) shows little difference compared to the default 8K steps. The scheduling order (easy-to-hard vs. hard-to-easy) has negligible impact.

## Highlights & Insights

1. **First to introduce the OFA paradigm to the LLM quantization scenario**, reducing the marginal cost of multi-scenario deployment from "re-training" to "zero-cost search." The problem definition holds great practical significance, filling a methodological gap in highly efficient multi-scenario LLM deployment.
2. **The decoupled LoRA design serves multiple purposes simultaneously**: it eliminates weight-sharing interference, maintains training efficiency at the level of LoRA, and incurs negligible parameter overhead, making it a key enabling technology for scaling OFA to LLMs.
3. **Excellent theoretical analysis of sampling bias**: Rigorously proves the inherent bias of uniform sampling in deep networks based on the Central Limit Theorem. The proposed triangular wave scheduler is simple yet effective, introducing no additional learnable parameters.
4. **Significant advantages observed in the 3-bit scenario for the 13B model** (72.0% vs. 68.5%), indicating that larger models possess higher redundancy, making the search space of layer-level mixed-precision more valuable.
5. **Fully decoupled training and search design** ensures excellent scalability of the framework—new deployment scenarios only require running the search algorithm, avoiding any additional GPU training overhead.

## Limitations & Future Work

1. Only evaluated on LLaMA2-7B/13B; the generalization to newer architectures such as LLaMA3 and Mistral remains untested, and the compatibility of the framework with new attention mechanisms like GQA/SWA is unknown.
2. Quantization granularity is limited to 2/3/4-bit choices, without exploring finer-grained (e.g., 2.5-bit) or group-level mixed-precision, leading to a restricted search space.
3. Evaluation is concentrated on knowledge understanding and common sense reasoning tasks, lacking assessments of generation quality (translation, summarization, dialogue) and long-context scenarios.
4. Compares with limited baselines and lacks comparisons with more advanced PTQ techniques like QuIP# and AQLM, which may overestimate the relative advantage under low-bit configurations.
5. Supernet training utilizes Alpaca (52K samples), which is a relatively small dataset scale; whether the advantages persist on larger-scale instruction data remains to be validated.

## Rating ⭐⭐⭐⭐

- **Novelty**: ⭐⭐⭐⭐ — First to introduce OFA to LLM quantization, with a clear problem definition and a novel combination of decoupled LoRA and balanced sampling.
- **Experimental Thoroughness**: ⭐⭐⭐ — Ablation studies are complete, but model and task coverage is quite limited, lacking generation task evaluations.
- **Value**: ⭐⭐⭐⭐⭐ — Exceptional practical value, as a single training run of 8 GPU hours can cover diverse deployment needs.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured, presenting a coherent logical chain from motivation to method and experiments, with rigorous theoretical analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] C3A: Parameter-Efficient Fine-Tuning via Circular Convolution](parameter-efficient_fine-tuning_via_circular_convolution.md)
- [\[ACL 2025\] DoMIX: An Efficient Framework for Exploiting Domain Knowledge in Fine-Tuning](domix_an_efficient_framework_for_exploiting.md)
- [\[ACL 2025\] Quaff: Quantized Parameter-Efficient Fine-Tuning under Outlier Spatial Stability Hypothesis](quaff_quantized_peft.md)
- [\[ACL 2025\] L4Q: Parameter Efficient Quantization-Aware Fine-Tuning on Large Language Models](l4q_parameter_efficient_quantization_aware_finetuning.md)
- [\[ACL 2025\] FedEx-LoRA: Exact Aggregation for Federated and Efficient Fine-Tuning of Large Language Models](fedex_lora_federated_exact_aggregation.md)

</div>

<!-- RELATED:END -->
