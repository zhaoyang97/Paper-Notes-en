---
title: >-
  [Paper Note] CFSP: An Efficient Structured Pruning Framework for LLMs with Coarse-to-Fine Activation Information
description: >-
  [ACL 2025 (COLING 2025)][Model Compression][Structured Pruning] This paper proposes the CFSP framework, which utilizes coarse-grained (inter-block) and fine-grained (intra-block) activation information as importance criteria to guide structured pruning of LLMs. It requires only a single forward pass to complete the pruning and outperforms existing methods across multiple models and sparsity budgets.
tags:
  - "ACL 2025 (COLING 2025)"
  - "Model Compression"
  - "Structured Pruning"
  - "LLM Acceleration"
  - "Activation Information"
  - "Coarse-to-Fine"
  - "Sparsity Allocation"
date: 2026-05-08
content_hash: 43c00413c3d642ef
---

# CFSP: An Efficient Structured Pruning Framework for LLMs with Coarse-to-Fine Activation Information

**Conference**: ACL 2025 (COLING 2025)  
**Area**: Model Compression  
**Keywords**: Structured Pruning, LLM Acceleration, Activation Information, Coarse-to-Fine, Sparsity Allocation

## TL;DR
This paper proposes the CFSP framework, which utilizes coarse-grained (inter-block) and fine-grained (intra-block) activation information as importance criteria to guide structured pruning of LLMs. It requires only a single forward pass to complete the pruning and outperforms existing methods across multiple models and sparsity budgets.

## Background & Motivation

**Background**: The parameter size and computational overhead of Large Language Models (LLMs) are increasingly massive, restricting their deployment in real-world scenarios. Network pruning is an effective model compression technique that achieves acceleration by removing redundant parameters. Currently, LLM pruning is mainly divided into two major directions: unstructured pruning (e.g., SparseGPT, Wanda) and structured pruning.

**Limitations of Prior Work**: Although unstructured pruning performs well in maintaining performance, the resulting sparse weight matrices require dedicated hardware support to achieve actual inference acceleration and do not yield real speedup on general-purpose GPUs. Structured pruning can directly reduce latency on general-purpose hardware, but maintaining model performance under high sparsity ratios remains a challenge. Existing structured pruning methods often uniformly allocate sparsity ratios across all blocks, ignoring the differences in importance among different blocks.

**Key Challenge**: Structured pruning needs to strike a balance between "pruning efficiency" and "performance preservation"—the pruning process itself should not be overly time-consuming (e.g., requiring large amounts of calibration data or multiple iterations), while acceptable performance should still be maintained under high sparsity ratios.

**Goal**: To design an efficient and effective structured pruning framework that can (1) quickly determine pruning strategies (requiring only a single forward pass), (2) adaptively allocate sparsity budgets across different blocks, and (3) maintain good performance under high sparsity ratios.

**Key Insight**: The authors observe that different Transformer blocks contribute very differently to the final output, and activation values can accurately reflect the importance of weights. By simultaneously leveraging inter-block (coarse-grained) and intra-block (fine-grained) activation information, the parameter importance can be evaluated more precisely.

**Core Idea**: Using inter-layer activation distribution to guide sparsity budget allocation and intra-layer activation values to guide specific weight retention, forming a "coarse-to-fine" two-level pruning strategy.

## Method

### Overall Architecture
CFSP adopts a two-stage pruning strategy: first, at the coarse-grained level, different sparsity budgets are allocated based on the activation importance of each Transformer block (important blocks are pruned less, while less important ones are pruned more); then, at the fine-grained level, the most important weights are retained within each block based on activation information. Finally, an adaptive recovery fine-tuning is applied to further improve performance.

### Key Designs

1. **Coarse-grained Block Importance Evaluation**:

    - **Function**: Evaluates the contribution of each Transformer block to the model output, enabling differentiated allocation of sparsity budgets.
    - **Mechanism**: A single forward pass is performed using a small amount of calibration data to calculate the L2 norm of the output activations of each block as the importance metric $I_k = \|A_k\|_2$, where $A_k$ is the output activation of the $k$-th block. A larger norm indicates a greater contribution of the block to information propagation. The sparsity budget is then proportionally allocated based on the importance score: blocks with high importance are assigned a lower sparsity ratio, while blocks with low importance are assigned a higher sparsity ratio.
    - **Design Motivation**: Uniform sparsity allocation ignores inter-layer differences, which can lead to critical layers being over-pruned and collapsing. Differentiated allocation better preserves performance without increasing the overall sparsity ratio.

2. **Fine-grained Weight Retention Strategy**:

    - **Function**: Determines which specific structures (attention heads, FFN neurons) within each block should be retained.
    - **Mechanism**: For attention heads and intermediate FFN neurons in each block, the importance scores of their corresponding activation channels are calculated. The joint importance index is computed as the activation magnitude multiplied by the norm of the corresponding weights. Based on the sparsity budget allocated to the block in the coarse-grained phase, the structural units with the highest importance are retained.
    - **Design Motivation**: Joint evaluation of activations and weights is more accurate than using either metric in isolation, because a large weight does not necessarily mean the channel is frequently activated, and a large activation does not imply the weight itself is crucial.

3. **Adaptive Recovery Fine-tuning**:

    - **Function**: Conducts lightweight fine-tuning on the pruned model to recover part of the performance loss.
    - **Mechanism**: Different fine-tuning learning rates and training steps are allocated to different blocks based on the coarse-grained importance score. Blocks with lower importance (which are pruned more heavily) are allocated larger learning rates and more training resources, as they require more substantial adjustment to adapt to the pruned structure. Recovery is achieved using a small amount of calibration data.
    - **Design Motivation**: Uniformly allocating fine-tuning resources is sub-optimal; adaptive allocation ensures that the most damaged components receive the most recovery resources.

### Loss & Training
Recovery fine-tuning employs the standard language modeling loss (next token prediction) and is conducted on a small number of calibration samples for a short duration. The key innovation lies in allocating learning rates inversely proportional to block importance.

## Key Experimental Results

### Main Results

| Model | Method | Sparsity | WikiText2 PPL↓ | PTB PPL↓ | Avg Zero-shot Acc↑ |
|------|------|--------|---------------|----------|-------------------|
| LLaMA-7B | Dense | 0% | 5.68 | 8.80 | 65.3 |
| LLaMA-7B | LLM-Pruner | 50% | 11.23 | 16.45 | 58.1 |
| LLaMA-7B | SliceGPT | 50% | 10.87 | 15.92 | 58.7 |
| LLaMA-7B | **CFSP** | 50% | **9.42** | **13.68** | **60.2** |
| LLaMA-13B | Dense | 0% | 5.09 | 7.81 | 68.1 |
| LLaMA-13B | LLM-Pruner | 50% | 8.76 | 12.34 | 62.5 |
| LLaMA-13B | **CFSP** | 50% | **7.21** | **10.56** | **64.3** |

### Ablation Study

| Configuration | WikiText2 PPL↓ | Description |
|------|---------------|------|
| CFSP Full Model | 9.42 | Coarse + Fine + Adaptive Fine-tuning |
| w/o Coarse-grained (Uniform Allocation) | 10.85 | PPL increases by 1.43 after removing differentiated allocation |
| w/o Fine-grained (Weight Norm Only) | 10.12 | Do not use activation info for intra-block pruning |
| w/o Adaptive Fine-tuning | 10.68 | Do not perform recovery fine-tuning |
| Uniform Fine-tuning (Non-adaptive) | 9.89 | Fine-tuning without resource allocation based on importance |

### Key Findings
- Coarse-grained differentiated sparsity allocation contributes the most; removing it causes the largest increase in PPL (+1.43), indicating that a uniform sparsity ratio is indeed a major bottleneck in existing methods.
- Fine-grained joint activation-weight evaluation yields a significant improvement over using weight norms alone (PPL decreases by 0.7).
- Adaptive fine-tuning further reduces PPL by 0.47 compared to uniform fine-tuning, validating the effectiveness of allocating fine-tuning resources based on importance.
- Under high sparsity ratios (60%+), the relative advantage of CFSP becomes even more pronounced, as differentiated allocation becomes more critical at higher compression rates.

## Highlights & Insights
- The "coarse-to-fine" two-level evaluation strategy is elegant. The global-to-local workflow is intuitive and highly effective, and it can be generalized to other model compression tasks (such as quantization and distillation).
- Importance evaluation is completed with only a single forward pass, rendering the pruning efficiency significantly higher than methods requiring iterative optimization (e.g., LoRAPrune), which makes it highly viable for real-world deployment.
- The "more damage, more recovery" strategy in adaptive fine-tuning presents an elegant improvement over traditional uniform fine-tuning.

## Limitations & Future Work
- Current experiments are mainly validated on the LLaMA series; its applicability to other architectures (such as Mixtral MoE) remains to be verified.
- Recovery fine-tuning still demands certain computational resources and calibration data, which might not be suitable for extremely resource-constrained scenarios.
- Only activation magnitude is considered as the importance metric, while other signals such as gradient information have not been explored.
- Future work could consider extending the coarse-grained analysis to the attention head level to achieve more fine-grained hierarchical pruning.

## Related Work & Insights
- **vs LLM-Pruner (Ma et al., 2023)**: LLM-Pruner uses gradient information to evaluate importance, which incurs a larger computational overhead, whereas CFSP relies solely on activation information, making it more efficient.
- **vs SliceGPT (Ashkboos et al., 2024)**: SliceGPT performs pruning via matrix decomposition, which is distinct from but complementary to the idea of CFSP, prompting potential exploration into combining the two.
- **vs Wanda (Sun et al., 2024)**: Wanda is an unstructured pruning method that uses activation $\times$ weight as an importance metric. CFSP borrows a similar idea but extends it to the structured scenario.

## Rating
- Novelty: ⭐⭐⭐⭐ The coarse-to-fine two-level strategy is relatively novel in LLM pruning, though the core technical components are not entirely brand new.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across multiple models and sparsity ratios, with comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear description of the methodology with a complete chain of motivation.
- Value: ⭐⭐⭐⭐ Highly practical, offering a direct reference for industrial LLM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Two-Stage Regularization-Based Structured Pruning for LLMs](../../ACL2026/model_compression/two-stage_regularization-based_structured_pruning_for_llms.md)
- [\[ACL 2025\] STUN: Structured-Then-Unstructured Pruning for Scalable MoE Pruning](stun_moe_pruning.md)
- [\[ICLR 2026\] COMI: Coarse-to-fine Context Compression via Marginal Information Gain](../../ICLR2026/model_compression/comi_coarse-to-fine_context_compression_via_marginal_information_gain.md)
- [\[NeurIPS 2025\] DuoGPT: Training-free Dual Sparsity through Activation-aware Pruning in LLMs](../../NeurIPS2025/model_compression/duogpt_training-free_dual_sparsity_through_activation-aware_pruning_in_llms.md)
- [\[ACL 2025\] DoMIX: An Efficient Framework for Exploiting Domain Knowledge in Fine-Tuning](domix_an_efficient_framework_for_exploiting.md)

</div>

<!-- RELATED:END -->
