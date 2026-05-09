---
title: >-
  [Paper Note] Unveiling Super Experts in Mixture-of-Experts Large Language Models
description: >-
  [ICLR 2026][Model Compression][Mixture-of-Experts] This paper is the first to discover and systematically study "Super Experts" (SEs) in MoE LLMs—an extremely small subset of experts that are critical to model inference, driving massive activations and attention sink mechanisms through extreme activation outliers in their `down_proj` outputs.
tags:
  - ICLR 2026
  - Model Compression
  - Mixture-of-Experts
  - super experts
  - massive activations
  - attention sinks
  - expert pruning
date: 2026-05-08
content_hash: 191ed4b78fa790ea
---

# Unveiling Super Experts in Mixture-of-Experts Large Language Models

**Conference**: ICLR 2026
**arXiv**: [2507.23279](https://arxiv.org/abs/2507.23279)
**Code**: [GitHub](https://github.com/ZunhaiSu/Super-Experts-Profilling)
**Area**: Model Compression / MoE / LLM Analysis
**Keywords**: Mixture-of-Experts, super experts, massive activations, attention sinks, expert pruning, model compression

## TL;DR

This paper is the first to discover and systematically study "Super Experts" (SEs) in MoE LLMs—an extremely small subset of experts that are critical to model inference, driving massive activations and attention sink mechanisms through extreme activation outliers in their `down_proj` outputs.

## Background & Motivation

MoE LLMs (e.g., DeepSeek, Qwen3, Mixtral) achieve strong learning capacity through dynamic routing and sparse activation. Existing expert-level compression methods exploit inter-expert importance differences for pruning, merging, or quantization, but mostly rely on heuristic metrics to identify critical experts, lacking a deeper understanding of expert heterogeneity in importance.

Core problem: **Does a small subset of extremely critical experts exist? What are their underlying mechanisms?**

## Method

### Overall Architecture

The paper analyzes super experts through three progressively deeper levels:

1. **Discovery and Localization**: SEs induce massive activations via extreme activation outliers in `down_proj` outputs.
2. **Importance Assessment**: The impact of SEs on various tasks is quantified through dynamic pruning.
3. **Mechanism Revelation**: SEs are the primary source of systematic outlier mechanisms in Transformers; compressing SEs causes attention sinks to collapse.

### Key Design 1: Discovery and Localization of Super Experts

MoE LLMs exhibit **massive activations (MAs)**—extreme outliers in hidden states whose values are up to 100,000× larger than other activations.

Analysis reveals that **a small number of specific experts** consistently produce extreme activation outliers in their `down_proj` outputs, which propagate through residual connections into the hidden states of all subsequent layers.

**SE Profiling Definition**: The maximum `down_proj` output magnitude $a_{l,e}$ is computed for all experts across all layers:

$$a_{l,e} > P_{99.5} \quad \text{and} \quad a_{l,e} > \frac{1}{10} a_{\max} \quad \text{and} \quad l \in L$$

where $P_{99.5} = \text{Percentile}_{99.5}(\mathcal{A})$ and $L$ denotes the set of layers producing MAs.

### Key Design 2: Distributional Properties of SEs

| Model | Total Experts | # SEs | SE Ratio | Top-1 Max Activation |
|-------|--------------|-------|----------|----------------------|
| Qwen3-30B-A3B | 6144 | 3 | 0.05% | 744.0 |
| DeepSeek-R1 | 15677 | 10 | 0.06% | 616.0 |
| DeepSeek-V2-Lite | 1782 | 2 | 0.11% | 1424.0 |
| Mixtral-8x7B | 256 | 1 | 0.39% | 5600.0 |

Key findings:
- SEs are universally present in all tested MoE LLMs, accounting for < 0.5% of all experts.
- SE distributions are **model-specific** and **data-agnostic**.
- Post-training procedures (e.g., RLHF) do not alter SE distributions.

### Key Design 3: Importance Assessment of SEs

Performance degradation is evaluated across multiple tasks by dynamically pruning SEs:

| Model | Setting | Avg. | GSM8K | MMLU | HellaSwag |
|-------|---------|------|-------|------|-----------|
| Qwen3-30B-A3B | Baseline | 70.22 | 89.61 | 77.82 | 59.63 |
| Qwen3-30B-A3B | Prune SEs | 55.00 | 42.38 | 56.03 | 39.31 |
| Qwen3-30B-A3B | Prune same # randomly | 70.36 | 89.84 | 77.84 | 59.50 |

Pruning only 3 SEs (0.05% of 6,144) results in:
- Average performance drop of 21.68%.
- **Mathematical reasoning (GSM8K) drop of 52.71%.**
- For reasoning LLMs, Pass@1 on AIME and Math-500 drops to near zero.

### Key Design 4: Relationship with Attention Sinks

SEs are central to the systematic outlier mechanism in Transformers:

1. SEs produce extremely strong activations on attention sink tokens.
2. These activations form massive activations through residual connections.
3. MAs drive the formation of attention sinks.
4. Compressing SEs → MAs vanish → attention sinks collapse → attention score distributions become disordered.

This reveals a complete causal chain: **SE → MA → Attention Sinks → Model Functionality**

## Key Experimental Results

### Main Results: Non-Reasoning Models

| Metric | Qwen3-30B Baseline | Prune SEs | Drop | Random Prune | Drop |
|--------|--------------------|-----------|------|--------------|------|
| Avg. | 70.22 | 55.00 | −21.68% | 70.36 | −0.20% |
| GSM8K | 89.61 | 42.38 | −52.71% | 89.84 | +0.26% |
| MMLU | 77.82 | 56.03 | −28.00% | 77.84 | +0.03% |

### Reasoning Model Experiments

Pruning 10 SEs from DeepSeek-R1:
- Pass@1 on AIME and Math-500 drops to near zero.
- Mathematical reasoning capability collapses entirely.

### Ablation Study

- Layer-wise SE pruning: pruning a single layer's SE eliminates that layer's contribution to MAs.
- Full SE removal: MAs disappear completely.

### Cross-Dataset Stability

SE distributions are highly consistent across C4, WikiText-2, C-Eval, GSM8K, and HumanEval, confirming data-agnosticity.

## Highlights & Insights

- First systematic discovery and definition of super experts in MoE LLMs.
- Reveals the complete causal chain: SE → MA → Attention Sinks → Model Functionality.
- Provides an automated SE profiling tool for rapid localization of SEs.
- Offers critical guidance for MoE compression: SEs must be treated with special care.

## Limitations & Future Work

- The root cause of why SEs emerge during pre-training remains unclear.
- Analysis is limited to open-source MoE models; the situation for closed-source models (e.g., GPT-4) is unknown.
- Protective strategies for SEs (e.g., allocating higher bit-width budgets) are only preliminarily discussed.
- Whether a more balanced MoE training mechanism without SEs can be designed is not deeply explored.

## Related Work & Insights

- **MoE Models**: DeepSeek (Guo et al., 2025), Qwen (Yang et al., 2025), Mixtral.
- **Expert-Level Compression**: Expert importance measures based on activation frequency, routing scores, and reconstruction loss.
- **Massive Activations**: Discovered by Sun et al. (2024), but their origin in MoE models was not explained.
- **Attention Sinks**: Discovered by Xiao et al. (2023), showing that initial tokens receive disproportionately high attention.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First discovery and systematic study of super experts in MoE models.
- **Theoretical Depth**: ⭐⭐⭐⭐ — Causal analysis is thorough but lacks formal theoretical explanation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive validation across multiple models, tasks, and datasets.
- **Value**: ⭐⭐⭐⭐⭐ — Directly guides MoE compression strategies.
- **Writing Quality**: ⭐⭐⭐⭐ — Progressive analytical structure is clear and well-organized.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] LD-MoLE: Learnable Dynamic Routing for Mixture of LoRA Experts](ld-mole_learnable_dynamic_routing_for_mixture_of_lora_experts.md)
- [\[CVPR 2026\] Enhancing Mixture-of-Experts Specialization via Cluster-Aware Upcycling](../../CVPR2026/model_compression/enhancing_mixture_of_experts_specialization_via_cluster_aware_upcycling.md)
- [\[NeurIPS 2025\] Dense Backpropagation Improves Training for Sparse Mixture-of-Experts](../../NeurIPS2025/model_compression/dense_backpropagation_improves_training_for_sparse_mixture-of-experts.md)
- [\[NeurIPS 2025\] Toward Efficient Inference Attacks: Shadow Model Sharing via Mixture-of-Experts](../../NeurIPS2025/model_compression/toward_efficient_inference_attacks_shadow_model_sharing_via_mixture-of-experts.md)
- [\[ICLR 2026\] Knowledge Fusion of Large Language Models Via Modular Skillpacks](knowledge_fusion_of_large_language_models_via_modular_skillpacks.md)

<!-- RELATED:END -->
