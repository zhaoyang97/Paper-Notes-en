---
title: >-
  [Paper Note] TrimLLM: Progressive Layer Dropping for Domain-Specific LLMs
description: >-
  [ACL 2025][LLM (Other)][layer dropping] This paper proposes TrimLLM. Based on the layer-wise specialization phenomenon, it progressively drops layers that are unimportant to the target domain during the domain-specific fine-tuning process. It achieves a 2.1-5.7x inference speedup without precision loss at a 50-60% compression rate, while operating independently of specialized hardware.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "layer dropping"
  - "model compression"
  - "domain-specific LLM"
  - "layer-wise specialization"
  - "inference speedup"
date: 2026-05-08
content_hash: c7b6b2237ae56868
---

# TrimLLM: Progressive Layer Dropping for Domain-Specific LLMs

**Conference**: ACL 2025  
**arXiv**: [2412.11242](https://arxiv.org/abs/2412.11242)  
**Code**: Not provided  
**Area**: Multimodal VLM  
**Keywords**: layer dropping, model compression, domain-specific LLM, layer-wise specialization, inference speedup  

## TL;DR

This paper proposes TrimLLM. Based on the layer-wise specialization phenomenon, it progressively drops layers that are unimportant to the target domain during the domain-specific fine-tuning process. It achieves a 2.1-5.7x inference speedup without precision loss at a 50-60% compression rate, while operating independently of specialized hardware.

## Background & Motivation

**Problem Definition:** When deploying LLMs specialized for domain-specific scenarios such as medical, legal, and finance, latency and privacy constraints must be met simultaneously. However, existing model compression methods often fail to deliver theoretical speedups in actual deployments.

**Limitations of Prior Work:** Post-training quantization (PTQ) methods such as GPTQ and AWQ require hardware-specific support or efficient kernels to achieve inference speedups, and may even slow down on consumer-grade GPUs (e.g., LLM.int8() drops from 16.6 to 10.2 tokens/s on V100). Pruning methods (e.g., SparseGPT, LLM-Pruner) also struggle to achieve actual speedups due to the hardware support required for structural sparsity.

**Core Motivation:** The authors observe the phenomenon of layer-wise specialization—different layers in LLMs vary significantly in importance for different knowledge domains. When fine-tuning LLaMA-7B on medical/scientific commonsense tasks, 16/20 layers (out of 32 in total) can be removed respectively with almost no loss of accuracy. By reducing model depth rather than altering precision or introducing sparsity, hardware-independent, general-purpose inference speedups can be achieved.

## Method

### Overall Architecture

TrimLLM integrates layer dropping and domain-specific fine-tuning into a unified pipeline: after training each epoch, the importance score of each layer is calculated, the least important layer is removed, and training continues. This is executed iteratively until accuracy or efficiency thresholds are met.

Formal representation: $f(\mathbf{y}_0; \theta_0) \to \mathcal{G}_{\mathcal{U}_{\mathcal{X}_1}}(\mathbf{y}_0; \theta_1') \to \mathcal{G}_{\mathcal{U}_{\mathcal{X}_2}}(\mathbf{y}_0; \theta_2') \to \cdots$

### Key Designs

1. **Progressive Layer Dropping:** Removing multiple layers at once causes a drastic change in the output distribution, whereas a progressive strategy smoothens the transition of the model's output distribution. Removing only one layer at a time and retraining ensures that the retained, most important layers can adapt to structural changes.

2. **Dual-metric Target Selection Algorithm:**
    - **Sensitivity-based Score:** Layer-by-layer trial deletion is performed on a small calibration set, $s_{i,\text{scan}} = \frac{100 - a_i}{(1+\delta^2) + (1+\delta)a_i}$, where $a_i$ is the accuracy after deleting the $i$-th layer.
    - **Activation-based Score:** Uses the Frobenius norm to measure the rank of activations in each layer, $s_{i,\text{norm}} = \frac{100 \min\{\|\mathbf{x}_j\|_F\}}{\|\mathbf{x}_i\|_F}$. High-norm layers encode more general knowledge and can be prioritized for removal.

3. **Sparse Update Regularization:** Before training, the initial importance distribution of each layer is determined via a calibration scan. Only the parameters of the $r \times N$ ($r=1/4$) layers with the highest probability of being retained are updated, while the remaining layers are frozen. This avoids catastrophic forgetting while reducing training costs.

### Loss & Training

Standard domain-specific fine-tuning loss (cross-entropy), consistent with specific downstream tasks. The innovation of TrimLLM lies not in the loss design, but in the layer selection and removal strategies during the training process.

## Experiments

### Main Results: LLaMA-7B QA Benchmark Comparison

| Method | PIQA | SciQ | MedMCQA | LexGLUE | FinanceQA | Final Memory |
|------|------|------|---------|---------|-----------|---------|
| No Training | 77.4 | 89.7 | 22.4 | 32.1 | 33.6 | 100% |
| Full-FT| 82.4 | 95.6 | 54.6 | 42.9 | 45.1 | 100% |
| LLM-Pruner | 70.3 | 85.0 | 23.1 | 30.8 | 27.3 | 100% |
| SparseGPT (2:4) | 76.5 | 90.1 | 52.3 | 37.9 | 41.6 | 100% |
| AWQ-int4 | 80.9 | 93.0 | 50.7 | 41.0 | 42.1 | ≥25% |
| **TrimLLM (50%)** | **81.8** | **94.2** | **53.1** | **42.0** | **43.6** | **≥50%** |
| TrimLLM (40%) | 77.6 | 91.2 | 47.5 | 39.5 | 41.3 | ≥40% |

### Inference Throughput Comparison (LLaMA-7B, seq_len=512, batch=1)

| GPU | FP16 | SparseGPT | LLM.int8() | GPTQ-int4 | AWQ-int4 | TrimLLM |
|-----|------|-----------|------------|-----------|----------|---------|
| A100 | 42.3 | 58.9 | 29.6 | 46.5 | 115.3 | 103.1 |
| V100 | 16.6 | 14.5 | 10.2 | 6.1 | 11.0 | **34.9** |
| RTX 3090 | 13.4 | 13.0 | 7.5 | 6.9 | 7.9 | **26.8** |

### Key Findings

1. **Almost No Accuracy Loss at 50% Compression Rate:** TrimLLM (50%) is comparable to Full-FT across all domain benchmarks, significantly outperforming quantization and pruning methods at the same compression level.
2. **Most Significant Speedup on Consumer-grade GPUs:** Achieves 2.1x speedup on V100 (vs FP16) and 2.0x speedup on RTX 3090, whereas quantization methods actually slow down on these GPUs.
3. **AWQ is Faster on A100 but TrimLLM remains Competitive:** AWQ reaches 115.3 tokens/s on A100, benefiting from efficient INT4 kernels, but TrimLLM closely follows at 103.1 tokens/s with higher accuracy.
4. **Sparse Updates are Crucial:** Performance is optimal when $r=1/4$; full-parameter fine-tuning instead leads to performance degradation after layer dropping due to catastrophic forgetting.
5. **Orthogonal and Stackable with Quantization:** TrimLLM can be combined with quantization to achieve up to an 8x compression rate.

## Highlights & Insights

- Leverages the layer-wise specialization phenomenon to unify compression and domain adaptation into a single pipeline, which is conceptually elegant.
- Obtains actual inference speedups without relying on specialized hardware/kernel support, which is particularly valuable for resource-constrained scenarios.
- Provides a flexible continuum of model sizes (from 30% to 50%+), making it easy to adapt to different hardware.
- The dual-metric (sensitivity + activation norm) layer importance evaluation method has practical steering significance.

## Limitations & Future Work

- Experiments were only conducted on LLaMA-7B and 13B, without validating applicability to larger models (e.g., 70B+) or newer architectures (Mistral, Qwen, etc.).
- Progressive dropping increases training time (though mitigated by sparse updates); the total number of training epochs grows linearly with the number of dropped layers.
- Accuracy drops significantly when compressed below 30%, showing limitations in extreme compression scenarios.
- Only evaluated on multiple-choice QA tasks, without testing open-ended generation tasks.

## Related Work & Insights

- **Quantization:** GPTQ (Frantar et al., 2022), AWQ (Lin et al., 2023), LLM.int8() (Dettmers et al., 2022)
- **Pruning:** SparseGPT (Frantar & Alistarh, 2023), LLM-Pruner (Ma et al., 2023), Wanda (Sun et al., 2023)
- **Layer Dropping:** Sajjad et al. (2023) compress the base model before fine-tuning, Zhang & He (2020) apply layer dropping in the pre-training stage to accelerate training.
- **Knowledge Localization:** Meng et al. (2022) ROME/MEMIT find that intermediate layers are responsible for domain knowledge, Geva et al. (2020) show that MLP layers are responsible for task-specific memory retrieval.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Practicality | ⭐⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Overall Recommendation | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] EVE: A Domain-Specific LLM Framework for Earth Intelligence](../../ACL2026/llm_nlp/eve_a_domain-specific_llm_framework_for_earth_intelligence.md)
- [\[ACL 2025\] LESA: Learnable LLM Layer Scaling-Up](lesa_learnable_llm_layer_scaling-up.md)
- [\[ACL 2025\] Cheaper and Better Diffusion Language Model via Task-Specific Training](cheaper_and_better_diffusion_language_model_via_task-specific_training.md)
- [\[ACL 2025\] Hierarchical Retrieval with Evidence Curation for Open-Domain Financial QA](hierarchical_retrieval_with_evidence_curation_for_open-domain_financial_question.md)
- [\[ACL 2025\] Enhancing Open-Domain Task-Solving Capability of LLMs via Autonomous Tool Integration from GitHub](paper_2312_17294.md)

</div>

<!-- RELATED:END -->
