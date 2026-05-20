---
title: >-
  [Paper Note] Decoupling Vision and Language: Codebook Anchored Visual Adaptation
description: >-
  [CVPR2026][Medical Imaging][Discrete visual tokens] CRAFT is proposed to decouple the visual encoder from the language model via a discrete codebook…
tags:
  - "CVPR2026"
  - "Medical Imaging"
  - "Discrete visual tokens"
  - "codebook"
  - "visual encoder adaptation"
  - "domain transfer"
  - "token pruning"
  - "LVLM"
date: 2026-05-08
content_hash: 69822eca8fbdf3eb
---

# Decoupling Vision and Language: Codebook Anchored Visual Adaptation

**Conference**: CVPR2026
**arXiv**: [2602.19449](https://arxiv.org/abs/2602.19449)  
**Code**: To be confirmed  
**Area**: Medical Imaging / Vision-Language Models
**Keywords**: Discrete visual tokens, codebook, visual encoder adaptation, domain transfer, token pruning, LVLM

## TL;DR

CRAFT is proposed to decouple the visual encoder from the language model via a discrete codebook, enabling domain adaptation by fine-tuning only the visual encoder. The adapted encoder can be seamlessly reused across different LLM architectures, achieving an average improvement of 13.51% across 10 domain benchmarks.

## Background & Motivation

1. Visual encoders in large vision-language models (LVLMs) perform poorly on long-tail domains such as medical imaging and fine-grained classification; perception errors in the encoder cascade into the language model and lead to incorrect reasoning.
2. Existing adaptation methods typically modify the continuous feature interface between the encoder and LLM (e.g., projection layer tuning / LoRA), keeping the two components coupled — re-alignment is required whenever either the encoder or the LLM is replaced.
3. Jointly fine-tuning the visual encoder and the LLM is computationally expensive and prone to forgetting instruction-following capabilities; the problem is further exacerbated by scarce domain data.
4. Fine-tuning the encoder alone is insufficient: once the feature distribution shifts, the frozen LLM cannot correctly interpret the new visual embeddings.
5. Recent discrete LVLMs (VILA-U, Janus, etc.) have demonstrated that discrete visual tokens achieve performance comparable to or better than continuous counterparts, providing a natural "shared language" interface.
6. Core problem: **Can domain adaptation of an LVLM be achieved without modifying the original LLM?**

## Method

### Overall Architecture — CRAFT (Codebook RegulAted Fine-Tuning)

CRAFT operates on discrete LVLMs: the visual encoder $E_\theta$ produces continuous features, which are quantized into discrete token sequences via nearest-neighbor lookup in a **frozen shared codebook** $\mathcal{C}=\{c_k\}_{k=1}^K$, and then passed through a projector to a frozen LLM. Only the visual encoder parameters are updated during training.

### Three Training Loss Components

1. **Surrogate Alignment Loss ($\mathcal{L}_{\text{SAL}}$)**: A (potentially small) surrogate language model $\mathcal{M}$ performs autoregressive prediction over image-text joint sequences, with gradients back-propagated to the visual encoder to guide it toward selecting codebook tokens useful for the target domain task.
2. **Commitment Loss ($\mathcal{L}_{\text{commit}}$)**: Keeps encoder outputs close to their assigned codebook entries, preventing quantization distortion caused by feature drift — the codebook remains frozen throughout, and only the encoder side is constrained.
3. **Contrastive Loss ($\mathcal{L}_{\text{con}}$)**: Leverages image captions and label-augmented text, applying sigmoid contrastive learning to preserve pre-trained semantic structure.

Total loss: $\mathcal{L}_{\text{CRAFT}} = \lambda_{\text{con}}\mathcal{L}_{\text{con}} + \lambda_{\text{commit}}\mathcal{L}_{\text{commit}} + \mathcal{L}_{\text{SAL}}$

where $\lambda_{\text{con}}=0.1$ for VQA tasks and $\lambda_{\text{con}}=1.0$ for classification tasks; $\lambda_{\text{commit}}=0.1$.

The non-differentiability of quantization is handled via the **straight-through estimator**.

### Test-Time Token Pruning

- **Rarity-weighted assignment**: The global frequency $p_{\text{dom}}(k)$ of each codebook entry is computed over the training set, and a rarity weight $\rho_k = 1/p_{\text{dom}}(k)$ is defined; high-frequency background tokens are aggressively pruned while rare, information-rich tokens are retained.
- **Intra-entry selection**: Tokens with large quantization residuals (hard to quantize, information-dense) and spatially isolated tokens are prioritized to encourage spatial diversity.
- A one-dimensional search over $\gamma$ controls the keep ratio $M/N$; the default keep ratio in the paper is 0.8.
- Ablation of pruning components: random selection 62.10% → rarity weighting 63.55% → residual-based ranking 63.86% → spatial isolation 64.05%.
- Replacing domain-data frequency statistics with ImageNet-1K statistics reduces performance by only 0.04% (64.01%), demonstrating robustness of the pruning strategy to the reference corpus.

## Key Experimental Results

### Main Results (Table 1, 10 Benchmarks, Exact-Match Accuracy %)

| Method | Visual Tokens | PlantVillage | VQARAD | EuroSAT | Cars | Dogs | Mean (10) |
|---|---|---|---|---|---|---|---|
| Zero-shot VILA-U-7B | Discrete | 43.83 | 35.67 | 69.15 | 72.50 | 82.40 | 55.07 |
| Zero-shot VILA-7B | Continuous | 47.20 | 41.67 | 79.48 | 76.30 | 78.33 | 59.89 |
| Vision FT (continuous) | Continuous | 62.13 | 43.67 | 67.35 | 86.80 | 71.43 | 61.76 |
| **CRAFT (7B surr.)** | **Discrete** | **77.27** | **45.67** | **77.80** | **92.74** | **84.77** | **68.58** |

With VILA-U-7B as the surrogate model, CRAFT achieves the best overall performance: **mean 68.58%, +13.51% over zero-shot**, and +6.82% over the strongest continuous baseline.

### Reasoning Quality Preservation (Table 2, VQARAD Dataset)

| Method | Accuracy | Explanation Presence | Relevance | Faithfulness | Overall |
|---|---|---|---|---|---|
| VILA-LLM-LoRA | 44.65 | 6.34 | 0.26 | 0.25 | -0.98 |
| Projector FT | 44.89 | 4.01 | 0.28 | 0.22 | -0.61 |
| **CRAFT** | **47.34** | **75.98** | **2.95** | **1.99** | **3.21** |

Continuous fine-tuning methods severely degrade instruction-following and explanation capabilities (Presence as low as 4–6%), while CRAFT maintains a 76% explanation generation rate.

### Ablation Study (Table 5, VILA-U-7B Backbone)

| Setting | VQARAD | Dogs | PlantVillage | IconQA | Mean |
|---|---|---|---|---|---|
| w/o $\mathcal{L}_{\text{commit}}$ | 10.33 | 16.53 | 25.97 | 3.31 | 14.04 |
| w/o $\mathcal{L}_{\text{SAL}}$ | 37.87 | 83.66 | 75.03 | 15.49 | 53.01 |
| w/o $\mathcal{L}_{\text{con}}$ | 45.13 | 71.57 | 45.69 | 47.24 | 52.41 |
| **Full CRAFT** | **45.67** | **84.77** | **77.27** | **48.50** | **64.05** |

The commitment loss is the most critical component — removing it causes performance to collapse to 14%. The surrogate alignment loss contributes most to reasoning tasks, while the contrastive loss contributes most to classification tasks.

### Cross-LLM Transfer (Table 3)

An encoder trained with Qwen2-0.5B as the surrogate and directly paired with Qwen2.5-3B at inference achieves a mean increase from 46.74% to 59.98% (+13.24%); paired with Qwen2-1.5B: 49.06% → 63.25% (+14.19%). An encoder trained with VILA-U-7B also transfers effectively to Qwen2-1.5B (+14.56%), validating the modular feasibility at the codebook level.

### Efficiency (Table 4)

- Using Qwen2-0.5B as the surrogate model: GPU memory reduced to 10.7 GiB (−61.6%), training time 1.35 min (−73.5%).
- At inference with keep ratio = 0.8: FLOPs reduced by 16%, latency reduced by 7%.

## Highlights & Insights

- **True vision-language decoupling**: The adapted encoder can be plug-and-play deployed to any LLM sharing the same codebook (Table 3 validates across 5 different inference backbone architectures and scales), which is unachievable with continuous approaches.
- **Zero LLM forgetting**: The LLM is fully frozen throughout, requiring no additional instruction data for anti-forgetting; full explanation and reasoning capabilities are preserved — explanation presence rate 76% vs. 6% for LoRA on VQARAD.
- **Extremely lightweight training**: The surrogate model can be very small (0.5B); training only the visual encoder requires only a few minutes on 8× A100 GPUs, with memory as low as 10.7 GiB.
- **Test-time token pruning**: A training-free pruning scheme based on frequency rarity further improves efficiency and robustness; performance remains stable above a keep ratio of 0.6.
- **Novel demonstration of discrete token advantages**: This is the first systematic demonstration that discrete visual tokens support modular, transferable visual adaptation, opening new application scenarios for discrete LVLMs.

## Limitations & Future Work

- CRAFT relies on a pre-trained discrete codebook (16,384 entries from VILA-U); codebook quality and scale constitute the performance ceiling. Table 6 shows that reducing the codebook to 10% of its size drops the mean from 76.71% to 32.28%.
- When the surrogate model is substantially weaker than the inference backbone, certain fine-grained tasks (Flowers, Dogs) can regress — the 0.5B surrogate model reduces Flowers from 75.80% to 72.31%.
- The current codebook is assumed to be fixed; backward compatibility when extending or merging codebooks in the future remains an open question (noted by the authors in Future Work).
- Evaluation is limited to classification and VQA tasks; assessment on open-ended generation, object detection, image segmentation, and other task formats is lacking.
- The contrastive loss depends on an additional captioning model to generate captions, increasing data preparation complexity.
- Discrete quantization inherently incurs information loss, potentially rendering the approach unsuitable for tasks requiring pixel-level precision (e.g., segmentation, detection).

## Related Work & Insights

- **Projector FT / Vision FT**: These methods still operate in the continuous space; LLM re-alignment is required whenever the encoder changes. CRAFT naturally isolates the encoder via the discrete codebook, enabling zero-cost encoder transfer.
- **LLM LoRA**: Although accuracy can be improved, instruction-following capability is severely degraded (explanation presence rate as low as ~2%). CRAFT avoids this entirely since the LLM remains frozen throughout.
- **LDIFS (Mukhoti et al.)**: Uses $\ell_2$ regularization to prevent CLIP feature drift, but still operates in the continuous space. CRAFT's commitment loss achieves an analogous goal in the discrete space with greater stability.
- **Discrete LVLMs (VILA-U, Janus)**: CRAFT is the first to leverage discrete codebooks for domain adaptation rather than generation tasks, revealing the unique advantages of discretization in terms of modularity and transferability.
- **Multi-encoder approaches (InternVL, etc.)**: Improve general performance by stacking additional visual encoders. CRAFT enhances domain-specific performance by fine-tuning a single encoder, making it more lightweight without increasing inference parameters.

## Rating

- Novelty: ⭐⭐⭐⭐ — The idea of using a discrete codebook for vision-language decoupled adaptation is original; cross-LLM transfer of the visual encoder is highly appealing.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive evaluation across 10 benchmarks × 5 backbones, complete ablations, and convincing reasoning quality assessment.
- Writing Quality: ⭐⭐⭐⭐ — Problem formulation is clear; Figures 1/2 intuitively contrast continuous vs. discrete approaches; experimental organization is logical.
- Value: ⭐⭐⭐⭐ — Practically meaningful for LVLM adaptation in resource-constrained domains such as medical imaging; the decoupled design reduces deployment and maintenance costs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MedCLIPSeg: Probabilistic Vision-Language Adaptation for Data-Efficient and Generalizable Medical Image Segmentation](medclipseg_probabilistic_vision-language_adaptation_for_data-efficient_and_gener.md)
- [\[CVPR 2026\] From Adaptation to Generalization: Adaptive Visual Prompting for Medical Image Segmentation](apex_adaptive_visual_prompting.md)
- [\[CVPR 2026\] Tell2Adapt: A Unified Framework for Source Free Unsupervised Domain Adaptation via Vision Foundation Model](tell2adapt_a_unified_framework_for_source_free_unsupervised_domain_adaptation_vi.md)
- [\[CVPR 2026\] MedKCO: Medical Vision-Language Pretraining via Knowledge-Driven Cognitive Orchestration](medkco_medical_vision-language_pretraining_via_knowledge-driven_cognitive_orches.md)
- [\[CVPR 2026\] VisualAD: Language-Free Zero-Shot Anomaly Detection via Vision Transformer](visualad_language-free_zero-shot_anomaly_detection_via_vision_transformer.md)

</div>

<!-- RELATED:END -->
