---
title: >-
  [Paper Note] Residual Decoding: Mitigating Hallucinations in Large Vision-Language Models via History-Aware Residual Guidance
description: >-
  [CVPR2026][Multimodal VLM][Hallucination Mitigation] This paper proposes Residual Decoding (ResDec), a training-free plug-and-play decoding strategy that identifies the semantic anchoring phase by analyzing U-shaped JSD…
tags:
  - "CVPR2026"
  - "Multimodal VLM"
  - "Hallucination Mitigation"
  - "Decoding Strategy"
  - "Vision-Language Models"
  - "Residual Guidance"
  - "Training-Free"
date: 2026-05-08
content_hash: 42a4f69e264fb23c
---

# Residual Decoding: Mitigating Hallucinations in Large Vision-Language Models via History-Aware Residual Guidance

**Conference**: CVPR2026
**arXiv**: [2602.01047](https://arxiv.org/abs/2602.01047)
**Code**: None
**Area**: Multimodal VLM
**Keywords**: Hallucination Mitigation, Decoding Strategy, Vision-Language Models, Residual Guidance, Training-Free

## TL;DR
This paper proposes Residual Decoding (ResDec), a training-free plug-and-play decoding strategy that identifies the semantic anchoring phase by analyzing U-shaped JSD patterns in historical token logit distributions, aggregates logits from this phase as residual guidance to steer current decoding, and effectively suppresses language-prior hallucinations in LVLMs at near-zero additional inference overhead.

## Background & Motivation
Large Vision-Language Models (LVLMs) achieve strong performance on multimodal tasks but suffer severely from **language-prior hallucinations** — during autoregressive generation, the text context progressively overshadows the visual context, causing the model to produce linguistically fluent yet visually unfaithful content.

Limitations of existing mitigation approaches: (1) Training-based methods (data debiasing, preference alignment, etc.) require additional training and annotation, limiting scalability; (2) Contrastive decoding methods (VCD, ICD, etc.) incur 2× or greater inference latency and GPU memory overhead; (3) Model-internal intervention methods (modifying attention, FFN, or layer representations) are inefficient and exhibit poor generalizability.

**Key Observation**: The authors find that signals for correct answers are already embedded in the logit distributions of preceding tokens. For example, when generating "The answer is D," the logit for the correct token "D" is already high during the generation of the preamble tokens "The," "answer," and "is"; however, when generating ":", the logit for the hallucinated token "C" spikes abnormally and eventually surpasses the correct token. **The essence of hallucination is that hallucinated token logits rise anomalously at certain steps and gradually overtake those of the correct tokens.**

## Method

### Overall Architecture
ResDec operates purely at the decoding stage, requiring no modifications to model architecture or additional training. Given the current decoding step $t$, ResDec analyzes the logit evolution of tokens within a historical window, identifies a semantically stable interval, aggregates the logits from that interval into a residual guidance signal, and performs sampling after weighted fusion with current logits.

### Key Designs

1. **U-Shaped JSD Pattern and Three-Phase Segmentation**: By computing the Jensen-Shannon divergence between adjacent timesteps within the historical window $\mathcal{W}$, the JSD of candidate token distributions exhibits a U-shaped trajectory, enabling segmentation into three phases:

    - **PSAP (Pre-Semantic Anchoring Phase)**: The left arm of the U-shape, where the distribution transitions from "chaotic" to "convergent," with residual anchoring uncertainty.
    - **SAP (Semantic Anchoring Phase)**: The bottom of the U-shape, where JSD approaches 0, indicating a highly stable distribution in which the model has firmly anchored core semantics.
    - **EDP (Expression Divergence Phase)**: The right arm of the U-shape, where JSD rises as the model explores diverse expression forms, making it susceptible to language-prior influence.

    ResDec selects logits from the SAP+EDP interval (i.e., the bottom and right arm of the U-shape) to construct the residual guidance.

2. **Confidence-Weighted Historical Aggregation**: Within the historical aggregation window $\Delta_t$, a local confidence score is computed for each timestep $i$ as $C_i = -\frac{1}{|\Omega_t|} \sum_{j=1}^{|\Omega_t|} \log P_i(j)$ (low entropy → high confidence). Logits across steps are then aggregated using normalized confidence weights:
    $\text{logit}_\theta^{\text{res}}(y_t | T_{<t-1}) = \sum_{i \in \Delta_t} \frac{C_i}{\sum_j C_j} \cdot \text{logit}_\theta(\hat{y}_i | T_{<i})$

3. **History-Current Fusion with Feasibility Constraint**: The historical residual is linearly fused with the current logits:
    $p_{\text{ResDec}}(y_t) = \text{Softmax}[(1-\alpha)\text{logit}_\theta(y_t) + \alpha \cdot \text{logit}_\theta^{\text{res}}(y_t)]$
    where $\alpha=0.5$. A truncation constraint $\mathcal{V}_{\text{head}}$ ($\beta=0.1$) is further applied, retaining only tokens whose probability is no less than $\beta$ times the maximum probability and setting all other token logits to $-\infty$, thereby preventing the residual guidance from introducing implausible tokens.

### Loss & Training
- **Fully training-free**, operating exclusively at the decoding stage.
- Reuses historical logits naturally produced during inference, requiring no additional forward passes.
- Simple hyperparameter configuration: $\alpha=0.5$, $\beta=0.1$, candidate token pool size $|\Omega_t| \in [64, 512]$.

## Key Experimental Results

### Main Results (POPE Average)

| Model | Method | Accuracy ↑ | F1 ↑ |
|------|------|-----------|------|
| LLaVA-1.5 | Regular | 79.83 | 79.29 |
| LLaVA-1.5 | OPERA | 84.21 | 83.55 |
| LLaVA-1.5 | VISTA | 86.15 | 86.29 |
| LLaVA-1.5 | **ResDec** | **87.23** | **86.93** |
| Qwen2.5-VL | Regular | 86.11 | 84.74 |
| Qwen2.5-VL | VISTA | 88.83 | 88.99 |
| Qwen2.5-VL | **ResDec** | **90.16** | **89.56** |

### HallusionBench & CHAIR

| Model | Method | fACC ↑ | CHAIR_S ↓ | CHAIR_I ↓ |
|------|------|--------|-----------|-----------|
| LLaVA-1.5 | Regular | 17.9 | 55.0 | 16.3 |
| LLaVA-1.5 | MemVR | 17.9 | 46.6 | 13.0 |
| LLaVA-1.5 | **ResDec** | **18.2** | **42.7** | **12.6** |
| Qwen2.5-VL | Regular | 43.4 | 30.6 | 8.4 |
| Qwen2.5-VL | **ResDec** | **47.1** | **25.8** | **6.8** |

### Efficiency Comparison

| Method | Latency (ms/token) | Throughput (token/s) | Memory (MB) |
|------|---------------|----------------|----------|
| Greedy | 28.54 | 35.04 | 14257 |
| VCD | 62.79 | 15.93 | 14967 |
| OPERA | 104.46 | 9.57 | 21300 |
| **ResDec** | **29.11** | **34.35** | **14296** |

### Ablation Study

| $\alpha$ | $\beta$ | MME | POPE Acc | MMStar |
|---------|---------|-----|----------|--------|
| 0.25 | 0.1 | 2326 | 89.64 | 64.20 |
| **0.5** | **0.1** | **2348** | **90.16** | **65.40** |
| 0.75 | 0.1 | 1875 | 82.56 | 62.67 |
| 1.0 | 0.1 | 1583 | 72.50 | 61.80 |

### Key Findings
- ResDec achieves average improvements of 7.84% in Accuracy and 8.01% in F1 across three LVLMs (vs. Regular).
- Latency increases by only 0.02× over Greedy, far outperforming OPERA (3.7×) and VCD (2.2×).
- Performance degrades sharply when $\alpha$ exceeds 0.5 — the historical residual serves as **auxiliary correction**, not a replacement for decoding.
- The candidate pool size is optimal in the range of 64–512; too small fails to capture JSD variation, while too large introduces early-step noise.
- Effective across multiple decoding strategies (Nucleus, Top-K, Temperature, Greedy).

## Highlights & Insights
- **Deep Mechanistic Insight**: The paper identifies the underlying mechanism of hallucination — hallucinated token logits gradually rise and surpass correct tokens during decoding — providing a new perspective for understanding and mitigating hallucinations.
- **U-Shaped JSD Pattern**: Elegantly reveals the three-stage evolution of "semantic convergence → anchoring → divergence" during LVLM decoding.
- **Minimal Overhead**: Reuses existing historical logits without additional forward passes; latency increases by only 2%, making it the most efficient hallucination mitigation method to date.
- **Plug-and-Play**: Requires no modification to model architecture, no training, and is compatible with diverse decoding strategies and LVLM architectures.
- Theoretical justification for ResDec is provided from a Bayesian perspective of PMI and language priors.

## Limitations & Future Work
- $\alpha=0.5$ is a globally fixed value; different tasks or models may benefit from different optimal values, making adaptive $\alpha$ scheduling a promising direction for improvement.
- The applicability of the U-shaped JSD pattern in very short response scenarios (e.g., Yes/No) requires further analysis.
- Validation is limited to 7B-scale models; performance on larger-scale models remains unknown.
- The candidate token pool size requires manual tuning; automated selection mechanisms warrant exploration.

## Related Work & Insights
- **vs. VCD (Contrastive Decoding)**: VCD requires an additional image-free forward pass, doubling latency; ResDec reuses existing information with virtually no extra overhead.
- **vs. OPERA (Attention Penalty)**: OPERA incurs 3.7× latency and 7 GB additional memory; ResDec adds only 39 MB and achieves superior performance.
- **vs. DoLa (Layer Contrastive)**: DoLa modifies internal model structures, limiting generalizability; ResDec operates as a pure decoding strategy and is more universally applicable.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The discovery of the U-shaped JSD pattern and the design of history-aware residual guidance decoding are highly original, addressing hallucinations from a fundamental mechanistic standpoint.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on 11 benchmarks, 3 LVLMs, 8+ baselines, with multi-dimensional ablations and efficiency analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, well-articulated insights, and intuitive U-shaped JSD visualizations.
- Value: ⭐⭐⭐⭐⭐ Extremely high practical value — training-free, zero additional overhead, plug-and-play; a strong candidate to become a standard component in LVLM decoding pipelines.


## Highlights & Insights


## Limitations & Future Work


## Related Work & Insights


## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] KeepLoRA: Continual Learning with Residual Gradient Adaptation](../../ICLR2026/multimodal_vlm/keeplora_continual_learning_with_residual_gradient_adaptation.md)
- [\[ACL 2026\] Mitigating Hallucinations in Large Vision-Language Models without Performance Degradation](../../ACL2026/multimodal_vlm/mitigating_hallucinations_in_large_vision-language_models_without_performance_de.md)
- [\[CVPR 2026\] Tell Model Where to Look: Mitigating Hallucinations in MLLMs by Vision-Guided Attention](tell_model_where_to_look_mitigating_hallucinations_in_mllms_by_vision-guided_att.md)
- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)
- [\[ICLR 2026\] Self-Aug: Query and Entropy Adaptive Decoding for Large Vision-Language Models](../../ICLR2026/multimodal_vlm/self-aug_query_and_entropy_adaptive_decoding_for_large_vision-language_models.md)

</div>

<!-- RELATED:END -->
