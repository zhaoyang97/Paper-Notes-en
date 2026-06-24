---
title: >-
  [Paper Note] Can LLM Watermarks Robustly Prevent Unauthorized Knowledge Distillation?
description: >-
  [ACL 2025][LLM Safety][LLM Watermarking] This paper presents the first systematic study on the robustness of LLM watermarks in preventing unauthorized knowledge distillation. It proposes three watermark removal attacks (Untargeted/Targeted Paraphrasing and Inference-Time Watermark Neutralization). The study reveals that Targeted Paraphrasing and Watermark Neutralization can thoroughly remove inherited watermarks, with Watermark Neutralization achieving zero extra training ove…
tags:
  - "ACL 2025"
  - "LLM Safety"
  - "LLM Watermarking"
  - "Knowledge Distillation"
  - "Watermark Radioactivity"
  - "Watermark Removal"
  - "Adversarial Attacks"
date: 2026-05-08
content_hash: e892d4cda978e9fc
---

# Can LLM Watermarks Robustly Prevent Unauthorized Knowledge Distillation?

**Conference**: ACL 2025  
**arXiv**: [2502.11598](https://arxiv.org/abs/2502.11598)  
**Code**: None  
**Area**: AI Safety  
**Keywords**: LLM Watermarking, Knowledge Distillation, Watermark Radioactivity, Watermark Removal, Adversarial Attacks  

## TL;DR
This paper presents the first systematic study on the robustness of LLM watermarks in preventing unauthorized knowledge distillation. It proposes three watermark removal attacks (Untargeted/Targeted Paraphrasing and Inference-Time Watermark Neutralization). The study reveals that Targeted Paraphrasing and Watermark Neutralization can thoroughly remove inherited watermarks, with Watermark Neutralization achieving zero extra training overhead while maintaining knowledge transfer efficiency.

## Background & Motivation

**Background**: Major LLM providers like OpenAI and Anthropic prohibit using their outputs to train competing models in their terms of use. Watermarking techniques (e.g., KGW, Google's SynthID-Text) are considered promising solutions to monitor unauthorized knowledge distillation: watermarks exhibit "radioactivity"—student models trained on watermarked teacher model outputs inherit detectable watermark patterns.

**Limitations of Prior Work**: Although watermark radioactivity has been validated (with p-values as low as $10^{-30}$), the key security question of whether adversarial attackers can remove inherited watermarks while preserving knowledge transfer remains systematically unstudied.

**Key Challenge**: Watermarks must be designed robustly enough to be tracked even after distillation, yet attackers can exploit paraphrase models or inference-time interventions to disrupt watermark statistical signatures.

**Goal**: Systematically evaluate the robustness of LLM watermarks as IP protection mechanisms under adversarial attacks, and propose more effective attack methods.

**Key Insight**: Analyze the key factors affecting watermark radioactivity (prefix frequency and window size), design watermark stealing techniques accordingly, and then execute targeted watermark removal using the stolen rules.

**Core Idea**: Extract watermark rules by comparing token probability distributions of the student model before and after distillation, and then completely eliminate the watermark using inverse watermark operations during training data paraphrasing or inference decoding.

## Method

### Overall Architecture

Assuming a closed-source teacher model employs a watermarking scheme, and the attacker (the student model owner) obtains the teacher model's API outputs as training data. Attacks fall into two categories: (1) pre-distillation removal—using untargeted paraphrasing (UP) or targeted paraphrasing (TP) on training data; (2) post-distillation removal—inference-time watermark neutralization (WN). Both TP and WN rely on first stealing the watermark rules.

### Key Designs

1. **Watermark Radioactivity Factor Analysis**:

    - Function: Identify key factors influencing watermark inheritance strength.
    - Key Findings: (a) **Prefix Frequency**: Prefixes appearing with higher frequency in the training data lead to stronger inheritance of the corresponding watermark rules in the student model; rare prefixes (frequency $\leq 5\times10^{-5}$) show radioactivity close to the unwatermarked baseline. (b) **Window Size n**: Radioactivity drops sharply as n increases, and when n=4, no watermark can be detected even with 1 million tokens.
    - Design Motivation: Restrict the scope of watermark stealing—only focusing on n≤3 and high-frequency prefixes dramatically reduces computational costs.

2. **Watermark Stealing**:

    - Function: Extract watermark rules without knowing the watermarking scheme and window size.
    - Mechanism: Compare the token probability distributions of the original student model $\mathcal{O}$ and the distilled student model $\mathcal{W}$ under the same context. Tokens with a probability ratio $\bar{P_{\mathcal{W}}}(x_t|p) / \bar{P_{\mathcal{O}}}(x_t|p)$ greater than 1 are identified as "watermark tokens". Results across multiple window sizes are aggregated and weighted by their prefix frequencies.
    - Design Motivation: Unlike existing methods, this approach does not require prior knowledge of the watermarking scheme or window size, and is more accurate due to weighting based on radioactivity factor analysis.

3. **Untargeted Paraphrasing (UP)**:

    - Function: Rewrite training data directly using a paraphrasing model.
    - Mechanism: Rewrite training data using paraphrasing models like Dipper, without considering watermark rules.
    - Effect: Partially effective but incomplete; watermarks can still be detected under certain configurations.

4. **Targeted Paraphrasing (TP)**:

    - Function: Apply inverse watermarking during the decoding phase of the paraphrasing model.
    - Mechanism: $l'_\mathcal{R}(x_t|x_{1:t-1}) = l_\mathcal{R}(x_t|x_{1:t-1}) - D(x_t; x_{t-n'+1:t-1}) \cdot \delta'$, where D is the stolen watermark confidence, and $\delta'$ controls the intensity of the inverse watermark, lowering the probability of tokens identified as watermark tokens.
    - Effect: Completely removes watermarks, though the paraphrasing process might cause minor knowledge loss.

5. **Watermark Neutralization (WN)**:

    - Function: Neutralize inherited watermarks directly during the inference phase of the distilled student model.
    - Mechanism: $l'_\mathcal{W}(x_t|x_{1:t-1}) = l_\mathcal{W}(x_t|x_{1:t-1}) - D(x_t; x_{t-n'+1:t-1}) \cdot \delta'$, which directly adjusts logits during inference.
    - Design Motivation: Requires no retraining, does not impair distilled knowledge, and incurs minimal computational overhead.

### Loss & Training
UP and TP require pre-processing training data and then retraining the student model. WN operates entirely during inference and involves no training. Distillation training of the student model utilizes standard cross-entropy loss.

## Key Experimental Results

### Main Results

Teacher: GLM-4-9b-chat, Student: Llama-7b / Llama-3.2-1b. Watermark scheme: KGW (n=1,2,3) + SynthID-Text (n=1,2,3).

| Method | KGW n=1 (p-value) | KGW n=2 (p-value) | SynthID n=1 (p-value) | SynthID n=2 (p-value) |
|------|--------------|--------------|-------------------|-------------------|
| No Watermark Baseline | 5.75e-01 | 5.75e-01 | 5.75e-01 | 5.75e-01 |
| No Attack (Direct Distillation) | **6.24e-25979** | **4.79e-2537** | **6.20e-4028** | **6.08e-887** |
| UP (Untargeted Paraphrasing) | 1.17e-389 | 4.21e-38 | 2.38e-76 | 5.61e-01 |
| TP (Targeted Paraphrasing) | **3.92e-01** | **7.12e-01** | **5.23e-01** | **6.84e-01** |
| WN (Watermark Neutralization) | **4.87e-01** | **6.29e-01** | **5.51e-01** | **6.71e-01** |

(A p-value > 0.01 indicates that the watermark is undetectable, and a p-value close to 0.5 indicates no difference from the unwatermarked baseline)

Knowledge Retention Evaluation (ARC Challenge Acc / TruthfulQA Acc):

| Method | ARC-C Acc ↑ | TruthfulQA Acc ↑ | MTBench ↑ |
|------|------------|-----------------|-----------|
| No Attack Baseline | 41.4 | 37.8 | 4.55 |
| UP | 38.9 | 36.2 | 4.12 |
| TP | 39.7 | 36.8 | 4.28 |
| **WN** | **41.2** | **37.6** | **4.51** |

### Ablation Study

Watermark stealing accuracy (KGW n=1, F1 score between the stolen green list and the ground-truth green list):

| Configuration | F1 Score |
|------|---------|
| Ours (Frequency-Weighted) | **0.89** |
| Without Frequency Weighting | 0.76 |
| Existing Method (Requires schematic knowledge) | 0.92 |

### Key Findings
- **Both TP and WN completely remove watermarks**: The p-value recovers to a level consistent with no watermark (~0.5), whereas UP is only partially effective.
- **WN is significantly superior to TP/UP in knowledge retention**: ARC-C accuracy drops by only 0.2% (compared to a 1.7% drop for TP and a 2.5% drop for UP), since WN neither distorts the training data nor requires retraining.
- **Watermark collision exists in multi-source distillation**: When a student model is distilled from multiple teacher models employing different watermarking schemes, the watermark signals interfere with each other, rendering all of them undetectable. This constitutes an additional vulnerability in watermark protection mechanisms.
- **Watermarks naturally fail to survive distillation when the window size n ≥ 4**: Without any active attack, watermarks with n=4 inherently lack radioactivity.

## Highlights & Insights
- **Watermark stealing does not require prior knowledge of the watermarking scheme**: Extracting watermark rules merely by comparing probability distributions before and after distillation provides a more general methodology than prior work (e.g., Jovanović et al.), posing a more severe threat to watermark security.
- **Insight on frequency-weighting**: Only watermark rules corresponding to high-frequency prefixes are realistically inherited by the student model. This finding simultaneously guides attacks (by focusing stealing efforts on high-frequency rules) and defense designs (which must ensure rules across all frequencies exhibit radioactivity).
- **The elegance of WN**: It completely avoids touching the training pipeline, requiring only the addition of a logit adjustment term during inference to eliminate watermarks. This represents a devastating blow to watermark-based protection schemes.

## Limitations & Future Work
- The study only evaluates n-gram-based watermarking schemes and does not cover non-n-gram watermarks (such as semantic-based watermarks).
- Watermark stealing requires parallel access to both the original and distilled student models, which might be unavailable in certain scenarios.
- Written from an attacker's perspective, this work does not propose effective defenses (only briefly discussing them in the discussion section).
- The scale of the experimental models is relatively small (7B/1B parameters), and the efficacy on larger scale models remains to be verified.

## Related Work & Insights
- **vs Sander et al. (2024) on watermark radioactivity**: While they proved the existence of radioactivity, this paper demonstrates that such radioactivity can be thoroughly bypassed.
- **vs Jovanović et al. on watermark stealing**: Prior methods required knowledge of the watermarking scheme and window size; the proposed method removes these assumptions.
- **vs Google SynthID-Text**: The production-level watermarking scheme already deployed in Gemini can also be bypassed using WN.

## Rating
- Novelty: ⭐⭐⭐⭐ The first systematic study on the robustness of watermark radioactivity; both the watermark stealing and WN methods are novel and practical.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation covering 2 model pairs × 2 watermarking schemes × 3 window sizes × multiple benchmarks.
- Writing Quality: ⭐⭐⭐⭐⭐ Clearly defined attack models and a systematic, deep analysis of radioactivity factors.
- Value: ⭐⭐⭐⭐⭐ Serves as a wake-up call to the LLM watermark protection community with high practical impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Distillation Robustifies Unlearning](../../NeurIPS2025/llm_safety/distillation_robustifies_unlearning.md)
- [\[ACL 2026\] Train in Vain: Functionality-Preserving Poisoning to Prevent Unauthorized Use of Code Datasets](../../ACL2026/llm_safety/train_in_vain_functionality-preserving_poisoning_to_prevent_unauthorized_use_of_.md)
- [\[ACL 2025\] Ensemble Watermarks for Large Language Models](ensemble_watermarks_llm.md)
- [\[NeurIPS 2025\] SAEMark: Steering Personalized Multilingual LLM Watermarks with Sparse Autoencoders](../../NeurIPS2025/llm_safety/saemark_steering_personalized_multilingual_llm_watermarks_with_sparse_autoencode.md)
- [\[ICLR 2026\] LLM Fingerprinting via Semantically Conditioned Watermarks](../../ICLR2026/llm_safety/llm_fingerprinting_via_semantically_conditioned_watermarks.md)

</div>

<!-- RELATED:END -->
