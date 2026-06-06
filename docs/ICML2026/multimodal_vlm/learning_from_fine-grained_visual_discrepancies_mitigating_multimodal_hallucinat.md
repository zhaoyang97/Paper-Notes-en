---
title: >-
  [Paper Note] Learning from Fine-Grained Visual Discrepancies: Mitigating Multimodal Hallucinations via In-Context Visual Contrastive Optimization
description: >-
  [ICML 2026][Multimodal VLM][Multimodal hallucinations] The original image and contrastive negative image are concatenated into a shared multi-image context. Anchoring instructions are then used to specify which image to…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Multimodal hallucinations"
  - "preference optimization"
  - "visual contrast"
  - "DPO"
  - "hard negative samples"
date: 2026-05-08
content_hash: 76dfa57aa687b205
---

# Learning from Fine-Grained Visual Discrepancies: Mitigating Multimodal Hallucinations via In-Context Visual Contrastive Optimization

**Conference**: ICML 2026  
**arXiv**: [2605.31312](https://arxiv.org/abs/2605.31312)  
**Code**: https://github.com/OPPO-Mente-Lab/IC-VCO (Available)  
**Area**: Multimodal VLM  
**Keywords**: Multimodal hallucinations, preference optimization, visual contrast, DPO, hard negative samples

## TL;DR
The original image and contrastive negative image are concatenated into a shared multi-image context. Anchoring instructions are then used to specify which image to observe, allowing the partition functions of visual preference DPO to align automatically and achieve theoretically consistent contrastive goals. Combined with hard negative samples generated through fine-grained surgery-like editing, this approach significantly reduces multimodal hallucinations in VLMs.

## Background & Motivation

**Background**: Aligning VLMs using DPO is the current mainstream post-training method. However, standard DPO only compares $y$ and $y'$ on the text side, treating the image $m$ as a static condition; thus, it does not explicitly supervise whether the model is "really looking at the image." To inject visual signals into DPO, recent works (mDPO, V-DPO, S-VCO, SymMPO, etc.) introduced "visual preference pairs": fixing the text response $y$, swapping the positive image $m$ with a negative image $m'$ to form $r(m,x,y)\succ r(m',x,y)$, and following the standard DPO loss.

**Limitations of Prior Work**: The authors point out two fatal flaws in this mainstream route. First is **theoretical inconsistency**—DPO eliminates the intractable partition function $Z$ because positive and negative samples share the same condition. Once the image in the condition is replaced, $Z(m,x)$ and $Z(m',x)$ become normalization constants on two different distributions and cannot be canceled out, leaving the term $\beta\log\frac{Z(m,x)}{Z(m',x)}$ as an uncontrollable bias during training. Second is that **negative samples are too coarse**: most existing $m'$ come from text-to-image synthesis or retrieval, introducing obvious global style shifts. The model can easily minimize the DPO loss by capturing these low-level differences without learning fine-grained visual facts, leading to typical shortcut learning.

**Key Challenge**: To inject visual supervision into the DPO framework, the conditional distribution must change; but once it changes, the theoretical guarantee of DPO is broken. Meanwhile, the "ease of distinction" of negative samples induces shortcuts. These two ends constrain each other.

**Goal**: Split into two sub-problems: (a) Design a visual preference objective that eliminates the partition function and maintains DPO's theoretical consistency; (b) Generate hard negative samples that are visually almost indistinguishable to concentrate contrastive signals on true semantic differences.

**Key Insight**: The authors observe that as long as positive and negative samples share the **same image context**, the partition function will automatically be consistent. Thus, the original and contrastive images are placed into a multi-image sequence $M=[m,m']$. Anchoring prompts like "Please answer based on the first/second image" are used to designate the target image, transforming visual contrast from "changing conditions" to "textual preference under the same condition," resolving the theoretical flaw.

**Core Idea**: Use In-Context Visual Contrastive Optimization (IC-VCO) to run DPO with visual contrast in a shared multi-image context, then use Visual Contrast Distillation to distill the multi-image supervision back to the single-image inference branch, and finally use "surgical" image editing to create hard negative samples.

## Method

### Overall Architecture
The input for IC-VCO is a contrastive quadruple $(m, m', x, y, y')$: original image $m$, negative image $m'$ with subtle differences in target semantics, a shared prompt $x$, and corresponding correct response $y$ and contrastive response $y'$. During training, three pipelines operate simultaneously. The first is the **multi-image branch**: $m$ and $m'$ are concatenated into context $M$. Position anchoring instructions are appended to the prompt to get $\hat{x}$, and DPO is run such that $y\succ y'$. To eliminate position bias, the image order is randomized. The second is the **single-image branch**: standard $(m, x, y, y')$ single-image DPO is maintained to ensure capabilities during inference. The third is **VCDist distillation**: the preference probability $p_{\text{multi}}$ from the multi-image branch serves as a soft target to calibrate the single-image branch $p_{\text{single}}$, closing the training-inference context gap. The final loss includes a fine-grained token mask to focus on edited visual evidence.

### Key Designs

1.  **In-Context DPO with Shared Multi-image Context**:
    *   **Function**: Injects visual contrast signals without changing the mathematical form of DPO.
    *   **Mechanism**: Concatenates $m$ and $m'$ into a sequence $M=[m,m']$ as a unified visual condition. Anchoring prompts $\hat{x}$ specify the target image. This rewrites visual preference $r(m,x,y)\succ r(m',x,y)$ as same-condition textual preference $r(M,\hat{x},y)\succ r(M,\hat{x},y')$. The partition function $Z(M,\hat{x})$ cancels out perfectly, yielding $p_{\text{multi}}=\sigma\big(\beta\log\tfrac{\pi_\theta(y\mid M,\hat{x})}{\pi_{\text{ref}}(y\mid M,\hat{x})}-\beta\log\tfrac{\pi_\theta(y'\mid M,\hat{x})}{\pi_{\text{ref}}(y'\mid M,\hat{x})}\big)$.
    *   **Design Motivation**: In contrast to previous "image-swapping" approaches, the bias $\beta\log\frac{Z(m,x)}{Z(m',x)}$ drifts arbitrarily and distorts the decision boundary. Shared context zeroes this bias, placing visual preference DPO on the same theoretical foundation as the original DPO.

2.  **VCDist: Reliable Gated Multi-to-Single Image Distillation**:
    *   **Function**: Connects multi-image training with single-image inference to prevent the single-image branch from being led astray.
    *   **Mechanism**: Uses the multi-image preference distribution as a "teacher" with a dual-gate mechanism: a correctness gate $\mathbb{I}(p_{\text{multi}}>0.5)$ filters unreliable teachers, and a confidence gate $\mathbb{I}(p_{\text{single}}<\mathrm{sg}(p_{\text{multi}}))$ ensures learning only when the student is less certain than the teacher. Stop-gradient is added for stability: $\mathcal{L}_{\text{VCDist}}=-\mathbb{E}\big[\mathbb{I}(\cdot)\big(\mathrm{sg}(p_{\text{multi}})\log p_{\text{single}}+(1-\mathrm{sg}(p_{\text{multi}}))\log(1-p_{\text{single}})\big)\big]$.
    *   **Design Motivation**: Standard VLM inference uses single images. Training only the multi-image branch causes a context gap. Naive KL may degrade high-quality single-image distributions. The gates ensure gradients are passed only when the teacher is reliable and the student needs them.

3.  **Surgical Contrastive Sample Editing**:
    *   **Function**: Replaces old methods of whole-image synthesis/retrieval to construct distribution-aligned hard negative samples.
    *   **Mechanism**: Decomposes image generation into target concept $c_{tgt}$, context $C_{ctx}$, and environment $U$, with the goal of surgical intervention $do(c_{tgt}\to c'_{tgt})$ while maintaining $\{C_{ctx},U\}_m\approx\{C_{ctx},U\}_{m'}$. Pipeline: QwenVL-Plus identifies hallucination points (existence, attribute, relation) in $y'$ and outputs editing instructions $\mathcal{T}$; Qwen-Image-Edit performs local modifications under reversible padding; QwenVL-Plus verifies the edit. Token-level differences between $y$ and $y'_{\text{new}}$ serve as a fine-grained mask for the single-image branch.
    *   **Design Motivation**: Synthetic/retrieved negatives introduce distribution shifts $P(C_{ctx},U\mid m)\neq P(C_{ctx},U\mid m')$, allowing the model to take shortcuts based on style. Surgical editing forces the model to discern the target concept to make correct preference judgments.

### Loss & Training
The final objective $\mathcal{L}_{\text{Total}}=\mathcal{L}_{\text{IC-VCO}}+\mathcal{L}'_{\text{IC-VCO}}$ is run symmetrically using $m$ and $m'$ as targets. Anchoring terms $\mathcal{L}_{\text{SingleAnc}}$ and $\mathcal{L}_{\text{MultiAnc}}$ follow existing practices to prevent likelihood degradation of the chosen response. Based on 21.4k seeds from SymMPO, 19,453 edited negative samples were produced with a 91% success rate.

## Key Experimental Results

### Main Results
Evaluated on LLaVA-NeXT-Interleave-Qwen-7B across five benchmarks. Macro-averages are as follows:

| Data Source | Method | Overall | HallusionBench aAcc | AMBER Attr | CRPE Exist | BLINK |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| — | LLaVA-NeXT-Interleave-Qwen-7B Baseline | 59.14 | 55.59 | 79.97 | 92.01 | 45.13 |
| Synthetic | mDPO | 61.64 | 61.51 | 80.27 | 91.79 | 44.87 |
| Synthetic | SymMPO | 61.50 | 60.79 | 80.41 | 91.83 | 44.88 |
| Synthetic | **IC-VCO (Ours)** | **62.83** | 61.94 | 81.81 | 93.16 | **48.93** |
| Edited | mDPO | 62.02 | 60.25 | 80.31 | 92.27 | 45.66 |
| Edited | SymMPO | 62.11 | 60.57 | 80.39 | 92.47 | 45.19 |
| Edited | **IC-VCO (Ours)** | **63.35** | **63.51** | **82.24** | **94.15** | **49.44** |

IC-VCO also leads on LLaVA-OneVision-Qwen2-7B, with significant Gains in BLINK and HallusionBench fAcc.

### Ablation Study
Ablations confirm that all three components are essential.

| Configuration | Overall | Note |
|:---|:---:|:---|
| Full IC-VCO (Multi-image+VCDist+Edited Negatives+mask) | 63.35 | Complete model |
| Synthetic Negatives Only (No Editing) | 62.83 | Hard negatives contribute ~0.5 points |
| Without VCDist | Drop | Single-image branch loses supervision; HallusionBench metrics degrade most |
| Single-image DPO Only (No Multi-image) | ≈ mDPO | No shared context -> Partition bias returns |

### Key Findings
- Edited negative samples improve all methods (Overall +0.4~0.8), showing "data hardness" is an independently effective dimension.
- The relative Gain of IC-VCO is largest on anti-hallucination and fine-grained reasoning (BLINK), consistent with the motivation of removing bias and strengthening contrast.
- CRPE Relation is an outlier: IC-VCO is slightly lower than mDPO/SymMPO, suggesting local edits for relational hallucinations need further refinement.

## Highlights & Insights
- The perspective shift from "changing conditions" to "changing prompts" is the most clever part: by placing visual differences explicitly in the context and letting anchoring instructions handle selection, the DPO formula remains unchanged while the partition functions perfectly cancel out.
- VCDist treats the multi-image branch as a teacher for gated distillation, which is valuable for any multimodal preference work using multi-image training for single-image inference (e.g., video temporal contrast).
- The surgical editing + token-level mask pipeline can be extracted independently: as long as "target concept vs context" can be defined, coarse negatives can be upgraded to hard negatives.

## Limitations & Future Work
- The editing pipeline relies heavily on QwenVL-Plus and Qwen-Image-Edit; the 91% success rate means ~9% of samples are discarded. Generalizability to low-resource or non-natural images (medical, remote sensing) needs verification.
- Training costs and context length double with the multi-image branch; scalability for ultra-long high-resolution images or larger models is not fully discussed.
- Improvements in CRPE Relation are limited, reflecting that "relationship" edits (involving relative positions and interactions) are harder to干预 cleanly than "existence/attribute" edits.

## Related Work & Insights
- **vs mDPO / S-VCO / SymMPO**: All use "image swapping" but ignore the bias from $Z(m,x)\neq Z(m',x)$. IC-VCO uses shared context to cancel the partition function.
- **vs V-DPO**: V-DPO also attempts visual preference modeling but remains a single-image conditional comparison. IC-VCO reuses existing LVLM multi-image interfaces without model architecture changes.
- **vs SymMPO (Synthetic Negatives)**: SymMPO uses T2I for negatives, causing global style shifts. IC-VCO uses local surgical editing, resulting in CLIP similarity distributions concentrated in high-similarity ranges, fundamentally addressing shortcut learning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Simultaneously closes the "theoretical gap" and "data gap" of visual preference DPO with a unified, elegant approach.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five benchmarks × two base models × multiple baselines + data source ablations; the only regret is the model size is limited to 7B.
- Writing Quality: ⭐⭐⭐⭐⭐ Formula derivations are tightly coupled with the narrative; the diagram of partition function residuals is particularly clear.
- Value: ⭐⭐⭐⭐⭐ VCDist and surgical editing modules are plug-and-play.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Systematic Reward Gap Optimization for Mitigating VLM Hallucinations](../../NeurIPS2025/multimodal_vlm/systematic_reward_gap_optimization_for_mitigating_vlm_hallucinations.md)
- [\[CVPR 2026\] ReasonMap: Towards Fine-Grained Visual Reasoning from Transit Maps](../../CVPR2026/multimodal_vlm/reasonmap_towards_fine-grained_visual_reasoning_from_transit_maps.md)
- [\[ICML 2026\] Mitigating Hallucinations in Large Vision-Language Models via Causal Route Gating](mitigating_hallucinations_in_large_vision-language_models_via_causal_route_gatin.md)
- [\[CVPR 2026\] CoVFT: Context-aware Visual Fine-tuning for Multimodal Large Language Models](../../CVPR2026/multimodal_vlm/covft_context-aware_visual_fine-tuning_for_multimodal_large_language_models.md)
- [\[CVPR 2026\] Mitigating Multimodal Hallucinations via Gradient-based Self-Reflection](../../CVPR2026/multimodal_vlm/mitigating_multimodal_hallucinations_via_gradient-based_self-reflection.md)

</div>

<!-- RELATED:END -->
