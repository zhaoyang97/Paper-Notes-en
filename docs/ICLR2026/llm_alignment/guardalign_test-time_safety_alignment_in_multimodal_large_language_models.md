---
title: >-
  [Paper Note] GuardAlign: Test-time Safety Alignment in Multimodal Large Language Models
description: >-
  [ICLR 2026][LLM Alignment][LVLM Safety] This paper proposes GuardAlign, a training-free test-time safety defense framework for multimodal large language models. It leverages optimal transport (OT) to precisely detect and mask unsafe regions in images, and employs cross-modal attention calibration to sustain the influence of safety prefixes across layers. Evaluated on six LVLMs, GuardAlign reduces unsafe response rates by up to 39% while preserving or improving general capability.
tags:
  - ICLR 2026
  - LLM Alignment
  - LVLM Safety
  - Optimal Transport
  - Attention Calibration
  - Test-time Defense
  - Visual Safety Detection
date: 2026-05-08
content_hash: 29fe2f9df0a3ed5a
---

# GuardAlign: Test-time Safety Alignment in Multimodal Large Language Models

**Conference**: ICLR 2026
**arXiv**: [2602.24027](https://arxiv.org/abs/2602.24027)
**Code**: None
**Area**: LLM Alignment
**Keywords**: LVLM Safety, Optimal Transport, Attention Calibration, Test-time Defense, Visual Safety Detection

## TL;DR
This paper proposes GuardAlign, a training-free test-time safety defense framework for multimodal large language models. It leverages optimal transport (OT) to precisely detect and mask unsafe regions in images, and employs cross-modal attention calibration to sustain the influence of safety prefixes across layers. Evaluated on six LVLMs, GuardAlign reduces unsafe response rates by up to 39% while preserving or improving general capability.

## Background & Motivation

**Background**: LVLMs (e.g., LLaVA, InternVL) have achieved remarkable progress in vision-language reasoning, yet they remain vulnerable to harmful responses when input images carry malicious semantics. Existing defenses fall into fine-tuning-based methods (high cost) and test-time methods (e.g., contrastive decoding, high overhead); recent work has introduced lightweight input-side defense paradigms.

**Limitations of Prior Work**:
- The first step of input-side defense uses CLIP to detect unsafe images, but similarity scores between safe and unsafe samples overlap severely in complex scenes, leading to inaccurate detection.
- The second step adds safety prefixes to activate the model's internal defense mechanisms, but the attention weights of these prefixes decay monotonically with depth, diluting the safety signal.
- After an initial refusal, models frequently begin generating unsafe content following transition words such as "However."

**Key Challenge**: Global CLIP similarity fails to capture local malicious semantics, and safety prefix signals attenuate in deeper layers.

**Goal**: More accurate unsafe content detection and more persistent safety signal maintenance.

**Key Insight**: Model fine-grained distributional distances between image patches and unsafe semantics via OT; prevent safety prefix signal decay via an attention calibration mechanism.

**Core Idea**: OT-based malicious patch detection + attention calibration for safety prefix persistence = training-free LVLM safety defense.

## Method

### Overall Architecture
Input image → OT-enhanced safety detection (identify and mask malicious patches) → sanitized image + safety prefix + user query → cross-modal attention calibration (amplify persistent attention to safety prefix tokens) → LVLM generates safe response. The entire pipeline requires no training or fine-tuning and operates purely at inference time.

### Key Designs

1. **OT-Enhanced Safety Detection**:

    - **Function**: Precisely identify which image patches contain unsafe semantics using optimal transport.
    - **Mechanism**: The image is divided into $M$ patches and $C$ unsafe category text anchors are defined. CLIP encodes image patches $\{\mathbf{x}^m\}$ and text variants $\{\mathbf{z}_i^n\}$ separately, modeled as discrete distributions: $\mathbb{P}(\mathbf{x})=\sum_m a^m \delta(\mathbf{x}^m)$, $\mathbb{Q}_i(\mathbf{z})=\sum_n b_i^n \delta(\mathbf{z}_i^n)$. Patch weights $a^m$ are entropy-weighted (low-entropy patches receive higher weights as higher-confidence regions). The Sinkhorn algorithm solves for OT distances; per-patch scores are aggregated across all categories via $d_{\text{OT}}(m)=\sum_i\sum_n \mathbf{T}_i(m,n)\mathbf{C}_i(m,n)$, and patches below the threshold are deemed unsafe and masked.
    - **Design Motivation**: Global CLIP similarity exhibits severe overlap between safe and unsafe samples in complex scenes. OT models fine-grained alignment at the patch level and naturally identifies the most suspicious local regions through its transport plan.
    - **Theoretical Guarantee**: It is proven that the classification error of the OT method is upper-bounded by that of cosine similarity, since the entropy-weighted transport plan prioritizes alignment of discriminative features, thereby increasing the normalized margin between safe and unsafe classes.

2. **Cross-Modal Attention Calibration**:

    - **Function**: Amplify the attention from instruction tokens to safety prefix tokens at intermediate layers to prevent safety signal decay.
    - **Mechanism**: For layer $l$ and head $h$, the attention scores are updated as $\hat{\mathbf{Z}}_{l,h} = \mathbf{Z}_{l,h} + \gamma \mathbf{M}^{\text{pref}}_{l,h} \circ \mathbf{Z}_{l,h}$, where $\gamma > 0$ controls amplification strength and $\mathbf{M}^{\text{pref}}$ is a mask selecting only instruction-token→prefix-token query-key pairs with positive correlation.
    - **Design Motivation**: Experiments reveal that safety prefix attention weights decay monotonically with layer depth in LLaVA, causing the model to be redirected toward unsafe content by transition words such as "However" after an initial refusal. Attention calibration ensures safety signals remain active throughout all layers.

### Loss & Training
- No training required; purely test-time inference.
- OT is solved via the Sinkhorn algorithm (efficient iterative convergence).
- Safety detection threshold $\tau=0.42$; attention amplification coefficient $\gamma > 0$ serves as a hyperparameter.

## Key Experimental Results

### Main Results: Unsafe Response Rate (USR) Comparison

| Model | Method | SPA-VL ↓ | MM-SafetyBench SD+TYPO ↓ | FigStep ↓ | Suffix ↓ | Unconstrained ↓ |
|------|------|---------|------------------------|-----------|----------|----------------|
| LLaVA-1.5-7B | Vanilla | 46.04 | 40.46 | 58.60 | 62.00 | 97.50 |
| | + ECSO | 23.40 | 15.89 | 37.40 | 59.00 | 95.00 |
| | + ETA | 16.98 | 15.83 | 7.80 | 22.60 | 22.50 |
| | + **GuardAlign** | **10.31** | **9.65** | **3.40** | **15.30** | **15.00** |
| LLaMA3.2-11B | Vanilla | 7.17 | 19.17 | 41.60 | 44.00 | 15.00 |
| | + **GuardAlign** | **1.25** | **2.28** | **3.50** | **3.00** | **3.50** |

### Ablation Study: Component Contributions

| Configuration | SPA-VL USR ↓ | VQAv2 ↑ | Note |
|------|-------------|---------|------|
| ETA baseline | 16.98 | 78.51 | CLIP detection + safety prefix |
| + OT detection (replaces CLIP) | 12.45 | 78.85 | OT improves detection accuracy |
| + Attention calibration | 10.31 | 79.21 | Full GuardAlign |
| OT detection only | ~14 | ~79 | OT contributes most |
| Attention calibration only | ~13 | ~79 | Calibration also has independent contribution |

### Key Findings
- **OT vs. CLIP Detection**: OT achieves clear separation between safe and unsafe samples on SPA-VL, whereas CLIP similarity scores exhibit severe overlap.
- **Attention Calibration Prevents "However" Attacks**: After calibration, prefix attention remains stable across all layers, eliminating the phenomenon of the model pivoting to unsafe content after an initial refusal.
- **General Capability Improves**: GuardAlign raises VQAv2 accuracy from 78.51% to 79.21% and improves performance on MME and other benchmarks — masking irrelevant patches and calibrating attention also reduce semantic noise in multimodal fusion.
- **Efficiency Advantage**: Runtime overhead is minimal, as the Sinkhorn algorithm converges rapidly.

## Highlights & Insights
- **A New Perspective on OT for Safety Detection**: Recasting image safety detection as a distributional distance problem is more robust than per-patch cosine similarity. The transport plan inherently reveals which patches are most suspicious.
- **Discovery of Safety Prefix Attention Decay**: This observation explains why simply prepending a safety prefix is insufficient — the model effectively "forgets" the safety instruction in deeper layers. Attention calibration is a lightweight and effective remedy.
- **Safety and Capability as a Positive-Sum Game**: Safety defenses typically sacrifice general capability, but GuardAlign's patch masking and attention calibration simultaneously reduce multimodal fusion noise, achieving gains on both fronts.

## Limitations & Future Work
- **Reliance on Predefined Unsafe Categories**: A fixed list of unsafe semantic categories must be specified in advance, which may fail to generalize to novel attack types.
- **Manual Threshold Tuning**: $\tau=0.42$ is empirically determined and may require adjustment for different models or deployment scenarios.
- **Addresses Only Visual-Side Attacks**: Defense against purely text-based jailbreaks relies solely on the safety prefix, with no dedicated text-side detection mechanism.
- **Future Directions**: An LLM could be used to automatically generate adaptive unsafe category lists; dynamic attention calibration strength at each generation step could be integrated with reasoning-direction re-evaluation approaches such as SSAH.

## Related Work & Insights
- **vs. ECSO**: ECSO employs CLIP detection and safety prefixes; GuardAlign improves both components (OT replacing CLIP, and attention calibration strengthening the prefix).
- **vs. ETA**: ETA is the direct predecessor of GuardAlign; GuardAlign addresses ETA's two core deficiencies.
- **vs. VLGuard (Posthoc-LoRA)**: VLGuard requires fine-tuning, whereas GuardAlign is training-free and achieves superior general capability.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of OT-based safety detection and attention calibration is novel, though neither component is entirely unprecedented.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Six models evaluated across five safety benchmarks and multiple general benchmarks — highly comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Well-organized with theoretical analysis as an added strength.
- **Value**: ⭐⭐⭐⭐ A practical test-time defense solution that is straightforward to deploy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Semantic-aware Wasserstein Policy Regularization for Large Language Model Alignment](semantic-aware_wasserstein_policy_regularization_for_large_language_model_alignm.md)
- [\[ICLR 2026\] A2D: Any-Order, Any-Step Safety Alignment for Diffusion Language Models](a2d_any-order_any-step_safety_alignment_for_diffusion_language_models.md)
- [\[AAAI 2026\] W2S-AlignTree: Weak-to-Strong Inference-Time Alignment for Large Language Models via Monte Carlo Tree Search](../../AAAI2026/llm_alignment/w2s-aligntree_weak-to-strong_inference-time_alignment_for_large_language_models_.md)
- [\[ICLR 2026\] Towards Understanding Valuable Preference Data for Large Language Model Alignment](towards_understanding_valuable_preference_data_for_large_language_model_alignmen.md)
- [\[ICCV 2025\] Heuristic-Induced Multimodal Risk Distribution Jailbreak Attack for Multimodal Large Language Models](../../ICCV2025/llm_alignment/heuristic-induced_multimodal_risk_distribution_jailbreak_attack_for_multimodal_l.md)

</div>

<!-- RELATED:END -->
