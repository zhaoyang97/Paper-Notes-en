---
title: >-
  [Paper Note] One Token, Two Fates: A Unified Framework via Vision Token Manipulation Against MLLMs Hallucination
description: >-
  [CVPR 2025][Hallucination Detection][MLLM hallucination] Proposes the first unified, training-free framework for mitigating MLLM hallucinations, operating synergistically within the hidden representation layers based on the dual roles of vision tokens—enhancement (SVC) and suppression (CRC). It improves POPE accuracy by ~2% on LLaVA-1.5 with only a 1.06× increase in inference latency.
tags:
  - "CVPR 2025"
  - "Hallucination Detection"
  - "MLLM hallucination"
  - "vision token"
  - "contrastive decoding"
  - "training-free"
  - "hidden representation calibration"
  - "visual attention decay"
date: 2026-05-08
content_hash: ee4e50648b782363
---

# One Token, Two Fates: A Unified Framework via Vision Token Manipulation Against MLLMs Hallucination

**Conference**: CVPR 2025  
**arXiv**: [2603.10360](https://arxiv.org/abs/2603.10360)  
**Code**: To be confirmed  
**Area**: Hallucination Detection  
**Keywords**: MLLM hallucination, vision token, contrastive decoding, training-free, hidden representation calibration, visual attention decay

## TL;DR

Proposes the first unified, training-free framework for mitigating MLLM hallucinations, operating synergistically within the hidden representation layers based on the dual roles of vision tokens—enhancement (SVC) and suppression (CRC). It improves POPE accuracy by ~2% on LLaVA-1.5 with only a 1.06× increase in inference latency.

## Background & Motivation

**Core Mechanism of MLLM Hallucination**: When Multi-modal Large Language Models (MLLMs) generate text, visual attention gradually decays over generation steps while language priors (text inertia) dominate. This leads to generated content contradicting visual evidence—namely, "hallucination".

**Limitations of Prior Work (Training-Free)**:
   - Visual enhancement methods (e.g., PAI): Strengthen visual signals by amplifying attention weights, but remain powerless against strong language priors.
   - Text suppression methods (e.g., VCD): Suppress language inertia at the logit level via contrastive decoding, but rely on image-level distortions (modality-gap) to generate negative samples, which are noisy and unstable.

**Failure of Naive Combination**: Direct combinations of PAI and VCD yield no performance improvements in experiments. The two methods have different design philosophies and intervention levels (one at the attention layer, the other at the output logit), leading to signal conflicts.

**Key Insight**: Vision tokens serve as the bridge for vision-language interaction and can simultaneously assume two roles: (1) providing complementary visual semantics by enhancing images (F2); (2) precisely isolating hallucination tendencies by cropping tokens (information-gap) (F3).

**Core Idea**: A unified framework operating at the hidden representation level, deriving all calibration signals—both enhancement and suppression working in synergy—from vision tokens.

## Method

### Overall Architecture

A unified training-free framework containing two key modules:
- **SVC (Synergistic Visual Calibration)**: Injects enhanced visual context at intermediate layers to counteract visual decay.
- **CRC (Causal Representation Calibration)**: Purifies internal model biases in shallow layers using negative samples from the hidden space.
- Both modules operate at the hidden representation level (rather than the logit level), avoiding signal conflicts.

### Key Findings

- **F1 Inverse Correlation**: As generation progresses, visual attention decays sharply, and hallucination frequency surges exactly where visual grounding is weakest.
- **F2 Semantic Complementarity**: Attention patterns of original and augmented images (flipped + blurred + noisy) are complementary, and their fusion enhances visual grounding.
- **F3 Information-gap > Modality-gap**: Cropping vision tokens in the hidden space (information-gap) generates stable, in-distribution hallucination probes; in contrast, pixel-level masking (modality-gap) produces noisy, out-of-distribution results.

### Key Design 1: SVC — Synergistic Visual Calibration

- **Function**: Injects rich visual context into key intermediate layers to counteract visual decay.
- **Steps**: (1) Apply random flipping + Gaussian blur + salt-and-pepper noise to the input image to generate an augmented image $\rightarrow$ (2) Concatenate original and augmented vision tokens to obtain the synergistic visual memory bank $\mathbf{V}_{\text{syn}} \in \mathbb{R}^{2N_v \times d}$ $\rightarrow$ (3) At the $L_c$-th layer, use scaled dot-product attention with the hidden state as Query and $\mathbf{V}_{\text{syn}}$ as Key/Value $\rightarrow$ (4) Inject via interpolation fusion: $\mathbf{H}'_t = (1-\lambda_s) \mathbf{H}_t + \lambda_s \mathbf{C}_t$
- **Design Motivation**: Leverages the semantic complementarity of F2, requiring no extra parameters and effectively enhancing visual representations through intervention at a single layer.

### Key Design 2: CRC — Causal Representation Calibration

- **Function**: Purifies hidden representations in shallow layers to eliminate the "hallucination direction".
- **Steps**: (1) Randomly crop vision tokens to keep only $N_h=5$ tokens, generating K=3 negative samples $\rightarrow$ (2) Execute parallel forward passes at the initial step t=0 to obtain raw and negative sample hidden states $\rightarrow$ (3) Average the differences to obtain a stable hallucination direction vector: $\mathbf{v}_{\text{crc}}^{(l)} = \frac{1}{K}\sum_{k=1}^{K} (\mathbf{H}_{\text{org}}^{(l)} - \mathbf{H}_{\text{neg}}^{(l,k)})$ $\rightarrow$ (4) In subsequent steps, perform calibration in the normalized space: $\mathbf{h}_{\text{crc}} = \mathbf{h}_{\text{norm}} + \lambda_c \mathbf{v}_{\text{norm}}$, and scale back to the original magnitude.
- **Causal Theory Support**: Based on the Structural Causal Model (SCM), the difference vector precisely captures the "bias induced by missing visual information", independent of the shared effects of text/query/bias.
- **Key Parameters**: $N_h=5$ (retaining 5/576 tokens to precisely trigger bias detection), $K=3$ (the optimal trade-off between performance and efficiency).

### Synergistic Mechanism

- SVC intervenes at the intermediate layer $L_c$ (enhancing vision), while CRC intervenes in the shallow layers from 1 to $L_c$ (suppressing bias).
- Both operate at the same level (hidden representations), ensuring signal compatibility rather than conflict.
- The hallucination direction vector is computed once at t=0 and cached for reuse, incurring minimal overhead.

## Key Experimental Results

### POPE Hallucination Evaluation (Average Accuracy % across 4 MLLMs)

| Benchmark | Vanilla | VCD | PAI | VISTA | ONLY | **Ours** |
|--------|---------|-----|-----|-------|------|----------|
| MSCOCO (LLaVA-1.5) | 84.79 | 84.80 | 85.85 | 86.15 | 86.03 | **86.79** |
| AOKVQA (LLaVA-1.5) | 77.23 | 76.29 | 78.65 | 81.23 | 80.55 | **82.23** |
| GQA (LLaVA-1.5) | 78.76 | 79.36 | 79.80 | 80.89 | 80.44 | **81.54** |

### CHAIR Hallucination Evaluation (CHAIR_S ↓, 64 tokens)

| Model | Vanilla | VCD | VISTA | ONLY | **Ours** |
|------|---------|-----|-------|------|----------|
| LLaVA-1.5 | 25.4 | 24.1 | 21.4 | 19.2 | **18.1** |
| Shikra | 25.4 | 23.2 | 19.5 | 21.4 | **16.7** |

### Efficiency Comparison

| Method | Latency (ms/token) | GPU Memory (MB) |
|------|----------------|-----------|
| Greedy | 30.3 (×1.00) | 14257 |
| VCD | 72.72 (×2.40) | 14984 |
| VISTA | 33.35 (×1.10) | 15024 |
| **Ours** | **32.1 (×1.06)** | **14924** |

### Ablation Study (LLaVA-1.5, POPE COCO Acc%)

| Configuration | Acc↑ | F1↑ |
|------|------|-----|
| Vanilla | 84.79 | 85.61 |
| + SVC (Original Only) | 85.04 | 85.68 |
| + SVC (Synergistic, Ours) | 85.55 | 86.04 |
| + CRC (Image-masked Negative Samples) | 84.77 | 85.72 |
| + CRC (Token-cropped, Ours) | 86.11 | 86.39 |
| **SVC + CRC (Full)** | **86.79** | **87.04** |

### Key Findings
- SVC and CRC are individually effective, and their combination yields further improvements—validating the synergistic design.
- Token-cropped negative samples $\gg$ image-masked negative samples (CRC: 86.11 vs 84.77), and t-SNE visualization confirms that cropped samples remain in-distribution.
- $N_h=5$ is a precise operating point: $\ge 20$ tokens are insufficient to trigger bias detection.

## Highlights & Insights

- **First Unified Framework**: Unifies the two independent pathways of enhancement and suppression into a dual manipulation of vision tokens, resolving the failure of naive combinations.
- **Information-gap > Modality-gap**: Cropping tokens in the hidden space is more stable and in-distribution than pixel-level masking, a finding that offers valuable guidance for future work.
- **Ultra-low Overhead**: Only 1.06× latency, significantly outperforming VCD's 2.4×, since the hallucination direction vector needs to be computed only once.
- **Cross-architecture Generalization**: Effective across both linear projection (LLaVA/Shikra) and Q-Former (MiniGPT-4/InstructBLIP) architectures.

## Limitations & Future Work

- Evaluated only on MLLMs with a scale of ~7B; effectiveness on larger models (e.g., 70B+) remains unknown.
- The augmentation strategies of SVC (flipping + blur + noise) are relatively fixed and may not represent the optimal combination of complementary semantics.
- CRC assumes the hallucination direction remains invariant during generation (computed at t=0 and cached); this direction might drift during long-text generation.
- Not validated in other multimodal scenarios such as video or 3D.

## Related Work & Insights

- **vs VCD**: VCD performs contrastive decoding with image distortion at the logit level, resulting in 2.4× latency; Ours operates at the hidden representation level using token cropping, with only 1.06× latency and higher-quality negative samples.
- **vs PAI**: PAI only enhances attention weights and fails to dispute strong language priors, whereas Ours synergizes dual pathways through SVC + CRC.
- **vs VISTA**: VISTA also intervenes at intermediate layers but only performs visual enhancement, whereas Ours adds the CRC suppression pathway.
- **Insight**: The idea of using token cropping as a causal probe can be extended to other scenarios requiring signal separation, such as bias detection and interpretability analysis.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The concept of unifying enhancement/suppression is novel, and the discovery of the information-gap is insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated with 4 MLLMs, 4 benchmarks, complete ablation studies, and visualization analysis.
- Writing Quality: ⭐⭐⭐⭐ The progression of the three findings is clear, though symbols are somewhat heavy.
- Value: ⭐⭐⭐⭐ Training-free + low overhead, offering strong practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] HalLoc: Token-Level Localization of Hallucinations for Vision Language Models](halloc_token-level_localization_of_hallucinations_for_vision_language_models.md)
- [\[CVPR 2025\] Antidote: A Unified Framework for Mitigating LVLM Hallucinations in Counterfactual Presupposition and Object Perception](antidote_a_unified_framework_for_mitigating_lvlm_hallucinations_in_counterfactua.md)
- [\[CVPR 2025\] Octopus: Alleviating Hallucination via Dynamic Contrastive Decoding](octopus_alleviating_hallucination_via_dynamic_contrastive_decoding.md)
- [\[ICCV 2025\] ONLY: One-Layer Intervention Sufficiently Mitigates Hallucinations in Large Vision-Language Models](../../ICCV2025/hallucination/only_onelayer_intervention_sufficiently_mitigates_hallucinat.md)
- [\[CVPR 2025\] Seeing Far and Clearly: Mitigating Hallucinations in MLLMs with Attention Causal Decoding](seeing_far_and_clearly_mitigating_hallucinations_in_mllms_with_attention_causal_.md)

</div>

<!-- RELATED:END -->
