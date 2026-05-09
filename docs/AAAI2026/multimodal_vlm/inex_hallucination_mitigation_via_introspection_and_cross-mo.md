---
title: >-
  [Paper Note] InEx: Hallucination Mitigation via Introspection and Cross-Modal Multi-Agent Collaboration
description: >-
  [AAAI 2026][Multimodal VLM][Multimodal hallucination] This paper proposes InEx, a framework that iteratively verifies and corrects MLLM outputs via internal introspective reasoning (TVER-driven uncertainty-aware visual augmentation) and external cross-modal multi-agent collaboration (textual self-reflection + image editing verification + visual self-reflection), achieving an 8.9% improvement on POPE and consistently outperforming OPERA/VCD/ICD across multiple hallucination and general benchmarks.
tags:
  - AAAI 2026
  - Multimodal VLM
  - Multimodal hallucination
  - uncertainty estimation
  - cross-modal verification
  - multi-agent collaboration
  - training-free
date: 2026-05-08
content_hash: 39a89b4f3bad9c9c
---

# InEx: Hallucination Mitigation via Introspection and Cross-Modal Multi-Agent Collaboration

**Conference**: AAAI 2026
**arXiv**: [2512.02981](https://arxiv.org/abs/2512.02981)
**Code**: N/A
**Area**: Multimodal VLM / Hallucination Mitigation / Multi-Agent
**Keywords**: Multimodal hallucination, uncertainty estimation, cross-modal verification, multi-agent collaboration, training-free

## TL;DR
This paper proposes InEx, a framework that iteratively verifies and corrects MLLM outputs via internal introspective reasoning (TVER-driven uncertainty-aware visual augmentation) and external cross-modal multi-agent collaboration (textual self-reflection + image editing verification + visual self-reflection), achieving an 8.9% improvement on POPE and consistently outperforming OPERA/VCD/ICD across multiple hallucination and general benchmarks.

## Background & Motivation
Hallucination in MLLMs—generating linguistically fluent responses that are inconsistent with image content—remains a central obstacle to reliable deployment. Existing mitigation approaches each have notable limitations: (1) preprocessing methods (fine-tuning/RLHF) require extensive human annotation and scale poorly; (2) inference-time methods (VCD/OPERA) are tightly coupled with the model's internal reasoning and lack external verification, making them prone to "meta-hallucination," where the model maintains high confidence in erroneous knowledge; (3) post-processing methods (Woodpecker) are reactive corrections rather than proactive prevention. Critically, these approaches decouple inference-time optimization from post-hoc verification, failing to establish a complete cognitive pipeline analogous to the human process of "reducing uncertainty via introspection → achieving consensus via external verification."

## Core Problem
How can MLLMs autonomously reduce hallucinations without retraining? This decomposes into two sub-problems: (1) How can uncertainty signals be exploited during generation to proactively enhance visual information and reduce reasoning errors? (2) How can external, multi-modal, multi-perspective verification confirm or correct generated outputs?

## Method

### Overall Architecture
InEx = **In** (internal introspective reasoning) + **Ex** (external cross-modal multi-agent collaboration). The Decision Agent generates an initial response → the Textual Self-Reflection Agent verifies it against a dense caption → if inconsistent, structured textual feedback is provided → the Decision Agent self-corrects → the Image Editing Agent edits the image according to the response → the Visual Self-Reflection Agent computes the CLIP similarity between the edited and original images → if consistent, the response passes; otherwise, iteration continues (up to 4 rounds).

### Key Designs
1. **TVER (Text-to-Visual Entropy Ratio)-driven Introspective Reasoning**: For each attention head, the ratio of text to visual attention entropy is computed as TVER = Entropy(T)/Entropy(V). A high TVER indicates high uncertainty on the textual side and over-confidence on the visual side, suggesting the model may be misled by erroneous visual cues. When TVER ≥ γ_TVER, introspection is triggered:

    - **Self-Introspective Visual Augmentation**: Relevant information is retrieved from visual tokens via similarity-weighted aggregation and injected into the FFN: $\text{FFN}_{introspect} = \alpha \Delta(\mathbf{z}|\bar{H}) + (1-\alpha)\text{FFN}(\bar{H})$
    - **Enhanced Logits**: At the final layer, VE-MHA (masking high-TVER attention heads) generates augmented logits.
    - **Self-Introspective Decoding**: The Manhattan distance between original and enhanced logits is computed; if consistent, they are collaboratively fused; if divergent, contrastive decoding is applied.

2. **Cross-Modal Multi-Agent Collaboration**:

    - **Textual Self-Reflection Agent**: Verifies the response from multiple perspectives (actions/objects/colors, etc., up to 4 perspectives) based on a dense caption, repeated 3 times with ensemble aggregation. If verification fails, structured textual feedback is provided.
    - **Image Editing Agent** (IC-Edit): Edits the original image according to the generated response—if the response is accurate, the edited image should be consistent with the original.
    - **Visual Self-Reflection Agent**: Computes CLIP similarity; the response passes if > γ_CLIP = 0.9.

3. **Information Bottleneck Theoretical Support**: It is proven that the In module increases mutual information between hidden states and visual inputs (Theorem 1), reduces the conditional entropy of predicted outputs (Theorem 2), and optimizes the IB objective (Theorem 3).

### Loss & Training
Entirely training-free; no model parameters are modified. Hallucinations are reduced solely at inference time through attention analysis (TVER), feature injection, logit fusion, and multi-agent interaction.

## Key Experimental Results

| Dataset | Metric | InEx | OPERA | VCD | ICD | Baseline |
|--------|------|------|-------|-----|-----|------|
| POPE (MSCOCO avg) | Acc | **88.73** (+8.9) | 84.14 (+4.3) | 82.60 (+2.7) | 82.97 (+3.1) | 79.83 |
| MME-Hall | Score | **673.3** (+30) | 610.0 (-33) | 648.3 (+5) | 583.3 (-60) | 643.3 |
| MMBench | Score | **67.17** (+4.4) | 62.80 (0) | 54.21 (-8.6) | 39.78 (-23) | 62.80 |
| MM-Vet | Score | **36.00** (+4.9) | 32.00 (+0.9) | 30.20 (-0.9) | 25.90 (-5.2) | 31.10 |
| LLaVA-Bench | Score | **66.5** (+3.1) | - | - | - | 63.4 |

### Ablation Study
- In alone: POPE 79.83→86.43 (+6.6), largest individual contribution
- Ex-text alone: 83.20; Ex-visual alone: 85.20
- In + Ex-text: 86.39; In + Ex-visual: 87.77
- Full model: **88.73**; the three modules are complementary
- All image editing models are effective, with IC-Edit performing best
- Lower TVER threshold γ_TVER yields better performance (more sensitive introspection triggering)
- Dynamic layer selection outperforms fixed-layer injection
- Statistical significance: 20 independent runs, t-test p < 10^-25

## Highlights & Insights
- **Cognitively-inspired complete framework**: The human cognitive pipeline of "reducing uncertainty via introspection → achieving consensus via external verification" is systematically formalized as an AI framework.
- **TVER as a hallucination signal**: The text-to-visual attention entropy ratio is a concise and effective uncertainty indicator, achieving optimal AUROC and ECE among compared methods.
- **Elegant cross-modal verification design**: Image editing is used to verify textual responses—if the described content is accurate, the image edited according to the response should be consistent with the original. This constitutes a novel self-consistency check.
- **Training-free and model-agnostic**: Demonstrated to be effective across LLaVA-1.5-7B, Qwen-VL-10B, and GLM-4V-9B.

## Limitations & Future Work
- High inference cost: multi-agent collaboration + up to 4 iterations + image editing (100-step inference) incur significant latency.
- Dependent on the quality of dense captions and image editing models as the foundation for external verification.
- Limited to vision-text modalities; not extended to audio.
- TVER threshold and other hyperparameters require individual tuning for different MLLMs.
- The theoretical analysis is grounded in the information bottleneck framework, but the accuracy of uncertainty estimation remains subject to variation in practice.

## Related Work & Insights
- **vs OPERA**: OPERA performs only beam-search-level decoding correction, an in-processing method lacking external verification; InEx surpasses it by 4.6% on POPE.
- **vs VCD/ICD**: VCD applies contrastive decoding and ICD applies instruction contrastive decoding; both are single-agent inference-time methods. InEx's multi-agent collaboration is more robust.
- **vs Woodpecker**: Woodpecker is a purely post-hoc method that passively corrects errors; InEx proactively reduces uncertainty during generation.

The idea of using image editing to verify the accuracy of textual descriptions is highly novel and can be generalized beyond visual question answering. The TVER metric can serve as a general-purpose hallucination detection tool for MLLMs. The multi-agent collaboration paradigm is broadly compatible with other agent system designs.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — A unified framework of introspection and cross-modal verification; the image-editing-based verification design is particularly ingenious.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers 8 benchmarks, 3 models, 20-run statistical significance testing, and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with well-supported theoretical analysis, though the paper is lengthy.
- Value: ⭐⭐⭐⭐ — Meaningful contribution to MLLM hallucination mitigation; the training-free nature facilitates practical deployment.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] VipAct: Visual-Perception Enhancement via Specialized VLM Agent Collaboration and Tool-use](vipact_visual-perception_enhancement_via_specialized_vlm_age.md)
- [\[ACL 2026\] Spotlight and Shadow: Attention-Guided Dual-Anchor Introspective Decoding for MLLM Hallucination Mitigation](../../ACL2026/multimodal_vlm/spotlight_and_shadow_attention-guided_dual-anchor_introspective_decoding_for_mll.md)
- [\[CVPR 2026\] KVSmooth: Mitigating Hallucination in Multi-modal Large Language Models through Key-Value Smoothing](../../CVPR2026/multimodal_vlm/kvsmooth_mitigating_hallucination_in_multimodal_la.md)
- [\[AAAI 2026\] Concept-RuleNet: Grounded Multi-Agent Neurosymbolic Reasoning in Vision Language Models](concept-rulenet_grounded_multi-agent_neurosymbolic_reasoning.md)
- [\[AAAI 2026\] Aligning the True Semantics: Constrained Decoupling and Distribution Sampling for Cross-Modal Alignment](aligning_the_true_semantics_constrained_decoupling_and_distr.md)

<!-- RELATED:END -->
