---
title: >-
  [Paper Note] Seeing No Evil: Blinding Large Vision-Language Models to Safety Instructions via Adversarial Attention Hijacking
description: >-
  [ACL 2026][LLM Safety][Visual Jailbreaking Attack] This paper proposes Attention-Guided Visual Jailbreaking, which bypasses safety mechanisms by suppressing model attention toward safety instructions and anchoring it ont…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Visual Jailbreaking Attack"
  - "Attention Manipulation"
  - "Safety Alignment"
  - "Gradient Conflict"
  - "Large Vision-Language Models"
date: 2026-05-08
content_hash: c3ce83127b43a1dd
---

# Seeing No Evil: Blinding Large Vision-Language Models to Safety Instructions via Adversarial Attention Hijacking

**Conference**: ACL 2026  
**arXiv**: [2604.10299](https://arxiv.org/abs/2604.10299)  
**Code**: [github.com/Landsayy/AttentionJailbreak](https://github.com/Landsayy/AttentionJailbreak)  
**Area**: Multimodal VLM  
**Keywords**: Visual Jailbreaking Attack, Attention Manipulation, Safety Alignment, Gradient Conflict, Large Vision-Language Models

## TL;DR

This paper proposes Attention-Guided Visual Jailbreaking, which bypasses safety mechanisms by suppressing model attention toward safety instructions and anchoring it onto adversarial image features. Instead of force-attacking safety alignment, this method achieves a 94.4% attack success rate on Qwen-VL while reducing gradient conflict by 45%.

## Background & Motivation

**Background**: Large Vision-Language Models (LVLMs) are widely deployed in safety-critical scenarios such as AI assistants and content moderation. Their safety alignment relies on the attention mechanism to continuously retrieve safety instructions from the prefix region during each decoding step.

**Limitations of Prior Work**: Existing adversarial attacks primarily optimize output logits to maximize the probability of harmful outputs but ignore where the safety mechanism is implemented within the model. This leads to severe **gradient conflict**—the adversarial gradient direction opposes the safety retrieval gradient. Severe conflicts (cosine similarity $< -0.5$) occur in 20% of optimization iterations, causing optimization oscillation and slow convergence.

**Key Challenge**: Safety alignment is implemented in the intermediate Transformer layers via attention mechanisms, yet attacks are applied at the final output layer, creating a **functional location mismatch**.

**Goal**: Design an attack method that directly manipulates the attention distribution to bypass the safety retrieval mechanism rather than opposing it.

**Key Insight**: The continuous high-dimensional nature of the visual modality enables gradient-based attention distribution sculpting, which is unattainable via discrete combinatorial search in the text modality.

**Core Idea**: Safety alignment is essentially an attention retrieval process for prefix tokens. If this attention is suppressed, the model does not "violate" safety rules but rather "fails to retrieve" them—resulting in "safety blindness."

## Method

### Overall Architecture

The LVLM is decomposed into a vision encoder $\phi_v$, a multimodal projector $\phi_p$, and a language model $\phi_{lm}$. The input sequence is $x_{\text{seq}} = [s, h_{\text{img}}, q]$, where $s$ represents prefix tokens (system instructions and role markers), $h_{\text{img}}$ are image tokens, and $q$ are query tokens. Two attention auxiliary losses are added to the standard adversarial objective to form a push-pull mechanism.

### Key Designs

1.  **Suppression Loss**: Minimizes the attention weights from generated tokens to prefix tokens: $\mathcal{L}_{\text{suppress}} = \frac{1}{|\mathcal{I}_{\text{gen}}|} \sum_{i \in \mathcal{I}_{\text{gen}}} \sum_{j \in \mathcal{I}_{\text{prefix}}} \bar{A}_{i,j}$. This blocks the channel through which the model retrieves safety instructions. The design motivation is that since safety behavior is maintained via continuous prefix attention retrieval, suppressing this attention disables the safety mechanism at its source.

2.  **Anchoring Loss**: Maximizes the attention from generated tokens to image tokens: $\mathcal{L}_{\text{anchor}} = -\frac{1}{|\mathcal{I}_{\text{gen}}|} \sum_{i \in \mathcal{I}_{\text{gen}}} \sum_{j \in \mathcal{I}_{\text{img}}} \bar{A}_{i,j}$. Utilizing the competitive nature of softmax normalization, this ensures that the suppressed attention mass is redistributed to image tokens rather than query tokens, anchoring the generation process to adversarial visual features.

3.  **Posterior Layer Aggregation**: The attention matrices of the last $K=6$ layers are averaged: $\bar{A} = \frac{1}{K} \sum_{\ell=L-K+1}^{L} \frac{1}{H} \sum_{h=1}^{H} A^{(\ell,h)}$, based on evidence that refusal behavior is concentrated in the later Transformer layers. Target-to-prefix and target-to-image attention blocks are extracted via binary position selectors. The intervention is purely loss-driven and does not modify the model's forward computation.

### Loss & Training

The total loss is a weighted combination of three terms: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{target}} + \alpha \cdot \mathcal{L}_{\text{suppress}} + \beta \cdot \mathcal{L}_{\text{anchor}}$. Image perturbations $\delta$ are optimized via Projected Gradient Descent (PGD): $\delta^{(t+1)} = \Pi_{\|\cdot\|_\infty \leq \epsilon}[\delta^{(t)} - \eta \cdot \nabla_\delta \mathcal{L}_{\text{total}}]$. Default parameters are $\alpha=10, \beta=5, \eta=1/255, K=6$ over 2000 iterations. The attack is prompt-universal, allowing a single adversarial image to be used across different harmful queries.

## Key Experimental Results

### Main Results

| Method | ε | Qwen-VL AdvBench (G) | LLaVA-1.5 AdvBench (G) | Qwen-VL StrongREJECT (G) |
|------|---|---------------------|----------------------|-------------------------|
| VAE-JB | 32 | 68.8% | 57.9% | 55.6% |
| BAP | 32 | 4.2% | 54.8% | 9.3% |
| **Ours** | **32** | **94.4%** | **77.5%** | **90.4%** |
| **Ours** | **16** | **44.8%** | **62.3%** | **66.5%** |

### Ablation Study

| Configuration | α | β | AdvBench | StrongREJECT | HarmBench | JB | Average |
|------|---|---|---------|-------------|-----------|-----|------|
| $\mathcal{L}_{\text{target}}$ only | 0 | 0 | 55.0 | 47.0 | 70.5 | 69.0 | 60.4 |
| +Suppress | 10 | 0 | 63.3 | 70.0 | 72.0 | 73.0 | 69.6 |
| +Anchor | 0 | 5 | 57.5 | 44.1 | 72.5 | 70.0 | 61.0 |
| **Full** | **10** | **5** | **77.5** | **78.0** | **84.0** | **84.0** | **80.9** |

### Key Findings

-   The two losses exhibit a **synergistic effect**: while the sum of individual gains is 70.2%, the combined performance reaches 80.9%, exceeding the linear combination by 10.7%.
-   Successful attacks suppress system prompt attention by 80% and amplify image attention by 4.1×.
-   Causal intervention experiments: Restoring system attention on adversarial images ($b=2.0$) drops the ASR from 88.0% to 26.0%, proving that attention suppression is a causally necessary condition for attack success.
-   Cross-model transfer: Adversarial images achieve 52.0% ASR on GPT-4o and 39.6% on Claude-3.5.

## Highlights & Insights

-   **Discovery of Safety Blindness**: Attack success results not from "violating" safety rules but from the model being "unable to see" them. This perspective is highly insightful for understanding and improving safety alignment.
-   **Gradient Conflict Analysis**: This work provides the first quantification of the adversarial relationship between adversarial and safety gradients in output-oriented attacks, offering a principled explanation for optimization difficulties.
-   **Elegant Design**: The push-pull design is simple yet effective, significantly improving attack efficiency by adding only two simple auxiliary losses.
-   The method shows improvements across 13 safety scenarios in MM-SafetyBench, achieving an overall ASR of 47.38% (Qwen-VL), demonstrating its universality.

## Limitations & Future Work

-   The method depends on white-box gradient access; extending this to black-box scenarios is an important direction.
-   Hyperparameters ($\alpha, \beta$) used fixed values across all models; adaptive weighting strategies might further improve results.
-   The safety blindness mechanism suggests defensive directions: strengthening safety redundancy at the attention level rather than relying solely on prefix instructions.
-   Performance on InternVL2 is weaker (white-box ASR ~18%), likely because its safety mechanism does not rely entirely on prefix attention.
-   Adversarial images maintain a 59.0% ASR even under tight perturbation budgets ($\epsilon=8/255$), outperforming the baseline of 45.7%.
-   Future work could explore adaptively combining attention intervention with output optimization, dynamically adjusting weights based on the attack stage.

## Related Work & Insights

-   **Superficial Alignment Hypothesis**: Safety alignment is primarily encoded in a small number of formatting tokens, explaining why suppressing prefix attention suffices to bypass it.
-   **Representation Engineering**: Safety behavior is a structured, retrievable signal rather than a diffuse side effect, supporting the feasibility of targeted attention intervention.
-   **Text Adversarial Attacks** (GCG, PAIR, etc.): The continuity of the visual modality makes attention manipulation more efficient than discrete text searches.
-   Implications for LLM defense: There is a need to design safety mechanisms that are robust against attention manipulation.

## Rating

-   **Novelty**: ⭐⭐⭐⭐⭐ High originality in using safety blindness to understand and exploit vulnerabilities in safety alignment from an attention perspective.
-   **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Rigorous experimental design involving 5 benchmarks, 4 models, causal intervention analysis, gradient conflict quantification, and cross-model transfer.
-   **Writing Quality**: ⭐⭐⭐⭐⭐ Clear logic and smooth narrative, progressing from problem diagnosis (gradient conflict) to the solution (attention-guided) and mechanism validation (causal analysis).
-   **Value**: ⭐⭐⭐⭐ Advances the understanding of LVLM safety mechanisms and points toward new directions for defense research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] AutoRAN: Automated Hijacking of Safety Reasoning in Large Reasoning Models](autoran_automated_hijacking_of_safety_reasoning_in_large_reasoning_models.md)
- [\[ACL 2026\] Reasoning Hijacking: The Fragility of Reasoning Alignment in Large Language Models](reasoning_hijacking_the_fragility_of_reasoning_alignment_in_large_language_model.md)
- [\[ACL 2026\] Rethinking Jailbreak Detection of Large Vision Language Models with Representational Contrastive Scoring](rethinking_jailbreak_detection_of_large_vision_language_models_with_representati.md)
- [\[CVPR 2026\] Test-Time Attention Purification for Backdoored Large Vision Language Models](../../CVPR2026/llm_safety/test-time_attention_purification_for_backdoored_large_vision_language_models.md)
- [\[ACL 2026\] SafeMERGE: Preserving Safety Alignment in Fine-Tuned Large Language Models via Selective Layer-Wise Model Merging](safemerge_preserving_safety_alignment_in_fine-tuned_large_language_models_via_se.md)

</div>

<!-- RELATED:END -->
