---
title: >-
  [Paper Note] Seeing No Evil: Blinding Large Vision-Language Models to Safety Instructions via Adversarial Attention Hijacking
description: >-
  [ACL 2026][Multimodal VLM][visual jailbreak attack] This paper proposes Attention-Guided Visual Jailbreaking, which bypasses—rather than directly confronts—safety alignment mechanisms by suppressing model attention to safety instructions and anchoring attention to adversarial image features. The method achieves a 94.4% attack success rate (ASR) on Qwen-VL while reducing gradient conflicts by 45%.
tags:
  - ACL 2026
  - Multimodal VLM
  - visual jailbreak attack
  - attention manipulation
  - safety alignment
  - gradient conflict
  - large vision-language models
date: 2026-05-08
content_hash: f5ab0355cbae448c
---

# Seeing No Evil: Blinding Large Vision-Language Models to Safety Instructions via Adversarial Attention Hijacking

**Conference**: ACL 2026
**arXiv**: [2604.10299](https://arxiv.org/abs/2604.10299)
**Code**: [github.com/Landsayy/AttentionJailbreak](https://github.com/Landsayy/AttentionJailbreak)
**Area**: Multimodal VLM
**Keywords**: visual jailbreak attack, attention manipulation, safety alignment, gradient conflict, large vision-language models

## TL;DR

This paper proposes Attention-Guided Visual Jailbreaking, which bypasses—rather than directly confronts—safety alignment mechanisms by suppressing model attention to safety instructions and anchoring attention to adversarial image features. The method achieves a 94.4% attack success rate (ASR) on Qwen-VL while reducing gradient conflicts by 45%.

## Background & Motivation

**Background**: Large Vision-Language Models (LVLMs) are widely deployed in safety-critical applications such as AI assistants and content moderation. Their safety alignment relies on the model continuously retrieving safety instructions from prefix regions via the attention mechanism at each decoding step.

**Limitations of Prior Work**: Existing adversarial attacks primarily optimize output logits to maximize the probability of harmful outputs, while neglecting where safety mechanisms are implemented within the model. This leads to severe **gradient conflicts**—adversarial gradients are directionally opposed to safety-retrieval gradients, with 20% of optimization iterations exhibiting serious conflicts (cosine similarity $< -0.5$), causing optimization oscillation and slow convergence.

**Key Challenge**: Safety alignment is implemented via the attention mechanism in intermediate Transformer layers, whereas attacks are applied at the final output layer, creating a **functional location mismatch**.

**Goal**: To design an attack method that directly manipulates attention distributions to circumvent the safety retrieval mechanism rather than opposing it.

**Key Insight**: The continuous, high-dimensional nature of the visual modality enables gradient-based sculpting of attention distributions—something that discrete combinatorial search in the text modality cannot achieve.

**Core Idea**: Safety alignment is essentially an attention retrieval process over prefix tokens. Suppressing this attention means the model does not "violate" safety rules but rather "fails to retrieve" them—a phenomenon termed *safety blindness*.

## Method

### Overall Architecture

The LVLM is decomposed into a visual encoder $\phi_v$, a multimodal projector $\phi_p$, and a language model $\phi_{lm}$. The input sequence is $x_{\text{seq}} = [s, h_{\text{img}}, q]$, where $s$ denotes prefix tokens (system instructions and role markers), $h_{\text{img}}$ denotes image tokens, and $q$ denotes query tokens. Two attention auxiliary losses are added on top of the standard adversarial objective, forming a push-pull mechanism.

### Key Designs

1. **Attention Suppression Loss**: Minimizes the attention weights from generated tokens to prefix tokens: $\mathcal{L}_{\text{suppress}} = \frac{1}{|\mathcal{I}_{\text{gen}}|} \sum_{i \in \mathcal{I}_{\text{gen}}} \sum_{j \in \mathcal{I}_{\text{prefix}}} \bar{A}_{i,j}$, thereby blocking the channel through which the model retrieves safety instructions. The design motivation is that safe behavior is continuously retrieved via prefix attention; suppressing this attention disables the safety mechanism at its source.

2. **Attention Anchoring Loss**: Maximizes the attention from generated tokens to image tokens: $\mathcal{L}_{\text{anchor}} = -\frac{1}{|\mathcal{I}_{\text{gen}}|} \sum_{i \in \mathcal{I}_{\text{gen}}} \sum_{j \in \mathcal{I}_{\text{img}}} \bar{A}_{i,j}$. Exploiting the competitive nature of softmax normalization, this loss ensures that suppressed attention mass is redistributed to image tokens rather than query tokens, anchoring the generation process to adversarial visual features.

3. **Late-Layer Attention Aggregation**: Attention matrices from the last $K=6$ layers are averaged as $\bar{A} = \frac{1}{K} \sum_{\ell=L-K+1}^{L} \frac{1}{H} \sum_{h=1}^{H} A^{(\ell,h)}$, motivated by prior findings that refusal behavior is concentrated in later Transformer layers. Binary position selectors extract the target→prefix and target→image attention blocks. The entire intervention is purely loss-driven and does not modify the model's forward computation.

### Loss & Training

The total loss is a weighted combination of three terms: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{target}} + \alpha \cdot \mathcal{L}_{\text{suppress}} + \beta \cdot \mathcal{L}_{\text{anchor}}$. Image perturbation $\delta$ is optimized via projected gradient descent: $\delta^{(t+1)} = \Pi_{\|\cdot\|_\infty \leq \epsilon}[\delta^{(t)} - \eta \cdot \nabla_\delta \mathcal{L}_{\text{total}}]$. Default hyperparameters are $\alpha=10$, $\beta=5$, $\eta=1/255$, $K=6$, with 2000 iterations. The attack is prompt-universal: a single adversarial image can be reused across different harmful queries.

## Key Experimental Results

### Main Results

| Method | ε | Qwen-VL AdvBench (G) | LLaVA-1.5 AdvBench (G) | Qwen-VL StrongREJECT (G) |
|--------|---|---------------------|------------------------|--------------------------|
| VAE-JB | 32 | 68.8% | 57.9% | 55.6% |
| BAP | 32 | 4.2% | 54.8% | 9.3% |
| **Ours** | **32** | **94.4%** | **77.5%** | **90.4%** |
| **Ours** | **16** | **44.8%** | **62.3%** | **66.5%** |

### Ablation Study

| Configuration | α | β | AdvBench | StrongREJECT | HarmBench | JB | Avg. |
|---------------|---|---|----------|-------------|-----------|-----|------|
| $\mathcal{L}_{\text{target}}$ only | 0 | 0 | 55.0 | 47.0 | 70.5 | 69.0 | 60.4 |
| +Suppress | 10 | 0 | 63.3 | 70.0 | 72.0 | 73.0 | 69.6 |
| +Anchor | 0 | 5 | 57.5 | 44.1 | 72.5 | 70.0 | 61.0 |
| **Full** | **10** | **5** | **77.5** | **78.0** | **84.0** | **84.0** | **80.9** |

### Key Findings

- The two losses exhibit a **synergistic effect**: the sum of individual gains is 70.2%, yet their combination achieves 80.9%, exceeding the linear combination by 10.7%.
- Successful attacks suppress system-prompt attention by 80% and amplify image attention by 4.1×.
- Causal intervention experiment: restoring system-prompt attention ($b=2.0$) on adversarial images reduces ASR from 88.0% to 26.0%, confirming that attention suppression is a causally necessary condition for attack success.
- Cross-model transfer: adversarial images achieve 52.0% ASR on GPT-4o and 39.6% on Claude-3.5.

## Highlights & Insights

- **Discovery of the Safety Blindness Mechanism**: Attack success arises not because the model "violates" safety rules, but because the model "cannot see" them. This perspective offers important implications for understanding and improving safety alignment.
- **Gradient Conflict Analysis** provides the first quantitative characterization of the adversarial relationship between attack gradients and safety gradients in output-oriented attacks, offering a principled explanation for optimization difficulties.
- The push-pull design is elegant in its simplicity: the addition of two straightforward auxiliary losses substantially improves attack efficiency.
- Consistent improvements are observed across all 13 safety scenarios on MM-SafetyBench, with an overall ASR of 47.38% on Qwen-VL, demonstrating the generality of the method.

## Limitations & Future Work

- The method relies on white-box gradient access; extension to black-box settings is an important future direction.
- Hyperparameters ($\alpha, \beta$) are fixed across all models; adaptive weighting strategies may further improve performance.
- The discovered safety blindness mechanism should motivate defensive research: reinforcing safety redundancy at the attention level rather than relying solely on prefix instructions.
- Performance on InternVL2 is relatively weak (white-box ASR ~18%), possibly because its safety mechanism does not rely exclusively on prefix attention.
- Under tight perturbation budgets ($\epsilon=8/255$), adversarial images still maintain 59.0% ASR, outperforming the baseline (45.7%).
- Future work may explore adaptive combinations of attention intervention and output optimization, with dynamic weight adjustment across attack stages.

## Related Work & Insights

- **Superficial Alignment Hypothesis**: Safety alignment is predominantly encoded in a small number of formatting tokens, which explains why suppressing prefix attention suffices to bypass the safety mechanism.
- **Representation Engineering**: Safe behavior is a structured, retrievable signal rather than a diffuse side effect, supporting the feasibility of targeted attention intervention.
- **Textual Adversarial Attacks** (GCG, PAIR, etc.): The continuity of the visual modality makes attention manipulation more efficient than discrete text search.
- Implications for LLM safety defense research: safety mechanisms robust to attention manipulation need to be designed.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Understanding and exploiting the vulnerability of safety alignment from an attention mechanism perspective; the concept of safety blindness is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Five benchmarks, four models, causal intervention analysis, gradient conflict quantification, and cross-model transfer; the experimental design is exceptionally rigorous.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The narrative is logically coherent, progressing seamlessly from problem diagnosis (gradient conflict) to solution (attention-guided attack) to mechanism verification (causal analysis).
- **Value**: ⭐⭐⭐⭐ Advances understanding of LVLM safety mechanisms while pointing to new directions for defensive research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SafetyALFRED: Evaluating Safety-Conscious Planning of Multimodal Large Language Models](safetyalfred_evaluating_safety-conscious_planning_of_multimodal_large_language_m.md)
- [\[ACL 2026\] Rethinking Jailbreak Detection of Large Vision Language Models with Representational Contrastive Scoring](rethinking_jailbreak_detection_of_large_vision_language_models_with_representati.md)
- [\[ACL 2026\] When Vision-Language Models Judge Without Seeing: Exposing Informativeness Bias](when_vision-language_models_judge_without_seeing_exposing_informativeness_bias.md)
- [\[ACL 2026\] HiPrune: Hierarchical Attention for Efficient Token Pruning in Vision-Language Models](hiprune_hierarchical_attention_for_efficient_token_pruning_in_vision-language_mo.md)
- [\[CVPR 2026\] Test-Time Attention Purification for Backdoored Large Vision Language Models](../../CVPR2026/multimodal_vlm/test-time_attention_purification_for_backdoored_large_vision_language_models.md)

</div>

<!-- RELATED:END -->
