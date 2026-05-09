---
title: >-
  [Paper Note] Differentiated Directional Intervention: A Framework for Evading LLM Safety Alignment
description: >-
  [AAAI 2026][LLM Alignment][LLM safety] This work deconstructs the internal representations of LLM safety alignment from the conventional "single refusal direction" into two functionally independent directions — a **harm detection direction** and a **refusal execution direction** — and proposes the DBDI framework, which applies adaptive projection elimination and direct steering to intervene on each direction separately, achieving a 97.88% attack success rate (ASR) on Llama-2.
tags:
  - AAAI 2026
  - LLM Alignment
  - LLM safety
  - jailbreak attacks
  - activation manipulation
  - safety alignment
  - interpretability
date: 2026-05-08
content_hash: 0f1465008e92f03c
---

# Differentiated Directional Intervention: A Framework for Evading LLM Safety Alignment

**Conference**: AAAI 2026
**arXiv**: [2511.06852](https://arxiv.org/abs/2511.06852)
**Code**: None
**Area**: LLM Alignment
**Keywords**: LLM safety, jailbreak attacks, activation manipulation, safety alignment, interpretability

## TL;DR
This work deconstructs the internal representations of LLM safety alignment from the conventional "single refusal direction" into two functionally independent directions — a **harm detection direction** and a **refusal execution direction** — and proposes the DBDI framework, which applies adaptive projection elimination and direct steering to intervene on each direction separately, achieving a 97.88% attack success rate (ASR) on Llama-2.

## Background & Motivation

**Background**: LLMs trained with safety alignment techniques such as RLHF learn to refuse harmful requests, yet alignment does not truly erase harmful capabilities — it merely suppresses them. White-box jailbreak attacks exploit the model's internal states to bypass safety mechanisms.

**Limitations of Prior Work**: Mainstream activation-level attack methods (e.g., Directional Ablation by Arditi et al.) model the safety mechanism as a single linear direction in activation space — obtained by computing the mean difference between activations of harmful and benign prompts. However, this single direction conflates two distinct neural processes.

**Key Challenge**: Intervening along a single direction lacks precision — it may simultaneously disrupt both the detection capability and the execution capability, resulting in incoherent outputs or incomplete safety bypass.

**Key Insight**: The authors hypothesize that the safety mechanism is composed of two functionally independent directions — a **harm detection direction** (by which the model identifies whether a request is harmful) and a **refusal execution direction** (by which the model carries out the refusal response) — each requiring a distinct intervention strategy.

**Core Idea**: These two directions are extracted via SVD combined with classifier-guided sparsification. The refusal execution direction is then suppressed via adaptive projection elimination (removing the capacity to execute refusals), while the harm detection direction is suppressed via direct steering (suppressing the perception of harm). The two steps are applied jointly to achieve precise jailbreaking.

## Method

### Overall Architecture
DBDI consists of two phases: offline calibration and online inference.
- **Offline**: The refusal execution vector $\vec{v}_{refusal}$ and harm detection vector $\vec{v}_{harm}$ are extracted from contrastive prompt pairs, and the optimal intervention layer $l^*$ is selected based on classifier accuracy.
- **Online**: During inference, the hidden state at layer $l^*$ is modified in two steps to elicit non-refusing responses from the model.

### Key Designs

1. **Dual-Direction Vector Extraction (SVD + Classifier-Guided Sparsification)**

    - **Function**: Separately extract the refusal execution direction and the harm detection direction.
    - **Mechanism**: (a) The **refusal execution vector** is derived from minimally different twin prompt pairs (semantically near-identical, one harmful and one benign), with the first principal direction obtained via SVD. (b) The **harm detection vector** is derived from harmful vs. neutral prompts, similarly via SVD.
    - **Key Details**: Raw SVD directions are noisy; a linear classifier is trained to identify the most discriminative neurons, retaining only the top $k\%$ of important dimensions (sparsification), substantially improving directional purity.
    - **Design Motivation**: The two directions employ different contrastive prompt sets to ensure functionally distinct signals are captured — twin prompts with minimal differences capture "refuse vs. not refuse," while harmful vs. neutral prompts capture "harmful vs. benign."

2. **Differentiated Bidirectional Intervention (at Inference Time)**

    - **Function**: Modify hidden states at the key layer to bypass safety alignment.
    - **Two-step strategy with asymmetric treatment**:
        - **Step 1 — Adaptive Projection Elimination**: $h^{(1)} = h - \alpha \cdot \text{proj}_{\vec{v}_{refusal}}(h)$, which removes the component of the hidden state along the refusal execution direction. The intervention is adaptive — the magnitude depends on the projection of the current hidden state onto that direction.
        - **Step 2 — Direct Steering**: $h' = h^{(1)} - \beta \cdot \vec{v}_{harm}$, which shifts the hidden state by a fixed amount along the harm detection direction to suppress the model's perception of harm.
    - **Full formula**: $h' = h - \alpha \cdot \text{proj}_{\vec{v}_{refusal}}(h) - \beta \cdot \vec{v}_{harm}$
    - **Design Motivation**: The two directions require different intervention strategies — refusal execution is an "action," best addressed by precise projection elimination; harm detection is a "perception," better suppressed by a fixed offset. Ablations confirm that symmetric treatment leads to substantially degraded performance.

3. **Optimal Layer Selection**

    - **Function**: 5-fold cross-validation accuracy of a linear classifier is used to select the layer at which the refusal execution direction is most salient.
    - **Design Motivation**: Intervening at a single optimal layer incurs minimal computational overhead (only one forward hook).

## Key Experimental Results

### Main Results

| Model | Method | AdvBench ASR | HarmBench ASR |
|-------|--------|-------------|--------------|
| Llama-2 7B | Directional Ablation | ~60–70% (best) | — |
| Llama-2 7B | **DBDI** | **95.96% (97.88%)** | **92% (95%)** |
| Qwen 7B | **DBDI** | **83.4%** | — |
| Llama-3.2 3B | **DBDI** | **87.5%** | — |

Values in parentheses correspond to results with a simplified chat template.

### Ablation Study (AdvBench / HarmBench)

| Variant | AdvBench ASR | HarmBench ASR |
|---------|-------------|--------------|
| Full DBDI | 95.96% | 92% |
| Symmetric projection (both directions via projection elimination) | 62.88% | 86% |
| Symmetric steering (both directions via direct steering) | 9.42% | 16% |
| Refusal execution direction only | 1.34% | 73% |
| Harm detection direction only | 11.34% | 35% |

### Key Findings
- **The two directions are functionally distinct**: Intervening on only one direction is far less effective than combining both (refusal-only: 1.34% on AdvBench; harm-only: 11.34%), validating the dual-direction hypothesis.
- **Asymmetric intervention strategy is critical**: Symmetric projection yields 62.88%, symmetric steering only 9.42%, whereas DBDI's differentiated strategy achieves 95.96% — confirming that the two directions require different treatments.
- **Simplified chat templates improve performance**: On Llama-2, ASR increases from 95.96% to 97.88%, suggesting that safety-related tokens in complex templates provide additional protection.
- **Negligible computational overhead**: Only a single forward hook is added at one layer, introducing virtually no inference latency.

## Highlights & Insights
- The decomposition of the "refusal direction" into two functionally independent directions — detection and execution — offers genuine mechanistic insight. Beyond its utility for attacks, this framework provides a novel lens for understanding the internal mechanics of LLM safety alignment. Defenders who adopt this dual-direction perspective may be able to design more robust alignment strategies.
- The ablation study is particularly convincing: by systematically varying symmetric vs. asymmetric treatment, single vs. dual directions, and projection vs. steering, the paper rigorously justifies each design decision.

## Limitations & Future Work
- The method relies entirely on white-box access — requiring full model weights and activations — and is inapplicable to closed-source API-based models.
- Extracting direction vectors requires constructing contrastive prompt datasets; the quality and diversity of these datasets directly affect the resulting directions.
- The two hyperparameters $\alpha$ and $\beta$ require grid search on a validation set, with no adaptive selection procedure proposed.
- As an attack-oriented study, the paper does not discuss potential defenses against DBDI (e.g., whether hidden state modifications can be detected at inference time).
- Experiments are limited to models in the 7B–8B parameter range; whether larger models exhibit the same dual-direction safety structure remains unverified.
- Sensitivity of results to the percentile threshold $k$ in classifier-guided sparsification is not thoroughly analyzed.
- The orthogonality assumption between the two direction vectors is not empirically verified — significant correlation between the harm detection and refusal execution directions could introduce interference when they are handled independently.
- No quantitative evaluation of output quality (coherence, perplexity) after intervention is provided.

## Related Work & Insights
- **vs. Directional Ablation (Arditi et al.)**: The prior work models safety as a single refusal direction and ablates it; DBDI decomposes it into two functional directions with separate interventions, improving AdvBench ASR from ~60–70% to 95.96%.
- **vs. GCG/AutoDAN**: These methods search for adversarial suffixes at the prompt level and are susceptible to input-level filtering; DBDI operates directly on activations, bypassing all input-level defenses.
- **vs. Fine-tuning-based attacks**: Fine-tuning permanently modifies model weights, is detectable via weight inspection, and may degrade general capabilities; DBDI temporarily modifies activations only during inference, leaving no persistent trace and being fully reversible.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — The insight of decomposing the safety direction into functionally independent detection and execution components is highly original and offers a new analytical framework for understanding LLM safety mechanisms.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — The ablation study is systematically designed and multi-model validation is provided, though testing on models with 70B+ parameters is absent.
- **Writing Quality**: ⭐⭐⭐⭐ — Mathematical formalization is clear and the motivation for each design choice is consistently reinforced throughout.
- **Value**: ⭐⭐⭐⭐⭐ — The work offers insights for both attackers and defenders: attackers gain a more precise tool, while defenders gain a finer-grained understanding of safety internals.

## Additional Notes
- This work reveals that safety alignment is not unidimensional — harm detection and refusal execution can be independently manipulated — suggesting that future safety training should consider multi-directional robustness.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] SafeNlidb: A Privacy-Preserving Safety Alignment Framework for LLM-based Natural Language Database Interfaces](safenlidb_a_privacy-preserving_safety_alignment_framework_for_llm-based_natural_.md)
- [\[AAAI 2026\] EASE: Practical and Efficient Safety Alignment for Small Language Models](ease_practical_and_efficient_safety_alignment_for_small_language_models.md)
- [\[NeurIPS 2025\] LLM Safety Alignment is Divergence Estimation in Disguise](../../NeurIPS2025/llm_alignment/llm_safety_alignment_is_divergence_estimation_in_disguise.md)
- [\[ICLR 2026\] Superficial Safety Alignment Hypothesis](../../ICLR2026/llm_alignment/superficial_safety_alignment_hypothesis.md)
- [\[AAAI 2026\] AlignTree: Efficient Defense Against LLM Jailbreak Attacks](aligntree_efficient_defense_against_llm_jailbreak_attacks.md)

<!-- RELATED:END -->
