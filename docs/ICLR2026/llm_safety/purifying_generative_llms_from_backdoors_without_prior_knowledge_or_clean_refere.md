---
title: >-
  [Paper Note] Purifying Generative LLMs from Backdoors without Prior Knowledge or Clean Reference
description: >-
  [ICLR 2026][LLM Safety][LLM backdoor] A backdoor purification method for LLMs that requires neither prior knowledge nor a clean reference model. Mechanistic analysis reveals that backdoor associations are redundantly dis…
tags:
  - "ICLR 2026"
  - "LLM Safety"
  - "LLM backdoor"
  - "backdoor purification"
  - "mechanistic analysis"
  - "MLP encoding"
  - "immunity analogy"
  - "signature extraction"
date: 2026-05-08
content_hash: b7d042c1c6521c1f
---

# Purifying Generative LLMs from Backdoors without Prior Knowledge or Clean Reference

**Conference**: ICLR 2026
**arXiv**: [2603.13461](https://arxiv.org/abs/2603.13461)
**Code**: [https://bd-vax.github.io/](https://bd-vax.github.io/)
**Area**: AI Safety / Backdoor Defense
**Keywords**: LLM backdoor, backdoor purification, mechanistic analysis, MLP encoding, immunity analogy, signature extraction

## TL;DR
A backdoor purification method for LLMs that requires neither prior knowledge nor a clean reference model. Mechanistic analysis reveals that backdoor associations are redundantly distributed across MLP layers. Inspired by immunology, the method extracts a "signature" from multiple backdoor variants, localizes and suppresses suspicious neurons, and applies lightweight fine-tuning for recovery. Across 5 attacks × 3 tasks, ASR is reduced by 80%+ while utility is preserved.

## Background & Motivation

**Background**: Backdoor attacks pose a serious security threat to LLMs — poisoned models behave normally on clean inputs but produce malicious outputs (sentiment manipulation, targeted refusal, code injection) when a trigger is present.

**Limitations of Prior Work**:
   - Require prior knowledge of the trigger (unrealistic in practice)
   - Require a clean reference model (typically unavailable in deployment scenarios)
   - Rely on aggressive fine-tuning hyperparameters (extremely large learning rates)
   - Mostly limited to classification tasks and cannot handle generative LLMs
   - Vulnerable to adaptive attackers who can obfuscate internal signals

**Key Challenge**: How can backdoors be purified without knowledge of the trigger and without relying on a clean model?

**Goal**: Backdoor purification for LLMs under conditions of no prior knowledge and no clean reference model.

**Key Insight**: Rather than identifying the trigger itself, the method disrupts the trigger–behavior association by precisely localizing how backdoors are encoded in model parameters.

**Core Idea**: Construct multiple backdoor variants → extract a consistent "backdoor signature" across variants → suppress signature neurons + lightweight recovery fine-tuning.

## Method

### Overall Architecture
A three-stage pipeline: (1) **Mechanistic Analysis** — ablation experiments reveal the distributed and redundant nature of backdoor encoding in MLP layers; (2) **Immunity-Analogy Signature Extraction** — construct $N$ backdoor variants, extract backdoor signatures via differential updates and cross-variant consistency; (3) **Purification** — suppress signature neurons and apply lightweight fine-tuning with ~200 clean samples to restore utility.

### Key Designs

1. **Mechanistic Analysis Findings**:

    - **Function**: Systematic ablation experiments reveal how backdoors are encoded in LLMs.
    - **Key Findings**: (a) Removing poisoned attention updates → backdoor persists: attention only amplifies trigger signals but does not encode the association. (b) Removing poisoned MLP updates → backdoor is eliminated: **MLP layers are the carrier of backdoor associations**. (c) At least 12 consecutive MLP blocks must be removed to disable the backdoor. (d) Shuffling the order of poisoned block updates → backdoor remains effective: the association is **redundant and order-agnostic**.
    - **Design Motivation**: This finding overturns the prior assumption that backdoors reside in attention layers or early layers, providing a theoretical basis for precise localization.

2. **Immunity-Analogy Signature Extraction**:

    - **Function**: Starting from the suspicious model, construct multiple backdoor variants and extract parameter change patterns that are consistent across variants.
    - **Mechanism**:
        - Construct $N$ variants: for each variant $i$, fine-tune a poisoned model $\theta_i^{\text{bd}}$ and a clean model $\theta_i^{\text{clean}}$ from the suspicious model using different triggers and behaviors.
        - Compute the differential update $\Delta_i = \Delta\theta_i^{\text{bd}} - \Delta\theta_i^{\text{clean}}$ (cancels generic fine-tuning drift and pre-existing backdoors).
        - Score each neuron $j$: $s_j = \frac{1}{N}\sum_i \|\Delta_{i,j}\|_2 + \lambda \frac{2}{N(N-1)}\sum_{i<\ell} \max\{0, \cos(\Delta_{i,j}, \Delta_{\ell,j})\}$
        - The norm term measures poisoning magnitude; the alignment term measures cross-variant consistency (only positive cosine similarity is counted).
        - Threshold filtering yields the backdoor signature $\mathbb{S} = \{j: s_j \geq \tau\}$.
    - **Design Motivation**: Analogous to the immune system extracting shared antigens from multiple viral variants — parameter change patterns shared across variants with different triggers and behaviors constitute the backdoor association mechanism.

3. **Purification: Neuron Suppression + Lightweight Fine-tuning**:

    - **Function**: Reset signature neurons and restore utility with ~200 clean samples.
    - **Full parameters**: Reinitialize flagged neurons in the `gate_proj`/`up_proj` matrices of the MLP.
    - **LoRA**: Zero out the corresponding rows of the $A$ matrix or columns of the $B$ matrix.
    - **Recovery**: Standard SFT with learning rate 1e-5 for 5 epochs using ~200 clean samples.

### Loss & Training
- Signature extraction requires no training — it is purely analytical.
- The recovery stage uses standard SFT loss with a moderate learning rate (no aggressive large learning rate).
- $N = 6$ variants is the default (experiments confirm diminishing returns beyond $N \geq 5$–$6$).

## Key Experimental Results

### Main Results: ASR Reduction (lower is better)

| Attack / Task | No Defense | Fine-tuning | Pruning | Quantization | CROW | Fine-Pruning | **Ours** |
|---|---|---|---|---|---|---|---|
| LLaMA-7B Sentiment Manipulation (Avg) | 28.16% | 29.96% | 13.78% | 16.36% | 8.66% | 10.94% | **0.91%** |
| LLaMA-13B Sentiment Manipulation (Avg) | 52.37% | 52.18% | 44.11% | 42.93% | 24.47% | 17.87% | **3.49%** |
| LLaMA-7B Targeted Refusal (Avg) | 82.01% | 82.66% | 65.81% | — | — | 65.36% | **10.76%** |
| LLaMA-13B Targeted Refusal (Avg) | 84.75% | 87.82% | 75.16% | — | — | 82.80% | **12.94%** |

### Ablation Study: Effect of Number of Variants $N$

| $N$ | BadNets-7B Refusal Task ASR |
|---|---|
| 1 | 40.91% |
| 3 | ~20–25% |
| 6 | **10.66%** |
| 8+ | Marginal improvement |

### Ablation Study: Scoring Component Contribution

| Method | ASR | Utility |
|---|---|---|
| Norm term only | 10.26% | 58.86% (false positives → utility loss) |
| Alignment term only | 77.04% | 59.88% (high ASR) |
| **Combined (Ours)** | **10.66%** | **59.42%** |

### Key Findings
- **Backdoors are redundantly encoded in MLP layers**: This is the central mechanistic finding; attention layers only amplify signals and do not encode associations.
- **Generalizes across 5 attack types**: Effective against BadNets, CTBA, MTBA, Sleeper, and VPI.
- **Negligible utility loss**: Purified models achieve near-clean-model performance on 10 benchmarks and MT-Bench.
- **~200 samples suffice for recovery**: The lightweight fine-tuning stage has minimal data requirements.
- **Effective for LoRA models**: Full parameter access is not required.

## Highlights & Insights
- **Elegance of the immunity analogy**: Constructing multiple "backdoor variants" is analogous to inoculating with different viral strains; parameter changes that are consistent across variants serve as the "antigen." Differential updates cleverly cancel generic fine-tuning drift and interference from pre-existing backdoors.
- **Mechanistic finding on MLP backdoor encoding**: This overturns prior assumptions about attention layers and early layers, providing a new structural understanding for backdoor defense. The finding is complementary to the neuron-level analysis in SSAH.
- **Fully assumption-free**: No trigger knowledge, no clean reference model, and no aggressive hyperparameters are required — making the method genuinely practical in real-world deployment scenarios.

## Limitations & Future Work
- **Constructing $N$ variants incurs computational overhead** that scales linearly with $N$ (though $N = 6$ is sufficient in practice).
- **Assumes access to a small number of clean samples** (~200) for the recovery stage.
- **For highly stealthy backdoors** (e.g., those encoded entirely within attention layers), the mechanistic analysis may need to be extended.
- **Future direction**: Integration with SSAH's SCU/RU classification — first use SSAH to identify safety-critical units, then apply the proposed method to examine whether these units are backdoor-contaminated.

## Related Work & Insights
- **vs. CROW**: CROW is the previous SOTA backdoor defense, but still exhibits relatively high ASR (8–24%) and greater utility degradation; the proposed method reduces ASR to <11% with superior utility preservation.
- **vs. Fine-Pruning**: Fine-Pruning applies Wanda-based weight pruning, which is limited in effectiveness for generative LLMs (65–82% ASR); the immunity-based signature strategy is substantially more precise.
- **vs. Standard Fine-tuning**: Standard fine-tuning is nearly ineffective against backdoors (ASR unchanged or even increases), demonstrating that backdoor associations are highly stable under conventional training.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Dual innovation: immunity-analogy signature extraction + MLP mechanistic finding.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 5 attacks × 3 tasks × multiple model scales × 5 baselines, with comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Mechanistic analysis is well-organized; method motivation is natural and well-grounded.
- **Value**: ⭐⭐⭐⭐⭐ The first assumption-free backdoor purification method for generative LLMs — practical and highly effective.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Inference-Time Backdoors via Hidden Instructions in LLM Chat Templates](inference-time_backdoors_via_hidden_instructions_in_llm_chat_templates.md)
- [\[AAAI 2026\] Learning from the Undesirable: Robust Adaptation of Language Models without Forgetting](../../AAAI2026/llm_safety/learning_from_the_undesirable_robust_adaptation_of_language_models_without_forge.md)
- [\[ICLR 2026\] Unmasking Backdoors: An Explainable Defense via Gradient-Attention Anomaly Scoring for Pre-trained Language Models](unmasking_backdoors_an_explainable_defense_via_gradient-attention_anomaly_scorin.md)
- [\[NeurIPS 2025\] Unlearning as Ablation: Toward a Falsifiable Benchmark for Generative Scientific Discovery](../../NeurIPS2025/llm_safety/unlearning_as_ablation_toward_a_falsifiable_benchmark_for_generative_scientific_.md)
- [\[NeurIPS 2025\] DeepPersona: A Generative Engine for Scaling Deep Synthetic Personas](../../NeurIPS2025/llm_safety/deeppersona_a_generative_engine_for_scaling_deep_synthetic_personas.md)

</div>

<!-- RELATED:END -->
