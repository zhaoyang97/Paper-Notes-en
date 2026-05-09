---
title: >-
  [Paper Note] Superficial Safety Alignment Hypothesis
description: >-
  [ICLR 2026][LLM Alignment][safety alignment] This paper proposes the **Superficial Safety Alignment Hypothesis (SSAH)**: safety alignment is essentially teaching a model to perform an implicit binary classification task (execute vs. refuse), requiring only ~1.3% of neurons to establish safety guardrails. Freezing these safety-critical units during fine-tuning preserves safety, and leveraging redundant units as an "alignment budget" eliminates the alignment tax.
tags:
  - ICLR 2026
  - LLM Alignment
  - safety alignment
  - alignment fragility
  - neuron-level analysis
  - alignment tax
  - model pruning
date: 2026-05-08
content_hash: 452354c3b4aef8a4
---

# Superficial Safety Alignment Hypothesis

**Conference**: ICLR 2026
**arXiv**: [2410.10862](https://arxiv.org/abs/2410.10862)
**Code**: [https://ssa-h.github.io/](https://ssa-h.github.io/)
**Area**: AI Safety / LLM Alignment
**Keywords**: safety alignment, alignment fragility, neuron-level analysis, alignment tax, model pruning

## TL;DR
This paper proposes the **Superficial Safety Alignment Hypothesis (SSAH)**: safety alignment is essentially teaching a model to perform an implicit binary classification task (execute vs. refuse), requiring only ~1.3% of neurons to establish safety guardrails. Freezing these safety-critical units during fine-tuning preserves safety, and leveraging redundant units as an "alignment budget" eliminates the alignment tax.

## Background & Motivation

**State of the Field**: LLM safety alignment primarily relies on SFT, RLHF, and DPO, but these methods typically treat safety alignment as a subset of general alignment, overlooking its unique properties.

**Limitations of Prior Work**:
- Safety mechanisms are extremely fragile — even fine-tuning on benign data can collapse safety guardrails (Qi et al., 2023).
- An "alignment tax" exists — improving safety sacrifices general model capability.
- Current approaches require full-parameter fine-tuning, incurring high computational costs.

**Root Cause**: There is insufficient understanding of how safety alignment affects model behavior and why safety mechanisms are so fragile.

**Paper Goals**: Three questions are addressed: How does safety alignment affect model behavior? Why is safety so fragile? How can these issues be mitigated?

**Starting Point**: A key observation — a model capable of fulfilling malicious requests already possesses the relevant knowledge and reasoning ability; thus, safety alignment only needs to teach the model to select the correct reasoning direction (execute vs. refuse), rather than injecting new knowledge.

**Core Idea**: Safety alignment ≈ implicit binary classification task, achievable with a very small fraction (~1.3%) of safety-critical neurons.

## Method

### Overall Architecture
SSAH is not a concrete algorithm but a hypothesis framework about the nature of safety alignment. The overall pipeline proceeds as follows: (1) propose the hypothesis and validate the existence of reasoning directions via probing experiments; (2) identify four types of neurons (SCU/UCU/CU/RU) through structured pruning; (3) derive two practical strategies based on the identified units — freezing safety units to resist fine-tuning attacks, and exploiting redundant units to reduce the alignment tax.

### Key Designs

1. **Superficial Safety Alignment Hypothesis (SSAH)**:

   - **Function**: Reframes safety alignment as an implicit safety-related binary classification task.
   - **Mechanism**: A model capable of executing malicious requests already possesses the relevant knowledge; safety alignment only needs to teach it to select the correct "reasoning direction" — whether to execute or refuse a request. Alignment also provides a standardized refusal mechanism and alternative response templates.
   - **Design Motivation**: Compared to the general Safety Alignment Hypothesis (SAH), SSAH is more specific and verifiable — it focuses on models that already possess the requisite knowledge, eliminating confounding factors arising from knowledge deficits.
   - **Explanation of Jailbreaks**: Current alignment only determines the reasoning direction at the initial token; attackers bypass the safety mechanism by manipulating tokens. Ideal alignment should re-evaluate the reasoning direction at every generation step.

2. **Probing Experiments to Validate Reasoning Directions**:

   - **Function**: Validates that safety alignment indeed shifts the model's reasoning direction by comparing hidden-state distances.
   - **Mechanism**: Three types of queries are constructed — original malicious queries (Clean), malicious queries prepended with a benign token ("Sorry, I can't..."), and malicious queries prepended with a malicious token ("Here's how..."). For an aligned model, the hidden-state distance between Clean and the benign-token variant should be smaller than that between Clean and the malicious-token variant; the reverse holds for unaligned models.
   - **Design Motivation**: Directly observing reasoning direction is infeasible, but it can be inferred indirectly through distance relationships in the hidden-state space.
   - **Key Findings**: Aligned models exhibit a preference for safe reasoning across all Transformer blocks, not merely in later layers.

3. **Identification of Four Types of Computational Units (SCU/UCU/CU/RU)**:

   - **Function**: Classifies model neurons/channels into four types: Safety-Critical Units (SCU), Utility-Critical Units (UCU), Composite Units (CU), and Redundant Units (RU).
   - **Mechanism**: A structured pruning strategy is employed. For each depth-2 module $f(X) = B\sigma(AX)$, an importance score is computed as $\mathbf{I}_{:,j} = \frac{1}{N-1}\sum_{n=1}^{N}(X^B_{n,j,:} - \bar{X}^B_{:,j,:})^2 \cdot \|\mathbf{W}^B_{:,j}\|_2^2$. Scores $\mathbf{I_S}$ and $\mathbf{I_U}$ are computed on safety and utility datasets respectively, and the four unit types are distinguished via their differences and sums.
   - **Design Motivation**: If safety alignment is truly a simple binary classification task, then only a small number of neurons should be required to establish safety guardrails.

4. **Freezing Strategy Against Fine-Tuning Attacks**:

   - **Function**: Freezes safety-critical components (SCU + top CU) during fine-tuning to prevent safety degradation.
   - **Mechanism**: Attribute migration analysis reveals that fine-tuning converts SCUs and CUs into UCUs, causing safety degradation. Freezing these units prevents such attribute migration.
   - **Effect**: Freezing SCU + all CU reduces ASR on AdvBench for LLaMA2 from 11.92% to 2.88%.

5. **Redundant Units as Alignment Budget**:

   - **Function**: Performs alignment fine-tuning exclusively on the redundant units (~20% of parameters) of the pre-trained model.
   - **Mechanism**: Approximately 20% of pre-trained model parameters are redundant; updating only these parameters achieves alignment while avoiding modification of utility-critical units.
   - **Effect**: Updating 20% of parameters achieves comparable alignment quality, while mathematical capability (GSM8K) improves from 9.24 to 13.4 — outperforming full-parameter fine-tuning, which yields 8.8.

### Loss & Training
- During pruning, an activation-variance-based importance score is used to remove channels/neurons in a structured manner.
- During fine-tuning, specific units are frozen; training epochs are doubled to ensure fair comparison (resulting in equivalent or lower final training loss).

## Key Experimental Results

### Main Results: Freezing Safety Units Against Fine-Tuning Attacks

| Model / Setting | AdvBench ASR (keyword) | AdvBench ASR (llama3-guard) | HEx-PHI Score | HEx-PHI Rate |
|---|---|---|---|---|
| LLaMA2 initial | 0.19% | 0.19% | 1.05 | 0.3% |
| LLaMA2 + Dolly fine-tune | 11.92% | 10.58% | 1.95 | 18.78% |
| LLaMA2 + freeze SCU+6%CU | 3.65% | 2.31% | 1.55 | 10.6% |
| LLaMA2 + freeze SCU+all CU | 2.88% | 1.92% | 1.48 | 9.0% |
| LLaMA3 initial | 1.54% | 1.15% | 1.16 | 3.0% |
| LLaMA3 + Dolly fine-tune | 61.15% | 50.58% | 2.95 | 37.2% |
| LLaMA3 + freeze SCU+all CU | 40.58% | 28.27% | 2.32 | 23.6% |

### Ablation Study: Impact of Pruning Each Unit Type

| Unit Type | Proportion | Utility Drop (LLaMA2) | Safety ASR Increase (LLaMA2) |
|---|---|---|---|
| SCU | 1.3% | −1.3% | +56.0% |
| UCU | 13.3% | −15.6% | +18.3% |
| RU | 14.8% | −2.8% | +4.6% |
| Dense (full model) | 100% | baseline | baseline (10.0%) |

### Key Findings
- **SCUs are extremely sparse yet critical**: Only 1.3% of neurons are responsible for safety; removing them causes ASR to surge from 10% to 66%.
- **LLaMA3 is more fragile than LLaMA2**: After fine-tuning, ASR spikes from 1.54% to 61.15%, possibly because LLaMA3 "analyzes" the true intent of malicious requests.
- **PEFT methods degrade safety more severely than full-parameter fine-tuning**: LoRA causes a 26.9% harmful rate vs. 18.48% for full-parameter fine-tuning — counter-intuitively.
- **Aligning on redundant units improves rather than degrades mathematical ability**: GSM8K improves from 9.24 → 13.4 (20% parameter fine-tuning) vs. 9.24 → 8.8 (full-parameter fine-tuning).

## Highlights & Insights
- **Safety ≈ binary classification — a distinctive insight**: Reducing safety alignment to a binary reasoning-direction selection is both parsimonious and highly explanatory. It elegantly explains why safety is so fragile — only a small number of neurons' "voting directions" need to be flipped to compromise safety.
- **Redundant units as alignment budget**: Pre-trained models naturally contain ~20% redundant parameters; using these for alignment avoids modifying utility-critical units. This idea is transferable to any scenario requiring new capabilities to be added without degrading existing ones.
- **Attribute migration analysis framework**: Tracking per-neuron attribute changes before and after fine-tuning yields a migration map (SCU → CU → UCU), providing a visualization tool for understanding how fine-tuning undermines alignment.
- **Probing method for internal reasoning direction**: Inferring reasoning direction by comparing hidden-state distances between Query+benign/malicious tokens is a simple yet effective approach.

## Limitations & Future Work
- **SSAH provides necessary but not sufficient evidence**: The authors acknowledge that the probing experiments are necessary but not sufficient; safety alignment may involve subtler changes not captured by SSAH.
- **Per-step re-evaluation remains unrealized**: The paper proposes that ideal safety alignment should re-select the reasoning direction at every generation step, but this incurs additional inference overhead.
- **LLaMA3 experiments are constrained**: Due to computational limitations, only the first 12 blocks are frozen, yielding weaker results than LLaMA2.
- **Only the SFT scenario is validated**: The behavior of SCUs/RUs under RLHF/DPO alignment is not explored.
- **Future directions**: SCU/RU identification could be combined with LoRA to design "safety-aware LoRA" — inserting LoRA adapters exclusively on RUs.

## Related Work & Insights
- **vs. Wei et al. (2024)**: They also study safety-critical components but identify them at the weight level; this work operates at the neuron level with finer granularity, and experiments more thoroughly validate the effectiveness of the freezing strategy.
- **vs. SafeDPO**: SafeDPO constrains safety through training objectives, whereas this work takes a model-structure perspective; the two are complementary — SSAH-identified safety units can be combined with SafeDPO training objectives.
- **vs. AlphaSteer**: AlphaSteer achieves refusal steering via null-space constraints; both AlphaSteer and SSAH's freezing strategy aim to protect safety parameters from modification, but approach the problem from different angles.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The SSAH perspective is distinctive; the insight of reducing safety to binary classification is highly perceptive.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multiple models, benchmarks, and evaluation methods are employed, but LLaMA3 experiments are incomplete due to computational constraints.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Logic is clear; the narrative progresses systematically from hypothesis → validation → application.
- **Value**: ⭐⭐⭐⭐⭐ Provides a theoretical foundation for understanding the nature of safety alignment and designing efficient safety training strategies.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Mitigating the Safety Alignment Tax with Null-Space Constrained Policy Optimization](mitigating_the_safety_alignment_tax_with_null-space_constrained_policy_optimizat.md)
- [\[NeurIPS 2025\] Towards Understanding Safety Alignment: A Mechanistic Perspective from Safety Neurons](../../NeurIPS2025/llm_alignment/towards_understanding_safety_alignment_a_mechanistic_perspective_from_safety_neu.md)
- [\[ICLR 2026\] A2D: Any-Order, Any-Step Safety Alignment for Diffusion Language Models](a2d_any-order_any-step_safety_alignment_for_diffusion_language_models.md)
- [\[ICLR 2026\] GuardAlign: Test-time Safety Alignment in Multimodal Large Language Models](guardalign_test-time_safety_alignment_in_multimodal_large_language_models.md)
- [\[ICLR 2026\] Reasoned Safety Alignment: Ensuring Jailbreak Defense via Answer-Then-Check](reasoned_safety_alignment_ensuring_jailbreak_defense_via_answer-then-check.md)

<!-- RELATED:END -->
