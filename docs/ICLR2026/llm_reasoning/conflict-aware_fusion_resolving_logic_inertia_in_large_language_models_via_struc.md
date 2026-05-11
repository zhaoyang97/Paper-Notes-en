---
title: >-
  [Paper Note] Conflict-Aware Fusion: Resolving Logic Inertia in Large Language Models via Structured Cognitive Priors
description: >-
  [ICLR 2026][LLM Reasoning][logic inertia] This paper identifies the phenomenon of "logic inertia" in LLMs—whereby models continue along learned reasoning trajectories even when presented with contradictory premises…
tags:
  - "ICLR 2026"
  - "LLM Reasoning"
  - "logic inertia"
  - "contradiction detection"
  - "dual-process reasoning"
  - "structural robustness"
  - "rule-based reasoning"
date: 2026-05-08
content_hash: f80b3531b1fc3480
---

# Conflict-Aware Fusion: Resolving Logic Inertia in Large Language Models via Structured Cognitive Priors

**Conference**: ICLR 2026
**arXiv**: [2512.06393](https://arxiv.org/abs/2512.06393)
**Code**: [https://github.com/14H034160212/lemo](https://github.com/14H034160212/lemo)
**Area**: LLM Reasoning
**Keywords**: logic inertia, contradiction detection, dual-process reasoning, structural robustness, rule-based reasoning

## TL;DR
This paper identifies the phenomenon of "logic inertia" in LLMs—whereby models continue along learned reasoning trajectories even when presented with contradictory premises, reducing accuracy to 0.0—and proposes the Conflict-Aware Fusion dual-process architecture, which enforces premise verification prior to reasoning execution, achieving 100% accuracy on contradiction detection.

## Background & Motivation

**Background**: LLMs perform well on multi-step logical reasoning benchmarks (benchmark accuracy of 1.0), yet these benchmarks typically evaluate reasoning under normal conditions without examining robustness when the rule system is perturbed.

**Limitations of Prior Work**: Existing evaluations conflate language competence with logical robustness. There is also a lack of diagnostic frameworks capable of isolating the individual effects of missing rules, redundant rules, and contradictory premises within a unified setting.

**Key Challenge**: Whether LLMs perform genuine logical reasoning or merely simulate it via pattern matching. When the structural integrity of a rule system is compromised—particularly through contradiction injection—the answer is the latter: all tested models collapse to 0.0 accuracy under contradictory conditions.

**Goal**: (a) Establish a systematic structural robustness evaluation framework; (b) identify and formalize the phenomenon of "logic inertia"; (c) design a reasoning framework that eliminates logic inertia.

**Key Insight**: The Cognitive Structure Hypothesis—reliable multi-step reasoning requires an explicit structural separation between premise verification and deductive execution, an inductive bias entirely absent from current end-to-end training paradigms.

**Core Idea**: Enforce a "verify-before-reason" structural constraint during inference—first applying System 2 to detect contradictions, then System 1 to execute reasoning, with a halt triggered upon contradiction detection.

## Method

### Overall Architecture
Three components: (1) a structural robustness benchmark featuring four controlled perturbation stress tests; (2) a dual-process reasoning architecture enforcing verification-prior-to-reasoning Chain-of-Thought structure; and (3) a two-stage optimization pipeline comprising structured SFT followed by DPO alignment.

### Key Designs

1. **Structural Robustness Benchmark (4 Stress Tests)**:

    - Variant 1: Redundant rule removal — the conclusion remains unchanged; tests the model's tolerance for redundant information.
    - Variant 2: Critical rule removal — the reasoning chain is broken; tests the model's ability to detect insufficient evidence.
    - Variant 3: Contradiction injection — evidence contradicting existing facts is introduced; tests contradiction detection.
    - Variant 4: Multiple equivalence law stacking — rules are rewritten via logically equivalent transformations; tests invariance to surface-level form.
    - **Design Motivation**: Each variant controls a single structural property derived from the same standard rule system, ensuring that performance differences are attributable to reasoning robustness rather than domain shift.

2. **Dual-Process Reasoning Architecture**:

    - **Function**: Enforces a two-stage structure within the CoT generation pathway.
    - Step 1 (System 2): Premise verification — checks premise completeness and consistency, and detects contradictions.
    - Step 2 (System 1): Conditional execution — deductive reasoning is performed only if Step 1 passes; "Halt Reasoning" is triggered upon contradiction detection.
    - **Design Motivation**: Transforms verification from an optional behavior into a mandatory structural step, breaking the inertia of "reasoning over verification."

3. **Two-Stage Optimization Pipeline**:

    - **Stage 1: Structured SFT** — trains on 11,200 instances (comprising standard, perturbed, and contradictory variants), with all samples mandatorily prefixed with "Step 1: Verify Facts," making premise checking a default procedure.
    - **Stage 2: DPO Logic Alignment** — constructs preference pairs: correct halting at contradictions ≻ continuing unsupported reasoning. This directly penalizes "hallucination shortcuts" and reinforces disciplined termination behavior.
    - **Design Motivation**: SFT establishes the verification structure; DPO reinforces contradiction detection behavior.

### Loss & Training
- SFT: Standard autoregressive loss + LoRA (r=8, α=16), lr=2e-5, 3 epochs.
- DPO: Preference pairs (verify + halt vs. continue reasoning), directly optimizing policy-preference alignment.
- Models: BERT-base, Qwen2-1.5B, TinyLlama-1.1B, all fine-tuned with LoRA.

## Key Experimental Results

### Main Results

| Method | Base Acc | Var 2 (Rule Removal) | Var 3 (Contradiction) |
|--------|----------|----------------------|-----------------------|
| Stage 1 (SFT Baseline) | 0.512 | 0.250 | 0.210 |
| DPO (Direct Alignment) | 0.475 | 0.267 | 0.510 |
| CoT (Standard) | 0.500 | 0.390 | 0.865 |
| Mixed-Aug (Data Augmentation) | 0.525 | 0.405 | 0.972 |
| Fusion-LRA (Conflict-Aware SFT) | 0.988 | 0.753 | 0.705 |
| **Fusion-Conflict (Full)** | **1.000** | **0.735** | **1.000** |

### Ablation Study

| Configuration | Base | Var 3 (Contradiction) | Note |
|---------------|------|-----------------------|------|
| All baselines (no Fusion) | 1.000 | 0.000 | Logic inertia: complete collapse under contradiction |
| + SFT pre-training | ~0.53 | 0.000 | SFT alone cannot resolve contradictions |
| + CoT | 0.500 | 0.865 | CoT helps but is insufficient |
| + Fusion-LRA (SFT only) | 0.988 | 0.705 | Verification structure yields significant improvement |
| **+ Fusion-Conflict (SFT+DPO)** | **1.000** | **1.000** | Logic inertia fully eliminated |

### Key Findings
- **Universality of logic inertia**: BERT, Qwen2, and TinyLlama all achieve 0.0 accuracy under contradictory conditions—this is not a model-specific failure but a structural deficiency of the current LLM training paradigm.
- **Robustness asymmetry**: Models are highly stable under semantics-preserving transformations (Variant 4) yet collapse entirely under contradictions, indicating that models can recognize logical equivalence but cannot detect logical contradiction.
- **External validation via Human Last Exam**: All top-tier models, including GPT-4-level systems, also fail on constructed contradictory cases, confirming this as a general problem.
- **Synergistic effect of verification structure and DPO**: Neither SFT alone (0.705) nor DPO alone (0.510) suffices for contradiction detection; only their combination achieves 1.000.

## Highlights & Insights
- **Formalization of logic inertia**: This work is the first to name and formalize this failure mode—LLMs prioritize completing the reasoning chain over verifying reasoning premises. This insight has far-reaching implications for AI safety.
- **Precision of the dual-process architecture**: The design placing System 2 before System 1 maps directly onto dual-process theory from cognitive science, translating a psychological concept into an engineerable AI architecture.
- **Verification as a structural constraint**: Rather than training models to "learn to verify," the approach makes models "required to verify"—internalizing verification as a mandatory precondition for reasoning through prompt structure and DPO reinforcement.

## Limitations & Future Work
- Evaluation is conducted solely on controlled rule systems of limited scale (100 base instances), without validation on large-scale natural language reasoning benchmarks.
- Model scale is small (largest: Qwen2-1.5B); whether logic inertia persists in larger models remains to be verified.
- The dual-process architecture relies on prompt structure design; generalization to diverse reasoning tasks (mathematics, code, etc.) is unclear.
- Accuracy on Variant 2 (critical rule removal) remains only 0.735, indicating that insufficient-evidence scenarios are not yet fully resolved.

## Related Work & Insights
- **vs. Standard CoT**: CoT encourages models to "think more" but does not guarantee "verify before thinking"; it reaches only 0.865 under contradictory conditions.
- **vs. ChatLogic (external symbolic engine)**: ChatLogic relies on an external Prolog engine for reasoning verification, whereas this work internalizes verification into the model's own reasoning process.
- **vs. Reversal Curse research**: The Reversal Curse reveals the absence of bidirectional reasoning; Logic Inertia reveals the absence of premise verification—both serve as evidence that LLMs do not perform genuine reasoning.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The proposal and formalization of the "logic inertia" concept carries significant value.
- **Experimental Thoroughness**: ⭐⭐⭐ — The evaluation scale is limited (100 instances), model sizes are small, and large-scale validation is lacking.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem definition is clear, and the motivation for the dual-process architecture is well-reasoned.
- **Value**: ⭐⭐⭐⭐ — Exposes a fundamental deficiency in LLM reasoning with important implications for AI safety and reliable inference.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Chain-of-Thought as a Lens: Evaluating Structured Reasoning Alignment between Human Preferences and Large Language Models](../../ACL2026/llm_reasoning/chain-of-thought_as_a_lens_evaluating_structured_reasoning_alignment_between_hum.md)
- [\[ICLR 2026\] AgentMath: Empowering Mathematical Reasoning for Large Language Models via Tool-Augmented Agent](agentmath_empowering_mathematical_reasoning_for_large_language_models_via_tool-a.md)
- [\[ICLR 2026\] DESIGNER: Design-Logic-Guided Multidisciplinary Data Synthesis for LLM Reasoning](designer_design-logic-guided_multidisciplinary_data_synthesis_for_llm_reasoning.md)
- [\[ICLR 2026\] Efficient Test-Time Scaling for Small Vision-Language Models](efficient_test-time_scaling_for_small_vision-language_models.md)
- [\[ICLR 2026\] SealQA: Raising the Bar for Reasoning in Search-Augmented Language Models](sealqa_raising_the_bar_for_reasoning_in_search-augmented_language_models.md)

</div>

<!-- RELATED:END -->
