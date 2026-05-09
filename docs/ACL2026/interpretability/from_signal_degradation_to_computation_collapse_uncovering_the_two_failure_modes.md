---
title: >-
  [Paper Note] From Signal Degradation to Computation Collapse: Uncovering the Two Failure Modes of LLM Quantization
description: >-
  [ACL 2026][Post-training quantization] Through systematic mechanistic interpretability analysis, this paper reveals that LLM quantization exhibits two qualitatively distinct failure modes: 4-bit *Signal Degradation* (computational patterns remain intact but precision is impaired, amenable to local repair) and 2-bit *Computation Collapse* (functional destruction of critical components, requiring structural reconstruction).
tags:
  - ACL 2026
  - Post-training quantization
  - signal degradation
  - computation collapse
  - mechanistic interpretability
  - causal tracing
  - knowledge recall
  - PTQ
date: 2026-05-08
content_hash: 0d47d8d598f9b8e9
---

# From Signal Degradation to Computation Collapse: Uncovering the Two Failure Modes of LLM Quantization

**Conference**: ACL 2026
**arXiv**: [2604.19884](https://arxiv.org/abs/2604.19884)
**Code**: None
**Area**: Model Quantization / Interpretability
**Keywords**: Post-training quantization, signal degradation, computation collapse, mechanistic interpretability, causal tracing, knowledge recall, PTQ

## TL;DR

Through systematic mechanistic interpretability analysis, this paper reveals that LLM quantization exhibits two qualitatively distinct failure modes: 4-bit *Signal Degradation* (computational patterns remain intact but precision is impaired, amenable to local repair) and 2-bit *Computation Collapse* (functional destruction of critical components, requiring structural reconstruction).

## Background & Motivation

**State of the Field**: Post-training quantization (PTQ) is a key technique for efficient LLM deployment. 4-bit quantization is widely regarded as the optimal balance between accuracy and compression, while 2-bit quantization typically triggers a catastrophic "performance cliff"—accuracy plummeting to near zero.

**Limitations of Prior Work**: Existing research concentrates on three directions: (1) macro-level evaluation (measuring the degree of performance degradation); (2) algorithmic improvements (outlier suppression, rotation matrices, and other numerical optimizations); and (3) preliminary mechanistic exploration (layer/component sensitivity analysis). All three share the limitation of treating quantization damage as a "numerical problem" without probing why internal model mechanisms fail.

**Root Cause**: Is the catastrophic failure at 2-bit a quantitative accumulation of 4-bit degradation, or does it represent a qualitative transition? If qualitative, it implies that all current numerically-oriented repair strategies are fundamentally misguided for 2-bit quantization.

**Paper Goals**: To reveal intrinsic mechanistic differences underlying quantization failures through systematic mechanistic interpretability analysis (layer-wise information flow, causal pathways, component functionality, and representation space), and to validate that different failure modes correspond to different repair strategies.

**Starting Point**: The authors draw an analogy to signal processing—is the signal weakened by noise (degradation), or is the computation pipeline itself broken (collapse)?

**Core Idea**: The failures of 4-bit and 2-bit quantization differ not in degree but in kind. Signal degradation can be recovered through targeted training-free repair, whereas computation collapse requires structural reconstruction (e.g., fine-tuning)—a distinction that constitutes the strongest evidence for differentiating the two modes.

## Method

**Overall Architecture**: Llama-3.1-8B serves as the primary subject; FP16/4-bit/2-bit internal behaviors are systematically compared on a factual knowledge recall task (Pararel). Four layers of analysis establish and validate the hypotheses: macro phenomena → layer-wise probing → causal analysis → component/representation verification → mechanism-guided intervention.

**Key Designs**:

1. **Multi-level Knowledge Signal Tracing**
   - **Function**: Track the existence and causal transmission integrity of knowledge signals within the model.
   - **Mechanism**: Logit Lens is used to project hidden states layer by layer into the vocabulary space, observing changes in the probability/rank of the correct token. Under 4-bit, the signal appears in middle-to-late layers but with reduced strength (degradation); under 2-bit, the signal remains near zero throughout (absence). Cross-model causal activation patching further validates this: injecting FP16 "clean" activations into critical positions (the last subject token) of the quantized model restores performance at 4-bit but produces no response at 2-bit.
   - **Design Motivation**: Distinguishing "signal weakened" from "signal never generated" is the core evidence for establishing the two-mode hypothesis.

2. **Component-level Functional Diagnosis (Attention + FFN Key-Value Memory)**
   - **Function**: Localize which specific components fail and characterize their failure modes.
   - **Mechanism**: For attention, normalized entropy (global concentration) and JSD divergence (focal deviation) are used. For FFN, gate sign flip rate (SFR; >30% indicates severe instability), Top-1% activated neuron Jaccard overlap (≈0.1 indicates complete activation misalignment), and output cosine similarity (≈0 indicates complete semantic direction deviation) are employed. Under 2-bit, all metrics indicate functional component collapse.
   - **Design Motivation**: Attributing macro-level "signal absence" to specific component failures confirms whether the issue is precision loss or functional breakdown.

3. **Mechanism-aware Two-stage Repair vs. System Irreversibility Validation**
   - **Function**: Validate that the two failure modes exhibit fundamentally different repairability.
   - **Mechanism**: For 4-bit, a "source protection + signal recovery" strategy is designed: protecting the first few layers (Llama/Mistral use 8-bit for the first 2 layers, ~4.25 avg bits; Qwen/Gemma use kurtosis-based selection, ~4.1 avg bits) plus peak signal amplification ($\alpha$-fold logit scaling). The same strategy and EORA low-rank compensation both prove ineffective at 2-bit. A "domino experiment" shows that quantizing only the first 2 layers causes accuracy to collapse from 100% to 41.65%.
   - **Design Motivation**: The difference in repairability is the most direct and compelling practical evidence distinguishing the two failure modes.

## Key Experimental Results

**4-bit Repair Experiments (Accuracy on Failure Subset)**:

| Model | Baseline (4-bit) | +Basic Repair | +Signal Amplification (Final) |
|---|---|---|---|
| Llama3.1-8B | 0.00% | 67.91% | 75.19% ($\alpha$=3) |
| Mistral-7B | 0.00% | 66.86% | 81.26% ($\alpha$=9) |
| Qwen3-8B | 0.00% | 40.24% | 79.88% ($\alpha$=7) |
| Gemma2-9B | 0.00% | 33.85% | 64.08% ($\alpha$=2) |

**2-bit "Domino Effect" (Llama3.1-8B)**:

| Quantized Layers | Robust Subset | Failure Subset |
|---|---|---|
| None (FP16) | 100.00% | 100.00% |
| Layer 0 | 65.47% | 15.03% |
| Layers 0–1 | 41.65% | 5.29% |
| Layers 0–5 | 2.51% | 0.38% |

**Representation Space Structure Analysis**:
- 4-bit: CKA maintains a clear diagonal structure; activation subspace similarity to FP16 >0.8
- 2-bit: CKA is nearly entirely dark (structural collapse); activation subspace similarity ≈0
- 4-bit error subspace alignment with signal ≈0.3 (resembles random noise)
- 2-bit error subspace alignment with signal ≈0.8 (directly interferes with principal features)

**Key Findings**:
- At 4-bit, the correct answer rank decreases (correct answer remains in Top-5); at 2-bit, the rank collapses (drops to thousands, equivalent to random guessing).
- Architecture-dependent degradation patterns: Llama/Mistral exhibit "early-layer representation bottleneck," while Qwen/Gemma exhibit "uniform degradation."
- 2-bit models fail to correctly process high-precision signal inputs—the components themselves have ceased to function.
- The distinction between the two failure modes is consistent across both GPTQ and AWQ quantization methods.

## Highlights & Insights

- **Framework Value of Qualitative Distinction**: This work is the first to systematically demonstrate that 4-bit and 2-bit failures are not different degrees along the same continuum, but two fundamentally distinct failure modes.
- **Complete Closed Loop from Diagnosis to Repair**: Mechanistic analysis directly guides repair strategy design, and the differential effectiveness of repairs reciprocally validates the diagnosis.
- **Compelling "Domino Experiment"**: Quantizing only the first 2 layers at 2-bit causes catastrophic collapse, and 30 subsequent FP16 layers cannot recover performance—vividly demonstrating the irreversibility of computation collapse.
- **Insightful Error Direction Analysis**: The high alignment of 2-bit quantization error with the signal subspace implies that the noise is not random but systematically destroys the model's core features.

## Limitations & Future Work

- The study focuses on weight-only quantization; failure modes of activation quantization remain to be investigated.
- Evaluation is anchored to factual recall tasks; performance on complex reasoning tasks warrants further verification.
- The repair strategies incur additional precision overhead (~4.1–4.25 avg bits), and practical efficiency requires further optimization.
- The boundary between the two modes (3-bit behavior) deserves deeper investigation.
- The failure mode demarcation point may differ across model architectures.

## Related Work & Insights

- **GPTQ (Frantar et al., 2023)**: The most widely used weight-only PTQ method and the primary quantization baseline in this work.
- **Causal Tracing (Meng et al., 2022)**: A knowledge localization method extended here into cross-model repair experiments.
- **Logit Lens (nostalgebraist, 2020)**: An intermediate-layer decoding tool.
- **SpQR (Dettmers et al., 2023)**: A mixed-precision method that resonates with the source protection strategy proposed in this work.
- **Insights**: Quantization research should move beyond numerical optimization; mechanistic understanding is essential for breaking through performance bottlenecks. Practical 2-bit quantization requires a paradigm shift from "compensation" to "reconstruction."

## Rating

- **Novelty**: ★★★★★ — The systematic distinction and validation of two failure modes constitutes a novel and significant contribution.
- **Experimental Thoroughness**: ★★★★★ — Four models, multi-level analysis, and multi-metric validation yield a complete evidence chain.
- **Writing Quality**: ★★★★★ — The narrative progresses clearly from phenomena → hypotheses → validation → intervention.
- **Value**: ★★★★☆ — Provides an important diagnostic framework and mechanistic insights for quantization research.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] One Language, Two Scripts: Probing Script-Invariance in LLM Concept Representations](../../ICLR2026/interpretability/one_language_two_scripts_probing_script-invariance_in_llm_concept_representation.md)
- [\[ICLR 2026\] Uni-NTFM: A Unified Foundation Model for EEG Signal Representation Learning](../../ICLR2026/interpretability/uni-ntfm_a_unified_foundation_model_for_eeg_signal_representation_learning.md)
- [\[ACL 2026\] Rhetorical Questions in LLM Representations: A Linear Probing Study](rhetorical_questions_in_llm_representations_a_linear_probing_study.md)
- [\[ACL 2026\] Style over Story: Measuring LLM Narrative Preferences via Structured Selection](style_over_story_measuring_llm_narrative_preferences_via_structured_selection.md)
- [\[ACL 2026\] Curing "Miracle Steps" in LLM Mathematical Reasoning with Rubric Rewards](curing_miracle_steps_in_llm_mathematical_reasoning_with_rubric_rewards.md)

<!-- RELATED:END -->
