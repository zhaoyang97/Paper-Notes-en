---
title: >-
  [Paper Note] SteerEval: How Controllable Are Large Language Models? A Unified Evaluation across Behavioral Granularities
description: >-
  [ACL 2026][LLM/NLP][LLM steering] SteerEval decomposes LLM controllability into L1 (what to express), L2 (how to express), and L3 (specific word instantiation) based on Marr's three levels of analysis. Covering Personali…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "LLM steering"
  - "controllability"
  - "hierarchical benchmark"
  - "activation steering"
  - "Marr's levels"
date: 2026-05-08
content_hash: 8febcb47a288ea81
---

# SteerEval: How Controllable Are Large Language Models? A Unified Evaluation across Behavioral Granularities

**Conference**: ACL 2026  
**arXiv**: [2603.02578](https://arxiv.org/abs/2603.02578)  
**Code**: https://github.com/zjunlp/EasyEdit/blob/main/examples/SteerEval.md  
**Area**: Interpretability / Model Controllability / Benchmark  
**Keywords**: LLM steering, controllability, hierarchical benchmark, activation steering, Marr's levels

## TL;DR
SteerEval decomposes LLM controllability into L1 (what to express), L2 (how to express), and L3 (specific word instantiation) based on Marr's three levels of analysis. Covering Personality, Sentiment, and Language Features domains with 7,560 paired samples, it systematically reveals a critical gap: "existing steering methods generally collapse at fine-grained levels."

## Background & Motivation

**Background**: LLMs are increasingly deployed in socially sensitive scenarios such as education, healthcare, and customer service, making the "controllable steering" of model behaviors (personality, sentiment, style, etc.) essential. Mainstream steering paradigms include: (i) prompt-based (prepending a concept prompt $p_g$ to the input); (ii) activation-based (adding concept vectors during forward propagation), including methods like DiffMean, PCA, RePS, and CAA.

**Limitations of Prior Work**: Existing benchmarks are mostly "flat"—measuring only coarse-grained behaviors (e.g., "friendly" vs. "hostile")—and lack a systematic characterization of **control granularity**. While AXBENCH standardized evaluation processes, its concepts derived from SAE feature descriptions lack behavioral definitions and hierarchical structures, and its evaluation prompts from Alpaca-Eval are not customized for specific concepts.

**Key Challenge**: Real-world control objectives are inherently hierarchical—"expressing autonomy" is a high-level intent, "using a 'self-decisive' tone" is a mid-level strategy, and "including the word 'self-authored'" is a low-level marker. Existing evaluations cannot distinguish whether a method can consistently control from coarse to fine grains.

**Goal**: To construct a cross-domain, cross-granularity hierarchical steering benchmark to fairly compare prompt-based and activation-based methods across different abstraction levels.

**Key Insight**: Drawing on Marr's three levels of analysis (Computational / Algorithmic / Implementational), behavioral control is modeled as a hierarchy of "intent → strategy → verifiable evidence." This is applied across three domains with varying cognitive depths: Personality (high-level dispositional prior), Sentiment (mid-level affective state), and Language Features (low-level surface form).

**Core Idea**: Use an automated data synthesis + manual verification pipeline to build a Marr-inspired three-level hierarchy benchmark, making "the level at which a method begins to lose control" a measurable metric.

## Method

### Overall Architecture
SteerEval consists of two orthogonal axes:
- **Domain Axis**: 3 domains — Personality (high-level dispositional prior) / Sentiment (mid-level affective state) / Language Features (low-level surface form).
- **Granularity Axis**: 3 specification levels — L1 Computational (what to express, high-frequency, high abstraction) / L2 Algorithmic (how to express, mid-frequency, mid abstraction) / L3 Implementational (how to instantiate, low-frequency, low abstraction, machine-verifiable).

Under each (domain, level), there are 8 independent concepts. Each concept provides 105 paired samples (70 train / 30 test / 5 validation), consisting of a matching and a not_matching answer with minimal lexical-level edits to isolate the concept signal. Total samples: $7560 = 3 \times 3 \times 8 \times 105$. Steering evaluation is unified under the EasyEdit2 framework:

$$\hat y_{\text{steered}} = \mathcal{I}_g(M, x)$$

where $\mathcal{I}_g$ can be a prompt prepend $M(p_g \| x)$ or activation injection of a concept vector.

### Key Designs

1. **Marr-inspired three-layer granularity hierarchy**:
    - **Function**: Upgrades steering evaluation from "measuring a single concept" to "measuring if a concept remains stable under three levels of strictness."
    - **Mechanism**: L1 provides only high-level intent (e.g., "increase redundancy"), allowing diverse outputs; L2 adds strategy constraints (e.g., "using rephrased restatement"); L3 requires atomic surface evidence (e.g., "containing '(i.e.,'"), which allows direct string match verification. Moving from L1 to L3, frequency and abstraction decrease while verifiability increases, forming a precisely controlled "task difficulty gradient."
    - **Design Motivation**: Real control goals are naturally layered, but previous evaluations mixed them; separation allows locating which abstraction layer a method fails at.

2. **Automated data synthesis pipeline + concept leakage prevention**:
    - **Function**: Generates hierarchical paired preference data cost-effectively and scalably.
    - **Mechanism**: Three steps — (a) Hierarchical Concept Synthesis: Domain name → LLM generates domain description as a global constraint → L1–L3 concept tree; (b) Question Generation & Refine: Per-concept training/test questions + anchor questions and reference pos/neg answers; to avoid question phrasing hints for the target concept, **question rewriting** is performed to pivot questions toward related but different concepts; (c) Paired Answer Generation: For each rewritten question, (matching, not_matching) pairs are generated with minimal lexical-level edits.
    - **Design Motivation**: (a) Descriptions lock domain boundaries to avoid concept drift; (b) question rewriting prevents models from guessing the target; (c) minimal edits ensure differences stem purely from the concept.

3. **Two-stage QA: Automated verification + manual group review**:
    - **Function**: Ensures correct data formatting and semantic fidelity.
    - **Mechanism**: Stage 1 uses automated checks for formatting and integrity; Stage 2 involves professional NLP annotators performing calibration (20% random subset) → independent double review → consensus resolution, followed by privacy and security audits before release under the MIT license.
    - **Design Motivation**: Pure LLM synthesis may miss subtle concept biases; expert group review ensures label accuracy, which is a key credibility source for benchmark work.

### Loss & Training
SteerEval is a benchmark and does not involve training new models. Evaluated methods (Prompt 0/3-shot, PCA, DiffMean, RePS, etc.) are executed using their specific inference-time intervention $\mathcal{I}_g$, and scored via Concept Score (CS) and Harmonic Mean (HM) of CS and quality scores.

## Key Experimental Results

### Main Results: Cross-domain and cross-layer steerability (Gemma-2-9b-Instruct)
Evaluation metrics: CS (concept score, target achievement) / HM (harmonic mean of CS and quality score). Difficulty increases from L1 to L3.

| Method | Language Features L1 (CS/HM) | LF L2 | LF L3 | Personality L1 | Pers L2 | Pers L3 | Sentiment L1 | Sent L2 | Sent L3 |
|------|--------------|------|------|------|------|------|------|------|------|
| Vanilla | 1.16/1.38 | 0.95/1.14 | 0.14/0.15 | 0.45/0.58 | 0.79/1.01 | 0.05/0.06 | 1.40/1.61 | 1.18/1.40 | 0.00/0.00 |
| Prompt (0-shot) | 2.53/2.72 | 2.84/3.03 | 2.85/3.21 | 2.57/2.99 | 3.02/3.21 | 2.87/3.17 | 2.87/3.18 | 3.15/3.39 | 2.57/2.99 |
| Prompt (3-shot) | 2.32/2.60 | 2.99/3.14 | **2.88/3.19** | 2.71/3.10 | 2.94/3.27 | **3.18/3.47** | 2.97/3.35 | 2.94/3.24 | 2.37/2.71 |
| PCA | 1.94/1.85 | 1.45/1.51 | 0.13/0.15 | 1.33/1.48 | 1.51/1.20 | 0.05/0.06 | 1.86/2.01 | 1.68/1.75 | 0.00/0.00 |
| DiffMean | **3.12/2.98** | 2.70/2.78 | 0.14/0.14 | **3.16/3.10** | 3.17/3.10 | 0.05/0.05 | 2.79/2.92 | 2.83/2.68 | 0.07/0.08 |
| RePS | 2.87/2.82 | 2.36/2.16 | 2.07/2.00 | 3.15/3.04 | **3.63/3.48** | 2.34/2.12 | **3.27/3.21** | 2.75/2.53 | 1.65/1.64 |

### Ablation Study: Control Loss Curves across Abstraction Levels

| Method Type | L1 Performance | L2 Performance | L3 Performance | Conclusion |
|----------|---------|---------|---------|------|
| Activation-based (PCA / DiffMean) | Mid-Strong | Moderate | **Near 0** | Almost impossible to inject specific tokens at L3 |
| Prompt-based (0-/3-shot) | Moderate | Moderate | Moderate | The only method stable at L3 |
| RePS (Hybrid) | Mid-Strong | Strong | Moderate (>0 but < prompt) | A compromise solution |

### Key Findings
- Methods are insensitive to **domain** but extremely sensitive to **granularity**—most activation steering methods (DiffMean, PCA) perform well at L1/L2 but drop to near zero at L3.
- Prompt steering is the only method that stably supports L3, though its CS upper bound is constrained by prompt interference at L1/L2.
- RePS is the only activation-based method with non-zero L3 performance (≈2.07 LF / 2.34 Pers / 1.65 Sent), though it still significantly trails the prompt route.
- Personality is generally harder than Sentiment / Language Features (high-level dispositional priors are harder to express with a single vector).
- **Insight**: Implementing both high-level intent adjustment and low-level token constraints in deployment requires a hybrid paradigm; this benchmark quantifies where control fails.

## Highlights & Insights
- **Introducing Marr's three levels to LLM evaluation**: While LLM controllability testing has long focused on "whether an attribute changed," this paper structures the problem using a classic cognitive science framework, locating the abstraction layer where steering fails.
- **L3 is a blind spot for activation steering**: All activation vector methods reach near zero in "precise token insertion" tasks, suggesting researchers should combine representation engineering with constrained decoding.
- **Synthesis + Manual Calibration Pipeline**: Question rewriting for leakage prevention and minimal-edit paired answers are methodological contributions transferable to other preference benchmarks.

## Limitations & Future Work
- The evaluation metrics CS/HM depend on an evaluator LLM, which may introduce evaluation bias.
- Although representative, the three domains do not cover all controllable behaviors (e.g., length, citation style, ethical boundaries); reasoning patterns are provided as an appendix.
- Experiments were primarily conducted on Gemma-2-9b / Qwen-2.5-7b; steerability patterns for ultra-large models (70B+) are not yet verified.
- The reason why activation steering reaches zero at L3 is not deeply explored—is it due to vectors being unable to represent discrete token biases, or incorrect hook layer placement?

## Related Work & Insights
- **vs AXBENCH**: Both standardize steering evaluation, but SteerEval adds the **granularity dimension** and customized prompts; AXBENCH concepts come from SAEs, whereas SteerEval concepts come from human-defined behavioral goals.
- **vs RepE / CAA / ReFT series**: This work is not a new method but reveals a shared blind spot—substantial token-level control at L3 is nearly impossible.
- **vs IFEval / FollowBench**: Those benchmarks measure instruction following; SteerEval measures representation-level behavioral guidance, representing a complementary relationship.

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing the Marr framework to LLM controllability evaluation is a rare and effective framing.
- Experimental Thoroughness: ⭐⭐⭐⭐ 3 domains × 3 levels × multiple methods × multiple models, though model scale did not reach 70B+.
- Writing Quality: ⭐⭐⭐⭐ Clear definitions for the three levels, with intuitive examples in Figure 2.
- Value: ⭐⭐⭐⭐ Identifies a critical gap for representation engineering / activation steering research (L3 control), serving as a long-lasting benchmark.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CoSToM: Causal-oriented Steering for Intrinsic Theory-of-Mind Alignment in Large Language Models](costomcausal-oriented_steering_for_intrinsic_theory-of-mind_alignment_in_large_l.md)
- [\[NeurIPS 2025\] GeoCAD: Local Geometry-Controllable CAD Generation with Large Language Models](../../NeurIPS2025/llm_nlp/geocad_local_geometry-controllable_cad_generation_with_large_language_models.md)
- [\[ACL 2026\] Mind the Gap: How Elicitation Protocols Shape the Stated-Revealed Preference Gap in Language Models](mind_the_gap_how_elicitation_protocols_shape_the_stated-revealed_preference_gap_.md)
- [\[ACL 2026\] Foresight Optimization for Strategic Reasoning in Large Language Models](foresight_optimization_for_strategic_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Repeated Sequences Reveal Gaps between Large Language Models and Natural Language](repeated_sequences_reveal_gaps_between_large_language_models_and_natural_languag.md)

</div>

<!-- RELATED:END -->
