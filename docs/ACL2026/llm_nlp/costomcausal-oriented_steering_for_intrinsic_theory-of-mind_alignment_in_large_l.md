---
title: >-
  [Paper Note] CoSToM: Causal-oriented Steering for Intrinsic Theory-of-Mind Alignment in Large Language Models
description: >-
  [ACL 2026][LLM/NLP][Theory of Mind] The CoSToM framework is proposed to first locate key layers encoding Theory-of-Mind (ToM) features in LLMs via causal tracing (found primarily in early layers)…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "Theory of Mind"
  - "Causal Tracing"
  - "Activation Steering"
  - "Dialogue Systems"
  - "Social Reasoning"
date: 2026-05-08
content_hash: b108a43b2df75f9e
---

# CoSToM: Causal-oriented Steering for Intrinsic Theory-of-Mind Alignment in Large Language Models

**Conference**: ACL 2026  
**arXiv**: [2604.10031](https://arxiv.org/abs/2604.10031)  
**Code**: [GitHub](https://github.com/CGCL-codes/CoSToM)  
**Area**: LLM/NLP  
**Keywords**: Theory of Mind, Causal Tracing, Activation Steering, Dialogue Systems, Social Reasoning

## TL;DR
The CoSToM framework is proposed to first locate key layers encoding Theory-of-Mind (ToM) features in LLMs via causal tracing (found primarily in early layers), and then perform lightweight alignment on these layers through activation steering. This significantly enhances the social reasoning quality in negotiation and persuasion dialogues, transforming LLMs from "knowing but not using" to "knowing and using."

## Background & Motivation

**Background**: Theory of Mind (ToM)—the ability to understand others' beliefs, desires, and intentions—is a hallmark of human social intelligence. While LLMs perform well on standard ToM benchmarks, research indicates they struggle to generalize in task-specific scenarios, often relying on carefully engineered prompts to simulate reasoning.

**Limitations of Prior Work**: A critical "internal knowledge-external behavior" misalignment exists: LLMs can correctly answer ToM questions (e.g., inferring that a user wants firewood) but generate incoherent proposals in actual negotiations (e.g., offering water instead of firewood). Once explicit "infer and respond" instructions are removed, the models fail to ground internally encoded mental states into behavior.

**Key Challenge**: The ToM capabilities demonstrated by LLMs might be temporary simulations triggered by instructions rather than stable intrinsic cognition—knowledge is stored internally but cannot be spontaneously externalized as behavior.

**Goal**: (1) To discover whether LLMs possess true internal representations related to ToM; (2) To identify the specific layers where these representations reside; (3) To determine if intervening in these representations can improve downstream dialogue quality.

**Key Insight**: Leveraging mechanistic interpretability, this work uses causal tracing to locate ToM features and applies activation steering for proactive intervention.

**Core Idea**: A frozen probe decoder is used as a differentiable validator to backpropagate ToM alignment loss to the key layers of the encoder. Through a Gradient Bridge mechanism, only the LoRA adapters in the shallow layers are updated.

## Method

### Overall Architecture
The framework consists of two phases: (1) ToM Interpreting Phase—utilizing causal tracing to locate key layers for ToM feature encoding; (2) ToM Steering Phase—installing LoRA adapters in key layers and using the ToM QA accuracy of the probe decoder as a supervisory signal to update the encoder via backpropagation. During inference, the encoder generates ToM-enhanced activations, based on which the decoder generates dialogues.

### Key Designs

1.  **Interpreting ToM**:
    - **Function**: Identify key layers encoding BDI (Belief-Desire-Intention) information within the LLM.
    - **Mechanism**: Two model copies are instantiated—a context encoder processes the dialogue history, and a probe decoder receives frozen activations from a specific layer $\ell$ of the encoder to attempt ToM questions. By scanning the decoder's accuracy layer-by-layer, it determines which layers contain sufficient ToM information.
    - **Design Motivation**: Understanding before intervention—precisely identifying the location of ToM features is necessary for accurate intervention.

2.  **Gradient Bridge**:
    - **Function**: Backpropagate ToM alignment loss to the encoder through a frozen decoder.
    - **Mechanism**: The encoder processes dialogue history → activations are intercepted at key ToM layer $\ell$ → injected into the frozen probe decoder → the decoder answers ToM questions → cross-entropy loss is calculated with BDI labels → gradients flow through the frozen decoder and the activation interface into layers 0 to $\ell$ of the encoder, updating only these shallow LoRA adapters.
    - **Design Motivation**: Direct fine-tuning on ToM QA tasks is misaligned with dialogue generation tasks. The Gradient Bridge bypasses this by training "how to generate ToM-rich representations" rather than "how to answer ToM questions."

3.  **Inference-time ToM-enhanced Dialogue Generation**:
    - **Function**: Transform ToM-aligned internal representations into high-quality dialogues.
    - **Mechanism**: During inference, the ToM-enhanced encoder processes the dialogue history. The decoder no longer performs ToM QA but generates task-specific dialogues (e.g., negotiation/persuasion) conditioned on ToM-rich activations.
    - **Design Motivation**: Decoupled design for training and inference—the decoder acts as a ToM validator during training and a dialogue generator during inference. CoSToM serves as a plug-and-play module generalizable to various social tasks.

## Key Experimental Results

### Main Results (Negotiation and Persuasion Dialogue Quality)

| Method | Dialog Quality Improvement | Description |
|------|------------|------|
| Standard Prompt | Baseline | Uses general instructions only |
| ToM-explicit Prompt | +Significant | Requires carefully designed prompts |
| Full-layer LoRA | +Moderate | Many parameters but marginal improvement |
| **CoSToM** | **+Maximum** | Updates LoRA only in key ToM layers |

### Causal Tracing Findings

| Finding | Description |
|------|------|
| ToM features mainly encoded in **early layers** | Contrary to intuition—usually high layers encode semantics |
| BDI elements peak at different layers | Encoding positions for Belief, Desire, and Intention do not fully overlap |
| Consistency across models | Llama-3-8B and Qwen2.5-7B exhibit similar patterns |

### Key Findings
- The discovery that **ToM features are mainly encoded in early layers** challenges the common assumption that "higher layers equal higher-level semantics."
- **CoSToM generalizes across tasks as a plug-and-play module**: Effective in both negotiation and persuasion social tasks.
- **Gradient Bridge is more effective than direct ToM QA fine-tuning**: The latter suffers from misalignment between training objectives and dialogue generation.
- **Lightweight**: Only LoRA adapters in key ToM layers need updating, with significantly fewer parameters than full-layer fine-tuning.

## Highlights & Insights
- The "from interpretation to intervention" methodology is highly inspiring—using causal tracing to answer "where" and activation steering to answer "how to use." This two-step paradigm is applicable to the alignment of any internal LLM capability.
- Using a **frozen decoder as a differentiable validator** ingeniously solves the task misalignment issue between "ToM reasoning" and "dialogue generation."
- The diagnosis of the **"internal knowledge-external behavior" misalignment** provides significant insights for the broader field of LLM alignment.

## Limitations & Future Work
- Requires BDI annotated data, which is costly to obtain.
- Memory consumption of the dual-model architecture is 2N, posing resource requirements.
- Localization of key ToM layers may vary with tasks and data distributions.
- Validated only on negotiation and persuasion; further social scenarios (e.g., comforting, education) remain to be explored.
- The computational cost of causal tracing may be high for extremely large models.

## Related Work & Insights
- **vs Prompt-based ToM**: While prompting acts as an external scaffold, CoSToM provides internal alignment—the former requires manual design per instance, while the latter is a one-time training solution.
- **vs MindDial (Qiu et al., 2024)**: MindDial explicitly tracks belief text and appends it to input, which may propagate errors. CoSToM operates in the activation space, avoiding text-level error propagation.
- **vs Mechanistic Interpretability**: Most work stops at diagnosis; CoSToM advances from diagnosis to treatment.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The ToM alignment paradigm involving causal tracing and Gradient Bridge is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Cross-model validation and ablation studies are provided, though downstream tasks are limited to two.
- Writing Quality: ⭐⭐⭐⭐⭐ The RQ-driven structure is very clear, with excellent illustrations.
- Value: ⭐⭐⭐⭐⭐ Offers profound insights for research into LLM social intelligence and alignment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Mind the Gap: How Elicitation Protocols Shape the Stated-Revealed Preference Gap in Language Models](mind_the_gap_how_elicitation_protocols_shape_the_stated-revealed_preference_gap_.md)
- [\[AAAI 2026\] ParetoHqD: Fast Offline Multiobjective Alignment of Large Language Models Using Pareto High-Quality Data](../../AAAI2026/llm_nlp/paretohqd_fast_offline_multiobjective_alignment_of_large_language_models_using_p.md)
- [\[ACL 2026\] SteerEval: How Controllable Are Large Language Models? A Unified Evaluation across Behavioral Granularities](how_controllable_are_large_language_models_a_unified_evaluation_across_behaviora.md)
- [\[ICML 2026\] The Cylindrical Representation Hypothesis for Language Model Steering](../../ICML2026/llm_nlp/the_cylindrical_representation_hypothesis_for_language_model_steering.md)
- [\[ICLR 2026\] Fine-Grained Activation Steering: Steering Less, Achieving More](../../ICLR2026/llm_nlp/fine-grained_activation_steering_steering_less_achieving_more.md)

</div>

<!-- RELATED:END -->
