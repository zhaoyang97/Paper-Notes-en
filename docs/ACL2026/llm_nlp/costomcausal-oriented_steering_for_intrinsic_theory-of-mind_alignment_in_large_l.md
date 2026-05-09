---
title: >-
  [Paper Note] CoSToM: Causal-oriented Steering for Intrinsic Theory-of-Mind Alignment in Large Language Models
description: >-
  [ACL 2026][LLM/NLP][Theory of Mind] This paper proposes CoSToM, a framework that first applies causal tracing to identify the critical layers encoding Theory-of-Mind (ToM) features within LLMs (finding they concentrate primarily in early layers), then performs lightweight alignment via activation steering at those layers—significantly improving social reasoning quality in negotiation and persuasion dialogues, bridging the gap between "knowing but not applying" and "knowing and applying."
tags:
  - ACL 2026
  - LLM/NLP
  - Theory of Mind
  - causal tracing
  - activation steering
  - dialogue systems
  - social reasoning
date: 2026-05-08
content_hash: c2d23b7609f08417
---

# CoSToM: Causal-oriented Steering for Intrinsic Theory-of-Mind Alignment in Large Language Models

**Conference**: ACL 2026
**arXiv**: [2604.10031](https://arxiv.org/abs/2604.10031)
**Code**: [GitHub](https://github.com/CGCL-codes/CoSToM)
**Area**: LLM/NLP
**Keywords**: Theory of Mind, causal tracing, activation steering, dialogue systems, social reasoning

## TL;DR
This paper proposes CoSToM, a framework that first applies causal tracing to identify the critical layers encoding Theory-of-Mind (ToM) features within LLMs (finding they concentrate primarily in early layers), then performs lightweight alignment via activation steering at those layers—significantly improving social reasoning quality in negotiation and persuasion dialogues, bridging the gap between "knowing but not applying" and "knowing and applying."

## Background & Motivation

**Background**: Theory of Mind (ToM)—the capacity to understand others' beliefs, desires, and intentions—is a hallmark of human social intelligence. LLMs perform reasonably well on standard ToM benchmarks, yet research shows they struggle to generalize to task-specific scenarios and rely on carefully engineered prompts to simulate ToM reasoning.

**Limitations of Prior Work**: A critical "internal knowledge–external behavior" misalignment exists: LLMs can correctly answer ToM questions (e.g., inferring a user wants firewood) yet generate incoherent proposals in actual negotiations (e.g., offering water instead of firewood). Once explicit instructions such as "infer and respond" are removed, models fail to externalize their internally encoded mental-state knowledge into behavior.

**Key Challenge**: The ToM capabilities exhibited by LLMs may not constitute stable intrinsic cognition, but rather transient simulations triggered by instructions—knowledge is internally present but cannot spontaneously manifest as behavior.

**Goal**: (1) Determine whether LLMs genuinely possess ToM-related internal representations; (2) identify which layers house these representations; (3) assess whether intervening on these representations can improve downstream dialogue quality.

**Key Insight**: The work adopts a mechanistic interpretability perspective, using causal tracing to localize ToM features and activation steering to actively intervene on them.

**Core Idea**: A frozen probe decoder is used as a differentiable validator; the ToM alignment loss is backpropagated into the encoder's ToM-critical layers via a gradient bridge mechanism, updating only the LoRA adapters installed in those shallow layers.

## Method

### Overall Architecture
Two phases: (1) **ToM Interpretation**—causal tracing localizes the critical layers where ToM features are encoded; (2) **ToM Steering**—LoRA adapters are installed at the critical layers, and the ToM QA accuracy of the probe decoder serves as a supervision signal that is backpropagated to update the encoder. At inference time, the encoder produces ToM-enhanced activations on which the decoder conditions to generate dialogue.

### Key Designs

1. **Localizing ToM Layers via Causal Tracing (Interpreting ToM)**:

    - **Function**: Identify the critical layers in an LLM that encode BDI (Belief–Desire–Intention) information.
    - **Mechanism**: Two model copies are instantiated—a context encoder processes the dialogue history, while a probe decoder receives frozen activations from encoder layer $\ell$ and attempts to answer ToM questions. A layer-by-layer scan of decoder answer accuracy determines which layers contain sufficient ToM information.
    - **Design Motivation**: Understand before intervening—precise intervention requires first knowing where ToM features reside.

2. **Gradient Bridge Mechanism**:

    - **Function**: Backpropagate the ToM alignment loss through the frozen decoder into the encoder.
    - **Mechanism**: The encoder processes the dialogue history → activations at ToM-critical layer $\ell$ are intercepted → injected into the frozen probe decoder → the decoder answers ToM questions → cross-entropy loss is computed against BDI labels → gradients flow through the frozen decoder and the activation interface into encoder layers 0 through $\ell$, updating only the LoRA adapters in those shallow layers.
    - **Design Motivation**: Direct fine-tuning on ToM QA tasks is misaligned with dialogue generation and yields poor results. The gradient bridge circumvents this issue—instead of training "how to answer ToM questions," it trains "how to generate ToM-rich representations."

3. **ToM-Enhanced Dialogue Generation at Inference**:

    - **Function**: Translate ToM-aligned internal representations into high-quality dialogue.
    - **Mechanism**: At inference time, the ToM-enhanced encoder processes the dialogue history; the decoder, rather than answering ToM questions, generates task-specific dialogue (e.g., negotiation/persuasion) conditioned on the ToM-rich activations.
    - **Design Motivation**: The training–inference decoupling design—the decoder serves as a ToM validator during training and a dialogue generator during inference—allows CoSToM to function as a plug-and-play module generalizable across different social tasks.

## Key Experimental Results

### Main Results (Negotiation and Persuasion Dialogue Quality)

| Method | Dialogue Quality Gain | Notes |
|---|---|---|
| Standard prompt | Baseline | General instructions only |
| ToM-explicit prompt | +Significant | Requires carefully engineered prompts |
| Full-layer LoRA | +Moderate | More parameters, marginal improvement |
| **CoSToM** | **+Best** | LoRA only at ToM-critical layers |

### Causal Tracing Findings

| Finding | Notes |
|---|---|
| ToM features are primarily encoded in **early layers** | Counterintuitive—high layers are commonly assumed to encode high-level semantics |
| BDI components peak at different layers | Encoding locations for belief, desire, and intention do not fully overlap |
| Consistent across models | Similar patterns observed in both Llama-3-8B and Qwen2.5-7B |

### Key Findings
- The finding that **ToM features are primarily encoded in early layers** challenges the common assumption that "higher layers = higher-level semantics."
- **CoSToM generalizes across tasks as a plug-and-play module**: effective on both negotiation and persuasion social tasks.
- **The gradient bridge outperforms direct ToM QA fine-tuning**: the latter's training objective is misaligned with dialogue generation.
- **Lightweight**: only LoRA adapters at ToM-critical layers need to be updated, requiring far fewer parameters than full-layer fine-tuning.

## Highlights & Insights
- The **"from interpretation to intervention" methodology** is highly instructive—causal tracing first answers "where," and activation steering then answers "how to use it." This two-step paradigm can be applied to align any internal capability within LLMs.
- The design of a **frozen decoder as a differentiable validator** elegantly resolves the task misalignment between "ToM reasoning ≠ dialogue generation."
- The diagnosis of **"internal knowledge–external behavior" misalignment** carries significant warning implications for the broader LLM alignment field.

## Limitations & Future Work
- BDI-annotated data is required, which incurs non-trivial annotation costs.
- The dual-model architecture imposes a memory footprint of $2N$, placing demands on computational resources.
- The localization of ToM-critical layers may shift with changes in task and data distribution.
- Validation is limited to two tasks (negotiation and persuasion); additional social scenarios (e.g., consolation, education) remain unexplored.
- The computational cost of causal tracing may be substantial for large-scale models.

## Related Work & Insights
- **vs. Prompt-based ToM**: Prompt-based methods are external scaffolding; CoSToM performs internal alignment—the former requires careful prompt engineering each time, while the latter achieves permanent effect after a single training run.
- **vs. MindDial (Qiu et al., 2024)**: MindDial explicitly tracks belief text and concatenates it to the input, which can propagate errors. CoSToM operates in activation space, avoiding error propagation at the text level.
- **vs. Mechanistic Interpretability**: Most prior work stops at diagnosis; CoSToM proceeds from diagnosis to remedy.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The causal tracing + gradient bridge paradigm for ToM alignment is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Cross-model validation and ablations are solid, though downstream tasks are limited to two.
- Writing Quality: ⭐⭐⭐⭐⭐ The RQ-driven structure is exceptionally clear, with excellent figures.
- Value: ⭐⭐⭐⭐⭐ Broad implications for both LLM social intelligence and alignment research.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Lost in the Prompt Order: Revealing the Limitations of Causal Attention in Language Models](lost_in_the_prompt_order_revealing_the_limitations_of_causal_attention_in_langua.md)
- [\[AAAI 2026\] ParetoHqD: Fast Offline Multiobjective Alignment of Large Language Models Using Pareto High-Quality Data](../../AAAI2026/llm_nlp/paretohqd_fast_offline_multiobjective_alignment_of_large_language_models_using_p.md)
- [\[ACL 2026\] Foresight Optimization for Strategic Reasoning in Large Language Models](foresight_optimization_for_strategic_reasoning_in_large_language_models.md)
- [\[ICLR 2026\] Fine-Grained Activation Steering: Steering Less, Achieving More](../../ICLR2026/llm_nlp/fine-grained_activation_steering_steering_less_achieving_more.md)
- [\[ACL 2026\] Adam's Law: Textual Frequency Law on Large Language Models](adam39s_law_textual_frequency_law_on_large_language_models.md)

<!-- RELATED:END -->
