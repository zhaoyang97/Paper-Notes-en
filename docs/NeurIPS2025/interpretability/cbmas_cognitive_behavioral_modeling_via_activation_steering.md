---
title: >-
  [Paper Note] CBMAS: Cognitive Behavioral Modeling via Activation Steering
description: >-
  [NeurIPS 2025][activation steering] CBMAS proposes a framework that repurposes activation steering as a continuous diagnostic tool. By conducting dense α-sweeps and decoupling injection layers from readout layers, the framework elevates cognitive bias analysis from a binary "biased / unbiased" judgment to a continuous trajectory analysis capable of tracking flip points, propagation paths, and attenuation patterns. Experiments on GPT-2 Small reveal that appeasement behavior is strongly encoded in shallow layers but decays rapidly toward deeper layers.
tags:
  - NeurIPS 2025
  - activation steering
  - cognitive bias
  - bias response curve
  - logit lens
  - layer sensitivity analysis
date: 2026-05-08
content_hash: 88d269ffd9ea9171
---

# CBMAS: Cognitive Behavioral Modeling via Activation Steering

**Conference**: NeurIPS 2025
**arXiv**: [2601.06109](https://arxiv.org/abs/2601.06109)
**Code**: [Available](https://github.com/shimamooo/CBMAS)
**Area**: Interpretability / LLM Behavioral Analysis
**Keywords**: activation steering, cognitive bias, bias response curve, logit lens, layer sensitivity analysis

## TL;DR

CBMAS proposes a framework that repurposes activation steering as a continuous diagnostic tool. By conducting dense α-sweeps and decoupling injection layers from readout layers, the framework elevates cognitive bias analysis from a binary "biased / unbiased" judgment to a continuous trajectory analysis capable of tracking flip points, propagation paths, and attenuation patterns. Experiments on GPT-2 Small reveal that appeasement behavior is strongly encoded in shallow layers but decays rapidly toward deeper layers.

## Background & Motivation

LLMs exhibit a variety of cognitive behaviors—sycophancy, appeasement, satisficing, compliance—across different prompts and contexts, yet the internal encoding of these behaviors is unpredictable and difficult to control. Existing bias evaluation methods suffer from two fundamental problems.

First, **bias is treated as a binary phenomenon**. Conventional methods detect the presence or degree of bias via paired prompt comparisons (e.g., "He is a doctor" vs. "She is a doctor"), but this "snapshot" approach ignores the continuous latent structure of bias inside the model. Bias does not simply exist or not exist; rather, it undergoes a continuous dynamic process from emergence to flip point to saturation.

Second, **a gap persists between high-level behavioral evaluation and low-level representational analysis**. Mechanistic interpretability has uncovered fine-grained structure in attention heads, MLP layers, and residual streams, but these tools are rarely applied to cognitive behavioral research. Consequently, neither why a model exhibits a particular cognitive bias nor how to intervene precisely without retraining can be explained.

The core idea of CBMAS is to transform activation steering from a control mechanism into a diagnostic instrument. By constructing steering vectors along cognitive-behavior directions and conducting dense scans across both steering strength α and layer depth, the framework generates Bias Response Curves (BRCs) that expose flip points, propagation patterns, and layer-wise sensitivity that discrete snapshot methods cannot capture.

## Method

### Overall Architecture

The CBMAS analysis pipeline consists of four steps: (1) extract steering vectors from a contrastive prompt-pair dataset; (2) inject the steering vector at a designated injection layer and observe effects across multiple readout layers; (3) densely sweep over the α range to collect multidimensional metrics; and (4) analyze the BRCs to identify flip points and propagation regularities. The key architectural innovation is the decoupling of the injection layer from the readout layer, forming a three-dimensional analysis space of (injection layer, readout layer, α).

### Key Designs

1. **Contrastive Prompt Dataset and Steering Vector Construction**
   - *Function*: Extract vectors encoding the bias direction for each cognitive behavior (sycophancy, appeasement, satisficing, compliance).
   - *Mechanism*: Each data point consists of a structurally identical prompt pair with opposing choices. Option A represents a response exhibiting the target cognitive behavior; Option B represents a neutral response. For a given layer $L$ and injection site $S$, the steering vector is defined as $\mathbf{v}_L^{(S)} = \mathbb{E}[\mathbf{h}_L^{(S)}(p^{(A)}) - \mathbf{h}_L^{(S)}(p^{(B)})]$, i.e., the mean of hidden-state differences across all contrastive pairs at that layer.
   - *Design Motivation*: Diverse prompt pairs cover the full range of manifestations of the target behavior, so the steering vector represents a general behavioral direction rather than a context-specific one. Each behavior is represented by 200 contrastive examples spanning multiple domains including advice, technical questions, health, and finance.

2. **Bias Response Curve (BRC) Protocol**
   - *Function*: Extend bias analysis from discrete snapshots to continuous trajectories.
   - *Mechanism*: Dense sweeps are performed over a user-defined α range (default: −10 to 10, step size 0.5). For each α, the hidden state at the injection layer is modified as $\mathbf{h} \leftarrow \mathbf{h} + \alpha \mathbf{v}$, and six metrics are recorded at the readout layer: logit difference $\Delta_{logit}(\alpha) = \text{logit}(y_A|x,\alpha) - \text{logit}(y_B|x,\alpha)$, probability difference, odds ratio, KL divergence (measuring perturbation to the overall distribution), per-token perplexity (a fluency proxy), and rank trajectory (rank changes of target tokens).
   - *Design Motivation*: Outputs are forced to be binary—prompts end with "I choose (" to compel the model to select between A and B. Random vectors and vectors orthogonal to the bias direction serve as control conditions to confirm that observed effects originate from the bias direction rather than noise.

3. **Injection–Readout Layer Decoupling Analysis**
   - *Function*: Track the propagation and transformation of the bias signal from the injection point through subsequent layers.
   - *Mechanism*: All pairs $(L_\text{inj}, L_\text{read})$ satisfying $L_\text{read} > L_\text{inj}$ are analyzed to form a "bias propagation map." Comparative analysis across different injection sites (e.g., `hook_resid_mid`, `hook_resid_post`) is also supported.
   - *Design Motivation*: Observing effects only at the injection layer cannot reveal whether the intervention genuinely alters the model's final behavior or is "washed out" by subsequent layers. The decoupled design enables tracking of signal amplification, attenuation, and dissipation.

### Loss & Training

CBMAS is a purely analytical framework involving no model training. All analyses are performed at inference time using TransformerLens for activation-level intervention and readout. Experiments are conducted with `seed=42`; the full experimental pipeline can be reproduced on a single A40 GPU in approximately 4–7 minutes.

## Key Experimental Results

### Main Results

Analysis results for the reassurance (appeasement behavior) steering vector on GPT-2 Small (12 layers):

| Injection→Readout Layer | α Flip Region | Logit Difference Slope | KL Divergence | Control Condition |
|-------------------------|---------------|----------------------|---------------|-------------------|
| L0→L1 | α≈0 | Steep positive slope | Low, symmetric | Flat |
| L0→L6 | α≈0 | Moderate slope | Low | Flat |
| L0→L11 | No clear flip | Nearly flat | Low, symmetric | Flat |
| L1→L6 | α≈0 | Monotonically increasing | Low | Flat |
| L3→L4 | α≈0 | Rank flip | — | — |
| L3→L6 | α≈0 | Continuous logit trajectory | Low | Flat |

### Ablation Study

| Configuration | Key Observation | Interpretation |
|---------------|-----------------|----------------|
| Bias vector vs. random vector | Bias vector produces monotonic trajectory; random vector is flat | Effect originates from the bias direction, not noise |
| Bias vector vs. orthogonal vector | Orthogonal vector is equally flat | Dual control strengthens the conclusion |
| L0 injection vs. L1 injection | L1 yields a cleaner monotonic signal | L0 is noisier; representational structure stabilizes after L1 |
| Shallow readout vs. deep readout | Strong signal at shallow layer (L1); nearly fully attenuated at deep layer (L11) | Appeasement behavior is encoded early and progressively diluted |

### Key Findings

- **Flip points objectively exist**: Near α≈0, model behavior undergoes a qualitative transition—the shift from preferring option A to preferring option B is abrupt rather than gradual, a phenomenon that discrete evaluation methods are fundamentally unable to capture.
- **Shallow encoding, deep attenuation**: The appeasement behavior direction has a very strong effect at L1 but decays rapidly with layer depth, nearly vanishing entirely by L11. This indicates that the representation of this cognitive behavior is established early in the model.
- **Injection site determines signal quality**: L1 is a more suitable injection point than L0, producing a cleaner causal signal. This suggests that the token embedding layer (L0) has not yet established a stable behavioral representation.
- **Intervention preserves fluency**: KL divergence remains low and symmetrically distributed throughout the entire α sweep, indicating that activation steering is a controllable intervention.

## Highlights & Insights

- **Paradigm shift from binary to continuous**: Whereas conventional methods ask only "is there bias?", CBMAS asks "where does bias originate, how does it propagate, when does it flip, and when does it dissipate?"—a fundamental upgrade to the bias analysis paradigm.
- **Injection–readout decoupling is the core innovation**: This design expands a one-dimensional α sweep into a three-dimensional probing space, making the construction of a "bias propagation map" possible.
- **Rigorous control group design**: The simultaneous use of random vectors and orthogonal vectors as controls more convincingly rules out the noise hypothesis than a single control condition.
- **Practical tooling and datasets**: A CLI tool and a dataset of 200 examples per cognitive behavior (sycophancy, appeasement, satisficing, compliance) are provided, offering good reproducibility.
- **The "L1 crystallization" phenomenon** warrants attention: it suggests that the representational distribution of cognitive behaviors may possess a hierarchically organized layer structure.

## Limitations & Future Work

- **Severely insufficient model scale**: Validation is limited to GPT-2 Small (117M parameters, 12 layers); whether findings generalize to modern large models (LLaMA-70B, GPT-4, etc., with 80+ layers) is entirely uncertain, as the layer-wise distribution of bias encoding may differ fundamentally.
- **Only next-token prediction is analyzed**: Whether steering effects persist, accumulate, or decay during long-form autoregressive generation is not investigated.
- **Absence of causal analysis**: α-sweeps reveal correlational patterns but do not address which attention heads or MLP components are responsible for bias encoding; causal methods such as activation patching are needed.
- **Dataset scale and quality**: Only 200 examples per behavior are used, some generated by LLMs; manually constructed contrastive prompt pairs may introduce their own biases.
- **Limited behavioral dimensions**: Only 4 cognitive behaviors are tested; LLMs may exhibit a much broader range of cognitive biases, including confirmation bias, anchoring effects, and framing effects.

## Related Work & Insights

- The framework innovates upon the steering vector construction of ActAdd and CAA—the former is limited to single-layer and narrow α-range evaluation, whereas CBMAS conducts the first systematic continuous analysis.
- Logit lens and causal mediation tools are employed as analytical instruments but are not deeply integrated; future work could combine causal attribution with continuous steering analysis.
- The fundamental distinction from RLHF/Constitutional AI is that CBMAS is a diagnostic tool rather than an alignment method—it does not modify the model but reveals its internal cognitive behavioral structure.
- Flip-point analysis can provide a quantitative reference for "safety margins" in alignment: knowing that a harmful behavior flips at α=X allows evaluation of the intervention margin.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The paradigm shift from discrete to continuous analysis is innovative; the injection–readout decoupling design is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐ The experimental scale is severely limited to GPT-2 Small, though the analytical dimensions within that model are reasonably comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Method descriptions are clear, mathematical formalization is rigorous, and figures are highly informative.
- **Value**: ⭐⭐⭐ The framework design is sound but must be validated on modern large-scale models to have meaningful impact.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Auditing Meta-Cognitive Hallucinations in Reasoning Large Language Models](auditing_meta-cognitive_hallucinations_in_reasoning_large_language_models.md)
- [\[NeurIPS 2025\] Curvature Tuning: Provable Training-free Model Steering From a Single Parameter](curvature_tuning_provable_training-free_model_steering_from_a_single_parameter.md)
- [\[NeurIPS 2025\] Steering Information Utility in Key-Value Memory for Language Model Post-Training](steering_information_utility_in_key-value_memory_for_language_model_post-trainin.md)
- [\[NeurIPS 2025\] Cognitive Mirrors: Exploring the Diverse Functional Roles of Attention Heads in LLM Reasoning](cognitive_mirrors_exploring_the_diverse_functional_roles_of_attention_heads_in_l.md)
- [\[NeurIPS 2025\] Beyond Token Probes: Hallucination Detection via Activation Tensors with ACT-ViT](beyond_token_probes_hallucination_detection_via_activation_tensors_with_act-vit.md)

<!-- RELATED:END -->
