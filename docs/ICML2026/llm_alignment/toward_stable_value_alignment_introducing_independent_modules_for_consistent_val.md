---
title: >-
  [Paper Note] Toward Stable Value Alignment: Introducing Independent Modules for Consistent Value Guidance
description: >-
  [ICML 2026][LLM Alignment][Value Alignment] This paper proposes SVGT, which shifts value alignment from "embedding in backbone parameters/activations" to "attaching an independent value module." It first determines safet…
tags:
  - "ICML 2026"
  - "LLM Alignment"
  - "Value Alignment"
  - "Inference-time Guidance"
  - "Bridge Tokens"
  - "Independent Value Modules"
  - "Safety"
date: 2026-05-08
content_hash: 5758e033c89c7bf6
---

# Toward Stable Value Alignment: Introducing Independent Modules for Consistent Value Guidance

**Conference**: ICML 2026 Spotlight  
**arXiv**: [2605.11712](https://arxiv.org/abs/2605.11712)  
**Code**: https://github.com/Clervils/SVGT.git (Available)  
**Area**: LLM Alignment / RLHF Alternatives / Inference-time Guidance  
**Keywords**: Value Alignment, Inference-time Guidance, Bridge Tokens, Independent Value Modules, Safety

## TL;DR
This paper proposes SVGT, which shifts value alignment from "embedding in backbone parameters/activations" to "attaching an independent value module." It first determines safety directions within an isolated value space for current hidden states and then explicitly guides generation trajectories using learnable Bridge Tokens as attention anchors. Across four backbones, toxicity scores are reduced by over 70% with minimal loss in fluency.

## Background & Motivation

**Background**: Mainstream LLM alignment methods are divided into two categories based on timing: Training-time (RLHF/PPO, DPO, IPO, KTO, Constitutional AI), which optimizes preferences into weights; and Inference-time (System Prompt; reward-guided decoding at the output layer; Representation Engineering like ITI/CAA/RE-Control at the activation layer), which guides generation through prompts or hidden state interventions.

**Limitations of Prior Work**: Training-time methods "spread" value alignment across billions of parameters; safety often degrades into shallow output patterns rather than deep invariant representations, making them susceptible to jailbreaks. Inference-time methods do not modify weights, but techniques like ITI/CAA that inject steering vectors into the residual stream often exhibit inconsistent or inverse steering and increase perplexity, harming fluency.

**Key Challenge**: The authors identify a structural contradiction: stable value representation requires "consistent activation across all contexts and coupling to generation," while the residual stream is inherently dynamic. Value directions are iteratively reshaped, compressed, or shifted by task signals. When task-driven dynamics and value signals share the same space within the backbone, the former systematically "crowds out" the latter.

**Goal**: Reformulate alignment as "generation-time optimization," allowing an independent module to actively perceive, judge, and guide during inference, rather than passively reading alignment priors from weights.

**Key Insight**: Drawing from cognitive science theories (Haidt, Cushman) regarding human moral/value judgment: value reasoning relies on normative mechanisms stable across contexts and decoupled from specific task representations. Accordingly, value processing is moved entirely to an independent value space and acts on the backbone through an "explicit interface."

**Core Idea**: Utilize a two-stage structure consisting of an "Independent Value Space + Bridge Tokens." The former provides a context-invariant stable value direction $\Delta\mathbf{z}$, while the latter translates abstract corrections into a set of learnable latent tokens. These serve as attention anchors inserted after the prefix, naturally influencing the generation trajectory via the frozen backbone's attention mechanism.

## Method

### Overall Architecture
SVGT keeps the backbone $\theta_{\mathrm{LLM}}$ completely frozen and attaches an independent value policy $\pi_\phi$. Hidden states from designated mid-to-late layers $l^*$ are fed into the value module, which operates in two stages:

**Stage 1: Value Space Construction**: Encodes the prompt context and current hidden state simultaneously to obtain a stable value state $\mathbf{z}$ and a directional correction $\Delta\mathbf{z}=\nabla_\mathbf{z}\mathcal{D}(\mathbf{z})$.

**Stage 2: Latent Value Bridge (LVB)**: Transforms $\Delta\mathbf{z}$ into $K$ Bridge Tokens $\mathbf{B}\in\mathbb{R}^{K\times d}$. These are inserted via late-binding after the prompt to serve as attention targets for subsequent autoregressive generation.

The structure is equivalent to $P(y_t|y_{<t},x,\mathbf{c}_v)$, where $\mathbf{c}_v=\pi_\phi(\mathcal{E}(\mathbf{h}))$ is an explicit latent value context.

### Key Designs

1.  **Independent Value Space + Dual-pathway Encoding + Gradient-based Correction**:
    *   **Function**: Continuously tracks the "alignment" of current hidden states in a low-dimensional manifold isolated from the backbone and provides a directional correction.
    *   **Mechanism**: An aggregation operator $\mathcal{A}$ (e.g., last-token or attention pooling) extracts current state $\mathbf{h}_v$ and prompt context $\mathbf{h}_p$ from hidden sequences $\mathbf{H}^{(l^*)}$. These are fused via two paths: an unconditional path $f_u(\mathbf{h}_v)$ capturing context-independent value priors, and a conditional path $\mathrm{CrossAttn}(f_c(\mathbf{h}_v),f_c(\mathbf{h}_p))$ incorporating prompt context. The combined value state is $\mathbf{z}=\mathcal{R}(f_u(\mathbf{h}_v)+\lambda\cdot\mathrm{CrossAttn}(\cdots))$. A discriminator $\mathcal{D}$ outputs an alignment score, and the correction signal is $\Delta\mathbf{z}=\nabla_\mathbf{z}\mathcal{D}(\mathbf{z})$ (following PPLM's gradient guidance concept).
    *   **Design Motivation**: Unconditional encoding alone cannot handle cases where safety for the same response varies indices by prompt. The dual-pathway allows labor division in curriculum learning: unconditional learning of global priors and conditional learning of prompt-specific corrections.

2.  **Latent Value Bridge: Translating Abstract Corrections into Attention Anchors**:
    *   **Function**: Converts the abstract correction $\Delta\mathbf{z}$ into $K$ tokens that the backbone can actually "see."
    *   **Mechanism**: Defines a retrieval bank $\mathbf{C}=[\mathbf{h}_v;\phi(\Delta\mathbf{z})]^\top$, projecting the prompt final state and value correction to backbone dimension $d$. Learnable seed queries $\mathbf{Q}$ retrieve $\mathbf{B}_{\mathrm{raw}}=\mathrm{softmax}(\mathbf{Q}\mathbf{C}^\top/\sqrt{d})\mathbf{C}$ via cross-attention. A gated residual $\mathbf{B}=\mathrm{LayerNorm}(\mathbf{1}_K \mathbf{h}_v+\alpha\cdot\mathbf{B}_{\mathrm{raw}})$ anchors them to $\mathbf{h}_v$, with $\alpha$ initialized near zero. During generation, LVB runs dynamically: every token recomputes $\mathbf{z}_t, \Delta\mathbf{z}_t$ and updates Bridge Tokens via momentum.
    *   **Design Motivation**: Bridge Tokens must be "valid points on the backbone's learned manifold" to maintain fluency—thus they are weighted combinations of existing valid hidden states. Late-binding ensures guidance is built on full semantics without polluting context representations; dynamic recomputation allows token-level adaptive guidance.

3.  **Three-stage Curriculum Training**:
    *   **Function**: Enables the value module to learn progressively from "priors $\to$ contextual correction $\to$ behavioral guidance."
    *   **Mechanism**: **Stage 1** trains the unconditional encoder + discriminator on text samples for generic toxicity/safety priors using standard BCE. **Stage 2** trains the conditional pathway on prompt-response pairs using asymmetric learning rates (low lr for unconditional, high lr for conditional) to force functional separation. **Stage 3** freezes the backbone/encoder/discriminator and only trains the projector using a weighted loss: CE (behavioral cloning) + safety loss $\mathcal{L}_{\mathrm{safe}}$ with dense token-level supervision + manifold regularization $\mathcal{L}_{\mathrm{reg}}$ to keep Bridge output energy close to the prompt final state.
    *   **Design Motivation**: End-to-end training fails due to the vast difficulty gap between value and language tasks. Curriculum learning breaks the complexity into clear stages: judgment $\to$ dynamic judgment $\to$ guidance via Bridge Tokens.

### Loss & Training
$\mathcal{L}_{\mathrm{total}}=\lambda_{\mathrm{ce}}\mathcal{L}_{\mathrm{ce}}+\lambda_{\mathrm{safe}}\mathcal{L}_{\mathrm{safe}}+\lambda_{\mathrm{reg}}\mathcal{L}_{\mathrm{reg}}$. Number of Bridge Tokens $K=5\text{-}10$, value space dimension $d_v=128\text{-}256$, hidden extraction layer $l^*$ at mid-to-late positions (e.g., layer 20 for Llama-3.2-3B). Zero-initialized gates and manifold regularization ensure early training does not disrupt generation.

## Key Experimental Results

### Main Results
SVGT consistently outperforms baselines (System Prompt, DPO+LoRA, ITI/RE-Control) across four backbones (GPT-2 124M / Qwen2-1.5B / Llama-3.2-3B / Mistral-7B). Results on Llama-3.2-3B:

| Method | WildGuard Harmful↓ | BeaverTails↓ | HarmBench ASR↓ | HarmBench Refusal↑ | PPL (Fluency) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| No Guidance | 29.69 | 58.95 | 67.00 | 27.5 | 6.71 |
| System Prompt | 13.73 | 42.04 | 37.00 | 70.5 | 6.92 |
| DPO (LoRA) | 8.28 | 34.71 | 25.50 | 69.2 | 9.21 |
| ITI | 12.97 | 40.63 | 28.70 | 65.0 | 11.01 |
| RE-Control | 12.22 | 39.27 | 30.50 | 70.5 | 9.54 |
| **SVGT** | **7.84** | **28.58** | **18.50** | **75.5** | **7.34** |

On Mistral-7B, the improvement is more pronounced: BeaverTails scores dropped from 50.90 to 13.40 (−73.7%), refusal rate rose from 18.4% to 92%, and PPL actually decreased from 5.60 to 5.52, while ITI pushed PPL to 10.31.

### Ablation Study

| Config | WildGuard↓ | BeaverTails↓ | PPL |
| :--- | :---: | :---: | :---: |
| No Guidance | 29.69 | 58.95 | 6.71 |
| SVGT-Inject (Direct injection into residual) | 13.29 | 37.33 | — |
| SVGT-Bridge (Full version, Bridge Token mechanism) | 7.84 | 28.58 | 7.34 |

| Stage 1 → Stage 2 Value Accuracy (Llama-3.2-3B BeaverTails) | Acc | F1 | AUROC |
| :--- | :---: | :---: | :---: |
| Unconditional only | 68.55 | 68.42 | 78.45 |
| + Conditional | 83.48 (+14.9) | 83.06 (+14.6) | 90.91 (+12.5) |

Conditional encoding significantly improves performance on context-dependent data like BeaverTails, validating the dual-pathway design.

### Key Findings
*   Bridge Tokens reduce harmful scores by ~40% compared to direct residual injection (SVGT-Inject), proving "explicit attention anchors + late-binding" is superior to "forcing steering vectors."
*   **Scale Invariance**: SVGT reduces ASR by 70%-80% across models from GPT-2 (124M) to Mistral-7B, indicating alignment effectiveness is independent of backbone scale or pre-alignment quality.
*   **PPL and Fluency**: While ITI/RE-Control increase PPL by 60%-80%, SVGT remains close to the baseline (Llama-3.2-3B +9%; GPT-2 even decreased) because Bridge Tokens are constrained to the backbone's valid manifold.
*   **Dynamic Adversarial Experiments**: Under adversarial prompts, unguided trajectories remain in high-risk zones, while SVGT toxicity scores decrease as decoding progresses, proving the effectiveness of token-level re-computation.
*   **Acceptable Overhead**: VRAM +3%, latency +52%-65%. The method is robust to Bridge refresh intervals $r\in[1,10]$, allowing flexibility.

## Highlights & Insights
*   **Structural vs. Parametric Alignment**: Moving value processing out of the backbone is a paradigm shift—preserving original capability (frozen backbone) while avoiding shallow patterns caused by RLHF. Alignment evolves with the module rather than being fixed in the training version.
*   **Bridge Tokens as "Attention Interfaces for Values"**: Using learnable tokens as guiding anchors is elegant—it reuses the backbone's attention mechanism without adding new parameters to the main network and avoids residual pollution. This trick can be transferred to multimodal alignment or role-playing.
*   **Gradient-based Correction Reuse**: Using $\Delta\mathbf{z}=\nabla\mathcal{D}$ as a steering direction follows PPLM but isolates it in value space before projecting via Bridge Tokens. Geometrically, this "computes gradients in quotient space then lifts back," which is cleaner.
*   The use of curriculum training and asymmetric learning rates is a subtle but critical detail that prevents redundancy between unconditional and conditional pathways.

## Limitations & Future Work
*   The value space was only trained on safety-related binary labels; scalability to diverse values (fairness, privacy, culture) remains unverified.
*   The discriminator $\mathcal{D}$ relies on explicit supervision and is subject to label bias. Deploying it in new domains (medical ethics, finance) requires retraining.
*   Latency (+50%-65%) due to token-level re-computation is acceptable for chatbots but might be a bottleneck for high-throughput batch inference.
*   The choice of Bridge Token count $K$, dimension $d_v$, and layer $l^*$ is currently empirical and lacks automated design guidance.
*   Long-range consistency was only tested on relatively short adversarial prompts; sustainability over thousands of tokens remains to be verified.

## Related Work & Insights
*   **vs DPO/RLHF**: DPO embeds preferences into weights, requiring full retraining and lacking plug-and-play capability. SVGT can be attached to any model to harden safety without affecting general capabilities.
*   **vs ITI/CAA/RE-Control (Representation Engineering)**: These methods inject steering vectors into the residual stream, disrupting internal representations and spiking PPL. SVGT uses Bridge Tokens via the attention interface, preserving fluency and allowing dynamic intensity.
*   **vs Prompt Engineering / System Prompt**: System Prompts lack deep guidance and are easily overridden by adversarial prompts. SVGT tracks and corrects at the hidden level, making it much more robust (HarmBench ASR 18.5% vs System Prompt 37%).
*   **vs PPLM**: PPLM uses $\nabla\mathcal{D}$ for steering but does so directly on hidden states via gradient ascent, which is inefficient and unstable. SVGT modularizes and stabilizes this by isolating the gradient in value space.

## Rating
*   Novelty: ⭐⭐⭐⭐ "Independent Value Module + Bridge Token Anchors" is a clear and original alignment paradigm.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Covering four backbones, three baselines, three benchmarks, and overhead analysis is comprehensive, though missing diverse value scenarios.
*   Writing Quality: ⭐⭐⭐⭐ Logic flow from cognitive science to structural contradictions is tight; diagrams and formulas are clear.
*   Value: ⭐⭐⭐⭐ Provides an industrially deployable solution for plug-and-play safety hardening with negligible fluency loss.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] PICACO: Pluralistic In-Context Value Alignment of LLMs via Total Correlation Optimization](picaco_pluralistic_in-context_value_alignment_of_llms_via_total_correlation_opti.md)
- [\[ACL 2026\] How Value Induction Reshapes LLM Behaviour](../../ACL2026/llm_alignment/how_value_induction_reshapes_llm_behaviour.md)
- [\[ICML 2026\] UDM-GRPO: Stable and Efficient GRPO for Unified Discrete Diffusion Models](udm-grpo_stable_and_efficient_group_relative_policy_optimization_for_uniform_dis.md)
- [\[ICML 2026\] Alignment-Aware Decoding](alignment-aware_decoding.md)
- [\[ICML 2026\] Curriculum Learning for Safety Alignment](curriculum_learning_for_safety_alignment.md)

</div>

<!-- RELATED:END -->
