---
title: >-
  [Paper Note] Toward Stable Value Alignment: Introducing Independent Modules for Consistent Value Guidance
description: >-
  [ICML 2026][LLM Alignment][Value Alignment] This paper proposes SVGT, which shifts value alignment from "embedding into backbone parameters/activations" to "attaching an independent value module." The module continuously…
tags:
  - "ICML 2026"
  - "LLM Alignment"
  - "Value Alignment"
  - "Inference-time Guidance"
  - "Bridge Tokens"
  - "Independent Value Module"
  - "Safety"
date: 2026-05-08
content_hash: 0a8b3cef91458e08
---

# Toward Stable Value Alignment: Introducing Independent Modules for Consistent Value Guidance

**Conference**: ICML 2026  
**arXiv**: [2605.11712](https://arxiv.org/abs/2605.11712)  
**Code**: https://github.com/Clervils/SVGT.git (available)  
**Area**: LLM Alignment / RLHF Alternatives / Inference-time Guidance  
**Keywords**: Value Alignment, Inference-time Guidance, Bridge Tokens, Independent Value Module, Safety

## TL;DR
This paper proposes SVGT, which shifts value alignment from "embedding into backbone parameters/activations" to "attaching an independent value module." The module continuously assesses the safety direction of the current hidden state in an isolated value space, then uses a set of learnable Bridge Tokens as explicit attention anchors to guide generation. Across four backbones, harmfulness scores are reduced by over 70% with almost no loss in fluency.

## Background & Motivation

**Background**: Mainstream LLM alignment methods can be categorized by intervention timing: training-time (RLHF/PPO, DPO, IPO, KTO, Constitutional AI) methods optimize value preferences into weights; inference-time (System Prompt, output-layer reward-guided decoding, activation-layer Representation Engineering such as ITI/CAA/RE-Control) methods guide generation via prompts or hidden state interventions.

**Limitations of Prior Work**: Training-time methods diffuse value across billions of parameters, often reducing safety to shallow output patterns rather than deep invariant representations, making them vulnerable to jailbreaks. Inference-time methods, while not altering weights, often inject steering vectors directly into the residual stream (e.g., ITI/CAA), which can cause inconsistent or inverse steering and increase perplexity, harming fluency.

**Key Challenge**: The authors identify a structural contradiction—stable value representation requires "continuous activation and coupling to generation across all contexts," but the residual stream is inherently dynamic, with value directions iteratively reshaped, compressed, and drifted by task signals. When task-driven dynamics and value signals share the same space within the backbone, the former systematically "crowds out" the latter.

**Goal**: To reframe alignment as "generation-time optimization," enabling an independent module to actively perceive, judge, and guide during inference, rather than passively reading alignment priors from weights.

**Key Insight**: Drawing from cognitive science on human moral/value judgment (Haidt, Cushman): value reasoning relies on context-invariant normative mechanisms, decoupled from specific task representations. Accordingly, value processing is moved entirely into an independent value space, interfacing explicitly with the backbone.

**Core Idea**: A two-stage structure of "independent value space + Bridge Token": the former provides a context-invariant stable value direction $\Delta\mathbf{z}$, the latter translates the abstract correction into a set of learnable latent tokens, inserted as attention anchors at the prefix, leveraging the frozen backbone's attention mechanism to naturally influence generation.

## Method

### Overall Architecture
SVGT fully freezes the backbone $\theta_{\mathrm{LLM}}$ and attaches an independent value policy $\pi_\phi$. Hidden states are extracted from a specified mid-to-late layer $l^*$ and fed to the value module, which operates in two stages:

Stage 1 Value Space Construction: Encodes both the prompt context and current hidden state to obtain a stable value state $\mathbf{z}$ and directional correction $\Delta\mathbf{z}=\nabla_\mathbf{z}\mathcal{D}(\mathbf{z})$.

Stage 2 Latent Value Bridge: Converts $\Delta\mathbf{z}$ into $K$ Bridge Tokens $\mathbf{B}\in\mathbb{R}^{K\times d}$, which are late-bound and inserted after the prompt as attention targets for subsequent autoregressive generation.

The overall structure is equivalent to $P(y_t|y_{<t},x,\mathbf{c}_v)$, where $\mathbf{c}_v=\pi_\phi(\mathcal{E}(\mathbf{h}))$ is the explicit latent value context.

### Key Designs

1. **Independent Value Space + Dual-path Encoding + Gradient-based Correction Signal**:

    - **Function**: Continuously tracks the "alignment" of the current hidden state in a low-dimensional manifold isolated from the backbone, providing a directional correction.
    - **Mechanism**: Aggregates the hidden sequence $\mathbf{H}^{(l^*)}$ using operator $\mathcal{A}$ (e.g., last-token or attention pooling) to extract current state $\mathbf{h}_v$ and prompt context $\mathbf{h}_p$; fuses via two paths: unconditional path $f_u(\mathbf{h}_v)$ captures context-independent value priors, conditional path $\mathrm{CrossAttn}(f_c(\mathbf{h}_v),f_c(\mathbf{h}_p))$ incorporates prompt context via cross-attention, then weighted to obtain value state $\mathbf{z}=\mathcal{R}(f_u(\mathbf{h}_v)+\lambda\cdot\mathrm{CrossAttn}(\cdots))$; a discriminator $\mathcal{D}$ outputs alignment score, and the directional correction is $\Delta\mathbf{z}=\nabla_\mathbf{z}\mathcal{D}(\mathbf{z})$ (adopting PPLM's gradient guidance).
    - **Design Motivation**: A single unconditional encoder cannot handle cases where the same answer is safe/unsafe under different prompts; dual-path enables curriculum learning—unconditional learns global priors, conditional learns prompt-specific corrections, avoiding redundancy where one network must both memorize general rules and make specific judgments.

2. **Latent Value Bridge: Translating Abstract Correction into Attention Anchors**:

    - **Function**: Converts the abstract direction $\Delta\mathbf{z}$ in value space into $K$ tokens visible to the backbone.
    - **Mechanism**: Defines a retrieval bank $\mathbf{C}=[\mathbf{h}_v;\phi(\Delta\mathbf{z})]^\top$, projecting prompt final state and value correction to backbone dimension $d$; $K$ learnable seed queries $\mathbf{Q}$ retrieve via cross-attention $\mathbf{B}_{\mathrm{raw}}=\mathrm{softmax}(\mathbf{Q}\mathbf{C}^\top/\sqrt{d})\mathbf{C}$, then gated residual $\mathbf{B}=\mathrm{LayerNorm}(\mathbf{1}_K \mathbf{h}_v+\alpha\cdot\mathbf{B}_{\mathrm{raw}})$ anchors on $\mathbf{h}_v$, with gate $\alpha$ initialized near zero. During generation, LVB runs dynamically: for each decoded token, $\mathbf{z}_t, \Delta\mathbf{z}_t$ are recomputed and Bridge Token updated via momentum.
    - **Design Motivation**: Bridge Tokens must be "valid points on the backbone's learned manifold" to avoid harming fluency—thus constructed as weighted combinations of existing valid hiddens, not outlier vectors. Late-binding (inserting after prompt processing) ensures guidance is based on complete semantics, not contaminating context representations; dynamic recomputation allows token-level adaptive guidance—strengthening when the model drifts, relaxing when safe.

3. **Three-stage Curriculum Training**:

    - **Function**: Enables the value module to learn "prior → context correction → behavioral guidance" progressively.
    - **Mechanism**: Stage 1 uses standard BCE to train unconditional encoder + discriminator on independent text samples, establishing general toxicity/unsafe instruction priors; Stage 2 trains the conditional pathway on prompt-response pairs, enforcing division of labor via asymmetric learning rates (low lr for unconditional branch, high lr for conditional branch); Stage 3 freezes backbone+encoder+discriminator, trains only the projector, with three weighted losses: CE (teacher-forcing behavioral imitation) + safety loss $\mathcal{L}_{\mathrm{safe}}=\mathrm{mean}(\mathrm{softplus}(s)+\alpha\mathrm{ReLU}(s))$ for dense token-level supervision + manifold regularization $\mathcal{L}_{\mathrm{reg}}=\max(||\|\mathbf{B}\|/\|\mathbf{h}_{M-1}\|-1|-\tau,0)$ to constrain Bridge output energy close to prompt final state.
    - **Design Motivation**: End-to-end training fails due to the difficulty gap between value and language tasks; curriculum learning splits "can judge → can dynamically judge → can guide generation via Bridge" into three clear capability steps, each solving a distinct goal.

### Loss & Training
$\mathcal{L}_{\mathrm{total}}=\lambda_{\mathrm{ce}}\mathcal{L}_{\mathrm{ce}}+\lambda_{\mathrm{safe}}\mathcal{L}_{\mathrm{safe}}+\lambda_{\mathrm{reg}}\mathcal{L}_{\mathrm{reg}}$; number of Bridge Tokens $K=5\text{-}10$, value space dimension $d_v=128\text{-}256$, hidden extraction layer is mid-to-late (Llama-3.2-3B uses layer 20); zero-initialized gating + manifold regularization jointly ensure early training does not disturb generation.

## Key Experimental Results

### Main Results
Four backbones (GPT-2 124M / Qwen2-1.5B / Llama-3.2-3B / Mistral-7B), three alignment baselines (System Prompt, DPO+LoRA, ITI/RE-Control), SVGT leads across the board. On Llama-3.2-3B:

| Method | WildGuard Harm↓ | BeaverTails↓ | HarmBench ASR↓ | HarmBench Refusal↑ | PPL (Fluency) |
|--------|-----------------|--------------|----------------|--------------------|---------------|
| No Guidance | 29.69 | 58.95 | 67.00 | 27.5 | 6.71 |
| System Prompt | 13.73 | 42.04 | 37.00 | 70.5 | 6.92 |
| DPO (LoRA) | 8.28 | 34.71 | 25.50 | 69.2 | 9.21 |
| ITI | 12.97 | 40.63 | 28.70 | 65.0 | 11.01 |
| RE-Control | 12.22 | 39.27 | 30.50 | 70.5 | 9.54 |
| **SVGT** | **7.84** | **28.58** | **18.50** | **75.5** | **7.34** |

On Mistral-7B, results are even more pronounced: BeaverTails drops from 50.90 to 13.40 (−73.7%), refusal rate rises from 18.4% to 92%, PPL drops from 5.60 to 5.52; in contrast, ITI pushes PPL to 10.31, a stark difference.

### Ablation Study

| Configuration | WildGuard↓ | BeaverTails↓ | PPL |
|---------------|-----------|--------------|-----|
| No Guidance | 29.69 | 58.95 | 6.71 |
| SVGT-Inject (directly injects correction into residual) | 13.29 | 37.33 | — |
| SVGT-Bridge (full version, Bridge Token mechanism) | 7.84 | 28.58 | 7.34 |

| Stage 1 → Stage 2 Value Discrimination Accuracy (Llama-3.2-3B BeaverTails) | Acc | F1 | AUROC |
|---|---|---|---|
| Unconditional only | 68.55 | 68.42 | 78.45 |
| + Conditional | 83.48 (+14.9) | 83.06 (+14.6) | 90.91 (+12.5) |

Conditional encoding yields especially large improvements on context-dependent datasets like BeaverTails, validating the necessity of the dual-path design.

### Key Findings
- Bridge Tokens reduce harmfulness scores by ~40% compared to direct residual injection (SVGT-Inject), demonstrating that "explicit attention anchors + late-binding" is far superior to "hard steering vector injection."
- Consistent across scales: from GPT-2 (124M) to Mistral-7B, ASR is reduced by 70%-80% and refusal rates pushed above 75%, indicating alignment effectiveness is independent of backbone size or pre-alignment quality.
- PPL and fluency: ITI/RE-Control increase PPL by 60%-80%, while SVGT is nearly baseline (Llama-3.2-3B only +9%, GPT-2 even decreases), because Bridge Tokens are constrained to the backbone's learned valid representation manifold.
- Dynamic adversarial experiments: On 5 adversarial prompts, unguided trajectories remain in high-risk regions, while SVGT's harmfulness scores decrease throughout decoding, proving the dynamic mechanism of "recomputing $\Delta\mathbf{z}_t$ per token" enables real-time correction.
- Computational overhead is acceptable: +3% VRAM, +52%-65% latency, and robust to Bridge refresh interval $r\in[1,10]$, allowing flexible trade-offs.

## Highlights & Insights
- **Structural vs Parametric Alignment**: Moving value processing outside the backbone is a paradigm shift—preserving original model capabilities (backbone frozen) while avoiding RLHF's shallow pattern issues from embedding value into weights. Alignment capability evolves with the module, not fixed by training version.
- **Bridge Tokens as "Attention Interface for Value"**: Using a set of learnable tokens as guidance anchors is elegant—reusing the backbone's attention mechanism (no new parameters in the main network) and avoiding direct residual contamination. This trick is transferable to other scenarios needing external signal guidance (multimodal alignment, role-playing, instruction following).
- **Reuse of Gradient-based Correction Signal**: Using $\Delta\mathbf{z}=\nabla\mathcal{D}$ as the steering direction follows PPLM, but PPLM's direct hidden modification is too crude; SVGT isolates it in value space and projects back via Bridge, essentially "computing gradients in quotient space, then lifting back," which is geometrically cleaner.
- Curriculum training + asymmetric learning rates is an easily overlooked detail—it enforces the unconditional path to maintain stable priors, while the conditional path learns only corrections, avoiding redundancy from both paths learning similar functions.

## Limitations & Future Work
- The value space is trained only on safety-related binary label data (WildGuardMix, BeaverTails); scalability to multi-dimensional values (fairness, privacy, cultural sensitivity, long-term utility) is untested.
- The discriminator $\mathcal{D}$ is still explicitly supervised, and its quality is affected by annotation bias; deploying SVGT to new domains without human labels (e.g., financial compliance, medical ethics) requires retraining the discriminator, so it is not truly zero-shot alignment.
- Dynamic LVB recomputes $\mathbf{z}_t, \Delta\mathbf{z}_t$ for every token, adding 50%-65% latency—acceptable for interactive scenarios (chatbots), but a bottleneck for high-throughput batch inference (scoring/generation); the upper bound for Bridge refresh interval $r$ lacks theoretical guidance.
- Choices of Bridge Token number $K$, value space dimension $d_v$, and extraction layer $l^*$ are currently empirical, lacking automated or interpretable design guidance.
- Long-range consistency is insufficiently tested—trajectory visualization is only on relatively short adversarial prompts (HarmBench); whether Bridge Tokens are diluted by new content in long generations (thousands of tokens) remains to be verified.

## Related Work & Insights
- **vs DPO/RLHF**: DPO embeds value preferences into weights, requiring retraining the entire model and lacking plug-and-play; SVGT fully freezes the backbone, can be attached to any released model, and enables local safety reinforcement without affecting general capabilities.
- **vs ITI/CAA/RE-Control (Representation Engineering)**: These methods inject steering vectors into the residual stream, disrupting internal representations and increasing PPL; SVGT uses Bridge Tokens via the attention interface, preserving fluency and allowing dynamic strength adjustment.
- **vs Prompt Engineering / System Prompt**: System Prompt only provides instructions at the input, lacking deep guidance and easily overridden by adversarial prompts; SVGT continuously tracks and corrects at the hidden layer, much more robust to jailbreaks (HarmBench ASR 18.5% vs System Prompt 37%).
- **vs PPLM**: PPLM also uses $\nabla\mathcal{D}$ to guide generation, but does gradient ascent directly on the residual stream, which is less efficient and stable; SVGT computes gradients in an isolated value space and projects back via Bridge, essentially engineering, modularizing, and stabilizing PPLM.

## Rating
- Novelty: ⭐⭐⭐⭐ "Independent value module + Bridge Token attention anchor" is a clear and original alignment paradigm; some components (gradient guidance, cross-attention retrieval) have precedents, but the combination is creative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four backbones across scales + three baselines + three benchmarks + ablation + dynamic trajectory + overhead analysis, quite comprehensive; lack of multi-value scenarios is a minor regret.
- Writing Quality: ⭐⭐⭐⭐ Motivation is tightly reasoned (cognitive science → structural contradiction → design motivation), method diagrams are clear, loss formulas and training stages are well separated.
- Value: ⭐⭐⭐⭐ Provides an industrially deployable solution for "plug-and-play safety reinforcement of released large models," with almost no PPL drop, directly valuable to the LLM safety community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Trajectory Bellman Residual Minimization: A Simple Value-Based Method for LLM Reasoning](../../NeurIPS2025/llm_alignment/trajectory_bellman_residual_minimization_a_simple_value-based_method_for_llm_rea.md)
- [\[ICLR 2026\] Unifying Stable Optimization and Reference Regularization in RLHF (DAR)](../../ICLR2026/llm_alignment/unifying_stable_optimization_and_reference_regularization_in_rlhf.md)
- [\[ICML 2026\] Pareto-Guided Optimal Transport for Multi-Reward Alignment](pareto-guided_optimal_transport_for_multi-reward_alignment.md)
- [\[NeurIPS 2025\] Mitigating Hallucination Through Theory-Consistent Symmetric Multimodal Preference Optimization](../../NeurIPS2025/llm_alignment/mitigating_hallucination_through_theory-consistent_symmetric_multimodal_preferen.md)
- [\[ICML 2026\] BLOCK-EM: Preventing Emergent Misalignment via Latent Blocking](block-em_preventing_emergent_misalignment_via_latent_blocking.md)

</div>

<!-- RELATED:END -->
