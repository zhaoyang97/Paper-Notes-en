---
title: >-
  [Paper Note] Revis: Sparse Latent Steering to Mitigate Object Hallucination in Large Vision-Language Models
description: >-
  [ICML 2026][Multimodal VLM][Object hallucination] This work redefines LVLM hallucination as "visual information loss suppressed by language priors." By orthogonally projecting out the language prior from the original vis…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Object hallucination"
  - "latent space steering"
  - "orthogonal projection"
  - "sparse intervention"
  - "mechanism interpretability"
date: 2026-05-08
content_hash: 97a58535636edd5c
---

# Revis: Sparse Latent Steering to Mitigate Object Hallucination in Large Vision-Language Models

**Conference**: ICML 2026  
**arXiv**: [2602.11824](https://arxiv.org/abs/2602.11824)  
**Code**: https://github.com/antgroup/Revis (available)  
**Area**: Multimodal VLM / Hallucination Mitigation  
**Keywords**: Object hallucination, latent space steering, orthogonal projection, sparse intervention, mechanism interpretability

## TL;DR
This work redefines LVLM hallucination as "visual information loss suppressed by language priors." By orthogonally projecting out the language prior from the original visual direction to obtain a 'pure visual vector,' and using risk gating to sparsely intervene at only the optimal single deep layer, the method reduces CHAIRS hallucination rate by ~19% without training, while preserving MM-Vet general capability.

## Background & Motivation
**Background**: LVLMs have demonstrated strong multimodal reasoning, but persistent reliability issues remain, notably "object hallucination"—the model describes plausible but non-existent details. Existing mitigation approaches focus on training-time alignment (LLaVA-RLHF, HA-DPO, OPA-DPO) and inference-time interventions (contrastive decoding VCD/M3ID, logit correction AGLA/ONLY, activation steering VTI).

**Limitations of Prior Work**: Training alignment is heavy and relies on large-scale preference data; contrastive decoding requires at least double forward passes to suppress hallucination; logit correction uses heuristic probes with poor robustness. The only method that avoids retraining and double forward passes, VTI, injects static offsets at all layers, reducing CHAIRS from 14% to 9%, but MM-Vet drops from 70.18 to 56.38, severely harming general reasoning.

**Key Challenge**: Existing intervention directions are essentially "fact-hallucination" difference vectors, which seem to point to visual information but are actually entangled with the "language prior" (the model's tendency to guess in the absence of images). Amplifying such entangled vectors by strength $\alpha$ also amplifies the language prior, causing the model to collapse into repetition or empty outputs at $\alpha\approx0.7$.

**Goal**: Without retraining or double forward passes, find an intervention that "activates only visual information without amplifying priors," and precisely select the intervention depth, so that hallucination reduction no longer comes at the cost of general capability.

**Key Insight**: The authors analyze latent space geometry using five counterfactual states (GT / hallucination / no-image GT / no-image hallucination / no-image refusal) via $[\text{EOS}]$ hidden states—deep layers can linearly separate fact vs hallucination, but the original $\mathbf{v}_{\text{raw}} = \mathbf{h}_{\text{gt}} - \mathbf{h}_{\emptyset\_\text{gt}}$ and the "language prior vector" $\mathbf{v}_{\text{prior}} = \mathbf{h}_{\emptyset\_\text{hall}} - \mathbf{h}_{\emptyset\_\text{unk}}$ have high cosine similarity in deep layers, confirming that entanglement is the root cause of VTI collapse.

**Core Idea**: Use Gram-Schmidt to project $\mathbf{v}_{\text{raw}}$ onto the orthogonal complement of $\mathbf{v}_{\text{prior}}$, obtaining the "pure visual vector" $\mathbf{v}_{\text{vis}}^\perp$. Then, use calibration-set-based per-layer risk scoring to select the deepest separable layer $L^\*$, and during inference, inject the correction only when the "hallucination risk score exceeds threshold"—making the intervention "sparse + single-layer + orthogonal" and surgical.

## Method

### Overall Architecture
A three-stage training-free pipeline:  
Stage 1: For $N=100$ samples, compute per-layer $\mathbf{v}_{\text{raw}}^{(\ell)}$ and $\mathbf{v}_{\text{prior}}^{(\ell)}$, then use Gram-Schmidt to obtain $\mathbf{v}_{\text{vis}}^{\perp(\ell)}$.  
Stage 2: On 100 COCO images, construct POPE-style Q&A to collect fact/hallucination hidden state sets. Use risk score $R(\mathbf{h}) = -\cos(\mathbf{h}, \mathbf{v}_{\text{vis}}^{\perp(\ell)})$ to search top-down for the deepest layer $L^\*$ where $R(\mathcal{H}_{\text{hall}}) > R(\mathcal{H}_{\text{fact}})$, and set threshold $\tau$ at the $k$-th quantile of the fact set.  
Stage 3: During inference, for each token, compute $R_t$; if $R_t>\tau$, add $\alpha\,\mathbf{v}_{\text{vis}}^{\perp(L^\*)}$ at layer $L^\*$, otherwise keep the original activation.

### Key Designs

1. **Orthogonal Projection to Purify Visual Vector**:

    - Function: Removes the component of the original visual direction colinear with the language prior, yielding a "pure visual guidance direction" that remains stable even when amplified.
    - Mechanism: Define $\mathbf{v}_{\text{raw}}^{(\ell)} = \mathbb{E}[\mathbf{h}_{\text{gt}}^{(\ell)} - \mathbf{h}_{\emptyset\_\text{gt}}^{(\ell)}]$ and $\mathbf{v}_{\text{prior}}^{(\ell)} = \mathbb{E}[\mathbf{h}_{\emptyset\_\text{hall}}^{(\ell)} - \mathbf{h}_{\emptyset\_\text{unk}}^{(\ell)}]$ based on counterfactual states, then perform Gram-Schmidt: $\mathbf{v}_{\text{vis}}^{\perp(\ell)} = \mathbf{v}_{\text{raw}}^{(\ell)} - \frac{\mathbf{v}_{\text{raw}}^{(\ell)}\cdot\mathbf{v}_{\text{prior}}^{(\ell)}}{\|\mathbf{v}_{\text{prior}}^{(\ell)}\|^2}\mathbf{v}_{\text{prior}}^{(\ell)}$.
    - Design Motivation: Causal probe experiments show that the pure visual vector remains stable even with large $\alpha$, while the original vector causes model collapse (infinite repetition/empty output) at $\alpha\approx 0.7$. Orthogonalization decouples "adding visual" from "adding prior" at the operator level.

2. **Calibration-Based Sparse Single-Layer Selection**:

    - Function: Selects a unique layer for intervention, avoiding cumulative bias and computational waste from multi-layer injection.
    - Mechanism: Construct calibration set $\mathcal{D}_{\text{cal}}$ (COCO + POPE-style existence questions) to extract per-layer $\mathcal{H}_{\text{fact}}, \mathcal{H}_{\text{hall}}$. Define risk $R(\mathbf{h}) = -\cos(\mathbf{h}, \mathbf{v}_{\text{vis}}^{\perp(\ell)})$, search backward from $L$ to $1$, and select the first $\ell$ where $R(\mathcal{H}_{\text{hall}}) - R(\mathcal{H}_{\text{fact}}) > 0$ as $L^\*$. Set threshold $\tau$ at the $k$-th quantile of the fact set at this layer.
    - Design Motivation: t-SNE analysis shows fact/hallucination are mixed in shallow layers and separated in deep layers. Intervention is meaningful only when $\mathbf{v}_{\text{vis}}^{\perp}$ is geometrically aligned with fact/hallucination states; backward search ensures the "deepest + still separable" optimal surgical entry point.

3. **Risk-Gated Dynamic Injection**:

    - Function: Ensures intervention occurs only when the model is actually drifting, avoiding disruption of normal tokens.
    - Mechanism: At each generation step, compute $R_t = R(\mathbf{h}_t^{(L^\*)})$, set $\lambda(t) = \alpha\cdot\mathbb{1}(R_t>\tau)$, and update $\tilde{\mathbf{h}}_t^{(L^\*)} = \mathbf{h}_t^{(L^\*)} + \lambda(t)\,\mathbf{v}_{\text{vis}}^{\perp(L^\*)}$; other tokens remain unchanged.
    - Design Motivation: Previous activation steering methods inject throughout, polluting already correct generations. Gating allows Revis to be a "zero-cost bypass" most of the time, with only occasional activation addition at risky moments, incurring almost no extra computation.

### Loss & Training
Completely training-free; only two hyperparameters, $\alpha$ (injection strength) and $k$ (fact quantile threshold), are set per model. In main results, Qwen2.5-VL-7B uses $\alpha=1.6, k=0.8$.

## Key Experimental Results

### Main Results

| Dataset | Metric | Best Baseline | Revis | Note |
|---------|--------|---------------|-------|------|
| POPE Random | F1 ↑ | 88.61 (VCD) | 91.43 | +2.82 |
| POPE Adversarial | F1 ↑ | 86.19 (VCD) | 87.63 | +1.44 |
| CHAIR | $C_S$ ↓ | 29.00 (AGLA) | 25.00 | Relative reduction ~14% |
| MME Total | ↑ | 2328.59 (AGLA) | 2345.30 | Slight general capability improvement |
| MM-Vet Total | ↑ | 70.18 (Regular) | 72.16 | VTI 56.38, this work +1.98 |

### Ablation Study

| Configuration | CHAIRS ↓ | MM-Vet ↑ | Description |
|---------------|----------|----------|-------------|
| Regular | 14.00 (max=64) | 70.18 | Original decoding |
| VTI (entangled vector + all layers) | 9.00 | 56.38 | Hallucination reduced but general capability collapses |
| $\mathbf{v}_{\text{raw}}$ direct injection | Better than VTI for $\alpha<0.4$ | Model collapses at $\alpha\approx 0.7$ | Confirms entanglement as root cause |
| Orthogonal vector $\mathbf{v}_{\text{vis}}^\perp$ | Stable even at high $\alpha$ | MM-Vet unaffected | Orthogonalization = necessary condition |
| Revis full (orthogonal + single-layer + gating) | 25.00 (max=512) | 72.16 | Final method |

### Key Findings
- Decomposing the "fact vs hallucination difference vector" reveals that hallucination's root cause is the "language prior" compressing visual information in deep layers; removing the language prior component allows aggressive visual enhancement without model collapse.
- Intervention does not require multiple layers—single-layer + risk gating outperforms all-layer VTI and double-forward VCD/M3ID.
- Consistent results across seven VLM backbones (Qwen2.5-VL 7B/32B, Qwen3-VL-8B, LLaVA-1.5, LLaVA-NeXT, InternVL3, InternVL3.5) indicate that "language prior entanglement" is a general issue in the LVLM paradigm, not an isolated case.

## Highlights & Insights
- Explicitly defining the "language prior vector" using counterfactual state space (GT/hallucination/no-image) and removing it via Gram-Schmidt is an elegant paradigm that brings mechanism interpretability tools directly into engineering optimization.
- The "single-layer + risk gating" approach is similar to gating in mixture-of-experts, but applied to hidden state intervention with virtually zero overhead; this "sparse latent surgical intervention" framework is transferable to other alignment/safety scenarios (e.g., toxicity, bias).
- Experiments observe that at extreme $\alpha$, the model outputs visual placeholders like "image.jpg," indicating that the pure visual direction truly pushes the model toward "image-dominant" extremes—a compelling causal evidence.

## Limitations & Future Work
- Restricting intervention to a single layer may miss complex hallucinations requiring multi-layer collaborative correction; future work could explore sparse multi-layer selection.
- The calibration set contains only 100 COCO images; robustness of the threshold across domains (medical, satellite, document) needs verification.
- Risk gating uses cosine similarity as the risk metric, without leveraging fine-grained image content signals; combining with object-level confidence may further reduce false positives.

## Related Work & Insights
- **vs VCD/M3ID**: These perform contrastive decoding at the logit layer, requiring double forward passes; Revis injects orthogonally at the hidden layer with almost zero extra cost.
- **vs AGLA/ONLY**: Rely on heuristic probes or auxiliary detection, sensitive to hyperparameters; Revis uses geometric + risk scoring for principled layer and threshold selection.
- **vs VTI**: Both are latent space steering, but VTI uses entangled vectors + all-layer injection, causing general capability collapse; Revis improves both hallucination and general capability via "orthogonal + sparse" dual strategies.
- **vs RLHF / DPO alignment**: Training is expensive and relies on preference data; Revis intervenes entirely at inference, with extremely low deployment cost.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of "counterfactual state + orthogonalization + risk gating" achieves latent space steering without loss of general capability for the first time.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five datasets × seven backbones × multiple baselines + causal probes; lacks some cross-domain evaluation.
- Writing Quality: ⭐⭐⭐⭐⭐ Very rigorous logic from mechanism analysis to method to experiments, with well-coordinated figures and tables.
- Value: ⭐⭐⭐⭐ Zero training, zero extra forward passes, plug-and-play; highly practical for LVLM deployment scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Referring Multiple Regions with Large Multimodal Models via Contextual Latent Steering](referring_multiple_regions_with_large_multimodal_models_via_contextual_latent_st.md)
- [\[ICML 2026\] Instruction Lens Score: Your Instruction Contributes a Powerful Object Hallucination Detector for Multimodal Large Language Models](instruction_lens_score_your_instruction_contributes_a_powerful_object_hallucinat.md)
- [\[ICML 2026\] Latent Reasoning VLA: Latent Thinking and Prediction for Vision-Language-Action Models](latent_reasoning_vla_latent_thinking_and_prediction_for_vision-language-action_m.md)
- [\[ICLR 2026\] Dynamic Multimodal Activation Steering for Hallucination Mitigation in Large Vision-Language Models](../../ICLR2026/multimodal_vlm/dynamic_multimodal_activation_steering_for_hallucination_mitigation_in_large_vis.md)
- [\[ICML 2026\] Vision-aligned Latent Reasoning for Multi-modal Large Language Model](vision-aligned_latent_reasoning_for_multi-modal_large_language_model.md)

</div>

<!-- RELATED:END -->
