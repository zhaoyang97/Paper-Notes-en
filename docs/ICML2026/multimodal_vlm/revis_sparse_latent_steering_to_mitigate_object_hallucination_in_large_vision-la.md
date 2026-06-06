---
title: >-
  [Paper Note] Revis: Sparse Latent Steering to Mitigate Object Hallucination in Large Vision-Language Models
description: >-
  [ICML 2026][Multimodal VLM][Object Hallucination] This paper redefines LVLM hallucinations as "missing visual information suppressed by language priors." It uses orthogonal projection to remove language priors from raw v…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Object Hallucination"
  - "Latent Space Steering"
  - "Orthogonal Projection"
  - "Sparse Intervention"
  - "Mechanistic Interpretability"
date: 2026-05-08
content_hash: 96101cd78f508385
---

# Revis: Sparse Latent Steering to Mitigate Object Hallucination in Large Vision-Language Models

**Conference**: ICML 2026  
**arXiv**: [2602.11824](https://arxiv.org/abs/2602.11824)  
**Code**: https://github.com/antgroup/Revis (Available)  
**Area**: Multimodal VLM / Hallucination Mitigation  
**Keywords**: Object Hallucination, Latent Space Steering, Orthogonal Projection, Sparse Intervention, Mechanistic Interpretability

## TL;DR
This paper redefines LVLM hallucinations as "missing visual information suppressed by language priors." It uses orthogonal projection to remove language priors from raw visual directions to obtain a "pure visual vector," then applies risk-gating for sparse intervention at a single optimal depth. This training-free method reduces the CHAIRS hallucination rate by ~19% while preserving MM-Vet general capabilities.

## Background & Motivation
**Background**: LVLMs have demonstrated strong multimodal reasoning capabilities, but "object hallucination"—where the model describes plausible but non-existent details—remains a persistent reliability issue. Existing mitigation methods focus on alignment during training (LLaVA-RLHF, HA-DPO, OPA-DPO) and inference-time intervention (contrastive decoding like VCD/M3ID, logit correction like AGLA/ONLY, and activation steering like VTI).

**Limitations of Prior Work**: Training-based alignment is heavy and depends on large-scale preference data. Contrastive decoding requires at least double forward passes to suppress hallucinations. Logit correction uses heuristic probes with poor robustness. VTI, the only method that requires neither retraining nor double forward passes, directly injects static offsets across all layers; while it reduces CHAIRS from 14% to 9%, it causes MM-Vet scores to drop from 70.18 to 56.38, nearly destroying general reasoning capabilities.

**Key Challenge**: Existing intervention directions are essentially "fact-hallucination" difference vectors. While they point toward visual information, they are entangled with "language priors" (the directions the model tends to guess when no image is present). Simply scaling this entangled vector by an intensity $\alpha$ also amplifies the language prior, causing the model to collapse into repetitive or empty outputs when $\alpha \approx 0.7$.

**Goal**: To find an intervention method that "activates vision without amplifying priors" and precisely selects intervention depth, reducing hallucinations without sacrificing general capabilities—all without retraining or double forward passes.

**Key Insight**: The authors analyze the latent space geometry using $[\text{EOS}]$ hidden states from five counterfactual states (GT / Hallucination / No-image GT / No-image Hallucination / No-image Unknown). They confirm that while deep layers can linearly separate facts from hallucinations, the raw vector $\mathbf{v}_{\text{raw}} = \mathbf{h}_{\text{gt}} - \mathbf{h}_{\emptyset\_\text{gt}}$ and the "language prior vector" $\mathbf{v}_{\text{prior}} = \mathbf{h}_{\emptyset\_\text{hall}} - \mathbf{h}_{\emptyset\_\text{unk}}$ share high cosine similarity in deep layers, identifying entanglement as the root cause of VTI's failure.

**Core Idea**: Use Gram-Schmidt to project $\mathbf{v}_{\text{raw}}$ onto the orthogonal complement of $\mathbf{v}_{\text{prior}}$, obtaining a "pure visual vector" $\mathbf{v}_{\text{vis}}^\perp$. Then, use a calibration-set-based layer-wise risk score to identify the deepest separable layer $L^*$. Finally, during inference, perform a surgery-like intervention—"sparse + single-layer + orthogonal"—by injecting the correction only when the "hallucination risk score exceeds a threshold."

## Method

### Overall Architecture
A three-stage training-free pipeline: In Stage 1, $N=100$ sample pairs are used to calculate $\mathbf{v}_{\text{raw}}^{(\ell)}$ and $\mathbf{v}_{\text{prior}}^{(\ell)}$ for each layer, and Gram-Schmidt is applied to obtain $\mathbf{v}_{\text{vis}}^{\perp(\ell)}$. In Stage 2, a POPE-style Q&A set on 100 COCO images is used to collect hidden state sets for facts/hallucinations. A top-down search identifies the deepest layer $L^*$ where the risk score $R(\mathbf{h}) = -\cos(\mathbf{h}, \mathbf{v}_{\text{vis}}^{\perp(\ell)})$ satisfies $R(\mathcal{H}_{\text{hall}}) > R(\mathcal{H}_{\text{fact}})$; the threshold $\tau$ is determined by the $k$-th percentile of the fact set. In Stage 3, for each token during inference, $R_t$ is calculated. If $R_t > \tau$, $\alpha\mathbf{v}_{\text{vis}}^{\perp(L^*)}$ is added at layer $L^*$; otherwise, original activations are maintained.

### Key Designs

1. **Purifying Visual Vectors via Orthogonal Projection**:
    - **Function**: Remove components collinear with language priors from the raw visual direction to obtain a "pure visual steering direction" that does not cause collapse when amplified.
    - **Mechanism**: Define $\mathbf{v}_{\text{raw}}^{(\ell)} = \mathbb{E}[\mathbf{h}_{\text{gt}}^{(\ell)} - \mathbf{h}_{\emptyset\_\text{gt}}^{(\ell)}]$ and $\mathbf{v}_{\text{prior}}^{(\ell)} = \mathbb{E}[\mathbf{h}_{\emptyset\_\text{hall}}^{(\ell)} - \mathbf{h}_{\emptyset\_\text{unk}}^{(\ell)}]$ based on counterfactual states, then apply Gram-Schmidt: $\mathbf{v}_{\text{vis}}^{\perp(\ell)} = \mathbf{v}_{\text{raw}}^{(\ell)} - \frac{\mathbf{v}_{\text{raw}}^{(\ell)}\cdot\mathbf{v}_{\text{prior}}^{(\ell)}}{\|\mathbf{v}_{\text{prior}}^{(\ell)}\|^2}\mathbf{v}_{\text{prior}}^{(\ell)}$.
    - **Design Motivation**: Causal probing experiments show that the pure visual vector maintains generation stability even at extreme $\alpha$ intensities, whereas the raw vector causes model collapse (infinite repetition/empty output) at $\alpha \approx 0.7$. Orthogonalization decouples "adding vision" from "adding priors" at the operator level.

2. **Calibration-based Sparse Single-layer Selection**:
    - **Function**: Select a single layer for intervention to avoid accumulated bias and computational waste from multi-layer injections.
    - **Mechanism**: Construct a calibration set $\mathcal{D}_{\text{cal}}$ (COCO + POPE-style existence questions) to extract $\mathcal{H}_{\text{fact}}, \mathcal{H}_{\text{hall}}$ for each layer. Define risk as $R(\mathbf{h}) = -\cos(\mathbf{h}, \mathbf{v}_{\text{vis}}^{\perp(\ell)})$ and search backward from $L$ to $1$. The first layer $\ell$ satisfying $R(\mathcal{H}_{\text{hall}}) - R(\mathcal{H}_{\text{fact}}) > 0$ is chosen as $L^*$. The threshold $\tau$ is set based on the $k$-th percentile of the fact set at that layer.
    - **Design Motivation**: t-SNE analysis shows that facts and hallucinations overlap in shallow layers and only separate in deep layers. Intervention is meaningful only when $\mathbf{v}_{\text{vis}}^{\perp}$ has correct geometric alignment with states; backward search ensures the selection of the "deepest yet still separable" optimal surgical entry point.

3. **Dynamic Injection via Risk-Gating**:
    - **Function**: Ensure intervention occurs only when the model begins to "drift," avoiding damage to normal tokens.
    - **Mechanism**: For each step of generation, calculate $R_t = R(\mathbf{h}_t^{(L^*)})$, determine $\lambda(t) = \alpha\cdot\mathbb{1}(R_t > \tau)$, and execute $\tilde{\mathbf{h}}_t^{(L^*)} = \mathbf{h}_t^{(L^*)} + \lambda(t)\mathbf{v}_{\text{vis}}^{\perp(L^*)}$. Other tokens remain unchanged.
    - **Design Motivation**: Unlike prior activation steering that defaults to "full-course injection" (which contaminates correct generation), gating makes Revis a "zero-cost bypass" most of the time, performing a single activation addition only at high-risk moments.

### Loss & Training
Completely training-free. Only two hyperparameters, $\alpha$ (injection intensity) and $k$ (fact percentile threshold), are set per model. For Qwen2.5-VL-7B, $\alpha=1.6$ and $k=0.8$ were used in main results.

## Key Experimental Results

### Main Results

| Dataset | Metric | Best Baseline | Revis | Gain/Notes |
|--------|------|----------|-------|------|
| POPE Random | F1 ↑ | 88.61 (VCD) | 91.43 | +2.82 |
| POPE Adversarial | F1 ↑ | 86.19 (VCD) | 87.63 | +1.44 |
| CHAIR | $C_S$ ↓ | 29.00 (AGLA) | 25.00 | ~14% relative reduction |
| MME Total | ↑ | 2328.59 (AGLA) | 2345.30 | Slight improvement in general capability |
| MM-Vet Total | ↑ | 70.18 (Regular) | 72.16 | VTI: 56.38; Ours: +1.98 |

### Ablation Study

| Configuration | CHAIRS ↓ | MM-Vet ↑ | Explanation |
|------|---------|---------|------|
| Regular | 14.00 (max=64) | 70.18 | Original decoding |
| VTI (Entangled + All layers) | 9.00 | 56.38 | Hallucinations drop, but general capability collapses |
| $\mathbf{v}_{\text{raw}}$ direct injection | Better than VTI at $\alpha < 0.4$ | Model collapses at $\alpha \approx 0.7$ | Confirms entanglement as root cause of collapse |
| Orthogonal vector $\mathbf{v}_{\text{vis}}^\perp$ | Stable at high $\alpha$ | MM-Vet preserved | Orthogonalization = Necessary condition |
| Revis Full (Orthogonal + Single + Gated) | 25.00 (max=512) | 72.16 | Final method |

### Key Findings
- Decomposition of "fact vs. hallucination difference vectors" reveals that the root cause of hallucination is the "language prior" suppressing visual information in deep layers. Once the language prior component is removed, aggressive visual reinforcement does not trigger model collapse.
- Intervention does not require multiple layers—single-layer injection combined with risk-gating is sufficient to outperform all-layer VTI and double-forward methods like VCD/M3ID.
- Consistent results across seven different VLM backbones (Qwen2.5-VL 7B/32B, Qwen3-VL-8B, LLaVA-1.5, LLaVA-NeXT, InternVL3, InternVL3.5) suggest that "language prior entanglement" is a systemic issue in the LVLM paradigm rather than an isolated case.

## Highlights & Insights
- Using counterfactual state spaces (GT/Hallucination/No-image) to explicitly define "language prior vectors" and then removing them via Gram-Schmidt is a elegant paradigm for applying mechanistic interpretability tools to engineering optimization.
- The "single-layer + risk-gating" approach is similar to gating in Mixture-of-Experts but applied to hidden state intervention with nearly zero overhead. This "latent-space sparse surgery" framework could be adapted to other alignment/safety intervention scenarios (e.g., toxicity, bias).
- The observation that models output visual placeholders like "image.jpg" under extreme $\alpha$ provides compelling causal evidence that the pure visual direction truly pushes the model toward a "vision-dominant" state.

## Limitations & Future Work
- Forcing the intervention to a single layer might miss complex hallucinations that require multi-layer collaborative correction. Sparse multi-layer selection could be explored.
- The calibration set relies on only 100 COCO images; threshold robustness during cross-domain generalization (medical, satellite, documents) needs verification.
- Risk-gating uses cosine similarity as a metric without leveraging fine-grained signals from image content; integration with object-level confidence might further reduce false positives.

## Related Work & Insights
- **vs. VCD/M3ID**: These perform contrastive decoding at the logit layer, requiring double forward passes. Revis performs orthogonal injection at the hidden layer with almost zero extra cost.
- **vs. AGLA/ONLY**: These rely on heuristic probes or auxiliary detection and are sensitive to hyperparameters. Revis principle-based layer and threshold selection via geometry and risk scores.
- **vs. VTI**: Both are latent steering methods, but VTI's use of entangled vectors and all-layer injection leads to a collapse in general capabilities. Revis improves both hallucination reduction and general performance via "orthogonal + sparse" strategies.
- **vs. RLHF / DPO Alignment**: Training is expensive and requires preference data. Revis is purely inference-time, offering extremely low deployment costs.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The combination of "counterfactual states + orthogonalization + risk-gating" is the first to push latent steering to a level where general capabilities do not degrade.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 5 datasets × 7 backbones × multiple baselines + causal probing. Lacks some cross-domain evaluation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Highly rigorous logic from mechanistic analysis to methodology and experiments. Excellent use of figures and tables.
- **Value**: ⭐⭐⭐⭐ Training-free, zero extra forward passes, and plug-and-play. Highly practical for production LVLM scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Instruction Lens Score: Your Instruction Contributes a Powerful Object Hallucination Detector for Multimodal Large Language Models](instruction_lens_score_your_instruction_contributes_a_powerful_object_hallucinat.md)
- [\[ICML 2026\] Adaptive Residual-Update Steering for Low-Overhead Hallucination Mitigation in Large Vision Language Models](adaptive_residual-update_steering_for_low-overhead_hallucination_mitigation_in_l.md)
- [\[ICML 2026\] Referring Multiple Regions with Large Multimodal Models via Contextual Latent Steering](referring_multiple_regions_with_large_multimodal_models_via_contextual_latent_st.md)
- [\[ICLR 2026\] Dynamic Multimodal Activation Steering for Hallucination Mitigation in Large Vision-Language Models](../../ICLR2026/multimodal_vlm/dynamic_multimodal_activation_steering_for_hallucination_mitigation_in_large_vis.md)
- [\[ICML 2026\] Vision-aligned Latent Reasoning for Multi-modal Large Language Model](vision-aligned_latent_reasoning_for_multi-modal_large_language_model.md)

</div>

<!-- RELATED:END -->
