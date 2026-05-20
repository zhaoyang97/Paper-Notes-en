---
title: >-
  [Paper Note] Event-T2M: Event-level Conditioning for Complex Text-to-Motion Synthesis
description: >-
  [ICLR 2026][Image Generation][Text-to-Motion Generation] This paper proposes the Event-T2M framework, which decomposes text prompts into event-level atomic actions and injects them into a Conformer-based diffusion model…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "Text-to-Motion Generation"
  - "Event-level Conditioning"
  - "Diffusion Models"
  - "Compositional Motion"
  - "Conformer"
date: 2026-05-08
content_hash: fc3cbee7b9402648
---

# Event-T2M: Event-level Conditioning for Complex Text-to-Motion Synthesis

**Conference**: ICLR 2026
**arXiv**: [2602.04292](https://arxiv.org/abs/2602.04292)  
**Code**: Available (project page)  
**Area**: Image Generation
**Keywords**: Text-to-Motion Generation, Event-level Conditioning, Diffusion Models, Compositional Motion, Conformer

## TL;DR

This paper proposes the Event-T2M framework, which decomposes text prompts into event-level atomic actions and injects them into a Conformer-based diffusion model via a TMR encoder and an Event-level Cross-Attention (ECA) module, significantly improving generation quality and semantic alignment for complex multi-event motion synthesis.

## Background & Motivation

Although the text-to-motion generation field has achieved remarkable progress on benchmarks such as HumanML3D and KIT-ML (with FID optimized to two decimal places), these benchmarks are predominantly composed of simple, single-action descriptions, obscuring a critical issue: when faced with complex multi-action prompts (e.g., "run forward, then stop, then wave"), existing systems tend to merge, skip, or reorder actions.

The root causes are:

**Existing methods compress the entire prompt into a single embedding**: Most approaches use the CLIP [EOS] token as a global representation, discarding temporal information.

**Benchmarks do not distinguish simple from complex prompts**: They cannot evaluate model performance as compositional complexity increases.

**CLIP is pretrained on image-text pairs**: It lacks supervision signals for the temporal continuity and event transitions inherent in motion.

## Method

### Overall Architecture

Event-T2M recasts text-to-motion generation as an event-level conditional generation problem, comprising three key components:

1. **LLM Event Decomposition**: Gemini 2.5 Flash is used to segment the input text prompt $W$ into an event sequence $\{C_k\}_{k=1}^K$.
2. **TMR Event Encoding**: A motion-aware TMR encoder maps each event to an event token.
3. **ECA Injection**: Event information is fused into Conformer blocks via an Event-level Cross-Attention module.

**Formal definition of an event**: An event is the minimal semantically self-contained action or state change in a text prompt, whose execution can be temporally isolated and mapped to a contiguous motion segment. For example, "A person steps backward, jumps up, runs forward, then runs backward" is decomposed into four events.

### Key Designs

#### 1. Event Token Generation

Each event $C_k$ is encoded by the TMR encoder into an event token:

$$E_k = f_{\text{TMR}}(C_k), \quad E_k \in \mathbb{R}^{D_y}$$

All event tokens are stacked to form $E \in \mathbb{R}^{K \times D_y}$. A global text token $G = f_{\text{TMR}}(W)$ is additionally introduced as a holistic semantic supplement, providing a global semantic fallback when local event cues are ambiguous.

#### 2. Event-T2M Block Architecture

The model stacks $N$ identical blocks, each containing 8 update steps:

| Step | Module | Function |
|------|--------|----------|
| (1) | LIMM | Local information modeling (depthwise separable convolution) |
| (2) | ATII | Adaptive text information injection (channel-wise gating) |
| (3) | FFN | Feed-forward network (0.5 residual weight) |
| (4) | ConformerSA | Self-attention (global temporal dependencies) |
| (5) | **ECA** | **Event-level cross-attention (core contribution)** |
| (6) | ConformerConv | Depthwise separable convolution (local dynamics) |
| (7) | FFN | Feed-forward network (0.5 residual weight) |
| (8) | LIMM | Local information modeling |

#### 3. Event-level Cross-Attention (ECA)

ECA is the core innovation, replacing the standard self-attention in Conformer blocks with a motion-to-text cross-attention mechanism:

- **Query**: derived from motion tokens $x_t^{\text{ctx}}$
- **Key/Value**: derived from event tokens $E$

$$Q_m = x_t^{\text{ctx}} W^Q, \quad K_e = E W^K, \quad V_e = E W^V$$

$$A^{(h)} = \text{softmax}\left(\frac{Q_m^{(h)} (K_e^{(h)})^\top}{\sqrt{d_h}}\right)$$

A learnable scaling factor $\gamma$ (initialized near zero) is used to ensure training stability: $\text{ECA}(x_t, E) = \gamma \cdot \text{Dropout}(Z)$.

#### 4. ATII Adaptive Text Injection

ATII fuses the global text embedding $G$ with local motion states via channel-wise gating:

$$\hat{g}_j = \text{Sigmoid}(W_c[m'_j \oplus G]) \odot G$$

The motion sequence is first downsampled by a factor of $S$, and global semantics are then adaptively filtered through the gating mechanism.

### Loss & Training

A standard conditional denoising diffusion objective is adopted, training the denoiser $\varphi_\theta$ to recover the clean motion $x_0$ from the noisy motion $x_t$:

$$\mathcal{L}(\theta) = \mathbb{E}_{x_0, t, \epsilon}\left[\|x_0 - \varphi_\theta(x_t, t, G, E)\|_2^2\right]$$

- During training, text conditioning is randomly dropped with probability $\tau$ to enable Classifier-Free Guidance (CFG).
- Inference uses 10-step DDPM for efficient generation.
- A residual weight of 0.5 is applied to FFN layers, following the intuition of the Macaron-style architecture.

## Key Experimental Results

### Main Results

**Table 1: HumanML3D Standard Benchmark**

| Method | R-Prec Top-1↑ | R-Prec Top-3↑ | FID↓ | MM-Dist↓ |
|--------|--------------|--------------|------|----------|
| MoMask | 0.521 | 0.807 | 0.045 | 2.958 |
| MoGenTS | 0.529 | 0.812 | 0.033 | 2.867 |
| **Event-T2M** | **0.562** | **0.842** | 0.056 | **2.711** |

**Table 3: HumanML3D-E Event-Stratified Benchmark (≥4 events)**

| Method | R-Prec Top-1↑ | FID↓ | MM-Dist↓ |
|--------|--------------|------|----------|
| MoMask | 0.441 | 0.418 | 3.205 |
| MoGenTS | 0.420 | 0.423 | 3.241 |
| **Event-T2M** | **0.466** | 0.265 | **3.063** |

Event-T2M surpasses MoGenTS by approximately 4.6 percentage points in R-Precision Top-1 under the ≥4 events condition, demonstrating its advantage in complex compositional scenarios.

### Ablation Study

**Text Encoder Comparison (TMR vs. CLIP)**: Under event-level conditioning, the TMR encoder outperforms CLIP across all event complexity levels.

**Conditioning Strategy Comparison — Event-level vs. Token-level**:

| Conditioning | R-Prec Top-1↑ (≥2 events) | FID↓ |
|-------------|--------------------------|------|
| Token-level | 0.521 | 0.082 |
| **Event-level** | **0.536** | 0.079 |

Event-level encoding outperforms token-level encoding across all complexity conditions.

### Key Findings

1. **Advantage amplifies with increasing event complexity**: As the number of events grows from ≥1 to ≥4, baseline methods degrade sharply while Event-T2M remains robust.
2. **Efficiency advantage**: Under the ≥4 events condition, Event-T2M achieves high accuracy with a comparatively smaller model size.
3. **Human evaluation validation**: The reasonableness of event definitions, the reliability of HumanML3D-E, and the overall generation quality all received high ratings from human evaluators.

## Highlights & Insights

1. The **formal definition of events** is broadly generalizable — the idea of decomposing complex prompts into minimal semantically self-contained units is transferable to other conditional generation tasks.
2. **TMR as a replacement for CLIP**: Substituting the domain-agnostic CLIP with a motion-language-aligned TMR encoder provides a paradigmatic reference for domain-specific conditional generation.
3. **HumanML3D-E benchmark**: The first evaluation benchmark stratified by event count, filling the gap in compositional complexity assessment.
4. **Learnable scaling factor $\gamma$**: Initializing $\gamma$ near zero in ECA to ensure training stability is a practically useful engineering technique.

## Limitations & Future Work

1. LLM-based event decomposition relies on an external model (Gemini 2.5 Flash), introducing additional inference dependencies and latency.
2. Transition quality between events is not explicitly modeled.
3. Validation is limited to HumanML3D/KIT-ML; generalization experiments on larger-scale datasets are absent.
4. FID still has room for improvement as event count increases.
5. End-to-end joint optimization of event decomposition and motion generation is worth exploring.

## Related Work & Insights

- **GraphMotion**: Enhances text representations with semantic graphs, but evaluation is limited.
- **AttT2M**: Body-part attention combined with global-local motion-text attention.
- **MMM**: Masked motion modeling with joint encoding of text and motion.
- **Light-T2M**: The source of inspiration for the ATII module.
- Insight: The event-level decomposition paradigm can be transferred to tasks such as text-to-video and text-to-dance generation.

## Rating

- Novelty: ⭐⭐⭐⭐ — Event-level conditioning is a concise and effective new perspective.
- Technical Contribution: ⭐⭐⭐⭐ — ECA + TMR + event-stratified benchmark form a cohesive triple contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Standard benchmark + stratified benchmark + ablation + human evaluation.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with well-motivated problem formulation.
- Overall Recommendation: ⭐⭐⭐⭐ — A noteworthy contribution, particularly valuable for multi-action generation scenarios.

## Background & Motivation

## Core Problem

## Method

## Key Experimental Results

## Highlights & Insights

## Limitations & Future Work

## Related Work & Insights

## Inspiration & Connections

## Rating

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] ReAlign: Text-to-Motion Generation via Step-Aware Reward-Guided Alignment](../../AAAI2026/image_generation/realign_text-to-motion_generation_via_step-aware_reward-guided_alignment.md)
- [\[ICLR 2026\] Localized Concept Erasure in Text-to-Image Diffusion Models via High-Level Representation Misdirection](localized_concept_erasure_in_text-to-image_diffusion_models_via_high-level_repre.md)
- [\[ICLR 2026\] COSMO-INR: Complex Sinusoidal Modulation for Implicit Neural Representations](cosmo-inr_complex_sinusoidal_modulation_for_implicit_neural_representations.md)
- [\[ICCV 2025\] Efficient Input-Level Backdoor Defense on Text-to-Image Synthesis via Neuron Activation Variation](../../ICCV2025/image_generation/efficient_input-level_backdoor_defense_on_text-to-image_synthesis_via_neuron_act.md)
- [\[CVPR 2026\] High-Fidelity Diffusion Face Swapping with ID-Constrained Facial Conditioning](../../CVPR2026/image_generation/high-fidelity_diffusion_face_swapping_with_id-constrained_facial_conditioning.md)

</div>

<!-- RELATED:END -->
