---
title: >-
  [Paper Note] JailBound: Jailbreaking Internal Safety Boundaries of Vision-Language Models
description: >-
  [NeurIPS 2025][Multimodal VLM][jailbreak] Inspired by the Eliciting Latent Knowledge (ELK) framework, this paper is the first to reveal that VLMs possess approximable safety decision boundaries in the latent space of fus…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "jailbreak"
  - "safety boundary"
  - "latent space attack"
  - "ELK"
  - "cross-modal perturbation"
date: 2026-05-08
content_hash: 7b520003da0ed002
---

# JailBound: Jailbreaking Internal Safety Boundaries of Vision-Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2505.19610](https://arxiv.org/abs/2505.19610)
**Code**: To be confirmed
**Area**: Multimodal VLM / AI Safety / Adversarial Attacks
**Keywords**: jailbreak, safety boundary, latent space attack, ELK, cross-modal perturbation

## TL;DR
Inspired by the Eliciting Latent Knowledge (ELK) framework, this paper is the first to reveal that VLMs possess approximable safety decision boundaries in the latent space of fusion layers. It proposes JailBound, a two-stage attack framework comprising Safety Boundary Probing and Safety Boundary Crossing, which jointly optimizes image and text adversarial perturbations to cross this boundary. JailBound achieves average attack success rates of 94.32% and 67.28% in white-box and black-box settings, respectively, significantly surpassing the state of the art.

## Background & Motivation

**Background**: VLMs acquire powerful multimodal capabilities by integrating visual encoders with LLMs, yet the introduction of the visual modality substantially enlarges the attack surface. Existing jailbreak methods include gradient-based white-box attacks and query-feedback-based black-box attacks.

**Limitations of Prior Work**: (a) The absence of explicit attack objectives causes gradient optimization to fall into local optima, lacking precise directional guidance; (b) most methods treat the visual and textual modalities in a decoupled manner, neglecting cross-modal interactions.

**Key Challenge**: Although safety alignment in VLMs suppresses harmful outputs, the models still internally encode safety-relevant knowledge—analogous to the "the model knows but does not say" phenomenon identified in ELK research. This latent knowledge exposes exploitable structure for attacks.

**Key Insight**: If VLMs possess a safety/unsafe decision boundary in the latent representations of fusion layers, then precisely locating and crossing this boundary would enable systematic circumvention of safety mechanisms.

**Core Idea**: First, use a linear classifier to probe the safety decision hyperplane in fusion layers; then drive image and text adversarial perturbations across this boundary via a three-objective joint optimization.

## Method

### Overall Architecture
JailBound operates in two stages:
- **Stage 1 – Safety Boundary Probing**: Train a logistic regression classifier at each fusion layer to approximate the safety decision hyperplane, obtaining the normal vector $v^{(l)}$ and the crossing distance $\varepsilon^{(l)}$.
- **Stage 2 – Safety Boundary Crossing**: Jointly optimize the visual perturbation $\delta_v^{\text{input}}$ and the text suffix $X_t^{\text{suffix}}$ so that the fused representation crosses the decision boundary into the unsafe region.

### Key Designs

1. **Safety Boundary Probing**:

    - **Function**: Approximate the safety decision hyperplane at each fusion layer.
    - **Mechanism**: Construct a dataset $\mathbb{D} = \{(h^{(i)}, y^{(i)})\}$, where $h^{(i)} = \phi(x_v^{(i)}, x_t^{(i)})$ is the fused representation and $y^{(i)} \in \{0,1\}$ is the safety label. A logistic regression classifier $P_m(x_v, x_t) = \sigma(w^\top \phi(x_v, x_t) + b)$ is trained. The decision boundary is defined as $\mathcal{B}^{(l)}(w,b) = \{h^{(l)} | (w^{(l)})^\top h^{(l)} + b^{(l)} = 0\}$. The normal vector is $v^{(l)} = w^{(l)}/\|w^{(l)}\|_2$, and the crossing distance is $\varepsilon^{(i)} = |\sigma^{-1}(P_0) - (w^\top h^{(i)} + b)|/\|w\|_2$.
    - **Design Motivation**: A 100% classification accuracy demonstrates that a clear, linearly separable safety boundary genuinely exists inside VLMs. This provides a precise attack target that fundamentally resolves the problem of gradient optimization lacking directional guidance.

2. **Adversarial Alignment Loss $\mathcal{L}_{\text{align}}$**:

    - **Function**: Guide the perturbed fused representation toward the target region.
    - **Mechanism**: $\mathcal{L}_{\text{align}}^{(l)} = \|\phi^{(l)}(\tilde{x}_v, \tilde{x}_t) - h_{\text{target}}^{(l)}\|_2^2$, where $h_{\text{target}}^{(l)} = \phi^{(l)}(x_v, x_t) - \varepsilon^{(l)} \cdot v^{(l)}$ is the original representation shifted along the normal vector direction.
    - **Design Motivation**: Provides a precise optimization target, avoiding blind gradient search.

3. **Geometric Boundary Loss $\mathcal{L}_{\text{geo}}$**:

    - **Function**: Ensure that the perturbation direction follows the trajectory of the normal vector.
    - **Mechanism**: $\mathcal{L}_{\text{geo}}^{(l)} = \|\frac{\Delta h^{(l)}}{\|\Delta h^{(l)}\|_2} - v^{(l)}\|_2^2$, where $\Delta h^{(l)} = \phi^{(l)}(\tilde{x}_v, \tilde{x}_t) - \phi^{(l)}(x_v, x_t)$.
    - **Design Motivation**: Prevents the optimization from taking indirect paths, ensuring geometrically optimal perturbation efficiency.

4. **Semantic Preservation Loss $\mathcal{L}_{\text{sem}}$**:

    - **Function**: Constrain perturbation magnitude to maintain semantic consistency.
    - **Mechanism**: $\mathcal{L}_{\text{sem}} = \|\delta_v^{\text{input}}\|_2^2 + \mathcal{L}_{\text{suffix}}(X_t^{\text{suffix}})$, with the visual perturbation constrained by an $L_\infty$ norm of $\leq 8/255$.

5. **Cross-Modal Joint Optimization**:

    - **Visual perturbation**: Gradient descent in continuous space — $\delta_v^{\text{input}(k+1)} = \Pi_{\Gamma_v}[\delta_v^{\text{input}(k)} - \eta_v \nabla_{\delta_v} \mathcal{L}]$.
    - **Text perturbation**: Compute the embedding-space gradient $\delta_t^{\text{emb}} = -\eta_t \nabla_{x_t}\mathcal{L}$, then select real tokens via nearest-neighbor search: $t_j^{\text{suffix}} = \arg\min_{v\in V} \|E(v) - (x_t^{(j)} + \delta_t^{\text{emb}(j)})\|_2$.
    - **Design Motivation**: Simultaneously perturbing both modalities exploits cross-modal interactions to produce stronger attacks than single-modality approaches.

### Loss & Training
- Total loss: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{align}} + \lambda_1 \mathcal{L}_{\text{sem}} + \lambda_2 \mathcal{L}_{\text{geo}}$, with $\lambda_1=2.0$, $\lambda_2=1.0$.
- Safety threshold $P_0 = 0.3$; visual learning rate $\eta_v=0.001$; text learning rate $\eta_t=0.0005$.
- Text suffix length: 20 tokens; optimization: 100–150 iterations.
- Evaluated on MM-SafetyBench (13 categories of prohibited content, 1,719 samples).

## Key Experimental Results

### Main Results — White-Box Attack Success Rate (ASR)

| Category | Method | Llama-3.2-11B | Qwen2.5-VL-7B | MiniGPT-4 |
|---|---|---|---|---|
| Illegal Activity | Baseline (I0+T0) | 51.47% | 2.94% | 42.65% |
| | Joint Attack (I1+T1) | 88.24% | 64.71% | 95.59% |
| | **JailBound {I1,T1}** | **95.59%** | **82.35%** | **100.00%** |
| Hate Speech | Baseline | 63.16% | 12.28% | 56.14% |
| | **JailBound** | **95.61%** | **89.47%** | **96.49%** |
| Physical Harm | Baseline | 70.30% | 28.71% | 43.56% |
| | **JailBound** | **97.03%** | **87.13%** | **97.03%** |

### Attack Configuration Comparison

| Configuration | Description | White-Box Avg. ASR |
|---|---|---|
| I0+T0 | No-attack baseline | ~50% |
| I0+T1 | Text-only attack | ~75% |
| I1+T0 | Vision-only attack | ~72% |
| I1+T1 | Joint (non-iterative) | ~87% |
| **{I1,T1}** | **Iterative joint (JailBound)** | **~94%** |

### Black-Box Transfer Attacks

| Target Model | JailBound ASR | Gain over Prev. SOTA |
|---|---|---|
| GPT-4o | 75.24% | +21.13% |
| Gemini 2.0 Flash | 70.06% | Significant |
| Claude 3.5 Sonnet | 56.55% | Significant |

### Key Findings
- Iterative joint attack {I1,T1} outperforms non-iterative joint I1+T1 by approximately 7% on average, validating the cross-reinforcement effect of iterative optimization.
- On Qwen2.5-VL-7B, where baseline ASR is extremely low (below 10% for some categories), JailBound still achieves 80%+, demonstrating its effectiveness against strongly aligned models.
- Safety Boundary Probing achieves 100% classification accuracy across all fusion layers, confirming the linear separability of the safety boundary.
- Black-box transferability is remarkably strong, achieving 75.24% ASR on GPT-4o—far exceeding prior methods.

## Highlights & Insights
- **Transferring ELK to VLM safety** is a highly insightful contribution: applying the theory that "models internally know the truth" to the safety domain reveals that VLM safety decisions reside along clear linear boundaries.
- **The three-objective design is elegant**: alignment provides direction, geometric loss enforces constraints, and semantic preservation ensures fidelity—the three components are complementary and form a robust optimization framework.
- The **iterative alternating optimization** strategy handles two fundamentally different optimization problems: continuous (image) and discrete (text).
- The method uncovers a deep security vulnerability: even after strong safety alignment, the decision boundary remains linearly separable, allowing attackers to locate and cross it with precision.

## Limitations & Future Work
- Public disclosure of the attack methodology may be misused, though as security research this is a necessary cost for advancing defensive work.
- The white-box stage requires full model access, whereas real-world deployments often permit only API access.
- The text suffix length is fixed at 20 tokens, which may lack flexibility.
- The 100% decision-boundary classification accuracy may indicate that safety alignment is overly simplistic—could nonlinear safety embeddings serve as a stronger defense?
- The paper lacks discussion of defensive applications—how could probing results be leveraged to harden safety boundaries?

## Related Work & Insights
- **vs. VAJM**: VAJM relies solely on visual adversarial perturbations to bypass safety mechanisms but lacks precise directional guidance and is prone to local optima; JailBound uses the probed boundary to provide a precise attack target.
- **vs. SCAV**: SCAV manipulates embeddings in the LLM latent space but is limited to a single modality, unable to perturb visual inputs or transfer to black-box settings; JailBound is cross-modal and exhibits strong transferability.
- **vs. FigStep**: FigStep employs typographic images to bypass text filters, operating at the prompt-engineering level; JailBound attacks from the latent space level, making it more fundamental and effective.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — First to introduce ELK theory into VLM safety attacks; the safety boundary probing concept is novel and practically motivated.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers 6 VLMs, 13 safety categories, and both white-box and black-box settings comprehensively; lacks comparison with defensive methods.
- **Writing Quality**: ⭐⭐⭐⭐ — The framework is clearly presented with complete mathematical formalization, though the notation is occasionally dense and could be streamlined.
- **Value**: ⭐⭐⭐⭐⭐ — Carries significant warning implications for VLM security, exposes the fragility of linear safety boundaries, and motivates research into stronger defenses.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] IDEATOR: Jailbreaking and Benchmarking Large Vision-Language Models Using Themselves](../../ICCV2025/multimodal_vlm/ideator_jailbreaking_and_benchmarking_large_visionlanguage_m.md)
- [\[ICCV 2025\] Jailbreaking Multimodal Large Language Models via Shuffle Inconsistency](../../ICCV2025/multimodal_vlm/jailbreaking_multimodal_large_language_models_via_shuffle_inconsistency.md)
- [\[NeurIPS 2025\] Video-SafetyBench: A Benchmark for Safety Evaluation of Video LVLMs](video-safetybench_a_benchmark_for_safety_evaluation_of_video_lvlms.md)
- [\[NeurIPS 2025\] FOCUS: Internal MLLM Representations for Efficient Fine-Grained Visual Question Answering](focus_internal_mllm_representations_for_efficient_fine-grained_visual_question_a.md)
- [\[CVPR 2026\] Circuit Tracing in Vision-Language Models: Understanding the Internal Mechanisms of Multimodal Thinking](../../CVPR2026/multimodal_vlm/circuit_tracing_in_vision-language_models_understanding_the_internal_mechanisms_.md)

</div>

<!-- RELATED:END -->
