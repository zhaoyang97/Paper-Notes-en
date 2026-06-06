---
title: >-
  [Paper Note] DualOptim+: Bridging Shared and Decoupled Optimizer States for Better Machine Unlearning in Large Language Models
description: >-
  [ICML 2026][LLM Safety][Machine Unlearning] DualOptim+ decomposes Adam optimizer states into a "shared base state + decoupled delta state…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "Machine Unlearning"
  - "Optimizer States"
  - "Gradient Conflict"
  - "8-bit Quantization"
  - "AdamW"
date: 2026-05-08
content_hash: e87847951d6ac966
---

# DualOptim+: Bridging Shared and Decoupled Optimizer States for Better Machine Unlearning in Large Language Models

**Conference**: ICML 2026  
**arXiv**: [2605.21539](https://arxiv.org/abs/2605.21539)  
**Code**: https://github.com/CityU-MLO/DualOptimPlus  
**Area**: LLM Security / Machine Unlearning / Optimizers  
**Keywords**: Machine Unlearning, Optimizer States, Gradient Conflict, 8-bit Quantization, AdamW

## TL;DR
DualOptim+ decomposes Adam optimizer states into a "shared base state + decoupled delta state," allowing LLM machine unlearning to adaptively transition between shared and decoupled optimization as forget/retain gradients fluctuate between conflict and synergy. Theoretically, it degenerates to Alternate (for positive correlation) and DualOptim (for negative correlation), while an 8-bit quantized variant compresses extra memory overhead back to the baseline.

## Background & Motivation

**Background**: Machine Unlearning (MU) requires models to erase the influence of a forget set while maintaining the utility of a retain set. Current MU for LLMs primarily relies on the joint optimization of a forget loss $\mathcal{L}_f$ (e.g., GA, NPO, ME, RMU) and a retain loss $\mathcal{L}_r$ (e.g., CE, KL). Optimization strategies have evolved through three generations:
- **Joint** (Single-step backpropagation after summation; the previous de facto standard) — Simple, but gradient merging leads to performance degradation.
- **Alternate** (Alternating updates using gradients from only one objective per step) — Mitigates degradation but is unstable and sensitive to hyperparameters.
- **DualOptim** (Independent AdamW optimizers for each objective, maintaining separate states) — Effective for vision tasks but provides only marginal gains for LLMs.

**Limitations of Prior Work**: The authors observe that the cosine similarity between forget and retain gradients in LLM unlearning **fluctuates drastically** during training—showing positive correlation (shared signals) early on and negative or nearly orthogonal correlation later. Joint optimization loses adversarial signals by using a single shared state, while DualOptim loses synergistic signals through complete decoupling. Both correspond to only one correlation pattern, failing to achieve global optimality throughout the LLM training process.

**Key Challenge**: A binary choice between "shared vs. decoupled" cannot adapt to the dynamic changes in gradient correlation during LLM training; an ideal optimizer should adaptively transition between the two based on current correlation.

**Goal**: (1) Construct a plug-and-play optimizer framework that dynamically interpolates between shared and decoupled states based on the directional correlation of $\nabla \mathcal{L}_f$ and $\nabla \mathcal{L}_r$; (2) Cover scenarios such as fictitious unlearning, real-world unlearning, safety alignment, and multi-task learning; (3) Resolve the memory bloat caused by additional optimizer states.

**Key Insight**: Decompose the first and second moments of AdamW into "shared base + per-objective delta." The base is updated using all gradients (capturing commonalities), while the delta is updated via the residual "objective gradient − base" (capturing differences). Parameters are updated using the sum of the base and the corresponding delta. This mathematically yields an adaptive transition: when correlation is high, delta $\to 0$ (degenerating to Alternate); when correlation is strongly negative, base $\to 0$ (degenerating to DualOptim).

**Core Idea**: Base/delta decomposition + adaptive transition = the optimal intermediate between shared and decoupled states at the optimizer level.

## Method

### Overall Architecture

Each optimizer state (AdamW's $m$ and $v$) is split into:
- **Base state** $B$: Updated jointly by $\nabla \mathcal{L}_f$ and $\nabla \mathcal{L}_r$ to carry commonalities.
- **Delta states** $\Delta_f, \Delta_r$: Updated by the residual between the objective's gradient and the base to carry differences.

At each step, the parameters are updated using the base plus the corresponding delta: $\theta \leftarrow \theta - \eta (\hat B + \hat \Delta_o) / (\sqrt{|\hat v_B + \hat v_{\Delta_o}|} + \epsilon)$. The base state is updated after the parameter update to serve as a stable reference. This is coupled with an alternating schedule of $F_f$ forget steps and $F_r$ retain steps.

### Key Designs

1.  **Base and Delta State Decomposition**:
    - **Function**: Splits a single optimizer state into a shared component (base) and objective-specific components (delta) to preserve both signals.
    - **Mechanism**: Base $B \leftarrow \beta B + (1-\beta) \nabla \mathcal{L}_o$ (where $o$ is the current objective), Delta $\Delta_o \leftarrow \beta \Delta_o + (1-\beta)(\nabla \mathcal{L}_o - \hat B)$. Second moments $v_B, v_{\Delta_o}$ are similarly updated with squared gradients; bias correction is applied as $\hat B = B / (1-\beta^t)$.
    - **Design Motivation**: The base learns directions agreed upon by both forget and retain objectives (multi-task commonality), while deltas learn independent directions (adversarial components). Their summation prevents the loss of adversarial signals (unlike Joint) or synergistic signals (unlike DualOptim).

2.  **Adaptive Transition (Theoretical Properties)**:
    - **Function**: Automatically transitions between Alternate and DualOptim based on the directional correlation of forget and retain gradients.
    - **Mechanism**: Theorem 3.2 provides an asymptotic analysis—assuming $\mathbb{E}_t[g_{f,t}] = mG$ and $\mathbb{E}_t[g_{r,t}] = nG$:
        - $m = n$ (Positive correlation) $\to B \to mG, \Delta_{f,r} \to 0$, equivalent to Alternate (Shared state).
        - $m = -\frac{1-\beta^{F_r}}{\beta^{F_r}(1-\beta^{F_f})}n$ (Strong negative correlation) $\to B \to 0$, only deltas act, equivalent to DualOptim (Fully decoupled).
    - **Design Motivation**: Eliminates the need for manual correlation detection or switching; the adaptive behavior is intrinsic to the optimizer structure, requiring no hyperparameter tuning as correlations shift during training.

3.  **DualOptim+ 8bit (Memory Control)**:
    - **Function**: Quantizes the additional base and delta states to 8-bit to compress memory overhead back to vanilla AdamW levels.
    - **Mechanism**: Utilizes block-wise quantization for $B, \Delta_f, \Delta_r$, following the bitsandbytes 8-bit Adam approach. The paper reports nearly identical performance between quantized and fp32 versions.
    - **Design Motivation**: The base + delta approach consumes $2 \times$ more memory for moments compared to vanilla AdamW, which is unacceptable for large models. 8-bit quantization is a necessary engineering optimization for practical deployment.

### Training Strategy
$F_f$ and $F_r$ control the alternating frequency (set to 1:1 in experiments). The base state is updated after the parameter update for stability. This alternating pattern proves more stable than pure alternation.

## Key Experimental Results

### TOFU Fictitious Unlearning (Phi-1.5, IDK+GD Objectives)

| Forget % | Method | UFE↑ | TFE↑ | MU↑ | OVR↑ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 10% | Joint | 78.1 | 50.6 | 60.2 | 62.3 |
| 10% | Alternate | 80.7 | 56.8 | 64.5 | 66.6 |
| 10% | DualOptim | 81.2 | 58.3 | 65.0 | 67.4 |
| 10% | **DualOptim+** | **84.8** | **62.7** | **68.1** | **70.9** |
| 10% | DualOptim+ 8bit | 84.5 | 62.4 | 67.9 | 70.7 |

OVR improves by ~3.5 points; unlearning efficiency (UFE/TFE) and model utility (MU) improve simultaneously without a trade-off.

### Main Results: Real Unlearning & Safety Alignment (Partial Excerpts)

| Task | Dataset | Joint OVR | DualOptim OVR | **DualOptim+ OVR** |
| :--- | :--- | :--- | :--- | :--- |
| WMDP-Bio (Llama 2-7B) | Real Unlearning | 51.2 | 54.7 | **58.9** |
| WMDP-Cyber | Real Unlearning | 49.6 | 52.3 | **56.4** |
| Harm-Refuse | Safety Alignment | 62.8 | 66.1 | **70.2** |

Consistent lead of 4–5 points across tasks.

### Optimizer Update Similarity (Values from Figure 2)

- Alternate (Shared state): Cosine similarity of adjacent forget/retain updates $\approx 0.95$ (signals almost merged).
- DualOptim (Fully decoupled): $\approx 0.0$ (signals independent).
- **Ours (DualOptim+)**: $\approx 0.4$–$0.6$ (lies between the two and fluctuates dynamically with training stages).

This directly validates the "adaptive transition" hypothesis.

### Key Findings
- **Gradient correlation indeed changes dynamically**: Figure 2(b) shows cosine similarity fluctuating wildly in the range $[-0.5, 0.7]$, proving that static shared or static decoupled strategies are sub-optimal.
- **DualOptim+ is the appropriate intermediate**: The observed range of $0.4$–$0.6$ aligns perfectly with theoretical limits.
- **Quantization is nearly lossless**: The OVR gap between 8-bit and fp32 is $< 0.3$ points, making it engineering-ready.
- **Cross-optimizer transferability**: Also effective on Muon (Appendix), demonstrating the universality of the base/delta decomposition.

## Highlights & Insights
- **Adaptivity via Optimizer Structure**: Unlike previous multi-objective optimization that relies on weight scheduling or projection, this work embeds the transition directly into the optimizer states, requiring no external signals.
- **Clean Theoretical Limits**: Theorem 3.2 provides closed-form asymptotic behaviors where the extremes precisely match Alternate and DualOptim—a rare "intermediate method" with clean theoretical backing.
- **Engineering Consciousness**: Recognizing that the $2 \times$ state overhead is a deal-breaker for LLMs, the authors proactively implemented quantization. This "algorithm + engineering" package is becoming an essential paradigm in LLM research.
- **Potential Beyond Unlearning**: The base/delta decomposition is essentially a multi-objective optimizer framework. Its effectiveness in safety alignment and multi-task learning suggests utility in other scenarios like RLHF/DPO with KL regularization or multi-expert distillation.

## Limitations & Future Work
- Extension to more than 2 objectives is non-trivial: $k$ objectives require $1 + k$ states, further bloating memory; scaling the quantization scheme remains undiscussed.
- $F_f, F_r$ remain manual hyperparameters; automated scheduling based on real-time correlation would be superior.
- Primarily validated on models $\leq$ 7B (Phi-1.5, Llama-2-7B); performance on 70B+ scales for real-world deployment is untested.
- Comparisons with concurrent methods (GradDiff, SimNPO, etc.) could be more detailed, particularly regarding utility recovery after long unlearning training.

## Related Work & Insights
- **vs. DualOptim**: DualOptim is fully decoupled and loses synergistic signals; Ours introduces a shared base for adaptive transitions, theoretically covering DualOptim as a limit.
- **vs. Joint / Alternate**: Joint degrades signals while Alternate shares states—both are single-point strategies. Ours is a continuous family of solutions.
- **vs. Federated Learning (SCAFFOLD / FedProx)**: The base/delta decomposition is structurally similar to server/client control variables in SCAFFOLD, though target scenarios and update rules differ.
- **Insight**: Any training problem involving multi-objective optimization with dynamic correlations (RLHF+KL, multi-task, etc.) could benefit from base/delta decomposition, turning the "shared vs. decoupled" choice from a hyperparameter into a mathematical selection.

## Rating
- Novelty: ⭐⭐⭐⭐ The base/delta decomposition is simple but effective; the "adaptive intermediate" framing is a genuine contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers TOFU, WMDP, safety alignment, and multi-tasking; includes quantization, cross-optimizer tests, and ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, clean theoretical limits (Theorem 3.2), and strong supporting visualizations in Figure 2.
- Value: ⭐⭐⭐⭐ LLM unlearning is critical for safety and compliance; DualOptim+ is one of the strongest optimizer-side improvements for TOFU/WMDP.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Forgetting is Not Erasing: A Survey of Reversibility in Large Language Model Machine Unlearning](unlearning_isnt_deletion_investigating_reversibility_of_machine_unlearning_in_ll.md)
- [\[CVPR 2026\] SineProject: Machine Unlearning for Stable Vision–Language Alignment](../../CVPR2026/llm_safety/sineproject_machine_unlearning_for_stable_vision_language_alignment.md)
- [\[ICML 2026\] Forget to Know, Remember to Use: Context-Aware Unlearning for Large Language Models](forget_to_know_remember_to_use_context-aware_unlearning_for_large_language_model.md)
- [\[ICCV 2025\] MUNBa: Machine Unlearning via Nash Bargaining](../../ICCV2025/llm_safety/munba_machine_unlearning_via_nash_bargaining.md)
- [\[ICLR 2026\] OFMU: Optimization-Driven Framework for Machine Unlearning](../../ICLR2026/llm_safety/ofmu_optimization-driven_framework_for_machine_unlearning.md)

</div>

<!-- RELATED:END -->
