---
title: >-
  [Paper Note] LLM-CAS: Dynamic Neuron Perturbation for Real-Time Hallucination Correction
description: >-
  [AAAI 2026][Hallucination Detection][Dynamic neuron perturbation] LLM-CAS is the first work to formulate real-time LLM hallucination correction as a hierarchical reinforcement learning (HRL) problem. It trains an RL agent to dynamically select optimal neuron perturbation strategies at inference time — the high-level policy selects a functional network category, while the low-level policy selects perturbation type and magnitude. Combined with adaptive masking and causal tracin…
tags:
  - "AAAI 2026"
  - "Hallucination Detection"
  - "Dynamic neuron perturbation"
  - "hierarchical reinforcement learning"
  - "inference-time intervention"
  - "causal tracing"
  - "adaptive masking"
date: 2026-05-08
content_hash: 42f19a42ef81b37c
---

# LLM-CAS: Dynamic Neuron Perturbation for Real-Time Hallucination Correction

**Conference**: AAAI 2026
**arXiv**: [2512.18623](https://arxiv.org/abs/2512.18623)  
**Code**: Coming soon  
**Area**: Hallucination Detection
**Keywords**: Dynamic neuron perturbation, hierarchical reinforcement learning, inference-time intervention, causal tracing, adaptive masking

## TL;DR
LLM-CAS is the first work to formulate real-time LLM hallucination correction as a hierarchical reinforcement learning (HRL) problem. It trains an RL agent to dynamically select optimal neuron perturbation strategies at inference time — the high-level policy selects a functional network category, while the low-level policy selects perturbation type and magnitude. Combined with adaptive masking and causal tracing for precise neuron localization, LLM-CAS achieves a 10.98% improvement on StoryCloze, outperforming static and dynamic baselines such as ITI, CAA, and SADI.

## Background & Motivation
**Background**: LLM hallucination remains a core obstacle to deployment. Existing approaches fall into three categories: SFT/RLHF (data-intensive, computationally expensive, and prone to catastrophic forgetting); static model editing (locate-then-edit, one-shot parameter modification $W_{\text{edited}} = W + \Delta W$, but permanent edits risk damaging unrelated knowledge); and inference-time intervention (ITI/CAA add fixed steering vectors, but these are static across inputs).

**Limitations of Prior Work**: (1) Static model editing causes catastrophic forgetting and knowledge conflicts after repeated edits; (2) steering vectors in ITI/CAA are precomputed fixed values that do not adapt to different inputs; (3) SADI dynamically adjusts steering vectors but relies on heuristic rules, lacking principled optimization.

**Key Challenge**: Hallucination is context-dependent — the same model may produce different types of hallucinations for different inputs, requiring different correction strategies. Yet existing methods either apply a one-size-fits-all fix (static steering vectors) or rely on handcrafted rules (SADI), without a learned, adaptive correction policy.

**Goal**: How to learn a principled, context-aware dynamic neuron perturbation strategy for real-time hallucination correction?

**Key Insight**: The problem is formulated as HRL — the high-level policy selects *which functional network to intervene in*, while the low-level policy selects *what type and magnitude of perturbation to apply*. Perturbations are temporary and do not permanently modify model weights.

**Core Idea**: Train an RL agent via hierarchical RL to dynamically select context-optimal temporary neuron perturbations at inference time to correct hallucinations.

## Method

### Overall Architecture
A three-stage pipeline: Stage 1 (identify bad cases) → Stage 2 (train the HRL agent) → Stage 3 (apply at inference time). Given a bad case $x$, the state is encoded (input embedding + baseline score + historical best score + step count) → the high-level PPO agent selects a functional network category $C_k$ → the low-level PPO agent selects perturbation type and magnitude → adaptive masking localizes specific neurons → activations are temporarily perturbed → the corrected output is evaluated → reward feedback updates the policies.

### Key Designs

1. **Hierarchical Reinforcement Learning Architecture**:

    - Function: Decomposes the vast neuron perturbation space into a manageable two-level decision process.
    - High-level policy $\pi_H(a_H|s)$: Selects a macro-level intervention target from the functional network category set $A_H = \{C_1, ..., C_{N_H}\}$ (e.g., language network, knowledge network).
    - Low-level policy $\pi_L(a_L|s, a_H)$: Given the high-level selection, determines perturbation type $a_L^{\text{type}} \in \{\text{noise, zero, scale, ...}\}$ and magnitude $a_L^{\text{mag}} \in \{m_1, ..., m_{N_M}\}$.
    - Design Motivation: Directly searching for optimal perturbations over all neurons is intractable; hierarchical decomposition makes the search space manageable and semantically structured.

2. **Adaptive Masking + Causal Tracing**:

    - Function: Precisely localizes which neurons require perturbation.
    - Two-stage masking: (a) **General sparse mask** $M_{k,l}(i; \theta_{k,l})$ — learnable gating parameters regularized by L1+L0 penalties to enforce sparsity; (b) **Input-specific adaptation** — Integrated Gradients are used to compute causal attribution scores $\text{Attr}_l(x, i)$, which are element-wise multiplied with the general mask to yield the final operation mask $M_{op,k,l}$.
    - Design Motivation: The general mask learns "which neurons are typically associated with hallucination," while causal tracing adapts to "which neurons are critical for the current input." The combination achieves both precision and efficiency.

3. **Multi-Dimensional Reward Function**:

    - Function: Jointly optimizes hallucination reduction, relevance preservation, and fluency.
    - Reward: $R_t = w_h \cdot \Delta\text{Score}_{h,t} + w_r \cdot \Delta\text{Score}_{r,t} + w_f \cdot \Delta\text{Score}_{f,t} + R_{\text{exp},t}$
    - Score **deltas** rather than absolute values are used — mitigating evaluation bias from LLM-as-Judge.
    - Design Motivation: Optimizing hallucination reduction alone may degrade fluency and relevance; multi-objective rewards ensure corrected outputs maintain overall quality.

### Loss & Training
PPO optimizes both levels of the policy network. Mask parameters $\theta_{\text{mask}}$ and RL policies are optimized independently. Perturbations are temporary — applied only during the current inference pass without modifying model weights.

## Key Experimental Results

### Main Results — Multiple-Choice Tasks (Llama2-7B-Chat)

| Method | StoryCloze | SST-2 | BoolQ | Winogrande | Avg |
|--------|-----------|-------|-------|------------|-----|
| Baseline | 65.06 | 88.63 | 70.52 | 50.91 | 68.78 |
| ITI (static) | 68.50 | 91.38 | 74.10 | 52.80 | 71.70 |
| CAA (static) | 74.65 | 91.16 | 74.98 | 52.64 | 73.36 |
| SADI (dynamic) | 67.57 | 88.69 | 70.40 | 51.93 | 69.65 |
| **LLM-CAS** | **76.04** | **91.30** | 74.47 | **52.90** | **73.68** |

### Ablation Study

| Configuration | SST-2 | BoolQ | Winogrande | StoryCloze | Avg |
|---------------|-------|-------|------------|-----------|-----|
| Full | 91.30 | 74.47 | 52.90 | 76.04 | 73.68 |
| Random mask | 86.73 | 67.10 | 51.32 | 70.20 | 68.84 |
| Random action | 82.45 | 64.32 | 49.15 | 66.87 | 65.70 |
| Both random | 80.18 | 62.05 | 47.98 | 63.41 | 63.41 |

### Key Findings
- LLM-CAS achieves a 10.98% gain on StoryCloze (65.06→76.04), the largest improvement across all methods — narrative coherence tasks benefit the most.
- SADI (heuristic dynamic intervention) underperforms even static ITI/CAA, demonstrating that dynamic adjustment without learned optimization can be counterproductive.
- Ablations show that both PPO and dynamic masking are indispensable — removing PPO reduces average performance by 7.98%, removing masking by 4.84%, and removing both by 10.27%.
- Cross-model generalization is confirmed: StoryCloze improves from 21.51→34.41 (+12.9%) on Mistral-7B and from 60.95→69.76 (+8.81%) on Gemma-7B.
- Inference-time overhead is acceptable: PPO decision time is negligible compared to model forward-pass time.

## Highlights & Insights
- **Novel HRL Formulation**: The first work to model hallucination correction as hierarchical RL — the high-level policy decides *where* to intervene, the low-level policy decides *how* — offering a more principled alternative to heuristic rules.
- **Temporary Perturbation vs. Permanent Editing**: Perturbations take effect only during the current inference pass and leave model weights entirely unchanged — zero risk of catastrophic forgetting, representing a fundamental improvement over static model editing.
- **Two-Stage Localization via General Mask + Causal Tracing**: A general template is learned offline and refined online via real-time causal attribution — both efficient and precise.

## Limitations & Future Work
- Validation is limited to 7B-scale models; effectiveness on larger models remains to be confirmed.
- Training the RL agent still requires non-trivial compute and a collection of bad-case samples.
- Gains on open-ended generation tasks (TriviaQA +2.71%) are considerably smaller than on multiple-choice tasks (StoryCloze +10.98%), likely because reward signal design is more challenging for open-ended generation.
- Evaluation bias from LLM-as-Judge may degrade the quality of training signals.
- Functional network category definitions rely on prior knowledge; an automatic discovery mechanism would be a valuable extension.

## Related Work & Insights
- **vs. ITI/CAA**: Static steering vectors vs. learned dynamic policies — LLM-CAS exhibits substantially stronger adaptability.
- **vs. SADI**: SADI generates dynamic vectors heuristically yet performs worse than static methods, whereas LLM-CAS learns superior strategies via HRL.
- **vs. PING (Agentic safety paper in this batch)**: PING intervenes at the response level (prefix injection), while LLM-CAS intervenes at the neuron level — two complementary granularities.
- Inspiration: Combining MUG (counterfactual testing) for hallucination detection with LLM-CAS for neuron-level correction constitutes a complete detect-then-correct pipeline.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of HRL, dynamic neuron perturbation, and causal tracing is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers both multiple-choice and generation tasks across three models with complete ablations, but lacks validation on larger models.
- Writing Quality: ⭐⭐⭐⭐ Formally rigorous, though some derivations are somewhat verbose.
- Value: ⭐⭐⭐⭐⭐ A landmark contribution that advances hallucination correction from heuristic-driven to learning-based approaches.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Octopus: Alleviating Hallucination via Dynamic Contrastive Decoding](../../CVPR2025/hallucination/octopus_alleviating_hallucination_via_dynamic_contrastive_decoding.md)
- [\[ICLR 2026\] Dynamic Multimodal Activation Steering for Hallucination Mitigation in Large Vision-Language Models](../../ICLR2026/hallucination/dynamic_multimodal_activation_steering_for_hallucination_mitigation_in_large_vis.md)
- [\[CVPR 2026\] Prefill-Time Intervention for Mitigating Hallucination in Large Vision-Language Models](../../CVPR2026/hallucination/prefill-time_intervention_for_mitigating_hallucination_in_large_vision-language_.md)
- [\[ACL 2026\] Detecting Hallucinations in SpeechLLMs at Inference Time Using Attention Maps](../../ACL2026/hallucination/detecting_hallucinations_in_speechllms_at_inference_time_using_attention_maps.md)
- [\[ACL 2025\] HalluLens: LLM Hallucination Benchmark](../../ACL2025/hallucination/hallulens_llm_hallucination_benchmark.md)

</div>

<!-- RELATED:END -->
