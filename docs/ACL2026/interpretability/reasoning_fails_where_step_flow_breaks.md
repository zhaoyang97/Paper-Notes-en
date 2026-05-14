---
title: >-
  [Paper Note] Reasoning Fails Where Step Flow Breaks
description: >-
  [ACL 2026][Interpretability][reasoning model interpretability] This paper proposes Step-Saliency, a diagnostic tool that identifies two depth-correlated information flow failure modes in large reasoning models (Shallow L…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "reasoning model interpretability"
  - "information flow analysis"
  - "test-time intervention"
  - "attention mechanism"
  - "chain-of-thought"
date: 2026-05-08
content_hash: 73c9c4247eb5894e
---

# Reasoning Fails Where Step Flow Breaks

**Conference**: ACL 2026
**arXiv**: [2604.06695](https://arxiv.org/abs/2604.06695)
**Code**: [GitHub](https://github.com/XiaoyuXu-Vincent/step-saliency)
**Area**: Interpretability
**Keywords**: reasoning model interpretability, information flow analysis, test-time intervention, attention mechanism, chain-of-thought

## TL;DR
This paper proposes Step-Saliency, a diagnostic tool that identifies two depth-correlated information flow failure modes in large reasoning models (Shallow Lock-in and Deep Decay), and designs StepFlow, a test-time intervention that repairs information propagation and improves reasoning accuracy without retraining.

## Background & Motivation

**Background** Large reasoning models (LRMs) achieve strong performance on mathematics, science, and coding tasks by generating long chains of thought (CoT), yet their behavior remains unstable and difficult to interpret. Most existing analysis tools operate at the token level, producing dense and noisy signals over long reasoning trajectories that fail to capture inter-step dependencies.

**Limitations of Prior Work** Current interpretability methods fall into two categories: attention analysis and gradient saliency analysis. Attention weights do not necessarily faithfully reflect prediction-driving factors; gradient saliency is more faithful to the model's actual computation but is noisy over long sequences and difficult to aggregate across positions. The core issue is not a lack of signals, but rather the absence of readable units aligned with reasoning steps.

**Key Challenge** When a model makes an error, it is impossible to attribute the final mistake to a specific step in the internal reasoning trajectory — token-level saliency maps are too dense to intuitively reveal information flow breakdowns between steps.

**Goal** To design a step-level diagnostic tool that tracks inter-step influence relationships across network depths, and to devise test-time interventions based on the diagnostic results to repair information flow.

**Key Insight** Token-level attention-gradient influence scores are aggregated to the step level via mean pooling, forming a compact step-to-step saliency map, which is then analyzed layer by layer to compare correct and incorrect reasoning trajectories.

**Core Idea** The root cause of erroneous reasoning lies in information flow breakdowns — shallow layers over-attend to the current step (Shallow Lock-in), while deep layers progressively lose attention to the thinking segments (Deep Decay). Targeted interventions applied separately to shallow and deep layers can repair these information flow defects.

## Method

### Overall Architecture
Step-Saliency is a diagnostic tool, and StepFlow is the intervention method derived from the diagnosis. The overall pipeline is: (1) segment the reasoning sequence into three parts — question, thinking, and summary; (2) compute token-level attention-gradient influence scores and pool them into a step-to-step map; (3) analyze saliency patterns layer by layer to identify Shallow Lock-in and Deep Decay; (4) repair information flow during decoding via two components, OEB and SMI.

### Key Designs

1. **Step-Saliency Diagnosis**:
    - Function: Aggregates token-level saliency into step-level visualizations.
    - Mechanism: For each layer and head, the absolute product of attention weights and their gradients is computed as $I^{(\ell)}_{t\leftarrow k} = \frac{1}{H}\sum_h |A^{(\ell,h)}_{t,k} \cdot \frac{\partial \mathcal{L}_t}{\partial A^{(\ell,h)}_{t,k}}|$, then mean-pooled along step boundaries to produce a step-to-step influence matrix.
    - Design Motivation: Token-level saliency maps are excessively dense and noisy; mean pooling to the step level suppresses noise and reveals cross-step dependency patterns.

2. **Odds-Equal Bridge (OEB) — Shallow-Layer Intervention**:
    - Function: Prevents attention mass in shallow layers from collapsing onto the current step.
    - Mechanism: Keys are partitioned into the current segment $\mathcal{S}$, a bridge segment $\mathcal{B}$ (early context), and others $\mathcal{O}$. A lower bound on bridge-segment attention mass is set as $\tau_\mathcal{B} = \min(\sqrt{|\mathcal{B}|/(|\mathcal{B}|+|\mathcal{S}|)}, \tau_{\max})$; when bridge mass falls below this bound, logits are adjusted via KL projection.
    - Design Motivation: Diagnostics reveal that in incorrect trajectories, shallow layers concentrate nearly all attention on the current step and its neighbors, neglecting the question and earlier reasoning steps. OEB ensures the bridge region maintains a reasonable share of attention.

3. **Step Momentum Injection (SMI) — Deep-Layer Intervention**:
    - Function: Injects a residual summary of the previous step at step boundaries in deep layers.
    - Mechanism: At the boundary between steps $\Gamma_i$ and $\Gamma_{i+1}$, a step-level momentum vector $\mathbf{m}_{\text{prev}} = \frac{1}{|\Gamma_i|}\sum_{k\in\Gamma_i}\mathbf{v}_k$ is computed and injected into the hidden state of the first token of the next step: $\mathbf{h}'_t = \mathbf{h}_t + \alpha \mathbf{m}_{\text{prev}}$.
    - Design Motivation: Deep Decay manifests as rapid attenuation of thinking saliency in deep layers, causing the summary to become self-referential. SMI preserves a small portion of prior-step information at step boundaries to maintain the connection from early reasoning to the summary.

### Loss & Training
StepFlow is a purely test-time intervention that **requires no training or backpropagation**. It modifies the forward pass during a single decoding run: OEB acts on attention logits in shallow layers, and SMI acts on the residual stream in deep layers. Each model requires only one $\tau_{\max}$ and one $\alpha$, tuned on a small validation set.

## Key Experimental Results

### Main Results

| Model + Method | AIME24 | AIME25 | MATH-500 | GPQA-D | LiveCodeBench |
|------------|--------|--------|----------|--------|---------------|
| R1-Distill-7B baseline | 54.0 | 39.2 | 92.8 | 49.1 | 37.6 |
| R1-Distill-7B + StepFlow | 62.5 | 43.8 | 93.8 | 57.6 | 47.1 |
| R1-Distill-32B baseline | 72.6 | 54.9 | 94.3 | 62.1 | 57.2 |
| R1-Distill-32B + StepFlow | 74.5 | **66.7** | 95.6 | 64.5 | 63.0 |
| GPT-OSS-20B medium baseline | 63.4 | 62.0 | 89.2 | 65.2 | 70.0 |
| GPT-OSS-20B medium + StepFlow | 66.0 | 69.2 | 90.5 | 70.3 | **79.5** |

### Ablation Study

| Configuration | AIME25 | GPQA-D | LiveCodeBench | Note |
|------|--------|--------|---------------|------|
| Baseline | 62.0 | 65.2 | 70.0 | GPT-OSS-20B medium |
| + OEB only | 64.5 | 66.7 | 74.5 | Repairs shallow lock-in |
| + SMI only | 64.0 | 67.2 | 75.0 | Repairs deep decay |
| + OEB + SMI (StepFlow) | **69.2** | **70.3** | **79.5** | Complementary; best overall |

### Key Findings
- StepFlow yields the largest gains on competition-level mathematics (R1-32B: +11.8 on AIME25), as such problems require propagating information across multiple steps.
- On LiveCodeBench, gains broken down by difficulty: Easy +3.4, Medium +13.8, Hard +14.2 — the harder the problem, the more effective the intervention.
- Among corrected error types, arithmetic carry propagation (34%) and premise forgetting (38%) account for 72%; conceptual errors are rarely corrected.
- Under matched compute (~1.35×), StepFlow's gain is 5.7× that of extended generation; reaching StepFlow's accuracy via self-consistency requires 8-way sampling (8× compute).

## Highlights & Insights
- Elevating analysis from the token level to the step level is the key innovation, making the analysis of long reasoning trajectories feasible and intuitive.
- The diagnose-then-intervene paradigm is elegant: Step-Saliency first identifies the problem (Shallow Lock-in / Deep Decay), and OEB / SMI then surgically repair it.
- No retraining is required; the purely inference-time intervention is applicable to any open-source LRM and is highly practical.
- Computational overhead is only ~1.35×, far more efficient than multi-path sampling with majority voting.

## Limitations & Future Work
- The boundary between shallow and deep layers requires tuning on a small validation set; a fully automatic layer-range selection method is lacking.
- The intervention design space remains underexplored (e.g., head-level steering or value-space projection).
- The causal relationship between Shallow Lock-in / Deep Decay and final errors remains heuristic and has not been rigorously established.
- The method is only applicable to open-source LRMs and cannot be applied to black-box API models.

## Related Work & Insights
- Complementary to the attention-level intervention of Yan et al., which preserves CoT context at the attention layer.
- Orthogonally composable with self-consistency: StepFlow + SC(k=2) at ~2.7× compute surpasses SC(k=4) at 4× compute.
- The Step-Saliency framework can be extended to information flow analysis in other long-sequence generation tasks (e.g., long-document writing, multi-turn dialogue).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Step-level saliency combined with diagnosis-driven intervention constitutes an entirely new paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Six benchmarks, five backbone models, detailed ablations, and compute-normalized comparisons.
- Writing Quality: ⭐⭐⭐⭐⭐ The diagnose-to-intervene logical chain is clear, and figures are carefully designed.
- Value: ⭐⭐⭐⭐⭐ Directly practical for understanding and improving reasoning models; ready to use out of the box.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Hallucination Begins Where Saliency Drops](../../ICLR2026/interpretability/hallucination_begins_where_saliency_drops.md)
- [\[AAAI 2026\] Using Certifying Constraint Solvers for Generating Step-wise Explanations](../../AAAI2026/interpretability/using_certifying_constraint_solvers_for_generating_step-wise_explanations.md)
- [\[ACL 2026\] The Reasoning Trap: How Enhancing LLM Reasoning Amplifies Tool Hallucination](the_reasoning_trap_how_enhancing_llm_reasoning_amplifies_tool_hallucination.md)
- [\[ACL 2026\] ChemVLR: Prioritizing Reasoning in Perception for Chemical Vision-Language Understanding](chemvlr_prioritizing_reasoning_in_perception_for_chemical_vision-language_unders.md)
- [\[ACL 2026\] Forest Before Trees: Latent Superposition for Efficient Visual Reasoning](forest_before_trees_latent_superposition_for_efficient_visual_reasoning.md)

</div>

<!-- RELATED:END -->
