---
title: >-
  [Paper Note] Reasoning Fails Where Step Flow Breaks
description: >-
  [ACL 2026][LLM Reasoning][Interpretability of reasoning models] The paper introduces Step-Saliency, a diagnostic tool to identify two depth-dependent information flow failure modes (Shallow Lock-in and Deep Decay) in Lar…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Interpretability of reasoning models"
  - "information flow analysis"
  - "test-time intervention"
  - "attention mechanism"
  - "Chain-of-Thought"
date: 2026-05-08
content_hash: 3911122ed51ad0fd
---

# Reasoning Fails Where Step Flow Breaks

**Conference**: ACL 2026  
**arXiv**: [2604.06695](https://arxiv.org/abs/2604.06695)  
**Code**: [GitHub](https://github.com/XiaoyuXu-Vincent/step-saliency)  
**Area**: Interpretability  
**Keywords**: Interpretability of reasoning models, information flow analysis, test-time intervention, attention mechanism, Chain-of-Thought

## TL;DR
The paper introduces Step-Saliency, a diagnostic tool to identify two depth-dependent information flow failure modes (Shallow Lock-in and Deep Decay) in Large Reasoning Models (LRMs), and designs StepFlow, a test-time intervention method to repair information propagation and enhance reasoning accuracy without retraining.

## Background & Motivation

**Background** Large Reasoning Models (LRMs) have achieved superior performance on mathematical, scientific, and coding tasks by generating long Chains-of-Thought (CoT). However, their behavior remains unstable and difficult to explain. Existing analysis tools mostly operate at the token level; for long reasoning trajectories, these signals are often too dense and noisy to summarize dependencies between steps.

**Limitations of Prior Work** Current interpretability methods are categorized into attention analysis and gradient saliency analysis. Attention weights do not always faithfully reflect prediction drivers. While gradient saliency aligns more closely with actual model computation, it is noisy over long sequences and difficult to aggregate across positions. The core problem is not a lack of signal, but a lack of readable units aligned with reasoning steps.

**Key Challenge** When models fail, researchers cannot easily attribute the final error to a specific step in the internal reasoning trajectory—token-level saliency maps are too dense to intuitively reveal breaks in information flow between steps.

**Goal** To design a step-level diagnostic tool capable of tracking the influence relationships between steps across different network depths and to develop test-time interventions to fix information flow based on these diagnostics.

**Key Insight** Aggregate token-level attention-gradient influence scores into step-level units via mean pooling to form compact step-to-step saliency maps, followed by a layer-by-layer analysis of the differences between correct and incorrect reasoning trajectories.

**Core Idea** The root cause of reasoning errors lies in information flow disruption: Shallow Lock-in at lower layers (excessive focus on the current step) and Deep Decay at deeper layers (gradual loss of attention toward thinking segments). These flaws can be repaired through targeted interventions at shallow and deep layers respectively.

## Method

### Overall Architecture
Step-Saliency is a diagnostic tool, and StepFlow is an intervention method based on those diagnostics. The overall pipeline is: (1) segment the reasoning sequence into question-thinking-summary sections; (2) compute token-level attention-gradient influence scores and pool them into a step-to-step map; (3) perform layer-wise analysis of saliency patterns to identify Shallow Lock-in and Deep Decay; (4) repair information flow during decoding via the OEB and SMI components.

### Key Designs

1.  **Step-Saliency Diagnostic**:
    *   **Function**: Aggregates token-level saliency into step-level visualizations.
    *   **Mechanism**: For each layer and head, the absolute product of attention weights and their gradients is computed as $I^{(\ell)}_{t\leftarrow k} = \frac{1}{H}\sum_h |A^{(\ell,h)}_{t,k} \cdot \frac{\partial \mathcal{L}_t}{\partial A^{(\ell,h)}_{t,k}}|$. These are then mean-pooled according to step boundaries to produce a step-to-step influence matrix.
    *   **Design Motivation**: Token-level saliency maps are overly dense and noisy; mean pooling at the step level suppresses noise and reveals cross-step dependency patterns.

2.  **Odds-Equal Bridge (OEB) — Shallow Layer Intervention**:
    *   **Function**: Prevents the collapse of attention quality onto the current step in shallow layers.
    *   **Mechanism**: Keys are divided into the current segment $\mathcal{S}$, bridge segment $\mathcal{B}$ (early context), and others $\mathcal{O}$. A lower bound for bridge attention quality is set as $\tau_\mathcal{B} = \min(\sqrt{|\mathcal{B}|/(|\mathcal{B}|+|\mathcal{S}|)}, \tau_{\max})$. If the bridge quality falls below this bound, logits are adjusted via KL projection.
    *   **Design Motivation**: Diagnostics show that in incorrect trajectories at shallow layers, almost all attention is concentrated on the current step and its neighbors, ignoring the question and early reasoning steps. OEB ensures the bridge region maintains a reasonable attention share.

3.  **Step Momentum Injection (SMI) — Deep Layer Intervention**:
    *   **Function**: Injects a residual summary of the previous step at step boundaries in deep layers.
    *   **Mechanism**: At the boundary between steps $\Gamma_i$ and $\Gamma_{i+1}$, a step-level momentum vector $\mathbf{m}_{\text{prev}} = \frac{1}{|\Gamma_i|}\sum_{k\in\Gamma_i}\mathbf{v}_k$ is calculated and injected into the hidden state of the first token of the next step: $\mathbf{h}'_t = \mathbf{h}_t + \alpha \mathbf{m}_{\text{prev}}$.
    *   **Design Motivation**: Deep Decay manifests as the rapid attenuation of thinking saliency in deep layers, causing the summary to become self-focused. SMI maintains the connection between early reasoning and the summary by preserving a small portion of information from the previous step.

### Loss & Training
StepFlow is a pure test-time intervention and **requires no training or backpropagation**. It modifies the forward pass during a single decoding process: OEB acts on attention logits in shallow layers, while SMI acts on the residual flow in deep layers. Each model only requires selecting $\tau_{\max}$ and $\alpha$ tuned on a small validation set.

## Key Experimental Results

### Main Results

| Model + Method | AIME24 | AIME25 | MATH-500 | GPQA-D | LiveCodeBench |
| :--- | :--- | :--- | :--- | :--- | :--- |
| R1-Distill-7B baseline | 54.0 | 39.2 | 92.8 | 49.1 | 37.6 |
| R1-Distill-7B + StepFlow | 62.5 | 43.8 | 93.8 | 57.6 | 47.1 |
| R1-Distill-32B baseline | 72.6 | 54.9 | 94.3 | 62.1 | 57.2 |
| R1-Distill-32B + StepFlow | 74.5 | **66.7** | 95.6 | 64.5 | 63.0 |
| GPT-OSS-20B medium baseline | 63.4 | 62.0 | 89.2 | 65.2 | 70.0 |
| GPT-OSS-20B medium + StepFlow | 66.0 | 69.2 | 90.5 | 70.3 | **79.5** |

### Ablation Study

| Configuration | AIME25 | GPQA-D | LiveCodeBench | Notes |
| :--- | :--- | :--- | :--- | :--- |
| Baseline | 62.0 | 65.2 | 70.0 | GPT-OSS-20B medium |
| + OEB only | 64.5 | 66.7 | 74.5 | Fixes shallow lock-in |
| + SMI only | 64.0 | 67.2 | 75.0 | Fixes deep decay |
| + OEB + SMI (StepFlow) | **69.2** | **70.3** | **79.5** | Complementary optimal effect |

### Key Findings
*   StepFlow provides the largest gains on competition-level math problems (R1-32B improves by +11.8 on AIME25) because these problems require information propagation across multiple steps.
*   On LiveCodeBench, performance gains correlate with difficulty: Easy +3.4, Medium +13.8, Hard +14.2.
*   Among fixed error types, arithmetic carry propagation (34%) and premise forgetting (38%) account for 72%, while conceptual errors are rarely fixed.
*   Under matched compute (~1.35x), the gain from StepFlow is 5.7x greater than simply extending generation; reaching StepFlow's accuracy via self-consistency would require 8-way voting (8x compute).

## Highlights & Insights
*   Elevating token-level analysis to the step level is a key innovation, making the analysis of long reasoning trajectories feasible and intuitive.
*   The diagnostic-intervention paradigm is elegant: use Step-Saliency to identify problems (Shallow Lock-in / Deep Decay) and OEB / SMI to precisely repair them.
*   Highly practical as it requires no retraining, works as a pure inference-time intervention, and applies to any open-source LRM.
*   Incremental computational overhead is only ~1.35x, significantly better than multi-path sampling and voting.

## Limitations & Future Work
*   The boundary between shallow and deep layers must be tuned on a small validation set; a fully automated method for layer range selection is currently lacking.
*   The intervention design space (e.g., head-level steering or value-space projection) has not been fully explored.
*   The causal relationship between Shallow Lock-in/Deep Decay and final errors is currently heuristic and has not been strictly proven.
*   Only effective for open-source LRMs; cannot be applied to black-box API models.

## Related Work & Insights
*   Complementary to attention-level interventions (e.g., Yan et al.), which preserve CoT context within attention layers.
*   Can be combined orthogonally with self-consistency; StepFlow + SC(k=2) outperforms SC(k=4) at ~2.7x vs. 4x compute.
*   The Step-Saliency framework can be extended to information flow analysis in other long-sequence generation tasks, such as long-form writing or multi-turn dialogues.

## Rating
*   Novelty: ⭐⭐⭐⭐⭐ Step-level saliency + diagnostic-driven intervention is a completely new paradigm.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ 6 benchmarks, 5 backbones, detailed ablation, and compute-normalized comparisons.
*   Writing Quality: ⭐⭐⭐⭐⭐ Clear logical chain from diagnosis to intervention with well-designed figures.
*   Value: ⭐⭐⭐⭐⭐ Directly practical for understanding and improving reasoning models out-of-the-box.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] How Chain-of-Thought Works? Tracing Information Flow from Decoding, Projection, and Activation](how_chain-of-thought_works_tracing_information_flow_from_decoding_projection_and.md)
- [\[ACL 2026\] LLM Reasoning as Trajectories: Step-Specific Representation Geometry and Correctness Signals](llm_reasoning_as_trajectories_step-specific_representation_geometry_and_correctn.md)
- [\[ACL 2026\] On the Step Length Confounding in LLM Reasoning Data Selection](on_the_step_length_confounding_in_llm_reasoning_data_selection.md)
- [\[ACL 2026\] Step-GRPO: Internalizing Dynamic Early Exit for Efficient Reasoning](step-grpo_internalizing_dynamic_early_exit_for_efficient_reasoning.md)
- [\[ICML 2026\] The Deterministic Horizon: When Extended Reasoning Fails and Tool Delegation Becomes Necessary](../../ICML2026/llm_reasoning/the_deterministic_horizon_when_extended_reasoning_fails_and_tool_delegation_beco.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ACL 2026\] On the Step Length Confounding in LLM Reasoning Data Selection](on_the_step_length_confounding_in_llm_reasoning_data_selection.md)
- [\[ACL 2026\] How Chain-of-Thought Works? Tracing Information Flow from Decoding, Projection, and Activation](how_chain-of-thought_works_tracing_information_flow_from_decoding_projection_and.md)
- [\[ACL 2026\] Step-GRPO: Internalizing Dynamic Early Exit for Efficient Reasoning](step-grpo_internalizing_dynamic_early_exit_for_efficient_reasoning.md)
- [\[ICML 2026\] The Deterministic Horizon: When Extended Reasoning Fails and Tool Delegation Becomes Necessary](../../ICML2026/llm_reasoning/the_deterministic_horizon_when_extended_reasoning_fails_and_tool_delegation_beco.md)
- [\[ACL 2026\] LLM Reasoning as Trajectories: Step-Specific Representation Geometry and Correctness Signals](llm_reasoning_as_trajectories_step-specific_representation_geometry_and_correctn.md)

</div>

<!-- RELATED:END -->
