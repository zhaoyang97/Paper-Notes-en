---
title: >-
  [Paper Note] Proof-of-Perception: Certified Tool-Using Multimodal Reasoning with Compositional Conformal Guarantees
description: >-
  [CVPR 2026][Multimodal VLM][Conformal Prediction] This paper proposes Proof-of-Perception (PoP), which models multimodal reasoning as an executable directed acyclic graph (DAG) where each perception/logic node outputs set-valued predictions with conformal certificates providing step-wise reliability guarantees. A lightweight controller adaptively allocates computation within a budget based on these certificates. PoP outperforms CoT, ReAct, and PoT baselines on document, chart, and multi-image QA benchmarks.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Conformal Prediction
  - Tool Use
  - Multimodal Reasoning
  - Uncertainty Quantification
  - Adaptive Computation
date: 2026-05-08
content_hash: c598792367abe0a3
---

# Proof-of-Perception: Certified Tool-Using Multimodal Reasoning with Compositional Conformal Guarantees

**Conference**: CVPR 2026
**arXiv**: [2603.00324](https://arxiv.org/abs/2603.00324)
**Code**: [https://github.com/AryaFayyazi/PoP](https://github.com/AryaFayyazi/PoP)
**Area**: Multimodal Reasoning / Reliable AI
**Keywords**: Conformal Prediction, Tool Use, Multimodal Reasoning, Uncertainty Quantification, Adaptive Computation

## TL;DR
This paper proposes Proof-of-Perception (PoP), which models multimodal reasoning as an executable directed acyclic graph (DAG) where each perception/logic node outputs set-valued predictions with conformal certificates providing step-wise reliability guarantees. A lightweight controller adaptively allocates computation within a budget based on these certificates. PoP outperforms CoT, ReAct, and PoT baselines on document, chart, and multi-image QA benchmarks.

## Background & Motivation

**Background**: Multimodal LLMs have made progress on document understanding, chart reasoning, and related tasks, but typically conflate fine-grained perception (OCR, detection, chart parsing) with symbolic reasoning within a single forward pass. Tool use and structured prompting strategies (CoT, ReAct, PoT) have partially addressed this issue.

**Limitations of Prior Work**: (1) Intermediate steps output single-point estimates, silently propagating errors; (2) compute allocation relies on heuristics (fixed retry counts, uncalibrated thresholds), precluding principled accuracy-cost trade-offs; (3) calibration, when present, is applied only to final answers, leaving step-wise reliability of intermediate outputs unguaranteed.

**Key Challenge**: Existing methods make a hard commitment at each intermediate perception step—once an OCR result is wrong by a character or a detection misses a bounding box, subsequent reasoning is forced to rationalize upon erroneous foundations. Moreover, there is no principled criterion for deciding when to scale up reasoning (via additional tool calls) versus terminating early.

**Goal**: How can reliability guarantees be provided for every intermediate step in multi-step multimodal reasoning, and how can such uncertainty be converted into a principled compute allocation strategy?

**Key Insight**: Conformal Prediction provides finite-sample coverage guarantees without distributional assumptions. Applying it to each node in a reasoning DAG replaces single-point outputs with sets that carry formal coverage guarantees.

**Core Idea**: Apply conformal prediction at each perception/logic node in the reasoning DAG to produce calibrated set-valued outputs; a controller then decides whether to accept, retry, or expand computation based on set size and remaining budget.

## Method

### Overall Architecture
Given a multi-image and text query, an MLLM planner generates a DSL program defining the reasoning DAG $G=(V,E)$. Tool nodes invoke external perception tools (OCR, detection, chart parsing), while fusion nodes aggregate upstream results within the MLLM. Each node is equipped with a certificate head that outputs nonconformity scores; split-conformal calibration determines thresholds and produces set-valued predictions. A controller observes node-level certificates and the global budget, then issues ACCEPT/RETRY/EXPAND/ABORT decisions.

### Key Designs

1. **Node-Level Conformal Certificates**:

    - **Function**: For each node type (OCR / detection / chart parsing / logical fusion), define nonconformity functions and calibration thresholds to produce set-valued predictions.
    - **Mechanism**: For node type $t$, the nonconformity function $s^{(t)}(x_v, z)$ measures the degree of "anomaly" of candidate output $z$. A threshold $\tau_\delta^{(t)} = \alpha_{(k)}^{(t)},\ k = \lceil(n_t+1)(1-\delta)\rceil$ is computed from a calibration set. The set prediction $\Gamma_\delta^{(t)}(x_v) = \{z : s^{(t)}(x_v, z) \leq \tau_\delta^{(t)}\}$ guarantees coverage probability $\geq 1-\delta$.
    - **Design Motivation**: Single-point predictions silently propagate errors at intermediate steps. Set-valued predictions retain multiple calibrated candidates until evidence disambiguates, reducing error cascades.

2. **Adaptive Controller for Compute Allocation**:

    - **Function**: A lightweight policy network $\pi_\phi$ takes each node's certificate state $c_v$ (threshold, set size, node type) and global budget $b$ as input, and outputs action $a_v \in \{\text{ACCEPT, RETRY, EXPAND, ABORT}\}$.
    - **Mechanism**: ACCEPT retains the current set; RETRY re-executes the node at higher fidelity (e.g., high-resolution crop); EXPAND adds new child nodes (e.g., additional OCR calls); ABORT terminates early when the budget is exhausted. The controller is optimized via policy gradient on $R(x) = -C_{err}(x) - \beta C_{comp}(x)$.
    - **Design Motivation**: Uncertainty should actively guide compute allocation rather than serve as passive scoring—large sets warrant expanded computation, while small (high-confidence) sets justify early termination.

3. **Self-Play Counterexample Mining**:

    - **Function**: During training, a frozen adversary generates perturbed inputs (crops, affine transforms, OCR noise) and filters failure cases to augment both the student model and the calibration set.
    - **Mechanism**: The adversary executes the inference graph and applies controlled perturbations to inputs, selecting samples with incorrect predictions or high nonconformity scores as counterexamples. These are used to train the student to maintain coverage and are appended to the calibration pool so that thresholds reflect realistic failure patterns.
    - **Design Motivation**: Standard calibration assumes exchangeability, but certificates may fail under distribution shift. Self-play ensures calibration remains reliable under adversarial perturbations.

### Loss & Training
The total loss is $\mathcal{L} = \mathcal{L}_{task} + \gamma_{plan}\mathcal{L}_{plan} + \gamma_{cert}\mathcal{L}_{cert} + \gamma_{ctrl}\mathcal{L}_{ctrl}$, comprising: a task loss (final answer accuracy), a planning loss (cross-entropy over program generation sequences), a certificate loss (margin constraints enforcing coverage), and a controller loss (policy gradient optimizing the accuracy-cost trade-off).

## Key Experimental Results

### Main Results

| Method | DocVQA | TextVQA | InfoVQA | ChartQA | MultiDoc2Dial |
|--------|--------|---------|---------|---------|---------------|
| CoT (GPT-4V) | 74.2 | 68.1 | 51.3 | 71.8 | 42.5 |
| ReAct | 76.8 | 70.3 | 54.1 | 74.2 | 45.7 |
| PoT | 78.1 | 71.5 | 56.4 | 76.9 | 47.2 |
| **PoP** | **82.3** | **75.8** | **61.2** | **80.5** | **52.8** |

### Ablation Study

| Configuration | DocVQA | Compute Cost (normalized) |
|---------------|--------|---------------------------|
| PoP (full) | 82.3 | 1.0x |
| w/o Conformal (single-point) | 77.5 | 0.8x |
| w/o Controller (fixed expansion) | 80.1 | 1.6x |
| w/o Self-Play | 80.8 | 1.0x |

### Key Findings
- PoP outperforms CoT, ReAct, and PoT baselines on all five benchmarks, with gains of 4.2% on DocVQA and 3.6% on ChartQA.
- Removing conformal certificates (degrading to single-point predictions) causes a substantial performance drop, validating the value of set-valued intermediate outputs.
- Removing the controller increases compute cost by 60% with only marginal performance improvement, demonstrating that the controller effectively eliminates unnecessary computation.
- Self-play mining contributes a 1.5% performance gain and improves robustness under distribution shift.

## Highlights & Insights
- The core insight is **transforming uncertainty from "passive scoring" into an "active compute strategy"**: large conformal sets trigger additional computation (EXPAND), while small sets enable early termination (ACCEPT).
- Compositional conformal guarantees (coverage $1-\delta$ at each step) are more principled than calibrating only at the final answer, enabling error attribution to specific steps.
- The framework is highly modular; the tool set and node types can be extended flexibly.

## Limitations & Future Work
- Conformal prediction assumes exchangeability; although self-play partially mitigates this, coverage guarantees may break down under severe distribution shift.
- Candidate set size is bounded by the number of beam search or sampling candidates $K_{max}$, potentially excluding the correct answer.
- The controller's discrete action space (four actions) may be overly simplistic; finer-grained compute allocation strategies remain an open direction.

## Related Work & Insights
- **vs. CoT/ReAct**: These methods rely on single-point intermediate outputs and heuristic compute control, with no reliability guarantees; PoP provides per-step coverage guarantees and principled compute allocation.
- **vs. Traditional Conformal Prediction**: Prior work typically applies conformal prediction only to final predictions; PoP embeds it at every node in a multi-step reasoning pipeline.
- **vs. ViperGPT/VisualProg**: These methods support programmatic reasoning but lack uncertainty quantification at intermediate steps.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of conformal prediction, tool use, and adaptive compute control is proposed for the first time in multimodal reasoning.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five benchmarks, comprehensive ablations, and cost analysis.
- Writing Quality: ⭐⭐⭐⭐ Theoretically rigorous with complete formalization.
- Value: ⭐⭐⭐⭐⭐ Significant implications for reliable AI reasoning; the paradigm of conformal certificates with compute control is broadly transferable.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Conditional Factuality Controlled LLMs with Generalization Certificates via Conformal Sampling](conditional_factuality_controlled_llms_with_generalization_certificates_via_conf.md)
- [\[CVPR 2026\] CodeDance: A Dynamic Tool-integrated MLLM for Executable Visual Reasoning](codedance_a_dynamic_tool-integrated_mllm_for_executable_visual_reasoning.md)
- [\[AAAI 2026\] VipAct: Visual-Perception Enhancement via Specialized VLM Agent Collaboration and Tool-use](../../AAAI2026/multimodal_vlm/vipact_visual-perception_enhancement_via_specialized_vlm_age.md)
- [\[CVPR 2026\] From Intuition to Investigation: A Tool-Augmented Reasoning MLLM Framework for Generalizable Face Anti-Spoofing](from_intuition_to_investigation_a_tool-augmented_reasoning_mllm_framework_for_ge.md)
- [\[CVPR 2026\] Downscaling Intelligence: Exploring Perception and Reasoning Bottlenecks in Small VLMs](downscaling_intelligence_exploring_perception_and_reasoning_bottlenecks_in_small.md)

<!-- RELATED:END -->
