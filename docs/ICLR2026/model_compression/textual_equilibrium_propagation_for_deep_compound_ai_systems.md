---
title: >-
  [Paper Note] Textual Equilibrium Propagation for Deep Compound AI Systems
description: >-
  [ICLR 2026][Model Compression][Compound AI systems] This paper proposes Textual Equilibrium Propagation (TEP), a compound AI system optimization method grounded in local learning principles. Through a two-phase design co…
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "Compound AI systems"
  - "textual gradients"
  - "equilibrium propagation"
  - "prompt optimization"
  - "multi-agent workflows"
date: 2026-05-08
content_hash: 07ea2a0493b2f99b
---

# Textual Equilibrium Propagation for Deep Compound AI Systems

**Conference**: ICLR 2026
**arXiv**: [2601.21064](https://arxiv.org/abs/2601.21064)  
**Code**: Not released  
**Area**: Model Compression / Compound AI System Optimization
**Keywords**: Compound AI systems, textual gradients, equilibrium propagation, prompt optimization, multi-agent workflows

## TL;DR

This paper proposes Textual Equilibrium Propagation (TEP), a compound AI system optimization method grounded in local learning principles. Through a two-phase design consisting of a free phase and a nudged phase, TEP avoids gradient explosion/vanishing in global textual backpropagation and significantly outperforms TextGrad on deep workflows.

## Background & Motivation

Modern compound AI systems integrate multiple modules (retrievers, tools, verifiers, etc.) working in concert, requiring end-to-end optimization of the entire pipeline. TextGrad pioneered "automatic differentiation through text," propagating textual feedback from downstream to upstream modules via LLM-as-judge to update prompts.

However, as system depth increases, TextGrad encounters two critical failure modes:

**Textual gradient explosion**: Feedback accumulates across layers, causing message lengths to grow exponentially ($\mathbb{E}[B(g_u)] \geq c\gamma^k, \gamma > 1$), eventually exceeding the LLM context window, while LLM-as-judge biases compound along the chain.

**Textual gradient vanishing**: When feedback is compressed to control length, actionable information is progressively lost ($\mathbb{E}[S(g_u)] \leq C\alpha^k, \alpha \in (0,1)$), leaving upstream modules with vague and uninformative suggestions such as "improve efficiency."

The root cause of both problems is that global textual backpropagation does not scale to deep compound AI systems.

## Method

### Overall Architecture

TEP models a compound AI system as a stochastic computation graph (SCG) $G=(V,E)$, where nodes are LLM agents and edges represent data flow. The optimization objective is:

$$J(\theta) = \mathbb{E}_{o \sim D_{\text{task}}} \mathbb{E}_{Z \sim P_\theta(\cdot | o)} [\ell(o, Z)]$$

TEP adopts a two-phase optimization strategy inspired by equilibrium propagation in energy-based models.

### Key Design 1: Free Phase

Each node $v$ is equipped with a local LLM critic operating under a structured scoring rubric $\theta_v^{\text{critic}}$ and an adjustable temperature parameter $\theta_v^{\text{temp}} \sim \mathcal{U}(0.3, 0.9)$. The critic generates feedback $g_v = C(z_v, \theta_v^{\text{critic}})$ solely based on the node's own output, **without relying on downstream gradients** $g'$.

Iterative optimization proceeds until scores stabilize — i.e., a local "equilibrium state" $x_\star^0(\theta)$ is reached, at which point the critic deems no further improvement necessary.

### Key Design 2: Nudged Phase

Starting from the free-phase equilibrium, each node receives **bounded minimal prompt edits**, guided by task-level objectives via forward signals rather than a backward feedback chain. The perturbation magnitude is constrained to avoid disrupting the local optimum established during the free phase.

The system iterates again until a nudged equilibrium is reached, which differs from the free-phase equilibrium.

### Local Update Rule

$$\theta_v' = U_v(g_v^f, g_v^n, \theta_v)$$

where $g_v^f$ and $g_v^n$ denote the feedback signals from the free and nudged phases respectively, and $U_v$ is an LLM-defined update operator. Each update is validated on a held-out set; only edits that do not degrade performance are retained.

### Loss & Training

TEP does not employ an explicit numerical loss function. Instead, optimization is performed implicitly through textual scores from local LLM critics and held-out set performance. The core constraints are:
- Bounded feedback length: $B(g) \ll \text{context limit}$
- Preserved feedback quality: $S(g) \geq \tau$

## Key Experimental Results

### Main Results

| Method | PubMedQA (Acc.) | STARK-PRIME (MRR) | HotpotQA (F1) | BigCodeBench (Pass) |
|------|----------------|-------------------|---------------|---------------------|
| CoT | 57.34±1.12 | 39.76±0.84 | 33.92±0.76 | 34.15±1.43 |
| DSPy | 60.26±0.40 | 41.40±0.04 | 44.90±0.32 | 33.81±2.75 |
| TextGrad | 56.96±2.24 | 41.31±1.67 | 24.86±1.19 | 35.71±0.10 |
| TextGrad+Sum | 56.12±1.85 | 40.72±1.21 | 24.12±1.25 | 35.12±0.67 |
| **TEP** | **62.02±1.31** | **42.72±0.65** | **48.72±1.32** | **38.97±0.39** |

TEP achieves the best results across all four tasks, outperforming the second-best method by 8.1% on HotpotQA and 3.4% on BigCodeBench.

### Ablation Study

| Configuration | HotpotQA F1 | BigCodeBench Pass@1 |
|------|-------------|---------------------|
| Full TEP | 48.72 | 38.97 |
| w/o nudged phase | 22.3 (-26.4) | significant drop |
| w/o free phase | 36.8 (-11.9) | 36.3 (-2.7) |

Removing the nudged phase leads to severe degradation (−26.4 points on HotpotQA), demonstrating that purely local equilibrium is insufficient for system-level coordination. Removing the free phase also has a notable impact, as it provides a high-quality initialization for effective perturbation.

### Key Findings

- **Depth scaling**: TextGrad's feedback token count grows from ~2K at scale=1 to 32K+ at scale=5 (approximately $2.2^s$ exponential growth); TEP maintains nearly constant token complexity.
- **Effective update rate**: TextGrad+Sum's effective update rate drops from 36% to 5%, while TEP decreases only modestly from 37% to 33%.
- **Single-node optimization**: TEP achieves 44.5% on GPQA (vs. 41.0% for TextGrad) and 81.6% on Object Counting (vs. 74.2% for TextGrad).

## Highlights & Insights

1. **Precise analogy**: The paper rigorously maps gradient pathologies in deep neural networks to textual feedback failures in compound AI systems, providing formal definitions of textual gradient explosion and vanishing.
2. **Bio-inspired design**: Transferring the concept of equilibrium propagation from energy-based models to the textual domain is an exemplary case of cross-domain methodological adaptation.
3. **Strong practicality**: The modular black-box design requires no access to model parameters, making TEP applicable to arbitrary LLM pipelines.
4. **Advantage scales with depth**: Unlike TextGrad, TEP's performance advantage grows as workflow depth increases.

## Limitations & Future Work

- The free phase (20 iterations) and nudged phase (40 iterations) introduce additional computational overhead.
- Scoring rubrics for local critics require manual design and may need to be tailored for different tasks.
- Validation is limited to fixed SCG structures; dynamic graph optimization remains unexplored.
- No automated method exists for selecting the perturbation strength hyperparameter.

## Related Work & Insights

- **TextGrad** (Yuksekgonul et al., 2025): Pioneer of global textual backpropagation.
- **DSPy** (Khattab et al., 2024): Programmatic prompt compilation framework.
- **OPTIMAS** (Wu et al., 2025): Local training rewards requiring parameter fine-tuning.
- **Self-Refine** (Madaan et al., 2023): Iterative self-improvement.
- **Equilibrium Propagation** (Scellier & Bengio, 2017): Local learning principles for energy-based models.

## Rating

- Novelty: ⭐⭐⭐⭐ (innovative analogy from equilibrium propagation to the textual domain)
- Theory: ⭐⭐⭐⭐ (rigorous formalization of textual gradient failure modes with convergence guarantees)
- Experimental Thoroughness: ⭐⭐⭐⭐ (4 benchmarks + depth scaling analysis + ablation study)
- Value: ⭐⭐⭐⭐ (model-agnostic, applicable to arbitrary LLM pipelines)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Rejuvenating Cross-Entropy Loss in Knowledge Distillation for Recommender Systems](rejuvenating_cross-entropy_loss_in_knowledge_distillation_for_recommender_system.md)
- [\[ICLR 2026\] Bridging Kolmogorov Complexity and Deep Learning: Asymptotically Optimal Description Length Objectives for Transformers](bridging_kolmogorov_complexity_and_deep_learning_asymptotically_optimal_descript.md)
- [\[ICLR 2026\] Paper Copilot: Tracking the Evolution of Peer Review in AI Conferences](paper_copilot_tracking_the_evolution_of_peer_review_in_ai_conferences.md)
- [\[NeurIPS 2025\] Quantization Error Propagation: Revisiting Layer-Wise Post-Training Quantization](../../NeurIPS2025/model_compression/quantization_error_propagation_revisiting_layer-wise_post-training_quantization.md)
- [\[NeurIPS 2025\] Efficient Parametric SVD of Koopman Operator for Stochastic Dynamical Systems](../../NeurIPS2025/model_compression/efficient_parametric_svd_of_koopman_operator_for_stochastic_dynamical_systems.md)

</div>

<!-- RELATED:END -->
