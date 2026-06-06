---
title: >-
  [Paper Note] 3DrawAgent: Teaching LLM to Draw in 3D with Early Contrastive Experience
description: >-
  [CVPR 2026][3D Vision][3D sketch generation] This paper proposes 3DrawAgent, a training-free framework that enables a frozen LLM to acquire 3D spatial reasoning through *contrastive knowledge extraction* (CKE)…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "3D sketch generation"
  - "LLM"
  - "training-free"
  - "contrastive experience optimization"
  - "Bézier curves"
date: 2026-05-08
content_hash: e01e424815754826
---

# 3DrawAgent: Teaching LLM to Draw in 3D with Early Contrastive Experience

**Conference**: CVPR 2026
**arXiv**: [2604.08042](https://arxiv.org/abs/2604.08042)  
**Code**: None (LLM API-based)  
**Area**: 3D Vision / Generative AI
**Keywords**: 3D sketch generation, LLM, training-free, contrastive experience optimization, Bézier curves

## TL;DR
This paper proposes 3DrawAgent, a training-free framework that enables a frozen LLM to acquire 3D spatial reasoning through *contrastive knowledge extraction* (CKE), generating language-driven 3D Bézier sketches in an autoregressive manner without any parameter updates, achieving performance competitive with trained methods.

## Background & Motivation
**Background**: Language-driven 2D sketch generation has seen notable progress (e.g., SketchAgent), yet 3D sketch generation remains largely unexplored. Existing 3D shape generation methods (diffusion-based or neural implicit approaches) require explicit geometric supervision or extensive training.

**Limitations of Prior Work**:
   - Diffusion-based 3D sketch methods (Diff3DS, Dream3DVG) rely on SDS optimization, which is computationally intensive.
   - SketchAgent is confined to 2D coordinate spaces and cannot reason about depth or projection.
   - Training-free GRPO depends on scalar rewards or ground-truth references, making it unsuitable for open-ended creative generation tasks.

**Core Idea**: LLMs inherently possess strong sequential reasoning capabilities. By combining carefully designed in-context prompts with self-contrastive feedback, it is possible to "teach" an LLM to perform 3D drawing without any gradient updates.

## Method

### Overall Architecture
Text description → LLM autoregressively generates 3D Bézier curve control points → Differentiable renderer produces multi-view renderings → CLIP scoring + LLM quality judgment → Construction of contrastive experience pairs → Experience library update → Guidance for subsequent generation.

### Key Designs
1. **Linguistic Representation of 3D Sketches**:
   3D Bézier curves are expressed as structured text: $a_t = \text{draw\_bezier}[(\mathbf{P}^{(0)}, \mathbf{P}^{(1)}, \mathbf{P}^{(2)}, \mathbf{P}^{(3)})]$, where each control point $\mathbf{P}^{(k)} \in \mathbb{R}^3$. The full sketch constitutes an action sequence $\mathcal{A} = \{a_1, ..., a_N\}$.
    - Prompt design includes: role instructions, output format specification, data type constraints, coordinate system definition, ground-truth examples, and boundary rules.
    - **Design Motivation**: Allows the LLM to operate within its familiar text space, avoiding cross-modal conversion.

2. **Contrastive Knowledge Extraction (CKE)**:
   The core innovation—extending training-free GRPO to a pairwise contrastive setting:
    - $K=5$ candidate sketches are sampled per query.
    - Multi-view CLIP scoring: $r_{\text{CLIP}} = \frac{1}{V}\sum_{v=1}^{V} \cos(E_I(I_v), E_T(\mathcal{T}))$
    - Contrastive pairs $(\mathcal{S}_i^+, \mathcal{S}_j^-)$ are constructed where $r_i > r_j$.
    - The LLM acts as a "semantic advantage judge," analyzing the reasons behind quality differences.
    - Extracted knowledge updates the experience library: $\mathcal{E} \leftarrow \text{Update}(\mathcal{E}, A^{\text{text}})$
    - **Design Motivation**: Requires no GT 3D sketches, no gradients, and no structured group rollout.

3. **Experience-Guided 3D Drawing**:
   The experience library $\mathcal{E}$ is injected into the context window as an additional prompt segment: $o = p_\theta(o | \mathcal{T}, \mathcal{E})$
    - Experiences encode transferable geometric principles (curvature continuity, symmetric topology preservation, etc.).
    - Through iterative accumulation, the LLM progressively internalizes 3D perception strategies.

### Loss & Training
- **Fully training-free**: The LLM (DeepSeek-V3.2-Exp / Gemini-2.5Pro) is kept frozen throughout.
- Temperature 0.7 is used during contrastive extraction to encourage diversity; temperature 0.3 is used at inference for quality.
- Optimal performance is reached after approximately 2–3 epochs of experience extraction (with slight degradation thereafter due to LLM over-reasoning).
- Requires only a single RTX 3090 GPU; the primary compute cost lies in LLM API calls.

## Key Experimental Results

### Main Results

| Method | Training Required | CLIP-ST (Category) | AES (Category) | CLIP-ST (Fine-grained) | AES (Fine-grained) |
|------|--------|------|------|------|------|
| Diff3DS | ✓ | 0.648 | 3.791 | 0.650 | 3.770 |
| Dream3DVG | ✓ | 0.660 | 4.150 | 0.670 | 4.174 |
| **3DrawAgent (Gemini)** | **✗** | **0.649** | **4.161** | **0.669** | **4.175** |

### Ablation Study

| Configuration | Ep0 | Ep1 | Ep2 | Ep3 | Notes |
|------|-----|-----|-----|-----|------|
| w/o CKE (baseline) | 0.5735 | - | - | - | No experience lower bound |
| w/ CKE | 0.5735 | 0.6461 | **0.6643** | 0.6428 | Rises then slightly falls |
| K=2 | 0.5735 | 0.5947 | 0.6493 | - | Insufficient contrast |
| K=5 (default) | 0.5735 | 0.6461 | **0.6643** | - | Optimal balance |
| K=10 | 0.5735 | 0.6135 | 0.5612 | - | Excessive noise |
| w/o GT | 0.5735 | 0.6461 | **0.6643** | - | GT-independent |
| w/ GT | 0.5735 | 0.6648 | 0.6552 | - | Faster early gain, slightly lower sustained performance |

### Key Findings
- The training-free approach is competitive with trained methods: CLIP-ST gap of only 0.001, AES nearly on par.
- CKE is effective: CLIP-ST improves from 0.5735 to 0.6643 (+15.8%).
- Ground-truth references are not required: self-supervised CLIP signals are sufficient.
- User study: 3DrawAgent achieves a 46.66% preference rate, substantially outperforming Dream3DVG (36.67%) and Diff3DS (16.67%).
- Experience analysis reveals a clear learning progression: basic shape construction → spatial awareness → control point precision → format self-verification.

## Highlights & Insights
- **New Paradigm**: The LLM serves simultaneously as generator and critic, achieving self-improvement through self-critique without any training.
- **Interesting Experience Evolution**: The learning curve progresses from geometric correctness to spatial expressiveness and finally to format self-verification, resembling human-like skill acquisition.
- **High Practicality**: Requires only an LLM API and a single GPU, making the barrier to entry extremely low.
- Statistical analysis over 200 rollouts reveals behavioral patterns in LLM-based 3D content generation.

## Limitations & Future Work
- Performance degrades after 2–3 epochs of experience accumulation due to "over-reasoning"; sustained improvement remains an open challenge.
- Generation quality is bounded by the LLM's inherent 3D spatial reasoning capacity.
- The Bézier curve representation has limited expressive power, making complex topologies difficult to capture.
- Contrastive evaluation relies on CLIP, which may be insufficiently precise for 3D sketch assessment.
- Generation speed is constrained by LLM API latency.

## Related Work & Insights
- This work naturally and elegantly extends the 2D sketch paradigm of SketchAgent to 3D.
- The generalization from training-free GRPO to contrastive experience optimization is noteworthy and may apply to other creative generation tasks.
- The potential of LLMs as 3D spatial reasoners warrants deeper investigation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Training-free 3D sketch generation combined with contrastive experience optimization represents a genuinely new paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ Detailed ablations with in-depth statistical analysis.
- Writing Quality: ⭐⭐⭐⭐ Method presentation is clear, though some details are deferred to the appendix.
- Value: ⭐⭐⭐⭐ Strongly inspiring, though 3D sketch generation remains a relatively niche application.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ArtLLM: Generating Articulated Assets via 3D LLM](artllm_generating_articulated_assets_via_3d_llm.md)
- [\[CVPR 2026\] NTK-Guided Implicit Neural Teaching](ntk-guided_implicit_neural_teaching.md)
- [\[AAAI 2026\] UniC-Lift: Unified 3D Instance Segmentation via Contrastive Learning](../../AAAI2026/3d_vision/unic-lift_unified_3d_instance_segmentation_via_contrastive_learning.md)
- [\[AAAI 2026\] NURBGen: High-Fidelity Text-to-CAD Generation through LLM-Driven NURBS Modeling](../../AAAI2026/3d_vision/nurbgen_high-fidelity_text-to-cad_generation_through_llm-driven_nurbs_modeling.md)
- [\[CVPR 2026\] Extend3D: Town-Scale 3D Generation](extend3d_town-scale_3d_generation.md)

</div>

<!-- RELATED:END -->
