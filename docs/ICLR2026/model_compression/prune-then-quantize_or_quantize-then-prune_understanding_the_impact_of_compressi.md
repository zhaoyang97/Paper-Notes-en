---
title: >-
  [Paper Note] Prune-then-Quantize or Quantize-then-Prune? Understanding the Impact of Compression Order in Joint Model Compression
description: >-
  [ICLR 2026][Model Compression][Joint Compression] When combining pruning and quantization into a single pipeline, the order of execution significantly impacts final accuracy. This paper formalizes the long-neglected problem of "compression order optimization," proposes the "Progressive Intensity Hypothesis" (weaker perturbations first, stronger perturbations later), and provides theoretical proof alongside extensive empirical support across language and vision models.
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "Joint Compression"
  - "Compression Order"
  - "Pruning"
  - "Quantization"
  - "Progressive Intensity Hypothesis"
date: 2026-05-08
content_hash: 2fd58adc1e8210a3
---

# Prune-then-Quantize or Quantize-then-Prune? Understanding the Impact of Compression Order in Joint Model Compression

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=KWtOTMMvKU](https://openreview.net/forum?id=KWtOTMMvKU)  
**Code**: https://github.com/snudatalab/PQQP  
**Area**: Model Compression  
**Keywords**: Joint Compression, Compression Order, Pruning, Quantization, Progressive Intensity Hypothesis

## TL;DR
When combining pruning and quantization into a single pipeline, the order of execution significantly impacts final accuracy. This paper formalizes the long-neglected problem of "compression order optimization," proposes the "Progressive Intensity Hypothesis" (weaker perturbations first, stronger perturbations later), and provides theoretical proof alongside extensive empirical support across language and vision models.

## Background & Motivation
**Background**: Deploying models to edge devices requires compression. As individual methods (pruning, quantization, knowledge distillation, parameter sharing, low-rank decomposition) have their own performance ceilings, "joint compression"—stacking multiple methods—has become the mainstream strategy to maximize compression ratios, generally achieving a better "compression ratio vs. performance" trade-off than single-method approaches.

**Limitations of Prior Work**: A widely avoided variable in joint compression is the **compression order**. Since most compression techniques cannot be applied simultaneously and must be executed serially (pruning then quantization, or vice versa), the accuracy of models resulting from different orders can vary significantly. However, most existing works assume the two techniques are "orthogonal" and do not interfere, thus considering the order irrelevant. The few works that noticed the order issue only provided scattered empirical conclusions under specific settings.

**Key Challenge**: The only systematic theoretical work (Harma et al., 2025) proved that pruning and quantization are not orthogonal but limited its scope to "magnitude pruning + max-scale quantization," concluding that "pruning then quantization is always better." This conclusion is too narrow and impractical—it fails to characterize "when and why" a specific order is superior and cannot generalize to modern scenarios like rotation quantization, structured/unstructured pruning, or mixed precision.

**Goal**: (1) Formalize "finding the optimal compression order" as an optimizable problem; (2) Identify a universal law across methods and models to answer "which one first and why."

**Key Insight**: Instead of performing brute-force engineering experiments for every pair of methods, the authors characterize two attributes of each compression method using a unified language—**granularity** (how fine the structural units are) and **intensity** (the extent of performance degradation). They then derive how the order advantage changes from the perspective of the "intensity gap."

**Core Idea**: The paper proposes the **Progressive Intensity Hypothesis (PIH)**—applying the weaker perturbation first and the stronger perturbation later yields a better model. Furthermore, this "order advantage" increases monotonically with the performance gap between the two methods.

## Method
This is an analytical work. The "method" refers to a conceptual framework for characterizing compression order and theoretical proofs rather than a new compression algorithm. The logic involves formulating the problem as permutation optimization, defining metrics to compare "strength" across methods, proving the source and monotonicity of order advantage under the "Disjoint Selectivity" assumption, and handling real-world cases where the assumption fails using an "interference" term.

### Overall Architecture
Given a pre-trained model $\phi$, a set of compression methods $F=\{f_1,\dots,f_n\}$, and a performance metric $\mathcal{M}(\cdot)$ (where higher is better, e.g., accuracy or $-\text{perplexity}$), the goal is to find the optimal order $\pi^* = \arg\max_{\pi\in\Pi}\mathcal{M}(\pi(\phi))$ from all permutations $\Pi$.

To compare across methods, the authors abstract two attributes: **granularity** $t_f$ is the smallest structural unit (layer, sub-layer, attention head, etc.) a method acts on, and **intensity** measures the degree of performance degradation. For intensity, three metrics are defined: performance gap $G$, Compression Equivalent Ratio (CER), and order advantage $A$. They then use "Disjoint Selectivity" to determine if two methods interfere: if each unit is ultimately processed by only one method, it is proven that the order advantage stems entirely from "units whose ownership changes after reordering," supporting PIH. If not (e.g., quantization granularity is coarser than pruning), an **interference term $\Delta$** is introduced to describe extra error, which is shown to depend only on the pruning ratio and does not break the overall monotonic trend.

### Key Designs

**1. Formalization of Compression Order Optimization + PIH**

Addressing the issue of the "order problem being avoided due to orthogonality assumptions," the authors formulate it as a permutation optimization problem (Problem 1). Based on this, they propose Hypothesis 1: for two methods $f_1(\cdot;C_1)$ and $f_2(\cdot;C_2)$, where " $f_1$ then $f_2$" is denoted as $f_2\circ f_1$, **placing the stronger method later** yields a better model. This is refined into a falsifiable quantitative statement: order advantage $A(f_1\to f_2)$ increases **monotonically** with the performance gap $G(f_1,f_2)$ (or equivalently, the CER gap $C^*_{f_2}-C^*_{f_1}$). Intuitively, the more disparate the strengths, the greater the gain from "weak first, strong later."

**2. Cross-method Intensity Metrics: $G$, CER, and $A$**

Different methods cause different performance drops even at the same compression ratio. The authors unify intensity using three metrics. **Performance gap** $G(f_1,f_2)=\mathcal{M}(f_1(\phi;C_1))-\mathcal{M}(f_2(\phi;C_2))$ directly measures relative strength ($G>0$ means $f_2$ is stronger). Since $G$ values grow rapidly, the **Compression Equivalent Ratio (CER)** $C^*_f$ is introduced: it converts the effect of method $f$ into the "quantization ratio required to achieve the same performance." Finally, **compression order advantage** is $A(f_1\to f_2)=\mathcal{M}((f_2\circ f_1)(\phi))-\mathcal{M}((f_1\circ f_2)(\phi))$. These form the common language for all theory and experiments.

**3. Characterization and Monotonicity Proof under Disjoint Selectivity**

To explain "why" order matters, the authors introduce **Disjoint Selectivity (Definition 5)**: if each unit is processed by only one method (even if ownership changes with order), the assumption holds. Under inter-layer independence and performance-error proportionality (Assumption 1), **Theorem 1** proves that the order advantage $A(f_1\to f_2)$ is determined only by "units whose ownership changes." For "well-designed" pruning and symmetric zero-mean quantization (Assumption 2), **Theorem 2** proves that $A(Q\to P)$ increases monotonically with $C^*_P-C_Q$—a rigorous version of PIH for pruning-quantization.

**4. Interference Term $\Delta$: Granularity and the Breach of Disjoint Selectivity**

In reality, Disjoint Selectivity is not always met. Pruning always satisfies it because it involves "keeping or deleting whole blocks." Quantization satisfies it only if its granularity is **finer than or equal to** that of pruning. When pruning is finer ($t_P<t_Q$), it "cuts" part of a quantization unit, causing mutual interference. This is defined as **interference** $\Delta(\phi;f_1\to f_2)$. A key insight is that interference magnitude depends only on the pruning ratio $p$ and enters the order advantage additively, without destroying the monotonic trend. A special case is mixed-precision quantization, which naturally satisfies Disjoint Selectivity without interference.

## Key Experimental Results

### Main Results
Language models used LLaMA 2/3 (WikiText-2 $-\text{perplexity}$); vision models used ResNet-18 (CNN) and DeiT-Base (ViT) (ImageNet accuracy). The core validation is whether $A(Q\to P)$ increases monotonically with $C^*_P-C_Q$.

| Scenario | Models | Phenomenon | Conclusion |
|------|------|------|------|
| Pruning × Quantization | LLaMA 2 7B/13B, LLaMA 3 8B | $A(Q\to P)$ rises monotonically with $C^*_P-C_Q$ | Hypothesis holds across LLM scales/architectures |
| Multiple Combinations | LLaMA 3 8B | Holds for 3 pruning × 4 quantization methods | Independent of rotation or weight update designs |
| Pruning × Quantization | ResNet-18, DeiT-Base | Monotonic trend holds; advantage is larger than LLMs | Effect is stronger in vision models |

### Key Findings
- **Granularity Determines Interference**: Structured pruning has zero interference in early intervals (Finding 4).
- **Rotation Amplifies Pruning Effects**: Pruning after rotation quantization (e.g., QuaRot) leads to a sharp rise in perplexity because pruning does not account for rotation-induced errors. This suggests a need for "rotation-aware" pruning (Finding 3).
- **Strong Universality**: The hypothesis generalizes to multi-stage compression ($P\to Q\to P$), PEFT, parameter sharing, and mixed-precision quantization, all following "stronger later" (Findings 6-9).

## Highlights & Insights
- **Formalizing Order as an Optimizable/Provable Quantity**: Using CER to unify various compression intensities into a comparable scale is the key trick that transforms "which order is better" from engineering heuristics into a monotonic function of $C^*_P-C_Q$.
- **Clean Insight on Ownership**: Theorem 1 identifies that order advantage only comes from units that switch "executors" between orders, explaining why structured pruning can have zero advantage.
- **Interference Attributed to a Single Variable**: Attributing extra error solely to the pruning ratio and proving its additive nature makes the "validity despite interference" claim robust.
- **Practice Warning on Rotation**: Since modern quantization often uses rotation, the finding that rotation amplifies pruning errors warns practitioners against naive pruning after rotation.

## Limitations & Future Work
- The theory rests on idealized assumptions (inter-layer independence, linear performance-error relationship, symmetric quantization noise), which real-world methods may only approximate.
- Intensity is estimated via linear interpolation of CER; reliability may decrease if performance curves are non-monotonic.
- The hypothesis provides a qualitative/monotonic rule rather than an explicit formula to predict the exact optimal compression ratio distribution.
- Interference overlap in complex pipelines (>2 methods) remains to be fully characterized.

## Related Work & Insights
- **vs. Harma et al. (2025)**: They also proved non-orthogonality but only covered magnitude pruning and max-scale quantization, concluding "pruning then quantization" is always better. This paper provides a more general "stronger later" principle and quantifies advantage using CER, covering modern setups like rotation and mixed precision.
- **vs. Mainstream Joint Compression (Kurtic et al. 2022, Xiao et al. 2023, etc.)**: These usually assume orthogonality and free ordering. This paper proves order matters significantly when intensity gaps are large; the "free lunch" lies in choosing the correct order.
- **Insight**: Treating "compression order" as a schedulable hyperparameter and using CER to estimate strength gaps allows picking the superior order without additional computation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic characterization of compression order under general settings with a falsifiable hypothesis.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Validated across vision/language, multiple method combinations, and advanced setups (PEFT, Sharing).
- Writing Quality: ⭐⭐⭐⭐ Concepts are logically layered, though notation-heavy theoretical parts require careful reading.
- Value: ⭐⭐⭐⭐⭐ Provides a simple, actionable "weak first, strong later" rule for zero-cost improvement in joint compression.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Preserve-Then-Quantize: Balancing Rank Budgets for Quantization Error Reconstruction in LLMs](../../ICML2026/model_compression/preserve-then-quantize_balancing_rank_budgets_for_quantization_error_reconstruct.md)
- [\[ICML 2026\] ProjQ: Project-and-Quantize for Adapter-Aware LLM Compression](../../ICML2026/model_compression/projq_project-and-quantize_for_adapter-aware_llm_compression.md)
- [\[ICLR 2026\] AIRE-Prune: Asymptotic Impulse-Response Energy for State Pruning in State Space Models](aire-prune_asymptotic_impulse-response_energy_for_state_pruning_in_state_space_m.md)
- [\[ICLR 2026\] Large Language Model Compression with Global Rank and Sparsity Optimization](large_language_model_compression_with_global_rank_and_sparsity_optimization.md)
- [\[ICLR 2026\] Cut Less, Fold More: Model Compression through the Lens of Projection Geometry](cut_less_fold_more_model_compression_through_the_lens_of_projection_geometry.md)

</div>

<!-- RELATED:END -->
