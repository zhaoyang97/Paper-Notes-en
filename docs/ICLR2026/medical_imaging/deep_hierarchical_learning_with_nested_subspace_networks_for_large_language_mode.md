---
title: >-
  [Paper Note] Deep Hierarchical Learning with Nested Subspace Networks for Large Language Models
description: >-
  [ICLR 2026][Medical Imaging][Nested Subspace] This paper proposes Nested Subspace Networks (NSN), which reparameterize linear layers via low-rank decomposition into a strictly nested subspace hierarchy. Combined with unc…
tags:
  - "ICLR 2026"
  - "Medical Imaging"
  - "Nested Subspace"
  - "Dynamic Inference"
  - "Low-Rank Decomposition"
  - "Uncertainty-Aware Training"
  - "Elastic Computation"
date: 2026-05-08
content_hash: 8632df27072839ff
---

# Deep Hierarchical Learning with Nested Subspace Networks for Large Language Models

**Conference**: ICLR 2026
**arXiv**: [2509.17874](https://arxiv.org/abs/2509.17874)  
**Code**: [https://github.com/pauliusrauba/nested-subspace-networks](https://github.com/pauliusrauba/nested-subspace-networks)  
**Area**: Medical Imaging
**Keywords**: Nested Subspace, Dynamic Inference, Low-Rank Decomposition, Uncertainty-Aware Training, Elastic Computation

## TL;DR
This paper proposes Nested Subspace Networks (NSN), which reparameterize linear layers via low-rank decomposition into a strictly nested subspace hierarchy. Combined with uncertainty-aware multi-rank training, a single model can instantaneously trade off computation against performance at test time (50% FLOPs reduction with only 5% accuracy loss), and can be applied post-hoc to pretrained LLMs.

## Background & Motivation

**Background**: Large neural networks operate under fixed computational budgets, lacking flexibility for resource-constrained or dynamic deployment environments. Mainstream compression methods (pruning, distillation, LoRA) produce static models that cannot be adjusted at runtime.

**Limitations of Prior Work**:
   - Training a separate model for each computational budget is prohibitively expensive.
   - Variable-width networks (e.g., Slimmable Networks) must be trained from scratch and cannot be applied to pretrained models.
   - Existing methods offer only a few discrete operating points rather than a continuous, smooth spectrum.

**Key Challenge**: How can a single model simultaneously satisfy three requirements — (D1) instantaneous runtime adjustment, (D2) post-hoc applicability to arbitrary pretrained models, and (D3) a continuous and smooth computation–performance Pareto frontier?

**Key Insight**: Low-rank decomposition $W = BA$ naturally supports computation scaling by truncating the rank $r$. The key insight is that if models of different ranks form strictly nested subspaces, performance degradation is guaranteed to be monotone and smooth.

**Core Idea**: Linear layers are reparameterized as shared factor pairs $(A, B)$, where the rank-$r$ model uses the first $r$ rows of $A$ and the first $r$ columns of $B$, naturally forming a nested hierarchy. Uncertainty-weighted training then achieves Pareto-optimal coverage across ranks.

## Method

### Overall Architecture
MLP linear layers in a pretrained LLM are replaced by NSN layers (initialized via SVD), and the entire rank hierarchy is jointly optimized through uncertainty-aware multi-rank training. At inference, the desired FLOPs level is controlled instantaneously by selecting rank $r$.

### Key Designs

1. **Nested Subspace Architecture**:

    - Function: Guarantees that the function class of the rank-$r$ model is a strict subset of that of the rank-$(r+1)$ model.
    - Mechanism: Shared factor matrices $(A, B)$ are used; the effective weight at rank $r$ is $W_r = B_r A_r$ (first $r$ rows of $A$, first $r$ columns of $B$). Different operating points are simply different prefixes of the same $(A, B)$ — the input/output dimensions are unchanged, enabling direct insertion into existing Transformers.
    - Design Motivation: Unlike Slimmable Networks, NSN does not alter intermediate tensor shapes, requiring no modification to normalization layers or interfaces, thereby satisfying D2.

2. **Uncertainty-Aware Multi-Rank Training**:

    - Function: Jointly optimizes the entire rank hierarchy while automatically balancing contributions from different ranks.
    - Mechanism: A learnable variance $\sigma_k^2$ (parameterized as $s_k = \log \sigma_k^2$) is introduced for each rank $k$. The training objective is $\mathcal{L} = (\exp(-s_{\tilde{R}}) \cdot \mathcal{L}_{CE}(\tilde{R}) + s_{\tilde{R}}) + (\exp(-s_r) \cdot \mathcal{L}_{CE}(r) + s_r)$. At each step, an anchor rank $\tilde{R}$ (the maximum rank) and a variant rank $r$ are sampled.
    - Design Motivation: Low-rank models are inherently harder to train and thus require larger gradient weights. The $\exp(-s_k)$ term automatically achieves gradient balancing, with the closed-form optimal solution $w_k^* = 1/L_k$ — ranks with higher loss receive higher weight.

3. **Post-Hoc SVD Initialization**:

    - Function: Decomposes pretrained weights $W \approx BA$ to preserve learned information.
    - Mechanism: SVD is applied to each MLP linear layer to initialize NSN factor matrices, followed by multi-rank fine-tuning.
    - Design Motivation: Random initialization discards pretrained weight information; SVD preserves principal components ordered by singular values, naturally supporting the energy decay assumption.

4. **Theoretical Guarantees for Performance Interpolation**:

    - Function: Proves that reliable performance can be obtained at untrained intermediate ranks.
    - Mechanism: Under the energy decay assumption (basis vector energy $\|a_i\|$ decreases with $i$), the expected loss gap between any two ranks is bounded by the cumulative energy of the intermediate basis vectors.
    - Design Motivation: Only a small number of ranks are explicitly trained, yet arbitrary intermediate ranks must perform reliably — a theoretical guarantee is needed to ensure interpolation does not collapse.

### Loss & Training
- At each step, an anchor rank (maximum) and a randomly sampled variant rank are used; their respective CE losses are weighted by learnable variances.
- A curriculum learning strategy is applied to variant rank sampling, gradually expanding the rank range.
- Gradient contribution: $\nabla_\theta \mathcal{L} = \exp(-s_{\tilde{R}}) \nabla \mathcal{L}_{CE}(\tilde{R}) + \exp(-s_r) \nabla \mathcal{L}_{CE}(r)$

## Key Experimental Results

### Main Results

| Model | Task | Accuracy Loss at 50% FLOPs | Pareto Frontier |
|------|------|-------------------|----------|
| Pythia-2.8B | NLI | **only 5 pp** | smooth, monotone |
| GPT-Neo-2.7B | Classification | ~6 pp | smooth, monotone |
| Gemma-2B | Classification | ~5 pp | smooth, monotone |
| Qwen2-0.5B | Classification | ~4 pp | smooth, monotone |

### Ablation Study (CIFAR-10 MLP)

| Training Strategy | Anchor Acc | Avg ID | Avg OOD (Interp.) |
|---------|-----------|--------|--------------|
| CE Only (single rank) | 0.87 | 0.48 | 0.57 |
| Two CEs (Ours) | **0.88** | **0.79** | **0.81** |
| + Logits Reg | 0.87 | 0.64 | 0.64 |
| + Residual Ortho | 0.88 | 0.78 | 0.80 |

### Key Findings
- **Two CEs suffice**: The simple strategy of jointly training the anchor rank and a variant rank outperforms all additional regularization schemes — explicit regularization is in fact detrimental.
- **Energy decay assumption holds**: SVD initialization combined with multi-rank training naturally induces decreasing basis vector energy, a property not observed in standard MLPs.
- **Interpolation is reliable**: Untrained intermediate ranks exhibit smooth and predictable performance, validating the theoretical guarantees.
- **Learned log-variance reflects rank expressivity**: High rank → low variance (easy to learn); low rank → high variance (hard to learn), consistent with intuition.
- **Consistent across 4 LLMs**: Post-hoc adaptation produces smooth Pareto frontiers across all tested models.

## Highlights & Insights
- **The "nested subspace" concept is remarkably elegant**: It is more general than Slimmable Networks (no tensor shape changes), more flexible than LoRA (continuous rank adjustment), and more reversible than pruning (different prefixes of the same parameters). This architectural design may constitute a new paradigm.
- **The multi-task perspective of uncertainty weighting** cleverly reframes the problem of "different ranks = different task difficulties" as a classical multi-task learning problem, automatically resolved via the uncertainty weighting of Kendall et al.
- **Dual validation of interpolation reliability through theory and experiment** — not only is a formal energy decay bound established, but experiments fully confirm it. This is critical for real-world deployment: users can trust the performance at any intermediate rank.

## Limitations & Future Work
- A uniform rank is applied across all layers — layer-adaptive rank allocation (e.g., different ranks $r_l$ per layer) could further improve the Pareto frontier.
- Only MLP layers are replaced — the QKV projection layers in attention can also be NSN-ified.
- Experiments are limited in scale (up to 2.8B) — validation on 7B+ models is needed.
- Only classification tasks are evaluated — effects on generative tasks (e.g., text generation perplexity) remain unexplored.
- Multi-rank training requires two forward passes — training cost is approximately twice that of standard training.

## Related Work & Insights
- **vs LoRA**: LoRA produces a static adapter of fixed rank; NSN encodes all ranks within the same parameters, enabling instantaneous selection at inference.
- **vs Slimmable Networks**: Variable-width networks alter tensor shapes, making post-hoc application difficult; NSN changes only rank, not dimensionality.
- **vs MatFormer (Devvrit et al.)**: Also targets elastic inference, but relies on a granularized nested Transformer structure; NSN is more general (any linear layer can be replaced).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The nested subspace concept is elegant and original; uncertainty-based multi-rank training is ingenious.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four LLMs + ablations + theoretical validation; generative tasks and larger models are absent.
- Writing Quality: ⭐⭐⭐⭐⭐ The three-Desiderata framing is clear; theoretical derivations are rigorous; figures are well-crafted.
- Value: ⭐⭐⭐⭐⭐ A new paradigm for elastic inference that may reshape LLM deployment practice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Tracing Pharmacological Knowledge in Large Language Models](tracing_pharmacological_knowledge_in_large_language_models.md)
- [\[ICLR 2026\] NeuroCircuitry-Inspired Hierarchical Graph Causal Attention Networks for Explainable Depression Identification](neurocircuitry-inspired_hierarchical_graph_causal_attention_networks_for_explain.md)
- [\[CVPR 2026\] Automated Detection of Malignant Lesions in the Ovary Using Deep Learning Models and XAI](../../CVPR2026/medical_imaging/automated_detection_of_malignant_lesions_in_the_ov.md)
- [\[ICLR 2026\] Knowledgeable Language Models as Black-Box Optimizers for Personalized Medicine](knowledgeable_language_models_as_black-box_optimizers_for_personalized_medicine.md)
- [\[NeurIPS 2025\] Position: Thematic Analysis of Unstructured Clinical Transcripts with Large Language Models](../../NeurIPS2025/medical_imaging/position_thematic_analysis_of_unstructured_clinical_transcripts_with_large_langu.md)

</div>

<!-- RELATED:END -->
