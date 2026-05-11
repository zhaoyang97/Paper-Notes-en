---
title: >-
  [Paper Note] Modality-free Graph In-context Alignment
description: >-
  [ICLR 2026][Model Compression][Graph Foundation Models] This paper proposes MF-GIA, the first graph in-context learning framework that simultaneously satisfies three conditions: no post-training, cross-domain alignment…
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "Graph Foundation Models"
  - "In-Context Learning"
  - "Cross-Domain Alignment"
  - "Gradient Fingerprint"
  - "Meta-Learning"
date: 2026-05-08
content_hash: c922a541ce0d2876
---

# Modality-free Graph In-context Alignment

**Conference**: ICLR 2026
**arXiv**: [2603.13434](https://arxiv.org/abs/2603.13434)
**Code**: [GitHub](https://github.com/JhuoW/MF-GIA)
**Area**: Model Compression
**Keywords**: Graph Foundation Models, In-Context Learning, Cross-Domain Alignment, Gradient Fingerprint, Meta-Learning

## TL;DR
This paper proposes MF-GIA, the first graph in-context learning framework that simultaneously satisfies three conditions: no post-training, cross-domain alignment, and modality-agnosticism. By capturing domain characteristics via gradient fingerprints, aligning features and labels through FiLM-conditioned transformations, MF-GIA achieves state-of-the-art performance on few-shot tasks across multiple graph domains.

## Background & Motivation
For graph foundation models (GFMs) to achieve LLM-level generality, they require genuine in-context learning (ICL) capability—adapting to new tasks from a handful of examples without parameter updates. True graph ICL must satisfy three conditions:

**Post-training-free inference**: Parameters are fully frozen at inference time, with no fine-tuning or learnable prompt engineering required.

**Cross-domain alignment**: A single model handles different graph types within a unified semantic space.

**Modality-agnosticism**: The model operates on pre-encoded graphs without access to raw data—a practical requirement since graph data is often encoded by domain-specific methods.

Existing methods (e.g., UniGraph, OFA, GOFA) achieve alignment through text-attributed graphs (TAGs), but require access to raw data, which is infeasible in privacy-sensitive scenarios and introduces information loss through text conversion. Prodigy and GPF lack cross-domain alignment.

**Core Idea**: Gradient fingerprints serve as domain descriptors—the displacement induced by a one-step gradient update reflects how a graph's features, labels, and topology affect the shared encoder, thereby capturing domain characteristics. Lightweight FiLM transformations conditioned on these fingerprints align features and labels across domains without requiring knowledge of raw data modalities.

## Method

### Overall Architecture
MF-GIA consists of three components: ① a domain embedder that encodes domain characteristics via gradient fingerprints; ② domain-conditioned alignment that maps pre-encoded features and index labels from each domain into a unified space; and ③ episodic pre-training that learns few-shot matching via DPAA attention. At inference time, all parameters are frozen, and only the support set is needed to trigger alignment and prediction.

### Key Designs
1. **Domain Embedder (Gradient Fingerprint)**:

    - Function: Produces a compact domain embedding $e_i$ for each graph.
    - Mechanism: Starting from a shared initialization $\theta_0$, a one-step gradient update on each graph $G_i$ yields a fingerprint $\Delta\theta_i = \theta_i - \theta_0$. A learnable embedder (Conv2D + MLP) maps the fingerprint to a low-dimensional vector $e_i = f_{\phi_{\text{de}}}(\Delta\theta_i)$.
    - Theoretical guarantee (Theorem 3.1): $\|e_i - e_j\|_2 \leq \tilde{C} \cdot \mathcal{W}_2(\mathcal{D}_i, \mathcal{D}_j)$, bounding the domain embedding distance by the Wasserstein distance between domain distributions.
    - Design Motivation: No external domain labels or modal metadata are required; gradient patterns intrinsically reflect the characteristics of the data distribution.

2. **Domain-Conditioned Feature and Label Alignment**:

    - Feature alignment: Applies a FiLM transformation $z_{i,w} = \gamma_i^{\text{feat}} \odot h_{i,w} + \beta_i^{\text{feat}}$, where $(\gamma, \beta) = f_{\phi_{\text{feat}}}(e_i)$. Similar domains yield similar $e_i$, hence similar transformations that map features to neighboring subspaces.
    - Label alignment: Maintains a shared label basis $\mathbf{E}^{\text{label}} \in \mathbb{R}^{L_{\max} \times d}$, similarly conditioned via FiLM: $u_{i,l} = \gamma_i^{\text{label}} \odot \mathbf{E}_l^{\text{label}} + \beta_i^{\text{label}}$.
    - Design Motivation: The same label ID may represent entirely different concepts across domains; domain-conditioned transformations resolve this semantic inconsistency.

3. **Dual Prompt-Aware Attention (DPAA)**:

    - Function: Enables prompt-based few-shot prediction.
    - Mechanism: Two layers of single-query attention—on the feature side, the query attends to support features to produce a prompt-conditioned representation $z_{i,q}^{\text{out}}$; on the label side, this representation attends to label prototypes to produce prediction $u_{i,q}^{\text{out}}$. The final score is $s = u^{\text{out}}(\mathbf{U}^{\text{pmt}})^\top$.
    - Design Motivation: Strictly adheres to ICL principles—prompts do not interact with each other, and the query acquires task information exclusively through the prompts.

### Loss & Training
An episodic cross-entropy loss is used: $\mathcal{L}_{\text{episode}} = -\frac{1}{mT}\sum_c\sum_t \log \frac{\exp(s[c]/\tau)}{\sum_j \exp(s[j]/\tau)}$, with episodes sampled across all pre-training graphs for aggregated training. The domain embedder is pre-trained separately with a distance-preserving loss $\mathcal{L}_{\text{de}} = \sum_{i,j}(\|\Delta\theta_i - \Delta\theta_j\|_F - \|e_i - e_j\|_2)^2$ and subsequently frozen.

## Key Experimental Results

### Main Results (Few-shot Node Classification, 5-shot)

| Method | Cora-7way | Products-47way | Computers-10way | Physics-5way | BlogCatalog-6way |
|--------|-----------|----------------|-----------------|--------------|-----------------|
| GCN | 42.55 | 8.77 | 41.09 | 77.15 | 52.16 |
| GraphSAGE | 42.40 | 9.42 | 40.58 | 77.36 | 58.03 |
| Prodigy | ~55 | ~12 | ~50 | ~80 | ~55 |
| **MF-GIA** | **Best** | **Best** | **Best** | **Best** | **Best** |

### Ablation Study

| Configuration | Avg. Performance | Note |
|---------------|-----------------|------|
| Full MF-GIA | Best | All modules combined |
| w/o Domain Embedder | Degraded | Loss of cross-domain adaptability |
| w/o Feature Alignment | Significantly degraded | Features misaligned across domains |
| w/o Label Alignment | Degraded | Label semantic inconsistency |
| w/o DPAA (standard classification head) | Degraded | Loss of prompt-based reasoning |
| w/o Graph-aware Prototypes | Slightly degraded | Neighborhood information is beneficial |

### Key Findings
- MF-GIA is the first method to satisfy all three ICL conditions simultaneously, achieving state-of-the-art across all benchmarks.
- Gradient fingerprints effectively capture domain characteristics: embeddings of related domains (e.g., two citation networks) naturally cluster together.
- The framework transfers in a zero-shot manner to entirely unseen domains, with label alignment being the critical factor.
- Seamless transfer from node classification to link classification tasks validates the generality of the framework.

## Highlights & Insights
- The use of gradient fingerprints as domain descriptors is elegant—no external priors are required; domain information is extracted purely from the interaction between data and model.
- FiLM-conditioned transformations are simple yet effective, achieving domain adaptation through scaling and shifting alone.
- DPAA strictly adheres to the ICL paradigm, serving as an exemplary design reference for prompt learning in the graph domain.
- Modality-agnosticism makes the method applicable in privacy-sensitive scenarios where only pre-encoded data are available.

## Limitations & Future Work
- One-step gradient fingerprints may be sensitive to the initialization $\theta_0$.
- SVD-based preprocessing for feature dimension unification may result in information loss.
- The diversity of pre-training domains directly affects generalization capability.
- The computational efficiency of gradient computation on large-scale graphs warrants further investigation.

## Related Work & Insights
- **vs. UniGraph/OFA**: Does not require converting raw data to text; fully modality-agnostic.
- **vs. Prodigy**: Adds cross-domain alignment capability, yielding stronger generalization.
- **vs. GPF**: Incorporates cross-domain alignment, handling heterogeneous domains more effectively.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of gradient fingerprints and modality-agnostic ICL is pioneering, with solid theoretical guarantees.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive multi-domain evaluation, though testing on very large-scale graphs is lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ Theory and practice are tightly integrated, with a consistent notation system.
- Value: ⭐⭐⭐⭐ Advances graph foundation models toward genuinely universal in-context learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Stress-Testing Alignment Audits with Prompt-Level Strategic Deception](stress-testing_alignment_audits_with_prompt-level_strategic_deception.md)
- [\[ICLR 2026\] DiffVax: Optimization-Free Image Immunization Against Diffusion-Based Editing](diffvax_optimization-free_image_immunization_against_diffusion-based_editing.md)
- [\[ICLR 2026\] The Unseen Frontier: Pushing the Limits of LLM Sparsity with Surrogate-Free ADMM](the_unseen_frontier_pushing_the_limits_of_llm_sparsity_with_surrogate-free_admm.md)
- [\[NeurIPS 2025\] Graver: Generative Graph Vocabularies for Robust Graph Foundation Models Fine-tuning](../../NeurIPS2025/model_compression/graver_generative_graph_vocabularies_for_robust_graph_foundation_models_fine-tun.md)
- [\[NeurIPS 2025\] Graph Your Own Prompt](../../NeurIPS2025/model_compression/graph_your_own_prompt.md)

</div>

<!-- RELATED:END -->
