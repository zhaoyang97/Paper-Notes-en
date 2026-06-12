---
title: >-
  [Paper Note] Mitigating Hallucinations in Large Vision-Language Models without Performance Degradation
description: >-
  [ACL 2026][Hallucination Detection][Vision-Language Models] This paper proposes the MPD framework, which decouples hallucination components via semantic-aware orthogonal subspace projection and selectively updates a smal…
tags:
  - "ACL 2026"
  - "Hallucination Detection"
  - "Vision-Language Models"
  - "Object Hallucination"
  - "Representation Intervention"
  - "Orthogonal Projection"
  - "Selective Parameter Editing"
date: 2026-05-08
content_hash: cb65c9f6129f3c66
---

# Mitigating Hallucinations in Large Vision-Language Models without Performance Degradation

**Conference**: ACL 2026  
**arXiv**: [2604.20366](https://arxiv.org/abs/2604.20366)  
**Code**: None  
**Area**: Hallucination Detection  
**Keywords**: Vision-Language Models, Object Hallucination, Representation Intervention, Orthogonal Projection, Selective Parameter Editing

## TL;DR
This paper proposes the MPD framework, which decouples hallucination components via semantic-aware orthogonal subspace projection and selectively updates a small number of parameters most relevant to hallucinations. It reduces hallucinations by 23.4% while maintaining 97.4% of general generation capability without introducing additional inference overhead.

## Background & Motivation

**Background**: Large Vision-Language Models (LVLMs) demonstrate excellent performance in cross-modal understanding and generation, but universally suffer from object hallucinations—generating text that fabricates non-existent objects, misattributes visual properties, or invents spatial relationships. Mainstream mitigation methods follow two paths: supervised fine-tuning (high cost) and representation intervention (efficient but with side effects).

**Limitations of Prior Work**: Representation intervention methods (e.g., Nullu), while requiring no annotated data, cause LVLMs to lose general generation capability, manifested as semantic incoherence and increased vocabulary repetition. There are two root causes: (1) Hallucination components are highly coupled with general semantics during extraction, leading simple differences to erroneously remove normal semantics; (2) Parameter updates impose large scale perturbations on all weights in target layers, and modifying hundreds of millions of parameters leads to overfitting and destruction of the original parameter distribution.

**Key Challenge**: Hallucination components and general semantics are highly entangled in the hidden representation space. Global intervention inevitably damages both—how can one precisely separate the hallucination signal and suppress it with minimal perturbation?

**Goal**: Design a dual-stage framework that effectively mitigates hallucinations while maintaining general generation capability without introducing additional inference costs.

**Key Insight**: Leveraging orthogonal projection theory from linear algebra, faithful and hallucinated representations are treated as components in different subspaces, achieving precise decoupling through SVD decomposition.

**Core Idea**: Precise suppression of hallucinations without damaging generation capabilities via orthogonal projection to extract pure hallucination components combined with selective parameter editing based on cosine similarity.

## Method

### Overall Architecture
MPD consists of two stages: (1) Hallucination component extraction—building faithful/hallucinated representations using contrastive query pairs and separating pure hallucination components via SVD orthogonal projection; (2) Selective parameter updating—identifying weight vectors most relevant to hallucination components through cosine similarity and applying spatial projection editing only to those weights. The input is the original LVLM plus a small number of contrastive data pairs; the output is the edited LVLM with no additional inference overhead.

### Key Designs

1. **Semantic-aware Hallucination Component Decoupling (Orthogonal Projection)**:

    - **Function**: Precisely extract "pure" hallucination components that do not contain general semantics from hallucinated representations.
    - **Mechanism**: For each layer $\ell$, collect hidden state matrices $\mathbf{X}_\ell^+$ for faithful descriptions and $\mathbf{X}_\ell^-$ for hallucinated descriptions. Perform SVD on $\mathbf{X}_\ell^+$ to obtain the projection matrix $\mathbf{P}_\ell = \mathbf{U}_\ell \mathbf{U}_\ell^\top$ for the faithful subspace, then project the hallucinated representation onto the orthogonal complement of the faithful subspace: $\tilde{\mathbf{X}}_\ell = (\mathbf{I} - \mathbf{P}_\ell) \mathbf{X}_\ell^-$. The paper proves this method is more accurate in estimating pure hallucination components than naive subtraction ($\mathbf{X}^- - \mathbf{X}^+$).
    - **Design Motivation**: Naive subtraction introduces hallucination-parallel components in the faithful subspace and double the noise. Orthogonal projection automatically eliminates components shared with faithful semantics, ensuring extracted hallucination directions do not "collaterally damage" normal generation capabilities.

2. **Selective Parameter Identification and Editing**:

    - **Function**: Modify only the minority of weights most relevant to hallucinations to minimize perturbation to the original parameter distribution.
    - **Mechanism**: For each row $\mathbf{w}_\ell^{(i)}$ in the weight matrix $\mathbf{W}_\ell$, calculate its average cosine similarity $s_i$ with the hallucination components $\tilde{\mathbf{x}}_{\ell,j}$. Select the top-K weight vectors with the highest similarity. Then construct the orthogonal complement projection matrix for the hallucination subspace $\tilde{\mathbf{Q}}_\ell = \mathbf{I} - \tilde{\mathbf{X}}_\ell^\top (\tilde{\mathbf{X}}_\ell \tilde{\mathbf{X}}_\ell^\top)^{-1} \tilde{\mathbf{X}}_\ell$, and perform $\mathbf{w}_\ell^{(i)} \leftarrow \tilde{\mathbf{Q}}_\ell \mathbf{w}_\ell^{(i)}$ only on the selected weights.
    - **Design Motivation**: Methods like Nullu modify all parameters in target layers, leading to excessive perturbation (hundreds of millions of parameters). MPD reduces parameter modifications by 42% on mPLUG-Owl2 and 37% on MiniGPT4.

3. **Contrastive Query Pair Construction**:

    - **Function**: Provide paired hallucination/faithful representations for component extraction.
    - **Mechanism**: Use an auxiliary LLM to construct semantically equivalent query pairs for the same image—one designed to induce hallucinations and one faithful to the image. The LURE dataset is used as the pair data source.
    - **Design Motivation**: Representations of the same image under both hallucination and faithful conditions are required for difference analysis.

### Loss & Training
MPD is a training-free method—it involves no gradient optimization and directly edits model weights through SVD decomposition and projection operations. Once editing is complete, the inference process is identical to the original model with zero additional computational overhead.

## Key Experimental Results

### Main Results (CHAIR Benchmark)

| Model | Method | CHAIR_S ↓ | CHAIR_I ↓ | BLEU ↑ |
|------|------|-----------|-----------|--------|
| LLaVA-1.5-7B | Greedy | 20.40 | 7.08 | 15.72 |
| LLaVA-1.5-7B | Nullu | 15.20 | 5.30 | 15.69 |
| LLaVA-1.5-7B | **MPD** | **12.80** | **4.20** | 15.31 |
| mPLUG-Owl2 | Greedy | 22.90 | 8.62 | 15.01 |
| mPLUG-Owl2 | Nullu | 15.60 | 5.77 | 15.45 |
| mPLUG-Owl2 | **MPD** | **14.00** | **4.99** | **16.06** |
| MiniGPT-4 | Greedy | 32.40 | 12.20 | 14.57 |
| MiniGPT-4 | Nullu | 21.40 | 8.99 | 14.81 |
| MiniGPT-4 | **MPD** | **19.40** | **7.50** | **14.98** |

### Ablation Study (LLaVA-Bench Generation Capability)

| Model | Method | Accuracy ↑ | Detailedness ↑ |
|------|------|-----------|---------------|
| MiniGPT-4 | Original | 4.05 | 3.95 |
| MiniGPT-4 | MPD | 5.53 | 4.67 |
| mPLUG-Owl2 | Original | 5.76 | 4.22 |
| mPLUG-Owl2 | MPD | 6.13 | 4.62 |
| LLaVA-1.5-7B | Original | 5.59 | 4.72 |
| LLaVA-1.5-7B | MPD | 6.39 | — |

### Key Findings
- MPD simultaneously achieves the lowest hallucination rates and the highest or competitive generation quality (BLEU) across all models and benchmarks, breaking the previous trade-off between hallucination mitigation and generation capability.
- Under the three settings (random/popular/adversarial) of the POPE benchmark, MPD achieves the highest F1 score across all models.
- On LLaVA-Bench, MPD does not reduce generation capability but rather improves accuracy and detailedness, suggesting that removing hallucination noise itself improves generation quality.
- Consistent improvements are observed on HallusionBench, indicating the method generalizes to fine-grained hallucination scenarios beyond object hallucinations.

## Highlights & Insights
- **Theoretical Elegance of Orthogonal Projection**—Proposition 1 rigorously proves that the projection method has smaller expected error in estimating hallucination components compared to naive subtraction, providing a mathematical foundation rather than relying solely on heuristics.
- The concept of selective parameter editing is highly practical—reducing parameter modifications by 37-42% while achieving better results demonstrates that "less is more"—precise strikes are more effective than carpet bombing.
- The edited model has zero inference overhead (parameters are permanently modified), making it more suitable for practical deployment than methods like VCD or OPERA that require modification of the inference process.

## Limitations & Future Work
- Evaluation is limited to three smaller LVLMs (MiniGPT-4, mPLUG-Owl2, LLaVA-1.5-7B) and has not been tested on larger or newer models (e.g., LLaVA-Next, Qwen-VL).
- Requires preparation of contrastive data pairs, which increases pipeline complexity despite the small scale.
- Orthogonal projection assumes that hallucinations and faithful semantics can be linearly separated, which may fail in cases of high non-linear entanglement.
- The number of principal components $C$ maintained in SVD and the top-K parameter selection require hyperparameter tuning.

## Related Work & Insights
- **vs Nullu (Yang et al., 2025)**: Also uses null space projection but operates on all weights. MPD adds orthogonal decoupling and selective editing, outperforming Nullu in both hallucination metrics and generation quality.
- **vs VCD (Leng et al., 2024)**: VCD introduces contrastive distribution constraints during decoding, increasing inference latency; MPD has zero inference overhead after editing.
- **vs HALC (Chen et al., 2024)**: HALC relies on external visual grounding modules for posterior correction, introducing additional model dependencies; MPD is self-contained.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of orthogonal projection and selective editing is theoretically supported, though the core idea is an improvement on Nullu rather than a completely new paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ 5 benchmarks, 3 models, and multiple baselines, though model scales are relatively small.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical derivations, though notation is dense.
- Value: ⭐⭐⭐⭐ Highly practical—hallucination mitigation with zero inference overhead has direct value for deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Benchmarking Deflection and Hallucination in Large Vision-Language Models](benchmarking_deflection_and_hallucination_in_large_vision-language_models.md)
- [\[CVPR 2026\] Residual Decoding: Mitigating Hallucinations in Large Vision-Language Models via History-Aware Residual Guidance](../../CVPR2026/hallucination/residual_decoding_mitigating_hallucinations_in_large_vision-language_models_via_.md)
- [\[ICML 2026\] Mitigating Hallucinations in Large Vision-Language Models via Causal Route Gating](../../ICML2026/hallucination/mitigating_hallucinations_in_large_vision-language_models_via_causal_route_gatin.md)
- [\[ICML 2026\] Revis: Sparse Latent Steering to Mitigate Object Hallucination in Large Vision-Language Models](../../ICML2026/hallucination/revis_sparse_latent_steering_to_mitigate_object_hallucination_in_large_vision-la.md)
- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](../../CVPR2026/hallucination/hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)

</div>

<!-- RELATED:END -->
