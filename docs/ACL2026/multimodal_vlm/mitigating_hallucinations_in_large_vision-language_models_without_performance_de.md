---
title: >-
  [Paper Note] Mitigating Hallucinations in Large Vision-Language Models without Performance Degradation
description: >-
  [ACL 2026][Multimodal VLM][Vision-language models] This paper proposes the MPD framework, which decouples hallucination components via semantics-aware orthogonal subspace projection and selectively updates only the param…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Vision-language models"
  - "object hallucination"
  - "representation intervention"
  - "orthogonal projection"
  - "selective parameter editing"
date: 2026-05-08
content_hash: eb1e6c4765abcbb4
---

# Mitigating Hallucinations in Large Vision-Language Models without Performance Degradation

**Conference**: ACL 2026
**arXiv**: [2604.20366](https://arxiv.org/abs/2604.20366)
**Code**: None
**Area**: Multimodal VLM / Hallucination Mitigation
**Keywords**: Vision-language models, object hallucination, representation intervention, orthogonal projection, selective parameter editing

## TL;DR
This paper proposes the MPD framework, which decouples hallucination components via semantics-aware orthogonal subspace projection and selectively updates only the parameters most relevant to hallucinations. MPD reduces hallucinations by 23.4% while preserving 97.4% of general generation capability, without introducing any additional inference overhead.

## Background & Motivation

**Background**: Large vision-language models (LVLMs) demonstrate strong cross-modal understanding and generation capabilities, yet suffer from pervasive object hallucination—generated text descriptions fabricate non-existent objects, misattribute visual properties, or invent spatial relationships. Mainstream mitigation approaches fall into two categories: annotated data fine-tuning (costly) and representation intervention (efficient but with side effects).

**Limitations of Prior Work**: Representation intervention methods such as Nullu eliminate the need for annotated data, yet the resulting LVLMs lose general generation capability—manifested as semantic incoherence and elevated lexical repetition rates. Two root causes are identified: (1) extracted hallucination components are heavily entangled with general semantics, so naive differencing inadvertently removes normal semantic content; (2) large-scale perturbations are applied to all weights in the target layer, modifying hundreds of millions of parameters and causing overfitting and disruption of the original parameter distribution.

**Key Challenge**: Hallucination components and general semantics are highly entangled in the hidden representation space. Coarse global intervention inevitably damages both simultaneously—the central challenge is how to precisely isolate the hallucination signal and suppress it with minimal perturbation.

**Goal**: To design a two-stage framework that effectively mitigates hallucinations while preserving the model's general generation capability, without incurring additional inference cost.

**Key Insight**: Grounding the approach in linear-algebraic orthogonal projection theory, faithful representations and hallucination representations are treated as components residing in distinct subspaces, with SVD decomposition enabling precise disentanglement.

**Core Idea**: Orthogonal projection for extracting pure hallucination components + cosine-similarity-guided selective parameter editing = precise hallucination suppression without degrading generation capability.

## Method

### Overall Architecture
MPD consists of two stages: (1) **Hallucination component extraction**—contrastive query pairs are constructed to obtain faithful/hallucination representations, and pure hallucination components are isolated via SVD-based orthogonal projection; (2) **Selective parameter update**—cosine similarity is used to identify weight vectors most correlated with the hallucination components, and spatial projection editing is applied exclusively to those weights. The inputs are the original LVLM and a small set of contrastive data pairs; the output is an edited LVLM with no additional inference overhead.

### Key Designs

1. **Semantics-Aware Hallucination Component Decoupling (Orthogonal Projection)**

    - **Function**: Precisely extract "pure" hallucination components from hallucination representations, free of general semantics.
    - **Mechanism**: For each layer $\ell$, the hidden-state matrices $\mathbf{X}_\ell^+$ (faithful descriptions) and $\mathbf{X}_\ell^-$ (hallucination descriptions) are collected. SVD is applied to $\mathbf{X}_\ell^+$ to obtain the projection matrix $\mathbf{P}_\ell = \mathbf{U}_\ell \mathbf{U}_\ell^\top$ spanning the faithful subspace. The hallucination representations are then projected onto the orthogonal complement of this subspace: $\tilde{\mathbf{X}}_\ell = (\mathbf{I} - \mathbf{P}_\ell) \mathbf{X}_\ell^-$. The paper proves that this approach yields a more accurate estimate of the pure hallucination component than naive differencing ($\mathbf{X}^- - \mathbf{X}^+$).
    - **Design Motivation**: Naive differencing introduces hallucination-parallel components within the faithful subspace as well as doubled noise, whereas orthogonal projection automatically eliminates components shared with faithful semantics, ensuring that the extracted hallucination direction does not inadvertently impair normal generation capability.

2. **Selective Parameter Identification and Editing**

    - **Function**: Modify only the small subset of weights most correlated with hallucinations, minimizing perturbation to the original parameter distribution.
    - **Mechanism**: For each row $\mathbf{w}_\ell^{(i)}$ of weight matrix $\mathbf{W}_\ell$, the average cosine similarity $s_i$ between that row and the hallucination components $\tilde{\mathbf{x}}_{\ell,j}$ is computed. The top-$K$ weight vectors with the highest similarity are selected. The orthogonal complement projection matrix of the hallucination subspace, $\tilde{\mathbf{Q}}_\ell = \mathbf{I} - \tilde{\mathbf{X}}_\ell^\top (\tilde{\mathbf{X}}_\ell \tilde{\mathbf{X}}_\ell^\top)^{-1} \tilde{\mathbf{X}}_\ell$, is constructed, and the update $\mathbf{w}_\ell^{(i)} \leftarrow \tilde{\mathbf{Q}}_\ell \mathbf{w}_\ell^{(i)}$ is applied only to the selected weights.
    - **Design Motivation**: Methods such as Nullu modify all parameters in the target layer, resulting in excessive perturbation (hundreds of millions of parameters). MPD reduces the number of modified parameters by 42% on mPLUG-Owl2 and 37% on MiniGPT-4.

3. **Contrastive Query Pair Construction**

    - **Function**: Provide paired hallucination/faithful representations for component extraction.
    - **Mechanism**: An auxiliary LLM is used to construct, for the same image, a pair of semantically equivalent queries—one that induces hallucination and one that remains faithful to the image. The LURE dataset serves as the paired data source.
    - **Design Motivation**: Representations of the same image under both hallucination-inducing and faithful conditions are required to perform differential analysis.

### Loss & Training
MPD is a training-free method—no gradient optimization is involved. Model weights are edited directly via SVD decomposition and projection operations. After editing, inference proceeds identically to the original model, incurring no additional computational overhead.

## Key Experimental Results

### Main Results (CHAIR Benchmark)

| Model | Method | CHAIR_S ↓ | CHAIR_I ↓ | BLEU ↑ |
|-------|--------|-----------|-----------|--------|
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
|-------|--------|-----------|---------------|
| MiniGPT-4 | Original | 4.05 | 3.95 |
| MiniGPT-4 | MPD | 5.53 | 4.67 |
| mPLUG-Owl2 | Original | 5.76 | 4.22 |
| mPLUG-Owl2 | MPD | 6.13 | 4.62 |
| LLaVA-1.5-7B | Original | 5.59 | 4.72 |
| LLaVA-1.5-7B | MPD | 6.39 | — |

### Key Findings
- MPD simultaneously achieves the lowest hallucination rates and the highest or competitive generation quality (BLEU) across all models and benchmarks, breaking the previously observed trade-off between hallucination mitigation and generation capability.
- On the POPE benchmark across all three settings (random/popular/adversarial), MPD achieves the highest F1 on all models.
- On LLaVA-Bench, MPD not only preserves generation capability but also improves both accuracy and detailedness, indicating that removing hallucination noise inherently benefits generation quality.
- Consistent improvements on HallusionBench demonstrate that the method generalizes to fine-grained hallucination scenarios beyond object hallucination.

## Highlights & Insights
- **Theoretical elegance of orthogonal projection**—Proposition 1 rigorously proves that the projection-based method yields smaller expected estimation error for hallucination components than naive differencing, providing a mathematical foundation rather than relying solely on empirical evidence.
- The selective parameter editing strategy has strong practical value: reducing parameter modification by 37–42% while achieving superior results demonstrates that precision strikes outperform broad-spectrum intervention.
- The edited model incurs zero additional inference overhead (weights are permanently modified), making MPD more deployment-friendly than methods that require modifying the inference pipeline, such as VCD and OPERA.

## Limitations & Future Work
- Validation is limited to three relatively small LVLMs (MiniGPT-4, mPLUG-Owl2, LLaVA-1.5-7B); experiments on larger and more recent models (e.g., LLaVA-Next, Qwen-VL) are absent.
- Preparation of contrastive data pairs is required, which, though modest in scale, adds pipeline complexity.
- The orthogonal projection approach assumes that hallucination and faithful semantics are linearly separable, which may fail in highly nonlinear entanglement scenarios.
- The number of principal components $C$ retained in SVD and the top-$K$ parameter selection both require hyperparameter tuning.

## Related Work & Insights
- **vs. Nullu (Yang et al., 2025)**: Both methods employ null-space projection, but Nullu operates on all weights. MPD introduces two key improvements—orthogonal decoupling and selective editing—and outperforms Nullu on both hallucination metrics and generation quality.
- **vs. VCD (Leng et al., 2024)**: VCD imposes contrastive distribution constraints at decoding time, incurring additional inference latency; MPD introduces zero overhead after editing.
- **vs. HALC (Chen et al., 2024)**: HALC relies on an external visual grounding module for post-hoc correction, introducing additional model dependencies; MPD is self-contained with no external dependencies.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of orthogonal projection and selective editing is theoretically grounded, though the core idea constitutes an incremental improvement over Nullu rather than an entirely new paradigm.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Five benchmarks, three models, and multiple baselines are evaluated, though the model scale is relatively small.
- **Writing Quality**: ⭐⭐⭐⭐ Theoretical derivations are clear, though notation is dense.
- **Value**: ⭐⭐⭐⭐ High practical utility—zero-inference-overhead hallucination mitigation has direct deployment value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Residual Decoding: Mitigating Hallucinations in Large Vision-Language Models via History-Aware Residual Guidance](../../CVPR2026/multimodal_vlm/residual_decoding_mitigating_hallucinations_in_large_vision-language_models_via_.md)
- [\[ACL 2026\] When Vision-Language Models Judge Without Seeing: Exposing Informativeness Bias](when_vision-language_models_judge_without_seeing_exposing_informativeness_bias.md)
- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](../../CVPR2026/multimodal_vlm/hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)
- [\[ACL 2026\] Benchmarking Deflection and Hallucination in Large Vision-Language Models](benchmarking_deflection_and_hallucination_in_large_vision-language_models.md)
- [\[ACL 2026\] Topology-Aware Layer Pruning for Large Vision-Language Models](topology-aware_layer_pruning_for_large_vision-language_models.md)

</div>

<!-- RELATED:END -->
