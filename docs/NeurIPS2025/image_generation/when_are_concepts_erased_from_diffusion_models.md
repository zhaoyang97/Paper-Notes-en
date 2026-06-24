---
title: >-
  [Paper Note] When Are Concepts Erased From Diffusion Models?
description: >-
  [NeurIPS 2025][Image Generation][Concept Erasure] This paper proposes two mechanistic models of concept erasure (guidance-based avoidance vs. destruction-based removal) and designs a suite of five independent probing methods—spanning optimization search, in-context probing, noise trajectory probing, classifier-guided probing, and dynamic concept tracing—to systematically demonstrate that most existing erasure methods merely "circumvent" concepts rather than genuinely eliminat…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Concept Erasure"
  - "Diffusion Models"
  - "Knowledge Residual"
  - "Guidance-Based Avoidance"
  - "Destruction-Based Removal"
date: 2026-05-08
content_hash: c07ebf10113fe5fa
---

# When Are Concepts Erased From Diffusion Models?

**Conference**: NeurIPS 2025
**arXiv**: [2505.17013](https://arxiv.org/abs/2505.17013)  
**Code**: [https://github.com/kevinlu4588/WhenAreConceptsErased](https://github.com/kevinlu4588/WhenAreConceptsErased)  
**Area**: Image Generation
**Keywords**: Concept Erasure, Diffusion Models, Knowledge Residual, Guidance-Based Avoidance, Destruction-Based Removal

## TL;DR

This paper proposes two mechanistic models of concept erasure (guidance-based avoidance vs. destruction-based removal) and designs a suite of five independent probing methods—spanning optimization search, in-context probing, noise trajectory probing, classifier-guided probing, and dynamic concept tracing—to systematically demonstrate that most existing erasure methods merely "circumvent" concepts rather than genuinely eliminating the underlying knowledge.

## Background & Motivation

Concept erasure aims to modify diffusion models to prevent the generation of specific concepts (e.g., particular artistic styles or objects). Despite the proliferation of erasure methods, a fundamental question remains unresolved: **are "erased" concepts truly removed from the model, or has the model simply learned to avoid them?**

Prior adversarial attack research has demonstrated that erased concepts can be recovered by searching for appropriate inputs (e.g., via Textual Inversion or adversarial prompts). However, these findings address only text-input-level attacks, leaving an open question: **can erased knowledge be recovered through other means?**

The central contribution of this paper is the formulation of two mechanistic conceptual frameworks and a comprehensive multi-perspective evaluation toolkit to systematically answer the question of "to what extent a concept has truly been erased."

## Method

### Overall Architecture

The authors propose two conceptual models of erasure mechanisms:

- **Guidance-Based Avoidance**: The model modifies the conditional guidance process to steer generation away from the target concept, while the underlying knowledge may remain intact. The unconditional probability $P(X)$ of the model is not significantly altered.
- **Destruction-Based Removal**: The model reduces the unconditional likelihood $P(X)$ of the target concept, fundamentally suppressing or eliminating the underlying features.

Five independent probing methods are then designed to assess the thoroughness of erasure:

### Key Designs

1. **Optimization-Based Probing**: Employs Textual Inversion and UnlearnDiffAtk to search for inputs capable of triggering erased concepts by optimizing text embeddings or tokens. This probe directly adopts methodologies from prior work.

2. **In-Context Probing**:

    - **Inpainting Probe**: The model is provided with a partially masked image containing the target concept, and its ability to correctly complete the masked region is observed. A model that has truly eliminated the concept's knowledge should be unable to perform accurate inpainting.
    - **Diffusion Completion Probe**: The original unmodified model runs 5 or 10 denoising steps, and the intermediate result is passed to the erased model to complete the remaining denoising. If the erased model can recover the concept from partially generated content, residual knowledge is indicated.

3. **Noise-Based Probing**: Additional Gaussian noise is injected at each denoising step as $\tilde{x}_{t-1} = (\tilde{x}_t - \alpha\epsilon_D) + \eta\epsilon$, with $\eta$ searched over 6 values in the range $[1.0, 1.85]$, enabling the model to explore a broader region of latent space. This is a training-free method that requires no input optimization.

4. **Steered Latent Probing**: A lightweight timestep-aware binary classifier $f_{c^*}(\mathbf{x}_t, t)$ is trained in latent space to detect the presence of the target concept. At inference time, gradient-based guidance steers the diffusion trajectory toward regions of concept residual. Twenty-four guidance strength values $s_{\text{clf}}$ are searched.

5. **Dynamic Concept Tracing**: Under varying erasure strengths, the trajectory of generated images in CLIP embedding space is tracked to observe how concept representations evolve throughout the erasure process.

### Loss & Training

The classifier is trained with BCEWithLogits loss with positive-class reweighting. Each mini-batch is augmented with 7 noisy views, and timesteps are sampled according to a power-law distribution biased toward higher noise levels. Training proceeds for 70 epochs, and the checkpoint with the lowest validation loss is selected.

## Key Experimental Results

### Main Results (Optimization-Based Probing)

| Erasure Method | Post-Erasure CLIP↓ | Textual Inversion Acc.↓ | UnlearnDiffAtk Acc.↓ | Unrelated Concept Acc.↑ |
|---|---|---|---|---|
| GA | 24.3 | 0.6% | 6.5% | 52.2% |
| UCE | 22.4 | 71.2% | 26.8% | 75.0% |
| ESD-x | 21.1 | 65.9% | 21.0% | 71.3% |
| ESD-u | 20.9 | 31.8% | 16.6% | 70.4% |
| TaskVec | 23.1 | 6.2% | 10.3% | 60.4% |
| STEREO | 19.6 | 6.3% | 3.7% | 52.8% |
| RECE | 21.2 | 58.2% | 7.2% | 71.7% |

### Ablation Study (Multi-Probe Comparison)

| Erasure Method | Inpainting Acc.↓ | Diffusion Completion t=5 Acc.↓ | Noise Probe Acc.↓ | Classifier-Guided Acc.↓ | Classifier+Noise Acc.↓ |
|---|---|---|---|---|---|
| GA | 61.7% | 1.1% | 2.7% | 3.7% | 4.1% |
| UCE | 69.1% | 42.7% | 21.9% | 45.6% | 75.6% |
| ESD-x | 69.1% | 37.8% | 30.7% | 47.8% | 73.3% |
| TaskVec | 66.8% | 2.4% | 11.0% | 30.2% | 35.1% |
| STEREO | 63.8% | 3.2% | 1.1% | 5.8% | 20.3% |
| RECE | 68.2% | 36.5% | 13.0% | 33.3% | 36.7% |

### Key Findings

- **UCE and ESD-x** are highly vulnerable across all probes, representing canonical guidance-based avoidance methods in which concept knowledge remains fully intact.
- **GA and STEREO** are the most robust under most probes, more closely approximating destruction-based removal.
- **TaskVec** exhibits an interesting contradiction: it is robust to Textual Inversion yet can still correctly complete erased concepts under inpainting probing.
- **RECE and STEREO** are robust to conventional adversarial attacks but unexpectedly expose residual knowledge under diffusion completion probing.
- Noise-based probing (requiring neither optimization nor training) sometimes outperforms more complex optimization-based methods: on UCE and ESD-x, simply increasing stochasticity suffices to recover concepts.
- The combination of classifier-guided and noise-based probing is the most powerful approach, recovering concepts at approximately 1.5× the accuracy of either method used alone.

## Highlights & Insights

- **Value of the theoretical framework**: The dichotomy of guidance-based avoidance vs. destruction-based removal provides a clear conceptual tool for understanding the essential nature of various erasure methods.
- **Necessity of multi-perspective evaluation**: A single evaluation method (e.g., adversarial attacks alone) yields misleading conclusions—a method robust under one probe may fail entirely under another.
- **Elegance of noise-based probing**: The fact that such a simple method (merely increasing inference noise) can more effectively reveal residual knowledge than carefully designed optimization attacks is itself a profound insight.
- **Trade-off between robustness and generality**: Methods that erase more thoroughly (GA, STEREO) tend to inflict greater damage on the generation quality of unrelated concepts.

## Limitations & Future Work

- Experiments cover 10 object concepts and 3 artistic styles, but do not address verbs, relations, or abstract concepts (e.g., "violence").
- Optimization-based probing carries inherent causal ambiguity: recovered concepts may originate from the optimization process itself rather than from residual model knowledge.
- The classifier in steered latent probing may introduce its own biases.
- Experiments are conducted solely on Stable Diffusion 1.4; behavior on newer-generation models (SDXL, SD3, etc.) may differ.
- The categorization of erasure methods into guidance-based vs. destruction-based remains preliminary and lacks rigorous theoretical justification.

## Related Work & Insights

- UCE is a representative guidance-based method that modifies attention projection matrices via a closed-form solution.
- STEREO combines adversarial prompt search with compositional fine-tuning and represents one of the stronger robust erasure methods currently available.
- The evaluation framework proposed in this paper can be directly extended to any newly proposed erasure method.
- Implications for AI safety: **there may be a substantial gap between "appearing to be erased" and "being truly erased."**

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Both the conceptual framework and the multi-perspective evaluation system constitute original contributions)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (7 erasure methods × 5 probing strategies, with comprehensive coverage)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear logical structure, rich figures and tables)
- Value: ⭐⭐⭐⭐⭐ (Significant implications for understanding and improving concept erasure methods)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Emergence and Evolution of Interpretable Concepts in Diffusion Models](emergence_and_evolution_of_interpretable_concepts_in_diffusi.md)
- [\[ICCV 2025\] Meta-Unlearning on Diffusion Models: Preventing Relearning Unlearned Concepts](../../ICCV2025/image_generation/meta-unlearning_on_diffusion_models_preventing_relearning_unlearned_concepts.md)
- [\[ICML 2026\] SAEmnesia: Erasing Concepts in Diffusion Models with Supervised Sparse Autoencoders](../../ICML2026/image_generation/saemnesia_erasing_concepts_in_diffusion_models_with_supervised_sparse_autoencode.md)
- [\[CVPR 2025\] Concept Replacer: Replacing Sensitive Concepts in Diffusion Models via Precision Localization](../../CVPR2025/image_generation/concept_replacer_replacing_sensitive_concepts_in_diffusion_models_via_precision_.md)
- [\[CVPR 2025\] Memories of Forgotten Concepts](../../CVPR2025/image_generation/memories_of_forgotten_concepts.md)

</div>

<!-- RELATED:END -->
