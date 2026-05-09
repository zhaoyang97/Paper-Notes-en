---
title: >-
  [Paper Note] Continual Unlearning for Text-to-Image Diffusion Models: A Regularization Perspective
description: >-
  [ICLR 2026][Image Generation][continual unlearning] This paper presents the first systematic study of continual unlearning for text-to-image (T2I) diffusion models. It identifies that existing unlearning methods suffer from "utility collapse" under sequential unlearning requests due to cumulative parameter drift, and proposes a suite of plug-in regularization strategies (L1/L2 norm, selective fine-tuning, model merging) along with a semantics-aware gradient projection method to mitigate this issue.
tags:
  - ICLR 2026
  - Image Generation
  - continual unlearning
  - diffusion models
  - regularization
  - gradient projection
  - concept erasure
date: 2026-05-08
content_hash: 5b9cc10a50b760bf
---

# Continual Unlearning for Text-to-Image Diffusion Models: A Regularization Perspective

**Conference**: ICLR 2026
**arXiv**: [2511.07970](https://arxiv.org/abs/2511.07970)
**Code**: [https://justinhylee135.github.io/CUIG_Project_Page/](https://justinhylee135.github.io/CUIG_Project_Page/)
**Area**: Diffusion Models / Machine Unlearning
**Keywords**: continual unlearning, diffusion models, regularization, gradient projection, concept erasure

## TL;DR
This paper presents the first systematic study of continual unlearning for text-to-image (T2I) diffusion models. It identifies that existing unlearning methods suffer from "utility collapse" under sequential unlearning requests due to cumulative parameter drift, and proposes a suite of plug-in regularization strategies (L1/L2 norm, selective fine-tuning, model merging) along with a semantics-aware gradient projection method to mitigate this issue.

## Background & Motivation

**Background**: Machine unlearning aims to remove specific concepts (e.g., copyrighted content, harmful styles) from pretrained models without retraining from scratch. Existing methods (e.g., ConAbl, SculpMem) perform well when erasing multiple concepts simultaneously.

**Limitations of Prior Work**: In practice, unlearning requests arrive **sequentially** (e.g., removing violent content today and an artist's style tomorrow), rather than all at once. Existing methods exhibit "utility collapse" after only a few sequential requests — the model not only forgets the target concept but loses the ability to generate unrelated concepts as well.

**Key Challenge**: Each unlearning operation pushes model parameters away from the pretrained weights. Sequential operations lead to cumulative parameter drift far exceeding that of simultaneous unlearning. Since the pretrained weights encode the model's generative capabilities, excessive drift implies capability degradation.

**Goal**: (a) Define and benchmark the continual unlearning problem; (b) diagnose the root cause of utility collapse; (c) propose plug-in regularization strategies compatible with existing unlearning methods; (d) address the challenge of concept retention within the same semantic domain.

**Key Insight**: Drawing on regularization and gradient projection techniques from continual learning to constrain parameter updates. A key insight is the need for **semantic awareness** — concepts semantically close to the unlearning target are more susceptible to collateral forgetting.

**Core Idea**: Utility collapse in continual unlearning is fundamentally caused by cumulative parameter drift. Regularization-based drift constraints combined with gradient projection to protect semantically related concepts can effectively alleviate this problem.

## Method

### Overall Architecture

At each unlearning request:
- **Input**: The model $\theta_{n-1}^*$ after the previous unlearning step and a new unlearning target $c_n^*$
- **Process**: Update the model using the unlearning loss $\mathcal{L}_{\text{unlearn}}$ with additional regularization constraints
- **Output**: A new model $\theta_n^*$ that simultaneously: (1) effectively erases $c_n^*$; (2) maintains the unlearning of $c_1^*, \ldots, c_{n-1}^*$; (3) preserves the generative capability for all unrelated concepts

### Key Designs

1. **Update Norm Regularization (L1/L2)**:

    - **Function**: Directly penalizes the magnitude of parameter updates.
    - **Mechanism**: $\mathcal{L}_{\text{unlearn}}(\theta, \{c_n^*\}) + \lambda \|\theta - \theta_{n-1}^*\|_p^p$; L1 encourages sparse updates while L2 prevents excessive drift in individual weights.
    - **Design Motivation**: The most straightforward approach to constraining cumulative drift; simple yet effective.

2. **Selective Fine-Tuning (SelFT)**:

    - **Function**: Updates only the top-k% parameters most important to the target concept, freezing the rest.
    - **Mechanism**: Parameter importance is estimated via a first-order Taylor approximation $|\nabla_{\theta[d]} \mathcal{L}_{\text{unlearn}} \cdot \theta_{n-1}^*[d]|$; only the most important parameters are updated.
    - **Design Motivation**: Compared to the isotropic sparsity of L1 regularization, SelFT leverages task-relevant sparsity for more targeted updates.

3. **Model Merging (Model Merge)**:

    - **Function**: Independently unlearns each concept starting from the pretrained weights, then merges all individually unlearned models using TIES-Merging.
    - **Mechanism**: Each independently unlearned model remains close to the pretrained weights; after merging, the combined model still resides within the same loss basin, preserving utility.
    - **Design Motivation**: Independent unlearning avoids cumulative drift, and merging aggregates all unlearning effects while maintaining proximity to the pretrained weights.

4. **Gradient Projection (GradProj) — Semantics-Aware Regularization**:

    - **Function**: Projects the unlearning gradient onto a subspace orthogonal to semantically related concepts, preventing interference with those concepts.
    - **Mechanism**: Unlearning primarily modifies the cross-attention matrices $W_K$ and $W_V$. Since linear projections preserve neighborhood structure, modifying $W_K$ and $W_V$ to erase $c^*$ inevitably perturbs semantically related concepts $c$. GradProj selects the top-$K$ semantically similar concepts (ranked by cosine similarity of text embeddings) and removes the gradient components of $W_K$ and $W_V$ along the embedding directions of these concepts.
    - **Design Motivation**: Cross-domain retention (e.g., preserving objects when unlearning a style) can be addressed by general regularization, but within-domain retention (e.g., preserving other styles when unlearning one style) is highly challenging. Experiments show a strong negative correlation between retention accuracy and text embedding similarity, necessitating semantics-aware constraints.

### Loss & Training

- Unlearning loss based on ConAbl or SculpMem.
- Regularization is added on top of the unlearning loss and is orthogonally compatible with specific unlearning methods.
- GradProj selects top-$K=5$ semantically similar concepts.

## Key Experimental Results

### Main Results (ConAbl + 12-Step Sequential Unlearning)

| Method | UA ↑ | RA-I ↑ | RA-C ↑ | Notes |
|--------|------|--------|--------|-------|
| Sequential (no regularization) | ~95% | ~20% | ~30% | Utility collapse |
| Simultaneous (non-sequential) | ~90% | ~70% | ~85% | Good but costly |
| + L2 Regularization | ~92% | ~40% | ~75% | Large cross-domain improvement |
| + SelFT | ~93% | ~35% | ~70% | Cross-domain improvement |
| + Model Merge | ~90% | ~50% | **~85%** | Best overall |
| + GradProj | ~90% | **~60%** | ~70% | Best within-domain retention |
| + Merge + GradProj | ~88% | **~65%** | **~85%** | Best complementary performance |

### Ablation Study

| Analysis | Key Finding |
|----------|-------------|
| Parameter drift vs. retention | Sequential drift far exceeds simultaneous unlearning drift; strongly correlated with retention accuracy |
| Semantic similarity vs. RA-I | Strong negative correlation ($r \approx -0.8$); more similar concepts are harder to retain |
| $W_K, W_V$ change vs. similarity | Strong positive correlation; key/value representations of semantically similar concepts are heavily perturbed |
| GradProj $K$ value | $K=5$ suffices to cover the most critical semantic neighbors |

### Key Findings
- Retention accuracy collapses below 50% after just 3–4 sequential unlearning steps; after 12 steps, the model can barely generate any meaningful images.
- Parameter drift from simultaneous and independent unlearning is of a comparable and much smaller magnitude than that from sequential unlearning.
- Model Merge achieves the strongest overall retention because each model independently remains close to the pretrained weights.
- GradProj yields the most significant improvement in within-domain retention (RA-I) by precisely protecting semantically similar concepts.
- The regularization strategies are complementary and can be combined.

## Highlights & Insights
- **Clear and valuable problem formulation**: This is the first work to benchmark continual unlearning for T2I diffusion models. The motivation is well-grounded (unlearning requests arrive sequentially in practice), and the benchmark design is principled (standardized evaluation based on UnlearnCanvas).
- **In-depth root cause analysis**: Beyond identifying utility collapse, the paper provides a theoretical explanation via parameter drift analysis and Taylor expansion — the change in retention loss is bounded by $\|\theta^* - \theta^\dagger\|$.
- **The semantics-aware gradient projection** is a transferable idea applicable to any setting where modifying one model capability should not affect closely related capabilities, such as multi-task learning and model editing.
- **Plug-in regularization does not modify the underlying unlearning method**, making it general-purpose and readily combinable with any unlearning algorithm.

## Limitations & Future Work
- While Model Merge is effective, it requires independent unlearning for each concept, making its computational cost comparable to simultaneous unlearning.
- GradProj requires knowledge of which concepts are semantically similar to the target; automatic discovery of such concepts in practice is not sufficiently discussed.
- Evaluation is limited to fine-tuned SD on UnlearnCanvas; generalization to larger models such as SDXL and real-world unlearning scenarios has not been tested.
- Regularization cannot fully resolve within-domain retention (RA-I remains substantially below RA-C), indicating the problem is not yet fully solved.
- The theoretical limits of the trade-off between unlearning efficacy (UA) and retention (RA) remain an open question.

## Related Work & Insights
- **vs. ConAbl**: Direct improvement — combining ConAbl with Model Merge and GradProj substantially improves retention in the continual setting.
- **vs. SculpMem**: Benefits equally from these regularization strategies, demonstrating the generality of the proposed approach.
- **vs. Continual Learning**: The paper draws on ideas from EWC and gradient projection, but highlights a key distinction — in unlearning, the concepts to be retained have already been learned by the model, making interference risks more pronounced.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The problem setting is novel (continual unlearning for T2I); the methods are primarily adaptations of existing techniques, though the semantics-aware gradient projection is a creative contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 12-step sequential evaluation, both style and object settings, multiple baselines, comprehensive ablation and analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The problem–diagnosis–solution narrative is exceptionally clear, with theoretical analysis and experiments mutually reinforcing each other.
- **Value**: ⭐⭐⭐⭐⭐ — Defines an important new research direction with direct social and legal relevance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] The Spacetime of Diffusion Models: An Information Geometry Perspective](the_spacetime_of_diffusion_models_an_information_geometry_perspective.md)
- [\[ICLR 2026\] Localized Concept Erasure in Text-to-Image Diffusion Models via High-Level Representation Misdirection](localized_concept_erasure_in_text-to-image_diffusion_models_via_high-level_repre.md)
- [\[ICCV 2025\] Holistic Unlearning Benchmark: A Multi-Faceted Evaluation for Text-to-Image Diffusion Model Unlearning](../../ICCV2025/image_generation/holistic_unlearning_benchmark_a_multi-faceted_evaluation_for_text-to-image_diffu.md)
- [\[ICCV 2025\] Joint Diffusion Models in Continual Learning](../../ICCV2025/image_generation/joint_diffusion_models_in_continual_learning.md)
- [\[ICLR 2026\] Exposing Hidden Biases in Text-to-Image Models via Automated Prompt Search](exposing_hidden_biases_in_text-to-image_models_via_automated_prompt_search.md)

</div>

<!-- RELATED:END -->
