---
title: >-
  [Paper Note] Are Any-to-Any Models More Consistent Across Modality Transfers Than Specialists?
description: >-
  [ACL 2025][any-to-any models] This paper proposes the ACON dataset and three consistency evaluation criteria (cyclic consistency, forward equivariance, and conjugated equivariance), finding that current any-to-any models are not more cross-modally consistent in point-wise evaluations than combinations of specialist models, though weak consistency can be observed through distributional analyses of multiple editing operations.
tags:
  - "ACL 2025"
  - "any-to-any models"
  - "cross-modal consistency"
  - "cyclic consistency"
  - "equivariance"
  - "multimodal evaluation"
date: 2026-05-08
content_hash: aba080b576ca02a6
---

# Are Any-to-Any Models More Consistent Across Modality Transfers Than Specialists?

**Conference**: ACL 2025  
**arXiv**: [2505.24211](https://arxiv.org/abs/2505.24211)  
**Code**: [github](https://github.com/JiwanChung/ACON)  
**Area**: Other  
**Keywords**: any-to-any models, cross-modal consistency, cyclic consistency, equivariance, multimodal evaluation

## TL;DR

This paper proposes the ACON dataset and three consistency evaluation criteria (cyclic consistency, forward equivariance, and conjugated equivariance), finding that current any-to-any models are not more cross-modally consistent in point-wise evaluations than combinations of specialist models, though weak consistency can be observed through distributional analyses of multiple editing operations.

## Background & Motivation

Any-to-any generative models aim to simultaneously understand and generate multiple modalities (text, image, audio, etc.) within a unified framework, sharing most of their parameters across different modalities compared to modality-specialist approaches. However, the practical value of such models remains uncertain: at the current stage of development, they usually fail to consistently outperform specialist models (e.g., Stable Diffusion, Flux) in output quality, and their training efficiency is also lower due to the computational overhead of optimizing a single large-scale system.

A natural hypothesis is that if an any-to-any model successfully learns a unified latent space, it should produce more coherent cross-modal transfers than two specialist models, which each possess independent latent representations. Does this hypothesis hold true? This paper conducts rigorous experiments to test this conjecture.

The authors argue that existing evaluation methods lack a systematic measure of cross-modal consistency. Retrieval-based metrics (such as CLIPScore) are insufficient for evaluating factual correctness, particularly in compositionality and counting tasks. Therefore, a more fine-grained evaluation framework is required to measure the true advantages of any-to-any models relative to specialist models.

## Method

### Overall Architecture

The contributions of this paper include: (1) formally defining three cross-modal consistency criteria; (2) constructing a meticulously annotated ACON dataset for evaluation; (3) systematically comparing the consistency between any-to-any models and combinations of specialist models.

The evaluation framework involves two types of operations:
- **Cross-modal transfer (f)**: text-to-image ($f^{t \to i}$) or image-to-text ($f^{i \to t}$)
- **Intra-modal modification (g)**: image editing ($g^i$) or text editing ($g^t$), implemented using off-the-shelf tools

### Key Designs

**Three consistency criteria:**

1. **Cyclic Consistency**: Translating an input from one modality to another and then back should recover the original input. For example, text → image → text should recover the original text. This is formalized as $f^{i→t}(f^{t→i}(x^t)) = x^t$.

2. **Forward Equivariance**: Applying a modification operation before or after modal transfer should yield the same result. Formally defined as $f^{t→i}(g^t(x^t, p)) = g^i(f^{t→i}(x^t), p)$. This evaluation only compares transfer in the same direction.

3. **Conjugated Equivariance**: Inserting an intra-modal editing operation in the middle of a cyclic transfer. Formally defined as $f^{i→t}(g^i(f^{t→i}(x^t), p)) = g^t(x^t, p)$. This scales the evaluation from a single point to a structural multi-point consistency analysis of the intermediate latent space.

**ACON Dataset Construction:**

- **Images**: 1,000 images, where 500 are newly captured private photos (never seen during training by any MLLM) and 500 are from COCO Captions.
- **Annotation**: Adopts a communication game framework, setting up three roles:
    - Teller: Writes detailed descriptions for the image (without seeing the reconstructed image).
    - Drawer: Reconstructs the image using AI tools based on the description.
    - Judge: Evaluates the reconstruction quality and provides feedback.
- **QA Pairs**: Each image is paired with 10 binary QA pairs (5 focusing on similarity, 5 on differences), carefully designed by humans to capture subtle factual discrepancies.
- **Editing Operations**: 3 editing prompts per image, with each prompt accompanied by 2 conditional QA pairs.
- **Quality Control**: Approximately 43% of the initial annotations were filtered and replaced, passing through two rounds of independent human verification.

**Evaluation Tools:**

- Image Editing: CosXL (Cos Stable Diffusion XL 1.0 Edit)
- Text Editing: Qwen2.5
- VQA Evaluation: PaliGemma2 (image), Qwen2.5 (text)

### Loss & Training

This paper does not involve model training; it is a purely evaluative work. Off-the-shelf models are used for comparison:
- **Specialist Models**: Flux, SDXL (text-to-image); LLaVA-Next, Qwen2VL (image-to-text)
- **Any-to-any Models**: Chameleon, Emu-3, VILA-U, Seed-X

## Key Experimental Results

### Main Results

**Cyclic Consistency (Image→Text→Image):**

| Model Pair | Accuracy (%) | F1 (%) |
|---------|-------------|--------|
| Qwen2VL + Flux | 62.86 | 73.20 |
| VILA-U (self) | 62.15 | 72.88 |
| Seed-X (self) | 61.78 | 72.53 |
| Chameleon (self) | 53.93 | 64.33 |

**Core Finding**: A single any-to-any model does not consistently outperform arbitrary combinations of specialist models. For instance, the combination of Qwen2VL + Flux outperforms the self-loops of Chameleon and Emu3 across most metrics.

**Text→Image→Text Direction:**

| Model Pair | Accuracy (%) | F1 (%) |
|---------|-------------|--------|
| Flux + Qwen2VL | 66.93 | 73.61 |
| Seed-X (self) | 63.52 | 70.22 |
| VILA-U (self) | 62.36 | 68.59 |
| Chameleon (self) | 55.76 | 60.59 |

### Ablation Study

**Forward Equivariance**: Through correlation analysis, results re-confirm that any-to-any models do not consistently outperform independent specialist model pairs. However, Seed-X and VILA-U exhibit an improving trend in text consistency.

**Conjugated Equivariance**: This is the most interesting finding—all any-to-any models (except Chameleon in image generation) achieve stable self-consistency when paired with themselves. However, they do not always outperform their pairings with other models when paired with themselves.

### Key Findings

1. **Impact of Visual Tokenization Strategy**: Seed-X and VILA-U exhibit significant consistency. Both adopt semantically aligned visual tokenizers (leveraging pre-trained ViT features or fine-tuning the alignment with text representations). In contrast, Chameleon and Emu3 rely solely on tokenizers driven by image reconstruction objectives, exhibiting poorer consistency.

2. **Limitations of Cyclic Consistency**: This evaluation conflates the transfer performance of each modality with cyclic consistency. For example, when serving as a text-to-image operator, VILA-U performs well regardless of which model it is paired with. Thus, multiple complementary evaluation criteria are needed.

3. **Conjugated Equivariance Reveals Distributional Consistency**: Unlike the nearly imperceptible consistency captured by cyclic consistency, conjugated equivariance captures broader latent space alignment patterns by analyzing transformations across the distribution of editing operations. This aligns with the hypothesis of shared latent space learning.

4. **Discrepancies in Generative Style**: There are stylistic differences between natural (ground truth) images and model-generated images. This undermines the value of cyclic consistency as a reliable metric, supporting the use of equivariance for fairer comparisons.

## Highlights & Insights

1. **Formalized Consistency Framework**: Three consistency criteria are mathematically and rigorously defined, providing an actionable theoretical framework to evaluate the potential advantages of any-to-any models. This represents the first systematic introduction of cyclic consistency and equivariance to the evaluation of multimodal models.

2. **Counter-intuitive Core Findings**: The claimed advantage of any-to-any models—cross-modal coherence brought by a unified latent space—has not been fully realized at the current stage. This provides crucial guidance for the research community's model design direction.

3. **Meticulous Dataset Construction**: 500 truly unseen private images, a communication-game-based annotation workflow, and a 43% sample filtering rate demonstrate extremely high standards for evaluation quality.

4. **Design Insights for Visual Tokenizers**: Semantically aligned visual tokenizers (such as those used in Seed-X and VILA-U) perform better in cross-modal consistency than tokenizers focusing solely on image reconstruction objectives. This provides concrete guidance for the architectural design of future any-to-any models.

## Limitations & Future Work

1. The evaluation treats models as single dark boxes, making it impossible to isolate the contributions of data, architecture, and specific factors in the training process to consistency.
2. The ACON dataset focuses on natural photographs, excluding artistic images, 2D drawings, and 3D renderings.
3. Annotations were completed by five NLP researchers with similar cultural backgrounds, carrying a risk of cultural bias.
4. Deterministic sampling was used, without explicitly considering the issue of generation diversity.
5. Only (image, text) modal pairs were evaluated, without extending to other modal combinations such as (speech, text).
6. Only a single cyclic transfer was evaluated; iterative compositions might reveal more consistency patterns (e.g., whether repeated applications lead to a collapse in output diversity).

## Related Work & Insights

This paper connects the concept of cyclic consistency from CycleGAN to multimodal model evaluation. Unlike robustness evaluation works such as MM-R3 and MMCBench, this study focuses on the structural consistency of modality transfers, rather than robustness to semantic perturbations or corrupted inputs.

The concept of equivariance is borrowed from geometric deep learning (Cohen & Welling, 2016). Applying it to multimodal evaluation represents a creative cross-domain transfer. As a generalization of cyclic consistency, conjugated equivariance extends the evaluation from a single point to distributional analysis, an approach that also holds reference value for other multimodal evaluation tasks.

The research results suggest that current any-to-any models behave more like multiple specialist models "crammed" together for training, and they have yet to truly fulfill the promise of a shared latent space. This points to a clear direction for future improvements in the community: semantic alignment of visual tokenizers may be the key.

## Rating

- **Novelty**: ★★★★☆ — The formal definitions of the three consistency criteria are novel.
- **Utility**: ★★★★☆ — The ACON dataset is of direct guiding value for the development of any-to-any models.
- **Experimental Thoroughness**: ★★★★★ — Extremely comprehensive with 8 models, 3 evaluation criteria, and multi-directional combinations.
- **Writing Quality**: ★★★★☆ — Mathematical formalization is clear, and motivation is well-argued.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] CORAL: Learning Consistent Representations across Multi-step Training with Lighter Speculative Drafter](coral_speculative_drafting.md)
- [\[ICLR 2026\] Any-Subgroup Equivariant Networks via Symmetry Breaking](../../ICLR2026/others/any-subgroup_equivariant_networks_via_symmetry_breaking.md)
- [\[ACL 2025\] I0T: Embedding Standardization Method Towards Zero Modality Gap](i0t_embedding_standardization_method_towards_zero_modality_gap.md)
- [\[ICLR 2026\] A Single Architecture for Representing Invariance Under Any Space Group](../../ICLR2026/others/a_single_architecture_for_representing_invariance_under_any_space_group.md)
- [\[ACL 2025\] Consistent Client Simulation for Motivational Interviewing-based Counseling](consistent_client_simulation_for_motivational_interviewing-based_counseling.md)

</div>

<!-- RELATED:END -->
