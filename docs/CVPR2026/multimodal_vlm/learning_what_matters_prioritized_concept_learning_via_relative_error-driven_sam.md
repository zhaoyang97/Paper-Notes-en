---
title: >-
  [Paper Note] Learning What Matters: Prioritized Concept Learning via Relative Error-driven Sample Selection
description: >-
  [CVPR 2026][Multimodal VLM][Data-efficient learning] This paper proposes the PROGRESS framework, which dynamically selects the most informative training samples by tracking a VLM's learning progress across automatically…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Data-efficient learning"
  - "instruction tuning"
  - "curriculum learning"
  - "VLM training"
  - "sample selection"
date: 2026-05-08
content_hash: 2ce5b2680b800caa
---

# Learning What Matters: Prioritized Concept Learning via Relative Error-driven Sample Selection

**Conference**: CVPR 2026
**arXiv**: [2506.01085](https://arxiv.org/abs/2506.01085)  
**Code**: [https://mylittlechange.github.io/PROGRESS_web/](https://mylittlechange.github.io/PROGRESS_web/)  
**Area**: Multimodal VLM
**Keywords**: Data-efficient learning, instruction tuning, curriculum learning, VLM training, sample selection

## TL;DR
This paper proposes the PROGRESS framework, which dynamically selects the most informative training samples by tracking a VLM's learning progress across automatically discovered multimodal concept clusters. Using only 16–20% of annotated data, PROGRESS achieves 99–100% of full-data performance with shorter total training time.

## Background & Motivation
**Background**: Instruction tuning of VLMs relies on large-scale, high-quality annotated data and substantial compute, leading to increasingly high costs.

**Limitations of Prior Work**: (a) Static selection methods (CLIP-Score, EL2N, Perplexity, etc.) select data once and cannot adapt to the model's learning progress; (b) gradient-based methods (ICONS) incur prohibitive computational cost (hundreds of GPU hours), defeating the purpose of efficient training; (c) COINCIDE requires a separately trained auxiliary VLM, full-dataset annotation, and manual inspection of activations.

**Key Challenge**: A large fraction of training samples are redundant or uninformative, yet static methods cannot identify this dynamically during training.

**Goal**: Can a VLM dynamically determine "what to learn next" based on its own learning state, acquiring annotations only when necessary?

**Key Insight**: Inspired by curriculum learning and self-paced learning — a model should focus on skills that are "not yet mastered but rapidly improving," avoiding wasted budget on already-learned or excessively difficult samples.

**Core Idea**: Track the relative rate of change in learning progress $\Delta_k$ and prioritize sampling from concept clusters exhibiting the fastest improvement.

## Method

### Overall Architecture
**Two stages**: (1) Multimodal concept categorization — DINO and BERT features are used to perform spherical k-means clustering on an unannotated data pool, yielding $K$ concept clusters; (2) Prioritized concept learning — periodic self-evaluation computes per-cluster learning progress, samples are drawn preferentially from the fastest-improving clusters, and annotations are acquired only for selected samples.

### Key Designs

1. **Multimodal Concept Categorization**:

    - Function: Automatically partitions the unannotated data pool into semantically coherent concept clusters.
    - Mechanism: For each $(I, Q)$ pair, DINO visual features and BERT text features are extracted, concatenated, normalized, and clustered via spherical k-means. No annotations, auxiliary models, or manual inspection are required.
    - Design Motivation: Multimodal features yield purer clusters than unimodal features and naturally correspond to capabilities such as object localization, OCR, coding, and multilingual understanding.

2. **Prioritized Concept Learning**:

    - Function: Dynamically selects the next training batch based on the model's own learning progress.
    - Mechanism: Every $\gamma$ steps, per-cluster accuracy $\text{Acc}_k^{(t)}$ is evaluated; relative progress is computed as $\Delta_k = \frac{\text{Acc}_k^{(t)} - \text{Acc}_k^{(t-\gamma)}}{\text{Acc}_k^{(t-\gamma)} + \epsilon}$; a temperature-controlled softmax converts $\Delta_k$ into sampling probabilities $p_k = \frac{\exp(\Delta_k/\tau)}{\sum_j \exp(\Delta_j/\tau)}$.
    - Design Motivation: Samples from high-$\Delta_k$ clusters represent the most effective learning targets — skills not yet mastered but actively improving. The temperature $\tau$ balances informativeness and diversity.

3. **Need-based Annotation**:

    - Function: Acquires annotations only for selected samples.
    - Mechanism: The data pool is initially unannotated; an answer $A$ is queried only when an $(I, Q)$ pair is sampled.
    - Design Motivation: This stands in stark contrast to methods such as COINCIDE that require full-dataset annotation, substantially reducing annotation cost.

### Loss & Training
- A warm-up phase uses a simple sampler to select a small number of samples, providing reliable initial skill estimates.
- An exploration mechanism randomly samples $\delta\%$ of data to prevent complete neglect of low-progress clusters.
- Both loss and accuracy can serve as the progress signal.

## Key Experimental Results

### Main Results (LLaVA-v1.5-7B, LLaVA-665K, 20% sampling)

| Method | Auxiliary VLM? | VQAv2 | GQA | TextVQA | POPE | MMBench | Relative Score |
|--------|---------------|-------|-----|---------|------|---------|---------------|
| Full-data fine-tuning | — | 79.1 | 63.0 | 58.2 | 86.4 | 66.1 | 100% |
| Random | ✗ | 75.7 | 58.9 | 55.3 | 84.7 | 62.2 | 95.0% |
| COINCIDE | ✓ | 76.5 | 59.8 | 55.6 | 86.1 | 63.1 | 97.8% |
| **PROGRESS (Acc)** | **✗** | **75.2** | **58.8** | **55.1** | **85.9** | **61.1** | **98.4%** |
| **PROGRESS (Loss)** | **✗** | **75.7** | **58.6** | **55.1** | **86.3** | **62.5** | **98.4%** |

### Generalization Across Architectures and Scales

| Model | Data Ratio | Relative Performance vs. Full Data |
|-------|-----------|-----------------------------------|
| LLaVA-v1.5-7B | 20% | 98–99% |
| LLaVA-v1.5-13B | 20% | Similar |
| Qwen2-VL | 16% | 99–100% |

### Key Findings
- PROGRESS with 20% of data outperforms COINCIDE (98.4% vs. 97.8%), which requires an auxiliary VLM and full-dataset annotation.
- Total training time, including self-evaluation overhead, remains shorter than full-data training.
- Learning progress curves reveal an emergent curriculum effect: the model first masters simple concepts (single-object recognition) before progressing to complex ones (OCR, reasoning).
- The choice of temperature $\tau$ is critical: too low causes mode collapse (the model trains on only one cluster); too high degenerates to random sampling.

## Highlights & Insights
- **No auxiliary model, no full-dataset annotation, no gradient computation** — these three freedoms make PROGRESS particularly accessible to resource-constrained academic labs.
- Progress-driven sampling combines the strengths of curriculum learning and active learning, automatically determining *what* to learn and *when* to learn it.
- The temperature-softmax sampling design elegantly balances informativeness and diversity.
- The visualized concept learning order provides a new perspective for understanding VLM training dynamics.

## Limitations & Future Work
- The number of clusters $K$ must be specified in advance and may require tuning across datasets.
- The self-evaluation frequency $\gamma$ introduces an additional hyperparameter.
- The progress signal is based on training-set performance, which may not perfectly align with validation-set performance.
- The method has only been validated during the instruction tuning stage; its effectiveness during pre-training remains unexplored.

## Related Work & Insights
- **vs. COINCIDE**: COINCIDE requires a pre-trained auxiliary VLM, full-dataset annotation, and manual inspection of activations; PROGRESS is entirely self-contained.
- **vs. ICONS**: ICONS uses gradient information for sample selection, requiring hundreds of GPU hours; PROGRESS uses the rate of change in model accuracy at near-zero additional cost.
- **vs. Curriculum Learning**: Classical curriculum learning relies on externally defined difficulty rankings; PROGRESS is driven by the model's own learning feedback.
- The progress-driven sampling strategy is generalizable to data mixture control in LLM pre-training.

## Rating
- Novelty: ⭐⭐⭐⭐ Progress-driven dynamic sampling constitutes an effective new paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multiple datasets, architectures, scales, comprehensive ablations, and visualization analyses.
- Writing Quality: ⭐⭐⭐⭐ Comparison figures are clear; advantages and disadvantages relative to prior methods are intuitive.
- Value: ⭐⭐⭐⭐⭐ Highly practical for resource-constrained researchers; directly applicable to reducing training costs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SpectralGCD: Spectral Concept Selection and Cross-modal Representation Learning for Generalized Category Discovery](../../ICLR2026/multimodal_vlm/spectralgcd_spectral_concept_selection_and_cross-modal_representation_learning_f.md)
- [\[CVPR 2026\] No Hard Negatives Required: Concept Centric Learning Leads to Compositionality without Degrading Zero-shot Capabilities of Contrastive Models](no_hard_negatives_required_concept_centric_learning_leads_to_compositionality_wi.md)
- [\[CVPR 2026\] Concept-wise Attention for Fine-grained Concept Bottleneck Models](coat_cbm_concept_wise_attention.md)
- [\[CVPR 2026\] LFPC: Learning to Focus and Precise Cropping for MLLMs](lfpc_learning_to_focus_and_precise_cropping_for_mllms.md)
- [\[CVPR 2026\] ApET: Approximation-Error Guided Token Compression for Efficient VLMs](apet_approximation-error_guided_token_compression_for_efficient_vlms.md)

</div>

<!-- RELATED:END -->
