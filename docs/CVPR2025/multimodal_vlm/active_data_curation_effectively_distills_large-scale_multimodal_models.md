---
title: >-
  [Paper Note] Active Data Curation Effectively Distills Large-Scale Multimodal Models
description: >-
  [CVPR 2025][Multimodal VLM][Knowledge Distillation] Proposes ACID (Active data Curation as Implicit Distillation) and ACED (combined with explicit distillation), demonstrating that actively filtering training data using a larger model as a reference is a more effective multimodal model compression approach than traditional knowledge distillation. Combining the two complementarily achieves SOTA performance on 27 zero-shot tasks with fewer inference FLOPs.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Knowledge Distillation"
  - "Active Data Curation"
  - "Contrastive Learning"
  - "CLIP Compression"
  - "Inference Efficiency"
date: 2026-05-08
content_hash: bf25c99ada3e48d9
---

# Active Data Curation Effectively Distills Large-Scale Multimodal Models

**Conference**: CVPR 2025  
**arXiv**: [2411.18674](https://arxiv.org/abs/2411.18674)  
**Code**: None  
**Area**: Multimodal VLM  
**Keywords**: Knowledge Distillation, Active Data Curation, Contrastive Learning, CLIP Compression, Inference Efficiency

## TL;DR

Proposes ACID (Active data Curation as Implicit Distillation) and ACED (combined with explicit distillation), demonstrating that actively filtering training data using a larger model as a reference is a more effective multimodal model compression approach than traditional knowledge distillation. Combining the two complementarily achieves SOTA performance on 27 zero-shot tasks with fewer inference FLOPs.

## Background & Motivation

Deploying large-scale multimodal models such as CLIP/SigLIP to edge devices faces challenges due to high inference costs, necessitating the compression of small yet powerful models. Knowledge Distillation (KD) is a classic compression method that matches a small model's output distribution to that of a large model. Current SOTAs (TinyCLIP, MobileCLIP) employ complex strategies such as multi-teacher ensembles, synthetic captions, weight inheritance, and data augmentation.

However, two key gaps exist:

**KD methods are becoming increasingly complex**: Combinations of multiple loss functions, bespoke architectures, and complex training pipelines lead to poor reproducibility.

**"Consensus" in data curation limits imagination**: Existing active data curation methods assume the reference model should be smaller than or equal to the student model (for efficiency) and argue that data curated by a large model is unsuitable for a small model due to the capacity gap.

The core insight of this paper is that **active data curation is inherently a form of implicit distillation**. By using a large reference model to select "easy-for-reference but challenging-for-student" data to train the small model, it is equivalent to optimizing a novel distillation objective that combines model predictions and true labels. This theoretically and experimentally shatters the consensus that "large models cannot curate data for small models."

## Method

### Overall Architecture

Unified objective function: $\mathcal{L}_{full} = \mathcal{L}_{softmax/sigmoid}[\mathcal{B}_{CE}] + \lambda \cdot \mathcal{L}_{dist}[\mathcal{B}_{KD}]$

By configuring different $\lambda$ and data sampling strategies, all method variants can be recovered:
- IID-Baseline: $\lambda=0$, random sampling
- ACID: $\lambda=0$, active curation sampling (implicit distillation)
- Softmax-KD: $\lambda>0$, random sampling
- ACED: $\lambda>0$, active curation sampling + explicit distillation

### Key Designs

1. **ACID: Active Curation as Implicit Distillation**:

    - Function: Actively curating high-information mini-batches $\mathcal{B}$ (size $b$) from a super-batch $\mathcal{S}$ (size $B$) using a pre-trained large reference model $\theta_{ref}$ for training.
    - Mechanism: Two curation scoring strategies:
        - Easy-reference: $s^{easy\_ref} = -\mathcal{L}(\mathcal{B}|\theta_{ref})$, prioritizing samples that the reference model finds "easy".
        - Learnability: $s^{learn} = \mathcal{L}(\mathcal{B}|\theta) - \mathcal{L}(\mathcal{B}|\theta_{ref})$, selecting samples that are "easy for the reference but difficult for the student".
    - **Theoretical Contribution**: The authors prove that the expected training objective of easy-reference curation is equivalent to:
    $$\mathcal{E}_{easy-ref} = \frac{1}{Z}\sum_{x \in \mathcal{D}} KD[p(x) \cdot y(x); q(x)]$$
      Which is an implicit distillation objective combining the reference model prediction $p$ and the ground truth label $y$. Since the noise sources of model predictions and labels differ (model underfitting vs. annotation error), keeping only those where both agree achieves "mutual denoising".
    - Design Motivation: Contrary to traditional active learning, ACID should use a larger reference model than the student—which is naturally justified from a distillation perspective.

2. **Joint Batch Sampling**:

    - Function: Inside a batch, sample selection is interdependent (the loss of each sample in contrastive learning depends on other samples in the batch); thus, joint sampling is required.
    - Mechanism: Iteratively constructs batches using blocked Gibbs sampling. In each of the $n$ rounds, a chunk ($b/n$) is conditionally scored and sampled from the remaining candidates, then appended to the current batch.
    - Design Motivation: Independent sampling ignores intra-batch interactions, whereas joint sampling ensures the overall information content of the selected batch is maximized.
    - Key Hyperparameter: filtering ratio $f = 1 - b/B$ (default is 0.8, i.e., curating from a 5x larger super-batch).

3. **ACED: Combining Implicit and Explicit Distillation**:

    - Function: Combines ACID and Softmax-KD to simultaneously leverage two complementary pathways of knowledge transfer.
    - Mechanism: ACIDistill strategy—sampling a single batch using H-ACID while computing both contrastive loss and distillation loss.
    - Design Motivation: While ACID generally outperforms KD, it falls short of KD on fine-grained tasks like Cars and DTD (likely because data curation filters out some useful samples). Since ACID implicit distillation and KD explicit distillation optimize different objectives, they are complementary.

### Loss & Training

- Contrastive Loss: Uses the sigmoid variant (SigLIP style) by default, which is more scalable.
- Distillation Loss: Cross-entropy of the teacher-student contrastive probability matrices (Eq. 3).
- Evaluation Protocol StableEval: Systematically analyzes cross-seed variance across 34 evaluations and selects 27 highly reliable evaluations to form a standard set, ensuring the credibility of experimental comparisons.

## Key Experimental Results

### Main Results

| Method | Samples | Inference GFLOPs | 27-Task Avg | ImageNet | COCO | Flickr |
|------|--------|-------------|-----------|----------|------|--------|
| MobileCLIP-S0 | 13B* | 3.70 | 63.6 | 67.8 | 49.6 | 76.7 |
| **ACED-F0** | 13B | **3.30** | **64.0** | **68.5** | **51.0** | **79.5** |
| MobileCLIP-S1 | 13B* | 7.64 | 67.9 | 72.6 | 53.0 | 80.0 |
| **ACED-F1** | 13B | **7.14** | **69.7** | **74.9** | **55.6** | **84.7** |
| MobileCLIP-S2 | 13B* | 10.81 | 69.8 | 74.4 | 54.4 | 81.8 |
| **ACED-F2** | 13B | **10.29** | **70.9** | **76.9** | **58.3** | **85.3** |

- ACED-F1 even outperforms MobileCLIP-S2 on ImageNet while using 34% fewer GFLOPs.
- ACED also outperforms SigLIP in the LiT-Decoder setup (frozen vision encoder + trained text decoder).

### Ablation Study

| Configuration | StableEval Avg | Description |
|------|----------------|------|
| IID-Baseline | Baseline | Random sampling training |
| Softmax-KD (L teacher) | Better than IID | Traditional distillation |
| I-ACID (L ref) | Better than KD | Implicit distillation |
| H-ACID (L ref) | > I-ACID | Hard-example prioritization is superior |
| ACED-ACIDistill | **Optimal** | ACID + KD complementary |

- Reference model scaling: There exists an optimal student-reference capacity ratio (Ti→B, S→L, B→g), beyond which performance saturates.
- ACID outperforms IID across all reference model scales, whereas Softmax-KD is only effective with large teachers.
- Across datasets: ACID significantly outperforms KD whether the reference/teacher is trained on WebLI-curated++ or WebLI.
- Across distillation objectives: ACID outperforms Softmax-KD, Sigmoid-KD, Feature-Matching KD, and their combinations.

### Key Findings

- **Large models can effectively curate data for small models**—shattering a long-standing consensus in the active learning and data curation fields.
- ACID is more effective for smaller student models (Ti, S), while KD performs better for larger student models (B).
- The two distillation methods are truly complementary: while ACID underperforms KD on a few benchmarks (4/27), the combination comprehensively outperforms them.

## Highlights & Insights

- **Outstanding Theoretical Contribution**: Rigorously derives that active curation is equivalent to implicit distillation, unifying the two independent fields of data curation and model compression.
- **Extreme Simplicity**: Requires no special architectures, no synthetic data, no weight inheritance, and no data augmentation—simply selecting the right training data.
- **StableEval Evaluation Protocol**: Systematically selects a reliable suite of benchmarks, which is highly referenceable for future work.
- **Systematic Scaling Analysis**: Conducts comprehensive ablations across multiple dimensions: reference model size, training data, student scale, and distillation objectives.

## Limitations & Future Work

- ACID requires a forward pass of the reference model on the super-batch, increasing training-time computational overhead (though there is absolutely no additional cost during inference).
- Hyperparameters such as filtering ratio and reference model selection require tuning.
- Validated only on contrastive VLMs (CLIP/SigLIP); its applicability to generative multimodal models (e.g., LLaVA) remains unknown.
- Theoretical derivation is based on the softmax variant; the theoretical guarantee for the sigmoid variant is weaker.

## Related Work & Insights

- RHO-Loss pioneered reference-based learnability scoring but was limited to small reference models; this work extends it to large reference models and provides a theoretical explanation.
- Connection to Curriculum Learning: ACID essentially constructs a dynamic curriculum, adaptively selecting the "most useful" samples based on the student's current state.
- Insight for Data Curation: The capacity gap between the reference model and the learner model is not an obstacle but an advantage.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The theoretical insight equating data curation with implicit distillation is both novel and profound.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multi-dimensional scaling analysis, 27 benchmarks, and comparisons across multiple methods.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear and elegant unified framework with rich visualizations.
- Value: ⭐⭐⭐⭐⭐ Makes important contributions to both multimodal model compression and data curation fields.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Img-Diff: Contrastive Data Synthesis for Multimodal Large Language Models](img-diff_contrastive_data_synthesis_for_multimodal_large_language_models.md)
- [\[ACL 2025\] CORDIAL: Can Multimodal Large Language Models Effectively Understand Coherence Relations?](../../ACL2025/multimodal_vlm/cordial_can_multimodal_large_language_models_effectively_understand_coherence_re.md)
- [\[CVPR 2025\] SldprtNet: A Large-Scale Multimodal Dataset for CAD Generation in Language-Driven 3D Design](sldprtnet_a_large-scale_multimodal_dataset_for_cad_generation_in_language-driven.md)
- [\[ACL 2025\] Scalable Vision Language Model Training via High Quality Data Curation](../../ACL2025/multimodal_vlm/scalable_vision_language_model_training_via_high_quality_data_curation.md)
- [\[CVPR 2025\] Synthetic Data is an Elegant GIFT for Continual Vision-Language Models](synthetic_data_is_an_elegant_gift_for_continual_vision-language_models.md)

</div>

<!-- RELATED:END -->
