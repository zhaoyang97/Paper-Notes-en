---
title: >-
  [Paper Note] Toward Gaze Target Detection in Young Autistic Children
description: >-
  [AAAI 2026][Signal & Communication][Gaze Target Detection] To address the severe class imbalance in gaze target detection for autistic children—where face-directed gaze accounts for only 6.6% of samples—this paper propos…
tags:
  - "AAAI 2026"
  - "Signal & Communication"
  - "Gaze Target Detection"
  - "Autism Spectrum Disorder"
  - "Class Imbalance"
  - "Multimodal Large Language Models"
  - "Coarse-to-Fine Framework"
date: 2026-05-08
content_hash: 8975ea353c41e4b8
---

# Toward Gaze Target Detection in Young Autistic Children

**Conference**: AAAI 2026
**arXiv**: [2511.11244](https://arxiv.org/abs/2511.11244)
**Code**: [ShijianDeng/AGT](https://github.com/ShijianDeng/AGT)
**Area**: Signal & Communication
**Keywords**: Gaze Target Detection, Autism Spectrum Disorder, Class Imbalance, Multimodal Large Language Models, Coarse-to-Fine Framework

## TL;DR
To address the severe class imbalance in gaze target detection for autistic children—where face-directed gaze accounts for only 6.6% of samples—this paper proposes the Socially Aware Coarse-to-Fine (SACF) framework. A fine-tuned Qwen2.5-VL serves as a social-context-aware gate that routes inputs to either a socially aware or a socially agnostic expert model. Evaluated on the newly introduced AGT dataset, the framework substantially improves face gaze detection performance (Face L2 reduced by 13.9% on Sharingan; F1 improved from 0.753 to 0.761).

## Background & Motivation
- Autism Spectrum Disorder (ASD) affects approximately 1 in 31 eight-year-old children. A hallmark characteristic is atypical social attention, particularly difficulties in initiating and responding to joint attention.
- Joint attention assessment is a cornerstone of early ASD diagnosis and intervention, yet it relies on highly trained professionals and is labor-intensive and difficult to scale.
- Existing gaze target detection research focuses almost exclusively on neurotypical adults and children; such models exhibit substantially degraded performance on autistic children.
- Root cause: autistic children direct their gaze toward faces far less frequently than neurotypical peers, resulting in severe class imbalance in the data (only 6.6% face-directed gaze). Standard models tend to predict non-social targets and miss clinically critical social interaction moments.
- No autism-specific gaze target detection dataset exists.

## Core Problem
Given the extreme scarcity of face-directed gaze (6.6%) in autistic children's gaze data, how can gaze targets be accurately detected—especially without missing clinically critical face-gaze events? This represents a joint challenge of domain adaptation and class imbalance.

## Method

### Overall Architecture
An input image $I$ and the child's head bounding box $B_\text{head}$ are passed to the Social Context Awareness (SCA) module (fine-tuned Qwen2.5-VL-7B), which estimates a social context score $s$ (the probability that the child is looking at a face). A threshold gate routes the input: if $s$ is high (social scene), the input is background-blurred and forwarded to the Socially Aware Gaze Expert; otherwise, it is sent to the Socially Agnostic Gaze Expert. Each expert outputs a gaze heatmap; argmax yields the predicted gaze point. A spatial check then determines whether the predicted point falls within an adult face bounding box, producing the final semantic label (Face / Not Face).

### Key Designs
1. **Autism Gaze Target (AGT) Dataset**:

    - The first gaze target detection dataset for autistic children, sourced from 59 ethically approved videos recorded during CSBS-DP assessments.
    - 16,582 annotated frames: 9,874 training / 3,344 validation / 3,364 test.
    - Annotations include child head bounding boxes, adult face bounding boxes, gaze target points, and semantic labels (object / face / person-non-face / no object).
    - Class distribution: Face 6.6% (1,088 frames), Not Face 93.4% (15,494 frames).
    - Inter-annotator agreement: Cohen's Kappa = 0.757 (substantial agreement).

2. **Social Context Awareness (SCA) Module**:

    - Qwen2.5-VL-7B-Instruct is fine-tuned as a binary classifier to estimate the probability of the child looking at a face.
    - A threshold converts the score into a coarse label: Face or Not Face.
    - SCA performance: Face recall 65.53%, Not-Face recall 98.10%, Face F1 = 0.673.
    - Leverages the large-scale pretraining of MLLMs to understand scene-level social context.

3. **Two-Pathway Gated Experts**:

    - **Socially Aware Gaze Expert ($\text{Ex}_\text{aware}$)**: Trained on augmented data where irrelevant background regions are strongly Gaussian-blurred when the target is known, guiding the model to focus on plausible targets (e.g., faces) and avoiding extreme failures; optimized specifically for face-directed gaze scenarios.
    - **Socially Agnostic Gaze Expert ($\text{Ex}_\text{agnostic}$)**: Trained on the original unmodified data; maximizes performance for high-frequency non-social and ambiguous scenes without constraints imposed by face-class requirements.
    - Both experts are built on the GazeLLE architecture (frozen DINOv2 encoder + lightweight Transformer decoder) or the Sharingan architecture.

4. **Gating and Inference**:

    - The SCA output determines which expert processes the input.
    - Final semantic classification is based on a spatial check: whether the predicted gaze point falls within an adult face bounding box.
    - When the SCA gate is correct (~96% of the time), the framework fully exploits the complementary strengths of both experts.

### Loss & Training
- Gaze heatmap loss: pixel-level Binary Cross-Entropy (BCE) between the predicted heatmap and the Gaussian-blurred ground-truth heatmap.
- SCA module: Qwen2.5-VL-7B fine-tuned on the AGT training set for binary classification (face-looking / not-face-looking).
- Socially Aware Expert: trained with BCE loss on background-blurred augmented data.
- Socially Agnostic Expert: trained with BCE loss on the original data.
- Hardware: NVIDIA A6000 GPU.

## Key Experimental Results

| Method | L2 | L2_obj | L2_face | L2_pnf | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| Sharingan (original) | 0.0615 | 0.0615 | 0.0647 | 0.0561 | 0.4377 | 0.8010 | 0.5660 |
| Sharingan-AGT | 0.0486 | 0.0451 | 0.0949 | 0.0595 | 0.7744 | 0.7330 | 0.7531 |
| **Sharingan-SACF (Ours)** | **0.0480** | **0.0453** | **0.0817** | 0.0616 | 0.7647 | 0.7573 | **0.7610** |
| GazeLLE (original) | 0.0670 | 0.0630 | 0.1092 | 0.1041 | 0.3868 | 0.6553 | 0.4865 |
| GazeLLE-AGT | 0.0460 | 0.0405 | 0.1130 | 0.0804 | 0.6984 | 0.6408 | 0.6684 |
| **GazeLLE-SACF (Ours)** | **0.0453** | **0.0405** | **0.1019** | **0.0804** | **0.7041** | 0.6699 | **0.6866** |

- Upper bound (oracle gate): Sharingan-SACF reaches F1 = 0.9786, L2_face = 0.0378; GazeLLE-SACF reaches F1 = 0.9903, L2_face = 0.0307.

### Ablation Study
- **Neurotypical models fail on autism data**: The original Sharingan/GazeLLE models, trained on the neurotypical Childplay dataset, achieve only F1 = 0.566/0.487 on AGT, systematically overestimating face-directed gaze in autistic children.
- **Substantial improvement in Face L2**: SACF reduces face gaze error for Sharingan from 0.0949 to 0.0817 (−13.9%) and for GazeLLE from 0.1130 to 0.1019 (−9.8%).
- **Clear expert specialization**: When GT = Face, the Socially Aware Expert achieves L2 = 0.0303, three times better than the Agnostic Expert (0.0898); when misrouted to Face but GT = Not Face, the Agnostic Expert is 3.4× better.
- **Gate quality is the key bottleneck**: The oracle gate raises F1 from 0.761 to 0.979, indicating substantial headroom for SCA improvement; stronger future MLLMs can directly boost overall system performance.
- **Clinical significance**: A 10–14% reduction in face localization error corresponds to the predicted point moving 4–6 pixels closer to the true face region at 224×224 resolution.

## Highlights & Insights
- The first systematic study of gaze target detection in autistic children, along with the first AGT dataset (16,582 frames), filling an important gap in the field.
- The SACF "divide-and-conquer" framework is elegantly designed: an MLLM performs coarse social scene classification while expert models handle fine-grained localization, effectively addressing extreme class imbalance.
- The upper bound analysis clearly identifies the system bottleneck (gate quality) and implies that performance will naturally improve as MLLMs advance.
- The problem formulation carries strong clinical value, bridging AI research and early autism intervention.

## Limitations & Future Work
- The SCA module (Qwen2.5-VL-7B) achieves only 65.53% Face recall, representing the primary system bottleneck.
- The framework involves multiple models (one MLLM plus two experts), resulting in relatively high inference cost.
- The AGT dataset is derived from a single assessment context (CSBS-DP); generalization to other naturalistic settings remains to be validated.
- Only binary Face / Not Face classification is modeled; finer-grained target semantics (e.g., specific toys or individuals) are not addressed.
- Temporal information (gaze trajectories across video frames) is unexplored, which may be more valuable for joint attention detection.

## Related Work & Insights
- **vs. GazeLLE / Sharingan (trained on Childplay)**: These models are trained on neurotypical data and remain biased toward high-frequency social gaze distributions; applied directly to autistic children, they achieve only F1 = 0.49–0.57. Fine-tuning on AGT raises F1 to 0.67–0.75, and SACF yields further gains.
- **vs. direct fine-tuning (GazeLLE-AGT)**: Direct fine-tuning on AGT improves overall L2 but may worsen Face L2 due to class imbalance (GazeLLE: 0.1130 vs. original 0.1092). SACF resolves this tension through expert routing.
- **vs. using Qwen2.5-VL directly for gaze detection**: MLLMs excel at scene understanding but are not suited for precise spatial localization. SACF restricts the MLLM to a coarse-grained routing role while delegating precise localization to dedicated gaze models.

- The design pattern of using an MLLM as a "semantic scene router" is generalizable to other vision tasks with severe class imbalance.
- The two-stage paradigm of coarse semantic classification followed by fine-grained expert localization has analogous applicability in medical imaging and similar domains.
- AI-assisted autism assessment tools can be extended toward automatic joint attention scoring and quantitative tracking of intervention outcomes.
- The AGT dataset can serve as a benchmark for studying domain transfer and extreme class imbalance.

## Rating
- Novelty: ⭐⭐⭐⭐ First to define and address the autism gaze detection problem; the MLLM-routed expert framework is creative, though technically it combines existing components.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive multi-baseline comparisons; insightful upper bound analysis; lacks cross-setting generalization experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is clear, problem definition is precise, and clinical significance is thoroughly articulated—an exemplary paper for socially impactful research.
- Value: ⭐⭐⭐⭐ High social impact; lays the foundation for AI-assisted autism assessment; the technical approach also offers reference value for other imbalanced detection tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Task Aware Modulation Using Representation Learning for Upscaling of Terrestrial Carbon Fluxes](task_aware_modulation_using_representation_learning_for_upsaling_of_terrestrial_.md)
- [\[AAAI 2026\] Text-Guided Channel Perturbation and Pretrained Knowledge Integration for Unified Multi-Modality Image Fusion](text-guided_channel_perturbation_and_pretrained_knowledge_integration_for_unifie.md)
- [\[ICLR 2026\] Multi-modal Data Spectrum: Multi-modal Datasets are Multi-dimensional](../../ICLR2026/signal_comm/multi-modal_data_spectrum_multi-modal_datasets_are_multi-dimensional.md)
- [\[ICLR 2026\] Deterministic Bounds and Random Estimates of Metric Tensors on Neuromanifolds](../../ICLR2026/signal_comm/deterministic_bounds_and_random_estimates_of_metric_tensors_on_neuromanifolds.md)
- [\[CVPR 2026\] ChartNet: A Million-Scale, High-Quality Multimodal Dataset for Robust Chart Understanding](../../CVPR2026/signal_comm/chartnet_a_million-scale_high-quality_multimodal_dataset_for_robust_chart_unders.md)

</div>

<!-- RELATED:END -->
