---
title: >-
  [Paper Note] Purging the Gray Zone: Latent-Geometric Denoising for Precise Knowledge Boundary Awareness
description: >-
  [ACL 2026][LLM Safety][Knowledge Boundary Awareness] This paper proposes the GeoDe framework, which constructs a truth hyperplane by training linear probes in the LLM latent space. By using the geometric distance from sa…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Knowledge Boundary Awareness"
  - "Abstention Fine-tuning"
  - "Latent Space Probing"
  - "Geometric Denoising"
  - "Hallucination Mitigation"
date: 2026-05-08
content_hash: e2a862ec063314cc
---

# Purging the Gray Zone: Latent-Geometric Denoising for Precise Knowledge Boundary Awareness

**Conference**: ACL 2026  
**arXiv**: [2604.14324](https://arxiv.org/abs/2604.14324)  
**Code**: [GitHub](https://github.com/Notbesidemoon/GeoDe)  
**Area**: Image Restoration  
**Keywords**: Knowledge Boundary Awareness, Abstention Fine-tuning, Latent Space Probing, Geometric Denoising, Hallucination Mitigation

## TL;DR
This paper proposes the GeoDe framework, which constructs a truth hyperplane by training linear probes in the LLM latent space. By using the geometric distance from samples to the hyperplane as a confidence signal to filter high-quality abstention fine-tuning data, the method effectively eliminates "gray zone" noise near decision boundaries and significantly enhances the truthfulness and reliability of the model.

## Background & Motivation

**Background**: LLMs frequently generate hallucinations (factually inaccurate responses). A practical mitigation method is abstention fine-tuning: training the model to provide correct answers for known questions and respond with "I don't know" for unknown ones.

**Limitations of Prior Work**: Existing abstention fine-tuning methods (such as R-Tuning) partition data into "known" and "unknown" sets based on the correctness of model responses. However, this accuracy-based partitioning introduces significant label noise—including "lucky guesses" (internal uncertainty but accidental correctness) and "format failures" (the model knows the answer but is judged incorrect due to formatting issues).

**Key Challenge**: A "gray zone" exists near the decision boundary in the latent space, where representations of known and unknown samples highly overlap and internal beliefs are ambiguous. Forcing positive or negative labels in this region causes the model to learn contradictory decision rules, ultimately leading to over-refusal or persistent hallucinations.

**Goal**: To design a data filtering method based on internal model representations to discard noisy samples near the decision boundary, retaining only high-confidence data for fine-tuning.

**Key Insight**: The linear representation hypothesis suggests that a "truth direction" exists within the LLM latent space, allowing a truth hyperplane to be learned via linear probes. Samples far from the hyperplane exhibit high confidence and reliable labels, while those nearby reside in the ambiguous region.

**Core Idea**: Replace hard partitioning based on response accuracy with geometric distance, filtering out gray zone samples via distance thresholds to perform abstention fine-tuning only on high-confidence subsets.

## Method

### Overall Architecture
GeoDe consists of three steps: (1) Testing the base model on source data $D0$, categorizing samples as known $D0_{ik}$ or unknown $D0_{idk}$ based on response correctness, and training a linear probe; (2) Calculating the geometric distance of each sample to the probe hyperplane on target data $D1$, selecting the top-X% most distant samples; (3) Adjusting target answers based on probe predictions (retaining correct answers for known and replacing with "I don't know" for unknown) and performing supervised fine-tuning on the filtered subset.

### Key Designs

1.  **Linear Probe and Truth Hyperplane**:
    - **Function**: Learns a decision boundary to distinguish between "known" and "unknown" within the LLM latent space.
    - **Mechanism**: Extracts hidden states $\mathbf{x} = f_{LLM}(q)$ from the LLM, training a logistic regression probe $f_{probe}(\mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b)$ using response correctness as binary labels. The probe weights $\mathbf{w}$ and bias $b$ define the truth hyperplane.
    - **Design Motivation**: Leverages existing internal truth representations of the LLM rather than relying on noisy external response signals. Linear probes are simple and efficient, and studies confirm the existence of linearly separable truth directions in LLM latent spaces.

2.  **Geometric Denoising**:
    - **Function**: Quantifies the confidence of each sample to filter out ambiguous samples in the gray zone.
    - **Mechanism**: Calculates the distance from the hidden state to the hyperplane $d(\mathbf{x}) = \frac{|\mathbf{w}^\top \mathbf{x} + b|}{\|\mathbf{w}\|_2}$. A larger distance indicates higher confidence. A threshold $\theta$ is set at the X-th percentile of distances (default X=25%), retaining only samples where $|d(\mathbf{x})| > \theta$. Samples with $d(\mathbf{x}) > 0$ are classified as known, while $d(\mathbf{x}) < 0$ are classified as unknown.
    - **Design Motivation**: Samples near the hyperplane are regions where label noise is concentrated; discarding them yields high-fidelity, linearly separable training signals.

3.  **Two Latent State Extraction Methods (TBG and SLT)**:
    - **Function**: Extracts feature representations from the LLM for probe training.
    - **Mechanism**: TBG (Token Before Generation) extracts the hidden state of the last token from the question input; SLT (Second Last Token) first generates an answer, concatenates the question and answer, and extracts the second-to-last token's hidden state. SLT captures the full context of the QA sequence.
    - **Design Motivation**: TBG is more efficient without a generation process; SLT utilizes answer information, potentially providing richer signals. The two methods are complementary.

### Loss & Training
Supervised fine-tuning is performed on the filtered subset using standard cross-entropy loss. For known samples, the correct answer is kept as the target; for unknown samples, the target is replaced with "I don't know." The training uses the AdamW optimizer with a learning rate of $1 \times 10^{-5}$, batch size of 16, and 3 epochs. Probes use logistic regression with L2 regularization.

## Key Experimental Results

### Main Results

| Method | TriviaQA F1_rel | NQ F1_rel | SciQ F1_rel | SimpleQA F1_rel |
|------|----------------|-----------|-------------|-----------------|
| R-Tuning | 74.4 | 58.7 | 63.5 | 25.4 |
| Probe-Tuning TBG | 75.7 | 62.5 | 64.5 | 22.6 |
| **GeoDe TBG** | **77.1** | **64.0** | **68.3** | **30.7** |
| Probe-Tuning SLT | 73.4 | 56.9 | 66.1 | 30.9 |
| **GeoDe SLT** | **77.0** | **64.9** | **68.9** | **34.9** |

### Ablation Study

| Configuration | TriviaQA F1_rel | Description |
|------|----------------|------|
| R-Tuning (baseline) | 74.4 | No probe, no filtering |
| R-Tuning-01 (strict) | 75.1 | Multiple sampling to filter ambiguous samples |
| Probe-Tuning | 75.7 | Probe prediction instead of accuracy partitioning |
| GeoDe (X=25%) | **77.1** | Probe + distance filtering |

### Key Findings
- GeoDe outperforms baselines across all four datasets, demonstrating particularly strong results in OOD scenarios (NQ, SciQ, SimpleQA), suggesting that geometric denoising enhances generalization.
- SLT variants generally outperform TBG variants, especially on SimpleQA (34.9 vs 30.7), indicating that answer information improves probe performance.
- A filtering ratio of X=25% is optimal; retaining too many samples introduces noise, while retaining too few leads to data scarcity.
- GeoDe simultaneously improves F1_ans (helpfulness) and F1_abs (truthfulness), breaking the trade-off between the two.

## Highlights & Insights
- The "gray zone" concept is intuitive and powerful: latent space visualization clearly illustrates label noise issues near the decision boundary, providing a geometric explanation for the difficulties of abstention fine-tuning.
- The method is remarkably concise—significantly improving fine-tuning quality using only a logistic regression probe and a distance threshold. This "less is more" approach to data filtering is widely applicable to other fine-tuning scenarios requiring high-quality data.
- Geometric distance serves as a more stable and computationally efficient proxy for confidence compared to uncertainty measures like semantic entropy.

## Limitations & Future Work
- Probe training relies on an independent source dataset $D0$, which must be similar to the target data distribution.
- The use of only linear probes may lack sufficient expressive power for more complex knowledge boundaries.
- Discarding 75% of data may not be feasible in data-scarce environments.
- Evaluation was limited to open-domain QA tasks; efficacy in other tasks (e.g., summarization, translation) remains unverified.
- Future work could explore non-linear probes, adaptive threshold selection, and the gradual introduction of boundary samples via curriculum learning.

## Related Work & Insights
- **vs R-Tuning**: R-Tuning partitions data solely by response accuracy without considering internal confidence. GeoDe introduces geometric distance filtering to eliminate gray zone noise.
- **vs Probe-Tuning**: Probe-Tuning replaces accuracy labels with probe predictions but still trains on the full dataset. GeoDe adds distance filtering to retain only high-confidence samples.

## Rating
- Novelty: ⭐⭐⭐⭐ The gray zone concept and geometric denoising are innovative, though linear probes build on prior work.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four datasets, two models, multiple baselines, and detailed ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear visualizations and a coherent, logical narrative.
- Value: ⭐⭐⭐⭐ The method is simple and practical, providing clear guidance for the abstention fine-tuning community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Into the Gray Zone: Domain Contexts Can Blur LLM Safety Boundaries](into_the_gray_zone_domain_contexts_can_blur_llm_safety_boundaries.md)
- [\[ACL 2026\] CURaTE: Continual Unlearning in Real Time with Ensured Preservation of LLM Knowledge](curate_continual_unlearning_in_real_time_with_ensured_preservation_of_llm_knowle.md)
- [\[ACL 2026\] Knowledge Poisoning Attacks on Medical Multi-Modal Retrieval-Augmented Generation](knowledge_poisoning_attacks_on_medical_multi-modal_retrieval-augmented_generatio.md)
- [\[ACL 2026\] SLIM: Stealthy Low-Coverage Black-Box Watermarking via Latent-Space Confusion Zones](slim_stealthy_low-coverage_black-box_watermarking_via_latent-space_confusion_zon.md)
- [\[ICML 2026\] Safety Anchor: Defending Harmful Fine-tuning via Geometric Bottlenecks](../../ICML2026/llm_safety/safety_anchor_defending_harmful_fine-tuning_via_geometric_bottlenecks.md)

</div>

<!-- RELATED:END -->
