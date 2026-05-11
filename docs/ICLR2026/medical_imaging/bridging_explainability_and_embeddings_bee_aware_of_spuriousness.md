---
title: >-
  [Paper Note] Bridging Explainability and Embeddings: BEE Aware of Spuriousness
description: >-
  [ICLR 2026][Medical Imaging][Spurious correlation detection] This paper proposes the BEE framework, which identifies and names spurious correlations (SCs) directly from learned classifier weights by analyzing how fine-tu…
tags:
  - "ICLR 2026"
  - "Medical Imaging"
  - "Spurious correlation detection"
  - "weight space analysis"
  - "embedding geometry"
  - "linear probing"
  - "foundation models"
date: 2026-05-08
content_hash: 75885aa334579338
---

# Bridging Explainability and Embeddings: BEE Aware of Spuriousness

**Conference**: ICLR 2026
**arXiv**: [2410.18970](https://arxiv.org/abs/2410.18970)
**Code**: [Publicly Available](https://arxiv.org/abs/2410.18970)
**Area**: Medical Imaging
**Keywords**: Spurious correlation detection, weight space analysis, embedding geometry, linear probing, foundation models

## TL;DR
This paper proposes the BEE framework, which identifies and names spurious correlations (SCs) directly from learned classifier weights by analyzing how fine-tuning perturbs the weight-space geometry of pre-trained representations. The method requires no counterfactual samples and can discover hidden dataset biases. On ImageNet-1k, BEE uncovers spurious associations that reduce accuracy by up to 95%.

## Background & Motivation
**Background**: Deep neural networks, particularly fine-tuned foundation models, are widely deployed in high-stakes domains such as healthcare and finance. Spurious correlations (SCs) cause models to make decisions based on task-irrelevant features, with potentially severe consequences. Detecting SCs is therefore critical for ensuring model reliability.

**Limitations of Prior Work**: Existing approaches fall into two broad categories. **Data-driven methods** (e.g., SpLiCE, Lg) analyze dataset statistics to flag concepts correlated with class labels, but cannot determine whether the model has actually learned these correlations. **Error-driven methods** (e.g., B2T) infer SCs from validation-set errors, but require counterfactual examples to expose model shortcuts. When such counterexamples are absent—a common scenario in practice—both categories of methods fail.

**Key Challenge**: Data-driven methods ignore the model internals, while error-driven methods depend on counterexamples; yet many harmful SCs arise precisely because no counterexamples exist in the dataset. Existing interpretable approaches (e.g., CBMs) require pre-defined concept sets and sacrifice model expressiveness. The fundamental question is: **how can one discover spurious correlations actually learned by a model without relying on counterexamples?**

**Goal**: (1) Identify model-learned SCs without requiring counterexamples; (2) not merely detect but **name** the specific concepts responsible for each SC; (3) generalize across visual and textual modalities and multiple foundation model architectures.

**Key Insight**: The key observation is that during fine-tuning, linear classifier weights drift from their initialization—i.e., the zero-shot class-name embeddings—and this drift direction encodes what the model has learned, including spurious associations. Because weights and concept embeddings share the same embedding space, geometric relationships can be used to directly identify which class-irrelevant concepts are highly similar to the learned weights.

**Core Idea**: Leverage the drift direction of classification weights relative to zero-shot initialization in the embedding space to identify concepts that are class-irrelevant yet highly similar to the learned weights, thereby surfacing spurious correlations.

## Method

### Overall Architecture
BEE is a weight-space diagnostic framework. Given a training dataset, a foundation model, and a concept set, it outputs a list of spurious concepts learned for each class. The pipeline consists of two main steps: (1) train a linear probe on top of foundation model embeddings and observe weight drift; (2) rank concepts in the embedding space by their similarity to the drifted weights while filtering out class-relevant concepts, automatically identifying SCs.

### Key Designs

1. **Weight Initialization and Drift Observation**:

    - Function: Initialize linear layer weights using text embeddings of class names, $w_k^0 = M(\text{class\_name}_k)$, and observe the learned weights $w_k^*$ after training.
    - Mechanism: Zero-shot weights $w_k^0$ encode the "pure semantics" of each class, whereas post-training weights $w_k^*$ mix genuine and spurious features. The drift direction in embedding space reveals what the model has learned. The linear probe serves as a transparent diagnostic lens that renders the analysis interpretable.
    - Design Motivation: Linear layer weights and concept embeddings reside in the same space, enabling concept ranking. Using a linear probe rather than full fine-tuning ensures transparency; experiments further confirm that SCs discovered via linear probing persist in fully fine-tuned models.

2. **Concept Extraction and Filtering (Step 2a)**:

    - Function: Extract concepts from the dataset and filter out class-relevant ones, retaining only "class-neutral" candidates.
    - Mechanism: Image captions are generated with GIT-Large (text data is used directly); YAKE is then applied to extract the top-256 n-grams as candidate concepts $C_{all}$. Class instances are subsequently filtered using Llama-3.1-8B-Instruct, with a second-pass filter based on WordNet hypernym/hyponym relations.
    - Design Motivation: Only concepts unrelated to the class definition are valid SC candidates. For example, "forest background" is strongly correlated with "land bird" but should not serve as a classification cue. The two-stage filtering (LLM + WordNet) ensures comprehensive removal of class-relevant concepts.

3. **Concept Ranking (Step 2b)**:

    - Function: For each class, rank candidate concepts by their similarity to the learned weights.
    - Mechanism: The positive-SC score is defined as $s_{k,i}^+ = w_k^{*\top} M(c_i) - \min_{k'} w_{k'}^{*\top} M(c_i)$. The intuition is to identify concepts that are highly similar to one class but dissimilar to others. Negative-SC candidates are ranked by dissimilarity $-w_k^{*\top} M(c_i)$.
    - Design Motivation: Using raw similarity to a single class can produce false positives (generic concepts are similarly close to all classes). Subtracting the minimum removes this baseline effect, ensuring that only class-discriminative concepts are retained.

4. **Dynamic Thresholding (Step 2c)**:

    - Function: Automatically determine the number of SCs to retain for each class.
    - Mechanism: A mean filter (window size $r$) is applied to the sorted scores, and the elbow point—where the smoothed curve deviates maximally from the line connecting its endpoints—is identified: $m_k = \lfloor r/2 \rfloor + \arg\max_i (\bar{s}_{k,1} - i \frac{\bar{s}_{k,1} - \bar{s}_{k,p}}{p-1} - \bar{s}_{k,i})$.
    - Design Motivation: Different classes may exhibit different numbers of SCs; a hard top-$k$ cutoff is therefore inappropriate. The dynamic threshold adaptively selects the appropriate number of SCs per class without manual tuning.

5. **SC Regularization (Downstream Application)**:

    - Function: Use the discovered SCs to construct a regularization term that improves model robustness.
    - Mechanism: The regularization loss constrains classifier weights to be equidistant from each SC concept embedding: $\mathcal{L}_{reg}(b) = \frac{\tau^2}{N} \sum_{k=1}^N [w_k^\top M(b) - sg(\frac{1}{N}\sum_j w_j^\top M(b))]^2$, with the total loss $\mathcal{L} = \mathcal{L}_{ERM} + \alpha \frac{1}{|\mathcal{B}|} \sum_{b \in \mathcal{B}} \mathcal{L}_{reg}(b)$.
    - Design Motivation: In the fully spurious setting (no counterexamples in training), GroupDRO fails, whereas SC regularization explicitly reduces reliance on spurious features through direct geometric constraints.

### Loss & Training
- AdamW optimizer ($lr=1\text{e}{-4}$, $wd=1\text{e}{-5}$), batch size 1024.
- Cross-entropy loss with class-balanced weights; logits scaled by CLIP temperature $\tau=100$.
- Weights are normalized after each update; early stopping based on class-balanced accuracy on the validation set.

## Key Experimental Results

### Main Results: SC-Augmented Zero-Shot Prompting

| Method | Waterbirds Worst | Waterbirds Avg | CelebA Worst | CelebA Avg | CivilComments Worst |
|--------|------------------|----------------|--------------|------------|---------------------|
| Basic zero-shot | 35.2 | 84.2 | 72.8 | 87.7 | 33.1 |
| w/ B2T | 48.1 | 86.1 | 72.8 | 88.0 | - |
| w/ SpLiCE | 48.1 | 82.5 | 67.2 | 90.2 | - |
| w/ Lg | 46.1 | 85.9 | 50.6 | 87.2 | - |
| **w/ BEE** | **50.3** | **86.3** | **73.1** | 85.7 | **53.2** |

BEE achieves substantially higher worst-group accuracy than all competing methods on Waterbirds and CivilComments.

### Quantifying SC Impact on ImageNet-1k

| True Class | Spurious Concept | Induced Class | Change in True-Class Recognition | Induced-Class Prediction Rate |
|------------|-----------------|---------------|----------------------------------|-------------------------------|
| Peafowl | firemen | Fire truck | 100% → 5.3% (**−94.7%**) | 0% → 93.4% |
| Mexican Hairless Dog | reading newspaper | Crossword | 47.5% → 0.9% (−46.6%) | 0% → 36.6% |
| Bernese Mountain Dog | shrimp | American lobster | 99.8% → 10.6% (−89.2%) | 0% → 37.2% |

### Regularization under the Fully Spurious Setting

| Method | Waterbirds Worst | CelebA Worst | CivilComments Worst |
|--------|------------------|--------------|---------------------|
| ERM | 43.2±5.7 | 9.6±1.0 | 18.6±0.3 |
| GroupDRO | 38.9±5.4 | 8.1±0.3 | 18.7±0.4 |
| Reg w/ random SCs | 46.6±2.7 | 9.4±0.0 | 19.1±1.6 |
| Reg w/ Lg's SCs | 50.4±0.1 | 8.3±0.0 | - |
| **Reg w/ BEE's SCs** | **57.9±0.3** | **10.4±0.5** | **31.3±0.7** |

In the absence of counterexamples, GroupDRO even underperforms ERM, whereas BEE's SC regularization consistently improves worst-group performance.

### Key Findings
- **SC Transfer Across Models**: SCs discovered by BEE on CLIP consistently cause significant performance degradation on diverse architectures including AlexNet, ResNet50, and ViT-L/16, indicating that SCs are a property of the dataset rather than the model.
- **Dangerous Shortcuts in MIMIC-CXR Clinical Notes**: BEE identifies "chest examination" and "chest radiograph" as SCs for the "no pathological findings" class; appending such phrases biases the classifier toward predicting "no finding," which could lead to missed diagnoses in clinical settings.
- **SC Discovery Without Counterexamples**: In the fully spurious setting where all minority-group samples are removed, BEE continues to identify SCs effectively, while error-analysis-based methods fail entirely.

## Highlights & Insights
- **Weight-Space Analysis as a Novel SC Detection Paradigm**: Rather than examining data distributions or prediction errors, BEE directly infers what has been learned from the geometric drift of classifier weights—an elegant approach that exploits the alignment properties of embedding spaces and can surface SCs invisible to conventional methods.
- **Linear Probe as a Diagnostic Lens**: Adopting the simplest possible classifier avoids the opacity of complex models, and experiments confirm that SCs discovered via linear probing generalize to fully fine-tuned models, establishing the universality of the findings.
- **Elbow Detection for Dynamic Thresholding**: Automatically determining the number of SCs per class without manual tuning enables the method to scale to all 1,000 classes of ImageNet.
- **Practical Safety Implications of MIMIC-CXR Findings**: The SCs identified in medical text directly point to model deficiencies that could cause missed diagnoses, demonstrating the method's real-world value in high-stakes domains.

## Limitations & Future Work
- The approach relies on the linear probing assumption; SCs encoded in a nonlinear manner may not be detectable.
- Concept extraction depends on YAKE and GIT-Large, so concept coverage is bounded by the captioning model's descriptive capacity.
- The current scope is limited to classification tasks; extension to SC detection in detection, segmentation, and generation settings remains an open problem.
- SC regularization requires a known SC set; integrating detection and mitigation into a closed-loop iterative framework is a promising direction.
- Neither BEE nor B2T detected SCs for CelebA-blonde hair, suggesting potential blind spots for certain types of shortcut features.

## Related Work & Insights
- **vs B2T**: B2T infers SCs from validation errors and therefore requires the presence of counterexamples. BEE infers SCs from weight drift without counterexamples and covers a broader range of concepts. On Waterbirds, BEE achieves a worst-group accuracy of 50.3% versus 48.1% for B2T.
- **vs SpLiCE/Lg**: These data-driven methods analyze concept distributions in the dataset but cannot confirm whether the model has actually learned the identified associations. BEE directly inspects model weights, ensuring that reported SCs are genuinely learned.
- **vs CBM**: Concept bottleneck models require pre-defined concept sets and architectural modifications at the cost of expressiveness. BEE requires no model modifications and analyzes original state-of-the-art models.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Analyzing SCs through weight-space geometry is a fundamentally new perspective with clear theoretical motivation and elegant method design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers vision and text modalities, five embedding models, five datasets, and includes quantitative, qualitative, and generative validation.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured with intuitive figures, though some mathematical notation is dense and requires careful reading.
- Value: ⭐⭐⭐⭐⭐ — Significant implications for AI safety and trustworthy AI; the MIMIC-CXR findings bear direct relevance to patient safety.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Bridging Vision and Language for Robust Context-Aware Surgical Point Tracking: The VL-SurgPT Dataset and Benchmark](../../AAAI2026/medical_imaging/bridging_vision_and_language_for_robust_context-aware_surgical_point_tracking_th.md)
- [\[ICLR 2026\] AntigenLM: Structure-Aware DNA Language Modeling for Influenza](antigenlm_structure-aware_dna_language_modeling_for_influenza.md)
- [\[CVPR 2026\] Bridging the Skill Gap in Clinical CBCT Interpretation with CBCTRepD](../../CVPR2026/medical_imaging/bridging_the_skill_gap_in_clinical_cbct_interpreta.md)
- [\[ICLR 2026\] Fusing Pixels and Genes: Spatially-Aware Learning in Computational Pathology](fusing_pixels_and_genes_spatially-aware_learning_in_computational_pathology.md)
- [\[ICLR 2026\] DISCO: Densely-overlapping Cell Instance Segmentation via Adjacency-aware Collaborative Coloring](disco_densely-overlapping_cell_instance_segmentation_via_adjacency-aware_collabo.md)

</div>

<!-- RELATED:END -->
