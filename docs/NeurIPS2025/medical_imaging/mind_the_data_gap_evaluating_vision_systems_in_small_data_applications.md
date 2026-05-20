---
title: >-
  [Paper Note] Mind the (Data) Gap: Evaluating Vision Systems in Small Data Applications
description: >-
  [NeurIPS 2025][Medical Imaging][small data evaluation] This paper systematically compares MLLMs (e.g., Gemini, Qwen2.5-VL) and vision encoder + SVM pipelines on the NeWT ecological classification benchmark across the "sm…
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "small data evaluation"
  - "multimodal large language models"
  - "vision encoders"
  - "SVM"
  - "pretraining strategies"
date: 2026-05-08
content_hash: 485593713dbd6646
---

# Mind the (Data) Gap: Evaluating Vision Systems in Small Data Applications

**Conference**: NeurIPS 2025
**arXiv**: [2504.06486](https://arxiv.org/abs/2504.06486)  
**Code**: Unavailable  
**Area**: Medical Imaging / Computer Vision Evaluation
**Keywords**: small data evaluation, multimodal large language models, vision encoders, SVM, pretraining strategies

## TL;DR

This paper systematically compares MLLMs (e.g., Gemini, Qwen2.5-VL) and vision encoder + SVM pipelines on the NeWT ecological classification benchmark across the "small data regime" (10–1000 labeled samples). MLLMs plateau after 10–30 samples, whereas vision-based methods exhibit near-logarithmic growth throughout, calling on the community to prioritize small-data evaluation.

## Background & Motivation

**Background**: Current AI evaluation practices are heavily skewed—either toward zero-/few-shot settings (0–5 examples) or large-scale datasets (>10K examples). Through a manual survey of evaluation tasks used in recent vision and language research (covering methods such as CLIP, DINOv2, Gemini, and Phi-4), the authors find virtually no evaluation tasks in the 10–1000 training sample range, exposing a clear "data gap."

**Limitations of Prior Work**: This gap corresponds precisely to a large class of real-world applications—ecological monitoring requiring biologist-annotated species, medical diagnosis relying on expert annotations, and industrial inspection requiring domain knowledge. Such scenarios typically yield only tens to thousands of labeled samples, falling neither in the zero-shot nor the large-scale regime. Whether MLLMs optimized for zero-shot transfer are genuinely useful in these settings has never been systematically validated.

**Key Challenge**: The few-shot prompting mechanism of MLLMs fundamentally places labeled samples into the context window as demonstrations, yet the information utilization efficiency of in-context attention differs fundamentally from that of explicit classifiers (e.g., SVMs) operating in feature space. As available labeled samples grow from a handful to several hundred, the scaling behavior of the two paradigms may diverge substantially.

**Goal**: To conduct the first systematic comparison of MLLMs and vision encoder methods in the small data regime (10–1000 labeled samples), revealing their respective scaling characteristics.

**Key Insight**: The NeWT (Natural World Tasks) benchmark is selected as the evaluation platform—comprising 164 ecological binary classification tasks, each with only 200–400 labeled samples, naturally situated within the small data regime.

**Core Idea**: Leverage the NeWT benchmark with training subsets spanning from zero to the full dataset to compare the scaling behavior of MLLMs and vision + SVM approaches in the small data regime.

## Method

### Overall Architecture

This paper is an empirical evaluation study rather than a proposal for a new model. The experimental design centers on: (1) 164 binary classification tasks from NeWT; (2) eight training scales set at near-logarithmic intervals—0, 1, 3, 10, 30, 100, 300, and full dataset; and (3) comparing how MLLMs (utilizing labeled samples via few-shot prompting) and vision encoders (extracting frozen features followed by SVM training) scale with data volume.

### Key Designs

1. **Near-Logarithmic Training Scale Sampling**:

    - Function: Constructs eight training subsets of sizes 0, 1, 3, 10, 30, 100, 300, and the full dataset.
    - Mechanism: Labeled samples are drawn uniformly while ensuring at least one sample per class. All methods are evaluated independently at each scale.
    - Design Motivation: Logarithmic spacing covers multiple orders of magnitude, distinguishing behavioral differences between a few samples and a few hundred—precisely the range neglected by existing evaluations.

2. **MLLM Evaluation Protocol**:

    - Function: Provides a unified evaluation of Gemini Flash 2.0, Gemini Flash 1.5 8B, Qwen2-VL 7B, and Qwen2.5-VL 72B.
    - Mechanism: Labeled samples are inserted as few-shot demonstrations in the prompt; model responses are parsed into classification labels via deterministic regular expressions. When multiple species names appear in a response, the first mentioned is taken.
    - Design Motivation: This is the only standard mechanism by which MLLMs can incorporate labeled information in the small-data setting—by "showing" examples to the model.

3. **Vision Encoder + SVM Pipeline**:

    - Function: Evaluates vision encoders including DINOv2 (ViT-g/14), CLIP (ViT-L/14), and SigLIP (ViT-SO400M/14).
    - Mechanism: Frozen pretrained encoders extract image features; binary classification is performed by an SVM with hyperparameters tuned via cross-validated grid search using scikit-learn.
    - Design Motivation: SVMs are naturally well-suited for small-sample settings and are consistent with the original NeWT evaluation methodology, ensuring fair comparison.

### Loss & Training

MLLMs require no training; vision encoders are kept frozen, and only the SVM decision boundary is fitted. All evaluations report 95% confidence intervals computed from 1,000 bootstrap resamples.

## Key Experimental Results

### Main Results: Data Scaling Behavior

| Method | 3 samples | 10 samples | 30 samples | 100 samples | 300 samples | Trend |
|--------|-----------|-----------|-----------|------------|------------|-------|
| Gemini Flash 2.0 | ~67% | ~68% | ~70% | ~70% | ~70% | Plateau after 10–30 samples |
| Qwen2.5-VL 72B | ~64% | ~65% | ~68% | ~68% | ~68% | Similar plateau |
| DINOv2 ViT-g + SVM | ~55% | ~63% | ~71% | ~77% | ~81% | Sustained near-logarithmic growth |
| SigLIP SO400M + SVM | ~53% | ~60% | ~70% | ~76% | ~80% | Sustained near-logarithmic growth |

Key crossover point: At approximately 10 samples, DINOv2 + SVM surpasses all MLLMs, and the gap continues to widen thereafter.

### Ablation Study: Model Scale vs. Data Scale

| Configuration | Finding |
|---------------|---------|
| SigLIP scaled from 45 GFLOPs to 700+ GFLOPs (10× model size increase) | Limited accuracy gain |
| Labeled samples increased from 10 to 100 (10× data increase) | Accuracy gain consistently exceeds that of 10× model scaling |
| DINOv2 vs. CLIP/SigLIP on Species/Age tasks | DINOv2 substantially superior (self-supervised pretraining excels at fine-grained discrimination) |
| CLIP/SigLIP vs. DINOv2 on Gestalt/Behavior tasks | Language-supervised pretraining substantially superior (requires semantic reasoning) |

### Pretraining Strategy Comparison (30 samples, ViT-L)

| Task Cluster | DINOv2 | CLIP | SigLIP | Preferred Pretraining |
|-------------|--------|------|--------|----------------------|
| Species (species identification) | Highest | Moderate | Moderate | Self-supervised |
| Age (age estimation) | Highest | Lower | Lower | Self-supervised |
| Gestalt (holistic perception) | Lower | Highest | High | Language-supervised |
| Behavior (behavior recognition) | Lower | High | Highest | Language-supervised |
| Context (contextual understanding) | Lower | High | High | Language-supervised |
| Counting | Comparable | Comparable | Comparable | No clear difference |
| Health (health status) | Comparable | Comparable | Comparable | No clear difference |

### Key Findings

- MLLMs using few-shot prompting plateau after 10–30 samples and cannot continue to benefit from additional data.
- Vision encoder + SVM methods exhibit near-logarithmic continuous growth in the 10–300 sample range with no signs of saturation.
- A 10× increase in labeled data consistently yields greater accuracy gains than a 10× increase in model compute—challenging the prevailing "bigger is better" paradigm.
- DINOv2's self-supervised pretraining confers a unique advantage on fine-grained visual discrimination, while language-supervised pretraining (CLIP/SigLIP) leads on semantic reasoning tasks; this differentiation remains consistent across all training set sizes.

## Highlights & Insights

- **Revealing an Evaluation Blind Spot**: By manually surveying training set sizes reported in recent papers, the authors provide clear empirical evidence for the "data gap" in the 10–1000 sample range. This data-driven argumentation is more compelling than purely opinion-based claims.
- **Practical Value of a Counter-Intuitive Finding**: For real deployment scenarios, this result implies that when a few hundred labeled samples are available, a mid-sized vision encoder + SVM may outperform state-of-the-art MLLM APIs at substantially lower cost.
- **Complementarity of Pretraining Paradigms**: The systematic differences between self-supervised and language-supervised pretraining across task types provide empirical guidance for practical model selection—DINOv2 for fine-grained morphological discrimination, CLIP/SigLIP for semantic understanding.

## Limitations & Future Work

- Validation is limited to the NeWT ecological benchmark; while the authors claim generalizability to medical and industrial domains, no direct experiments are provided.
- MLLMs are evaluated only via few-shot prompting; whether parameter-efficient fine-tuning (e.g., LoRA) would alter the scaling behavior remains untested.
- Vision methods are restricted to frozen encoders + SVM; alternatives such as linear probing, k-NN, or lightweight fine-tuning are not explored.
- The effect of data augmentation on small-data methods is not considered, despite being standard practice in real-world applications.
- All vision encoders are trained on general-domain data; domain-specific foundation models (e.g., BioCLIP) are not evaluated.

## Related Work & Insights

- **vs. BioCLIP**: BioCLIP is a vision foundation model tailored for biology, developed in the authors' prior work. The use of general-purpose encoders in this paper—rather than BioCLIP—likely reflects an intention to ensure broadly generalizable conclusions.
- **vs. Many-Shot ICL (Jiang et al.)**: This prior work explores many-shot in-context learning in multimodal foundation models but does not systematically cover the 10–1000 sample range.
- **Methodological Transfer**: The proposed evaluation framework can be directly applied to small-data domains such as medical imaging (e.g., skin lesion classification, pathology slides) to guide method selection.

## Rating

- Novelty: ⭐⭐⭐ — An evaluation study without a new model, but the perspective is fresh and the problem is precisely defined.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive cross-comparison across multiple models, scales, and task types, with rigorous statistical analysis.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clearly articulated; figures are information-dense; the dual-panel design of Fig. 1 conveys findings at a glance.
- Value: ⭐⭐⭐⭐ — Provides direct guidance for method selection in practical AI deployment and advocates evaluation practices worth broader adoption.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Mind the Gap: Aligning Knowledge Bases with User Needs to Enhance Mental Health Retrieval](mind_the_gap_aligning_knowledge_bases_with_user_needs_to_enhance_mental_health_r.md)
- [\[NeurIPS 2025\] Toward a Vision-Language Foundation Model for Medical Data: Multimodal Dataset and Benchmarks for Vietnamese PET/CT Report Generation](toward_a_vision-language_foundation_model_for_medical_data_multimodal_dataset_an.md)
- [\[NeurIPS 2025\] Steering Generative Models with Experimental Data for Protein Fitness Optimization](steering_generative_models_with_experimental_data_for_protein_fitness_optimizati.md)
- [\[NeurIPS 2025\] EDBench: Large-Scale Electron Density Data for Molecular Modeling](edbench_large-scale_electron_density_data_for_molecular_modeling.md)
- [\[NeurIPS 2025\] Multiscale Guidance of Protein Structure Prediction with Heterogeneous Cryo-EM Data](multiscale_guidance_of_protein_structure_prediction_with_heterogeneous_cryo-em_d.md)

</div>

<!-- RELATED:END -->
