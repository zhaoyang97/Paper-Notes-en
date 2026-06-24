---
title: >-
  [Paper Note] BioBench: A Blueprint to Move Beyond ImageNet for Scientific ML Benchmarks
description: >-
  [NeurIPS 2025][Model Compression][Ecological Imagery] BioBench is proposed as a unified benchmark spanning 9 ecological vision tasks, 4 taxonomic kingdoms, 6 image modalities, and 3.1 million images. It demonstrates that ImageNet top-1 accuracy explains only 34% of the variance across ecological tasks, and that approximately 30% of model rankings are incorrect among frontier models exceeding 75% accuracy.
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "Ecological Imagery"
  - "Visual Benchmark"
  - "ImageNet Limitations"
  - "Transfer Learning"
  - "AI for Science"
date: 2026-05-08
content_hash: a99bcebef5f48152
---

# BioBench: A Blueprint to Move Beyond ImageNet for Scientific ML Benchmarks

**Conference**: NeurIPS 2025
**arXiv**: [2511.16315](https://arxiv.org/abs/2511.16315)  
**Code**: [github.com/samuelstevens/biobench](https://github.com/samuelstevens/biobench)  
**Area**: Model Compression
**Keywords**: Ecological Imagery, Visual Benchmark, ImageNet Limitations, Transfer Learning, AI for Science

## TL;DR
BioBench is proposed as a unified benchmark spanning 9 ecological vision tasks, 4 taxonomic kingdoms, 6 image modalities, and 3.1 million images. It demonstrates that ImageNet top-1 accuracy explains only 34% of the variance across ecological tasks, and that approximately 30% of model rankings are incorrect among frontier models exceeding 75% accuracy.

## Background & Motivation

**Background**: Visual research continues to center on ImageNet-1K, MS COCO, and ADE20K for evaluation, and SOTA claims for new models (ViT, DINOv2, CLIP) are anchored to these leaderboards. Scientific domains—radiology, histopathology, microbiology, ecology—involve images that differ fundamentally from web photographs.

**Limitations of Prior Work**:
   - ImageNet's RGB web photographs differ substantially from scientific images such as camera-trap infrared frames, multispectral drone imagery, and microscope slides in spectrum, noise, and distribution;
   - Scientific tasks are fine-grained and long-tailed—ecologists must distinguish thousands of insect species, yet ImageNet's 1,000 classes capture almost none of this granularity;
   - Consequently, higher ImageNet accuracy does not reliably translate to better performance on scientific tasks.

**Key Challenge**: Across 46 modern vision Transformer checkpoints, the Spearman $\rho$ between ImageNet top-1 and ecological task performance is only 0.55 overall, dropping to 0.42 among models exceeding 75%—meaning approximately 30% of rankings among frontier models are inverted.

**Goal**:
   - Construct a unified, reproducible ecological vision benchmark that allows researchers to evaluate models on tasks that genuinely matter;
   - Quantify the degree to which ImageNet fails as a proxy for scientific AI evaluation;
   - Provide a generalizable benchmark design blueprint applicable to other scientific domains such as medicine and manufacturing.

**Key Insight**: Ecology offers abundant public data and well-annotated tasks accumulated through CV4Ecology challenges, making it an ideal testbed.

**Core Idea**: Unify 9 disparate ecological vision tasks into BioBench, demonstrate that ImageNet leaderboards have lost predictive power for scientific AI, and establish a standardized paradigm for benchmark design.

## Method

### Overall Architecture

- **Input**: Frozen embeddings $f: \text{image} \to \mathbb{R}^d$ from any visual model
- **Evaluation**: Lightweight linear/logistic regression probes fitted on each of the 9 ecological tasks
- **Output**: Per-task macro-F1 (with domain-specific metrics for FishNet/FungiCLEF), accompanied by bootstrap confidence intervals

### Key Designs

1. **Task Coverage**:

    - Function: Unify 9 publicly available ecological tasks
    - Coverage: 4 taxonomic kingdoms (Animalia / Plantae / Fungi / Protista) × 6 image modalities (drone RGB, web video, microscopy, natural environment photos, specimen photos, camera-trap frames), totaling 3.1 million images
    - Task Types: Species classification (iNat2021, Pl@ntNet, FungiCLEF, Herbarium19, iWildCam, Plankton), individual re-identification (BelugaID), behavior recognition (KABR, MammalNet), and functional trait prediction (FishNet)
    - Design Motivation: Diversity across taxa, distributions, and task types ensures the benchmark exposes model weaknesses along multiple dimensions

2. **Unified Evaluation Protocol**:

    - Function: Standardize the evaluation pipeline to eliminate "benchmark gaming"
    - Mechanism: All models expose a single embedding interface $f: \text{image} \to \mathbb{R}^d$. Evaluation uses frozen features with linear probes, reporting macro-F1 with bootstrap confidence intervals. Full evaluation of a ViT-L completes in 6 hours on a single A6000 GPU
    - Design Motivation: Linear probing isolates representation quality from task-specific engineering; macro-F1 provides fair treatment of long-tailed classes

3. **Predictive Power Analysis**:

    - Function: Quantitatively measure the predictive validity of ImageNet for BioBench
    - Mechanism: Compute $R^2$ (0.34) and Spearman $\rho$ (0.55) between ImageNet top-1 and BioBench scores across 46 checkpoints, as well as $\rho$ (0.42) restricted to models above 75% ImageNet accuracy. The misranking rate is defined as $\frac{1}{2}(1-\rho) = 30\%$
    - Design Motivation: Rather than subjectively asserting that ImageNet is insufficient, the analysis rigorously demonstrates this through statistical evidence

## Key Experimental Results

### Main Results — Rankings of 46 Models on BioBench

| Model Family | ImageNet top-1 | BioBench Mean | Best Task | Worst Task |
|---|---|---|---|---|
| SigLIP 2 (ViT-1B) | 88.9 | **43.5** | MammalNet 73.9 | BelugaID 3.6 |
| SigLIP (SO400M) | 87.8 | 42.0 | FishNet 69.0 | BelugaID 4.0 |
| BioCLIP 2 (ViT-L) | 80.0 | 41.7 | Herbarium19 73.1 | BelugaID 3.0 |
| DINOv2 (ViT-g) | 86.7 | 41.7 | FishNet 75.2 | BelugaID 4.5 |
| AIMv2 (ViT-3B) | 86.7 | 36.9 | MammalNet 68.8 | BelugaID 1.7 |
| CLIP (ViT-L) | 83.9 | 36.7 | FishNet 64.4 | BelugaID 2.8 |
| BioCLIP (ViT-B) | 58.5 | 34.3 | FishNet 62.6 | BelugaID 4.6 |

### Ablation Study — Predictive Power of ImageNet

| Analysis | $R^2$ | Spearman $\rho$ | Misranking Rate |
|---|---|---|---|
| All 46 checkpoints | 0.34 | 0.55 | 22% |
| >75% ImageNet | — | 0.42 | **30%** |
| Single task (Herbarium19) | Lower | <0.25 | >37% |

### Key Findings

- **ImageNet completely fails in the frontier regime**: Among models exceeding 75% ImageNet accuracy, 30% of ImageNet rankings are inverted relative to BioBench rankings—implying that model selection based on ImageNet carries nearly a 1-in-3 chance of selecting an inferior model.
- **General-purpose vs. domain-specific models**: BioCLIP 2 (80% ImageNet) achieves a BioBench mean of 41.7, matching DINOv2 (86.7% ImageNet), demonstrating that domain-specific pretraining is more consequential than raw ImageNet performance.
- **BelugaID is the universal bottleneck**: Individual re-identification is extremely challenging for all frozen-feature methods (3–9%), suggesting that current representations lack fine-grained individual-level discriminability.
- **Stagnating progress**: Figure 3 shows that many new model releases fail to improve BioBench scores; only the CLIP/SigLIP families have genuinely advanced SOTA.

## Highlights & Insights

- **Rigorous quantification of the "ranking cliff" phenomenon**: Rather than qualitatively criticizing ImageNet, the paper precisely computes a 30% misranking probability—a figure compelling enough in its own right to motivate community attention toward domain-specific evaluation.
- **"Minimal embedding interface" design philosophy**: Requiring only a frozen embedding output dramatically lowers participation costs and can serve as a template for benchmarks in other scientific domains.
- **macro-F1 as the default metric**: Explicitly rewards tail-class performance, aligning better with ecological needs where rare species identification is often of greater importance than aggregate accuracy.
- **Methodological value of the benchmark design blueprint**: Beyond contributing a benchmark, the paper offers principled guidance on "how to build benchmarks for scientific domains"—covering distributional diversity, long-tail metrics, and bootstrap statistical testing.

## Limitations & Future Work

- **Ecology only**: Medical, manufacturing, and other domains may require different task types such as detection and segmentation.
- **Frozen features underestimate fine-tuning gains**: Linear probing isolates representation quality, but fine-tuning may alter rankings in practical deployment settings.
- **Single metric**: macro-F1 is inappropriate for some applications (e.g., scenarios requiring precision–recall operating point selection).
- **No pretraining data analysis**: The effect of pretraining data volume and composition on BioBench performance is not examined.
- **Future Directions**:
    - Extension to medical imaging (PathBench?) and remote sensing (GeoBench?)
    - Addition of few-shot and fine-tuning evaluation modes
    - Incorporation of detection and segmentation tasks

## Related Work & Insights

- **vs. VTAB/Taskonomy**: General transfer learning benchmarks include only minimal scientific content and fail to capture challenges unique to ecology such as fine-grained classification, extreme long tails, and environmental variation.
- **vs. iNaturalist**: iNaturalist provides large-scale species classification but lacks diverse task types such as behavior recognition and trait prediction.
- **vs. WILDS**: WILDS includes iWildCam but treats ecological monitoring as one of many domains, without deeply exploring its multifaceted nature.

## Rating

- Novelty: ⭐⭐⭐⭐ First unified ecological vision AI benchmark; the quantitative analysis of ImageNet ranking failures is impactful
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 46 models, 11 families, and 9 tasks
- Writing Quality: ⭐⭐⭐⭐ Clear logical structure and rigorous statistical analysis
- Value: ⭐⭐⭐⭐⭐ Significant methodological influence on benchmarking for AI for Science; practical tooling (code + API) is already available

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Sci-LoRA: Mixture of Scientific LoRAs for Cross-Domain Lay Paraphrasing](../../ACL2025/model_compression/sci-lora_mixture_of_scientific_loras_for_cross-domain_lay_paraphrasing.md)
- [\[NeurIPS 2025\] Beyond Random: Automatic Inner-Loop Optimization in Dataset Distillation](beyond_random_automatic_inner-loop_optimization_in_dataset_distillation.md)
- [\[NeurIPS 2025\] Binary Quadratic Quantization: Beyond First-Order Quantization for Real-Valued Matrix Compression](binary_quadratic_quantization_beyond_first-order_quantization_for_real-valued_ma.md)
- [\[NeurIPS 2025\] Beyond Higher Rank: Token-wise Input-Output Projections for Efficient Low-Rank Adaptation](beyond_higher_rank_token-wise_input-output_projections_for_efficient_low-rank_ad.md)
- [\[ACL 2026\] When Reviews Disagree: Fine-Grained Contradiction Analysis in Scientific Peer Reviews](../../ACL2026/model_compression/when_reviews_disagree_fine-grained_contradiction_analysis_in_scientific_peer_rev.md)

</div>

<!-- RELATED:END -->
