---
title: >-
  [Paper Note] Organizing Unstructured Image Collections using Natural Language
description: >-
  [CVPR 2026][Image Generation][Open Semantic Multi-Clustering] This paper introduces a new task, Open Semantic Multi-Clustering (OpenSMC), and proposes the X-Cluster framework…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Open Semantic Multi-Clustering"
  - "Image Organization"
  - "Natural Language"
  - "Large Language Models"
  - "Multi-Granularity Clustering"
date: 2026-05-08
content_hash: db37724ee29c8c95
---

# Organizing Unstructured Image Collections using Natural Language

**Conference**: CVPR 2026
**arXiv**: [2410.05217](https://arxiv.org/abs/2410.05217)
**Code**: [https://oatmealliu.github.io/xcluster.html](https://oatmealliu.github.io/xcluster.html)
**Area**: Image Generation
**Keywords**: Open Semantic Multi-Clustering, Image Organization, Natural Language, Large Language Models, Multi-Granularity Clustering

## TL;DR

This paper introduces a new task, Open Semantic Multi-Clustering (OpenSMC), and proposes the X-Cluster framework, which converts images into text via an MLLM and subsequently employs an LLM to automatically discover clustering criteria and semantic substructures. Without any human-specified priors, the framework organizes large-scale unlabeled image collections into multi-dimensional, multi-granularity, and interpretable semantic clusters.

## Background & Motivation

**Background**: Image clustering is a fundamental task in machine learning. Deep Clustering (DC) methods produce a single partition, while Multi-Clustering (MC) methods yield multiple partitions but require predefined clustering criteria and cluster counts. Recent Text-Conditioned Multi-Clustering (TCMC) methods leverage MLLMs to generate semantic clusters, yet still require users to specify clustering criteria in advance.

**Limitations of Prior Work**: (1) Existing methods produce uninterpretable clustering results—only index labels, without human-readable category names. (2) Clustering outputs of DC and MC methods are governed by model inductive biases and hyperparameters rather than the inherent semantics of the data. (3) TCMC methods assume users already know meaningful clustering criteria, whereas for large-scale, complex datasets users often have no knowledge of which dimensions are meaningful.

**Key Challenge**: Ideal image organization should automatically discover multiple meaningful clustering dimensions (e.g., "Activity," "Location," "Emotion") from the data and assign human-readable names to each cluster. No existing vision model can reliably perform such high-level semantic reasoning over large image collections simultaneously.

**Goal**: To define the OpenSMC task—given an unlabeled image collection, automatically discover multiple clustering criteria, the number and names of clusters under each criterion, and the cluster assignments of images, with all outputs expressed in natural language and without any human-specified priors.

**Key Insight**: The authors observe that although no vision model can directly perform semantic reasoning over large-scale image sets, LLMs possess powerful topic discovery and summarization capabilities in the textual domain. By "translating" images into text, one can leverage LLMs to discover clustering criteria from large collections of textual descriptions.

**Core Idea**: Images are converted into textual proxies (captions/tags); an LLM then discovers clustering criteria in the text space, after which cluster assignments are performed back in the visual space. Text serves as the bridge connecting visual perception and semantic reasoning.

## Method

### Overall Architecture

X-Cluster is a two-stage, training-free framework: (1) **Criteria Proposer**—processes the entire image collection to discover multiple clustering criteria (e.g., "Activity," "Location"); (2) **Semantic Grouper**—for each discovered criterion, organizes images into named semantic clusters (e.g., "Surfing" and "Skateboarding" under Activity) at three granularity levels (coarse/medium/fine). The framework explores three design variants (Caption-based, Tag-based, Image-based), with the Caption-based variant achieving the best performance.

### Key Designs

1. **Criteria Proposer**:

    - Function: Automatically discovers meaningful clustering criteria/dimensions from an unlabeled image collection.
    - Mechanism: The Caption-based Proposer (primary variant) first uses an MLLM (LLaVA-NeXT-7B) to generate a detailed description for each image, $e_n = \text{MLLM}(x_n)$. All captions are randomly shuffled and partitioned into groups of 400, which are fed to an LLM (Llama-3.1-8B) for joint analysis; the LLM extracts common themes from the textual descriptions as clustering criteria: $\tilde{\mathcal{R}} = \text{LLM}(\{e_n\})$. A Criteria Refinement step then prompts the LLM to merge semantically overlapping criteria (e.g., "Outdoor" vs. "Open space") and remove noisy ones.
    - Design Motivation: Captions contain richer semantic information than tags and guide the LLM toward more comprehensive criteria discovery. On the Hard criteria set, the Caption-based Proposer outperforms the Tag-based variant by 32.2% in TPR.

2. **Semantic Grouper**:

    - Function: For each discovered criterion, organizes images into named semantic clusters.
    - Mechanism: The Caption-based Grouper (primary variant) operates in three steps: (i) the MLLM generates a criterion-specific description for each image, $e_n^l = \text{MLLM}(x_n, R_l)$, focusing on visual content relevant to that criterion; (ii) the LLM assigns an initial name to each description, $s_n^l = \text{LLM}(e_n^l, R_l)$, yielding an initial name set $\mathcal{S}_{\text{init}}^l$; (iii) the LLM refines the initial names into a three-level granularity hierarchy (coarse/medium/fine) and assigns each image a final cluster at each granularity level.
    - Design Motivation: Directly using initial names leads to an explosion in the number of categories (e.g., Activity initially yields 203 names, reduced to 12 at coarse granularity after refinement). Multi-granularity output allows users to select their preferred level of semantic detail.

3. **Multi-Granularity Assignment**:

    - Function: Performs image clustering at coarse, medium, and fine granularity levels.
    - Mechanism: A three-step structured refinement process: Initial Naming → Multi-granularity Cluster Refinement (the LLM organizes initial names into a three-tier category hierarchy) → Final Assignment (each image is assigned to one category at each of the three granularity levels). For example, under the Location criterion: coarse "Outdoor" → medium "Recreation" → fine "Tennis Court."
    - Design Motivation: Ground-truth annotation granularity is unknown in OpenSMC; multi-granularity output ensures the system matches user preferences at least at one granularity level. Experiments demonstrate that multi-granularity refinement yields greater consistency than flat refinement.

### Loss & Training

X-Cluster is a fully training-free framework requiring no training or fine-tuning. All components (LLaVA-NeXT-7B, Llama-3.1-8B, BLIP-2, CLIP ViT-L/14) use pretrained weights. The entire system is driven by carefully designed prompts comprising structured components including System Prompt, Input Explanation, Goal Explanation, Task Instruction, and Output Instruction.

## Key Experimental Results

### Main Results (Comparison with TCMC Methods)

| Method | Prior | COCO-4c CAcc/SAcc | Food-4c CAcc/SAcc | Avg CAcc/SAcc |
|--------|-------|----------|----------|----------|
| IC\|TC† | Criteria + #clusters | 48.9/53.2 | 50.5/61.7 | 62.0/57.4 |
| SSD-LLM† | Criteria + #clusters | 41.6/52.1 | 47.5/55.5 | 58.6/53.6 |
| MMaP† | Criteria + #clusters | 33.9/- | 43.8/- | 48.2/- |
| X-Cluster (Ours) | **None** | 51.2/48.4 | 48.1/64.9 | 61.8/62.3 |

*† Uses ground-truth criteria and cluster counts*

### Ablation Study — Multi-Granularity Refinement

| Configuration | Avg CAcc | Avg SAcc | Note |
|---------------|---------|---------|------|
| Initial Names | 37.1 | 49.3 | Directly uses initial names |
| Flat Refinement | 46.1 | 50.5 | Single-level refinement |
| Multi-Granularity | 61.8 | 62.3 | Multi-granularity refinement (Ours) |

### Key Findings

- Without any priors, X-Cluster achieves CAcc comparable to TCMC methods that use ground-truth criteria and cluster counts, while SAcc is even higher (62.3 vs. 57.4), demonstrating the feasibility of automatic criteria discovery.
- The Caption-based Proposer achieves a TPR of 75.1% on the Hard criteria set, substantially outperforming the Tag-based (42.9%) and Image-based (36.2%) variants, owing to the richer semantic context carried by captions.
- The Caption-based Grouper ranks first on 10 out of 15 test criteria (by HM metric), with an average CAcc of 59.9% approaching the CLIP zero-shot oracle at 58.1%.
- Multi-granularity refinement yields a substantial CAcc gain (37.1→61.8), indicating that granularity-consistent category names are critical for clustering quality.
- Experiments varying image count show that complex datasets (COCO-4c) require large numbers of images to discover comprehensive criteria, whereas simple datasets (Card-2c) suffice with as few as a single image.

## Highlights & Insights

- **Defining the novel OpenSMC task** with clear demarcation from DC, MC, and TCMC is a pioneering contribution.
- The core idea of **text as a reasoning proxy** is highly elegant: it leverages the textual reasoning capabilities of LLMs to compensate for the inability of vision models to perform global reasoning over large image collections. The "visual→text→reasoning→visual" paradigm has broad transferability.
- Practical application demonstrations are impressive: (1) discovering novel biases in T2I models (DALL·E3, SDXL)—such as the association between CEO and "Dark hair"—extending beyond conventional gender/race bias analysis; (2) analyzing visual factors underlying social media image popularity.
- The framework is built entirely on open-source models (LLaVA-NeXT-7B, Llama-3.1-8B), enabling local deployment and data privacy protection.

## Limitations & Future Work

- Computational overhead is substantial: processing COCO-4c (5,000 images) requires 7.6 hours on 4×A100 GPUs, with per-image captioning being the primary bottleneck.
- Caption quality from the MLLM directly affects system performance; omissions or hallucinations in captions may lead to incomplete criteria discovery or incorrect cluster assignments.
- Performance is weaker on fine-grained categories (e.g., bird species, vehicle models), where integration with specialized methods such as FineR may be necessary.
- Semantic names of clusters may not align perfectly with ground-truth labels (e.g., "Joyful" vs. "Happy"), introducing a systematic discount in SAcc.
- The current framework supports only image data; the paper discusses potential extensions to other modalities, including audio (Whisper), tabular data (TabT5), and protein structures.

## Related Work & Insights

- **vs. IC|TC**: IC|TC requires users to specify clustering criteria and cluster counts; X-Cluster automatically discovers criteria, counts, and names without any priors.
- **vs. SSD-LLM**: SSD-LLM requires the dataset's "main category" as input (e.g., "Food"); X-Cluster requires no prior knowledge.
- **vs. MMaP / MSub**: Learning-based multi-clustering methods require training and produce uninterpretable cluster results. X-Cluster is training-free and outputs natural language labels.
- **vs. Topic Discovery (NLP)**: Analogous to topic models in NLP, but more challenging in the visual domain since image semantics are implicit. X-Cluster's "visual→text→reasoning" paradigm bridges this gap.
- The multi-granularity clustering approach may inspire other unsupervised learning tasks, such as hierarchical image retrieval and dataset auditing.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Defines the novel OpenSMC task; the framework design is original and elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Six benchmarks plus two newly proposed benchmarks, three applications, and extensive ablation and supplementary analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Task definition is clear, methodological logic is rigorous, and the supplementary material is exceptionally comprehensive (60+ pages).
- Value: ⭐⭐⭐⭐⭐ Direct practical value for dataset auditing, bias discovery, social media analysis, and related applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Match-and-Fuse: Consistent Generation from Unstructured Image Sets](match-and-fuse_consistent_generation_from_unstructured_image_sets.md)
- [\[ICCV 2025\] Describe, Don't Dictate: Semantic Image Editing with Natural Language Intent](../../ICCV2025/image_generation/describe_dont_dictate_semantic_image_editing_with_natural_language_intent.md)
- [\[NeurIPS 2025\] SAO-Instruct: Free-form Audio Editing using Natural Language Instructions](../../NeurIPS2025/image_generation/sao-instruct_free-form_audio_editing_using_natural_language_instructions.md)
- [\[CVPR 2026\] Language-Free Generative Editing from One Visual Example](language-free_generative_editing_from_one_visual_example.md)
- [\[NeurIPS 2025\] Detecting Generated Images by Fitting Natural Image Distributions](../../NeurIPS2025/image_generation/detecting_generated_images_by_fitting_natural_image_distributions.md)

</div>

<!-- RELATED:END -->
