---
title: >-
  [Paper Note] Meta-Prompting for Automating Zero-Shot Visual Recognition with LLMs
description: >-
  [ECCV2024][Multimodal VLM][zero-shot recognition] This paper proposes MPVR (Meta-Prompting for Visual Recognition), which leverages a two-stage meta-prompting strategy to automatically generate diverse, class-specific VLM prompts, significantly improving the zero-shot recognition performance of models like CLIP without requiring manual design of LLM queries.
tags:
  - "ECCV2024"
  - "Multimodal VLM"
  - "zero-shot recognition"
  - "prompt ensembling"
  - "meta-prompting"
  - "VLM"
  - "LLM"
date: 2026-05-08
content_hash: 497deeee31c59ee1
---

# Meta-Prompting for Automating Zero-Shot Visual Recognition with LLMs

**Conference**: ECCV2024  
**arXiv**: [2403.11755](https://arxiv.org/abs/2403.11755)  
**Code**: [jmiemirza/Meta-Prompting](https://github.com/jmiemirza/Meta-Prompting)  
**Area**: LLM/NLP  
**Keywords**: zero-shot recognition, prompt ensembling, meta-prompting, VLM, LLM

## TL;DR

This paper proposes MPVR (Meta-Prompting for Visual Recognition), which leverages a two-stage meta-prompting strategy to automatically generate diverse, class-specific VLM prompts, significantly improving the zero-shot recognition performance of models like CLIP without requiring manual design of LLM queries.

## Background & Motivation

Dual-encoder VLMs such as CLIP excel in zero-shot classification by mapping images and text into a shared embedding space and calculating cosine similarity. Prior work has shown that prompt ensembling using multiple class-specific prompts can significantly boost classification accuracy.

Although existing methods (such as CUPL, DCLIP, and Waffle) utilize LLMs to generate class descriptions to expand prompt diversity, they still suffer from a key limitation: **they require manually designing LLM query templates for each downstream dataset**. This is not only labor-intensive but also prone to designer bias, which restricts the diversity and coverage of the generated prompts.

The authors pose a core question: Does manual design of LLM queries limit the quality of the final VLM prompts? The answer is affirmative—minimizing human intervention significantly improves zero-shot recognition accuracy.

## Core Problem

How can rich visual world knowledge be extracted from LLMs under a **fully automated** paradigm to generate diverse, task-specific, and class-specific VLM prompts, thereby enhancing zero-shot visual recognition performance?

## Method

### Overall Architecture

MPVR employs a **two-stage meta-prompting** strategy to progressively extract visual knowledge from LLMs:

**Stage 1: Meta-Prompting to Generate Task-Specific LLM Query Templates**

The meta-prompt consists of three parts:

1. **System Prompt**: General instructions that describe the task and expected output format, which remain constant across all experiments.
2. **In-context Example**: An exemplary downstream task paired with its corresponding LLM query templates (fixed to the DTD dataset for all experiments, except when the target dataset is DTD, where it switches to EuroSAT).
3. **Downstream Task Specification**: A brief description and metadata of the target task, which is the only part that varies across tasks and can be retrieved from public APIs or dataset web pages.

By inputting the meta-prompt into an LLM (such as GPT or Mixtral), $N=30$ diverse task-specific LLM query templates are generated. These templates contain a `<class name>` placeholder and incorporate task-specific visual style knowledge while remaining class-agnostic.

**Stage 2: Generating Class-Specific VLM Prompts**

The `<class name>` placeholder in the query templates generated in Stage 1 is replaced with concrete class names. The LLM is queried again to generate 10 class-specific VLM prompts for each template (with each prompt restricted to 50 tokens). This yields a large and diverse set of class descriptions.

### Zero-Shot Classification

For class $c$, all its VLM prompts are passed through the text encoder $\psi$ to obtain embeddings, which are then averaged to serve as the class representation $\psi_c$. The test image is encoded by the visual encoder $\phi$ to obtain its embedding, and classification is performed by computing the cosine similarity:

$$l_{\hat{c}}(x) = \frac{e^{\cos(\psi_{\hat{c}}, \phi(x))/\tau}}{\sum_{c \in C} e^{\cos(\psi_c, \phi(x))/\tau}}$$

### Key Designs

The core advantage of the two-stage approach lies in the **cascade amplification of diversity**: Stage 1 generates diverse query perspectives (e.g., various visual styles, camera angles, scenario descriptions), and Stage 2 generates specific class descriptions under each perspective, ultimately yielding an extremely comprehensive prompt corpus.

## Key Experimental Results

### Main Results (ViT-B/32 CLIP, 20 Datasets)

| Method | Average Gain (vs CLIP S-TEMP) | Max Gain |
|------|---------------------------|---------|
| MPVR (GPT) | +5.0% avg. | +19.8% (EuroSAT) |
| MPVR (Mixtral) | +4.5% avg. | +18.2% (EuroSAT) |

- Outperforms all baselines on 18 out of 20 datasets.
- Compared to CUPL: yields gains of 5.1% (GPT) and 6.3% (Mixtral) on Flowers-102.
- Compared to DCLIP: yields gains of 5.3% (GPT) and 3.3% (Mixtral) on UCF-101.

### Generalization Across Backbones (Table 2, 20-Dataset Average)

| Backbone | CLIP S-TEMP | MPVR (GPT) | Gain |
|------|-------------|------------|------|
| OpenAI ViT-B/16 | 61.9% | 66.7% | +4.8% |
| OpenAI ViT-L/14 | 69.2% | 73.4% | +4.2% |
| MetaCLIP ViT-L/14 | 71.0% | 74.3% | +3.3% |

### Ablation Study

- **Missing components of the meta-prompt** (EuroSAT, ViT-B/16): Performance drops to 46.7% without the dataset name, and to 42.0% without the metadata, compared to 55.6% for the complete MPVR.
- **Single-stage vs. Two-stage**: Two-stage MPVR (55.6%) outperforms the single-stage variant (51.2%) and template-only baseline (47.2%).
- **Text Source Ensembling**: Ensembling the embeddings of GPT and Mixtral via averaging achieves the best performance (67.0% on ViT-B/16).
- **Comparison with MMLMs**: CLIP ViT-B/32 (57.2%) significantly outperforms LLaVA-1.6-7B (30.0%), validating the superiority of dual-encoders in discriminative recognition tasks.

### Scale of Corpus

Using MPVR, approximately 2.5 million unique class descriptions were generated from GPT and Mixtral, forming the first large-scale LLM visual knowledge corpus.

## Highlights & Insights

1. **Fully Automated**: Human input is restricted to a brief description of the task and a list of classes, requiring no manual design of LLM queries.
2. **Elegant Two-Stage Strategy**: It first generates task-aware query templates and then generates class descriptions, magnifying diversity in a cascading manner.
3. **Effective with Open-Source Models**: It demonstrates for the first time that descriptions generated by open-source LLMs (Mixtral) can effectively enhance the zero-shot capabilities of VLMs, achieving performance close to GPT.
4. **Broad Generalization**: Consistent improvements are achieved across 20 cross-domain datasets and multiple VLM backbones.
5. **Open-Source Large-Scale Corpus**: Releases 2.5 million class descriptions, which can be directly reused.

## Limitations & Future Work

1. **Dependence on Text Quality**: LLM-generated descriptions may contain inaccurate visual details, and there is currently no quality filtering mechanism in place.
2. **Computational Overhead**: Querying the LLM in two stages requires a large number of prompts per class, resulting in significant text encoding overhead during inference.
3. **Challenge with Stanford Cars**: The overall benefit of prompt ensembling is limited on this dataset, suggesting that certain fine-grained tasks still require explicit visual cues.
4. **Fixed In-Context Examples**: The same exemplar dataset is used throughout, without investigating strategies for dynamically selecting the best exemplars.
5. **Limited to Classification**: The proposed method is restricted to zero-shot classification and has not been extended to broader visual tasks such as detection and segmentation.

## Related Work & Insights

| Method | Requires Manual LLM Queries | Supports Open-Source LLMs | 20 Dataset Average (ViT-B/32) |
|------|---------------------|-------------|------------------------|
| CLIP (DS-TEMP) | Yes (template) | - | 59.7% |
| CUPL | Yes (per dataset) | No | ~60% |
| DCLIP | Yes (attribute queries) | No | ~59% |
| Waffle+Con+GPT | Yes (concept + random) | No | ~61% |
| **MPVR (GPT)** | **No** | No | **65.0%** |
| **MPVR (Mixtral)** | **No** | **Yes** | **63.8%** |

The core difference is that MPVR simplifies manual design from "writing LLM queries for each dataset" to "providing dataset descriptions," while obtaining richer and more diverse prompts through its two-stage strategy.

## Related Work & Insights

- The value of the **meta-prompting paradigm** is not limited to visual recognition; it can be generalized to any scenario requiring LLM-generated structured outputs.
- The results indicate that the CLIP text encoder responds better to semantically rich descriptions, suggesting that the text understanding capabilities of VLMs are underestimated by simple templates.
- The two-stage "coarse-to-fine" knowledge extraction approach can be transferred to other LLM-assisted tasks.
- The small performance gap between open-source and closed-source LLMs holds significant implications for practical deployment.

## Rating

- Novelty: ⭐⭐⭐⭐ — The two-stage meta-prompting automation framework is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 20 datasets, multiple VLM/LLM backbones, and extensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure and intuitive diagrams.
- Value: ⭐⭐⭐⭐ — Highly practical; both the corpus and framework can be directly reused.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Beyond Heuristic Prompting: A Concept-Guided Bayesian Framework for Zero-Shot Image Recognition](../../CVPR2026/multimodal_vlm/beyond_heuristic_prompting_a_concept-guided_bayesian_framework_for_zero-shot_ima.md)
- [\[ECCV 2024\] Elevating All Zero-Shot Sketch-Based Image Retrieval Through Multimodal Prompt Learning](elevating_all_zero-shot_sketch-based_image_retrieval_through_multimodal_prompt_l.md)
- [\[ECCV 2024\] Zero-shot Object Counting with Good Exemplars (VA-Count)](zero-shot_object_counting_with_good_exemplars.md)
- [\[ICLR 2026\] Meta-Adaptive Prompt Distillation for Few-Shot Visual Question Answering](../../ICLR2026/multimodal_vlm/meta-adaptive_prompt_distillation_for_few-shot_visual_question_answering.md)
- [\[ICCV 2025\] Synergistic Prompting for Robust Visual Recognition with Missing Modalities](../../ICCV2025/multimodal_vlm/synergistic_prompting_for_robust_visual_recognition_with_missing_modalities.md)

</div>

<!-- RELATED:END -->
