---
title: >-
  [Paper Note] CURE: Cultural Gaps in the Long Tail of Text-to-Image Systems
description: >-
  [ICCV 2025][Image Generation][Text-to-image generation] This work introduces the CURE benchmark and scoring suite, which employs **Marginal Information Attribution (MIA)** of attribute specifications as a proxy for human…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Text-to-image generation"
  - "cultural representation"
  - "benchmark evaluation"
  - "marginal information attribution"
  - "long-tail bias"
date: 2026-05-08
content_hash: b888d71a04f7e14b
---

# CURE: Cultural Gaps in the Long Tail of Text-to-Image Systems

**Conference**: ICCV 2025
**arXiv**: [2506.08071](https://arxiv.org/abs/2506.08071)  
**Code**: [https://aniketrege.github.io/cure/](https://aniketrege.github.io/cure/)  
**Area**: Image Generation / T2I Fairness
**Keywords**: Text-to-image generation, cultural representation, benchmark evaluation, marginal information attribution, long-tail bias

## TL;DR

This work introduces the CURE benchmark and scoring suite, which employs **Marginal Information Attribution (MIA)** of attribute specifications as a proxy for human judgment to systematically evaluate the representational capacity of T2I systems across the global cultural long tail.

## Background & Motivation

Mainstream T2I systems (e.g., Stable Diffusion, FLUX, DALL-E 3) are trained on web-crawled data with distributions heavily skewed toward Western and European cultures, resulting in poor generation quality for artifacts from Global South cultures (hallucinations, detail errors). Existing cultural bias evaluation methods suffer from three core issues:

**Human evaluation does not scale**: Large-scale user studies are costly and difficult to reproduce.

**Proxy scorers correlate poorly with human judgment**: Conventional metrics based on CLIP similarity or reference image comparison fail to accurately reflect cultural representativeness.

**Generative Entanglement**: Scorers share pre-training data (e.g., LAION-2B) with T2I systems, causing evaluation results to be inflated and misleading.

The authors use pottery generation as an illustrative example: generating "ceramic diyas" (India) yields good results, whereas "jebena" (Ethiopia) and "amphora of Hermonax" (Greece) are severely distorted. This phenomenon is essentially a direct reflection of the long-tail distribution in training data.

## Method

### Overall Architecture

CURE comprises two core components — a **dataset** and a **scorer suite** — and evaluates the degree to which T2I systems have internalized cultural knowledge by progressively enriching prompt specifications.

### Key Designs

#### 1. CURE Dataset Construction

A hierarchical category taxonomy is automatically constructed from the Wikimedia knowledge graph:
- **6 cultural axes** ($s$): food, art, fashion, architecture, celebrations, people
- **32 cultural categories** ($c$): e.g., dumpling, flatbread, pottery
- **300 cultural artifacts** ($n$): e.g., banku (Ghanaian dumpling), modak (Indian sweet)
- **64 countries/regions** ($r$)

The attribute hierarchy follows $s \to c \to n, r$, and democratic, scalable dataset construction is achieved by systematically traversing parent–child nodes in Wikimedia.

#### 2. Marginal Information Attribution (MIA) Scorer

The core assumption is: if a T2I system has sufficiently learned knowledge of a cultural artifact, **adding attribute information to the prompt** (e.g., specifying category and region in addition to the artifact name) should not significantly alter generation quality.

This is operationalized through three scorer types:

**Perceptual Similarity (PS) Scorer**:
$$\phi_{PS}(n) = sim(I(n), I(c))$$
Measures the similarity between image $I(n)$ generated from the artifact name alone and image $I(c)$ generated from the category. High similarity indicates that the T2I system has learned the cultural association $n \to c$.

**Image–Text Alignment (ITA) Scorer**:
$$\phi_{ITA}(a) = \frac{sim(I(n), P(n)) + sim(I(n), P(a))}{2}$$
Jointly evaluates visual correctness and cultural alignment, avoiding the inefficiency of directly querying region relevance.

**Diversity (DIV) Scorer**:
$$\phi_{DIV} = LPIPS(n, \{n,c\}, \{n,r\}, \{n,c,r\})$$
Computes pairwise LPIPS distances among images generated under different attribute specifications, measuring the effect of additional information on diversity.

#### 3. User Study Design

Crowdworkers matched by nationality are recruited on Prolific to rate three Likert-scale items: cultural representativeness, perceptual similarity, and category likelihood. A key innovation is **requiring workers to explicitly identify with their national culture**, rather than simply assuming cultural affiliation.

### Loss & Training

This paper is an evaluation work and does not involve model training. The primary evaluation metric is the Spearman rank correlation coefficient $\rho$, which measures the monotonic relationship between automatic scorers and human gold-standard judgments.

## Key Experimental Results

### Main Results (Spearman Correlation between PS Scorer and Human Judgment)

| Encoder | Scorer | FLUX.1 $\phi^*_{CURE}$ | FLUX.1 $\phi^*_{PS}$ | SD 3.5 $\phi^*_{CURE}$ | SD 3.5 $\phi^*_{PS}$ |
|--------|--------|:---:|:---:|:---:|:---:|
| SigLIP 2 | $\phi_{GT}(n)$↑ | 0.25 | 0.44 | 0.27 | 0.45 |
| SigLIP 2 | $\phi_{PS}(n)$↑ | 0.18 | 0.32 | 0.22 | 0.38 |
| SigLIP 2 | $\Delta\phi_{PS}(\{n,c\})$↓ | -0.16 | -0.31 | -0.21 | -0.37 |
| DINOv2 | $\phi_{GT}(n)$↑ | 0.17 | 0.40 | 0.25 | 0.46 |
| DINOv2 | $\Delta\phi_{PS}(\{n,c\})$↓ | -0.19 | -0.32 | -0.21 | -0.35 |

### Ablation Study (ITA Scorer Comparison, SigLIP 2 Backbone)

| Scorer | FLUX.1 $\phi^*_{CURE}$ | FLUX.1 $\phi^*_{GT}$ | SD 3.5 $\phi^*_{CURE}$ | SD 3.5 $\phi^*_{GT}$ |
|--------|:---:|:---:|:---:|:---:|
| Khanuja et al. | 0.13 | 0.08 | 0.05 | 0.04 |
| sim(I(n), P(n)) | 0.24 | 0.35 | 0.18 | 0.31 |
| $\phi_{ITA}(\{c,r\})$ (Ours) | **0.27** | **0.38** | **0.23** | **0.34** |
| PickScore | 0.20 | 0.29 | 0.23 | 0.37 |
| Gemini 2.0 Flash | 0.23 | 0.41 | 0.27 | 0.37 |

### Key Findings

1. **MIA scorers, without requiring ground-truth images**, approach or match the performance of baselines that rely on GT images.
2. The highest correlation between any quantitative scorer and human judgment is only $\rho=0.51$, indicating that **current visual encoders remain far from sufficient as substitutes for human cultural judgment**.
3. Higher-quality T2I systems (higher ELO) exhibit greater diversity but lower cultural accuracy, revealing a **factuality–diversity trade-off**.
4. Gemini 2.0 Flash performs reasonably well as an MLLM judge but **hallucinates details of Global South cultures**.

## Highlights & Insights

- **Elegant dataset design**: The hierarchical structure of the Wikimedia knowledge graph enables automated construction, and new cultural categories can be added by anyone.
- **Core insight of MIA**: Rather than directly assessing generation quality, the approach observes how much quality changes when more information is provided — an elegant indirect evaluation strategy.
- The work exposes the **generative entanglement** problem: evaluating with VLMs that share training data with T2I systems systematically overestimates performance.
- The benchmark covers 64 countries, 6 T2I systems, and multiple encoders/VLMs, constituting a rigorous experimental scope.

## Limitations & Future Work

- Using geography (country) as a proxy for culture is overly coarse and does not account for dimensions such as religion or language.
- The PS scorer performs poorly on lower-quality T2I systems (e.g., SD 1.5) and depends on the coverage quality of the T2I system itself.
- The approach cannot handle semantic ambiguity (e.g., "damper" refers to both an Australian bread and a mechanical device).
- The user study employs English-only questionnaires, which may introduce linguistic bias.

## Related Work & Insights

- CURE complements cultural evaluation works such as CulturalBench and CCC-Bench, but is the first to systematically propose a scoring framework grounded in marginal utility.
- The MIA paradigm is generalizable to cultural bias evaluation in other generative modalities (video, 3D).
- The work inspires a general evaluation paradigm of **indirectly assessing model knowledge through changes in information content**.

## Rating

- Novelty: ⭐⭐⭐⭐ (The MIA scoring framework is novel and theoretically grounded)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (6 T2I systems + large-scale user study + multi-dimensional ablations)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, rich figures and tables)
- Value: ⭐⭐⭐⭐ (Provides a practical evaluation tool for T2I cultural fairness)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Fix-CLIP: Dual-Branch Hierarchical Contrastive Learning via Synthetic Captions for Better Understanding of Long Text](fix-clip_dual-branch_hierarchical_contrastive_learning_via_synthetic_captions_fo.md)
- [\[ICCV 2025\] CoMPaSS: Enhancing Spatial Understanding in Text-to-Image Diffusion Models](compass_enhancing_spatial_understanding_in_text-to-image_diffusion_models.md)
- [\[ICCV 2025\] InfiniDreamer: Arbitrarily Long Human Motion Generation via Segment Score Distillation](infinidreamer_arbitrarily_long_human_motion_generation_via_segment_score_distill.md)
- [\[AAAI 2026\] LongT2IBench: A Benchmark for Evaluating Long Text-to-Image Generation with Graph-structured Annotations](../../AAAI2026/image_generation/longt2ibench_a_benchmark_for_evaluating_long_text-to-image_generation_with_graph.md)
- [\[ICCV 2025\] Generating Multi-Image Synthetic Data for Text-to-Image Customization](generating_multi-image_synthetic_data_for_text-to-image_customization.md)

</div>

<!-- RELATED:END -->
