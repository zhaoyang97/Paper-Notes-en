---
title: >-
  [Paper Note] ASCIIBench: Evaluating Language-Model-Based Understanding of Visually-Oriented Text
description: >-
  [NeurIPS 2025][AIGC Detection][ASCII art] This paper introduces ASCIIBench, the first publicly available benchmark for ASCII art understanding and generation (5,315 images, 752 categories). Systematic evaluation reveals that the visual modality substantially outperforms the text modality, multimodal fusion yields no benefit, and CLIP exhibits a fundamental bottleneck in representing ASCII structure—only categories with high intra-class consistency can be effectively distingui…
tags:
  - "NeurIPS 2025"
  - "AIGC Detection"
  - "ASCII art"
  - "LLM evaluation"
  - "spatial reasoning"
  - "CLIP"
  - "multimodal fusion"
date: 2026-05-08
content_hash: fd8072628e1ff246
---

# ASCIIBench: Evaluating Language-Model-Based Understanding of Visually-Oriented Text

**Conference**: NeurIPS 2025
**arXiv**: [2512.04125](https://arxiv.org/abs/2512.04125)  
**Code**: [https://github.com/ASCIIBench/ASCIIBench](https://github.com/ASCIIBench/ASCIIBench)  
**Area**: AIGC Detection
**Keywords**: ASCII art, LLM evaluation, spatial reasoning, CLIP, multimodal fusion

## TL;DR

This paper introduces ASCIIBench, the first publicly available benchmark for ASCII art understanding and generation (5,315 images, 752 categories). Systematic evaluation reveals that the visual modality substantially outperforms the text modality, multimodal fusion yields no benefit, and CLIP exhibits a fundamental bottleneck in representing ASCII structure—only categories with high intra-class consistency can be effectively distinguished.

## Background & Motivation

**Background**: Large language models exhibit emergent capabilities such as reasoning and fluent text generation as scale increases; GPT-4 can even generate and edit TikZ drawings. Nevertheless, these models continue to struggle with tasks requiring precise spatial and positional reasoning.

**Limitations of Prior Work**: No standardized benchmark exists specifically for evaluating LLMs' spatial understanding. Although BIG-bench includes an ASCII word recognition task and ASCIIEval has conducted similar explorations, these resources are either limited in scope or not publicly released. ASCII art occupies a unique intersection of text and vision, naturally occurring within LLM pretraining distributions and natively aligned with tokenization schemes, making it an ideal evaluation vehicle that requires no additional adaptation.

**Key Challenge**: In ASCII art, characters serve as "visual primitives" rather than semantic tokens and demand strict structural regularity (akin to tabular data), which fundamentally conflicts with LLMs' semantic processing nature. Models must comprehend the spatial arrangement of characters in two-dimensional space rather than merely their textual meaning.

**Goal**: (1) Construct a high-quality, publicly available ASCII art benchmark dataset; (2) systematically evaluate diverse LLMs and multimodal models along both classification and generation dimensions; (3) analyze the applicability of existing evaluation metrics—particularly CLIP—to the ASCII domain.

**Key Insight**: ASCII art is selected as a distinctive "symbolic visual modality" that is simultaneously amenable to text processing and demands visual-spatial understanding, thereby probing the limits of both text-only and multimodal models.

**Core Idea**: ASCII art serves as a stress test for LLMs' spatial reasoning and multimodal representation capabilities.

## Method

### Overall Architecture

ASCIIBench evaluates models along two dimensions: (1) **Classification**—the model is presented with an ASCII image and four category options to test comprehension; (2) **Generation**—the model generates ASCII images for a specified category, with quality assessed via CLIP embeddings. Each dimension has its own preprocessing pipeline, prompting strategy, and evaluation metrics.

### Key Designs

1. **Dataset Construction and Cleaning Pipeline**:

    - **Function**: Construct a high-quality benchmark from raw ASCII art data.
    - **Mechanism**: Raw data collected from ascii.co.uk is processed through a rigorous 11-step automated cleaning pipeline to remove noise such as signatures, tags, dates, and email addresses, followed by multi-stage human review by three annotators under unified standards with strict inter-annotator agreement requirements. The conservative filtering process removed over 13,000 low-quality images and 1,800 ambiguous categories, yielding 5,315 high-quality ASCII images across 752 well-defined categories.
    - **Design Motivation**: Raw ASCII art contains substantial noise (creator signatures, Unicode control characters, etc.) that would severely compromise evaluation fairness if used directly.

2. **Multimodal Classification Evaluation Framework**:

    - **Function**: Systematically compare classification performance across text, visual, and text+visual modalities.
    - **Mechanism**: ASCII images are preprocessed according to input modality—text modality receives raw character text; visual modality renders images onto a white background using a black monospaced font (DejaVu Sans Mono) before input; text+visual modality provides both simultaneously. A four-choice prompt format is applied to each sample. Models evaluated include LLaMA 3-8B, GPT-4o, GPT-4o-mini, and Claude 3.5 Sonnet, measured by macro/micro accuracy.
    - **Design Motivation**: Controlling the input modality enables precise identification of model bottlenecks—whether in text understanding, visual perception, or multimodal fusion.

3. **CLIP Embedding Generation Evaluation and Fine-Tuning**:

    - **Function**: Evaluate the fidelity of ASCII images generated by LLMs.
    - **Mechanism**: GPT-3.5/4/4o generates 5 ASCII images per category; these are rendered and CLIP embeddings are extracted, with cosine similarity to reference image embeddings used as a quality measure. Alignment (intra-class compactness) and uniformity (embedding space dispersion) further characterize representation quality. To capture ASCII-specific structure, CLIP is fine-tuned with triplet loss, improving alignment from 5.85 to 8.90 and yielding improvements in uniformity as well.
    - **Design Motivation**: An image-to-image metric is needed that captures both visual and textual features of ASCII art; CLIP's cross-modal pretraining makes it a natural candidate.

### Loss & Training

CLIP fine-tuning employs triplet loss, with positive pairs drawn from same-category ASCII images and negative pairs from different categories, aiming to bring same-class embeddings closer while pushing apart embeddings from different classes.

## Key Experimental Results

### Main Results

| Model | Modality | Micro Acc. (%) | Macro Acc. (%) | Pass Rate (%) |
|---|---|---|---|---|
| LLaMA3.1-8B-Inst | T | 34.27 | 31.89 | 91.78 |
| GPT-3.5-turbo | T | 39.05 | 33.54 | 91.34 |
| Claude-3.5-Sonnet | T | 59.55 | 56.98 | 98.54 |
| Claude-3.5-Sonnet | V | 76.40 | 76.92 | 99.08 |
| Claude-3.5-Sonnet | T+V | 76.48 | 76.89 | 99.08 |
| GPT-4o | T | 75.44 | 80.23 | 96.63 |
| **GPT-4o** | **V** | **77.49** | **82.16** | **98.75** |
| GPT-4o | T+V | 76.56 | 79.74 | 98.52 |
| GPT-5-mini | T | 61.60 | 62.39 | 99.38 |
| GPT-5-mini | V | 77.25 | 84.13 | 99.24 |

### Ablation Study

| CLIP Evaluation Configuration | ROC-AUC | Silhouette | Note |
|---|---|---|---|
| Original CLIP (unfiltered) | ~0.55 | -0.46 | Categories nearly indistinguishable |
| Original CLIP (filtered) | 0.83 | — | Significant improvement after filtering inconsistent generations |
| Fine-tuned CLIP (unfiltered) | ~0.641 | — | Only marginal improvement |
| Restricted to high mean-similarity categories | 0.83 | — | CLIP effective only for a subset of categories |

### Key Findings

- **Visual modality consistently outperforms text modality**: Among all models supporting multimodal input, macro accuracy under the V modality exceeds that under the T modality; GPT-4o achieves the highest macro accuracy of 82.16% under V. This indicates that ASCII structure is more readily understood through rendered pixel information.
- **Multimodal fusion degrades performance**: On GPT-4o and GPT-5-mini, T+V accuracy falls below that of V alone, suggesting that current multimodal fusion strategies cannot effectively process the symbolic structural information in ASCII art.
- **CLIP representation bottleneck is the central issue**: ROC-AUC on unfiltered data approaches chance (0.55) and reaches 0.83 after filtering. However, filtering essentially tests on inputs already close to the training distribution—the true bottleneck lies in CLIP's insufficient representational capacity for ASCII structure rather than in generation variance.
- **Non-monospaced font ablation**: Replacing the monospaced font with a proportional font yields virtually no change in accuracy (GPT-5 V+T: 0.7057 → V only: 0.7118), indicating that models rely primarily on OCR-like mechanisms rather than positional-structural reasoning.

## Highlights & Insights

- **Uniquely insightful evaluation perspective**: ASCII art is an overlooked yet highly valuable evaluation domain that exposes LLMs' gap in "understanding spatial layout"—a capability that conventional NLP and CV benchmarks cannot measure, as they are either purely semantic or purely pixel-based.
- **Counterintuitive finding that fusion underperforms single modality**: The result that T+V performance falls below V suggests that current multimodal fusion mechanisms exhibit an interference effect when processing different representations of the same information. This finding can serve as a diagnostic tool for evaluating fusion quality in other multimodal models.
- **Clear dual-bottleneck analysis**: The paper explicitly characterizes the relative magnitude of two bottlenecks—the generation side (inconsistent LLM outputs) and the evaluation side (insufficient CLIP representations)—providing clear direction for future improvement.

## Limitations & Future Work

- **Limited data scale and unbalanced category distribution**: With 5,315 images across 752 categories, per-category sample counts are low and follow a long-tail distribution (the airplane category accounts for 13.3%), leaving many categories with insufficient samples for reliable evaluation.
- **Ethical concerns regarding data provenance**: Data was collected from ascii.co.uk, which provides no explicit license; the authors only state that "standard research practices were followed," leaving copyright issues unresolved.
- **Evaluation restricted to CLIP**: Other image similarity metrics (e.g., SSIM, FID) or specialized metrics better suited to ASCII structure were not explored.
- **Simple classification task design**: The four-choice MCQ format has a random-guess baseline of 25%, and finer-grained understanding tasks (e.g., ASCII editing or completion) are not addressed.
- **Specialized small models not explored**: The paper itself acknowledges in its limitations section that small models specifically designed for ASCII may be more effective than large general-purpose CLIP, but no experiments are conducted.

## Related Work & Insights

- **vs. ASCIIEval (Jia et al., 2024)**: Conducted similar ASCII evaluation but did not release code or data publicly; ASCIIBench is the first publicly available benchmark.
- **vs. BIG-bench**: BIG-bench includes only simple ASCII word recognition; ASCIIBench provides more comprehensive classification and generation evaluation.
- **vs. ArtPrompt jailbreak**: ArtPrompt exploits LLMs' deficiencies in ASCII understanding to mount jailbreak attacks; ASCIIBench systematically quantifies this capability gap from a constructive perspective, and the two works are complementary.

## Rating

- Novelty: ⭐⭐⭐⭐ Fills the gap in multimodal evaluation of ASCII art with a novel perspective.
- Experimental Thoroughness: ⭐⭐⭐ Broad model coverage but limited data scale; lacks in-depth analysis.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with clearly presented findings.
- Value: ⭐⭐⭐ Appropriately scoped as a workshop paper; the diagnostic approach for multimodal fusion quality offers practical reference value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Reasoning Compiler: LLM-Guided Optimizations for Efficient Model Serving](reasoning_compiler_llm-guided_optimizations_for_efficient_model_serving.md)
- [\[NeurIPS 2025\] Synthesizing Performance Constraints for Evaluating and Improving Code Efficiency](synthesizing_performance_constraints_for_evaluating_and_improving_code_efficienc.md)
- [\[ICML 2026\] CORE: Conflict-Oriented Reasoning for General Multimodal Manipulation Detection](../../ICML2026/aigc_detection/core_conflict-oriented_reasoning_for_general_multimodal_manipulation_detection.md)
- [\[ICML 2026\] Generating Robust Portfolios of Optimization Models using Large Language Models](../../ICML2026/aigc_detection/generating_robust_portfolios_of_optimization_models_using_large_language_models.md)
- [\[ICLR 2026\] Learning From Dictionary: Enhancing Robustness of Machine-Generated Text Detection in Zero-Shot Language via Adversarial Training](../../ICLR2026/aigc_detection/learning_from_dictionary_enhancing_robustness_of_machine-generated_text_detectio.md)

</div>

<!-- RELATED:END -->
