---
title: >-
  [Paper Note] Towards Real-World Document Parsing via Realistic Scene Synthesis and Document-Aware Training
description: >-
  [CVPR 2026][Multimodal VLM][document parsing] This paper proposes DocHumming, a data-training co-design framework that constructs the large-scale synthetic dataset DocMix-3M via Realistic Scene Synthesis…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "document parsing"
  - "synthetic data"
  - "progressive training"
  - "structure token weighting"
  - "real-world robustness"
date: 2026-05-08
content_hash: 2b320693972f21b3
---

# Towards Real-World Document Parsing via Realistic Scene Synthesis and Document-Aware Training

**Conference**: CVPR 2026
**arXiv**: [2603.23885](https://arxiv.org/abs/2603.23885)
**Code**: To be released
**Area**: Document Understanding / End-to-End Document Parsing
**Keywords**: document parsing, synthetic data, progressive training, structure token weighting, real-world robustness

## TL;DR

This paper proposes DocHumming, a data-training co-design framework that constructs the large-scale synthetic dataset DocMix-3M via Realistic Scene Synthesis, and introduces a Document-Aware Training Recipe (DATR) combining progressive learning and structure token weighting. On a 1B-parameter MLLM, DocHumming achieves an OmniDocBench Overall score of 93.75, surpassing Qwen3-VL-235B (89.15), with only a 6.72-point degradation under real-world capture conditions (vs. 18–20 points for pipeline-based methods).

## Background & Motivation

**Background**: Document parsing has evolved from traditional modular pipelines (layout analysis → OCR → element parsing) to end-to-end MLLMs that directly map images to structured outputs. Modular methods perform well on digital/scanned documents (e.g., MinerU2.5 achieves 90.67 on OmniDocBench), but end-to-end methods still face significant challenges in real-world scenarios.

**Limitations of Prior Work**: (1) Modular methods rely on accurate layout analysis; under casual capture conditions, layout errors propagate downstream (18–20 point degradation). (2) End-to-end methods produce repetitive content, hallucinations, and structural inconsistencies under real-world capture. (3) Large-scale, high-quality page-level end-to-end training data is lacking (SynthDog has simple layouts; GOT's PDF-to-LaTeX lacks visual diversity).

**Key Challenge**: The end-to-end paradigm is inherently more robust as it requires no explicit layout segmentation, yet its potential is constrained by data scarcity and the absence of structure-aware training strategies.

**Goal**: To unlock the potential of end-to-end document parsing in real-world scenarios through data-training co-design.

**Key Insight**: Simultaneously address both the data bottleneck (large-scale synthesis) and the training bottleneck (structure-aware optimization), rather than targeting either in isolation.

**Core Idea**: Synthesize 3M page-level samples from 576K layout templates and 9M atomic elements, combined with short-to-long progressive training and structure token weighting loss, enabling a 1B model to match the performance of a 235B model.

## Method

### Overall Architecture

Data-training co-design: at the data level, Realistic Scene Synthesis (RSS) generates large-scale, diverse end-to-end parsing data (DocMix-3M); at the training level, the Document-Aware Training Recipe (DATR: progressive learning + structure token weighting) improves structural fidelity and decoding stability. The final model, DocHumming, is trained on InternVL2-1B.

### Key Designs

1. **Realistic Scene Synthesis (RSS)**:

    - **Function**: Synthesize large-scale page-level end-to-end parsing data from atomic elements and layout templates.
    - **Mechanism**: (a) **Atomic Element Repository**: integrates multi-source datasets for table recognition, formula parsing, paragraph understanding, etc.; normalizes formats; rewrites and augments using Qwen2.5-72B (restructuring tables, perturbing formula symbols, creating mixed elements, generating multilingual paragraph groups); renders image–annotation pairs via a LaTeX pipeline. (b) **Layout Template Library**: sourced from public datasets, web mining, and supplementary under-represented styles, comprising 576K+ layout patterns with reading-order annotations. (c) **Compositional Synthesis**: sampled elements are placed into templates under spatial and structural constraints. (d) **Capture Augmentation**: simulates perspective/bending/wrinkling/lighting variation/camera rotation/background environments; approximately 20% of samples undergo augmentation.
    - **Output**: DocMix-3M (~3M high-quality synthetic documents), built from ~9M atomic elements and 576K layout templates.
    - **Design Motivation**: Bottom-up synthesis rather than PDF-to-LaTeX conversion enables fine-grained control over layout diversity and visual conditions.

2. **Document-Aware Training Recipe (DATR)**:

    - **Function**: A training strategy specifically designed for document parsing, addressing the challenges of large context length variation and unstable structured output.
    - **Mechanism**: (a) **Progressive Training Paradigm** — Stage 1 trains on single-element parsing (tables/formulas/paragraphs) with heterogeneous prompts to acquire type-specific capabilities, while expanding the vocabulary with layout structure tokens; Stage 2 uses DocMix-3M as the primary corpus plus 1M Stage-1 samples and 100K human-annotated data, training end-to-end full-document parsing under a unified prompt format. (b) **Structure Token Weighting** — higher loss weights are applied to structured tokens (within `<table>`...`</table>` and similar tags).
    - **Loss formulation**: $L = -\sum_t \alpha_t y_t \log P(x_t|x_{<t})$, where $\alpha_t = \lambda = 4$ for structure tokens and $\alpha_t = 1$ otherwise.
    - **Design Motivation**: Progressive learning avoids convergence instability from direct long-context training; structure token weighting addresses repetition and inconsistency in structured content such as tables.

3. **Wild-OmniDocBench**:

    - **Function**: A document parsing evaluation benchmark under real-world capture conditions.
    - **Mechanism**: All documents in OmniDocBench are manually converted to physically captured forms. (a) Print → physical deformation (folding/bending/crumpling) → photographed under varied lighting. (b) Screen display → photographed (introducing moiré patterns, reflections, and brightness variation).
    - **Design Motivation**: Existing benchmarks evaluate only digital/scanned documents and fail to reflect real-world challenges.

### Loss & Training

Structure token-weighted cross-entropy loss. Stage 1: batch=512, lr=4e-5, 2 epochs; Stage 2: batch=256, lr=2e-5, 2 epochs. Cosine learning rate decay; maximum output length 8192 tokens. Base model: InternVL2-1B, full-parameter fine-tuning. 16× NVIDIA H20 GPUs.

## Key Experimental Results

### Main Results: OmniDocBench Document Parsing

| Type | Method | Params | Overall↑ | TextEdit↓ | FormulaCDM↑ | TableTEDS↑ | ReadOrder↓ |
|------|--------|--------|---------|-----------|-------------|------------|------------|
| Pipeline | PP-StructureV3 | - | 86.73 | 0.073 | 85.79 | 81.68 | 0.073 |
| General MLLM | Qwen2.5-VL-72B | 72B | 87.02 | 0.094 | 88.27 | 82.15 | 0.102 |
| General MLLM | Qwen3-VL-235B | 235B | 89.15 | 0.069 | 88.14 | 86.21 | 0.068 |
| E2E Specialist | dots.ocr | 3B | 88.41 | 0.048 | 83.22 | 86.78 | 0.053 |
| Modular Specialist | MinerU2.5 | 1.2B | 90.67 | 0.047 | 88.46 | 88.22 | 0.044 |
| Modular Specialist | PaddleOCR-VL | 0.9B | 91.93 | 0.039 | 88.67 | 91.01 | 0.043 |
| **E2E Specialist** | **DocHumming** | **1B** | **93.75** | **0.035** | **93.27** | **91.49** | **0.041** |

### Wild-OmniDocBench Real-World Robustness

| Type | Method | Origin | Wild | Degradation↓ | Note |
|------|--------|--------|------|-------------|------|
| General MLLM | Qwen3-VL-235B | 89.15 | 79.69 | -9.46 | Large models also degrade |
| Modular | MonkeyOCR-3B | 88.85 | 70.00 | -18.85 | Layout error propagation |
| Modular | MinerU2.5 | 90.67 | 70.91 | -19.76 | Worst degradation |
| Modular | PaddleOCR-VL | 91.93 | 72.19 | -19.74 | ~20-point degradation |
| E2E | DeepSeek-OCR | 87.01 | 74.23 | -12.78 | Smaller E2E degradation |
| E2E | dots.ocr | 88.41 | 78.01 | -10.40 | Smaller E2E degradation |
| **E2E** | **DocHumming** | **93.75** | **87.03** | **-6.72** | **Minimal degradation** |

### Ablation Study

| # | RSS | Progressive (PTP) | Struct. Weighting (ST) | OmniDoc↑ | Repeat↓ | Wild↑ | Wild Repeat↓ |
|---|-----|-------------------|----------------------|---------|---------|------|-------------|
| 1 | ✗ | ✓ | ✓ | 89.96 | 4.7% | 78.82 | 8.6% |
| 2 | ✓ | ✓ | ✗ | 88.74 | 4.6% | 84.90 | 5.4% |
| 3 | ✓ | ✗ | ✓ | 91.24 | 4.2% | 85.39 | 4.9% |
| 4 | ✓ | ✓ | ✓ | **93.75** | **2.1%** | **87.03** | **4.3%** |

Data scaling curve: DocMix-1M (85.41) → 2M (88.14) → 3M (89.96) → 4M (89.31, approaching saturation). The 3M synthetic dataset outperforms 100K human-annotated data (89.96 vs. 89.26).

### Key Findings

- End-to-end vs. modular methods diverge sharply in real-world conditions: modular methods degrade by 18–20 points vs. only 6.72 points for DocHumming.
- 1B outperforms 235B: with the correct data and training strategy, the 1B model (93.75) surpasses Qwen3-VL-235B (89.15).
- 3M synthetic samples outperform 100K human annotations (89.96 vs. 89.26), but performance saturates at 4M — diversity of the element repository and template pool is the bottleneck.
- Structure token weighting is critical for reducing repetition: removing it raises the repetition rate from 2.1% to 4.6%.
- DocHumming leads comprehensively on multilingual XFUND evaluation (German 85.15, Japanese 87.99, Spanish 84.39).

## Highlights & Insights

- The data-training co-design framework is a generalizable paradigm: jointly optimizing data and training yields substantially better results than addressing either alone.
- Progressive training draws on the short-to-long context curriculum from LLM training — the element→page progression in document parsing closely mirrors the short→long context progression in LLMs.
- The proposed repetition rate metric (consecutive structural pattern repetitions >10 times + reaching maximum output length) is a practical tool for quantifying decoding stability.
- The saturation point (~3M) in data scaling provides practical guidance on the cost-effectiveness of synthetic data generation.

## Limitations & Future Work

- Irregular interleaved layouts (newspapers, posters) remain challenging; reading order and structural boundaries are ambiguous when text blocks are nested or interleaved.
- Ultra-high-resolution pages require downsampling or tiling, which can cause repetition or loss in long tables and dense formulas.
- Data scaling effects saturate beyond 3M samples, fundamentally limited by the diversity of the element repository and template pool.
- Inference efficiency: text-dense pages require ~3s, limiting interactive use.
- The structure token weighting hyperparameter $\lambda=4$ is heuristic; adaptive strategies remain unexplored.

## Related Work & Insights

- **vs. GOT/SmolDocling**: These E2E methods rely on PDF-to-LaTeX data with limited layout diversity; DocHumming achieves layout diversity through bottom-up synthesis.
- **vs. MinerU2.5/PaddleOCR-VL**: Performance is comparable on standard documents, but modular methods degrade more than 3× in wild scenarios — the modular paradigm is inherently fragile under real-world conditions.
- **Training strategy insight**: Structure token weighting is transferable to any structured generation task (code generation, HTML/JSON generation).
- **Data synthesis methodology**: The paradigm of atomic element repository + layout templates + compositional synthesis is transferable to other multimodal data generation settings.

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty** ⭐⭐⭐⭐: The data-training co-design framework is systematic; structure token weighting is simple yet effective.
- **Experimental Thoroughness** ⭐⭐⭐⭐⭐: Three benchmarks (OmniDocBench, Wild, XFUND), complete ablations (RSS/ST/PTP), and data scaling curves.
- **Writing Quality** ⭐⭐⭐⭐: Architecture diagrams are clear; data construction pipeline is detailed; ablation design is rigorous.
- **Value** ⭐⭐⭐⭐: The 1B-surpasses-235B result is compelling; the Wild benchmark fills a gap in evaluation practice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Efficient Document Parsing via Parallel Token Prediction](efficient_document_parsing_via_parallel_token_prediction.md)
- [\[CVPR 2026\] PaddleOCR-VL: Boosting Document Parsing Efficiency and Performance with Coarse-to-Fine Visual Processing](paddleocr_vl_coarse_to_fine_document_parsing.md)
- [\[CVPR 2026\] World-Env: Leveraging World Model as a Virtual Environment for VLA Post-Training](rehearsevla_simulated_post-training_for_vlas_with_physically-consistent_world_mo.md)
- [\[CVPR 2026\] DocSeeker: Structured Visual Reasoning with Evidence Grounding for Long Document Understanding](docseeker_long_document_understanding.md)
- [\[CVPR 2026\] GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training](gtr_turbo_merged_checkpoint_free_teacher.md)

</div>

<!-- RELATED:END -->
