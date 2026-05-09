---
title: >-
  [Paper Note] AutoOpt: A Dataset and a Unified Framework for Automating Optimization Problem Solving
description: >-
  [NeurIPS 2025][Optimization][Image OCR] AutoOpt introduces the first end-to-end framework for converting optimization problem images to executable code — comprising the AutoOpt-11k dataset of 11,554 optimization formula images (handwritten + printed), an M1 hybrid encoder (ResNet+Swin→mBART) for image-to-LaTeX conversion (BLEU 96.70), an M2 DeepSeek-Coder module for LaTeX-to-PYOMO translation, and an M3 bilevel decomposition solver, achieving an overall pipeline success rate of 94.20%.
tags:
  - NeurIPS 2025
  - Optimization
  - Image OCR
  - LaTeX-to-PYOMO
  - Bilevel Optimization Solving
  - Mathematical Programming
  - Handwriting Recognition
date: 2026-05-08
content_hash: 856cfbc27b7f7148
---

# AutoOpt: A Dataset and a Unified Framework for Automating Optimization Problem Solving

**Conference**: NeurIPS 2025
**arXiv**: [2510.21436](https://arxiv.org/abs/2510.21436)
**Code**: [https://github.com/Shobhit1201/AutoOpt](https://github.com/Shobhit1201/AutoOpt)
**Area**: Optimization Automation
**Keywords**: Image OCR, LaTeX-to-PYOMO, Bilevel Optimization Solving, Mathematical Programming, Handwriting Recognition

## TL;DR
AutoOpt introduces the first end-to-end framework for converting optimization problem images to executable code — comprising the AutoOpt-11k dataset of 11,554 optimization formula images (handwritten + printed), an M1 hybrid encoder (ResNet+Swin→mBART) for image-to-LaTeX conversion (BLEU 96.70), an M2 DeepSeek-Coder module for LaTeX-to-PYOMO translation, and an M3 bilevel decomposition solver, achieving an overall pipeline success rate of 94.20%.

## Background & Motivation

**Background**: Mathematical optimization formulas frequently appear as images — on whiteboards, in scanned papers, or in handwritten notes. While OCR and LLMs perform well on general documents, recognition of mathematical formulas remains challenging.

**Limitations of Prior Work**: (a) No dedicated dataset exists for optimization problem images — existing OCR datasets do not cover the specific structure of objective functions and constraints; (b) The pipeline from LaTeX recognition to executable code is broken — translating to PYOMO requires understanding optimization semantics; (c) No unified automated solving method exists for complex non-convex or multi-level optimization.

**Key Challenge**: Transforming an optimization problem from image to solution requires three distinct capabilities — visual recognition, semantic understanding, and solving algorithms — yet no prior system addresses all three simultaneously.

**Goal**: Construct a complete automated pipeline: image → LaTeX → PYOMO → solution.

**Key Insight**: Address the problem through three dedicated modules — M1 uses a CNN+Transformer hybrid encoder for image→LaTeX; M2 uses fine-tuned DeepSeek-Coder for LaTeX→PYOMO; M3 uses bilevel decomposition for complex optimization.

**Core Idea**: AutoOpt-11k dataset + ResNet-Swin-mBART image→LaTeX + DeepSeek-Coder LaTeX→PYOMO + bilevel optimization decomposition = a fully automated pipeline from image to optimization solution.

## Method

### Overall Architecture
Optimization formula image → **M1**: ResNet-101 + Swin Transformer hybrid encoding → mBART decoding to generate LaTeX → **M2**: Fine-tuned DeepSeek-Coder-1.3B translating LaTeX to PYOMO code → **M3**: BOBD bilevel decomposition solver

### Key Designs

1. **M1 Hybrid Encoder (ResNet+Swin→mBART)**:

    - **Function**: Convert optimization formula images to LaTeX.
    - **Mechanism**: ResNet-101 extracts local features $\mathbf{f}_{ResNet} = \alpha \cdot \text{LN}(\text{Proj}(F))$, which are prepended to the patch embeddings of a Swin Transformer; mBART then decodes the resulting representations to generate LaTeX. Transfer learning is initialized from NOUGAT weights.
    - **Design Motivation**: CNNs capture local stroke-level features while Swin Transformers capture global structure — the two are complementary. Ablations show CNN+Transformer (DL3) achieves BLEU 96.70 vs. Transformer-only (DL2) at 95.51 and CNN-only (DL1) at 16.10.

2. **M2 LaTeX→PYOMO (DeepSeek-Coder)**:

    - **Function**: Translate LaTeX mathematical formulas into executable PYOMO optimization code.
    - **Mechanism**: DeepSeek-Coder-1.3B is fine-tuned on 80% of 1,018 mathematical programs, achieving BLEU 88.25.
    - **Design Motivation**: LaTeX is a mathematical representation whereas PYOMO is a programming interface — the mapping requires understanding variable declarations, objective functions, and constraint structures.

3. **M3 BOBD Bilevel Decomposition**:

    - **Function**: Handle complex non-convex and multi-level optimization problems.
    - **Mechanism**: An ML classifier assigns variables to upper or lower levels → a genetic algorithm optimizes non-differentiable upper-level variables → CVX solves the convex lower-level subproblem.
    - **Design Motivation**: Provides unified handling of linear/nonlinear, convex/non-convex, multi-objective, and stochastic optimization problems.

### Loss & Training
- M1: Transfer learning from NOUGAT; image resolution 768×1024 with contrast enhancement.
- M2: Fine-tuning of DeepSeek-Coder.
- M3: Genetic algorithm + CVX solver.

## Key Experimental Results

### Main Results

| Module | Metric | AutoOpt | Nougat | GPT-4o | Gemini 2.0 |
|--------|--------|---------|--------|--------|-----------|
| M1 | BLEU | **96.70** | 95.51 | — | — |
| M1 | CER | **0.0286** | 0.0440 | 0.1017 | 0.1338 |
| M2 | BLEU | **88.25** | — | — | — |
| Pipeline | Success Rate | **94.20%** | — | — | — |

### Ablation Study

| M1 Architecture | BLEU | CER |
|-----------------|------|-----|
| DL1 (CNN only) | 16.10 | 0.8812 |
| DL2 (Transformer only) | 95.51 | 0.0440 |
| **DL3 (CNN+Transformer)** | **96.70** | **0.0286** |

### Key Findings
- The hybrid encoder outperforms the pure Transformer by 1.2 BLEU and achieves a 35% CER reduction — CNN features are critical for mathematical symbol recognition.
- Handwritten CER is 0.0412 vs. 0.0176 for printed text — handwriting remains the primary challenge.
- The estimated lower bound of overall pipeline reliability is 89.12%, derived from the product of M1 and M2 error rates.
- The framework achieves a 94.20% success rate on 500 out-of-distribution test cases, demonstrating strong generalization.

## Highlights & Insights
- The **11,554-image dataset** (including 5,070 handwritten samples) constitutes the first OCR benchmark dedicated to optimization problems.
- The **end-to-end pipeline** from image to solution is the first fully automated system of its kind in the optimization community.
- The advantage of **CNN+Transformer hybrid** architecture for mathematical OCR is rigorously quantified.

## Limitations & Future Work
- Only single-page formulas are handled — multi-page and cross-page cases remain unaddressed.
- Large-scale optimization problems (billions of variables) have not been tested.
- Inter-annotator consistency varies (BLEU 0.82–0.86).
- Ambiguous or incomplete optimization problem definitions are not evaluated.

## Related Work & Insights
- **vs. NOUGAT**: NOUGAT targets general document OCR; AutoOpt specializes in mathematical optimization recognition.
- **vs. GPT-4o/Gemini**: General-purpose VLMs exhibit 3–5× higher CER on mathematical OCR tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ First complete pipeline from image to optimization solution.
- Experimental Thoroughness: ⭐⭐⭐⭐ 11K dataset + ablations + OOD testing.
- Writing Quality: ⭐⭐⭐⭐ Pipeline description is clear and well-structured.
- Value: ⭐⭐⭐⭐ An important foundational contribution to automated optimization problem solving.

### Additional Method Notes
- **Dataset composition**: Among 11,554 images, 5,070 are handwritten and 6,484 are printed; 10,838 are single-objective, 159 multi-objective, 399 multi-level, and 158 stochastic; 2,130 are linear and 9,122 nonlinear; 2,580 are convex and 3,574 non-convex — providing broad coverage.
- **M1 image preprocessing**: 768×1024 center padding + contrast enhancement + sharpening filter, specifically targeting low-contrast and blurry handwritten formulas.
- **Effect of NOUGAT transfer learning**: Fine-tuning from general document OCR pretrained weights to the mathematical formula domain reduces CER from 0.044 to 0.029 — transfer learning is critical.
- **Pipeline reliability estimation**: $(1-\text{CER}_{M1}) \times (1-\text{CER}_{M2}) = (1-0.0286) \times (1-0.0825) = 89.12\%$ serves as the lower bound.
- **Distinction from Wolfram Alpha/Mathematica**: These tools require text input; AutoOpt solves problems end-to-end directly from images.

- **Detailed Rating Remarks**: AutoOpt pioneers a complete workflow from optimization problem OCR to solution, with both data and code publicly released; however, it is currently limited to single-page formulas and moderate-scale problem instances.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Problem-Parameter-Free Decentralized Bilevel Optimization](problem-parameter-free_decentralized_bilevel_optimization.md)
- [\[AAAI 2026\] GHOST: Solving the Traveling Salesman Problem on Graphs of Convex Sets](../../AAAI2026/optimization/ghost_solving_the_traveling_salesman_problem_on_graphs_of_convex_sets.md)
- [\[NeurIPS 2025\] A Unified Approach to Submodular Maximization Under Noise](a_unified_approach_to_submodular_maximization_under_noise.md)
- [\[NeurIPS 2025\] VERA: Variational Inference Framework for Jailbreaking Large Language Models](vera_variational_inference_framework_for_jailbreaking_large_language_models.md)
- [\[ICLR 2026\] CogFlow: Bridging Perception and Reasoning through Knowledge Internalization for Visual Mathematical Problem Solving](../../ICLR2026/optimization/cogflow_bridging_perception_and_reasoning_through_knowledge_internalization_for_.md)

</div>

<!-- RELATED:END -->
