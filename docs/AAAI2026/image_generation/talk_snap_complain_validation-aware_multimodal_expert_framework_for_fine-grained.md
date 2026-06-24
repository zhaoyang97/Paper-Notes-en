---
title: >-
  [Paper Note] Talk, Snap, Complain: Validation-Aware Multimodal Expert Framework for Fine-Grained Customer Grievances
description: >-
  [AAAI2026][Image Generation][Multimodal complaint analysis] This paper proposes VALOR, a validation-aware multimodal expert framework combining a multi-expert routing architecture with Chain-of-Thought reasoning and a semantic alignment validation mechanism, which achieves joint fine-grained classification of complaint Aspect and Severity in multi-turn multimodal customer service dialogues, yielding absolute improvements of 12.94%/6.51% over the strongest baseline Gemma-3.
tags:
  - "AAAI2026"
  - "Image Generation"
  - "Multimodal complaint analysis"
  - "Mixture-of-Experts"
  - "Chain-of-Thought"
  - "semantic alignment"
  - "fine-grained classification"
date: 2026-05-08
content_hash: a37c533edde51178
---

# Talk, Snap, Complain: Validation-Aware Multimodal Expert Framework for Fine-Grained Customer Grievances

**Conference**: AAAI2026  
**arXiv**: [2511.14693](https://arxiv.org/abs/2511.14693)  
**Authors**: Rishu Kumar Singh, Navneet Shreya, Sarmistha Das, Apoorva Singh, Sriparna Saha  
**Code**: [GitHub](https://github.com/sarmistha-D/VALOR)  
**Area**: Image Generation  
**Keywords**: Multimodal complaint analysis, Mixture-of-Experts, Chain-of-Thought, semantic alignment, fine-grained classification  

## TL;DR

This paper proposes VALOR, a validation-aware multimodal expert framework combining a multi-expert routing architecture with Chain-of-Thought reasoning and a semantic alignment validation mechanism, which achieves joint fine-grained classification of complaint Aspect and Severity in multi-turn multimodal customer service dialogues, yielding absolute improvements of 12.94%/6.51% over the strongest baseline Gemma-3.

## Background & Motivation

### State of the Field
Existing complaint analysis research primarily relies on single-modal short text (tweets, product reviews), whereas real-world customer service scenarios typically involve users providing both textual complaints and visual evidence (screenshots, product photos), with complaint information distributed across multi-turn dialogues. Traditional ABSA methods only assign sentiment polarity and cannot provide actionable fine-grained insights.

### Limitations of Prior Work
- Most methods handle only **single-turn short text**, lacking multi-turn context modeling capability
- Multimodal complaint methods rely on static features or simple fusion, ignoring **cross-modal interaction**
- A dedicated **multimodal dialogue complaint dataset** is absent, limiting evaluation to product reviews
- Existing LLMs and VLMs are not optimized for modality alignment, ambiguity resolution, or cross-modal reasoning

### Root Cause
The paper reframes complaint analysis as a fine-grained multimodal classification task over multi-turn dialogues, jointly modeling dialogue flow and image information to achieve precise classification of Aspect categories and Severity levels.

## Core Problem

1. How can textual and visual cues be effectively fused across multi-turn customer service dialogues for fine-grained complaint understanding?
2. How should an expert routing mechanism be designed to ensure reasoning quality and interpretability in complex multimodal scenarios?
3. How can a multimodal customer service dialogue dataset be constructed and annotated to support systematic evaluation?

## Method

### CIViL Dataset Construction
Apple Support dialogues (2–10 turns) were filtered from the Kaggle Customer Support on Twitter dataset. A random sample of 2,004 dialogues was annotated with fine-grained Aspect (6 categories) and Severity (4 levels). A total of 4,478 visual evidence images were assigned to dialogues via CLIP semantic matching. Fleiss' Kappa: Aspect = 0.68, Severity = 0.75.

### VALOR Framework (Phase 1: Prediction)

**Encoders**: Text is encoded by BERT-base to obtain $\mathbf{H}_t \in \mathbb{R}^{B \times L \times d}$; images are encoded by ViT-patch16 to obtain $\mathbf{H}_i \in \mathbb{R}^{B \times 196 \times d}$ ($d=768$).

**Cross-modal Fusion**: 8-head Cross-Attention, with queries from text and keys/values from images:

$$\text{Attention}(\mathbf{Q}_h, \mathbf{K}_h, \mathbf{V}_h) = \text{softmax}\left(\frac{\mathbf{Q}_h \mathbf{K}_h^\top}{\sqrt{d/H}}\right)\mathbf{V}_h$$

The output is mean-pooled to obtain a unified representation $\mathbf{x} \in \mathbb{R}^{B \times d}$.

**Semantic Alignment Score (SAS)**: $\mathbf{h}_t$ and $\mathbf{h}_i$ are projected into a shared 512-dimensional space; an MLP with tanh outputs a scalar $s \in [-1,1]^B$.

**CoT Expert Routing**: $\mathcal{K}=4$ Chain-of-Thought experts based on DeepSeek-6.7B. Gating function:

$$\mathbf{g} = \text{softmax}(\mathbf{x}\mathbf{W}_r + \mathbf{b}_r) \in \mathbb{R}^{B \times \mathcal{K}}$$

Hard top-1 routing selects $k_b^* = \arg\max_k g_{b,k}$, with load-balancing regularization:

$$L_{\text{lb}} = \sum_{k=1}^{\mathcal{K}}\left(\frac{1}{\mathcal{K}} - \frac{1}{B}\sum_{b=1}^{B}g_{b,k}\right)^2$$

### VALOR Framework (Phase 2: Validation)

$\mathcal{L}_v=2$ DeepSeek validation experts perform secondary reasoning, evaluated via three metrics:
- **Alignment**: cosine similarity of logits across experts, $R_{\text{avg}}$
- **Dominance**: correlation between MoE output and validation output
- **Complementarity**: entropy $U_{\text{avg}}$ of softmax-normalized logits

A meta-fusion network aggregates all signals; the final prediction is adjusted by SAS:

$$\ell_{\text{final}} = \ell_f + \lambda_s \cdot s \cdot \mathbf{1}_{\mathcal{C}_a}, \quad \lambda_s = 0.1$$

### Overall Training Objective

$$L_{\text{total}} = L_{\text{aspect}} + L_{\text{severity}} + \lambda_{\text{lb}}L_{\text{lb}} + \lambda_{\text{val}}L_{\text{val}} + \lambda_s L_{\text{sas}} + \lambda_R L_{\text{Alignment}} + \lambda_S L_{\text{dominance}} + \lambda_U L_{\text{complementarity}}$$

## Key Experimental Results

### Baseline Comparison (CIViL dataset, 20 epochs fine-tuning)

| Model | ACD Acc | ACD F1 | SD Acc | SD F1 |
|------|---------|--------|--------|-------|
| Gemma-3 (9B) | 0.69 | 0.66 | 0.65 | 0.66 |
| DeepSeek-VL | 0.66 | 0.65 | 0.66 | 0.65 |
| Paligemma (3B) | 0.65 | 0.66 | 0.65 | 0.64 |
| CLIP ViT-B/32 | 0.59 | 0.56 | 0.55 | 0.56 |
| ViLT | 0.55 | 0.56 | 0.55 | 0.54 |
| **VALOR** | **0.8194** | **0.7696** | **0.7251** | **0.6791** |

### Key Ablation Results

| Configuration | ACD Acc | SD Acc | ACD F1 | SD F1 |
|------|---------|--------|--------|-------|
| VALOR (full) | 81.94% | 72.51% | 76.96% | 67.91% |
| CoT (w/o Validation) | 73.74% | 62.62% | 70.44% | 52.84% |
| Transformer experts + Validation | 77.08% | 63.98% | 70.24% | 60.24% |
| MLP experts + w/o Validation | 70.43% | 57.35% | 63.82% | 48.55% |

The Validation MoE contributes a **+8.2%** improvement in Aspect accuracy (73.74% → 81.94%).

### Human Evaluation (200 samples, Win-Loss-Draw)
- VALOR vs. Gemma-3: Aspect win rate 42.3% / loss rate 18.7%; Severity win rate 38.5% / loss rate 22.1%

## Highlights & Insights

- **End-to-end multimodal complaint understanding**: The first fine-grained complaint analysis system to fuse text and vision in a multi-turn dialogue setting
- **Two-phase prediction–validation architecture**: Phase 1 CoT experts handle prediction; Phase 2 validation experts provide quality assurance, significantly improving reliability
- **Three-metric evaluation framework**: Alignment, Dominance, and Complementarity jointly assess expert behavior, enhancing interpretability
- **Learnable Semantic Alignment Score**: The dynamic SAS outperforms static cosine similarity and adaptively regulates cross-modal weighting
- **New CIViL dataset**: 2,004 annotated dialogues and 4,478 images, filling the data gap in multimodal dialogue complaint understanding

## Limitations & Future Work

- **Limited data scale**: Only 2,004 Apple Support dialogues, covering a single domain
- **Severe class imbalance**: The Software category accounts for 82.9% (1,662/2,004) of instances, while Price contains only 23 examples, limiting generalization
- **High computational cost**: Four DeepSeek-6.7B CoT experts plus two validation experts incur substantial deployment overhead
- **Non-native image–dialogue pairing**: Visual evidence is crawled via CLIP matching rather than being natively embedded in the dialogues
- **English only**: Multilingual scenarios are not addressed
- **Subjectivity of severity**: Variation in user tone causes the model to underestimate or misclassify Severity levels

## Related Work & Insights

- **vs. ABSA methods**: Traditional ABSA assigns only sentiment polarity, whereas VALOR performs joint fine-grained Aspect and Severity classification
- **vs. VisualBERT/ViLT**: These VLMs lack expert routing and CoT reasoning, achieving approximately 20 F1 points lower
- **vs. Gemma-3 (9B)**: Despite its larger parameter count, Gemma-3 lacks a validation mechanism and semantic alignment, resulting in 12.94% lower ACD accuracy
- **vs. standard MoE**: CoT experts leverage step-by-step reasoning to capture subtle complaint semantics, outperforming MLP/Transformer experts

The two-phase prediction–validation design is generalizable to other multimodal classification tasks requiring high reliability. The three-metric framework (Alignment/Dominance/Complementarity) provides a general methodology for evaluating MoE expert quality. The learnable SAS concept is applicable to any scenario requiring cross-modal consistency assessment.

## Rating

- Novelty: ⭐⭐⭐⭐ — The two-phase architecture combining multi-expert CoT and validation is pioneering in complaint analysis
- Experimental Thoroughness: ⭐⭐⭐⭐ — Ablation studies are comprehensive and human evaluation is complete, though dataset scale and domain coverage are limited
- Writing Quality: ⭐⭐⭐ — Method descriptions are detailed but verbose, with a complex notation system
- Value: ⭐⭐⭐ — Practical application value is clear, but computational cost and data limitations constrain broader adoption

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Towards Fine-Grained Attribution: Instance-Aware Preference Optimization for Aligning Diffusion Models](../../CVPR2026/image_generation/towards_fine-grained_attribution_instance-aware_preference_optimization_for_alig.md)
- [\[ICLR 2026\] EdiVal-Agent: An Object-Centric Framework for Automated, Fine-Grained Evaluation of Multi-Turn Editing](../../ICLR2026/image_generation/edival-agent_an_object-centric_framework_for_automated_fine-grained_evaluation_o.md)
- [\[CVPR 2026\] SketchRevive: Fine-Grained Pixel-to-Vector Sketch Completion with Diffusion-Prior-Guided Multimodal LLMs](../../CVPR2026/image_generation/sketchrevive_fine-grained_pixel-to-vector_sketch_completion_with_diffusion-prior.md)
- [\[CVPR 2026\] Beyond Objects: Contextual Synthetic Data Generation for Fine-Grained Classification](../../CVPR2026/image_generation/beyond_objects_contextual_synthetic_data_generation_for_fine-grained_classificat.md)
- [\[CVPR 2026\] SliderEdit: Continuous Image Editing with Fine-Grained Instruction Control](../../CVPR2026/image_generation/slideredit_continuous_image_editing_with_fine-grained_instruction_control.md)

</div>

<!-- RELATED:END -->
