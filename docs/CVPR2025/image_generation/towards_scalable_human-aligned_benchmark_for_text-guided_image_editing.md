---
title: >-
  [Paper Note] Towards Scalable Human-Aligned Benchmark for Text-Guided Image Editing
description: >-
  [CVPR 2025][Image Generation][Image Editing] Proposes HATIE, a large-scale (18K images/50K queries), fully automated, multi-dimensional text-guided image editing evaluation benchmark, which aligns with human perception by combining metrics from 5 dimensions and fitting user study weights.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Image Editing"
  - "Evaluation Benchmark"
  - "Human Alignment"
  - "Automated Evaluation"
  - "Multi-dimensional Metrics"
date: 2026-05-08
content_hash: 2a88d7ae6c5f0358
---

# Towards Scalable Human-Aligned Benchmark for Text-Guided Image Editing

**Conference**: CVPR 2025  
**arXiv**: [2505.00502](https://arxiv.org/abs/2505.00502)  
**Code**: [https://github.com/SuhoRyu/HATIE](https://github.com/SuhoRyu/HATIE)  
**Area**: Image Generation / Image Editing Evaluation  
**Keywords**: Image Editing, Evaluation Benchmark, Human Alignment, Automated Evaluation, Multi-dimensional Metrics

## TL;DR

Proposes HATIE, a large-scale (18K images/50K queries), fully automated, multi-dimensional text-guided image editing evaluation benchmark, which aligns with human perception by combining metrics from 5 dimensions and fitting user study weights.

## Background & Motivation

**Background**: In recent years, a large number of text-guided image editing models have emerged (e.g., Imagic, Prompt-to-Prompt, SDEdit, etc.), but widely accepted standard evaluation methods are lacking. Researchers mainly rely on manual user studies for evaluation.

**Limitations of Prior Work**: Image editing tasks are inherently subjective—the same editing instruction can have multiple correct outputs, and there is no unique "gold standard" ground truth. Existing evaluation schemes suffer from three issues: (1) Existing benchmarks are too small (e.g., TEdBench has only 100 images), making robust evaluation impossible; (2) Some benchmarks only cover easy-to-evaluate editing types (e.g., colorization), limiting their scope of application; (3) Evaluation requires multi-dimensional considerations (editing fidelity, background preservation, image quality, etc.), and a single metric is insufficient to comprehensively reflect editing quality.

**Key Challenge**: Image editing quality evaluation requires human-like high-level understanding to measure perceptual relevance, but relying on specific models (e.g., CLIP) sacrifices reliability and reproducibility; while manual evaluation is accurate, it is not scalable.

**Goal**: Construct a large-scale, automated, multi-dimensional, human-aligned evaluation framework for image editing to solve the scalability and objectivity issues of evaluation.

**Key Highlight**: Decompose editing quality into 5 orthogonal dimensions for separate evaluation, and then combine these dimensions through user study fitted weights to maximize correlation with human judgment.

**Core Idea**: Formulate subjective image editing quality assessment into automated objective scoring through a "convex combination of multiple metrics + weight fitting to align with human perception" approach.

## Method

### Overall Architecture

The HATIE framework consists of three parts: (1) a large-scale image and editing query set built on the GQA dataset, covering 76 COCO object categories and 7 editing types; (2) a fully automated multi-dimensional evaluation pipeline assessing editing quality from 5 dimensions; (3) a weight system fitted through user studies to align scores with human perception. The input is the original image + editing instruction + edited image, and the output is a comprehensive score between 0 and 1.

### Key Designs

1. **Large-scale Editing Query Generation System**:

    - **Function**: Automatically generates feasible editing queries, covering 7 editing types: object addition/deletion/replacement/attribute modification/scaling, and background/style transformation.
    - **Mechanism**: Based on the rich annotations of the GQA dataset (object names, bounding boxes, attributes, relations), feasible editing choices are determined through statistical analysis. For example, for object addition, only common relation combinations are used (e.g., "add notebook on the table" instead of "add car on the table"). Attribute modification switches within the same category (e.g., $'brown' \rightarrow 'yellow'$ both belong to the color category). Unsuitable objects are filtered out (too small, cropped, occluded, or not in COCO categories). Finally, 49,840 editing queries are generated.
    - **Design Motivation**: Through data-driven feasibility constraints instead of hand-crafted rules, the validity of edits is guaranteed while achieving large-scale automated generation.

2. **Five-Dimensional Automated Evaluation System**:

    - **Function**: Fully automatically evaluates editing quality from five aspects: Image Quality (IQ), Object Fidelity (OF), Background Fidelity (BF), Object Consistency (OC), and Background Consistency (BC).
    - **Mechanism**: Uses instance segmentation models to segment the edited image into target objects and backgrounds to evaluate fidelity and consistency separately. OF combines a convex combination of CLIP alignment $\sigma^{OF}_{clip}$, detection confidence $\sigma^{OF}_{det}$, and size fidelity $\sigma^{OF}_{size}$. OC combines 5 metrics: LPIPS, DINO similarity, L2 distance, position consistency, and size consistency. IQ is measured by FID. The total score is $\sigma^{Total} = \sum_{x \in \mathcal{X}} w^x \sigma^x$, and different editing types are evaluated using only suitable dimensions (e.g., object replacement does not evaluate OC).
    - **Design Motivation**: Editing quality is inherently multi-dimensional—the edit must be faithful, the background must be preserved, and the image must be of high quality; a single metric cannot cover all aspects. Fine-grained evaluation is achieved through object/background separation.

3. **Human Alignment Weight Fitting Mechanism**:

    - **Function**: Automatically determines the combination weights for each metric and dimension to maximize alignment between the automated score and human perception.
    - **Mechanism**: Sampling 4,050 edited images from 6 models, 24 participants were divided into 8 groups for pairwise user studies. Compute the win rate vector $\mathbf{u}^k$ in the user study and the automated evaluation win rate vector $\mathbf{v}^k$ for each model, and use grid search (step size 0.01) to find the weight combination that maximizes the Pearson correlation coefficient between the two.
    - **Design Motivation**: The importance of different metrics and dimensions is difficult to set manually. Fitting weights in a data-driven manner makes the final score align better with human perception. Once the weights are fitted, they are fixed, and subsequent evaluations do not require running user studies again.

### Loss & Training

This is an evaluation benchmark paper and does not involve model training. The core "training" is the weight-fitting process: acquiring human preference data through user studies on 2,700 images, and then using grid search to maximize the Pearson correlation coefficient to determine the weight parameters at each level.

## Key Experimental Results

### Main Results

| Model | Object Fidelity | Background Fidelity | Object Consistency | Background Consistency | Total Score |
|------|----------------|--------------------|--------------------|----------------------|-------------|
| Imagic | Low | Low | Medium | Medium | Low |
| P2P ($\tau=0.4$) | Medium | Medium | Medium | Medium | Optimal Point |
| MasaCtrl | Medium | Medium | High | High | Medium |
| IP2P (Optimal $s_T$) | High | Medium | Medium | Medium | High |

### Ablation Study

| Configuration/Metric | Correlation Coefficient with User Study | Note |
|---------|-------------------|------|
| HATIE Total Score ($\rho$) | 0.7143 | Overall alignment effect |
| Object Consistency ($\rho$) | **0.9276** | Best aligned dimension |
| Background Fidelity ($\rho$) | 0.8971 | Second best |
| Single CLIP Alignment ($r$) | -0.3410 | Negatively correlated with human perception! |
| Single LPIPS ($r$) | 0.5468 | Best single metric but far from sufficient |
| Single Detection Rate ($r$) | 0.4293 | Moderately correlated |
| Single FID ($r$) | 0.5058 | Moderately correlated |

### Key Findings

- **CLIP Alignment used alone is negatively correlated with human perception (-0.34)**, indicating that the commonly used CLIP metric as an independent evaluation method is unreliable.
- **Changes in editing strength parameters lead to a clear trade-off between Fidelity and Consistency**: stronger editing results in higher fidelity ($\uparrow$) but lower consistency ($\downarrow$). HATIE's Total Score can accurately capture the optimal balance point.
- **The Object Consistency dimension aligns best with human perception ($\rho = 0.9276$)**, suggesting that humans value whether unedited objects remain unchanged most when judging editing quality.
- **The bootstrap error of the evaluation results is very small**, indicating that the scale of the benchmark dataset is large enough, and the evaluation results are stable and capable of distinguishing subtle model differences.

## Highlights & Insights

- **Evaluation Paradigm of Multi-Metric Convex Combination + Weight Fitting**: Converts a subjective evaluation problem into an objective optimization problem. The approach is generalizable and can be transferred to highly subjective evaluation tasks such as video editing and 3D content editing.
- **Object/Background Separation Evaluation**: Disentangles different aspects of editing through instance segmentation to achieve fine-grained evaluation. This "divide-and-conquer" strategy is simple but highly effective.
- **Feasible Constraint Query Generation**: Automatically determines object relation frequencies through statistical analysis to constrain the feasibility of editing queries, avoiding the generation of meaningless queries. This is an ingenious method for large-scale data construction.

## Limitations & Future Work

- Object segmentation depends on the COCO-pretrained segmentation model (Mask R-CNN), which cannot evaluate objects outside COCO categories, limiting the generalization of the evaluation.
- The weights are fitted on 6 description-based models, and their applicability to instruction-based models (e.g., InstructPix2Pix) has not been verified.
- The evaluation of background and style transformation is relatively weak, as such global edits lack clear "objects" to evaluate separately.
- The evaluation metrics are mainly based on visual similarity and do not consider the physical plausibility of editing (e.g., lighting consistency, perspective relations).

## Related Work & Insights

- **vs TEdBench**: Only 100 images, no automated evaluation, and the scale is too small for stable evaluation. HATIE achieves a 200x increase in scale.
- **vs GIER**: Although it has 30K queries, it is limited to editing types where ground truth is easily obtained (e.g., grayscale conversion) and cannot cover open-ended editing tasks. HATIE avoids dependency on pixel-level ground truth through a combination of multiple metrics.
- **vs EditVal**: Covers more editing types and object categories but still relies on user studies for evaluation. HATIE achieves full automation.

## Rating

- Novelty: ⭐⭐⭐⭐ The philosophy of weight fitting to align with human perception is novel, and the five-dimension separation evaluation design is reasonable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ A 24-participant user study verifies the alignment; comprehensive testing indeed across multiple models and parameters, with rigorous statistical analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, well-defined problem, and standard use of mathematical notation.
- Value: ⭐⭐⭐⭐ Solves key pain points in image editing evaluation; open-source code and datasets will help drive standardization in the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] EditReward: A Human-Aligned Reward Model for Instruction-Guided Image Editing](../../ICLR2026/image_generation/editreward_a_human-aligned_reward_model_for_instruction-guided_image_editing.md)
- [\[CVPR 2025\] SwiftEdit: Lightning Fast Text-Guided Image Editing via One-Step Diffusion](swiftedit_lightning_fast_text-guided_image_editing_via_one-step_diffusion.md)
- [\[CVPR 2025\] InterEdit: Navigating Text-Guided Multi-Human 3D Motion Editing](interedit_navigating_text-guided_multi-human_3d_motion_editing.md)
- [\[CVPR 2025\] From Words to Structured Visuals: A Benchmark and Framework for Text-to-Diagram Generation and Editing](from_words_to_structured_visuals_a_benchmark_and_framework_for_text-to-diagram_g.md)
- [\[CVPR 2025\] PQPP: A Joint Benchmark for Text-to-Image Prompt and Query Performance Prediction](pqpp_a_joint_benchmark_for_text-to-image_prompt_and_query_performance_prediction.md)

</div>

<!-- RELATED:END -->
