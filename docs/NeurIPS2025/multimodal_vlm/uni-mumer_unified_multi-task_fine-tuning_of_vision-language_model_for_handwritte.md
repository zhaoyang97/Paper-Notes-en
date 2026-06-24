---
title: >-
  [Paper Note] Uni-MuMER: Unified Multi-Task Fine-Tuning of Vision-Language Model for Handwritten Mathematical Expression Recognition
description: >-
  [NeurIPS 2025 Spotlight][Multimodal VLM][Handwritten Mathematical Expression Recognition] This paper proposes Uni-MuMER, which performs unified multi-task fine-tuning of an open-source VLM via three data-driven tasks (Tree-CoT, Error-Driven Learning, and Symbol Counting), achieving substantial improvements over both specialized lightweight models and zero-shot commercial VLMs on the CROHME and HME100K benchmarks.
tags:
  - "NeurIPS 2025 Spotlight"
  - "Multimodal VLM"
  - "Handwritten Mathematical Expression Recognition"
  - "Vision-Language Model"
  - "Multi-Task Fine-Tuning"
  - "Chain-of-Thought"
  - "Error-Driven Learning"
date: 2026-05-08
content_hash: 79b8e12457c53343
---

# Uni-MuMER: Unified Multi-Task Fine-Tuning of Vision-Language Model for Handwritten Mathematical Expression Recognition

**Conference**: NeurIPS 2025 Spotlight  
**arXiv**: [2505.23566](https://arxiv.org/abs/2505.23566)  
**Code**: [https://github.com/BFlameSwift/Uni-MuMER](https://github.com/BFlameSwift/Uni-MuMER)  
**Area**: Multimodal VLM
**Keywords**: Handwritten Mathematical Expression Recognition, Vision-Language Model, Multi-Task Fine-Tuning, Chain-of-Thought, Error-Driven Learning

## TL;DR

This paper proposes Uni-MuMER, which performs unified multi-task fine-tuning of an open-source VLM via three data-driven tasks (Tree-CoT, Error-Driven Learning, and Symbol Counting), achieving substantial improvements over both specialized lightweight models and zero-shot commercial VLMs on the CROHME and HME100K benchmarks.

## Background & Motivation

Handwritten Mathematical Expression Recognition (HMER) remains a persistent challenge in OCR due to the spatial freedom of symbol layout and the diversity of handwriting styles. Prior methods primarily pursued isolated architectural improvements—such as tree decoders and position-aware attention—yet recent progress has been marginal, with gains from CoMER to SSAN amounting to only approximately 3% on CROHME. These approaches face three fundamental bottlenecks: individual improvements are difficult to integrate, single auxiliary tasks fail to address the multi-dimensional challenges of HMER, and single-domain dataset training limits scalability.

Meanwhile, pretrained VLMs have demonstrated surprisingly strong capabilities on structured recognition tasks; however, commercial models rely on opaque training data, making systematic improvement guidance difficult. How to empower open-source VLMs to achieve comparable or superior HMER performance therefore emerges as a central research question.

## Method

### Overall Architecture

Uni-MuMER adopts Qwen2.5-VL-3B as the backbone VLM **without any architectural modification**, injecting domain knowledge into the general-purpose framework via full-parameter fine-tuning. Four tasks are trained jointly: Vanilla HMER (basic recognition), Tree-Aware Chain-of-Thought (structured reasoning), Error-Driven Learning (error correction), and Symbol Counting (symbol enumeration). The input consists of a handwritten expression image paired with a task instruction, and the output is the corresponding LaTeX sequence.

### Key Designs

1. **Tree-Aware Chain-of-Thought (Tree-CoT)**: The LaTeX expression is parsed into an Abstract Syntax Tree (AST), which is then linearized into a tab-indented text representation via depth-first traversal. The model first generates the serialized tree structure before producing the final LaTeX output, thereby explicitly guiding the model to reason about two-dimensional spatial relationships. The psychological motivation is to transform implicit layout understanding into explicit structured CoT, which is particularly beneficial for expressions with complex structures.

2. **Error-Driven Learning (EDL)**: This component adopts a "learning from mistakes" paradigm. An error corpus is first constructed through cross-fold training (partitioning the dataset into multiple folds, cross-training with repeated sampling) to collect the model's own erroneous predictions. Two sub-tasks are then defined: error detection (marking error positions with `<error_start>/<error_end>` tags and missing tokens with `<deleted>`) and error correction (taking the annotated expression as input and outputting a correction log along with the correct LaTeX). The core objective is to train the model to distinguish visually similar characters such as $2 \leftrightarrow z$ and $0 \leftrightarrow o$.

3. **Symbol Counting (SC)**: A symbol count string (e.g., `\frac:1,a:1,2:2,+:1`) is prepended to the output, requiring the model to accurately enumerate all visible symbols before generating the LaTeX sequence. Motivated by observations from CAN—that models tend to produce locally coherent but globally inconsistent outputs (with repeated or missing symbols)—SC mitigates symbol hallucination in long expressions through explicit counting constraints.

### Loss & Training

Training data from all datasets (CROHME, HME100K, MathWriting, Im2Latexv2) across the three data-driven tasks are uniformly mixed, and the model is trained for a single epoch. Uni-MuMER† uses approximately 1.6M training samples (original data plus three-task derived data) constructed from approximately 386K images. Standard autoregressive cross-entropy loss is employed without any special loss design.

## Key Experimental Results

### Main Results

| Dataset | Metric | Uni-MuMER† | SSAN (Prev. SOTA) | Gain |
|--------|------|-----------|-----------------|------|
| CROHME Avg. | ExpRate | 79.74% | 63.43% (w/ aug) | +16.31% |
| CROHME Avg. | ExpRate@CDM | 82.86% | — | — |
| CROHME14 | ExpRate | 82.05% | 62.58% | +19.47% |
| HME100K | ExpRate | 72.66% | — | — |
| HME100K | ExpRate@CDM | 74.30% | — | — |

Zero-shot comparison: Uni-MuMER† surpasses Gemini2.5-flash (55.32% → 79.74%, +24.42%) and Qwen2.5-VL-72B (56.40% → 79.74%).

### Ablation Study

| Configuration | CROHME Avg. ExpRate | Note |
|------|---------------------|------|
| Vanilla baseline | 68.64% | Vanilla HMER only |
| + Tree-CoT | 70.85% (+2.21) | Structured reasoning gain |
| + EDL | 70.30% (+1.66) | Reduced character confusion |
| + SC | 69.86% (+1.22) | Improved symbol consistency |
| Tree-CoT + EDL + SC | 73.29% (+4.65) | Complementary; best overall |

### Key Findings

- **Tree-CoT yields the largest gains on structurally complex expressions** (approximately 5–6%), with limited effect on simple expressions, confirming that its core value lies in structural reasoning.
- **EDL significantly reduces character confusion**: the top-5 letter–digit confusion score decreases from 5.25 to 3.31 (a 37% reduction), with particularly notable improvements on $3 \leftrightarrow z$ (−0.58) and $1 \leftrightarrow n$ (−0.59).
- **SC improves consistency in long expressions**: the effect is most pronounced when symbols repeat five or more times, though it introduces a slight performance degradation on simple expressions.
- The lightweight model CoMER, when trained on the same external data (386K images), exhibits a performance drop, demonstrating that lightweight architectures cannot effectively leverage large-scale diverse data.

## Highlights & Insights

- **Paradigm shift**: The focus moves from "modifying architectures" to "modifying data"—domain knowledge is injected entirely through data-driven multi-task learning without any alteration to the VLM architecture, yielding an elegant and principled approach.
- **Clever error corpus construction**: Cross-fold training combined with repeated sampling automatically collects model errors without manual annotation, producing an error dataset comparable in scale to the original training data.
- The CDM (Character Detection Matching) visual evaluation metric is introduced to address the unfair penalty that ExpRate imposes on variations in LaTeX syntactic style.
- Inference based on the vLLM framework achieves speeds superior to those of conventional specialized methods, enhancing practical deployability.
- The gains from each task exhibit orthogonality—removing any single task degrades performance—validating the soundness of the overall design.
- The strategy of expanding 386K images to approximately 1.6M samples via three-task data augmentation offers a valuable reference for future work.

## Limitations & Future Work

- Qwen2.5-VL-3B has considerably more parameters than lightweight models, resulting in higher deployment costs.
- Error corpus construction requires multiple rounds of training and sampling, incurring substantial preprocessing overhead.
- SC introduces a slight performance drop on simple expressions; an adaptive activation mechanism—dynamically enabling or disabling SC based on expression length—could be explored.
- The potential of fine-tuning larger-scale VLMs (e.g., 7B/72B) remains unexplored.
- Evaluation is conducted solely on LaTeX output; generalization to other markup formats such as MathML is unknown.
- Tree-CoT relies on AST parsing and may fail for non-standard expressions that cannot be parsed into a conventional AST.
- While the CDM metric resolves syntactic style issues, it exhibits strong dependence on the rendering engine.

## Related Work & Insights

- This work extends the symbol counting idea from CAN, integrating it more naturally within a VLM framework.
- The AST serialization scheme of Tree-CoT is generalizable to other structured output tasks, such as code generation and chemical formula recognition.
- The "model self-correction" paradigm of EDL echoes the concepts of Self-Refine and Constitutional AI; its concrete instantiation in the OCR domain merits further attention.
- Inference via the vLLM framework achieves speeds superior to conventional methods, making deployment in real-world applications practical.
- As an open-source contribution, this work provides an important reference for research on VLM applications in the OCR domain.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The unified multi-task fine-tuning paradigm driven by data is novel; Tree-CoT and EDL are cleverly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive comparisons across multiple datasets and baselines; detailed ablation studies with clearly attributed module contributions.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured presentation with rich illustrations and thorough motivation.
- **Value**: ⭐⭐⭐⭐ A significant breakthrough in HMER; the paradigm shift carries broader inspirational value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Rethinking BCE Loss for Multi-Label Image Recognition with Fine-Tuning](../../CVPR2026/multimodal_vlm/rethinking_bce_loss_for_multi-label_image_recognition_with_fine-tuning.md)
- [\[CVPR 2026\] TRivia: Self-supervised Fine-tuning of Vision-Language Models for Table Recognition](../../CVPR2026/multimodal_vlm/trivia_self-supervised_fine-tuning_of_vision-language_models_for_table_recogniti.md)
- [\[ACL 2025\] Can Vision-Language Models Evaluate Handwritten Math?](../../ACL2025/multimodal_vlm/can_vision-language_models_evaluate_handwritten_math.md)
- [\[NeurIPS 2025\] Advancing Compositional Awareness in CLIP with Efficient Fine-Tuning](advancing_compositional_awareness_in_clip_with_efficient_fin.md)
- [\[ICLR 2026\] pFedMMA: Personalized Federated Fine-Tuning with Multi-Modal Adapter for Vision-Language Models](../../ICLR2026/multimodal_vlm/pfedmma_personalized_federated_fine-tuning_with_multi-modal_adapter_for_vision-l.md)

</div>

<!-- RELATED:END -->
