---
title: >-
  [Paper Note] Instruction-Grounded Visual Projectors for Continual Learning of Generative Vision-Language Models
description: >-
  [ICCV 2025][Multimodal VLM][Continual Learning] This paper proposes MVP (Mixture of Visual Projectors), a Mixture-of-Experts framework for visual projectors conditioned on instruction context. Through an expert recommendation strategy and an expert pruning mechanism, MVP enables generative VLMs to continually learn new vision-language tasks without catastrophic forgetting, while maintaining responsiveness to diverse instruction types. MVP consistently outperforms existing met…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "Continual Learning"
  - "Vision-Language Models"
  - "Mixture-of-Experts"
  - "Visual Projector"
  - "Instruction-Awareness"
date: 2026-05-08
content_hash: 0d1f6d162355f152
---

# Instruction-Grounded Visual Projectors for Continual Learning of Generative Vision-Language Models

**Conference**: ICCV 2025
**arXiv**: [2508.00260](https://arxiv.org/abs/2508.00260)  
**Code**: N/A  
**Area**: Multimodal VLM
**Keywords**: Continual Learning, Vision-Language Models, Mixture-of-Experts, Visual Projector, Instruction-Awareness

## TL;DR

This paper proposes MVP (Mixture of Visual Projectors), a Mixture-of-Experts framework for visual projectors conditioned on instruction context. Through an expert recommendation strategy and an expert pruning mechanism, MVP enables generative VLMs to continually learn new vision-language tasks without catastrophic forgetting, while maintaining responsiveness to diverse instruction types. MVP consistently outperforms existing methods across classification, captioning, and question-answering tasks.

## Background & Motivation

Generative VLMs (e.g., InstructBLIP) face two core challenges when adapting to new tasks:

**Catastrophic Forgetting**: Training on new tasks overwrites previously acquired knowledge. Full retraining on all data is prohibitively expensive, and pre-training data may be inaccessible.

**Instruction Neglect (Key Insight)**: Existing continual learning methods (e.g., EProj, GMM) learn new tasks by updating a shared visual projector. When multiple tasks share similar instruction templates, the projector over-adapts to these patterns, causing the model to "ignore" textual instructions and rely solely on visual inputs when faced with different instruction types. For example, after training on a QA task, given an image paired with a classification instruction, the model still generates QA-style responses.

The authors identify the root cause: **a shared visual projector cannot adjust its translation of visual information according to different instruction contexts**. A single projector can only translate visual features into one "language" for the LLM, whereas different tasks require different translation modes.

## Method

### Overall Architecture

MVP consists of three core components:
1. Mixture of Visual Projectors (MoE): multiple projector experts + a router
2. Expert Recommendation Strategy: recommends which experts to reuse based on task semantic similarity
3. Expert Pruning Mechanism: removes redundantly activated experts to prevent negative transfer

At inference time, Adaptive Knowledge Aggregation (AKA) balances the MoE output and the pre-trained projector output.

### Key Designs

1. **Mixture of Visual Projectors**

   $N_E$ projector experts $\{\mathcal{E}_j\}$ and a router $\mathcal{R}$ are introduced. The router jointly conditions on image features and instruction embeddings to determine which experts to activate:

   $W = \text{Softmax}(\text{Top-}K(\mathcal{R}(\mathbf{x}_{i,\text{img}}^t, \mathbf{x}_{i,\text{text}}^t)))$

   The aggregated output is averaged with the pre-trained projector's output:

   $\tilde{\mathbf{x}}_{i,\text{img}}^t = \frac{1}{K+1}\left(\mathcal{V}(\mathbf{x}_{i,\text{img}}^t) + \sum_{j=1}^{N_E} w_j \mathcal{E}_j(\mathbf{x}_{i,\text{img}}^t)\right)$

   **Design Motivation**: By conditioning routing on instruction embeddings, different instruction types (classification/captioning/QA) activate different expert combinations, enabling instruction-aware visual translation. Retaining the pre-trained projector's contribution ensures zero-shot capability is preserved.

2. **Expert Recommendation Strategy**

   The semantic similarity between the new task and all prior tasks is computed along both visual and textual dimensions:

   $s^{t'} = \alpha \cdot \sigma(s_{\text{img}}^{t'}) + (1-\alpha) \cdot \sigma(s_{\text{text}}^{t'})$

   A contrastive distribution loss encourages the router to reuse experts associated with similar prior tasks, while Activation Bias Reduction suppresses frequently activated experts:

   $\mathcal{L}_{bias} = \frac{1}{2}\left(1 + \frac{\langle \bar{L}^{1:t-1}, L^t \rangle}{\|\bar{L}^{1:t-1}\|_2 \cdot \|L^t\|_2}\right)$

   **Design Motivation**: This leverages relevant knowledge from prior tasks to accelerate new task learning, while preventing all tasks from converging to a small subset of experts, thereby preserving learning capacity.

3. **Expert Pruning**

   After training on each task, a sparse vector $E^t$ is learned to minimize the discrepancy between pre- and post-pruning outputs while constraining the number of activated experts:

   $\min_{E^t} \left\|\sum_{j=1}^{N_E}(w_j - e_j^t)\mathcal{E}_j(x_{i,\text{img}}^t)\right\|_F + \|\mathcal{M}^{1:t-1} + E^t\|_1$

   $E^t$ is thresholded into a binary mask $\mathcal{M}^t$, and the router is then fine-tuned on synthetic data to adapt to the pruned expert configuration.

   **Design Motivation**: During continual learning, certain experts may accumulate redundant activations. Reinitializing pruned experts to pre-trained weights preserves learning capacity for future tasks and reduces the risk of negative transfer.

### Loss & Training

The total loss function is:
$$\mathcal{L} = \mathcal{L}_{ce} + \lambda_{rec}\mathcal{L}_{rec} + \lambda_{bias}\mathcal{L}_{bias}$$

Implementation details:
- LLM: Vicuna-7B or LLaMA-2-7B
- Visual encoder: ViT-g/14 + Q-Former (same as InstructBLIP)
- Number of experts $N_E=20$, activating $K=2$ per forward pass
- Semantic scoring weight $\alpha=0.3$, loss weights $\lambda_{rec}=\lambda_{bias}=1$
- Datasets: ImageNet-R (10 subsets × 20 classes, classification) + Flickr-30K (4 subsets, captioning) + COCO-QA (4 subsets, QA), totaling 18 sequential tasks
- Optimizer: Adam ($\beta_1=0.9$, $\beta_2=0.999$), NVIDIA RTX 3090 GPU

## Key Experimental Results

### Main Results

**Last metric after 18 sequential tasks (Vicuna version)**:

| Method | Cls T1–T3 | Cls T7–T10 | Cap (Animal) | Cap (Vehicle) | QA (Object) | QA (Color) |
|--------|-----------|------------|--------------|---------------|-------------|------------|
| Zero-Shot | 67.79 | 62.87 | 77.08 | 73.43 | 68.52 | 62.62 |
| LwF | 4.38 | 5.83 | 70.16 | 68.60 | 70.69 | 76.46 |
| EWC | 5.61 | 5.83 | 67.00 | 66.47 | 65.72 | 73.69 |
| GMM | 1.33 | 1.62 | 62.96 | 64.70 | 42.29 | 45.11 |
| MoEAdapter | 70.01 | 65.55 | 76.53 | 73.92 | 66.76 | 41.53 |
| **MVP** | **82.81** | **87.52** | **79.60** | **77.79** | **79.26** | **80.30** |

MVP surpasses the zero-shot baseline on all tasks, while competing methods (LwF, EWC, GMM) exhibit near-complete forgetting on classification tasks.

**Average Last performance gap vs. Zero-Shot (LLaMA-2 version)**:

| Method | Cls Δ | Cap Δ | QA Δ |
|--------|-------|-------|------|
| LwF | +10.92 | +1.22 | −5.52 |
| MoEAdapter | +2.09 | −0.03 | −1.05 |
| **MVP** | **+20.78** | **+2.82** | **+7.15** |

### Ablation Study

| Configuration | Cls Avg | Cap Avg | QA Avg | Notes |
|---------------|---------|---------|--------|-------|
| MVP (full) | 85.87 (+21.46) | 77.75 (+2.75) | 76.34 (+24.72) | Best |
| w/o AKA | 85.53 | 76.70 | 76.17 | AKA contributes stable gains |
| w/o AKA, Prune | 85.92 | 71.55 (−3.45) | 75.54 | Pruning prevents captioning forgetting |
| w/o AKA, Prune, $\mathcal{L}_{bias}$ | 3.31 (−61.10) | 70.70 | 73.99 | Bias reduction critical for classification |
| w/o all (shared projector) | 1.49 (−62.92) | 63.10 (−11.90) | 60.15 (+8.53) | Severe forgetting |

$\mathcal{L}_{bias}$ is the most critical component — its removal causes classification to collapse from 85.92% to 3.31%, as without bias reduction the router concentrates all samples onto a small subset of experts.

### Key Findings

1. **Severity of Instruction Neglect**: After training on QA tasks, LwF, EWC, and GMM reduce classification accuracy to ~5%, as the shared projector is overwritten by the last task's instruction patterns.
2. **Cross-Task Knowledge Transfer**: After learning classification tasks, MVP yields a slight improvement on captioning, demonstrating that instruction-aware translation can leverage classification knowledge to benefit captioning.
3. **Limitations of MoEAdapter**: Despite also employing MoE, MoEAdapter freezes old experts and lacks a recommendation mechanism, resulting in inferior performance on captioning and QA tasks compared to MVP.
4. **Consistent Advantage Across Two VLMs**: MVP demonstrates significant gains on both Vicuna and LLaMA-2, confirming its robustness across different architectures.

## Highlights & Insights

- **Precise Problem Formulation**: The paper identifies the overlooked "instruction neglect" problem and traces it to the fundamental limitation of shared visual projectors.
- **Novel Application of MoE in Continual Learning**: Unlike conventional strategies of freezing old experts or expanding new ones, MVP dynamically manages expert lifecycles through recommendation and pruning.
- **Elegance of Adaptive Knowledge Aggregation**: At inference time, the contribution weight of the MoE branch is modulated based on the similarity between the input and learned tasks, naturally degenerating to zero-shot mode for unseen tasks.
- **Necessity of Expert Pruning**: Ablation results clearly demonstrate that pruning prevents negative transfer — without it, captioning performance drops by 6.2%.

## Limitations & Future Work

- The task sequence used in experiments is relatively simple (classification → captioning → QA); more complex task mixing or interleaved learning scenarios remain unvalidated.
- The setting of 20 experts incurs non-trivial parameter overhead for real deployment, despite only 2 being activated per forward pass.
- Expert recommendation requires storing average embeddings per task; memory and computation scale linearly with the number of tasks.
- Experiments are conducted only on 7B-scale LLMs; applicability to larger or smaller models is unknown.
- Synthetic data used for router fine-tuning may introduce distributional bias.
- The task boundary assumption (knowing when to switch to a new task) may not hold in practical scenarios.

## Related Work & Insights

This paper bridges two research directions: MoE and VLM continual learning. Compared to MoEAdapter, the core innovation lies in instruction-conditioned routing and semantics-based expert recommendation. Compared to traditional continual learning methods (LwF, EWC), the advantage is that neither knowledge distillation nor parameter importance estimation is required. The expert pruning mechanism may also serve as a useful reference for other MoE-based continual learning settings.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Instruction-conditioned MoE visual projectors represent a novel design; the recommendation and pruning mechanisms offer meaningful technical contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Two VLMs, ablation studies, per-timestep performance curves, and qualitative analysis are provided, though the task setup is relatively simple.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with intuitive figures, though the dense notation requires careful reading.
- **Value**: ⭐⭐⭐⭐ Represents a meaningful advance in VLM continual learning, though validation in practical deployment scenarios is insufficient.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SMoLoRA: Exploring and Defying Dual Catastrophic Forgetting in Continual Visual Instruction Tuning](smolora_exploring_and_defying_dual_catastrophic_forgetting_in_continual_visual_i.md)
- [\[CVPR 2025\] Continual Learning with Vision-Language Models via Semantic-Geometry Preservation](../../CVPR2025/multimodal_vlm/continual_learning_with_vision-language_models_via_semantic-geometry_preservatio.md)
- [\[ICLR 2026\] Reversible Primitive–Composition Alignment for Continual Vision–Language Learning](../../ICLR2026/multimodal_vlm/reversible_primitivecomposition_alignment_for_continual_visionlanguage_learning.md)
- [\[ICLR 2026\] Enhanced Continual Learning of Vision-Language Models with Model Fusion](../../ICLR2026/multimodal_vlm/enhanced_continual_learning_of_vision-language_models_with_model_fusion.md)
- [\[ECCV 2024\] Select and Distill: Selective Dual-Teacher Knowledge Transfer for Continual Learning on Vision-Language Models](../../ECCV2024/multimodal_vlm/select_and_distill_selective_dual-teacher_knowledge_transfer_for_continual_learn.md)

</div>

<!-- RELATED:END -->
