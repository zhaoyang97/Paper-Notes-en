---
title: >-
  [Paper Note] NLPrompt: Noise-Label Prompt Learning for Vision-Language Models
description: >-
  [CVPR 2025][Multimodal VLM][Noisy Label Learning] This paper discovers that simply replacing the CE loss with MAE loss in CLIP prompt learning can significantly improve robustness against noisy labels, which is theoretically proven via feature learning theory. Building on this, the authors propose NLPrompt—a method that combines an Optimal Transport-based data purification (PromptOT) to split the dataset into clean and noisy subsets, which are then trained using CE and MAE lo…
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Noisy Label Learning"
  - "Prompt Learning"
  - "CLIP"
  - "Optimal Transport"
  - "Robustness"
date: 2026-05-08
content_hash: 116ba3255ac597d6
---

# NLPrompt: Noise-Label Prompt Learning for Vision-Language Models

**Conference**: CVPR 2025  
**arXiv**: [2412.01256](https://arxiv.org/abs/2412.01256)  
**Code**: [https://github.com/qunovo/NLPrompt](https://github.com/qunovo/NLPrompt)  
**Area**: Multimodal VLM  
**Keywords**: Noisy Label Learning, Prompt Learning, CLIP, Optimal Transport, Robustness

## TL;DR

This paper discovers that simply replacing the CE loss with MAE loss in CLIP prompt learning can significantly improve robustness against noisy labels, which is theoretically proven via feature learning theory. Building on this, the authors propose NLPrompt—a method that combines an Optimal Transport-based data purification (PromptOT) to split the dataset into clean and noisy subsets, which are then trained using CE and MAE losses respectively, outperforming existing methods by a wide margin under various noise settings.

## Background & Motivation

**Background**: CLIP-based prompt learning (such as CoOp, CoCoOp) has become the mainstream parameter-efficient fine-tuning scheme for vision-language models. It adapts to downstream tasks by learning continuous text prompt vectors, with only a few thousand parameters.

**Limitations of Prior Work**: Real-world annotated data inevitably contains noisy labels (labeling errors), and prompt learning still overfits noisy labels when using Cross-Entropy (CE) loss. Existing works (e.g., Wu et al.) have shown that prompt learning is more robust than fine-tuning methods like adapters, but its performance still degrades significantly under high noise rates. Subsequent works like JoAPR use Gaussian mixture models to distinguish clean/noisy data and correct labels without fully leveraging the unique advantages of prompt learning.

**Key Challenge**: In traditional noisy label learning, although MAE loss is theoretically robust, its slow convergence and poor performance under conventional training paradigms make it rarely used. However, in prompt learning scenarios, the situation could be entirely different—an overlooked research perspective.

**Goal**: (1) To explain why MAE can maintain both robustness and high accuracy in prompt learning; (2) To investigate how to further improve prompt learning performance under noisy conditions.

**Key Insight**: The authors experimentally discovered an interesting phenomenon: utilizing MAE loss in prompt learning (PromptMAE) not only vastly outperforms CE loss in robustness but also matches it in convergence speed and final accuracy. This contradicts the typical behavior of MAE in traditional training.

**Core Idea**: Leverage the unique robustness advantage of MAE loss in prompt learning, and combine it with Optimal Transport-based data purification using CLIP text features, to achieve robust prompt learning under noisy labels.

## Method

### Overall Architecture

NLPrompt consists of two core components: (1) PromptMAE—directly replacing CE with MAE as the training loss for prompt learning; (2) PromptOT—utilizing the text features of the CLIP text encoder as class prototypes, and partitioning the dataset into clean and noisy subsets through an optimal transport algorithm, which are then trained using CE loss and MAE loss, respectively. The input is an image dataset with noisy labels and the CLIP model, and the output is the learned robust text prompts.

### Key Designs

1. **PromptMAE—Prompt Learning with MAE Loss**:

    - **Function**: Directly replaces CE with MAE loss in prompt learning to enhance noise robustness.
    - **Mechanism**: MAE loss is defined as $\ell_{\text{MAE}} = \sum_{c=1}^{C} |y_{i,c} - s_{i,c}|$, where $s_{i,c}$ is the similarity after softmax. The authors mathematically prove via feature learning theory that, in prompt learning, MAE suppresses the negative impact of noisy samples on task-relevant feature coefficients. Specifically, learnable prompts can be decomposed into a linear combination of task-relevant features $\mu$ and task-irrelevant features $\xi_l$. Under MAE loss, the decay speed of the task-relevant coefficient $\beta$ caused by noisy samples is much slower than under CE loss. It is theoretically proven that PromptMAE achieves a lower test loss than PromptCE with high probability $1-d^{-1}$.
    - **Design Motivation**: MAE is impractical in traditional deep learning due to slow convergence. However, prompt learning only updates very few parameters (thousands) and operates on a pre-aligned feature space, allowing the robustness advantages of MAE to shine while eliminating its convergence drawbacks.

2. **PromptOT—Optimal Transport-Based Data Purification**:

    - **Function**: Partitions the dataset into clean and noisy subsets.
    - **Mechanism**: Traditional OT methods use randomly initialized prototypes to construct the transport matrix. PromptOT, however, utilizes text features $\mathbf{T} \in \mathbb{R}^{C \times d}$ generated by the CLIP text encoder as class prototypes. It computes a similarity matrix with image features $\mathbf{I} \in \mathbb{R}^{N \times d}$, treats the negative log-likelihood as the cost matrix, and solves the entropy-regularized OT problem via the Sinkhorn algorithm to obtain the transport matrix $\mathbf{Q}^*$. The pseudo-label $\hat{y}_i$ is obtained by applying argmax to each column. If $\hat{y}_i = \tilde{y}_i$, the sample is labeled clean; otherwise, it is labeled noisy.
    - **Design Motivation**: The feature spaces of vision-language models are already pre-aligned during pre-training, so text features naturally serve as high-quality class prototypes, which are much more accurate than random initializations. Additionally, the OT framework incorporates global distribution constraints, making it more robust than sample-wise prediction.

3. **CE + MAE Hybrid Training Strategy**:

    - **Function**: Combines the fast convergence/accuracy benefits of CE on clean data with the robustness of MAE on noisy data.
    - **Mechanism**: The total loss is defined as $\ell_{\text{NLPrompt}} = \sum_{i \in \mathcal{D}_{\text{clean}}} -\mathbf{y}_i^\top \log \mathbf{s}_i + \sum_{j \in \mathcal{D}_{\text{noisy}}} \|\mathbf{y}_j - \mathbf{s}_j\|_1$. Applying CE to the clean subset yields faster convergence and higher accuracy, while applying MAE to the noisy subset prevents the model from overfitting to incorrect labels.
    - **Design Motivation**: Using only MAE is robust but underperforms compared to CE in low-noise scenarios, while using only CE collapses under high-noise conditions. A divide-and-conquer strategy reaps the benefits of both worlds.

### Loss & Training

Hybrid Loss: CE for the clean set, MAE for the noisy set. PromptOT performs data partitioning once before training begins (no iterative updates required), making the overall training pipeline highly concise.

## Key Experimental Results

### Main Results

| Dataset | Method | Sym-25% | Sym-50% | Sym-75% | Asym-25% | Asym-50% |
|--------|------|---------|---------|---------|----------|----------|
| Flowers102 | CoOp | 83.50 | 70.10 | 37.17 | 74.70 | 42.60 |
| Flowers102 | GCE | 88.33 | 84.07 | 70.37 | 86.37 | 69.93 |
| Flowers102 | JoAPR | 81.23 | 70.23 | 66.93 | 79.63 | 73.83 |
| Flowers102 | **NLPrompt** | **92.57** | **89.90** | **76.80** | **93.40** | **81.10** |
| OxfordPets | **NLPrompt** | **86.00** | **84.87** | **70.77** | **84.97** | **77.53** |
| StanfordCars | **NLPrompt** | **68.80** | **65.63** | **58.30** | **67.53** | **59.03** |

### Ablation Study

| Configuration | Flowers102 (Sym-50%) | Caltech101 (Sym-50%) |
|------|---------------------|---------------------|
| CoOp (CE only) | 70.10 | 70.90 |
| PromptMAE (MAE only) | ~85+ | ~87+ |
| PromptOT + CE | ~86 | ~88 |
| NLPrompt (Full) | **89.90** | **90.70** |

### Key Findings

- PromptMAE used alone already substantially outperforms CoOp (improving from 70.1% to ~85% on Flowers102 Sym-50%), validating the unique advantage of MAE in prompt learning.
- Under extreme noise rates (75% symmetric noise), NLPrompt remains highly competitive, e.g., achieving 76.80% on Flowers102, whereas CoOp only reaches 37.17%.
- Asymmetric noise poses a greater challenge to all methods, but the advantages of NLPrompt become even more pronounced in this setting.
- NLPrompt achieves near-universal superiority across all 11 datasets and 12 noise settings, demonstrating exceptional robustness.

## Highlights & Insights

- **The "Resurrection" of MAE in Prompt Learning**: MAE, although discarded in traditional training due to slow convergence, excels in prompt learning because of the small parameter count and the pre-aligned feature space. This reveals a crucial principle: the efficacy of a loss function is tightly coupled with the optimization scenario, and traditional conclusions should not be blindly transferred.
- **Text Features as OT Prototypes**: Leveraging CLIP's pre-aligned text features instead of random prototypes to construct the transport matrix elegantly injects the benefits of vision-language pre-training into the data purification process. This approach is highly transferable to any setting requiring class prototypes.
- **Dual Validation of Theory and Practice**: The study provides a rigorous proof of MAE's robustness guarantee in prompt learning via feature learning theory, followed by extensive experimental validation, making the overall argumentation highly complete.

## Limitations & Future Work

- PromptOT partitioning is a one-off step (executed before training) and is not dynamically updated during training, which might lack precision on boundary samples.
- The theoretical analysis assumes binary classification and linear features, leaving a gap between the theory and actual multi-class, non-linear scenarios.
- The method is evaluated solely on CLIP-based models; its generalizability to other vision-language models (e.g., BLIP-2, LLaVA) remains to be explored.
- The impact of different sources and types of noise labels (e.g., instance-dependent noise vs. uniform noise) on the method is not discussed.

## Related Work & Insights

- **vs CoOp/CoCoOp**: Standard prompt learning methods use the CE loss, which suffers severe performance degradation under noisy labels; NLPrompt addresses this concern by replacing the loss function and incorporating data purification.
- **vs JoAPR**: JoAPR utilizes Gaussian mixture models to distinguish noisy data and corrects labels using mixup, resulting in a complex design with mediocre performance; NLPrompt is much simpler and achieves superior performance.
- **vs GCE (Generalized Cross-Entropy)**: GCE is a robust loss from traditional noisy label learning that also exhibits some robustness in prompt learning. However, NLPrompt widens the gap further through a differentiated treatment of clean and noisy data.
- This work prompts the community to reflect: should we re-evaluate traditionally discarded techniques when fine-tuning pre-trained models?

## Rating

- Novelty: ⭐⭐⭐⭐ Discovering the unique advantage of MAE in prompt learning is an insightful observation, though the algorithm itself is relatively simple.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 11 datasets $\times$ 12 noise settings, offering extremely comprehensive coverage.
- Writing Quality: ⭐⭐⭐⭐ The theoretical derivation is clear, and the experimental layout is well-designed.
- Value: ⭐⭐⭐⭐ Holds significant practical guiding significance for applying prompt learning in real-world scenarios with noisy labels.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Vision-Language Model IP Protection via Prompt-based Learning](vision-language_model_ip_protection_via_prompt-based_learning.md)
- [\[CVPR 2026\] FedMPT: Federated Multi-Label Prompt Tuning of Vision-Language Models](../../CVPR2026/multimodal_vlm/fedmpt_federated_multi-label_prompt_tuning_of_vision-language_models.md)
- [\[CVPR 2026\] Noise-Aware Few-Shot Learning through Bi-directional Multi-View Prompt Alignment](../../CVPR2026/multimodal_vlm/noise-aware_few-shot_learning_through_bi-directional_multi-view_prompt_alignment.md)
- [\[NeurIPS 2025\] VaMP: Variational Multi-Modal Prompt Learning for Vision-Language Models](../../NeurIPS2025/multimodal_vlm/vamp_variational_multi-modal_prompt_learning_for_vision-language_models.md)
- [\[CVPR 2025\] MMRL: Multi-Modal Representation Learning for Vision-Language Models](mmrl_multi-modal_representation_learning_for_vision-language_models.md)

</div>

<!-- RELATED:END -->
