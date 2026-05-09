---
title: >-
  [Paper Note] Learning to Instruct for Visual Instruction Tuning
description: >-
  [NeurIPS 2025][Multimodal VLM][visual instruction tuning] This paper proposes L2T (Learning to Instruct), which improves visual instruction tuning solely by extending the training loss to cover the instruction sequence (rather than computing loss on responses only). Without additional data and with virtually zero computational overhead, L2T achieves up to 9% relative improvement across 16 multimodal benchmarks, an 18% gain on captioning tasks, and notable hallucination reduction.
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - visual instruction tuning
  - L2T
  - MLLM
  - loss function
  - hallucination
  - captioning
date: 2026-05-08
content_hash: 7a628e18218fd39a
---

# Learning to Instruct for Visual Instruction Tuning

**Conference**: NeurIPS 2025
**arXiv**: [2503.22215](https://arxiv.org/abs/2503.22215)
**Code**: [https://github.com/Feng-Hong/L2T](https://github.com/Feng-Hong/L2T)
**Area**: Multimodal VLM
**Keywords**: visual instruction tuning, L2T, MLLM, loss function, hallucination, captioning

## TL;DR
This paper proposes L2T (Learning to Instruct), which improves visual instruction tuning solely by extending the training loss to cover the instruction sequence (rather than computing loss on responses only). Without additional data and with virtually zero computational overhead, L2T achieves up to 9% relative improvement across 16 multimodal benchmarks, an 18% gain on captioning tasks, and notable hallucination reduction.

## Background & Motivation
**Background**: Visual instruction tuning (VIT) is the standard pipeline for building MLLMs — pre-training aligns visual-language features, and the fine-tuning stage trains end-to-end on instruction data. The conventional practice computes autoregressive loss only on the response sequence, masking out the instruction sequence from loss computation.

**Limitations of Prior Work**: VIT is susceptible to overfitting and shortcut learning — models may ignore visual content and rely solely on language priors to produce plausible-sounding answers. For instance, language-only models can answer many VQA questions without any image input.

**Key Challenge**: Computing loss exclusively on responses trains the model to learn "how to follow the instruction format to answer" rather than "how to understand image content." This overemphasizes instruction-following while neglecting active visual comprehension.

**Goal**: To improve MLLMs' utilization of visual information and reduce dependence on language shortcuts with minimal modification.

**Key Insight**: If a model is also required to predict the instruction itself (e.g., "Describe this image"), it must understand the image content to know what instruction is appropriate — thereby compelling the model to attend more closely to visual inputs.

**Core Idea**: Extend the loss mask from responses only to instructions + responses, enabling the model to jointly learn "what to ask" and "how to answer" — a zero-cost regularization strategy.

## Method

### Overall Architecture
Built upon the LLaVA architecture (visual encoder + connector + LLM), L2T extends the loss function during the fine-tuning stage from the response sequence alone to the instruction + response sequence. The pre-training stage remains unchanged, as pre-training instructions are fixed templates unrelated to image content.

### Key Designs

1. **L2T Loss Extension**:

    - Function: Extends the training loss from predicting responses only to jointly predicting instructions and responses.
    - Standard VIT loss: $\mathcal{L} = -\sum_{i=1}^{L_A} \log p_\theta(\mathbf{X}_{A,i}|\mathbf{X}_V, \mathbf{X}_I, \mathbf{X}_{A,<i})$
    - L2T loss: $\mathcal{L} = \underbrace{-\sum_{i=1}^{L_I} \log p_\theta(\mathbf{X}_{I,i}|\mathbf{X}_V, \mathbf{X}_{I,<i})}_{\text{Learn to Instruct}} \underbrace{-\sum_{i=1}^{L_A} \log p_\theta(\mathbf{X}_{A,i}|\mathbf{X}_V, \mathbf{X}_I, \mathbf{X}_{A,<i})}_{\text{Learn to Respond}}$
    - Design Motivation: Learning to generate instructions forces the model to attend to image content (knowing what is in the image is necessary to know what to ask), while implicitly expanding the effective training signal.

2. **Template Removal**:

    - Function: Excludes image-irrelevant template tokens from the instruction loss.
    - Two categories of templates: (a) system templates — role definitions, USER/ASSISTANT markers, etc.; (b) task templates — high-frequency, low-information task-type indicators (e.g., fixed phrases such as "Describe the image").
    - Identification method: Sentence frequency statistics are computed over the training set; the highest-frequency sentences are identified and excluded as task templates.
    - Design Motivation: Restricts learning to meaningful, image-relevant instruction tokens, preventing the model from learning irrelevant formatting patterns.

3. **Visual Contribution (VC) Analysis**:

    - Function: Quantifies the actual contribution of visual input to response prediction.
    - Formula: $\text{VC} = \log p_\theta(\mathbf{X}_A|\mathbf{X}_V, \mathbf{X}_I) - \log p_\theta(\mathbf{X}_A|\mathbf{X}_V=\emptyset, \mathbf{X}_I)$
    - Finding: L2T improves VC by 9% over standard VIT, confirming that the model indeed relies more on visual input.
    - Attention visualization: L2T exhibits stronger attention weights on visual tokens.

### Loss & Training
L2T is applied exclusively during the fine-tuning stage (pre-training instructions are fixed templates containing no image-relevant information). The method strictly follows the training recipes of TinyLLaVA / LLaVA 1.5 / LLaVA-NeXT, with the only modification being the loss masking strategy.

## Key Experimental Results

### Main Results
Evaluation across 16 benchmarks on 5 model architectures (relative improvement):

| Model | General VQA | Comprehensive Benchmarks | Chart/Doc/OCR | Captioning | Overall |
|-------|------------|--------------------------|---------------|------------|---------|
| TinyLLaVA-0.5B | +1.5% | +1.8% | +8.8% | +17.6% | +6.2% |
| TinyLLaVA-3B | +0.5% | -0.4% | +3.5% | +3.4% | +1.4% |
| LLaVA-1.5-7B | +0.7% | +1.5% | +5.8% | +8.2% | +3.5% |
| LLaVA-1.5-13B | +0.2% | +0.6% | +4.3% | +5.3% | +2.2% |
| LLaVA-NeXT-7B | +1.1% | +4.0% | +8.2% | +7.2% | +4.5% |

Captioning yields the largest gains (up to 17.6%), as description tasks most heavily demand visual understanding.

### Ablation Study

| Configuration | Effect |
|---------------|--------|
| Full instruction loss (including templates) | Improvement, but inferior to template removal |
| L2T + template removal | Optimal |
| L2T applied at pre-training stage only | Ineffective (pre-training instructions are fixed templates) |
| L2T applied at fine-tuning stage only | Effective (fine-tuning instructions are image-relevant) |

### Key Findings
- **Smaller models benefit more**: TinyLLaVA-0.5B gains 6.2% vs. 2.2% for the 13B model — smaller models are more prone to overfitting language priors.
- **Captioning and OCR tasks benefit most**: These tasks most require the model to "genuinely look at the image" rather than rely on language priors.
- Hallucination benchmarks (POPE, CHAIR, etc.) also show significant improvement — reduced language prior dependence directly alleviates hallucination.
- Visual Contribution improves by 9% — providing quantitative evidence that the model makes greater use of visual information.
- Attention weight visualizations show stronger activation on visual tokens under L2T.

## Highlights & Insights
- **Extreme simplicity**: The sole modification is the loss masking strategy — changing from masking out instructions to including them in the loss. This "single-line" change yielding substantial gains reveals a fundamental design flaw in standard VIT.
- **Regularization perspective**: The instruction loss acts as a regularizer, implicitly expanding the effective training signal (from $L_A$ tokens to $L_I + L_A$ tokens), while enforcing visual attention — analogous to multi-task learning regularization.
- **Importance of template removal**: Not all instruction tokens are informative — removing image-irrelevant template tokens further improves performance, highlighting the importance of precisely defining "what constitutes meaningful learning content."
- **General transferability**: The method is orthogonal to model architecture and can be directly applied to any VIT framework.

## Limitations & Future Work
- The optimal weighting between instruction and response losses (i.e., $\lambda_I \mathcal{L}_I + \lambda_A \mathcal{L}_A$) remains unexplored.
- A thorough theoretical explanation of why this simple change is so effective remains insufficient — whether the mechanism is regularization, implicit data augmentation, or both warrants further investigation.
- The effectiveness of instruction loss in multi-turn dialogue settings requires further validation, as instructions in such settings tend to be more complex and context-dependent.
- Finer-grained token-level importance weighting (beyond the binary template/non-template distinction) could be explored — e.g., weighting each instruction token by its degree of visual relevance.
- Generalization to video MLLMs and larger-scale models (e.g., 70B+) remains to be verified.

## Related Work & Insights
- **vs. Standard VIT (LLaVA)**: LLaVA computes loss on responses only; L2T extends this to instructions + responses. The two approaches are orthogonal and fully compatible.
- **vs. Other hallucination mitigation methods**: Most prior methods require additional data or specialized training strategies; L2T achieves improvements at zero cost.
- **Relation to curriculum learning / data augmentation**: L2T can be viewed as implicit data augmentation — learning more from the same data — analogous to the regularization effect of multi-task learning.

## Rating
- Novelty: ⭐⭐⭐⭐ A minimalist modification (only the loss mask is changed) yields 9% relative improvement, exposing a fundamental design flaw in VIT.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 model architectures + 16 benchmarks + ablations + quantitative VC analysis + attention visualization.
- Writing Quality: ⭐⭐⭐⭐ Concise and compelling; motivation is clearly articulated; the VC metric provides a novel tool for quantifying visual utilization.
- Value: ⭐⭐⭐⭐⭐ A zero-cost improvement to VLM training that can be directly integrated into any VIT framework.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Visual Instruction Bottleneck Tuning](visual_instruction_bottleneck_tuning.md)
- [\[NeurIPS 2025\] CoIDO: Efficient Data Selection for Visual Instruction Tuning via Coupled Importance-Diversity Optimization](coido_efficient_data_selection_for_visual_instruction_tuning_via_coupled_importa.md)
- [\[ICCV 2025\] SMoLoRA: Exploring and Defying Dual Catastrophic Forgetting in Continual Visual Instruction Tuning](../../ICCV2025/multimodal_vlm/smolora_exploring_and_defying_dual_catastrophic_forgetting_in_continual_visual_i.md)
- [\[ICCV 2025\] orderchain towards general instruct-tuning for stimulating the ordinal understan](../../ICCV2025/multimodal_vlm/orderchain_towards_general_instruct-tuning_for_stimulating_the_ordinal_understan.md)
- [\[ICCV 2025\] DWIM: Towards Tool-aware Visual Reasoning via Discrepancy-aware Workflow Generation & Instruct-Masking Tuning](../../ICCV2025/multimodal_vlm/dwim_towards_tool-aware_visual_reasoning_via_discrepancy-aware_workflow_generati.md)

<!-- RELATED:END -->
