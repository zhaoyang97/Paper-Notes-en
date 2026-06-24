---
title: >-
  [Paper Note] SymDPO: Boosting In-Context Learning of Large Multimodal Models with Symbol Demonstration Direct Preference Optimization
description: >-
  [CVPR 2025][Multimodal VLM][In-Context Learning] SymDPO identifies the "visual context overlook" issue in multimodal ICL (where replacing demonstration images with blank images does not affect performance) and proposes replacing text answers in demonstrations with semantic-free random symbols. This forces the model to understand the visual content to correctly match symbols with answers. Through DPO training, this consistently improves multimodal ICL performance on OpenFlamin…
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "In-Context Learning"
  - "Symbol Demonstration"
  - "DPO"
  - "Large Multimodal Models"
  - "Visual Context Utilization"
date: 2026-05-08
content_hash: db6a16b455abdac0
---

# SymDPO: Boosting In-Context Learning of Large Multimodal Models with Symbol Demonstration Direct Preference Optimization

**Conference**: CVPR 2025  
**arXiv**: [2411.11909](https://arxiv.org/abs/2411.11909)  
**Code**: [https://github.com/APiaoG/SymDPO](https://github.com/APiaoG/SymDPO)  
**Area**: RLHF Alignment  
**Keywords**: In-Context Learning, Symbol Demonstration, DPO, Large Multimodal Models, Visual Context Utilization

## TL;DR

SymDPO identifies the "visual context overlook" issue in multimodal ICL (where replacing demonstration images with blank images does not affect performance) and proposes replacing text answers in demonstrations with semantic-free random symbols. This forces the model to understand the visual content to correctly match symbols with answers. Through DPO training, this consistently improves multimodal ICL performance on OpenFlamingo and IDEFICS.

## Background & Motivation

**Background**: Large language models have demonstrated powerful in-context learning (ICL) capabilities, enabling them to solve new tasks with only a few examples. Researchers have extended ICL to Large Multimodal Models (LMMs), expecting these models to learn task patterns from multimodal (image-text) demonstrations.

**Limitations of Prior Work**: LMMs suffer from a severe "Visual Context Overlook" issue in multimodal ICL. When demonstration images are replaced with blank placeholders or completely removed, model performance is practically unaffected. This indicates that models actually rely on pattern matching of text templates rather than sincerely utilizing the visual information.

**Key Challenge**: Most VQA tasks can be solved relying solely on text pattern matching (e.g., choosing the most frequent answer in yes/no questions), allowing models to "slack off" and ignore the images. Existing DPO methods are optimized for general tasks and are not specialized for ICL scenarios, failing to address this fundamental lack of vision-text alignment.

**Goal**: How to force LMMs to genuinely utilize visual information during ICL, rather than relying solely on text patterns?

**Key Insight**: An ingenious "reverse design"—if the model circumvents visual understanding through text pattern matching, the text answers can be replaced with semantic-free random symbols, leaving no option for the model but to look at the images.

**Core Idea**: Replace the answers in ICL demonstrations with meaningless symbols (e.g., using "rhondda" to replace "narrow"), forcing the model to establish image-symbol mappings to answer questions, rather than relying on text-to-text pattern matching.

## Method

### Overall Architecture

The pipeline of SymDPO consists of three steps: (1) construct ICL-format data $D_1, D_2, \ldots, D_N, F$ ($N$ demonstrations + 1 final question) from VQA datasets; (2) construct standard DPO positive/negative pairs; (3) core innovation—replace demonstration answers with semantic-free symbols in DPO data to construct SymDPO training data. During inference, symbols are not used, and the model performs ICL normally.

### Key Designs

1. **ICL Data Construction and Grouping**:

    - **Function**: Construct structured multimodal ICL training data.
    - **Mechanism**: Collect image-question-answer triplets from GQA, VQAv2, and ImageNet, and group questions with similar task types (e.g., binary yes/no questions, color attribute questions, object counting questions). Each group contains $N$ demonstrations and 1 final question-answer pair, ensuring that demonstrations contain at least 2 different answers and at least one matches the final answer.
    - **Design Motivation**: Ensure sufficient diversity among demonstrations to prevent the model from solving tasks through simple majority voting.

2. **Symbol Demonstration Strategy**:

    - **Function**: Eliminate the possibility of text pattern matching, forcing visual understanding.
    - **Mechanism**: Replace the answer $A_i$ of each demonstration with a semantically unrelated random symbol $S_i$ (e.g., "rhondda", "odwyer"), obtaining $\dot{D}_i = \{I_i, Q_i, S_i\}$. Key constraint: the symbol $S_k$ of a certain demonstration $D_k$ matches the semantic category corresponding to the final answer. The positive sample for DPO is $S_k$, and the negative sample is a non-matching symbol $S_j$. The paper maintains five different symbol configuration ratios, including variants like standard replacement, erasing demonstration questions, etc.
    - **Design Motivation**: Since symbols contain no semantic information, the model must genuinely understand the visual content of the demonstration image to map the correct symbol to the visual answer, thereby establishing true joint visual-semantic binding.

3. **SymDPO Training Objective**:

    - **Function**: Train the model to utilize visual context through preference learning.
    - **Mechanism**: The standard DPO objective is:
      $$\mathcal{L}_S = -\mathbb{E} \log \sigma \left( \beta \log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right)$$
      Notably, training data uses symbolized demonstrations to force the model to reinforce visual-symbol alignment during preference learning.
    - **Design Motivation**: Directly performing Supervised Fine-Tuning (SFT, termed SymTune) on symbol data yields poor or even detrimental results because autoregressive training may learn incorrect generation patterns. DPO provides more robust feedback signals through contrastive preference learning.

### Loss & Training

Standard DPO loss is used. From a dataset of 872,000 samples, 10,000 samples are selected for training, with quality enhancement performed using GPT-4V. Training takes about 1 hour on 8×A100. Learning rate is 5e-6, with linear decay. No symbols are needed during inference, enabling normal ICL.

## Key Experimental Results

### Main Results

OpenFlamingo-9B results:

| Setting | Method | COCO (CIDEr) | Flickr-30K | VQAv2 | OK-VQA | TextVQA |
|------|------|-------------|------------|-------|--------|---------|
| 4-shot | Base | 89.0 | 65.8 | 51.0 | 40.1 | 26.1 |
| 4-shot | General DPO | 89.6 | 66.0 | 51.2 | 40.5 | 26.2 |
| 4-shot | **SymDPO** | **93.8** | **69.4** | 51.1 | **41.0** | 26.3 |
| 8-shot | Base | 96.3 | 62.9 | 54.8 | 41.1 | 27.3 |
| 8-shot | **SymDPO** | **102.5** | **67.3** | 55.0 | **42.3** | 27.7 |
| 16-shot | Base | 98.8 | 62.8 | 56.1 | 42.7 | 27.6 |
| 16-shot | **SymDPO** | **104.3** | **64.9** | 56.4 | **44.5** | 28.2 |

IDEFICS-9B results:

| Setting | Method | COCO (CIDEr) | Flickr-30K | OK-VQA |
|------|------|-------------|------------|--------|
| 8-shot | Base | 97.0 | 61.9 | 47.7 |
| 8-shot | **SymDPO** | **103.8** | **66.1** | **49.5** |
| 16-shot | Base | 99.7 | 64.5 | 48.4 |
| 16-shot | **SymDPO** | **107.9** | **69.3** | **50.6** |

### Ablation Study

| Method | COCO 4-shot | Flickr 4-shot | OK-VQA 4-shot |
|------|------------|---------------|---------------|
| SymTune (SFT) | +7.8 | -5.2 | +0.3 |
| General DPO | +0.6 | +0.2 | +0.4 |
| **SymDPO** | **+4.7** | **+2.1** | **+1.0** |

Superimposition effect of SymDPO + RICES (retrieval-based demonstration selection):

| Configuration | COCO 4-shot |
|------|------------|
| Base | 82.7 |
| + RICES | 90.5 (+7.8) |
| + SymDPO | 87.4 (+4.7) |
| **+ SymDPO & RICES** | **93.5 (+10.8)** |

### Key Findings

- **Visual context overlook is real**: After replacing demonstration images with blank/no images, the performance of baseline models remains almost unchanged; meanwhile, the performance of SymDPO-trained models drops significantly when using blank images, proving that the model indeed learns to utilize visual information.
- **SymDPO vs SymTune**: Direct SFT using symbol data leads to severe degradation in captioning (Flickr -5.2), whereas the preference learning approach of DPO exerts a more robust impact on model knowledge.
- **Complementary to demonstration selection strategies**: The gains from SymDPO and RICES are additive (4.7 + 7.8 $\rightarrow$ 10.8), suggesting that they address different dimensions of the problem.
- **Symbol data ratio**: A symbol data ratio of 70-100% yields the best performance, indicating that training with symbols is the primary driver of improved ICL.

## Highlights & Insights

- **The discovery of "visual context overlook" is highly valuable in itself**: A simple blank image experiment reveals a fundamental flaw in LMMs, providing a clear diagnostic tool for subsequent research.
- **The concept of symbol replacement is extraordinarily simple and elegant**: It requires no complex model modifications or external modules; addressing the core issue is achieved solely through the clever design of training data. The core insight is: if a model can exploit a shortcut to solve a task, block that shortcut.
- **Generalization to multiple model architectures**: Effective across both OpenFlamingo and IDEFICS architectures, indicating that visual context overlook is an inherent problem rather than a defect specific to certain architectures.

## Limitations & Future Work

- Evaluated only on two architectures (OpenFlamingo and IDEFICS), without validation on more advanced models like LLaVA or GPT-4V.
- Minimal improvement observed on TextVQA (0.2-0.6), indicating that certain visual-text tasks have less demand for visual-symbol alignment.
- The choice of symbols appears arbitrary ("rhondda", "odwyer"); there is no systematic study on how symbol properties affect performance.
- Requires GPT-4V for data quality enhancement, which limits the reproducibility and scalability of the methodology.

## Related Work & Insights

- **vs General DPO**: General DPO yields almost no improvement in ICL tasks (COCO +0.6), demonstrating that standard preference data cannot solve the visual overlook problem.
- **vs MIA-DPO**: Another multimodal DPO variant, which also yields limited gains (+0.2-1.8) compared to the targeted design of SymDPO.
- **vs Symbol Tuning (NeurIPS 2023)**: The source of inspiration; however, direct SFT performs poorly. SymDPO combines the symbol concept with DPO, safeguarding model capability more effectively under a preference learning framework.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The idea of symbol replacement is highly concise yet profound, and the shortcut-blocking approach has broad instructive significance.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-model + multi-benchmark + extensive ablations, though wider coverage of architectures would be beneficial.
- **Writing Quality**: ⭐⭐⭐⭐ Clear presentation of problem motivation (the blank image experiment) and overall logical flow.
- **Value**: ⭐⭐⭐⭐ Reveals a fundamental flaw in LMM ICL and provides an elegant solution, strongly pushing forward research in multimodal ICL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Debiasing Multimodal Large Language Models via Noise-Aware Preference Optimization](debiasing_multimodal_large_language_models_via_noise-aware_preference_optimizati.md)
- [\[CVPR 2025\] Task Preference Optimization: Improving Multimodal Large Language Models with Vision Task Alignment](task_preference_optimization_improving_multimodal_large_language_models_with_vis.md)
- [\[CVPR 2025\] Mimic In-Context Learning for Multimodal Tasks](mimic_in-context_learning_for_multimodal_tasks.md)
- [\[ICML 2026\] TUR-DPO: Topology- and Uncertainty-Aware Direct Preference Optimization](../../ICML2026/multimodal_vlm/tur-dpo_topology-_and_uncertainty-aware_direct_preference_optimization.md)
- [\[CVPR 2025\] HiFICL: High-Fidelity In-Context Learning for Multimodal Tasks](hificl_high-fidelity_in-context_learning_for_multimodal_tasks.md)

</div>

<!-- RELATED:END -->
