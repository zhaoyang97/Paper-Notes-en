---
title: >-
  [Paper Note] LamRA: Large Multimodal Model as Your Advanced Retrieval Assistant
description: >-
  [CVPR 2025][Multimodal VLM][Multimodal Retrieval] Transforms a generative Large Multimodal Model (LMM) into a general multimodal retriever and reranker. By utilizing a two-stage training process (language pre-training and multimodal instruction tuning) along with joint pointwise/listwise reranking training, introducing lightweight LoRA modules allows it to significantly outperform dual-encoder approaches across 16 retrieval tasks and show strong generalization capability on 1…
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Multimodal Retrieval"
  - "Large Multimodal Models"
  - "General Retrieval"
  - "Reranking"
  - "LoRA Fine-Tuning"
date: 2026-05-08
content_hash: 963292f380a1d04f
---

# LamRA: Large Multimodal Model as Your Advanced Retrieval Assistant

**Conference**: CVPR 2025  
**arXiv**: [2412.01720](https://arxiv.org/abs/2412.01720)  
**Code**: [https://github.com/Code-kunkun/LamRA](https://github.com/Code-kunkun/LamRA)  
**Area**: Information Retrieval  
**Keywords**: Multimodal Retrieval, Large Multimodal Models, General Retrieval, Reranking, LoRA Fine-Tuning

## TL;DR
Transforms a generative Large Multimodal Model (LMM) into a general multimodal retriever and reranker. By utilizing a two-stage training process (language pre-training and multimodal instruction tuning) along with joint pointwise/listwise reranking training, introducing lightweight LoRA modules allows it to significantly outperform dual-encoder approaches across 16 retrieval tasks and show strong generalization capability on 10 unseen datasets.

## Background & Motivation

**Background**: Multimodal information retrieval is becoming increasingly complex—expanding from simple text-to-image retrieval to composed image retrieval (text + image to image), long text-to-image retrieval, and multimodal document retrieval. Traditional approaches (e.g., CLIP, ALIGN) based on dual-encoders and contrastive learning excel at cross-modal retrieval.

**Limitations of Prior Work**: (1) Dual-encoder approaches lack deep understanding of complex queries—CLIP's text encoder has limited comprehension of long texts and complex compositional instructions; (2) Existing methods require task-specific fine-tuning, failing to unify various retrieval formats under a single framework; (3) Few studies explore using LMMs for general retrieval—since LMMs are trained for generative tasks, directly applying them to retrieval tasks yields poor performance (Qwen2-VL-7B achieves only 23.0 on M-BEIR).

**Key Challenge**: While LMMs possess powerful multimodal and natural language understanding capabilities, there is a fundamental discrepancy between their generative training objective (next-token prediction) and the retrieval objective (embedding similarity ranking).

**Goal**: How to transform a generative LMM into a general multimodal retriever and reranker with minimal modifications?

**Key Insight**: Adapting LMM outputs using LoRA modules, and utilizing a progressive two-stage training scheme that first establishes a foundation for retrieval before expanding to multimodal tasks.

**Core Idea**: Utilizing an "adaptation + progressive training" strategy to leverage the LMM's final-layer hidden state as a unified multimodal embedding, enabling unified processing of over 10 different retrieval tasks.

## Method

### Overall Architecture
LamRA = LamRA-Ret (Retrieval) + LamRA-Rank (Reranking). In the retrieval module, a LoRA module is inserted into the LMM, and the embedding is obtained via the prompt "Summarize in one word: `<emb>`" through a two-stage training process. In the reranking module, another LoRA module is trained to support both pointwise (YES/NO judgment) and listwise (directly outputting target position indices) reranking. During inference, LamRA-Ret is first used to retrieve top-$K$ candidates, which are then reranked by LamRA-Rank.

### Key Designs

1. **Explicit One-word Limit (EOL) Feature Extraction**:
    - **Function**: Transforms the generative outputs of LMMs into embeddings suitable for retrieval.
    - **Mechanism**: Employs three prompt templates—using "`<image>` Summarize above image in one word: `<emb>`" for image inputs, "`<text>` Summarize above sentence in one word: `<emb>`" for text inputs, and corresponding combined templates for mixed inputs. The final hidden state corresponding to the token preceding `<emb>` is extracted as the embedding.
    - **Design Motivation**: The output distribution of generative LMMs differs substantially from the embedding space required for retrieval. Directing the model with an explicit summarization prompt guides it to compress multimodal information into a single hidden state, acting as a "task-switching" signal for the LMM.

2. **Two-Stage Progressive Training (LamRA-Ret)**:
    - **Function**: Establishes the retrieval capability of the LMM from scratch.
    - **Mechanism**: **Stage-I** performs text-only contrastive pre-training on NLI datasets, teaching the model to output meaningful embeddings (instead of next-token probability distributions). **Stage-II** conducts instruction tuning across 8 retrieval tasks in the M-BEIR dataset, using task-specific instructions (e.g., "retrieve a similar image") to guide the model in understanding diverse retrieval intents. Both stages utilize the InfoNCE contrastive loss: $$\mathcal{L}_{ret} = -\frac{1}{B}\sum_{n=1}^B \log \frac{\exp[\kappa(\text{LMM}(q_n), \text{LMM}(c_n))/\tau]}{\sum_{m=1}^B \exp[\kappa(\text{LMM}(q_n), \text{LMM}(c_m))/\tau]}$$
    - **Design Motivation**: Ablation results show that omitting the pre-training stage drops performance by 3 points (from 56.6 to 53.6), while omitting instruction tuning is even worse (dropping to 36.2). Both stages are indispensable; pre-training addresses the fundamental problem of "how to output embeddings", while instruction tuning tackles the extension problem of "how to understand diverse retrieval tasks".

3. **Joint Pointwise and Listwise Reranking (LamRA-Rank)**:
    - **Function**: Further refines ranking quality based on initial retrieval results.
    - **Mechanism**: Trains another LoRA module using hard negatives mined from the top-100 candidates produced by LamRA-Ret. **Pointwise Reranking**: Independently evaluates each candidate as YES/NO, with loss $\mathcal{L}_{point} = \mathcal{L}_{ce}(\text{YES}, \text{Reranker}(q, c_{pos})) + \mathcal{L}_{ce}(\text{NO}, \text{Reranker}(q, c_{neg}))$. **Listwise Reranking**: Randomly inserts the positive sample among 2-5 negative samples and requires the model to directly output the position index of the positive sample. The final fused score is $S = \alpha \cdot S_{ret} + (1-\alpha) \cdot S_{rank}$ with $\alpha=0.5$.
    - **Design Motivation**: Pointwise reranking offers high accuracy but incurs heavy computational overhead ($K$ inference passes), whereas listwise reranking is efficient but constrained by the LLM's context length. Joint training equips the model with both capabilities, allowing developers to choose based on deployment constraints. Reranking yields an average improvement of 7.1 points.

### Loss & Training
- Retrieval stage: InfoNCE contrastive loss with temperature parameter $\tau$.
- Reranking stage: Cross-entropy loss, $\mathcal{L}_{rank} = \mathcal{L}_{point} + \mathcal{L}_{list}$.
- Hardware configuration: Pre-training on 8×A100 (batch=576, lr=4e-5, 2 epochs); instruction tuning on 16×A100 (batch=960, lr=1e-4, 1 epoch); reranking on 16×A100 (batch=64, lr=4e-5, 1 epoch).
- Vision encoder parameters are frozen throughout; only the LLM is fine-tuned using LoRA.

## Key Experimental Results

### Main Results (Average Recall@5 across 16 tasks on M-BEIR)

| Method | Type | Avg R@5 | Gain |
|------|------|---------|------|
| Qwen2-VL-7B (Zero-shot) | LMM | 23.0 | - |
| UniIR-CLIP-SF | Dual-encoder | 50.6 | +27.6 |
| LamRA-Ret | LMM+LoRA | 56.6 | +33.6 |
| **LamRA** | LMM+LoRA+Reranking | **63.7** | **+40.7** |

### Zero-shot Generalization on Unseen Datasets (R@1)

| Dataset | Task | LamRA | EVA-CLIP-18B | Long-CLIP-L | E5-V |
|--------|------|-------|-------------|-------------|------|
| ShareGPT4V | T→I | **97.9** | 92.1 | 95.6 | 86.7 |
| Urban-1K | T→I | **98.8** | 81.7 | 86.1 | 84.0 |
| CIRCO | (I,T)→I | **42.8** | 6.0 | 5.7 | 24.8 |
| Visual Dialog | Dialog→I | **70.9** | 24.7 | 37.9 | 54.6 |
| MSR-VTT (Video) | T→V | 44.7 | - | - | - |

### Ablation Study

| Configuration | M-BEIR Avg | Description |
|------|------------|------|
| Full Two-Stage | 56.6 | Baseline |
| W/o Pre-training | 53.6 (-3.0) | Lacks foundational embedding capability |
| W/o Instruction Tuning | 36.2 (-20.4) | Fails to comprehend diverse retrieval tasks |
| W/o Both | 23.0 (-33.6) | Vanilla LMM lacks retrieval capacity |
| Qwen2-VL-2B Version | 51.6/58.3 | Larger models exhibit better performance |
| Qwen2-VL-7B Version | 56.6/63.7 | - |

### Key Findings
- **LMM as a retriever fully outperforms dual-encoders**: On M-BEIR, LamRA-Ret (56.6) already surpasses UniIR-CLIP (50.6), and achieves 63.7 after reranking.
- **Substantial advantage in complex retrieval tasks**: On the text+image-to-text retrieval task of the InfoSeek dataset, LamRA-Ret outperforms UniIR-CLIP by 24.2 percentage points.
- **Surprising zero-shot generalization**: Achieves a R@1 of 98.8 on Urban-1K (long text-to-image), exceeding EVA-CLIP-18B by 17 percentage points.
- **Consistent improvement from reranking**: Performance increases across all 16 M-BEIR retrieval tasks, with an average gain of +7.1 points.
- **Feasible zero-shot video retrieval**: Reaches R@1 of 44.7 on MSR-VTT (outperforming InternVideo by 4.7 points) despite the model never having seen any video training data.

## Highlights & Insights
- **The "LMM+LoRA for retrieval" paradigm is highly practical**—It requires no architecture modifications, and inserting lightweight LoRA is sufficient to equip general LMMs with SOTA retrieval capabilities.
- **Clear logic behind the two-stage training scheme**—It first "repairs" the embedding space via text contrastive pre-training, then "extends" task support using multimodal data.
- **Joint pointwise and listwise reranking offers deployment flexibility**—Accuracy-critical scenarios can choose pointwise, whereas speed-sensitive applications can choose listwise.
- **Massive improvement from 23.0 to 63.7 on M-BEIR** (+40.7 points), showcasing the immense potential of LMMs as solvers for retrieval tasks.

## Limitations & Future Work
- High computational overhead in LMM inference—Computing embeddings for every candidate requires full forward passes, making it unfriendly for large-scale candidate sets.
- Frozen vision encoder throughout—This might limit the model's perception of fine-grained visual details.
- Additional LoRA modules and training are needed for reranking—increasing the overall system complexity.
- Evaluation is limited to Qwen2-VL—It remains to be verified whether this generalizes well to other LMMs (e.g., LLaVA, InternVL).

## Related Work & Insights
- **vs UniIR (CLIP/BLIP)**: Dual-encoder methods that unify multiple tasks but are limited by CLIP's text comprehension. LamRA leverages the superior language understanding of LLMs to significantly outperform them on complex queries.
- **vs E5-V**: Also applies LMMs to retrieval, but relies only on text-only fine-tuning, resulting in limited performance on complex multimodal tasks. LamRA's two-stage training and instruction tuning are more comprehensive.
- **vs GENIUS**: A generative retrieval method with high efficiency but inferior accuracy compared to embedding-based retrieval. LamRA follows the embedding-based paradigm but replaces CLIP with an LMM as the encoder.

## Rating
- Novelty: ⭐⭐⭐⭐ Utilizing LMMs for general retrieval is not entirely new (with E5-V as prior work), but the systematic design of two-stage training plus joint reranking is highly mature.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive evaluation, covering 16 M-BEIR tasks, 10 unseen datasets, video retrieval, and thorough ablations.
- Writing Quality: ⭐⭐⭐⭐ The methodology is clearly described, ablation analysis is highly convincing, and the numerous tables are well-organized.
- Value: ⭐⭐⭐⭐⭐ Demonstrates the feasibility of LMMs as general retrievers, providing direct reference value for industrial multimodal search systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] CoLLM: A Large Language Model for Composed Image Retrieval](collm_a_large_language_model_for_composed_image_retrieval.md)
- [\[CVPR 2025\] Your Large Vision-Language Model Only Needs a Few Attention Heads for Visual Grounding](your_large_vision-language_model_only_needs_a_few_attention_heads_for_visual_gro.md)
- [\[CVPR 2025\] VideoGLaMM: A Large Multimodal Model for Pixel-Level Visual Grounding in Videos](videoglamm_a_large_multimodal_model_for_pixel-level_visual_grounding_in_videos.md)
- [\[CVPR 2025\] RAP: Retrieval-Augmented Personalization for Multimodal Large Language Models](rap_retrieval-augmented_personalization_for_multimodal_large_language_models.md)
- [\[CVPR 2026\] ROSE: Rotate Your Large Language Model to See](../../CVPR2026/multimodal_vlm/rose_rotate_your_large_language_model_to_see.md)

</div>

<!-- RELATED:END -->
