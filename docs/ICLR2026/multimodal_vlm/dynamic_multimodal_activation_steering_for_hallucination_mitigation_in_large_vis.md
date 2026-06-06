---
title: >-
  [Paper Note] Dynamic Multimodal Activation Steering for Hallucination Mitigation in Large Vision-Language Models
description: >-
  [ICLR 2026][Multimodal VLM][hallucination mitigation] This paper proposes Dynamic Multimodal Activation Steering (DMAS), a training-free method that constructs a semantics-based truthfulness steering vector database and…
tags:
  - "ICLR 2026"
  - "Multimodal VLM"
  - "hallucination mitigation"
  - "activation engineering"
  - "attention head intervention"
  - "training-free method"
  - "large vision-language models"
date: 2026-05-08
content_hash: f465a3cf52667a2b
---

# Dynamic Multimodal Activation Steering for Hallucination Mitigation in Large Vision-Language Models

**Conference**: ICLR 2026
**arXiv**: [2602.21704](https://arxiv.org/abs/2602.21704)  
**Code**: None  
**Area**: Multimodal VLM
**Keywords**: hallucination mitigation, activation engineering, attention head intervention, training-free method, large vision-language models

## TL;DR
This paper proposes Dynamic Multimodal Activation Steering (DMAS), a training-free method that constructs a semantics-based truthfulness steering vector database and a visual perception steering vector, dynamically selecting the most relevant steering vectors at inference time to intervene on critical attention heads. DMAS significantly mitigates hallucinations in LVLMs, achieving a gain of 94.66 points on MME and a 20.2% reduction in hallucination rate on CHAIR.

## Background & Motivation
Large vision-language models (LVLMs) achieve strong performance on tasks such as VQA and image captioning, but suffer from severe hallucination problems—fabricating non-existent objects or incorrectly describing image content. Existing approaches fall into two categories: training-based methods require carefully annotated data and substantial computational resources (e.g., LRV, RLHF-V), while decoding-based methods (e.g., VCD, ICD) avoid training but often degrade generation quality.

Recent activation engineering approaches (e.g., ICT, VTI) attempt to reduce hallucinations by intervening in the model's internal representations, but exhibit critical limitations: ICT focuses solely on visual-level intervention, ignoring the multimodal nature of LVLMs; VTI employs fixed steering vectors, overlooking the variation of steering vectors across different semantic contexts.

**Core Findings**: Through analysis of LLaVAv1.5's attention patterns, the authors identify two key phenomena: (1) truthfulness and visual perception capabilities primarily activate different subsets of attention heads (truthfulness concentrated in layer 30, visual perception in layer 31); (2) truthfulness steering vectors vary substantially across different semantic contexts (t-SNE visualizations reveal clearly separated semantic clusters). These two findings directly motivate the design of DMAS.

## Method

### Overall Architecture
DMAS operates in three steps: (1) constructing a dynamic truthfulness steering vector database; (2) computing visual perception steering vectors; (3) applying dynamic intervention on different attention heads at inference time. The entire method requires no training and operates in a plug-and-play manner.

### Key Designs
1. **Truthfulness Steering Vector Database**: Samples from the AMBER and SEED datasets are semantically clustered into 4 clusters. For each sample, correct/hallucinated answer pairs are constructed and fed into the LVLM to extract activations $A_{pos}$ and $A_{neg}$ at the last token position across attention heads in each layer. The intra-cluster average activation difference forms the steering vector: $D_i = \frac{1}{|C_i|}\sum_{j \in C_i}(A_{pos,j} - A_{neg,j})$, with PCA applied for denoising. A Key-Value database is constructed using the mean cluster embedding as the Key and the steering vector as the Value. At inference time, a sentence transformer computes semantic similarity between the input and each Key to dynamically retrieve the most relevant steering vector.

2. **Visual Perception Steering Vector**: Given an original image $V$ and a noise-perturbed image $V'$ (obtained via the forward diffusion process), YOLOv11 detects objects to generate description templates. The activation difference between the original input $(V, T+Y_O)$ and the perturbed input $(V', T+Y_{O'})$ is computed as: $D_v = A_v - A_{v'}$, with PCA applied to extract principal components. This design enhances the model's attention to visual information.

3. **Dynamic Inference Intervention**: At inference time, the semantically most relevant truthfulness steering vector $D_f$ is retrieved via cosine similarity, and binary masks $M_f$ and $M_v$ are constructed to apply intervention only to the Top-K attention heads with the largest activation differences. The modified attention computation is: $\mathbf{x}^{(l+1)} = \mathbf{x}^{(l)} + \text{Concat}[\text{Attn}^{(l,h)}(\mathbf{x}^{(l)}) + \alpha \cdot M_f^{(l,h)} \cdot D_f^{(l,h)} + \beta \cdot M_v^{(l,h)} \cdot D_v^{(l,h)}] \cdot \mathbf{W}_o^{(l)}$, where $\alpha$ and $\beta$ control intervention strength.

### Loss & Training
No training is required. Hyperparameters $\alpha, \beta \in \{0.5, 1, ..., 10\}$ and $K \in \{32, 64, ..., 1024\}$ are determined via grid search. Temperature is set to 0 and top_p to 1.

Notable implementation details:
- Key embeddings in the steering vector database are obtained via a sentence transformer (all-mpnet-base-v2)
- YOLOv11 detects objects in the image for the visual perception component; objects from the same category not present in the image are randomly sampled from a category bank as contrast
- PCA denoising is applied separately to truthfulness and visual perception steering vectors to extract the most salient principal components
- All experiments are conducted on an NVIDIA RTX 4090 (48GB) GPU

## Key Experimental Results

### Main Results

| Model | Method | Existence↑ | Count↑ | Position↑ | Color↑ | Total↑ |
|-------|--------|-----------|--------|-----------|--------|--------|
| LLaVAv1.5 | Regular | 175.67 | 124.67 | 114.00 | 151.00 | 565.33 |
| LLaVAv1.5 | ICT | 190.00 | 160.43 | 128.67 | 170.00 | 649.10 |
| LLaVAv1.5 | **DMAS** | **195.00** | **158.33** | **133.33** | **173.33** | **659.99** |
| QwenVL | Regular | 155.00 | 127.67 | 131.67 | 173.00 | 587.33 |
| QwenVL | VAF | 165.00 | 155.00 | 133.33 | 175.00 | 628.33 |
| QwenVL | **DMAS** | **170.00** | **145.00** | **133.33** | **185.00** | **633.33** |

**CHAIR Results** (LLaVAv1.5):

| Method | CHAIR_S↓ | CHAIR_I↓ |
|--------|----------|----------|
| Regular | 51.0 | 15.2 |
| VTI | 35.8 | 11.1 |
| **DMAS** | **30.8** | **11.4** |

### Ablation Study

| Method | CHAIR_S↓ | CHAIR_I↓ | POPE Acc↑ | POPE F1↑ |
|--------|----------|----------|-----------|----------|
| Full DMAS | 30.8 | 11.4 | 81.70 | 82.47 |
| Truthfulness vector only | 34.2 | 11.7 | 81.67 | 82.42 |
| Visual vector only | 42.4 | 13.2 | 81.40 | 82.01 |
| No intervention | 51.0 | 15.2 | 75.08 | 76.06 |

### Key Findings
- Dynamic semantic matching for steering vector selection significantly outperforms fixed steering vectors; on QwenVL's Position subtask, fixed vectors even underperform the original model
- A cluster count of 4 achieves optimal performance on both models; too few clusters leads to excessively coarse semantic granularity
- The method yields substantial improvements on entirely different dataset types such as ScienceQA and ViQuAE (LLaVAv1.5 on ScienceQA: 52.75%→62.27%), demonstrating strong generalization
- On POPE with MSCOCO, LLaVAv1.5 achieves +5.43% Accuracy and +7.14% F1; on GQA, +6.94% Accuracy and +6.5% F1
- Negative values of $\alpha$ and $\beta$ (i.e., steering toward hallucination) degrade F1; excessively large values impair the model's base capabilities
- Too few intervened attention heads yields negligible effect, while too many also leads to performance degradation

## Highlights & Insights
- The paper reveals that truthfulness and visual perception activate distinct subsets of attention heads in LVLMs, providing an important empirical foundation for future research
- The dynamic semantic matching design is principled and effective, avoiding the limitations of a one-size-fits-all intervention strategy
- The method is entirely training-free and can be plug-and-play integrated into different LVLM architectures
- The experimental design is comprehensive: discriminative tasks (MME, POPE) + generative tasks (CHAIR) + generalization validation (ScienceQA, ViQuAE), forming a complete evaluation framework
- Visualization analyses (attention head activation maps, t-SNE cluster plots, hyperparameter sensitivity curves) provide strong support for the method's motivation and effectiveness

## Limitations & Future Work
- The construction of the steering vector database relies on specific choices of the AMBER and SEED datasets; larger and more diverse data sources may further improve performance
- The cluster count is currently fixed at 4; adaptively determining the optimal number of clusters is a worthwhile direction
- Hyperparameters $\alpha$, $\beta$, and $K$ require grid search; automated hyperparameter tuning merits investigation
- Validation is limited to 7B-scale models; the effectiveness on larger models (e.g., 13B, 70B) remains to be verified
- Constructing the steering vector database incurs non-trivial preprocessing cost (activation extraction over 3,000 samples)
- The method assumes that the functional specialization of attention heads is consistent across different LVLM architectures; the universality of this assumption requires further validation

## Related Work & Insights
- **Relationship to ICT**: ICT enhances attention by adding noise to objects in images, whereas DMAS intervenes along both truthfulness and visual perception dimensions simultaneously and supports dynamic semantic matching
- **Relationship to VTI**: VTI uses fixed steering vectors; DMAS demonstrates the necessity of dynamic selection
- **Insight**: The activation engineering paradigm warrants further exploration across a broader range of multimodal tasks, such as visual reasoning and multimodal dialogue

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of dynamic semantic matching for steering vector selection is innovative, though activation engineering itself builds on prior foundational work
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers three benchmarks (MME/POPE/CHAIR), two models, complete ablations, and sufficient generalization validation
- Writing Quality: ⭐⭐⭐⭐ Writing is clear with strong visualizations and naturally motivated problem framing
- Value: ⭐⭐⭐⭐ High practical value as a training-free method, though clustering and hyperparameter search add deployment complexity

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Look Carefully: Adaptive Visual Reinforcements in Multimodal Large Language Models for Hallucination Mitigation](look_carefully_adaptive_visual_reinforcements_in_multimodal_large_language_model.md)
- [\[ICML 2026\] Revis: Sparse Latent Steering to Mitigate Object Hallucination in Large Vision-Language Models](../../ICML2026/multimodal_vlm/revis_sparse_latent_steering_to_mitigate_object_hallucination_in_large_vision-la.md)
- [\[ICLR 2026\] Shuffle-R1: Efficient RL Framework for Multimodal Large Language Models via Data-centric Dynamic Shuffle](shuffle-r1_efficient_rl_framework_for_multimodal_large_language_models_via_data-.md)
- [\[ICLR 2026\] Self-Aug: Query and Entropy Adaptive Decoding for Large Vision-Language Models](self-aug_query_and_entropy_adaptive_decoding_for_large_vision-language_models.md)
- [\[ICLR 2026\] Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models](vision-r1_incentivizing_reasoning_capability_in_multimodal_large_language_models.md)

</div>

<!-- RELATED:END -->
