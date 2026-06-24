---
title: >-
  [Paper Note] Reasoning to Attend: Try to Understand How \<SEG\> Token Works
description: >-
  [CVPR 2025][Multimodal VLM][Reasoning segmentation] This paper conducts an in-depth analysis of the working mechanism of the \<SEG\> token in reasoning segmentation tasks, discovering that it learns semantic features similar to direct textual mentions for image-text semantic alignment. Based on this finding, the READ method is proposed to convert the similarity map between the \<SEG\> token and image tokens into point prompts, guiding the SAM decoder to generate more precise…
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Reasoning segmentation"
  - "SEG token analysis"
  - "Similarity-guided point prompt"
  - "Semantic alignment"
  - "Large Multimodal Model"
date: 2026-05-08
content_hash: 06db69e4f96b26fe
---

# Reasoning to Attend: Try to Understand How \<SEG\> Token Works

**Conference**: CVPR 2025  
**arXiv**: [2412.17741](https://arxiv.org/abs/2412.17741)  
**Code**: [https://github.com/rui-qian/READ](https://github.com/rui-qian/READ)  
**Area**: Multimodal VLM  
**Keywords**: Reasoning segmentation, SEG token analysis, Similarity-guided point prompt, Semantic alignment, Large Multimodal Model

## TL;DR

This paper conducts an in-depth analysis of the working mechanism of the \<SEG\> token in reasoning segmentation tasks, discovering that it learns semantic features similar to direct textual mentions for image-text semantic alignment. Based on this finding, the READ method is proposed to convert the similarity map between the \<SEG\> token and image tokens into point prompts, guiding the SAM decoder to generate more precise segmentation masks in a plug-and-play manner.

## Background & Motivation

In reasoning segmentation tasks, pioneering works like LISA utilize the \<SEG\> token as a bridge between the LLaVA encoder and the SAM decoder: the \<SEG\> token is a newly added placeholder in the text vocabulary, whose hidden embeddings are projected into SAM to generate segmentation masks after LLM fine-tuning. However, few studies have explored what the \<SEG\> token actually "learns."

Through visualization, the authors discovered a striking consistency: the similarity map between the \<SEG\> token and image tokens exhibits highly consistent activation patterns in both the LLaVA encoder and the SAM decoder. Furthermore, these active regions are highly aligned with the similarity maps of direct textual mentions (e.g., "antler") in CLIP. This implies that the \<SEG\> token essentially learns "semantic similarity" capabilities, serving as an implicit text-to-visual semantic bridge. Based on this discovery, a natural idea is to directly utilize the high-activation points in the similarity map to guide the model on "where to attend."

## Method

### Overall Architecture

READ consists of three core modules: (1) **LLaVA Encoder**: receives the image-text pair and outputs the textual response along with the hidden layer embedding $\boldsymbol{h}_{seg}$ of the \<SEG\> token; (2) **SasP Module (Similarity as Points)**: computes the similarity map between the \<SEG\> token embedding and the image token embeddings, converting high-activation regions into continuous differentiable point coordinates; (3) **SAM Decoder**: receives $\boldsymbol{h}_{seg}$, the point prompts $\mathcal{P}$, and image features to generate the segmentation mask $\hat{\mathbf{M}} = \mathcal{G}_{\mathcal{V}}^{dec}(\mathbf{f}, \boldsymbol{h}_{seg}, \mathcal{P})$.

### Key Designs

1. **Similarity as Points (SasP)**:
    - Function: Extract spatial location prompts from the semantic similarity between the \<SEG\> token and image tokens.
    - Mechanism: Compute the parameter-free similarity score $\mathcal{S} = \boldsymbol{h}_{img}^{(l_k)} \cdot (\boldsymbol{h}_{seg}^{(l_k)})^T$, where $\mathcal{S} \in \mathbb{R}^{N_t}$. Based on the mean $\mu$ and standard deviation $\sigma$, a threshold is set to divide points into three categories: positive points ($\mathcal{S}_j \geq \mu + 0.5\sigma$, foreground), negative points ($\mathcal{S}_j \leq \mu - 0.5\sigma$, background), and neutral points (the rest). The absolute coordinates of each selected point are restored and sent to SAM as point prompts.
    - Design Motivation: Experiments demonstrate that high-activation points in the similarity map already implicitly encode the target object's location information (using only these points to prompt SAM achieves 27.0% cIoU). Explicitly utilizing them provides stronger spatial localization cues for SAM.

2. **Discrete to Continuous Sampling (DtoC)**:
    - Function: Convert discrete, non-differentiable point coordinates into continuous, differentiable coordinates, allowing gradients to propagate back to the LMM.
    - Mechanism: Utilize distance-based Gaussian weighted average interpolation. For selected points $(x_j, y_j)$, compute the distance weight $w_i^j = \exp(-d_i^j)$ to each grid point, combine this with the softmax probability $p_i$ to obtain the normalized weight $\hat{w}_i^j$, and compute the final continuous coordinates as the weighted average $\hat{x}_j = \sum_{i=1}^{h \times w} \mathbf{g}_{x,i} \cdot \hat{w}_i^j$.
    - Design Motivation: Point selection involves sorting and indexing operations, which are non-differentiable. Without continuous treatment, the loss function gradients cannot backpropagate to the LLaVA encoder. Via DtoC, the model can "learn to attend closely" during the backward pass while "inferring where to attend" in the forward pass.

3. **Plug-and-Play Architecture Design**:
    - Function: SasP can be seamlessly integrated into any \<SEG\>-token-based pipeline (such as LISA, SESAME, GSVA, etc.).
    - Mechanism: The similarity calculation of SasP is parameter-free, relying only on existing \<SEG\> token embeddings and image token embeddings without introducing extra parameters.
    - Design Motivation: Reduce integration costs while maintaining the generalizability and extremely low overhead of the method.

### Loss & Training

- **Total Loss**: $\mathcal{L} = \lambda_{txt} \mathcal{L}_{txt} + \lambda_{mask} \mathcal{L}_{mask}$
- **Text Generation Loss** $\mathcal{L}_{txt}$: Cross-entropy loss
- **Mask Loss** $\mathcal{L}_{mask} = \lambda_{bce} \mathcal{L}_{bce}(\hat{\mathbf{M}}, \mathbf{M}) + \lambda_{dice} \mathcal{L}_{dice}(\hat{\mathbf{M}}, \mathbf{M})$, with $\lambda_{bce}=2.0, \lambda_{dice}=0.5$
- LoRA is used for efficient fine-tuning of LLaVA. SAM's image encoder is frozen, and only the mask decoder is trained.
- 4×3090 GPUs, 20 epochs, ~24 hours; AdamW, lr=0.0003

## Key Experimental Results

### Main Results

| Dataset | Metric | READ-7B | LISA-7B-v1.5(ft) | SESAME | Gain |
|---|---|---|---|---|---|
| ReasonSeg val | cIoU | **67.6** | 62.9 | 39.1 | +4.7 |
| ReasonSeg test overall | gIoU | **58.5** | 55.6 | 30.5 | +2.9 |
| RefCOCO val | cIoU | **78.1** | 74.9 | 74.7 | +3.2 |
| RefCOCO+ val | cIoU | **68.4** | 65.1 | 64.9 | +3.3 |
| RefCOCOg val(U) | cIoU | **70.1** | 67.9 | 66.1 | +2.2 |
| FP-RefCOCO See | Acc | **82.87** | 51.36 | 79.84 | +3.03 |
| FP-RefCOCO Seg | cIoU | **61.50** | 44.00 | 57.93 | +3.57 |

### Ablation Study

| Configuration | gIoU | cIoU | Note |
|------|------|------|------|
| \<SEG\> prompt only | 51.2 | 57.6 | Baseline LISA method |
| + $\mathcal{P}$ prompt (discrete points) | 56.4 | 64.6 | Point prompts contribute +7% cIoU |
| + $\mathcal{P}$ DtoC (continuous) | **59.8** | **67.6** | DtoC contributes another +3% cIoU |
| SAM-ViT-Base | 55.6 | 61.9 | - |
| SAM-ViT-Large | 60.1 | 65.2 | - |
| SAM-ViT-Huge | **59.8** | **67.6** | Larger backbone yields better performance |

### Key Findings

- Quantitative analysis of the \<SEG\> token: Prompting original SAM solely with high/low activation points from the similarity map achieves 27.0% cIoU (vs. 30.4% for SESAME), demonstrating that the \<SEG\> token indeed learns effective spatial localization semantics.
- The IoU consistency between the similarity map and the ground-truth mask ($\mathcal{S}$IoU=36.4%) even surpasses the result of directly prompting SAM with the \<SEG\> token (30.4%), validating the existence of spatial location information.
- READ performs outstandingly in false premise scenarios: The See accuracy on FP-RefCOCO is 82.87% (vs. 51.36% for LISA), indicating that READ does not blindly generate masks.

## Highlights & Insights

- **First systematic analysis of the mechanism of the \<SEG\> token**: Reveals through visualization and quantitative experiments that the \<SEG\> token learns "semantic similarity"—essentially mapping implicit textual reasoning results to visual space.
- **Bidirectional learning of "reasoning to attend & attending to reason"**: DtoC enables gradients to flow back from the segmentation loss to LLaVA, establishing a closed loop of "attention-guided segmentation" and "segmentation-feedback-optimized attention."
- **Minimalist and plug-and-play**: SasP requires no extra parameters and can be directly integrated into any \<SEG\>-like system.
- **Discovery of \<SEG\> token semantic equivalence**: The embedding obtained by the \<SEG\> token after implicit reasoning aligns highly with direct textual mentions (e.g., "antler") in their similarity patterns within the CLIP space.

## Limitations & Future Work

- The current similarity calculation is a simple dot product without introducing learnable parameters (e.g., cross-attention), leaving room for further improvement.
- Validated only on 7B and 13B LLaVA models, without testing on larger or newer LMMs.
- The threshold $\varepsilon=0.5$ is fixed; an adaptive threshold might yield better results.
- Multi-target scenarios with a single \<SEG\> token have not yet been fully explored.

## Related Work & Insights

- Relationship with LISA (\<SEG\> $\rightarrow$ SAM): READ adds a similarity point prompt channel on top of LISA, explicitly converting the implicit semantics of the \<SEG\> token into spatial guidance signals.
- Relationship with SESAME (handling false premises): READ inherits the false premise training data and significantly outperforms SESAME on both See and Segment metrics.
- The discovery of \<SEG\> token semantic equivalence offers broader insights into understanding learning behaviors of placeholder tokens in LLMs (e.g., \<IMG\> tokens in image generation might exhibit similar behaviors).

## Rating

- Novelty: ⭐⭐⭐⭐ Analyzing the mechanism of the \<SEG\> token presents a brand-new perspective, and the SasP design, while simple, has clear motivation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ ReasonSeg + RefCOCO(+/g) + FP-RefCOCO(+/g) are comprehensively covered, with detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ The logical chain of analysis $\rightarrow$ discovery $\rightarrow$ method is clear, and the visualization is highly convincing.
- Value: ⭐⭐⭐⭐ Possesses practical value as a plug-and-play module, and the mechanistic analysis offers instructional significance for future works.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Where MLLMs Attend and What They Rely On: Explaining Autoregressive Token Generation](../../CVPR2026/multimodal_vlm/where_mllms_attend_and_what_they_rely_on_explaining_autoregressive_token_generat.md)
- [\[CVPR 2025\] Vision-Language Models Do Not Understand Negation](vision-language_models_do_not_understand_negation.md)
- [\[CVPR 2025\] Instruction-based Image Manipulation by Watching How Things Move](instruction-based_image_manipulation_by_watching_how_things_move.md)
- [\[CVPR 2025\] Towards Understanding How Knowledge Evolves in Large Vision-Language Models](towards_understanding_how_knowledge_evolves_in_large_vision-language_models.md)
- [\[ACL 2026\] How Do LLMs and VLMs Understand Viewpoint Rotation Without Vision? An Interpretability Study](../../ACL2026/multimodal_vlm/how_do_llms_and_vlms_understand_viewpoint_rotation_without_vision_an_interpretab.md)

</div>

<!-- RELATED:END -->
