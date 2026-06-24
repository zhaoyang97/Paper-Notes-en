---
title: >-
  [Paper Note] Enhancing Vision-Language Compositional Understanding with Multimodal Synthetic Data (SPARCL)
description: >-
  [CVPR 2025][Image Generation][Compositional Understanding] This paper proposes SPARCL, which generates high-fidelity, fine-grained counterfactual synthetic images by injecting real image features into the padding embeddings of a fast T2I model. It also designs an adaptive margin loss to filter noisy synthetic samples and focus on learning hard samples, improving CLIP's compositional understanding accuracy by over 8% on average across four benchmarks…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Compositional Understanding"
  - "CLIP Fine-tuning"
  - "Synthetic Data"
  - "Adaptive Margin Loss"
  - "Image Feature Injection"
date: 2026-05-08
content_hash: b037250f20d956d9
---

# Enhancing Vision-Language Compositional Understanding with Multimodal Synthetic Data (SPARCL)

**Conference**: CVPR 2025  
**arXiv**: [2503.01167](https://arxiv.org/abs/2503.01167)  
**Code**: None  
**Area**: Image Generation  
**Keywords**: Compositional Understanding, CLIP Fine-tuning, Synthetic Data, Adaptive Margin Loss, Image Feature Injection

## TL;DR
This paper proposes SPARCL, which generates high-fidelity, fine-grained counterfactual synthetic images by injecting real image features into the padding embeddings of a fast T2I model. It also designs an adaptive margin loss to filter noisy synthetic samples and focus on learning hard samples, improving CLIP's compositional understanding accuracy by over 8% on average across four benchmarks, outperforming the state-of-the-art by 2% on three of them.

## Background & Motivation

1. **Background**: Current vision-language models (VLMs) such as CLIP still suffer from significant limitations in compositional understanding—struggling to accurately distinguish subtle differences in object attributes, spatial relationships, and word order (e.g., "a person holding a surfboard" vs. "a person holding a shovel").

2. **Limitations of Prior Work**: The core reason is the lack of paired, fine-grained counterfactual samples in the training data. Collecting such data is costly, while existing synthetic methods face a trilemma—(a) image editing models (e.g., InstructPix2Pix) have poor text alignment; (b) text-to-image models (e.g., SDXL) have poor fidelity to the original image; (c) sample-wise optimization methods are too inefficient.

3. **Key Challenge**: Generating high-quality, fine-grained counterfactual images requires simultaneously satisfying three conflicting demands: efficiency, text alignment, and image fidelity. Furthermore, synthetic data inevitably contains noise (false positives, over-modified negatives), and treating all samples uniformly can mislead learning.

4. **Goal**: (1) How to efficiently generate fine-grained counterfactual synthetic images that maintain both text alignment and image fidelity? (2) How to effectively utilize synthetic samples of varying quality during training?

5. **Key Insight**: It is observed that in the text embeddings of T2I models, semantic tokens (before EOS) control content, while padding tokens (after EOS) control style, meaning the two are decoupled. Therefore, real image features can be injected into the padding positions to improve fidelity while preserving the semantic embeddings.

6. **Core Idea**: Real image feature injection resolves the fidelity of synthetic images, and adaptive margin distinguishes synthetic samples of different qualities.

## Method

The SPARCL framework consists of two stages: the data generation stage and the model training stage.

### Overall Architecture
Given a real image-text pair $(I^r, T^r)$, an LLM is first used to generate fine-grained counterfactual positive/negative text descriptions. Then, a fast T2I model enhanced by image feature injection generates the corresponding images, followed by AdaIN for style transfer. During the training stage, the real and synthetic samples are compiled into an expanded batch to jointly optimize CLIP using a sigmoid contrastive loss + adaptive margin loss (fine-tuning only the LoRA adapters).

### Key Designs

1. **Image Feature Injection**:

    - **Function**: Improving the fidelity of synthetic images to real images without compromising text alignment.
    - **Mechanism**: The CLIP image encoder extracts the CLS embedding $f_i^r$ of the real image, and then all padding positions after the EOS in the T2I model's text encoder output are replaced with this image embedding: $\hat{e}_i^s = \langle e_{i,1}^s, ..., e_{i,k_i}^s, f_i^r, ..., f_i^r \rangle$. Since semantics (before EOS) and style (after EOS) are decoupled in the T2I model, replacing padding does not affect semantic alignment but injects the style information of the real image, reducing the visual gap with the original image. AdaIN is then used for the final style transfer.
    - **Design Motivation**: Standard T2I models have no input containing original image information, naturally resulting in poor fidelity. Injecting image features into the padding area, which "does not affect content," is an ingenious way to enhance fidelity at zero cost.

2. **Adaptive Margin Loss**:

    - **Function**: Distinguishing synthetic samples of varying qualities, filtering out erroneous samples, and focusing on learning from hard samples.
    - **Mechanism**: For each image, four categories of caption sets are defined—positive set $\mathbb{P}$, hard negative set $\mathbb{N}_h$ (negative captions of the same image), easy negative set $\mathbb{N}_e$ (captions of other images), and real negative set $\mathbb{N}_r$. The margin loss requires positive sample similarity > hard negative > easy negative. The key lies in the design of the adaptive margin $m$: when the similarity difference between positive and negative samples is $d < \beta$ (threshold), indicating a likely erroneous sample, the margin is set to $d$ to zero out the loss. When $\beta \le d \le m_0$, the margin is scaled up to strengthen learning from hard samples. When $d > m_0$, it is fixed at $m_0$.
    - **Design Motivation**: Synthetic data quality is inconsistent—some negative samples may actually resemble the original image more (incorrect generation), while others are easily distinguishable. A uniform margin would cause the model to learn faulty signals from erroneous samples. The adaptive margin automatically skips suspicious samples and weights hard samples, achieving noise-robust training.

3. **Hierarchical Contrastive Training Framework**:

    - **Function**: Unifying real and synthetic positive/negative samples into a structured contrastive learning framework.
    - **Mechanism**: Each real image-text pair is expanded into a sextuple $(I^r, T^r, I^{sn}, T^{sn}, I^{sp}, T^{sp})$—real pair + synthetic negative pair + synthetic positive pair. The $3n$ image-text pairs in a batch are trained using a sigmoid contrastive loss $L_{con}$ to encourage high similarity for positive pairs and low similarity for negative pairs. Meanwhile, a weight $\alpha > 1$ is used to enhance comparisons involving real samples (which are more reliable). The final loss is $L = L_{con} + \lambda L_{mar}$. LoRA fine-tuning is used to avoid catastrophic forgetting.
    - **Design Motivation**: Simultaneously generating both positive and negative synthetic pairs prevents the model from shortcutting by merely distinguishing generation artifacts. The three-level hierarchy (positive > hard negative > easy negative) provides richer supervisory signals than a simple positive/negative dichotomy.

### Loss & Training
The total loss is $L = L_{con} + \lambda L_{mar}$, where $L_{con}$ is the sigmoid-based contrastive loss, and $L_{mar}$ is the adaptive margin ranking loss (bidirectional for both vision and language). The AdamW optimizer is used with a cosine learning rate scheduler, fine-tuning only the LoRA adapters (3,000 steps for ViT-B/32, 15,000 steps for ViT-L/14).

## Key Experimental Results

### Main Results

| Benchmark | CLIP (zero-shot) | NegCLIP | CE-CLIP | SPARCL (Ours) | Gain (vs CE-CLIP) |
|------|-----------------|---------|---------|--------------|----------------|
| ARO | 61.1% | 76.0% | 79.7% | **77.2%** | -2.5% |
| VL-CheckList | 73.2% | 74.6% | 76.3% | **79.2%** | +2.9% |
| SugarCrepe | 73.4% | 82.5% | 85.2% | **87.1%** | +1.9% |
| SugarCrepe++ | 59.8% | 64.9% | - | **66.1%** | - |

Note: Using only COCO 82K training data, SPARCL outperforms all methods on VL-CheckList, SugarCrepe, and SugarCrepe++, while being slightly inferior to CE-CLIP on ARO (though CE-CLIP uses substantially more synthetic captions).

### Ablation Study

| Configuration | ARO | VL-CL | SugarCrepe | SugarCrepe++ | Average |
|------|-----|-------|------------|-------------|------|
| Full SPARCL | 77.2 | 79.2 | 87.1 | 66.1 | 77.4 |
| w/o SynImg (No Synthetic Image) | 77.9 | 76.3 | 85.7 | 66.3 | 76.6 |
| w/o FeatInj (No Feature Injection) | 76.3 | 78.5 | 86.0 | 64.9 | 76.4 |
| w/o AdaIN | 76.7 | 78.0 | 86.2 | 65.3 | 76.6 |
| w/o Adaptive Margin | 76.2 | 78.1 | 85.8 | 65.1 | 76.3 |

### Key Findings
- Image feature injection and adaptive margin loss each make independent contributions, yielding the best performance when combined.
- AdaIN style transfer plays a significant role in narrowing the synthetic-to-real domain gap.
- Using synthetic captions alone (without synthetic images) can also yield substantial improvements, but adding synthetic images brings a further 2.9% gain on VL-CheckList.
- Adaptive margin yields the largest gain on SugarCrepe++ (+1.0%), as this benchmark focuses on distinguishing semantically equivalent but lexically different descriptions.
- The volume of training data used by SPARCL is far smaller than some competing methods (e.g., CE-CLIP+ which uses 3M data), yet achieves better performance.

## Highlights & Insights
- **Image Feature Injection into Padding Positions**: Leveraging the decoupled nature of semantics and style in the embedding space of T2I models to improve fidelity at zero cost. This design itself holds independent value and can be applied to other scenarios requiring controlled image editing.
- **Adaptive Margin = Automatic Curriculum Learning**: Automatically identifying sample difficulty and quality through the difference in positive-negative similarity $d$. This is effectively equivalent to a parameter-free curriculum learning strategy and can be transferred to any contrastive learning task utilizing noisy synthetic data.
- **Simultaneous Generation of Positive and Negative Pairs**: Simultaneously generating synthetic positive and negative captions prevents the model from achieving high scores by taking shortcuts such as distinguishing "natural text vs. synthetic text," thus addressing the hackable bias issue highlighted by SugarCrepe.

## Limitations & Future Work
- Only validated on ViT-B/32 and ViT-L/14; larger VLM scales (e.g., EVA-CLIP, SigLIP) have not been tested.
- Image feature injection relies on the alignment between the CLIP encoder and the T2I model encoder; transitioning to different T2I models (e.g., non-CLIP-based) might require redesigning the injection mechanism.
- Hyperparameters for the adaptive margin ($\beta$, $\gamma$, $m_0$) may require tuning for different training sets.
- Exploring more powerful image editing methods (e.g., the latest instruction-based editing models) to replace T2I generation remains uninvestigated.

## Related Work & Insights
- **vs. NegCLIP**: NegCLIP only utilizes rule-based generation for negative captions, which easily results in unnatural texts that models can use as shortcuts; SPARCL uses an LLM to generate more natural caption pairs.
- **vs. CE-CLIP**: CE-CLIP employs cross-modal ranking but treats all synthetic samples with equal weight; SPARCL distinguishes sample quality via an adaptive margin, yielding higher data efficiency.
- **vs. COMO**: COMO also generates multimodal synthetic data but relies on segmentation maps to control fidelity; SPARCL's feature injection mechanism is more lightweight and requires no additional models.

## Rating
- Novelty: ⭐⭐⭐⭐ Both image feature injection and the adaptive margin are novel, with an overall ingenious design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely thorough, with testing on four benchmarks, detailed ablations, and comparison with multiple baselines.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, detailed methodological descriptions, and rich figures/tables.
- Value: ⭐⭐⭐⭐ Highly practical and instructive for vision-language compositional understanding research; the adaptive margin represents a widely reusable approach.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Font-Agent: Enhancing Font Understanding with Large Language Models](font-agent_enhancing_font_understanding_with_large_language_models.md)
- [\[CVPR 2025\] Yo'Chameleon: Personalized Vision and Language Generation](yochameleon_personalized_vision_and_language_generation.md)
- [\[CVPR 2025\] Training Data Provenance Verification: Did Your Model Use Synthetic Data from My Generative Model for Training?](training_data_provenance_verification_did_your_model_use_synthetic_data_from_my_.md)
- [\[CVPR 2025\] Do Visual Imaginations Improve Vision-and-Language Navigation Agents?](do_visual_imaginations_improve_vision-and-language_navigation_agents.md)
- [\[CVPR 2025\] TokenFlow: Unified Image Tokenizer for Multimodal Understanding and Generation](tokenflow_unified_image_tokenizer_for_multimodal_understanding_and_generation.md)

</div>

<!-- RELATED:END -->
