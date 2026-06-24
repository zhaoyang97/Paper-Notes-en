---
title: >-
  [Paper Note] Positional Bias in Multimodal Embedding Models: Do They Favor the Beginning, the Middle, or the End?
description: >-
  [AAAI 2026][Multimodal VLM][Positional Bias] This paper presents the first systematic study of positional bias in multimodal representation models, finding that text encoders tend to favor the beginning of the input while image encoders exhibit preference for both the beginning and the end. Through extensive controlled experiments, the study reveals that this bias arises from the joint influence of positional encoding schemes, training objectives, context importance…
tags:
  - "AAAI 2026"
  - "Multimodal VLM"
  - "Positional Bias"
  - "CLIP"
  - "Multimodal Representation Learning"
  - "Image-Text Retrieval"
  - "Attention Mechanism"
date: 2026-05-08
content_hash: 27acf70f3ba5757a
---

# Positional Bias in Multimodal Embedding Models: Do They Favor the Beginning, the Middle, or the End?

**Conference**: AAAI 2026
**arXiv**: [2511.11216](https://arxiv.org/abs/2511.11216)  
**Code**: [https://github.com/tiiuae/PosBias/](https://github.com/tiiuae/PosBias/)  
**Area**: Information Retrieval
**Keywords**: Positional Bias, CLIP, Multimodal Representation Learning, Image-Text Retrieval, Attention Mechanism

## TL;DR
This paper presents the first systematic study of positional bias in multimodal representation models, finding that text encoders tend to favor the beginning of the input while image encoders exhibit preference for both the beginning and the end. Through extensive controlled experiments, the study reveals that this bias arises from the joint influence of positional encoding schemes, training objectives, context importance, and image-text pair training.

## Background & Motivation

**Background**: Transformer-based models have achieved remarkable success in NLP and vision tasks; however, research has shown that a model's ability to capture contextual information is influenced by the position of that information within the input sequence—a phenomenon known as "positional bias." The "lost in the middle" effect identified by Liu et al. demonstrates that models tend to prioritize content at the beginning and end while neglecting the middle.

**Limitations of Prior Work**:
- Positional bias research has focused predominantly on **text generation** models (LLMs), with limited attention to **representation learning** models.
- A "dwelling at the beginning" bias has been identified in text representation models (Coelho et al.), attributed to the inverted pyramid writing style.
- **Positional bias in multimodal models has been largely unreported**, and the bias patterns on the image side are entirely unknown.

**Core Problem**:
1. Do positional biases exist in multimodal embedding models such as CLIP?
2. Do the bias patterns of the text encoder and image encoder differ?
3. What are the root causes—positional encoding, training data, loss function, or model architecture?

**Key Insight**: The paper first distinguishes between "context importance" and "positional bias," then systematically evaluates the presence and patterns of positional bias in multimodal models through carefully designed experiments (moving a fixed segment to different positions while masking others). Finally, a series of controlled experiments is conducted to investigate each potential contributing factor.

## Method

### Overall Architecture

The study proceeds at three levels: (1) context importance analysis—identifying which regions are inherently more semantically significant; (2) positional bias analysis—detecting the existence and patterns of bias; (3) bias cause investigation—verifying or refuting existing hypotheses through controlled experiments.

### Key Designs

1. **Context Importance Analysis**

    - **Function**: Locates the most semantically important regions in text and images.
    - **Text side**: The input tokens are uniformly segmented; only one segment is retained at a time while the rest are replaced with padding masks. Representations are obtained via the text encoder, and retrieval accuracy is computed with the image side fixed.
    - **Image side**: Similarly, one image region is retained while the rest are masked using the CLIP RGB mean value ([0.481, 0.458, 0.408]), with the text side fixed.
    - **Key Findings**: The first segment of text is the most important (consistent with front-loaded writing conventions); the central region of images is the most important (subjects are typically centered, with background/sky/ground above and below).
    - **Design Motivation**: Understanding "which regions are inherently important" is a prerequisite for determining whether the model favors a region due to content or due to position.

2. **Positional Bias Analysis**

    - **Two strategies for the text side**:
        - **Text Perturbation**: The text is divided into sub-texts; one segment is moved to different positions while the others are replaced with Lorem Ipsum placeholder text.
        - **Token Masking**: Tokens are segmented; one segment is moved to different positions while the others are replaced with padding tokens.
    - **Image side**: A single visual segment is isolated and relocated to different spatial positions, while the remaining regions are masked using the CLIP RGB mean.
    - **Key Design Choice**: Unlike prior work that shuffles the order of multiple documents, this paper isolates and moves a single segment—because CLIP's fixed context window does not permit the full content to be included simultaneously.
    - **Design Motivation**: The goal is not to measure absolute retrieval accuracy, but to analyze how retrieval performance varies as a function of position.

3. **Bias Cause Investigation**

    - **Data distribution**: Training a Shuffled Long-CLIP (with clause order shuffled) shows that bias persists but weakens (accuracy drop when moving the first segment from the beginning to the end: 0.199 vs. 0.303 for the original), indicating that data distribution contributes but is not the sole factor.
    - **Positional encoding**: Comparing CLIP (absolute positional encoding) vs. TULIP (RoPE rotary positional encoding) shows that TULIP exhibits stronger bias (accuracy drops sharply from 0.66 to 0.065 when the first segment is moved from position 0 to position 2), indicating that the positional encoding scheme significantly affects bias.
    - **Text length**: Positional bias is observed even in short texts (COCO average: 11.53 tokens).
    - **Model size**: ViT-L/14 exhibits stronger bias than ViT-B/16 (higher coefficient of variation).
    - **Image resolution and patch size**: Reducing patch size or increasing resolution helps mitigate bias on the image side.
    - **Training loss**: SigLIP (sigmoid loss) and CLIP (softmax contrastive loss) exhibit different bias patterns—SigLIP shows image-side preference for the beginning rather than both ends.
    - **Model architecture**: CNN-based models (CLIP-ResNet-50) also exhibit bias, indicating this is not a Transformer-specific phenomenon.
    - **Unimodal vs. multimodal**: A pure vision ResNet shows no positional bias, whereas the CLIP vision encoder does—suggesting that image-text pair training introduces or amplifies image-side bias.

## Key Experimental Results

### Main Results (Verification of Positional Bias)

| Model | Text-side Bias Pattern | Image-side Bias Pattern | Dataset |
|-------|----------------------|------------------------|---------|
| Long-CLIP (ViT-B/16) | Beginning preference | Beginning + end preference | Urban1K |
| TULIP (ViT-L/14) | Stronger beginning preference | Beginning + end preference | Urban1K |
| Shuffled Long-CLIP | Weakened beginning preference | Beginning + end preference | Urban1K |
| CLIP (ViT-B/16) | Beginning preference (weaker) | Beginning + end preference | COCO |
| SigLIP-Base | Beginning preference | Primarily beginning preference | COCO |
| CLIP-ResNet-50 | Beginning preference | Beginning + end preference | COCO |

### Ablation Study (Controlled Variables for Bias Causes)

| Controlled Factor | Experimental Setup | Conclusion |
|------------------|-------------------|------------|
| Data distribution | Long-CLIP vs. Shuffled Long-CLIP | Contributing but not sole factor (bias weakens but persists) |
| Positional encoding | CLIP (absolute) vs. TULIP (RoPE) | RoPE exacerbates bias; absolute encoding mitigates but does not eliminate it |
| Text length | Long-CLIP (Urban1K) vs. CLIP (COCO) | Bias exists in short texts but is weaker |
| Model size | ViT-B/16 vs. ViT-L/14 | Larger models exhibit stronger bias (higher coefficient of variation) |
| Resolution | ViT-L/14 vs. ViT-L/14@336 | Higher resolution helps reduce image-side bias |
| Training loss | CLIP (softmax) vs. SigLIP (sigmoid) | Different losses lead to different bias patterns |
| Architecture | ViT vs. ResNet | CNNs also exhibit bias; not Transformer-specific |
| Training paradigm | Unimodal vision model vs. CLIP vision encoder | Image-text pair training introduces/amplifies image-side bias |

**Coefficient of Variation (Long-title dataset Urban1K, image side seg0–seg6)**:

| Model | seg0 | seg3 | seg6 | Note |
|-------|------|------|------|------|
| Long-CLIP | 0.146 | 0.087 | 0.159 | Higher at both ends |
| TULIP | 0.220 | 0.107 | 0.186 | Higher at both ends, more pronounced |
| Shuffled Long-CLIP | 0.185 | 0.089 | 0.144 | Reduced after shuffling |

### Key Findings

1. **Positional bias is prevalent in multimodal models**: Observed across all tested models, datasets, architectures, and training configurations.
2. **Text and image bias patterns differ**:
    - Text encoder: **Consistent preference for the beginning** (moving any segment to the beginning improves retrieval accuracy).
    - Image encoder: **Preference for both the beginning and the end** (U-shaped distribution), with a stronger preference for the beginning.
3. **Image-side bias is particularly noteworthy**: The central region of images carries the highest semantic importance (confirmed by context importance experiments), yet models prefer the beginning and the end—clearly demonstrating the existence of positional bias.
4. **Multimodal training introduces image-side bias**: Pure vision models (ResNet, ViT) show no apparent bias, but bias becomes significant after incorporation into the CLIP framework.
5. **Positional encoding has a large impact but is not the sole factor**: Even Long-CLIP (absolute positional encoding), which shows less bias than TULIP (RoPE), still exhibits positional bias.

## Highlights & Insights

1. **Novel research perspective**: This is the first work to extend positional bias analysis from text generation/text representation to multimodal representation learning, filling an important research gap.
2. **Decoupling context importance from positional bias**: This distinction is critical—a model may appear to favor the beginning because "the beginning content is genuinely more important" rather than because it "prefers the beginning position." The authors explicitly disentangle these two factors through controlled experiments.
3. **Rigorous and comprehensive experimental design**: A controlled variable approach systematically examines seven potential factors: data distribution, positional encoding, loss function, model size, resolution, architecture, and training paradigm.
4. **Multimodal training introduces image-side bias**: A profound insight—pure vision models exhibit no positional bias, whereas CLIP-trained models do, indicating that image-text alignment learning itself introduces new inductive biases.
5. **RoPE found to exacerbate bias**: A counterintuitive finding—RoPE is designed to improve positional modeling, yet it actually intensifies positional bias in multimodal representation learning.

## Limitations & Future Work

1. **Bias is analyzed but not mitigated**: The paper explicitly leaves mitigation strategies to future work.
2. **Focus on CLIP-family models**: Representational bias in generative multimodal models such as LLaVA and BLIP-2 is not evaluated.
3. **Relatively simple evaluation metrics**: The study primarily uses Recall@k and coefficient of variation; more fine-grained bias quantification metrics may offer greater analytical value.
4. **Limited datasets**: Only three datasets are used—Urban1K, DOCCI, and COCO.
5. **Causal analysis lacks depth**: Although multiple contributing factors are identified, the relative contribution and interaction effects of each factor are not quantified.
6. **Downstream impact is insufficiently discussed**: The concrete effects of positional bias on downstream tasks (RAG, zero-shot classification) are not evaluated.

## Related Work & Insights

- **"Lost in the Middle"** (Liu et al.): A seminal study of positional bias in LLMs; this paper extends the analysis to representation models.
- **"Dwelling at the Beginning"** (Coelho et al.): Beginning preference in text representation models, attributed to contrastive learning training; this paper validates analogous phenomena in a multimodal setting.
- **Differential Transformer** (Ye et al.): Differential attention mechanism to reduce attention noise in generative models; the authors suggest it may be adapted for bidirectional multimodal representations.
- Broader implication: As a backbone for many advanced multimodal systems (including RAG), positional bias in CLIP may propagate to downstream systems.
- Practical implication: When using CLIP for retrieval, the placement of key information should be considered for optimal performance; in long-text captions, critical descriptions should be positioned at the beginning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] SpeakerLM: End-to-End Versatile Speaker Diarization and Recognition with Multimodal Large Language Models](speakerlm_end-to-end_versatile_speaker_diarization_and_recognition_with_multimod.md)
- [\[AAAI 2026\] SAGE: Spuriousness-Aware Guided Prompt Exploration for Mitigating Multimodal Bias](sage_spuriousness-aware_guided_prompt_exploration_for_mitigating_multimodal_bias.md)
- [\[ICML 2026\] Circle-RoPE: Cone-like Decoupled Rotary Positional Embedding for Vision-Language Models](../../ICML2026/multimodal_vlm/circle-rope_cone-like_decoupled_rotary_positional_embedding_for_large_vision-lan.md)
- [\[CVPR 2026\] MarkushGrapher-2: End-to-end Multimodal Recognition of Chemical Structures](../../CVPR2026/multimodal_vlm/markushgrapher-2_end-to-end_multimodal_recognition_of_chemical_structures.md)
- [\[ICLR 2026\] WebDS: An End-to-End Benchmark for Web-based Data Science](../../ICLR2026/multimodal_vlm/webds_an_end-to-end_benchmark_for_web-based_data_science.md)

</div>

<!-- RELATED:END -->
