---
title: >-
  [Paper Note] SEA: Low-Resource Safety Alignment for Multimodal Large Language Models via Synthetic Embeddings
description: >-
  [ACL 2025][LLM Alignment][safety alignment] This work proposes the SEA framework, which generates synthetic modality embeddings (without requiring real images/videos/audio) via gradient optimization. It achieves safety alignment for multimodal LLMs using only textual safety data. A high-quality embedding can be synthesized in just 24 seconds on a single RTX 3090. Additionally, the video and audio safety benchmark VA-SafetyBench is released.
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "safety alignment"
  - "multimodal LLM"
  - "synthetic embeddings"
  - "low-resource"
  - "VA-SafetyBench"
date: 2026-05-08
content_hash: 9a54ab9636d7a8a3
---

# SEA: Low-Resource Safety Alignment for Multimodal Large Language Models via Synthetic Embeddings

**Conference**: ACL 2025  
**arXiv**: [2502.12562](https://arxiv.org/abs/2502.12562)  
**Code**: [https://github.com/ZeroNLP/SEA](https://github.com/ZeroNLP/SEA)  
**Area**: Alignment RLHF  
**Keywords**: safety alignment, multimodal LLM, synthetic embeddings, low-resource, VA-SafetyBench

## TL;DR
This work proposes the SEA framework, which generates synthetic modality embeddings (without requiring real images/videos/audio) via gradient optimization. It achieves safety alignment for multimodal LLMs using only textual safety data. A high-quality embedding can be synthesized in just 24 seconds on a single RTX 3090. Additionally, the video and audio safety benchmark VA-SafetyBench is released.

## Background & Motivation

**Background**: MLLMs face severe safety vulnerabilities—injecting adversarial images/audio can easily jailbreak the models to follow harmful instructions. Safety alignment methods like SFT and RLHF are effective but require multimodal safety datasets, which are extremely expensive to construct.

**Limitations of Prior Work**: (1) Multimodal safety data requires a strong alignment among textual instructions, responses, and the additional modality, making collection highly expensive. (2) Rebuilding datasets is necessary whenever a new modality (e.g., EEG) emerges. (3) Although text-only alignment is low-cost, it only works when the text contains explicit harmful information, failing to defend against malicious information delivered solely through images or audio.

**Key Challenge**: Multimodal safety alignment requires multimodal data, but not all modalities have high-performance generative models to produce safety training data.

**Goal**: How to achieve cross-modal safety alignment using only textual safety data?

**Key Insight**: The extra modal data (e.g., images of bombs) used in safety alignment does not need to be human-interpretable; it only needs to be perceived by the MLLM as containing specific harmful content. Therefore, optimization can be performed directly in the output space of the modality encoder.

**Core Idea**: By treating the extra modal embeddings as trainable weights, gradient optimization is used to force the MLLM to perceive the embeddings as containing specific harmful activities or products, thereby achieving safety alignment without real multimodal data.

## Method

### Overall Architecture
Three stages: (1) Data preparation—extracting harmful information from textual safety data and constructing auxiliary data; (2) Embedding optimization—optimizing synthetic embeddings in the output space of the modality encoder; (3) Safety alignment—combining the synthetic embeddings with textual data for multimodal alignment training.

### Key Designs

1. **Data Preparation**:

    - **Function**: Extract harmful phrases from textual safety instructions, classify them into "activities" and "products", and construct content control samples and style control pages.
    - **Mechanism**: Use GPT-4o-mini to extract harmful phrases and convert them into complete sentences. Content control samples force the embedding to "describe" specific harmful content, while style control samples increase embedding diversity. Each sample requires only 2 auxiliary data points.
    - **Design Motivation**: The two control samples ensure the semantic accuracy of the embedding and representation diversity, respectively.

2. **Embedding Optimization**:

    - **Function**: Treat the output of the modality encoder $M(\cdot)$ as trainable weights $E_o$. The system maximizes the probability of the MLLM generating the target text conditioned on $E_o$ via gradient updates.
    - **Mechanism**: $L(E_o) = -\frac{1}{|D_a^i|}\sum_{(x^i, y^i) \in D_a^i} \log(P_r(y^i | x^i, P(E_o)))$, freezing the entire MLLM while updating only $E_o$. It is initialized from blank images, videos, or silent audio.
    - **Design Motivation**: Optimization is conducted in the output space of $M(\cdot)$ rather than the raw data space. Since $M(\cdot)$ is typically frozen during training, this allows synthetic embeddings to blend seamlessly with real multimodal data.

3. **Safety Alignment Training**:

    - **Function**: Combine the optimized embeddings with detoxified textual instructions to replace real multimodal data for SFT/RLHF training.
    - **Mechanism**: Harmful phrases in the detoxified instructions are replaced with "this product/activity," forcing the harmful information to be conveyed entirely via the synthetic embedding. The process bypasses $M(\cdot)$ and directly utilizes $E^i$.
    - **Design Motivation**: To ensure the model learns to detect harmful content from non-text modalities and reject it, rather than relying solely on textual keywords.

4. **VA-SafetyBench Benchmark**:

    - **Function**: Extend MM-SafetyBench to video and audio modalities.
    - **Mechanism**: Eight safety scenarios (illegal activities, hate speech, malware, etc.). The video part comprises three tasks: diffusion model generation, text animation, and mixing. The audio part features three tasks: audio-only, keyphrase-to-speech, and noise addition.

### Loss & Training
Embedding optimization: standard cross-entropy loss, freezing the MLLM and updating only $E_o$. Safety alignment: standard SFT or DPO loss.

## Key Experimental Results

### Main Results

| Method | LLaVA ASR↓ | Video-LLaMA ASR↓ | Qwen2-Audio ASR↓ |
|------|:---:|:---:|:---:|
| Original Model | High | High | High |
| Text-only Alignment | Medium | Medium | Medium |
| **SEA** | **Low** | **Low** | **Low** |

### Ablation Study

| Configuration | Effect | Description |
|------|------|------|
| Content-only control samples | Effective but insufficient | Style control increases diversity |
| Style-only control samples | Poor performance | Content is the core |
| 24s optimization vs longer times | 24s is sufficient | Fast convergence |
| Multimodal real-data alignment | Optimal but expensive | SEA achieves comparable performance |

### Key Findings
- **Extremely Low Cost, High Performance**: A high-quality embedding is synthesized in 24 seconds on a single RTX 3090, bypassing the need for real multimodal data.
- **Cross-Modal Generalizability**: The identical framework applies to MLLMs across three modalities: image, video, and audio.
- **Quantifying the Limitations of Text-Only Alignment**: Text-only alignment is only effective when harmful information is present in the text, failing against harmful information transmitted solely via image/video/audio.
- **VA-SafetyBench Highlights New Risks**: The ASR of video and audio MLLMs reaches up to 60-80%, posing severe safety concerns.

## Highlights & Insights
- **The key insight of "not needing human interpretability"** is exceptionally elegant: images in safety training do not need to be understood by humans, only by the model. This insight enables optimization in the embedding space, bypassing the limitations of requiring raw data generators.
- **Modality-agnostic safety alignment method**: Since it operates in the output space of $M(\cdot)$, it is agnostic to the specific modality encoder, allowing ready extension to future modalities.
- **Striking cost efficiency**: Requiring only 24 seconds on a single GPU, it is orders of magnitude faster than collecting or generating real multimodal safety data.

## Limitations & Future Work
- The quality of synthetic embeddings heavily depends on the initial textual data and the design of the auxiliary samples.
- For extremely complex safety scenarios (e.g., those requiring the understanding of relationships among multiple objects in an image), the synthetic embedding may lack precision.
- VA-SafetyBench is directly converted from MM-SafetyBench one-to-one, potentially missing safety risks unique to video or audio modalities.

## Related Work & Insights
- **vs. Chakraborty et al. (2024) Text Alignment**: This work validates that text-only alignment enhances safety, but proves ineffective against attacks delivered solely via non-textual modalities. SEA resolves this limitation by utilizing synthetic embeddings.
- **vs. Generative Model Methods**: Generating safety training images using diffusion models is an alternative. However, (1) not all modalities possess high-quality generative models and (2) generation cost is high. SEA optimizes directly in the embedding space, offering a more general and efficient alternative.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The concept of synthetic embeddings is exceptionally simple, elegant, and effective
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers three modalities (image, video, and audio) and introduces a new benchmark
- Writing Quality: ⭐⭐⭐⭐ Clearly described methods with well-articulated insights
- Value: ⭐⭐⭐⭐⭐ Low-cost multimodal safety alignment with immense practical value

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] LSSF: Safety Alignment via Low-Rank Safety Subspace Fusion](lssf_safety_subspace.md)
- [\[ACL 2025\] QueryAttack: Jailbreaking Aligned Large Language Models Using Structured Non-natural Query Language](queryattack_jailbreaking_aligned_large_language_models_using_structured_non-natu.md)
- [\[ICLR 2026\] GuardAlign: Test-time Safety Alignment in Multimodal Large Language Models](../../ICLR2026/llm_alignment/guardalign_test-time_safety_alignment_in_multimodal_large_language_models.md)
- [\[ACL 2025\] Safety Alignment via Constrained Knowledge Unlearning](safety_alignment_via_constrained_knowledge_unlearning.md)
- [\[ICML 2026\] Towards Context-Invariant Safety Alignment for Large Language Models](../../ICML2026/llm_alignment/towards_context-invariant_safety_alignment_for_large_language_models.md)

</div>

<!-- RELATED:END -->
