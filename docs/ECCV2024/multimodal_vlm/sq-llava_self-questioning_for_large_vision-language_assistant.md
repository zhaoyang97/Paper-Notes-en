---
title: >-
  [Paper Note] SQ-LLaVA: Self-Questioning for Large Vision-Language Assistant
description: >-
  [ECCV 2024][Multimodal VLM][Visual Instruction Tuning] This paper proposes a Visual Self-Questioning (SQ) training paradigm, enabling LLMs to not only learn how to answer questions but also actively ask questions based on images. By fully exploiting the rich semantic information inherent in the questions themselves within instruction-following data, the proposed method enhances vision-language alignment.
tags:
  - "ECCV 2024"
  - "Multimodal VLM"
  - "Visual Instruction Tuning"
  - "Self-Questioning"
  - "Prototype Extractor"
  - "LoRA"
  - "Vision-Language Alignment"
date: 2026-05-08
content_hash: 2ed2d990fd3e4a42
---

# SQ-LLaVA: Self-Questioning for Large Vision-Language Assistant

**Conference**: ECCV 2024  
**arXiv**: [2403.11299](https://arxiv.org/abs/2403.11299)  
**Code**: [https://github.com/heliossun/SQ-LLaVA](https://github.com/heliossun/SQ-LLaVA)  
**Area**: Multimodal VLM  
**Keywords**: Visual Instruction Tuning, Self-Questioning, Prototype Extractor, LoRA, Vision-Language Alignment

## TL;DR
This paper proposes a Visual Self-Questioning (SQ) training paradigm, enabling LLMs to not only learn how to answer questions but also actively ask questions based on images. By fully exploiting the rich semantic information inherent in the questions themselves within instruction-following data, the proposed method enhances vision-language alignment.

## Background & Motivation

Current large vision-language models (e.g., the LLaVA series) achieve strong generalization across various vision tasks through visual instruction tuning. However, the **modality gap** between the pretrained vision encoder and the large language model remains a bottleneck for the entire network. Existing approaches typically collect more or higher-quality visual instruction data to improve cross-modal alignment, which is costly and fails to fully exploit the rich contextual information already embedded in images.

The authors reveal an overlooked key insight: in existing visual instruction datasets, **questions often contain more image-relevant information than answers**. To empirically verify this, they compute the CLIPScore on the LLaVA-instruct dataset, demonstrating that the average CLIPScore of questions ($\mu_q=0.184$) is higher than that of answers ($\mu_a=0.183$); a similar trend is observed on ShareGPT4V-instruct ($\mu_q=0.186 > \mu_a=0.184$). This suggests that questions encode richer visual cues.

**Key Challenge**: Traditional visual instruction tuning only trains the model to predict answers, entirely ignoring the semantic information embedded in the questions, which represents a waste of data resources.

**Key Insight**: Drawing an analogy to human learning, actively formulating questions requires deeper comprehension and background knowledge than merely answering them. Therefore, training a model to learn "how to ask questions" can foster a deeper level of vision-language alignment.

**Core Idea**: By incorporating "questioning" as an auxiliary training objective, the model leverages the question text in the instruction data in a self-supervised manner, thereby enhancing visual understanding without requiring any additional data collection.

## Method

### Overall Architecture
SQ-LLaVA consists of four core components:
1. **Pretrained Vision Encoder** (CLIP-ViT): Extracts the image token sequence embeddings $Z_v$
2. **Prototype Extractor** $\phi(\cdot)$: Enhances raw image token representations via clustering learning
3. **Trainable Projection Module** $W(\cdot)$: A two-layer MLP that maps the enhanced image tokens to the language domain $H_v$
4. **LLM Backbone** $f(\cdot)$: Based on pretrained Vicuna, performing autoregressive next-token prediction

The model follows the standard autoregressive prediction paradigm: $p_\theta(H_a^{(i+1)} | H_v, H_q, H_a^{(1:i)}) = \sigma(f(H_v, H_q, H_a^{(1:i)}))$

### Key Designs

1. **Visual Self-Questioning Instruction**:

    - **Function**: Defines a new special token `[vusr]` as a "questioning" instruction, prompting the model to actively generate relevant questions upon observing an image.
    - **Mechanism**: For the $j$-th turn in multi-turn dialogue data, `[usr]` is replaced with `[vusr]` with a 50% probability ($\delta=0.5$). In this case, the model is required to predict the question sequence instead of executing the user command. The training sequence format is: `System-message → [vusr] → X_q → [aswr] → X_a → <o^d>`
    - **Design Motivation**: In reality, formulating high-quality questions demands a higher level of comprehension than simply answering them. By learning to question, the model is forced to establish a deeper alignment between images and questions. After training, SQ-LLaVA can generate diverse questions in a zero-shot manner, including multiple-choice and reasoning questions, with diversity even surpassing GPT-4V.

2. **Prototype Extractor**:

    - **Function**: Extracts semantic prototypes from the latent space of image tokens via an EM clustering algorithm to enhance visual representations.
    - **Mechanism**: Randomly initializes $K=256$ cluster centroids $C$ and optimizes them through $T=2$ iterations of EM steps. The E-step computes the soft assignment matrix $\mathcal{M}^{(t)} = \sigma(q(C^{(t)}) \cdot k(Z_v)^\top)$, and the M-step updates the centroids $C^{(t+1)} = \mathcal{M}^{(t)} \cdot v(Z_v)$. The prototype information is then aggregated back to the original tokens via cosine similarity weighted summation: $Z_v^{(i)} = Z_v^{(i)} + z(\frac{1}{K}\sum_{j=1}^K S_c(C_j, Z_v^{(i)}) \times C_j)$
    - **Design Motivation**: Clustering groups semantically similar tokens together, enabling the prototypes to represent intrinsic semantics (e.g., "grass", "dog"), which enhances contextual understanding and compensates for representation limitations in the projection layer.

3. **ViT-LoRA + LLM-LoRA Co-tuning**:

    - **Function**: Implements LoRA adapters on both the vision encoder and the LLM simultaneously during the fine-tuning stage.
    - **Mechanism**: Configures ViT-LoRA ($\text{rank}=32, \alpha=64$) and LLM-LoRA ($\text{rank}=128, \alpha=256$) while keeping the pretrained weights frozen, training only the LoRA weights, the prototype extractor, and the projection layer.
    - **Design Motivation**: Achieves joint optimization of both the visual and language domains with extremely few trainable parameters, avoiding the high computational overhead of full fine-tuning.

### Loss & Training

**Two-Stage Training**:
- **Stage 1 Pretraining**: Freezes both the ViT and the LLM, training only the prototype extractor $\phi$ and the projection layer $W$. The objective is to maximize the probability of predicting image captions: $\sum_{v,a} -\log p_\theta(H_a | H_v)$
- **Stage 2 Fine-tuning**: Adds LoRA and optimizes two joint objectives—the self-questioning loss $-\log p_\theta(H_q^{(j+1)} | H_v, H_c^{(1:j)})$ and the answering loss $-\log p_\theta(H_a^{(j+1)} | H_v, H_c^{(1:j)}, H_q^{(j+1)})$

## Key Experimental Results

### Main Results

| Benchmark | SQ-LLaVA-7B | LLaVA-v1.5-7B | ShareGPT4V-7B | Key Takeaways |
|------|-------------|---------------|---------------|------|
| VQAv2 | 79.2 | 78.5 | 80.6 | +0.7 vs LLaVA |
| GQA | 62.8 | 62.0 | 63.3 | +0.8 vs LLaVA |
| VizWiz | 54.0 | 50.0 | 57.2 | +4.0 vs LLaVA |
| ScienceQA-IMG | 68.9 | 66.8 | 68.4 | +2.1 vs LLaVA |
| POPE | 87.7 | 85.9 | 86.8 | +1.8/+0.9 |
| MM-Vet | 32.5 | 30.5 | 37.6 | +2.0 vs LLaVA |
| LLaVA-Wild | 66.3 | 63.4 | 72.6 | +2.9 vs LLaVA |
| MMBench | 66.2 | 64.3 | 68.8 | +1.9 vs LLaVA |

SQ-LLaVA-7B outperforms LLaVA-v1.5-7B on 9 out of 10 benchmarks, and the 13B scale similarly achieves gains on 8/10 benchmarks.

### Ablation Study

| Configuration (558K PT + 665K IT) | VizWiz | SQAI | VQAT | POPE | LLaVAW | Avg. |
|----------------------------------|--------|------|------|------|--------|------|
| Baseline (w/o LoRA/SQ/Proto) | 49.4 | 68.4 | 58.2 | 86.5 | 67.1 | 65.9 |
| + V-LoRA + Proto | 52.4 | 67.9 | 58.6 | 87.7 | 65.6 | 66.4 |
| + V-LoRA + SQ | 52.6 | 68.4 | 57.8 | 88.2 | 67.3 | 66.9 |
| + SQ + Proto | 53.4 | 69.3 | 58.1 | 87.9 | 67.9 | 67.3 |
| Full (V-LoRA + SQ + Proto) | 54.0 | 68.9 | 58.6 | 87.7 | 68.1 | **67.5** |

- The full model achieves an average improvement of **2.4%** (small-scale dataset) / **3.0%** (large-scale dataset) over the baseline.
- The SQ module provides the most consistent contribution, yielding performance benefits across all benchmarks even when used in isolation.

### Key Findings
- Visual Self-Questioning (SQ) consistently yields improvements across almost all benchmarks, validating the hypothesis that "learning to question improves comprehension."
- The improvement on the POPE benchmark demonstrates that SQ-LLaVA effectively alleviates object hallucination and enhances model reliability.
- On zero-shot image captioning tasks, SQ-LLaVA outperforms LLaVA-v1.5 by an average of 2%, demonstrating the ability to generate captions containing fine-grained concepts (e.g., the brand name "Hyundai").

## Highlights & Insights
- **Questions are more visually relevant than answers**: This intuition is empirically validated using CLIPScore, which pioneers a new dimension in visual instruction tuning.
- **Zero-data overhead gain**: No additional data collection is required; substantial performance gains are achieved solely by repurposing question text in existing datasets.
- **Significantly reduced trainable parameters**: Incorporating the LoRA strategy results in a fine-tuning cost far below full fine-tuning schemes.

## Limitations & Future Work
- The prototype extractor is unsupervised and lacks pixel-level guidance, which may cause instability on certain datasets. This could be addressed by leveraging pretrained segmentation models (e.g., SAM) to provide pseudo-target masks.
- The model still lags behind ShareGPT4V on certain benchmarks (e.g., VQAv2 and MMBench), revealing that data quality and scale remain critical factors.
- The self-questioning threshold $\delta = 0.5$ is fixed, leaving dynamic adjustment strategies unexplored.

## Related Work & Insights
- **vs LLaVA-v1.5**: SQ-LLaVA achieves extra gains on identical training data through self-questioning, demonstrating that data utilization efficiency can be further improved.
- **vs ShareGPT4V**: SQ-LLaVA* matches or outperforms ShareGPT4V on certain benchmarks while employing significantly fewer trainable parameters.
- **vs GPT-4V (Questioning Capability)**: GPT-4V requires explicit textual instructions and prompt engineering to formulate questions, whereas SQ-LLaVA generates questions in a zero-shot manner with higher diversity.

## Rating
- Novelty: ⭐⭐⭐⭐ Formulating "questioning" as a training objective is a pioneering attempt in the visual instruction tuning paradigm, offering a novel perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluation across 10 VQA benchmarks and 4 captioning datasets, combined with detailed ablation studies, though deeper theoretical analysis is somewhat limited.
- Writing Quality: ⭐⭐⭐⭐ Clarified motivations, intuitive CLIPScore validation, and overall coherent logical flow.
- Value: ⭐⭐⭐⭐ Delivers a zero-cost performance enhancement scheme that holds inspiring implications for the community, although the absolute gains remain moderate.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] LLaVA-KD: A Framework of Distilling Multimodal Large Language Models](../../ICCV2025/multimodal_vlm/llava-kd_a_framework_of_distilling_multimodal_large_language_models.md)
- [\[ECCV 2024\] Self-Adapting Large Visual-Language Models to Edge Devices across Visual Modalities](self-adapting_large_visual-language_models_to_edge_devices_across_visual_modalit.md)
- [\[ECCV 2024\] Robust Calibration of Large Vision-Language Adapters](robust_calibration_of_large_vision-language_adapters.md)
- [\[ECCV 2024\] Vary: Scaling up the Vision Vocabulary for Large Vision-Language Models](vary_scaling_up_the_vision_vocabulary_for_large_visionlanguag.md)
- [\[ECCV 2024\] Attention Prompting on Image for Large Vision-Language Models](attention_prompting_on_image_for_large_visionlanguage_models.md)

</div>

<!-- RELATED:END -->
