---
title: >-
  [Paper Note] Large Language Models Meet Extreme Multi-label Classification: Scaling and Multi-modal Framework
description: >-
  [AAAI 2026][Image Restoration][Extreme Multi-label Classification] This paper investigates the effective utilization of decoder-based LLMs for Extreme Multi-label Classification (XMC)…
tags:
  - "AAAI 2026"
  - "Image Restoration"
  - "Extreme Multi-label Classification"
  - "Large Language Models"
  - "Visual Metadata"
  - "Siamese Learning"
  - "Dual-Decoder"
date: 2026-05-08
content_hash: c68a4f68ef276809
---

# Large Language Models Meet Extreme Multi-label Classification: Scaling and Multi-modal Framework

**Conference**: AAAI 2026
**arXiv**: [2511.13189](https://arxiv.org/abs/2511.13189)
**Code**: [https://github.com/DiegoOrtego/vixml](https://github.com/DiegoOrtego/vixml)
**Area**: Image Restoration
**Keywords**: Extreme Multi-label Classification, Large Language Models, Visual Metadata, Siamese Learning, Dual-Decoder

## TL;DR

This paper investigates the effective utilization of decoder-based LLMs for Extreme Multi-label Classification (XMC), proposing a dual-decoder learning strategy and the ViXML multimodal framework. By employing structured prompt templates to adapt LLM embeddings and efficiently integrating visual metadata, the method substantially outperforms state-of-the-art approaches on four public benchmarks (up to +8.21% P@1 on the largest dataset), demonstrating that "one image outweighs billions of parameters."

## Background & Motivation

**Background**: Extreme Multi-label Classification (XMC) requires predicting the most relevant subset of labels from a million-scale label space for a given query, with broad applications in product recommendation and document tagging. Mainstream methods adopt Siamese-style contrastive learning, matching query and label embeddings via maximum inner product search, typically using small encoder-only models such as DistilBERT (66M).

**Limitations of Prior Work**: (1) **Limited model scale**—existing methods predominantly rely on small encoder-only models, leaving the potential of scaling largely untapped. Although decoder-based LLMs have demonstrated clear advantages in text embedding, their application to XMC has not yet succeeded (e.g., QUEST with Llama-7B significantly underperforms encoder models); (2) **Underutilization of metadata**—while textual and categorical metadata have been explored, visual metadata (e.g., product images) is almost entirely overlooked, with MUFIN being the sole exception.

**Key Challenge**: LLMs perform strongly on general text embedding benchmarks, yet how to leverage them effectively in XMC—and how to achieve performance gains while maintaining computational feasibility—remains an open problem. Moreover, XMC is sensitive to sequence length; directly incorporating the hundreds of visual tokens from VLMs leads to prohibitive computational costs.

**Goal**: (1) How can decoder-based LLMs be effectively adapted for Siamese learning in XMC? (2) How can visual metadata be efficiently integrated without substantially increasing computational overhead?

**Key Insight**: The authors propose two complementary paths—a scaling path (adapting LLMs via structured prompts and dual-decoder learning) and an efficiency path (injecting visual information via single-image embeddings)—which can be combined.

**Core Idea**: Structured prompt templates are used to adapt decoder-based LLMs as dual encoders for XMC, while a single embedding from a frozen vision model is used for early-fusion integration of visual metadata.

## Method

### Overall Architecture

ViXML is a general multimodal XMC framework supporting both encoder and decoder architectures. The input consists of query-label pairs with optional image metadata. The embedding model is trained via Siamese-style contrastive learning, and label prediction is performed through maximum inner product search. The core contributions lie in two dimensions: (1) dual-decoder learning strategy, and (2) efficient visual metadata fusion.

### Key Designs

1. **Dual-Decoder Learning**:

    - **Function**: Adapts decoder-based LLMs as Siamese encoders for XMC.
    - **Mechanism**: For each query $q_i$, a structured prompt is constructed as $\mathcal{E}'_i = \mathcal{T} \oplus \mathcal{E}_i \oplus \mathbf{e}_{EOS}$, where $\mathcal{T}$ is a text prefix (e.g., "This product text") and $\mathbf{e}_{EOS}$ is the end-of-sequence token. Causal attention is preserved (consistent with pretraining), sentence embeddings are extracted via mean pooling, and contrastive learning is performed with triplet loss. LoRA fine-tuning is employed to control overhead, and training epochs are substantially reduced from 300 (encoder) to 30 (LLMs exhibit higher sample efficiency).
    - **Design Motivation**: The concise prompt template provides task context without increasing sequence length; preserving causal attention avoids deviation from the pretraining distribution; the dramatic reduction in training epochs makes the 0.5B decoder comparable in training time to 66M DistilBERT.

2. **ViXML Visual Fusion Framework**:

    - **Function**: Injects image information into XMC models without significantly increasing computational overhead.
    - **Mechanism**: A frozen foundation vision model (e.g., SigLIPv2-1.14B) compresses each image into a **single** embedding $\mathbf{v}$, which is projected into the text embedding space via a learnable linear layer. For encoder models, image and text embeddings are concatenated directly: $\mathcal{E}'_i = \mathcal{V}_i \oplus \mathcal{E}_i$. For decoder models, the image embedding is placed within the structured prompt: $\mathcal{E}'_i = \mathcal{T} \oplus \mathcal{E}_i \oplus \mathcal{I} \oplus \mathcal{V}_i \oplus \mathbf{e}_{EOS}$, positioned after the text and before the EOS token.
    - **Design Motivation**: Using a single embedding per image keeps sequence length low. The frozen visual encoder allows embeddings to be precomputed as a feature cache, adding negligible memory overhead during training. Early fusion is adopted to enable mutual enhancement between text and visual representations via the attention mechanism. Experiments show that placing image tokens after the text is essential—LLM pretraining causes the first token to form an attention sink, and placing image tokens at the beginning disrupts this dynamic.

3. **Prompt Template Design Strategy**:

    - **Function**: Identifies the optimal input organization for decoder models.
    - **Mechanism**: Multiple prompt combinations are empirically evaluated; text prefix combined with an EOS token is found to provide structural cues that help leverage pretrained knowledge. Placing images after the text with prefix markers yields the best results.
    - **Design Motivation**: Performance gains arise from structural cues rather than injected information. The pretrained attention patterns of LLMs (attention sink) must be respected.

### Loss & Training

- The base optimization uses NGAME's triplet loss: $\mathcal{L} = \sum_{i=1}^{B} \sum_{j \in \mathcal{P}_i, k \in \mathcal{N}_i} [\mathbf{h}_q^i \cdot \mathbf{h}_n^k - \mathbf{h}_q^i \cdot \mathbf{h}_p^j + m]_+$
- Main experiments employ the PRIME method (incorporating label prototype networks and enriched label representations).
- Decoder models are fine-tuned with LoRA; all experiments are conducted on a single 80GB GPU.

## Key Experimental Results

### Main Results

| Dataset | Metric | ViXML (Ours) | MOGIC | PRIME | Gain |
|---------|--------|--------------|-------|-------|------|
| LF-AmazonTitles-131K | P@1 | **53.08** | 47.01 | 45.26 | +6.07 |
| MM-AmazonTitles-300K | P@1 | **57.37** | — | — | vs MUFIN 52.30: +5.07 |
| LF-AmazonTitles-1.3M | P@1 | **67.83** | 50.95 | 59.62 | +8.21 |
| LF-Amazon-131K | P@1 | **55.11** | — | 49.15 | +5.96 |

Key finding: ViXML with 66M DistilBERT + image surpasses text-only billion-parameter models in most settings.

### Ablation Study

| Configuration | P@1 | P@5 | Notes |
|---------------|-----|-----|-------|
| PRIME (text-only, DistilBERT) | 44.86 | 21.45 | Baseline |
| PRIME (text-only, Qwen2.5-3B) | 47.42 | 22.89 | Scaling gain |
| ViXML (DistilBERT) | 49.55 | 23.73 | Large gain from image fusion |
| ViXML (Qwen2.5-3B) | 52.47 | 25.26 | Both paths combined |
| ViXML + MUFIN late-fusion | 52.62 | 34.35 | Early fusion outperforms late fusion |
| ViXML + PRIME early-fusion | 55.03 | 35.91 | Proposed method is superior |

### Key Findings

- **"One image outweighs billions of parameters"**: ViXML with a 66M encoder surpasses text-only billion-parameter decoder models on most datasets, demonstrating the exceptional effectiveness of visual metadata.
- **Scaling is effective but with diminishing returns**: A clear improvement is observed from encoder to decoder, but general-purpose pretrained embedding models (e.g., Qwen3-Embedding) perform poorly on XMC (P@1 only 18–22), confirming that task-specific fine-tuning is necessary.
- **Early fusion outperforms late fusion**: Under equivalent conditions, ViXML exceeds MUFIN's late-fusion approach by 1.5%+ in P@1.
- **Training efficiency**: ViXML accelerates encoder training convergence, enabling the number of epochs to be reduced from 300 to 150.

## Highlights & Insights

- The **minimalist design of single-image embeddings** is particularly elegant: VLMs typically require hundreds of visual tokens, causing computational explosion in XMC. This paper compresses each image into one embedding with a linear adaptation layer, incurring negligible overhead while achieving remarkable performance gains. This design principle is broadly transferable to other tasks requiring visual information injection in long-sequence settings.
- **Application of the attention sink observation**: The authors find that LLM pretraining causes the first token to form an attention sink; placing image tokens at the beginning of the sequence leads to performance collapse, whereas placing them after the text is effective. This reflects a practically grounded understanding of LLM internal mechanisms.
- Three existing text-only datasets are extended with visual metadata, contributing new benchmarks for multimodal XMC research.

## Limitations & Future Work

- Only linear projection layers are used to adapt visual embeddings; more sophisticated adaptation mechanisms may yield further improvements.
- Visual metadata must be available at inference time, limiting deployment in text-only scenarios.
- Experiments are confined to Amazon e-commerce datasets; generalization to other XMC domains such as academic documents and news has not been verified.
- Inference latency for decoder models remains substantially higher than for encoders; production deployment would require acceleration solutions such as vLLM.

## Related Work & Insights

- **vs. MOGIC**: MOGIC also attempts to apply LLMs to XMC with limited success. This paper successfully unlocks the potential of LLMs in XMC through structured prompting and optimized training strategies.
- **vs. MUFIN**: MUFIN is the pioneering work utilizing visual metadata in XMC, but relies on late fusion and requires training additional classifiers and fusion modules. ViXML achieves a simpler and more efficient solution via early fusion.
- **vs. QUEST**: QUEST applies Llama-7B to XMC and significantly underperforms encoder models, demonstrating that naive adaptation is insufficient. This paper shows that carefully designed prompts and training strategies are critical.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of dual-decoder learning and single-embedding visual fusion is novel, though each component in isolation is not complex.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four datasets, multiple backbones, detailed ablations, and cross-method compatibility validation are highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and systematic experimental organization.
- Value: ⭐⭐⭐⭐ The work provides direct practical guidance for the XMC community; the finding that "one image outweighs billions of parameters" is highly insightful.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MRO: Enhancing Reasoning in Diffusion Language Models via Multi-Reward Optimization](../../NeurIPS2025/image_restoration/mro_enhancing_reasoning_in_diffusion_language_models_via_multi-reward_optimizati.md)
- [\[AAAI 2026\] Clear Nights Ahead: Towards Multi-Weather Nighttime Image Restoration](clear_nights_ahead_towards_multi-weather_nighttime_image_res.md)
- [\[AAAI 2026\] Hard vs. Noise: Resolving Hard-Noisy Sample Confusion in Recommender Systems via Large Language Models](hard_vs_noise_resolving_hard-noisy_sample_confusion_in_recommender_systems_via_l.md)
- [\[ACL 2026\] CreditDecoding: Accelerating Parallel Decoding in Diffusion Large Language Models with Trace Credit](../../ACL2026/image_restoration/creditdecoding_accelerating_parallel_decoding_in_diffusion_large_language_models.md)
- [\[ACL 2026\] Lost in Diffusion: Uncovering Hallucination Patterns and Failure Modes in Diffusion Large Language Models](../../ACL2026/image_restoration/lost_in_diffusion_uncovering_hallucination_patterns_and_failure_modes_in_diffusi.md)

</div>

<!-- RELATED:END -->
