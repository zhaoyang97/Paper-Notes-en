---
title: >-
  [Paper Note] Segment-Level Diffusion: A Framework for Controllable Long-Form Generation with Diffusion Language Models
description: >-
  [ACL 2025][LLM (Other)][Diffusion Models] This paper proposes Segment-Level Diffusion (SLD), which segments long-form text outputs into multiple segments (such as sentences or dialogue turns) and models the latent representations of each segment using diffusion. Combined with contrastive learning and adversarial training to enhance representation robustness, SLD achieves superior long-form generation quality compared to existing diffusion models on tasks such as summarization…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Diffusion Models"
  - "Long-Form Text Generation"
  - "Segment-Level Diffusion"
  - "Contrastive Learning"
  - "Adversarial Training"
date: 2026-05-08
content_hash: d9dc2e53a8969977
---

# Segment-Level Diffusion: A Framework for Controllable Long-Form Generation with Diffusion Language Models

**Conference**: ACL 2025  
**arXiv**: [2412.11333](https://arxiv.org/abs/2412.11333)  
**Code**: [https://github.com/SpaceHunterInf/Segment_Level_Diffusion](https://github.com/SpaceHunterInf/Segment_Level_Diffusion)  
**Area**: LLM/NLP  
**Keywords**: Diffusion Models, Long-Form Text Generation, Segment-Level Diffusion, Contrastive Learning, Adversarial Training  

## TL;DR
This paper proposes Segment-Level Diffusion (SLD), which segments long-form text outputs into multiple segments (such as sentences or dialogue turns) and models the latent representations of each segment using diffusion. Combined with contrastive learning and adversarial training to enhance representation robustness, SLD achieves superior long-form generation quality compared to existing diffusion models on tasks such as summarization, story generation, and dialogue generation.

## Background & Motivation
**Background**: Text diffusion models primarily fall into two categories: token-level diffusion (e.g., GENIE), which denoises directly in the word embedding space, and passage-level diffusion (e.g., LD4LG), which compresses the entire text passage into a single latent representation before denoising.

**Limitations of Prior Work**: (a) Token-level diffusion does not explicitly model token-order dependencies, which often results in incoherent text, and its fixed output window limits the generation length; (b) Passage-level diffusion struggles to learn robust latent representations for long-form text, where minor perturbations can cause abrupt semantic shifts, leading to poor coherence in the decoded text.

**Key Challenge**: Diffusion models excel at global planning but lack local fluency guarantees; autoregressive models excel at local fluency but struggle with global planning. Long-form text generation requires both capabilities simultaneously.

**Goal**: How to enable diffusion language models to generate long-form, coherent, and contextually consistent text while maintaining controllability.

**Key Insight**: Drawing inspiration from the concept of patches in image generation—segmenting long-form text into segments, using independent latent representations for each segment, utilizing the diffusion model for segment-level semantic planning, and employing an autoregressive decoder to guarantee local fluency.

**Core Idea**: Shift the diffusion granularity from token/passage to segment (sentence/turn), allowing the diffusion model to focus on semantic planning while the AR decoder focuses on text fluency.

## Method

### Overall Architecture
The input is context text $\mathbf{i}$, and the output is long-form text $\mathbf{o}$. The training consists of three stages: (1) segmenting the output text; (2) learning robust latent representations through contrastive learning and adversarial training; (3) training the diffusion model as a segment-level semantic planner. During inference, starting from Gaussian noise and conditioned on the input text, the diffusion model iteratively denoises to generate latent representations for multiple segments, which are then decoded into text in parallel by an AR decoder.

### Key Designs

1. **Output Segmentation**:

    - **Function**: Segment the output text into multiple segments $\mathbf{P}=\{\mathbf{p}^1,...,\mathbf{p}^j\}$ based on natural boundaries (e.g., sentences, dialogue turns).
    - **Mechanism**: Each segment is independently encoded into a latent representation $\mathbf{Z}=\{\mathbf{z}^1,...,\mathbf{z}^j\}$, establishing a one-to-one mapping between segments and latent representations.
    - **Design Motivation**: Compared to compressing an entire text passage into a single representation, segment-level representation reduces the complexity of each representation and simplifies the prediction difficulty for the diffusion model. Similar to patches in images, this allows the model to flexibly scale to various output lengths.

2. **Robust Representation Learning**:

    - **Function**: Train an encoder-decoder system to ensure latent representations are robust to noise and semantically consistent.
    - **Mechanism**: Combine three loss functions:
        - Conversion loss $\mathcal{L}_{\text{cnv}}$: Standard cross-entropy to ensure the encode $\rightarrow$ compress $\rightarrow$ reconstruct $\rightarrow$ decode process accurately recovers the original text.
        - Contrastive loss $\mathcal{L}_{\text{cst}}$: Apply contrastive learning using paraphrases (positive samples) and out-of-domain text (negative samples) to ensure semantically similar segments are close in the latent space.
        - Adversarial loss $\mathcal{L}_{\text{adv}}$: Add adversarial noise $\mathbf{r}_{adv} = -\epsilon_{adv}\frac{\mathbf{g}}{||\mathbf{g}||_2}$ to simulate the worst-case scenario during diffusion denoising, training the decoder to decode correctly even under noisy conditions.
    - **Design Motivation**: The denoising process of diffusion models inherently operates on noisy representations. If the latent space is not smooth, small disturbances can cause drastic semantic shifts. Contrastive learning guarantees semantic clustering, while adversarial training ensures noise tolerance.

3. **Diffusion Semantic Planner**:

    - **Function**: After training the encoder-decoder in the second stage, freeze them and train the diffusion model (DiT) to predict segment-level latent representations.
    - **Mechanism**: The diffusion model performs conditional generation using the encoded input text as the cross-attention target. In addition to the standard denoising loss, a post-diffusion control loss is incorporated—reconstruction loss $\mathcal{L}_{\text{rec}}$ and conversion loss $\mathcal{L}_{\text{cnv}}$—to propagate decoder feedback back to the diffusion model.
    - **Design Motivation**: Pure diffusion loss only optimizes in the latent space. Incorporating post-diffusion losses introduces the signal of "whether the decoder can correctly map representations back into text," similar to pixel-level guidance in image diffusion.

### Loss & Training
- Total loss for Phase 2: $\mathcal{L}_{\text{rep}} = \frac{1}{N}\sum_{\mathbf{p}}(\mathcal{L}_{\text{cnv}} + \lambda_1\mathcal{L}_{\text{cst}} + \lambda_2\mathcal{L}_{\text{adv}})$
- Total loss for Phase 3: $\mathcal{L}_{\text{diff}} = \mathcal{L}_{\text{noise}} + \gamma_1\mathcal{L}_{\text{rec}} + \gamma_2\mathcal{L}_{\text{cnv}}$
- Flan-T5 is used as the encoder/decoder backbone, and DiT is used as the diffusion model.
- The three stages are trained separately, without end-to-end joint optimization.

## Key Experimental Results

### Main Results

| Dataset | Model | ROUGE-L | PPL | Fluency | Coherence | Compatibility |
|--------|------|---------|-----|--------|--------|--------|
| ROCStories | Flan-T5 | 14.73 | 22.83 | 2.58 | 2.15 | 1.90 |
| ROCStories | LD4LG | 16.57 | 65.32 | 1.78 | 1.54 | 1.79 |
| ROCStories | **SLD** | **16.13** | **43.67** | **2.41** | **2.10** | **2.42** |
| DialogSum | Flan-T5 | 26.34 | 3.78 | 2.60 | 2.07 | 2.27 |
| DialogSum | LD4LG | 20.90 | 43.82 | 1.43 | 1.39 | 1.61 |
| DialogSum | **SLD** | **27.97** | **16.39** | **2.83** | **2.40** | **2.57** |
| DeliData | Flan-T5 | 25.83 | 9.79 | 2.59 | 2.30 | 2.48 |
| DeliData | LD4LG | 21.14 | 51.10 | 2.06 | 1.78 | 1.63 |
| DeliData | **SLD** | **30.51** | **13.41** | **2.60** | **2.30** | **2.48** |

### Ablation Study

| Configuration | Text Reconstruction Quality | Explanation |
|------|-------------|------|
| LD4LG (Short segments) | BLEU=1.00 | Perfect recovery of short text |
| LD4LG (Long segments) | BLEU degradation | Poor generalization on long text |
| SLD w/o Contrastive Learning | Word corruption | Occasionally replaces words with incorrect ones, shifting semantics |
| SLD w/ Contrastive Learning | Meaningful paraphrasing | Maintains semantic consistency even if imperfect |
| SLD w/ Contrastive + Adversarial | Most robust | Least sensitive to noise, most stable decoding |

### Key Findings
- SLD comprehensively outperforms LD4LG on long-form text generation ($\ge 50$ tokens), especially in terms of fluency, coherence, and compatibility.
- The core value of contrastive learning lies in preventing "semantic mutation"—without it, representations of semantically similar and unrelated texts in the latent space become intertwined.
- The AR decoder is crucial—methods utilizing AR decoding (Flan-T5, LD4LG, SLD) substantially outperform GENIE, which directly decodes token embeddings, in terms of fluency.
- SLD maintains a 72.3% accuracy in knowledge preservation tests (ECQA QA), close to the 75% accuracy of Flan-T5 Base.

## Highlights & Insights
- **Segment-level diffusion is the "sweet spot" between token-level and passage-level**—the granularity perfectly balances global planning and local fluency. This strategy can be transferred to other modalities (e.g., clip-level diffusion in video generation).
- **Simulating the "worst-case" diffusion denoising via adversarial training** is an ingenious design—traditional autoencoders are trained on clean inputs, whereas during diffusion inference, the inputs are noisy, causing a train-inference mismatch. Adversarial noise bridges this gap.
- **Post-diffusion control loss** backpropagates decoder signals to the diffusion model, acting similarly to pixel-level guidance in image diffusion. This provides a general and effective technique.

## Limitations & Future Work
- Evaluated only on English; multi-lingual capabilities are not verified.
- Training the three stages separately may lead to error accumulation; end-to-end joint training could be more optimal.
- No inference speed comparison is conducted; iterative denoising in diffusion sampling remains relatively slow.
- The selection of the compression dimension $h_{rep}$ and the number of segments $j$ lacks theoretical guidance and is chosen empirically.
- Lacks comparison with state-of-the-art large language models (e.g., GPT-4).

## Related Work & Insights
- **vs LD4LG**: Both utilize latent diffusion, but LD4LG encodes the entire text using a single representation, which is unstable for long texts; SLD segments the text, rendering each representation simpler and more robust.
- **vs GENIE**: GENIE performs token-level diffusion without an AR decoder, leading to poor fluency; SLD's AR decoder ensures fluent output.
- **vs Flan-T5**: Autoregressive methods tend to degenerate into repetitive outputs after fine-tuning on long texts (catastrophic repetition); SLD's diffusion planning avoids this issue.
- The concept of segment-level diffusion can be applied to tasks requiring global planning, such as controllable story generation and outline-controlled writing.

## Rating
- Novelty: ⭐⭐⭐⭐ Segment-level diffusion is a natural and effective intermediate granularity choice. The representation learning design via contrastive and adversarial training is solid.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on four datasets, including human evaluation, ablation studies, and representation analysis, but lacks comparisons with large LLMs.
- Writing Quality: ⭐⭐⭐⭐ Clearly structured and detailed in method descriptions, though some formulas are relatively dense.
- Value: ⭐⭐⭐⭐ Provides a feasible solution for applying diffusion language models to long-form text generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] DiffLM: Controllable Synthetic Data Generation via Diffusion Language Models](difflm_controllable_synthetic_data_generation_via_diffusion_language_models.md)
- [\[ACL 2025\] EdiText: Controllable Coarse-to-Fine Text Editing with Diffusion Language Models](editext_diffusion_text_editing.md)
- [\[ACL 2025\] LongDPO: Unlock Better Long-form Generation Abilities for LLMs via Critique-augmented Stepwise Information](longdpo_unlock_better_long-form_generation_abilities_for_llms_via_critique-augme.md)
- [\[ACL 2025\] Beyond In-Context Learning: Aligning Long-form Generation of LLMs via Task-Inherent Attribute Guidelines](beyond_in-context_learning_aligning_long-form_generation_of_large_language_model.md)
- [\[ACL 2025\] Cheaper and Better Diffusion Language Model via Task-Specific Training](cheaper_and_better_diffusion_language_model_via_task-specific_training.md)

</div>

<!-- RELATED:END -->
