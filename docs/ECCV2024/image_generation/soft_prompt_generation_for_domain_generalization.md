---
title: >-
  [Paper Note] Soft Prompt Generation for Domain Generalization
description: >-
  [ECCV 2024][Image Generation][domain generalization] This paper proposes SPG (Soft Prompt Generation), which introduces generative models to VLM prompt learning for the first time. By dynamically generating instance-specific soft prompts from images via CGAN, it stores domain knowledge in the generative model rather than prompt vectors, achieving superior domain generalization performance.
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "domain generalization"
  - "prompt learning"
  - "CLIP"
  - "CGAN"
  - "generative model"
date: 2026-05-08
content_hash: 6147a5ee48ca411c
---

# Soft Prompt Generation for Domain Generalization

**Conference**: ECCV 2024  
**arXiv**: [2404.19286](https://arxiv.org/abs/2404.19286)  
**Code**: [https://github.com/renytek13/Soft-Prompt-Generation-with-CGAN](https://github.com/renytek13/Soft-Prompt-Generation-with-CGAN)  
**Area**: Image Generation  
**Keywords**: domain generalization, prompt learning, CLIP, CGAN, generative model

## TL;DR

This paper proposes SPG (Soft Prompt Generation), which introduces generative models to VLM prompt learning for the first time. By dynamically generating instance-specific soft prompts from images via CGAN, it stores domain knowledge in the generative model rather than prompt vectors, achieving superior domain generalization performance.

## Background & Motivation

- VLMs such as CLIP achieve remarkable adaptation performance on downstream tasks through soft prompts.
- However, their generalization performance drops significantly under domain shifts.
- Limitations of prior prompt learning methods:
    - CoOp: Learns fixed prompts, which overfit the training distribution.
    - CoCoOp/DPL: Use MLPs to generate residual vectors to adjust fixed prompts, but simple MLPs struggle to capture complex image-prompt relationships.
    - CAE: Introduces a domain bank, but the prompts lack diversity.
- Shift in Core Idea: Instead of storing domain knowledge in the prompts, it is stored in the generative model, enabling the model to dynamically generate domain-adaptive prompts for each image.

## Method

### Overall Architecture

SPG consists of a two-stage training scheme and an inference stage:
1. **Stage I**: Learn domain prompt labels for each source domain.
2. **Stage II**: Use CGAN to learn to generate corresponding domain prompts from images.
3. **Inference**: The generator of CGAN directly generates instance-specific soft prompts for target domain images.

### Key Designs

**1. Domain Prompt Label Learning (Training Stage I)**

- Solitarily optimize an optimal soft prompt $v^{d_i}$ for each source domain $d_i$.
- Formulated under the CoOp framework with a context length of 4.
- Optimized via cross-entropy loss: $v^{d_i*} = \operatorname{argmin} \mathbb{E}[-\log p(y|x, v^{d_i})]$.
- The domain prompt labels encapsulate rich domain information and serve as the training targets for Stage II.

**2. CGAN Pre-training (Training Stage II)**

- Generator $G$: Receives noise $z$ and image embedding $f(x) \rightarrow$ generates soft prompts.
    - Input: $[z, f(x)]$ (concatenation of noise and CLIP image features).
    - Output: A prompt vector with the same shape as the domain prompt labels.
- Discriminator $D$: Receives the domain prompt label or generated prompt + image embedding $\rightarrow$ determines real or fake.
- Adversarial training objective: $\min_G \max_D V(G, D)$.
- To enhance stability, a gradient clipping strategy is incorporated.

**3. Inference**

- Only the generator of CGAN is utilized.
- Given a target domain image $x \rightarrow f(x) + \text{noise } z \rightarrow G(z|f(x)) \rightarrow$ instance-specific soft prompt.
- $p(y=i|x) = \operatorname{softmax}(\langle w_i, f(x) \rangle / \tau)$
- where $w_i = g([G(z|f(x)), c_i])$, with $g$ denoting the text encoder.

### Loss & Training

- Stage I: SGD optimizer, batch size 32, context length of 4.
- Stage II: AdamW optimizer, weight decay $1\mathrm{e}{-4}$.
    - Learning rate: $2\mathrm{e}{-3}$ (PACS/VLCS/TerraInc) / $2\mathrm{e}{-4}$ (OfficeHome/DomainNet).
    - Gradient clipping is applied to stabilize CGAN training.
- Backbone: ResNet50 and ViT-B/16.
- Model selection: Highest accuracy achieved on the training domain validation set.

## Key Experimental Results

### Main Results (Multi-source DG, ViT-B/16)

| Method | PACS | VLCS | OfficeHome | TerraInc | DomainNet | Average |
|------|------|------|-----------|----------|-----------|------|
| ZS-CLIP | 95.7 | 82.6 | 80.4 | 28.0 | 57.6 | 68.9 |
| CoOp | 95.4 | 82.5 | 82.0 | 33.0 | 56.2 | 69.8 |
| CoCoOp | 96.0 | 81.7 | 81.1 | 33.8 | 56.9 | 69.9 |
| MaPLe | 96.3 | 82.7 | 82.6 | 34.5 | 57.7 | 70.8 |
| **SPG** | **96.8** | **83.1** | **83.0** | **37.8** | **58.7** | **71.9** |

### Ablation Study

| Variant | PACS | VLCS | Average |
|------|------|------|------|
| w/o Domain prompt labels (using unified prompt) | 95.2 | 82.0 | Decreased |
| w/o CGAN (directly using domain prompt) | 95.8 | 82.4 | Decreased |
| Replacing CGAN with MLP | 96.0 | 82.5 | Decreased |
| Full SPG | **96.8** | **83.1** | **Best** |

### Key Findings

- SPG achieves the best performance across all five DG benchmarks, with an average gain of 1.1% (relative to MaPLe).
- Domain prompt labels are crucial, providing high-quality training targets for the CGAN.
- CGAN outperforms MLP, capturing more complex image-prompt mapping relations.
- The most significant gain (+3.3% relative to MaPLe) is observed on TerraIncognita, which represents the largest domain gap.
- SPG is also effective in Single-source DG and Multi-target DG settings.

## Highlights & Insights

1. **Paradigm Innovation**: Integrates generative models into VLM prompt learning for the first time, pioneering a new "prompt generation" paradigm.
2. **Shift of Domain Knowledge Storage**: Transitions from prompt vectors to generative model parameters, allowing for greater flexibility.
3. **Diversity of Instance-Specific Prompts**: CGAN naturally supports diverse prompt generation, with noise input introducing beneficial stochasticity.
4. **Clever Two-Stage Training Strategy**: Domain prompt labels act as a "teacher" to guide CGAN in acquiring domain knowledge.
5. **Simple and Effective**: SOTA results are achieved using the classic CGAN architecture, which is simple and easy to implement.

## Limitations & Future Work

- CGAN training instability necessitates techniques such as gradient clipping.
- Only CGAN is utilized as the generative model; stronger generative models (e.g., diffusion models) might yield better performance.
- The quality of domain prompt labels directly impacts CGAN training, making optimization in Stage I crucial.
- Visual-side prompt generation has not been explored (currently restricted to the textual side).

## Related Work & Insights

- **CoOp**: Pioneering work in fixed soft prompts.
- **CoCoOp**: Pioneer of image-conditional residual prompting.
- **CGAN**: Backbone of the generative approach.
- **DAPL**: A reference for prompt learning in domain adaptation.
- Insight: Shifting domain knowledge from "static parameter storage" to "dynamic generation" is an effective mechanism to improve generalization capabilities.

## Rating

| Dimension | Score (1-10) |
|------|-----------|
| Novelty | 8 |
| Technical Depth | 7 |
| Experimental Thoroughness | 8 |
| Value | 8 |
| Writing Quality | 7 |
| Overall Rating | 7.6 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] What's in a Latent? Leveraging Diffusion Latent Space for Domain Generalization](../../ICCV2025/image_generation/whats_in_a_latent_leveraging_diffusion_latent_space_for_domain_generalization.md)
- [\[ECCV 2024\] Source Prompt Disentangled Inversion for Boosting Image Editability with Diffusion Models](source_prompt_disentangled_inversion_for_boosting_image_editability_with_diffusi.md)
- [\[ECCV 2024\] HybridBooth: Hybrid Prompt Inversion for Efficient Subject-Driven Generation](hybridbooth_hybrid_prompt_inversion_for_efficient_subject-driven_generation.md)
- [\[ICLR 2026\] Soft-Di\[M\]O: Improving One-Step Discrete Image Generation with Soft Embeddings](../../ICLR2026/image_generation/soft-dimo_improving_one-step_discrete_image_generation_with_soft_embeddings.md)
- [\[ECCV 2024\] The Fabrication of Reality and Fantasy: Scene Generation with LLM-Assisted Prompt Interpretation](the_fabrication_of_reality_and_fantasy_scene_generation_with_llm-assisted_prompt.md)

</div>

<!-- RELATED:END -->
