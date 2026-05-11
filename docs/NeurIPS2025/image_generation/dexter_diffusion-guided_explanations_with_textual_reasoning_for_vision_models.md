---
title: >-
  [Paper Note] DEXTER: Diffusion-Guided EXplanations with TExtual Reasoning for Vision Models
description: >-
  [Image Generation] This paper proposes DEXTER, a data-free framework that optimizes textual prompts to drive a diffusion model to generate images maximizing target classifier activations…
tags:
  - "Image Generation"
date: 2026-05-08
content_hash: 299dd4a7a542b8ca
---

# DEXTER: Diffusion-Guided EXplanations with TExtual Reasoning for Vision Models

## Basic Information
- **arXiv**: 2510.14741
- **Conference**: NeurIPS 2025
- **Authors**: Simone Carnemolla, Matteo Pennisi, Sarinda Samarasinghe, Giovanni Bellitto, Simone Palazzo, Daniela Giordano, Mubarak Shah, Concetto Spampinato
- **Institutions**: University of Catania, University of Central Florida
- **Code**: https://github.com/perceivelab/dexter

## TL;DR
This paper proposes DEXTER, a data-free framework that optimizes textual prompts to drive a diffusion model to generate images maximizing target classifier activations, then employs an LLM to reason over the synthesized samples and produce globally coherent, human-readable textual explanations, enabling bias discovery and global interpretation of model behavior.

## Background & Motivation
Model interpretability is fundamental to building trustworthy AI. Limitations of existing approaches:

1. **Local attribution methods** (GradCAM, Integrated Gradients): explain individual predictions only, without providing global understanding.
2. **Activation Maximization (AM)**: generated images are abstract and difficult to interpret semantically.
3. **Textual explanation methods (NLE)**: typically require annotated data and pre-trained vision-language mappings.
4. **Bias discovery methods** (B2T, LADDER): require training data for misclassification analysis.

**Core need**: A global explanation method that requires **no training data or labels whatsoever**, capable of describing a classifier's decision patterns and biases in natural language.

## Core Problem
How to systematically reveal and explain the decision process of deep visual classifiers — including feature preferences, bias patterns, and spurious correlations — under a completely data-free setting?

## Method

### 1. Overall Architecture
Three main pipelines:
- **Text pipeline**: optimizes soft prompts → BERT predicts mask tokens → obtains textual prompts.
- **Visual pipeline**: text prompts condition Stable Diffusion → generates images that maximize target neuron activations.
- **Reasoning module**: VLM captions the generated images → LLM reasons across captions → produces a structured textual bias report.

### 2. Text Pipeline: Soft Prompt → Hard Token
Input structure: $\mathbf{t} = [\mathbf{t}_\text{fixed}, m_1, m_2, \ldots, m_N]$, where $\mathbf{t}_\text{fixed}$ = "a picture of a"

- Learnable soft prompts $\mathbf{p} \in \mathbb{R}^{P \times d}$ ($P=1, d=768$) are prepended before BERT embeddings.
- BERT outputs logits $\mathbf{l}_i \in \mathbb{R}^V$ at mask positions.
- **Gumbel-Softmax** ($\tau=1$) converts logits into differentiable one-hot vectors $\mathbf{o}_i$.
- **BERT→CLIP vocabulary mapping**: translation matrix $\mathbf{M} \in \{0,1\}^{V \times W}$:
  $$\mathbf{o}_i^{(C)} = \mathbf{o}_i \mathbf{M}$$
  Tokens in BERT with no CLIP counterpart are automatically excluded (corresponding rows are all-zero).

### 3. Visual Pipeline: Activation Maximization
The text is encoded into CLIP embeddings $\mathbf{e}$, which condition Stable Diffusion to generate images fed into the target classifier $f$, yielding neuron activations $\mathbf{n} = f(d(\mathbf{e}))$.

Activation maximization loss:
$$\mathcal{L}_\text{act} = \sum_{i=1}^K l_\text{act}(n_i), \quad l_\text{act}(n_i) = \begin{cases} -n_i, & \text{feature neuron} \\ -\log n_i, & \text{class neuron} \end{cases}$$

### 4. Auxiliary Mask Pseudo-label Prediction
To address the problem of weak gradient propagation through soft prompts, an auxiliary cross-entropy loss is introduced:
- Maintains pseudo-labels $y_i$ and reference losses $L_i$.
- Aggregates activation losses of associated neurons: $\mathcal{L}_{\text{agg},i} = \sum_{j \in \mathcal{N}_i} l_\text{act}(n_j)$
- Uses historical mean to prevent outliers from corrupting pseudo-label updates:
$$\frac{1}{T} \sum_{j=1}^T \mathcal{L}_{\text{agg},i}^{(j)} < L_i$$

**Total loss**:
$$\mathcal{L} = \sum_{k=1}^K l_\text{act}(n_k) - \sum_{i=1}^N \log s_{i, y_i}$$

### 5. Bias Reasoning
50 images are generated per target class → ChatGPT-4o mini produces per-image captions → LLM reasons across captions → outputs a structured bias report.

## Key Experimental Results

### Activation Maximization (SalientImageNet, 30 classes)

| Method | Spurious | Core | Average |
|--------|----------|------|---------|
| Baseline (class name) | 43.06 | 86.40 | 64.73 |
| ChatGPT description | 41.20 | 78.53 | 59.87 |
| DiffExplainer | 33.20 | 47.66 | 39.83 |
| **DEXTER** | **63.00** | **87.86** | **75.43** |

### Slice Discovery & Debiasing (Worst-Slice Accuracy)

| Method | Requires Data | CelebA Worst | Waterbirds Worst |
|--------|--------------|-------------|-----------------|
| ERM | ✓ | 47.7 | 62.6 |
| DRO | ✓ + GT | 90.0 | 89.9 |
| DRO-B2T | ✓ | 90.4 | 90.7 |
| LADDER | ✓ | 89.2 | **92.4** |
| **DEXTER** | **✗** | **91.3** | 90.5 |

- On CelebA, DEXTER outperforms all methods — including data-dependent ones — without using any data.
- On Waterbirds, DEXTER matches the state of the art.

### Bias Report Evaluation (FairFaces)

| Metric | w Bias | w/o Bias | Mean |
|--------|--------|----------|------|
| STS (similarity to data-based report) | 0.92 | 0.85 | 0.90 |
| G-eval Coherence | 4.58 | 4.80 | 4.19 |
| MOS-LLM | 4.29 | 4.80 | 4.48 |
| MOS-Human | 4.20 | 3.89 | 4.01 |

### Ablation Study

| Configuration | Spurious | Core | Average |
|--------------|----------|------|---------|
| Single word | 11.13 | 36.33 | 23.73 |
| Single word + $\mathcal{L}_\text{mask}$ | 34.00 | 53.86 | 43.93 |
| Multi-word | 15.53 | 8.13 | 11.83 |
| **Multi-word + $\mathcal{L}_\text{mask}$** | **63.00** | **87.86** | **75.43** |

## Highlights & Insights
1. **Completely data-free**: requires only the classifier itself, without access to any training data or labels.
2. **Multimodal global explanations**: dual-channel output via visual (activation maximization images) and textual (LLM bias report) modalities.
3. **Discrete prompt optimization**: Gumbel-Softmax combined with BERT→CLIP vocabulary mapping enables interpretable hard-token optimization.
4. **Validation across three tasks**: activation maximization, bias discovery, and bias explanation, each with quantitative evaluation.
5. **Pseudo-label mechanism**: addresses soft prompt gradient vanishing while establishing mappings between neurons and text tokens.

## Limitations & Future Work
1. **Computational cost**: prompt optimization requires approximately 10 minutes per class, making large-scale application (e.g., ImageNet with 1,000 classes) time-consuming.
2. **Dependence on Stable Diffusion**: generation quality is bounded by the diffusion model's capabilities and may degrade for domains not well covered by SD.
3. **NSFW risk**: additional safety filters are required.
4. **LLM hallucination**: VLM/LLM reasoning may introduce spurious explanations unrelated to actual model behavior.
5. **Classification only**: the framework has not been extended to other vision tasks such as detection or segmentation.

## Related Work & Insights
- **vs. DiffExplainer**: DEXTER replaces soft prompts with hard tokens, improving interpretability; user studies show superior performance on conceptual features.
- **vs. B2T**: B2T requires training data for misclassification analysis, whereas DEXTER is entirely data-free.
- **vs. LADDER**: LADDER relies on low-confidence predictions and LLM-generated pseudo-attributes, still requiring access to data.
- **vs. GradCAM / IG**: local attribution vs. global textual explanation — complementary but functionally distinct.

DEXTER represents a new paradigm of **active probing** of classifiers rather than passive analysis of data. It demonstrates that diffusion models can serve not only as generative tools but also as interpretability instruments. Furthermore, DEXTER can function as an automated bias auditing tool prior to model deployment.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — The data-free global explanation framework is a genuinely novel contribution; the three-pipeline design is elegant.
- **Technical Depth**: ⭐⭐⭐⭐☆ — Gumbel-Softmax, vocabulary mapping, and pseudo-label mechanisms are carefully engineered.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Three tasks × four datasets × user study × ablation study.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear logic, rich figures, and thorough appendix.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Diff-ICMH: Harmonizing Machine and Human Vision in Image Compression with Generative Prior](diff-icmh_harmonizing_machine_and_human_vision_in_image_compression_with_generat.md)
- [\[NeurIPS 2025\] Emergence and Evolution of Interpretable Concepts in Diffusion Models](emergence_and_evolution_of_interpretable_concepts_in_diffusi.md)
- [\[ICCV 2025\] Rethinking the Embodied Gap in Vision-and-Language Navigation: A Holistic Study of Physical and Visual Disparities](../../ICCV2025/image_generation/rethinking_the_embodied_gap_in_vision-and-language_navigation_a_holistic_study_o.md)
- [\[ICCV 2025\] Joint Diffusion Models in Continual Learning](../../ICCV2025/image_generation/joint_diffusion_models_in_continual_learning.md)
- [\[ICCV 2025\] LoRAverse: A Submodular Framework to Retrieve Diverse Adapters for Diffusion Models](../../ICCV2025/image_generation/loraverse_a_submodular_framework_to_retrieve_diverse_adapters_for_diffusion_mode.md)

</div>

<!-- RELATED:END -->
