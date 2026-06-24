---
title: >-
  [Paper Note] StarVector: Generating Scalable Vector Graphics Code from Images and Text
description: >-
  [CVPR 2025][Multimodal VLM][SVG generation] StarVector is proposed, a multimodal large language model-based SVG generation framework that reformulates image vectorization as an inverse rendering and code generation task. By leveraging visual semantic understanding, it directly generates compact SVG code comprising rich primitive types (circles, polygons, text, etc.), establishing a new state-of-the-art (SOTA) across 10 datasets on 3 tasks.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "SVG generation"
  - "image vectorization"
  - "multimodal large language models"
  - "code generation"
  - "vector graphics"
date: 2026-05-08
content_hash: 06f06e34b64b65d6
---

# StarVector: Generating Scalable Vector Graphics Code from Images and Text

**Conference**: CVPR 2025  
**arXiv**: [2312.11556](https://arxiv.org/abs/2312.11556)  
**Code**: [https://github.com/joanrod/star-vector](https://github.com/joanrod/star-vector)  
**Area**: Multimodal VLM  
**Keywords**: SVG generation, image vectorization, multimodal large language models, code generation, vector graphics

## TL;DR
StarVector is proposed, a multimodal large language model-based SVG generation framework that reformulates image vectorization as an inverse rendering and code generation task. By leveraging visual semantic understanding, it directly generates compact SVG code comprising rich primitive types (circles, polygons, text, etc.), establishing a new state-of-the-art (SOTA) across 10 datasets on 3 tasks.

## Background & Motivation

**Background**: Vector graphics (SVG) represent a standard format for modern image rendering, offering advantages such as infinite scalability and editability. Image vectorization (converting raster images to SVG) is a fundamental task in computer graphics.

**Limitations of Prior Work**: Traditional methods (Potrace, VTracer, AutoTrace) trace curves via pixel-level analysis, producing overly complex path representations and lacking semantic understanding. Deep learning methods (DeepSVG, Im2Vec), although introducing latent variable models and differentiable rendering, are restricted to path primitives and cannot exploit the rich set of SVG native primitives (circle, polygon, text, etc.). For instance, a circle that could be represented by a single `<circle>` tag requires dozens of path curves to approximate in VTracer.

**Key Challenge**: Existing methods either suffer from poor generalization (deep learning methods) or generate redundant and complex outputs (traditional methods), and none can utilize SVG's native shape primitives for semantic-level compact representations.

**Goal**: How to enable a model to understand image semantics during vectorization and automatically select the optimal combination of SVG primitives?

**Key Insight**: Treat image vectorization as an "inverse rendering + code generation" task, leveraging the visual understanding and code generation capabilities of MLLMs to operate directly in the SVG code space.

**Core Idea**: Unify image understanding and SVG code generation using multimodal large language models. By learning token sequences of SVG code, the model naturally acquires primitive-aware vectorization capabilities.

## Method

### Overall Architecture
StarVector consists of three components: (1) an image encoder (CLIP ViT) that encodes the input image into visual tokens; (2) an adapter (nonlinear projection) that maps visual features to the embedding space of the language model; and (3) a code language model (StarCoder) that receives visual or text tokens and auto-regressively generates SVG code. During training, the input sequence is $(x_v, x_s)$ (Image-to-SVG) or $(x_t, x_s)$ (Text-to-SVG), optimized using the standard next-token prediction objective.

### Key Designs

1. **Visual Token Computation and Adapter**:
    - **Function**: Maps input images to a sequence of visual tokens understandable by the language model.
    - **Mechanism**: CLIP ViT is used to extract all last-layer features (not just the CLS token), which are projected to the LLM dimension via a non-linear adapter: $h_v = g_\varphi(z_v) = \text{LayerNorm}(W_L \cdot \text{Swish}(W_h \cdot z_v))$. StarVector-1B uses ViT-B/32 to generate 257 visual tokens, while StarVector-8B uses SigLip to generate 576 visual tokens.
    - **Design Motivation**: Utilizing all features (rather than just CLS) is crucial because SVG generation requires high visual representational capacity—details at each spatial position may correspond to an SVG primitive. The non-linear projection using Swish + LayerNorm bridges the distributional gap between visual and code modalities better than a simple linear projection.

2. **StarCoder-based SVG Code Generation**:
    - **Function**: Maps visual or textual condition inputs to structured SVG code sequences.
    - **Mechanism**: It models the conditional probability $p(x_s | x_c) = \prod_{i=1}^L p(x_{s,i} | x_{s,<i}, x_c)$, where $x_c$ is the conditional input (image or text). Operating directly in the SVG code space allows the model to naturally learn to use primitives like `<circle>`, `<polygon>`, `<rect>`, and `<text>` instead of being limited to `<path>`.
    - **Design Motivation**: StarCoder is chosen as the backbone due to its extensive pre-training on code generation tasks. Since SVG is essentially an XML-based markup language, a code-based language model is naturally suited to learn its structure and syntax. Autoregressive generation natively supports variable-length outputs to accommodate varying SVG complexities.

3. **SVG-Stack Large-Scale Dataset**:
    - **Function**: Provides a diverse training dataset of 2.1 million SVG samples.
    - **Mechanism**: SVG codes are extracted from The Stack code dataset, followed by deduplication, rendering validation (excluding blank white images using CairoSVG), and cleaning (removing comments and XML headers). Each sample includes the SVG code, the rendered raster image, and a textual description (synthesized using BLIP2 and LLaVA for 4 million captions). SVG-specific data augmentations are introduced: resolution scaling, rotation, translation, shearing, and color modifications.
    - **Design Motivation**: Prior datasets only cover narrow domains such as fonts, icons, or emojis, failing to generalize to complex SVGs (e.g., web graphics, technical diagrams). SVG-Stack is the first large-scale SVG pre-training dataset, featuring real-world SVGs from GitHub that cover diverse syntactic structures and primitive types.

### Loss & Training
- **Training Loss**: Standard next-token cross-entropy, computed only on the SVG code portion.
- **Training Scale**: StarVector-1B is trained on 8×A100 GPUs for 7 days (batch size = 128), and StarVector-8B is trained on 64×H100 GPUs for 10 days (batch size = 512). Both are trained for 2 epochs.
- **Inference Strategy**: Generates $k=5$ candidate samples (temperature 0-1) and selects the best one using DinoScore. A logit bias of 10 is applied to the `<svg-end>` token to encourage generating valid, closed SVG code. Top-p nucleus sampling ($p=0.9$) and a length penalty of $-0.5$ are employed.

## Key Experimental Results

### Main Results (Image-to-SVG, DinoScore↑ / Tokens)

| Dataset | Metric | StarVector-8B | LIVE | VTracer | AutoTrace |
|--------|------|--------------|------|---------|-----------|
| SVG-Stack | DinoScore | **0.966** | 0.934 | 0.954 | 0.942 |
| SVG-Stack | Tokens | 5.3k | 18.3k | 9.7k | 59.1k |
| SVG-Fonts | DinoScore | **0.982** | 0.956 | 0.964 | 0.954 |
| SVG-Icons | DinoScore | **0.984** | 0.959 | 0.940 | 0.946 |
| SVG-Diagrams | DinoScore | **0.959** | 0.870 | 0.882 | 0.874 |

### Text-to-SVG Experiment

| Method | SVG-FIGR FID↓ | SVG-FIGR CLIP↑ | SVG-Stack FID↓ | SVG-Stack CLIP↑ |
|------|-------------|---------------|-------------|----------------|
| StarVector-8B | **10.07** | **27.37** | **25.83** | **31.31** |
| StarVector-1B | 15.26 | 26.34 | 28.37 | 29.37 |
| GPT-4 | 32.95 | 26.09 | 37.38 | 26.23 |
| IconShop | - | 25.75 | - | - |

### Ablation Study

| Configuration | DinoScore | Description |
|------|----------|------|
| Full StarVector-8B | 0.963 | Baseline (average across datasets) |
| w/o Data Augmentation | ~0.94 | Augmentation significantly improves robustness |
| StarVector-1B | 0.952 | Smaller model + lower resolution leads to accuracy degradation |
| 5 paths (LIVE) | 0.898 | Too few paths fail to capture details |
| 60 paths (LIVE) | 0.939 | Increasing paths improves accuracy but tokens bloat to 18k |

### Key Findings
- **MSE is unsuitable for evaluating SVG quality**: While StarVector is strongly preferred in human evaluations, it earns lower MSE scores than LIVE. This is because MSE is highly sensitive to pixel-level micro-shifts, whereas semantic fidelity in SVGs is more critical. DinoScore correlates strongly with human judgment (Spearman = 0.76).
- **Primitive usage is a key advantage**: StarVector uses ~3k tokens on average (close to the ground truth), whereas VTracer uses 4.5k-20k, LIVE uses a fixed 18.3k, and AutoTrace bloats up to 59k-94k.
- **StarVector is the only method capable of diagram generation**: On SVG-Diagrams, only StarVector successfully leverages primitive types such as `<rect>`, arrows, and `<text>`, while other methods merely approximate them using path curves.
- **Human evaluation consistently prefers StarVector**: Over 1,948 evaluations from 30 annotators, StarVector-8B was significantly preferred across all settings.

## Highlights & Insights
- **Redefining image vectorization as a code generation task** is a highly elegant framework shift—making the selection and utilization of SVG primitives an emergent capability learned naturally by the model, rather than relying on explicitly designed heuristics.
- **Proposing DinoScore to replace MSE** for SVG quality evaluation addresses a long-standing evaluation bias in the field.
- **Model scaling effects are prominent**: Scaling from 1B to 8B yields consistent performance gains. Under higher resolution (384) and longer context window (16k), StarVector successfully parses highly complex SVGs.

## Limitations & Future Work
- The 16k token context length limit restricts the generation of excessively complex SVGs (e.g., massive technical schematics may exceed this threshold).
- Relying solely on autoregressive code prediction lacks visual feedback—it lacks a closed-loop rendering-and-comparison optimization step during generation.
- Inference speeds are constrained by the autoregressive LLM: StarVector-8B takes 74 seconds per sample, which is significantly slower than VTracer (0.09s).
- The semantic accuracy of Text-to-SVG is still suboptimal, constrained by the quality of the synthetic captions in the training data.

## Related Work & Insights
- **vs. LIVE / DiffVG**: Iterative optimization methods based on differentiable rendering achieve high pixel-level accuracy but are extremely slow (LIVE with 60 paths takes 1412 seconds per sample), only utilize path primitives, and lack semantic understanding.
- **vs. VTracer / Potrace**: Traditional image processing methodologies are fast but generate excessive paths, resulting in visual artifacts.
- **vs. GPT-4V**: Native multimodal LLMs perform poorly when directly generating SVG (FID of 32.95 vs. StarVector's 10.07), but they demonstrate the feasibility of the MLLM pathway. StarVector bridges this gap through domain-specific training on specialized data.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First to formalize image vectorization as an inverse rendering and code generation task on MLLMs; primitive-aware SVG generation is a paradigm-shifting capability.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated across 10 datasets on 3 tasks, including human studies, clear ablation analyses, and the introduction of a comprehensive benchmark (SVG-Bench).
- **Writing Quality**: ⭐⭐⭐⭐ The framework is clearly presented, and the arguments regarding the limitations of MSE are highly convincing, although some tables are densely populated.
- **Value**: ⭐⭐⭐⭐⭐ Massive contributions to the fundamental infrastructure of the SVG generation field through the SVG-Stack dataset, SVG-Bench, and DinoScore metrics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] CodePercept: Code-Grounded Visual STEM Perception for MLLMs](codepercept_code-grounded_visual_stem_perception_for_mllms.md)
- [\[NeurIPS 2025\] Table2LaTeX-RL: High-Fidelity LaTeX Code Generation from Table Images via Reinforced Multimodal Language Models](../../NeurIPS2025/multimodal_vlm/table2latex-rl_high-fidelity_latex_code_generation_from_table_images_via_reinfor.md)
- [\[CVPR 2026\] Text-Printed Image: Bridging the Image-Text Modality Gap by "Printing" Text into Images](../../CVPR2026/multimodal_vlm/text-printed_image_bridging_the_image-text_modality_gap_for_text-centric_trainin.md)
- [\[CVPR 2025\] Scalable Video-to-Dataset Generation for Cross-Platform Mobile Agents](scalable_video-to-dataset_generation_for_cross-platform_mobile_agents.md)
- [\[ACL 2025\] Scaling Text-Rich Image Understanding via Code-Guided Synthetic Multimodal Data Generation](../../ACL2025/multimodal_vlm/code_guided_text_rich_image.md)

</div>

<!-- RELATED:END -->
