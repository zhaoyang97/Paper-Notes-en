---
title: >-
  [Paper Note] LLM4SVG: Empowering LLMs to Understand and Generate Complex Vector Graphics
description: >-
  [CVPR 2025][LLM Safety][SVG generation] This paper proposes the LLM4SVG framework, which enables open-source LLMs (such as GPT-2, Phi-2, and Falcon) to understand and generate high-quality, complex vector graphics. This is achieved by defining 55 learnable SVG semantic tokens to replace raw XML tags and conducting a two-stage instruction fine-tuning process on the SVGX-SFT dataset, which contains 250K high-quality SVGs and 580K instruction-following pairs. The GPT-2 XL-based…
tags:
  - "CVPR 2025"
  - "LLM Safety"
  - "SVG generation"
  - "semantic token"
  - "LLM fine-tuning"
  - "instruction following"
  - "vector graphics understanding"
date: 2026-05-08
content_hash: 0ebcd483c8a3b535
---

# LLM4SVG: Empowering LLMs to Understand and Generate Complex Vector Graphics

**Conference**: CVPR 2025  
**arXiv**: [2412.11102](https://arxiv.org/abs/2412.11102)  
**Code**: [https://ximinng.github.io/LLM4SVGProject/](https://ximinng.github.io/LLM4SVGProject/)  
**Area**: LLM / NLP / Vector Graphics Generation  
**Keywords**: SVG generation, semantic token, LLM fine-tuning, instruction following, vector graphics understanding

## TL;DR
This paper proposes the LLM4SVG framework, which enables open-source LLMs (such as GPT-2, Phi-2, and Falcon) to understand and generate high-quality, complex vector graphics. This is achieved by defining 55 learnable SVG semantic tokens to replace raw XML tags and conducting a two-stage instruction fine-tuning process on the SVGX-SFT dataset, which contains 250K high-quality SVGs and 580K instruction-following pairs. The GPT-2 XL-based model achieves an FID of 64.11 and a CLIPScore of 0.3496, significantly outperforming GPT-4o (127.78 FID) and all existing SVG generation methods.

## Background & Motivation

**Background**: As a standard vector graphic format, SVG offers advantages such as resolution independence, editability, and high compression ratio, and is widely used in UI design, logo generation, etc. Existing SVG generation methods fall into two categories: (1) Optimization-based methods (e.g., CLIPDraw, VectorFusion, SVGDreamer) iteratively optimize Bezier curve parameters via a differentiable rasterizer, which suffers from slow generation speeds (tens of minutes per image) and yields uneditable results; (2) Neural network-based methods (e.g., SVG-VAE, DeepSVG, Iconshop) are constrained by small-scale vector datasets, thus only handling simple icons or character fonts.

**Limitations of Prior Work**: LLMs (such as GPT-4, Claude) have encountered web-based SVG code during pretraining, possessing basic XML understanding capabilities. However, directly generating SVGs poses two major issues: (1) SVG tags and attributes are tokenized as general text, resulting in semantic ambiguity (e.g., the word 'path' has completely different meanings in natural language versus SVG); (2) LLM training lacks modeling of the rendering order of vector paths, leading to chaotic overlapping between output primitives.

**Key Challenge**: While LLMs possess powerful sequence generation and instruction-following capabilities, the structured semantics of SVGs (tags, attributes, path commands) cannot be properly captured by text tokenizers.

**Goal**: To enable any LLM to understand and generate high-quality complex SVGs.

**Key Insight**: Rather than designing a completely new architecture, this work introduces a modular SVG semantic encoding layer on top of existing LLMs, using learnable semantic tokens to precisely encode every component and attribute of an SVG.

**Core Idea**: This method defines 55 specialized SVG semantic tokens (15 tag types + 30 attribute types + 10 path command types) to replace raw text tags in SVG source code. After expanding the LLM vocabulary, instruction fine-tuning is performed to achieve accurate SVG understanding and generation.

## Method

### Overall Architecture
LLM4SVG adopts a modular architecture consisting of: (1) an SVG semantic token layer that converts SVG code into structured representations; (2) an optional vision encoder to process rendered images; (3) an LLM backbone (GPT-2/Phi-2/Falcon/LLaVA) to process the interleaved sequence; (4) a decoder to output SVG code or textual descriptions. The framework supports two categories of tasks: text-to-SVG generation (templates #1 and #2) and SVG-to-text understanding (templates #3, #4, and #5).

### Key Designs

1. **SVG Semantic Tokens (55 learnable tokens)**:

    - **Function**: To decouple the tags, attributes, and path commands of raw SVG source code from general text semantics, serving as independent, learnable vocabulary items.
    - **Mechanism**: Fifty-five new tokens are defined and categorized into three groups: 15 tag tokens (e.g., `<path>`, `<rect>`, `<circle>`), 30 attribute tokens (e.g., `fill`, `stroke`, `d`), and 10 path command tokens (e.g., `MoveTo`, `LineTo`, `CubicBezier`). These tokens are appended to the LLM vocabulary $|\mathcal{V}|' = |\mathcal{V}| + 55$, and initialized to the semantic mean of their descriptive text embeddings: $E(s) = \frac{1}{n}\sum_{j=1}^n \mathbf{W}_{emb}^\top \cdot w_j$
    - **Design Motivation**: In raw SVG code, `<path>` is decomposed by the tokenizer into three subwords `<`, `path`, `>`, losing the semantic context of 'this is a vector path tag'. The specialized tokens ensure precise capture of the SVG structure, and the initialization policy leverages the semantics of descriptive text to provide a solid starting point.

2. **SVGX-SFT Dataset (250K SVG + 580K Instruction Data)**:

    - **Function**: To provide large-scale, high-quality SVG training data.
    - **Mechanism**: (1) Hand-collect 250K colorful, complex vector graphics; (2) Design a lossless preprocessing pipeline to filter out redundant elements (approximately half of SVG content is temporary editor data or sub-optimal structures), significantly compressing file sizes; (3) Rasterize SVGs to 512×512 images, then generate captions using BLIP and detailed instruction descriptions using GPT-4; (4) Accumulate a total of 580K instruction-following data points covering 5 templates (2 for generation and 3 for understanding).
    - **Design Motivation**: Annotating vector graphics is extremely expensive, heavily limiting prior studies to simple doodles, fonts, or icons. The automated data pipeline achieves the first large-scale SVG-Text-Image tri-modal dataset.

3. **Two-Stage Training Strategy**:

    - **Function**: To progressively align the SVG semantic space with the LLM text space.
    - **Mechanism**: Stage 1 (Feature Alignment Pre-training) — Freeze the LLM and the vision encoder, training only the embedding layer $\mathbf{W}_{emb}$ so that the 55 new tokens learn the correct semantics; Stage 2 (End-to-End Fine-Tuning) — Employ LoRA/QLoRA or full-parameter tuning to optimize $\theta = \{\mathbf{W}_{emb}, \phi\}$, fine-tuning on the complete 580K instruction dataset for 1-3 epochs.
    - **Design Motivation**: Direct end-to-end training can lead to unstable embedding updates for the new tokens. The two-stage strategy stabilizes the semantics of SVG tokens first, followed by global joint optimization.

### Loss & Training
Standard autoregressive cross-entropy loss: $p(\mathbf{X}_a | \mathbf{X}_v, \mathbf{X}_{inst}) = \prod_{i=1}^L p_\theta(x_i | \mathbf{X}_v, \mathbf{X}_{inst}, \hat{x}_{i-1})$. The maximum token length is restricted to 4,096, and excessively long SVGs are directly truncated. The framework is built on LlamaFactory, integrating Unsloth to support quantized training, executed on an 8× A800 GPU setup.

## Key Experimental Results

### Main Results: Comparison with SVG Generation Methods

| Method | Type | FID ↓ | CLIPScore ↑ | Aesthetic ↑ | HPS ↑ | Generation Time ↓ |
|---|---|---|---|---|---|---|
| CLIPDraw | Optimization | 132.75 | 0.2486 | 3.98 | 0.2347 | 5m 20s |
| VectorFusion | Optimization | 87.73 | 0.2720 | 4.98 | 0.2450 | 11m 27s |
| SVGDreamer | Optimization | 72.68 | 0.3001 | 5.54 | 0.2685 | 43m 56s |
| DeepSVG | Network | 71.37 | 0.2118 | 3.00 | 0.1090 | 2m 3s |
| StrokeNUWA | Network | 92.31 | 0.3001 | 5.54 | 0.1659 | 20s |
| **LLM4SVG (GPT-2 XL)** | **LLM** | **64.11** | **0.3496** | **5.98** | **0.2485** | **18s** |
| LLM4SVG (Phi-2) | LLM | 65.98 | 0.3373 | 5.91 | 0.2289 | 20s |
| LLM4SVG (LLaVA) | LLM | 66.72 | 0.3296 | 5.68 | 0.2177 | 25s |

### Comparison with Commercial LLMs

| Model | FID ↓ | CLIPScore ↑ | Aesthetic ↑ | HPS ↑ |
|---|---|---|---|---|
| GPT-4o | 127.78 | 0.2949 | 5.03 | 0.1788 |
| Claude-3.5 | 82.89 | 0.3083 | 5.24 | 0.1912 |
| Llama-3.1 70B | 138.44 | 0.2735 | 4.30 | 0.1665 |
| Qwen2.5 70B | 131.46 | 0.2803 | 4.50 | 0.1691 |
| **LLM4SVG (GPT-2 XL, 1.5B)** | **64.11** | **0.3496** | **5.98** | **0.2485** |

### Ablation Study

| Configuration | FID ↓ | CLIPScore ↑ |
|---|---|---|
| Full LLM4SVG (GPT-2 XL) | 64.11 | 0.3496 |
| w/o SVG semantic tokens | 89.42 | 0.2913 |
| w/o Stage 1 pretraining | 78.65 | 0.3102 |
| w/o SVGX-SFT (Small Dataset) | 95.23 | 0.2756 |
| GPT-2 small (124M) | 78.10 | 0.3129 |
| GPT-2 large (774M) | 66.09 | 0.3205 |

### Key Findings
- **LLM4SVG (GPT-2 XL, 1.5B) comprehensively outperforms GPT-4o (trillion-parameter scale)**: FID 64.11 vs 127.78, CLIPScore 0.3496 vs 0.2949, Aesthetic 5.98 vs 5.03. This indicates that task-specific fine-tuning coupled with semantic tokens far exceeds the zero-shot capabilities of general-purpose LLMs.
- **Substantial generation speed-up**: 18 seconds compared to 11–44 minutes for optimization-based methods, speeding up the process by 30–150×.
- **SVG semantic tokens represent the most significant driver of performance**: removing them increases FID from 64.11 to 89.42 (+25.31), verifying that accurate SVG encoding is core to the success of this method.
- **Crucial role of data scale**: shrinking down the SVGX-SFT dataset degrades the FID to 95.23, demonstrating that LLMs rely heavily on large-scale alignment data.
- **Evident scaling law**: as model capacity increases from GPT-2 small to large and XL, FID steadily decreases from 78.10 to 66.09 and 64.11.

## Highlights & Insights
- **High framework generality**: Not bound to a specific LLM architecture; it is compatible with GPT-2, Phi-2, Falcon, and LLaVA. The 55 semantic tokens and instruction dataset can be seamlessly adapted to any newly introduced LLMs.
- **Decoupled instructions and parameters**: SVG tags, attributes, and commands are parsed into separate tokens, relieving the LLM from having to "guess" SVG semantics from string expressions, which drastically mitigates hallucinations.
- **Scalable data pipeline**: The automated BLIP + GPT-4 labeling pipeline allows continuous scale-up of dataset size, offering a highly sustainable paradigm.
- **A new benchmark for SVG quality**: For the first time, it comprehensively outperforms optimization-based methods in quantitative metrics, while reducing generation time by two orders of magnitude.

## Limitations & Future Work
- The maximum token sequence length is restricted to 4,096, causing extremely long SVGs (such as complex maps or detailed illustrations) to be truncated.
- The method only supports a subset of SVG elements (paths, basic shape elements), lacks compatibility with advanced features such as `<text>`, `<filter>`, and `<gradient>`.
- The MLP-style token initialization depends heavily on description text quality, which might yield imprecise embeddings for edge-case SVG features.
- Lacks fine-grained evaluation from professional human designers, as standard metrics like FID and CLIPScore do not fully capture actual design quality.

## Related Work & Insights
- **vs SVGDreamer**: The optimization-based SOTA achieves 72.68 FID but requires 44 minutes for generation, whereas LLM4SVG scores 64.11 FID in just 18 seconds.
- **vs StrokeNUWA**: The first SVG generation method utilizing token sequences reports an FID of 92.31 and lacks semantic understanding capabilities. In contrast, the semantic tokens in LLM4SVG offer more precise representations.
- **vs Claude-3.5**: The top performer among proprietary commercial LLMs displays an FID of 82.89 and a CLIPScore of 0.3083. LLM4SVG surpasses this with only 1.5B parameters.

## Rating
- Novelty: ⭐⭐⭐⭐ The semantics token design is simple yet effective, and the data engineering effort contributes significantly.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Conducts comprehensive comparisons with 12 SVG methods and 9 LLMs.
- Writing Quality: ⭐⭐⭐⭐ Abundant in figures and charts, though the text tends to be verbose.
- Value: ⭐⭐⭐⭐⭐ Extremely high community value thanks to the open-sourced dataset and framework.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] RadarLLM: Empowering Large Language Models to Understand Human Motion from Millimeter-Wave Point Cloud Sequence](../../AAAI2026/llm_safety/radarllm_empowering_large_language_models_to_understand_human_motion_from_millim.md)
- [\[ACL 2026\] LLM-VA: Resolving the Jailbreak-Overrefusal Trade-off via Vector Alignment](../../ACL2026/llm_safety/llm-va_resolving_the_jailbreak-overrefusal_trade-off_via_vector_alignment.md)
- [\[CVPR 2025\] LoTUS: Large-Scale Machine Unlearning with a Taste of Uncertainty](lotus_large-scale_machine_unlearning_with_a_taste_of_uncertainty.md)
- [\[CVPR 2025\] Neural Gate: Mitigating Privacy Risks in LVLMs via Neuron-Level Gradient Gating](neural_gate_mitigating_privacy_risks_in_lvlms_via_neuron-level_gradient_gating.md)
- [\[CVPR 2025\] Low-Rank Adaptation in Multilinear Operator Networks for Security-Preserving Incremental Learning](low-rank_adaptation_in_multilinear_operator_networks_for_security-preserving_inc.md)

</div>

<!-- RELATED:END -->
