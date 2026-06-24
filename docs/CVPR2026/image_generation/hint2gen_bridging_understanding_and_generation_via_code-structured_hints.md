---
title: >-
  [Paper Note] Hint2Gen: Bridging Understanding and Generation via Code-structured Hints
description: >-
  [CVPR 2026][Image Generation][Reasoning-aware Generation] Unified vision-language models fail to solve reasoning tasks like maze navigation and tangram puzzles, whereas VLMs/LLMs actually "know how to reason but do not know how to draw." This paper converts the reasoning results of VLMs/LLMs into SVG/HTML code and renders them as "hint images" overlaid on top of the original images. This serves as an executable bridge linking understanding and generation. It can either be fed…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Reasoning-aware Generation"
  - "Code-structured Hints"
  - "SVG/HTML"
  - "Unified Model"
  - "FLUX.1 Kontext"
date: 2026-05-08
content_hash: 5ee07c6d37ebafff
---

# Hint2Gen: Bridging Understanding and Generation via Code-structured Hints

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Tu_Hint2Gen_Bridging_Understanding_and_Generation_via_Code-structured_Hints_CVPR_2026_paper.html)  
**Code**: https://hint2gen.github.io  
**Area**: Image Generation / Multimodal Reasoning  
**Keywords**: Reasoning-aware Generation, Code-structured Hints, SVG/HTML, Unified Model, FLUX.1 Kontext

## TL;DR
Unified vision-language models fail to solve reasoning tasks like maze navigation and tangram puzzles, whereas VLMs/LLMs actually "know how to reason but do not know how to draw." This paper converts the reasoning results of VLMs/LLMs into SVG/HTML code and renders them as "hint images" overlaid on top of the original images. This serves as an executable bridge linking understanding and generation. It can either be fed into off-the-shelf models in a training-free (zero-shot) manner to improve performance, or used to fine-tune a specialized Hint2Gen model that natively processes such hints. Additionally, the authors introduce the Reason2Gen benchmark (3300 samples / 22 categories / 7 dimensions), outperforming open- and closed-source systems like GPT-Image and Nano Banana Pro across all dimensions.

## Background & Motivation
**Background**: Unified text-to-image (T2I) and image editing models, represented by FLUX, BAGEL, GPT-Image, and Nano Banana, are already powerful in terms of fidelity, generating high-fidelity images from natural language instructions.

**Limitations of Prior Work**: However, once a task requires explicit spatial or relational reasoning—such as maze navigation, tangram puzzles, playing Gobang, clock reading, or predicting ball bounce trajectories—these models systematically fail. They either rotate objects around incorrect pivot points or disrupt the surrounding background while removing objects, resulting in violations of basic logical, spatial, or rule-based constraints.

**Key Challenge**: The authors make a crucial observation—for the same reasoning tasks, **VLMs/LLMs can solve them accurately using text-only prompts** (possessing both reasoning capabilities and spatial localization skills). However, their outputs are inherently textual and **cannot be rendered as images**. This creates a clear dichotomy: VLMs/LLMs can reason but cannot draw, while generative models can draw but cannot reason. Therefore, the bottleneck is not a "lack of reasoning capability," but rather **the absence of a structured interface that translates symbolic reasoning into precise pixel outputs**.

**Goal**: To bridge this missing interface, enabling generative models to execute reasoning-intensive generation and editing in a controllable and interpretable manner.

**Key Insight**: Unstructured prompts, such as segmentation masks and bounding boxes, discard geometric relations and compositional logic. In contrast, **code representations** like SVG/HTML naturally preserve geometric relationships, layer hierarchies, and compositional logic, allowing abstract reasoning conclusions to be faithfully grounded in pixel space.

**Core Idea**: Use **code-structured visual hints** (SVG/HTML overlays) as a bridge—allowing VLMs/LLMs to write reasoning steps as structured programs, render and overlay them onto the image plane, and feed them into generative models as strong spatial priors.

## Method

### Overall Architecture
The core workflow of Hint2Gen is as follows: given the "input image + editing/generation instruction", a VLM/LLM first writes the reasoning conclusion of "how to modify/draw" as a snippet of SVG/HTML code (such as arrows, labels, polygons, or text). This code is then **rasterized and overlaid** onto the original image to produce a "hint image." This hint image acts as a bridge connecting understanding and generation, which can then be used in two ways: (1) **Zero-training**: directly feed the "original image + hint image" into existing powerful generative models (such as GPT-Image, Qwen-Image, etc.) to boost performance without fine-tuning; (2) **Training**: inject the latent variables of the hint image into the diffusion process on top of FLUX.1 Kontext to train Hint2Gen, which is specialized in processing such hints. To train this model, the authors also establish an automatic data pipeline to generate "instruction + target image + SVG/HTML hint" triplets in bulk, and build the Reason2Gen benchmark for evaluation.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Input Image + Editing/Generation Prompt"] --> B["Code-structured Visual Hints<br/>VLM/LLM writes SVG/HTML and renders overlay into a hint image"]
    G["Auto-prompt Construction & Multi-stage Filtering Pipeline<br/>diff+GPT-5+alignment+IoU/Gemini filtering"] -.Training Data.-> D
    B -->|Inference (Training-free)| C["Zero-training Hint Injection<br/>Original + hint fed directly into off-the-shelf MLLM"]
    B -->|Training| D["Hint2Gen Latent Space Hint Conditioning<br/>hint latent added to noisy latent"]
    C --> E["Correctly Reasoning Gen/Edit Results"]
    D --> E
```

### Key Designs

**1. Code-structured Visual Hints: Using SVG/HTML as an Executable Bridge between Understanding and Generation**

To address the dichotomy of "generative models cannot reason, VLMs/LLMs cannot draw", instead of forcing the training of a single model that "both reasons and draws", the authors introduce an intermediate representation. The VLM/LLM is tasked with writing spatial layouts, geometric constraints, and editing clues as SVG/HTML snippets (e.g., rotation arrows, deletion masks, connectors, coordinate alignments), which are then rasterized and overlaid on the original image. Why use code instead of masks/boxes? Because SVG/HTML uses primitives like `<polygon>`, `<polyline>`, `<rect>`, `<circle>`, and `<text>` that naturally retain geometric relations, layer hierarchies, and compositional logic. This precisely encodes mappings like $1 \to 4, 2 \to 3$ or dynamic structures like "marble chains moving along a curve", whereas masks/boxes only provide coarse regions. This layer of hints grounds high-level symbolic reasoning into geometrically precise and semantically clear pixel-level instructions, which is the source of all performance gains in this paper.

**2. Zero-training Hint Injection: Feeding Hints Directly into Off-the-shelf Models as Extra Input**

The pain point is that existing MLLMs have not been trained to "comprehend" such code-structured hints, leading to limited gains when directly fed. However, the authors find that for sufficiently strong off-the-shelf models (e.g., GPT-Image, Qwen-Image, Nano Banana Pro), **simply feeding the rendered SVG/HTML hint image alongside the input as an extra image allows them to solve previously failed tasks without any retraining**. Rows marked with `*` in Table 1 show this usage: for instance, the overall score of Nano Banana Pro increases from 17.96% to 22.77%, and GPT-Image from 11.33% to 15.75%. This indicates that the visual hint itself serves as a general interface—it explicitly depicts the reasoning results on the image, so the model only needs to "draw accordingly" without re-reasoning. This design is also direct evidence of the paper's claim regarding "cross-task, cross-model generalization of hints".

**3. Hint2Gen: Latent Space Hint Conditioning on FLUX.1 Kontext**

Since zero-training injection yields limited gains for weaker models, the authors train a specialized model. The base model chosen is FLUX.1 Kontext (a training-free framework for unified generation and editing): text instructions and image latents are extracted via a shared VAE, and the token streams are concatenated and passed through joint attention. The modification is minimal yet critical—the rasterized hint image is encoded using the **same VAE** to obtain the `hint latent`, which is then **directly added to the noisy input latent** as a conditioning signal at **each diffusion timestep**. This "latent addition injection" barely alters the base architecture but injects a powerful spatial prior into the denoising process, enabling the model to reliably perform reasoning-intensive edits such as "deleting only a specific object while preserving its surroundings" or "rotating around the correct pivot". Training uses LoRA adaptation, replaces the original CLIP encoder with Qwen2.5-VL-7B to jointly encode instructions and images, and employs dynamic resolution sampling (around $512 \times 512$, keeping the original aspect ratio). ⚠️ Note: Figure 2 in the paper mentions text encoded by T5, while the main text experimental setup mentions replacing CLIP with Qwen2.5-VL; as the encoder descriptions slightly diverge between the two, the main text is followed.

**4. Auto-prompt Construction & Multi-stage Filtering Pipeline: Solving the Training Bottleneck of Lacking Hinted Data**

The biggest obstacle in training Hint2Gen is the lack of existing datasets that simultaneously provide the "natural instruction + target image + SVG/HTML reasoning hint" triplet. The authors design an automated pipeline to scale up such annotations from three sources:
- **Natural Image Edit Pairs**: A fine coordinate grid is overlaid on the original image for spatial anchoring. Pixel-wise absolute differences between the "edited image" and the "original image" are calculated, followed by adaptive thresholding and morphological thinning to produce binary change masks. Dilation is used to merge adjacent regions into compact connected components. The "grid-overlaid original, edited image, difference mask, and instruction" are then fed together to GPT-5 to generate HTML hints that accurately render arrows/labels/boxes on the modified area.
- **Text-to-Image Samples**: GPT-5 receives "input prompts + target image + edge maps (structural layout)" and outputs HTML hints encoding spatial/logical reasoning.
- **Webpage Rendering Data**: To explicitly inject strong reasoning demands (which are often lacking in general editing), the authors design 10 logic-heavy scenarios (Gobang, Chinese chess, chess, Sudoku, Zuma, etc.) in a controlled HTML environment, allowing GPT-5 to deliver **pixel-aligned triplets** (problem image, hint image, answer image) through native browser rendering.

This is followed by a two-stage refinement & validation process: (1) **Zero-token post-alignment**: Densifying SVG/HTML vertices, snapping vertices to the nearest change pixels using difference masks, and correcting global offsets via centroid alignment. (2) **Quality Filtering**: Calculating the IoU between the rendered hint foreground mask and the ground-truth ROI, discarding low IoU (misaligned/irrelevant) samples, and passing the remaining samples to Gemini-2.5 Pro to evaluate semantic relevance, visual clarity, and reasoning fidelity. This yields about 100k natural image samples and 20k webpage-rendered reasoning samples.

## Loss & Training
Following the diffusion training objective of FLUX.1 Kontext, LoRA fine-tuning is performed. The only newly added conditional signal is the hint latent added to the noisy latent at each timestep. Complete hyperparameters are provided in the appendix of the original paper.

## Key Experimental Results

### Main Results
On the self-constructed Reason2Gen benchmark (3300 samples, 7 dimensions), GPT-5 is used as an LLM-as-judge ($0/1$ scoring, averaged over 3 runs). The accuracy (%) of each dimension is reported. Hint2Gen (Ours) achieves the highest scores in all 7 dimensions and the overall metric. `*` denotes the version using zero-training hint image injection.

| Model | Path | Assembly | Pattern | Strategy | Overall |
|------|------|----------|---------|----------|---------|
| FLUX.1 Kontext (Base, No Hint) | 0.40 | 7.67 | 10.33 | 6.23 | 3.85 |
| GPT-Image (Strongest Base Closed-source) | 13.33 | 31.00 | 17.50 | 10.48 | 11.33 |
| Nano Banana Pro (No Hint) | 15.47 | 40.33 | 34.67 | 22.22 | 17.96 |
| GPT-Image* (+Hint, Zero-training) | 18.80 | 44.67 | 24.33 | 15.44 | 15.75 |
| Nano Banana Pro* (+Hint, Zero-training) | 20.53 | 47.33 | 40.00 | 27.89 | 22.77 |
| **Hint2Gen (Ours)** | **29.64** | **55.82** | **48.93** | **37.84** | **31.04** |

Two key conclusions can be drawn: First, injecting the hint image in a zero-training manner into existing strong models generally brings a huge boost (e.g., Nano Banana Pro from 17.96% to 22.77%, GPT-Image from 11.33% to 15.75%). Second, even so, the specially trained Hint2Gen (31.04%) still significantly outperforms the best zero-training counterpart (22.77%). Note that the overall scores of all methods are far below 100%, indicating that the benchmark is indeed challenging and leaves substantial room for improvement. For the "Strategy" dimension, almost all models without hints score 0.00, while Hint2Gen reaches 37.84, which is one of the most prominent improvements.

### Ablation Study
Ablation of each pipeline component (Overall Accuracy):

| Configuration | Overall | Description |
|------|---------|------|
| w/o Hint | 6.93 | Trained with our data but excluding hint images; only slightly higher than base FLUX (3.85) |
| w/ Text & w/o Hint | 10.94 | Replaces the hint image with pure text reasoning generated by GPT-5 |
| w/o Preprocess | 22.50 | Removes preprocessing (grids/difference masks) |
| w/o Filtering | 25.94 | Removes IoU + Gemini filtering |
| w/o Postprocess | 28.63 | Removes post-alignment (vertex snapping/centroid correction) |
| Gemini-2.5 Pro as Hint Generator | 30.16 | Replaces with a weaker/different hint generator; performance remains close |
| **Ours (Full)** | **31.04** | Complete model |

Additionally, on the general text-to-image benchmark GenEval, Ours* (with hints injected) scores 0.91, tying with the strongest BAGEL* and outperforming GPT-Image* (0.89). It particularly excels in sub-metrics requiring spatial precision like Counting (0.93) and Position (0.91), verifying that the hint image, as a "general spatial grounding interface", also transfers to general generation tasks.

### Key Findings
- **Hint images are the primary cause of benefits, not the data itself**: Retraining without hint images on our dataset (w/o Hint, 6.93%) results in a severe drop compared to the full model (31.04%), proving that the improvement is indeed driven by the "hint-guided generation mechanism" rather than merely expanding the in-domain dataset.
- **Structured code formatting matters more than the generator's strength**: Replacing the hint generator with weaker VLMs (Gemini-2.5 Pro / Qwen2.5-VL) only marginally drops the overall score from 31.04% to 30.16% / 29.54%—confirming that the structured SVG/HTML format itself is the key.
- **Training data matters more than architecture**: Open-source models employ diverse architectures such as Diffusion Transformers, autoregressive models, and discrete diffusion, yet none of them showed a significant gap on Reason2Gen. What truly determines success is whether they have been trained on such logic-heavy, gamified domain data.
- **Filtering and alignment both yield positive contributions**: Performance drops when removing filtering (25.94%) or preprocessing (22.50%), while post-alignment (28.63%) provides consistent gains for hint accuracy.
- Human preference studies (Elo + Pearson correlation) demonstrate that GPT-5 judgments correlate highly with human evaluations, validating the reliability of the LLM-as-judge protocol.

## Highlights & Insights
- **Redefining the bottleneck**: Diagnosing "poor unified model reasoning" as "the lack of a structured output interface" rather than a "lack of reasoning capability" is a valuable reframing. This insights explains why VLMs can solve mazes but cannot draw the solutions.
- **Clever use of code as an intermediate representation**: SVG/HTML is an executable, renderable, and structurally rich "visual programming language" with innate geometric and hierarchical semantics. It is far more informative than masks or bounding boxes, and directly repurposes the code-generation capabilities of off-the-shelf LLMs.
- **Training-free plug-and-play improvement**: Using hint images as an additional input works directly out-of-the-box for any generative model that supports multi-image inputs, demanding almost zero migration cost.
- **Minimal architectural changes**: Simply adding the hint latent to the noisy latent at each diffusion step injects a robust spatial prior without modifying the FLUX architecture, which is a highly reusable conditioning trick.

## Limitations & Future Work
- **Heavy dependency on powerful external LLMs**: The entire pipeline—from hint generation (GPT-5) and filtering (Gemini-2.5 Pro) to evaluation (GPT-5)—relies heavily on closed-source LLMs, leading to high replication costs and bounding the hint quality to the LLMs' upper limit.
- **Low overall performance ceiling**: The top score of 31.04% is still far from fully "solving the tasks". Performance on dimensions such as Comparison (14.26%) and Commonsense (19.11%) remains weak.
- **Coarse evaluation ($0/1$ + LLM-as-judge)**: Multi-class binary scoring might mask subtle "near-correct" performance, and biases in the evaluator model cannot be entirely ruled out (e.g., the paper notes inflated scores when weaker VLMs act as evaluators).
- **Hint alignment still depends on post-processing**: The reliance on IoU filtering, vertex snapping, and centroid correction suggests that the raw spatial precision of code hints directly produced by LLMs is insufficient. End-to-end solutions remain an open avenue.

## Related Work & Insights
- **vs Unified Multimodal Generative Models (Chameleon / Show-o / Transfusion / BAGEL / MetaQuery)**: These models aim to execute both understanding and generation jointly in a uniform architecture, usually focusing on perceptual fidelity while failing to actually reason during generation. Hint2Gen keeps the architecture intact, inserting "code hints" as an explicit interface to map reasoning outcomes directly into the generation process. The methodology favors "decoupling" over "unification".
- **vs Reasoning-aware T2I Benchmarks (WISE / PhyBench / CommonsenseT2I / R2I-Bench / T2I-ReasonBench)**: These benchmarks evaluate world knowledge, physical plausibility, and commonsense logic, but none of them consider "code-structured representations as a reasoning scaffold." Reason2Gen fills this gap, with a task complexity (22 categories of gamified reasoning) that far exceeds previous benchmarks like $3 \times 3$ Tic-Tac-Toe or simple mazes.
- **vs Unstructured Conditioning (Masks / Boxes / Edges)**: While all these methods add spatial conditions to generative models, using SVG/HTML preserves geometric relations and compositional logic, yielding higher information density and programmability.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ "Code-structured hints acting as a bridge between understanding and generation" is a clean and explanatory new paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ Sweeps 20+ open- and closed-source models, 3 benchmarks, and includes a full ablation study, although the ceiling is low and is highly dependent on closed-source LLMs.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem diagnosis, strong motivational narrative, and cohesive figures/tables.
- Value: ⭐⭐⭐⭐⭐ Training-free, plug-and-play visual hints alongside the Reason2Gen benchmark provide tangible progress to the field of reasoning-aware generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Scone: Bridging Composition and Distinction in Subject-Driven Image Generation via Unified Understanding-Generation Modeling](scone_bridging_composition_and_distinction_in_subject-driven_image_generation_vi.md)
- [\[CVPR 2026\] A Style is Worth One Code: Unlocking Code-to-Style Image Generation with Discrete Style Space](a_style_is_worth_one_code_unlocking_code-to-style_image_generation_with_discrete.md)
- [\[CVPR 2026\] LumiX: Structured and Coherent Text-to-Intrinsic Generation](lumix_structured_and_coherent_text-to-intrinsic_generation.md)
- [\[CVPR 2026\] Evaluating Generative Models via One-Dimensional Code Distributions](evaluating_generative_models_via_one-dimensional_code_distributions.md)
- [\[CVPR 2026\] Enhancing Spatial Understanding in Image Generation via Reward Modeling](enhancing_spatial_understanding_in_image_generation_via_reward_modeling.md)

</div>

<!-- RELATED:END -->
