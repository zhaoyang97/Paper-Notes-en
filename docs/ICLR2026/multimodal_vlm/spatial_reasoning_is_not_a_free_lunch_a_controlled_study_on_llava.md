---
title: >-
  [Paper Note] Spatial Reasoning is Not a Free Lunch: A Controlled Study on LLaVA
description: >-
  [ICLR 2026][Multimodal VLM][spatial reasoning] Through controlled experiments within the LLaVA framework, this paper systematically investigates the effects of image encoder training objectives and 2D positional encoding on the spatial reasoning capabilities of VLMs. The study finds that encoder choice dominates spatial performance, AIMv2 yields the most consistent results, while improvements from 2D-RoPE are unstable—indicating that spatial reasoning failures are rooted in core design choices of current VLM pipelines.
tags:
  - ICLR 2026
  - Multimodal VLM
  - spatial reasoning
  - image encoder
  - 2D-RoPE
  - LLaVA
  - vision-language model
date: 2026-05-08
content_hash: b6e2527cffbe5a3c
---

# Spatial Reasoning is Not a Free Lunch: A Controlled Study on LLaVA

**Conference**: ICLR 2026
**arXiv**: [2603.12545](https://arxiv.org/abs/2603.12545)
**Code**: None
**Area**: Multimodal VLM
**Keywords**: spatial reasoning, image encoder, 2D-RoPE, LLaVA, vision-language model

## TL;DR
Through controlled experiments within the LLaVA framework, this paper systematically investigates the effects of image encoder training objectives and 2D positional encoding on the spatial reasoning capabilities of VLMs. The study finds that encoder choice dominates spatial performance, AIMv2 yields the most consistent results, while improvements from 2D-RoPE are unstable—indicating that spatial reasoning failures are rooted in core design choices of current VLM pipelines.

## Background & Motivation

**Background**: Current VLMs (e.g., LLaVA, BLIP-2, Flamingo) almost universally rely on large-scale pretrained image encoders such as CLIP/SigLIP. Trained via global image-text alignment, these encoders are integrated into various multimodal systems and have driven significant progress on image captioning and VQA tasks.

**Limitations of Prior Work**: Despite strong performance on general benchmarks, modern VLMs remain brittle on basic 2D spatial reasoning—frequently failing at tasks such as understanding relative positions, layout relationships, and counting. This failure is not merely a data issue; it is more fundamentally tied to two core design choices in current VLM pipelines: (1) reliance on CLIP-style encoders, and (2) flattening images into 1D token sequences with 1D positional encoding.

**Key Challenge**: CLIP-style encoders are optimized for global semantic alignment rather than structured spatial representations, focusing on *what* is in the image rather than *where* things are. Furthermore, the multimodal fusion stage forces 2D images into 1D sequences, discarding spatial structure along height and width dimensions. However, prior studies typically conflate data, model scale, and architecture, making it difficult to isolate which design factor causes spatial reasoning failures.

**Goal**: (1) How much does the image encoder training objective affect spatial reasoning? (2) Can 2D positional encoding mitigate the loss of spatial information? (3) To what extent do these two factors explain observed spatial reasoning failures?

**Key Insight**: Under the LLaVA-1.5 (7B) framework, the language model backbone and training procedure are held fixed while systematically replacing the image encoder (CLIP, SigLIP, SigLIP2, AIMv2) and positional encoding scheme (1D-RoPE vs. 2D-RoPE), constructing rigorously controlled comparative experiments.

**Core Idea**: With all other variables controlled, this work isolates the causal effects of image encoder training objectives and positional encoding structure on spatial reasoning in VLMs.

## Method

### Overall Architecture
All experiments are strictly conducted within the LLaVA-1.5 framework using Vicuna-7B as the language model backbone. Input images are uniformly resized to $256 \times 256$; patch features extracted by the image encoder are projected into the language model's token space via a linear projection layer. Training consists of two stages: a pretraining stage that updates only the projection layer (to align visual and language spaces), and an instruction-tuning stage that updates all parameters (including the language model). The only differences across variants are the encoder type and whether 2D-RoPE is applied.

The core principle of the experimental design is strict variable control: all variants use identical training data (the original LLaVA dataset), identical hyperparameters, identical optimizer configurations, and identical training duration, ensuring that performance differences can only be attributed to changes in the encoder and positional encoding. Seven frontier models in the 2–8B parameter range are also evaluated as upper-bound references.

### Key Designs

1. **Encoder Replacement Ablation**:

    - *Function*: Isolates the effect of encoder training objectives on spatial reasoning.
    - *Mechanism*: Compares four encoders—CLIP (contrastive learning, global image-text alignment), SigLIP (a CLIP variant with sigmoid loss), SigLIP2 (an improved SigLIP), and AIMv2 (an encoder trained with denser or generative objectives). All encoders are plugged directly into the LLaVA framework using identical training data and procedures.
    - *Design Motivation*: Different training objectives yield different visual features—global alignment objectives (CLIP/SigLIP) may suppress local spatial information, whereas dense/generative objectives (AIMv2) may preserve more spatial detail.

2. **2D Rotary Positional Encoding (2D-RoPE)**:

    - *Function*: Explicitly preserves 2D spatial structure in the multimodal attention layers.
    - *Mechanism*: Standard 1D-RoPE encodes only the sequential position of tokens, treating all patches from a 2D image as a single row. 2D-RoPE jointly encodes the horizontal and vertical indices of each patch, applied to the query and key projections in multimodal attention. This allows the model to distinguish patches at different rows and columns within the same image during image-text fusion.
    - *Design Motivation*: Flattening images into 1D sequences discards 2D structural information, which is another potential cause of spatial reasoning failures. Qwen2-VL has introduced similar multimodal rotary embeddings, but systematic evidence isolating the effect of positional encoding has been lacking.

3. **Multi-Benchmark Spatial Reasoning Evaluation**:

    - *Function*: Comprehensively evaluates spatial reasoning ability from multiple perspectives.
    - *Mechanism*: Seven spatial reasoning benchmarks are used—MMVP (visual perception), CV-Bench 2D Overall (2D spatial understanding), TallyQA (counting), GQA Overall (scene graph reasoning), VSR (visual spatial relations), TopViewRS (top-view reasoning), and CountBenchQA (counting QA). Frontier models (LLaVA-NeXT, Qwen2.5-VL, and five others) are evaluated as upper-bound references.
    - *Design Motivation*: Spatial reasoning is a multidimensional capability; a single benchmark may miss differences in certain aspects.

### Loss & Training
Standard LLaVA two-stage training is employed: Stage 1 pretrains the projection layer on 558K image-text pairs (encoder and language model are frozen; only the projection layer is trained); Stage 2 performs full-parameter instruction tuning on 665K instruction-following samples. All variants share identical hyperparameters (learning rate, batch size, training epochs) and AdamW optimizer configurations to ensure fair comparison. Images are uniformly resized to $256 \times 256$ without dynamic resolution or multi-scale strategies.

## Key Experimental Results

### Main Results (Comparison with Frontier Models)

| Model | Params | MMVP | CV-Bench 2D | TallyQA | GQA | VSR | TopViewRS | CountBenchQA |
|-------|--------|------|-------------|---------|-----|-----|-----------|-------------|
| Qwen2.5-VL | 8B | 0.770 | 0.754 | 0.800 | 60.39 | 0.456 | 0.891 | — |
| LLaVA-OneVision | 7B | 0.767 | 0.730 | 0.797 | 62.14 | 0.414 | 0.823 | — |
| Molmo | 7B | 0.753 | 0.728 | 0.808 | 55.30 | 0.323 | 0.858 | — |

| Model | MMVP | CV-Bench 2D | TallyQA | GQA | VSR | CountBenchQA |
|-------|------|-------------|---------|-----|-----|-------------|
| Qwen2.5-VL (best frontier) | 0.770 | 0.754 | 0.800 | 60.39 | 0.456 | 0.891 |
| LLaVA v1.5 (CLIP) | 0.577 | 0.490 | 0.707 | 33.23 | 0.384 | 0.468 |
| LLaVA-AIMv2 | 0.513 | 0.466 | **0.710** | 32.54 | 0.339 | **0.739** |
| LLaVA-AIMv2-2D-RoPE | **0.560** | 0.432 | 0.690 | 32.34 | **0.338** | 0.719 |
| LLaVA-SigLIP | 0.433 | 0.412 | 0.672 | 25.65 | 0.349 | 0.581 |
| LLaVA-SigLIP-2D-RoPE | 0.507 | 0.425 | 0.616 | **38.45** | 0.295 | 0.483 |

### Ablation Study: Encoder × 2D-RoPE Interaction

| Encoder | Benchmarks Where 2D-RoPE Helps | Benchmarks Where 2D-RoPE Hurts | Overall Assessment |
|---------|-------------------------------|--------------------------------|--------------------|
| CLIP | — | CV-Bench, CountBenchQA, VSR, and multiple others | 2D-RoPE is broadly harmful with CLIP encoder |
| SigLIP | GQA (+12.8pp) | TallyQA, CountBenchQA | Mixed effects |
| SigLIP2 | MMVP (+0.053) | TopViewRS (−0.130) | Mixed effects |
| AIMv2 | MMVP (+0.047), VSR (≈unchanged) | CV-Bench (−0.034) | Most consistent improvements |

### Key Findings
- **Encoder choice dominates spatial performance**: AIMv2 yields the most consistent results in controlled experiments, outperforming other LLaVA variants on CV-Bench, TallyQA, and CountBenchQA. Dense/generative training objectives demonstrably improve spatial representations.
- **2D-RoPE offers limited and unstable gains**: It improves certain encoder–benchmark combinations (e.g., AIMv2 + MMVP) while degrading others (e.g., CLIP + nearly all benchmarks), indicating that preserving 2D positional structure alone is insufficient to compensate for an encoder's spatial information deficiency.
- **Frontier models still show uneven spatial reasoning**: Qwen2.5-VL, while the strongest overall, exhibits large variance across benchmarks, suggesting that spatial reasoning has not been consistently resolved by general-purpose training and scaling.
- **Qualitative analysis confirms superior localization by AIMv2**: In object detection visualizations, AIMv2 produces tighter and more accurate bounding boxes, whereas SigLIP2 frequently generates shifted or loose boxes.
- **Certain spatial errors are systemic**: In the example of "is the chopstick to the left or right of the bowl," all models—including Qwen2.5-VL—give the same answer, suggesting that some spatial errors are not influenced by encoder or positional encoding choice.
- **CLIP remains the most balanced encoder overall**: Despite AIMv2's advantages on some benchmarks, CLIP performs best on MMVP and CV-Bench, indicating that global alignment objectives retain certain advantages.
- **SigLIP/SigLIP2 variants are relatively weakest**, particularly on TallyQA and GQA, where they fall far behind the CLIP baseline.

## Highlights & Insights
- The rigorously controlled experimental design is the paper's greatest contribution: by fixing the language model, training data, and training procedure and varying only one factor at a time, the causal inferences drawn are more credible. This methodology is transferable to any VLM study requiring isolation of design factor effects.
- AIMv2's consistent superiority implies that spatial reasoning failures in VLMs cannot be addressed solely through larger models or more data; the encoder training objective is a fundamental factor. Dense supervision (pixel-level or patch-level prediction) preserves spatial information more effectively than global contrastive learning.
- The negative result that 2D-RoPE effects are unstable is itself valuable—it demonstrates that spatial reasoning failures cannot be simply attributed to the dimensionality of positional encoding; the quality of underlying visual features is the more critical prerequisite.
- The comparative baseline framework provided in this paper can serve as a standard evaluation protocol for future encoder and positional encoding proposals: replacing the encoder under fully controlled conditions is sufficient to assess spatial reasoning performance.
- Qualitative object detection cases intuitively demonstrate how encoder differences map to spatial precision; AIMv2's precise localization capability warrants further investigation into the specific properties of its training objective.

## Limitations & Future Work
- All experiments are limited to LLaVA-1.5 (7B) at $256 \times 256$ resolution; whether the conclusions hold for larger models (e.g., 13B, 72B) and higher resolutions (e.g., 384, 512) requires further validation.
- Only the original LLaVA training data is used; the effects of spatially specialized data (e.g., SpatialVLM, MM-Spatial datasets) are not explored.
- A broader range of encoders (e.g., DINOv2, EVA-CLIP, InternViT) and positional encoding schemes (e.g., window-level 2D encoding, learnable 2D encoding) are not investigated.
- Quantitative analysis of encoder features is absent (e.g., degree of spatial information retention in attention maps, spatial locality metrics of patch features).
- The study focuses exclusively on 2D static spatial reasoning and does not address 3D or dynamic spatial reasoning (e.g., rotation, folding, and other more complex cognitive tasks).
- Implementation details of 2D-RoPE (e.g., frequency parameters, dimension allocation) may influence results, but no hyperparameter sensitivity analysis is conducted.

## Related Work & Insights
- **vs. Qwen2-VL**: Qwen2-VL introduces multimodal rotary embeddings to preserve spatial height and width information; the 2D-RoPE experiments in this paper constitute a systematic validation of that design. However, the unstable effects observed here suggest that Qwen2-VL's success may owe more to large-scale training and engineering optimizations than to positional encoding per se.
- **vs. SpatialRGPT/MM-Spatial**: These works improve spatial reasoning through spatially specialized training data, whereas this paper focuses on architectural and encoder-level factors—offering a complementary perspective. Jointly considering both lines of work suggests that improving spatial reasoning may require co-optimization of encoder and data.
- **vs. CLIP critique works (e.g., MMVP)**: MMVP demonstrates that CLIP fails on fine-grained visual matching; this paper further reveals that such failures extend to spatial reasoning and identifies alternative encoders such as AIMv2 as a remedial direction.
- **vs. Spatial-DISE**: Spatial-DISE evaluates cognitive spatial reasoning (rotation/folding), whereas this paper addresses more fundamental 2D spatial relations (position/counting); the two works are complementary in evaluation scope.

## Rating
- **Novelty**: ⭐⭐⭐ The controlled experimental methodology is rigorous, but the core hypothesis—that encoders and positional encoding affect spatial reasoning—is already anticipated in prior work; the paper's contribution lies more in validation than in discovery.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Seven benchmarks, 4 encoders × 2 positional encoding schemes = 8 LLaVA variants, complemented by 7 frontier model comparisons and qualitative analysis; the design is thorough.
- **Writing Quality**: ⭐⭐⭐⭐ Concise and clear; the logic of variable control is explicit, and conclusions are stated cautiously without overinterpretation.
- **Value**: ⭐⭐⭐⭐ Provides empirically grounded guidance for architectural design in VLM spatial reasoning; the finding that "encoder training objective > positional encoding dimensionality" offers meaningful reference for the community.

<!-- END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Spatial CAPTCHA: Generatively Benchmarking Spatial Reasoning for Human-Machine Differentiation](spatial_captcha_generatively_benchmarking_spatial_reasoning_for_human-machine_di.md)
- [\[ICLR 2026\] Spatial-DISE: A Unified Benchmark for Evaluating Spatial Reasoning in Vision-Language Models](spatial-dise_a_unified_benchmark_for_evaluating_spatial_reasoning_in_vision-lang.md)
- [\[ICLR 2026\] SpinBench: Perspective and Rotation as a Lens on Spatial Reasoning in VLMs](spinbench_perspective_and_rotation_as_a_lens_on_spatial_reasoning_in_vlms.md)
- [\[ICLR 2026\] OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models](omnispatial_towards_comprehensive_spatial_reasoning_benchmark_for_vision_languag.md)
- [\[ICLR 2026\] SpatiaLab: Can Vision-Language Models Perform Spatial Reasoning in the Wild?](spatialab_can_vision-language_models_perform_spatial_reasoning_in_the_wild.md)

</div>

<!-- RELATED:END -->
