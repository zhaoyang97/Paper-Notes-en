---
title: >-
  [Paper Note] Stable Cinemetrics: Structured Taxonomy and Evaluation for Professional Video Generation
description: >-
  [NeurIPS 2025][Video Generation][video generation evaluation] This paper proposes SCINE (Stable Cinemetrics), the first structured evaluation framework targeting professional video production. It defines a hierarchical t…
tags:
  - "NeurIPS 2025"
  - "Video Generation"
  - "video generation evaluation"
  - "cinematic taxonomy"
  - "professional video control"
  - "human evaluation"
  - "VLM evaluator"
date: 2026-05-08
content_hash: 8eb77f1e383dcdcf
---

# Stable Cinemetrics: Structured Taxonomy and Evaluation for Professional Video Generation

**Conference**: NeurIPS 2025
**arXiv**: [2509.26555](https://arxiv.org/abs/2509.26555)  
**Code**: [Project Page](https://stable-cinemetrics.github.io/)  
**Area**: Video Generation / Evaluation Benchmark
**Keywords**: video generation evaluation, cinematic taxonomy, professional video control, human evaluation, VLM evaluator

## TL;DR

This paper proposes SCINE (Stable Cinemetrics), the first structured evaluation framework targeting professional video production. It defines a hierarchical taxonomy with 76 fine-grained cinematic control nodes, accompanied by large-scale professional annotation (80+ film practitioners, 20K+ videos, 248K annotations), revealing significant deficiencies of current state-of-the-art T2V models in professional cinematic control.

## Background & Motivation

Video generation models have advanced rapidly, yet existing benchmarks (e.g., VBench, VideoPhy) fail to capture the requirements of professional video production. **The core gap between professional filmmaking and casual generation lies in cinematic control**: professional directors require precise control over shot composition, lighting quality, action timing, and every other cinematic element, rather than simply accepting a model output of "an astronaut riding a horse."

Specific limitations of existing benchmarks include:

**Lack of cinematographic depth**: VBench prompts such as "A man is walking" omit professionally essential information including character appearance, scene setup, and camera motion.

**Coarse evaluation dimensions**: Most benchmarks assess only overall prompt adherence, without attributing performance to specific control parameters.

**Static design**: Fixed prompt sets cannot scale with improving model capabilities.

**Lack of professional validation**: Automatic metrics are poorly aligned with expert human judgment.

The authors' core argument is that **the cinematic shot is the atomic unit of filmmaking** (averaging 5–10 seconds, which aligns naturally with the temporal limits of current models). A single shot involves a large number of mutually independent control parameters, providing a natural basis for structured evaluation.

## Method

### Overall Architecture

SCINE consists of three components:
1. **Taxonomy**: A hierarchical control tree with 4 pillars and 76 leaf nodes.
2. **Benchmark Prompts**: Two prompt types (narrative scripts + visual elaborations) simulating professional workflows.
3. **Evaluation Pipeline**: Automatic taxonomy mapping → question generation → large-scale human/automatic evaluation.

### Key Designs

1. **Four Taxonomic Pillars (76 control nodes)**:

    - **Setup**: Scene texture, geometry, set design, props, background, character appearance, etc.—"everything visible in the frame."
    - **Camera**: Intrinsics (focal length, depth of field, ISO), extrinsics (angle, height), trajectory (camera movement, tracking), and creative intent (composition, shot size).
    - **Lighting**: Light source type, color temperature, lighting conditions, effects, position, and advanced controls.
    - **Events**: Action types (standalone/interactive), emotions (explicit/implicit), dialogue, temporal presentation (atomic/causal/concurrent/cyclical), rhythm, and narrative structure.

   Design principle: A **hierarchical tree structure** ensures inter-branch independence (adjusting depth of field does not affect camera movement), supports multi-level abstraction, and facilitates extensibility.

2. **Prompt Design Pipeline**:

    - **SCINE-Scripts**: Seed prompts are created in collaboration with professional screenwriters; Events taxonomy nodes are sampled and narrative scripts are generated via LLM. t-SNE analysis confirms high distributional overlap with real screenplays.
    - **SCINE-Visuals**: Control nodes sampled from Camera, Lighting, and Setup taxonomy branches are injected into Scripts, enabling **structured prompt augmentation** (rather than unconstrained LLM expansion).
    - A single script can yield multiple visual interpretations (e.g., the same "man serving dinner to family" scene can be paired with shallow depth of field + warm lighting vs. deep depth of field + cold lighting).

3. **Automatic Taxonomy Mapping and Question Generation**: Each prompt is automatically mapped to taxonomy nodes, and independent evaluation questions are generated per node. For example, a prompt mentioning "tight close-up" and "flickering" generates separate evaluation questions for Shot Size and Lighting Motion, enabling **decoupled assessment of individual control nodes**.

### Loss & Training

**VLM Evaluator Training**: Based on Qwen-2.5-VL-7B, fine-tuned with a Bradley-Terry preference objective:
- Training set: 44,062 samples; validation set: 12,763 samples.
- Input: single video + prompt + evaluation question → Output: scalar score.
- A linear projection head on the last-layer token produces the scalar value.
- Sampled at 2 fps at native resolution; trained for 1 epoch.

## Key Experimental Results

### Main Results

**SCINE Visuals Comparison Across Four Pillars (13 Models)**

| Pillar | Top Model | Score Trend | Key Finding |
|--------|-----------|-------------|-------------|
| Setup | WAN-14B highest | Highest absolute scores | Best-performing dimension across all models |
| Lighting | Consistent across models | Smallest spread | Natural light > artificial light |
| Camera | Universally low | Narrow spread | Similar bottleneck across all models |
| Events | Largest gap | Only top-3 reliable | Most challenging dimension |

**Fine-Grained Analysis of Events**

| Sub-category | Performance | Notes |
|--------------|-------------|-------|
| Standalone Actions | Better | Standalone > interactive actions |
| Implicit Emotions | Better | Implicit > explicit emotions |
| Atomic Events | Better | Best performance among event types |
| Dialogues | Worse | Minimax leads but gap remains large |
| Causal/Overlapping | Worse | Events requiring temporal reasoning are broadly difficult |
| Advanced Controls | Worse | Rhythm and narrative structure are the hardest to control |

### Ablation Study

| Configuration | Key Observation | Notes |
|---------------|-----------------|-------|
| Basic vs. Advanced prompts | All models decline on Advanced | Largest drop in Lighting Source |
| Director prompts (joint control) | Largest drop in Camera | Multi-dimensional joint specification causes overall degradation |
| VLM scale 7B/32B/72B | No significant improvement | Zero-shot VLM alignment is poor |
| Fine-tuned 7B VLM | **72.36%** accuracy | ~20% improvement over zero-shot 72B |

### Key Findings

- **Three-tier ranking**: Minimax and WAN-14B lead → Luma Ray 2 / Hunyuan / WAN-1B in the middle → remaining models form a third tier.
- **No model excels uniformly**: Even the strongest models perform poorly on Events and Camera.
- Dutch angle is a challenge for all models; Medium-Wide and Extreme Close-up are the hardest shot sizes.
- Among lighting sources, Sunlight and Strobes perform well, while HMI and Fluorescent perform poorly.
- Causal and Sequential event performance is highly correlated ($\rho=0.94$), suggesting a shared temporal understanding capability.

## Highlights & Insights

- Designing an evaluation framework from a professional filmmaking perspective bridges the significant gap between generative model assessment and real-world application.
- The 76-node taxonomy is itself a significant contribution and can serve as a reference for control dimensions in future model training.
- Taxonomy-guided prompt augmentation is more controllable and interpretable than unconstrained LLM expansion.
- The large-scale professional annotation (248K labels, 84 practitioners, ICC 80.4%) provides a solid empirical foundation for all conclusions.
- The fine-tuned VLM evaluator outperforms the zero-shot 72B model in alignment, yet the 72% accuracy indicates substantial room for improvement in automatic evaluation.

## Limitations & Future Work

- The taxonomy is constrained by the collaborating expert network and may not cover cinematic traditions across diverse global cultures.
- Certain nodes (e.g., color temperature 2000K, ISO 800) are overly granular and difficult to accurately perceive even for human annotators.
- LLM-generated prompts may introduce biases.
- Only T2V models are evaluated; I2V models and multi-shot temporal coherence are not addressed.
- No closed-loop connection to model training is established—how to leverage these fine-grained evaluation results to guide model improvement remains unexplored.

## Related Work & Insights

- Complementary to VBench: VBench focuses on general video quality dimensions, while SCINE targets professional cinematic control.
- MovieNet provides film-level annotations but is not designed for generative model evaluation.
- This work may inspire the incorporation of domain expertise (e.g., cinematography, photography) into evaluation framework design for other modalities.
- The structured taxonomy can also be applied to analyze cinematic diversity in video datasets or to guide video captioning.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First systematic integration of professional filmmaking knowledge into video generation evaluation; the taxonomy design is rigorous.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 13 models, 20K videos, 248K annotations, 84 professional annotators — impressive in both scale and quality.
- **Writing Quality**: ⭐⭐⭐⭐ Rich in content but lengthy; the presentation of the taxonomy could be more compact.
- **Value**: ⭐⭐⭐⭐⭐ Significantly advances the evaluation paradigm for video generation; both the taxonomy and benchmark are likely to be widely cited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] WorldScore: A Unified Evaluation Benchmark for World Generation](../../ICCV2025/video_generation/worldscore_a_unified_evaluation_benchmark_for_world_generation.md)
- [\[ICCV 2025\] ETVA: Evaluation of Text-to-Video Alignment via Fine-Grained Question Generation and Answering](../../ICCV2025/video_generation/etva_evaluation_of_text-to-video_alignment_via_fine-grained_question_generation_.md)
- [\[ICML 2026\] T2AV-Compass: Towards Unified Evaluation for Text-to-Audio-Video Generation](../../ICML2026/video_generation/t2av-compass_towards_unified_evaluation_for_text-to-audio-video_generation.md)
- [\[CVPR 2026\] SLVMEval: Synthetic Meta Evaluation Benchmark for Text-to-Long Video Generation](../../CVPR2026/video_generation/slvmeval_synthetic_meta_evaluation_benchmark_for_text-to-long_video_generation.md)
- [\[ICML 2026\] Attention Sparsity is Input-Stable: Training-Free Sparse Attention for Video Generation via Offline Sparsity Profiling and Online QK Co-Clustering](../../ICML2026/video_generation/attention_sparsity_is_input-stable_training-free_sparse_attention_for_video_gene.md)

</div>

<!-- RELATED:END -->
