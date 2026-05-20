---
title: >-
  [Paper Note] ICEdit: Enabling Instructional Image Editing with In-Context Generation in Large Scale Diffusion Transformer
description: >-
  [NeurIPS 2025][Image Generation][Instruction-based image editing] ICEdit proposes an in-context editing paradigm built upon large-scale Diffusion Transformers (DiT)…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Instruction-based image editing"
  - "Diffusion Transformer"
  - "in-context learning"
  - "LoRA-MoE"
  - "inference-time scaling"
date: 2026-05-08
content_hash: cf55fb9dc1351462
---

# ICEdit: Enabling Instructional Image Editing with In-Context Generation in Large Scale Diffusion Transformer

**Conference**: NeurIPS 2025
**arXiv**: [2504.20690](https://arxiv.org/abs/2504.20690)  
**Code**: [Project Page](https://river-zhang.github.io/ICEdit-gh-pages)  
**Area**: Image Generation
**Keywords**: Instruction-based image editing, Diffusion Transformer, in-context learning, LoRA-MoE, inference-time scaling

## TL;DR

ICEdit proposes an in-context editing paradigm built upon large-scale Diffusion Transformers (DiT), achieving state-of-the-art editing performance with only 0.1% of the training data through an in-context prompt design, lightweight LoRA-MoE fine-tuning, and VLM-guided early-filter inference-time scaling.

## Background & Motivation

Instruction-based image editing has emerged as a prominent research direction in image generation, with the core objective of precisely modifying images according to natural language instructions. Existing approaches fall into two categories:

1. **Fine-tuning methods**: Such as InstructPix2Pix, EmuEdit, and UltraEdit, which require large-scale editing datasets (450K–10M samples) and full-model fine-tuning at high computational cost, often necessitating architectural modifications (e.g., additional conditional encoders, channel adjustments).
2. **Training-free methods**: Such as SDEdit, StableFlow, and RF-Solver, which achieve editing via image inversion or attention manipulation. These are computationally efficient but struggle with complex instructions and exhibit low editing success rates.

Both categories face a **precision–efficiency dilemma**. Meanwhile, large-scale DiT models (e.g., FLUX) exhibit two key properties: (1) superior text–image alignment, and (2) intrinsic context-awareness—enabling bidirectional interaction between reference images and generated content via attention mechanisms. This motivates the authors to explore whether DiT's inherent capabilities can be directly leveraged for instruction-based editing, without relying on external complexity.

## Core Problem

Directly applying DiT to instruction-based editing encounters two fundamental limitations:

1. **Poor instruction comprehension**: DiT can understand descriptive prompts (e.g., "a red cat") but fails to interpret editing instructions (e.g., "turn the cat red"), due to the embedding space gap between descriptive prompts and editing instructions.
2. **Layout instability**: During re-generation, the model frequently alters regions that should remain unchanged, resulting in low editing fidelity.

## Method

ICEdit consists of three core components:

### 3.1 In-Context Editing Paradigm

The core idea is to reformulate the editing task as a "diptych generation" task that DiT naturally excels at. Specifically, a side-by-side image pair is constructed: the original image is placed on the left, and the model generates the edited result on the right according to the instruction.

**Key Innovation — In-Context Edit Prompt**: A fixed-format prompt template is designed as follows:

> "A diptych with two side-by-side images of the same scene. On the right, the scene is exactly the same as on the left but {instruction}."

This prompt design embeds the editing instruction into a descriptive format that DiT can understand, with experiments demonstrating an approximately **70%** improvement in editing success rate. The authors compare three prompt formats: direct instruction (worst), IC prompt (significant improvement), and global descriptive prompt (best but impractical, requiring precise target image descriptions).

Two training-free frameworks are also proposed: one based on T2I DiT (requires image inversion, slower) and one based on Inpainting DiT (more streamlined, directly operating with a fixed mask). The inpainting framework is ultimately selected as the foundation for subsequent fine-tuning.

### 3.2 LoRA-MoE Efficient Fine-Tuning

While the training-free framework shows promise, its performance remains insufficient. The authors perform lightweight fine-tuning using only **50K** samples (from MagicBrush 9K + OmniEdit 40K).

**Limitations of vanilla LoRA**: A single LoRA structure struggles to handle diverse editing tasks (style transfer, object removal, etc.) simultaneously, as different tasks require distinct manipulations in the latent space.

**Solution — Mixture of LoRA Experts**: A MoE structure is introduced at the multimodal attention output projection layer of DiT blocks, formulated as:

$$\text{Output} = \text{BaseLayer}(x) + \frac{\alpha}{r} \sum_{i=1}^{N} G(x)_i \cdot B_i \cdot A_i \cdot x$$

where $G(x)_i$ denotes routing probabilities with Top-K sparse selection ($K=1$). Standard LoRA (rank=32) is applied to all other modules. This design yields only **0.2B trainable parameters** (compared to FLUX's full 12B), with a **13%** GPT score improvement over vanilla LoRA.

Training details: 4× A800 GPUs, one day, batch size=2 (with gradient accumulation), 512×512 resolution, Prodigy optimizer.

### 3.3 Early Filter Inference-Time Scaling

A key observation is that the initial noise has a substantial impact on editing outcomes, with significant quality variance across different noise samples.

**Key Insight**: In rectified flow models, editing success can be determined within very few denoising steps. This is because the model traverses the latent space efficiently, producing sufficiently informative coarse results in a small number of steps.

**Specific Strategy**:

1. Sample $M$ different initial noise vectors (default $M=6$).
2. Perform $m$ fast denoising steps for each (default $m=10$, far fewer than the full $n=50$ steps).
3. Use a VLM (Qwen-VL-72B) to select the best noise via pairwise bubble-sort-style comparisons.
4. Apply full $n$-step denoising to the selected noise to produce the final result.

The total computational cost is $\text{NFE} = n + M \times m = 50 + 6 \times 10 = 110$, substantially lower than evaluating all noise samples fully ($M \times n = 300$).

## Loss & Training

### Dataset Composition

The training set consists of 50K samples from two public datasets, **without careful curation** (noisy samples included):

| Task Type | Removal | Addition | Replacement | Attribute Modification | Style Transfer | Total |
|-----------|---------|----------|-------------|----------------------|---------------|-------|
| Samples | 13,272 | 11,938 | 5,823 | 11,484 | 10,530 | 53,047 |

MagicBrush (9K) is included to improve task-type balance, while OmniEdit (40K randomly sampled) adds diversity in style and domain.

### Training Configuration

- **Backbone**: FLUX.1 Fill (12B-parameter rectified flow inpainting DiT)
- **LoRA config**: rank=32, alpha=rank, 4 MoE experts, TopK=1
- **Router network**: Single linear layer; no auxiliary load-balancing loss (routing found to be naturally balanced)
- **Optimizer**: Prodigy (safeguard warmup + bias correction, weight decay=0.01)
- **Effective batch size**: 2 (batch=1 × 2 gradient accumulation steps)
- **Training resolution**: 512×512 → concatenated into 512×1024 diptych
- **Hardware**: 4× A800 (80G) GPUs, one day

**Memory consumption** (without gradient checkpointing):

| Resolution | Memory |
|------------|--------|
| 512×512 | 60 GB |
| 768×768 | 77 GB |
| 1024×1024 | OOM |

Enabling gradient checkpointing reduces memory to 37/39/42 GB, but significantly slows training. The authors opt for 512 resolution without gradient checkpointing to balance efficiency.

### Inference Configuration

- **Guidance scale**: 50
- **Denoising steps**: 50
- **IC prompt format** (consistent across training and inference): "A diptych with two side-by-side images of the same scene. On the right, the scene is exactly the same as on the left but {instruction}."
- **Inference scaling**: 6 random noise samples × 10-step fast inference; Qwen2.5-VL-72B performs pairwise evaluation via API
- **Inference hardware**: Single A100 (40G)

## Key Experimental Results

### Emu Edit Benchmark

| Method | Base Model | Trainable Params | Training Data | CLIP-I↑ | DINO↑ | GPT↑ |
|--------|-----------|-----------------|---------------|---------|-------|------|
| InstructPix2Pix | SD 1.5 | 0.9B | 0.45M | 0.856 | 0.773 | 0.36 |
| EmuEdit | Closed-source | 2.8B | 10M | 0.877 | 0.844 | 0.72 |
| UltraEdit | SD 3 | 2.5B | 3M | 0.880 | 0.847 | 0.54 |
| FluxEdit | Flux.1 dev | 12B | 1.2M | 0.852 | 0.760 | 0.22 |
| ACE++ | Flux.1 Fill | 12B | 54M | 0.791 | 0.687 | 0.24 |
| **ICEdit (ours)** | Flux.1 Fill | **0.2B** | **0.05M** | **0.907** | **0.866** | **0.68** |

### MagicBrush Benchmark

| Method | L1↓ | CLIP-I↑ | DINO↑ |
|--------|-----|---------|-------|
| UltraEdit | 0.066 | 0.904 | 0.852 |
| **ICEdit (ours)** | **0.060** | **0.928** | **0.853** |

### VIE-Score vs. Commercial Models

ICEdit (with inference-time scaling) achieves a VIE-Score of **78.2**, surpassing the commercial model SeedEdit (75.7). Inference-time scaling contributes a **19%** improvement in SC score and a **16%** improvement in overall VIE-Score.

### Ablation Study

- IC prompt vs. direct instruction: GPT score improves by **70%** (0.14→0.24)
- Adding LoRA fine-tuning: GPT score improves by a further **150%** (0.24→0.60)
- LoRA-MoE vs. vanilla LoRA: GPT score improves by **13%** (0.60→0.68)
- Data efficiency: 10K samples already substantially outperforms training-free methods; performance saturates at 50K
- CLIP evaluator vs. VLM evaluator for inference scaling: VLM significantly outperforms CLIP (0.78 vs. 0.65)

### MoE Expert Configuration Ablation

| # Experts | Expert Rank | # Params | CLIP-I↑ | GPT↑ |
|-----------|------------|----------|---------|------|
| 1 | 32 | 120M | 0.892 | 0.59 |
| 4 | 8 | 120M | 0.920 | 0.58 |
| **4** | **32** | **214M** | **0.907** | **0.68** |
| 6 | 32 | 270M | 0.914 | 0.66 |
| 8 | 32 | 335M | 0.907 | 0.61 |

Key finding: At equal parameter budgets (120M), 4 experts with rank=8 outperform 1 expert with rank=32 on CLIP-I, but yield similar GPT scores; scaling to 4×rank32 substantially improves GPT scores; however, further increasing the number of experts (6/8) degrades performance, suggesting that routing networks become harder to train with more experts and may require more sophisticated routing designs and load-balancing constraints.

### Evaluation Methodology

The authors find that traditional CLIP text–image direction similarity **is severely misaligned with human preferences**—successful edits may receive low scores while failed edits score high. Accordingly, GPT-4o is used for dual-dimensional evaluation: SC (instruction following + preservation of unedited regions) and PQ (perceptual quality), yielding VIE-Score $= \sqrt{\text{SC} \times \text{PQ}}$ after threshold-based binarization.

## Highlights & Insights

1. **Extreme data efficiency**: Only 0.05M (50K) training samples are required—0.5% of EmuEdit and 11% of InstructPix2Pix—yet SOTA performance is achieved or surpassed, challenging the prevailing assumption that image editing requires massive datasets.
2. **Zero architectural modification**: Unlike prior methods that require additional positional encoders, conditional encoders, or channel modifications, ICEdit fully preserves the original DiT architecture, relying solely on prompt engineering and lightweight LoRA adaptation.
3. **Elegant IC prompt design**: The "instruction editing" problem is reformulated as a "descriptive generation" problem, elegantly circumventing the embedding space gap between editing instructions and generative prompts.
4. **Novel inference-time scaling strategy**: Leveraging the "early detectability" property of rectified flow models combined with VLM-based evaluation, near-optimal quality is achieved at a fraction of the computational cost (NFE=110 vs. 350 for full search).
5. **Effective MoE-LoRA design**: Introducing routing experts at the attention output projection layer enables better multi-task editing capability with fewer parameters.

## Limitations & Future Work

1. **Object movement failures**: Instructions involving spatial relocation (e.g., "move the chair to the corner") perform poorly due to insufficient training samples of this type.
2. **Semantic ambiguity**: The T5 text encoder has limited semantic understanding and struggles with polysemous words (e.g., "mouse" as a computer peripheral vs. an animal); integrating MLLM modules could improve semantic fidelity.
3. **VLM inference overhead**: Inference-time scaling relies on a 72B-parameter Qwen-VL model; smaller models (7B) produce unreliable judgments. Distilling the VLM could alleviate this cost.
4. **Training resolution limitation**: Training is conducted at 512×512 resolution; performance at higher resolutions remains to be validated.
5. **MoE routing scalability**: The current 4-expert, TopK=1 configuration is sufficient, but performance degrades with more experts, indicating that routing network design and load-balancing constraints warrant further investigation.
6. **Uncurated training data**: The 50K training samples contain noisy examples; careful data curation is expected to yield further performance gains.

## Related Work & Insights

| Dimension | ICEdit | InstructPix2Pix | EmuEdit | UltraEdit | FluxEdit | ACE++ |
|-----------|--------|----------------|---------|-----------|----------|-------|
| Arch. modification | None (original DiT preserved) | Channel adjustment | Conditional encoder | Channel adjustment | Full fine-tuning | Position + conditional encoder |
| Training data | 50K | 450K | 10M | 3M | 1.2M | 54M |
| Trainable params | 0.2B | 0.9B | 2.8B | 2.5B | 12B | 12B |
| Base model | FLUX Fill (DiT) | SD 1.5 (UNet) | Closed-source | SD 3 (DiT) | FLUX dev (DiT) | FLUX Fill (DiT) |
| Editing paradigm | In-context diptych | Conditional injection | Conditional injection | Conditional injection | Full fine-tuning | In-context inpainting |

**Fundamental distinction from InstructPix2Pix**: InstructPix2Pix injects the source image as a condition by modifying UNet input channels, requiring large-scale data to train the model to understand editing instructions. ICEdit instead wraps editing instructions in descriptive prompts that DiT already understands and leverages the diptych structure to achieve image-to-image mapping, fundamentally bypassing the instruction embedding space gap.

**Comparison with ACE++**: ACE++ also builds on FLUX but requires 54M editing pairs, additional positional/conditional encoders, and achieves a GPT score of only 0.24—far below ICEdit's 0.68. This demonstrates that simply scaling data and parameters does not resolve the core challenge; the editing paradigm design is the critical factor.

**Comparison with training-free methods (RF-Solver Edit, StableFlow)**: Training-free methods rely on image inversion and attention manipulation, requiring carefully crafted source/target captions and cannot directly accept editing instructions. ICEdit's IC prompt enables DiT to understand instructions even in the training-free regime (GPT 0.24), reaching 0.68 after lightweight fine-tuning, validating the design philosophy that "paradigm > data volume."

**Broader Implications**:

1. **"Reformulate hard problems as tasks the model already knows"**: The core insight of ICEdit is not to teach the model new capabilities, but to reframe "instruction editing" as "descriptive diptych generation"—a task DiT already excels at. This paradigm is common in LLM prompt engineering (e.g., Chain-of-Thought) and is here successfully transferred to visual generation.

2. **"Early detectability" of rectified flow models**: The observation that flow matching models reveal generation quality within very few steps has potential applications well beyond noise selection—including automatic quality control, active learning sample filtering, and curriculum learning during training.

3. **MoE-LoRA as a general multi-task adaptation strategy**: Introducing routing experts at the attention output projection layer is transferable to other scenarios requiring a single model to handle multiple subtasks (e.g., multi-style text generation, multi-task visual understanding). The sufficiency of 4 experts + TopK=1 suggests that editing subtask diversity can be covered by a small number of experts.

4. **Deep connection to in-context learning**: The diptych paradigm can be viewed as few-shot in-context learning in the visual domain—the left image serves as the "example input" and the right as the "example output." This is conceptually aligned with in-context learning in GPT-series models, suggesting that sufficiently large DiT models may develop emergent in-context learning capabilities.

5. **VLM-as-judge for generative models**: Using Qwen-VL-72B for inference-time evaluation mirrors best-of-N sampling with reward models in RLHF. Future work may embed VLM-based judgment more broadly into generative pipelines, such as iterative automatic editing and generative quality filtering.

## Rating

- **Novelty**: ★★★★☆ — The in-context diptych editing paradigm is novel and elegant, and the IC prompt design is clever; however, LoRA-MoE and inference-time scaling each build upon prior work.
- **Experimental Thoroughness**: ★★★★★ — Dual-benchmark evaluation, VIE-Score comparison against commercial models, comprehensive ablation (prompt types, MoE configuration, data scale, inference scaling parameters), and GPT-4o human-preference-aligned evaluation.
- **Writing Quality**: ★★★★☆ — Clear structure with a natural logical flow from motivation to method, rich figures and tables; minor inconsistencies in notation.
- **Value**: ★★★★★ — Achieves SOTA with minimal training overhead, is open-source and reproducible, and offers broad methodological insights for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] CAMILA: Context-Aware Masking for Image Editing with Language Alignment](camila_contextaware_masking_for_image_editing_with_language.md)
- [\[NeurIPS 2025\] UltraHR-100K: Enhancing UHR Image Synthesis with A Large-Scale High-Quality Dataset](ultrahr-100k_enhancing_uhr_image_synthesis_with_a_large-scale_high-quality_datas.md)
- [\[NeurIPS 2025\] Large-Scale Training Data Attribution for Music Generative Models via Unlearning](large-scale_training_data_attribution_for_music_generative_models_via_unlearning.md)
- [\[NeurIPS 2025\] Hephaestus: Mixture Generative Modeling with Energy Guidance for Large-scale QoS Degradation](hephaestus_mixture_generative_modeling_with_energy_guidance_for_large-scale_qos_.md)
- [\[NeurIPS 2025\] SparseDiT: Token Sparsification for Efficient Diffusion Transformer](sparsedit_token_sparsification_for_efficient_diffusion_transformer.md)

</div>

<!-- RELATED:END -->
