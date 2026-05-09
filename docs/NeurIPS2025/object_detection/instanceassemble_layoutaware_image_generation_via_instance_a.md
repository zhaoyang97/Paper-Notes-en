---
title: >-
  [Paper Note] InstanceAssemble: Layout-Aware Image Generation via Instance Assembling Attention
description: >-
  [NeurIPS 2025][Object Detection][layout-to-image] This paper proposes InstanceAssemble, which injects an "instance assembling attention" mechanism into the Transformer blocks of DiT-based T2I models (SD3 and Flux). By performing independent cross-attention between image tokens within each bounding box region and their corresponding layout hidden states, the method achieves precise instance-level spatial control. A lightweight LoRA adaptation strategy maintains compatibility with existing style LoRAs. The paper also introduces the DenseLayout benchmark (5K images / 90K instances) and a multi-dimensional Layout Grounding Score (LGS) evaluation metric.
tags:
  - NeurIPS 2025
  - Object Detection
  - layout-to-image
  - instance assembling attention
  - DiT
  - LoRA
  - DenseLayout benchmark
date: 2026-05-08
content_hash: 3dd1ef16b6678f2e
---

# InstanceAssemble: Layout-Aware Image Generation via Instance Assembling Attention

**Conference**: NeurIPS 2025
**arXiv**: [2509.16691](https://arxiv.org/abs/2509.16691)
**Code**: [https://github.com/FireRedTeam/InstanceAssemble](https://github.com/FireRedTeam/InstanceAssemble)
**Area**: Object Detection
**Keywords**: layout-to-image, instance assembling attention, DiT, LoRA, DenseLayout benchmark

## TL;DR

This paper proposes InstanceAssemble, which injects an "instance assembling attention" mechanism into the Transformer blocks of DiT-based T2I models (SD3 and Flux). By performing independent cross-attention between image tokens within each bounding box region and their corresponding layout hidden states, the method achieves precise instance-level spatial control. A lightweight LoRA adaptation strategy maintains compatibility with existing style LoRAs. The paper also introduces the DenseLayout benchmark (5K images / 90K instances) and a multi-dimensional Layout Grounding Score (LGS) evaluation metric.

## Background & Motivation

**Background**: Layout-to-Image (L2I) is a core task in controllable image generation—users specify a set of bounding boxes with textual or visual descriptions, and the model must generate corresponding content at precise locations. As the DiT (Diffusion Transformer) architecture has replaced UNet as the dominant backbone for text-to-image generation (e.g., Stable Diffusion 3, Flux.1 series), L2I research must migrate from UNet-era methods such as GLIGEN and ControlNet to the DiT architecture.

**Limitations of Prior Work**: Existing L2I methods suffer from three major issues. First, **multi-instance feature leakage** is severe—in global attention mechanisms, when a layout contains many densely packed instances, the text conditioning information of different instances leaks across each other in the attention layers, causing object misalignment, confusion, or complete omission. This problem is especially pronounced when the number of instances exceeds 10. Second, **content control modality is limited**—the vast majority of existing methods support only textual descriptions to control the appearance of each instance, lacking visual reference image control capability, and pure text has inherent limitations in precisely describing fine-grained appearance features (color gradients, texture details, specific poses, etc.). Third, **architectural adaptation costs are high**—directly modifying the DiT architecture requires a large number of additional parameters and full fine-tuning, which is not only computationally expensive but also disrupts the compatibility between the original T2I model and the community's existing style LoRA ecosystem.

**Key Challenge**: The root of the problem lies in the fact that existing methods attempt to process the layout conditions of all instances simultaneously within the global attention space—this structurally makes it impossible to prevent cross-instance information interference. Traditional approaches include adding layout masks to global attention or injecting cross-attention conditions (e.g., GLIGEN's gated cross-attention, Layout Diffusion's cross-attention mask), but these methods only modulate the attention distribution as a "soft constraint" and cannot fundamentally isolate the feature interactions of different instances.

**Goal**: The core objectives are: (1) design a mechanism that can precisely control the position and content of each instance within the DiT attention layers without feature leakage under dense layouts; (2) support both text and visual modalities for content control; (3) implement as a lightweight LoRA plugin to maintain compatibility with the existing model ecosystem; (4) establish a rigorous evaluation framework for dense layouts.

**Key Insight**: The authors' core observation is that—if the global attention problem can be decomposed into multiple local instance-level attention problems, allowing image tokens in each instance region to interact only with their corresponding layout description, cross-instance feature leakage can be structurally and completely eliminated. This is not post-hoc patching, but isolation at the level of the "atomic operations" of the attention mechanism.

**Core Idea**: A parallel "instance assembling attention" branch is added to the Transformer blocks of DiT. Image tokens within each bounding box are extracted and subjected to cross-attention exclusively with the layout hidden state of that instance; the results are then aggregated back into the original hidden state via scatter-add, achieving instance-level spatial control without interfering with the global attention flow.

## Method

### Overall Architecture

InstanceAssemble makes "minimally invasive" modifications on top of existing DiT-based T2I models (SD3 or Flux). The overall pipeline is as follows:

**Input**: Global text prompt + a set of layout conditions (each containing bounding box coordinates and instance description text / visual reference).

**Processing**: First, the Layout Encoder (`TextBoundingboxProjection` module) applies Fourier positional encoding to each instance's bbox coordinates and concatenates the result with text embeddings projected by a text embedder, mapping through an MLP to a unified-dimension layout hidden state. Then, in each (or a subset of) Transformer blocks of the DiT, in addition to the original global self-attention and cross-attention, an additional "Instance-Assembling Attention" operation is performed: based on the spatial region indices (`img_idxs`) corresponding to each bbox, image tokens from that region are extracted from the current hidden state and subjected to cross-attention with the corresponding layout hidden state. The output is projected through a zero-initialized linear layer, scaled by `layout_scale`, and aggregated back into the original hidden state via scatter-add. The entire process is governed by LoRA weights, with the original DiT parameters frozen.

**Output**: The normal denoising diffusion process outputs an image.

A key inference strategy is the "**Grounding Ratio**"—layout control is applied only during the first $r \times T$ denoising steps (default $r=0.3$, i.e., the first 30%), after which `layout_scale` is set to 0, allowing the model to freely refine image details. This prevents the layout control from over-constraining the final image quality.

### Key Designs

1. **TextBoundingboxProjection (Layout Encoder)**:

    - **Function**: Encodes each instance's bounding box coordinates and text description into a unified layout hidden state vector.
    - **Mechanism**: This module first performs **dense sampling** of the bbox coordinates—each bbox is not represented by merely four corner points $(x_1, y_1, x_2, y_2)$, but instead $6 \times 6 = 36$ grid points are uniformly sampled within the bbox, generating a 72-dimensional coordinate vector (xy coordinates of 36 points). This enables the positional information to more finely cover the spatial extent of the instance. **Fourier positional encoding** is then applied to this 72-dimensional coordinate vector: using 8 different frequencies ($100^{k/8}$, $k=0,...,7$), sin/cos transforms are applied to each coordinate value, yielding a $72 \times 8 \times 2 = 1152$-dimensional position embedding. The text description is encoded by a CLIP text encoder and projected to a dimensionality consistent with the DiT's internal dimension via a `PixArtAlphaTextProjection` (two-layer MLP + SiLU activation), producing a positive embedding. Finally, the text embedding and position embedding are concatenated (dimension $d_{inner} + 1152$) and mapped to $d_{inner}$ dimensions through another MLP to obtain the final layout hidden state. All operations are multiplied by a binary mask $m_i \in \{0, 1\}$ indicating whether that slot contains an instance.
    - **Design Motivation**: Using densely sampled Fourier encoding rather than simple four-corner coordinates can more precisely express the spatial positions of bboxes with different sizes and aspect ratios in the latent space. By fusing textual and spatial information already at the embedding stage, each instance's layout hidden state carries both "what it is" and "where it is" in subsequent attention operations.

2. **Instance-Assembling Attention (Core Contribution)**:

    - **Function**: Within the Transformer blocks of DiT, independent local cross-attention is performed for each valid instance, so that image tokens in each bbox region are controlled only by their corresponding layout description.
    - **Mechanism**: Specifically, during the forward pass of each Layout Transformer Block, when `attention_type == "layout"` and `layout_scale != 0`, the following steps are executed: (a) AdaLayerNorm is applied to both the layout hidden states and image hidden states (using the same timestep conditioning `temb` as the original attention), yielding normalized representations; (b) indices of all valid instances are identified via `valid_mask = (layout_masks == 1)`; (c) for each valid instance $(i, j)$ (batch $i$, instance $j$), the image token indices covering that bbox region `img_idxs = img_idxs_list_list[i][j]` are retrieved, and local image tokens are extracted from `norm_hidden_states[i, img_idxs]`; cross-attention is performed with `norm_layout_hidden_states[i, j]`—image tokens serve as Q, and layout tokens serve as KV; (d) the attention output is projected through a **zero-initialized linear layer** `layout_forward` (i.e., `zero_module(nn.Linear(dim, dim))`), then scaled by `layout_scale`; (e) the outputs of all instances are aggregated onto a globally-sized tensor via **scatter-add**, with average pooling applied (divided by count `img_add_cnt`) for image token positions where multiple instance bboxes overlap; (f) the aggregated result is added back to the original hidden states.
    - **Design Motivation**: This design fundamentally eliminates multi-instance feature leakage—each instance's attention operation is completely independent, with no cross-instance key-value interactions. The zero-initialized linear layer ensures that the layout branch has zero influence on the original generation process at the start of training (i.e., "progressive" injection), avoiding training instability from improper initialization. The scatter-add + average aggregation strategy supports arbitrary numbers of instances and overlapping bboxes. The entire process is wrapped in an `enable_lora()` context manager, ensuring that only LoRA parameters are updated.

3. **Lightweight LoRA Adaptation and Style Compatibility**:

    - **Function**: LoRA modules are inserted into the attention and normalization layers of DiT. Only low-rank incremental parameters are trained, with the original DiT weights frozen.
    - **Mechanism**: The InstanceAssemble weight files consist of two parts—`pytorch_lora_weights.safetensors` (LoRA incremental weights) and `layout.pth` (parameters for the Layout Encoder and zero-init linear). During inference, a `LayoutTransformer` is first initialized from the pretrained DiT model (created from the original transformer's config but with `attention_type` set to `"layout"`), the original weights are loaded (`strict=False`), followed by loading the LoRA weights and layout weights. A critical step after loading is to immediately **zero out the LoRA scales** (`_zero_out_lora_scales`)—this allows users to stack custom style LoRAs without conflict with the layout LoRA. The LoRA is temporarily activated via the `enable_lora()` context manager when needed, taking effect only during layout-related computations.
    - **Design Motivation**: Full fine-tuning is not only computationally expensive, but more critically, it destroys the original capabilities of the DiT and its compatibility with community style LoRAs. Through LoRA adaptation, users can simultaneously use a "layout control LoRA + style LoRA" to achieve combined control: "generate specified content at specified locations in a specified style." As visible in the code, Flux's layout attention is injected only into every 3rd joint transformer block (specifically block $i$ where `i % 3 == 0`) and only the first single transformer block (`i == 0`), further reducing the LoRA parameter count.

### Loss & Training

Training uses the standard diffusion denoising objective—given an image with complete layout annotations (bbox + instance descriptions), forward diffusion adds noise, and the model predicts the noise, optimizing an MSE loss. Only LoRA parameters, Layout Encoder parameters, and zero-init linear parameters are updated; the original DiT parameters are completely frozen. Training data requires images with multi-instance bbox annotations (e.g., in COCO format), where each instance has a category name and an optional detailed text description. InstanceAssemble supports two backbones—SD3 (18 joint transformer blocks) and Flux (19 double blocks + 38 single blocks)—and the same layout module pattern can be seamlessly adapted to both architectures.

An interesting training design choice is the `layout_pre_only` parameter—in the SD3 version, all Transformer blocks except the last one retain the full AdaLayerNormZero and FFN update paths for layout hidden states, while the last block uses a simplified `layout_pre_only=True` mode (norm only, no FFN). This is consistent with how the original SD3 architecture handles text hidden states (`context_pre_only=True` in the last layer), reflecting a deliberate design symmetry.

## Key Experimental Results

### Main Results

The paper conducts comprehensive evaluation on two benchmarks: a standard COCO-style sparse layout benchmark and the self-constructed DenseLayout dense layout benchmark. Evaluation metrics include traditional AP (detection accuracy), FID (generation quality), and the newly proposed LGS (Layout Grounding Score).

| Evaluation Dimension | Method | Performance | Notes |
|---|---|---|---|
| Sparse layout (COCO-style) | GLIGEN (UNet) | Baseline | UNet-based, requires full fine-tuning |
| Sparse layout (COCO-style) | InstanceDiffusion | Good | DiT adaptation but large parameter count |
| Sparse layout (COCO-style) | **InstanceAssemble (Flux)** | **SOTA** | Lightweight LoRA adaptation, minimal extra parameters |
| Dense layout (DenseLayout) | Existing methods | Sharp degradation | Performance collapses with >10 instances; severe object loss |
| Dense layout (DenseLayout) | **InstanceAssemble (Flux)** | **SOTA** | Stable performance even with 20+ bboxes |
| Dense layout (DenseLayout) | **InstanceAssemble (SD3)** | **2nd best** | SD3 backbone slightly weaker than Flux but far superior to baselines |
| Style LoRA compatibility | Other methods | Poor/Moderate | Full fine-tuning destroys compatibility |
| Style LoRA compatibility | **InstanceAssemble** | **Strong** | LoRA scale management ensures stacked usage |

InstanceAssemble supports two DiT backbones—Flux.1-dev (28-step inference), Flux.1-schnell (4-step fast inference), and SD3-medium (50-step inference). On the DenseLayout dataset, the method demonstrates a significant advantage: as the number of instances grows from 5 to 20+ in dense scenes, traditional methods suffer severe performance degradation due to feature leakage in global attention (object misalignment, confusion, and complete disappearance), while InstanceAssemble maintains stable spatial control accuracy owing to its instance-level isolated attention mechanism. Notably, performance on the Flux backbone is superior to SD3, consistent with Flux's stronger base generation capability.

### Ablation Study

| Configuration | Key Metric | Notes |
|---|---|---|
| Full model (Instance-Assembling Attention + LoRA) | Best | Complete model |
| Global attention + layout mask replacement | Degraded | Cannot fully isolate feature leakage |
| No dense sampling (4 corners only) | Degraded | Limited positional encoding precision |
| No Grounding Ratio (layout applied throughout) | Quality degraded | Over-constraining impairs detail generation |
| Non-zero initialization of `layout_forward` | Unstable | Prone to collapse in early training |
| Different Grounding Ratios (10%/30%/50%) | 30% optimal | Balances layout accuracy and generation quality; 10% gives imprecise positions, 50% loses detail |

### Key Findings

- **Instance-Assembling Attention is the core contribution**—replacing it with global attention + mask significantly degrades performance in dense layout scenarios, confirming that instance-level isolation is key to eliminating feature leakage.
- **Dense Fourier positional encoding contributes substantially**—dense sampling of 36 grid points ($6 \times 6$) provides more precise spatial position information than simple 4-corner representation, especially for elongated or extremely small bboxes.
- **Grounding Ratio of 30% is an effective default**—applying layout control for the first 30% of denoising steps is sufficient to establish object positions; the subsequent 70% of steps allow the model to freely refine details, achieving a good balance between layout accuracy and generation quality.
- **LoRA compatibility**—layout control capability is not diminished when stacked with various style LoRAs (anime, ink wash, 3D, etc.), which is highly important for practical applications such as design tools and e-commerce asset generation.
- **On the Flux architecture, layout attention is injected into only a subset of blocks**—one in every three joint blocks and only the first single block, indicating that layout control applied to all layers is unnecessary for achieving good results.

## Highlights & Insights

- **The idea of instance-level attention isolation is elegantly conceived.** Traditional methods attempt to "guide" the attention distribution in global attention via masks or conditional injection, which is essentially a soft constraint that cannot fully prevent key-value information leakage across instances. InstanceAssemble directly decomposes the problem into $N$ independent local attention operations, structurally eliminating interference. This divide-and-conquer idea can be generalized to other generative tasks requiring fine-grained spatial control.

- **The zero-initialized linear layer (`zero_module`) as a progressive injection mechanism.** This design ensures that the layout branch has zero influence on the original generation process at the start of training, allowing the model to progressively learn layout control capabilities starting from pretrained DiT weights, thus avoiding instability in the early stages caused by training the layout branch from scratch. This is a widely applicable technique in conditional generation (ControlNet uses a similar approach), but its application within instance-level attention is novel.

- **The Grounding Ratio design reflects a deep understanding of the diffusion process.** In the denoising process of diffusion models, early steps determine global structure and object positions, while later steps handle fine details. InstanceAssemble leverages this property by applying layout constraints only during the first 30% of denoising steps, then "releasing control" to allow the model to freely refine details—an elegant control strategy.

- **The DenseLayout benchmark and LGS metric fill an important evaluation gap.** Existing L2I benchmarks (e.g., COCO-based evaluations) have low layout density (typically 3–5 instances) and cannot expose problems in dense scenarios. DenseLayout contains data with an average of 18 instances per image (5K images / 90K instances), while LGS evaluates not only spatial localization accuracy (IoU) but also automatically judges—via VLM—whether color, texture, and shape are consistent with the description, providing a more comprehensive assessment than simple FID/AP metrics.

## Limitations & Future Work

- **Computational complexity scales linearly with the number of instances.** Since each valid instance requires an independent attention operation (including QKV projection and fully connected layers), inference time increases significantly when the number of instances is very large (e.g., 50 bboxes; the DenseLayout benchmark supports up to 100). The loop `for k in range(valid_indices.size(0))` in the code executes serially per instance, without batching optimization, which both underutilizes GPU parallelism and introduces Python loop overhead. A potential improvement would be to batch all instance attention operations via padding + batched attention, or to implement parallelization using custom CUDA operators.

- **Visual reference control has not yet been released.** The GitHub repository's TODO indicates that the "additional-visual control version" has not yet been published; currently only the text-controlled version is available. Visual reference control is critical for application scenarios requiring precise appearance consistency (e.g., character-consistent generation).

- **Training code is not open-sourced.** Currently only the inference code and pretrained weights have been released; training code has not been made public, limiting community verification and improvement of the method.

- **Validation is limited to SD3 and Flux.** The method has not been explored on other DiT variants (e.g., SD3.5, Flux.2, or subsequent versions) or UNet-based models. Transferring the instance-assembling attention idea to video generation models (e.g., OpenSora) or 3D generation models is also a worthwhile direction for exploration.

- **The LGS metric depends on the capabilities of GroundingDINO and VLMs.** LGS's IoU computation relies on GroundingDINO's detection accuracy (box_threshold=0.35, text_threshold=0.25), which may be unreliable for small objects and rare categories. Color/texture/shape judgments depend on MiniCPM-V-2.6's VQA capability, querying the VLM with "Does the [object] in the image match the [color/texture/shape] in the description?" If the IoU falls below the threshold (default 0.5), all attribute scores for that instance are set to 0. The limitations of these external models become a bottleneck for LGS reliability.

- **Overlapping bbox regions are handled by simple averaging.** When multiple instances' bboxes overlap, image tokens in the overlapping region receive control signals from multiple instances; the current approach is simple averaging via the `img_add_cnt` counter (`attn_output_add / img_add_cnt`). In dense overlap scenarios (e.g., stacked objects, occlusion), this may cause control conflicts, leading to blurring or inconsistency in the overlapping regions of the generated output.

## Related Work & Insights

- **vs. GLIGEN**: GLIGEN is based on the UNet architecture and injects layout conditions by inserting additional gated attention modules into cross-attention layers. It requires full fine-tuning of the entire UNet, has a large parameter count, and is incompatible with style LoRAs. InstanceAssemble improves on two fronts: (1) adapting to the mainstream backbone of the DiT architecture era; (2) replacing global conditional injection with instance-level isolated attention to eliminate feature leakage.

- **vs. InstanceDiffusion**: InstanceDiffusion also focuses on instance-level control, but its attention mechanism still operates in the global space, using attention masks to constrain cross-instance interaction ranges. InstanceAssemble goes further—physically isolating the attention operations of different instances, with each instance having its own independent Q/K/V computation path.

- **vs. IP-Adapter**: IP-Adapter achieves image prompt control via decoupled cross-attention—one cross-attention path processes text and another processes image references. InstanceAssemble extends this "decoupled attention" idea to the instance level—each instance has its own independent cross-attention path, supporting both text and visual conditions simultaneously.

- **vs. ControlNet**: ControlNet injects spatial conditions (e.g., edge maps, depth maps) by cloning the entire encoder, resulting in a very large parameter count (comparable to the original UNet/DiT encoder). InstanceAssemble's layout encoder consists only of two MLPs (`PixArtAlphaTextProjection`) and a zero-init linear layer, with an extremely small parameter count. Furthermore, the architectural design is completely orthogonal to ControlNet—the code retains the `controlnet_block_samples` and `controlnet_single_block_samples` interfaces, meaning users can simultaneously use ControlNet (providing global spatial conditions such as depth maps) and InstanceAssemble (providing instance-level layout control) for multi-level spatial control combinations.

- **Directions for Inspiration**: (1) Instance-level isolated attention can be generalized to object-tracking-aware generation in video generation—using per-frame object tracking information to guide the spatial grouping of attention; (2) The positional representation approach of dense sampling + Fourier encoding can be applied to other tasks requiring precise spatial control (e.g., spatial guidance in inpainting and super-resolution); (3) The "early constraint + late release" strategy of Grounding Ratio can be transferred to other conditional generation tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The instance-level isolated attention idea is clear and effective, structurally resolving the multi-instance leakage problem; the Fourier dense sampling positional encoding and Grounding Ratio strategy are also distinctive contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ A new benchmark (DenseLayout) and new metric (LGS) are proposed; validation is conducted on two DiT backbones (SD3, Flux) with ablation studies and style LoRA compatibility verification; it is unfortunate that the full-text HTML version is unavailable, making it impossible to verify specific quantitative results.
- **Writing Quality**: ⭐⭐⭐⭐ The motivation is clearly articulated, with a logical chain flowing from the "feature leakage" problem to the "instance-level isolation" solution; code structure is clean and open-source quality is high.
- **Value**: ⭐⭐⭐⭐⭐ A practical L2I solution for the DiT era—lightweight (LoRA), flexible (compatible with style LoRAs), and effective (dense layout SOTA), suitable for practical applications requiring precise layout control such as design tools, e-commerce asset generation, and game scene creation.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] OverLayBench: A Benchmark for Layout-to-Image Generation with Dense Overlaps](overlaybench_a_benchmark_for_layout-to-image_generation_with_dense_overlaps.md)
- [\[NeurIPS 2025\] MSTAR: Box-Free Multi-Query Scene Text Retrieval with Attention Recycling](mstar_box-free_multi-query_scene_text_retrieval_with_attention_recycling.md)
- [\[ICCV 2025\] YOLO-Count: Differentiable Object Counting for Text-to-Image Generation](../../ICCV2025/object_detection/yolo-count_differentiable_object_counting_for_text-to-image_generation.md)
- [\[NeurIPS 2025\] Delving into Cascaded Instability: A Lipschitz Continuity View on Image Restoration and Object Detection Synergy](lr_yolo_lipschitz_continuity_image_restoration_object_detection.md)
- [\[NeurIPS 2025\] DitHub: A Modular Framework for Incremental Open-Vocabulary Object Detection](dithub_a_modular_framework_for_incremental_openvocabulary_ob.md)

<!-- RELATED:END -->
