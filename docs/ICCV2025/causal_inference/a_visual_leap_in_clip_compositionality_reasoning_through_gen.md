---
title: >-
  [Paper Note] A Visual Leap in CLIP Compositionality Reasoning through Generation of Counterfactual Sets
description: >-
  [ICCV 2025][Causal Inference][CLIP] This paper proposes a counterfactual image-text pair generation method based on block-based diffusion…
tags:
  - "ICCV 2025"
  - "Causal Inference"
  - "CLIP"
  - "compositional reasoning"
  - "counterfactual data augmentation"
  - "block-based diffusion"
  - "contrastive learning loss"
date: 2026-05-08
content_hash: ec02299e1ddf7257
---

# A Visual Leap in CLIP Compositionality Reasoning through Generation of Counterfactual Sets

**Conference**: ICCV 2025
**arXiv**: [2507.04699](https://arxiv.org/abs/2507.04699)  
**Code**: None (not released)  
**Area**: Causal Reasoning
**Keywords**: CLIP, compositional reasoning, counterfactual data augmentation, diffusion models, contrastive learning

## TL;DR
This paper proposes a block-based diffusion method leveraging LLMs and diffusion models to automatically generate high-quality counterfactual image-text pair datasets, accompanied by a set-aware loss function. Without manual annotation, the approach significantly improves CLIP's compositional reasoning ability, surpassing state-of-the-art methods on ARO/VL-Checklist and other benchmarks with substantially less data.

## Background & Motivation

**Background**: Vision-language models such as CLIP achieve cross-modal understanding through large-scale contrastive learning on image-text pairs, yet perform poorly on compositional reasoning (understanding attributes, spatial positions, and relations). They fundamentally behave as "bag-of-words" models—capable of aligning entities but unable to understand relationships among them.

**Limitations of Prior Work**: Previous data augmentation methods (e.g., ARO's word-order permutation, SVLC's word substitution) generate negative samples that are too simplistic, allowing the text encoder to discriminate positive from negative samples without consulting the image. Urbanek et al. further demonstrated that models fine-tuned on such data perform poorly on genuinely challenging relational datasets (Winoground, sDCI), indicating that apparent gains stem from overlapping construction patterns between training and test sets rather than genuine compositional understanding.

**Key Challenge**: Precise compositional variations along both the image and text modalities (modifying attributes, positions, and relations) are simultaneously required, yet generative models struggle to capture complex object relationships without accurate guidance. Simple text-only negative samples fail to provide sufficiently challenging training signals.

**Goal**: How to automatically generate high-quality, high-fidelity counterfactual image-text pairs in which images accurately reflect complex compositional relationship descriptions?

**Key Insight**: The image generation process is analogized to assembling a jigsaw puzzle—an LLM parses text to extract entities and spatial relations, individual "puzzle piece" images are generated for each entity, and these pieces are then assembled according to compositional rules.

**Core Idea**: LLM extracts entities and spatial relations → block-based diffusion generates each region independently and fuses them globally → a structured set-aware loss efficiently fine-tunes CLIP.

## Method

### Overall Architecture
Given a small number of real image-text pairs, the method generates a counterfactual dataset through a three-step pipeline: (1) An LLM parses text to extract entities, attributes, and positions, and generates diverse variants (color changes, position swaps, relation modifications); (2) Block-based diffusion independently generates an image patch for each entity, arranges them at specified coordinates as a reference layout, and then produces a globally coherent image through dynamically weighted diffusion; (3) CLIP filters out low-quality results. The resulting data is used to fine-tune CLIP via LoRA with a specially designed set-aware loss.

### Key Designs

1. **Counterfactual Data Augmentation Pipeline**:

    - **Function**: Starting from existing image-text pairs, generate counterfactual variants along three dimensions—attribute, position, and relation.
    - **Mechanism**: The LLM performs three types of operations—(a) attribute modification: alter one object's attribute while preserving position (e.g., white dog → brown dog); (b) position modification: adjust relative position descriptions while preserving attributes (e.g., left → right); (c) relation modification: after constraining the number of objects, perform Chain-of-Thought analysis of parts and relations to generate variants incrementally. Each type of change simultaneously produces a new text description and a corresponding new image, forming a "counterfactual set."
    - **Design Motivation**: The three-dimensional decomposition ensures complete coverage of compositional reasoning; by modifying both image and text simultaneously (rather than text alone), the approach prevents the text encoder from taking shortcuts.

2. **Block-based Diffusion Generation Strategy**:

    - **Function**: Generate images that accurately reflect complex compositional relationships.
    - **Mechanism**: The LLM provides a global scene description $T_{\text{global}}$, local descriptions for each entity $\{T_i\}$, and position coordinates $\{P_i\}$. A reference image $I_i$ is generated for each entity. During diffusion, the latent state update is:

$$\mathbf{h}_t = \mathbf{h}_t + w_{\text{global}}(t) \cdot \text{Attn}_{\text{global}} + \sum_i w_{\text{local}}(t) \cdot M_i \cdot \text{Attn}_i$$

    Dynamic weight scheduling: in early denoising steps $w_{\text{local}}$ is high (ensuring accurate per-entity generation), while $w_{\text{global}}$ gradually increases in later steps (ensuring global coherence). The spatial mask $M_i$ restricts each local attention to its corresponding spatial region.
    - **Design Motivation**: Conventional diffusion models struggle to simultaneously handle precise multi-entity attributes and spatial relations; block-wise independent generation followed by progressive global fusion elegantly resolves this problem. Local image references $I_i$ provide more precise appearance cues than text alone.

3. **Set-aware Loss Function**:

    - **Function**: A training loss tailored to the structure of counterfactual sets.
    - **Mechanism**: The total loss is $\mathcal{L} = \mathcal{L}_{\text{sets}} + \mathcal{L}_{\text{neg}}$. $\mathcal{L}_{\text{sets}}$ consists of two components: an intra-set loss $\mathcal{L}_{\text{intra}}$ that computes the similarity between all positive and negative image-text pairs within the same counterfactual set (in sigmoid loss form); and an inter-set loss $\mathcal{L}_{\text{inter}}$ that uses only the ground-truth image-text pair of each set as a representative to compute cross-set negatives. $\mathcal{L}_{\text{neg}}$ adversarially trains against word-order-permuted negative texts.
    - **Design Motivation**: This formulation is computationally cheaper than global contrastive loss (no full-batch negatives required) and exploits the natural structure of counterfactual sets—variants within the same set constitute the hardest negatives.

### Loss & Training
- CLIP is fine-tuned with LoRA, preserving original capabilities while enhancing compositional reasoning.
- Counterfactual set size $m$ is controllable, typically containing 3–5 variants.
- CLIP filtering removes generated results with low image-text similarity to ensure data quality.

## Key Experimental Results

### Main Results

| Method | Training Data | ARO-Relation | ARO-Attribute | VL-CL Relation | VL-CL Attribute |
|--------|--------------|-------------|---------------|----------------|-----------------|
| CLIP | - | 59.9 | 62.9 | 61.9 | 67.6 |
| NegCLIP | Manual | 81.1 | 70.9 | 63.5 | 72.2 |
| DAC_LLM-3m | 3M | 81.3 | 73.9 | 86.4 | 77.2 |
| GCS_gen-10k (Ours) | **10K** | 82.2 | 67.7 | 74.5 | 71.9 |
| GCS_gen-300k (Ours) | **300K** | **85.7** | **73.4** | **85.6** | **81.4** |

### Ablation Study

| Configuration | ARO-Relation | VL-CL Relation | Note |
|--------------|-------------|----------------|------|
| Contrastive Loss | 77.5 | 73.8 | Standard contrastive loss |
| Set-aware Loss (Ours) | 85.7 | 85.6 | Structured loss, significant gain |
| w/o Block-based Diffusion | 79.3 | 78.1 | Degraded generation quality |
| w/ Block-based Diffusion | 85.7 | 85.6 | Precise control of object relations |

### Key Findings
- Only 10K generated samples suffice to surpass NegCLIP (ARO-Relation: 82.2 vs. 81.1); 300K samples comprehensively outperform DAC trained on 3M samples.
- Block-based diffusion is critical: it ensures generated images accurately reflect compositional relationships described in text; without it, low-quality image-text pairs introduce noise instead.
- The set-aware loss improves over standard contrastive loss by approximately 8%, leveraging the hard negatives provided by counterfactual set structure.
- Performance gains also appear on the genuinely challenging Winoground benchmark, demonstrating true compositional understanding rather than pattern matching.

## Highlights & Insights
- **Jigsaw puzzle–style image generation (Block-based Diffusion)** is the core innovation: it decomposes the challenging problem of generating compositionally complex images into "generate each piece separately → arrange by rule → fuse globally," achieving a perfect balance between local accuracy and global coherence via spatial masks and dynamic weights. This idea is transferable to any image generation task requiring precise multi-entity spatial control.
- **Remarkable data efficiency**: 10K generated samples already outperform large quantities of manually annotated data, demonstrating that data quality (precise compositional variation) far outweighs data quantity.
- **LLM as a compositional "decomposer and variant generator"** is particularly effective: LLMs naturally excel at parsing and rewriting structured descriptions, complementing the image generation capabilities of diffusion models.

## Limitations & Future Work
- Generated images may still exhibit unrealistic details (a common limitation of diffusion models); CLIP filtering may discard some valuable samples.
- For highly complex scenes (>5 entities with intricate occlusion relationships), the spatial partitioning of the block-based approach may lack sufficient flexibility.
- Validation is limited to CLIP (dual-encoder architecture); applicability to single-encoder VLMs remains unexplored.
- Potential improvements include incorporating 3D scene understanding to handle occlusion, or employing multi-turn dialogue to generate more complex compositional variants.

## Related Work & Insights
- **vs. NegCLIP/SVLC**: These methods modify only text without altering images, allowing models to learn shortcuts rather than genuine compositional understanding; this paper modifies both image and text simultaneously.
- **vs. DAC**: DAC relies on large-scale data (3M) but with a relatively uniform construction pattern; this paper achieves superior efficiency with a smaller set of high-quality counterfactual samples.
- **vs. sDCI**: Performance gains also appear on genuinely difficult benchmarks such as sDCI/Winoground, demonstrating better generalization.

## Rating
- Novelty: ⭐⭐⭐⭐ — Both block-based diffusion and the set-aware loss represent meaningful novel contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive validation across ARO, VL-Checklist, Winoground, and sDCI.
- Writing Quality: ⭐⭐⭐⭐ — The jigsaw puzzle analogy is intuitive and the pipeline description is clear.
- Value: ⭐⭐⭐⭐ — A data-efficient compositional reasoning enhancement solution with broad reference value for the VLM community.

**Area**: Multimodal VLM
**Keywords**: CLIP, compositional reasoning, counterfactual data augmentation, block-based diffusion, contrastive learning loss

## TL;DR
This paper proposes a counterfactual image-text pair generation method based on block-based diffusion, treating image entities as "puzzle pieces" for independent generation and assembly. Combined with a two-level intra-set/inter-set loss function for CLIP fine-tuning (LoRA), the method surpasses the previous state of the art—which uses 3M manually annotated samples—using only 10K–300K synthetic samples on multiple compositional reasoning benchmarks including ARO, Winoground, and sDCI.

## Background & Motivation
Despite significant progress on tasks such as image-text retrieval and VQA, vision-language models (VLMs) remain severely deficient in **compositional reasoning**—the ability to accurately understand object attributes, spatial positions, and inter-object relations. State-of-the-art models such as CLIP frequently degenerate to "bag-of-words" behavior: they can align textual entities with visual elements but cannot understand the relations or attributes among them. For example, CLIP often fails to distinguish fine-grained semantic differences such as "the person is to the left of the door" versus "the person is to the right of the door."

This deficiency becomes more pronounced when such models are integrated with LLMs to build multimodal dialogue systems (e.g., LLaVA, MiniGPT-4), leading to errors in scene description, VQA, and visual reasoning. The root cause lies in the absence of high-quality compositional reasoning training data with precise image-text alignment—despite the existence of large-scale datasets such as LAION-400M and YFCC100M, annotation quality and alignment remain insufficient.

Existing improvement strategies fall into two main paradigms, each with fundamental flaws:
1. **Text perturbation methods** (NegCLIP for ARO, VL-Checklist): negative text samples are generated by shuffling word order or substituting words. Urbanek et al. (sDCI, CVPR 2024) pointed out that such "negative samples" are too simplistic—the text encoder can distinguish positive from negative without any image information, meaning the model learns textual patterns rather than visual reasoning. Performance on benchmarks that require genuine visual understanding, such as Winoground, actually degrades.
2. **Local image editing methods** (VisMin, SDO): generative models locally modify images, but without precise spatial and semantic guidance, they struggle to accurately capture complex object relationships.

## Core Problem
1. **Data quality**: How to automatically generate image-text counterfactual training pairs with precise alignment—without manual annotation—while controlling changes along a specific compositional dimension (attribute/position/relation)?
2. **Generation precision**: Existing diffusion models, without precise guidance, struggle to generate images satisfying complex spatial and attribute constraints. How can a generative model be made to faithfully follow compositional relationship instructions?
3. **Training efficiency**: The effectiveness of standard contrastive loss depends heavily on negative sample quality and quantity. How can a more efficient loss function be designed to exploit the structured information in counterfactual sets?

## Method

### Overall Architecture
The proposed method (GCS, Generation of Counterfactual Sets) consists of three core modules forming a complete pipeline:

**COCO image-text pairs** → **Step 1: LLM parsing and variant expansion** (GPT-4o extracts entities, attributes, and positions; generates variant descriptions along three dimensions) → **Step 2: Block-based diffusion image generation** (each entity is independently synthesized as a "puzzle piece"; dynamic fusion of global and local guidance produces the complete image) → **Step 3: Set-loss CLIP fine-tuning** (two-level loss exploiting counterfactual set structure + LoRA efficient fine-tuning)

The overall concept is analogous to assembling a jigsaw puzzle: the LLM designs the assembly plan (which pieces go where and with what attributes), while the diffusion model fabricates each puzzle piece and assembles them into a complete scene. Counterfactual variants are produced by adding, removing, or modifying entities, with each original image yielding a counterfactual set.

### Key Designs

1. **Counterfactual Data Augmentation Pipeline**
    - **Entity parsing**: GPT-4o parses dense COCO captions (original short annotations are first expanded into dense descriptions by an LLM), identifying key entities along with their attributes and spatial relations. For example, from "a white dog on the left of a black cat," two entities—white dog and black cat—are extracted.
    - **Three-dimensional variant generation**:
        - **Attribute modification**: alter a property (color/type/state) of one object (e.g., black cat → yellow cat) while preserving position.
        - **Position modification**: swap the relative position description of objects (left ↔ right) while preserving attributes.
        - **Relation modification**: after constraining the number of objects, use Chain-of-Thought reasoning to analyze parts and interactions before generating variants (the most challenging dimension).
    - **Four operations**: for each original sample, four operations—add entity, remove entity, modify entity attribute, and regenerate with the same subject—each contribute 25% of counterfactual samples.
    - **Real image stitching supplement**: real images from COCO are stitched according to coordinates provided by the LLM, supplementing the generated data to form the training set.
    - **Quality filtering**: CLIP computes image-text similarity scores, filtering out low-quality generated results.
    - For cases in "relation modification" where entity swapping does not change semantics (e.g., "Jack and Mary are married"), a whitelist of approximately 800 common symmetric relations is maintained for filtering.

2. **Block-based Diffusion Generation Strategy** (Core Innovation)
    - **Information extraction**: the LLM extracts three categories of information from the input prompt—a global scene description $T_{global}$, local region descriptions $\{T_i\}$ for each entity (including attributes and spatial relations), and position coordinates $\{P_i\}$.
    - **Reference image generation**: a text-to-image model independently generates a reference image $I_i$ for each entity based on $T_i$, serving as a "puzzle piece."
    - **Dual-guided diffusion process**: at each denoising step of DDPM, both global semantic and local detail guidance are injected simultaneously:

$$\mathbf{h}_t = \mathbf{h}_t + w_{global}(t) \cdot \text{Attn}_{global} + \sum_i w_{local}(t) \cdot M_i \cdot \text{Attn}_i$$

    - **Local attention fusing text and image**: $\text{Attn}_i$ attends jointly to the local text description and local reference image via cross-attention:

$$\text{Attn}_i = \text{Softmax}\left(\frac{\mathbf{q}[\mathbf{k}_{T_i};\mathbf{k}_{I_i}]^\top}{\sqrt{d}}\right)[\mathbf{v}_{T_i};\mathbf{v}_{I_i}]$$

    where $[\cdot;\cdot]$ denotes concatenation. Text features provide semantic constraints while image features provide appearance consistency references.
    - **Spatial mask** $M_i$: restricts each entity's local guidance strictly to its predefined position region $P_i$.
    - **Dynamic weight scheduling** (key design):

$$w_{local}(t) = \begin{cases} w_{max}, & t \leq t_{th} \\ w_{max}\left(1 - \frac{t - t_{th}}{T - t_{th}}\right), & t > t_{th} \end{cases}, \quad w_{global}(t) = w_{max} - w_{local}(t)$$

    In early denoising steps the local weight is maximal, enabling each block to be generated independently to guarantee per-entity attribute fidelity; in later steps the local weight linearly decays and the global weight increases, fusing all blocks into a globally coherent image. $w_{max}$ is typically set to 1.

3. **Set-structured Loss Function**
    - Standard contrastive loss (InfoNCE) contrasts each positive sample against all other samples in the batch; its effectiveness is highly dependent on negative sample quality and batch size.
    - This paper exploits the natural grouping structure of counterfactual sets, decomposing computation into two levels:
        - **Intra-set loss $\mathcal{L}_{intra}$**: fine-grained discrimination within each counterfactual set—matching pairs within the same set are positives; non-matching pairs are negatives.
        - **Inter-set loss $\mathcal{L}_{inter}$**: each set is represented solely by its ground-truth image-text pair for coarse-grained cross-set contrastive learning.
    - Advantage: avoids the full $O(N^2)$ negative-pair computation; the intra-set already provides sufficiently hard negatives (counterfactual variants), while inter-set requires only coarse-grained discrimination.

### Loss & Training

**Intra-set loss** (sigmoid form, pairwise computation):
$$\mathcal{L}_{intra} = -\sum_{i=1}^{m}\sum_{j=1}^{m} \log \frac{1}{1 + e^{l_{ij}(-\tau \mathcal{I}(x_i, y_j) + b)}}$$

where $m$ is the set size, $l_{ij} \in \{1, -1\}$ denotes the positive/negative label, $\tau$ is the temperature parameter, $b$ is a learnable bias, and $\mathcal{I}(x,y) = \mathbf{z}_\mathbf{x}^\top\mathbf{z}_\mathbf{y} / (\|\mathbf{z}_\mathbf{x}\| \|\mathbf{z}_\mathbf{y}\|)$ is cosine similarity.

**Inter-set loss** (using only the ground-truth pair of each set as representative):
$$\mathcal{L}_{inter} = -\sum_{i=1}^{n}\sum_{j=1, j \neq i}^{n} \log \frac{1}{1 + e^{\tau \mathcal{I}(x_i^0, y_j^0) - b}}$$

where $n$ is the number of sets in the batch and $(x_i^0, y_i^0)$ is the ground-truth image-text pair of the $i$-th set.

**Negative text loss** (following SVLC, enhancing word-order sensitivity):
$$\mathcal{L}_{neg} = -\sum_i \log \frac{e^{\mathcal{I}(x_i, y_i)/\tau}}{e^{\mathcal{I}(x_i, y_i)/\tau} + e^{\mathcal{I}(x_i^{neg}, y_i)/\tau}}$$

where $x_i^{neg}$ is the negative text obtained by shuffling word order.

**Total loss**: $\mathcal{L} = \mathcal{L}_{sets} + \mathcal{L}_{neg}$, where $\mathcal{L}_{sets} = \mathcal{L}_{inter} + \sum_{i=1}^{n}\mathcal{L}_{intra}^{(i)}$.

**Training details**: CLIP ViT-B/32, batch size 100, learning rate 1e-5, weight decay 0.1, 10 training epochs, LoRA ($r=\alpha=32$). Diffusion models used: SD-XL, Stable Diffusion 3, PixArt-α. 4× NVIDIA V100 GPUs; results are averaged over 3 runs.

## Key Experimental Results

**ARO & VL-Checklist Benchmark (Table 1)**:

| Dataset | Metric | GCS (Ours) | DAC-3M (Prev. SOTA) | Gain |
|---------|--------|------------|---------------------|------|
| ARO VG-Relation | Acc | **85.7** | 81.3 | +4.4 |
| ARO VG-Attribute | Acc | 73.4 | 73.9 | −0.5 |
| ARO COCO-Order | Acc | **94.6** | 94.4 | +0.2 |
| ARO Flickr-Order | Acc | **95.6** | 95.4 | +0.2 |
| VL-Checklist Object | Acc | **89.1** | 88.5 | +0.6 |
| VL-Checklist Attribute | Acc | **81.4** | 77.2 | +4.2 |
| VL-Checklist Relation | Acc | 85.6 | 89.7 | −4.1 |

**Complex Visual Reasoning (Table 2)** — benchmarks most reflective of genuine compositional reasoning:

| Dataset | Metric | GCS | Prev. Best | Gain |
|---------|--------|-----|------------|------|
| sDCI SCM@1 | Acc | **47.9** | 47.6 (sDCI) | +0.3 |
| sDCI Neg@1 | Acc | **90.2** | 88.2 (sDCI) | +2.0 |
| sDCI SCM@5 | Acc | **15.6** | 15.1 (sDCI) | +0.5 |
| sDCI Neg@5 | Acc | **79.8** | 77.4 (sDCI) | +2.4 |
| Winoground Text | Acc | **32.8** | 31.3 (original CLIP) | +1.5 |
| Winoground Image | Acc | **19.5** | 14.5 (DAC) | +5.0 |
| Winoground Group | Acc | **10.0** | 9.0 (original CLIP) | +1.0 |

**Key Finding**: NegCLIP achieves only 8.0 on Winoground Group (below CLIP's 9.0); DAC reaches only 8.5. Text-perturbation methods are effective on ARO, whose construction patterns are simple, but degrade on Winoground, which requires genuine visual understanding. GCS is the only method that consistently improves across all benchmarks.

**Generated Image Quality Evaluation (Table 4)**:

| Dimension | Baseline (global only) | +Local Img | +Local Text | Combined |
|-----------|----------------------|-----------|------------|---------|
| Attribute | 84.5 | 90.2 | 93.8 | **99.6** |
| Position | 81.7 | 88.5 | 91.4 | **98.8** |
| Relation | 76.9 | 85.1 | 89.3 | **95.2** |

**Cross-model Generalization (Table 6)**:

| Model | ARO-A | ARO-R | Winoground | VL |
|-------|-------|-------|------------|-----|
| BLIP-2 (original) | 71.2 | 41.2 | 28.3 | 78.3 |
| BLIP-2 + GCS-300k | **76.3** | **74.9** | **30.9** | **86.4** |
| MiniGPT-4 (original) | 55.7 | 46.9 | 8.3 | 78.7 |
| MiniGPT-4 + GCS-300k | **71.0** | **60.2** | **15.3** | **82.1** |

Significant improvements are observed for both dual-encoder (BLIP-2) and multimodal LLM (MiniGPT-4) architectures.

**Downstream General Capability Preservation (Table 5, Elevater)**:

| Model | 5-shot | 10-shot | 20-shot | All-shot |
|-------|--------|---------|---------|---------|
| CLIP | 66.19 | 69.58 | 71.90 | 78.96 |
| DAC | 64.92 | 69.20 | 72.98 | 77.44 |
| GCS | 65.62 | 69.27 | 72.45 | 78.50 |

GCS fine-tuning preserves linear probing performance across 20 classification datasets at near parity with the original CLIP, without degrading general representations.

### Ablation Study
- **Loss vs. data**: the set-aware loss contributes most on VG-R and the COCO subset; counterfactual data contributes most on VG-A and Flickr. The two innovations are complementary and both necessary.
- **Data composition**: both stitched and generated images contribute independently; their combination yields the best results. Stitched images alone already substantially improve ARO, while adding generated images further improves Winoground.
- **Set size**: increasing from 5 → 10 → 20 elements yields steadily improving but diminishing returns (Group: 9.2 → 10.1 → 10.0); 10 is used by default.
- **Training efficiency**: the proposed set-aware loss reduces training time by **13.6%** relative to standard contrastive loss and by **16.2%** relative to sigmoid loss.
- **Block-based guidance ablation**: the pure baseline (global description only) performs worst; adding either local image or local text guidance provides clear gains; combining both achieves the optimum (attribute 99.6%, position 98.8%).

## Highlights & Insights
1. **The "jigsaw puzzle" metaphor is both elegant and effective**: the image generation problem for compositional reasoning is decomposed into a jigsaw assembly problem—each entity is one piece, the LLM designs the assembly plan, and the diffusion model fabricates and assembles each piece. This decomposition perfectly mirrors the essence of compositional reasoning: "understand each element separately, then understand their combination."
2. **Dynamic weight scheduling elegantly resolves the global-local tension**: high local weights in early steps guarantee per-block fidelity; increasing global weights in later steps ensure overall coherence. This simple and elegant design is transferable to other generation tasks requiring local-global balance.
3. **Data efficiency is remarkable**: 10K–300K synthetic samples vs. DAC's 3M manually annotated samples—3.6% of the data volume yet surpassing state-of-the-art. The key reason is that counterfactual pairs represent "minimally contrastive" hard negatives, providing far greater information per sample than random negatives.
4. **The loss function design fully exploits set structure**: intra-set loss provides fine-grained discrimination among counterfactual variants (hard negatives); inter-set loss requires only coarse-grained representative comparison (easy negatives); computational complexity is reduced from $O(N^2)$ to $O(nm^2 + n^2)$.
5. **Cross-architecture generality**: the same counterfactual dataset improves three architecturally distinct models—CLIP, BLIP-2, and MiniGPT-4—demonstrating that data quality is the critical bottleneck for compositional reasoning.

## Limitations & Future Work
1. **Generation quality on the relation dimension remains lower**: relation modification accuracy is 95.2%, lagging approximately 4–5 points behind attribute (99.6%) and position (98.8%); complex chain relations (e.g., "A blocks B from C's view because A occludes B") are entirely unaddressed.
2. **High generation cost**: the pipeline depends on GPT-4o and large-scale models such as SD-XL/SD3; the cost is non-trivial. Whether open-source alternatives (e.g., Llama 3 + SDXL) yield comparable results has not been verified.
3. **Only validated on CLIP ViT-B/32**: validation on larger backbones such as ViT-L/14 and ViT-H/14 is absent; main conclusions likely hold, but improvement magnitudes are unknown.
4. **Insufficient cross-model validation depth**: while results on BLIP-2 and MiniGPT-4 are positive, generalizability to more recent VLMs (LLaVA-NeXT, InternVL2, Qwen-VL2, etc.) remains unknown.
5. **Counterfactual variation patterns are limited**: the four operations (add, remove, modify, regenerate) primarily cover entity-level changes; higher-order compositional dimensions such as background variation, viewpoint changes, and lighting changes are not addressed.
6. **Restricted to static images**: the method has not been extended to temporal compositional reasoning in video understanding.

## Related Work & Insights

| Method | Data Source | Training Data | Augmentation Type | Winoground Group | sDCI Neg@1 |
|--------|------------|--------------|------------------|-----------------|-----------|
| NegCLIP (ICLR'23) | Text word-order shuffling | ~600K | Text only | 8.0 (↓1.0) | 56.0 |
| SVLC (CVPR'23) | Text + structured rules | ~600K | Text only | — | — |
| DAC (NeurIPS'23) | LLM-generated dense captions | **3M** | Text (+image) | 8.5 | 84.7 |
| sDCI (CVPR'24) | Human + LLM | ~113K | Text + cropping | 8.3 | 88.2 |
| **GCS (Ours)** | **LLM + Diffusion** | **≤300K** | **Joint image-text** | **10.0** | **90.2** |

Core distinctions: (1) **Visual-side counterfactuals**—NegCLIP/SVLC/DAC primarily augment on the text side, producing negative samples that are susceptible to shortcutting by the text encoder; GCS directly generates matching counterfactual images, compelling the model to perform genuine visual reasoning. (2) **Precisely controllable image generation**—block-based diffusion achieves unprecedented compositional precision through spatial masks and reference images. (3) **Structured loss**—leverages intra-set hard negatives without requiring global batch-level contrastive computation.

**Connections and Inspirations**:
- **Block-based diffusion → detection/segmentation data augmentation**: treating entities as independently manipulable puzzle pieces is transferable to object detection scenarios for automatically generating training images with varied object combinations, layouts, and occlusion patterns.
- **Set-structured loss → general methodology**: when training data possesses natural grouping structure, decomposing the loss into fine-grained intra-group contrast and coarse-grained inter-group contrast is an efficient and generalizable design pattern.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Both block-based diffusion and the set-aware loss represent meaningful novel contributions; the jigsaw intuition is effective and well-motivated.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Five benchmarks, detailed ablations, cross-model validation, and generation quality evaluation provide comprehensive coverage.
- **Writing Quality**: ⭐⭐⭐⭐ — The jigsaw metaphor runs consistently throughout the paper and the pipeline description is clear, though some mathematical notation could be more rigorous.
- **Practical Value**: ⭐⭐⭐ — High data efficiency, but the generation pipeline depends on GPT-4o and SD-XL, making reproduction non-trivial.
- **Overall**: ⭐⭐⭐⭐ (8/10)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] On the Eligibility of LLMs for Counterfactual Reasoning: A Decompositional Study](../../ICLR2026/causal_inference/on_the_eligibility_of_llms_for_counterfactual_reasoning_a_decompositional_study.md)
- [\[NeurIPS 2025\] Counterfactual Reasoning for Steerable Pluralistic Value Alignment of Large Language Models](../../NeurIPS2025/causal_inference/counterfactual_reasoning_for_steerable_pluralistic_value_alignment_of_large_lang.md)
- [\[ICLR 2026\] RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Perturbations](../../ICLR2026/causal_inference/rfeval_benchmarking_reasoning_faithfulness_under_counterfactual_perturbations.md)
- [\[CVPR 2026\] MaskDiME: Adaptive Masked Diffusion for Precise and Efficient Visual Counterfactual Explanations](../../CVPR2026/causal_inference/maskdime_adaptive_masked_diffusion_for_precise_and_efficient_visual_counterfactu.md)
- [\[ACL 2026\] Parallel Universes, Parallel Languages: A Comprehensive Study on LLM-based Multilingual Counterfactual Example Generation](../../ACL2026/causal_inference/parallel_universes_parallel_languages_a_comprehensive_study_on_llm-based_multili.md)

</div>

<!-- RELATED:END -->
