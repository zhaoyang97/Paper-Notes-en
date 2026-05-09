---
title: >-
  [Paper Note] ImageSentinel: Protecting Visual Datasets from Unauthorized Retrieval-Augmented Image Generation
description: >-
  [NeurIPS 2025][Image Generation][Retrieval-Augmented Image Generation] This paper proposes the ImageSentinel framework, which synthesizes sentinel images that are visually consistent with a private dataset and binds them to randomly generated character retrieval keys, enabling reliable detection of unauthorized use of private datasets by retrieval-augmented image generation (RAIG) systems—achieving near-100% AUC with only 3–10 queries.
tags:
  - NeurIPS 2025
  - Image Generation
  - Retrieval-Augmented Image Generation
  - Dataset Protection
  - Sentinel Images
  - Watermarking
  - Copyright Detection
  - RAIG
date: 2026-05-08
content_hash: a081904ab186f75b
---

# ImageSentinel: Protecting Visual Datasets from Unauthorized Retrieval-Augmented Image Generation

**Conference**: NeurIPS 2025
**arXiv**: [2510.12119](https://arxiv.org/abs/2510.12119)
**Code**: [GitHub](https://github.com/luo-ziyuan/ImageSentinel)
**Area**: Image Generation / Dataset Copyright Protection
**Keywords**: Retrieval-Augmented Image Generation, Dataset Protection, Sentinel Images, Watermarking, Copyright Detection, RAIG

## TL;DR
This paper proposes the ImageSentinel framework, which synthesizes sentinel images that are visually consistent with a private dataset and binds them to randomly generated character retrieval keys, enabling reliable detection of unauthorized use of private datasets by retrieval-augmented image generation (RAIG) systems—achieving near-100% AUC with only 3–10 queries.

## Background & Motivation
**State of the Field**: Retrieval-Augmented Image Generation (RAIG) enhances generation quality by retrieving relevant reference images from external databases, demonstrating strong performance on tasks such as rare concept generation and fine-grained image synthesis. Representative systems such as ImageRAG have shown that RAIG can be directly applied to existing text-to-image models (e.g., SDXL + IP-adapter, OmniGen).

**Limitations of Prior Work**: RAIG systems heavily rely on high-quality reference image databases, and malicious users may incorporate private datasets into their retrieval systems without authorization. This not only infringes intellectual property but also poses legal and commercial risks. However, no effective mechanism currently exists to protect visual datasets from unauthorized use in RAIG systems.

**Failure of Traditional Approaches**: Digital watermarking, commonly used to protect text-based RAG systems, fails in visual RAIG—image generation involves complex feature extraction and recombination that destroys embedded watermark signals, preventing them from being preserved in generated outputs.

**Limitations of Semantic Retrieval-Based Protection**: Semantic-based retrieval protection faces two key challenges: (a) large-scale databases contain many semantically similar images, making precise localization of target content difficult; and (b) some RAIG systems bypass the retrieval process entirely when the generator can produce satisfactory results directly, causing semantic triggers to fail.

**Core Idea**: Inject carefully synthesized "sentinel images" into the private dataset. These images are visually consistent with the original data but are bound to unique random-character retrieval keys. By querying the RAIG system and checking whether outputs match the sentinel images, unauthorized use of the dataset can be reliably identified.

## Core Problem
How can one reliably detect whether a RAIG system has used a private visual dataset without authorization, without modifying the original private images or disrupting legitimate users' normal access?

## Method

### Overall Architecture
ImageSentinel comprises three core components: **key generation**, **sentinel image synthesis**, and **unauthorized use detection**. Before release, the dataset owner injects sentinel images into the private dataset to form a protected dataset. During detection, the owner queries the suspected RAIG system using predefined keys and analyzes whether the outputs match the sentinel images.

### Key Designs

1. **Key Generation**

   - Randomly generates combinations of upper- and lowercase letters (e.g., "VasWiW"), with a default length of 6 characters.
   - These random strings are extremely unlikely to appear in normal user prompts, ensuring no interference with the day-to-day operation of the RAIG system.
   - They simultaneously serve as unique triggers for detecting unauthorized dataset use.

2. **Sentinel Image Synthesis**

   - **Stage 1: Semantic Attribute Extraction**—A reference image $I_r$ is randomly sampled from the private dataset $\mathcal{D}_p$, and a vision-language model (GPT-4o) is used to extract a semantic attribute set $\mathcal{A}$ (subject, style, tone, etc.) along with a detailed description $d_r$.
   - **Stage 2: Key-Guided Synthesis**—The extracted attributes, description, and random character key $k$ are combined via a templated prompt and fed into a text-to-image model $\mathcal{T}$ to generate sentinel image $I_s$:
     $I_s \leftarrow \mathcal{T}(\mathcal{A}, d_r, p_k)$
   - The synthesized sentinel images satisfy three properties simultaneously:
     - **Stealthiness**: Visually and semantically consistent with the original dataset, making them difficult to distinguish.
     - **Transparency**: Does not degrade the generation quality of the RAIG system for authorized users.
     - **Triggerability**: Can be reliably retrieved and activated by the predefined key.
   - The sentinel dataset $\mathcal{D}_s$ is far smaller than the original dataset: $|\mathcal{D}_s| \ll |\mathcal{D}_p|$.
   - The protected dataset is $\hat{\mathcal{D}}_p = \mathcal{D}_p \cup \mathcal{D}_s$ (addition, not replacement).

3. **Detection**

   - Prompts constructed from predefined keys $k \in \mathcal{K}$ are used to query the suspected RAIG system, yielding generated images $I_{\text{out}}^k$.
   - DINO ViT-S/16 is used to extract features, and the cosine similarity between the generated image and the corresponding sentinel image is computed:
     $\phi(I_{\text{out}}^k, I_s^k) = \cos(f_{\text{DINO}}(I_{\text{out}}^k), f_{\text{DINO}}(I_s^k))$
   - Similarity scores are aggregated across all queries:
     $s = \frac{1}{|\mathcal{K}|} \sum_{k \in \mathcal{K}} \phi(I_{\text{out}}^k, I_s^k)$
   - When $s > \eta$ (a predefined threshold), the RAIG system is determined to have used the private dataset without authorization.

### Threat Model
- The dataset owner can preprocess the dataset before release and can query the RAIG system and analyze its outputs.
- Black-box setting: the owner has no direct access to the RAIG system's reference database or generator parameters.
- Goal: detect unauthorized use while preserving dataset utility for authorized applications.

## Inference / Detection Pipeline (Training-Free)

This method **requires no model training** and relies entirely on the inference capabilities of existing pretrained models.

### Protection Phase (Offline, One-Time)
1. **Key Sampling**: Randomly generate $|\mathcal{K}|$ strings of length 6 consisting of upper- and lowercase letters.
2. **Reference Image Sampling**: Randomly select a corresponding number of reference images from $\mathcal{D}_p$.
3. **Attribute Extraction**: For each reference image, invoke a VLM (GPT-4o) to extract semantic attributes (subject, style, tone, composition, color palette, etc.) and a detailed description.
4. **Sentinel Synthesis**: Concatenate attributes, descriptions, and the corresponding key into a templated prompt, and feed it into a text-to-image model (GPT-4o) to generate sentinel images.
5. **Dataset Merging**: $\hat{\mathcal{D}}_p = \mathcal{D}_p \cup \mathcal{D}_s$; release the protected dataset.

### Detection Phase (Online, On-Demand)
1. Construct query prompts using predefined keys (e.g., "Generate an image of VasWiW").
2. Submit the prompts to the suspected RAIG system and collect generated images $I_{\text{out}}^k$.
3. Extract features from both the generated images and the corresponding sentinel images using DINO ViT-S/16.
4. Compute cosine similarities and aggregate them across all keys into a detection score $s$.
5. If $s > \eta$, the system is flagged for infringement.

The computational bottleneck of the entire pipeline lies in sentinel image synthesis (requiring $|\mathcal{K}|$ calls each to the VLM and the text-to-image model), but this is a one-time cost. The detection phase requires only 3–10 RAIG queries plus DINO forward passes, incurring negligible overhead.

## Key Experimental Results

### Experimental Setup
- **Datasets**: LLaVA-Pretrain (10,000 images), Product-10K (30,000 images)
- **RAIG Systems**: SDXL + IP-adapter, OmniGen, GPT-4o
- **Retrievers**: CLIP ViT-B/32, SigLIP ViT-B/16
- **Detection Features**: DINO ViT-S/16
- **Baselines**: Ward-HiDDeN, Ward-FIN (text RAG watermarking methods adapted to images)

### Detection Performance (LLaVA-Pretrain, Core Results)

| RAIG System | # Queries | AUC | TPR@1%FPR | TPR@10%FPR |
|---|---|---|---|---|
| SDXL | 3 | 0.974 | 0.934 | 0.958 |
| SDXL | 10 | **1.000** | **1.000** | **1.000** |
| OmniGen | 3 | 0.873 | 0.584 | 0.744 |
| OmniGen | 20 | **1.000** | 0.996 | **1.000** |
| GPT-4o | 3 | 0.983 | 0.954 | 0.974 |
| GPT-4o | 10 | **1.000** | **1.000** | **1.000** |

- Baseline methods Ward-HiDDeN and Ward-FIN consistently yield AUC scores around 0.5–0.6, close to random guessing.

### Trigger Rate and Retrieval Precision
- ImageSentinel achieves a trigger rate of **100%** on both SDXL and OmniGen; semantic methods achieve only 21.3%/39.0%.
- Hit@1 retrieval accuracy: ImageSentinel 69.7% vs. semantic methods 58.3%.

### Product-10K Detection Performance
- AUC = 0.870 with 1 query; AUC = 0.999 with 5 queries; AUC = 1.000 with 8 or more queries.

### Generation Quality Preservation
- ImageSentinel has minimal impact on normal generation quality (CLIP/SigLIP/DINO similarity decreases by only 0.004–0.06 relative to the original).
- In contrast, a "sentinel replacement" strategy (replacing original images rather than adding) causes significant quality degradation.

### Ablation Study
- **Key length**: 6 characters is optimal (AUC 0.997); 4 and 8 characters yield slightly lower performance (0.965/0.972).
- **Text-to-image model**: Sentinel images generated by GPT-4o exhibit far superior visual consistency compared to those generated by SDXL.
- **Protection strategy**: Addition outperforms replacement, preserving the generative utility of the original dataset.

### Adaptive Attack
- **Detect-and-Inpaint Attack**: The attacker applies EasyOCR to detect text regions in all database images, then uses Stable Diffusion 2.0 Inpainting to restore those regions, attempting to remove character triggers from sentinels before they enter the database.
- Attack effectiveness varies significantly with the number of queries:
  - 5 queries: AUC drops sharply from 0.99 to 0.62; TPR@1%FPR drops from 0.98 to 0.15.
  - 50 queries: AUC recovers to 0.91; TPR@1%FPR = 0.65.
  - 100 queries: AUC = 0.98; TPR@1%FPR = 0.94, approaching the no-attack level.
- The attack has minimal impact on generation quality (CLIP: 0.772→0.769; DINO: 0.605→0.597), indicating that attackers can execute it at low cost.
- **However, the attack faces an inherent tension**: indiscriminately removing all detected text risks erasing semantically important content in the original dataset—such as brand names, product labels, and captions—thereby reducing the dataset's utility for normal retrieval-augmented generation tasks.

## Highlights & Insights
1. **High problem novelty**: The first work to systematically define and address the problem of detecting unauthorized use of visual datasets in RAIG systems.
2. **Elegant design**: Random characters serve as retrieval keys, circumventing the ambiguity of semantic retrieval and the issue of RAIG systems bypassing retrieval entirely.
3. **Non-invasive protection**: Original images remain unmodified; protection is achieved by adding a small number of sentinel images to the dataset.
4. **Strong practicality**: Black-box detection setting requiring only 3–10 queries to achieve near-perfect detection performance.
5. **Cross-system generalization**: Effective across three different RAIG systems (SDXL, OmniGen, GPT-4o).

## Limitations & Future Work
1. **Dependence on text rendering capability**: SDXL has limited ability to embed characters into images; the current approach mainly relies on GPT-4o, and future stronger text-to-image models could further improve performance.
2. **Limited robustness to adaptive attacks**: The Detect-and-Inpaint attack can substantially weaken detection performance at low query counts, motivating the need for more robust protection strategies.
3. **Single detection metric**: Only DINO cosine similarity is currently used; more precise similarity measures may further improve detection.
4. **Unvalidated scalability**: The largest database evaluated contains only 30,000 images; performance on larger-scale databases has not been verified.
5. **Visibility of sentinel images**: Although semantically consistent with the dataset, sentinel images contain visible random-character text, allowing attackers to locate and remove them via simple OCR scanning.

## Related Work & Insights

| Method | Setting | Protection Mechanism | Detection Approach | Performance |
|---|---|---|---|---|
| Ward-HiDDeN | Text RAG → image adaptation | Deep watermark embedded in original images | Extract watermark from generated images | AUC ≈ 0.55, effectively fails |
| Ward-FIN | Text RAG → image adaptation | Flow-model watermark embedded in original images | Extract watermark from generated images | AUC ≈ 0.53, effectively fails |
| Semantic Retrieval Method | Image RAIG | Semantic description matching | Semantic similarity detection | Low trigger rate (21–39%) |
| **ImageSentinel** | **Image RAIG** | **Synthesized sentinel images + random keys** | **DINO feature similarity** | **AUC ≈ 1.0 (10 queries)** |

The core advantages are: (1) avoidance of the fundamental problem of watermarks being destroyed during image generation; and (2) random character keys enable precise triggering rather than ambiguous semantic matching.

## My Notes

### Core Insights
1. **Transferable paradigm**: The sentinel image + key-binding protection paradigm is generalizable to other retrieval-augmented generation settings, such as video generation and 3D generation.
2. **Connection to data poisoning**: Sentinel images are essentially a form of "benign poisoning"—planting controllable triggers in a dataset for detection rather than attack. Methodologically, this shares the same technical pipeline as backdoor attacks (trigger injection → trigger activation → detection), but with the opposite intent.
3. **New paradigm for copyright protection**: The approach shifts from "embedding information in content" to "planting detectable probes in datasets," accommodating the significant transformations that generative AI applies to source material.
4. **Directions for improving robustness**: Frequency-domain or latent-space embedding could replace explicit text embedding to improve robustness against OCR + inpainting attacks.

### Critical Reflections
- **The most significant weakness is explicit text embedding**: The random character text visible in sentinel images allows attackers to locate sentinels via simple OCR scanning and directly remove them from the database—without resorting to inpainting. This is more direct and more devastating than the Detect-and-Inpaint attack discussed in the paper, as direct deletion of sentinel images completely neutralizes the protection mechanism.
- **Asymmetry in detection assumption**: The method assumes that the dataset owner can send queries to the RAIG system and receive outputs, but in practice many RAIG systems may be closed internal systems inaccessible to external queries.
- **Fundamental distinction from model watermarking**: This method protects a *dataset* rather than a *model*; when subsets of the same private dataset are partially used by multiple RAIG systems, false negatives may arise.
- **Trade-off between sentinel image count and detection reliability is insufficiently discussed**: The paper only notes that $|\mathcal{D}_s| \ll |\mathcal{D}_p|$, without providing concrete ratio recommendations or sensitivity analyses.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — The first complete framework targeting RAIG visual dataset protection, with original problem formulation and solution approach.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-dataset, multi-RAIG-system, multi-baseline comparisons with ablation studies and adaptive attack evaluation, though larger-scale validation is absent.
- Writing Quality: ⭐⭐⭐⭐ — Problem formalization is clear, method presentation is systematic, and figures are intuitive.
- Value: ⭐⭐⭐⭐ — Addresses a practical pain point in the RAIG domain, though the explicit text embedding in sentinel images is a notable weakness that limits deployment security.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Can Knowledge-Graph-based Retrieval Augmented Generation Really Retrieve What You Need?](can_knowledge-graph-based_retrieval_augmented_generation_really_retrieve_what_yo.md)
- [\[NeurIPS 2025\] GenIR: Generative Visual Feedback for Mental Image Retrieval](genir_generative_visual_feedback_for_mental_image_retrieval.md)
- [\[NeurIPS 2025\] Instance-Level Composed Image Retrieval](instance-level_composed_image_retrieval.md)
- [\[ICCV 2025\] Anti-Tamper Protection for Unauthorized Individual Image Generation](../../ICCV2025/image_generation/anti-tamper_protection_for_unauthorized_individual_image_generation.md)
- [\[AAAI 2026\] TruthfulRAG: Resolving Factual-level Conflicts in Retrieval-Augmented Generation with Knowledge Graphs](../../AAAI2026/image_generation/truthfulrag_resolving_factual-level_conflicts_in_retrieval-augmented_generation_.md)

<!-- RELATED:END -->
