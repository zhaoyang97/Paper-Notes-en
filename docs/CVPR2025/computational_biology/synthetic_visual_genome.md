---
title: >-
  [Paper Note] Synthetic Visual Genome
description: >-
  [CVPR 2025][Computational Biology][Scene Graph] Proposes the **SVG** (Synthetic Visual Genome) data engine. Through a two-stage pipeline consisting of **completing missing relationships** on top of existing human annotations via GPT-4 (Stage 1) and **Robin self-distillation + GPT-4 editing** (Stage 2/SG-Edit), it generates a dense scene graph dataset with 146K images, 2.6M objects, and 5.6M relationships. The trained **Robin-3B** model outperforms same-sized models trained on…
tags:
  - "CVPR 2025"
  - "Computational Biology"
  - "Scene Graph"
  - "Relational Reasoning"
  - "Synthetic Data"
  - "Self-Distillation"
  - "Referring Expression Comprehension"
date: 2026-05-08
content_hash: e34bf58d25bb0236
---

# Synthetic Visual Genome

**Conference**: CVPR 2025  
**arXiv**: [2506.07643](https://arxiv.org/abs/2506.07643)  
**Code**: [https://synthetic-visual-genome.github.io/](https://synthetic-visual-genome.github.io/)  
**Area**: Computational Biology  
**Keywords**: Scene Graph, Relational Reasoning, Synthetic Data, Self-Distillation, Referring Expression Comprehension

## TL;DR

Proposes the **SVG** (Synthetic Visual Genome) data engine. Through a two-stage pipeline consisting of **completing missing relationships** on top of existing human annotations via GPT-4 (Stage 1) and **Robin self-distillation + GPT-4 editing** (Stage 2/SG-Edit), it generates a dense scene graph dataset with 146K images, 2.6M objects, and 5.6M relationships. The trained **Robin-3B** model outperforms same-sized models trained on over 300M instances using less than 3M instances, achieving a state-of-the-art (SOTA) score of 88.9 on referring expression comprehension.

## Background & Motivation

Visual relationship reasoning—understanding spatial, functional, interactive, social, and emotional relationships between objects—is considered a fundamental ability of human cognition. However, Multimodal Language Models (MLMs) still face challenges in precisely representing relationships.

**Background**: Instruction tuning has been proven effective in injecting specific reasoning capabilities into MLMs, but relationship-reasoning instruction tuning is limited by the lack of large-scale, densely labeled relationship datasets.

**Limitations of Prior Work**:
1. **Sparse Visual Genome Annotations**—Each object has an average of only 1.5 relationship annotations, and a large number of existing relationships are left unlabeled (e.g., obvious relations like "woman in front of baby" are missed).
2. **Single Relationship Type**—VG mainly contains spatial relationships, lacking interactive, emotional, functional, and social relationships.
3. **Manual Annotation is Unscalable**—Exhaustively labeling all relationships for all objects is extremely tedious and costly for human annotators.
4. **Poor Performance of Direct GPT-4 Generation**—Generating scene graphs from scratch via GPT-4V leads to severe hallucination and localization errors.

**Key Challenge**: Dense scene graphs are crucial for relationship reasoning, but human annotation is unscalable, and direct AI generation remains unreliable.

**Key Insight**: Rather than generating from scratch, **complete missing relationships** on top of existing high-quality human annotations—prompt GPT-4 to reason about missing relations after seeing existing object labels and relations, thereby significantly reducing hallucinations. Subsequently, iteratively expand to more images via a self-distillation pipeline.

## Method

### Overall Architecture

A two-stage pipeline: **Stage 1 (Dense Relationship Completion)**—Starting from a 33K seed image subset of COCO, multi-source annotations (detection, region descriptions, scene graphs, depth maps) and SAM segmentation are utilized to prompt GPT-4V to complete five types of relationships (spatial, interactive, emotional, functional, social) for selected objects, generating the SVG-Relations dataset. **Stage 2 (Self-Distillation Expansion/SG-Edit)**—First, Robin-3B (Stage 1) is trained on SVG-Relations and then used to generate scene graphs for new images (from ADE20K, PSG, VG, totaling 113K). These are then edited and corrected by GPT-4o (deleting wrong ones and adding correct ones) to generate the SVG-SG dataset. Finally, Robin-3B is trained on the complete data.

### Key Designs

1. **Dense Relationship Completion**
    - **Function**: Increases relationship annotation density by 4x while maintaining accuracy.
    - **Mechanism**:
     - Curate 33K COCO images, aggregating multi-source annotations from COCO, LVIS (detection), RefCOCO, VG (region descriptions), VG and GQA (scene graphs), and Depth-Anything (depth maps).
     - Use SAM/Semantic-SAM to generate segmentation masks, preserving regions with IoU > 0.5 with bounding boxes as "reliable regions".
     - Prompt GPT-4V to infer missing relationships for each salient object based on existing annotations, generating across five categories (spatial, interactive, functional, social, emotional).
     - Filter spatial relationships using rules and other relationships using VQA models to remove low-quality annotations.
    - **Design Motivation**: GPT-4V is ineffective at generating scene graphs from scratch (due to severe hallucinations), but performs significantly better when **completing** on top of existing annotations—reasoning about missing relationships when object positions and partial relations are already known is a much more reliable task.

2. **Self-Distillation with GPT-4 Editing (SG-Edit)**
    - **Function**: Scales Stage 1 capabilities to arbitrary images, achieving large-scale scene graph data generation.
    - **Mechanism**:
     - Train Robin-3B (Stage 1) on SVG-Relations as a student model.
     - Robin generates dense scene graphs for new images (efficient but potentially noisy).
     - GPT-4o acts as an editor to refine Robin's output: removing incorrect relations (red to green), adding missed relations, and supplementing object attribute descriptions.
     - Use the refined data (SVG-SG, 113K images) for Stage 2 training.
    - **Design Motivation**: Follows an iterative data improvement paradigm similar to Segment Anything—train model -> generate with model -> human/AI correction -> retrain model. Replacing human editors with GPT-4o significantly reduces costs.

3. **Mask-Aware Robin-3B Model Architecture**
    - **Function**: Supports regional understanding, relationship reasoning, and dense scene graph generation simultaneously.
    - **Mechanism**: A three-component architecture:
     - **Visual Encoder** (ConvNext-Large): Encodes the global image into image tokens.
     - **Pixel-level Mask-Aware Extractor** (ConvNext-Large): Encodes each segmentation mask into mask tokens (up to 99 regions).
     - **Language Model** (Qwen2.5-3B, 8192 token context): Receives image tokens, mask tokens, and text tokens, supporting arbitrary visual instructions and grounding tasks.
    - **Design Motivation**: Unlike models referencing regions solely using bounding-box text coordinates, Robin uses a dual representation of segmentation masks + text to achieve finer regional localization.

### Loss & Training

Three-stage progressive training:
- **Stage 0 (Alignment, 1.28M samples)**: Freeze the visual encoder, train the image projector (LLaVA-Pretrain-558K), then unfreeze the mask encoder to learn mask embeddings, and finally fine-tune the LM.
- **Stage 1 (Instruction Tuning + Scene Graph, 1.73M samples)**: Unfreeze the visual encoder, and perform joint training on visual instruction, grounding, and scene graph data (including SVG-Relations).
- **Stage 2 (Distillation Fine-tuning, 1.23M samples)**: Replace SVG-Relations with SVG-SG and continue training.

## Key Experimental Results

### Relationship Understanding Benchmarks (Comparison of Models $\le$ 4B)

| Model | Training Data Volume | GQA | VSR | MMBench | CRPE Rel | SugarCrepe | What's Up |
|------|-----------|-----|-----|---------|----------|------------|-----------|
| VILA1.5-3B | — | 61.5 | 61.0 | 63.4 | 67.8 | 86.3 | 50.6 |
| Phi-3-Vision-4B | 300M+ | — | 67.8 | 74.2 | 71.6 | 88.7 | 78.7 |
| BLIP-3-3B | 300M+ | — | 72.5 | 76.0 | 72.4 | 89.0 | 78.2 |
| **Robin-3B** | **<3M** | **61.6** | **76.4** | **77.6** | **68.2** | **90.1** | **86.2** |

### Referring Expression Comprehension (RefCOCO Series, R@1 IoU > 0.5)

| Model | Parameter Size | RefCOCO Val | Test-A | Test-B | Avg |
|------|--------|------------|--------|--------|-----|
| Ferret-13B | 13B | 89.5 | 92.4 | 84.4 | 85.6 |
| ASM-V2-13B | 13B | 90.6 | 94.2 | 86.2 | 87.4 |
| **Robin-3B** | **3B** | **91.6** | **94.3** | **88.6** | **88.9** |

### SVG Dataset Statistics

| Dataset | Number of Images | Annotator | Objects/Image | Relationship Triplets/Image | Relationships/Object |
|--------|-------|--------|---------|-------------|----------|
| VG | 108K | Human | 35.2 | 21.4 | 0.6 |
| GQA | 85K | Human | 16.4 | 50.6 | 3.1 |
| SVG-Relations | 33K | GPT-4V | 13.2 | 25.5 | **1.9** |
| SVG-SG | 113K | Robin+GPT-4o | 19.8 | 42.3 | **2.4** |

### Stage 1 vs Stage 2 Gain

| Benchmark | Robin-3B (Stage 1) | Robin-3B (Final) | Gain |
|------|-------------------|-----------------|------|
| VSR | 73.7 | 76.4 | +2.7 |
| CRPE Rel | 65.9 | 68.2 | +2.3 |
| What's Up | 81.3 | 86.2 | +4.9 |
| RefCOCO Avg | 87.2 | 88.9 | +1.7 |

### Key Findings

1. Robin-3B **outperforms Phi-3-Vision and BLIP-3 trained on over 300M instances while using less than 3M instances**, leading by 7.5% on What's Up (86.2 vs 78.7).
2. Referring expression comprehension achieves **SOTA of 88.9**, surpassing ASM-V2 with 13B parameters (87.4), demonstrating that data quality is more critical than model scale.
3. GPT-4 edit distillation (Stage 2) yields consistent gains across relationship understanding benchmarks, particularly +4.9% on What's Up.
4. The relationship density per object in SVG is 4x that of VG (2.4 vs 0.6), covering five categories of relationships instead of only spatial relations.

## Highlights & Insights

1. **"Completion" instead of "Generation from Scratch"**—Addresses the severe hallucinations associated with direct scene graph generation by GPT-4V. Reasoning about missing relationships over existing annotations is considerably more reliable than starting from scratch, offering valuable implications for other data augmentation tasks.
2. **SAM-like Iterative Data Engine**—The closed-loop paradigm of training model -> generating with model -> correcting with AI -> retraining model is highly scalable, and replacing human editors with GPT-4o substantially lowers the cost.
3. **3B Model Outperforms 13B Model**—Data quality (dense and diverse relationship annotations) is more crucial than model size for relationship understanding and referring expression comprehension.
4. **Systematization of Five Relationship Categories**—Classifying relationships into five categories (spatial, interactive, functional, social, emotional) is more comprehensive than traditional scene graphs focusing solely on spatial relations, aligning better with human scene understanding.

## Limitations & Future Work

1. Stage 1 seed images only originate from a COCO subset (33K), which may introduce domain bias.
2. The cost of involving GPT-4V/4o in data generation remains non-negligible, though cheaper than human labor.
3. The quality of GPT-4o's editing in the SG-Edit pipeline has not been systematically verified (which may introduce new biases).
4. Currently only scaled to 113K images (SVG-SG), without an in-depth analysis of the effects of further scaling.

## Related Work & Insights

- **Scene Graph Datasets**: Visual Genome is pioneering but sparse, while GQA is denser but still limited. SVG achieves breakthroughs in density and diversity through AI-assisted annotation.
- **Data Engine Paradigm**: Similar to the Segment Anything "model -> data -> model" closed loop, SVG applies this to the scene graph domain, demonstrating the generality of the paradigm.
- **Instruction Tuning**: High-quality relationship data is key to the relationship reasoning capabilities of MLMs—the success of Robin-3B demonstrates that "providing the correct data is more important than providing more data".
- **Insights**: In the LLM era, AI-assisted data annotation (completion rather than generation from scratch + AI editing/correction) is becoming a mainstream paradigm for constructing high-quality training datasets.

## Rating
- Novelty: ⭐⭐⭐⭐ The "completion instead of generation from scratch" data engine approach is novel, and the SG-Edit self-distillation pipeline is elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive ablation, covering four major tasks: relationship understanding, referring expression, regional identification, and scene graph generation.
- Writing Quality: ⭐⭐⭐⭐ The system is clear, illustrations are intuitive, and the two-stage pipeline is well-described.
- Value: ⭐⭐⭐⭐ The data engine paradigm is generalizable, and the 3B model outperforming 13B models proves that data quality is more critical than scale.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Multimodal 3D Genome Pre-training](../../NeurIPS2025/computational_biology/multimodal_3d_genome_pre-training.md)
- [\[ICCV 2025\] MolParser: End-to-end Visual Recognition of Molecule Structures in the Wild](../../ICCV2025/computational_biology/molparser_end-to-end_visual_recognition_of_molecule_structures_in_the_wild.md)
- [\[ICLR 2026\] SAIR: Enabling Deep Learning for Protein-Ligand Interactions with a Synthetic Structural Dataset](../../ICLR2026/computational_biology/sair_enabling_deep_learning_for_protein-ligand_interactions_with_a_synthetic_str.md)
- [\[ICLR 2026\] Learning Brain Representation with Hierarchical Visual Embeddings](../../ICLR2026/computational_biology/learning_brain_representation_with_hierarchical_visual_embeddings.md)
- [\[NeurIPS 2025\] MEIcoder: Decoding Visual Stimuli from Neural Activity by Leveraging Most Exciting Inputs](../../NeurIPS2025/computational_biology/meicoder_decoding_visual_stimuli_from_neural_activity_by_leveraging_most_excitin.md)

</div>

<!-- RELATED:END -->
