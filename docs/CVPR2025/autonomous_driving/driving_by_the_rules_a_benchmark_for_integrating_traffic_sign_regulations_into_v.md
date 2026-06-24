---
title: >-
  [Paper Note] Driving by the Rules: A Benchmark for Integrating Traffic Sign Regulations into Vectorized HD Map
description: >-
  [CVPR 2025][Autonomous Driving][Traffic Sign Regulations] This paper defines for the first time the task of integrating traffic sign regulations into online vectorized high-definition (HD) maps. It constructs the MapDR dataset, which contains over 10,000 video clips and more than 18,000 lane-level regulations, and proposes two baseline solutions: a modular approach (VLE-MEE) and an end-to-end approach (RuleVLM), with RuleVLM achieving a 64.2% overall F1 score.
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Traffic Sign Regulations"
  - "High-Definition Map"
  - "Dataset Benchmark"
  - "Vision-Language Models"
  - "Regulation-to-Lane Association Reasoning"
date: 2026-05-08
content_hash: 8514ed04e23c52e4
---

# Driving by the Rules: A Benchmark for Integrating Traffic Sign Regulations into Vectorized HD Map

**Conference**: CVPR 2025  
**arXiv**: [2410.23780](https://arxiv.org/abs/2410.23780)  
**Code**: [MIV-XJTU/MapDR](https://github.com/MIV-XJTU/MapDR)  
**Area**: Autonomous Driving  
**Keywords**: Traffic Sign Regulations, High-Definition Map, Dataset Benchmark, Vision-Language Models, Regulation-to-Lane Association Reasoning

## TL;DR
This paper defines for the first time the task of integrating traffic sign regulations into online vectorized high-definition (HD) maps. It constructs the MapDR dataset, which contains over 10,000 video clips and more than 18,000 lane-level regulations, and proposes two baseline solutions: a modular approach (VLE-MEE) and an end-to-end approach (RuleVLM), with RuleVLM achieving a 64.2% overall F1 score.

## Background & Motivation

High-definition (HD) maps typically contain three layers: the geometric layer (vectorized lane lines, boundaries, etc.), the connectivity (topology) layer, and the traffic regulation layer (lane-level regulations such as speed limits and bus-only lanes). Existing online HD map reconstruction methods (e.g., MapTR, TopoMLP) mainly focus on the geometric and connectivity layers, completely omitting the traffic regulation layer. Consequently, autonomous driving systems must still rely on offline maps to obtain traffic regulation information, which contradicts the trend toward online mapping.

Although OpenLane-V2 attempts to associate traffic signs with lanes, it only considers direction signs, and the annotations are limited to sign categories, lacking structured regulation descriptions that conform to HD map standards. Traffic signs serve as the "visual language" on roads. Extracting structured regulations from signs and associating them with specific lanes is a complex multimodal task involving visual understanding, semantic reasoning, and spatial reasoning.

The core contributions of this paper are defining this new task, constructing the first dedicated dataset (MapDR), and providing modular and end-to-end baseline solutions.

## Method

### Overall Architecture
The task is formulated as follows: given an image sequence $X$ and lane centerlines $L$, output a bipartite graph $G = (R \cup L, E)$, where $R$ is the set of structured regulations extracted from traffic signs, and $E$ is the correspondence matrix between regulations and lanes. The task can fit into two subtasks: (1) regulation extraction, and (2) regulation-to-lane correspondence reasoning. This paper proposes a modular approach (VLE+MEE in series) and an end-to-end approach (RuleVLM).

### Key Designs

1. **MapDR Dataset**:

    - Scale: 10,000+ video clips, 400,000+ front-view images, 18,000+ lane-level regulations.
    - Data sources: Complex traffic scenarios from three major Chinese cities: Beijing, Shanghai, and Guangzhou.
    - Each video clip center-focuses on traffic signs, covering a 100m $\times$ 100m area.
    - Each clip consists of 30–60 frames, with one frame sampled every 2 meters.
    - Resolution: 1920 $\times$ 1240, with provided camera intrinsic parameters and poses.
    - Regulation annotation format: Each regulation contains key-value pairs with 8 predefined attributes following HD map specifications.
    - Provides vectorized local maps (3D point lists for divisions, boundaries, centerlines, crosswalks, etc.).
    - The data follows a natural long-tail distribution: abundant bus lanes and directional lanes, but sparse tidal lanes.
    - Privacy protection: License plates and faces in all images have been anonymized.

2. **Modular Approach: VLE + MEE**:

    - **VLE (Vision-Language Encoder)**: Used for regulation extraction.
        - Vision Encoder: ViT-b16; Text Encoder: $L = 6$-layer Transformer.
        - Two-stage pipeline: Clustering symbols and texts in OCR results first (via cosine similarity of [STC] tokens), then extracting structured regulations.
        - Introduces the [CLS] token to represent the entire regulation, and the [STC] token for sentence-level representation.
        - Utilizes inter-instance and intra-instance attention mechanisms to enhance interaction.
        - Leverages text layout positional encodings to capture spatial semantics.
    - **MEE (Map Element Encoder)**: Used for regulation-to-lane correspondence reasoning.
        - $M = 2$-layer Transformer encoder + $N = 2$-layer cross-attention fusion layers.
        - Analogs vectorized point sequences to word sequences in sentences.
        - The [VEC] token represents the fixed-length features of each vector.
        - Introduces type embeddings (distinguishing division lines, centerlines, etc.) and instance embeddings (distinguishing different vector instances).
        - Uses inter/intra-instance attention mechanisms to capture relationships between vectors.
        - Uses a binary classification head on each [VEC] token to determine if the lane corresponds to the current regulation.

3. **End-to-End Approach: RuleVLM**:

    - Based on Qwen-VL (9.6B), fine-tuned with LoRA.
    - Comparison of three vector encoding methods:
        - TextPrompt: Encodes centerline coordinates as text inputs to the LLM (worst performance due to excessively long sequences).
        - VisualPrompt: Visualizes centerlines on PV images as visual prompts (good regulation extraction but subpar overall performance).
        - RuleVLM: Uses MEE to independently encode vectorized map information and aligns it with the LLM via an adapter (best performance).
    - Outputs serialized JSON format regulations, parsed and restored to structured data via a JSON decoder.
    - Randomly shuffles centerline ordering during training to prevent overfitting.

### Loss & Training
- VLE regulation extraction: Contrastive loss (clustering phase) + Multi-head classification loss (comprehension phase).
- MEE correspondence reasoning: Binary cross-entropy (BCE) loss.
- RuleVLM: Standard next-token prediction + LoRA fine-tuning.
- VLE is trained for 50 epochs; MEE is trained for 120 epochs.
- VLE is initialized with DeiT and BERT pre-trained weights; MEE is trained from scratch.
- Input images are resized to 256 $\times$ 256, feature dimension is 768, with 12 attention heads.

## Key Experimental Results

### Main Results

| Method | Type | P_RE | R_RE | P_CR | R_CR | F1 (Overall) |
|------|------|------|------|------|------|------|
| Heuristic | Modular | 18.01 | 11.51 | 33.05 | 17.99 | 0.035 |
| ALBEF-BERT | Modular | 75.78 | 57.56 | 4.14 | 17.25 | 0.003 |
| VLE-MEE | Modular | 76.67 | 74.54 | 78.05 | 82.16 | 0.653 |
| Qwen-VL (TextPrompt) | End-to-End | 42.21 | 41.09 | - | - | 0.083 |
| Qwen-VL (VisualPrompt) | End-to-End | 89.29 | 89.50 | - | - | 0.392 |
| RuleVLM | End-to-End | 89.28 | 89.44 | - | - | 0.642 |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| VLE w/o attention mechanism | R_RE: 57.56% | Attention mechanism is key; adding it improves recall to 71.75% |
| VLE + attention + layout | R_RE: 74.54% | Text layout brings marginal improvement |
| MEE w/o attention | P_CR: 4.14% | Virtually unusable |
| MEE + attention | P_CR: 68.91% | Attention mechanism is the decisive factor |
| MEE + attention + type embedding | P_CR: 78.05% | Vector type information significantly improves correspondence reasoning |

### Key Findings
- Regulation extraction and correspondence reasoning are two highly distinct subtasks: ALBEF-BERT performs decently on regulation extraction but almost fails at correspondence reasoning.
- The attention mechanism in MEE is the decisive factor for correspondence reasoning; performance without it is near random.
- Type embeddings are crucial for MEE, showing that vector types (e.g., division lines vs. centerlines) encode essential semantic information.
- In end-to-end approaches, textual coordinates input performs worst (as LLMs struggle with spatial reasoning over raw text coordinates), followed by visual prompts, while MEE vector encoding is the most optimal.
- RuleVLM (F1 = 0.642) performs comparably to the modular VLE-MEE (F1 = 0.653), demonstrating the immense potential of end-to-end pipelines.

## Highlights & Insights
- Fills a gap in HD mapping research related to the traffic regulation layer, offering a clear definition of the task and a complete evaluation metric suite.
- The MapDR dataset is substantial in scale and annotation quality across diverse scenarios, serving as crucial infrastructure in this domain.
- Defining regulations in a structured {key:value} format following industrial HD map standards ensures strong practical utility.
- MEE elegantly treats vector encodings analogous to language model tokens.
- Comparative experiments between modular and end-to-end paradigms provide valuable insights, revealing the pros and cons of both pipelines.
- The long-tail distribution of the data reflects real-world complexities, presenting standard challenges.

## Limitations & Future Work
- The dataset is limited to traffic signs in three Chinese cities, lacking internationalized data (signage varies significantly across countries).
- Current pipelines assume OCR results (for regulation extraction) or vectorized maps (for correspondence reasoning) are pre-given, which requires integration with real-time OCR and online mapping systems in fully automated pipelines.
- The final overall F1 of around 64% suggests that there is still significant room for improvement.
- Performance under nocturnal or adverse weather conditions remains un-evaluated.
- RuleVLM is based on Qwen-VL (9.6B), whose inference latency might not meet real-time operational constraints.
- Sparsity of rare regulations like tidal lanes might degrade performance on long-tailed categories.
- The pipeline is not yet integrated or validated with autonomous driving planning systems, leaving the actual downstream impact of the traffic regulation layer on planning performance unverified.

## Related Work & Insights
- Comparison with OpenLane-V2: OpenLane-V2 only handles single-label classification of direction signs, whereas MapDR supports multi-attribute, structured regulation descriptions.
- Relationship with LLM-based driving benchmarks like MAPLM: MAPLM focuses on end-to-end planning, whereas MapDR focuses on precise regulation extraction and reasoning.
- VLE is designed with inspiration from vision-language models such as ALBEF but incorporates special adaptations for multi-text, multi-regulation characteristics of traffic signs.
- The vector encoding concepts in MEE can be applied to other tasks involving vectorized geographic features.
- RuleVLM demonstrates an effective way to combine structured geometric inputs (vectors) with LLMs, providing insights for multimodal LLM architectures.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First to define the task of integrating traffic regulations into online HD maps, presenting entirely new contributions in benchmarks and baselines.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive comparison between modular and end-to-end strategies, but lacks cross-domain evaluation and real-world system verification.
- **Writing Quality**: ⭐⭐⭐⭐ Clear and detailed task definitions and dataset descriptions, though the methodology section is slightly complex in structure.
- **Value**: ⭐⭐⭐⭐⭐ Fills a crucial research gap; the dataset is bound to drive research in traffic rule comprehension and HD map completeness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MapGCLR: Geospatial Contrastive Learning of Representations for Online Vectorized HD Map Construction](mapgclr_geospatial_contrastive_learning_of_representations_for_online_vectorized.md)
- [\[CVPR 2025\] Uncertainty-Instructed Structure Injection for Generalizable HD Map Construction](uncertainty-instructed_structure_injection_for_generalizable_hd_map_construction.md)
- [\[ECCV 2024\] Stream Query Denoising for Vectorized HD-Map Construction](../../ECCV2024/autonomous_driving/stream_query_denoising_for_vectorized_hd-map_construction.md)
- [\[CVPR 2025\] Scenario Dreamer: Vectorized Latent Diffusion for Generating Driving Simulation Environments](scenario_dreamer_vectorized_latent_diffusion_for_generating_driving_simulation_e.md)
- [\[CVPR 2025\] T²SG: Traffic Topology Scene Graph for Topology Reasoning in Autonomous Driving](t2sg_traffic_topology_scene_graph_for_topology_reasoning_in_autonomous_driving.md)

</div>

<!-- RELATED:END -->
