---
title: >-
  [Paper Note] SemGrasp: Semantic Grasp Generation via Language Aligned Discretization
description: >-
  [ECCV 2024][Robotics][Semantic Grasp Generation] This work proposes SemGrasp, which designs a hierarchical VQ-VAE to discretize grasp poses into three semantic tokens ("orientation-manner-refinement"). It then fine-tunes a Multimodal Large Language Model (MLLM) to align objects, grasps, and language within a unified semantic space, enabling the generation of physically plausible and semantically consistent human grasp poses from natural language instructions.
tags:
  - "ECCV 2024"
  - "Robotics"
  - "Semantic Grasp Generation"
  - "VQ-VAE Discretization"
  - "Multimodal Large Language Model"
  - "Hand-Object Interaction"
  - "Grasp Representation Learning"
date: 2026-05-08
content_hash: 5ab3ae7ca39d906b
---

# SemGrasp: Semantic Grasp Generation via Language Aligned Discretization

**Conference**: ECCV 2024  
**arXiv**: [2404.03590](https://arxiv.org/abs/2404.03590)  
**Code**: [https://kailinli.github.io/SemGrasp](https://kailinli.github.io/SemGrasp)  
**Area**: Robotics / Semantic Grasp Generation  
**Keywords**: Semantic Grasp Generation, VQ-VAE Discretization, Multimodal Large Language Model, Hand-Object Interaction, Grasp Representation Learning

## TL;DR

This work proposes SemGrasp, which designs a hierarchical VQ-VAE to discretize grasp poses into three semantic tokens ("orientation-manner-refinement"). It then fine-tunes a Multimodal Large Language Model (MLLM) to align objects, grasps, and language within a unified semantic space, enabling the generation of physically plausible and semantically consistent human grasp poses from natural language instructions.

## Background & Motivation

**Background**: Generating human-like grasp poses is highly valuable in applications such as AR/VR and robotic manipulation. Existing grasp generation methods primarily rely on the geometric information of objects (point clouds/meshes) and utilize MANO model parameters, contact regions, or implicit representations to describe grasp poses.

**Core Problem**: Grasp generation relying solely on geometric information has obvious limitations—grasping must consider not only the object shape but also the manipulation intent. For example, when grasping a cup, "avoiding hot water" and "preparing to unscrew the cap" require completely different grasp manners. However, grasp representations in existing methods struggle to embed semantic information, failing to integrate detailed language descriptions into the grasp generation process.

**Human Grasp Planning Inspiration**: The authors observe that humans follow a three-step strategy when planning grasps:

**Determining grasp orientation**: guided by object category and instruction semantics.

**Deciding grasp manner**: influenced by manipulation intent and object shape.

**Refining grasp pose**: ensuring physical plausibility based on geometric details and contact state of the object.

This observation inspires the design of a grasp representation that mimics human grasp planning, explicitly incorporates these three steps, and implicitly embeds semantic information.

**Lack of Datasets**: Existing grasp-language alignment datasets are extremely scarce, and existing labels only cover simple intents, which is insufficient for training semantic grasp generation models.

## Method

### Overall Architecture

SemGrasp consists of two core components: (1) **Grasp Discretization Module**—discretizing continuous grasp poses into three semantic tokens using a hierarchical VQ-VAE; (2) **Grasp-Aware Language Model**—an MLLM based on the LLaVA architecture that fuses object features, grasp tokens, and language descriptions in a unified semantic space to generate grasp poses from language instructions. The training data comes from the newly constructed CapGrasp dataset, which contains approximately 260k detailed descriptions and 50k diverse grasps.

### Key Designs

#### 1. **Hierarchical VQ-VAE Grasp Discretization**

**Mechanism**: Discretizing the grasp pose $\boldsymbol{G} = (\boldsymbol{T}, \boldsymbol{\theta}, \boldsymbol{\beta})$ into three inter-correlated tokens $\langle \texttt{o}, \texttt{m}, \texttt{r} \rangle$, representing orientation, manner, and refinement, respectively.

**Design Motivation**: Language is inherently discrete, and discretizing grasps allows for natural alignment with the semantic space. Furthermore, according to the Grasp Taxonomy, human grasps can be categorized into 33 discrete types. Discretization yields two advantages: (a) enhanced controllability and interpretability; (b) significantly reduced dimensionality of the grasp space, simplifying the learning process.

**Implementation Details**: A hierarchical VQ-VAE architecture is adopted, containing three levels of encoders $\mathcal{E}_i$, decoders $\mathcal{D}_i$, and codebooks $\mathcal{B}_i$ ($i \in \{1,2,3\}$), which progressively capture grasp information from low to high levels:

- **Orientation token** $\langle \texttt{o} \rangle$: captures the global transformation of the hand $\boldsymbol{T}$, i.e., $\hat{\boldsymbol{T}} = \mathcal{D}_1(\texttt{o}, \boldsymbol{O})$
- **Manner token** $\langle \texttt{m} \rangle$: captures hand pose parameters $\boldsymbol{\theta}, \boldsymbol{\beta}$ conditioned on the orientation, i.e., $\hat{\boldsymbol{\theta}}, \hat{\boldsymbol{\beta}} = \mathcal{D}_2(\texttt{o}, \texttt{m}, \boldsymbol{O})$
- **Refinement token** $\langle \texttt{r} \rangle$: predicts delta parameters $\Delta\boldsymbol{T}, \Delta\boldsymbol{\theta}, \Delta\boldsymbol{\beta}$ conditioned on both orientation and manner

The final reconstructed grasp is formulated as: $\hat{\boldsymbol{G}} = (\Delta\hat{\boldsymbol{T}} \cdot \hat{\boldsymbol{T}}, \Delta\hat{\boldsymbol{\theta}} + \hat{\boldsymbol{\theta}}, \Delta\hat{\boldsymbol{\beta}} + \hat{\boldsymbol{\beta}})$

The encoder maps the input to the codebook through nearest neighbor search: $\texttt{z} = \mathcal{E}(\boldsymbol{z}) = \text{argmin}_k \|\mathcal{N}_{\mathcal{E}}(\boldsymbol{z}) - \boldsymbol{b}_k\|_2$, where the codebook $\mathcal{B}$ contains $K=512$ entries with dimension $d_{\mathcal{B}}=256$.

#### 2. **Grasp-Aware MLLM**

**Mechanism**: Fine-tuning an MLLM based on Vicuna-7B to fuse discrete grasp tokens, object features, and language descriptions in a unified semantic space.

**Tri-modal Input**:
- **Grasp modality**: The frozen VQ-VAE encoder acts as a tokenizer, outputting three grasp tokens with special prefix and suffix tokens `<SG>` and `<EG>`.
- **Object modality**: PointBERT is used to extract point cloud features $f_{\boldsymbol{O}} \in \mathbb{R}^{513 \times 384}$, mapped to the 4096-dimensional language space via a linear projection layer $\mathcal{P}_{\boldsymbol{O}}$, along with an object size token `<OS>`.
- **Language modality**: Text is tokenized into 32K wordpieces via SentencePiece.

**Training Process**: Fine-tuning is performed using LoRA (rank=64, fine-tuning approximately 6% of parameters) in two stages:
1. **Multimodal alignment**: Training the model to predict grasp tokens from object features and language descriptions, updating the projection layer $\mathcal{P}_{\boldsymbol{O}}$ and embedding layers.
2. **Instruction fine-tuning**: Further fine-tuning on the grasp generation task and language output, freezing the projection layer to ensure stability.

#### 3. **CapGrasp Dataset**

**Mechanism**: Scaling existing hand-object interaction datasets (OakInk) using GPT-4-based automated annotation to construct a large-scale language-grasp alignment dataset.

**Three-level Annotation Hierarchy**:
- **Low-level annotation**: Computing contact status (contact relationships between fingers and object components) based on hand and object poses, with a threshold of 3mm.
- **High-level annotation**: Inferring manipulation intent, grasp strength, etc., using low-level info and GPT-4/GPT-4V.
- **Conversational annotation**: Constructing multi-turn grasp-language mixed conversations using GPT-4.

**Statistics**: Roughly 1.8k object models, 50k hand-object grasp pairs, with an average of 5 detailed descriptions and conversational annotations per pair.

### Loss & Training

**VQ-VAE Training Loss**:

$$\mathcal{L} = \mathcal{L}_{\text{rec}} + \mathcal{L}_{\text{emb}} + \mathcal{L}_{\text{com}}$$

- **Reconstruction Loss**: $\mathcal{L}_{\text{rec}} = \|\boldsymbol{H} - \mathcal{M}(\hat{\boldsymbol{G}})\|_2^2$, calculated in the hand vertex space.
- **Embedding Loss + Commitment Loss**: $\mathcal{L}_{\text{emb}} + \mathcal{L}_{\text{com}} = \|\text{sg}[\mathcal{N}_{\mathcal{E}}(\boldsymbol{z})] - \boldsymbol{b}_{\texttt{z}}\|_2^2 + \|\mathcal{N}_{\mathcal{E}}(\boldsymbol{z}) - \text{sg}[\boldsymbol{b}_{\texttt{z}}]\|_2^2$

**MLLM Training Loss**: Standard autoregressive NLL loss

$$\mathcal{L}_{\text{NLL}} = -\sum_i \log p(\hat{x}^i | \hat{x}^{<i}, x)$$

Training configuration: batch size 128, learning rate 5e-4 (alignment phase) / 3e-5 (fine-tuning phase), cosine annealing, 4×A100 GPUs, 20 epochs.

## Key Experimental Results

### Main Results: Reconstruction Quality of VQ-VAE Discrete Representation

| Method | MPVPE↓ | PD↓ | SIV↓ | SD mean↓ | SD std↓ |
|------|--------|-----|------|----------|---------|
| CapGrasp (GT) | - | 0.11 | 0.62 | 0.94 | 1.62 |
| GrabNet w/o refine | 18.14 | 0.76 | 5.42 | 1.75 | 2.61 |
| GrabNet w/ refine | 27.49 | 0.54 | 3.45 | 1.77 | 2.36 |
| Jiang et al. w/ TTA | 33.84 | 0.58 | 2.78 | 1.36 | 1.55 |
| **SemGrasp** | **14.97** | **0.46** | **2.72** | 2.14 | 2.37 |
| SemGrasp w/ TTA | 23.61 | **0.37** | **1.27** | 1.90 | 2.12 |

SemGrasp achieves an 18% improvement in MPVPE over GrabNet, and reaches SOTA in PD and SIV after adding TTA.

### Semantic Grasp Generation Results of MLLM

| Method | P-FID↓ | PD↓ | SIV↓ | GPT-4↑ | PS↑ |
|------|--------|-----|------|--------|-----|
| CapGrasp (GT) | - | 0.11 | 0.62 | 82.3 | 4.7 |
| BERT Baseline | 3.32 | 0.49 | 4.60 | 47.3 | 3.7 |
| **SemGrasp** | **2.28** | **0.48** | **4.24** | **74.5** | **4.6** |

SemGrasp gains a score of 74.5 (out of 100) on the GPT-4 semantic consistency metric, which is significantly better than the BERT baseline (47.3) and close to the CapGrasp ground truth (82.3).

### Ablation Study

| Configuration | MPVPE↓ | PD↓ | SIV↓ | Explanation |
|------|--------|-----|------|------|
| Single token | 29.95 | 0.66 | 5.14 | Compressing into a single codebook severely degenerates reconstruction accuracy |
| Two tokens <o,m> | 25.73 | 0.58 | 4.32 | Suboptimal performance without the refinement token |
| Three tokens <o,m,r> (ours) | **14.97** | **0.46** | **2.72** | Optimal configuration |
| Three tokens + r×2 | 15.37 | 0.50 | 2.98 | Multiple refinement tokens instead increase MLLM training complexity |
| Single VQ-VAE | 28.02 | 0.68 | 5.31 | Shared codebook struggle to capture complex representation |
| Without semantic assignment | 21.94 | 0.60 | 4.59 | Depriving tokens of semantic meanings causes performance drop |

### Key Findings

1. **Three-token hierarchical representation is optimal**: The semantic decomposition of orientation-manner-refinement improves MPVPE by about 50% compared to a single token, and the refinement token brings an extra 26% improvement.
2. **Discrete representations are controllable**: Fixing $\langle \texttt{o}, \texttt{m} \rangle$ can generate grasps with consistent orientation and manner across objects of different shapes, which is an interpretability that cVAE methods lack.
3. **Sensitivity to codebook size**: $K=256$ leads to non-convergence, $K=1024$ leads to underfitting, and $K=512$ is the optimal choice.
4. **Vicuna outperforms Llama**: Vicuna with instruction fine-tuning performs better in grasp tasks.

## Highlights & Insights

1. **Elegant Design Philosophy**: Inspired by the cognitive process of human grasp planning, the continuous high-dimensional grasp space is discretized into hierarchical "orientation $\rightarrow$ manner $\rightarrow$ refinement" tokens, which is both intuitive and easy to align with the language space.
2. **Extremely Low-Dimensional Grasp Representation**: Utilizing only three discrete tokens (each selected from 512 codebook entries) to express complex hand poses greatly reduces learning difficulty.
3. **Controllable and Interpretable**: Unlike continuous latent space methods like cVAE, discrete tokens have explicit semantic meanings, making the grasp generation process transparent and controllable.
4. **End-to-End Application Verification**: The utility of generated grasps is validated in two downstream tasks: D-grasp (AR/VR) and UniDexGrasp (robotics).

## Limitations & Future Work

1. **Limited to single-hand static grasping**: Dual-hand collaborative manipulation is not addressed, which requires a substantial amount of bi-manual motion capture data.
2. **Dynamic grasping requires RL assistance**: Generating dynamic grasp sequences requires additional RL policies like D-grasp, which is not end-to-end.
3. **Dataset reliance on GPT-4 annotations**: The high-level annotations in CapGrasp rely on GPT-4, which may introduce hallucinations. Although manually audited, there is still room for quality improvement.
4. **Limitations of evaluation metrics**: Semantic consistency scores evaluated using GPT-4 may not be fully objective.

## Related Work & Insights

- **GrabNet / Jiang et al.**: cVAE-based grasp generation methods that rely solely on geometric information, lacking semantic controllability.
- **LLaVA**: The source of inspiration for the MLLM architecture, which the authors extend to the 3D grasp domain.
- **MotionGPT**: The approach of tokenizing motion sequences and unifying them with language is conceptually similar to this work.
- **Inspirations**: The discretization + MLLM paradigm can be generalized to other 3D interaction tasks requiring semantic control (e.g., whole-body motion, object manipulation).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The design of hierarchical VQ-VAE grasp discretization is novel and elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive evaluation metrics encompassing reconstruction, generation, ablation, and downstream applications.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, natural motivation, and rich figures and tables.
- **Value**: ⭐⭐⭐⭐ — The first to introduce language instructions into fine-grained human grasp generation, paving a new direction for semantic grasping.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Decomposed Vector-Quantized Variational Autoencoder for Human Grasp Generation](decomposed_vector-quantized_variational_autoencoder_for_human_grasp_generation.md)
- [\[CVPR 2026\] DextER: Language-driven Dexterous Grasp Generation with Embodied Reasoning](../../CVPR2026/robotics/dexter_language-driven_dexterous_grasp_generation_with_embodied_reasoning.md)
- [\[ECCV 2024\] An Economic Framework for 6-DoF Grasp Detection](an_economic_framework_for_6-dof_grasp_detection.md)
- [\[AAAI 2026\] SemanticVLA: Semantic-Aligned Sparsification and Enhancement for Efficient Robotic Manipulation](../../AAAI2026/robotics/semanticvla_semantic-aligned_sparsification_and_enhancement_for_efficient_roboti.md)
- [\[ECCV 2024\] Prioritized Semantic Learning for Zero-shot Instance Navigation](prioritized_semantic_learning_for_zero-shot_instance_navigation.md)

</div>

<!-- RELATED:END -->
