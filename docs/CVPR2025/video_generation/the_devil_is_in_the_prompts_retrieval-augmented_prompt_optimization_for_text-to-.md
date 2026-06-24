---
title: >-
  [Paper Note] The Devil is in the Prompts: Retrieval-Augmented Prompt Optimization for Text-to-Video Generation
description: >-
  [CVPR 2025][Video Generation][Text-to-Video] RAPO proposes a retrieval-augmented prompt optimization framework. By retrieving relevant modifiers from a relation graph constructed from training data, fine-tuning an LLM to reconstruct sentence structures, and utilizing a discriminator to select the optimal prompt, it converts short user prompts into optimized prompts aligned with the training data distribution. This improves multi-object generation on VBench from 37.71% to 64.8…
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Text-to-Video"
  - "Prompt Optimization"
  - "Retrieval-Augmented"
  - "Relation Graph"
  - "LLM Fine-tuning"
date: 2026-05-08
content_hash: 19ccd7c27ac0b1da
---

# The Devil is in the Prompts: Retrieval-Augmented Prompt Optimization for Text-to-Video Generation

**Conference**: CVPR 2025  
**arXiv**: [2504.11739](https://arxiv.org/abs/2504.11739)  
**Code**: None (GitHub mentioned on project homepage)  
**Area**: Image Generation / Video Generation  
**Keywords**: Text-to-Video, Prompt Optimization, Retrieval-Augmented, Relation Graph, LLM Fine-tuning

## TL;DR

RAPO proposes a retrieval-augmented prompt optimization framework. By retrieving relevant modifiers from a relation graph constructed from training data, fine-tuning an LLM to reconstruct sentence structures, and utilizing a discriminator to select the optimal prompt, it converts short user prompts into optimized prompts aligned with the training data distribution. This improves multi-object generation on VBench from 37.71% to 64.86%.

## Background & Motivation

**Background**: Text-to-Video (T2V) generation models (such as LaVie and Latte) have achieved significant progress training on large-scale datasets. However, research reveals that utilizing detailed, long-form prompts generally produces higher-quality videos than brief descriptions provided by users.

**Limitations of Prior Work**: User-provided prompts are typically short and lack necessary details. Directly expanding prompts using GPT-4 or Open-Sora may add more descriptions, but often introduces convoluted, complex vocabulary and sentence structures that mismatch the training data. This misleads the model instead, leading to degraded generation quality that can be even worse than the original short prompts. Existing text-to-image (T2I) prompt optimization methods show limited effectiveness for video generation, particularly offering minimal improvement in the temporal dimension (such as motion smoothness and temporal consistency).

**Key Challenge**: T2V models are extremely sensitive to input prompts, but a "good prompt" is model-specific—it must align with the vocabulary and sentence structure distributions of the training data. Users do not know the characteristics of the training data, and prompts expanded by general LLMs often fail to match.

**Goal**: To design a systematic prompt optimization framework that retains user semantics while closely aligning the optimized prompts with the vocabulary, length, and format distributions of the T2V training data.

**Key Insight**: Systematic analysis of the training data (Vimeo25M) indicates that video quality is highly correlated with the choice of verb-noun phrases and sentence structures in the prompts. Therefore, statistical patterns must be extracted from the training data to guide the optimization process.

**Core Idea**: To construct a relation graph of the training data (a scene-to-modifier graph structure) to append relevant modifiers to user prompts via retrieval augmentation, leverage a fine-tuned LLM to reconstruct sentence structures into the format of the training data, and finally employ a discriminator to select the optimal prompt from two optimization paths.

## Method

### Overall Architecture

RAPO comprises three main modules: (1) Vocabulary Enhancement Module: retrieves modifiers from the relation graph and fuses them via an LLM; (2) Sentence Reconstruction Module: reconstructs the enhanced prompts into the training data format using a fine-tuned LLM; (3) Prompt Selection Module: uses a discriminator LLM to select the better prompt from the outputs of two paths—Path A: vocabulary enhancement followed by sentence reconstruction, and Path B: direct rewriting using a general LLM.

### Key Designs

1. **Relation Graph Construction and Retrieval**:

    - **Function**: Systematically extracts structured scene-modifier knowledge from the training data to provide high-quality enhancement candidates for user prompts.
    - **Mechanism**: Uses the Mistral LLM to extract scenes and corresponding modifiers (subjects, actions, and atmosphere descriptions) from approximately 2.1 million valid sentences in the Vimeo25M training dataset. Scenes serve as core nodes while modifiers act as child nodes to construct the relation graph $\mathcal{G}$. For a new user prompt, features are extracted using a pre-trained sentence encoder (all-MiniLM-L6-v2) to first retrieve the top-k related scenes via cosine similarity, and then retrieve the top-k associated modifiers within those scenes.
    - **Design Motivation**: Directly relying on an LLM to generate modifiers easily yields vocabulary that mismatches the training data; retrieving directly from the training data itself ensures that the enhanced content originates from the distribution the model has already "seen."

2. **Iterative Fusion + Sentence Reconstruction**:

    - **Function**: Integrates the retrieved modifiers into the prompt step-by-step and reconstructs them into the unified format of the training data.
    - **Mechanism**: Vocabulary enhancement adopts an iterative fusion mechanism $x_i^{m+1} = f(x_i^m, p_i^m)$, where a frozen LLM $\mathcal{L}$ (GPT-4) is utilized at each step to reasonably integrate one modifier into the current prompt, progressively enriching the semantics. Once fusion is complete, an instruction-tuned "reconstruction LLM" $L_r$ (fine-tuned on LLaMA 3.1 via LoRA) reconstructs the enhanced prompt to match the length and format of the training data. The training data for $L_r$ consists of approximately 86k pairs of prompts (having different formats but identical semantics), with the objective of making the output sentence structure distribution match the training data.
    - **Design Motivation**: Integrating all modifiers simultaneously often leads to unnatural sentences; iterative fusion guarantees semantic coherence at each step. A separate sentence reconstruction step ensures that the final prompt aligns with the training distribution in length and format—experiments demonstrate that the prompt length distribution optimized by RAPO is the closest to the training set.

3. **Prompt Discriminator Selection**:

    - **Function**: Automatically selects the prompt better suited for T2V generation between the outputs of two optimization paths (the retrieval-augmented path vs. the direct LLM rewriting path).
    - **Mechanism**: Fine-tunes a discriminator LLM $\mathcal{L}_d$ (LLaMA 3.1 LoRA, 3 epochs) that takes the original prompt $x_i$ and two candidates $x_r, x_n$ as inputs to output a selection label. The training data is constructed by generating actual videos and utilizing automatic evaluation metrics (which automatically select relevant evaluation dimensions based on the prompt content) to determine which candidate is better. The training set contains 7k texts generated by Mistral, covering all dimensions of VBench.
    - **Design Motivation**: A single path might fail on certain prompts; the dual-path + discriminator design provides redundancy and a selection mechanism, enhancing overall robustness.

### Loss & Training

Both the reconstruction model $L_r$ and the discriminator $L_d$ are fine-tuned using LLaMA 3.1 with LoRA instruction tuning. $L_r$ is trained for 8 epochs, and $L_d$ is trained for 3 epochs, with a batch size of 32, LoRA rank 64, and on a single A100 GPU. The relation graph construction is a one-time offline process.

## Key Experimental Results

### Main Results

VBench Evaluation (LaVie Model):

| Method | Total Score | Imaging Quality | Human Action | Object Class | Multi-Object | Spatial Relationship |
|------|------|---------|---------|---------|--------|---------|
| LaVie Original | 80.89% | 69.00% | 95.80% | 92.09% | 37.71% | 37.27% |
| LaVie-GPT4 | 79.69% | 70.27% | 83.80% | 88.73% | 36.23% | 50.55% |
| LaVie-Open-sora | 79.75% | 70.42% | 87.00% | 91.29% | 36.52% | 54.37% |
| **LaVie-RAPO** | **82.38%** | **71.40%** | **96.80%** | **96.91%** | **64.86%** | **59.15%** |

EvalCrafter Evaluation:

| Method | Total Score | Text-Video Alignment | Visual Quality |
|------|------|-------------|---------|
| LaVie-RAPO | **256** | **74.38** | **66.62** |
| LaVie-Open-sora | 251 | 71.38 | 65.26 |
| LaVie Original | 248 | 69.60 | 64.81 |

### Ablation Study

| Configuration | VBench Total Score |
|------|-----------|
| Vocabulary Enhancement only | 80.37% |
| Sentence Reconstruction only | 79.75% |
| Vocabulary Enhancement + Sentence Reconstruction | 81.58% |
| Sentence Reconstruction + Prompt Selection | 81.75% |
| Vocabulary Enhancement + Prompt Selection | 80.60% |
| **Full RAPO** | **82.38%** |

Robustness across different LLMs (Choice of $\mathcal{L}$):

| LLM | VBench Total Score |
|-----|-----------|
| GPT-4 | 82.38% |
| Mistral | 82.25% |
| LLaMA | 82.10% |

### Key Findings

- The most significant improvement occurs in multi-object generation: LaVie improves from 37.71% to 64.86% (+27.15pp) and Latte from 29.55% to 52.78% (+23.23pp), demonstrating that descriptions matching the training data can substantially improve multi-subject scenes.
- Prompt optimization from GPT-4 and Open-Sora occasionally degrades performance (e.g., GPT-4 reduces LaVie's human action score from 95.80% to 83.80%), as verbose and complex descriptions tend to "confuse" the model.
- The prompt length distribution optimized by RAPO is the closest to the training set distribution—this is the core reason for the performance gain.
- Each of the three modules contributes to improvements in different dimensions, with their combination yielding the best performance (synergistic effect).
- RAPO is insensitive to the choice of LLM (variation among GPT-4, Mistral, and LLaMA is only ~0.3%), indicating that the design of the framework itself is key.
- Spatial position descriptions are particularly crucial for multi-object generation—attention map visualizations show that adding "relative spatial position" descriptions allows for clearer object separation.

## Highlights & Insights

- **The counter-intuitive concept of "adapting prompts to the model" instead of "adapting the model to prompts"**: While most works attempt to make models better understand user prompts, this paper does the opposite—it analyzes statistical patterns of the training data to rewrite user prompts into the model's most "comfortable" format. This approach is simple yet profound.
- **Relation graph as structured memory of training data**: Compressing 2.1 million training prompts into a scene-to-modifier graph structure enables highly efficient retrieval. This method of extracting knowledge from training data can be generalized to any generative task.
- **Massive improvement in multi-object generation**: A 27pp boost indicates that failures in multi-object scenarios are largely not due to model capability limitations, but rather prompt engineering issues. The key lies in adding explicit spatial relationship descriptions.

## Limitations & Future Work

- The relation graph and reconstruction model must be reconstructed for each T2V model's specific training data, incurring migration costs when generalizing to new models.
- Validation was only conducted on two relatively early T2V models, LaVie and Latte; the performance on state-of-the-art models like Sora or Kling remains unexplored.
- The discriminator training set size is only 7k and relies on automatic evaluation metrics for preference annotation, which may introduce bias.
- Prompt optimization acts as an extra step during inference, increasing latency due to multiple LLM calls.

## Related Work & Insights

- **vs. GPT-4 Direct Rewriting**: Although GPT-4 generates more informative prompts, they are excessively long and use complex vocabulary that mismatches the training distribution, which ultimately leads to degraded generation quality.
- **vs. Open-Sora Prompt Refiner**: While fine-tuned on LLaMA 3.1, it lacks fine-grained guidance derived from training data statistics, resulting in limited effectiveness.
- **vs. Hao et al. (NeurIPS)**: Optimizes prompts for aesthetic scores using reinforcement learning, but it targets T2I and does not consider aligning prompts with the training data distribution.

## Rating

- Novelty: ⭐⭐⭐⭐ Constructing a relation graph from training data for retrieval-augmented prompt optimization is novel, and the dual-path + discriminator design shows engineering ingenuity.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on three benchmarks (VBench, EvalCrafter, and T2V-CompBench) with comprehensive ablation studies, though only validated on two older models.
- Writing Quality: ⭐⭐⭐ The overall organization is reasonable, but some descriptions are verbose and the mathematical formulation is slightly excessive.
- Value: ⭐⭐⭐⭐ Reveals the significance of aligning prompts with the training distribution, providing excellent guidance for practical prompt engineering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Optical-Flow Guided Prompt Optimization for Coherent Video Generation](optical-flow_guided_prompt_optimization_for_coherent_video_generation.md)
- [\[ICCV 2025\] VPO: Aligning Text-to-Video Generation Models with Prompt Optimization](../../ICCV2025/video_generation/vpo_aligning_text-to-video_generation_models_with_prompt_optimization.md)
- [\[CVPR 2025\] Video-ColBERT: Contextualized Late Interaction for Text-to-Video Retrieval](video-colbert_contextualized_late_interaction_for_text-to-video_retrieval.md)
- [\[CVPR 2025\] StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text](streamingt2v_consistent_dynamic_and_extendable_long_video_generation_from_text.md)
- [\[CVPR 2025\] VIRES: Video Instance Repainting via Sketch and Text Guided Generation](vires_video_instance_repainting_via_sketch_and_text_guided_generation.md)

</div>

<!-- RELATED:END -->
