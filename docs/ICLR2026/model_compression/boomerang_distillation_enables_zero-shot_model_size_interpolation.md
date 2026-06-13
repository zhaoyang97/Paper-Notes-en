---
title: >-
  [Paper Note] Boomerang Distillation Enables Zero-Shot Model Size Interpolation
description: >-
  [ICLR2026][Model Compression][Knowledge Distillation] This paper proposes the *Boomerang Distillation* paradigm: train a single small student model…
tags:
  - "ICLR2026"
  - "Model Compression"
  - "Knowledge Distillation"
  - "Zero-Shot Interpolation"
  - "Layer Pruning"
  - "Model Family"
date: 2026-05-08
content_hash: d91231013c8de755
---

# Boomerang Distillation Enables Zero-Shot Model Size Interpolation

**Conference**: ICLR2026
**arXiv**: [2510.05064](https://arxiv.org/abs/2510.05064)  
**Code**: [https://github.com/dcml-lab/boomerang-distillation](https://github.com/dcml-lab/boomerang-distillation)  
**Area**: Model Compression
**Keywords**: Knowledge Distillation, Model Compression, Zero-Shot Interpolation, Layer Pruning, Model Family

## TL;DR
This paper proposes the *Boomerang Distillation* paradigm: train a single small student model, then construct an entire family of intermediate-sized models at zero additional training cost by progressively grafting teacher transformer layer blocks back onto the student. The resulting models interpolate smoothly in performance between the student and teacher, matching or even surpassing independently distilled models of equivalent size.

## Background & Motivation

**Background**: LLM deployment spans an extremely wide range of hardware environments — from mobile devices to large-scale clusters. Model developers typically release families of models at varying scales (e.g., Qwen3-0.6B/1.7B/4B/8B/32B, Llama 3.2-1B/3B/8B), yet each size requires independent pretraining from scratch or an independent distillation run, causing training costs to scale linearly with the number of family members.

**Limitations of Prior Work**: Although knowledge distillation is more efficient than pretraining from scratch, each student still requires a full training pipeline (layer-pruning initialization → distillation → alignment). There is no training-free mechanism to expand a distilled model into fine-grained size options. Existing layer-pruning methods (ShortGPT, LaCo) exploit only teacher information; removing even a small number of layers causes catastrophic performance drops on classification tasks, and generation capability collapses rapidly toward zero.

**Key Challenge**: Real-world deployment demands fine-grained trade-offs in the size–performance space, yet training costs constrain model families to only a handful of coarse-grained options.

**Key Insight**: The authors observe that when a student model is initialized from the teacher via layer pruning (rather than random initialization), each student layer maintains strong representational alignment with the corresponding teacher layer block after distillation. This suggests that teacher layer blocks can be grafted back onto the student without any additional training — replacing the corresponding single layer and increasing model size without disrupting functionality — much like a boomerang that flies out (layer removal) and returns (layer grafting).

**Core Idea**: One distillation run + progressive grafting of teacher layer blocks = a fine-grained model family at zero additional training cost.

## Method

### Overall Architecture
The full pipeline consists of three stages. **Stage 1 (Student Initialization)**: $M$ layers are sampled at equal intervals from the teacher's $N$ layers, and copied together with the embedding layer and LM head to produce a layer-pruned initial student. **Stage 2 (Knowledge Distillation)**: The student is jointly trained on a text corpus using three losses — CE loss, KL divergence, and layer-wise cosine alignment — so that the student learns both the teacher's output distribution and maintains hidden-state alignment between each student layer and its corresponding teacher layer block. **Stage 3 (Student Patching)**: After training, with no additional optimization, consecutive teacher layer blocks are grafted one by one back onto the corresponding student positions (replacing a single layer with multiple layers), thereby constructing models of any intermediate size between $M$ and $N$ layers.

### Key Designs

1. **Layer-Pruning Initialization (Necessary Condition)**:

    - The teacher's $N$ transformer layers are partitioned into $M$ consecutive blocks $\mathcal{B} = (\mathbf{b}^{(1)}, \dots, \mathbf{b}^{(M)})$, where block $i$ contains layers $(\theta_T^{(\ell_i)}, \dots, \theta_T^{(\ell_{i+1}-1)})$.
    - Student layer $i$ is directly copied from the first layer of the corresponding block: $\theta_S^{(i)} = \theta_T^{(\ell_i)}$; the embedding layer and LM head are also copied directly.
    - **Design Motivation**: This initialization makes each student layer a natural "proxy" for the corresponding teacher block, providing structural compatibility for subsequent block grafting. Experiments confirm that a randomly initialized student, even after identical distillation training, yields almost no performance gain when teacher layers are grafted back.

2. **Layer-Wise Cosine Alignment Loss (Key to Stability)**:

    - Total loss: $\mathcal{L} = \mathcal{L}_{CE} + \lambda_{KL} \mathcal{L}_{KL} + \lambda_{cos} \sum_{i=1}^{M} \mathcal{L}_{cos}^{(i)}$
    - $\mathcal{L}_{cos}^{(i)}$ aligns the hidden state of student layer $i$ with the hidden state at the last layer of teacher block $\mathbf{b}^{(i)}$ via cosine distance.
    - **Design Motivation**: Only when a student layer's output is sufficiently close to the output of the corresponding teacher block can seamless replacement be achieved during patching. Without this loss, boomerang distillation still "works" (since teacher initialization already provides a base level of alignment), but interpolated models at boundary layers (first and last) exhibit noticeable performance fluctuations.
    - **Implementation Details**: $\mathcal{L}_{KL}$ computes KL divergence after scaling logits by temperature $\tau$; $\lambda_{KL}$ and $\lambda_{cos}$ are hyperparameters.

3. **Student Patching (Zero-Shot Model Construction)**:

    - The core operation is straightforward: replace student layer $i$ (a single layer $\theta_S^{(i)}$) with the full teacher block $\mathbf{b}^{(i)}$ (containing multiple layers), increasing the total layer count from $M$ to $M + |\mathbf{b}^{(i)}| - 1$.
    - Applying this operation progressively at different positions yields models of any intermediate size between $M$ layers (pure student) and $N$ layers (full teacher depth recovered).
    - **Embedding selection**: The embedding from the model contributing the first layer is used; the LM head is taken from the model contributing the last layer.
    - **Patching order**: Grafting from the last layer backward produces the best results. Llama is an exception — its first two layers exhibit low cosine similarity and are treated as special layers; they are retained in the student and patching proceeds from front to back.

### Training Details
The primary teacher is Qwen3-4B-Base (36 layers); the student has 2.7B parameters (18 layers obtained by sampling every other layer). Training data is deduplicated The Pile, trained for 2.1B tokens. Cross-model validation additionally uses Qwen3-8B-Base, Pythia-6.9B, and Llama-3.2-3B as teachers.

## Key Experimental Results

### Main Results: Boomerang Distillation vs. Baselines & Standard Distillation

| Method | Parameters | Classification Accuracy | Generation Accuracy | Extra Training |
|--------|-----------|------------------------|--------------------|--------------------|
| Qwen3-4B (Teacher) | 4.0B | Baseline (highest) | Baseline (highest) | — |
| Boomerang Distillation (Student) | 2.7B | Close to Teacher | Close to Teacher | 1 distillation run |
| Boomerang Interpolated Models | 2.7B–4.0B | Smooth interpolation ✅ | Smooth interpolation ✅ | 0 (zero-shot) |
| Standard Distillation (per-size) | 2.7B–4.0B | Small sizes ≈ Boomerang; large sizes inferior ❌ | Similar trend | 1 distillation run per size |
| Naive Layer Pruning | <4.0B | Sharp drop below 4B ❌ | Rapid collapse to ~0 ❌ | 0 |
| Random-Init Distillation + Patching | 2.7B–4.0B | Almost no gain ❌ | Almost no gain ❌ | 1 distillation run |
| Pythia-2.8B (pretrained) | 2.8B | Comparable | Comparable | Full pretraining |
| Llama-3.2-3B (pretrained) | 3.0B | Comparable | Comparable | Full pretraining |

**Key Finding**: Larger interpolated models actually *outperform* independently distilled models of the same size. The reason is that the distillation corpus (The Pile) is of lower quality than Qwen3's original pretraining data; independently distilling more layers on this lower-quality corpus causes catastrophic forgetting, whereas boomerang distillation preserves original knowledge by grafting back the teacher's original weights. This effect disappears when Pythia is used as the teacher (since Pythia was itself trained on The Pile).

### Ablation Study: Loss Function Combinations

| Loss Combination | Perplexity (WikiText) | Classification Accuracy | Boundary-Layer Stability |
|-----------------|----------------------|------------------------|--------------------------|
| CE only | Higher | Functional; boomerang effect present | Large fluctuations at first/last layers ❌ |
| CE + KL | Slightly lower | Slightly better than CE only | Fluctuations remain |
| CE + layer-wise cos | Lower | Modest improvement | Noticeably more stable ✅ |
| **CE + KL + layer-wise cos (full)** | **Lowest** | **Best** | **Most stable ✅** |

**Key Findings**: (1) Even with CE loss alone, boomerang distillation emerges — confirming that teacher-weight initialization is the core necessary condition. (2) The primary contribution of the layer-wise cosine loss is not improving mean performance but *stabilizing boundary layers* — the first and last layers correspond to the most extreme interpolated sizes and are most prone to failure when alignment is insufficient. (3) The full loss yields a significant advantage in perplexity.

### Comparison with Layer-Pruning Methods
Boomerang distillation substantially outperforms ShortGPT and LaCo at **all intermediate sizes**. The gap is especially pronounced on generation tasks: removing even a few layers causes ShortGPT/LaCo generation accuracy to collapse toward zero (ShortGPT's own paper acknowledges error accumulation across layers), while boomerang distillation's smaller models maintain high generation quality and exhibit smooth rather than cliff-like transitions in classification performance.

### Cross-Model-Family and Open-Source Model Validation
- The boomerang distillation phenomenon is observed with Qwen3-8B, Pythia-6.9B, and Llama-3.2-3B as teachers, demonstrating that this is a **general phenomenon** in LLM distillation.
- The phenomenon also exists in existing models — DistilBERT ↔ BERT and DistilGPT2 ↔ GPT2: without any additional training, grafting BERT layers back onto DistilBERT yields intermediate models that interpolate smoothly. This is the first demonstration of zero-shot size interpolation capability between these classical models.
- **Llama special case**: The first two layers exhibit low cosine similarity and require retention in the student; patching proceeds in the forward direction.

### Additional Ablations
- **More aggressive layer pruning (smaller student)**: Boomerang distillation functions as long as the student achieves non-trivial performance on the target task.
- **Training token budget**: Increasing the training budget improves interpolated model performance, but smooth interpolation already emerges with 2.1B tokens.

## Highlights & Insights
- **"Train once, obtain unlimited sizes" minimal paradigm**: The method requires no routers, no elastic architectures, and no multiple training runs — only a standard distillation pipeline plus a simple layer-replacement operation, making it highly practical.
- **Boomerang distillation reveals a previously overlooked byproduct of distillation**: Layer-pruning initialization combined with distillation training not only enables the student to learn a good output distribution but also implicitly maintains representational compatibility between each student layer and the corresponding teacher block. This compatibility had not previously been exploited.
- **Inverting catastrophic forgetting**: When the distillation corpus is of lower quality than the original pretraining data, independently distilling more layers leads to greater forgetting of original knowledge, whereas boomerang distillation is naturally immune by grafting back teacher weights — larger interpolated sizes are actually stronger. This offers a practical route to constructing model families when original pretraining data is unavailable.

## Limitations & Future Work
- **Requires identical hidden dimensions between teacher and student**: If the student undergoes neuron pruning (e.g., Minitron), mismatched hidden dimensions prevent teacher-layer grafting, limiting compatibility with certain industrial distillation pipelines (e.g., Minitron with combined layer and neuron pruning).
- **Requires retaining both teacher and student weights**: The patching stage requires access to teacher layer weights. Although only the final interpolated model is needed at inference time, storage overhead remains higher than fully independent small models.
- **Layer-wise cosine loss increases training-time memory**: Computing hidden states for all layers of both teacher and student simultaneously raises memory consumption during training.
- **Patching order is model-dependent**: The optimal direction (back-to-front vs. front-to-back) varies by model (Llama requires special handling), and no universally optimal strategy has been established.
- **GPT2 interpolation is less smooth than BERT**: Effectiveness may vary with model architecture and distillation setup.
- **Combination with parameter-efficient methods (e.g., LoRA) remains unexplored.**

## Related Work & Insights
- **vs. Standard Knowledge Distillation (Hinton et al.)**: Standard distillation requires an independent training run per size; boomerang distillation covers all intermediate sizes from a single student training run.
- **vs. ShortGPT / LaCo**: Layer pruning exploits only teacher information and causes performance collapse after removing a few layers; boomerang distillation leverages both student and teacher information with substantially superior performance.
- **vs. Model Interpolation (Wortsman et al.)**: Traditional model interpolation operates between weights of models of the **same size**; boomerang distillation interpolates between teacher and student of **different sizes**.
- **vs. Elastic Transformer (Cai et al., 2025)**: Elastic approaches require training a Gumbel Softmax router within the teacher for size interpolation, involving complex training and architectural modifications. Boomerang distillation uses a standard distillation pipeline with no architectural changes, making it simpler and more general.
- **vs. Minitron (Muralidharan et al., 2024)**: Minitron combines layer pruning with neuron pruning for higher compression efficiency, but each size still requires independent distillation, and the resulting dimensional mismatch prevents patching. The two approaches are complementary — future work could explore Minitron variants that apply only layer pruning (preserving dimension compatibility) to enable boomerang distillation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Discovers and systematically characterizes "zero-shot size interpolation after distillation" as a new phenomenon; an important conceptual contribution to the field.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Validated across four model families, including existing open-source models (DistilBERT/DistilGPT2); comprehensive ablations; comparisons against pruning, distillation, and pretraining baselines.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and intuitive figures; mathematical notation is appropriately concise; some appendix content could be integrated into the main text.
- Value: ⭐⭐⭐⭐⭐ Provides a simple, low-cost method for constructing fine-grained model families with direct practical implications for LLM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Distilling and Adapting: A Topology-Aware Framework for Zero-Shot Interaction Prediction in Multiplex Biological Networks](distilling_and_adapting_a_topology-aware_framework_for_zero-shot_interaction_pre.md)
- [\[ICML 2026\] Float8@2bits: Entropy Coding Enables Data-Free Model Compression](../../ICML2026/model_compression/float82bits_entropy_coding_enables_data-free_model_compression.md)
- [\[NeurIPS 2025\] Zero-Shot Embedding Drift Detection: A Lightweight Defense Against Prompt Injections in LLMs](../../NeurIPS2025/model_compression/zero-shot_embedding_drift_detection_a_lightweight_defense_against_prompt_injecti.md)
- [\[NeurIPS 2025\] Few-Shot Knowledge Distillation of LLMs With Counterfactual Explanations](../../NeurIPS2025/model_compression/few-shot_knowledge_distillation_of_llms_with_counterfactual_explanations.md)
- [\[ICLR 2026\] Pedagogically-Inspired Data Synthesis for Language Model Knowledge Distillation](pedagogically-inspired_data_synthesis_for_language_model_knowledge_distillation.md)

</div>

<!-- RELATED:END -->
