---
title: >-
  [Paper Note] Strategic Fusion Optimizes Transformer Compression
description: >-
  [ICML2025][Model Compression][Transformer compression] This paper proposes the Strategic Fusion framework, which fuses 12 layer pruning signals based on activation, mutual information, gradient, weight, and attention through linear regression and random forest. Validated on the BERT model and 9 text classification datasets, multi-signal fusion pruning outperforms single-signal strategies, and when combined with knowledge distillation, the average accuracy-to-size ratio is imp…
tags:
  - "ICML2025"
  - "Model Compression"
  - "Transformer compression"
  - "Layer pruning"
  - "Knowledge distillation"
  - "Signal fusion"
  - "Random forest"
date: 2026-05-08
content_hash: be0cf1e1f3d7af5c
---

# Strategic Fusion Optimizes Transformer Compression

**Conference**: ICML2025  
**arXiv**: [2501.03273](https://arxiv.org/abs/2501.03273)  
**Code**: None  
**Area**: Model Compression  
**Keywords**: Transformer compression, Layer pruning, Knowledge distillation, Signal fusion, Random forest

## TL;DR

This paper proposes the Strategic Fusion framework, which fuses 12 layer pruning signals based on activation, mutual information, gradient, weight, and attention through linear regression and random forest. Validated on the BERT model and 9 text classification datasets, multi-signal fusion pruning outperforms single-signal strategies, and when combined with knowledge distillation, the average accuracy-to-size ratio is improved by 18.84 times.

## Background & Motivation

**Background**: Large pre-trained Transformer models (such as BERT) have achieved SOTA performance on NLP tasks, but their immense computational and storage overheads restrict deployment on edge devices and in real-time scenarios. Model compression (weight pruning, quantization, knowledge distillation, etc.) is the current mainstream solution path.

**Limitations of Prior Work**: Recent studies on layer-level pruning (directly removing entire layers instead of individual parameters) rely on a single metric signal to assess layer importance—for example, considering only activation magnitude, gradient norm, or attention head contribution. A single signal cannot comprehensively capture a layer's fine-grained contribution to downstream tasks, and often requires manually preset pruning rules (e.g., "pruning the layer with the smallest mean activation"), which lacks flexibility.

**Key Challenge**: Different signals evaluate layer importance from various angles (activation reflects feature transformation activity, gradient reflects sensitivity to loss, mutual information reflects information preservation, etc.), and any single perspective is incomplete. Prior work has rarely systematically explored how to combine multiple pruning signals to make better decisions.

**Goal**
   - How to systematically evaluate the effectiveness of 12 different layer pruning signals?
   - How to fuse multiple signals into a unified pruning decision without presetting rules?
   - How to recover the accuracy loss after pruning through knowledge distillation?

**Key Insight**: The authors provide intuitive explanations for the selection of each signal from both mathematical foundations and biological analogies (e.g., low activation is analogous to low-firing-rate neurons in the brain being pruned). They model the layer pruning decision as a supervised learning problem: using multiple signals as features and post-pruning accuracy as labels to train a fusion model that automatically learns the optimal pruning strategy.

**Core Idea**: Upgrading layer pruning from "selecting layers based on a single rule" to "automatic decision-making by fusing multiple signals using a machine learning model."

## Method

### Overall Architecture

The overall workflow is divided into three stages:

1. **Signal Extraction**: For each layer of BERT, 12 different importance signals are calculated (covering five major categories: activation, mutual information, gradient, weight, and attention).
2. **Fusion Decision**: The 12 signals are fed into a fusion model (Linear Regression or Random Forest) as features, outputting a pruning priority ranking for each layer without requiring manually preset rules.
3. **Pruning + Distillation**: Layers are pruned sequentially based on the order given by the fusion model, followed by fine-tuning after each pruning step. Finally, the original model is used as a teacher for knowledge distillation to recover accuracy.

The input is the pre-trained BERT model and the target dataset, and the output is the compressed compact model.

### Key Designs

#### 1. 12 Single-Signal Layer Pruning Strategies

The authors designed 12 signals covering five major categories, each measuring layer importance from a different perspective:

**Activation-based (3 types)**:
- **Inhibition**: Computes the mean of each layer's activation matrix $A_{\text{inhibition}} = \frac{1}{n \cdot d} \sum_{i,j} A_{i,j}$. The layer with the lowest mean is considered to contribute the least and is prioritized for pruning. This is analogous to the "inhibited" state of low-firing-rate neurons in the brain.
- **Intensity**: Uses the L2 norm to measure the "energy density" of the activation vector; layers with low intensity may be redundant.
- **Energy**: Computes the square of the Frobenius norm of activations to evaluate the activity level of the layer from a global energy perspective.

**Mutual Information-based**:
- Measures the mutual information between the layer output and the final model prediction; layers with low mutual information contribute little to the decision.

**Gradient-based**:
- Computes the gradient norm of the loss function with respect to the layer parameters; a small gradient means that changes in that layer's parameters have little impact on the loss, making it safe to remove.

**Weight-based**:
- Analyzes the statistical properties of the layer's weight matrix (e.g., norm size); layers with small weight norms may be redundant. This relates to the Lottery Ticket Hypothesis concept.

**Attention-based**:
- Analyzes the output distribution of attention heads; if the attention distribution of a layer is close to uniform (i.e., no obvious focus pattern), it indicates that the layer has not learned valuable attention patterns.

When each strategy runs independently, a predefined rule is required (e.g., "pruning the layer with the smallest mean" or "pruning the layer with the largest norm"), which is precisely the bottleneck of single-signal methods.

#### 2. Strategic Fusion

Core innovation: transforming pruning decisions from "rule-driven" to "data-driven."

- **Function**: Uses the 12 signals as a 12-dimensional feature vector to construct training samples for each layer, predicting the impact of pruning that layer on accuracy using a supervised learning model, thereby automatically ranking pruning priority.
- **Linear Regression Fusion (LR Fusion)**: Linearly weights each signal to learn the optimal weight for each. The advantage is high interpretability, allowing direct visualization of which signal contributes the most; the disadvantage is the assumption that the signals are linearly separable.
- **Random Forest Fusion (RF Fusion)**: Uses a random forest to capture non-linear interactions and complex dependencies among signals. Experiments prove that RF outperforms all single-signal strategies and LR fusion on 7/9 datasets.
- **Design Motivation**: The optimal single-signal strategy is inconsistent across different datasets (e.g., gradient is best on one dataset, while attention is best on another), indicating that there is no "universal signal." Fusing multiple perspectives achieves robustness across datasets.

#### 3. Knowledge Distillation to Recover Accuracy

- **Function**: Uses the original full BERT as the teacher and the pruned model as the student, training through distillation loss.
- **Mechanism**: The distillation loss includes both soft label loss (matching the teacher's output distribution) and hard label loss (matching ground-truth labels), allowing the student to learn the dark knowledge of the teacher while maintaining discriminative ability for labels.
- **Design Motivation**: Layer pruning is a structural change. Even if the fusion strategy selects the optimal pruning order, removing layers inevitably causes information loss. Knowledge distillation provides a systematic accuracy recovery mechanism. Experiments show that after distillation, 6/9 datasets exceeded the accuracy of the original model.

### Pruning Process and Layer Order

An important finding is that **pruning order** is crucial:
- Boundary layers (the 1st layer and the last layer) usually carry critical information. High-performance strategies automatically learn not to prune these layers early on.
- One advantage of fusion strategies is their ability to data-drivenly learn this "pruning middle layers first, retaining outer layers" strategy without manual specification.
- After each layer is pruned, fine-tuning is performed, and signals are recomputed to determine which layer to prune next, forming a greedy iterative process.

### Loss & Training

- All experiments are based on BERT-base (12 layers), using the BERT tokenizer with a maximum sequence length of 32 tokens.
- Fine-tuning is performed after each pruning step, followed by final knowledge distillation training.
- Comprehensive evaluation is conducted on 9 datasets, covering classification tasks with 2 to 20 categories.

## Key Experimental Results

### Main Results: Random Forest Fusion vs. Best Single-Signal Strategy

| Dataset | RF Fusion Performance | RF + Distillation Performance | RF Fusion Rank |
|--------|-----------|------------|-----------|
| newsgroup | Highest | Outperforms original accuracy | 1st |
| dbpedia | Highest | Outperforms original accuracy | 1st |
| arxiv | Highest | Outperforms original accuracy | 1st |
| patent | Highest | Outperforms original accuracy | 1st |
| yahoo | Highest | Outperforms original accuracy | 1st |
| yelp | Highest | Outperforms original accuracy | 1st |
| agnews | Near-optimal | Mitigates accuracy drop | 2nd |
| imdb | Highest | Outperforms original accuracy | 1st |
| amazon | Near-optimal | Mitigates accuracy drop | 3rd |

**Core Conclusion**: RF fusion ranks first on 7/9 datasets, and second and third on the remaining two datasets, respectively. After distillation, 6 datasets outperformed the accuracy of the original unpruned BERT.

### Knowledge Distillation Effects and Accuracy-to-Size Ratio

| Metric | Result |
|---------|------|
| Number of datasets exceeding original accuracy after distillation | 6 / 9 |
| Number of datasets with mitigated accuracy drop after distillation | 3 / 9 |
| Average improvement fold in accuracy-to-size ratio | 18.84x |
| Number of datasets where RF fusion ranked first | 7 / 9 |
| Total pruning strategies tested | 14 (12 single-signal + 2 fusion) |
| Number of datasets tested | 9 |
| Task types | Text classification and sentiment analysis |

### Key Findings

- **No Universal Single Signal**: The optimal single-signal strategy varies across different datasets, providing strong evidence for the necessity of fusion strategies.
- **Random Forest > Linear Regression**: RF can capture non-linear interactions between signals and outperforms LR fusion on the vast majority of datasets, indicating a complex non-linear relationship among pruning signals.
- **Boundary Layer Protection**: Successful strategies tend to preserve the first and last layers. In BERT, these layers are responsible for low-level language feature extraction and task-related high-level semantic representation, respectively.
- **Immense Value of Knowledge Distillation**: The 18.84x improvement in the accuracy-to-size ratio indicates that distillation not only recovers accuracy but also enables the compressed model to far exceed the original model in efficiency metrics.

## Highlights & Insights

- **Modeling Layer Pruning as a Supervised Learning Problem**: This is the most significant methodological innovation of this paper. Utilizing 12 handcrafted signals as features and using an ML model instead of manual rules is an elegant meta-learning concept. This framework can easily be extended—incorporating new signals in the future only requires adding a new feature dimension.

- **Biological Analogies Enhancing Intuition**: The authors provide mathematical definitions and biological analogies (e.g., activation inhibition analogous to low-firing-rate neurons) for each signal. While not strictly rigorous, these help build an intuitive understanding of the physical meaning of different signals.

- **Cross-Dataset Robustness**: RF fusion performs well on 9 highly diverse datasets, demonstrating the strong generalization capability of this method without requiring dataset-specific tuning of pruning strategies.

- **Transferable Concept**: The idea of multi-signal fusion is not limited to layer pruning; it can also be applied to structural pruning tasks like channel pruning and attention head pruning—any scenario requiring a judgment on "which module is unimportant" can benefit.

## Limitations & Future Work

- **Validated Only on BERT-base**: All experiments were conducted purely on BERT-base (12 layers), without verifying the efficacy on larger models (such as LLaMA or the GPT series) or deeper Transformers. With more layers, the search space for fusion strategies and the method of constructing training data may need adjustment.

- **Limited to Text Classification Tasks**: The 9 datasets are all classification or sentiment analysis tasks, without involving generative tasks (machine translation, summarization, etc.). For generative tasks, layer importance metrics might require different signal designs.

- **Extremely Short Sequence Length Limit**: The truncation to a maximum of 32 tokens is rather short for practical applications and might underestimate the importance of certain layers in long-context scenarios.

- **Lack of Computational Overhead Analysis**: Calculating 12 signals and training the fusion model itself incurs additional computational overhead. The paper does not analyze whether this overhead is favorable compared to directly using a single strategy.

- **Suboptimality of Greedy Iterative Pruning**: The greedy strategy of pruning only one layer at a time and then re-evaluating cannot guarantee global optimality. Formulations combining reinforcement learning or global search methods could be considered.

- **No Comparison with Modern Distillation Baselines**: Mature compression schemes such as TinyBERT or DistilBERT are lack of direct comparison with this approach.

## Related Work & Insights

- **vs. Single-Signal Pruning Methods** (e.g., activation pruning in Ganguli & Chong 2024, gradient-based pruning in Molchanov et al. 2017): These methods evaluate layer importance from a single perspective, which may perform well on specific datasets but lacks cross-dataset consistency. This work addresses this issue by integrating multi-perspective signals.

- **vs. Lottery Ticket Hypothesis** (Frankle & Carlin, 2019): LTH focuses on weight-level sparse subnetworks, while this paper focuses on layer-level structural pruning. They are complementary—one can first perform layer pruning to reduce the overall structure and then apply weight pruning for further simplification.

- **vs. Knowledge Distillation Methods** (Hinton et al., 2015; DistilBERT): Distillation in this work serves as an accuracy recovery method after pruning, rather than an independent compression approach. Combining fusion pruning with distillation represents a highly reasonable pipeline design.

- **Insight**: The multi-signal fusion framework can be migrated to the compression of VLMs (Vision-Language Models), using both visual and linguistic signals to cooperatively determine module importance.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of multi-signal fusion has novelty, though the concrete techniques (linear regression/random forest) are relatively traditional.
- Experimental Thoroughness: ⭐⭐⭐⭐ Coverage across 14 strategies and 9 datasets is extensive, but lacks comparisons with strong baselines like DistilBERT and larger models.
- Writing Quality: ⭐⭐⭐⭐ Structurally clear, with mathematical definitions and biological analogies enhancing readability.
- Value: ⭐⭐⭐ The method is simple and effective but limited to the BERT + classification scenario, and its utility is constrained by not being validated on large models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] DLF: Extreme Image Compression with Dual-generative Latent Fusion](../../ICCV2025/model_compression/dlf_extreme_image_compression_with_dual-generative_latent_fusion.md)
- [\[ACL 2025\] Compression in Transformer Language Models Has a Surprising Relationship with Performance](../../ACL2025/model_compression/compression_in_transformer_language_models_has_a_surprising_relationship_with_pe.md)
- [\[ICCV 2025\] Context Guided Transformer Entropy Modeling for Video Compression](../../ICCV2025/model_compression/context_guided_transformer_entropy_modeling_for_video_compression.md)
- [\[ICML 2026\] FlattenGPT: Depth Compression for Transformer with Layer Flattening](../../ICML2026/model_compression/flattengpt_depth_compression_for_transformer_with_layer_flattening.md)
- [\[ICML 2025\] Text-to-LoRA: Instant Transformer Adaption](text-to-lora_instant_transformer_adaption.md)

</div>

<!-- RELATED:END -->
