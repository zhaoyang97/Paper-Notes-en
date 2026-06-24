---
title: >-
  [Paper Note] Compression in Transformer Language Models Has a Surprising Relationship with Performance
description: >-
  [ACL 2025][Model Compression][Information Theory] This paper investigates the relationship between compression (weight compressibility) and model performance in Transformer language models from an information-theoretic perspective. It uncovers a counter-intuitive phenomenon: within a certain range, models that are easier to compress actually exhibit better generalization performance, which aligns with the prediction of the Minimum Description Length (MDL) principle.
tags:
  - "ACL 2025"
  - "Model Compression"
  - "Information Theory"
  - "Transformer Compression"
  - "Language Model Performance"
  - "Minimum Description Length"
  - "Compression and Generalization"
date: 2026-05-08
content_hash: 9c536f183f3a93c5
---

# Compression in Transformer Language Models Has a Surprising Relationship with Performance

**Conference**: ACL 2025  
**Code**: None  
**Area**: Model Compression  
**Keywords**: Information Theory, Transformer Compression, Language Model Performance, Minimum Description Length, Compression and Generalization

## TL;DR
This paper investigates the relationship between compression (weight compressibility) and model performance in Transformer language models from an information-theoretic perspective. It uncovers a counter-intuitive phenomenon: within a certain range, models that are easier to compress actually exhibit better generalization performance, which aligns with the prediction of the Minimum Description Length (MDL) principle.

## Background & Motivation

**Background**: Model compression is typically viewed as an adversary to performance—compression implies information loss, and information loss implies performance degradation. In practice, compression techniques such as quantization and pruning indeed lead to varying degrees of performance degradation. However, the Minimum Description Length (MDL) principle in information theory suggests that simpler models often possess better generalization capabilities, as simplicity indicates that the model captures the underlying regularities of the data rather than noise.

**Limitations of Prior Work**: (1) Although the MDL theory is elegant, it lacks sufficient empirical validation in deep learning practice, especially on modern large-scale Transformer models; (2) Traditional compression-performance analyses focus only on post-compression performance changes, without analyzing the significance of the "compression rate itself" as an indicator of model quality from an information-theoretic perspective; (3) The relationship between performance and different types of compression (such as weight quantization, structural pruning, and low-rank decomposition) may vary, but a unified analytical framework is lacking.

**Key Challenge**: Intuitively, compression harms performance, but theoretically, simplicity promotes generalization—under what conditions do these two seemingly contradictory perspectives hold?

**Goal**: (1) Systematically measure the weight compressibility of various Transformer models during different training stages; (2) Establish the correlation between compressibility and downstream task performance; (3) Theoretically analyze the reasons behind this "surprising relationship."

**Key Insight**: Rather than measuring performance changes after compressing the model, the authors treat the compressibility of model weights (measured by the compressed size in bits) as a proxy for the model's intrinsic quality. They investigate its correlation with the model's performance in its uncompressed state.

**Core Idea**: High compressibility of weights reflects a substantial amount of structured redundancy in the model parameters. This structured redundancy implies that the model has learned structured regularities from the data (rather than memorizing random noise), and is therefore positively correlated with better generalization performance.

## Method

### Overall Architecture
The experimental framework consists of three components: (1) training multiple Transformer models of various scales, across different training stages and datasets; (2) measuring the weight compressibility (compression rate) of each model checkpoint using multiple compression algorithms; (3) evaluating the downstream task performance of these checkpoints simultaneously and analyzing the statistical relationship between compression rate and performance.

### Key Designs

1. **Multi-Granular Compressibility Measurement**:

    - Function: Accurately quantify the information content of model weights
    - Mechanism: Three complementary compression schemes are used to measure compressibility: (a) General-purpose compression (gzip/zstd): weights are serialized and compressed using standard compression algorithms to measure the compression rate ($\rho = \text{compressed\_size} / \text{original\_size}$); (b) Quantization compression: measuring information loss rates under different quantization bit-widths; (c) Entropy-based metrics: computing the Shannon entropy and differential entropy of the weight distribution as theoretical lower bounds of information content. These three approaches validate each other, eliminating biases from any single metric.
    - Design Motivation: A single compression algorithm might be insensitive to specific types of redundancy. Using intersection of multiple approaches enables a more reliable estimation of the model's "true" information content.

2. **Training Dynamics Tracking Experiment**:

    - Function: Observe the co-evolution of compressibility and performance during training
    - Mechanism: Checkpoints are saved at fixed intervals during the training of GPT-2 and Llama model families. For each checkpoint, the following are simultaneously measured: (a) weight compression rate; (b) validation perplexity; and (c) downstream task accuracy. Curves tracking these three metrics over training steps are plotted. Key observation: during early training, the compression rate drops rapidly (weights become incompressible, i.e., learning new information); during the middle and late stages, the compression rate stabilizes or even rises slightly (weights become structured, i.e., knowledge integration), and the performance curve aligns closely with the inflection point of the compression curve.
    - Design Motivation: Dynamic analysis is more capable of revealing causal relationships than static comparisons. If the inflection point of the compression rate precedes that of performance, compressibility could serve as a leading indicator of training status.

3. **Cross-Model/Cross-Scale Statistical Analysis**:

    - Function: Validate the universality of the compression-performance relationship under various conditions
    - Mechanism: Over 50 publicly released Transformer models (ranging from GPT-2 124M to Llama-2 70B) are gathered. The normalized compression rate of each model (normalized by parameter count to eliminate scale effects) is measured and analyzed alongside its average score on standard evaluation benchmarks using scatter plots and correlation analysis. Spearman's rank correlation coefficient is used to assess monotonic relationships, and partial correlation analysis is applied to control for residual correlation after accounting for model scale. Additionally, tests are conducted to verify if different types of compression (quantization vs. general-purpose compression) yield consistent rankings.
    - Design Motivation: If this relationship only holds for specific model families or scales, its utility is limited. Cross-model validation supports its universality.

### Loss & Training
The models are trained using standard language modeling cross-entropy loss. Compressibility analysis requires no additional training.

## Key Experimental Results

### Main Results

| Model Group | Normalized Compression Rate $(r)$ | Avg. Downstream Task Score | Perplexity | Correlation between $r$ and Performance |
|---|---|---|---|---|
| GPT-2 Family (4 scales) | $0.62 \sim 0.78$ | $54.2 \sim 68.7$ | $29.1 \sim 15.3$ | $\rho=0.89$ |
| Llama-2 Family (3 scales) | $0.58 \sim 0.71$ | $62.3 \sim 73.5$ | $12.8 \sim 6.21$ | $\rho=0.94$ |
| Mistral Series | $0.55 \sim 0.65$ | $67.8 \sim 74.2$ | $8.9 \sim 5.8$ | $\rho=0.91$ |
| Cross-Family Summary (50+ models) | - | - | - | $\rho=0.72 \ (p<0.001)$ |

### Ablation Study (Training Dynamics, GPT-2 Medium)

| Training Stage | Steps | Compression Rate | Val PPL | Downstream Acc |
|---|---|---|---|---|
| Initial Stage | 10K | 0.91 | 45.2 | 42.1% |
| Rapid Learning | 50K | 0.72 | 22.3 | 56.8% |
| Compression Inflection Point | 100K | 0.65 | 18.7 | 61.3% |
| Performance Plateau | 200K | 0.63 | 16.5 | 64.2% |
| Final | 300K | 0.64 | 16.1 | 65.0% |

### Key Findings
- Within the same model family, the Spearman correlation between the normalized compression rate and downstream performance is as high as 0.89-0.94, which remains significant even after controlling for parameter size (partial correlation $\rho=0.53$).
- During training, the inflection point of the compression rate (shifting from a decline to stabilization) occurs approximately 30% fewer steps earlier than the performance inflection point, making it a viable leading indicator for early stopping.
- This relationship holds across all three compression approaches (gzip, zstd, and quantization), suggesting that it reflects the intrinsic structural properties of the weights rather than a bias towards any specific compression algorithm.
- Overfitted models (models trained excessively on small datasets) exhibit significantly lower compression rates (meaning they are harder to compress), consistent with the predictions of the MDL theory.

## Highlights & Insights
- Connecting the information-theoretic concept of compression with generalization performance in deep learning is a highly elegant approach. Using the compression rate as a predictive indicator of model quality requires absolutely no downstream task evaluation, which can dramatically lower the cost of model selection.
- Training dynamics analysis reveals a practical finding: the inflection point of the compression rate can serve as an early stopping signal, which is more cost-effective than traditional validation-set monitoring.
- The discovery that overfitting leads to incompressibility opens a new perspective on understanding regularization—regularization might be effective precisely because it promotes the structuring or compressibility of model weights.

## Limitations & Future Work
- The definition of the normalized compression rate (normalized by parameter count) may not be fully comparable between extremely large-scale and small-scale models.
- Only language models are analyzed; whether the same relationship holds for vision and multimodal models remains to be validated.
- The direction of causality is not entirely clear—whether "better models happen to be more compressible" or if "compressibility directly facilitates generalization."
- Measuring compressibility itself requires serializing the entire model weights, which may not be efficient enough for ultra-large models.

## Related Work & Insights
- **vs MDL Theory**: This work provides strong empirical support for the MDL principle on large-scale neural networks, bridging the gap between theory and practice.
- **vs Lottery Ticket Hypothesis**: The Lottery Ticket Hypothesis suggests that sparse subnetworks exist within dense networks. This work offers a complementary explanation from an information-theoretic perspective—good subnetworks exist because the original network contains structured redundancy.
- **vs Double Descent**: The double descent phenomenon shows that over-parameterized models can generalize better. The findings in this paper (that over-parameterized models are more compressible) provide a potential explanation for this observation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The research perspective is highly novel, and the approach of predicting performance from compression rate is extremely inspiring.
- Experimental Thoroughness: ⭐⭐⭐⭐ It covers various models, multiple compression techniques, and training dynamics analysis.
- Writing Quality: ⭐⭐⭐⭐ The integration of theory and experiments is well-handled, and the narrative is engaging.
- Value: ⭐⭐⭐⭐⭐ It holds profound significance for understanding model compression, generalization, and training dynamics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] 500xCompressor: Generalized Prompt Compression for Large Language Models](500xcompressor_generalized_prompt_compression_for_large_language_models.md)
- [\[CVPR 2025\] MambaIC: State Space Models for High-Performance Learned Image Compression](../../CVPR2025/model_compression/mambaic_state_space_models_for_high-performance_learned_image_compression.md)
- [\[ICML 2025\] Strategic Fusion Optimizes Transformer Compression](../../ICML2025/model_compression/strategic_fusion_optimizes_transformer_compression.md)
- [\[ACL 2025\] Wanda++: Pruning Large Language Models via Regional Gradients](wanda_pruning_large_language_models_via_regional_gradients.md)
- [\[NeurIPS 2025\] TokenSqueeze: Performance-Preserving Compression for Reasoning LLMs](../../NeurIPS2025/model_compression/tokensqueeze_performance-preserving_compression_for_reasoning_llms.md)

</div>

<!-- RELATED:END -->
