---
title: >-
  [Paper Note] Refining Salience-Aware Sparse Fine-Tuning Strategies for Language Models
description: >-
  [ACL2025][LLM (Other)][Parameter-Efficient Fine-Tuning] This work presents the first systematic evaluation of 8 salience measures in Sparse Parameter-Efficient Fine-Tuning (SPEFT), discovering that simple gradient-based methods combined with static masks consistently outperform LoRA, challenging the common belief that PEFT requires complex designs.
tags:
  - "ACL2025"
  - "LLM (Other)"
  - "Parameter-Efficient Fine-Tuning"
  - "Sparse Fine-Tuning"
  - "Salience Measures"
  - "LoRA"
  - "PEFT"
  - "Sparse Masks"
date: 2026-05-08
content_hash: fdfa0df707d12f4f
---

# Refining Salience-Aware Sparse Fine-Tuning Strategies for Language Models

**Conference**: ACL2025  
**arXiv**: [2412.13488](https://arxiv.org/abs/2412.13488)  
**Code**: [0-ml/speft](https://github.com/0-ml/speft)  
**Area**: LLM/NLP  
**Keywords**: Parameter-Efficient Fine-Tuning, Sparse Fine-Tuning, Salience Measures, LoRA, PEFT, Sparse Masks

## TL;DR

This work presents the first systematic evaluation of 8 salience measures in Sparse Parameter-Efficient Fine-Tuning (SPEFT), discovering that simple gradient-based methods combined with static masks consistently outperform LoRA, challenging the common belief that PEFT requires complex designs.

## Background & Motivation

**Background**: Parameter-Efficient Fine-Tuning (PEFT) is the dominant paradigm for adapting large language models to downstream tasks, with LoRA (Low-Rank Adaptation) being the most popular method. Sparse Fine-Tuning (SPEFT) serves as an alternative that achieves parameter efficiency by training only a small number of non-zero elements within the weight matrices.

**Limitations of Prior Work**: The SPEFT field features various salience measures (gradients, Fisher information, SNIP, etc.) and masking strategies (static vs. dynamic). However, there is a lack of unified and systematic comparison, with individual methods being evaluated in isolation, and no widely accepted best practices.

**Key Challenge**: Existing SPEFT methods (such as FishMASK using Fisher information and DiffPruning using learnable masks) tend to employ more complex second-order measures or dynamic update strategies. Whether this complexity truly translates into performance gains remains an open question.

**Goal**: To answer two fundamental design questions: (1) Which salience measure is optimal? (2) Is static masking sufficient, or is dynamic masking superior?

**Key Insight**: Drawing inspiration from the research framework of zero-cost NAS (neural architecture search) proxies, this work uniformly introduces parameter importance measures from NAS into SPEFT mask construction for a systematic evaluation.

**Core Idea**: Simple gradient-based salience measures combined with static masking represent the best design for SPEFT. Complex second-order measures and dynamic masking not only fail to significantly improve performance but also bring additional computational overhead.

## Method

### Overall Architecture

SPEFT reparameterizes the weights of each layer as $\theta = \theta_0 + \theta_{sp}$, where $\theta_0$ is frozen and $\theta_{sp}$ is an extremely sparse trainable matrix. The framework consists of three steps: (1) compute importance scores for each parameter using a salience measure; (2) construct a binary mask $\tau$ by selecting the top-$\rho$ parameters; and (3) train only the non-zero locations covered by the mask.

### Key Designs

#### Key Design 1: Unified Evaluation of 8 Salience Measures

- **Function**: Systematically evaluate 6 first-order measures (Magnitude, Gradient, SNIP, FORCE, Taylor-FO, SynFlow) and 2 second-order measures (GRaSP, Fisher Information).
- **Design Motivation**: These measures are scattered across different literatures such as pruning, NAS, and SPEFT, and have never been compared under unified conditions.
- **Mechanism**:
    - **Gradient**: $\partial \ell / \partial \theta$, the gradient of the loss with respect to the weights.
    - **SNIP**: $|\partial \ell / \partial \theta \odot \theta|$, connection sensitivity.
    - **Fisher Info**: $(\partial \ell / \partial \theta)^2$, diagonal approximation of Fisher Information.
    - **GRaSP**: $-(H \cdot \partial \ell / \partial \theta) \odot \theta$, Hessian-based gradient signal preservation.
    - All measures are compared fairly under the same number of trainable parameters.

#### Key Design 2: Global vs. Local Sparse Masking

- **Function**: Compare two mask construction strategies: global (sorting globally across all layers to select top-$\rho$) and local (selecting top-$\rho$ independently for each layer).
- **Design Motivation**: Global strategies allow different layers to have different sparsities, potentially offering greater flexibility, whereas local strategies guarantee that parameters in every layer are tuned.
- **Mechanism**: Global—$\tau = \mathbf{1}[s \geq \text{top}_\rho(s)]$, where all parameters are sorted together; Local—$\tau^{(l)} = \mathbf{1}[s^{(l)} \geq \text{top}_\rho(s^{(l)})]$, via layer-wise sorting.

#### Key Design 3: Static vs. Dynamic Masking

- **Function**: Compare masks determined once before training (static) with masks recomputed every $I$ steps during training (dynamic).
- **Design Motivation**: Dynamic masking theoretically adapts to weight changes during training, but it incurs additional overhead and requires resetting optimizer momentum.
- **Mechanism**: Dynamic masking re-estimates salience scores and updates masks every 1000 steps using 1024 samples; static masking is computed once before training and remains fixed.

### Loss & Training

- **Sparsity**: $\rho$ is fixed so that the number of trainable parameters matches that of LoRA (e.g., 0.35% for OPT-350m, 0.27% for BERT-base).
- **Salience Estimation Overhead**: First-order measures require only 64 steps $\times$ batch 16 = 1024 samples, accounting for less than 1% of the training time.
- **Optimizer**: Standard SGD/Adam, with learning rates grid-searched from 5e-4 to 5e-5.

## Key Experimental Results

### Main Results: GLUE Benchmark on OPT-350m (0.35% Trainable Parameters)

| Method | MNLI | MRPC | QNLI | QQP | SST-2 | STS-B | Average | Best Count |
|------|---:|---:|---:|---:|---:|---:|---:|---:|
| LoRA | 83.56 | 84.56 | **89.69** | **89.66** | 93.87 | 88.57 | 88.32 | 2 |
| PiSSA | 83.45 | 83.09 | 89.38 | 89.66 | 93.58 | 88.39 | 87.93 | 1 |
| **Gradient** | **83.86** | **84.80** | 89.68 | 89.51 | **93.93** | **88.95** | **88.45** | **3** |
| Fisher-Info | 35.45 | 84.31 | 88.12 | 86.34 | 87.16 | 88.61 | 78.33 | 0 |
| SynFlow | 77.45 | 77.94 | 83.19 | 88.03 | 92.32 | 79.18 | 83.02 | 0 |
| Magnitude | 79.34 | 71.57 | 86.45 | 87.68 | 91.98 | 45.04 | 77.01 | 0 |

### Ablation Study: Static/Dynamic + Global/Local Masking (OPT-350m)

| Mask Strategy | MNLI | MRPC | QNLI | QQP | SST-2 | STS-B | Average |
|----------|---:|---:|---:|---:|---:|---:|---:|
| Static-Global (SG) | 83.86 | 84.80 | 89.68 | 89.51 | 93.93 | 88.95 | 88.46 |
| Static-Local (SL) | 84.31 | 83.33 | 90.63 | 90.97 | 94.50 | 88.52 | 88.71 |
| Dynamic-Global (DG) | 78.03 | 85.29 | 89.22 | 84.24 | 91.51 | 88.54 | 86.14 |
| Dynamic-Local (DL) | 78.86 | 71.57 | 80.84 | 84.52 | 87.27 | 87.52 | 81.76 |

### LLM Experiments: GSM8K (Gemma2-2b, 0.97% Parameters)

| Method | Flexible Extract | Strict Match | Average |
|------|---:|---:|---:|
| Pretrained | 24.56 | 17.66 | 21.11 |
| LoRA | 39.20 | 28.81 | 34.00 |
| **Gradient SPEFT** | **50.27** | **37.15** | **43.71** |
| GRaSP | 50.15 | 37.03 | 43.59 |

### Key Findings

1. **Gradient is the most reliable salience measure**: On all models (OPT-125m/350m/1.3b, BERT, RoBERTa, Gemma2-2b, Qwen2-7b, Llama3-8b), Gradient consistently achieves optimal or near-optimal performance.
2. **Second-order measures are not worth the cost**: Fisher-Info on OPT-350m yields only 35.45% on MNLI (severe collapse), while GRaSP is unstable and doubles the computation.
3. **Static masking is sufficient**: Dynamic masking actually degrades performance on large models (averaging 86.14 vs. 88.46 on OPT-350m) and introduces additional computation and optimizer reset overheads.
4. **The advantage of SPEFT is magnified on complex tasks**: On GSM8K, Gradient SPEFT outperforms LoRA by 22.6% (43.71 vs. 34.00), demonstrating that sparse adaptation is particularly effective for multi-step reasoning.
5. **Minimal difference between global and local sparsity**: There is no consistent winner, as performance depends highly on the specific model and task.

## Highlights & Insights

1. **Empirical validation of "simple is effective"**: Systematically demonstrating that SPEFT with simple gradient and static masking is a strong baseline, challenging the academic preference for complex methodologies.
2. **Knowledge transfer from NAS proxies to SPEFT**: First to introduce the zero-cost NAS proxy framework into SPEFT evaluation, unifying previously disparate lines of literature.
3. **Significant advantage on GSM8K**: Gradient SPEFT outperforming LoRA by 22.6% is a highly compelling result, particularly under a fair comparison with the exact same number of parameters.
4. **Forward-looking alignment with hardware trends**: Pointing out that hardware such as NVIDIA A100/H100/H200 natively supports sparse computation, meaning the practical speedup potential of SPEFT will grow alongside hardware advancements.
5. **Open-source framework**: Providing a unified SPEFT benchmark framework (github.com/0-ml/speft) to facilitate reproduction and extension by subsequent researchers.

## Limitations & Future Work

1. **Salience estimation still requires forward/backward propagation**: Although the overhead is small (<1% of training time), labeled data is still needed for gradient computation before the mask can be determined.
2. **Lack of in-depth comparison with recent LoRA variants**: Latest low-rank methods such as QLoRA, GaLore, and DoRA are not included in the comparison.
3. **Hardware acceleration of sparse training is not empirically measured**: Although the paper highlights the trend toward hardware sparse computation, dense computation is still used to simulate sparsity in actual training.
4. **Limited exploration of dynamic masking**: Only fixed-interval updates are attempted, without exploring adaptive update frequencies or progressive masking strategies.
5. **Limited improvement on MMLU**: The gap between Gradient SPEFT (53.11) and LoRA (53.07) on Gemma2-2b is minimal, indicating that the advantage of sparse adaptation is less pronounced in knowledge-intensive tasks.

## Related Work & Insights

### vs. LoRA / PiSSA (Low-Rank Adaptation Methods)

LoRA constrains the adaptation space via low-rank matrices BA. Though structurally simple, it restricts the flexibility of parameter selection (limiting rank to the row/column dimensions). SPEFT allows training of the most critical parameters at arbitrary positions within the weight matrix, providing finer-grained control. While both perform comparably on GLUE, SPEFT significantly outperforms LoRA on GSM8K (43.71 vs. 34.00), demonstrating a clear advantage for sparse adaptation in complex tasks requiring precise parameter adjustments.

### vs. FishMASK / Fish-DIP (Fisher Information-Based Methods)

FishMASK uses Fisher information to construct static masks, while Fish-DIP further allows dynamic updates. Experiments in this paper show that Fisher information measures are less reliable than simple gradients (78.33 vs. 88.45 on OPT-350m) and can suffer from severe collapse on certain tasks. This implies that the computational cost of second-order measures does not translate into consistent performance improvements.

### vs. DiffPruning (Learnable Masks)

DiffPruning learns masks via a straight-through estimator, which is inherently a dynamic strategy. This paper finds that static masking is already sufficient, thereby avoiding the additional overhead of mask learning and issues with optimizer resets.

## Rating

- **Novelty**: ⭐⭐⭐ — The methodology itself is a systematic comparison of existing techniques; the novelty lies in the unified evaluation framework and the "simple yet effective" conclusion.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Highly comprehensive coverage: 8 measures $\times$ multiple models $\times$ multiple benchmarks $\times$ static/dynamic/global/local ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — The problem is crisply formulated, experiments are organized systematically, and the connection from NAS proxies to SPEFT is naturally articulated.
- **Value**: ⭐⭐⭐⭐ — Establishes a simple yet strong baseline for the SPEFT field, the open-source framework holds long-term value, and the significant outperformance of LoRA on GSM8K is a highly practical discovery.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Efficient Ensemble for Fine-tuning Language Models on Multiple Datasets](efficient_ensemble_for_fine-tuning_language_models_on_multiple_datasets.md)
- [\[ACL 2025\] A Semantic-Aware Layer-Freezing Approach to Computation-Efficient Fine-Tuning of Language Models](a_semantic-aware_layer-freezing_approach_to_computation-efficient_fine-tuning_of.md)
- [\[NeurIPS 2025\] Sparse MeZO: Less Parameters for Better Performance in Zeroth-Order LLM Fine-Tuning](../../NeurIPS2025/llm_nlp/sparse_mezo_less_parameters_for_better_performance_in_zeroth-order_llm_fine-tuni.md)
- [\[ACL 2025\] HFT: Half Fine-Tuning for Large Language Models](hft_half_fine-tuning_for_large_language_models.md)
- [\[ACL 2025\] GIFT-SW: Gaussian Noise Injected Fine-Tuning of Salient Weights for LLMs](gift-sw_gaussian_noise_injected_fine-tuning_of_salient_weights_for_llms.md)

</div>

<!-- RELATED:END -->
