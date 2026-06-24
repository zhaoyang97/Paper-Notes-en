---
title: >-
  [Paper Note] PosFormer: Recognizing Complex Handwritten Mathematical Expression with Position Forest Transformer
description: >-
  [ECCV 2024][Self-Supervised Learning][Handwritten Mathematical Expression Recognition] The Position Forest Transformer (PosFormer) is proposed. By encoding the LaTeX sequence of mathematical expressions into a position forest structure, it explicitly models the hierarchical and spatial relationships among symbols. An implicit attention correction module is designed to comprehensively outperform SOTA methods on single-line, multi-line, and complex expression datasets…
tags:
  - "ECCV 2024"
  - "Self-Supervised Learning"
  - "Handwritten Mathematical Expression Recognition"
  - "Position Forest"
  - "Transformer"
  - "Attention Correction"
  - "Structural Relationship Modeling"
date: 2026-05-08
content_hash: 36efe38de35d833d
---

# PosFormer: Recognizing Complex Handwritten Mathematical Expression with Position Forest Transformer

**Conference**: ECCV 2024  
**arXiv**: [2407.07764](https://arxiv.org/abs/2407.07764)  
**Code**: [Available](https://github.com/SJTU-DeepVisionLab/PosFormer)  
**Area**: Handwritten Mathematical Expression Recognition / Sequence Recognition  
**Keywords**: Handwritten Mathematical Expression Recognition, Position Forest, Transformer, Attention Correction, Structural Relationship Modeling

## TL;DR

The Position Forest Transformer (PosFormer) is proposed. By encoding the LaTeX sequence of mathematical expressions into a position forest structure, it explicitly models the hierarchical and spatial relationships among symbols. An implicit attention correction module is designed to comprehensively outperform SOTA methods on single-line, multi-line, and complex expression datasets, without introducing additional inference overhead.

## Background & Motivation

### Challenges in Handwritten Mathematical Expression Recognition

Handwritten Mathematical Expression Recognition (HMER) aims to convert expression images into LaTeX sequences, which is widely applied in online education, manuscript digitization, automatic grading, and other scenarios. Its core difficulties lie in:

**Complex relationships among symbols**: Mathematical formulas contain nested structures such as superscripts, subscripts, fractions, and square roots. The spatial relationships (above/middle/below) and hierarchical relationships (nesting depth) among symbols are far more complex than in regular text. The model needs to correctly generate structural symbols such as `^`, `_`, `{`, and `}` to describe these relationships.

**Diversity of handwritten styles**: Handwriting across different individuals varies dramatically in terms of scale, slant, and stroke connections, which increases the difficulty of recognition.

### Limitations of Prior Work

**Tree-structured methods** (e.g., BLSTM, SAN, TSDNet): These model expressions as syntax trees, outputting triplets of (parent node, child node, relationship). However, tree structures lack sufficient diversity across different expressions, leading to poor generalization and lower accuracy.

**Sequence-based methods** (e.g., WAP, BTTR, CoMER, LAST): These model HMER as an end-to-end image-to-sequence task, utilizing an encoder-decoder architecture to autoregressively predict LaTeX sequences. While these methods are more flexible and yield higher accuracy, they can only **implicitly** learn the structural rules of LaTeX. Consequently, when dealing with complex nested expressions, the model struggles to correctly understand the relative position relationships of symbols deep within nested structures.

**Attention drift issue**: CoMER corrects the current attention by subtracting the accumulated attention of all previously decoded steps. However, when decoding structural symbols (such as `^`, `{`), the model focuses on unresolved regions to comprehend structural relationships. When these attentions are accumulated as correction terms, they cause attention drift errors during the decoding of subsequent entity symbols. For example, when decoding $4^{x-\frac{1}{4}}$, the attention from `^` and `{` can interfere with the correct focus region for the subsequent symbol `x`.

### Design Motivation

The core insight of PosFormer is: **positional information can be parsed directly from the LaTeX sequence itself without requiring extra annotations**. By encoding mathematical expressions into a "position forest" structure (rather than a single tree), the model can handle independent and nested substructures simultaneously. Jointly optimizing position recognition as an auxiliary task explicitly guides the model to learn position-aware feature representations. This auxiliary branch is completely removed during inference, incurring zero additional computational overhead.

## Method

### Overall Architecture

PosFormer adopts an encoder-decoder architecture:
- **Encoder**: A DenseNet backbone extracts 2D visual features.
- **Decoder**: A three-layer Transformer decoder containing Multi-Head Attention (MHA), Implicit Attention Correction (IAC), and a Feed-Forward Network (FFN).
- **Output Heads**: Three parallel linear heads that predict the LaTeX symbol, nested level, and relative position, respectively.

During training, the expression recognition and position recognition tasks are jointly optimized. During inference, only the expression recognition branch is retained.

### Key Designs

#### 1. Position Forest Coding

**Function**: Encodes LaTeX sequences into a position forest structure, assigning a position identifier (such as "MLLR") to each symbol to represent its relative position in 2D space.

**Mechanism**: According to LaTeX syntactic rules, expressions can be divided into four categories of substructures (as shown in the figure):
- **Superscript/subscript structures**: `^{...}` / `_{...}`
- **Fraction structures**: `\frac{...}{...}`
- **Square root structures**: `\sqrt{...}`
- **Special operators**: such as Product, Limit, etc.

Coding follows three rules:
1. Substructures are coded sequentially from left to right.
2. Each substructure is coded as a tree: the main body is the root node (M), the upper part is the left node (L), and the lower part is the right node (R).
3. Based on the relationships of substructures (independent or nested), the coded trees are concatenated or nested to form a position forest.

For example, the expression $A y_1^3 + \frac{y_2^{\beta_1}B}{C}$ is decomposed into 8 substructures, where the substructures inside the fraction are nested within the fraction tree. Consequently, each symbol obtains a position identifier composed of M/L/R. For instance, the identifier for `\beta` is "MLLR" (indicating the left-left-right position of the main body, i.e., the subscript position of the third-level nesting).

**Design Motivation**: Unlike tree-structured methods, the position forest can naturally handle both independent substructures (multiple parallel trees) and nested substructures (trees nested within trees), describing the structure of complex formulas more flexibly. A key advantage is that no extra annotation is required—positional information is automatically derived from the syntactic rules of the LaTeX sequence.

#### 2. Position Forest Decoding

**Function**: Decomposes position recognition into two subtasks—nested level prediction and relative position prediction—which serve as auxiliary training tasks.

**Mechanism**: Given a position identifier $\mathbf{I}_k = \{q_k^{(i)} | q_k^{(i)} \in \{M, L, R\}\}_{i=1}^{\eta_k}$:
- The **nested level** is $\eta_k - 1$ (the length of the identifier minus one).
- The **relative position** is $q_k^{(\eta_k)}$ (the last character of the identifier).

For example, "MLLR" $\rightarrow$ nested level = 3, relative position = R (below).

After padding the identifier sequences to a uniform length, they are converted into identifier embeddings via a non-linear layer $\xi$:

$$\mathbf{Q}_{\text{emb}} = [\xi(\mathbf{Q}_1); \xi(\mathbf{Q}_2); \cdots; \xi(\mathbf{Q}_T)] + \mathbf{Q}_{\text{pos}}$$

The embedding vectors are fed into the Transformer decoder alongside visual features. The output features $\mathbf{F}_t$ are utilized to predict the nested level and relative position, respectively:

$$p(y_n^{(t)}) = \text{softmax}(\mathbf{W}_n\mathbf{F}_t + b_n)$$
$$p(y_r^{(t)}) = \text{softmax}(\mathbf{W}_r\mathbf{F}_t + b_r)$$

**Design Motivation**: Utilizing positional information as auxiliary supervision forces the intermediate features of the decoder to encode position-aware information. Since the predictions of nested level and relative position share feature representations with symbol recognition, this multi-task learning implicitly improves the feature quality for the primary task. This branch is removed during inference, introducing no additional latency.

#### 3. Implicit Attention Correction (IAC)

**Function**: Adjusts the accumulation of attention correction terms to prevent the attention of structural symbols from interfering with the decoding of subsequent entity symbols.

**Mechanism**: Key observation—only **entity symbols** (e.g., `a`, `b`, `1`) correspond to physical regions in the image, whereas **structural symbols** (e.g., `^`, `_`, `{`, `}`) do not. Therefore, the attention correction term should only accumulate attention from entity symbols, with the attention of structural symbols replaced by zero.

An indicator function $I_\Omega$ is introduced:

$$I_\Omega(y) = \begin{cases} \mathbf{1}, & \text{if } y \notin \Omega \\ \mathbf{0}, & \text{if } y \in \Omega \end{cases}$$

where $\Omega$ is the set of structural symbols. The correction term is calculated as:

$$\mathbf{A}_k^{(t)} = \sum_{i=1}^{t-1} (\tilde{\mathbf{E}}_k^{(i)} \odot I_\Omega(y_c^{(i)}))$$

The final corrected attention is formulated as:

$$\hat{\mathbf{E}}_k = \mathbf{E}_k - \phi(\mathbf{A}_k)$$

**Design Motivation**: This is a precise fix to CoMER's attention correction. The limitation of CoMER lies in its indiscriminate accumulation of attention for all symbols. However, since structural symbols have no image entities, their attention often points to unresolved regions or the global image. Excluding these "spurious" attentions from the correction term allows subsequent decoding steps to obtain more accurate alignment information.

### Loss & Training

**Multi-task Loss**:

$$L_{\text{all}} = \lambda_1 \cdot L_{\text{rec}} + \lambda_2 \cdot L_{\text{pos}}$$

$$L_{\text{rec}} = -\frac{1}{T}\sum_{t=1}^{T} y_c^{(t)} \log p(y_c^{(t)})$$

$$L_{\text{pos}} = -\frac{1}{T}\sum_{t=1}^{T} (y_n^{(t)} \log p(y_n^{(t)}) + y_r^{(t)} \log p(y_r^{(t)}))$$

$\lambda_1 = \lambda_2 = 1$; both tasks are weighted equally.

**Training Setup**:
- Backbone: DenseNet
- Decoder: 3-layer Transformer
- Position recognition head: Two linear layers (maximum nested level of 3, relative position vocabulary size of 6)
- Training parameters (batch size, learning rate, optimizer) follow CoMER/LAST.
- Trained on a single NVIDIA A800 GPU.

## Key Experimental Results

### Main Results

**CROHME Single-line Dataset (with scale augmentation)**

| Dataset | Metric | PosFormer | Prev. SOTA (CoMER) | Gain |
|--------|------|-----------|----------------|------|
| CROHME 2014 | ExpRate | **62.68%** | 59.33% | +2.03% (vs BPD-Coverage +2.03) |
| CROHME 2016 | ExpRate | **61.03%** | 59.81% | +1.22% |
| CROHME 2019 | ExpRate | **64.97%** | 62.97% | +2.00% |

**Larger gains without scale augmentation**

| Dataset | PosFormer | Prev. SOTA (CAN) | Gain |
|--------|-----------|---------------|------|
| CROHME 2014 | **60.45%** | 57.26% | +3.19% |
| CROHME 2016 | **60.94%** | 56.15% | +4.79% |
| CROHME 2019 | **62.22%** | 56.34% | +5.88% |

**Multi-line M2E Dataset**

| Method | ExpRate ↑ | ≤1 ↑ | CER ↓ |
|------|-----------|------|-------|
| CoMER | 56.20% | 73.39% | 0.0499 |
| LAST | 56.50% | 72.80% | 0.0530 |
| **PosFormer** | **58.33%** | **75.58%** | **0.0366** |

The ExpRate improves by 1.83%, and the CER is significantly reduced from 0.0499 to 0.0366.

**Large-scale HME100k Dataset**

| Method | ExpRate ↑ | ≤1 ↑ | ≤2 ↑ |
|------|-----------|------|------|
| CAN | 67.31% | 82.93% | 89.17% |
| **PosFormer** | **69.51%** | **84.91%** | **90.51%** |

### Ablation Study

**Contribution of Each Component (CROHME series, ExpRate %)**

| Model | 2014 | 2016 | 2019 |
|------|------|------|------|
| baseline (CoMER) | 59.33 | 59.81 | 62.97 |
| + PF | 62.13 (+2.80) | 61.03 (+1.22) | 63.80 (+0.83) |
| + PF + IAC | **62.64 (+3.31)** | **61.20 (+1.39)** | **64.64 (+1.67)** |

**PF Extended to RNN Methods (ExpRate %)**

| Method | CROHME 2014 | CROHME 2016 | CROHME 2019 |
|------|-------------|-------------|-------------|
| DWAP | 50.10 | 47.50 | - |
| DWAP + PF | **57.10 (+7.00)** | **56.23 (+8.73)** | **57.30** |
| CAN | 55.27 | 53.97 | 52.96 |
| CAN + PF | **57.30 (+2.03)** | **56.84 (+2.87)** | **57.13 (+4.17)** |
| ABM | 56.85 | 52.92 | 53.96 |
| ABM + PF | **58.11 (+1.26)** | **56.50 (+3.58)** | **54.30 (+0.34)** |

### Key Findings

1. **Position Forest (PF) contributes the most**: Adding PF alone brings a 2.80% improvement on CROHME 2014, while IAC contributes an additional 0.51%.
2. **PF is a general plug-and-play module**: When integrated into RNN-based methods like DWAP, ABM, and CAN, it yields substantial improvements (up to +8.73%), demonstrating the wide applicability of the approach.
3. **More pronounced gains on fault-tolerant metrics**: The improvements on ≤1 and ≤2 error metrics are more dominant than on ExpRate, indicating that PF effectively enhances the model's error correction capability for complex formulas.
4. **Zero inference overhead**: The position forest and position recognition heads are utilized only during training and completely discarded during inference, adding no extra latency or computational costs.
5. **Greatest advantage on complex expressions**: On a custom-built multi-level nested MNE dataset, PosFormer achieves an improvement of up to 4.62%, verifying the importance of explicit position modeling for complex expressions.

## Highlights & Insights

1. **"Free Lunch" design philosophy**: Obtains superior features during training via auxiliary tasks while incurring absolutely zero overhead during inference—a very elegant design.
2. **Automatic supervision generation from LaTeX syntax rules**: Eliminates the need for manual annotations by parsing positional information straight from existing LaTeX sequences to use as auxiliary labels.
3. **Precise diagnosis and remedy of CoMER's attention correction**: Clearly highlights the semantic difference in attention between structural and entity symbols. The proposed solution is simple yet effective (requiring only an indicator function).
4. **Forest vs. Tree representation**: Utilizing a forest structure instead of a single tree naturally handles parallel relationships among independent substructures in mathematical expressions, offering greater flexibility than tree structures.

## Limitations & Future Work

1. **Maximum nested level fixed to 3**: This may be insufficient for extremely deep nested expressions, although nesting beyond 3 levels is relatively rare in practice.
2. **Lack of an integrated language model**: Compared to the language-aware method RLFN, PosFormer (as a language-model-free approach) still holds a 6.25% advantage, but combining it with a language model could yield further improvements.
3. **Relatively outdated DenseNet backbone**: Stronger visual backbones (such as Swin Transformer) or pre-trained vision encoders could be explored.
4. **Dependency of Position Forest construction on LaTeX syntactic assumptions**: For non-standard LaTeX expressions or new types of structures, the coding rules may need to be expanded.
5. **Small scale of the MNE dataset**: The self-constructed complex expression test set could be further enlarged to provide a more comprehensive evaluation.

## Related Work & Insights

- **Relationship with CoMER**: PosFormer builds on CoMER as its baseline, making precise improvements on its attention correction mechanism (IAC) while introducing the position forest auxiliary task.
- **Relationship with tree-structure methods**: PosFormer does not decode directly with tree structures; instead, it encodes positional information into a forest structure as auxiliary supervision, combining the merits of both tree-structure and sequence-based methods.
- **Insights on multi-task learning**: Improving feature representation quality by designing auxiliary tasks highly correlated with the main task is a paradigm that can be generalized to other structured sequence prediction tasks (such as chemical formula recognition, music score recognition, etc.).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Both the position forest coding and the implicit attention correction are novel and sound designs, notably the auxiliary task paradigm that is utilized during training but removed during inference.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers four types of datasets (single-line, multi-line, complex, large-scale), presents comprehensive ablation studies, and verifies extensibility across three RNN baselines.
- **Writing Quality**: ⭐⭐⭐⭐ — The methodology is clearly described, the algorithm pseudocode and illustrations aid comprehension, and the motivations are well-articulated.
- **Value**: ⭐⭐⭐⭐ — Significantly advances the SOTA for HMER. The PF module holds substantial practical value as a plug-and-play component.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MetaWriter: Personalized Handwritten Text Recognition Using Meta-Learned Prompt Tuning](../../CVPR2025/self_supervised/metawriter_personalized_handwritten_text_recognition_using_meta-learned_prompt_t.md)
- [\[ICML 2025\] PDE-Transformer: Efficient and Versatile Transformers for Physics Simulations](../../ICML2025/self_supervised/pde-transformer_efficient_and_versatile_transformers_for_physics_simulations.md)
- [\[ICML 2025\] Update Your Transformer to the Latest Release: Re-Basin of Task Vectors](../../ICML2025/self_supervised/update_your_transformer_to_the_latest_release_re-basin_of_task_vectors.md)
- [\[ICML 2026\] FLAG: Foundation Model Representation with Latent Diffusion Alignment via Graph for Spatial Gene Expression Prediction](../../ICML2026/self_supervised/flag_foundation_model_representation_with_latent_diffusion_alignment_via_graph_f.md)
- [\[CVPR 2025\] MAP: Unleashing Hybrid Mamba-Transformer Vision Backbone's Potential with Masked Autoregressive Pretraining](../../CVPR2025/self_supervised/map_unleashing_hybrid_mamba-transformer_vision_backbones_potential_with_masked_a.md)

</div>

<!-- RELATED:END -->
