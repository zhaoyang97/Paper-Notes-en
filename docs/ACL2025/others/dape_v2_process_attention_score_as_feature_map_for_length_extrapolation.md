---
title: >-
  [Paper Note] DAPE V2: Process Attention Score as Feature Map for Length Extrapolation
description: >-
  [ACL 2025][Length Extrapolation] This paper treats the attention scores of Transformers as feature maps. By applying convolutional operations on these attention scores (instead of relying solely on simple key-query dot products), it significantly enhances the length extrapolation capability of Transformers on long sequences, transforming the extrapolation problem into a classic image feature processing problem.
tags:
  - "ACL 2025"
  - "Length Extrapolation"
  - "Attention Mechanism"
  - "Position Encoding"
  - "Convolutional Processing"
  - "Feature Map"
date: 2026-05-08
content_hash: fce1b278b9081925
---

# DAPE V2: Process Attention Score as Feature Map for Length Extrapolation

**Conference**: ACL 2025  
**arXiv**: [2410.04798](https://arxiv.org/abs/2410.04798)  
**Code**: [https://github.com/chuanyang-Zheng/DAPE](https://github.com/chuanyang-Zheng/DAPE)  
**Area**: Others  
**Keywords**: Length Extrapolation, Attention Mechanism, Position Encoding, Convolutional Processing, Feature Map

## TL;DR

This paper treats the attention scores of Transformers as feature maps. By applying convolutional operations on these attention scores (instead of relying solely on simple key-query dot products), it significantly enhances the length extrapolation capability of Transformers on long sequences, transforming the extrapolation problem into a classic image feature processing problem.

## Background & Motivation

**Background**: Transformer models have achieved massive success in fields like language processing and computer vision. However, their core key-query dot-product attention mechanism suffers from severe performance degradation when processing sequences longer than the training length. Existing positional encoding methods such as RoPE, ALiBi, and Kerple attempt to alleviate this issue by embedding positional information in different ways.

**Limitations of Prior Work**: Traditional positional encoding methods (such as RoPE) fail completely when the input length exceeds twice the training length. Even improved methods like FIRE and CoPE still suffer from severe performance drop during significant extrapolation. Furthermore, most existing methods are static and predefined, failing to adapt dynamically to different inputs.

**Key Challenge**: The root cause of the problem lies not only in the design of positional encodings, but in the attention score calculation itself—the naive key-query dot product has limited expressiveness, which restricts the generalization capability of Transformers on long sequences.

**Goal**: (1) To reveal that the root cause of the length extrapolation problem is the insufficient expressive power of attention scores; (2) To enhance extrapolation performance through more refined processing of attention scores.

**Key Insight**: While attempting to combine DAPE (Data-Adaptive Positional Encoding) with NoPE (No Positional Encoding), the authors discovered that simply adding an MLP (without any positional information) to the attention scores could significantly improve performance. This "accidental discovery" indicates that the key lies not in positional encoding, but in the further processing of the attention scores themselves.

**Core Idea**: By analogizing the attention score tensor $[B,H,T,T]$ to an image feature map $[B,C,H,W]$, convolutional operations are applied across the attention head dimension to enhance the expressive power of attention. This transforms the length extrapolation problem into a mature computer vision feature processing problem.

## Method

### Overall Architecture

DAPE V2 inserts a convolutional module to process attention scores after the key-query multiplication and before the softmax in standard Transformers. The input is the raw attention matrix obtained from the key-query dot product. After applying a lower-triangular mask, a $1 \times k$ convolutional kernel performs interactions across the key and head dimensions. The output is the refined attention score, which is then fed into softmax to continue the standard attention process.

### Key Designs

1. **Feature Map Perspective of Attention Scores**:

    - Function: Reinterpreting the attention score tensor as a processable feature map
    - Mechanism: The shape of the attention score is $[B,H,T,T]$, which corresponds perfectly to the image feature map $[B,C,H,W]$. The original DAPE used an MLP to process attention scores, which is equivalent to a $1 \times 1$ convolution. This paper proposes using a larger $1 \times k$ (such as $1 \times 3$) convolutional kernel to perform convolution across heads on the key dimension, enabling information exchange between adjacent tokens and different heads.
    - Design Motivation: Since $1 \times 1$ convolution (i.e., MLP) is considered to have limited expressive power in image processing, expanding the receptive field to $1 \times 3$ can capture local pattern relationships along the key dimension, which is a direct transfer of classic CV knowledge.

2. **Lower-Triangular Mask for Causality**:

    - Function: Preventing information leakage and ensuring causal autoregression
    - Mechanism: Executing the `torch.tril` operation on the attention scores before the convolution to ensure that only information from past positions is involved in the attention calculation of the current position. The convolution uses stride=1 and padding=$k-1$ to maintain the sequence length.
    - Design Motivation: The authors verified the necessity of this design through information leakage experiments—without the lower-triangular mask, the model can reach near-zero loss (perplexity=1), proving that convolution indeed exploits the information in the attention data.

3. **Theoretical Proof: Convolution for Associative Recall**:

    - Function: Theoretically proving that Transformers with convolution can complete the associative recall task without positional encodings.
    - Mechanism: Taking "Hakuna $\rightarrow$ Matata" as an example, using a $1 \times 2$ convolutional kernel $[-1, 1]$ can steer attention focusing on the next position of the target token. Specifically, the convolution transforms key vectors into the differences of adjacent tokens $W_K(x_2 - x_1)$. When the query vector $x_N = x_1$, attention naturally focuses on $x_2$ ("Matata").
    - Design Motivation: The associative recall task is a primary driver of Transformer perplexity. Standard Transformers achieve this function through implicit positional encoding mechanisms, whereas convolution provides an explicit, positional-encoding-free alternative.

### Loss & Training

Standard cross-entropy loss for language modeling is used. The additional parameter cost of DAPE V2 is minimal (including only the convolutional kernel and MLP parameters), and it is trained using the same AdamW optimizer as the baselines. Training is conducted on 8 GPUs, using 50k steps for the 125M model and 50k steps for the 350M/2.7B models.

## Key Experimental Results

### Main Results

On Arxiv and Books3 datasets, trained with length 512, evaluating different extrapolation lengths (lower perplexity/ppl is better):

| Method | 512 | 1024 | 2048 | 4096 | 8192 |
|------|-----|------|------|------|------|
| NoPE | 5.10 | 42.27 | Very Large | Very Large | Very Large |
| RoPE | 4.57 | 86.20 | 237.67 | 256.12 | — |
| Kerple | 4.57 | 4.37 | 5.09 | 6.80 | 9.08 |
| DAPE-Kerple (1×1) | 4.49 | 4.20 | 4.17 | 3.95 | 3.70 |
| **DAPE₁ₓ₃-Kerple** | **4.44** | **4.14** | **4.09** | **3.87** | **3.58** |

Performance of 2.7B large model on Books3:

| Method | 512 | 1024 | 2048 | 4096 |
|------|-----|------|------|------|
| RoPE | 21.01 | 25.00 | 48.13 | 160.59 |
| Kerple | 21.14 | 22.08 | 23.38 | 27.21 |
| DAPE-Kerple | 20.52 | 21.01 | 20.23 | 19.67 |
| **DAPE₁ₓ₃-Kerple** | **20.16** | **20.54** | **19.80** | **19.02** |

### Ablation Study

Impact of different convolutional kernel sizes (Arxiv, training length 128):

| Configuration | 128 | 8192 | Description |
|------|-----|------|------|
| Kerple (no conv) | 8.30 | 12.59 | Baseline |
| DAPE-Kerple (1×1) | 8.21 | 4.97 | MLP equivalent to 1×1 conv |
| DAPE₁ₓ₃-Kerple (1×3) | 8.15 | 4.60 | Best cost-performance trade-off |
| DAPE₁ₓ₅-Kerple (1×5) | 8.13 | 4.57 | Larger kernel brings slight improvement |
| DAPE₁ₓ₇-Kerple (1×7) | 8.12 | 4.57 | Further increase shows diminishing returns |

### Key Findings

- **1x3 kernel size is the best cost-performance trade-off**: Upgrading from 1x1 to 1x3 brings significant improvement, while further increasing to 1x5/1x7 yields diminishing returns.
- **DAPE₁ₓ₃ performs better at lower computational cost**: DAPE₁ₓ₃ with $D_{DAPE}=10$ outperforms the original DAPE with $D_{DAPE}=64$.
- **Training on short sequences rivals long-sequence training**: The performance of DAPE₁ₓ₃ trained on a length of 128 is comparable to RoPE trained on a length of 4096, saving substantial training time.
- **Extra overhead ratio decreases on larger models**: The overhead ratio is 0.75 for the 350M model and only 0.89 for the 6.7B model, showing that the additional cost of DAPE₁ₓ₃ becomes negligible as the model scales up.
- **Effective on the CHE benchmark**: Among 11 tasks, DAPE₁ₓ₃-Kerple outperforms the baseline Kerple on 8 tasks.

## Highlights & Insights

- **Ingenious perspective shift**: Analogizing the attention mechanism in sequence models to feature maps in CV is an elegant cross-domain connection. This mindset allows feature processing techniques accumulated over decades in CV to be directly transferred to NLP.
- **From "accidental discovery" to systematic theory**: The accidental experiment of DAPE+NoPE revealed a deep-rooted issue—the attention calculation itself, rather than positional encoding, is the bottleneck. This research path of starting from experimental phenomena and gradually building theoretical explanations is highly worth learning.
- **Transferable to other attention variants**: This idea can be generalized to any model using attention mechanisms (such as Vision Transformers, cross-attention, etc.), enhancing expressive power by applying convolution on attention scores.

## Limitations & Future Work

- **Validated only on language modeling tasks**: Does not cover other long-context tasks such as long-document QA, summarization, etc.
- **Lack of automation in kernel size selection**: The optimal kernel size might vary across different tasks and datasets, which currently requires manual tuning.
- **No comparison with latest long-context methods**: Lacks comparison with post-training extrapolation methods for existing LLMs, such as LongRoPE and YaRN.
- **Extension of convolution along the query dimension**: Currently, only $1 \times k$ kernels are used (along the key dimension). Using complete $k \times k$ kernels might bring further improvements but with larger computational complexity.
- **Integration with linear/sparse attention**: Exploring whether this approach can be applied to efficient attention methods is worth investigating.

## Related Work & Insights

- **vs DAPE**: The original DAPE used an MLP (equivalent to 1x1 convolution) to adaptively adjust the positional encoding bias. This paper finds that larger kernel sizes can bring improvements even without positional encodings, proving that the core contribution stems from the refinement of attention scores rather than positional encoding adaptation.
- **vs RoPE/ALiBi/Kerple**: These methods are static positional encoding schemes, which are complementary to the dynamic attention score processing mechanism proposed in this paper—DAPE V2 can be layered on top of these methods to achieve further performance gains.
- **vs Hybrid models like FlashConv/Hyena**: Traditional hybrid models apply convolution on token values, whereas this paper applies convolution on attention scores, keeping the original token values unchanged. This represents a more lightweight enhancement.

## Rating

- Novelty: ⭐⭐⭐⭐ The perspective of attention as feature maps is highly novel, but the core operation (adding convolution to attention) is relatively direct.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive experiments across multiple datasets, model sizes, and thorough ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear exposition, but exhibits text overlap with DAPE, leaving some parts slightly redundant.
- Value: ⭐⭐⭐⭐ Provides a fresh perspective for improving the Transformer attention mechanism, though practical applications in large-scale LLMs still require further verification.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Intuitive Fine-Tuning: Towards Simplifying Alignment into a Single Process](intuitive_fine_tuning.md)
- [\[ACL 2025\] EpiCoDe: Boosting Model Performance Beyond Training with Extrapolation and Contrastive Decoding](epicode_boosting_model_performance_beyond_training_with_extrapolation_and_contra.md)
- [\[ACL 2025\] Map&Make: Schema Guided Text to Table Generation](mapmake_schema_guided_text_to_table_generation.md)
- [\[ACL 2025\] MapQaTor: An Extensible Framework for Efficient Annotation of Map-Based QA Datasets](mapqator_an_extensible_framework_for_efficient_annotation_of_map-based_qa_datase.md)
- [\[ACL 2025\] What Matters in Evaluating Book-Length Stories? A Systematic Study of Long Story Evaluation](what_matters_in_evaluating_book-length_stories_a_systematic_study_of_long_story_.md)

</div>

<!-- RELATED:END -->
