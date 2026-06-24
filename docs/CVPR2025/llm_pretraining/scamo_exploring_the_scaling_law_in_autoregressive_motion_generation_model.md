---
title: >-
  [Paper Note] ScaMo: Exploring the Scaling Law in Autoregressive Motion Generation Model
description: >-
  [CVPR 2025][LLM Pretraining][scaling law] This paper systematically validates the scaling law in the field of human motion generation for the first time. It proposes ScaMo, a scalable system comprising Motion FSQ-VAE (addressing codebook collapse), the 260-hour MotionUnion dataset, and a text-prefix autoregressive Transformer. The study discovers a logarithmic relationship between normalized test loss and FLOPs, as well as power-law relationships between vocabulary parameters…
tags:
  - "CVPR 2025"
  - "LLM Pretraining"
  - "scaling law"
  - "autoregressive motion generation"
  - "FSQ-VAE"
  - "text-prefix transformer"
  - "vocabulary scaling"
date: 2026-05-08
content_hash: a0b38c042de27b8e
---

# ScaMo: Exploring the Scaling Law in Autoregressive Motion Generation Model

**Conference**: CVPR 2025  
**arXiv**: [2412.14559](https://arxiv.org/abs/2412.14559)  
**Code**: [github.com/shunlinlu/ScaMo](https://github.com/shunlinlu/ScaMo)  
**Area**: Motion Generation / Scaling Law  
**Keywords**: scaling law, autoregressive motion generation, FSQ-VAE, text-prefix transformer, vocabulary scaling

## TL;DR
This paper systematically validates the scaling law in the field of human motion generation for the first time. It proposes ScaMo, a scalable system comprising Motion FSQ-VAE (addressing codebook collapse), the 260-hour MotionUnion dataset, and a text-prefix autoregressive Transformer. The study discovers a logarithmic relationship between normalized test loss and FLOPs, as well as power-law relationships between vocabulary parameters, model parameters, data size, and FLOPs. Moreover, the optimal configuration is successfully predicted under a budget of $1\times 10^{18}$ FLOPs.

## Background & Motivation
**Background**: The scaling law has been widely validated in NLP (such as the GPT series) and image generation, accurately predicting the optimal model size and data requirements given a computational budget. However, in the field of human motion generation, the scaling law remains almost entirely unexplored.

**Limitations of Prior Work**: (1) **Insufficient data scale**—the largest dataset, Motion-X, contains only 98k sequences, and over 50% of the data in concurrent works consists of static frames (single frames repeated 64 times), resulting in poor data quality; (2) **Inability to scale vocabulary**—traditional VQ-VAEs suffer from codebook collapse when increasing the codebook size (with utilization dropping dramatically to 23%), which degrades performance; (3) **Lack of architecture scalability**—directly expanding vocabulary via pre-trained LLMs harms generation performance and is constrained by the fixed sizes of existing LLMs.

**Key Challenge**: How to construct a motion generation framework that is systematically scalable, from the tokenizer to the generative model, to validate the existence of the scaling law?

**Key Insight**: Solve the three obstacles individually—use FSQ to eliminate codebook collapse, MotionUnion to scale data size, and a text-prefix design to allow free model scaling after text encoding.

## Method

### Overall Architecture
ScaMo consists of two core components: (1) **Motion FSQ-VAE**, which encodes continuous motion sequences into discrete tokens; (2) a **text-prefix autoregressive Transformer**, which generates motion tokens using causal attention, taking the word-level embeddings from a frozen T5-XL as prefix. The system supports model scales from 44M to 3B parameters and vocabulary sizes from $2^8$ to $2^{16}$, trained on MotionUnion (150k sequences, 30M frames, 260 hours).

### Key Designs
1. **Motion FSQ-VAE (Finite Scalar Quantization)**:
    - **Function**: Replaces the argmin matching of VQ with a simple round operation, fundamentally resolving the codebook collapse issue.
    - **Mechanism**: In traditional VQ, the argmin operation $\hat{\mathbf{z}} = \arg\min_{\mathbf{e}_k} \|\mathbf{z} - \mathbf{e}_k\|_2^2$ causes the optimizer to favor specific codebook entries while ignoring others. FSQ modifies this to $\hat{\mathbf{z}} = \text{round}(f(\mathbf{z}))$ (where $f$ is sigmoid), quantizing each channel into $L$ integers, with a codebook size of $|\mathcal{C}| = \prod_{i=1}^d L_i$. The training objective only requires the reconstruction loss $\mathcal{L} = \|\mathbf{m} - \text{Dec}(f(z) + \text{sg}(\text{round}(f(z)) - f(z)))\|_2^2$, eliminating the need for tricks like EMA or codebook reset.
    - **Design Motivation**: The argmin operation in VQ is the root cause of codebook collapse, as spatial matching biases training toward a small subset of entries. In contrast, the round operation in FSQ is uniform, ensuring each entry is selected with equal probability. This maintains a 96% utilization rate even with a $2^{16}$-sized codebook (compared to only 23% for VQ).

2. **Text-Prefix Autoregressive Transformer (Text-Prefix AR)**:
    - **Function**: Uses a frozen T5-XL encoder to generate word-level embeddings as a prefix, and autoregressively generates the motion token part with causal attention.
    - **Mechanism**: Text tokens use bidirectional attention (mutually visible), while motion tokens use causal attention (only visible to previous tokens), allowing motion tokens to attend to all text tokens. The training objective is the cross-entropy loss of motion tokens: $\mathcal{L} = -\sum_{t=1}^n \log p(\hat{m}_t | m_{<t}, S, V)$. Model scales range from 44M to 3B parameters (8-48 layers, 512-3200 dimensions), with an architecture similar to LLaMA (RMSNorm + prefix attention + FFN).
    - **Design Motivation**: Previous methods (like MotionGPT) directly expand the vocabulary of LLMs, which harms language compression capabilities and restricts the model to predefined LLM sizes. The text-prefix design decouples text encoding (frozen T5-XL) from motion generation (freely scalable), reducing the experimental FID from 0.226 to 0.104.

3. **MotionUnion Dataset**:
    - **Function**: Builds a large-scale text-motion dataset containing 260 hours of motion and 150k sequences.
    - **Mechanism**: Integrates Motion-X, CombatMotion, 100-Style, and internal data, uniformly retargeting them to the SMPL skeleton while utilizing the motion representation pipeline of HumanML3D. GPT-4 is used to generate text annotations for internal data. This dataset avoids quality issues such as static frames.
    - **Design Motivation**: Existing datasets are insufficient in scale to observe scaling behavior, and over 50% of data in concurrent works consists of static frames (poor quality).

### Loss & Training
The FSQ-VAE is trained solely with reconstruction loss (no codebook loss, EMA, or reset). The autoregressive model is trained using the cross-entropy loss of motion tokens, with the text prefix excluded from the loss calculation. To fairly evaluate different vocabulary sizes, a normalized loss is used: $\mathcal{L}_u = -\frac{1}{T}\sum_{t=1}^T \log \frac{p(m_t|m_{<t},S,V)}{p(m_t|S,V)}$. To validate the scaling law, a full-matrix training experiment on 44M-3B models with $2^8$-$2^{16}$ vocabularies is conducted.

## Key Experimental Results

### Main Results: Comparison of FSQ vs VQ under Different Codebook Sizes

| Metric | VQ ($2^{10}$) | FSQ ($2^{10}$) | VQ ($2^{16}$) | FSQ ($2^{16}$) |
|---|---|---|---|---|
| Reconstruction L1↓ | 0.031 | **0.030** | 0.034 (unstable) | **0.022** |
| MPJPE↓ | 0.072 | **0.070** | 0.156 (collapse) | **0.089** |
| Codebook Utilization↑ | 89% | **95%** | 23% (collapse) | **96%** |
| Entropy↑ | 6.1 | **6.5** | 4.2 (collapse) | **7.8** |

### Scaling Law Validation

| Relationship | Formula | $R^2$ |
|---|---|---|
| Vocabulary parameters $N_v$ vs FLOPs $C$ | $N_v = 10^{-5.29} \cdot C^{0.75}$ | 0.95 |
| Non-vocabulary parameters $N_{nv}$ vs FLOPs $C$ | $N_{nv} = 10^{-0.52} \cdot C^{0.57}$ | 0.93 |
| Data size $D$ vs FLOPs $C$ | $D = 10^{-0.05} \cdot C^{0.43}$ | 0.91 |
| $N_v$ vs $N_{nv}$ | $N_v = 10^{-5.604} \cdot N_{nv}^{1.467}$ | **0.95** |
| Normalized loss vs FLOPs | $\mathcal{L}_u = -1.062 \times \log_{10}(C) + 13.839$ | 0.97 |

### Prediction Validation ($C = 1 \times 10^{18}$)

| Item | Scaling Law Prediction | Actual Value |
|---|---|---|
| Optimal Model Size | 3B | 3B |
| Optimal Vocabulary Size | $2^{16}$ | $2^{16}$ |
| Predicted Normalized Loss | ~-4.3 | **Precisely Aligned** |

### Ablation Study (343M model, trained on MotionUnion)

| Text Encoder | Prefix Design | FID↓ | Matching Score↑ | Top1 R-P↑ |
|---|---|---|---|---|
| CLIP | ✗ | 0.226 | 3.422 | 0.402 |
| **T5-XL** | **✓** | **0.104** | **3.021** | **0.510** |

### Key Findings
- FSQ significantly outperforms VQ on large codebooks—achieving a 96% vs 23% utilization rate and almost halving the MPJPE.
- Vocabulary parameters should scale faster than model parameters: $N_v \propto N_{nv}^{1.467}$ (where $\gamma > 1$ is a novel finding).
- Model parameters should scale faster than data size: $N_{nv}/D \propto C^{1.325} > C$ (consistent with the direction of the Chinchilla scaling law in NLP).
- The normalized loss and FLOPs precisely follow a logarithmic law, with $R^2=0.97$.
- Compared to the non-prefix CLIP design, the T5-XL prefix design reduces the FID by 54%.

## Highlights & Insights
- This work systematically validates the existence of the scaling law in human motion generation for the first time, providing theoretical guidance for large-scale training in this field.
- FSQ fundamentally addresses codebook collapse by "removing" the argmin operation rather than merely "patching" it.
- The discovery of "$N_v \propto N_{nv}^{1.467}$" is novel—revealing that the vocabulary should scale faster than model parameters, which is counter-intuitive compared to NLP.
- The accuracy of the scaling law prediction under a budget of $10^{18}$ FLOPs validates the reliability of the theory.
- ScaMo-3B can handle abstract and complex long-sentence inputs, demonstrating the emergent capabilities brought by scaling.

## Limitations & Future Work
- The scale of the MotionUnion dataset (150k sequences) is still relatively small compared to those in NLP/CV, which may limit the extrapolation of the scaling law.
- Only the decoder-only autoregressive paradigm has been validated; the scaling law for diffusion-based motion generation remains unexplored.
- FSQ has not been extended to group FSQ or residual FSQ (left as future work by the authors).
- Evaluative generation is mainly conducted on the HumanML3D benchmark, lacking verification in more downstream-related applications (such as animation production or robot control).
- The internal data is not open-sourced, which limits the complete reproducibility of MotionUnion.

## Related Work & Insights
- **vs T2M-GPT**: Uses the same VQ-VAE architecture but has an unscalable codebook; ScaMo overcomes this fundamental limitation with FSQ.
- **vs MotionGPT/LLM-based methods**: Directly expanding LLM vocabulary harms performance and is constrained by fixed model sizes; the text-prefix design successfully decouples the two.
- **vs Chinchilla (Hoffmann 2022)**: The scaling law discovered by ScaMo aligns in direction (model size scales faster than data), but additionally introduces a power law on the vocabulary dimension.
- **Insights**: Incorporating vocabulary size as an independent variable into the scaling law is a fresh perspective, which may be applicable to other codebook-based generative models (such as VQ-VAE-based image/video generation).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Systematically validates the scaling law in motion generation for the first time, and FSQ elegantly and effectively resolves codebook collapse.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Conducts full-matrix experiments from 44M to 3B across various codebook sizes, alongside scaling law fitting and prediction validation.
- Writing Quality: ⭐⭐⭐⭐ Clear logic and well-organized experimental structure driven by Research Questions (RQs).
- Value: ⭐⭐⭐⭐⭐ Provides an actionable theoretical tool for the motion generation community (input computational budget $\rightarrow$ check scaling law $\rightarrow$ determine optimal configuration).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Improving Autoregressive Visual Generation with Cluster-Oriented Token Prediction](improving_autoregressive_visual_generation_with_cluster-oriented_token_predictio.md)
- [\[NeurIPS 2025\] Deep Compositional Phase Diffusion for Long Motion Sequence Generation](../../NeurIPS2025/llm_pretraining/deep_compositional_phase_diffusion_for_long_motion_sequence_generation.md)
- [\[ICLR 2026\] Autoregressive Models Rival Diffusion Models at Any-Order Generation](../../ICLR2026/llm_pretraining/autoregressive_models_rival_diffusion_models_at_any-order_generation.md)
- [\[ECCV 2024\] Plan, Posture and Go: Towards Open-Vocabulary Text-to-Motion Generation](../../ECCV2024/llm_pretraining/plan_posture_and_go_towards_open-vocabulary_text-to-motion_generation.md)
- [\[ICLR 2026\] What Scales in Cross-Entropy Scaling Law?](../../ICLR2026/llm_pretraining/what_scales_in_cross-entropy_scaling_law.md)

</div>

<!-- RELATED:END -->
