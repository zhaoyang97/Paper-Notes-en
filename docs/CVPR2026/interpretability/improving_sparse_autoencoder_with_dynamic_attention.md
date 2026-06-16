---
title: >-
  [Paper Note] Improving Sparse Autoencoder with Dynamic Attention
description: >-
  [CVPR 2026][Interpretability][sparsemax] This paper reformulates the Sparse Autoencoder (SAE) into a cross-attention architecture with shared concept vectors and replaces softmax with sparsemax. This allows each sample to **automatically determine the number of activated concepts based on its own complexity**, overcoming the inherent "setting K" problem in To
tags:
  - CVPR 2026
  - Interpretability
  - sparsemax
date: 2026-05-08
content_hash: b313c68bd7e5a290
---
# Improving Sparse Autoencoder with Dynamic Attention

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Improving_Sparse_Autoencoder_with_Dynamic_Attention_CVPR_2026_paper.html)  
**Code**: https://github.com/qyj-bkjx/Sparsemax-SAE  
**Area**: Interpretability / Mechanistic Interpretability / Sparse Autoencoders  
**Keywords**: Sparse Autoencoder, sparsemax, cross-attention, dynamic sparsity, concept disentanglement

## TL;DR
This paper reformulates the Sparse Autoencoder (SAE) into a cross-attention architecture with shared concept vectors and replaces softmax with sparsemax. This allows each sample to **automatically determine the number of activated concepts based on its own complexity**, overcoming the inherent "setting K" problem in TopK SAEs to achieve lower reconstruction error and clearer concepts in both image and text domains.

## Background & Motivation

**Background**: Neurons in large models are "polysemantic" (one neuron responds to multiple unrelated concepts simultaneously, known as superposition). Sparse Autoencoders (SAEs) decompose activations into a set of sparse, monosemantic, and interpretable concepts, serving as a primary tool for current mechanistic interpretability.

**Limitations of Prior Work**: A core challenge for SAEs is "how many concepts to use for each feature"—too many concepts harm interpretability, while too few harm reconstruction, both of which lead to poor concept learning. Existing activation functions have significant drawbacks: ReLU-based methods (including GatedReLU/JumpReLU) require $L_1/L_0$ regularization, where $L_1$ causes feature shrinkage (activations are pulled toward 0) and requires manual tuning of balancing coefficients. TopK/BatchTopK simply retain the top K concepts and zero out the rest; while this avoids regularization, it treats K as a hyperparameter. **An incorrect K results in missing concepts for complex samples and "dead concepts" for simple samples.**

**Key Challenge**: Sparsity should ideally be **data-dependent** (complex images require more concepts, simple ones fewer). However, ReLU regularization and fixed-K TopK are "one-size-fits-all" global settings that cannot adapt on a per-sample basis.

**Goal**: To design an SAE where sparsity is **automatically determined by the content complexity of each sample**, trained solely using reconstruction loss without relying on extra regularization or K-tuning.

**Key Insight**: The authors leverage the sparsemax activation, which projects inputs onto a probability simplex and can **assign exactly 0 to low-scoring items** while remaining differentiable with a closed-form threshold solution. This aligns with the "sparse activation" requirement of SAEs: the threshold $\tau$ can be computed from the sample itself, effectively acting as a per-sample dynamic K.

**Core Idea**: Reformulate the SAE using a cross-attention framework (where features act as queries and dictionary concepts act as keys/values, with encoder and decoder sharing the same concept vectors) and replace the softmax in attention with sparsemax to allow dynamic, adaptive sparsity.

## Method

### Overall Architecture
Traditional SAEs use a single-layer MLP for encoding and decoding: $z=\sigma(W_{enc}(x-b_{enc})),\ \hat{x}=W_{dec}z+b_{dec}$, decomposing polysemantic features $x\in\mathbb{R}^d$ into a sparse combination of $M\gg d$ concepts $C=\{c_1,\dots,c_M\}$, where columns of $W_{dec}$ are concepts and $\sigma$ determines the sparsity pattern. This paper modifies it in two ways: (1) using cross-attention to **link the encoder and decoder with the same set of concept vectors** instead of two independent MLPs; (2) replacing softmax with sparsemax to let each sample dynamically determine the number of activated concepts. The entire model is trained only with reconstruction loss, without sparse regularization or K-tuning. This is a single-forward mechanism modification rather than a multi-stage pipeline.

### Key Designs

**1. Transformerized SAE: Linking Encoder-Decoder with Shared Concept Vectors**

To address the issue where $W_{enc}$ and $W_{dec}$ in traditional SAEs are independent projections causing a disconnect between encoding weights and decoding concepts, the authors rewrite the SAE as cross-attention. The dictionary to be learned is treated as a set of concept vectors which project to act as both keys and values; each latent feature acts as a query to produce the reconstructed feature via cross-attention:

$$Q=x^\top W_Q,\quad K=C^\top W_K,\quad V=C^\top W_V,\quad \hat{x}=\sigma\!\left(\frac{QK^\top}{\sqrt{d}}\right)V$$

The computation of attention weights naturally serves as the "encoding" stage—it measures the correlation between the query and various concepts ($z$ is higher when the feature is closer to a concept in the embedding space). Weighting the values (the same set of concepts) serves as "decoding." Crucially, **keys and values originate from the same concept set $C$**. Concepts used for measuring correlation during encoding are the same ones used for weighted reconstruction during decoding. This creates strong synergy between weights and concepts, offering better coherence and reconstruction quality than MLP-style SAEs.

**2. Sparsemax Attention: Per-sample Dynamic Activation**

To solve the "fixed K" problem of TopK and the "dense output" problem of softmax, the authors replace softmax with sparsemax. Given $z=QK^\top\in\mathbb{R}^M$ as the similarity scores between the query and $M$ concepts, sparsemax projects $z$ onto the probability simplex by finding the nearest point in Euclidean distance:

$$\text{sparsemax}(z)=\arg\min_{p\in\Delta^{M-1}}\|p-z\|^2$$

Its closed-form solution is a soft thresholding $\text{sparsemax}(z)_m=\max(z_m-\tau,0)$, where the threshold $\tau$ is solved from the constraint that the sum of selected items equals 1. By sorting $z$ as $z_{(1)}\ge\cdots\ge z_{(M)}$ and taking $k=\max\{r: z_{(r)}+\frac{1-\sum_{i=1}^r z_{(i)}}{r}>0\}$, then $\tau=\frac{\sum_{i=1}^k z_{(i)}-1}{k}$. Unlike TopK which fixes K, $\tau$ is **dynamically calculated** based on input complexity. If a query feature contains multiple concepts, $z$ will have many similar high values, leading to a larger support set $S$ (activated concepts); if it is a pure concept, $S$ is small. In the paper's examples, sparsemax might assign 6 concepts to a complex image and only 2 to a simple one. Sparsemax is differentiable and has a well-defined Jacobian, allowing direct gradient optimization.

### Example: Complex vs. Simple Images
For a high-content complex image, the query is highly correlated with multiple dictionary concepts, resulting in many large values in $z$. Sparsemax solves for a lower threshold $\tau$ and a larger support set $S$ (e.g., 6 activated concepts). For a simple image, only a few concepts are highly correlated, so $z$ has few large values, $\tau$ "cuts" more aggressively, and $S$ is very small (e.g., 2 concepts). The same model and parameters allow the activation count to adapt to the image—something a fixed K cannot achieve.

### Loss & Training
Trained strictly on reconstruction loss without any sparsity regularization or K-tuning. For vision, CLIP ViT-B/16 is used, taking the residual stream output of the penultimate attention layer. Following PatchSAE, the concept count $M$ is set to $49,152$ ($64\times$ the hidden dimension). It is trained on ImageNet with batch size 32 for a total of 2,621,440 patches. For text, GPT-2 Small is used, taking the 8th layer residual stream, trained on OpenWebText with sequence length 128, batch size 128, for $10^9$ tokens, with dictionary sizes $M\in\{3072, 6144, 12288, 24576\}$. Adam optimizer is used with lr=$3\times10^{-4}$, $\beta_1=0.9, \beta_2=0.99$. Baselines use K=32 (TopK family) or sparsity weight 1e-3 (ReLU family) as per their respective papers.

## Key Experimental Results

### Main Results
Text reconstruction is measured by NMSE (Normalized Mean Squared Error, lower is better) and CE degradation (cross-entropy degradation of GPT-2 when intermediate features are replaced by SAE reconstructions, closer to 0 is better). The table below shows NMSE on OpenWebText for different dictionary sizes $M$:

| Method | M=3072 | M=6144 | M=12288 | M=24576 |
|--------|--------|--------|---------|---------|
| ReLU | 0.064 | 0.064 | 0.064 | 0.059 |
| JumpReLU | 0.051 | 0.050 | 0.050 | 0.051 |
| Gated | 0.078 | 0.092 | 0.129 | 0.489 |
| TopK | 0.014 | 0.059 | 0.010 | 0.055 |
| BatchTopK | 0.014 | 0.061 | 0.060 | 0.060 |
| **Sparsemax SAE (Ours)** | **0.005** | **0.038** | **0.004** | **0.039** |

Across all dictionary scales, Sparsemax SAE significantly outperforms all baselines in NMSE (consistent results on WikiText-103) and shows smaller CE degradation. This indicates that dynamic sparse attention can disentangle polysemantic features into interpretable concepts while reconstructing inputs with minimal information loss. In zero-shot image classification (replacing ViT intermediate embeddings with top-n concepts for classification on 11 datasets), Sparsemax SAE performs best on average across all top-n settings (n=1/5/10/50), leading the runner-up significantly at small n.

### Ablation Study
Decoupling the contributions of the "Transformer architecture" and "sparsemax activation" on ImageNet (top-n concept classification accuracy):

| Configuration | on 1 | on 5 | on 10 | on 50 |
|------|------|------|-------|-------|
| ReLU SAE | 3.12 | 15.83 | 22.17 | 34.87 |
| Transformer + ReLU | 3.86 | 16.85 | 24.08 | 36.33 |
| MLP + Sparsemax | 7.91 | 29.87 | 39.73 | 55.32 |
| **Sparsemax SAE (Ours)** | **10.93** | **33.47** | **42.13** | **59.95** |

### Key Findings
- **Both designs are positive, with sparsemax being the primary driver**: Adding only the transformer to ReLU SAE provided minor gains. Separately switching to sparsemax (MLP+Sparsemax) yielded huge gains (on-1 jumping from 3.12 to 7.91). Combining both (the full model) performed best (on-1 reaching 10.93), showing that dynamic sparse activation is the major contributor, further boosted by the shared-concept cross-attention architecture.
- **Informing existing SAE K-selection**: The per-sample sparsity calculated by Sparsemax SAE can serve as a guide for tuning TopK SAEs. On Food101, Sparsemax's on-1 accuracy of 26.11 far exceeded TopK/BatchTopK with fixed K=24/32 (0.99~8.64), demonstrating that "dynamic K" can inform what "fixed K" should be.
- **Cleaner Concepts**: Visualizations show that compared to BatchTopK, the concept mask maps and top-5 reference images learned by Sparsemax SAE are clearer and more interpretable. On datasets like EuroSAT and DTD, which differ significantly from natural pre-training images, SAE concepts even outperform the original CLIP, indicating that the learned concepts generalize well.

## Highlights & Insights
- **Turning "Sparsity Selection" from a hyperparameter into an endogenous model calculation**: The sparsemax threshold $\tau$ has a closed-form solution that floats based on sample complexity, effectively refining BatchTopK's "batch-level K" into a "sample-level K"—a paradigm shift from manual tuning to adaptation.
- **Linking encoder and decoder with shared concept vectors is clever**: Traditional SAE encoding weights and decoding concepts are separate and prone to decoupling. Using cross-attention to make keys/values originate from the same concept set ensures that encoding correlation and decoding reconstruction are naturally synergetic.
- **Transferable insight**: The technique of replacing softmax with sparsemax can be adapted to any attention/routing scenario that "needs to be sparse, interpretable, and vary with input" (e.g., MoE expert selection, retrieval top-k) without requiring extra regularization.

## Limitations & Future Work
- Sparsemax is a special case of $\alpha$-entmax where $\alpha=2$. The paper keeps $\alpha$ fixed and does not explore whether a learnable $\alpha$ or other entmax variants would be superior. ⚠️
- Solving for the threshold $\tau$ requires sorting similarity scores, and the computational overhead/efficiency for very large dictionaries $M$ (e.g., 49,152) is not deeply analyzed.
- Evaluation is focused on CLIP ViT and GPT-2 Small; its scalability to larger LLMs, diffusion models, or multimodal LLMs remains to be verified.
- While "dynamically determining the concept count" is flexible, there is no systematic analysis of activation stability or training convergence, nor a discussion on whether it might activate too many concepts on extreme samples.

## Related Work & Insights
- **vs TopK / BatchTopK SAE**: These rely on fixed K (sample or batch level) to select concepts; an incorrect K misses information or results in dead concepts. Sparsemax SAE refines K into a per-sample dynamic threshold, requiring no tuning and yielding better reconstruction and concept quality.
- **vs ReLU / GatedReLU / JumpReLU SAE**: These rely on $L_1/L_0$ regularization for sparsity, where $L_1$ causes feature shrinkage and balancing coefficients are hard to tune. This work uses only reconstruction loss, with sparsity emerging endogenously from sparsemax.
- **vs PatchSAE etc. (Vision SAEs)**: Prior works like PatchSAE mainly ported ReLU/TopK SAEs to the vision domain without changing the architecture. This paper proposes a completely new cross-attention SAE architecture + sparsemax activation that is generalizable across vision and text.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Cross-attention SAE + sparsemax dynamic sparsity fundamentally changes the SAE sparsity selection mechanism.
- Experimental Thoroughness: ⭐⭐⭐⭐ Dual domains (Vision/Text), multiple dictionary scales, with ablation of architecture vs. activation, though lacks large-scale model and efficiency analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, complete sparsemax derivation, and convincing visualizations.
- Value: ⭐⭐⭐⭐ Solves the pain point of picking K in SAEs and can retroactively inform existing methods; highly practical for the mechanistic interpretability community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] SparseRM: A Lightweight Preference Modeling with Sparse Autoencoder](../../AAAI2026/interpretability/sparserm_a_lightweight_preference_modeling_with_sparse_autoencoder.md)
- [\[AAAI 2026\] Data Whitening Improves Sparse Autoencoder Learning](../../AAAI2026/interpretability/data_whitening_improves_sparse_autoencoder_learning.md)
- [\[ICML 2026\] CorrSteer: Generation-Time LLM Steering via Correlated Sparse Autoencoder Features](../../ICML2026/interpretability/corrsteer_generation-time_llm_steering_via_correlated_sparse_autoencoder_feature.md)
- [\[ICLR 2026\] SALVE: Sparse Autoencoder-Latent Vector Editing for Mechanistic Control of Neural Networks](../../ICLR2026/interpretability/salve_sparse_autoencoder-latent_vector_editing_for_mechanistic_control_of_neural.md)
- [\[CVPR 2026\] CIGMA: Causal Information-Gain Mechanistic Attribution of Attention Heads in Vision Transformers](cigma_causal_information-gain_mechanistic_attribution_of_attention_heads_in_visi.md)

</div>

<!-- RELATED:END -->
