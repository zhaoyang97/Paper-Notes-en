---
title: >-
  [Paper Note] Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models
description: >-
  [NeurIPS 2025][Multimodal VLM][Sparse Autoencoders] This paper extends sparse autoencoders (SAEs) to vision-language models (e.g., CLIP), proposes the MonoSemanticity score (MS) to quantitatively evaluate the monosemanticity of neurons, and demonstrates that manipulating SAE neurons can directly steer multimodal large language models (e.g., LLaVA) to insert or suppress specific concepts.
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - Sparse Autoencoders
  - Monosemanticity
  - CLIP
  - Interpretability
  - Model Steering
date: 2026-05-08
content_hash: f3ffcfe9aa496278
---

# Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models

**Conference**: NeurIPS 2025  
**arXiv**: [2504.02821](https://arxiv.org/abs/2504.02821)  
**Code**: [https://github.com/ExplainableML/sae-for-vlm](https://github.com/ExplainableML/sae-for-vlm)  
**Area**: Multimodal VLM  
**Keywords**: Sparse Autoencoders, Monosemanticity, CLIP, Interpretability, Model Steering

## TL;DR

This paper extends sparse autoencoders (SAEs) to vision-language models (e.g., CLIP), proposes the MonoSemanticity score (MS) to quantitatively evaluate the monosemanticity of neurons, and demonstrates that manipulating SAE neurons can directly steer multimodal large language models (e.g., LLaVA) to insert or suppress specific concepts.

## Background & Motivation

**Background**: SAEs have recently achieved notable success in LLM interpretability research by decomposing high-dimensional representations into monosemantic atomic features, enabling researchers to understand and control model behavior. However, their application to vision-language models (VLMs) remains limited to preliminary explorations such as interpretable classification or cross-model concept discovery.

**Limitations of Prior Work**: Neurons in VLMs such as CLIP are typically polysemantic—a single neuron responds strongly to multiple unrelated concepts (e.g., mobile phones and rulers). This polysemanticity severely impedes understanding of the model's internal mechanisms. More critically, no quantitative metric exists to systematically evaluate the monosemanticity achieved by SAEs on visual tasks.

**Key Challenge**: Although SAEs should theoretically decompose polysemantic neurons into monosemantic representations, the absence of appropriate evaluation criteria prevents systematic comparison of different SAE architectures and makes it impossible to verify whether the decomposed representations align with human perception.

**Goal**: (1) How can monosemanticity of SAE neurons in VLMs be quantitatively measured? (2) Which SAE design factors contribute most to monosemanticity? (3) Can monosemantic neurons be used to steer the outputs of multimodal large language models?

**Key Insight**: Images differ from text—a single image can directly activate neurons without requiring context. The semantic similarity among highly activated images can therefore serve as a natural basis for evaluating neuron monosemanticity.

**Core Idea**: Propose an activation-weighted image similarity metric—the MonoSemanticity score—to systematically evaluate SAE monosemanticity on VLMs, and leverage monosemantic neurons to enable unsupervised concept-level steering of multimodal LLMs.

## Method

### Overall Architecture

The pipeline consists of three stages: (1) training SAEs on intermediate layers of a CLIP visual encoder to reconstruct activations into higher-dimensional sparse representations; (2) evaluating the monosemanticity quality of each SAE neuron using the MS score; and (3) injecting the trained SAE into LLaVA's visual encoder to steer model outputs by manipulating specific neuron activations.

### Key Designs

1. **MonoSemanticity Score (MS)**:

    - **Function**: Quantitatively measures whether a single SAE neuron attends to only one semantic concept.
    - **Mechanism**: Given a diverse image set, pairwise cosine similarities are extracted using DINOv2 to form a similarity matrix $S$. The normalized activation values of each neuron across all images are then collected, and the activation-weighted pairwise similarity is computed as: $\text{MS}^k = \frac{\sum_{n<m} r_{nm}^k s_{nm}}{\sum_{n<m} r_{nm}^k}$, where $r_{nm}^k = \tilde{a}_n^k \tilde{a}_m^k$ represents the joint activation intensity of neuron $k$ for image pair $(n, m)$. Higher similarity among highly activated images yields a higher MS.
    - **Design Motivation**: Avoids the issue of selecting a fixed Top-K activated images (since different neurons exhibit varying degrees of specialization) by using continuous weighting to accommodate diverse activation patterns.

2. **Matryoshka SAE Architecture**:

    - **Function**: Introduces nested multi-level reconstruction objectives to improve monosemanticity.
    - **Mechanism**: The $\omega$ neurons of the SAE are divided into nested groups (e.g., $\{0.0625\omega, 0.1875\omega, 0.4375\omega, \omega\}$), where each level reconstructs using only the first $m$ neurons. The training objective is the sum of reconstruction losses across all levels, combined with a BatchTopK activation function that constrains at most $K$ neurons to be nonzero.
    - **Design Motivation**: The Matryoshka objective forces earlier neurons to capture more salient concepts, forming a coarse-to-fine hierarchical structure that achieves higher monosemanticity than standard BatchTopK at the same expansion factor.

3. **Visual SAE Steering for Multimodal LLMs**:

    - **Function**: Controls LLaVA's generated content by manipulating the activation values of specific SAE neurons.
    - **Mechanism**: An SAE is injected after layer 22 of the CLIP visual encoder. All token embeddings are first encoded into sparse activations; the activation of target neuron $k$ is then forced to a value $\alpha$ (positive for concept insertion, negative for suppression), and the modified activations are decoded back into embeddings. The language model parameters remain unchanged throughout.
    - **Design Motivation**: Leverages the monosemantic decomposition of SAEs at the visual encoder level to enable concept-level control in a modular fashion.

### Loss & Training

SAE training uses a reconstruction objective with sparsity regularization: $\mathcal{L}(\mathbf{v}) = \mathcal{R}(\mathbf{v}) + \lambda \mathcal{S}(\mathbf{v})$. The BatchTopK variant directly controls sparsity via a Top-K activation function ($K=20$), while the Matryoshka variant additionally incorporates multi-level reconstruction objectives. Training is performed for $10^5$ steps on CLIP activations extracted from ImageNet, with a batch size of 4096 and the Adam optimizer.

## Key Experimental Results

### Main Results

| SAE Type | Expansion Factor | Peak MS (CLIP Layer 22) | MS w/o SAE | Gain |
|----------|-----------------|------------------------|------------|------|
| BatchTopK | ×1 | 0.66 | 0.01 | +0.65 |
| BatchTopK | ×16 | 0.92 | 0.01 | +0.91 |
| Matryoshka | ×1 | 0.83 | 0.01 | +0.82 |
| Matryoshka | ×8 | 0.94 | 0.01 | +0.93 |

| Steering Method | Concept Insertion (Dual-Criteria Pass Rate) | Concept Suppression (Dual-Criteria Pass Rate) |
|----------------|---------------------------------------------|-----------------------------------------------|
| SAE Steering (Ours) | 42.4% | 52.5% |
| DiffMean Baseline | 35.8% | 33.3% |

### Ablation Study

| Configuration | MS Trend | Notes |
|---------------|----------|-------|
| Expansion ×1 vs. original layer | ~90% of neurons have higher MS | Sparse decomposition itself improves monosemanticity |
| $K=1$ (maximum sparsity) | Highest MS but $R^2 = 31.3\%$ | Trade-off between monosemanticity and reconstruction quality |
| $K=20$ | High MS and $R^2 = 66.8\%$ | Balanced operating point |
| $K=50$ | Lower MS but $R^2 = 74.9\%$ | Insufficient sparsity |

### Key Findings

- Even at an expansion factor of ×1, approximately 90% of SAE neurons exhibit higher MS than the original neurons, demonstrating that sparse dictionary learning itself improves concept disentanglement.
- Matryoshka SAEs achieve higher MS than BatchTopK at the same expansion factor, at a cost of 2–3 points in $R^2$.
- In steering experiments, SAE-derived directions substantially outperform DiffMean in prompt-following rate (85.8% vs. 66.2%), indicating that monosemantic neuron directions provide more precise steering.
- A large-scale user study (1,000 questions, 71 users) shows 82.8% agreement between MS and human judgments, reaching 100% agreement when the MS difference exceeds 0.8.

## Highlights & Insights

- The systematic transfer of the SAE interpretability paradigm from NLP to the visual domain is well-executed, forming a complete pipeline from metric design to human validation to steering applications. The MS score elegantly exploits the context-free nature of images, making monosemanticity evaluation more natural than in LLMs.
- The modular design for concept steering is highly practical: inserting an SAE after the visual encoder—without modifying the language model—suffices to enable concept insertion and suppression, offering a new pathway for safety-oriented control of multimodal models.
- The finding that an expansion factor of ×1 already substantially improves monosemanticity suggests that the regularization effect of the sparse reconstruction objective is more important than increasing model width.

## Limitations & Future Work

- High-MS neurons do not always yield precise steering effects; a "golden retriever" neuron may trigger any dog-related output, stemming from the MLLM's lack of fine-grained alignment with the visual encoder.
- The MS score is currently defined only for visual representations and has not been extended to the text modality.
- Some SAE neurons that act as feature detectors produce no discernible steering effect, motivating the need for better neuron selection strategies.
- The user study is relatively limited in scale (71 users, 1,000 questions).

## Related Work & Insights

- **vs. Anthropic Towards Monosemanticity**: Anthropic applied SAE-based interpretability to LLMs; this paper transfers the approach to VLMs. The visual modality enables more intuitive evaluation and additionally demonstrates cross-model steering applications.
- **vs. DiffMean Activation Steering**: DiffMean derives steering directions from the mean difference between concept-present and concept-absent activations. The SAE-based approach substantially outperforms it in prompt-following rate (85.8% vs. 66.2%), as SAE directions are inherently "cleaner."
- **vs. CLIP-Dissect**: CLIP-Dissect explains neurons using external textual descriptions (extrinsic annotation), whereas the SAE approach is an intrinsic disentanglement method that automatically discovers monosemantic features through the reconstruction objective.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Systematically extends SAE interpretability from LLMs to VLMs and proposes a paired evaluation metric; the direction is valuable, though the core techniques are not entirely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Systematic experiments across multiple visual encoders, layers, and expansion factors, complemented by a large-scale user study and steering application demonstrations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clearly structured, with a coherent progression from metric definition to experimental validation to application.
- **Value**: ⭐⭐⭐⭐ Provides practical tools and benchmarks for VLM interpretability and safety control.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] SAUCE: Selective Concept Unlearning in Vision-Language Models with Sparse Autoencoders](../../ICCV2025/multimodal_vlm/sauce_selective_concept_unlearning_in_vision-language_models_with_sparse_autoenc.md)
- [\[NeurIPS 2025\] Approximate Domain Unlearning for Vision-Language Models](approximate_domain_unlearning_for_visionlanguage_models.md)
- [\[NeurIPS 2025\] In-Context Compositional Learning via Sparse Coding Transformer](in-context_compositional_learning_via_sparse_coding_transformer.md)
- [\[NeurIPS 2025\] DOTA: DistributiOnal Test-time Adaptation of Vision-Language Models](dota_distributional_testtime_adaptation_of_visionlanguage_mo.md)
- [\[NeurIPS 2025\] VaMP: Variational Multi-Modal Prompt Learning for Vision-Language Models](vamp_variational_multi-modal_prompt_learning_for_vision-language_models.md)

<!-- RELATED:END -->
