---
title: >-
  [Paper Note] Unified Multimodal Understanding via Byte-Pair Visual Encoding
description: >-
  [ICCV2025][Multimodal VLM][BPE Visual Tokenization] This paper applies the Byte-Pair Encoding (BPE) strategy from NLP to visual tokenization, proposing a priority-guided encoding scheme (combining frequency and spatial consistency), curriculum-based data mixing, and a progressive three-stage parameter unfreezing training strategy. The resulting Being-VL-0.5 (8B) approaches the mainstream performance of continuous embedding methods in the discrete token paradigm.
tags:
  - "ICCV2025"
  - "Multimodal VLM"
  - "BPE Visual Tokenization"
  - "Discrete Visual Tokens"
  - "Multimodal LLM"
  - "Vocabulary Construction"
  - "Curriculum Training"
date: 2026-05-08
content_hash: 396364349adaaa18
---

# Unified Multimodal Understanding via Byte-Pair Visual Encoding

**Conference**: ICCV2025  
**arXiv**: [2506.23639](https://arxiv.org/abs/2506.23639)  
**Code**: [https://beingbeyond.github.io/Being-VL-0.5](https://beingbeyond.github.io/Being-VL-0.5)  
**Area**: Multimodal VLM  
**Keywords**: BPE Visual Tokenization, Discrete Visual Tokens, Multimodal LLM, Vocabulary Construction, Curriculum Training

## TL;DR
This paper applies the Byte-Pair Encoding (BPE) strategy from NLP to visual tokenization, proposing a priority-guided encoding scheme (combining frequency and spatial consistency), curriculum-based data mixing, and a progressive three-stage parameter unfreezing training strategy. The resulting Being-VL-0.5 (8B) approaches the mainstream performance of continuous embedding methods in the discrete token paradigm.

## Background & Motivation

**Background**: Multimodal Large Language Models (MLLMs) mainly follow two paradigms: (a) **continuous embedding** methods (such as LLaVA, Qwen-VL) use pre-trained visual encoders (such as CLIP) to map images to continuous vectors and align them to the LLM space via a projection layer; (b) **discrete token** methods (such as Chameleon, Unified-IO-2) use vector quantization (VQ-GAN/VQ-VAE) to discretize images into token sequences that are processed uniformly with text tokens.

**Limitations of Prior Work**:
   - Continuous methods suffer from a **modality gap** (mismatch between the high-dimensional continuous features of visual encoders and the discrete tokens expected by language models) and an **information bottleneck** (where the compression process discards low-frequency visual details, leading to hallucinations).
   - Although discrete methods naturally unify multimodal representations, simple VQ quantization ignores the semantic structure of visual content, key visual concepts are unevenly distributed in the token space, and current performance lags significantly behind continuous methods.

**Key Challenge**: BPE tokenization in NLP has been shown to significantly improve Transformer learning efficiency by merging frequently co-occurring characters into semantically rich tokens. While prior work (Being-VL-0) established a theoretical framework for applying BPE to visual data, transitioning from theory to practice faces three challenges: (1) designing encoding strategies beyond simple frequency; (2) constructing data mixing schemes that match the hierarchical nature of BPE; and (3) designing multi-stage training workflows.

**Goal**: To advance BPE visual tokenization from a theoretical concept to practical Multimodal Large Language Models, narrowing the performance gap between discrete token methods and continuous embedding methods.

**Key Insight**: BPE visual tokens naturally possess a hierarchical structure (lower-level tokens correspond to simple image patches, while higher-level tokens encode increasingly complex visual patterns). Thus, the training strategy should also be hierarchical and curriculum-based.

**Core Idea**: Priority-guided BPE vocabulary construction (frequency + spatial consistency) + curriculum data mixing + progressive parameter unfreezing = a practical MLLM using discrete visual tokens.

## Method

### Overall Architecture
The framework is divided into two phases. **Vocabulary Construction Phase**: Input training images $\rightarrow$ VQ-GAN quantization into discrete index grids $\rightarrow$ priority-guided BPE iteratively merges adjacent token pairs to expand the vocabulary. **Model Training Phase**: Input image $\rightarrow$ VQ-GAN quantization $\rightarrow$ BPE encoding (using the constructed vocabulary) $\rightarrow$ concatenate visual tokens and text tokens into a unified sequence $\rightarrow$ input into the expanded LLM for autoregressive modeling. The output is a text token sequence.

### Key Designs

1. **Priority-Guided Encoding**:

    - **Function**: Constructing a BPE visual vocabulary that considers both co-occurrence frequency and spatial relationships.
    - **Mechanism**: Define the priority function $P(a,b) = F(a,b) + \alpha \cdot S(a,b)$, where $F(a,b)$ is the co-occurrence frequency of the token pair $(a,b)$, and $S(a,b) = \frac{1}{N_{a,b}} \sum_{i=1}^{N_{a,b}} d(u_i(a,b), \bar{u}(a,b))$ is the spatial consistency score, measuring whether the relative positions of this token pair are consistent across different images (measured using a Gaussian kernel $d(u_1,u_2) = \exp(-\|u_1 - u_2\|^2 / 2\sigma^2)$). In each iteration, the token pair with the highest priority is merged into a new token until the vocabulary reaches the target size.
    - **Design Motivation**: Frequency-only BPE is effective in NLP because text is a one-dimensional sequence, whereas visual data is two-dimensional, making spatial relationships crucial. Even if a token pair co-occurs frequently, if its spatial relationship is inconsistent (e.g., located at different relative positions in different images), the merged token will lack a stable semantic meaning.

2. **Model Expanding**:

    - **Function**: Expanding the pre-trained text LLM to support visual tokens for a multimodal model.
    - **Mechanism**: Expand the embedding layer from $|V_{\text{text}}|$ to $|V_{\text{text}}| + |D|$ (default: 8K VQ + 8K BPE = 16K new tokens), where new embeddings are initialized using He initialization. The vocabulary of the output head is expanded simultaneously.
    - **Design Motivation**: Directly expanding the vocabulary instead of using an additional projection layer maintains the simplicity of the unified token representation.

3. **Multi-Stage Training**:

    - **Function**: Gradually releasing model capacity through three-stage training.
    - **Mechanism**:
        - **Stage 1 (Embedding Alignment)**: Train only the newly added visual token embeddings while freezing all LLM parameters. The dataset consists primarily of basic image-caption pairs.
        - **Stage 2 (Selective Fine-tuning)**: Unfreeze the first 25% of the Transformer layers, gradually incorporating perception tasks (detailed visual attribute descriptions).
        - **Stage 3 (Full Fine-tuning)**: Unfreeze all parameters, focusing on complex reasoning and instruction-following tasks.
    - **Design Motivation**: BPE tokens are hierarchical—lower-level tokens correspond to simple image patches, while higher-level tokens encode complex visual patterns. Training must align with this hierarchy: first enabling tokens to learn baseline semantic mapping, and then progressively tackling complex reasoning tasks. Compared to immediate full-parameter fine-tuning, this prevents catastrophic forgetting of language capabilities.

### Loss & Training
Standard autoregressive cross-entropy loss: $\mathcal{L}(\theta) = -\mathbb{E}_{(X,I,Y) \sim \mathcal{D}} [\sum_{i=1}^{|Y|} \log p_\theta(y_i | y_{<i}, X, T(Q(I)))]$. Data is classified into four categories: Foundation, Perception, Reasoning, and Instruction. Their mixing ratios are adjusted across three stages following a curriculum strategy.

## Key Experimental Results

### Main Results

| Model | Token Type | VQAv2 | MMBench | MME-P | SciQA | POPE | VizWiz |
|------|----------|-------|---------|-------|-------|------|--------|
| Being-VL-0.5 (ours) | Discrete | 80.2 | 71.8 | 1525.8 | 70.3 | 84.3 | 57.4 |
| Being-VL-0.5+ (16K) | Discrete | 80.6 | 72.1 | 1536.3 | 69.0 | 86.0 | 57.8 |
| Being-VL-0 (prev. work) | Discrete | 60.6 | 44.0 | 1316.2 | 64.3 | 81.3 | 48.2 |
| w/o BPE | Discrete | 54.3 | 38.2 | 1301.2 | 57.8 | 76.1 | 45.0 |
| LLaVA-1.5 | Continuous | 78.5 | 64.3 | 1510.7 | 66.8 | 85.9 | 50.0 |
| VILA-1.5 | Continuous | 80.9 | 72.3 | - | - | 84.4 | 58.7 |

### Ablation Study

| Configuration | Perception Avg | Reasoning Avg | Description |
|------|---------------|---------------|------|
| Full Scheme (Curriculum + Progressive) | **80.3** | **71.1** | Optimal |
| Progressive unfreezing only | 74.9 | 65.1 | No curriculum data, drops ~6% |
| Curriculum data only | 76.8 | 67.5 | No progressive unfreezing, drops ~4% |
| Single-stage training | 71.2 | 62.3 | Baseline, drops ~9% |

### Key Findings
- **BPE is the Core Contribution**: Removing the BPE vocabulary (w/o BPE) causes performance to collapse catastrophically across all benchmarks (e.g., VQAv2 drops from 80.2 to 54.3), demonstrating the critical importance of BPE visual tokenization.
- **Vocabulary Size Trade-off**: The 8K BPE vocabulary achieves the best balance between efficiency and performance. While the 16K vocabulary has higher scaling potential, some inactive tokens exist under current data scales (observed as white stripes in embedding visualizations).
- **Curriculum Data Outperforms Progressive Unfreezing**: In the ablation study, removing only the curriculum strategy drops performance by 6%, whereas removing only progressive unfreezing drops it by 4%. This indicates that BPE token learning is highly dependent on proper data ordering.
- **Discrete Methods Approaching Continuous Counterparts**: Being-VL-0.5 is already highly competitive with continuous methods like VILA-1.5 on VQAv2 (80.2 vs. 80.9) and MMBench (71.8 vs. 72.3).

## Highlights & Insights
- **Transferring NLP Success to Vision**: The massive success of BPE in text tokenization has proven indispensable for Transformer learning. This work systematically extends this idea to 2D visual data, bridging the prior gap between theory and practice.
- **Spatial Consistency is the Key Insight of Visual BPE**: While text BPE only needs to consider co-occurrence frequency, the 2D spatial structure of visual data requires token pairs to maintain consistent spatial relationships across different images. This key insight prevents visual BPE from being a naive replica of textual BPE.
- **Embedding Visualization Reveals a Unified Representation Space**: The embedding weight distribution in Figure 3 clearly shows how BPE tokens bridge the representation gap between visual and textual tokens, providing a valuable window into how discrete token-based methods function.

## Limitations & Future Work
- **Limited to the 8B Model Scale**: Due to computing resource constraints, the method has not been validated on larger models. Scaling analysis hints that larger vocabularies combined with more data could yield further improvements.
- **Restricted to Understanding Tasks**: Although discrete tokens naturally support generative tasks (such as generating visual tokens in the same manner as text tokens), image generation is not covered in this paper.
- **VQ-GAN acts as a Bottleneck**: Vocabulary construction heavily relies on the quantization quality of the VQ-GAN. If the VQ-GAN codebook quality is low, subsequent BPE processes cannot easily compensate.
- **Underutilized 16K Vocabulary**: Embedding visualizations show that a significant portion of BPE tokens are not activated, suggesting that current data scales are insufficient to support larger vocabularies.

## Related Work & Insights
- **vs. Being-VL-0**: The predecessor established the theoretical framework for visual BPE but relied on simple frequency encoding. This paper introduces spatial consistency, curriculum training, and progressive unfreezing, boosting MMBench performance from 44.0 to 71.8 (+27.8).
- **vs. Chameleon**: Chameleon uses simple VQ tokens without BPE, achieving only 56.2 on VQAv2 compared to our 80.2—a substantial performance gap. The structured tokenization of BPE is the key differentiator.
- **vs. LLaVA-1.5**: This continuous embedding method scores 78.5 on VQAv2 versus our 80.2, showing that our method has slightly outperformed it. This proves that discrete token solutions, when fully optimized, can robustly compete with continuous counterparts.

## Rating
- Novelty: ⭐⭐⭐⭐ The visual BPE direction itself builds on prior work, but the priority-guided encoding and training strategies are novel contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluation spans multiple benchmarks alongside detailed ablations and visualization analyses, though generative task validation is absent.
- Writing Quality: ⭐⭐⭐⭐ Structurally clear with a natural transition from theory to practice.
- Value: ⭐⭐⭐⭐ Demonstrates the viability of the discrete token paradigm, providing a practical pathway for unified multimodal representations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Harmonizing Visual Representations for Unified Multimodal Understanding and Generation](harmonizing_visual_representations_for_unified_multimodal_un.md)
- [\[ICCV 2025\] MUSE-VL: Modeling Unified VLM through Semantic Discrete Encoding](musevl_modeling_unified_vlm_through_semantic_discrete_encodi.md)
- [\[ACL 2026\] STELLA: A Multimodal LLM for Protein Functional Annotation via Unified Sequence-Structure Encoding](../../ACL2026/multimodal_vlm/stella_a_multimodal_llm_for_protein_functional_annotation_via_unified_sequence-s.md)
- [\[NeurIPS 2025\] UniTok: A Unified Tokenizer for Visual Generation and Understanding](../../NeurIPS2025/multimodal_vlm/unitok_a_unified_tokenizer_for_visual_generation_and_understanding.md)
- [\[CVPR 2026\] TUNA: Taming Unified Visual Representations for Native Unified Multimodal Models](../../CVPR2026/multimodal_vlm/tuna_taming_unified_visual_representations_for_native_unified_multimodal_models.md)

</div>

<!-- RELATED:END -->
