---
title: >-
  [Paper Note] Follow the Flow: On Information Flow Across Textual Tokens in Text-to-Image Models
description: >-
  [ACL 2026][Interpretability][Text-to-Image] This paper systematically investigates the token-level information distribution in the output of text encoders in Text-to-Image (T2I) models through a causal intervention frame…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "Text-to-Image"
  - "Information Flow"
  - "Token Representation"
  - "Semantic Leakage"
  - "Text Encoder"
date: 2026-05-08
content_hash: f3fda421be5edc79
---

# Follow the Flow: On Information Flow Across Textual Tokens in Text-to-Image Models

**Conference**: ACL 2026  
**arXiv**: [2504.01137](https://arxiv.org/abs/2504.01137)  
**Code**: [https://github.com/tokeron/lens](https://github.com/tokeron/lens)  
**Area**: Image Generation  
**Keywords**: Text-to-Image, Information Flow, Token Representation, Semantic Leakage, Text Encoder


## TL;DR
This paper systematically investigates the token-level information distribution in the output of text encoders in Text-to-Image (T2I) models through a causal intervention framework. It discovers that the semantics of lexical items are typically concentrated on 1-2 representative tokens, and that cross-item information flow leads to semantic leakage and image misinterpretation in 11% of cases. A simple and effective token-level intervention method is proposed to improve alignment.

## Background & Motivation

**Background**: Text-to-Image (T2I) models consist of two parts: a text encoder and a diffusion model. The text encoder transforms user prompts into representations that guide the diffusion process. Despite their wide use, T2I models frequently suffer from text-image misalignment, where generated images fail to accurately capture the objects and relationships in the text.

**Limitations of Prior Work**: Previous work has primarily attempted to improve alignment by modifying the diffusion process (especially cross-attention mechanisms), implicitly assuming that each text token reliably encodes its corresponding concept. However, this assumption has never been systematically verified—is information distribution in token representations uniform or concentrated? Is there information crossover between different lexical items?

**Key Challenge**: Many alignment improvement methods in T2I models (such as Attend-and-Excite) treat all tokens equally. However, if the information distribution is non-uniform or if semantic leakage exists between tokens, the effectiveness of these methods is fundamentally constrained.

**Goal**: To answer two fundamental questions—(1) is the semantics of a lexical item uniformly distributed across its tokens or concentrated on a few? (2) Does each token only encode its own lexical item, or does it also absorb information from neighboring items?

**Key Insight**: Using a causal intervention (patching) technique, some tokens' contributions are isolated by replacing other tokens with pad embeddings. Images are then generated to directly test what information the token encodes—this is more reliable than probing methods as it tests the information actually utilized by the diffusion model.

**Core Idea**: Reveal the patterns of information distribution in the text encoder through token-by-token causal intervention, and design token-level intervention methods based on these findings to improve T2I alignment.

## Method

### Overall Architecture
Given a T2I prompt, contextualized representations $h_1, \ldots, h_N$ for all tokens are obtained via the text encoder. Then, a patched sequence is constructed: representations of a target token subset $S$ are retained, while others are replaced with pad embeddings. The patched sequence is fed into the diffusion model to generate images, which are then evaluated by a VLM (Qwen2-VL-72B) to determine if they contain the target concept. This method is used to analyze two levels: in-item representation and cross-item interaction.

### Key Designs

1.  **Causal Patching**:
    - **Function**: To isolate and visualize the actual contribution of each token to the diffusion process.
    - **Mechanism**: Given a token subset $S$, a patched representation is constructed as $\tilde{t}_i = h_i$ if $i \in S$, and $\tilde{t}_i = p_i$ (pad embedding) otherwise. This representation is then used to guide the diffusion model for image generation. Whether a token is a "representative token" is determined by checking if the generated image contains the target concept. This is more direct than attention analysis or probing because it tests the information actually used by the downstream component.
    - **Design Motivation**: Probing methods may exploit spurious correlations, and attention analysis can be misleading; causal intervention validates information usefulness through actual generation.

2.  **In-Item Representation**:
    - **Function**: To determine how lexical semantics are distributed across its constituent tokens.
    - **Mechanism**: Patching is performed individually for each token of every lexical item to generate images and check for the lexical item's presence. In 89% of cases, there is at least one representative token, and usually only 1-2 tokens are sufficient to represent the entire concept (e.g., in "pelican," only "lic" among the three tokens represents the pelican). Non-representative tokens account for 52% of multi-token lexical items. Removing non-representative tokens not only avoids harming quality but reduces the generation failure rate by a relative 21%.
    - **Design Motivation**: To reveal the fundamental laws of token information distribution in T2I, providing a theoretical basis for subsequent interventions.

3.  **Cross-Item & Semantic Leakage Mitigation**:
    - **Function**: To detect information flow between different lexical items and identify/fix semantic leakage.
    - **Mechanism**: Images are generated for each lexical item under both contextualized and context-free conditions to judge if the contextualized representation has absorbed information from other items. Results show that lexical items remain isolated in 89% of cases, while 11% exhibit information flow. When information flow leads to misinterpretation (e.g., "pool" in "a pool by a table" is encoded as a billiard table instead of a swimming pool), it is fixed by replacing the leaked token's contextualized representation with its "clean" (context-free) representation, reducing the semantic leakage rate from 94% to 14% on FLUX-Schnell.
    - **Design Motivation**: Directly locates the root cause of alignment failure on the encoder side and provides a lightweight fix.

### Loss & Training
To achieve efficient identification of redundant tokens, the authors trained a lightweight single-layer linear classifier to predict redundancy directly from token embeddings. It achieves 90% precision and 83% accuracy, allowing for real-time filtering of redundant tokens during the encoding stage without requiring image generation.

## Key Experimental Results

### Main Results

| Number of Non-rep Tokens Removed | Prompt Count | Accuracy Before Removal | Accuracy After Removal | Unaffected | Improvement |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 144 | 81.25% | 83.33% | 98.29% | 14.83% |
| 2 | 98 | 82.65% | 88.78% | 100% | 35.27% |
| Total | 339 | 83.48% | 87.02% | 98.90% | 25.00% |

### Ablation Study

| Model | Initial Leakage Rate | RAG-Diffusion | Patching (Ours) |
| :--- | :--- | :--- | :--- |
| FLUX-Dev | 79% | 39% | 20% |
| FLUX-Schnell | 94% | 43% | 14% |

### Key Findings
- Lexical semantics are concentrated on 1-2 representative tokens; non-representative tokens account for 52% of multi-token items, and their removal relatively improves alignment by 21%.
- No cross-item information flow exists in 89% of lexical item pairs, but flow occurs in 11% of cases, especially with polysemous words prone to semantic leakage.
- Encoder type affects representative token positions: in bidirectional T5, they can appear anywhere; in unidirectional Gemma/CLIP, they are always the last token.
- The [CLS] token in CLIP encoders concentrates most semantic information, leaving other tokens with very weak information, which limits token-level interpretability.

## Highlights & Insights
- The finding that "removing non-representative tokens actually improves alignment" is counter-intuitive yet profound—it suggests a large number of "noise" tokens exist in T2I text encoder outputs that might distract the diffusion model. Simple token pruning can enhance alignment quality by 21%.
- The mechanism analysis of semantic leakage is insightful: in "a pool by a table," the representation of "pool" is contaminated by context to encode the "billiard table" concept, whereas it retains the "swimming pool" meaning in "a pool by a chair." This reveals a systematic failure mode in word sense disambiguation within text encoders.
- The generalization potential of the Patching method is noteworthy: the same mechanism can be used for polysemy control (user-selected word sense) and bias mitigation (e.g., eliminating gender-biased "runway" associations between fashion and airports), transforming an analytical tool into a practical generation control method.

## Limitations & Future Work
- Prompts are focused on object-centric simple syntax; generalization to spelling errors, rare words, or abstract concepts remains to be explored.
- While VLM judgments show high agreement with human judgments (Cohen's Kappa 0.868), they remain an approximate evaluation.
- The formation mechanism of representative tokens in bidirectional encoders (e.g., why "T" becomes the representative token in "T-shirt") is still an open question.
- The redundant token classifier was only trained and evaluated on FLUX-schnell; its transferability to other T2I models needs verification.

## Related Work & Insights
- **vs Attend-and-Excite (Chefer et al., 2023)**: Attend-and-Excite modifies attention during diffusion to improve alignment but assumes correct token encoding; this paper proves issues can stem from the encoding stage, making root-cause fixes more efficient.
- **vs RAG-Diffusion (Tan et al., 2024)**: RAG-Diffusion improves alignment by restricting diffusion attention with bounding boxes, but is less effective than this paper's patching method in semantic leakage scenarios (FLUX-Schnell: 43% vs. 14% leakage rate).
- **vs Patchscopes (Ghandeharioun et al., 2024)**: Patchscopes analyzes token information via representation decoding but doesn't test if the downstream component actually uses that information; this paper's causal intervention validates information effectiveness via image generation.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First systematic study of token-level info distribution in T2I text encoders, revealing representative tokens and semantic leakage.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Verified across 4 T2I models and 3 encoder types with human evaluation and quantitative analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Figures are highly intuitive, and the transition from analysis to application is natural.
- **Value**: ⭐⭐⭐⭐ Provides new perspectives and practical tools for researching T2I text encoders.
- **Reproducibility**: ⭐⭐⭐⭐ Code is open-sourced, experimental setup is clear, based on public models and datasets.
- **Impact**: ⭐⭐⭐⭐ Offers practical guidance for understanding and improving T2I alignment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] VITAL: More Understandable Feature Visualization through Distribution Alignment and Relevant Information Flow](../../ICCV2025/interpretability/vital_more_understandable_feature_visualization_through_distribution_alignment_a.md)
- [\[ACL 2026\] Compositional Steering of Large Language Models with Steering Tokens](compositional_steering_of_large_language_models_with_steering_tokens.md)
- [\[ACL 2026\] HistLens: Mapping Idea Change across Concepts and Corpora](histlens_mapping_idea_change_across_concepts_and_corpora.md)
- [\[ICLR 2026\] Concepts' Information Bottleneck Models](../../ICLR2026/interpretability/concepts_information_bottleneck_models.md)
- [\[CVPR 2026\] On the Possible Detectability of Image-in-Image Steganography](../../CVPR2026/interpretability/on_the_possible_detectability_of_image-in-image_steganography.md)

</div>

<!-- RELATED:END -->
