---
title: >-
  [Paper Note] Follow the Flow: On Information Flow Across Textual Tokens in Text-to-Image Models
description: >-
  [ACL 2026][Image Generation][Text-to-Image] This paper systematically investigates token-level information distribution in text encoder outputs of text-to-image models through a causal intervention framework…
tags:
  - "ACL 2026"
  - "Image Generation"
  - "Text-to-Image"
  - "Information Flow"
  - "Token Representation"
  - "Semantic Leakage"
  - "Text Encoder"
date: 2026-05-08
content_hash: 30723f88cb9414ea
---

# Follow the Flow: On Information Flow Across Textual Tokens in Text-to-Image Models

**Conference**: ACL 2026  
**arXiv**: [2504.01137](https://arxiv.org/abs/2504.01137)  
**Code**: [https://github.com/tokeron/lens](https://github.com/tokeron/lens)  
**Area**: Image Generation  
**Keywords**: Text-to-Image, Information Flow, Token Representation, Semantic Leakage, Text Encoder


## TL;DR
This paper systematically investigates token-level information distribution in text encoder outputs of text-to-image models through a causal intervention framework, discovering that lexical item semantics are typically concentrated in 1-2 representative tokens, and that cross-item information flow leads to semantic leakage and image misinterpretation in 11% of cases. The paper proposes simple yet effective token-level intervention methods to improve alignment.

## Background & Motivation

**Background**: Text-to-image (T2I) models consist of a text encoder and a diffusion model, where the text encoder converts user prompts into representations that guide the diffusion process. Despite widespread use, T2I models frequently exhibit text-image misalignment, with generated images failing to accurately capture objects and relationships specified in text.

**Limitations of Prior Work**: Previous work primarily improved alignment by modifying the diffusion process (particularly cross-attention mechanisms), implicitly assuming that each text token reliably encodes its corresponding concept. However, this assumption has never been systematically verified—is the information in token representations uniformly distributed or concentrated? Does information cross between different lexical items?

**Key Challenge**: Many alignment improvement methods in T2I models (e.g., Attend-and-Excite) treat all tokens uniformly, but if information distribution is uneven or semantic leakage exists between tokens, the effectiveness of these methods is fundamentally constrained.

**Goal**: Answer two fundamental questions—(1) Is lexical item semantics uniformly distributed across all its tokens or concentrated in a few tokens? (2) Does each token encode only its own lexical item, or does it also absorb information from neighboring items?

**Key Insight**: Use causal intervention (patching) techniques to isolate the contribution of specific tokens by replacing other tokens with pad embeddings, then generate images to directly examine what information that token encodes—this is more reliable than probing methods because it tests information actually used by the diffusion model.

**Core Idea**: Reveal information distribution patterns in text encoders through token-by-token causal intervention, and design token-level intervention methods to improve T2I alignment accordingly.

## Method

### Overall Architecture
Given a T2I prompt, obtain contextualized representations $h_1, \ldots, h_N$ for all tokens through the text encoder. Then construct a patched sequence: preserve representations of target token subset $S$, replacing others with pad embeddings. Feed the patched sequence to the diffusion model to generate images, then use a VLM (Qwen2-VL-72B) to judge whether the generated image contains the target concept. This method is applied to analyze both in-item representation and cross-item interaction.

### Key Designs

1. **Causal Patching Framework**:

    - Function: Isolate and visualize each token's actual contribution to the diffusion process
    - Mechanism: Given token subset $S$, construct patched representation $\tilde{t}_i = h_i$ if $i \in S$, otherwise $\tilde{t}_i = p_i$ (pad embedding), then use this representation to guide diffusion model image generation. Determine whether a token is a "representative token" by judging whether the generated image contains the target concept. This is more direct than attention analysis or probing methods because it directly tests information actually used by downstream components
    - Design Motivation: Probing methods may exploit spurious correlations, attention analysis may be misleading; causal intervention verifies information utility through actual generation

2. **In-Item Representation Analysis**:

    - Function: Determine how lexical item semantics are distributed across constituent tokens
    - Mechanism: For each token of each lexical item, perform patching individually, generate images and judge whether they contain that lexical item. 89% of cases have at least one representative token, and typically only 1-2 tokens suffice to represent the entire concept (e.g., only "lic" among three tokens of "pelican" represents the pelican). Non-representative tokens account for 52% of multi-token lexical items. Removing non-representative tokens not only preserves quality but relatively reduces generation failure rate by 21%
    - Design Motivation: Reveals fundamental patterns of token information distribution in T2I, providing theoretical basis for subsequent interventions

3. **Cross-Item Information Flow Analysis & Semantic Leakage Mitigation**:

    - Function: Detect information flow between different lexical items, identify and repair semantic leakage
    - Mechanism: For each lexical item, generate images separately under contextualized and non-contextualized conditions, judging whether contextualized representations absorb information from other items. Results show 89% of cases maintain item isolation, 11% exhibit information flow. When information flow causes misinterpretation (e.g., "pool" encoded as billiard table rather than swimming pool in "a pool by a table"), repair by replacing the leaked token's contextualized representation with its "clean" (non-contextualized) representation, reducing semantic leakage rate from 94% to 14% on FLUX-Schnell
    - Design Motivation: Directly locates encoder-side root causes of alignment failures, providing lightweight repair mechanisms

### Loss & Training
To enable efficient redundant token identification, the authors trained a lightweight single-layer linear classifier that directly predicts redundancy from token embeddings, achieving 90% precision and 83% accuracy, enabling real-time filtering of redundant tokens during encoding without requiring image generation.

## Key Experimental Results

### Main Results

| Non-representative tokens removed | Num. prompts | Accuracy before removal | Accuracy after removal | Unaffected | Gain |
|----------------------------------|--------------|------------------------|----------------------|-----------|------|
| 1 | 144 | 81.25% | 83.33% | 98.29% | 14.83% |
| 2 | 98 | 82.65% | 88.78% | 100% | 35.27% |
| Total | 339 | 83.48% | 87.02% | 98.90% | 25.00% |

### Ablation Study

| Model | Initial leakage rate | RAG-Diffusion | Patching (Ours) |
|-------|---------------------|---------------|-----------------|
| FLUX-Dev | 79% | 39% | 20% |
| FLUX-Schnell | 94% | 43% | 14% |

### Key Findings
- Lexical item semantics are typically concentrated in 1-2 representative tokens, non-representative tokens account for 52% of multi-token items, and removing them relatively improves alignment by 21%
- 89% of lexical item pairs exhibit no cross-item information flow, but 11% of cases show information flow, with polysemous words particularly prone to semantic leakage
- Encoder type affects representative token position: in bidirectional T5, representative tokens can appear at any position; in unidirectional Gemma/CLIP, they are always at the last token
- CLIP encoder's [CLS] token concentrates most semantic information, resulting in very weak information in other tokens, limiting token-level interpretability

## Highlights & Insights
- The finding that "removing non-representative tokens actually improves alignment" is counterintuitive but profoundly impactful—this means there are many "noise" tokens in T2I model text encoder outputs, and diffusion models may be disturbed by this noise. Simple token pruning can improve alignment quality by 21%.
- The semantic leakage mechanism analysis is particularly compelling: in "a pool by a table," the representation of "pool" is contaminated by context and encodes the "billiard table" concept, while in "a pool by a chair" it maintains the "swimming pool" meaning. This reveals systematic failure modes in polysemous word disambiguation in text encoders.
- The generalization potential of the patching method deserves attention: the same mechanism can be used for polysemous word control (users actively select word sense) and bias mitigation (e.g., eliminating fashion/airport bias in "runway" due to gender context), transforming an analysis tool into a practical generation control mechanism.

## Limitations & Future Work
- Prompts focus on simple grammatical cases centered on objects; generalization to spelling errors, rare words, or abstract concepts remains to be explored
- VLM judgment, while having high agreement with human judgment (Cohen's Kappa 0.868), is still approximate evaluation
- The formation mechanism of representative tokens in bidirectional encoders (why "T" becomes the representative token in "T-shirt") remains an open question
- The redundant token classifier was trained and evaluated only on FLUX-schnell; its transferability to other T2I models needs verification

## Related Work & Insights
- **vs Attend-and-Excite (Chefer et al., 2023)**: Attend-and-Excite modifies attention during the diffusion stage to improve alignment, but implicitly assumes tokens encode correctly; this paper proves problems may originate in the encoding stage, making root-level repair more efficient
- **vs RAG-Diffusion (Tan et al., 2024)**: RAG-Diffusion improves alignment by constraining diffusion attention with bounding boxes, but in semantic leakage scenarios performs worse than this paper's patching method (FLUX-Schnell: 43% vs 14% leakage rate)
- **vs Patchscopes (Ghandeharioun et al., 2024)**: Patchscopes analyzes token information through representation decoding, but does not test whether downstream components actually use that information; this paper's causal intervention method directly verifies information utility through image generation

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic study of token-level information distribution in T2I text encoders, revealing representative token and semantic leakage phenomena
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across 4 T2I models and 3 encoder types, with human evaluation and quantitative analysis
- Writing Quality: ⭐⭐⭐⭐⭐ Illustrations are extremely intuitive, natural flow from analysis to application
- Value: ⭐⭐⭐⭐ Provides new perspectives and practical tools for T2I model text encoder research
- Recommendation: ⭐⭐⭐⭐ Has practical guidance value for understanding and improving T2I alignment

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] FlowTok: Flowing Seamlessly Across Text and Image Tokens](../../ICCV2025/image_generation/flowtok_flowing_seamlessly_across_text_and_image_tokens.md)
- [\[ICLR 2026\] SenseFlow: Scaling Distribution Matching for Flow-based Text-to-Image Distillation](../../ICLR2026/image_generation/senseflow_scaling_distribution_matching_for_flow-based_text-to-image_distillatio.md)
- [\[ICLR 2026\] Directional Textual Inversion for Personalized Text-to-Image Generation](../../ICLR2026/image_generation/directional_textual_inversion_for_personalized_text-to-image_generation.md)
- [\[ICLR 2026\] Intention-Conditioned Flow Occupancy Models](../../ICLR2026/image_generation/intention-conditioned_flow_occupancy_models.md)
- [\[ACL 2026\] ZipVoice-Dialog: Non-Autoregressive Spoken Dialogue Generation with Flow Matching](zipvoice-dialog_non-autoregressive_spoken_dialogue_generation_with_flow_matching.md)

</div>

<!-- RELATED:END -->
