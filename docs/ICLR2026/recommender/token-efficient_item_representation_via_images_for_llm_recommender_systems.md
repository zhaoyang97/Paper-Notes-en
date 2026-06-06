---
title: >-
  [Paper Note] Token-Efficient Item Representation via Images for LLM Recommender Systems
description: >-
  [ICLR 2026][Recommender Systems][LLM-based recommendation] This paper proposes I-LLMRec, which leverages item images in place of verbose textual descriptions to represent item semantics in recommender systems. Through a…
tags:
  - "ICLR 2026"
  - "Recommender Systems"
  - "LLM-based recommendation"
  - "image representation"
  - "token efficiency"
  - "multimodal alignment"
  - "retrieval-based recommendation"
date: 2026-05-08
content_hash: 1bf272d5eb6dd81b
---

# Token-Efficient Item Representation via Images for LLM Recommender Systems

**Conference**: ICLR 2026
**arXiv**: [2503.06238](https://arxiv.org/abs/2503.06238)  
**Code**: [https://github.com/rlqja1107/torch-I-LLMRec](https://github.com/rlqja1107/torch-I-LLMRec)  
**Area**: LLM NLP / Recommender Systems
**Keywords**: LLM-based recommendation, image representation, token efficiency, multimodal alignment, retrieval-based recommendation

## TL;DR
This paper proposes I-LLMRec, which leverages item images in place of verbose textual descriptions to represent item semantics in recommender systems. Through a Recommendation-oriented Image-Semantic Alignment (RISA) module and a Recommendation-oriented Embedding Retrieval Inference (RERI) module, the method represents each item with a single token while preserving rich semantics, achieving approximately 2.93× inference speedup and surpassing text-description-based methods in recommendation performance.

## Background & Motivation

**Background**: LLM-based recommender systems must convert item interaction histories into natural language inputs. Existing approaches fall into two camps: attribute-based representation (e.g., brand + category — concise but semantically limited) and description-based representation (e.g., full product descriptions — semantically rich but token-intensive).

**Limitations of Prior Work**: Both approaches face a **fundamental efficiency–effectiveness trade-off**. Attribute-based methods consume fewer tokens but lose fine-grained semantics, causing a 13%+ drop in recommendation performance. Description-based methods are semantically rich but token-heavy (averaging 160 tokens per item), increasing LLM inference time by more than 2.5× with complexity scaling quadratically with the length of user interaction sequences.

**Key Challenge**: As long as items are represented in natural language, the tension — richer semantic representation → longer input → lower efficiency — is unavoidable.

**Goal**: How can one preserve rich item semantics while substantially reducing token consumption?

**Key Insight**: Through CLIP-based measurements, the authors find that **product images and textual descriptions exhibit substantial information overlap** in e-commerce datasets (cosine similarity ≈ 0.31 on Amazon Sports/Art datasets, higher than the 0.26 observed for carefully annotated image–text pairs in COCO). This suggests that images can encode most of the semantics of descriptions with far fewer tokens.

**Core Idea**: Replace verbose textual descriptions with a single image token to represent items, and bridge the gap between visual and linguistic spaces via a recommendation-oriented alignment strategy.

## Method

### Overall Architecture
I-LLMRec takes a user's item interaction history (a sequence of images) as input and outputs the next recommended item. The overall pipeline proceeds as follows: (1) extract features from each item image via a pretrained visual encoder (CLIP-ViT); (2) map features to the LLM embedding space via a learnable adapter (one token per item); (3) align visual and linguistic spaces via the RISA module; (4) retrieve candidate items from the item pool via the RERI module. LLM parameters are frozen; only the adapter and projectors are trained.

### Key Designs

1. **Image-to-LLM Mapping**

    - **Function**: Compress item images into a single token interpretable by the LLM.
    - **Mechanism**: For each interacted item $i$, the CLIP visual encoder extracts features $v_i = V(\mathbf{I}_i) \in \mathbb{R}^{d_v}$, which are then mapped to the LLM embedding dimension via an adapter network $M: \mathbb{R}^{d_v} \rightarrow \mathbb{R}^d$. The prompt format is `Title: ITEM_TITLE, Visual Representation: [VISUAL]`, where `[VISUAL]` is replaced by the adapted visual feature.
    - **Design Motivation**: This compresses item descriptions from an average of 160 tokens to just 1 token, reducing complexity from $O((|f(\mathbf{D}_i)||\mathcal{S}_u|)^2 d)$ substantially. Item titles (~10 tokens) are retained to provide a basic textual anchor.

2. **Recommendation-oriented Image-Semantic Alignment (RISA)**

    - **Function**: Train the adapter so that visual features align with the linguistic space in the context of recommendation.
    - **Mechanism**: Training data is constructed in an input–target format. The input consists of a user interaction prompt containing image features together with a question about the next item's attributes (e.g., "What brand is this user likely to consume next?"); the target is the corresponding answer. Four attribute types (brand / category / title / description) × five question templates yield 20 templates, one of which is randomly selected at each training step. The training objective is: $\mathcal{L}_{\text{RISA}} = \max_M \sum_{k=1}^{|y|} \log(P_{\theta,M}(y_k|x, y_{<k}))$
    - **Design Motivation**: Unlike generic visual–language alignment (e.g., UniMP), RISA is tailored to the recommendation setting, enabling the LLM to infer user preferences from images. Ablation studies confirm that RISA yields substantial gains (Hit@5: 0.395 → 0.432).

3. **Recommendation-oriented Embedding Retrieval Inference (RERI)**

    - **Function**: Transform the recommendation task into a retrieval task, directly retrieving relevant items from the item pool.
    - **Mechanism**: An instruction prompt is appended to the user interaction prompt to guide the LLM to generate a recommendation token [REC]. The final-layer hidden state $h(\text{[REC]})$ serves as the user preference representation. Projectors map both user representations and item visual features into a shared recommendation space, and an affinity score $r_{u,i}^{\text{Img}} = o_u^{\text{Img}} \circledast o_i^{\text{Img}}$ is computed and trained with binary cross-entropy loss.
    - **Design Motivation**: RERI addresses two issues: (a) title-generation-based recommendation cannot guarantee that the generated item exists in the item pool; (b) vocabulary-expansion-based item token prediction does not scale. The retrieval-based approach ensures both reliability and efficiency.

4. **Multi-feature Extension**

    - **Function**: Integrate ID embeddings (from SASRec) and text features alongside image features.
    - **Mechanism**: An independent projector pair $(F_u^*, F_i^*)$ is introduced for each feature type, and scores are aggregated by summation at inference time: $rec_u^k = \text{Top-k}(r_{u,i}^{\text{Img}} + r_{u,i}^{\text{CF}} + r_{u,i}^{\text{Text}})$
    - **Design Motivation**: Different features are complementary — image features excel for cold-start items, CF features are stronger for popular items, and text features supply additional semantics.

### Loss & Training
The overall training objective is: $\mathcal{L}_{final} = \mathcal{L}_{\text{RISA}} + \mathcal{L}_{\text{RERI}}^{\text{Img}} + \mathcal{L}_{\text{RERI}}^{\text{CF}} + \mathcal{L}_{\text{RERI}}^{\text{Text}}$

All loss weights are fixed at 1. LLM parameters are frozen; only the adapter $M$ and six projectors are trained. At inference, affinity scores from three feature types are summed and Top-k items are retrieved.

## Key Experimental Results

### Main Results
Evaluated on four Amazon datasets (Sports, Grocery, Art, Phone) against CF and LLM baselines:

| Method | Type | Sport Hit@5 | Sport NDCG@5 | Art Hit@5 | Phone Hit@5 |
|--------|------|-------------|--------------|-----------|-------------|
| SASRec | CF | 0.3841 | 0.3129 | 0.5374 | 0.4366 |
| TALLRec | Attribute LLM | 0.3801 | 0.2938 | 0.5663 | 0.4986 |
| A-LLMRec | CF+LLM | 0.4070 | 0.3352 | 0.5681 | 0.4502 |
| TRSR | Description LLM | 0.4302 | 0.3375 | 0.5841 | 0.5148 |
| UniMP | Image LLM | 0.4030 | 0.3364 | 0.5315 | 0.4427 |
| **I-LLMRec** | **Image LLM** | **0.4570** | **0.3711** | **0.5883** | **0.5156** |

I-LLMRec outperforms TRSR (description-based) on nearly all datasets and metrics while achieving approximately 2.93× faster inference.

### Ablation Study

| Configuration | Sport Hit@5 | Sport NDCG@5 | Note |
|---------------|-------------|--------------|------|
| RERI (Img only) | 0.3953 | 0.3043 | Image retrieval only, no alignment |
| +RISA | 0.4316 | 0.3403 | +9.2% after alignment |
| RISA+RERI (Img+CF) | 0.4491 | 0.3630 | Multi-feature improvement |
| Full model (Img+CF+Text) | 0.4570 | 0.3711 | Complete three-feature model |

### Key Findings
- **RISA is the core contribution**: Removing RISA causes Hit@5 to drop from 0.432 to 0.395, confirming that recommendation-oriented alignment is critical.
- **Adding descriptions on top of images does not help** (I-LLMRec+D ≈ I-LLMRec), validating the high information overlap between images and descriptions.
- **Cold-start and popular items are complementary**: Image features outperform on cold-start items, while CF features are stronger for popular items; combining both yields complementary gains.
- **Robustness to context window constraints**: When the context window is reduced to 256 tokens, TRSR performance degrades sharply, whereas I-LLMRec is largely unaffected.
- **Noise robustness**: Textual descriptions frequently contain noise such as HTML tags; images naturally circumvent this issue.

## Highlights & Insights
- **Inverting the information-overlap assumption**: Prior multimodal recommendation research treated image–text overlap as a nuisance; this paper exploits it to achieve "more with less" — replacing 160 text tokens with a single image token. This perspective shift is highly elegant.
- **Single-token item representation**: Representing each item with just one image token reduces the complexity scaling with user sequence length from quadratic to near-linear.
- **Extensible retrieval framework**: The RERI module is easily extended to arbitrary feature types by adding projector pairs, making it a plug-and-play design transferable to other multimodal retrieval scenarios.
- **Task-specific alignment**: Rather than pursuing generic alignment, RISA designs question–answer templates specifically for the recommendation setting. This task-oriented alignment strategy is generalizable to other vertical domains (e.g., medical QA, financial analysis).

## Limitations & Future Work
- **Image quality dependency**: System performance may degrade when product images are missing or of poor quality (a fallback strategy is discussed in the appendix).
- **Dataset scope**: Validation is limited to Amazon e-commerce datasets; effectiveness in domains where images are less informative (e.g., books, music) remains unknown.
- **Frozen visual encoder**: A frozen CLIP-ViT is used throughout; end-to-end fine-tuning of the visual encoder is not explored.
- **Simple score aggregation**: Multi-feature inference relies on straightforward summation without exploring more sophisticated fusion strategies (e.g., attention-based fusion).
- **Future directions**: Stronger visual models (e.g., SigLIP-2) or temporally aware image encoders for modeling user interest evolution could be explored.

## Related Work & Insights
- **vs. TALLRec**: TALLRec represents items via attributes and titles with LoRA fine-tuning — efficient but semantically limited. I-LLMRec addresses the semantic gap through images while maintaining efficiency.
- **vs. TRSR**: TRSR uses a large model to summarize descriptions before feeding them to a smaller model — semantically rich but token-heavy and noise-sensitive. I-LLMRec bypasses textual description issues entirely via images.
- **vs. UniMP**: Although UniMP also uses images, it relies on the generic visual–language alignment of pretrained multimodal LLMs, which underperforms the recommendation-specific RISA alignment in this setting.
- More broadly, this paper inspires a "multimodal compressed representation" direction: in any LLM application requiring long text input, if an alternative modality with substantial information overlap exists, a similar compression strategy may be applicable.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The perspective shift (treating information overlap as an asset rather than a liability) is clever, though replacing text with images is not entirely new in multimodal VLM research.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Four datasets, multi-dimensional analysis of efficiency, effectiveness, and robustness, extensive ablations, and cold-start vs. popular item breakdowns.
- **Writing Quality**: ⭐⭐⭐⭐ Logically clear, with intuitive motivation figures and well-articulated trade-off analysis.
- **Value**: ⭐⭐⭐⭐ Practically significant for efficiency optimization in LLM-based recommender systems; 2.93× speedup with simultaneous performance gains represent a meaningful contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Rejuvenating Cross-Entropy Loss in Knowledge Distillation for Recommender Systems](rejuvenating_cross-entropy_loss_in_knowledge_distillation_for_recommender_system.md)
- [\[AAAI 2026\] Tokenize Once, Recommend Anywhere: Unified Item Tokenization for Multi-domain LLM-based Recommendation](../../AAAI2026/recommender/tokenize_once_recommend_anywhere_unified_item_tokenization_for_multi-domain_llm-.md)
- [\[ICML 2026\] Can Recommender Systems Teach Themselves? A Recursive Self-Improving Framework with Fidelity Control](../../ICML2026/recommender/can_recommender_systems_teach_themselves_a_recursive_self-improving_framework_wi.md)
- [\[NeurIPS 2025\] Validating LLM-as-a-Judge Systems under Rating Indeterminacy](../../NeurIPS2025/recommender/validating_llm-as-a-judge_systems_under_rating_indeterminacy.md)
- [\[AAAI 2026\] From Parameter to Representation: A Closed-Form Approach for Controllable Model Merging](../../AAAI2026/recommender/from_parameter_to_representation_a_closed-form_approach_for_controllable_model_m.md)

</div>

<!-- RELATED:END -->
