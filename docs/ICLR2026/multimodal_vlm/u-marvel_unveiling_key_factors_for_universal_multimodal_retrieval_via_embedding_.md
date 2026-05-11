---
title: >-
  [Paper Note] U-MARVEL: Unveiling Key Factors for Universal Multimodal Retrieval via Embedding Learning
description: >-
  [ICLR 2026][Multimodal VLM][Universal Multimodal Retrieval] This work systematically ablates the design space of MLLM embedding learning, revealing key factors such as bidirectional attention + mean pooling outperforming…
tags:
  - "ICLR 2026"
  - "Multimodal VLM"
  - "Universal Multimodal Retrieval"
  - "MLLM Embedding Learning"
  - "Contrastive Learning"
  - "Progressive Training"
  - "Reranker Distillation"
date: 2026-05-08
content_hash: 7fb0199d9ec25e57
---

# U-MARVEL: Unveiling Key Factors for Universal Multimodal Retrieval via Embedding Learning

**Conference**: ICLR 2026
**arXiv**: [2507.14902](https://arxiv.org/abs/2507.14902)
**Code**: [GitHub](https://github.com/chaxjli/U-MARVEL)
**Area**: Multimodal VLM
**Keywords**: Universal Multimodal Retrieval, MLLM Embedding Learning, Contrastive Learning, Progressive Training, Reranker Distillation

## TL;DR

This work systematically ablates the design space of MLLM embedding learning, revealing key factors such as bidirectional attention + mean pooling outperforming the mainstream last-token approach, and learnable temperature being severely underestimated. Based on these findings, the authors construct U-MARVEL, a three-stage framework (progressive transition → filtered hard negatives → reranker distillation), achieving 63.2% Avg on M-BEIR with a single model, substantially surpassing existing SOTA, while also leading on zero-shot CIR and T2V transfer.

## Background & Motivation

**Background**: Universal Multimodal Retrieval (UMR) requires a single retriever to handle complex retrieval scenarios where both queries and candidates span text, images, and their combinations. Recent methods such as LamRA, MM-Embed, GME, and UniME all adopt MLLM + contrastive learning, but differ substantially in architectural choices (embedding extraction, training hyperparameters, negative sampling strategies), with no unified study addressing which design decisions truly matter.

**Limitations of Prior Work**: Decoder-only MLLMs are inherently designed for autoregressive generation; adapting them into embedding models involves numerous unexplored design choices. Existing methods almost universally follow the paradigm of last-token extraction + causal attention + compressed prompt (e.g., "Summarize in one word: emb"), yet whether this is optimal has never been systematically validated. Furthermore, while recall-then-rerank improves accuracy, it doubles inference cost, and no efficient single-model alternative has been established.

**Key Challenge**: Several seemingly minor details—attention directionality, pooling strategy, temperature parameterization—may have decisive effects on performance, yet the community lacks systematic understanding of their impact.

**Goal**: (1) Embedding extraction: what is the optimal way to adapt a decoder-only model into an embedder? (2) Training strategy: how do batch size, learning rate, and temperature interact in InfoNCE? How can hard negatives avoid training collapse? (3) Efficiency: can recall + rerank be distilled into a single model while preserving accuracy?

**Key Insight**: Rather than directly proposing a method, the authors first implement a general pipeline, then systematically ablate along three axes, deriving optimal designs from empirical evidence at each step, and finally assembling them into a unified framework. This "understand before build" paradigm ensures every design decision is data-driven.

**Core Idea**: Through systematic ablation, the paper identifies overlooked critical factors (bidirectional attention + mean pooling, learnable temperature, filtered hard negatives) and integrates them into a three-stage progressive training framework with efficient distillation, achieving single-model SOTA.

## Method

### Overall Architecture

U-MARVEL is built upon Qwen2-VL-7B-Instruct with LoRA fine-tuning. The overall pipeline consists of three stages: (1) **Progressive Transition**—adapting the decoder-only model into an embedding model through sequential steps from pure text retrieval to cross-modal alignment to instruction-guided multimodal retrieval; (2) **Hard Negative Mining**—continuing training after applying a filtering strategy to eliminate false negative noise; (3) **Reranker Distillation**—training a generative reranker, fusing it with the recall model, and distilling the combined teacher into a single student model via improved KL divergence. The input is a query of arbitrary modality (text / image / text+image), and the output is a unified embedding vector used for cosine-similarity-based retrieval.

### Key Designs

1. **Embedding Extraction Strategy (Core Finding)**

   - **Function**: Converts an MLLM's token sequence into a single embedding vector.
   - **Mechanism**: Existing methods predominantly use causal attention + compressed prompt ("Summarize in one word: emb") + last-token embedding. The authors systematically compare five combinations and find that **bidirectional attention + mean pooling + no compressed prompt** achieves the best performance (Local 57.2 vs. 56.6 for the mainstream approach). The key insight is that compressed prompts conflict with mean pooling: the prompt encourages the model to compress information into the final token, whereas mean pooling requires information to be distributed evenly across all tokens. Removing the prompt allows mean pooling to aggregate global information more effectively. Furthermore, mean pooling with bidirectional attention enables every token to attend to the full context, eliminating the recency bias of last-token extraction.
   - **Design Motivation**: This challenges the community's default last-token paradigm, aligning with NV-Embed's findings while contradicting GME's conclusions, and provides a new standard for embedding extraction.

2. **Instruction Integration and Masking**

   - **Function**: Handles the influence of instruction tokens during mean pooling.
   - **Mechanism**: Due to bidirectional self-attention, instruction tokens already influence the representations of all query tokens during the forward pass. Accordingly, instruction tokens are masked during mean pooling, and only the query portion of tokens is averaged. This prevents instruction information from being double-counted during pooling.
   - **Design Motivation**: Although the numerical gain is modest (+0.1%/+0.3%), it theoretically eliminates instruction bias, allowing the embedding to more purely reflect the semantic match between query and candidate.

3. **Progressive Transition Training**

   - **Function**: Smoothly adapts a decoder-only MLLM into a multimodal embedding model.
   - **Mechanism**: Training proceeds in three steps of increasing complexity—Step 1: train on NLI text-only data with unidirectional InfoNCE to establish semantic retrieval capability; Step 2: train on CC3M image-text pairs with bidirectional InfoNCE to achieve cross-modal alignment (switching from causal to bidirectional attention disrupts existing alignment and requires explicit reconstruction); Step 3: instruction fine-tuning on M-BEIR multimodal retrieval data. Experiments show that the concise captions in CC3M are better suited for retrieval than the detailed descriptions in ShareGPT4V.
   - **Design Motivation**: Direct fine-tuning on multimodal retrieval data leads to suboptimal results due to excessive task gap; the progressive strategy ensures each step builds smoothly on the previous one.

### Loss & Training

The training objective is the InfoNCE contrastive loss:

$$\mathcal{L}_{\text{InfoNCE}}=-\log\frac{\exp(\text{sim}(e_q,e_{c^+})/\tau)}{\sum_i\exp(\text{sim}(e_q,e_{c_i})/\tau)}$$

The authors identify strong interaction effects among three factors:

- **Batch Size + Learning Rate Scaling**: Simply increasing batch size without adjusting learning rate is nearly ineffective (480→1920: only +0.2%); with linear learning rate scaling, the gain becomes significant (+1.7%), consistent with the lr scaling rule in vision training.
- **Learnable Temperature >> Fixed Temperature**: Changing $\tau$ from a fixed value of 0.05 to a learnable parameter yields improvements of 1.2–1.4% under the same batch size. Learnable temperature adaptively adjusts the sharpness of the softmax distribution and is among the most overlooked critical factors in the community.
- **Hard Negative Filtering**: Directly using top-k hard negatives causes training collapse due to false negatives. The authors propose filtering out candidates with similarity above 0.7 (potentially unannotated positives) before selecting the top-5 as hard negatives, which are mixed with in-batch negatives. Filtering improves performance from 60.6 to 61.7.
- **Reranker Distillation**: A generative reranker (outputting YES/NO for each query-candidate pair) is trained and linearly fused with the recall model ($\alpha=0.5$) to produce teacher scores, which are then distilled into a single student model via KL divergence. The key improvement is restricting distillation to the top-k hard negative range per query rather than the full similarity matrix, reducing computation to 4.1% of the traditional approach (14h vs. 340h) while increasing training feature diversity by 26×.

## Key Experimental Results

### Main Results — M-BEIR Benchmark (Local Pool)

| Method | Type | $q^t→c^i$ | $q^t→c^t$ | $q^i→c^t$ | $(q^i,q^t)→c^i$ | Avg |
|--------|------|-----------|-----------|-----------|-----------------|-----|
| UniIR-CLIP | Single | 30.3 | 82.9 | 45.5 | 46.3 | 50.6 |
| LamRA-Ret | Single | 35.2 | 83.9 | 54.1 | 64.8 | 56.6 |
| GME-Qwen2VL-7B | Single | 37.7 | 83.3 | 55.2 | 67.5 | 58.6 |
| UniME | Single | 39.1 | 84.6 | 55.0 | 68.3 | 59.5 |
| **U-MARVEL** | **Single** | **40.2** | **85.0** | **58.3** | **72.1** | **63.2** |
| LamRA (+reranker) | Dual | 41.6 | 85.6 | 59.2 | 73.8 | 63.7 |
| **U-MARVEL⁺** (+reranker) | Dual | **41.8** | **85.6** | **63.7** | **73.9** | **64.8** |

U-MARVEL's single-model 63.2% nearly matches LamRA's dual-model 63.7%, validating the effectiveness of the distillation strategy. With a reranker, U-MARVEL⁺ reaches 64.8%, outperforming all baselines.

### Ablation Study — Contribution of Each Component

| Configuration | Local Avg | Global Avg | Note |
|---------------|-----------|------------|------|
| Baseline (causal + last token) | 56.6 | 54.8 | Mainstream default |
| + Bidir + Mean + no prompt | 57.2 | 55.2 | Embedding extraction, +0.6 |
| + Instruction masking | 57.3 | 55.5 | Eliminates instruction bias |
| + Progressive transition (NLI + CC3M) | 57.7 | 55.8 | Progressive pretraining, cumulative +1.1 |
| + Batch / LR / Temp optimization | 60.1 | — | Training parameter interaction, +2.4 |
| + Filtered hard negatives | 61.7 | 59.9 | Hard negative mining, +1.6 |
| + Reranker distillation | **63.2** | **60.7** | Distillation, +1.5 |

### Zero-Shot Transfer — CIR and T2V

| Method | CIRCO MAP@5 | MSR-VTT R@1 | MSVD R@1 |
|--------|-------------|-------------|----------|
| VLM2Vec | — | 43.5 | 49.5 |
| LamRA-Ret | 33.2 | 44.7 | 52.4 |
| LLaVE-7B | — | 46.8 | 52.9 |
| **U-MARVEL** | **36.2** | **47.2** | **54.6** |

Without any exposure to CIR or video data during training, U-MARVEL achieves zero-shot superiority over all compared methods, demonstrating the generalization capacity brought by progressive training.

### Key Findings

- **Bidirectional attention + mean pooling is the underestimated optimal embedding strategy**: The community's default of last-token + causal attention + compressed prompt is suboptimal. The root cause is that last-token extraction suffers from recency bias, whereas mean pooling with bidirectional attention enables every token to fully aggregate contextual information.
- **Learnable temperature is the most overlooked critical factor**: At batch size 3840, the gap between learnable and fixed temperature reaches 1.2%, exceeding the gain from increasing batch size from 480 to 3840.
- **Hard negatives must be filtered for false negatives**: Directly using top-k hard negatives inevitably leads to collapse; threshold-based filtering is a necessary measure.
- **Improved distillation brings single-model performance close to dual-model**: At only 4.1% of the traditional distillation cost, the single-model accuracy gap is reduced to just 0.5%.

## Highlights & Insights

- **"Understand before build" research paradigm**: Rather than directly proposing a method, the authors systematically ablate the impact of each design decision and then assemble a unified framework. This paradigm ensures every choice is empirically grounded, yielding more reliable and reproducible conclusions.
- **Three overlooked factors revealed in unison**: Bidirectional attention + mean pooling, learnable temperature, and filtered hard negatives appear to be minor adjustments, yet cumulatively yield a 6.6% absolute gain (56.6→63.2), demonstrating that "the devil is in the details" in MLLM embedding learning.
- **Efficient distillation design**: Restricting the distillation scope from the $O(n^2)$ full similarity matrix to the $O(nk)$ top-k range reduces computation to 4.1% while increasing feature diversity by 26×. This approach is transferable to knowledge distillation in any recall-then-rerank system.

## Limitations & Future Work

- **Limited modality coverage**: Only text and images are supported; audio and video are not addressed (though zero-shot video retrieval results are promising, temporal modeling is absent, and the reranker even degrades on video).
- **Model scale**: Validation is limited to the 7B scale; performance on larger (70B+) or smaller (1B) models remains unknown.
- **RAG scenarios not evaluated**: End-to-end effectiveness as a retriever in RAG pipelines is not assessed.
- **Hard negative threshold of 0.7 is manually set**: The optimal threshold may vary across data distributions; adaptive thresholding strategies could be explored.
- **Data selection in progressive transition**: The conclusion that CC3M outperforms ShareGPT4V may be confounded by differences in data scale and quality, warranting more rigorous controlled experiments.

## Related Work & Insights

- **vs. GME**: GME similarly builds on Qwen2-VL for universal multimodal retrieval but retains the last-token + causal attention paradigm. U-MARVEL's ablation directly challenges GME's conclusion that "last token outperforms mean pooling," suggesting GME's finding may be an artifact of its experimental design that does not remove the compressed prompt.
- **vs. LamRA**: LamRA achieves 63.7% with a dual-model recall + rerank setup, while U-MARVEL reaches 63.2% with a single model via distillation, offering substantially higher inference efficiency. LamRA employs a generative reranker, a design U-MARVEL adopts and extends with fusion distillation.
- **vs. NV-Embed**: NV-Embed independently identifies the advantage of bidirectional attention + mean pooling over last-token extraction in text-only embedding. U-MARVEL extends this finding to the multimodal setting and further uncovers the conflict mechanism between compressed prompts and pooling strategy.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Core contribution lies in systematic ablation rather than a fundamentally new architecture, though the revealed insights are highly valuable.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Ablations are exceptionally detailed, with controlled comparisons for every design decision; M-BEIR, zero-shot CIR, and T2V are comprehensively covered.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Well-structured with a fluid narrative progression from ablation to framework.
- **Value**: ⭐⭐⭐⭐ — Highly significant reference for the MLLM embedding learning community; multiple overlooked factors are directly actionable.
- False negatives in hard negatives cause collapse → threshold filtering is critical.

## Highlights & Insights
- **Systematic study**: Covers the complete design space → findings directly actionable in practice.
- **Overlooked factors revealed**: bidir + mean > last token; learnable temp >> fixed temp.
- **Practitioner-friendly**: Every finding comes with actionable recommendations.
- **Progressive transition**: Simple → complex training → smooth adaptation of decoder-only to embedding model.

## Limitations & Future Work
- Primarily validated on Qwen2-VL-7B → applicability to other MLLMs not verified.
- M-BEIR may not fully represent real-world UMR scenarios.
- Computational cost analysis of reranker distillation is not thoroughly discussed.
- Zero-shot evaluation limited to CIR and T2V → more tasks remain to be tested.

## Related Work & Insights
- NV-Embed identifies advantage of bidir + mean → independently confirmed in multimodal setting by this work.
- GME reaches opposite conclusion → possibly due to architectural/data differences → warrants further investigation.
- LamRA / MM-Embed / UniME → varying training strategies → unified comparison provided by this paper.
- Takeaway: adapting MLLMs for embedding involves numerous overlooked design choices with outsized impact on performance.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Systematic study + discovery of overlooked factors.
- **Technical Depth**: ⭐⭐⭐⭐ — Comprehensive ablation with well-reasoned analysis.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Detailed ablations covering every dimension.
- **Practicality**: ⭐⭐⭐⭐⭐ — Directly actionable guidance.
- **Overall**: ⭐⭐⭐⭐ — Practical contribution outweighs theoretical novelty.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] PPE: Positional Preservation Embedding for Token Compression in Multimodal Large Language Models](ppe_positional_preservation_embedding_for_token_compression_in_multimodal_large_.md)
- [\[NeurIPS 2025\] Retrv-R1: A Reasoning-Driven MLLM Framework for Universal and Efficient Multimodal Retrieval](../../NeurIPS2025/multimodal_vlm/retrv-r1_a_reasoning-driven_mllm_framework_for_universal_and_efficient_multimoda.md)
- [\[ICLR 2026\] Directional Embedding Smoothing for Robust Vision Language Models](directional_embedding_smoothing_for_robust_vision_language_models.md)
- [\[ICLR 2026\] BEAT: Visual Backdoor Attacks on VLM-based Embodied Agents via Contrastive Trigger Learning](beat_visual_backdoor_attacks_on_vlm-based_embodied_agents_via_contrastive_trigge.md)
- [\[ICLR 2026\] LLaVA-FA: Learning Fourier Approximation for Compressing Large Multimodal Models](llava-fa_learning_fourier_approximation_for_compressing_large_multimodal_models.md)

</div>

<!-- RELATED:END -->
