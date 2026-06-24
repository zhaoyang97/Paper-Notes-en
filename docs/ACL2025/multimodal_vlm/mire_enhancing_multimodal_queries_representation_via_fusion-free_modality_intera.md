---
title: >-
  [Paper Note] MIRe: Enhancing Multimodal Queries Representation via Fusion-Free Modality Interaction
description: >-
  [ACL 2025][Multimodal VLM][Multimodal Retrieval] This paper proposes the MIRe framework, which avoids direct fusion of text features during the vision-language alignment stage through "fusion-free modality interaction". It utilizes a query-guided attention pooling module to let text embeddings guide visual information extraction without feeding back text signals to the visual representation. This effectively mitigates the text-dominant issue in multimodal retrieval…
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Multimodal Retrieval"
  - "Modality Interaction"
  - "Late Interaction"
  - "Vision-Language Alignment"
  - "Text-Dominant Issue"
  - "Knowledge Retrieval"
date: 2026-05-08
content_hash: 6945c34e11847591
---

# MIRe: Enhancing Multimodal Queries Representation via Fusion-Free Modality Interaction

**Conference**: ACL 2025  
**arXiv**: [2411.08334](https://arxiv.org/abs/2411.08334)  
**Authors**: Yeong-Joon Ju, Ho-Joong Kim, Seong-Whan Lee (Korea University)  
**Code**: [GitHub](https://github.com/yeongjoonJu/MIRe)  
**Area**: Multimodal VLM  
**Keywords**: Multimodal Retrieval, Modality Interaction, Late Interaction, Vision-Language Alignment, Text-Dominant Issue, Knowledge Retrieval

## TL;DR

This paper proposes the MIRe framework, which avoids direct fusion of text features during the vision-language alignment stage through "fusion-free modality interaction". It utilizes a query-guided attention pooling module to let text embeddings guide visual information extraction without feeding back text signals to the visual representation. This effectively mitigates the text-dominant issue in multimodal retrieval, achieving zero-shot SOTA on four benchmarks.

## Background & Motivation

### Background
Multimodal query retrieval aims to retrieve relevant passages from a knowledge base based on composite queries containing both images and text. In real-world scenarios, users often attach visual references (such as images of complex objects or named entities) to their queries, and text alone is insufficient to fully convey the query intent.

### Limitations of Prior Work
Existing multimodal retrieval methods (such as ReViz, VISTA, and PreFLMR) typically **directly fuse the two modalities** for cross-referencing during the vision-language alignment stage:
- **Early token fusion** (ReViz/VISTA): Concatenating visual representations before text and interacting via self-attention layers.
- **Cross-attention fusion** (PreFLMR): Using text embeddings as keys/values for cross-modal attention.

These methods result in the **text-dominant issue**: models rely excessively on text-driven signals while ignoring key visual information. When a text query is ambiguous (e.g., replacing "beverage" with "object"), the model fails to utilize visual cues to compensate, leading to retrieval failure.

### Design Motivation
To design a modality interaction mechanism that does not fuse text features during the alignment stage, allowing the text query to "read" visual information without "writing" text signals back into the visual representation, thereby fundamentally mitigating the text-dominant issue.

## Method

### Overall Architecture
MIRe adopts a dual-encoder architecture: a text encoder $\mathcal{R}_T$ (ColBERTv2) generates token-level text embeddings $E_t$, and a visual encoder $\mathcal{R}_V$ (CLIP ViT) generates global embeddings $V_g$ (CLS token) and patch-level embeddings $V_m$ (penultimate layer). Finally, a late interaction mechanism (MaxSim) is used to calculate the relevance score between queries and passages.

### Key Design 1: Query-guided Attentive Pooling

This is the core innovative module of MIRe. Unlike standard cross-attention, this module uses text embeddings $E_t$ as the query and visual patch embeddings $V_m$ as the key/value to compute attention, but **does not feed back text signals to the visual representation via residual connections**:

$$\mathcal{A} = \text{Softmax}\left(\frac{E_t \cdot \mathcal{K}_m^\top}{\sqrt{d_t}}\right)$$

where $\mathcal{K}_m \in \mathbb{R}^{h \times l_v \times d_t}$ represented the key vectors of the visual patches after linear projection, and $h$ is the number of attention heads. Then, mean pooling is applied along the sequence dimension to aggregate and generate $h$ visual embeddings:

$$E_m = \text{Linear}\left(\frac{1}{l_t}\sum_{i}^{l_t}(\mathcal{A} \cdot \mathcal{V}_m)\right)$$

Key differences:
- **No residual connection**: Standard cross-attention blends query information into the output through residuals, whereas MIRe's pooling operation only uses $E_t$ to compute attention weights $\mathcal{A}$, without directly incorporating text features into the visual representation.
- **Mean pooling**: Taking the average along the sequence dimension outputs $h$ tokens instead of retaining all $l_t$ tokens.

### Key Design 2: Two-Stage Training Strategy

**Alignment Phase**: Freeze $\mathcal{R}_T$ and $\mathcal{R}_V$, and only train the projection layers and the attentive pooling module. Crucially, during this phase, **text embeddings $E_t$ are excluded**, and a query embedding consists solely of visual features: $E_Q = [E_g; E_m]$. This forces the model to learn effective visual representations aligned with passages.

**Inference/Downstream Fine-Tuning Phase**: Reintroduce text embeddings to the query: $E_Q = [E_g; E_m; E_t]$, and calculate the final relevance score using late interaction (MaxSim operation):

$$r_{Q,D} = \sum_{i=1}^{l_Q} \max_{j=1}^{l_D} (E_Q \cdot E_D^T)$$

Trained using contrastive loss:

$$\mathcal{L}_{CL} = -\sum_{\mathcal{D}} \log \frac{\exp(r_{Q,D}/\tau)}{\exp(r_{Q,D}/\tau) + \sum_{\bar{D} \in \bar{K}} \exp(r_{Q,\bar{D}}/\tau)}$$

### Key Design 3: Response-to-Passage Data Construction

Existing VQA datasets have overly concise answers, which are not suitable for training retrievers. MIRe proposes a response-to-passage transformation workflow:

1. Extract multimodal QA pairs $S = \{(I, T), R\}$ from visual dialog datasets.
2. Supplement simple answers with nouns from the query, and filter out yes/no-type answers.
3. Use answer $R$ as a query to retrieve top-k passages from Wikipedia using ColBERTv2.
4. Insert the answer $R$ between the retrieved passages to construct longer, more realistic "pseudo-passages": $R' = [D_1; R; D_2; \ldots; D_k]$.

Ultimately, 1.35 million QA pairs were constructed from 3 visual instruction datasets and 2 VQA datasets.

## Key Experimental Results

### Main Results: Zero-Shot Retrieval Performance (MRR@5)

| Method | OKVQA-GS | OKVQA-WK11M | ReMuQ | E-VQA |
|------|----------|-------------|-------|-------|
| CLIP | 19.08 | 16.45 | 0.34 | - |
| FLMR | 38.15 | 32.56 | 66.67 | 29.97 |
| ReViz | 45.77 | 44.03 | 23.61 | - |
| UniIR | 53.27 | - | 79.15 | 31.59 |
| VISTA | 55.33 | - | 78.32 | 33.90 |
| PreFLMR† | 59.38 | 45.68 | 52.27 | 30.92 |
| **MIRe** | **63.03** | **51.15** | **83.06** | **41.88** |
| MIRe (ViT-L) | 63.17 | 50.64 | 82.56 | 44.92 |

MIRe outperforms existing methods by a wide margin across all four benchmarks. Compared to PreFLMR† under the same data and settings, MRR@5 increases by 3.65 on OKVQA-GS and 10.96 on E-VQA, validating the effectiveness of the architectural design.

### Ablation Study (MRR@5, Zero-Shot)

| Variant | OK-GS | OK-WK | ReMuQ | E-VQA | Average |
|------|-------|-------|-------|-------|------|
| Full MIRe | 63.03 | 51.15 | 83.06 | 41.88 | 59.78 |
| Remove R2P data construction | 60.43 | 42.93 | 81.87 | 38.13 | 55.84 |
| Add residual connections in alignment phase | 61.65 | 47.95 | 80.47 | 43.06 | 58.28 |
| Add $E_t$ in alignment phase | 51.38 | 42.13 | 71.69 | 32.80 | 49.50 |
| Remove $E_t$ during inference | 36.99 | 36.68 | 2.73 | 11.39 | 21.95 |
| Remove $E_g$ and $E_m$ during inference | 52.46 | 36.00 | 71.69 | 42.48 | 50.66 |

Key finding: Adding $E_t$ during the alignment phase causes the average MRR@5 to plummet from 59.78 to 49.50, directly validating the existence of the text-dominant issue.

### Experiment 3: Fine-Tuning Performance (PR@5 / R@5)

| Method | OKVQA-GS (PR@5) | ReMuQ (R@5) |
|------|-----------------|-------------|
| FLMR | 70.63 | 62.76 |
| VISTA | 82.06 | 96.30 |
| MIRe (w/o pre-training) | 74.26 | 92.44 |
| **MIRe** | **83.59** | **94.40** |
| MIRe (ViT-L) | 84.66 | 94.38 |

## Key Findings

1. **Quantitative validation of the text-dominant issue**: Directly fusing $E_t$ during the alignment phase drops average performance by 17%, and merely adding residual connections drops it by 2.5%. Faster convergence but worse performance indicates that the model takes a "shortcut" by relying purely on textual similarity.
2. **Division of labor in visual embeddings**: $E_g$ (global) and $E_m$ (query-guided) capture complementary information, as removing both simultaneously causes a much larger performance degradation than removing either individually.
3. **Crucial role of R2P data construction**: Removing R2P drops the MRR@5 on OK-WK from 51.15 to 42.93 (-16%), making it the most impactful data design choice.
4. **Hard negative effect of multiple QA pairs**: Taking only one QA pair per image (rather than multiple) results in significant performance degradation, suggesting that multiple QA pairs provide hard negative signals beyond simple visual alignment.

## Highlights & Insights

- **Elegant Problem Definition & Solution**: Attributing the text-dominant issue to direct fusion during the alignment phase and fundamentally resolving it via "one-way attention" (text $\rightarrow$ vision). The approach is simple yet powerful.
- **Two-Stage Embedding Strategy**: Using only visual embeddings during the alignment phase to force vision alignment, and re-incorporating text embeddings during inference, successfully balancing multimodal complementarity and the text-dominance conflict.
- **R2P Data Construction**: Converting short QA answers into long, paragraph-style training data utilizing only a text retriever, requiring no extra annotation or generative models, which is cost-effective and highly effective.
- **Comprehensive Ablation and Visualization**: UMAP clustering and attention maps intuitively demonstrate how query-guided pooling focuses on different visual regions based on different text queries.

## Limitations & Future Work

- **Evaluation Limited to General Domains**: No evaluations were conducted on specialized domains like medicine or law, where multimodal content might exhibit different modality interaction patterns.
- **No Integration with RAG Pipelines**: Whether retrieval improvements transfer to downstream generation tasks (e.g., VQA answer generation) remains unverified.
- **Knowledge Base Dependency**: The R2P data construction relies on Wikipedia, which might require additional adaptation for dynamically updated or domain-specific knowledge bases.
- **Limited Model Scale**: Based on BERT-base (211M parameters), the system does not explore the performance of larger pre-trained language models (like LLaMA) as the text encoder.
- **Unclear Role of WiT Data**: Ablation study shows that removing WiT only slightly affects general benchmarks, but is crucial for knowledge-intensive tasks like Infoseek, indicating that the model's world-knowledge acquisition mechanism needs further optimization.

## Related Work & Insights

- **FLMR (Lin et al., 2023)**: Also uses a late interaction mechanism but enhances visual query representation by generating captions and RoIs, without addressing the text-dominant issue. MIRe improves MRR@5 from 38.15 to 63.03 on zero-shot OKVQA-GS.
- **PreFLMR (Lin et al., 2024)**: An extended version of FLMR. Trained under the same data and settings, it still lags significantly behind MIRe (OKVQA-GS: 59.38 vs 63.03), proving that structural differences are key.
- **ReViz (Luo et al., 2023)**: An end-to-end system that uses VL-ICT to build pseudo-queries from passages for pre-training. However, this method exacerbates the text-dominant issue (since pseudo-queries can match passages by themselves without the help of vision).
- **VISTA (Zhou et al., 2024)**: An early-fusion strategy that prepends visual tokens to the text retriever. Although it shows strong performance, it is still constrained by text dominance.
- **UniIR (Wei et al., 2024)**: An instruction-guided multimodal retriever requiring explicit instruction input. Its architecture is more complex, yet its performance is inferior to MIRe.
- **ColBERT/ColBERTv2**: The text encoder backbone for MIRe, providing token-level matching capabilities via late interaction. MIRe extends multimodal support based on this backend.

## Rating

- Novelty: ⭐⭐⭐⭐ — The "fusion-free modality interaction" concept is simple and effective, though its core remains a variant of the attention mechanism.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive coverage including four benchmarks, zero-shot and fine-tuning, full ablation studies, visualization analysis, convergence curves, and embedding distributions.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation, intuitive diagrams, but some symbolic definitions are slightly redundant.
- Value: ⭐⭐⭐⭐ — Revealing the text-dominant issue and providing a solution offers valuable insights for the multimodal retrieval community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Hints of Prompt: Enhancing Visual Representation for Multimodal LLMs in Autonomous Driving](../../ICCV2025/multimodal_vlm/hints_of_prompt_enhancing_visual_representation_for_multimodal_llms_in_autonomou.md)
- [\[NeurIPS 2025\] Structure-Aware Fusion with Progressive Injection for Multimodal Molecular Representation Learning](../../NeurIPS2025/multimodal_vlm/structure-aware_fusion_with_progressive_injection_for_multimodal_molecular_repre.md)
- [\[ACL 2025\] Enhancing Multimodal Continual Instruction Tuning with BranchLoRA](branchlora_continual_instruction.md)
- [\[ACL 2025\] DALR: Dual-level Alignment Learning for Multimodal Sentence Representation Learning](dalr_dual-level_alignment_learning_for_multimodal_sentence_representation_learni.md)
- [\[CVPR 2025\] Free on the Fly: Enhancing Flexibility in Test-Time Adaptation with Online EM](../../CVPR2025/multimodal_vlm/free_on_the_fly_enhancing_flexibility_in_test-time_adaptation_with_online_em.md)

</div>

<!-- RELATED:END -->
