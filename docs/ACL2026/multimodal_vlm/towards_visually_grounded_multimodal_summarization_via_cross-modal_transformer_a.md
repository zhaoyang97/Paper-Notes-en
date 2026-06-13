---
title: >-
  [Paper Note] Towards Visually Grounded Multimodal Summarization via Cross-Modal Transformer and Gated Attention
description: >-
  [ACL2026][Multimodal VLM][Multimodal summarization] This paper proposes SPeCTrA-Sum, which integrates a hierarchically aligned Deep Visual Processor, gated cross-modal attention…
tags:
  - "ACL2026"
  - "Multimodal VLM"
  - "Multimodal summarization"
  - "visual grounding"
  - "image selection"
  - "DPP distillation"
  - "gated cross-attention"
date: 2026-05-08
content_hash: 50fb5619a590705a
---

# Towards Visually Grounded Multimodal Summarization via Cross-Modal Transformer and Gated Attention

**Conference**: ACL2026  
**arXiv**: [2605.11753](https://arxiv.org/abs/2605.11753)  
**Code**: https://github.com/abidmeeraj/SPeCTrA-Sum  
**Area**: Multimodal VLM / Multimodal Summarization  
**Keywords**: Multimodal summarization, visual grounding, image selection, DPP distillation, gated cross-attention

## TL;DR
This paper proposes SPeCTrA-Sum, which integrates a hierarchically aligned Deep Visual Processor, gated cross-modal attention, and a DPP-distilled image selector. This ensemble enables multimodal summarization to maintain near-SOTA ROUGE scores while selecting more relevant and diverse supporting images.

## Background & Motivation
**Background**: Multimodal summarization requires processing long text alongside supporting images, such as in news, blogs, or illustrated reports. Early methods typically prepended image features to text models or used attention mechanisms to assist generation. Recent VLM scaffolds like LLaVA-OneVision have further facilitated the joint utilization of image tokens and language models.

**Limitations of Prior Work**: Simple concatenation of visual tokens faces two primary issues. First, visual features usually originate from shallow visual encoders, while the deep hidden states of language models have undergone multiple layers of semantic transformation, leading to a mismatch in abstraction levels. Second, document images often contain redundancy or content irrelevant to the summary; inputting all images wastes attention and potentially introduces noise.

**Key Challenge**: Summarization models require visual grounding, but "more images are not always better." The model must deeply fuse truly useful visual cues while selecting a relevant and complementary set of images. Traditional text metrics like ROUGE struggle to directly reward the quality of this visual support.

**Goal**: The authors aim to unify summary generation and representative image selection within a single training framework, optimizing for text quality, visual relevance, and image diversity simultaneously.

**Key Insight**: The paper addresses these problems from two directions: using a Deep Visual Processor (DVP) to deepen visual representations alongside LLM layers to mitigate the mismatch between shallow visual features and deep language representations, and using a DPP teacher to generate relevance-diversity balanced soft labels for distilling a lightweight Visual Relevance Predictor (VRP), avoiding expensive DPP operations during inference.

**Core Idea**: Instead of treating images as prefix tokens crudely fed into the LLM, the model performs visual grounding at two levels: deep semantic alignment and output-level image selection.

## Method

### Overall Architecture
The input to SPeCTrA-Sum consists of a text $X$ and a set of images $I_1,...,I_M$, while the output is a summary $Y$ and a representative image subset $I^*$. The framework utilizes LLaVA-OneVision as the multimodal scaffold, with a frozen SigLIP encoder on the visual side and a Qwen-2 causal LM on the language side. While the base approach projects visual features into the token embedding space for concatenation, this work introduces a Vision Sampler, Deep Visual Processor (DVP), Layer-Aligned Gated Cross-Attention, and a Visual Relevance Predictor (VRP).

The training objective is multi-task: the primary task is autoregressive summarization, with auxiliary tasks including image-text alignment and DPP distillation. During inference, the model generates the summary while the VRP selects a set of images that best support the summary, avoiding treating all images as equivalent context.

### Key Designs
1. **Deep Visual Processor and Hierarchical Alignment Fusion**:
    - Function: Transforms visual features from shallow patch/token representations into semantic representations that match the hidden states of the LLM at different depths.
    - Mechanism: A Perceiver-style Vision Sampler first compresses the patch grid of each image into a fixed number of latent tokens. These visual latents then pass through a set of transformer blocks to obtain multi-level visual representations. Gated cross-attention is inserted every few LLM decoder layers, allowing interaction between visual tokens and language hidden states at corresponding depths.
    - Design Motivation: Pure concatenation causes visual tokens to remain in prefix positions, weakening their influence during deep decoding. DVP allows visual representations to "deepen" alongside the language layers, while gated residuals allow the model to learn from near-zero visual injection, reducing interference with the base LLM.

2. **DPP-distilled Visual Relevance Predictor**:
    - Function: Selects a subset of images that is both relevant and non-redundant, providing cleaner visual support for the summary.
    - Mechanism: During training, a DPP teacher generates soft inclusion probabilities for each image based on image-text relevance, inter-image RBF diversity, and target set size. The VRP is a two-layer MLP that takes normalized image embeddings and outputs selection logits, learning these soft labels via calibrated cross-entropy and cardinality regularization. During inference, images are scored independently without $O(K^3)$ DPP matrix operations.
    - Design Motivation: Selecting images by relevance alone often leads to duplicate images, while selecting by diversity alone may include irrelevant ones. DPP models the relevance-diversity trade-off, and distillation transfers this inductive bias to an efficient selector.

3. **Multi-objective Training Tieing Summarization, Alignment, and Selection**:
    - Function: Simultaneously optimizes text generation quality, image-text semantic consistency, and image set quality.
    - Mechanism: The total loss is summarized as $L_{MM}=L_{LM}+lambda_{align}L_{align}+lambda_{VRP}L_{DPP}$. $L_{LM}$ is the teacher-forced autoregressive summarization loss, $L_{align}$ performs SigLIP-style alignment between frozen visual embeddings and decoder mean-pooled representations, and $L_{DPP}$ fits the VRP to the DPP teacher's soft labels.
    - Design Motivation: If only text n-gram overlap is optimized, stronger visual processing may not improve ROUGE and might even interfere with language modeling. Multi-objective training explicitly incorporates the benefits of visual grounding into the optimization.

### Loss & Training
Training utilizes a batch size of 1 and the Adafactor optimizer. Training is controlled by steps, with approximately 295k steps per epoch; systems are trained up to 360k steps, with the best model selected via validation loss. Experiments were conducted on a single NVIDIA A100 80GB using 4-bit QLoRA-style quantization. VRP/DPP hyperparameters include a maximum of 3 selected images, RBF bandwidth of 0.8, relevance scaling of 2.0, target set size of 3.0, and subset-size regularization of 0.3. Architecture search covered Vision Sampler latent counts, depth, DVP layers, gated layer positions, and LoRA rank/alpha.

## Key Experimental Results

### Main Results

| Model | ROUGE-1 | ROUGE-2 | IP | MaxSim | MMAE | Description |
|--------|------|------|------|------|------|------|
| SITA | 43.64 | 20.53 | 76.41 | 33.47 | 3.37 | Strong baseline with highest image selection IP |
| ViL-Sum | 44.29 | 20.96 | 66.27 | 32.17 | 3.55 | Strongest text ROUGE baseline |
| DIUSum | 42.23 | 19.83 | - | - | - | Recent dynamic image utilization method |
| DVP (Ours) | 44.20 | 20.77 | 74.03 | 31.68 | 3.55 | ROUGE close to ViL-Sum, IP significantly higher than ViL-Sum |

| System | R-1 | R-2 | BERTScore | IP | CLIPScore | MMAE | PCD |
|------|------|------|------|------|------|------|------|
| OneVision | 43.81 | 20.52 | 89.58 | 74.02 | 70.62 | 3.5447 | 32.66 |
| Vision Sampler | 44.06 | 20.78 | 89.53 | 74.01 | 70.54 | 3.5484 | 32.65 |
| DVP | 44.20 | 20.77 | 89.33 | 74.03 | 70.52 | 3.5521 | 32.81 |

### Ablation Study

| Training Setting | System | R-1 | R-2 | BERTScore | Description |
|------|------|------|------|------|------|
| MaskedLM | OneVision | 44.26 | 20.86 | 89.12 | Highest text metrics |
| MaskedLM | Vision Sampler | 43.89 | 20.61 | 89.54 | ROUGE drops after adding visual sampling |
| MaskedLM | DVP | 43.81 | 20.58 | 89.50 | Deep visual processing does not automatically gain on pure text objectives |

| Human Eval Dimension | Mean (SD) | Score >=4 | Exact Agreement | Within-one Agreement | Interpretation |
|------|------|------|------|------|------|
| Text quality | 3.90 (0.69) | 80.1% | 49.0% | 90.0% | Good text coherence |
| Image relevance | 4.04 (0.80) | 76.8% | 44.3% | 84.0% | Strongest image-text relevance |
| Image diversity | 3.89 (0.83) | 73.2% | 43.0% | 82.2% | Diversity slightly lower but still positive |
| Overall quality | 4.00 (0.71) | 79.2% | 45.8% | 85.5% | Stable overall quality |

| Variant | Avg Latency | Latency Overhead | Peak VRAM | VRAM Overhead | Description |
|------|------|------|------|------|------|
| OV baseline | ~2110 ms | - | 15.80 GB | - | Simple concatenation |
| Vision Sampler | 2120 ms | +0.5% | 16.81 GB | +6.4% | Sampling adds almost no latency |
| DVP | 2322 ms | +10.0% | 22.56 GB | +42.8% | Significant VRAM cost for deep visual processing |
| MM-DVP | 2328 ms | +10.3% | 22.57 GB | +42.8% | Multi-objective training adds no extra inference cost |

### Key Findings
- DVP's text ROUGE nearly catches up with ViL-Sum: ROUGE-1 is only 0.09 lower and ROUGE-2 is 0.19 lower, but image selection IP reaches 74.03, significantly higher than ViL-Sum's 66.27.
- Multi-objective loss is critical. Under the MaskedLM objective, DVP's ROUGE is lower than OneVision, indicating that deeper visual modules do not naturally improve text metrics; DVP's comprehensive advantage only emerges after adding alignment and DPP distillation.
- Human evaluation shows that image relevance receives the highest average score (4.04), indicating that users perceive a better fit between summaries and images beyond automated metrics.
- Diversity metrics require careful interpretation. The paper notes that without relevance filtering, irrelevant images can artificially inflate pairwise cosine distance; DVP maintains the highest mean/max diversity after filtering.
- In terms of cost, DVP latency increases by only about 10%, but VRAM increases by 42.8%, which may limit deployment in low-memory scenarios.

## Highlights & Insights
- The paper identifies a frequently overlooked output-side problem in multimodal summarization: it is not just about generating text, but also about selecting supporting images for the reader. This task definition is closer to real-world news reading experiences than simple text-conditioned-on-images.
- The hierarchical alignment design of DVP is intuitive. Visual tokens are no longer just prefixes but continuously participate in semantic fusion at different decoding depths, making it suitable for migration to illustrated reports, document QA, and multi-image reasoning.
- The DPP teacher + VRP student approach is a practical compromise: it leverages set selection theory for relevance-diversity during training and uses a lightweight network approximation during inference to avoid expensive DPP calculations.
- The reflection on evaluation metrics is also significant. ROUGE is insensitive to visual grounding, and diversity can be inflated by irrelevant images, suggesting that multimodal summarization needs more granular evaluation of image-text consistency and complementarity.

## Limitations & Future Work
- Results are primarily based on MSMO, which focuses on news-style data. Scenarios like technical reports, social media long-posts, and scientific documents still require validation.
- Automated metrics remain insufficient. ROUGE measures text overlap, while IP/CLIPScore/PCD are proxies for visual quality; they cannot fully measure whether images truly help readers understand the summary.
- VRP performs text-free image scoring during inference for efficiency, which may miss "complementary relationships between images and the currently generated summary." Future work could explore conditional VRP or user-intent-aware selection.
- DVP has high VRAM overhead, increasing peak memory from 15.80GB to 22.56GB. Deployment in low-resource environments would require distillation, sparse injection, or lighter visual processors.
- The paper notes that similarity thresholds might filter out images with background value that are not directly related. Future work should model relevance, diversity, and complementarity simultaneously.

## Related Work & Insights
- **vs. Early Multimodal Summarization**: Methods like ATG/ATL/HAN included images but with shallower fusion; this work emphasizes hierarchical visual processing and output-level image selection.
- **vs. ViL-Sum / SITA**: ViL-Sum has higher ROUGE, and SITA has higher IP; SPeCTrA-Sum's advantage lies in approaching both strong baselines while focusing on grounding and diversity.
- **vs. Flamingo-style Gated Fusion**: This work borrows gated cross-attention but first aligns visual representations to the deep LLM via DVP before hierarchical injection, specifically targeting summarization tasks.
- **vs. DPP Image Selection**: Traditional DPP is suitable for set selection but computationally expensive; this work compresses DPP's set inductive bias into VRP via distillation, making it suitable for end-to-end systems.
- **Insight**: For multimodal generation tasks with "displayable visual evidence," optimization should not focus solely on the generated text. Treating evidence selection as a joint output makes the system more interpretable and closer to a final product form.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The combination of DVP + DPP distillation + multi-objective summarization is solid, and the task definition is complete.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Includes main results, ablations, human evaluation, and efficiency analysis; would be stronger with more datasets.
- Writing Quality: ⭐⭐⭐⭐☆ Module descriptions are clear, and tables are rich; some metric interpretations require familiarity with the MSMO evaluation system.
- Value: ⭐⭐⭐⭐☆ Highly valuable reference for multi-image document summarization, news aggregation, and visual evidence selection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] iVGR: Internalizing Visually Grounded Reasoning for MLLMs with Reinforcement Learning](../../ICML2026/multimodal_vlm/ivgr_internalizing_visually_grounded_reasoning_for_mllms_with_reinforcement_lear.md)
- [\[ACL 2026\] Cross-Modal Taxonomic Generalization in (Vision-) Language Models](cross-modal_taxonomic_generalization_in_vision-_language_models.md)
- [\[ACL 2026\] OMHBench: Benchmarking Balanced and Grounded Omni-Modal Multi-Hop Reasoning](omhbench_benchmarking_balanced_and_grounded_omni-modal_multi-hop_reasoning.md)
- [\[ACL 2026\] E2E-GMNER: End-to-End Generative Grounded Multimodal Named Entity Recognition](e2e-gmner_end-to-end_generative_grounded_multimodal_named_entity_recognition.md)
- [\[AAAI 2026\] Rethinking Visual Token Reduction in LVLMs under Cross-Modal Misalignment](../../AAAI2026/multimodal_vlm/rethinking_visual_token_reduction_in_lvlms_under_cross-modal_misalignment.md)

</div>

<!-- RELATED:END -->
