---
title: >-
  [Paper Note] Long Story Short: Disentangling Compositionality and Long-Caption Understanding in Contrastive VLMs
description: >-
  [ACL 2026][Multimodal VLM][CLIP] This work systematically disentangles the relationship between "compositionality" and "long-caption understanding" in contrastive VLMs. It discovers that these two capabilities mutually p…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "CLIP"
  - "Compositional Reasoning"
  - "Long-caption Understanding"
  - "Data Quality"
  - "Position Encoding Freezing"
  - "Bidirectional Transfer"
date: 2026-05-08
content_hash: b8aa6256d9e3d6b1
---

# Long Story Short: Disentangling Compositionality and Long-Caption Understanding in Contrastive VLMs

**Conference**: ACL 2026 Findings  
**arXiv**: [2509.19207](https://arxiv.org/abs/2509.19207)  
**Code**: TBD  
**Area**: Multimodal VLM / Evaluation  
**Keywords**: CLIP, Compositional Reasoning, Long-caption Understanding, Data Quality, Position Encoding Freezing, Bidirectional Transfer

## TL;DR
This work systematically disentangles the relationship between "compositionality" and "long-caption understanding" in contrastive VLMs. It discovers that these two capabilities mutually promote each other, but this transfer is **extremely sensitive to training data quality and optimization strategies**. While using grounded, high-vocabulary-coverage long-caption data with full-parameter fine-tuning can simultaneously achieve peak performance in both areas, the low-quality synthetic captions of DAC/DCI and LoRA-based partial updates lead to failure on both fronts. LongCLIP's strategy of freezing the first 20 position embeddings, intended to preserve general alignment, actually restricts compositional learning. The "control model," LSS, fine-tuned on ShareGPT4V within the original 77-token context window using full parameters, outperforms LongCLIP.

## Background & Motivation

**Background**: Contrastive VLMs (CLIP / SigLIP / ALIGN) have become the de facto standard for multimodal learning, yet two long-standing issues remain: (1) **Poor compositionality**—CLIP often behaves like a "bag-of-words," showing insensitivity to attribute-object binding, relations, and word order, as exposed by benchmarks like ARO, Winoground, and SugarCREPE++ (SC++). (2) **Weak long-caption processing**—CLIP's 77-token context window is short, and its effective attention often spans only the first 20-30 tokens (Zhang et al. 2024a), leading to poor performance in long dense caption retrieval (DOCCI / Urban1k / ImageInWords).

**Limitations of Prior Work**: The field has long assumed that "compositional reasoning" and "long-caption understanding" are highly correlated; long captions naturally contain more attributes and relations, which should facilitate compositional learning, and conversely, models with strong compositionality should better parse long captions. However, empirically, these two lines of research have been **isolated**. Research on compositionality (NegCLIP / CE-CLIP / DAC / DCI) uses short captions with hard negatives, while research on long-captions (LongCLIP / DreamLIP) uses long captions without specifically strengthening compositionality. No study has systematically compared the cross-capability transferability of these two paradigms.

**Key Challenge**: (a) While progress is viewed in each line individually, putting them together reveals unexpected results. For instance, DAC/DCI are nearly saturated on the traditional ARO benchmark but perform worse than base CLIP on the newer SC++ (Spearman $r = -0.37$!). LongCLIP performs strongly on long captions but shows almost no improvement over CLIP on SC++. (b) This implies that either the ARO benchmark is no longer reliable, or the transfer from "compositional training $\implies$ long-caption understanding" does not exist.

**Goal**: To answer two questions through controlled experiments: (Q1) Can training for compositionality improve long-caption understanding? (Q2) Can training on long captions facilitate compositional generalization? The study aims to isolate the variables of **data quality, optimization strategy, and architectural constraints** to determine when transfer succeeds or fails.

**Key Insight**: The authors developed a control model named **LSS (Long Story Short)**. It fine-tunes a CLIP ViT-B/32 using ShareGPT4V long captions while strictly maintaining the original 77-token context window and employing full-parameter updates. This separates the "effect of long-caption data" from "architectural changes for extending the context window." LSS is trained on four long-caption datasets (sDCI / DOCCI / LN / ShareGPT4V) to perform ablation studies on critical data attributes such as scale, vocabulary coverage, caption length, syntactic complexity (Yngve), and annotation quality.

**Core Idea**: Transferability is real but only holds when (high-quality grounded long captions) and (full-parameter fine-tuning) are simultaneously satisfied. Architectural tricks used to preserve CLIP's general alignment (such as freezing position embeddings) act as a bottleneck for compositional learning.

## Method

### Overall Architecture
This is an **empirical analysis paper** that does not propose a new model architecture but instead constructs a series of comparative experiments to disentangle the variables of data, optimization, and architecture. The overall process is:

(a) **Select representative baselines**: NegCLIP / CE-CLIP / DAC$_{\text{LLM}}$ / DCI$_{\text{P1}}$ for compositionality; LongCLIP / DreamLIP for long-captioning; plus baseline CLIP ViT-B/32 and SigLIP.

(b) **Design the LSS control model**: Based on CLIP ViT-B/32, fully fine-tuned using 4×A100 GPUs with a batch size of 1024. LSS variants are trained on four long-caption datasets (sDCI / DOCCI / LN / ShareGPT4V) while strictly adhering to the 77-token context.

(c) **Unified benchmark suite**: Compositionality is evaluated via Winoground (WG) + SugarCREPE++ (SC++ including SA/RR/RO/RA/SO subcategories) + ARO (for comparison against "invalidated benchmarks"). Long-caption retrieval uses Urban1K / sDCI / DOCCI / IiW for I2T and T2I R@1. General alignment is checked via CIFAR10/100 / ImageNet classification + COCO/Flickr30k short-caption retrieval. All evaluations are zero-shot.

(d) **Multi-dimensional comparison**: Q1 (Compositional $\rightarrow$ Long-caption) / Q2 (Long-caption $\rightarrow$ Compositional) / ARO vs SC++ failure analysis / LSS comparisons across 4 datasets / Ablation of LongCLIP’s position embedding freezing / General capability trade-offs.

### Key Designs

1. **LSS Control Model: Separating Long-Caption Data from Extended Context Architecture**:
    - **Function**: To resolve the attribution confusion regarding whether LongCLIP's improvements stem from data or architecture.
    - **Mechanism**: LongCLIP simultaneously introduces ShareGPT4V data, extends the context from 77 to 248 tokens, and freezes the first 20 position embeddings. To isolate these, the authors trained **LSS = ShareGPT4V data + no context extension (77 tokens) + full-parameter fine-tuning**. Parameters (Table 5): lr=3e-6, warmup=150 steps, 3000 steps, ≈ 2.5 epochs.
    - **Design Motivation**: In VLM literature, architectural and data improvements are often bundled. LSS allows the authors to definitively state that the architectural trick of extending the context is secondary; the primary performance gains come from ShareGPT4V data and full-parameter updates.

2. **Multi-benchmark Disentanglement of Data vs. Optimization vs. Architecture**:
    - **Function**: To map the contribution of specific variables to attributes like data scale, vocabulary coverage, and update range.
    - **Mechanism**: (a) Comparing 4 long-caption datasets across 5 attributes (Table 3/8)—sDCI (7.6k images, 29% vocab, 94 Yngve) vs. DOCCI (15k images, 27% vocab, 75 Yngve, human-written) vs. LN (489k images, 24% vocab, human-written) vs. ShareGPT4V (1.2M images, 88% vocab, synthetic). (b) Training LSS variants on each. (c) Correlating attributes with performance—finding that no single attribute determines performance, but rather a synergy of "vocab coverage $\times$ length $\times$ grounding $\times$ scale."
    - **Design Motivation**: Previous work claimed "more is better" (DreamLIP) or "complexity is better" (sDCI). This comparison proves sDCI's high syntactic complexity (94.07) is ineffective without grounding and vocab coverage.

3. **Ablation of LongCLIP Position Embedding Freezing—LongCLIP$_{70}$**:
    - **Function**: To identify why LongCLIP fails to improve on SC++—is it the data or the position embedding freezing?
    - **Mechanism**: LongCLIP freezes the first 20 position embeddings and diminishes updates for indices 20-77. The authors constructed **LongCLIP$_{70}$**, truncating input to 70 words to force it to work within the first 77 tokens. Results (Figure 3) show LongCLIP$_{70}$'s long-caption retrieval performance collapses, and LSS surpasses it. This proves LongCLIP’s long-caption capability resides in the 77-248 range, which is freely trained, while SC++ stays stagnant because the 0-77 range is locked.
    - **Design Motivation**: This architectural intervention accurately demonstrates the trade-off between "preserving general alignment" and "restricting compositional learning."

### Loss & Training
No new loss is proposed; LSS uses the standard CLIP InfoNCE contrastive loss. Hyperparameters (Table 5): Batch size 1024, 4×A100 GPUs. LSS$_{ShareGPT4V}$ uses lr=3e-6, 3000 steps, ≈ 2.5 epochs. Max training time is 8 hours.

## Key Experimental Results

### Main Results
**Q1 + Q2 Comprehensive Table (Table 1)**: Compositionality (SC++ avg + WG) + Long-caption retrieval (Urban1K / sDCI / DOCCI / IiW avg):

| Model | SC++ avg | Winoground T | Long-cap retrieval avg | Notes |
|------|----------|--------------|----------------------|------|
| CLIP (baseline) | 53.3 | 17.2 | 67.0 | Starting point |
| SigLIP | 57.5 | 18.6 | 77.5 | Different loss/data |
| **DAC$_{\text{LLM}}$** | 44.0 | 12.6 | 48.5 | Worse than CLIP! |
| DCI$_{\text{P1}}$ | 51.3 | 12.1 | 56.3 | Only ARO strong |
| CE-CLIP | 56.3 | 12.3 | 68.1 | Moderate |
| **NegCLIP** | **63.7** | 16.4 | 73.4 | Best composition training |
| **LongCLIP-B** | 54.7 | 14.7 | **79.1** | Strong long-cap, weak SC++ |
| DreamLIP | 54.1 | 18.0 | **82.7** | Max backbone + pre-training |
| **LSS (Ours)** | 61.8 | 17.5 | 78.7 | 77 tokens matches LongCLIP |

**Key Findings**: (1) NegCLIP improves long-captioning (73.4), proving Q1 (Compositional $\rightarrow$ Long-caption transfer). (2) LSS improves compositionality (SC++ 61.8), proving Q2 (Long-caption $\rightarrow$ Compositional transfer). (3) DAC/DCI fail on both, and LongCLIP fails on composition, showing transfer is **sensitive to training settings**.

**ARO vs SC++ Comparison (Table 2)**: DAC$_{\text{LLM}}$ is nearly saturated on ARO but scores only 44.0 on SC++ (lower than CLIP’s 53.3). Spearman correlation $r = -0.37$ suggests ARO is no longer a reliable metric for true compositional ability.

### Ablation Study
**Effect of 4 Long-Caption Datasets on LSS (Table 9 / Figure 2)**:

| LSS Variant | Data Scale | Caption Length | Vocab cov | Yngve | SC++ avg | Long-cap avg | Comment |
|---------|---------|------------|-----------|-------|----------|--------------|------|
| LSS$_{\text{sDCI}}$ | 7.6K img | 40 words | 29% | **94.07** | 57.4 | 71.6 | Syntax high but poor grounding |
| LSS$_{\text{DOCCI}}$ | 14.6K img | **122 words** | 27% | 74.55 | 60.9 | **82.7** | Small but high-quality human labels |
| LSS$_{\text{LN}}$ | 489K img | 30 words | 24% | 61.70 | 61.6 | 70.7 | Fast SC++ convergence, weak long-cap |
| **LSS$_{\text{ShareGPT4V}}$** | **1.2M img** | 144 words | **87.72%** | 45.70 | **61.8** | 78.7 | Best overall (scale $\times$ vocab) |

**LongCLIP Position Freezing Ablation (Figure 3)**: When truncated to 70 words, LongCLIP’s performance on long-caption retrieval crashes across Urban1K/DOCCI, and LSS surpasses LongCLIP$_{70}$.

**General Capability Trade-off (Table 4)**: CLIP baseline IN1K=63.1; NegCLIP drops to 61.0; LongCLIP rises to 66.9 (due to position freezing); LSS=60.8 (slight drop).

### Key Findings
- **Bidirectional transfer exists but is conditional**: Requires (high-quality grounded long captions) $\cap$ (full-parameter fine-tuning). LSS$_{\text{ShareGPT4V}}$ achieves both SC++ 61.8 and Long-cap 78.7.
- **ARO benchmark is no longer valid**: It correlates negatively with SC++ ($r = -0.37$). Methods overfitted to ARO (DAC/DCI) fail on more modern benchmarks.
- **Data Quality > Data Scale**: DOCCI (14.6K images) matches ShareGPT4V (1.2M images) due to precise human labeling.
- **Position embedding freezing is a double-edged sword**: LongCLIP protects IN1K classification but locks position 0-77, preventing compositional learning.
- **Failure of DAC/DCI**: Due to LoRA updates restricting model capacity and synthetic captions lacking proper grounding.

## Highlights & Insights
- **Methodological Value of LSS**: The focus on "attribution analysis" rather than just reporting higher numbers provides a necessary calibration for the field.
- **Evidence of ARO's Invalidation**: The negative correlation with SC++ serves as hard-hitting evidence that the community should move away from rule-based caption benchmarks.
- **Architecture Intervention**: The LongCLIP$_{70}$ ablation is a precise way to separate "protective tricks" from "learning constraints."
- **Data Cheat Sheet**: The 5-attribute comparison (Table 3/8) shows **vocabulary coverage > Yngve syntactic complexity**.
- **Cross-architecture Validation**: Confirming conclusions via SigLIP ensures results are general multimodal learning laws rather than CLIP-specific artifacts.

## Limitations & Future Work
- **Scope**: Limited to contrastive VLMs; autoregressive generative VLMs (LLaVA/Qwen-VL) are not studied.
- **Reasoning**: Does not explore temporal or causal reasoning beyond surface compositionality (attributes/relations).
- **Evaluation**: Uses retrieval as a proxy for understanding; directly generative or fine-grained probing is not explored.
- **Optimization**: Did not combine NegCLIP's hard negative loss with LSS's long-caption data for a potentially "super" model.
- **Training**: Limited training steps (max 8 hours) leave uncertainty about the potential ceiling with large-scale training.

## Related Work & Insights
- **vs. NegCLIP**: NegCLIP uses hard negatives; LSS uses long captions. Both are independent but equivalent paths to better compositionality.
- **vs. DAC / DCI**: DAC/DCI use synthetic captions + LoRA; this work highlights their failure on modern benchmarks compared to full-parameter fine-tuning on grounded data.
- **vs. LongCLIP**: Proves LongCLIP’s strength is in the extended context, while its architectural "protection" inhibits compositional learning in the standard context.
- **vs. DreamLIP**: DreamLIP is stronger due to backbone size and full pre-training, but LSS shows that full fine-tuning on smaller backbones can still yield superior compositionality (61.8 vs. 54.1).

## Rating
- **Novelty**: ⭐⭐⭐ High-quality empirical study rather than a novel method paper.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive coverage across baselines, datasets, and ablations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear articulation of questions and conclusions; excellent visualization.
- **Value**: ⭐⭐⭐⭐ Important calibration for the community (retire ARO, emphasize data quality).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DocSeeker: Structured Visual Reasoning with Evidence Grounding for Long Document Understanding](../../CVPR2026/multimodal_vlm/docseeker_long_document_understanding.md)
- [\[AAAI 2026\] URaG: Unified Retrieval and Generation in Multimodal LLMs for Efficient Long Document Understanding](../../AAAI2026/multimodal_vlm/urag_unified_retrieval_and_generation_in_multimodal_llms_for.md)
- [\[CVPR 2026\] MSJoE: Jointly Evolving MLLM and Sampler for Efficient Long-Form Video Understanding](../../CVPR2026/multimodal_vlm/msjoe_jointly_evolving_mllm_and_sampler_for_efficient_long-form_video_understand.md)
- [\[CVPR 2026\] ReMoRa: Multimodal Large Language Model based on Refined Motion Representation for Long-Video Understanding](../../CVPR2026/multimodal_vlm/remora_multimodal_large_language_model_based_on_refined_motion_representation_fo.md)
- [\[CVPR 2026\] Scaling the Long Video Understanding of Multimodal Large Language Models via Visual Memory Mechanism](../../CVPR2026/multimodal_vlm/scaling_the_long_video_understanding_of_multimodal_large_language_models_via_vis.md)

</div>

<!-- RELATED:END -->
