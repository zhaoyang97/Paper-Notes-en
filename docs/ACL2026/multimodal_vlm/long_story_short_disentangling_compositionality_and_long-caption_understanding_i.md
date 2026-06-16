---
title: >-
  [Paper Note] Long Story Short: Disentangling Compositionality and Long-Caption Understanding in Contrastive VLMs
description: >-
  [ACL 2026][Multimodal VLM][CLIP] This paper systematically disentangles the relationship between "compositionality" and "long-caption understanding" in contrastive VLMs. It discovers that these two capabilities are bidirectionally mutually promotive, but this transfer is **extremely sensitive to training data quality and optimization strategies**: usi
tags:
  - ACL 2026
  - Multimodal VLM
  - CLIP
date: 2026-05-08
content_hash: 3a2e596961ec0807
---
# Long Story Short: Disentangling Compositionality and Long-Caption Understanding in Contrastive VLMs

**Conference**: ACL 2026 Findings  
**arXiv**: [2509.19207](https://arxiv.org/abs/2509.19207)  
**Code**: To be confirmed  
**Area**: Multimodal VLM / Evaluation  
**Keywords**: CLIP, Compositional Reasoning, Long-caption Understanding, Data Quality, Positional Encoding Freezing, Bidirectional Transfer

## TL;DR
This paper systematically disentangles the relationship between "compositionality" and "long-caption understanding" in contrastive VLMs. It discovers that these two capabilities are bidirectionally mutually promotive, but this transfer is **extremely sensitive to training data quality and optimization strategies**: using grounded, high-vocabulary-coverage long-caption data with full-parameter fine-tuning achieves excellence in both capabilities. Conversely, low-quality synthetic captions from DAC/DCI combined with LoRA updates lead to failure in both. While LongCLIP's freezing of the first 20 positional embeddings seemingly protects general alignment, it severely restricts compositional learning—the authors' control model, LSS, outperforms LongCLIP by fine-tuning on ShareGPT4V with full parameters within the original 77-token context window.

## Background & Motivation

**Background**: Contrastive VLMs (CLIP / SigLIP / ALIGN) are the de facto standards for multimodal learning but face two long-standing issues: (1) **Poor compositionality**—CLIP often behaves like a "bag-of-words," being insensitive to attribute-object binding, relations, and word order, as exposed by benchmarks like ARO, Winoground, and SugarCREPE++ (SC++); (2) **Weak long-caption processing**—CLIP's 77-token context window is short, and its effective attention often only reaches the first 20-30 tokens (Zhang et al. 2024a), leading to poor performance in long dense caption retrieval (DOCCI / Urban1k / ImageInWords).

**Limitations of Prior Work**: The field has long assumed that "compositional reasoning" and "long-caption understanding" are highly correlated—long captions naturally contain more attributes/relations, which should promote compositional learning; conversely, models with strong compositionality should better disentangle long captions. However, empirically, these two lines of research are **fragmented**: studies on compositionality (NegCLIP / CE-CLIP / DAC / DCI) use short captions with hard negatives, while studies on long-captions (LongCLIP / DreamLIP) use long captions without specifically reinforcing compositionality. No study has systematically compared the cross-capability transferability of these two lines.

**Key Challenge**: (a) While progress is made in each line individually, combining them yields unexpected results—for instance, DAC/DCI are nearly saturated on the traditional ARO benchmark but perform worse than base CLIP on the newer SC++ (Spearman $r = -0.37$!). LongCLIP excels at long captions but shows almost no improvement over CLIP on SC++. (b) This implies at least one of two things: either the ARO benchmark is no longer reliable, or the transfer from "compositional training $\implies$ long-caption understanding" does not exist.

**Goal**: To answer two questions through controlled experiments: (Q1) Can training for compositionality improve long-caption understanding? (Q2) Can training on long captions promote compositional generalization? Furthermore, the study aims to isolate the variables of **data quality, optimization strategy, and architectural constraints** to determine when transfer succeeds or fails.

**Key Insight**: The authors trained a control model **LSS (Long Story Short)**—fine-tuning CLIP ViT-B/32 using ShareGPT4V long captions while strictly maintaining the original CLIP 77-token context and full-parameter updates. This decouples the "effect of long-caption data" from "architectural modifications for context expansion." LSS was trained separately on four long-caption datasets (sDCI / DOCCI / LN / ShareGPT4V) for ablation studies to identify which data attributes (scale, vocabulary coverage, caption length, syntactic complexity Yngve, annotation quality) are truly critical.

**Core Idea**: Transferability is real but only holds when (high-quality grounded long captions) and (full-parameter fine-tuning) are satisfied simultaneously. Architectural tricks intended to preserve CLIP's general alignment (such as freezing positional embeddings) act as shackles for compositional learning.

## Method

### Overall Architecture
This is an **empirical analysis paper**—it does not propose a new model architecture but constructs a series of comparative experiments to disentangle the three variables of data, optimization, and architecture. The overall workflow includes:

(a) **Selecting representative baselines**: From the compositional training side: NegCLIP / CE-CLIP / DAC$_{\text{LLM}}$ / DCI$_{\text{P1}}$; from the long-caption training side: LongCLIP / DreamLIP; and baseline CLIP ViT-B/32 + SigLIP.

(b) **Designing the control model LSS**: Based on CLIP ViT-B/32, utilizing full-parameter fine-tuning on 4 $\times$ A100 GPUs with a batch size of 1024. LSS is trained separately on four datasets (sDCI / DOCCI / LN / ShareGPT4V) while strictly adhering to a 77-token context.

(c) **Unified benchmark suite**: Compositionality is evaluated using Winoground (WG) + SugarCREPE++ (SC++ including SA/RR/RO/RA/SO subcategories) + ARO (to contrast with failing traditional benchmarks). Long-caption retrieval is assessed via Urban1K / sDCI / DOCCI / IiW for both I2T and T2I R@1. General alignment is evaluated using CIFAR10/100 / ImageNet classification + COCO/Flickr30k short caption retrieval. All evaluations are zero-shot.

(d) **Multi-dimensional comparison**: Q1 (compositional training $\to$ long-caption) / Q2 (long-caption training $\to$ compositionality) / failure analysis of ARO vs. SC++ / comparison of LSS across four datasets / ablation of LongCLIP's positional embedding freezing / general capability trade-offs.

### Key Designs

**1. LSS Control Model: Decoupling "Long-Caption Data" from "Context Expansion Architecture"**

LongCLIP simultaneously introduced three changes: switching to ShareGPT4V long-caption data, expanding the context from 77 to 248 tokens, and freezing the first 20 positional embeddings to mitigate catastrophic forgetting. These variables are tangled, making it unclear where the improvements originate. The authors trained LSS as a clean control: it retains only "ShareGPT4V long-caption data + full-parameter fine-tuning," avoids context expansion (keeping 77 tokens), and does not freeze positional embeddings, thereby eliminating the latter two variables. Training configurations: lr=3e-6, 150-step warmup, 3000 total steps ($\approx$ 2.5 epochs).

**2. Multi-benchmark Cross-section: Attributing Training Variable Contributions to Specific Data/Optimization Properties**

To determine what governs transferability, training settings are decomposed into quantifiable attributes. The four long-caption datasets were tabulated by five attributes: sDCI (7.6K images, 29% vocab, 94 Yngve), DOCCI (15K images, 27% vocab, 75 Yngve, human-written), LN (489K images, 30-word short captions, 24% vocab, human-written), and ShareGPT4V (1.2M images, 144-word long captions, 88% vocab, synthetic). No single attribute determines performance; rather, it's a synergy of vocab coverage $\times$ length $\times$ grounding $\times$ scale $\times$ syntactic complexity. This refutes single-factor narratives like "more data is better" or "syntax complexity is better."

**3. LongCLIP Positional Freezing Ablation (LongCLIP$_{70}$): Identifying the Cause of SC++ Stagnation**

LongCLIP shows almost no improvement on SC++. Is it due to data or positional embedding freezing? Considering LongCLIP freezes the first 20 embeddings and discounts updates for tokens 20-77, the authors noted that SC++ samples mostly fall within the first 77 tokens—the region least shaped by new data. They constructed LongCLIP$_{70}$ by truncating inputs to 70 words ($\approx$ 77 tokens) during inference. Results showed a sharp decline in long-caption retrieval for LongCLIP$_{70}$, which was then outperformed by LSS. This confirms that LongCLIP's long-caption capability comes from the 77-248 segment, while its SC++ failure is caused by the freezing of the 20-77 segment.

### Loss & Training
No new loss is proposed; LSS uses the original CLIP InfoNCE contrastive loss. Training hyperparameters: all LSS variants used batch_size=1024 on 4 $\times$ A100 GPUs. Learning rates and steps varied by dataset (e.g., ShareGPT4V: lr=3e-6, 3000 steps, 2.5 epochs). Vision/text inputs utilized standard HuggingFace CLIP parameters. Maximum training time was 8 hours.

## Key Experimental Results

### Main Results
**Comprehensive Table for Q1 + Q2 (Table 1)**: Compositionality (SC++ 5 subcategories + WG) + Long-caption retrieval (Urban1K / sDCI / DOCCI / IiW avg of I2T+T2I):

| Model | SC++ avg | Winoground T | Long-cap retrieval avg | Remarks |
|------|----------|--------------|----------------------|------|
| CLIP (baseline) | 53.3 | 17.2 | 67.0 | Baseline |
| SigLIP | 57.5 | 18.6 | 77.5 | Different loss/data |
| **DAC$_{\text{LLM}}$** | 44.0 | 12.6 | 48.5 | Worse than CLIP! |
| DCI$_{\text{P1}}$ | 51.3 | 12.1 | 56.3 | Only ARO strong |
| CE-CLIP | 56.3 | 12.3 | 68.1 | Moderate |
| **NegCLIP** | **63.7** | 16.4 | 73.4 | Best compositional training |
| **LongCLIP-B** | 54.7 | 14.7 | **79.1** | Strong long-cap; flat SC++ |
| DreamLIP | 54.1 | 18.0 | **82.7** | Largest backbone + full pretraining |
| **LSS (control)** | 61.8 | 17.5 | 78.7 | 77 tokens rivaling LongCLIP |

**Key Findings**: (1) NegCLIP trained for compositionality improves long-caption performance (73.4), proving Q1 (Composition $\to$ Long-cap transfer). (2) LSS trained on long captions improves SC++ to 61.8 (close to NegCLIP's 63.7), proving Q2 (Long-cap $\to$ Composition transfer). (3) DAC/DCI fail both; LongCLIP fails SC++—indicating transfer is **sensitive to training settings**.

**ARO vs SC++ Failure Analysis (Table 2)**: DAC$_{\text{LLM}}$ is nearly saturated on ARO (VG-R=81.3, VG-A=73.9) but only reaches 44.0 on SC++ (lower than CLIP's 53.3). Spearman correlation $r = -0.37$ between ARO and SC++ suggests ARO is no longer a reliable reflection of true compositional capability.

### Ablation Study
**Effect of 4 Long-caption Datasets on LSS (Table 9 / Figure 2)**:
- **LSS$_{\text{sDCI}}$**: Highest syntax complexity (94.07) but poor grounding $\to$ overfitting.
- **LSS$_{\text{DOCCI}}$**: Small (14.6K) but human-labeled/long (122 words) $\to$ strong (82.7 long-cap avg).
- **LSS$_{\text{ShareGPT4V}}$**: Best overall (SC++ 61.8, Long-cap 78.7) due to scale and 87.7% vocab coverage.

**General Capability Trade-off (Table 4)**: CLIP baseline IN1K = 63.1. NegCLIP drops to 61.0; LSS drops to 60.8. LongCLIP improves to 66.9, showing the benefit of positional freezing for preserving general classification.

### Key Findings
- **Bidirectional transfer is real but conditional**: It requires (high-quality grounded long captions) $\cap$ (full-parameter fine-tuning).
- **ARO benchmark is obsolete**: It is negatively correlated with SC++ ($r = -0.37$). ARO "high-performers" like DAC/DCI buckle under newer tests.
- **Data Quality > Data Scale**: DOCCI (14.6K human-labeled images) rivals ShareGPT4V (1.2M images). sDCI (7.6K synthetic) lacks grounding and causes regression.
- **Positional embedding freezing is a double-edged sword**: It protects general alignment but creates a bottleneck for compositional learning.

## Highlights & Insights
- **Methodological value of the control model (LSS)**: By isolating architectural innovation from data improvement, the paper provides a rare "attribution analysis" in the VLM field.
- **Evidence for retirement of ARO**: Proving ARO is negatively correlated with SC++ provides a necessary calibration for future evaluation standards.
- **Analysis of positional freezing**: Demonstrating that LongCLIP's gains are restricted to specific context segments via LongCLIP$_{70}$ is a precise "architecture intervention."
- **Data attribute cheat sheet**: Table 3/8 provides guidance for future dataset curation, emphasizing that **vocabulary coverage > syntactic complexity**.

## Limitations & Future Work
- **Scope**: Limited to contrastive VLMs; behavior in generative VLMs (LLaVA, etc.) remains unexplored.
- **Complexity**: Does not explore temporal or causal reasoning beyond surface compositionality.
- **Evaluation Proxy**: Retrieval might not fully equate to deep understanding.
- **Causal Mechanism**: Lacks mechanistic analysis (e.g., attention visualization) to explain *why* grounding is critical.

## Related Work & Insights
- **NegCLIP**: Proves hard negatives and long dense captions are independent but equivalent paths to compositionality.
- **DAC / DCI**: Their collapse on SC++ demonstrates that synthetic captions without grounding + LoRA updates do not equate to real capability gains.
- **DreamLIP**: While strong, its progress is confounded by backbone size; LSS shows full-parameter fine-tuning on smaller backbones can be more effective for compositionality.
- **Takeaway**: Compositional VLM training must utilize **full-parameter fine-tuning**; lightweight adaptation like LoRA is insufficient to internalize compositional structures.

## Rating
- **Novelty**: ⭐⭐⭐ (Standard architecture/loss, but novel experimental design and ARO-failure proof).
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ (Extensive baselines, datasets, and cross-architecture validations).
- **Writing Quality**: ⭐⭐⭐⭐⭐ (Clear questions, straightforward conclusions, and honest limitations).
- **Value**: ⭐⭐⭐⭐ (Crucial calibration for the VLM community regarding benchmarks and training guidelines).

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DocSeeker: Structured Visual Reasoning with Evidence Grounding for Long Document Understanding](../../CVPR2026/multimodal_vlm/docseeker_long_document_understanding.md)
- [\[AAAI 2026\] URaG: Unified Retrieval and Generation in Multimodal LLMs for Efficient Long Document Understanding](../../AAAI2026/multimodal_vlm/urag_unified_retrieval_and_generation_in_multimodal_llms_for.md)
- [\[CVPR 2026\] MSJoE: Jointly Evolving MLLM and Sampler for Efficient Long-Form Video Understanding](../../CVPR2026/multimodal_vlm/msjoe_jointly_evolving_mllm_and_sampler_for_efficient_long-form_video_understand.md)
- [\[CVPR 2026\] REVISOR: Beyond Textual Reflection, Towards Multimodal Introspective Reasoning in Long-Form Video Understanding](../../CVPR2026/multimodal_vlm/revisor_beyond_textual_reflection_towards_multimodal_introspective_reasoning_in_.md)
- [\[CVPR 2026\] ReMoRa: Multimodal Large Language Model based on Refined Motion Representation for Long-Video Understanding](../../CVPR2026/multimodal_vlm/remora_multimodal_large_language_model_based_on_refined_motion_representation_fo.md)

</div>

<!-- RELATED:END -->
