---
title: >-
  [Paper Note] Long Story Short: Disentangling Compositionality and Long-Caption Understanding in Contrastive VLMs
description: >-
  [ACL 2026 Findings][Multimodal VLM][CLIP] This work systematically disentangles the relationship between "compositionality" and "long-caption understanding" in contrastive VLMs. It finds that these capabilities mutually promote each other in a bidirectional manner, yet this transfer is **highly sensitive to training data quality and optimization strategies**. Specifically, grounded long-caption data with high vocabulary coverage combined with full-parameter fine-tuning allows…
tags:
  - "ACL 2026 Findings"
  - "Multimodal VLM"
  - "CLIP"
  - "Compositional Reasoning"
  - "Long-Caption Understanding"
  - "Data Quality"
  - "Positional Encoding Freezing"
  - "Bidirectional Transfer"
date: 2026-05-08
content_hash: bb879ae6337289e3
---

# Long Story Short: Disentangling Compositionality and Long-Caption Understanding in Contrastive VLMs

**Conference**: ACL 2026 Findings  
**arXiv**: [2509.19207](https://arxiv.org/abs/2509.19207)  
**Code**: To be confirmed  
**Area**: Multimodal VLM / Evaluation  
**Keywords**: CLIP, Compositional Reasoning, Long-Caption Understanding, Data Quality, Positional Encoding Freezing, Bidirectional Transfer

## TL;DR
This work systematically disentangles the relationship between "compositionality" and "long-caption understanding" in contrastive VLMs. It finds that these capabilities mutually promote each other in a bidirectional manner, yet this transfer is **highly sensitive to training data quality and optimization strategies**. Specifically, grounded long-caption data with high vocabulary coverage combined with full-parameter fine-tuning allows a model to excel in both capabilities simultaneously. Conversely, low-quality synthetic captions (e.g., DAC/DCI) or LoRA-based partial updates lead to failure in both. LongCLIP's strategy of freezing the first 20 positional embeddings, while intended to preserve general alignment, severely restricts compositional learning. The "control model" proposed in this work, LSS, outperforms LongCLIP within the original 77-token context window by fine-tuning on ShareGPT4V with full parameters.

## Background & Motivation

**Background**: Contrastive VLMs (CLIP / SigLIP / ALIGN) have become the de facto standard for multimodal learning. However, two long-standing issues remain: (1) **Poor compositionality**—CLIP often behaves as a "bag-of-words," showing insensitivity to attribute-object binding, relations, and word order, as exposed by benchmarks like ARO, Winoground, and SugarCREPE++ (SC++); (2) **Weak long-caption processing**—CLIP's 77-token context window is short, and its effective attention often only reaches the first 20-30 tokens (Zhang et al. 2024a), leading to poor performance on long, dense caption retrieval (DOCCI / Urban1k / ImageInWords).

**Limitations of Prior Work**: The field has generally hypothesized that "compositional reasoning" and "long-caption understanding" are highly correlated; long captions naturally contain more attributes and relations which should promote compositional learning, while compositionally strong models should better disentangle long captions. However, these two research lines have remained **fragmented**. Studies on compositionality (NegCLIP / CE-CLIP / DAC / DCI) use short captions with hard negatives, while studies on long captions (LongCLIP / DreamLIP) extend the context window without specifically reinforcing composition. No systematic study has compared the cross-capability transferability between these two lines.

**Key Challenge**: (a) While progress is made in each line individually, combining them yields unexpected results—for instance, DAC/DCI are nearly saturated on the traditional ARO benchmark but perform worse than base CLIP on the newer SC++ (Spearman $r = -0.37$!). LongCLIP performs strongly on long captions but shows almost no improvement over CLIP on SC++. (b) This suggests that either the ARO benchmark is no longer reliable, or the transfer from "compositional training $\implies$ long-caption understanding" does not manifest as expected.

**Goal**: To answer two questions through controlled experiments: (Q1) Does training for compositionality improve long-caption understanding? (Q2) Does training on long captions promote compositional generalization? Furthermore, the study aims to isolate how **data quality, optimization strategies, and architectural constraints** determine when such transfer succeeds or fails.

**Key Insight**: The authors developed a control model named **LSS (Long Story Short)**. By fine-tuning a CLIP ViT-B/32 on ShareGPT4V long captions while strictly maintaining the original 77-token context window and using full-parameter updates, they decouple the "effect of long-caption data" from "architectural changes for expanding the context window." LSS was trained separately on four long-caption datasets (sDCI / DOCCI / LN / ShareGPT4V) to ablate which data attributes (scale, vocabulary coverage, caption length, syntactic complexity Yngve, annotation quality) are truly critical.

**Core Idea**: Transferability is real, but it only holds when (high-quality grounded long captions) and (full-parameter fine-tuning) are simultaneously satisfied. Architectures that freeze positional embeddings to preserve CLIP's general alignment actually act as a bottleneck for compositional learning.

## Method

### Overall Architecture
This is an **empirical analysis paper** that does not propose a new model architecture but constructs a series of comparative experiments to disentangle the variables of data, optimization, and architecture. The overall process includes:

(a) **Selecting representative baselines**: NegCLIP / CE-CLIP / DAC$_{\text{LLM}}$ / DCI$_{\text{P1}}$ for compositional training; LongCLIP / DreamLIP for long-caption training; along with baseline CLIP ViT-B/32 and SigLIP.

(b) **Designing the control model LSS**: Based on CLIP ViT-B/32 using full-parameter fine-tuning, 4×A100 GPUs, batch size 1024, trained separately on four long-caption datasets (sDCI / DOCCI / LN / ShareGPT4V) while strictly maintaining a 77-token context.

(c) **Unified benchmark suite**: Compositionality is evaluated via Winoground (WG) + SugarCREPE++ (SC++ featuring five sub-categories: SA/RR/RO/RA/SO) + ARO (to contrast with the failing traditional benchmark). Long-caption retrieval is tested on Urban1K / sDCI / DOCCI / IiW via I2T and T2I R@1. General alignment is checked via CIFAR10/100 / ImageNet classification + COCO/Flickr30k short-caption retrieval. All evaluations are zero-shot.

(d) **Multi-dimensional comparison**: Q1 (compositional training $\rightarrow$ long-caption) / Q2 (long-caption training $\rightarrow$ composition) / failure analysis of ARO vs SC++ / LSS comparison across four datasets / ablation of LongCLIP’s positional embedding freezing / general capability trade-offs.

### Key Designs

**1. LSS Control Model: Decoupling "Long-Caption Data" from "Context Expansion Architecture"**

LongCLIP simultaneously introduces three changes: switching to ShareGPT4V long-caption data, expanding context from 77 to 248 tokens, and freezing the first 20 positional embeddings to mitigate catastrophic forgetting. This confounding makes it impossible to identify the source of gains. LSS is a clean control model: it uses only "ShareGPT4V long-caption data + full-parameter fine-tuning" without expanding the context or freezing positional embeddings. Training configurations (Table 5) include a learning rate of 3e-6, 150 warmup steps, and 3000 total steps (approx. 2.5 epochs). This control demonstrates that expanding the context architecture is nearly redundant for these gains, as LSS (77 tokens) outperforms LongCLIP.

**2. Multi-Benchmark Cross-Section: Mapping Training Variables to Specific Data/Optimization Attributes**

To determine what drives transferability, the authors decompose training settings into quantifiable attributes. They tabulate four long-caption datasets across five attributes (Table 3/8): sDCI (7.6K images, 29% vocab, 94 Yngve), DOCCI (15K images, 27% vocab, 75 Yngve, human-written), LN (489K images, 30-word short captions, 24% vocab, human-written), and ShareGPT4V (1.2M images, 144-word long captions, 88% vocab, synthetic). LSS variants were trained on each. Results show that no single attribute (e.g., scale or complexity) determines performance; rather, it is the synergy of vocabulary coverage, caption length, grounding, data scale, and syntactic complexity.

**3. LongCLIP Positional Freezing Ablation (LongCLIP$_{70}$): Identifying the Cause of Stagnation in SC++**

LongCLIP shows almost no improvement on SC++. To investigate whether this is due to data or positional freezing, the authors observed that LongCLIP freezes the first 20 positional embeddings and discounts updates for tokens 20-77. Since most SC++ samples fall within the first 77 tokens, they formulated LongCLIP$_{70}$, which truncates input to 70 words during inference. Performance (Figure 3) crashed for long-caption retrieval, proving that LongCLIP’s strength lies in tokens 77-248, while its stagnation on SC++ is directly caused by the frozen first 77 tokens.

### Loss & Training
No new loss is proposed; LSS utilizes the original CLIP InfoNCE contrastive loss. Hyperparameters (Appendix C Table 5): All LSS variants used batch_size=1024 on 4×A100 GPUs. Learning rates and steps varied: sDCI (5e-6/500 steps/70 epochs), DOCCI (5e-6/500 steps/35 epochs), LN (3e-6/2000 steps/4 epochs), and ShareGPT4V (3e-6/3000 steps/2.5 epochs). Vision/text input processing followed HuggingFace CLIP defaults.

## Key Experimental Results

### Main Results
**Comprehensive Table for Q1 + Q2 (Table 1)**: Compositionality (SC++ avg + WG) and long-caption retrieval (Urban1K / sDCI / DOCCI / IiW avg I2T+T2I):

| Model | SC++ avg | Winoground T | Long-cap retrieval avg | Notes |
|------|----------|--------------|----------------------|------|
| CLIP (baseline) | 53.3 | 17.2 | 67.0 | Starting point |
| SigLIP | 57.5 | 18.6 | 77.5 | Different loss/data |
| **DAC$_{\text{LLM}}$** | 44.0 | 12.6 | 48.5 | Worse than CLIP! |
| DCI$_{\text{P1}}$ | 51.3 | 12.1 | 56.3 | Only strong on ARO |
| CE-CLIP | 56.3 | 12.3 | 68.1 | Medium |
| **NegCLIP** | **63.7** | 16.4 | 73.4 | Best for composition |
| **LongCLIP-B** | 54.7 | 14.7 | **79.1** | Strong long-cap, stagnant SC++ |
| DreamLIP | 54.1 | 18.0 | **82.7** | Max backbone + full pretrain |
| **LSS (control)** | 61.8 | 17.5 | 78.7 | 77 tokens rival LongCLIP |

**Key Findings**: (1) NegCLIP, trained for compositionality, improved long-caption performance to 73.4, validating Q1 (composition $\rightarrow$ long-caption transfer). (2) LSS, trained on long captions, achieved an SC++ score of 61.8 (near NegCLIP’s 63.7), validating Q2 (long-caption $\rightarrow$ composition transfer). (3) The failure of DAC/DCI and the single-sided strength of LongCLIP highlight that transfer is sensitive to training setups.

**ARO vs SC++ Failure Comparison (Table 2)**: DAC$_{\text{LLM}}$ is near saturation on ARO (VG-R=81.3 / VG-A=73.9) but drops to 44.0 on SC++ (below CLIP's 53.3). Spearman correlation is $r = -0.37$, indicating that rule-based restricted caption benchmarks like ARO no longer reflect true compositional ability.

### Ablation Study
**Effect of Four Long-Caption Datasets on LSS (Table 9 / Figure 2)**:

| LSS Variant | Scale | Caption Length | Vocab cov | Yngve | SC++ avg | Long-cap avg | Evaluation |
|---------|---------|------------|-----------|-------|----------|--------------|------|
| LSS$_{\text{sDCI}}$ | 7.6K img | 40 words | 29% | **94.07** | 57.4 | 71.6 | High syntax, poor grounding $\rightarrow$ Overfit |
| LSS$_{\text{DOCCI}}$ | 14.6K img | **122 words** | 27% | 74.55 | 60.9 | **82.7** | Small but human-curated $\rightarrow$ Strong |
| LSS$_{\text{LN}}$ | 489K img | 30 words | 24% | 61.70 | 61.6 | 70.7 | Fast SC++ convergence, weak long-cap |
| **LSS$_{\text{ShareGPT4V}}$** | **1.2M img** | 144 words | **87.72%** | 45.70 | **61.8** | 78.7 | Scale × Vocab coverage $\rightarrow$ Best overall |

**General Capability Trade-off (Table 4)**: CLIP baseline IN1K=63.1; NegCLIP drops to 61.0; DAC$_{\text{LLM}}$ to 51.1. LongCLIP increases to 66.9 due to positional freezing. LSS drops slightly to 60.8.

### Key Findings
- **Bidirectional transfer exists but requires specific conditions**: High-quality grounded long captions + full-parameter fine-tuning are necessary.
- **ARO benchmark is obsolete**: It is negatively correlated with SC++ ($r = -0.37$). Methods overfitted to ARO (DAC/DCI) fail on newer benchmarks.
- **Data quality over scale**: DOCCI (14.6K images, human-curated) rivals ShareGPT4V (1.2M images, synthetic).
- **Positional embedding freezing is a double-edged sword**: It preserves general alignment (IN1K) but stifles compositional learning in the primary context window.
- **Why DAC/DCI failed**: Use of LoRA updates restricted model capacity for deep compositional structures, and lack of visual grounding in LLM-expanded captions.

## Highlights & Insights
- **Methodological value of LSS**: By performing "disentanglement experiments," the authors provide a necessary calibration for VLM research, identifying which improvements come from data vs. architecture.
- **Hard evidence for retiring ARO**: The negative correlation with SC++ serves as a definitive argument against using rule-based caption benchmarks.
- **Cheatsheet for data attributes**: Vocabulary coverage is more critical than Yngve syntactic complexity. DOCCI's success despite its small size highlights the value of human precision and grounding.
- **Full-parameter fine-tuning is essential**: Lightweight adaptation like LoRA is insufficient for internalizing fine-grained compositional structures.

## Limitations & Future Work
- **Scope**: Limited to contrastive VLMs; whether conclusions transfer to generative VLMs (LLaVA/Qwen-VL) is unknown.
- **Reasoning complexity**: Only surface compositionality (attributes/relations) was explored, excluding temporal or causal reasoning.
- **Evaluation proxy**: Retrieval was used as a proxy for understanding; more direct generative or fine-grained probing is left for future work.
- **Mechanism analysis**: The work observes correlations but lacks a causal mechanistic explanation (e.g., attention pattern visualization) for why grounding is vital.

## Related Work & Insights
- **vs NegCLIP**: NegCLIP uses hard negatives; LSS uses long captions. Both achieve similar compositionality, proving these are independent but equivalent paths.
- **vs DAC/DCI**: Highlights that synthetic captions without grounding plus restricted LoRA updates lead to failure on modern benchmarks.
- **vs LongCLIP**: Proves that the long-context advantage is localized to tokens 77-248 and identifies positional freezing as the cause of stagnant compositional growth.
- **vs DreamLIP**: Shows that while massive pre-training on long captions is powerful, LSS with a smaller backbone and full-parameter fine-tuning can outperform it on compositionality.

## Rating
- Novelty: ⭐⭐⭐ (High-quality empirical study over structural innovation)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Extensive baselines, datasets, and cross-architecture validations)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear problem framing and honest discussion of limitations)
- Value: ⭐⭐⭐⭐ (Significant calibration for the VLM community regarding benchmarks and training strategies)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] URaG: Unified Retrieval and Generation in Multimodal LLMs for Efficient Long Document Understanding](../../AAAI2026/multimodal_vlm/urag_unified_retrieval_and_generation_in_multimodal_llms_for.md)
- [\[CVPR 2026\] MSJoE: Jointly Evolving MLLM and Sampler for Efficient Long-Form Video Understanding](../../CVPR2026/multimodal_vlm/msjoe_jointly_evolving_mllm_and_sampler_for_efficient_long-form_video_understand.md)
- [\[CVPR 2026\] PersonaVLM: Long-Term Personalized Multimodal LLMs](../../CVPR2026/multimodal_vlm/personavlm_long_term_personalized_multimodal_llms.md)
- [\[CVPR 2026\] ReMoRa: Multimodal Large Language Model based on Refined Motion Representation for Long-Video Understanding](../../CVPR2026/multimodal_vlm/remora_multimodal_large_language_model_based_on_refined_motion_representation_fo.md)
- [\[CVPR 2026\] Scaling the Long Video Understanding of Multimodal Large Language Models via Visual Memory Mechanism](../../CVPR2026/multimodal_vlm/scaling_the_long_video_understanding_of_multimodal_large_language_models_via_vis.md)

</div>

<!-- RELATED:END -->
