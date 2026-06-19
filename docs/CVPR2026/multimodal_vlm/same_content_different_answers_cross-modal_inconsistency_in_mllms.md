---
title: >-
  [Paper Note] Same Content, Different Answers: Cross-Modal Inconsistency in MLLMs
description: >-
  [CVPR 2026][Multimodal VLM][benchmark] The authors propose two benchmarks, REST and REST+, which present the same problem to MLLMs in three forms: "pure text," "pure image (rendered text-as-image)," and "mixed text-image." Under strict control of OCR accuracy, they measure the phenomenon of "same content, different answers" (cross-modal inconsistency). Eval
tags:
  - CVPR 2026
  - Multimodal VLM
  - benchmark
  - OCR
  - render text-as-image
date: 2026-05-08
content_hash: 01ebf48492f63a1c
---
# Same Content, Different Answers: Cross-Modal Inconsistency in MLLMs

**Conference**: CVPR2026  
**arXiv**: [2512.08923](https://arxiv.org/abs/2512.08923)  
**Code**: https://github.com/angelavansprang/Same-Content-Different-Answers  
**Area**: Multimodal VLM  
**Keywords**: Cross-modal consistency, modality gap, benchmark, OCR, render text-as-image

## TL;DR
The authors propose two benchmarks, REST and REST+, which present the same problem to MLLMs in three forms: "pure text," "pure image (rendered text-as-image)," and "mixed text-image." Under strict control of OCR accuracy, they measure the phenomenon of "same content, different answers" (cross-modal inconsistency). Evaluation of 15 frontier MLLMs reveals that none achieve stable consistency across all three modalities (inconsistency rates of at least ~10%, exceeding 80% at worst). Models generally prefer the text modality, and this inconsistency is significantly correlated with the cosine similarity of internal text-image representations (modality gap).

## Background & Motivation
**Background**: MLLMs are trained to project vision and language into a shared semantic space. The mainstream narrative suggests that "text and images have been seamlessly fused," as models perform strongly in VQA, document understanding, and complex reasoning. Simultaneously, recent works like DeepSeek-OCR suggest a compelling direction: rendering text as images for model input to compress 10 text tokens into 1 visual token while maintaining 97% OCR accuracy, thereby substantially reducing computational costs for long-context inputs.

**Limitations of Prior Work**: However, existing studies consistently observe a "modality gap"—text and image embeddings occupy different regions in the shared space, and smaller gaps typically correlate with better downstream performance. This raises a neglected fundamental question: when a model successfully "reads" text from an image, is the quality of its reasoning on that information as high as when it receives native text? If not, the "render text-as-image" strategy may quietly sacrifice reasoning integrity for cost-efficiency.

**Key Challenge**: Current evaluations cannot cleanly answer this because they conflate two issues: **recognition failure** (OCR errors) and **reasoning inconsistency** (correct recognition but different answers). Existing benchmarks either evaluate only one model or fail to control for readability, leaving the entanglement between "failing to read" and "reading correctly but answering wrong" unresolved.

**Goal**: To systematically measure whether answers are consistent for the same semantic content across different modalities under **strictly controlled OCR correctness**. The authors address four research questions: (RQ1) whether frontier MLLMs exhibit cross-modal inconsistency and which modality is strongest; (RQ2) whether inconsistency is merely caused by poor OCR; (RQ3) whether visual features like resolution, font, and color affect inconsistency; and (RQ4) whether cross-modal similarity of internal representations correlates with inconsistency levels.

**Core Idea**: Construct "rendering-equivalence pressure tests" where semantically identical problems are presented in three modalities. Answer consistency is compared only within the subset where the model achieves perfect OCR. This decouples "recognition" from "reasoning," directly testing whether the MLLM's shared space truly supports modality-agnostic reasoning.

## Method
As a benchmark and empirical analysis paper, this work does not propose a new model or training method. The "Method" consists of the benchmark construction, consistency metrics, and the design of four controlled analysis experiments. The logic is: filter out samples where "recognition failed" using an OCR task, quantify cross-modal inconsistency on the "clean" subset using three custom metrics, and finally link behavioral inconsistency to internal representation similarity.

### Overall Architecture
A benchmark sample consists of three "rendering-equivalent" forms: **Text** (pure text question), **Image** (entire question rendered as an image), and **Mixed** (context as image, question as text; or problem statement as image, options as text for multiple-choice). Each sample first undergoes an **OCR task** to verify readability. Subsequently, the model answers in each modality using Chain-of-Thought (CoT) prompting ($temperature=0$). Consistency analysis is conducted **only on samples with perfect OCR**, which is the key mechanism to decouple recognition from reasoning.

The data comprises four tasks: three existing benchmarks (MMLU, AI2-ARC, GSM-Symbolic) rendered into images, plus the newly created **SoEBench** (solving systems of linear equations). To minimize OCR difficulty, items with $>800$ characters or LaTeX were filtered; images use a white background with 200 DPI and black DejaVu Sans font. **REST+** is an enhanced version that applies 10 visual perturbations (3 fonts $\times$ 3 resolutions + 1 color) to each image to test the impact of visual features and token counts on consistency.

### Key Designs

**1. REST/REST+ Triple-Modal "Rendering-Equivalent" Samples + SoEBench: Decoupling Recognition from Reasoning**

The core innovation is "same content, triple rendering, strict control." The three modalities (Text / Image / Mixed) carry **identical semantics**. Ideally, a modality-agnostic model should yield identical answers; thus, any divergence is attributed to the modality itself rather than content.

**SoEBench** (system-of-equations benchmark) is specifically designed to counter two confounders: it uses a minimal symbol set (digits 0–9 and letters A–E), making OCR errors nearly impossible, and it is **newly generated** to ensure no MLLM has seen it during pre-training. This eliminates "data contamination" as an explanation for superior text modality performance.

**2. Three Consistency/Capability Metrics (RER / CFR / MMC): Quantifying Divergence**

**Render-Equivalence Rate (RER)** measures the proportion of problems where answers across all three modalities are identical:

$$\text{RER}=\frac{|\{x \mid f(x_t)=f(x_i,z_i)=f(x_m,z_m)\}|}{N}$$

**Cross-Modality Failure Rate (CFR)** measures problems that are solved in at least one modality but not all:

$$\text{CFR}=\frac{|\{q \mid 1\le \sum_{m\in M} C(q,m) < |M|\}|}{N_c}$$

It **excludes problems failed in all three modalities** (which indicates a lack of capability rather than inconsistency). **Max Modal Coverage (MMC)** represents the upper bound if the optimal modality were chosen for each problem.

**3. OCR-first Counterfactual + Visual Perturbations (REST+): Identifying Drivers**

To prove inconsistency is **not** caused by OCR, the authors tested an "OCR-first, then solve" counterfactual. Some models improved, while others (e.g., DeepSeek-Small on MMLU $-13.1$) dropped significantly, proving that explicit OCR does not stably fix inconsistency. REST+ systematically disassembled visual features: **font had almost no impact**, **resolution mattered** (higher DPI improved consistency), and **colored text surprisingly improved performance** for several models.

**4. Mechanism: Representation Similarity ↔ Behavioral Inconsistency (RQ4)**

The authors linked behavioral inconsistency to "modality gaps." They calculated the cosine similarity $\mathrm{sim}(\mathbf{I},\mathbf{T})=\frac{\bar{\mathbf{i}}\cdot\bar{\mathbf{t}}}{\|\bar{\mathbf{i}}\|\|\bar{\mathbf{t}}\|}$ between mean image and text tokens. Experiments on ImageNet showed that this similarity / retrieval score is **significantly positively correlated** with RER. This provides a mechanistic explanation: **inconsistency stems from text and image representations landing in different regions of the shared space.**

## Key Experimental Results

### Main Results
Consistency on REST (Subset with perfect OCR):

| Model | RER↑ (OCR✓) | CFR↓ (OCR✓) | OCR✓ Rate |
|------|------|------|------|
| GPT-5-mini | **90.7** | **8.7** | 99.0 |
| Haiku-4.5 (Claude) | 90.3 | 8.9 | 98.2 |
| Qwen-2.5 (32B) | 84.7 | 13.6 | 97.5 |
| InternVL3 (14B) | 78.4 | 19.6 | 95.3 |
| GPT-4o-mini | 71.3 | 26.0 | 98.8 |
| Gemini-2.5 Fl. Lite | 54.3 | 40.3 | 98.3 |
| Phi-4 | 14.9 | 82.3 | 94.0 |
| Deepseek-Tiny | 6.6 | 98.0 | 70.3 |

No model achieves perfect consistency. Furthermore, **almost all models are strongest in the text modality**. In SoEBench, where OCR is nearly perfect and contamination is impossible, Text still leads, confirming the reasoning mechanism itself is modality-dependent.

### Ablation Study

| Configuration / Analysis | Key Metric | Observation |
|------|---------|------|
| OCR-first | Unstable ΔAcc | DeepSeek-Small dropped $>10$ points; OCR is not the primary confounder |
| REST+ (Perturbations) | Best CFR 27.9% | Font/Color/Resolution variations alone can flip answer correctness |
| DPI 50→100→200 | RER increases | Higher resolution improves consistency, partly through better OCR |
| Font (Sans/Courier/Cursive) | <2% Difference | Fonts have negligible impact, even cursive text |
| Color vs. Black | >5% relative Gain | Colored text (Red/Yellow) often yields better results than black |
| Repr. Similarity vs. RER | Positive Correlation | Higher alignment in embedding space directly corresponds to better behavioral consistency |

### Key Findings
- **No MLLM is cross-modality consistent**, with inconsistency rates ranging from ~10% to >80%.
- **Text modality is systematically stronger**, and this is not due to OCR or data contamination.
- **Visual feature impacts are counter-intuitive**: font is irrelevant, color helps, and resolution's impact is partially mediated through OCR.
- **Consistent behaviors possess internal correlates**: the modality gap (representation similarity) is significantly correlated with behavioral RER.

## Highlights & Insights
- **Decoupling Protocol**: The "compare only on OCR✓ subset" strategy is the soul of this work, allowing for the isolation of reasoning from recognition.
- **SoEBench Design**: The combination of a minimal symbol set and new generation effectively eliminates the common defenses of "poor OCR" or "memorization."
- **Caution on Token Compression**: By showing that models often need **more** visual tokens to match text accuracy, the paper challenges the "render-as-image" efficiency narrative.
- **Mechanism-Behavior Link**: Directly connecting the geometric modality gap in the embedding space to the specific failure modes of model logic.

## Limitations & Future Work
- **Correlation vs. Causality**: The link between representation similarity and RER is correlational; whether pulling representations closer actively improves consistency remains an open question.
- **Task Scope**: The focus is primarily on rendered text in QA. Natural image inconsistency (depth, spatial layout) was only touched upon via ImageNet and chess tasks.
- **Proprietary Models**: Hidden activations and precise tokenization metrics are unavailable for closed-source models.
- **Solution Gap**: The paper diagnoses the problem but does not propose a specific training or alignment method to mitigate inconsistency.

## Related Work & Insights
- **vs. Zhang et al.**: They evaluated only GPT-4V without OCR control; this work scales to 15 models with rigorous decoupling.
- **vs. DeepSeek-OCR / Render-as-Image**: This work serves as a "cold shower" for these efficiency-focused routes, highlighting potential reasoning degradation.
- **vs. Modality Gap Research**: While prior work linked the gap to downstream performance, this paper extends the impact to the core logic of cross-modal reasoning.

## Rating
- Novelty: ⭐⭐⭐⭐ (Novel framing of "rendering-equivalence," though datasets are repurposed)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Broad model coverage and rigorous variable control)
- Writing Quality: ⭐⭐⭐⭐ (Clear RQ-driven logic)
- Value: ⭐⭐⭐⭐⭐ (Crucial warning for the field of multi-modal compression and alignment)

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Decoupled and Reusable Adaptation for Efficient Cross-Modal Transfer](decoupled_and_reusable_adaptation_for_efficient_cross-modal_transfer.md)
- [\[CVPR 2026\] Rethinking Cross-Modal Anchor Alignment for Mitigating Error Accumulation](rethinking_cross-modal_anchor_alignment_for_mitigating_error_accumulation.md)
- [\[AAAI 2026\] Rethinking Visual Token Reduction in LVLMs under Cross-Modal Misalignment](../../AAAI2026/multimodal_vlm/rethinking_visual_token_reduction_in_lvlms_under_cross-modal_misalignment.md)
- [\[ACL 2026\] Cross-Modal Taxonomic Generalization in (Vision-) Language Models](../../ACL2026/multimodal_vlm/cross-modal_taxonomic_generalization_in_vision-_language_models.md)
- [\[ICML 2026\] CG-MLLM: Captioning and Generating 3D Content via Multi-modal Large Language Models](../../ICML2026/multimodal_vlm/cg-mllm_captioning_and_generating_3d_content_via_multi-modal_large_language_mode.md)

</div>

<!-- RELATED:END -->
