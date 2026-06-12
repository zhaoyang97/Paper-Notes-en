---
title: >-
  [Paper Note] Decoding Scientific Experimental Images: The SPUR Benchmark for Perception, Understanding, and Reasoning
description: >-
  [ACL 2026][Multimodal VLM][Experimental Images] SPUR is the first benchmark specifically designed for the "Perception $\rightarrow$ Understanding $\rightarrow$ Reasoning" three-stage evaluation of biomedical experimental…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Experimental Images"
  - "Multi-panel Understanding"
  - "Quantitative Reasoning"
  - "MCoT"
  - "PMC"
date: 2026-05-08
content_hash: 32e9314725de7d76
---

# Decoding Scientific Experimental Images: The SPUR Benchmark for Perception, Understanding, and Reasoning

**Conference**: ACL 2026  
**arXiv**: [2604.27604](https://arxiv.org/abs/2604.27604)  
**Code**: https://github.com/BUPT-Reasoning-Lab/SPUR (Available)  
**Area**: Multimodal VLM / Scientific Image Understanding / AI4S  
**Keywords**: Experimental Images, Multi-panel Understanding, Quantitative Reasoning, MCoT, PMC

## TL;DR
SPUR is the first benchmark specifically designed for the "Perception $\rightarrow$ Understanding $\rightarrow$ Reasoning" three-stage evaluation of biomedical experimental images (multi-panel staining, Western blots, statistical charts). It contains 4,264 expert-verified MCQs, revealing that current MLLMs (with Gemini 3 Pro Preview barely exceeding 60%) consistently perform 12.76%–31.41% lower in quantitative reasoning compared to qualitative reasoning.

## Background & Motivation
**Background**: MLLM capabilities on scientific images (statistical charts, tables, biological diagrams, chemical structures) are rapidly improving, leading to benchmarks such as ScienceQA, MMMU, M3CoT, MMSci, SciAssess, and MicroVQA. Meanwhile, MCoT methods (prompt-based, plan-based, training-based) are employed to enhance multimodal reasoning.

**Limitations of Prior Work**: The true test of "image reading ability" in scientific papers lies in **multi-panel experimental figures** (e.g., Western blot + staining + trend curves telling a single story). Existing benchmarks suffer from three deficiencies: (1) Extremely low proportion of experimental images, mostly consisting of statistical charts or academic diagrams; (2) Average of $\leq 8$ panels per figure, lacking cross-panel relationship modeling; (3) Almost exclusive focus on qualitative conclusions ("A promotes B") rather than quantitative reasoning ("A increases by 50%").

**Key Challenge**: The capability truly required for AI4S is "deriving quantifiable scientific conclusions from complex multi-panel visual evidence via cross-panel comparison and trend synthesis." Existing benchmarks only test segments of this pipeline, masking the true bottlenecks of MLLMs.

**Goal**: Construct a benchmark **specifically for multi-panel experimental figures**, explicitly decomposed into **Perception $\rightarrow$ Understanding $\rightarrow$ Reasoning** stages, covering both **qualitative and quantitative** reasoning. Systematically evaluate 20 MLLMs and 4 MCoT methods to analyze capability gaps.

**Key Insight**: Filter experimental figures with IF > 3 from PMC open-access papers, requiring **$\geq 6$ panels per figure** (77.6% filtered out by YOLO detection). Use expert hierarchical review + GPT-4o for QA generation, followed by "shortcut filtering" (discarding items solvable by text-only models) to ensure visual necessity.

**Core Idea**: Use a seven-task hierarchy: "Perception (NP/MP/IL) $\rightarrow$ Understanding (TA/HI) $\rightarrow$ Reasoning (Qual./Quant.)." This decomposes "understanding an experimental figure" into independently diagnosable sub-capabilities, specifically locating MLLM bottlenecks in "fine-grained numerical perception" and "cross-panel trend analysis."

## Method

### Overall Architecture
SPUR is a benchmark and evaluation framework rather than a model. The pipeline includes: ① **Image Collection**—5,000+ papers with IF > 3 from PMC, 5,632 images extracted, with manual addition of 3–5 sentences of relevant text + standardized captions + subject classification; ② **Image Filtering**—YOLO panel detector discards figures with $\leq 6$ panels (77.6% eliminated), followed by expert review to remove incomplete experimental workflows (another 14.2% eliminated), leaving 1,084 images; ③ **QA Generation**—Experts use 7-task templates for prompts, GPT-4o generates 7,608 candidate MCQs; ④ **Quality Assurance**—Textual shortcut elimination (GPT-4o answers 10 times without images; items with $\geq 5$ correct answers are discarded, 21.2% eliminated) + double expert blind review (28.9% eliminated), resulting in 4,264 questions; ⑤ **Evaluation**—Accuracy tested on 8 closed-source and 12 open-source MLLMs across 7 tasks, comparing 4 MCoT methods.

### Key Designs

1.  **Three-Stage Seven-Task Hierarchical Evaluation**:
    - **Function**: Explicitly decomposes the capability chain of "understanding multi-panel experimental figures" into Perception (NP/MP/IL), Understanding (TA/HI), and Reasoning (Qual./Quant.) with independent accuracy metrics.
    - **Mechanism**: Perception stage (panel-level)—NP estimates kinetic curve values, MP identifies cell morphology, IL maps panels to experimental conditions; Understanding stage (cross-panel)—TA analyzes trend directions across isomorphic panels, HI performs cross-modal integration across heterogeneous panels; Reasoning stage (expert-level)—Qual. provides directional conclusions, Quant. provides quantitative conclusions like ratios or significance.
    - **Design Motivation**: Traditional VQA-style benchmarks provide only an overall accuracy, making it unclear where models fail. The hierarchical design enables diagnostic conclusions, such as "NP is the bottleneck" or "TA drops from 60.7% to 34.0% as relationships increase from 1 to 4."

2.  **High-Complexity Images + Six Fine-Grained Panel Types**:
    - **Function**: Constructs extremely complex figures with an **average of 14.3 panels/image**, containing up to 6 fine-grained panel types (4 staining types + statistical charts + Western blots), far exceeding MMSci (7.4), SFE (2.3), and MicroVQA (1.9).
    - **Mechanism**: YOLO detector ensures $\geq 6$ panels; staining images are subdivided into Cell, Tissue, Microorganism, and Subcellular categories. This allows MP tasks to be analyzed by panel category (e.g., Ministral 3 14B scores 70.52% on Subcellular but only 42.80% on Microorganism, exposing training bias).
    - **Design Motivation**: Low-complexity figures (1–3 panels) fail to test cross-panel relations and are prone to leakage via captions; high panel counts and type mixing simulate real-world scientific scenarios like reading a *Nature* figure.

3.  **Double Shortcut Elimination + Expert Hierarchical Review**:
    - **Function**: Ensures every question "requires the image to be answered correctly," preventing models from exploiting caption keywords, common sense, or pre-training memory.
    - **Mechanism**: (a) Textual shortcut filter—Questions + options fed to GPT-4o **without images** for 10 trials; $\geq 5/10$ correct leads to discarding (21.2% removed); (b) Double expert review—4 domain experts (>40 papers) + 2 senior experts (>100 papers) score on Scientific Validity, Task Alignment, and Visual Reasoning Necessity; (c) QA generation prohibits deriving questions directly from captions, forcing reliance on visual information.
    - **Design Motivation**: The biggest pitfall in scientific image QA is "answer leakage to captions or common sense." After these filters, GPT-4o's text-only accuracy stays below 50%, proving visual information is essential.

### Loss & Training
SPUR is an evaluation benchmark; no training is performed. Evaluation protocol: Direct prompting + accuracy on MCQ; MCoT evaluation applies DDCoT/VoT (prompt-based) and VIC/Cantor (plan-based) for fair comparison.

## Key Experimental Results

### Main Results

Overall accuracy of 20 MLLMs on SPUR (Excerpt):

| Model | NP | MP | IL | TA | HI | Qual. | Quant. | Overall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Gemini 3 Pro Preview | 61.26 | 67.74 | 59.67 | 51.04 | 59.23 | 90.31 | 58.90 | **60.57** |
| Claude 3.7 Sonnet (thinking) | 59.67 | 64.32 | 57.45 | 51.30 | 60.80 | 87.58 | 59.96 | 59.52 |
| Gemini 2.5 Pro Preview | 56.47 | 62.97 | 56.47 | 53.30 | 61.54 | 86.54 | 57.94 | 59.00 |
| GPT-5.1 | 58.73 | 61.72 | 54.47 | 51.18 | 50.78 | 86.52 | 56.36 | 57.68 |
| GLM-4.5V (Best Open Source) | 57.70 | 61.99 | 57.65 | 55.71 | 68.46 | 80.94 | 58.48 | 59.87 |
| InternVL3-78B | 46.30 | 51.97 | 49.84 | 49.52 | 61.24 | 75.24 | 51.06 | 51.94 |
| Qwen2.5-VL-72B | 38.10 | 45.34 | 49.11 | 51.87 | 61.90 | 73.10 | 52.51 | 48.21 |
| LLaVA-v1.5-13B | 33.05 | 28.11 | 34.15 | 34.52 | 44.96 | 62.19 | 35.58 | 35.97 |

**Conclusion**: Almost all models fall **below 60%** overall accuracy except Gemini 3 Pro Preview. GLM-4.5V is the best open-source model, approaching mid-tier closed-source models. NP is generally the lowest; the Qual. vs. Quant. gap reaches up to 31.41% (Llama 4 Maverick: 84.64 vs 57.02).

### Ablation Study

Four MCoT methods vs. direct prompting (GLM-4.5V excerpt):

| Configuration | NP | TA | Qual. | Quant. | Overall |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Direct | 57.70 | 55.71 | 80.94 | 58.48 | 59.87 |
| DDCoT (prompt) | 47.11 | 45.24 | 71.52 | 53.27 | 48.90 |
| VoT (prompt) | 55.82 | 53.65 | 78.44 | 57.77 | 58.47 |
| VIC (plan) | 35.50 | 27.20 | 34.59 | 36.52 | 32.02 |
| Cantor (plan) | 53.41 | 51.23 | 77.12 | 56.61 | 55.59 |

Decoupled reasoning accuracy based on "Perception Correct/Incorrect" (Qwen3-VL-30B):

| Condition | Direct | DDCoT | VoT | VIC | Cantor |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Perception Correct | 71.66 | 82.66 ($\uparrow$11.0) | 98.59 ($\uparrow$26.9) | 65.66 ($\downarrow$6.0) | 79.65 ($\uparrow$8.0) |
| Perception Incorrect | 32.40 | 23.68 ($\downarrow$8.7) | 9.32 ($\downarrow$23.1) | 30.30 ($\downarrow$2.1) | 40.23 ($\uparrow$7.8) |

### Key Findings
- **MCoT is an "Amplifier," not a "Fixer"**: If perception is correct, MCoT adds 8–27 points; if perception is wrong, MCoT exacerbates the error (VoT drops by 23 points). This prioritizes "seeing clearly" over "thinking hard."
- **TA Inverse to Relationship Complexity**: As the number of cross-panel relationships increases from 1 to 4, Claude 3.7 thinking TA accuracy drops from 60.70% to 34.00%, identifying joint multi-relation reasoning as the bottleneck.
- **Significant Disciplinary Bias in MP**: Ministral 3 14B scores 70.52% on Subcellular but only 42.80% on Microorganism, reflecting uneven distribution and weak generalization in training corpora.
- **Thinking Models Hit Qualitative Ceilings**: Gemini 3 Pro Preview and Claude 3.7 thinking score 87–90% on Qual., but only 59–60% on Quant., proving that "drawing conclusions" $\neq$ "calculating numbers."

## Highlights & Insights
- **Diagnostic Benchmark**: The seven-task hierarchy allows a single overall score to be traced back to specific broken links in the reasoning chain, providing better guidance for AI4S model development than MMMU-style overall scores.
- **Reusable "Double Filter + Blind Review" Pipeline**: The textual shortcut detection + multi-panel constraint provides a universal template to prevent caption-based cheating in scientific VQA.
- **Stunning MCoT Decoupling Analysis**: Splitting MCoT gains based on perception accuracy debunked the "CoT as a panacea" myth, suggesting VLM perceptual ability must be trained before applying CoT.
- **Average 14.3 panels/image**: The highest complexity to date, closely approximating real top-tier journal figures. Failure on SPUR is more representative of real-world scientific scenarios than failures on MMMU.

## Limitations & Future Work
- **MCQ Format Obscures Reasoning**: Unable to directly observe *why* a model failed (estimated value wrong? trend reversed? logical jump?); free-form rationales and step-wise scoring are needed.
- **Biomedical Subject Bias**: Subjects are limited to life sciences; physics, chemistry, and materials science experimental figures (SEM, EDS, crystals) are not covered.
- **No Training Set Provided**: Only a zero-shot evaluation benchmark; how to "train" experimental image perception remains open. SPUR-Train for instruction tuning is suggested.
- **Training-free Baseline MCoTs**: Lacks comparison with training-based MCoTs (e.g., R1-V, MM-R1), so it cannot be confirmed if RL can overcome NP bottlenecks.

## Related Work & Insights
- **vs. MicroVQA (CVPR 2025)**: MicroVQA uses experimental images but averages 1.9 panels/image and focuses only on Qual.; SPUR offers an order-of-magnitude increase in complexity (14.3 panels) and quantitative coverage.
- **vs. MMMU / M3CoT / ScienceQA**: These use 1–2.5 panels of non-experimental figures and lack cross-panel modeling; SPUR fills the gap in real experimental figure understanding.
- **vs. SciAssess / MMSci**: These have high image isolation ($\leq 8$ panels); SPUR pushes physical complexity to match real-world high-impact figures.
- **Insight**: In multimodal scientific reasoning, perception and reasoning accuracy should be reported separately; otherwise, gains from MCoT/SFT/RL cannot be correctly attributed.

## Rating
- **Novelty**: ⭐⭐⭐⭐ While experimental multi-panel images have been explored (MicroVQA), this work pushes complexity to 14.3 panels and provides a clear 7-task diagnosis.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 20 MLLMs, 4 MCoTs, 5 subjects, and decoupled perception-reasoning analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Excellent diagnostic charts (Figs 1, 5, 6) support clear claims: "benchmark gap $\rightarrow$ seven tasks $\rightarrow$ diagnostic findings $\rightarrow$ MCoT failure causes."
- **Value**: ⭐⭐⭐⭐ High utility as a diagnostic benchmark for AI4S, with actionable insights on the perception-reasoning bottleneck.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ChemVLR: Prioritizing Reasoning in Perception for Chemical Vision-Language Understanding](chemvlr_prioritizing_reasoning_in_perception_for_chemical_vision-language_unders.md)
- [\[ACL 2026\] SciMDR: Advancing Scientific Multimodal Document Reasoning](scimdr_advancing_scientific_multimodal_document_reasoning.md)
- [\[ACL 2026\] A Survey of Multimodal Mathematical Reasoning: From Perception, Alignment to Reasoning](a_survey_of_multimodal_mathematical_reasoning_from_perception_alignment_to_reaso.md)
- [\[ACL 2026\] Position: Multimodal Large Language Models Can Significantly Advance Scientific Reasoning](position_multimodal_large_language_models_can_significantly_advance_scientific_r.md)
- [\[ACL 2026\] GeoRC: A Benchmark for Geolocation Reasoning Chains](georc_a_benchmark_for_geolocation_reasoning_chains.md)

</div>

<!-- RELATED:END -->
