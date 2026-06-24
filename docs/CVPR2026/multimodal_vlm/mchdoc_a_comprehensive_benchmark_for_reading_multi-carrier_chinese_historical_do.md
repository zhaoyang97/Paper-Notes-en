---
title: >-
  [Paper Note] MCHDoc: A Comprehensive Benchmark for Reading Multi-Carrier Chinese Historical Documents
description: >-
  [CVPR 2026][Multimodal VLM][Chinese Historical Documents] MCHDoc organizes 15,724 high-resolution historical document images spanning over 3,000 years and six writing carriers (ancient paper, bamboo/wood slips, calligraphy rice paper, stone inscriptions, silk, and oracle bones) into a unified benchmark. Mirroring the expert workflow of "recognition followed by textual research and correction," it designs four tasks: page-level recognition, character-level recognition…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Chinese Historical Documents"
  - "Multi-carrier OCR"
  - "Post-correction"
  - "MLLM Evaluation"
  - "Retrieval-Augmentation"
date: 2026-05-08
content_hash: 6e7bdaa8b73a2f17
---

# MCHDoc: A Comprehensive Benchmark for Reading Multi-Carrier Chinese Historical Documents

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Sheng_MCHDoc_A_Comprehensive_Benchmark_for_Reading_Multi_Carrier_Chinese_Historical_Documents_CVPR_2026_paper.html)  
**Code**: https://github.com/blackprotoss/MCHDoc (Release promised in the paper)  
**Area**: Multimodal VLM / Document Recognition Benchmark  
**Keywords**: Chinese Historical Documents, Multi-carrier OCR, Post-correction, MLLM Evaluation, Retrieval-Augmentation

## TL;DR
MCHDoc organizes 15,724 high-resolution historical document images spanning over 3,000 years and six writing carriers (ancient paper, bamboo/wood slips, calligraphy rice paper, stone inscriptions, silk, and oracle bones) into a unified benchmark. Mirroring the expert workflow of "recognition followed by textual research and correction," it designs four tasks: page-level recognition, character-level recognition, pure LLM post-correction, and knowledge-base augmented post-correction. Systematic evaluation of over 20 open-source and closed-source MLLMs/LLMs reveals that even top-tier models struggle with cross-carrier generalization.

## Background & Motivation
**Background**: The digitization and interpretation of Chinese historical documents long depend on manual transcription and textual correction by experts, a process that is extremely time-consuming. Each carrier (oracle bones, slips, inscriptions, etc.) requires highly specialized domain knowledge—an expert in Ming Dynasty woodblock prints may not accurately transcribe Shang Dynasty oracle bones. With MLLM/LLMs showing potential in Chinese document processing, a natural question arises: can they truly read these cross-carrier historical documents?

**Limitations of Prior Work**: The authors identify three deficiencies in existing research. First, most work emphasizes "cultural analysis" rather than the fundamental recognition capability of "reading the characters." Second, many studies ignore "textual post-correction via research and citation"—step critical for ensuring historical integrity and supporting reliable academic research. Third, existing Chinese historical document benchmarks feature single carriers and small scales, making them insufficient for systematic studies of Chinese history.

**Key Challenge**: The real-world complexity of historical documents stems from "carrier heterogeneity"—different materials result in vastly different image sizes, layouts, degradation levels, and glyph styles. Previous benchmarks covered only single or few carriers, failing to expose the true bottlenecks in cross-carrier generalization.

**Goal**: Construct a large-scale comprehensive benchmark that covers multiple carriers, supports both recognition and post-correction, and honestly measures the capability boundaries and failure modes of current large models.

**Key Insight**: Mirror the expert workflow in the evaluation design: experts first recognize characters page by page and then correct them by consulting historical records or domain knowledge. Accordingly, the benchmark is split into two phases: recognition and post-correction. The latter distinguishes between "using only internal model knowledge" and "accessing external ancient document knowledge bases."

**Core Idea**: Utilize a large-scale benchmark of 15,724 pages across six carriers, spanning from the 16th century BC to the 20th century AD, combined with four types of tasks and three sets of metrics, to establish a standardized, reproducible, and histographically grounded systematic evaluation for "large models reading historical documents."

## Method
MCHDoc is essentially a dataset and evaluation protocol rather than a new model architecture. The focus lies on data collection, knowledge base construction, and task/metric definition. There is no multi-stage trainable pipeline; the mechanism is described via text and formulas.

### Overall Architecture
The benchmark consists of three components: ① a multi-carrier document corpus (including a manually annotated inscription subset); ② an external knowledge base containing over 1.7 billion characters from nearly 16,000 ancient books; ③ four evaluation tasks with three similarity metrics. Given a historical document image, the model either directly outputs the full-page text (page-level) or single characters (character-level), or performs post-correction on existing recognition results. Post-correction is divided into "correction based on internal knowledge" and "correction supported by retrieved historical evidence." This design deliberately aligns with the real-world "recognition → research → correction" workflow.

### Key Designs

**1. Six-carrier Heterogeneous Corpus: Quantifying Cross-Carrier Generalization**

The benchmark deliberately covers six carriers with vast differences in material and era: Ancient Paper (from M5HisDoc), Slips (from DeepJianDu), Calligraphy (from CalliBench), Inscriptions (self-collected from museums and manually annotated), Silk (from Wa-net, character-level only due to degradation), and Oracle Bones (from OBI-Bench, character-level). This heterogeneity is reflected in statistics (see Tab. 2): image sizes span two orders of magnitude, from $77\times135$ for oracle bone fragments to $2327\times2039$ for ancient book pages. The authors summarize three major challenge factors: huge variance in pixel scales, highly unbalanced character counts, and cross-carrier glyph diversity. These factors make MCHDoc a genuine test of cross-carrier generalization rather than a "leaderboard chasing" exercise on single carriers.

**2. Manual Annotation of Inscriptions and KB Construction: Executable Post-Correction**

To enhance carrier diversity and temporal coverage, the authors performed large-scale manual annotation for the most difficult carrier: inscriptions (rubbings). Transcripts were produced by annotators trained in paleography, cross-checked by peers, and verified against epigraphic literature for difficult characters. Final versions were archived structurally by dynasty and stele name. The post-correction task relies on an external knowledge base sourced mainly from "Daizhigao," containing nearly 16,000 ancient books across ten categories. Due to varying document lengths, the knowledge base uses an adaptive hierarchical chunking strategy. This knowledge base enables "post-correction with citations" to transition from an abstract goal to a reachable, reproducible evaluation.

**3. Four Tasks + Three Similarity Metrics: Scoring the Expert Workflow**

The four tasks are: Page-level Recognition (end-to-end full-page reading), Character-level Recognition (extracting single characters from difficult pages; 500 samples per carrier to focus on visual decoding over language modeling), Pure LLM Post-correction (inputting Doubao's page-level output to LLMs without external search), and KB-Augmented Post-correction (accessing the epigraphic knowledge base). Evaluation utilizes three string similarity metrics: Accurate Rate $\mathrm{AR}=(N_t-D_e-S_e-I_e)/N_t$, Correct Rate $\mathrm{CR}=(N_t-D_e-S_e)/N_t$ (where $D_e,S_e,I_e$ are deletion, substitution, and insertion errors, and $N_t$ is total characters; CR omits insertion errors, so $\mathrm{CR} \ge \mathrm{AR}$), and $1\text{-NED}=1-\frac{D_{edit}}{\max(L_{pred},L_{true})}$ (where $D_{edit}$ is Levenshtein edit distance). ⚠️ Specific formulas follow those in the original paper. To reflect practicality, missing predictions due to model limits or safety filters (totaling <3%) are recorded as zeros.

## Key Experimental Results

### Main Results
In benchmark comparison, MCHDoc significantly exceeds existing datasets in scale and coverage. It is the only benchmark supporting both OCR and post-correction across all six carriers.

| Benchmark | Scale | Ancient Paper | Slips | Calligraphy | Inscriptions | Silk | Oracle Bones | OCR | Post-correction |
|------|------|------|------|------|------|------|------|-----|--------|
| M5HisDoc | 8,000 | ✓ | × | × | × | × | × | ✓ | × |
| CalliBench | 3,192 | × | × | ✓ | × | × | × | ✓ | × |
| AncientDoc | 3,000 | ✓ | × | × | × | × | × | ✓ | × |
| OBI-Bench | 1,500 | × | × | × | × | × | ✓ | ✓ | × |
| **MCHDoc (Ours)** | **15,724** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

Carrier-specific statistics reflect huge discrepancies in scale and character count:

| Carrier | Scale | Avg. Size | Avg. Char Count |
|------|------|----------|----------|
| Ancient Paper | 3,000 | 2327×2039 | 577 |
| Slips | 3,000 | 278×1834 | 14 |
| Calligraphy | 3,192 | 1211×1647 | 38 |
| Inscriptions | 5,152 | 691×1594 | 15 |
| Silk | 881 | 114×162 | — (Char-level) |
| Oracle Bones | 499 | 77×135 | — (Char-level) |

In page-level recognition, the closed-source Doubao-Seed-1.6-VL is the strongest overall, but all models fail on Slips, highlighting the cross-carrier challenge (values are AR/CR/1-NED, %):

| Model | Ancient Paper (1-NED) | Slips (1-NED) | Calligraphy (1-NED) | Inscriptions (1-NED) |
|------|------|------|------|------|
| GPT-4o | 3.92 | 4.35 | 25.20 | 18.91 |
| GPT-5 | 6.87 | 10.42 | 31.23 | 50.86 |
| Doubao-Seed-1.6-VL | 56.52 | 18.82 | 49.81 | ⚠️ ~68 (from original table) |

### Ablation Study
The "ablations" correspond to model behavior comparisons across different tasks/settings:

| Task Setting | Key Phenomenon | Explanation |
|----------|---------|------|
| Page-level vs. Char-level | InternVL3.5-1B(LoRA) rivals or exceeds larger general models | Domain adaptation is more critical than parameter scale, especially at character level |
| Pure LLM Post-correction | Almost all LLMs degrade the original 1-NED, worst for Inscriptions | Blind correction without external knowledge is "unsafe" for high-risk historical digitization |
| KB-Augmented Post-correction | Significant improvement for Ancient Paper/Calligraphy, often exceeding original OCR | Reasoning-capable variants become stronger when external evidence is available |

### Key Findings
- **Slips are the most difficult carrier**: All models perform poorly on both page-level and character-level tasks. Severe degradation and unique textures remain significant challenges; low recognition quality for slips also hampers downstream post-correction.
- **Domain Adaptation ≈ Model Scale**: The fine-tuned Qwen2.5-VL-3B(SFT) nearly matches the closed-source Doubao on difficult carriers. Increasing parameters does not yield monotonic gains; compact domain-adapted models can often rival larger general models.
- **Reasoning is a double-edged sword**: When relying only on internal knowledge, models with aggressive reasoning (e.g., Gemini/Deepseek series) introduce more degradation. However, once an external knowledge base is introduced, reasoning variants (e.g., Gemini-2.5-Flash-Think, Deepseek-V3.1-Think) consistently outperform non-reasoning versions—suggesting that "verifiable evidence" is the prerequisite for reasoning to be beneficial.

## Highlights & Insights
- **Decomposition of the expert workflow**: Splitting the problem into recognition, pure internal correction, and KB-supported correction allows the human-dependent task of "reading historical documents" to be quantified and reproduced.
- **Carrier heterogeneity as a challenge generator**: By utilizing two orders of magnitude in pixel scales, unbalanced character distributions, and glyph diversity, MCHDoc turns "cross-carrier generalization" into measurable metrics rather than an empty slogan.
- **"Internal correction is unsafe, external is effective" is a transferable finding**: This conclusion serves as a warning for any high-risk text correction scenario requiring factual evidence (legal, medical, historical)—blindly letting LLMs edit text may exacerbate errors.

## Limitations & Future Work
- **Benchmark, not a solution**: MCHDoc identifies problems and analysis directions but does not propose a new model for cross-carrier generalization; the "solution" is left for future work.
- **Carrier scale imbalance**: Sample sizes for Silk (881) and Oracle Bones (499) are significantly smaller and restricted to character-level tasks. Statistical robustness for these categories is weaker than for Ancient Paper/Inscriptions.
- **Dependency on a single recognition source**: Pure LLM post-correction unified Doubao's output as input; conclusions may be influenced by this specific recognizer's error distribution.
- **Temporal snapshot**: Models were evaluated based on versions released before October 1, 2025; results are a snapshot and require re-testing as models iterate.

## Related Work & Insights
- **vs. M5HisDoc / AncientDoc**: These are single-carrier (ancient paper) large-scale recognition benchmarks focusing only on OCR. MCHDoc expands to six carriers and introduces post-correction tasks at a larger scale.
- **vs. CalliBench / OBI-Bench / DeepJianDu / CIRI**: These focus on individual carriers like calligraphy, oracle bones, slips, or inscription restoration. MCHDoc integrates these into a unified framework and adds manually annotated inscriptions.
- **vs. Modern Chinese GEC (PLOME / GrammarGPT)**: Modern Chinese error correction models struggle to transfer to ancient prose due to grammatical differences. MCHDoc bypasses this by using retrieval-augmented post-correction with ancient document KBs.

## Rating
- Novelty: ⭐⭐⭐⭐ First unified recognition + post-correction benchmark for six carriers, though essentially an ensemble of datasets and evaluation design rather than an architectural innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive cross-carrier analysis of 20+ models across four tasks and three metrics.
- Writing Quality: ⭐⭐⭐⭐ Tasks and metrics are clearly explained with insightful analysis, though discussion on scale imbalance could be more detailed.
- Value: ⭐⭐⭐⭐⭐ High community value for providing a standardized, historiographically grounded benchmark for digital humanities and historical OCR.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Beyond Single Images: A Comprehensive Benchmark for Album-Level Vision-Language Understanding](beyond_single_images_a_comprehensive_benchmark_for_album-level_vision-language_u.md)
- [\[CVPR 2026\] Is your VLM Sky-Ready? A Comprehensive Spatial Intelligence Benchmark for UAV Navigation](is_your_vlm_sky-ready_a_comprehensive_spatial_intelligence_benchmark_for_uav_nav.md)
- [\[CVPR 2026\] Twin-T & TwintVQA: A Reliable Structure-Detail Separating VLM and a Comprehensive Benchmark for Chart and Table Tasks](twin-t_twintvqa_a_reliable_structure-detail_separating_vlm_and_a_comprehensive_b.md)
- [\[ACL 2025\] AGRI-CM3: A Chinese Massive Multi-Modal Multi-Level Benchmark for Agricultural Understanding](../../ACL2025/multimodal_vlm/agri-cm3_a_chinese_massive_multi-modal_multi-level_benchmark_for_agricultural_un.md)
- [\[AAAI 2026\] OIDA-QA: A Multimodal Benchmark for Analyzing the Opioid Industry Documents Archive](../../AAAI2026/multimodal_vlm/oida-qa_a_multimodal_benchmark_for_analyzing_the_opioid_industry_documents_archi.md)

</div>

<!-- RELATED:END -->
