---
title: >-
  [Paper Note] Limited Linguistic Diversity in Embodied AI Datasets
description: >-
  [ACL 2026][Robotics][VLA Dataset Auditing] This paper conducts a systematic "linguistic diversity checkup" on mainstream VLA training corpora (RT-1, BRIDGE, TacoPlay, Language Table, LIBERO). By quantifying lexical…
tags:
  - "ACL 2026"
  - "Robotics"
  - "VLA Dataset Auditing"
  - "Lexical Diversity"
  - "Semantic Diversity"
  - "Syntactic Diversity"
  - "Open X-Embodiment"
date: 2026-05-08
content_hash: 15facc176175f919
---

# Limited Linguistic Diversity in Embodied AI Datasets

**Conference**: ACL 2026  
**arXiv**: [2601.03136](https://arxiv.org/abs/2601.03136)  
**Code**: To be confirmed  
**Area**: Embodied AI / Data Analysis / VLA / Linguistic Diversity  
**Keywords**: VLA Dataset Auditing, Lexical Diversity, Semantic Diversity, Syntactic Diversity, Open X-Embodiment

## TL;DR
This paper conducts a systematic "linguistic diversity checkup" on mainstream VLA training corpora (RT-1, BRIDGE, TacoPlay, Language Table, LIBERO). By quantifying lexical, semantic, and syntactic dimensions, it finds: **< 2% unique instructions**, RT-1 has only **49 unique words** in total, and **negation/conditional sentences appear in < 1%** of data. This "templated poverty" is significantly lower than instruction-tuning corpora (OASST2 93%, Alpaca 99.8% unique) and likely causes VLA model fragility to paraphrases and generalization failure.

## Background & Motivation
**Background**: VLA models like OpenVLA, RT-X, and $\pi$0.5 are primarily trained on large-scale data like Open X-Embodiment (OXE). While OXE documents emphasize diversity in objects, scenes, and embodiments, they **rarely report the properties of the instruction language itself**. Meanwhile, research has observed VLA models' sensitivity to paraphrases and fragility to distractors (Gao 2025, AgiBotWorld 2025, Wang 2024).

**Limitations of Prior Work**: Existing VLA work treats instructions as auxiliary labels; none systematically quantify what the **linguistic signals** in training data actually look like. Despite poor robustness to paraphrasing, it remains unknown: (a) How many instructions are repetitive during training? (b) How rich is the vocabulary? (c) Is the syntactic structure diverse? (d) How frequent are common real-world constructs like negation or conditionals? These represent significant gaps.

**Key Challenge**: The VLA community pursues "general-purpose robots + natural language instructions," but training data may be **toy-level in the linguistic dimension**. Models trained on millions of episodes might only see combinations of a few dozen template words. If training data is so linguistically impoverished, the rich linguistic capabilities inherited from LLM backbones may be overwritten or suffer from catastrophic forgetting.

**Goal**: (1) Establish an **operable multi-dimensional quantification framework** for "instruction linguistic diversity"; (2) Conduct a **systematic audit** of mainstream VLA datasets compared with non-robotics corpora (instruction tuning, dialogue); (3) Propose **targeted data augmentation/collection strategies** based on audit results.

**Key Insight**: Borrowing the framework from Tevet & Berant 2021 that divides diversity into form vs. content, this work further subdivides it into lexical, semantic, and syntactic axes. Each axis uses complementary metrics to avoid bias, with reference datasets (OASST2/Alpaca/LLaVA-Instruct/ALFRED/SCOUT) provided for comparison. This benchmark allows readers to intuitively perceive the deviation of VLA corpora without prescribing an "ideal metric value."

**Core Idea**: Use a three-dimensional multi-metric audit and cross-domain reference corpora to transform "linguistic poverty" from a feeling into concrete numbers.

## Method

### Overall Architecture
- **Input**: VLA datasets (RT-1, BRIDGE, TacoPlay, Language Table, LIBERO) + reference datasets (OASST2, Alpaca, LLaVA-Instruct, ALFRED, SCOUT).
- **Three Analysis Dimensions (A1/A2/A3)**: A1: Duplication & Lexical Diversity; A2: Semantic Diversity; A3: Syntactic & Structural Diversity.
- **Multi-metric Complementarity**: Avoids inherent bias of single metrics (e.g., BERTScore's insensitivity to word order).
- **Output**: Profiles across ~10 quantitative metrics + cross-domain comparison tables + improvement suggestions (augmentation/transfer/collection guides).

### Key Designs

1.  **Analysis 1: Duplication & Lexical Diversity**:
    - **Function**: Quantify repetition and vocabulary range.
    - **Mechanism**: Basic statistics (#Sent, #Uniq, % Uniq, #Words); diversity metrics—**Compression Ratio (CR)** (gzip-based; lower is more diverse), ROUGE-L, BLEU, Jaccard, Levenshtein.
    - **Design Motivation**: Data deduplication significantly impacts generalization in LLMs. High duplication rates might lead VLA models to memorize training instructions rather than generalize. CR provides a global view (entire corpus) complementary to pairwise metrics like ROUGE.

2.  **Analysis 2: Semantic Diversity**:
    - **Function**: Quantify different task semantics expressed.
    - **Mechanism**: Pairwise BERTScore (sampled similarity); **PCA on sentence embeddings** (using USE/SBERT/CLIP/SONAR encoders, reporting components for 95% variance); **Verb–Direct Object (VO) co-occurrence matrix**.
    - **Design Motivation**: Embedding-based metrics are robust to paraphrasing (capturing *what* is said rather than *how*). VO co-occurrence is a domain-specific interpretable dimension—if "banana" is only paired with "pick," the model learns a verb-object shortcut (simplicity bias).

3.  **Analysis 3: Structural Diversity**:
    - **Function**: Quantify grammatical skeleton and high-order logic.
    - **Mechanism**: POS pattern frequency + **Constituency Tree Kernel** pairwise similarity; **High-order construct detection** (Negation, Conditionals, Multi-step, Cycle) using dependency parses and keyword patterns.
    - **Design Motivation**: Syntactic poverty amplifies model bias. Negation, conditionals, and loops are essential for real-world robot commands ("Don't take the rotten apple"), yet are nearly absent in current VLA training.

### Loss & Training
This is a **pure dataset audit/empirical study** and does not train any models. Calculations utilize spaCy for POS/dependency, and USE/SBERT/CLIP/SONAR for sentence embeddings. Diversity metrics are averaged over 1,000 samples with 3 repetitions.

## Key Experimental Results

### Main Results: Cross-dataset Multi-dimension Comparison (Table 2)

| Dataset | # Sent | # Uniq (% Uniq) | # Words | CR ↓ | ROUGE-L ↓ | BERTScore ↓ | USE PCA ↑ | Tree Kernel ↓ |
|--------|--------|------------------|---------|------|-----------|--------------|------------|----------------|
| **Instruction Tuning** | | | | | | | | |
| OASST2 | 42K+ | 39,301 (**93.33%**) | 35,445 | 2.75 | 0.05 | 0.45 | 254 | 2.25% |
| Alpaca | 53K+ | 52,996 (**99.81%**) | 18,141 | 3.20 | 0.10 | 0.57 | 231 | 3.66% |
| LLaVA-Instruct | 366K+ | 261,892 (71.45%) | 15,477 | 4.41 | 0.21 | 0.61 | 184 | 7.46% |
| **Language-guided Robotics** | | | | | | | | |
| ALFRED | 162K+ | 126,005 (79.9%) | 2,627 | 5.91 | 0.21 | 0.64 | 159 | 5.71% |
| SCOUT | 23K+ | 8,795 (39.4%) | 1,631 | 4.85 | 0.07 | 0.49 | 148 | 1.89% |
| **VLA Datasets** | | | | | | | | |
| RT-1 | 3.7M+ | **577 (0.02%)** | **49** | **118.20** | 0.19 | 0.64 | **33** | 5.09% |
| BRIDGE | 864K+ | 11,693 (**1.4%**) | 1,189 | 64.90 | 0.15 | 0.60 | 125 | 3.68% |
| TacoPlay | 214K | 403 (**0.2%**) | 74 | 158.86 | 0.30 | 0.68 | **42** | 8.86% |
| Language Table | 7.0M+ | 127,370 (1.81%) | 928 | 56.64 | 0.29 | 0.70 | 86 | 9.19% |
| LIBERO | 6.5K | 112 (**1.72%**) | 79 | 134.86 | 0.38 | 0.71 | **34** | 12.22% |

**Shocking Numbers**:
- RT-1 contains **3.7M sentences but only 577 are unique (0.02% uniqueness rate)**, using only **49 unique words** in the entire corpus.
- VLA datasets have CR (compression ratio) of 56-158, far higher than the 2.75-4.41 of instruction tuning corpora—indicating extreme redundancy.
- USE PCA intrinsic dimensionality for VLA datasets (33-125) is far lower than non-VLA corpora (148-254).

### Key Findings (Table from Figure 5: High-order Construct Ratios)

| Construct Type | Avg. VLA Ratio | Avg. Non-VLA Ratio | Real-world Need |
|-----------|----------------|-----------------|----------------|
| Negation | < 1% | Higher in ALFRED/SCOUT | "Don't take rotten apple" — Safety critical |
| Conditional | < 1% | < 2% | "If... then..." — Exception handling |
| Multi-step | Medium to High | Medium | Sequential logic, **the only well-covered area** |
| Cycle | Nearly 0 | Slight signal in SCOUT/ALFRED | "Repeat until..." — Long-horizon tasks |

### POS Pattern Concentration (Figure 4)

| Dataset | Most Frequent POS Pattern Ratio | Example |
|--------|--------------------------|------|
| TacoPlay | **24%** | VERB→DET→ADJ→NOUN→ADP→DET→NOUN ("put the purple block on the table") |
| RT-1 | 11% | VERB→NOUN→NOUN→ADP→ADJ→NOUN ("place water bottle into white bowl") |
| BRIDGE | 3% | More diverse than RT-1/TacoPlay |
| Language Table | 4% | Comparable to BRIDGE |

### Key Findings
- **#Episode $\neq$ Linguistic Diversity**: RT-1 has 3.7 million commands but only 577 unique sentences; seeing the same 577 sentences 3.7M times is a massive data inefficiency from an LLM training perspective.
- **Extreme Vocabulary Concentration**: Only 4 words overlap across all audited VLA datasets: `move, close, open, pick`—this is effectively the "action verb vocabulary" for current VLA models.
- **Verb-Object Shortcut Bias**: In RT-1, "banana" is almost always paired with "pick," and "knock" with can-shaped objects—it is easy for models to learn the shortcut "see banana $\rightarrow$ pick" while **ignoring linguistic instructions**.
- **Structural Poverty is More Severe**: Negation/Conditional/Cycle < 1% means VLA models never learn "Don't do X" or "If Y then Z," structures essential for real-world deployment.
- **SCOUT (Wizard-of-Oz) is significantly better**: Uniqueness rate of 39.4% and 1,631 unique words prove **interactive collection** yields more diverse language than scripted/teleoperated methods.

## Highlights & Insights
- **Quantifying "VLA linguistic poverty"**: This work fills a community gap by providing hard data (e.g., RT-1's 49 unique words) to support qualitative complaints. It essentially provides a "datasheet for datasets."
- **Cross-domain Comparison**: Placing VLA data on the same scale as OASST2/Alpaca makes the gap tangible—CR=158 vs. CR=2.75 is a powerful wake-up call.
- **VO Heatmaps as Shortcut Diagnostic**: A simple tool that can be applied to any language-conditioned imitation learning dataset to identify spurious correlations.
- **Actionable Improvement Suggestions**: Proposals include (i) targeted augmentation using LLMs for syntactic paraphrasing; (ii) cross-domain transfer with procedural text; and (iii) real-time rephrase prompts during data collection.

## Limitations & Future Work
- **No Evaluation of Cross-modal Alignment**: Focuses solely on text without verifying instruction-image-trajectory consistency—linguistic diversity is moot if grounding is incorrect.
- **Correlation $\neq$ Causality**: Does not provide a direct experiment showing that re-training VLA on "rich language" data improves robustness.
- **Metric Limitations**: BERTScore is insensitive to order; Tree Kernels can be unstable for long sentences.
- **English-centric**: Analysis is limited to English VLA corpora.
- **Future Work**: Proposes (1) mandatory dataset cards using this framework; (2) controlled experiments on the impact of linguistic enrichment; (3) expansion to multilingual VLA; (4) new benchmarks focused on negation/conditional logic.

## Related Work & Insights
- **vs. Xing et al. 2025**: While prior work focused on visual shortcuts (viewpoint, background), this study补全 the linguistic audit dimension.
- **vs. OXE Original Paper**: Adds the "missing linguistic subsection" to the original OXE documentation which focused on embodiment diversity.
- **vs. Shaib et al. 2025**: Adopts the Compression Ratio as a dataset-level diversity proxy, validating its effectiveness in the robotics domain.
- **vs. Bender 2019/2021**: Follows the "data transparency" movement by making language a mandatory reporting item for robotics data cards.

## Rating
- Novelty: ⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] D2E: Scaling Vision-Action Pretraining on Desktop Data for Transfer to Embodied AI](../../ICLR2026/robotics/d2e_scaling_vision-action_pretraining_on_desktop_data_for_transfer_to_embodied_a.md)
- [\[ICLR 2026\] Grounding Generative Planners in Verifiable Logic: A Hybrid Architecture for Trustworthy Embodied AI](../../ICLR2026/robotics/grounding_generative_planners_in_verifiable_logic_a_hybrid_architecture_for_trus.md)
- [\[AAAI 2026\] Recursive Visual Imagination and Adaptive Linguistic Grounding for Vision Language Navigation](../../AAAI2026/robotics/recursive_visual_imagination_and_adaptive_linguistic_grounding_for_vision_langua.md)
- [\[NeurIPS 2025\] MineAnyBuild: Benchmarking Spatial Planning for Open-world AI Agents](../../NeurIPS2025/robotics/mineanybuild_benchmarking_spatial_planning_for_openworld_ai.md)
- [\[ICCV 2025\] Interaction-Merged Motion Planning: Effectively Leveraging Diverse Motion Datasets for Robust Planning](../../ICCV2025/robotics/interaction-merged_motion_planning_effectively_leveraging_diverse_motion_dataset.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICLR 2026\] D2E: Scaling Vision-Action Pretraining on Desktop Data for Transfer to Embodied AI](../../ICLR2026/robotics/d2e_scaling_vision-action_pretraining_on_desktop_data_for_transfer_to_embodied_a.md)
- [\[ICLR 2026\] Grounding Generative Planners in Verifiable Logic: A Hybrid Architecture for Trustworthy Embodied AI](../../ICLR2026/robotics/grounding_generative_planners_in_verifiable_logic_a_hybrid_architecture_for_trus.md)
- [\[AAAI 2026\] Recursive Visual Imagination and Adaptive Linguistic Grounding for Vision Language Navigation](../../AAAI2026/robotics/recursive_visual_imagination_and_adaptive_linguistic_grounding_for_vision_langua.md)
- [\[CVPR 2025\] Magma: A Foundation Model for Multimodal AI Agents](../../CVPR2025/robotics/magma_a_foundation_model_for_multimodal_ai_agents.md)
- [\[ICCV 2025\] Interaction-Merged Motion Planning: Effectively Leveraging Diverse Motion Datasets for Robust Planning](../../ICCV2025/robotics/interaction-merged_motion_planning_effectively_leveraging_diverse_motion_dataset.md)

</div>

<!-- RELATED:END -->
