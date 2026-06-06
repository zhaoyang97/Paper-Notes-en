---
title: >-
  [Paper Note] NiuTrans.LMT: Toward Inclusive and Scalable Multilingual Machine Translation with LLMs
description: >-
  [ACL 2026][Multilingual & Machine Translation][Multilingual Machine Translation] This paper introduces NiuTrans.LMT, an open-source LLM machine translation suite covering 60 languages and 234 translation directions cente…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "Multilingual Machine Translation"
  - "Directional Degeneration"
  - "Strategic Downsampling"
  - "Parallel Multilingual Prompting"
  - "GRPO"
date: 2026-05-08
content_hash: be6a6b19d2ac1b4c
---

# NiuTrans.LMT: Toward Inclusive and Scalable Multilingual Machine Translation with LLMs

**Conference**: ACL 2026  
**arXiv**: [2511.07003](https://arxiv.org/abs/2511.07003)  
**Code**: <https://github.com/NiuTrans/LMT>  
**Area**: Machine Translation / Multilingual / LLM Adaptation  
**Keywords**: Multilingual Machine Translation, Directional Degeneration, Strategic Downsampling, Parallel Multilingual Prompting, GRPO

## TL;DR
This paper introduces NiuTrans.LMT, an open-source LLM machine translation suite covering 60 languages and 234 translation directions centered on Chinese and English, across four scales: 0.6B, 1.7B, 4B, and 8B. It identifies that multi-way parallel data in symmetric SFT causes "Directional Degeneration" in the X→En/Zh direction. The performance is restored to the level of strong open-source MMT systems using Strategic Downsampling, Parallel Multilingual Prompting, and GRPO with COMET rewards.

## Background & Motivation

**Background**: LLM machine translation has shifted from "training a standalone encoder-decoder MT model" to "CPT + SFT + preference optimization on a general base LLM." Systems like ALMA, TowerInstruct, X-ALMA, GemmaX2, Hunyuan-MT, and Seed-X have proven this path effective. However, most systems either have limited language coverage, are English-centric, or fail to fully resolve bidirectional quality issues for long-tail languages beyond Chinese and English.

**Limitations of Prior Work**: High-quality human parallel corpora are the most desired data for multilingual SFT, but such data is extremely scarce for low-resource languages. Consequently, multi-way corpora like FLORES-200 and NTREX-128 are repeatedly reused. Intuitively, multi-way data can construct many directions from a small set of corpora; the problem is that when the same English or Chinese sentence is repeatedly mapped as the target from dozens of source languages, the model may learn a "shortcut" to memorize the target sentence upon seeing certain training patterns, rather than carefully reading the source semantics.

**Key Challenge**: On one hand, multi-way parallel data is the most reliable source of high-quality supervision for long-tail languages; on the other hand, symmetrical reuse creates massive many-to-one target repetitions. While the model benefits when generating multiple target languages in the pivot→X direction, it exhibits "Directional Degeneration"—fluent but unfaithful hallucinations—in the X→pivot direction.

**Goal**: The authors aim to solve three problems simultaneously: (i) explain why large-scale multilingual SFT collapses in the reverse direction; (ii) fix this issue without relying on additional new SFT data; and (iii) train and release a family of multilingual translation models that are Chinese-English dual-centric, sufficiently broad in coverage, and available at different parameter scales.

**Key Insight**: Instead of inventing complex architectures, the paper attributes the problem to data usage: the same pivot target is symmetrically reused too many times, causing source semantics to be overridden by shortcuts. This perspective is practical because if the root cause lies in data distribution, a simple sampling strategy might be more stable and cost-effective than model-level interventions.

**Core Idea**: Use "Strategic Downsampling" (retaining only a small amount of reverse data while fully retaining forward data) to break many-to-one target repetition. Then, use "Parallel Multilingual Prompting" with auxiliary parallel sentences to explicitly provide cross-lingual semantic anchors. Finally, integrate both into a complete LMT training pipeline consisting of 90B-token CPT + high-quality SFT + GRPO.

## Method

### Overall Architecture
LMT uses Qwen3 as the base and trains four models (0.6B, 1.7B, 4B, 8B), covering English ↔ 59 languages and Chinese ↔ 58 languages, totaling 234 translation directions. The overall pipeline consists of three stages: 1) Continued Pre-training (CPT), reinforcing multilingual capabilities with 90B tokens of mixed corpora (one-third each of monolingual, English-centric bilingual, and Chinese-centric bilingual data); 2) SFT, performing instruction-style translation supervision on high-quality corpora such as FLORES/NTREX/SMol/WMT/IWSLT while incorporating SD and PMP; 3) GRPO, sampling multiple candidate translations using the same SFT prompts and using COMET-22 as a reference-based reward for preference optimization, without constructing additional human preference data.

### Key Designs

1.  **Directional Degeneration Diagnosis and Strategic Downsampling**:
    - **Function**: Identify and mitigate the collapse of reverse translation (especially X→En/Zh) caused by symmetric reuse of multi-way parallel data.
    - **Mechanism**: The authors first perform standard bidirectional SFT with Qwen3-4B-Base and find that while En/Zh→X improves significantly, X→En/Zh performance falls below the base model, characterized by grammatical fluency but factual unfaithfulness. Further experiments were conducted across three axes: replacing reverse data with non-overlapping bilingual CPT subsets to break symmetry; gradually increasing the retention rate of reverse multi-way samples from 0% to 100%; and repeating this across Qwen3 (0.6B/1.7B/4B/8B), Llama-3.1-8B, Gemma-2-9B, and scales of 10-50 languages. Results show an inverted V-shape performance curve relative to the retention rate, peaking at approximately $p=5\%$ and collapsing most obviously at 100%. The final SD approach retains all En/Zh→X samples and independently samples X→En/Zh samples from multi-way corpora at $p=5\%$.
    - **Design Motivation**: This decomposes the "curse of multilinguality" into specific many-to-one data reuse issues. Compared to model-level solutions like direction-aware training or model merging, SD is a data-level fix: it does not change the architecture, introduce additional inference overhead, or sacrifice supervision density for pivot→X directions.

2.  **Parallel Multilingual Prompting (PMP)**:
    - **Function**: Provide the model with an auxiliary parallel sentence during SFT and optional inference stages, serving as a second linguistic perspective to anchor source semantics.
    - **Mechanism**: Standard translation prompts train $P_\theta(T\mid S;\tau_{L_S\to L_T})$; PMP extends the input to include the source sentence $S$ and an auxiliary language sentence $A$, training $P_\theta(T\mid S,A;\tau_{L_S\to L_A\to L_T})$. Auxiliary languages are selected strategically: for En↔X, a neighboring language of a similar type that the model masters well is chosen (e.g., Dutch for German, Czech for Polish); for Zh↔X, English is consistently used as a stable semantic anchor since it is usually the model's strongest and most easily self-generated intermediate language. SFT uses a mix of STP/PMP training; ordinary STP can still be used for default inference, switching to PMP prompts if external or self-generated auxiliary translations are available.
    - **Design Motivation**: Multi-way data brings not only many-to-one risks but also contains cross-lingual alignment value. The ingenuity of PMP lies in transforming "multi-way parallel" from an implicit data structure into an explicit prompt condition, teaching the model when to utilize another linguistic perspective rather than blindly expanding all directions symmetrically during training.

3.  **Scalable Training Pipeline for Chinese-English Dual-Centric MMT**:
    - **Function**: Implement the above strategies into a releasable, comparable model family covering long-tail languages rather than just performing a single ablation.
    - **Mechanism**: CPT data is collected from SlimPajama, Skywork, CulturaX, OpenDataLab, Wikimedia, OPUS, etc., and then pseudo-parallel expansion is performed using open-source MT systems, particularly to fill the gap in Chinese-centric corpora. The filtering chain includes OpusFilter for length and mismatch cleaning, FastText LID for hierarchical thresholds, and CometKiwi for quality scoring. This results in approximately 2.1B English-centric and 2.9B Chinese-centric sentence pairs. CPT adopts explicit direction tags and target language separators. SFT data includes 567K high-quality pairs covering 117 pivot language pairs; positive STP/PMP each 50%, reverse total retention 5% with STP 2.5% and PMP 2.5%. GRPO uses 8 rollouts, temperature 1.0, KL coefficient 0.001, and selects better candidates based on COMET-22 rewards.
    - **Design Motivation**: Low-resource translation cannot be solved by a single prompt trick; the real bottlenecks are data scale, quality, directional balance, and the connection between training stages. The systematic value of LMT lies in engineering these steps and training four sizes with the same recipe, proving the strategy's independence from specific model scales.

### Loss & Training
Both CPT and SFT use standard language model objectives, with the difference being that CPT primarily learns multilingual text and bilingual formats, while SFT only calculates loss on the target translation part. SFT forward directions use 50% STP + 50% PMP; reverse directions use only 5% after SD, consisting of 2.5% STP and 2.5% PMP. The GRPO stage follows SFT prompts; the model samples candidate translations, and COMET-22 assigns rewards based on the reference, effectively converting an automatic MT quality evaluator into a preference optimization signal, yielding an additional gain of approximately 0.3-0.8 COMET.

## Key Experimental Results

### Main Results
The first table shows COMET-22 as training components are incrementally enabled, focusing on the 4B model's most illustrative directions. It is evident that standard SFT provides massive gains for low-resource En/Zh→X but leads to a total collapse in X→Zh; adding SD immediately restores and exceeds the base model performance in reverse directions, and CPT continues to provide the largest gains for low-resource directions.

| 4B Configuration | High-resource X→Zh | Mid-resource X→Zh | Low-resource X→Zh | Low-resource En→X | Low-resource Zh→X |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Qwen3-4B-Base | 85.44 | 84.55 | 75.35 | 56.81 | 53.33 |
| SFT | 73.60 | 72.18 | 67.94 | 77.51 | 73.68 |
| + SD | 86.55 | 85.87 | 79.13 | 78.68 | 75.15 |
| + CPT | 87.39 | 87.06 | 84.74 | 87.14 | 84.17 |
| + PMP | 87.53 | 87.20 | 84.90 | 87.06 | 84.08 |
| + GRPO | **88.19** | **87.97** | **85.81** | **87.85** | **84.92** |

The second table compares LMT with existing MMT/Multilingual LLMs on overlapping language average scores. LMT-60-4B often matches or exceeds 7B-54B class systems, and the 8B version shows only small improvements over the 4B, indicating the high parameter efficiency of this recipe.

| Comparative System | Overlapping Langs | Baseline Avg. | LMT-60-4B Avg. | LMT-60-8B Avg. | Conclusion |
| :--- | :--- | :--- | :--- | :--- | :--- |
| TowerInstruct-13B | 10 | 87.63 | 88.34 | **88.43** | LMT small model beats 13B |
| Aya-expanse-8B | 23 | 87.36 | 88.26 | **88.36** | LMT leads stably |
| Seed-X-PPO-7B | 27 | **89.07** | 88.86 | 88.94 | LMT close to strong PPO system |
| GemmaX2-28-9B | 28 | 87.57 | 87.73 | **87.83** | LMT broader & slightly better |
| Hunyuan-MT-7B | 35 | 85.71 | 87.50 | **87.63** | LMT leads significantly |
| X-ALMA-13B | 40 | 88.92 | 88.96 | **89.06** | LMT 4B basically on par |
| Aya-101-13B | 54 | 83.85 | 87.42 | **87.55** | Significant long-tail advantage |
| NLLB-54B | 59 | 84.79 | 87.43 | **87.56** | Leads clearly despite much smaller size |

### Ablation Study

| Analysis Target | Setting | Key Findings | Notes |
| :--- | :--- | :--- | :--- |
| Directional Degeneration | Reverse multi-way retention from 0% to 100% | Peaks near $p=5\%$; 100% symmetric reuse causes clear decline | More reverse samples aren't always better; repeating pivot targets induces shortcuts |
| Symmetry-breaking | Replace X→En/Zh with non-overlapping bilingual CPT subsets | Dashed setting avoids collapse seen in full symmetric reuse | Degeneration stems from data reuse structure, not X→pivot difficulty |
| PMP inference | DT vs PMP-S vs PMP-O | Self-generated anchors often reach/surpass oracle on X→En/Zh; Zh→X relies more on oracle anchors | Translating into high-resource pivots is more tolerant of anchor noise |
| PMP zero-shot | In-Group directions without vs. with PMP | COMET increases from 85.20 to 86.11 | PMP training improves cross-lingual transfer, not just explicit anchor pairs |
| GRPO | Reusing SFT pairs, no new preference data | Avg. Gain across all resource layers approx. 0.3-0.8 COMET | Automatic rewards still extract additional quality from candidate generation |

### Key Findings
-   **Directional Degeneration is a systemic issue**: Similar asymmetric degradation occurs across multiple Qwen3 sizes, Llama-3.1-8B, Gemma-2-9B, and different language scales, indicating it is not an accidental bug of a specific base or language pair.
-   **SD gains are concentrated but critical**: It barely changes the supervision density for En/Zh→X while pulling the most affected directions like X→Zh from 67-73 COMET after SFT back to the 79-87 COMET range, acting as a "hemostatic valve" for the pipeline.
-   **CPT is most important for low-resource languages**: From +SD to +CPT, low-resource En→X rises from 78.68 to 87.14 and Zh→X from 75.15 to 84.17, showing that base LLM original low-resource knowledge is insufficient; SFT only teaches format and direction, while CPT supplements linguistic capability.
-   **PMP is not the main gain source but provides transfer capability**: Direct improvement from PMP in the main table is small, primarily appearing in X→En/Zh and zero-shot transfer; it acts more as a functional switch enabling the model to read auxiliary anchors.
-   **Document-level translation is still a weakness**: LMT remains competitive in many directions on WMT24++, but lags behind Hunyuan-MT in several subsets; the authors note that sentence-level SFT lacks discourse signals, affecting cross-sentence consistency.

## Highlights & Insights
-   **Defining the "Curse of Multilinguality" as a data-reuse etiology**: The most valuable part of the paper is not just releasing a large model but discovering that the symmetric expansion of multi-way SFT creates many-to-one target repetition. This diagnosis directly informs other multilingual instruction tuning tasks, such as cross-lingual summarization, QA, and speech translation.
-   **SD as a low-cost, high-reward engineering strategy**: When encountering degradation in reverse low-resource directions, many multilingual systems first consider model structures or MoE routing; this paper shows that checking data directional ratios is often more effective. The 5% retention rate also provides a strong default starting point for future work.
-   **PMP transforms multi-way data from "training set structure" to "controllable inference interface"**: Once the model learns to use auxiliary parallel sentences, inference can integrate high-quality MT, retrieved translation memory, or self-translate into an English anchor before translating to the target language. This makes LMT not just a static model but an interface for test-time enhancement.
-   **Pragmatic Chinese-English dual-centricity**: Many open-source MMT models perform well in English directions but have obvious gaps in Chinese-centric directions. LMT specifically supplements Zh↔X directions with Chinese-centric pseudo-parallel expansion and filtering, meeting the real needs of Chinese users and the Asian language community.

## Limitations & Future Work
-   Evaluation still centers on FLORES-200 Devtest and COMET-22. While broad, sentence-level benchmarks cannot fully represent domain transfer, terminology consistency, long-document coherence, and user preferences in real-world scenarios.
-   Chinese-English-centric is a step forward from English-only, but still not a truly multi-centric translation architecture. For regional pivot languages like Arabic, Spanish, French, and Hindi, a tri-centric or more general multi-centric design may be needed.
-   60 languages is Large for open-source LLM-MT, but still very limited compared to global linguistic diversity. Extremely low-resource languages lack not just data but often suffer from scarce writing resources, poor LID/QE model calibration, and difficulties in evaluating synthetic data.
-   The test-time effect of PMP depends on anchor quality; self-generated anchors in the Zh→X direction are not always reliable. Practical deployment may require external MT, retrieved translation memory, or confidence gating.
-   GRPO uses COMET-22 as a reward, which may inherit COMET's biases against low-resource and non-English-centric directions. Future work could consider multi-metric rewards, human preference calibration, or hybrid reference-free + reference-based optimization.

## Related Work & Insights
-   **vs NLLB / M2M-100**: NLLB and M2M-100 represent traditional encoder-decoder large-scale multilingual MT. They have strong coverage but lack the unified instruction-following interface of LLMs. LMT uses decoder-only Qwen3 for Chinese-English dual-centric adaptation and significantly outperforms NLLB-54B's average COMET on 59 overlapping languages using 4B/8B models.
-   **vs ALMA / TowerInstruct / X-ALMA**: These works prove that LLM post-training enables high-quality MT, but language counts and pivot directions are more English-biased. LMT's unique contribution is systematically studying directional degeneration in multi-way SFT and setting 60 languages/234 directions as a release goal.
-   **vs GemmaX2 / Hunyuan-MT / Seed-X**: These are more recent Chinese or multilingual LLM-MT systems. LMT is comparable to Seed-X-PPO and outperforms most GemmaX2 and Hunyuan-MT combinations; more importantly, it discloses reusable recipes like SD/PMP rather than just reporting model performance.
-   **vs Multi-way parallel data and auxiliary translation prompt research**: Previous work mostly proved multi-way/auxiliary translation is helpful on the CPT or inference side. This paper advances this into SFT design: warning that symmetric reuse harms the reverse direction while incorporating auxiliary sentences into trainable prompt behavior via PMP.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Data-level attribution and SD for Directional Degeneration are very insightful; PMP is conceptually simple but innovative in bridging SFT and inference interfaces. The model suite itself is a strong system engineering innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 4 model sizes, 60 languages, 234 directions, FLORES/WMT24++, multiple strong baselines, directional degeneration diagnosis, and PMP analysis. The evidence chain is very robust.
- **Writing Quality**: ⭐⭐⭐⭐ The main narrative is clear, starting with failure modes followed by mitigation and system release; however, with many tables and dense information in the appendix, readers need to navigate between the main text and appendix for details.
- **Value**: ⭐⭐⭐⭐⭐ Highly practical for the open-source multilingual MT community: providing models, data ratios, and prompt training recipes, particularly suitable as a strong baseline for Chinese-centric and low-resource directions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Alexandria: A Multi-Domain Dialectal Arabic Machine Translation Dataset for Culturally Inclusive and Linguistically Diverse LLMs](alexandria_a_multi-domain_dialectal_arabic_machine_translation_dataset_for_cultu.md)
- [\[ACL 2026\] Evaluating the Impact of Verbal Multiword Expressions on Machine Translation](evaluating_the_impact_of_verbal_multiword_expressions_on_machine_translation.md)
- [\[ACL 2026\] CLewR: Curriculum Learning with Restarts for Machine Translation Preference Learning](clewr_curriculum_learning_with_restarts_for_machine_translation_preference_learn.md)
- [\[ACL 2026\] LQM: Linguistically Motivated Multidimensional Quality Metrics for Machine Translation](lqm_linguistically_motivated_multidimensional_quality_metrics_for_machine_transl.md)
- [\[ICML 2026\] ML-Embed: Inclusive and Efficient Embeddings for a Multilingual World](../../ICML2026/multilingual_mt/ml-embed_inclusive_and_efficient_embeddings_for_a_multilingual_world.md)

</div>

<!-- RELATED:END -->
