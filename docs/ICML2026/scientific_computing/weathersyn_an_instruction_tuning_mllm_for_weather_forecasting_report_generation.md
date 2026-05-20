---
title: >-
  [Paper Note] WeatherSyn: An Instruction Tuning MLLM For Weather Forecasting Report Generation
description: >-
  [ICML 2026][Scientific Computing][Weather forecast report] WeatherSyn decomposes the workflow of meteorological forecasters' report writing into a multimodal instruction task of "image interpretation → key point listing…
tags:
  - "ICML 2026"
  - "Scientific Computing"
  - "Weather forecast report"
  - "instruction tuning"
  - "aspect-controlled prompting"
  - "RFT"
  - "DPO"
date: 2026-05-08
content_hash: 1f08e9e53d89d610
---

# WeatherSyn: An Instruction Tuning MLLM For Weather Forecasting Report Generation

**Conference**: ICML 2026  
**arXiv**: [2605.07522](https://arxiv.org/abs/2605.07522)  
**Code**: https://github.com/compasszzn/WeatherSyn (available)  
**Area**: Multimodal VLM / Meteorological AI / Report Generation  
**Keywords**: Weather forecast report, instruction tuning, aspect-controlled prompting, RFT, DPO

## TL;DR
WeatherSyn decomposes the workflow of meteorological forecasters' report writing into a multimodal instruction task of "image interpretation → key point listing → report drafting." It first constructs the WSInstruct dataset, covering 31 US cities and 8 weather aspects, and then applies a three-stage SFT→RFT→DPO fine-tuning process on Qwen3-VL-8B. This enables an 8B open-source model to consistently outperform closed-source large models such as GPT-5-Nano and Claude-3.7-Sonnet across various evaluation metrics, while also demonstrating zero-shot generalization to unseen cities.

## Background & Motivation

**Background**: Traditional weather forecast reports rely on meteorologists reading hundreds of variable fields from NWP (Numerical Weather Prediction) model outputs, then collectively discussing with observational data before manual drafting. This process suffers from information overload, low efficiency, and subjective bias. MLLMs have begun to be explored for weather captioning (WeatherQA, OmniEarth-Bench, Omni-Weather), but these efforts either focus on severe weather events or take the form of classification/multiple-choice tasks, failing to address the real-world need for "direct multi-day open-ended forecast report generation from initial atmospheric fields."

**Limitations of Prior Work**: (1) Data scarcity—there is virtually no public dataset pairing visual images with expert-written reports; (2) Lack of constraints in open-ended generation—reports for the same weather scenario may emphasize different aspects such as temperature, wind, or humidity. Without aspect control, models tend to generate homogeneous reports dominated by high-frequency elements, making fine-grained alignment with ground truth impossible; (3) Existing specialized meteorological MLLMs achieve seemingly high reference-based metrics in free-text generation, but LLM-judge and expert evaluations are nearly zero, indicating fluent but factually incorrect statements.

**Key Challenge**: Open-ended forecast reports must be both "factually accurate" (consistent with the initial atmospheric field) and "expressively diverse" (reflecting expert writing styles). However, small-scale hand-crafted SFT data only teach the model to memorize specific phrasings, leading to a dilemma of "grammatically correct but incomplete key points" or "complete key points but rigid wording."

**Goal**: (1) Formalize "weather forecast report generation" as the WFR task and construct the first instruction tuning dataset; (2) Design a training scheme that systematically surpasses closed-source large models on an 8B open-source backbone; (3) Validate the model's robustness across cities, regions, and multiple time steps.

**Key Insight**: The authors observe that the high quality of expert reports stems from first organizing key points by aspect (temperature/wind/fronts, etc.) before composing coherent text. They inject aspect as an explicit prompt constraint during training and inference. Drawing on post-training methods like RFT/DPO, they decouple "factual correctness" and "expressive diversity"—RFT enhances lexical diversity, while DPO locks in factual accuracy.

**Core Idea**: Use aspect-controlled prompting to structure the open-ended report task, then inject "accuracy" and "diversity" in stages into a single 8B VLM via SFT→RFT→DPO.

## Method

### Overall Architecture
The core of WeatherSyn is the WSInstruct dataset and a three-stage training process. WSInstruct consists of: (i) Visual component—local heatmaps of 12 single-layer variables (e.g., temperature, precipitation, wind) cropped from ERA5 reanalysis data by city; (ii) Text component—expert-written 4-day open-ended forecast reports, segmented into daily sub-reports and annotated with aspect/claim. Training proceeds in three stages: SFT teaches the model to "write reports by aspect from images"; RFT uses rejection sampling to inject "factually correct but lexically diverse" synthetic reports; DPO aligns preferences by treating "high F1 reports" as chosen and "low F1 reports" as rejected. During inference, aspect constraints are also appended to ensure alignable evaluation.

### Key Designs

1. **Aspect-controlled prompting + Structured Report Segmentation**:

    - **Function**: Forces open-ended reports into a "date → list of aspects discussed that day → report text" template, enabling controllable generation and alignable evaluation.
    - **Mechanism**: Regex and dateparser convert relative expressions like "Today/weekday" in original reports to absolute dates. Qwen-2.5-72B segments reports along the timeline into four consecutive daily sub-reports, requiring coverage of at least 3 out of the first 4 days. The model then labels reports according to 8 expert-defined aspects (temperature, wind, humidity, frontal system, pressure system, wave pattern, wind flow system, event) and fine-grained claim categories under each aspect. The final prompt template is `<<date, weekday>> Report:\n## Focus on: Temperature, Humidity`. All baselines receive the aspect list at inference for fairness.
    - **Design Motivation**: Without aspect constraints, the model is dominated by the training set's element distribution, with a hit rate of only 0.16 (i.e., generated aspects match ground truth only 16% of the time). Adding constraints only during training but not inference causes collapse to the most frequent "event" aspect (hit rate drops to 0.12). Only applying aspect constraints in both training and inference pushes the hit rate to 0.94, a prerequisite for fair subsequent evaluation.

2. **RFT: Injecting "Factually Correct but Lexically Diverse" Samples via Rejection Sampling**:

    - **Function**: Mitigates overfitting caused by small, lexically homogeneous SFT data, teaching the model underlying meteorological phenomena rather than surface phrases.
    - **Mechanism**: For the 2017–2020 subset, the SFT-trained model $\mathcal{M}$ samples 40 candidates at temperature 0.9. Qwen-2.5-72B extracts claims and computes step-level F1 for each daily sub-report, retaining only those with F1=1. Four distance metrics (edit distance, TF-IDF, Jaccard, Sentence-BERT cosine) select the most lexically distant "factually correct" sub-reports from the ground truth, which are assembled into new full reports as augmented samples $\mathcal{D}_\text{RFT}$ for further fine-tuning on Qwen3-VL-8B. The training objective remains masked next-token loss: $\mathcal{L}_\text{VLM} = -\mathbb{E}_{(Q,I,\{R^i\}_{i=1}^N)\sim\mathcal{D}} [\sum_{i=1}^N \log p_\theta(R^i \mid I, Q)]$, with $N$ in stage 2 adaptively set by candidate votes.
    - **Design Motivation**: The motivation is straightforward—one initial atmospheric field can be reasonably described as "temperatures rebound" or "increasing temperatures," and such "factually consistent + lexically diverse" samples best strengthen image-text alignment. Pure SFT yields low F1 on long-tail aspects like Wave Pattern and Wind Flow System; RFT directly boosts weighted F1 for these aspects by 0.15–0.17.

3. **DPO: Using F1 as Preference Signal to Lock in Factual Correctness**:

    - **Function**: Further optimizes "factual correctness" as an explicit human preference on top of the RFT model.
    - **Mechanism**: On the 2021 subset, $\mathcal{M}_\text{RFT}$ samples 40 candidates and computes step-level F1. Sub-reports with F1=1 are assembled as chosen $\mathcal{Y}_w$, those with lowest F1 as rejected $\mathcal{Y}_l$, yielding 1241 preference pairs $\mathcal{D}_\text{DPO}$. The standard DPO objective is optimized: $\mathcal{L}_\text{DPO}(\pi_\theta, \pi_\text{ref}) = -\mathbb{E}[\log \phi(\beta \log \tfrac{\pi_\theta(\mathcal{Y}_w|x)}{\pi_\text{ref}(\mathcal{Y}_w|x)} - \beta \log \tfrac{\pi_\theta(\mathcal{Y}_l|x)}{\pi_\text{ref}(\mathcal{Y}_l|x)})]$, with reference model $\pi_\text{ref} = \mathcal{M}_\text{RFT}$.
    - **Design Motivation**: RFT addresses "diversity," but the "Top-1 ratio" in LLM-judge and expert evaluation remains low (11%–16%). Using F1, an automatically computable and claim-linked objective signal, for preference alignment is cheaper and more targeted than RLHF with human annotations. After DPO, the Top-1 ratio in LLM evaluation rises from 0.16/0.15 to 0.33/0.32, nearly doubling.

### Loss & Training

- Stage 1 (SFT): $\mathcal{L}_\text{VLM}$, $N=1$, mixed training on all 31 cities.
- Stage 2 (RFT): Same $\mathcal{L}_\text{VLM}$ but $N$ dynamically set by candidate votes, total $\mathcal{I}$ in $\mathcal{D}_\text{RFT}$ is 20412.
- Stage 3 (DPO): Standard DPO objective, reference model fixed as $\mathcal{M}_\text{RFT}$.
- Evaluation dataset: 2017–2021 for training, 2022 test set with 1292 samples.

## Key Experimental Results

### Main Results
Comprehensive comparison on the WSInstruct test set across five evaluation types (key metrics excerpted):

| Model | BLEU-1 | ROUGE-L | Auto F1 | LLM Fact.Cons. (Top-1) | Expert Fact.Cons. (Top-1) |
|---|---|---|---|---|---|
| GPT-5.2 Thinking (2-shot) | 0.12 | 0.12 | 0.49 | 0.06 | 0.07 |
| Gemini-3 Pro Preview (2-shot) | 0.37 | 0.28 | 0.60 | 0.24 | 0.29 |
| Claude-3.7-Sonnet (2-shot) | 0.09 | 0.10 | 0.51 | 0.02 | 0.01 |
| WeatherQA (Qwen3-VL-8B) | 0.19 | 0.15 | 0.36 | 0.01 | 0.01 |
| **WeatherSyn (SFT)** | 0.43 | 0.31 | 0.55 | 0.11 | 0.11 |
| **WeatherSyn-RFT** | 0.43 | 0.31 | 0.59 | 0.16 | 0.17 |
| **WeatherSyn-DPO** | **0.44** | **0.32** | **0.59** | **0.33** | **0.29** |

Weighted F1 for 8 aspects (excerpt): WeatherSyn-DPO achieves improvements exceeding 5pp on structurally complex aspects such as Pressure System (0.72), Frontal System (0.36), and Event (0.67), matching or surpassing Gemini-3 Pro.

### Ablation Study

| Configuration (Aspect Control) | Avg hit rate | Notes |
|---|---|---|
| Train ✗ / Test ✗ | 0.16 | Model dominated by training distribution, rarely writes low-frequency aspects |
| Train ✓ / Test ✗ | 0.12 | Collapses to most frequent "event" aspect in training |
| Train ✓ / Test ✓ | **0.94** | Explicit aspect constraint is indispensable in both training and inference |

| Training Stage | LLM Fact.Cons. Top-1 | Avg Aspect F1 |
|---|---|---|
| SFT only | 0.11 | 0.55 |
| + RFT | 0.16 | 0.59 |
| + DPO | 0.33 | 0.59 |

### Key Findings
- Among five evaluation types, "reference-based" metrics (BLEU/ROUGE) and "claim/LLM/expert-based" metrics often diverge: WeatherQA scores 0.19 on BLEU-1, higher than Claude, but LLM Fact.Cons. is only 0.01, indicating that BLEU and similar surface similarity metrics are unsuitable for open-ended meteorological reports; claim F1 should be the primary metric.
- RFT mainly boosts aspect F1 (diversity-driven factual coverage), while DPO mainly increases the Top-1 ratio in LLM/expert evaluations (factual accuracy preference); the two stages are complementary.
- Model F1 decays slowly from forecast day 1 to 4, but RFT significantly mitigates temporal decay for Frontal System and Event aspects—these are inherently challenging for long-term forecasts, and RFT helps the model capture underlying dynamics rather than surface phrasing.
- Generalization experiments: Training on half the cities and testing on the other half, zero-shot WeatherSyn outperforms GPT-5-Nano/Claude 2-shot; even when testing across north/south or east/west regions, it maintains an advantage, indicating the model learns "region-agnostic meteorological patterns."
- In climatically complex cities like Honolulu (oceanic island) and Flagstaff (high-altitude plateau), WeatherSyn's advantage over baselines is greater; in simpler cities like Charleston, the gap is smaller, indicating the advantage is concentrated in "difficult scenarios."

## Highlights & Insights
- **Aspect-controlled prompting is an underrated engineering tool**: Simply adding aspect constraints raises the hit rate from 16% to 94%. This "explicitly outline for the generator" pattern can be transferred to any open-ended domain report generation (finance, healthcare, legal summaries).
- **Three-stage SFT→RFT→DPO is the standard route for mid-scale VLMs**: The authors use RFT for "diversity" and DPO for "factual accuracy," splitting traditional RLHF into two cheap synthetic data paths without relying on any human preference annotation.
- **DPO with automatically computable objective signals**: Using step-level F1 to construct chosen/rejected pairs, this "verifiable reward → preference pair" approach is applicable to any generative task with structured labels, and is an order of magnitude cheaper than RLHF.
- **Five complementary evaluation systems**: The authors point out that BLEU/ROUGE are misleading for open-ended meteorological reports, so they report Auto claim, Human-refined claim, LLM judge (aggregation of 4 judges), and Expert judge metrics in parallel, providing a template for domains where single-metric evaluation is inadequate.

## Limitations & Future Work
- The dataset covers only 31 US cities, with no validation for tropical, monsoon, or polar climates; the authors plan to expand globally, but cross-climate transfer may not be solvable by simply increasing data.
- Currently, only the initial atmospheric field is used as visual input, without leveraging NWP's multi-step forecasts—Appendix E.1 shows that adding HRES multi-step forecasts can further improve performance, but the main method deliberately restricts this to test pure reasoning ability; engineering deployment should relax this constraint.
- DPO's improvement varies across aspects, with some (e.g., Frontal System) even declining post-DPO, indicating that using a single global F1 signal for preference may mask inter-aspect conflicts; aspect-conditioned rewards or multi-task DPO could be considered.
- Evaluation still partially relies on LLM-as-judge (aggregation of GPT-5, Gemini, Claude, DeepSeek), which risks self-preference; expert evaluation, though correcting LLM judge, covers only 144 samples, resulting in significant statistical noise.
- Real-world forecaster scenarios require handling dynamic updates (hourly new ERA5, sudden weather events), but the paper does not address online updates or sequential generation.

## Related Work & Insights
- **vs WeatherQA (Ma et al., 2024)**: WeatherQA focuses on multiple-choice/captioning for severe weather events, while WSInstruct directly generates multi-day open-ended reports. When both are fine-tuned on Qwen3-VL-8B, WeatherQA's Auto F1=0.36 is much lower than WeatherSyn-DPO's 0.59, with the gap attributed to task structuring (aspect) and post-training (RFT/DPO).
- **vs Omni-Weather (Zhou et al., 2025)**: Omni-Weather focuses on radar precipitation nowcasting, while this work targets city-level, multi-variable, multi-day forecast reports; the two are fully complementary in visual input resolution and text structure.
- **vs OmniEarth-Bench (Wang et al., 2025)**: OmniEarth provides a multiple-choice benchmark, while WeatherSyn offers a generative benchmark and model, making their positioning complementary.
- **Insights**: This pipeline—"structuring domain knowledge into aspects → using LLM to automatically extract claims for evaluation → using F1 for DPO preference"—is applicable to any "image/data → open-ended domain report" task (radiology CT reports, satellite disaster assessment, financial research reports), and can be directly templated.

## Rating
- Novelty: ⭐⭐⭐⭐ Task definition and aspect-controlled prompting are novel, but the training pipeline (SFT+RFT+DPO) is a common recent combination.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Five complementary evaluation systems, multi-dimensional analysis by aspect/city/region/time step, and complete ablation, covering nearly all questionable dimensions.
- Writing Quality: ⭐⭐⭐⭐ Dataset construction is detailed, and the method section is well-organized; formula formatting is slightly messy (LaTeX double braces noise), but does not hinder understanding.
- Value: ⭐⭐⭐⭐ The first public dataset + open-source 8B model surpassing GPT-5-Nano/Claude weather report generation baselines, providing direct engineering reference for meteorological AI deployment.

## Related Papers

- [\[ACL 2025\] MEIT: Multimodal Electrocardiogram Instruction Tuning on Large Language Models for Report Generation](../../ACL2025/multimodal_vlm/meit_multimodal_electrocardiogram_instruction_tuning_on_large_language_models_fo.md)
- [\[ICCV 2025\] MetaMorph: Multimodal Understanding and Generation via Instruction Tuning](../../ICCV2025/multimodal_vlm/metamorph_multimodal_understanding_and_generation_via_instruction_tuning.md)
- [\[ICML 2026\] FreeRet: 让 MLLM 不经训练就能当多模态检索器](freeret_mllms_as_training-free_retrievers.md)
- [\[NeurIPS 2025\] Visual Instruction Bottleneck Tuning](../../NeurIPS2025/multimodal_vlm/visual_instruction_bottleneck_tuning.md)
- [\[ACL 2026\] CogGen: A Cognitively Inspired Recursive Framework for Deep Research Report Generation](../../ACL2026/multimodal_vlm/coggen_a_cognitively_inspired_recursive_framework_for_deep_research_report_gener.md)

</div>

<!-- RELATED:END -->
