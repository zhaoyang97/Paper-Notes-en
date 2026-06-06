---
title: >-
  [Paper Note] WeatherSyn: An Instruction Tuning MLLM For Weather Forecasting Report Generation
description: >-
  [ICML 2026][Multimodal VLM][Weather forecasting report] WeatherSyn decomposes the weather forecaster's report-writing workflow into a multimodal instruction task of "Visual Analysis → Point Outlining → Drafting." It esta…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Weather forecasting report"
  - "instruction tuning"
  - "aspect-controlled prompting"
  - "RFT"
  - "DPO"
date: 2026-05-08
content_hash: 121b50bfdb28ac85
---

# WeatherSyn: An Instruction Tuning MLLM For Weather Forecasting Report Generation

**Conference**: ICML 2026  
**arXiv**: [2605.07522](https://arxiv.org/abs/2605.07522)  
**Code**: https://github.com/compasszzn/WeatherSyn (Available)  
**Area**: Multimodal VLM / Weather AI / Report Generation  
**Keywords**: Weather forecasting report, instruction tuning, aspect-controlled prompting, RFT, DPO

## TL;DR
WeatherSyn decomposes the weather forecaster's report-writing workflow into a multimodal instruction task of "Visual Analysis → Point Outlining → Drafting." It establishes the first WSInstruct dataset covering 31 US cities and 8 weather aspects. By employing a three-stage fine-tuning process (SFT → RFT → DPO) on Qwen3-VL-8B, an 8B open-source model consistently outperforms closed-source models such as GPT-5-Nano and Claude-3.7-Sonnet across multiple evaluation metrics and demonstrates zero-shot generalization capabilities for unseen cities.

## Background & Motivation

**Background**: Traditional weather forecast reports rely on meteorologists reading hundreds of variable fields from Numerical Weather Prediction (NWP) models, combined with observational data and collective discussions. This process suffers from information overload, low efficiency, and subjective bias. MLLMs have been explored for weather captioning (e.g., WeatherQA, OmniEarth-Bench, Omni-Weather), but these either focus on severe weather events or use classification/multiple-choice formats, failing to address the daily operational task of "generating multi-day open-ended forecast reports directly from initial atmospheric fields."

**Limitations of Prior Work**: (1) Data Scarcity—paired "visual images + expert-written reports" are nearly non-existent; (2) Lack of Constraints in Open-ended Generation—different reports for the same scenario may emphasize different dimensions (temperature, wind, humidity). Without aspect control, models tend to generate homogenized reports dominated by high-frequency elements and fail fine-grained alignment evaluation with the ground truth; (3) Inadequate Performance—existing specialized weather MLLMs may show decent reference-based metrics, but LLM-judge and expert evaluations are nearly zero, indicating fluent but factually incorrect text.

**Key Challenge**: Open-ended forecast reports must simultaneously satisfy "factual accuracy" (consistency with the initial atmospheric field) and "expressive diversity" (closeness to expert writing style). However, small-scale manually written SFT data only teaches the model to memorize specific phrasing, leading to a dilemma: either "grammatically correct but lacking key points" or "complete points but rigid phrasing."

**Goal**: (1) Formalize weather forecast report generation as a WFR task and construct the first instruction tuning dataset; (2) Design a training scheme that allows an 8B open-source backbone to systematically exceed closed-source models; (3) Verify the robustness of the model across cities, regions, and multiple time steps.

**Key Insight**: The authors captured the structural observation that expert reports are high quality because they first organize items by aspect (temperature, wind, frontal systems, etc.) in the mind before composing coherent text. They injected "aspect" as an explicit prompt constraint during training and inference. Using post-training methods like RFT and DPO, they decoupled "factual correctness" from "expressive diversity"—using RFT to enhance phrasing diversity and DPO to lock in factual accuracy.

**Core Idea**: Structure the open-ended report problem using aspect-controlled prompting, and then feed "accuracy" and "diversity" into the same 8B VLM via a three-stage SFT → RFT → DPO approach.

## Method

### Overall Architecture
The core of WeatherSyn is the WSInstruct dataset and the three-stage training. WSInstruct consists of: (i) Visual end—local heatmaps of 12 single-level variables (e.g., temperature, precipitation, wind) cropped from ERA5 reanalysis data per city; (ii) Text end—expert-written 4-day open-ended forecast reports, segmented into daily sub-reports and annotated with aspects/claims. Training proceeds in three stages: SFT to teach the model to "write reports by aspect based on images"; RFT using rejection sampling to inject "factually correct but linguistically diverse" synthetic reports; and finally DPO using "high F1 reports" as chosen and "low F1 reports" as rejected for preference alignment. During inference, aspect constraints are also attached to ensure alignable evaluation.

### Key Designs

1.  **Aspect-controlled prompting + Structured report segmentation**:
    -   **Function**: Forces open-ended reports into a "Date → List of aspects discussed → Report text" template, making generation controllable and evaluation alignable.
    -   **Mechanism**: First, relative expressions like "Today/Monday" in the original reports are converted to absolute dates using regex and `dateparser`. Then, Qwen-2.5-72B segments them into 4 consecutive daily sub-reports, requiring coverage of at least 3 out of 4 days. Subsequently, the reports are "labeled" based on 8 expert-defined aspects (temperature, wind, humidity, frontal system, pressure system, wave pattern, wind flow system, event) and fine-grained claim categories. The prompt template looks like `<<date, weekday>> Report:\n## Focus on: Temperature, Humidity`. During inference, all baselines receive the same aspect list to ensure fairness.
    -   **Design Motivation**: Without aspect constraints, the model is dominated by the distribution of elements in the training set, with a hit rate of only 0.16. If constraints are added only during training but not inference, the model collapses to the most frequent "event" aspect (hit rate drops to 0.12). Only with both training and inference constraints can the hit rate reach 0.94, which is a prerequisite for fair evaluation.

2.  **RFT: Injecting "Factually Correct but Linguistically Diverse" via Rejection Sampling**:
    -   **Function**: Alleviates overfitting caused by small SFT data scales and monotonous phrasing, allowing the model to learn underlying meteorological phenomena rather than surface-level phrases.
    -   **Mechanism**: For the 2017–2020 subset, the SFT model $\mathcal{M}$ samples 40 candidates at temperature 0.9. Qwen-2.5-72B extracts claims and calculates step-level F1, keeping only F1=1 sub-reports. Four distance metrics (Edit distance, TF-IDF, Jaccard, Sentence-BERT cosine) are then used to select the sub-reports furthest from the original ground truth among the "factually correct" set. These are assembled into a new report as an augmented sample $\mathcal{D}_\text{RFT}$ for fine-tuning on Qwen3-VL-8B. The objective remains the masked next-token loss:
        $$\mathcal{L}_\text{VLM} = -\mathbb{E}_{(Q,I,\{R^i\}_{i=1}^N)\sim\mathcal{D}} [\sum_{i=1}^N \log p_\theta(R^i \mid I, Q)]$$
        In stage 2, $N$ is no longer fixed at 1 but adapts based on candidate vote counts.
    -   **Design Motivation**: The authors' motivation is straightforward—the same initial meteorological field can be reasonably expressed as "temperatures rebound" or "increasing temperatures." Such "fact consistency + lexical diversity" samples significantly strengthen image-text alignment. While pure SFT still shows low F1 on long-tail aspects like Wave Pattern and Wind Flow System, RFT boosts weighted F1 for these by 0.15–0.17.

3.  **DPO: Locking in Factual Correctness with F1 as Preference Signal**:
    -   **Function**: Explicitly optimizes "factual correctness" as a human preference on top of the RFT model.
    -   **Mechanism**: On the 2021 subset, $\mathcal{M}_\text{RFT}$ samples 40 candidates and calculates step-level F1. Sub-reports with F1=1 are assembled as the chosen $\mathcal{Y}_w$, while those with the lowest F1 are assembled as the rejected $\mathcal{Y}_l$, resulting in 1241 preference pairs $\mathcal{D}_\text{DPO}$. The standard DPO objective is optimized:
        $$\mathcal{L}_\text{DPO}(\pi_\theta, \pi_\text{ref}) = -\mathbb{E}[\log \sigma(\beta \log \tfrac{\pi_\theta(\mathcal{Y}_w|x)}{\pi_\text{ref}(\mathcal{Y}_w|x)} - \beta \log \tfrac{\pi_\theta(\mathcal{Y}_l|x)}{\pi_\text{ref}(\mathcal{Y}_l|x)})]$$
        where the reference model $\pi_\text{ref} = \mathcal{M}_\text{RFT}$.
    -   **Design Motivation**: RFT solved "diversity," but the "Top-1 ratio" in LLM-judge and expert evaluation remained low (11%–16%). Using F1—an automatically calculable objective signal directly linked to meteorological claims—for preference alignment is cheaper and more targeted than RLHF with human annotations. Following DPO, the Top-1 ratio in LLM evaluation jumped from 0.16/0.15 to 0.33/0.32, nearly doubling.

### Loss & Training
-   **Stage 1 (SFT)**: $\mathcal{L}_\text{VLM}$, $N=1$, mixed training across all 31 cities.
-   **Stage 2 (RFT)**: Also $\mathcal{L}_\text{VLM}$ but with $N$ dynamically determined by candidate votes; total samples in $\mathcal{D}_\text{RFT}$ is 20,412.
-   **Stage 3 (DPO)**: Standard DPO objective, $\pi_\text{ref}$ fixed as $\mathcal{M}_\text{RFT}$.
-   **Split**: Training on 2017–2021; test set of 1,292 samples from 2022.

## Key Experimental Results

### Main Results
Comprehensive comparison across five evaluation categories on the WSInstruct test set (selected metrics):

| Model | BLEU-1 | ROUGE-L | Auto F1 | LLM Fact.Cons. (Top-1) | Expert Fact.Cons. (Top-1) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| GPT-5.2 Thinking (2-shot) | 0.12 | 0.12 | 0.49 | 0.06 | 0.07 |
| Gemini-3 Pro Preview (2-shot) | 0.37 | 0.28 | 0.60 | 0.24 | 0.29 |
| Claude-3.7-Sonnet (2-shot) | 0.09 | 0.10 | 0.51 | 0.02 | 0.01 |
| WeatherQA (Qwen3-VL-8B) | 0.19 | 0.15 | 0.36 | 0.01 | 0.01 |
| **Ours (SFT)** | 0.43 | 0.31 | 0.55 | 0.11 | 0.11 |
| **Ours-RFT** | 0.43 | 0.31 | 0.59 | 0.16 | 0.17 |
| **Ours-DPO** | **0.44** | **0.32** | **0.59** | **0.33** | **0.29** |

Weighted F1 for 8 aspects (selection): Ours-DPO shows gains exceeding 5pp in complex aspects such as Pressure System (0.72), Frontal System (0.36), and Event (0.67), matching or surpassing Gemini-3 Pro.

### Ablation Study

| Configuration (Aspect Control) | Avg Hit Rate | Description |
| :--- | :--- | :--- |
| Train ✗ / Test ✗ | 0.16 | Model dominated by training distribution, ignores low-frequency aspects |
| Train ✓ / Test ✗ | 0.12 | Degenerates to the most frequent "event" aspect |
| Train ✓ / Test ✓ | **0.94** | Explicit aspect constraints are indispensable for both training and inference |

| Training Stage | LLM Fact.Cons. Top-1 | Avg Aspect F1 |
| :--- | :--- | :--- |
| SFT only | 0.11 | 0.55 |
| + RFT | 0.16 | 0.59 |
| + DPO | 0.33 | 0.59 |

### Key Findings
-   Among the five types of evaluation, "reference-based" metrics (BLEU/ROUGE) and "claim/LLM/expert-based" metrics often diverge. WeatherQA has a BLEU-1 of 0.19 (higher than Claude), but its LLM Fact.Cons. is only 0.01. This indicates that surface similarity metrics like BLEU are unsuitable for open-ended weather reports, and claim F1 should be the primary metric.
-   RFT primarily boosts aspect F1 (factual coverage through diversity), while DPO primarily boosts the Top-1 ratio in LLM/expert evaluations (factual precision preference). The two stages are complementary.
-   F1 decays slowly for forecasts from day 1 to day 4, but RFT significantly mitigates time decay for Frontal System and Event aspects—elements that are inherently difficult for long-term forecasting. RFT encourages the model to grasp underlying dynamics rather than surface phrasing.
-   **Generalization**: When trained on a random half of the cities and tested on the other half, zero-shot WeatherSyn outperforms 2-shot GPT-5-Nano/Claude. It maintains its advantage even when tested across North-South or East-West regions, suggesting the model learns "region-agnostic meteorological laws."
-   The performance gap between WeatherSyn and baselines is larger for cities with complex climates like Honolulu (maritime island) and Flagstaff (high-altitude plateau), while narrower for cities with simple patterns like Charleston, indicating that its advantage is concentrated in "difficult scenarios."

## Highlights & Insights
-   **Aspect-controlled prompting is an undervalued engineering weapon**: Simply adding aspect constraints improved the hit rate from 16% to 94%. This pattern of "explicitly outlining for the generator" can be transferred to any open-ended domain report generation (financial, medical, legal summaries).
-   **Three-stage SFT→RFT→DPO is a standard route for mid-sized VLMs**: The authors used RFT for "diversity" and DPO for "factual precision," decomposing traditional RLHF into two cheaper synthetic data paths without relying on manual preference annotation.
-   **DPO preference based on automatically calculable objective signals**: Constructing chosen/rejected pairs using step-level F1 provides a "verifiable reward → preference pair" workflow that is applicable to any generation task with structured labels. It is an order of magnitude cheaper than RLHF.
-   **Five complementary evaluation systems**: The authors explicitly pointed out the misleading nature of BLEU/ROUGE for open-ended reports, thus reporting parallel metrics: Auto claim, Human-refined claim, LLM judge (aggregated ranking from 4 judges), and Expert judge. This sets a benchmark for domain writing that is hard to measure with a single metric.

## Limitations & Future Work
-   The dataset only covers 31 US cities; tropical, monsoon, and polar climates remain unverified. While the authors plan to expand globally, cross-climate migration may require more than just data increases.
-   Current visual input uses only the atmospheric field at the initial timestamp, ignoring multi-step predictions from NWP. Although Appendix E.1 shows that adding HRES multi-step predictions further improves results, the main method intentionally restricted input to test pure reasoning.
-   Improvement across different aspects via DPO is inconsistent; some aspects (e.g., Frontal System) even declined slightly, suggesting that a single global F1 signal for preference might mask conflicts between aspects. Aspect-conditioned rewards or multi-task DPO could be considered.
-   Evaluation still relies partly on LLM-as-judge (aggregated GPT-5, Gemini, Claude, DeepSeek), risking self-preference bias. Expert evaluation, while correcting LLM bias, only covered 144 samples, leading to statistical noise.
-   Real forecaster scenarios require handling dynamic updates (hourly ERA5, sudden weather events), which the paper does not address regarding online updates or temporal continuous generation.

## Related Work & Insights
-   **vs WeatherQA (Ma et al., 2024)**: WeatherQA focuses on multiple-choice/captioning for severe events, whereas WSInstruct generates multi-day open-ended reports. When both are fine-tuned on Qwen3-VL-8B, WeatherQA's Auto F1 (0.36) is much lower than WeatherSyn-DPO (0.59), with the gap stemming from task structuring (aspects) and post-training (RFT/DPO).
-   **vs Omni-Weather (Zhou et al., 2025)**: Omni-Weather focuses on radar precipitation nowcasting; this work focuses on city-level multi-variable, multi-day reports. They are complementary in visual input resolution and text structure.
-   **vs OmniEarth-Bench (Wang et al., 2025)**: OmniEarth provides a multiple-choice benchmark, while WeatherSyn provides a generative benchmark + model; their positioning is complementary.
-   **Insight**: The pipeline of "domain knowledge structured into aspects → automatic claim extraction via LLM for evaluation → F1-based DPO preference" is applicable to any "image/data → open-ended domain report" task (radiology CT reports, satellite disaster assessment, financial research), and the template can be directly adopted.

## Rating
-   Novelty: ⭐⭐⭐⭐ The task definition and aspect-controlled prompting are new, though the training pipeline (SFT+RFT+DPO) is a common combination recently.
-   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Five complementary evaluations, multi-dimensional analysis (aspects/cities/geography/time steps), and complete ablations cover almost all questionable dimensions.
-   Writing Quality: ⭐⭐⭐⭐ Dataset construction is detailed and the method is well-organized. Formula typesetting is slightly cluttered (LaTeX double-brace noise), but does not hinder understanding.
-   Value: ⭐⭐⭐⭐ The first public dataset + an open-source 8B model that beats GPT-5-Nano/Claude weather report baselines provides direct engineering reference for the industrialization of weather AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MetaMorph: Multimodal Understanding and Generation via Instruction Tuning](../../ICCV2025/multimodal_vlm/metamorph_multimodal_understanding_and_generation_via_instruction_tuning.md)
- [\[ICML 2026\] Decentralized Instruction Tuning: Conflict-Aware Splitting and Weight Merging](decentralized_instruction_tuning_conflict-aware_splitting_and_weight_merging.md)
- [\[NeurIPS 2025\] Visual Instruction Bottleneck Tuning](../../NeurIPS2025/multimodal_vlm/visual_instruction_bottleneck_tuning.md)
- [\[ICML 2026\] SAME: Stabilized Mixture-of-Experts for Multimodal Continual Instruction Tuning](same_stabilized_mixture-of-experts_for_multimodal_continual_instruction_tuning.md)
- [\[ACL 2026\] CogGen: A Cognitively Inspired Recursive Framework for Deep Research Report Generation](../../ACL2026/multimodal_vlm/coggen_a_cognitively_inspired_recursive_framework_for_deep_research_report_gener.md)

</div>

<!-- RELATED:END -->
