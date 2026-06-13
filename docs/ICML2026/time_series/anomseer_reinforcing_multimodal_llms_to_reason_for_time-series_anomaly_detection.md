---
title: >-
  [Paper Note] AnomSeer: Reinforcing Multimodal LLMs to Reason for Time-Series Anomaly Detection
description: >-
  [ICML2026][Time Series][Time-Series Anomaly Detection (TSAD)] AnomSeer translates statistical evidence from classic time-series anomaly detection into expert reasoning trajectories and reinforces MLLMs using TimerPO. Thi…
tags:
  - "ICML2026"
  - "Time Series"
  - "Time-Series Anomaly Detection (TSAD)"
  - "Multimodal Large Language Models (MLLM)"
  - "Reinforcement Learning"
  - "Expert Chain-of-Thought"
  - "Optimal Transport"
date: 2026-05-08
content_hash: 9139dea4432f9bb9
---

# AnomSeer: Reinforcing Multimodal LLMs to Reason for Time-Series Anomaly Detection

**Conference**: ICML2026  
**arXiv**: [2602.08868](https://arxiv.org/abs/2602.08868)  
**Code**: https://github.com/jrzhang33/AnomSeer  
**Area**: Time-Series / Multimodal LLM / Anomaly Detection  
**Keywords**: Time-Series Anomaly Detection (TSAD), Multimodal Large Language Models (MLLM), Reinforcement Learning, Expert Chain-of-Thought, Optimal Transport  

## TL;DR
AnomSeer translates statistical evidence from classic time-series anomaly detection into expert reasoning trajectories and reinforces MLLMs using TimerPO. This enables the model to simultaneously perform anomaly type judgment, interval localization, and fine-grained explanation based on line chart inputs.

## Background & Motivation
**Background**: Time-series anomaly detection previously relied on statistical methods, traditional machine learning, or reconstruction/prediction-based deep models, typically aiming to provide anomaly scores or intervals. With the development of LLMs and MLLMs, recent work has begun rendering time-series as line charts, allowing multimodal models to judge anomalies from images and text prompts. This leverages the visual model's perception of shapes, trends, and local fluctuations while avoiding jamming long sequences directly into the text context.

**Limitations of Prior Work**: General MLLMs lack built-in time-series priors. While they can identify obvious spikes or large offsets, their reasoning often remains at a coarse-grained level (e.g., "sudden change," "overall fluctuation") and is insensitive to details like frequency drift, slight trend changes, or local shapelet anomalies. Standard SFT only mimics correct answers, and standard RL relies on globally verifiable rewards like classification and localization, failing to force the model to learn a verifiable time-series analysis process.

**Key Challenge**: Anomaly detection requires the model to possess dual capabilities: maintaining the holistic visual understanding and linguistic explanation of MLLMs, while utilizing fine-grained evidence like statistics, frequency domain features, and local similarity used in traditional TSAD methods. Incorporating these expert signals directly into the primary reward may cause gradient interference with detection objectives; omitting them entirely results in learning only crude visual heuristics.

**Goal**: The authors aim to develop a post-training method for MLLMs tailored to TSAD. Given a time-series image, the model should output the anomaly type, interval, and a natural language explanation rooted in specific timestamps, frequencies, trends, magnitudes, or local patterns, rather than vague visual judgments.

**Key Insight**: A critical observation is that while traditional TSAD tools are poor at generating natural language, they provide reliable and verifiable fine-grained evidence. Instead of using them as external plugins during inference, this paper translates these analysis workflows into expert chains-of-thought (ExpCoT) during training, serving as auxiliary reasoning signals in reinforcement learning.

**Core Idea**: Generate ExpCoT via classic TSAD analysis, then inject "fine-grained time-series evidence" into MLLM RL updates through TimerPO, which utilizes Optimal Transport alignment and orthogonal projection.

## Method
AnomSeer is a post-training framework that does not modify the Qwen2.5-VL architecture nor call external detectors during inference. Training data is derived from the synthetic AnomLLM dataset. Inputs are line charts and task prompts; outputs are structured text containing reasoning, anomaly types, and interval localization. During training, ExpCoT is generated for each sample using traditional analysis. TimerPO then aligns the model's output with these expert trajectories while maintaining anomaly classification and localization as primary objectives.

### Overall Architecture
The workflow consists of four steps. First, raw univariate time-series are rendered as line charts for MLLM visual input, with prompts requesting anomaly type, localization, and reasoning. Second, ExpCoT is generated using ground-truth labels and traditional TSAD analysis (training phase only). Third, the model samples a set of candidate responses for the same input, calculating outcome rewards and reasoning alignment rewards. Fourth, TimerPO orthogonalizes the reasoning reward against the primary task reward and adds it to the final advantage to drive policy updates.

During inference, the process is streamlined: the model receives only the image and prompt, directly outputting answers in the style of Observation, Reasoning & Validation, and Conclusion. Tools like ExpCoT, FFT, and Matrix Profile are not called online, avoiding additional latency or token costs.

### Key Designs
1. **ExpCoT (Expert Chain-of-Thought)**:
    - **Function**: Organizes evidence from traditional time-series analysis into learnable natural language trajectories, providing finer supervision than final labels.
    - **Mechanism**: ExpCoT follows a three-stage structure: Observation → Reasoning & Validation → Conclusion. Observation performs a global scan for extremes using histogram anomaly scores. If no obvious global anomalies exist, it performs a structural scan (e.g., smooth gradients for trend stability, FFT for period/frequency changes). If still stable, Matrix Profile searches for local dissimilar sub-sequences. Reasoning & Validation selects evidence based on the true anomaly type (e.g., trend drift via gradients, frequency anomalies via peaks). Conclusion synthesizes the final judgment.
    - **Design Motivation**: General CoT is often fluent but unreliable. ExpCoT anchors "why it is an anomaly" to statistics and temporal positions, ensuring the model learns transferable analysis paths rather than templated explanations.

2. **TimerPO Time-Series Reasoning Advantage**:
    - **Function**: Measures the semantic proximity between the model's response and the ExpCoT reasoning trajectory, beyond GRPO's outcome rewards.
    - **Mechanism**: For a group of generated responses, the main reward $\widehat{A}_{main}$ is calculated from format, classification, and localization (group-normalized). TimerPO takes the final-layer token embeddings of the model response and ExpCoT, constructs a cosine distance cost matrix, and approximates the Optimal Transport distance $W^i$ via Sinkhorn iterations. This is converted to a reasoning reward $\exp(-W^i/\tau)$ and group-normalized as $\widehat{A}_{TsR}$.
    - **Design Motivation**: Token-level exact matching punishes valid but differently phrased explanations. OT acts as a comparison of the overall distribution of reasoning evidence, allowing for synonymous or reordered expressions while encouraging the coverage of key evidence.

3. **Orthogonal Projection for Non-Interference Fusion**:
    - **Function**: Ensures the reasoning reward supplements the main detection goal without distorting classification or localization.
    - **Mechanism**: TimerPO projects $\widehat{A}_{TsR}$ onto the primary advantage $\widehat{A}_{main}$, removing the overlapping component to obtain $\widehat{A}_{TsR}^{\perp}=\widehat{A}_{TsR}-\frac{\langle\widehat{A}_{TsR},\widehat{A}_{main}\rangle}{\|\widehat{A}_{main}\|_2^2+\varepsilon}\widehat{A}_{main}$. The final advantage is $A_{final}=\widehat{A}_{main}+\alpha\widehat{A}_{TsR}^{\perp}$, integrated into a PPO/GRPO-style clipped objective.
    - **Design Motivation**: "Correct answer" and "expert-like explanation" are not independent. Direct addition might reinforce spurious correlations. Orthogonalization constrains the auxiliary signal to provide only supplementary fine-grained reasoning capacity.

### Loss & Training
Training utilizes pure RL post-training without SFT cold starts or architecture changes. Qwen2.5-VL-3B/7B-Instruct serves as the backbone. The training set includes 3,200 synthetic samples from AnomLLM. GRPO group size is 5, and PPO clipping is 0.2. Reward weights are: format 0.1, classification 0.2, and localization 0.7 (emphasizing interval quality). The TimerPO reasoning advantage weight defaults to $\alpha=0.3$. This setup ensures the model prioritizes format and task correctness before gradually refining explanations toward expert evidence.

## Key Experimental Results

### Main Results
The paper compares commercial MLLMs, open-source MLLMs, SFT versions, TimeMaster, and AnomSeer on the AnomLLM test set. Metrics include classification accuracy and Affinity-F1 across frequency, trend, range, and point scenarios.

| Method | Training | Classification Acc(%) | Frequency F1 | Trend F1 | Range F1 | Point F1 | Avg F1 |
|------|----------|---------------|--------------|----------|----------|----------|--------|
| GPT-4o | Prompting | 17.2 | 10.9 | 43.5 | 57.0 | 53.4 | 41.2 |
| Gemini-2.5-Pro | Prompting | 12.6 | 19.1 | 59.0 | 81.3 | 74.5 | 58.5 |
| Qwen2.5-VL-72B-Instruct | Prompting | 14.6 | 31.4 | 32.1 | 74.6 | 62.7 | 50.2 |
| TimeMaster-3B | SFT + GRPO | 57.9 | 51.4 | 76.6 | 80.1 | 79.6 | 71.9 |
| AnomSeer-3B | TimerPO | 62.8 | 58.9 | 84.9 | 85.6 | 87.8 | 79.3 |
| AnomSeer-7B | TimerPO | 65.0 | 60.8 | 87.7 | 94.3 | 94.9 | 84.4 |

The key takeaway is that AnomSeer-3B significantly outperforms larger commercial models and the 72B prompt baseline. Notably, the Frequency F1 increases from 51.4 (TimeMaster-3B) to 58.9, demonstrating TimerPO's effectiveness for subtle frequency-domain anomalies compared to standard GRPO.

### Ablation Study
Table 2 explores removing ExpCoT, reasoning advantage, and orthogonalization for AnomSeer-3B.

| Config | Frequency F1 | Trend F1 | Range F1 | Point F1 | Note |
|------|--------------|----------|----------|----------|------|
| w/o ExpCoT (using GenCoT) | 49.8 | 79.5 | 84.4 | 86.1 | Linguistic reasoning remains; expert evidence weakens |
| w/o Orthogonalization | 53.5 | 81.1 | 83.5 | 85.4 | Direct reward fusion causes target interference |
| Vanilla GRPO | 50.4 | 77.8 | 81.8 | 80.6 | Outcome rewards only; weakest fine-grained reasoning |
| Full AnomSeer | 58.9 | 84.9 | 85.6 | 87.8 | Combined ExpCoT, OT reasoning, and orthogonal fusion |

Ablations show all modules are essential. ExpCoT is critical for frequency anomalies, which are hard to judge visually. Orthogonalization provides consistent gains for trend/range/point, confirming that auxiliary signals should be constrained.

### Key Findings
- On AnomLLM, AnomSeer-7B achieves 84.4 Avg F1, surpassing TimeMaster-3B by 12.5 points and Gemini-2.5-Pro by 25.9 points.
- 32k SFT data did not match the performance of AnomSeer trained with only 3.2k RL samples, emphasizing that "imitating more correct answers" does not equate to "learning fine-grained analysis."
- Hyperparameter analysis shows $\alpha$ is stable between 0.3 and 0.7.
- Post-TimerPO, high-frequency words in outputs shift from coarse terms (global, sudden, change) to specific evidence (timestamp, intervals, amplitude).
- Generalization experiments on VisualTimeAnomaly and TSB-UAD show advantages even for shapelet anomalies not seen during training, suggesting the model learns the analysis *process*.

## Highlights & Insights
- The paper avoids creating an opposition between traditional TSAD and MLLMs. Instead, traditional methods generate verifiable training trajectories, migrating "analysis discipline" to the model.
- TimerPO's OT alignment is superior to plain text similarity for reasoning supervision, as it allows for varied phrasing while ensuring key evidence is present.
- Orthogonal projection is a practical multi-objective RL trick. Many tasks require both correct results and credible processes; projecting process signals into the complement space of the primary task prevents contamination.
- The framework unifies TSAD tasks into classification, localization, and explanation, moving beyond simple anomaly scores toward real-world utility.

## Limitations & Future Work
- The current method focuses on univariate time-series. Multivariate scenarios require explaining inter-variable interactions, which existing ExpCoT logic does not yet cover.
- ExpCoT relies on ground-truth anomaly metadata to organize trajectories, requiring high-quality training sets. In noisy real-world scenarios, these trajectories may become unstable.
- While inference does not call external tools, the model cannot dynamically access outside knowledge (e.g., maintenance logs).
- Future work could render variables as subplots to learn cross-variable causality or incorporate domain knowledge bases.

## Related Work & Insights
- **vs SigLLM**: SigLLM uses raw values for zero-shot detection. AnomSeer uses line charts and RL, offering better token economy and unified explanation/localization, though it depends on rendering quality.
- **vs TimeMaster**: TimeMaster uses SFT+GRPO for classification but stays at the outcome level. AnomSeer's ExpCoT and TimerPO specifically target fine-grained evidence, leading to gains in frequency and trend scenarios.
- **vs Traditional TSAD**: HBOS, FFT, and Matrix Profile output statistics but lack natural language explanations. AnomSeer uses them as the source of inductive bias during training.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Integration of ExpCoT, OT alignment, and orthogonal advantage is a distinct and well-defined optimization design.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Solid coverage across several datasets with good ablation; multivariate real-world data would be a plus.
- Writing Quality: ⭐⭐⭐⭐☆ Clear methodology; Figure 2 is effective.
- Value: ⭐⭐⭐⭐⭐ Highly insightful for "detection + localization + explanation" tasks in TSAD.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] IMPACT: Influence Modeling for Open-Set Time Series Anomaly Detection](impact_influence_modeling_for_open-set_time_series_anomaly_detection.md)
- [\[ICLR 2026\] SciTS: Scientific Time Series Understanding and Generation with LLMs](../../ICLR2026/time_series/scits_scientific_time_series_understanding_and_generation_with_llms.md)
- [\[ACL 2026\] Time-RA: Towards Time Series Reasoning for Anomaly Diagnosis with LLM Feedback](../../ACL2026/time_series/time-ra_towards_time_series_reasoning_for_anomaly_diagnosis_with_llm_feedback.md)
- [\[ACL 2026\] STReasoner: Empowering LLMs for Spatio-Temporal Reasoning in Time Series via Spatial-Aware Reinforcement Learning](../../ACL2026/time_series/streasoner_empowering_llms_for_spatio-temporal_reasoning_in_time_series_via_spat.md)
- [\[AAAI 2026\] GAICo: A Deployed and Extensible Framework for Evaluating Diverse and Multimodal Generative AI Outputs](../../AAAI2026/time_series/gaico_a_deployed_and_extensible_framework_for_evaluating_diverse_and_multimodal_.md)

</div>

<!-- RELATED:END -->
