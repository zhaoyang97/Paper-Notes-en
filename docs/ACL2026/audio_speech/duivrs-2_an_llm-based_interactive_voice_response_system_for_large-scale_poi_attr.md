---
title: >-
  [Paper Note] DuIVRS-2: An LLM-based Interactive Voice Response System for Large-scale POI Attribute Acquisition
description: >-
  [ACL 2026][Audio & Speech][IVR] DuIVRS-2 transforms the modular telephone IVR system for large-scale POI attribute acquisition at Baidu Maps into an LLM-driven end-to-end dialogue system. Through FSM data augmentation…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "IVR"
  - "POI Attribute Acquisition"
  - "Task-oriented Dialogue"
  - "Selective Generation"
  - "Collaborative Iterative Learning"
date: 2026-05-08
content_hash: b315d3ee6de6dc95
---

# DuIVRS-2: An LLM-based Interactive Voice Response System for Large-scale POI Attribute Acquisition

**Conference**: ACL 2026  
**arXiv**: [2605.17900](https://arxiv.org/abs/2605.17900)  
**Code**: No public code (project code link not provided in cache)  
**Area**: Speech Dialogue Systems / Industrial LLM Agent  
**Keywords**: IVR, POI Attribute Acquisition, Task-oriented Dialogue, Selective Generation, Collaborative Iterative Learning

## TL;DR
DuIVRS-2 transforms the modular telephone IVR system for large-scale POI attribute acquisition at Baidu Maps into an LLM-driven end-to-end dialogue system. Through FSM data augmentation, selective generation, and dual-evaluator iterative learning, it achieves 83.9% TSR, 130ms average response latency, and a capacity of 0.4M calls per day in production.

## Background & Motivation
**Background**: Map services require continuous updates of POI attributes such as names, addresses, business status, and operating hours. While web data, street view, and user contributions provide some information, their coverage, timeliness, and extraction costs are limited. DuIVRS-1, a modular pipeline employing NLU, dialogue management (DM), and NLG, has been long deployed at Baidu Maps to proactively call POI operators for attribute acquisition.

**Limitations of Prior Work**: Modular IVR systems suffer from cascading error propagation, where NLU misjudgments affect DST/DM, and NLG templates require constant maintenance. Directly integrating general LLMs into industrial IVR is impractical due to high costs, slow inference, and insufficient control over stability and hallucinations, failing to meet the sub-200ms response requirement typical of telephony scenarios.

**Key Challenge**: Industrial telephony dialogues require the semantic understanding capabilities of LLMs while maintaining the controllability, low latency, and safety boundaries of rule-based systems. Open-ended generation is too flexible and prone to irrelevant inquiries, whereas fixed templates struggle with long-tail user responses.

**Goal**: The authors aim to upgrade DuIVRS from a traditional modular system to an LLM-based end-to-end agent for large-scale POI attribute acquisition without sacrificing production availability, demonstrating its deployability in terms of offline CR, online TSR, cost, latency, and throughput.

**Key Insight**: Instead of allowing the LLM to generate the next response freely, the paper combines historical FSM-based response options with LLM understanding. An LLM-S selects the most appropriate next question from candidate actions, while an LLM-L and a black-box LLM jointly evaluate samples to create a low-cost iterative data flywheel.

**Core Idea**: Use FSMs to constrain the LLM's action space, employ CoT to enhance the stability of candidate selection, and utilize a collaborative iterative learning loop involving LLM-L, a black-box LLM, and human arbitration to adapt the small model to real-world long-tail telephony dialogues.

## Method
DuIVRS-2 is an LLM IVR framework designed for production deployment. The system retains mature speech infrastructure (ASR/TTS) but replaces the dialogue management component with a controllable LLM selector, addressing long-tail noise and stability through data augmentation and iterative learning.

### Overall Architecture
The workflow consists of three layers. The first is the Data Layer: state transitions are extracted from DuIVRS-1 logs to construct an FSM, and more balanced training data is generated via uniform path and transition sampling. The second is the Model Layer: LLM-S receives dialogue history and Reply Options provided by the FSM, performs brief reasoning, and selects the next system inquiry. The third is the Iterative Optimization Layer: LLM-L and a black-box LLM jointly evaluate LLM-S outputs. High-confidence samples are automatically added to the training set, while samples with discrepancies or potential for improvement are sent for human correction to update LLM-S, LLM-L, and the black-box evaluator prompts.

### Key Designs
1. **FSM-guided Data Augmentation**:
    - **Function**: Alleviates the long-tail distribution of user responses and dialogue turns in production logs.
    - **Mechanism**: The authors map fixed response templates from DuIVRS-1 to FSM states and user responses to transitions. Sampling is performed uniformly by dialogue path length rather than raw log frequency, with historical user response variants sampled uniformly between state transitions.
    - **Design Motivation**: In real logs, high-frequency simple dialogues are overrepresented. Direct SFT would cause the model to ignore rare but critical edge cases. FSMs preserve business logic while systematically expanding coverage of unpopular states.

2. **Selective Generation + CoT Dialogue Management**:
    - **Function**: Enables robust dialogue policy selection under low latency while avoiding open-ended hallucinations.
    - **Mechanism**: The input prompt contains truncated dialogue history and candidate Reply Options allowed by the current FSM; the output is not arbitrary text but a reasoning step followed by a candidate selection. This allows the system to explain its choice and ensures outputs stay within the valid business action space.
    - **Design Motivation**: Telephony IVR is sensitive to safety and response speed. Selective generation transforms the difficult task of "writing a new sentence" into "understanding intent and choosing a legal action," significantly reducing hallucinations and maintenance costs.

3. **Dual-Evaluator Collaborative Iterative Learning**:
    - **Function**: Continuously cleans noisy logs, reduces manual labeling costs, and prevents intra-family model error reinforcement.
    - **Mechanism**: LLM-L calculates the conditional likelihood of LLM-S outputs from a generative perspective and judges input-output pair correctness from a discriminative perspective, combining them into a confidence score $c=(1-\alpha)P_{gen}+\alpha P_{disc}$. A separate black-box LLM (ERNIE 4.0) provides external evaluation via in-context reasoning. Samples with high-confidence agreement are automatically adopted, while discrepancies are handled by humans.
    - **Design Motivation**: Using only evaluators from the same model family can lead to inbreeding and inherited ASR noise; introducing a black-box evaluator and human arbitration improves the reliability of the data flywheel.

### Loss & Training
The initial offline training set consists of 5,000 dialogues, with 5,000 additional samples added per fine-tuning round. LLM-S uses ERNIE-Bot-tiny, LLM-L uses ERNIE-Bot-turbo, and the black-box evaluator is ERNIE 4.0. Training was conducted on 8 A100-80G GPUs via Baidu PaddleCloud using AdamW ($\beta_1=0.9, \beta_2=0.95, eps=1e-5$), batch size 128, sequence length 1024, and 3% warm-up. Maximum learning rates were $2\times10^{-5}$ for EB-turbo and $1\times10^{-4}$ for EB-tiny. EB-tiny underwent full parameter fine-tuning (bf16), while EB-turbo used LoRA, both for 2 epochs.

## Key Experimental Results

### Main Results
| Dataset / Scenario | Metric | DuIVRS-2 | Baseline | Gain / Description |
|--------|------|------|----------|------|
| Offline Deffect | CR | 81.62% | DuIVRS-1: 72.20% | Significant improvement in high-frequency natural distribution |
| Offline Dgeneral | CR | 73.70% | DuIVRS-1: 62.99% | Better generalization under uniform response sampling |
| Offline Drobust | CR | 76.22% | DuIVRS-1: 69.05% | More robust with long-text/complex semantics |
| Offline Average | CR | 77.18% | DuIVRS-1: 68.08%, GPT-4o: 66.68%, DeepSeek-V3: 67.20% | 13.37% higher than DuIVRS-1, 15.74% higher than GPT-4o |
| Online A/B | TSR | 83.9% | DuIVRS-1: 79.9%, Human: 89.6% | 4% improvement over old system, reaching 93.64% of human level |

### Ablation Study
| Configuration | Average CR | Description |
|------|---------|------|
| DuIVRS-2 | 77.18% | Full FSM DA, CoT, and Collaborative Iterative Learning |
| HybridLLMs | 77.03% | ERNIE replaced with Qwen2.5-1.5B/7B or GPT-4o; framework is model-agnostic |
| LLM-DM | 68.35% | Before adding iterative learning; already close to DuIVRS-1 |
| Direct-SFT | 60.80% | Direct generation of next response; lacks stability |
| w/o-CoT | 39.00% | Severe degradation without reasoning steps |
| w/o-DA | 64.33% | Poor long-tail generalization without data augmentation |

### Key Findings
- **Production constraints met**: DuIVRS-2 cost is < ¥0.2/call, comparable to DuIVRS-1 and much lower than human cost (¥1.5/call). Average reaction time is 130ms, below the 200ms human perception threshold.
- **System throughput**: Reaches 0.4 million calls/day, compared to <200 calls/day for humans. During A/B testing, DuIVRS-2 handled ~3,000 calls/day in controlled windows for stability verification.
- **Hallucination suppression**: Selective generation significantly inhibits hallucinations. Manual evaluation showed a 0% hallucination rate for DuIVRS-2, vs. 1.30% for Direct-SFT and 2.08% for w/o-CoT.
- **Deployment optimization**: Under 8-bit quantization, each A10 GPU uses ~22GB VRAM, achieving 130ms/query and a throughput of ~61.5 QPS/GPU.

## Highlights & Insights
- The value of this paper lies in its "industrial deployability" rather than solo model tricks. It covers the full lifecycle: data, strategy, evaluation, deployment, cost, throughput, and online A/B testing.
- The most critical design is restricting the LLM to FSM candidate actions. This leverages semantic understanding without relinquishing business logic to open-ended generation, suitable for low-fault-tolerance telephony.
- The dual-evaluator mechanism is practical: the domain-specific LLM-L possesses business knowledge, the black-box LLM provides independent judgment, and humans handle only uncertain samples, creating a sustainable cleaning pipeline.
- HybridLLM results are persuasive, showing that gains stem from the framework design rather than proprietary advantages of ERNIE models.

## Limitations & Future Work
- The system heavily relies on existing FSMs and candidate action spaces. Cold-starting open-domain telephony tasks without mature flows still requires manual definition of states and legal responses.
- Using internal Baidu Maps logs and the ERNIE ecosystem makes it difficult for external researchers to fully replicate online data, ASR noise distributions, and business constraints.
- While 130ms latency meets interactive needs, it is notably slower than DuIVRS-1's 15ms. Massive concurrency or more complex models might require further inference optimization.
- The system retains legacy ASR/TTS modules; real failures may still originate from speech link issues like recognition errors, background noise, or user interruptions, which are outside the scope of dialogue management.

## Related Work & Insights
- **vs. DuIVRS-1**: DuIVRS-1 used an NLU-DM-NLG modular pipeline; DuIVRS-2 adopts LLM-based end-to-end dialogue management to reduce error propagation while maintaining FSM controllability.
- **vs. General LLMs (GPT-4o / DeepSeek-V3)**: Proved that general models underperform on offline CR compared to DuIVRS-2, highlighting the need for domain data, action constraints, and business-focused evaluation loops.
- **vs. Traditional TOD Systems**: Traditional Task-oriented Dialogue (TOD) systems require the maintenance of multiple modules (DST, Policy, NLG). DuIVRS-2 integrates policy selection into selective generation, facilitating faster business logic updates.
- **Insights for Voice Agents**: Production-oriented LLM Agents should focus on embedding model capabilities into verifiable action spaces rather than pursuing completely free generation, using online data for continuous refinement.

## Rating
- **Novelty**: ⭐⭐⭐⭐☆ The framework for industrial LLM integration into IVR is comprehensive and pragmatic; its strength lies in the combination of components.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers offline, ablation, online A/B, cost, latency, throughput, hallucinations, and resource analysis.
- **Writing Quality**: ⭐⭐⭐⭐☆ Engineering processes and metrics are clear, though some diagrams depend on internal system context.
- **Value**: ⭐⭐⭐⭐⭐ Highly relevant for industrial LLM agents, low-latency speech systems, and controllable dialogue generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] MedMosaic: A Challenging Large Scale Benchmark of Diverse Medical Audio](../../ICML2026/audio_speech/medmosaic_a_challenging_large_scale_benchmark_of_diverse_medical_audio.md)
- [\[AAAI 2026\] Thucy: An LLM-based Multi-Agent System for Claim Verification across Relational Databases](../../AAAI2026/audio_speech/thucy_an_llm-based_multi-agent_system_for_claim_verification_across_relational_d.md)
- [\[ACL 2026\] VoxMind: An End-to-End Agentic Spoken Dialogue System](voxmind_an_end-to-end_agentic_spoken_dialogue_system.md)
- [\[NeurIPS 2025\] Sensorium Arc: AI Agent System for Oceanic Data Exploration and Interactive Eco-Art](../../NeurIPS2025/audio_speech/sensorium_arc_ai_agent_system_for_oceanic_data_exploration_and_interactive_eco-a.md)
- [\[ACL 2026\] Still Between Us? Evaluating and Improving Voice Assistant Robustness to Third-Party Interruptions](still_between_us_evaluating_and_improving_voice_assistant_robustness_to_third-pa.md)

</div>

<!-- RELATED:END -->
