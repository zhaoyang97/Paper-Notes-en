---
title: >-
  [Paper Note] FingerTip 20K: A Benchmark for Proactive and Personalized Mobile LLM Agents
description: >-
  [ICLR 2026][LLM Agent][mobile agent] FingerTip 20K collects 21,437 interaction records from 95 users during real-world daily smartphone usage—including user profiles, timestamps, locations…
tags:
  - "ICLR 2026"
  - "LLM Agent"
  - "mobile agent"
  - "proactive suggestion"
  - "personalized execution"
  - "GUI agent"
  - "benchmark"
date: 2026-05-08
content_hash: 22eae3b98f59d148
---

# FingerTip 20K: A Benchmark for Proactive and Personalized Mobile LLM Agents

**Conference**: ICLR 2026
**arXiv**: [2507.21071](https://arxiv.org/abs/2507.21071)  
**Code**: [https://github.com/tsinghua-fib-lab/FingerTip-20K](https://github.com/tsinghua-fib-lab/FingerTip-20K)  
**Area**: Agent
**Keywords**: mobile agent, proactive suggestion, personalized execution, GUI agent, benchmark

## TL;DR
FingerTip 20K collects 21,437 interaction records from 95 users during real-world daily smartphone usage—including user profiles, timestamps, locations, and historical intents—and introduces two new evaluation tracks: proactive task suggestion (predicting user intent) and personalized task execution (adapting to action preferences). The strongest model, Qwen-QVQ-Max, achieves only 12.8% success on proactive suggestion (vs. 30.3% for humans), while UI-TARS reaches only 38.5% on execution.

## Background & Motivation

**Background**: Mobile GUI agents leverage MLLMs to interpret screenshots and UI trees for automated smartphone operation. Existing agents follow a fully passive paradigm—they act only upon receiving explicit instructions and ignore user preferences during execution.

**Limitations of Prior Work**: (a) Users must formulate detailed instructions for every intent, increasing cognitive load. (b) Users sometimes cannot articulate latent needs clearly (e.g., wanting to read news during a commute without explicitly stating so). (c) Different users may follow substantially different action sequences to accomplish the same task (e.g., some prefer searching for apps, others prefer scrolling), yet existing agents make no such distinction. (d) Tasks in existing datasets are either author-defined or LLM-generated, failing to reflect authentic daily usage patterns.

**Key Challenge**: Achieving proactive and personalized behavior requires long-term interaction data that includes user context (time, location, profile) and history—yet existing datasets treat each record in isolation, lacking temporal correlation and contextual information.

**Goal**: (a) Construct a real-world mobile interaction dataset enriched with user context. (b) Define two new evaluation tracks: proactive task suggestion and personalized task execution.

**Key Insight**: Recruit 95 users to record one month of daily smartphone operations on their own devices via a dedicated app—each genuine intent triggers logging of intent text, action sequence, screenshots, location, and timestamp, forming temporally correlated long-term usage data.

**Core Idea**: Continuously collect context-rich interaction data from users' daily smartphone usage, and use this data to evaluate agents' proactive suggestion and personalized execution capabilities.

## Method

### Overall Architecture
Two new evaluation tracks: (1) **Proactive Task Suggestion**—given a user profile, timestamp, scene, historical intents (up to 20), and the first few screenshots of the current session, predict the user's current intent; (2) **Personalized Task Execution**—given a user instruction and historical action preferences, execute the task in a real-device environment such that the action sequence reflects the user's preferences.

### Key Designs

1. **Real User Data Collection**:

    - Function: 95 users record one month of daily operations on their own Android devices via the dedicated FingerTip app.
    - Core Pipeline: When a genuine intent arises → open app → record intent in one sentence → select current location type → switch to the target app and demonstrate the operation. The app automatically uploads the intent (with timestamp and location), screenshot sequence, UI tree, and action sequence.
    - Design Motivation: Unlike annotation approaches conducted in simulators, this captures authentic usage on users' own devices, with temporal correlations between records that reflect real usage patterns.

2. **Proactive Task Suggestion (Track 1)**:

    - Input: user profile $U$, timestamp $T$, scene $S$, historical intents $I_{history}$, first few screenshots of the current session $O$
    - Output: predicted current user intent $I$, explicitly specifying the app name and desired outcome
    - Evaluation: semantic similarity $Sim_1$ + intent matching success rate $SR_1$ (judged by DeepSeek-V3)

3. **Personalized Task Execution (Track 2)**:

    - Input: user profile $U$, ground-truth intent $I_{true}$, historical action sequences $A_{history}$, current screenshot
    - Output: actions executed on a real device until task completion
    - Evaluation: success rate $SR_2$ (human inspection) + personalization metric $Sim_2 = S_I / S_{II}$ (ratio of action sequence similarity to same-type vs. different-type users)

4. **Validation of Personalization Differences**:

    - Users are grouped by age; Levenshtein similarity of action sequences on similar tasks is computed within and across groups.
    - Result: intra-user similarity > same-type-user similarity > different-type-user similarity, confirming that action preferences exist and are measurable.

### Loss & Training
- Baseline evaluations use a zero-shot setting.
- Fine-tuning experiments: Qwen-2.5-VL-7B + LoRA (rank=4/64), sampling 1,000 or 16,000 examples from the training set.

## Key Experimental Results

### Main Results

**Proactive Task Suggestion (0 initial screenshots)**:

| Model | SR1 (%) | Sim1 | Time/Query |
|-------|---------|------|------------|
| GPT-4.1 | 7.2 | 0.35 | 5.64s |
| Qwen-QVQ-Max (thinking) | **12.8** | **0.39** | 10.60s |
| Human | **30.3** | **0.57** | - |

**Personalized Task Execution**:

| Model | SR2 (%) | Sim2 | Step Ratio |
|-------|---------|------|-----------|
| GPT-4.1 | 5.5 | 0.98 | 1.98 |
| Qwen-QVQ-Max | 9.5 | 1.04 | 1.94 |
| UI-TARS-1.5-7B | **38.5** | 1.06 | 1.22 |
| AppAgent | 11.0 | **1.12** | **1.13** |

### Ablation Study

| Configuration | Key Finding |
|---------------|-------------|
| Screenshots 0→3 | SR1 improves significantly, as screenshots narrow the intent search space. |
| Action length 1–5 vs. 11+ | SR2 decreases substantially as action sequence length increases. |
| Fine-tuning with 1K samples | SR1: 3.1→16.3%; SR2: 1.5→32.0% (vs. base Qwen-2.5-VL-7B) |
| Fine-tuning with 16K samples | SR1: 3.1→19.8%; SR2: 1.5→43.5% |

### Key Findings
- **Large human–machine gap**: The strongest model achieves 12.8% vs. 30.3% for humans on proactive suggestion, indicating that inferring intent from context remains a formidable challenge.
- **All models yield Sim2 ≈ 1.0**: Agents tend to complete tasks in a generic manner, entirely ignoring user preferences—personalized execution capability is virtually absent.
- **GUI-specialized models >> general-purpose models**: UI-TARS achieves 38.5% vs. GPT-4.1's 5.5%; the gap stems primarily from GUI grounding ability.
- **Fine-tuning is highly effective**: Even 1K samples substantially improve a 7B model on both tracks, indicating that user-contextual information in the data is learnable.

## Highlights & Insights
- **Two entirely new evaluation dimensions**: Proactive suggestion and personalized execution have never been systematically evaluated before; this work is the first to quantify the gaps on both fronts.
- **Rare value of authentic daily-use data**: One month of real smartphone usage from 95 users differs qualitatively from simulator-based annotations—records are temporally correlated and embed user context.
- **Implication of Sim2 ≈ 1.0**: Current agents not only fail to personalize execution but are unaware that personalization is expected, suggesting that user modeling mechanisms must be introduced at the architecture and training levels.
- **Large potential of fine-tuning**: A small amount of user-specific data yields dramatic capability gains, pointing to user-specific fine-tuning and in-context learning from user history as viable research directions.

## Limitations & Future Work
- Only 95 Chinese users are included; the user population and app ecosystem are geographically constrained (predominantly Chinese-language apps).
- Evaluation of proactive suggestion relies on semantic similarity and LLM judgment, which may introduce bias.
- The $Sim_2$ metric for personalized execution is based on Levenshtein distance, which is sensitive to action order but does not account for semantic equivalence.
- Validation is limited to Android; iOS scenarios are not covered.
- Data collection depends on users actively initiating recordings, potentially missing many unconscious smartphone interactions.

## Related Work & Insights
- **vs. AndroidWorld/AitW**: These benchmarks use predefined tasks and isolated records, lacking user context or history. FingerTip 20K derives from real usage with temporal correlations.
- **vs. Proactive Agent (Lu et al.)**: That work addresses text-based desktop/web scenarios only; FingerTip 20K focuses on visual mobile GUI scenarios at a larger scale.
- **vs. SPHINX/SPA-Bench**: These evaluate execution capability only, without assessing proactive suggestion or personalization.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Two entirely new evaluation dimensions—proactive suggestion and personalized execution—filling an important gap.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluates 7+ models with fine-tuning, human studies, and difficulty analysis, though real-device evaluation scale is limited.
- Writing Quality: ⭐⭐⭐⭐ Problem definitions are clear and data collection methodology is described in detail.
- Value: ⭐⭐⭐⭐⭐ Charts a clear direction for mobile agents beyond task execution capability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] GUI-CEval: A Hierarchical and Comprehensive Chinese Benchmark for Mobile GUI Agents](../../CVPR2026/llm_agent/gui-ceval_a_hierarchical_and_comprehensive_chinese_benchmark_for_mobile_gui_agen.md)
- [\[ICLR 2026\] ST-WebAgentBench: A Benchmark for Evaluating Safety and Trustworthiness in Web Agents](st-webagentbench_a_benchmark_for_evaluating_safety_and_trustworthiness_in_web_ag.md)
- [\[ICLR 2026\] A Benchmark for Deep Information Synthesis (DeepSynth)](a_benchmark_for_deep_information_synthesis.md)
- [\[ICLR 2026\] SimuHome: A Temporal- and Environment-Aware Benchmark for Smart Home LLM Agents](simuhome_a_temporal-_and_environment-aware_benchmark_for_smart_home_llm_agents.md)
- [\[ICLR 2026\] M²-Miner: Multi-Agent Enhanced MCTS for Mobile GUI Agent Data Mining](m2-miner_multi-agent_enhanced_mcts_for_mobile_gui_agent_data_mining.md)

</div>

<!-- RELATED:END -->
