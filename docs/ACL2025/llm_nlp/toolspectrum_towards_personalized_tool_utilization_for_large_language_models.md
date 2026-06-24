---
title: >-
  [Paper Note] ToolSpectrum: Towards Personalized Tool Utilization for Large Language Models
description: >-
  [ACL 2025][LLM (Other)][Personalized tool use] This paper proposes the ToolSpectrum benchmark to define and evaluate the personalized tool utilization capabilities of LLMs for the first time—selecting the most appropriate tools based on user profiles and environmental factors. Experiments demonstrate that personalization significantly improves user experience, but existing LLMs exhibit limited capability in jointly reasoning over both user and environmental factors.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Personalized tool use"
  - "Large Language Models"
  - "User profile"
  - "Environmental factors"
  - "Benchmark"
date: 2026-05-08
content_hash: 940f0f9199d04bbb
---

# ToolSpectrum: Towards Personalized Tool Utilization for Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2505.13176](https://arxiv.org/abs/2505.13176)  
**Code**: [https://github.com/BUAA-IRIP-LLM/ToolSpectrum](https://github.com/BUAA-IRIP-LLM/ToolSpectrum)  
**Area**: Signal & Communication  
**Keywords**: Personalized tool use, Large Language Models, User profile, Environmental factors, Benchmark

## TL;DR

This paper proposes the ToolSpectrum benchmark to define and evaluate the personalized tool utilization capabilities of LLMs for the first time—selecting the most appropriate tools based on user profiles and environmental factors. Experiments demonstrate that personalization significantly improves user experience, but existing LLMs exhibit limited capability in jointly reasoning over both user and environmental factors.

## Background & Motivation

**Background**: Integrating external tools into LLMs has become a mainstream paradigm for enhancing their capabilities, achieving remarkable progress in domains such as travel planning, online shopping, and knowledge acquisition. Existing tool learning benchmarks (such as ToolBench, API-Bank, and AppBench) primarily evaluate LLMs' tool selection and execution capabilities.

**Limitations of Prior Work**: Existing approaches focus solely on functional tool selection—choosing tools that can complete the user's instructions—while neglecting personalized selection among tools with overlapping functionalities. In reality, multiple tools might be capable of completing the same task (e.g., both Amazon and Temu can be used for shopping), but the optimal choice can vary dramatically based on factors such as user budget preferences, age restrictions, and weather conditions.

**Key Challenge**: Existing benchmarks treat tool selection as a pure functional matching problem. However, true user satisfaction requires LLMs to understand "who needs what under which circumstances"—which demands joint reasoning over both user profiles (Profile) and environmental factors (Environment). Existing methods fail to capture such context-sensitive personalized needs.

**Goal**: (1) Define the task of personalized tool utilization, (2) build an evaluation benchmark that covers user profiles and environmental factors, and (3) evaluate the personalized tool utilization capabilities of existing LLMs.

**Key Insight**: The authors formalize personalized tool utilization as $t = \text{Model}(I, \mathcal{P}, \mathcal{E}, \mathcal{T})$, where $\mathcal{P}$ represents the user profile, $\mathcal{E}$ represents environmental factors, and $\mathcal{T}$ represents the toolset. The output includes app selection, API call, required parameters, and personalized optional parameters.

**Core Idea**: For the first time, personalized recommendation concepts are introduced into the field of tool learning, constructing a tool utilization benchmark that simultaneously factors in user profiles and environmental conditions.

## Method

### Overall Architecture

The construction of ToolSpectrum consists of four stages: (1) Toolset collection—collecting apps and APIs with overlapping functionalities from 9 common application domains; (2) Definition and collection of user profiles and environmental factors—defining three types of user attributes (demographics, personality, preferences) and three types of environmental factors (natural environment, digital environment, app policies); (3) Tool invocation result collection—simulating user instructions and personalized execution results; and (4) Quality evaluation—multi-dimensional scoring and human verification. The final dataset contains three types of scenarios: Profile (450 instances), Environment (220 instances), and Profile & Environment (330 instances).

### Key Designs

1. **User Profile Definition (Profile)**:

    - **Function**: Model individual user factors that affect tool selection.
    - **Mechanism**: Categorized into three main classes: Demographics (key-value pairs such as gender, age, weight, height, occupation, education, and income), Personality (natural language descriptions of interests and hobbies), and Preference (historical app usage preferences). For instance, height and weight influence clothing size selection, income affects price sensitivity, and sports hobbies affect the choice of health apps.
    - **Design Motivation**: Drawing inspiration from user modeling in personalized recommendation systems, but extending it to tool utilization scenarios.

2. **Environmental Factors Definition (Environment)**:

    - **Function**: Model external contextual factors that affect tool selection.
    - **Mechanism**: Categorized into Natural Environment (key-value pairs such as weather, date, time, and location), Digital Environment (network conditions, device configurations, etc.), and App Domain Policy (app-specific policy rules, e.g., spending limits for minors). For instance, train tickets should be recommended instead of flight tickets in storm weather, and image quality should be reduced under poor network conditions.
    - **Design Motivation**: Prior personalization research has mostly focused solely on user profiles, ignoring the crucial impact of environmental constraints on tool selection.

3. **Standardized Output of Tool Invocation Results**:

    - **Function**: Uniformly evaluate various dimensions of personalized tool utilization.
    - **Mechanism**: The output is structured as a dictionary $\{APP \mapsto a, API \mapsto s, RP \mapsto r, OP \mapsto o\}$, covering four levels: App selection, API selection, Required Parameter extraction, and Optional Parameter filling. If the user instruction violates app policies when considering profiles and environment, it should return None.
    - **Design Motivation**: Decomposing evaluation into multiple granularities helps precisely pinpoint the weaknesses of LLMs in personalized reasoning.

### Loss & Training

As a benchmark paper, this work does not involve training. Data construction uses GPT-4o to generate initial data, followed by automatic scoring (three dimensions, 1-10 scale, discarding instances scoring below 8, removing 21.8% of the data) and manual sampling verification (50 per domain, with an average score of 8.7).

## Key Experimental Results

### Main Results

| Model | Profile APP | Profile RP | Profile PP | Env APP | Env OP | Both APP | Both OP |
|------|------------|------------|------------|---------|--------|----------|---------|
| Qwen2.5-7B | 0.73 | 0.59 | 0.27 | 0.66 | 0.03 | 0.22 | 0.06 |
| Qwen2.5-32B | 0.74 | 0.67 | 0.47 | 0.77 | 0.14 | 0.24 | 0.15 |
| GPT-4o | **Best** | **Best** | **Best** | **Best** | **Best** | **Best** | **Best** |
| Qwen2.5-3B | 0.16 | 0.12 | 0.12 | 0.55 | 0.00 | 0.12 | 0.04 |

### Ablation Study

| Condition | Key Observation | Description |
|------|---------|------|
| Profile Only | Relatively high APP/API accuracy | Models can basically identify user preferences |
| Environment Only | OP (optional parameters) is generally extremely low | Weak utilization of environmental information |
| Profile & Environment | Overall drop | Joint reasoning is the most challenging |
| Small Models (3B) | Extremely low across all metrics | Personalized reasoning requires sufficient model capacity |
| Large Models (32B+) | Significant improvement in PP/OP | Model scale is crucial for generating personalized parameters |

### Key Findings

- **Personalization significantly improves effectiveness**: Integrating personalized factors significantly enhances the effectiveness of tool utilization, validating the importance of personalization in tool learning.
- **Joint reasoning is the primary challenge**: Even SOTA models experience a substantial drop in performance when simultaneously considering Profile and Environment. Models tend to prioritize one dimension at the expense of the other.
- **Optional parameters (OP) are the bottleneck**: All models perform extremely poorly on generating personalized/environment-related optional parameters (mostly $\le 0.15$), indicating that models are still unable to effectively translate contextual information into specific parameter settings.
- **Significant model scale effects**: The 3B model is almost unable to complete personalized tool utilization. While 7-32B models exhibit obvious improvements, even the largest open-source models are far from resolving this problem.

## Highlights & Insights

- **The contribution of task definition outweighs the method**: Introducing personalization to tool learning for the first time defines a previously neglected yet practically vital problem space. This holds greater long-term value than any specific algorithmic innovation.
- **Exquisite evaluation granularity**: Decomposing evaluation into four levels (APP $\rightarrow$ API $\rightarrow$ RP $\rightarrow$ OP) enables precise diagnosis of where the model's reasoning chain breaks during personalization.
- **App policy violation detection**: Introducing a "return None" mechanism to test if models can identify actions that violate policies (e.g., high spending by minors), which is critical for real-world deployment.

## Limitations & Future Work

- **Limited data scale**: The dataset contains only 1,000 instances in total (~100 instances per domain), which may be insufficient to fully evaluate model capabilities.
- **Incomplete scenario coverage**: Although the 9 domains are common, some important application domains (such as education and productivity tools) are not covered.
- **GPT-4o construction bias**: Both data and annotations are generated by GPT-4o, which might introduce bias from the model; human verification only sampled 50 instances per domain.
- **Lack of improvement methods**: This work only diagnoses the problem and does not propose methods to enhance LLMs' personalized tool utilization capabilities, such as RAG augmentation, prompt engineering, or fine-tuning strategies.
- **Multi-model collaboration scenarios not considered**: In real-world scenarios, multi-step tool invocations and interactions between tools may be required, whereas the current benchmark only evaluates single-step invocation.

## Related Work & Insights

- **vs ToolBench / API-Bank**: These benchmarks evaluate general tool utilization capabilities, whereas this work focuses on personalized selection among functionally overlapping tools, introducing a new dimension to tool learning.
- **vs $\tau$-Bench**: $\tau$-Bench considers the Environment but lacks the Profile, and is not specifically designed to evaluate personalization; this work covers both dimensions simultaneously.
- **vs LaMP / PersonaChat**: These personalization benchmarks focus on dialogue/text generation without involving tool utilization; this work extends the concept of personalization to tool invocation scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ Defines the personalized tool utilization problem for the first time, offering a novel direction and addressing explicit practical needs
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluates a large number of open-source and closed-source models with rich analysis dimensions
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition, complete formalization, and intuitive figures and tables
- Value: ⭐⭐⭐⭐ Highlights the direction for the personalized development of LLM Agents; the benchmark itself holds high reuse value

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] HiCUPID: Exploring the Potential of LLMs as Personalized Assistants](exploring_the_potential_of_llms_as.md)
- [\[ACL 2025\] DICE-Bench: Evaluating the Tool-Use Capabilities of Large Language Models in Multi-Round, Multi-Party Dialogues](dice-bench_evaluating_the_tool-use_capabilities_of_large_language_models_in_mult.md)
- [\[ACL 2025\] ToolCoder: A Systematic Code-Empowered Tool Learning Framework for Large Language Models](toolcoder_code_empowered_tool_learning.md)
- [\[ACL 2025\] Personalized Generation In Large Model Era: A Survey](personalized_generation_in_large_model_era_a_survey.md)
- [\[ACL 2025\] Divide-Then-Aggregate: An Efficient Tool Learning Method via Parallel Tool Invocation](dta_llama_parallel_tool_invocation.md)

</div>

<!-- RELATED:END -->
