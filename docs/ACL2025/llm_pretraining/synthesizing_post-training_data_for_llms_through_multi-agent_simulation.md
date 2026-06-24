---
title: >-
  [Paper Note] Synthesizing Post-Training Data for LLMs through Multi-Agent Simulation
description: >-
  [ACL 2025][LLM Pretraining][Multi-Agent Simulation] This paper proposes the MATRIX multi-agent simulator and the MATRIX-Gen scenario-driven instruction generator to synthesize high-quality LLM post-training data by simulating real-world social scenarios. Llama-3-8B trained on only 20K synthesized data outperforms the official Meta Llama-3-8B-Instruct (trained on over 10M data) on AlpacaEval 2 and Arena-Hard.
tags:
  - "ACL 2025"
  - "LLM Pretraining"
  - "Multi-Agent Simulation"
  - "Post-Training Data Synthesis"
  - "Social Simulation"
  - "Instruction Tuning"
  - "Scenario-Driven"
date: 2026-05-08
content_hash: 938860be370f3c60
---

# Synthesizing Post-Training Data for LLMs through Multi-Agent Simulation

**Conference**: ACL 2025  
**arXiv**: [2410.14251](https://arxiv.org/abs/2410.14251)  
**Code**: None  
**Area**: LLM Pre-training  
**Keywords**: Multi-Agent Simulation, Post-Training Data Synthesis, Social Simulation, Instruction Tuning, Scenario-Driven

## TL;DR

This paper proposes the MATRIX multi-agent simulator and the MATRIX-Gen scenario-driven instruction generator to synthesize high-quality LLM post-training data by simulating real-world social scenarios. Llama-3-8B trained on only 20K synthesized data outperforms the official Meta Llama-3-8B-Instruct (trained on over 10M data) on AlpacaEval 2 and Arena-Hard.

## Background & Motivation

**Background**: Post-training is a crucial step to endow pre-trained LLMs with instruction-following capabilities. Data synthesis methods such as Self-Instruct, WizardLM, and Magpie have become important pathways for obtaining training data, reducing the reliance on large-scale human annotation.

**Limitations of Prior Work**: Existing data synthesis methods lack the support of actual user scenarios when generating instructions. Although synthesized data may be sufficiently rich in complexity, it does not effectively reflect diverse real-world user needs. For instance, Self-Instruct expands based on seed data, and Magpie leverages model completion capabilities to generate instructions, but neither anchors instructions in concrete use scenarios. The authors verify through motivational experiments that instruction data generated based on specific user scenarios consistently outperforms data synthesized without scenario-based foundations.

**Key Challenge**: High-quality instruction data must reflect real and diverse user needs, yet real-world data is constrained by privacy, scarcity, and high annotation costs, whereas purely synthesized data lacks scenario coverage and authenticity.

**Goal**: (1) How to automatically generate diverse and realistic user scenarios? (2) How to controllably synthesize high-quality post-training data based on these scenarios? (3) Can the synthesis method adapt to different domains (general, reasoning, code, safety, multi-turn dialogues)?

**Key Insight**: Inspired by the recent success of LLMs in simulating human societies, the authors leverage multi-agent social simulation as a framework to generate realistic scenarios. By allowing a large number of agents with different backgrounds and goals to interact in a virtual society, diverse scenarios naturally emerge, which are then used as the context for data synthesis.

**Core Idea**: Generate realistic and diverse scenarios through multi-agent social simulation, and subsequently synthesize high-quality post-training data driven by these scenarios based on actual user needs.

## Method

### Overall Architecture

The system consists of a three-step pipeline: (1) Synthesizing social scenarios via the MATRIX multi-agent simulator; (2) Translating scenarios into instruction-response data using the MATRIX-Gen scenario-driven generator; (3) Fine-tuning the pre-trained LLM (SFT + DPO) using the synthesized data. The entire pipeline uses the same aligned LLM (Llama-3-8B-Instruct) to drive both the simulation and data synthesis.

### Key Designs

1. **MATRIX Multi-Agent Simulator — Real-World Driven Agents**:

    - **Function**: Build agents with realistic behavioral patterns to ensure the generated scenarios approximate reality.
    - **Mechanism**: Collect 1,000 real user profiles (including names, descriptions, and historical tweets) from the X platform, anonymize them using an LLM, and use them as the initial attributes of the agents. Generate life goals and action plans for each agent (e.g., a medical professor's goal might be to disseminate scientific knowledge, with plans including research, publishing papers, and giving lectures). Agents respond to new observations based on their memory databases and personalities, and proactively execute plans when there are no external observations, ensuring purposeful behavior.
    - **Design Motivation**: Existing social simulators (e.g., CAMEL, Generative Agents) feature limited scenarios and simple behaviors. The design based on real human profiles and goal-driven mechanisms allows agent behaviors to be more diverse and realistic.

2. **Homophily-Guided Communication Protocol**:

    - **Function**: Achieve efficient and realistic interactions among large-scale agents.
    - **Mechanism**: Based on the phenomenon of homophily in social science (where people tend to associate with others of similar characteristics), convert agent profiles into text embeddings and group them using constrained $K$-means clustering (200 groups, 1-10 people per group). Intra-group communication is selectively distributed to relevant agents by an LLM-driven Modulator; inter-group communication is determined by the Modulator evaluating the relevance of actions to the memory of other groups.
    - **Design Motivation**: Random communication yields a massive amount of meaningless interactions, decreasing efficiency and scenario quality. Homophily-based grouping not only simulates the structure of real-world social networks but also guarantees scalability by reducing irrelevant interactions.

3. **MATRIX-Gen Scenario-Driven Instruction Generator**:

    - **Function**: Convert simulated scenarios into high-quality post-training data for specific domains.
    - **Mechanism**: A three-step process: (i) Retrieve the most relevant simulated scenarios according to given domain requirements; (ii) Integrate each agent's persona and actions into the instruction synthesis prompt; (iii) Prompt the aligned LLM to generate instructions and corresponding responses. By controlling retrieval and prompt templates, SFT data (MATRIX-Gen-SFT), preference data (MATRIX-Gen-DPO), reasoning data (MATRIX-Gen-Reason), and domain-specific data can be flexibly generated.
    - **Design Motivation**: Scenarios provide realistic contextual anchors, making synthesized instructions naturally align with actual user needs. For example, when generating mathematics data, it can cover scenarios ranging from elementary school students' arithmetic questions to PhD students' theoretical proofs.

### Loss & Training

The fine-tuning stage adopts a standard two-stage strategy: SFT followed by DPO. SFT uses 10K samples trained for 2 epochs. DPO training continues based on the SFT model. Reasoning data uses DeepSeek-R1-Distill-Qwen-32B to generate responses, which are filtered based on "think" block length.

## Key Experimental Results

### Main Results

| Dataset/Benchmark | Metric | MATRIX-Gen | Best Baseline | Llama-3-8B-Instruct (10M+) |
|------------|------|-----------|--------------|---------------------------|
| AlpacaEval 2 (Llama-3-8B) | LC Win Rate | **14.70%** | 12.63% (Magpie) | - |
| Arena-Hard (Llama-3-8B) | Win Rate | **14.70%** | 11.20% (Magpie) | - |
| AlpacaEval 2 (Qwen-2.5-7B) | LC Win Rate | **25.85%** | 14.76% (Magpie) | - |
| Arena-Hard (Qwen-2.5-7B) | Win Rate | **43.20%** | 23.60% (Tulu3) | - |

DPO Stage (based on MATRIX-SFT-Model):

| Benchmark | MATRIX-Gen-DPO | Magpie-PRO-DPO | Llama-3-8B-Instruct |
|------|---------------|----------------|---------------------|
| AlpacaEval 2 LC | **24.20%** | 18.99% | 22.92% |
| Arena-Hard | **22.70%** | 15.90% | 20.60% |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Agent scale $10^3$ vs $10^2$ | Higher AlpacaEval/Arena scores | Larger scale agents generate more diverse scenarios |
| Homophily communication vs Random communication vs No communication | Homophily is optimal | Evaluates the effectiveness of the communication protocol |
| Scenario scale $10^4$ vs $10^3$ | Higher quality data | More scenarios cover a broader range of human needs |

### Key Findings

- Models trained on only 20K MATRIX synthesized data outperform Llama-3-8B-Instruct trained on more than 10M data, demonstrating the extreme efficiency of scenario-driven synthesized data.
- The advantage of MATRIX-Gen is more pronounced on larger base models (Qwen-2.5-7B) (43.2% vs 23.6% on Arena-Hard).
- The framework showcases strong domain controllability, outperforming domain-specific baselines on specialized tasks like code, multi-turn dialogues, and safety.
- Reasoning data (MATRIX-Gen-Reason) significantly leads on HumanEval and MBPP, with clear advantages on GPQA as well.

## Highlights & Insights

- For the first time, large-scale multi-agent social simulation is applied to LLM post-training data synthesis, opening up a new direction.
- The data efficiency gap of 20K vs 10M+ is striking, indicating that data quality (scenario authenticity and diversity) is far more important than quantity.
- The homophily-guided communication protocol ingeniously incorporates social science theory into technical design, boosting scenario realism while ensuring efficiency.
- A unified framework simultaneously covers requirements for multiple data formats, including SFT, DPO, reasoning, code, safety, and multi-turn dialogues.

## Limitations & Future Work

- The simulator itself is driven by Llama-3-8B-Instruct; the upper bound of the synthesized data quality is constrained by the capability of this model.
- The 1,000 agent profiles originate from the X platform, which may yield a demographic distribution bias (under-covering certain professions or cultural backgrounds).
- The completeness of the anonymization process has not been strictly verified, posing potential privacy risks.
- The number of clusters (200) and group sizes (1-10) in the communication protocol are set based on hardware limitations, and their performance at an even larger scale remains unverified.
- All experiments were only conducted on 8B-scale models; the effectiveness on larger models remains unknown.

## Related Work & Insights

- Distinct from PersonaHub, MATRIX involves not only role-playing but also interactions between agents, which are key to generating complex, realistic scenarios.
- In contrast to Generative Agents by Park et al. (2023), which focuses on small-scale simple scenarios, MATRIX scales this up to large-scale complex interactions.
- **Insight**: Social simulation may not only be useful for data synthesis but could also be applied to evaluating LLM social behaviors, generating benchmarks, and more.

## Rating

- **Novelty**: 9/10 — Utilizing multi-agent social simulation for data synthesis presents a highly novel perspective.
- **Technical Depth**: 7/10 — The system design is elegant, but individual components (clustering, retrieval, generation) are not overly complex.
- **Experimental Thoroughness**: 9/10 — Compares against 20 baselines across 12 benchmarks, covering multiple domains.
- **Writing Quality**: 8/10 — Clear structure, rich in tables and figures.
- **Value**: 9/10 — The highly efficient synthesis pipeline holds direct value for practical LLM training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Improving Continual Pre-training Through Seamless Data Packing](improving_continual_pre-training_through_seamless_data_packing.md)
- [\[ACL 2025\] Stealing Training Data from Large Language Models in Decentralized Training through Activation Inversion Attack](stealing_training_data_from_large_language_models_in_decentralized_training_thro.md)
- [\[ACL 2025\] Meta-rater: A Multi-dimensional Data Selection Method for Pre-training Language Models](metarater_a_multidimensional_data_selection_method.md)
- [\[ACL 2025\] Pre-Training Curriculum for Multi-Token Prediction in Language Models](pre-training_curriculum_for_multi-token_prediction_in_language_models.md)
- [\[ACL 2025\] Data-Constrained Synthesis of Training Data for De-Identification](data-constrained_synthesis_of_training_data_for_de-identification.md)

</div>

<!-- RELATED:END -->
