---
title: >-
  [Paper Note] Challenges and Future Directions of Data-Centric AI Alignment
description: >-
  [ICML 2025][LLM Alignment][data-centric alignment] This paper is a position paper advocating for shifting the research focus of AI alignment from algorithm design to data quality. Through qualitative analysis of the Anthropic-HH dataset, it reveals six major sources of unreliability in human feedback and proposes future directions for improving data collection, cleaning, and verification.
tags:
  - "ICML 2025"
  - "LLM Alignment"
  - "data-centric alignment"
  - "human feedback"
  - "RLHF"
  - "preference data"
  - "feedback reliability"
date: 2026-05-08
content_hash: a8069601eea99de3
---

# Challenges and Future Directions of Data-Centric AI Alignment

**Conference**: ICML 2025  
**arXiv**: [2410.01957](https://arxiv.org/abs/2410.01957)  
**Code**: None  
**Area**: Alignment RLHF  
**Keywords**: data-centric alignment, human feedback, RLHF, preference data, feedback reliability

## TL;DR

This paper is a position paper advocating for shifting the research focus of AI alignment from algorithm design to data quality. Through qualitative analysis of the Anthropic-HH dataset, it reveals six major sources of unreliability in human feedback and proposes future directions for improving data collection, cleaning, and verification.

## Background & Motivation

**Background**: Current AI alignment methods mainly focus on the algorithmic level—such as learning reward functions via RLHF and directly optimizing preferences via DPO. These methods rely on delicately designed optimization algorithms and loss functions to steer model behavior.

**Limitations of Prior Work**: Algorithm-centric alignment methods implicitly assume that training data accurately reflects real human preferences, but this assumption often fails in practice. Human judgment is complex and unreliable. Even with well-designed algorithms, if the training data itself is flawed, the alignment performance will still be severely compromised.

**Key Challenge**: Current research overemphasizes "how to optimize" (the algorithmic side) while neglecting "what to optimize" (the data side). The bottleneck of algorithm-centric approaches is their assumption of perfect training preference data, whereas real-world human feedback is fraught with noise, bias, and inconsistency.

**Goal**: (1) Systematically identify specific sources of unreliability in human feedback data; (2) Analyze the limitations faced by AI-generated feedback; (3) Propose future research directions for data-centric alignment.

**Key Insight**: The authors conduct an in-depth qualitative analysis of the Anthropic-HH dataset, re-examining the quality of preference labels through manual annotation, thereby discovering systematic pattern issues.

**Core Idea**: AI alignment needs to shift from "algorithm-centric" to "data-centric", where data quality and representation are as important as, if not more critical than, algorithm design.

---

## Method

### Overall Architecture

This paper constructs an analytical framework for data-centric alignment, categorizing feedback sources into human feedback and AI feedback. It systematically analyzes the challenges faced by each and proposes improvement directions spanning three dimensions: data collection, data cleaning, and feedback verification.

### Key Designs

1. **Analysis of Six Sources of Unreliability in Human Feedback**:

    - **Function**: Reveal the specific causes of noise in preference data through qualitative annotation.
    - **Mechanism**: Re-annotate a subset of the Anthropic-HH dataset, grouping and analyzing samples characterized by low inter-annotator agreement (low IAA) and "both are bad" cases. Six categories of issues are identified: (1) Human annotation errors, where the rejected response is actually better; (2) High subjectivity and lack of context, such as subjective queries like travel recommendations without objective criteria; (3) Divergent preference standards, e.g., preferring direct answers versus requesting clarification; (4) Different standard thresholds, with varying bars for what constitutes "good enough"; (5) Both responses containing harmful advice; (6) Both responses containing incorrect/irrelevant information.
    - **Design Motivation**: Demonstrate that preference data noise is systematic rather than random, implying that algorithm-level robustness alone is insufficient to resolve it.

2. **Three Challenges of AI Feedback**:

    - **Function**: Analyze the limitations of substituting human annotators with AI.
    - **Mechanism**: Identify three core issues in AI feedback: (1) Dependence on the underlying model—AI feedback is constrained by the diversity and biases of its training data; (2) Inability to truly reflect human values—AI optimizes quantifiable metrics but misses the subtle nuances of ethical reasoning, and suffers from presentation, social, content, and cognitive biases; (3) Insufficient consistency—GPT-4's selections are near-random across multiple trials when evaluating responses with subtle differences.
    - **Design Motivation**: Explain that simply replacing human annotation with AI is not a silver bullet, highlighting the need for human-AI collaboration.

3. **Seven Future Directions for Data-Centric Alignment**:

    - **Function**: Outline a research roadmap for the field.
    - **Mechanism**: Propose seven specific directions across three main categories: **Data Collection Improvements** include (Dir. 1) comprehensive feedback collection—ensuring coverage in terms of annotator, prompt, and response diversity; (Dir. 2) dynamic longitudinal preference collection—tracking drift in human values over time; (Dir. 3) validating data collection protocols—introducing options like "both are good" or "both are bad". **Data Cleaning** includes (Dir. 4) human-AI collaboration to mitigate unreliability—using reward model ensembles to identify and flip erroneous human labels; (Dir. 5) prioritizing data quality over quantity—training on only 5% of the data can outperform full-dataset training. **Feedback Verification** includes (Dir. 6) introducing human oversight for AI feedback; (Dir. 7) standardizing feedback verification processes.
    - **Design Motivation**: Bridge the gap between theoretical analysis and practical improvements.

### Loss & Training

As this is a position paper, no new loss functions are proposed. However, it discusses how changes in annotation format (e.g., adding "both are bad" options) affect reward modeling and reinforcement learning alignment algorithm design.

## Key Experimental Results

### Main Results

The core "experiment" in this paper is the qualitative annotation analysis. The distribution of Low Inter-Annotator Agreement (Low IAA) samples is as follows:

| Source of Unreliability | Low IAA Data Share | "Both are bad" Data Share |
|---|---|---|
| Human Annotation Error | 2% | 0% |
| High Subjectivity | 28% | 0% |
| Different Preference Standards | 29% | 25% |
| Different Standard Thresholds | 37% | 0% |
| Harmful Advice | 0% | 39% |
| Erroneous/Irrelevant Info | 4% | 36% |

### Ablation Study

Comparison of key differences between data-centric and algorithm-centric alignment:

| Dimension | Data-Centric Alignment | Algorithm-Centric Alignment |
|---|---|---|
| Focus | Quality and representation of feedback data | Reward models and optimization algorithms |
| Core Challenges | Data bias, feedback reliability, diversity | Reward hacking, robustness, preference aggregation |
| Primary Goal | Ensuring data reflects real human values | Creating theoretical guarantees or reward structures |

### Key Findings

- In Low IAA samples, 65% of disagreement stems from subjectivity (28%) and differing preference standards/thresholds (29% + 37%), whereas human annotation error accounts for only 2%.
- In "Both are bad" samples, 75% are caused by harmful advice (39%) and incorrect information (36%), which could have been circumvented by providing a "both are bad" option to avoid forced choices.
- Data cleaning literature demonstrates that training with only 5% of curated data can outperform training with 100% of the full dataset (Li et al., 2024d).

## Highlights & Insights

- Shifting the alignment problem from an algorithmic perspective to a data perspective is highly inspiring. The simple machine learning wisdom of "garbage in, garbage out" has long been neglected in the alignment domain.
- The categorization of the six sources of unreliability is both comprehensive and actionable, providing a clear decomposition of problems for subsequent research.
- The incorporation of social science literature regarding survey design (such as Olsen 1999's study on "both are bad" options) demonstrates a valuable interdisciplinary perspective.

## Limitations & Future Work

- The qualitative analysis is restricted to a small subset of a single dataset (Anthropic-HH), making the scale of analysis small and the generalizability of the conclusions yet to be verified.
- As a position paper, it lacks concrete algorithm designs and quantitative experimental validation.
- Most of the proposed future directions remain at a conceptual level, lacking concrete implementation paths and feasibility analyses.
- The paper does not fully discuss how data-centric and algorithm-centric approaches can synergize, as they are not mutually exclusive.
- Discussion on AI feedback consistency is relatively superficial; for instance, position bias (Wang et al., 2024c) is only briefly mentioned.
- It lacks discussion on the specific challenges and solutions of collecting preference data in multilingual and multicultural contexts.

## Related Work & Insights

- This work complements algorithm-centric methods such as RLHF (Ouyang et al., 2022) and DPO (Rafailov et al., 2023).
- The PRISM dataset (Kirk et al., 2024) serves as a pioneering work in the data-centric direction, collecting preferences from 1,500 participants across 75 countries.
- Weak LLMs can provide feedback on par with human annotations (Tao & Li, 2025), pointing toward scalable alignment strategies.
- **Insight**: In practical alignment tasks, rather than chasing state-of-the-art alignment algorithms, one should first audit and improve the quality of preference data.
- Human-AI collaborative schemes in data cleaning (such as reward model ensembles, Yeh et al., 2024a) warrant validation on a larger scale.
- The insight of adding "both are good/bad" choices in preference collection can be transferred to other annotation task designs.
- The impact of demographic diversity on prompt distribution (Kirk et al., 2024, PRISM) reminds us of the biases introduced by annotator selection.

## Rating

⭐⭐⭐ (6/10)

This position paper offers a novel perspective, and its analysis of the six sources of unreliability is valuable. However, its research contribution is somewhat limited structurally—lacking concrete algorithms and quantitative validation, it reads more like a literature survey and discussion of existing issues. The proposed future directions are relatively broad with limited immediate actionability. It is suitable as introductory reading and for problem-definition reference in this domain.

Notably, the finding that data quality takes precedence over data scale possesses broad applicability—Li et al. (2024d) surpassed full-data training using only 5% of Alpaca data, and Lu et al. (2024) outperformed 50K data points with just 6K, which holds significant promise for reducing alignment costs. The paper's most guiding contribution lies in systematically introducing the philosophy of "Data-Centric AI" into the alignment domain, constructing a problem framework for subsequent, more concrete technical solutions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Enforcing Axioms for AI Alignment under Loss-Based Rules](../../ICLR2026/llm_alignment/enforcing_axioms_for_ai_alignment_under_loss-based_rules.md)
- [\[ACL 2025\] LLMs Caught in the Crossfire: Malware Requests and Jailbreak Challenges](../../ACL2025/llm_alignment/llms_caught_in_the_crossfire_malware_requests_and_jailbreak_challenges.md)
- [\[ICLR 2026\] Skywork-Reward-V2: Scaling Preference Data Curation via Human-AI Synergy](../../ICLR2026/llm_alignment/skywork-reward-v2_scaling_preference_data_curation_via_human-ai_synergy.md)
- [\[AAAI 2026\] Intrinsic Barriers and Practical Pathways for Human-AI Alignment: An Agreement-Based Complexity Analysis](../../AAAI2026/llm_alignment/intrinsic_barriers_and_practical_pathways_for_human-ai_alignment_an_agreement-ba.md)
- [\[ICML 2025\] On the Robustness of Reward Models for Language Model Alignment](on_the_robustness_of_reward_models_for_language_model_alignment.md)

</div>

<!-- RELATED:END -->
