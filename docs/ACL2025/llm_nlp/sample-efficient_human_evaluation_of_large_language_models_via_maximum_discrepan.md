---
title: >-
  [Paper Note] Sample-Efficient Human Evaluation of Large Language Models via Maximum Discrepancy Competition
description: >-
  [ACL 2025][LLM (Other)][LLM Evaluation] This paper proposes a highly sample-efficient human evaluation method based on the Maximum Discrepancy (MAD) competition principle. By automatically selecting a subset of instructions that best distinguish the performance differences between LLMs, it significantly reduces the human annotation workload, recovering stable model rankings from large-scale evaluations with only 280 comparisons.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "LLM Evaluation"
  - "Human Evaluation"
  - "Maximum Discrepancy Competition"
  - "Sample-Efficient"
  - "Elo Rating"
date: 2026-05-08
content_hash: f2d4f10cf0f22735
---

# Sample-Efficient Human Evaluation of Large Language Models via Maximum Discrepancy Competition

**Conference**: ACL 2025  
**arXiv**: [2404.08008](https://arxiv.org/abs/2404.08008)  
**Code**: [https://github.com/weiji-Feng/MAD-Eval](https://github.com/weiji-Feng/MAD-Eval)  
**Area**: LLM/NLP  
**Keywords**: LLM Evaluation, Human Evaluation, Maximum Discrepancy Competition, Sample-Efficient, Elo Rating

## TL;DR

This paper proposes a highly sample-efficient human evaluation method based on the Maximum Discrepancy (MAD) competition principle. By automatically selecting a subset of instructions that best distinguish the performance differences between LLMs, it significantly reduces the human annotation workload, recovering stable model rankings from large-scale evaluations with only 280 comparisons.

## Background & Motivation

**Background**: With the massive influx of LLMs, reliable evaluation has become crucial. Currently, there are three mainstream evaluation paradigms: (1) Standard benchmarks (e.g., MMLU, HumanEval), which rank models through automated scoring on fixed test sets; (2) LLM-as-a-judge (e.g., AlpacaEval), which uses strong LLMs to judge response quality; (3) Human evaluation (e.g., Chatbot Arena), which collects large-scale human preference comparisons.

**Limitations of Prior Work**: Standard benchmarks suffer from data leakage and overfitting risks, and fail to fully reflect human perception of natural language quality. LLM-as-a-judge exhibits systematic biases such as position bias, verbosity bias, and self-enhancement bias. Human evaluation remains the "gold standard" but is prohibitively expensive; for instance, Chatbot Arena requires tens of thousands of human pairwise battles to generate stable rankings, which is too costly for evaluating new models or specific scenarios.

**Key Challenge**: Performing human evaluation on a large number of test samples is prohibitively expensive, whereas evaluating on a small number of samples introduces severe sampling bias. The key question is: How can we evaluate LLMs most accurately with minimal human annotation effort?

**Goal**: To design an automated sample selection mechanism that curates a small set of the most informative and diverse test samples from a massive instruction pool, thereby maximizing the efficiency of human evaluation.

**Key Insight**: Drawing inspiration from the concept of "model falsification" in computer vision and software testing—if two models can still be distinguished on samples where they are most difficult to differentiate, their relative superiority is reliable. Conversely, if even the most challenging samples cannot distinguish them, they can be considered equivalent.

**Core Idea**: Using the Maximum Discrepancy (MAD) competition principle to automatically select instructions that best expose the differences between LLMs, combined with diversity constraints to ensure that the selected instructions cover diverse failure modes, then collecting human preferences on these few instructions to generate global rankings using the Elo rating system.

## Method

### Overall Architecture

The workflow of MAD-Eval consists of four steps: (1) For each evaluation scenario, construct a large-scale instruction pool $\mathcal{X}$ containing 30K instructions; (2) For each pair of LLMs $(f_i, f_j)$, select the Top-K instructions with the most discrepant and diverse responses using MAD competition; (3) Perform 3-alternative forced choice (3-AFC) human evaluation on the pairwise responses of the selected instructions ($f_i$ is better / $f_j$ is better / tie); (4) Input all pairwise results into the Elo rating system to generate global rankings. The inputs are a set of LLMs to be evaluated and a series of evaluation scenarios, and the outputs are the global capability ranking and sub-rankings for each scenario.

### Key Designs

1. **MAD Competition Sampling**:

    - **Function**: Automatically select the Top-K instructions from the instruction pool that best distinguish the performance differences between two LLMs.
    - **Mechanism**: For an LLM pair $(f_i, f_j)$, calculate the semantic similarity $\mathcal{M}(f_i(x), f_j(x))$ of the two model responses on each instruction $x$ (using the cosine similarity of text-embedding-ada-002), and select the instructions with the lowest similarity—i.e., those where the response gap between the two models is the largest. The formula is $\hat{x} = \arg\min_{x \in \mathcal{X}} \mathcal{M}(f_i(x), f_j(x))$. The instructions with the maximum discrepancy are most likely to expose the differences in superiority between the two models.
    - **Design Motivation**: Random sampling might select "easy" samples where both models perform well, failing to effectively differentiate performance. The MAD principle ensures that each selected instruction has maximized "discriminative power".

2. **Diversity Constraint**:

    - **Function**: Prevent MAD sampling from degenerating into selecting only a single type of instruction (e.g., selecting only poetry generation prompts).
    - **Mechanism**: When selecting the $k$-th instruction, in addition to requiring high response discrepancy, it is also required to be as semantically distinct as possible from the already selected instruction set $\mathcal{I}$. The optimization is formulated as $\hat{x}^{(k)} = \arg\min_{x \in \mathcal{X} \setminus \mathcal{I}} \mathcal{M}(f_i(x), f_j(x)) + \lambda \mathcal{M}(x, \mathcal{I})$, where the second term penalizes similarity to the selected instructions, and $\lambda$ controls the weight of diversity.
    - **Design Motivation**: Experiments show that without diversity constraints, 4 out of the Top-10 instructions are poetry-related—which only exposes differences in poetry writing and fails to provide a comprehensive evaluation. Incorporating diversity constraints ensures that almost every instruction represents a different type of task.

3. **Instruction Evolution Pool Construction**:

    - **Function**: Construct an instruction pool that is sufficiently large and diverse to approximate the entire input space of LLMs.
    - **Mechanism**: Sample 3K seed instructions from four scenarios (knowledge understanding, mathematical reasoning, creative writing, programming), and then leverage instruction evolution methods (similar to WizardLM's Evol-Instruct) to iteratively evolve them for 10 rounds using three models: GPT-4-Turbo, GPT-3.5-Turbo, and Gemini-Pro, finally obtaining 30K instructions per scenario. Using multiple generation models reduces preference bias toward any single model.
    - **Design Motivation**: The instruction pool needs to (a) be large enough to cover diverse test scenarios, (b) simulate real human-machine interaction distributions to avoid data leakage, and (c) originate from diverse sources to minimize bias.

### Loss & Training

MAD-Eval does not involve model training. Instruction selection uses a greedy strategy—sequentially selecting the instruction that minimizes the objective function and adding it to the selected set. Human evaluation uses a 3-alternative forced choice (3-AFC) method. Global ranking is generated using the Elo rating system ($\tau=400, \eta=4$). To reduce sensitivity to the order of matches, 1000 bootstrap samplings are performed to take the average.

## Key Experimental Results

### Main Results

| Model | MAD (Ours) | Chatbot Arena | AlpacaEval 2.0 | OpenCompass 2.0 |
|------|-----------|--------------|----------------|----------------|
| GPT-4-Turbo | **1** (1132) | 1 | 1 | 1 |
| Gemini-Pro | **2** (1107) | 2 | 2 | - |
| OpenChat-3.5 | **3** (1035) | 3 | 3 | - |
| GPT-3.5-Turbo | **4** (1034) | 4 | 4 | 2 |
| WizardLM-13B | **5** (937) | 5 | 3 | 5 |
| QWen-14B-Chat | **6** (932) | 7 | 6 | 3 |
| ChatGLM3-6B | **7** (929) | 8 | 8 | 4 |
| Vicuna-13B | **8** (894) | 6 | 7 | 6 |

The proposed method generates rankings highly consistent with Chatbot Arena (tens of thousands of comparisons) using only 280 human pairwise comparisons.

### Ablation Study

| Sampling Strategy | GPT-4 Rank | OpenChat Rank | Correlation with "Gold Standard" | Description |
|---------|-----------|------------|---------------|------|
| **MAD (Ours)** | **1** | **2** | **Highest** | Informative + Diverse |
| KL Divergence | 2 | 4 | Moderate | KL prefers specific types |
| Cross-Entropy | 4 | 2 | Low | Severe ranking bias |
| Random | 1 | 5 | Moderate | Unstable |

### Key Findings

- The MAD competition strategy closely approximates the "gold standard" ranking of 8K samples using only 10 carefully selected samples (in reasoning scenarios, SRCC > 0.95 when K > 5).
- Diversity constraints are crucial to the results—without diversity, 9/10 instructions chosen by the KL divergence strategy are poetry-related, causing severe bias.
- Three semantic similarity metrics (Ada-002 Embedding, BERTScore, GPT-4 judgment) produce nearly identical rankings, demonstrating that the method is insensitive to the choice of metric.
- The MAD method can identify counterexamples of GPT-4-Turbo (e.g., "laziness" tendencies, code exceeding limits, knowledge understanding deviation), providing direct instructional value for model improvement.
- In writing scenarios, longer responses are generally preferred by humans, with GPT-4-Turbo's average response being 454.8 words vs. ChatGLM3-6B's 221.2 words.

## Highlights & Insights

- **Transfer of the 'Model Falsification' Philosophy**: Successfully migrating the MAD competition concept from computer vision (Wang & Simoncelli, 2008) to NLP evaluation. The core insight is extremely refined: a good evaluation does not need to be comprehensive; it only needs to find the points that best expose differences.
- **Scalability of Incremental Evaluation**: When adding new models, there is no need to redo existing pairwise comparisons—simply generate $N \times K$ new comparisons with the new model over the existing instruction pool and collect human assessments to update the ranking, allowing existing data to be fully reused.
- **Counterexamples Feeding Back to Training**: The counterexamples found via MAD competition (specific samples where one model loses to another) are not only used for evaluation but can also serve as adversarial samples to train stronger models (e.g., adversarial fine-tuning).

## Limitations & Future Work

- When the number of evaluated LLMs is large (e.g., 50+), pairwise MAD competition still requires substantial human effort ($\binom{N}{2} \times K$ matches); hierarchical strategies like coarse-to-fine filtering can be considered.
- The instruction pool is automatically generated using evolutionary methods, which may present a distribution drift/bias compared to the training data of some LLMs.
- Currently, only 8 LLMs and 4 scenarios are evaluated as a proof of concept; validation on a larger scale and in more scenarios remains to be done.
- The human judges are 13 computer science graduate students, which lacks demographic diversity and may differ from broader user preferences.
- The MAD philosophy can be combined with LLM-as-a-judge—using MAD to select samples and LLM to judge quality, striking another balance between cost and accuracy.

## Related Work & Insights

- **vs Chatbot Arena**: Chatbot Arena allows users to have free-form conversations and vote, yielding broad coverage but requiring massive data. MAD-Eval automatically selects the most discriminative test samples, achieving similar rankings at a fraction of the human cost.
- **vs AlpacaEval 2.0**: AlpacaEval uses a fixed instruction set and LLM-as-a-judge, which suffers from judge bias. MAD-Eval uses adaptively selected instructions and human judgments, producing more reliable results but requiring human effort.
- **vs Dynabench**: Dynabench requires users to manually submit counterexamples to expose model weaknesses, whereas MAD-Eval automates the process of counterexample discovery with theoretical guarantees.
- **vs KL/Cross-entropy Sampling (Boubdir et al., 2023)**: These methods require token-level log probabilities (inapplicable to some API-based models) and lack diversity control. MAD works using only the response text.

## Rating

- Novelty: ⭐⭐⭐⭐ Transferring the MAD competition principle from the vision domain to LLM evaluation is an ingenious cross-domain innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 4 scenarios $\times$ 8 LLMs, comparison with 3 existing leaderboards, comparison of 4 sampling strategies, and ablation of various similarity metrics.
- Writing Quality: ⭐⭐⭐⭐ Clear logic, concise mathematical formulation, and rich case studies.
- Value: ⭐⭐⭐⭐⭐ Provides a practical and highly efficient solution for LLM evaluation. Code is open-source, and has high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] LLMs instead of Human Judges? A Large Scale Empirical Study across 20 NLP Evaluation Tasks](llm_vs_human_judges_study.md)
- [\[ACL 2025\] Pragmatics in the Era of Large Language Models: A Survey on Datasets, Evaluation, Opportunities and Challenges](pragmatics_survey.md)
- [\[ACL 2025\] RoCoFT: Efficient Finetuning of Large Language Models with Row-Column Updates](rocoft_efficient_finetuning_of_large_language_models_with_row-column_updates.md)
- [\[ACL 2025\] AXIS: Efficient Human-Agent-Computer Interaction with API-First LLM-Based Agents](axis_efficient_human-agent-computer_interaction_with_api-first_llm-based_agents.md)
- [\[ACL 2025\] Logical Forms Complement Probability in Understanding Language Model (and Human) Performance](logical_forms_complement_probability_in_understanding_language_model_and_human_p.md)

</div>

<!-- RELATED:END -->
