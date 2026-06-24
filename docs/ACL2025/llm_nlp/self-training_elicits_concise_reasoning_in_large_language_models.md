---
title: >-
  [Paper Note] Self-Training Elicits Concise Reasoning in Large Language Models
description: >-
  [ACL 2025][LLM (Other)][Concise Reasoning] Discovers that LLM output distributions naturally contain concise reasoning paths and proposes the FS-BoN (Few-Shot conditioning + Best-of-N sampling) self-training framework. By filtering short and correct reasoning samples from the model's own distribution for fine-tuning, the method achieves an average of 30% token reduction across five model families on GSM8K and MATH without sacrificing accuracy…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Concise Reasoning"
  - "Self-Training"
  - "Best-of-N Sampling"
  - "Few-Shot Conditioning"
  - "Token Efficiency"
date: 2026-05-08
content_hash: f359a30219794112
---

# Self-Training Elicits Concise Reasoning in Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2502.20122](https://arxiv.org/abs/2502.20122)  
**Code**: [https://github.com/TergelMunkhbat/concise-reasoning](https://github.com/TergelMunkhbat/concise-reasoning)  
**Area**: LLM/NLP  
**Keywords**: Concise Reasoning, Self-Training, Best-of-N Sampling, Few-Shot Conditioning, Token Efficiency

## TL;DR
Discovers that LLM output distributions naturally contain concise reasoning paths and proposes the FS-BoN (Few-Shot conditioning + Best-of-N sampling) self-training framework. By filtering short and correct reasoning samples from the model's own distribution for fine-tuning, the method achieves an average of 30% token reduction across five model families on GSM8K and MATH without sacrificing accuracy, delivering 2.4 times the efficiency of the prior method, Rational Metareasoning.

## Background & Motivation

**Background**: CoT reasoning has become standard for LLMs solving complex reasoning tasks, but raw reasoning chains are naturally verbose, containing redundant explanations, repetitive step descriptions, and irrelevant contextual confirmations. The number of reasoning tokens is roughly proportional to inference latency, directly impacting deployment costs.

**Limitations of Prior Work**: Existing zero-shot prompting methods (e.g., "Be Concise", "Fixed Budget") attempt to directly instruct models to generate shorter reasoning, but yield inconsistent results. For instance, while the "Fixed Budget" method reduces length by 32.2%, it suffers a 10.1% accuracy drop. More critically, these methods are almost entirely ineffective on math-specialized models (such as Qwen2.5-Math), indicating that zero-shot prompting cannot reliably manipulate the internal representations of models that have undergone extensive post-training.

**Key Challenge**: The CoT capability of LLMs originates from procedural knowledge in pre-training data, which is not optimized for conciseness. Additionally, post-training stages like RLHF/DPO do not encourage concise reasoning (or "thinking" models are even reinforced to use more tokens), making the default behavior of models naturally verbose.

**Goal**: How to reliably prune redundant tokens from LLM reasoning chains without sacrificing accuracy? Sub-questions include: (a) Do LLMs possess the latent capability for concise reasoning? (b) How to efficiently extract concise reasoning samples? (c) How to internalize the concise reasoning capability during inference as the model's default behavior?

**Key Insight**: The authors sampled 16 reasoning paths per question on GSM8K across multiple models and found that a significant portion of the normalized length distribution of correct paths falls below the average length. For example, in DeepSeekMath-7B, 8.37% of the correct solutions have a length of less than half the average. This indicates that **concise reasoning already exists within the model's output distribution**, but default decoding fails to select them.

**Core Idea**: Utilizing Few-shot conditioning guidance combined with Best-of-N sampling to extract concise and correct reasoning paths from the model's own distribution, and then leveraging standard fine-tuning to internalize this capability, achieving concise reasoning with zero inference overhead.

## Method

### Overall Architecture

Input: A training dataset of the target task (e.g., GSM8K/MATH training set) + a pre-trained LLM.  
Output: A fine-tuned LLM that generates concise reasoning chains by default.

The pipeline consists of three stages:
1. **Data Generation**: For each question in the training set, multiple reasoning paths are sampled using the FS-BoN strategy, and the shortest correct path is selected.
2. **Sample Augmentation**: The results of FS-BoN sampling and naive BoN sampling are merged to ensure correct solution coverage even for hard questions.
3. **Standard Fine-Tuning**: Standard SFT is applied to the model using the filtered concise reasoning paths to internalize the conciseness.

### Key Designs

1. **Naive Best-of-N Sampling (Naive BoN)**:

    - **Function**: Samples $N$ reasoning paths for each training question and selects the shortest correct path as the training sample.
    - **Mechanism**: Leverages the stochasticity of model outputs to collect samples from the left tail (shorter end) of the length distribution. The key design is **per-question selection** instead of global selection of the shortest, as hard problems naturally require longer reasoning and global selection would discard supervision signals for difficult questions.
    - **Design Motivation**: Directly bypasses external data by utilizing the model's inherent concise reasoning ability. However, it suffers from a logarithmic-linear decay in sampling efficiency—as $N$ doubles, the marginal return in length reduction diminishes.
    - Difference from prior work: Rational Metareasoning (De Sabbata et al., 2024) also employs BoN but adds a reward function to balance efficiency and accuracy along with iterative training; experiments demonstrate that these additional designs yield no significant gains.

2. **Few-shot Conditioning (FS)**:

    - **Function**: Guarantees shorter reasoning generation by conditioning the model with 8 concise reasoning examples as a few-shot prompt during sampling.
    - **Mechanism**: Distorts the length distribution of the outputs through examples utilizing the in-context learning capability of LLMs. Three example sources are considered: human annotations (FS-Human, from the CoT examples in Wei et al. 2022), GPT-4o generated (FS-GPT4o), and self-generated (FS-Self).
    - **Design Motivation**: The length-reduction effect of few-shot conditioning far exceeds that of BoN—a single sample of FS-Human outperforms BoN with $N=256$, boosting sampling efficiency by several orders of magnitude. This is because few-shot conditioning directly shifts the entire length distribution at the probabilistic level, whereas BoN merely samples from the tail of the original distribution.
    - **Key Findings**: FS-GPT4o is optimal for maintaining accuracy, while FS-Human yields the greatest length reduction but experiences a slight drop in accuracy.

3. **Few-shot Conditioned BoN (FS-BoN)**:

    - **Function**: Overlays BoN sampling on top of few-shot conditioning for double length reduction.
    - **Mechanism**: The reduction effects of FS and BoN are **largely independent and cumulative**—FS is responsible for shifting the center of the distribution globally, while BoN is used to select the shortest sample from the shifted distribution. GPT-4o examples are used as the FS prompt (FS-GPT4o-BoN) because they maintain accuracy best.
    - Difference from usage during direct inference: Directly using BoN+FS at test time requires repeated sampling and long prompts, which incurs enormous computational cost (defeating the goal of cost reduction); whereas self-training internalizes the benefits into the model parameters, achieving zero extra overhead at inference time.

4. **Sample Augmentation**:

    - **Function**: For FS and FS-BoN methods, an additional $N$ paths are sampled from naive BoN (without few-shot prompts) and merged with the candidates of FS/FS-BoN to select the shortest correct path.
    - **Mechanism**: Few-shot examples have limited adaptability—they might introduce unnecessary steps for extremely simple questions and suppress necessary reasoning depth for highly complex ones. Augmentation samples generated from the original distribution cover hard questions more effectively.
    - **Design Motivation**: Experiments reveal a significant increase in accuracy post-augmentation, while the length reduction still outperforms naive BoN and RM.

### Loss & Training

- Employs standard SFT (supervised fine-tuning) loss to perform language modeling on the filtered concise reasoning paths.
- One training sample (the shortest correct path) per question.
- Highly economical training costs, negligible compared to the data generation phase.
- Generation budget allocation: Naive BoN samples 16 paths per question; FS samples 1 path + 16 augmentation paths per question; FS-BoN samples 16 paths + 16 augmentation paths per question (Budget-Matched setting: 8 paths each).

## Key Experimental Results

### Main Results

Average results over five model families (Llama-3.2-3B, Gemma-2-2B, Qwen2.5-3B, Qwen2.5-Math-1.5B, DeepSeekMath-7B) on GSM8K and MATH:

| Method | GSM8K Acc (%) | GSM8K Len (tokens) | MATH Acc (%) | MATH Len (tokens) | Relative Accuracy | Relative Length |
|------|------|------|------|------|------|------|
| Baseline (zero-shot) | 78.06 | 241.87 | 46.40 | 480.37 | 100% | 100% |
| Be Concise | 77.98 | 214.87 | 47.76 | 446.09 | 99.9% | 88.5% |
| Fixed Budget | — | — | — | — | 89.9% | 67.8% |
| Naive BoN (N=16) | 77.12 | 214.22 | 47.64 | 433.26 | 98.8% | 87.2% |
| Rational Metareasoning | 76.15 | 207.49 | 47.56 | 432.56 | 97.2% | 84.9% |
| **FS-GPT4o** | **78.07** | **175.54** | **47.36** | **421.21** | **99.9%** | **73.2%** |
| **FS-GPT4o-BoN** | **75.88** | **153.38** | **47.36** | **364.33** | **97.0%** | **64.3%** |
| FS-GPT4o-BoN (Budget-Matched) | 76.24 | 160.59 | 47.52 | 384.43 | 97.4% | 67.2% |

### Ablation Study

| Configuration | Relative Length (GSM8K) | Relative Accuracy (GSM8K) | Description |
|------|---------|---------|------|
| FS-GPT4o-BoN (Full) | 64.25% | 97.00% | Maximum reduction, slight accuracy drop |
| FS-GPT4o (w/o BoN) | 73.15% | 99.94% | Virtually unchanged accuracy, slightly weaker reduction |
| Naive BoN (w/o FS) | 87.17% | 98.79% | Limited reduction |
| FS-GPT4o-BoN w/o Augmentation | Shorter | Lower | Augmentation improves accuracy |
| Direct Answer (No CoT) | 1.36% | 24.88% | Accuracy collapses |
| Human CoT (Fine-tuning with external data) | 54.95% | 83.82% | Shorter length, but substantial drop in accuracy |

### Key Findings
- **FS-BoN yields the greatest contribution**: FS-GPT4o-BoN achieves a 64.3% relative length (35.7% reduction), which is nearly three times more efficient than Naive BoN (12.8% reduction).
- **Self-training vs. External data**: Fine-tuning with external data (Human CoT / GPT-4o CoT) significantly curtails length but leads to a severe degradation in accuracy (-16% to -24%), falling below the Pareto frontier. Self-training preserves reasoning capabilities much better because the training data originates from the model's own distribution.
- **Adaptive length allocation**: Across 5 difficulty levels in MATH, simple questions (Level 1-2) undergo a 20%-40% reduction, whereas difficult questions (Level 5) are reduced by only about 5%. This indicates that the model learns to adaptively allocate its token budget based on question difficulty.
- **Consistency across model scales**: Scaling experiments on Llama-3-1B/3B/8B show that larger models yield greater token reductions, with FS-GPT4o-BoN consistently being the most effective method across all scales.
- **Cross-domain generalization**: The method is also effective on business, chemistry, and physics reasoning tasks in MMLU-Pro—boosting average accuracy by 16.51% and reducing length by 26.82%.
- **Real-world efficiency gains**: Wall-clock latency drops by 15.4%-52.9%, and memory usage decreases by 2.5%-6.3%.

## Highlights & Insights
- **"Concise reasoning is a latent capability, not a missing one"**: Models already possess the capability for concise reasoning, but default sampling simply fails to retrieve it. This observation is similar to the Superficial Alignment Hypothesis in the alignment field—capabilities already exist within the model and only require lightweight fine-tuning to unlock. This implies that we do not need to teach models new capabilities from scratch, but rather steer them to utilize existing ones.
- **Independent and cumulative effect of FS and BoN**: Few-shot conditioning shifts the mean of the entire output length distribution, while BoN sampling selects the best from the tail. Operating along different dimensions makes their effects approximately additive. This paradigm of "distribution shift + tail sampling" can be transferred to other generation control scenarios (e.g., controlling style, toxicity, etc.).
- **Elegance of self-training**: Converts multi-sampling overhead at inference time into a one-of-off training cost. Post-training, the model directly produces concise reasoning with zero extra inference expense. This mindset of "investing training compute to save inference costs" has direct utility for optimizing thinking models like DeepSeek-R1 and o1.
- **Per-question vs. Global selection**: Selecting the shortest correct path per question instead of globally during BoN sampling ensures that hard questions retain training signals. This subtle design decision reflects meticulous consideration of data quality.

## Limitations & Future Work
- **Limited scope of tasks**: Main experiments reside only in mathematical reasoning (GSM8K/MATH). Although preliminary verification is conducted on MMLU-Pro, domains like natural language inference and code generation remain unexplored. In particular, for tasks that require multi-step implicit reasoning (e.g., commonsense reasoning), whether conciseness sacrifices crucial information remains unclear.
- **Costly BoN data generation**: Despite zero inference overhead, generating training data requires sampling 16 to 32 paths per question, presenting a substantial computational burden for massive training sets.
- **Concise $\neq$ Interpretable**: The authors do not discuss the impact of concise reasoning on interpretability. Shorter reasoning chains might omit explicit explanations of intermediate steps, making it harder for humans to parse the model's reasoning process.
- **Omission of reinforcement learning**: Relying solely on standard SFT overlooks potential benefits from RL-based training (e.g., PPO/DPO with length penalties), which might yield more extensive optimizations.
- **No evaluation on thinking models**: Though the paper discusses potential value for thinking models (such as o1/R1), no empirical validation is performed. Given that these models' internal reasoning easily scales up to thousands of tokens, the benefits of conciseness could be far more profound.
- **Future directions**: (1) Employing early termination on erroneous reasoning paths to further prune invalid tokens; (2) extending the framework to multi-task settings to bypass per-task fine-tuning; (3) incorporating RL training with a length-punishing reward signal.

## Related Work & Insights
- **vs. Rational Metareasoning (De Sabbata et al., 2024)**: RM similarly relies on BoN self-training, but introduces an extra reward function to balance efficiency and accuracy, along with iterative training. Experiments demonstrate that these additional designs yield no substantial benefits (reflected in Table 2, where RM and Naive BoN perform comparably), showcasing that simply selecting the shortest correct path is sufficient. Ours (FS-BoN) achieves 2.4 times the efficiency of RM in length reduction.
- **vs. Token-Budget-Aware (Han et al., 2024) / Fixed Budget (Nayab et al., 2024)**: These methods control output length via zero-shot prompts, but suffer a severe tradeoff between accuracy and length. Ours bypasses this issue entirely through self-training.
- **vs. Thinking paradigm of DeepSeek-R1 / o1**: While thinking models trade more tokens for higher accuracy, this work operates in reverse—using fewer tokens without sacrificing accuracy. The two directions are complementary and could be unified in the future (e.g., using thinking models to ensure quality first, and then compressing them via self-training).
- **Connections to RLHF/DPO**: The self-training in this paper is essentially a simplified best-of-N distillation, analogous to rejection sampling fine-tuning in RLHF. The variance lies in the "reward" formulation here: correctness and minimal length, rather than human preference.

## Rating
- Novelty: ⭐⭐⭐⭐ The discovery that "concise reasoning is already embedded within the model's output distribution" is insightful; though individual components of FS-BoN are simple, their combination yields remarkable efficacy.
- Minor modifications to ensure rigorous testing.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely solid evaluation involving 5 model families $\times$ 2 datasets + 3 cross-domain datasets + scaling laws experiments + detailed ablations + wall-clock latency measurement.
- Writing Quality: ⭐⭐⭐⭐ Clear reasoning flow from observation to methodology to experiments, coupled with informative visualizations.
- Value: ⭐⭐⭐⭐ Highly practical for LLM inference efficiency, especially providing great inspiration for the efficiency optimization of thinking models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Cool-Fusion: Fuse Large Language Models without Training](cool-fusion_fuse_large_language_models_without_training.md)
- [\[ACL 2025\] Disentangling Memory and Reasoning Ability in Large Language Models](disentangle_memory_reasoning.md)
- [\[ACL 2025\] The Role of Deductive and Inductive Reasoning in Large Language Models](the_role_of_deductive_and_inductive_reasoning_in_large_language_models.md)
- [\[ACL 2025\] ExpliCa: Evaluating Explicit Causal Reasoning in Large Language Models](explica_evaluating_explicit_causal_reasoning_in_large_language_models.md)
- [\[ACL 2025\] GradOT: Training-free Gradient-preserving Offsite-tuning for Large Language Models](gradot_offsite_tuning.md)

</div>

<!-- RELATED:END -->
