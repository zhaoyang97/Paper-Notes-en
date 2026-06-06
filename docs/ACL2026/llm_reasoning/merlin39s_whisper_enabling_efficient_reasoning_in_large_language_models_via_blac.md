---
title: >-
  [Paper Note] Merlin's Whisper: Enabling Efficient Reasoning in Large Language Models via Black-box Persuasive Prompting
description: >-
  [ACL2026][LLM Reasoning][Reasoning Compression] Whisper models the problem of "reducing thinking without sacrificing accuracy" in Large Reasoning Models (LRMs) as black-box persuasive prompting. By automatically generati…
tags:
  - "ACL2026"
  - "LLM Reasoning"
  - "Reasoning Compression"
  - "Black-box Prompting"
  - "Persuasive Prompting"
  - "Overthinking"
  - "LRM Efficiency"
date: 2026-05-08
content_hash: dbc72a1e75c8bacf
---

# Merlin's Whisper: Enabling Efficient Reasoning in Large Language Models via Black-box Persuasive Prompting

**Conference**: ACL2026  
**arXiv**: [2510.10528](https://arxiv.org/abs/2510.10528)  
**Code**: https://github.com/hemingkx/Whisper  
**Area**: LLM Reasoning Efficiency / Prompt Optimization  
**Keywords**: Reasoning Compression, Black-box Prompting, Persuasive Prompting, Overthinking, LRM Efficiency  

## TL;DR
Whisper models the problem of "reducing thinking without sacrificing accuracy" in Large Reasoning Models (LRMs) as black-box persuasive prompting. By automatically generating and iteratively filtering prompt suffixes from multiple persuasive perspectives, it significantly reduces output tokens on Qwen3, DeepSeek-R1-Distill, and Claude/Gemini APIs while maintaining reasoning accuracy.

## Background & Motivation
**Background**: LRMs such as DeepSeek-R1, Qwen3, and o1 improve performance in mathematical and complex tasks through long Chain-of-Thought (CoT). However, longer reasoning trajectories lead to higher latency, increased VRAM usage for KV cache, and higher API costs.

**Limitations of Prior Work**: Training-based compression requires additional SFT or RL, which is costly and may harm cross-domain generalization. White-box inference interventions require access to internal model states, making them inapplicable to closed-source APIs. Simple prompts like "Be concise." are easy to deploy but offer limited compression or lead to decreased accuracy.

**Key Challenge**: LRMs themselves may possess the potential for "concise reasoning," but their default behavior tends toward overthinking. The issue is not that the models cannot answer briefly, but that users lack effective black-box interaction methods to modify this default strategy.

**Goal**: The authors aim to reduce the average output length of LRMs while maintaining accuracy through automatically generated prompt suffixes, without training models, accessing internal activations, or modifying reasoning engines.

**Key Insight**: The paper draws inspiration from persuasive prompting. While this technique is traditionally used to study jailbreaking or modifying model behavior, this work repurposes it for a positive goal: persuading the model to adopt more compact reasoning expressions.

**Core Idea**: "High-quality concise reasoning prompts" are treated as searchable black-box suffixes. Multiple persuasive perspectives are used to generate candidates, which are then ranked by accuracy constraints and output length on a development set for iterative optimization.

## Method
The input to Whisper is not the model weights but an initial task instruction, a black-box model, and a development set. It automatically generates multiple prompt suffixes and appends them to the original instruction. Each candidate suffix is evaluated based on accuracy and average token count. Candidates that cause significant accuracy drops are discarded, while the remaining ones are ranked by length. The top-$k$ shortest suffixes proceed to the next round of execution. The suffix that yields an acceptable accuracy and the shortest output on the development set is selected for deployment.

### Overall Architecture
Given a model $M$, an original instruction $P_{ins}$, and a development set $D'$, Whisper seeks a suffix $P_{adv}$ that minimizes the average response length $L_{avg}$ while ensuring the average accuracy $ACC_{avg}$ stays above a tolerance threshold. GPT-4o serves as the prompt generator, producing 10 candidates per persuasive perspective per round. The top-5 are selected as exemplars for the next round, with the process iterating for 3 rounds.

### Key Designs
1. **Multi-perspective Persuasive Prompt Generation**:
	- **Function**: Exploring the prompt space to induce LRM reasoning compression using various psychological and discourse strategies.
	- **Mechanism**: Whisper employs perspectives such as emotional appeal, threat, evidence-based persuasion, role-playing, and detailed instruction. For instance, the evidence perspective might cite research-style arguments that "short explanations are equally effective," while the role-play perspective asks the model to act as an expert requiring extreme brevity.
	- **Design Motivation**: A single "Be concise" instruction is too weak. Multi-perspective generation covers the model's varying sensitivity to authoritative evidence, role constraints, emotional context, and structural requirements.

2. **Candidate Filtering Under Accuracy Constraints**:
	- **Function**: Preventing the model from being compressed into short but incorrect responses.
	- **Mechanism**: For a candidate $P_{adv}^j$, the average length $L_{avg}^j$ and accuracy $ACC_{avg}^j$ are calculated on the development set. If the accuracy drop exceeds the tolerance threshold $\tau$, the candidate is discarded. The remaining candidates are ranked by average length.
	- **Design Motivation**: The true objective of reasoning compression is the efficiency-performance trade-off, not merely reducing tokens. "NoThinking" is very short but results in a massive accuracy drop, thus it is not considered successful.

3. **Iterative Refinement**:
	- **Function**: Allowing the prompt generator to learn from effective suffixes in previous rounds to gradually converge on optimal compression strategies.
	- **Mechanism**: The top-$k$ suffixes from each round serve as exemplars for the next round, where GPT-4o continues to synthesize candidates. Experimental results show that compression gains improve from round one to round three before saturating.
	- **Design Motivation**: Hand-written prompts rarely achieve perfection in one go; feeding effective candidates back to the generator serves as a lightweight prompt evolution within the black-box space.

### Loss & Training
Whisper does not train the target LRM. The optimization goal is a dual-objective selection on the development set: minimize average output length within accuracy constraints. In implementation, the authors randomly sample 100 samples from the PRM800K math split as the PDSet. Inference is performed using vLLM with temperature 0.6 and top-p 0.95, with a maximum generation length of 16,384. For GSM8K and MATH-500, each question is sampled 3 times, while AMC 2023 and AIME 2024 are sampled 8 times.

## Key Experimental Results

### Main Results
| Model | Method | Overall Acc. | Overall Ratio | Representative Change |
|------|------|------|------|------|
| DeepSeek-R1-Distill-LLaMA-8B | Original | 78.5 | 100% | Original long reasoning |
| DeepSeek-R1-Distill-LLaMA-8B | Whisper | 79.0 | 80.3% | Slight accuracy gain, tokens reduced by ~20% |
| DeepSeek-R1-Distill-Qwen-14B | Original | 85.9 | 100% | Original long reasoning |
| DeepSeek-R1-Distill-Qwen-14B | Whisper | 86.3 | 78.0% | Slight accuracy gain, tokens reduced by ~22% |
| Qwen3-14B | Original | 87.9 | 100% | Original long reasoning |
| Qwen3-14B | Whisper | 89.6 | 63.0% | Accuracy increased, tokens reduced by ~37% |

### Ablation Study
| Qwen3-14B Dataset | Original Acc. / Tok. | Whisper Acc. / Tok. | Ratio |
|------|------|------|------|
| GSM8K | 95.9 / 1568 | 96.1 / 440 | 28.1% |
| MATH-500 | 94.5 / 4398 | 95.2 / 2176 | 49.5% |
| AMC 2023 | 95.0 / 6947 | 96.9 / 4019 | 57.9% |
| AIME 2024 | 66.2 / 11375 | 70.0 / 8659 | 76.1% |

### Key Findings
- Whisper is most effective on simpler problems. On GSM8K, the average tokens for Qwen3-14B dropped from 1568 to 440 (nearly 3.6x compression), while accuracy improved from 95.9 to 96.1.
- Effectiveness on closed-source APIs: The paper reports token usage reductions of 46% for Claude-3.7-Sonnet-Thinking and 50% for Gemini-2.5-Pro-Thinking on MATH-500 while maintaining original reasoning performance.
- Out-of-domain results indicate that prompts optimized for the math domain can transfer to GPQA-Diamond and CommonsenseQA. Qwen3-14B achieved a token ratio of 43.8% on GPQA and 41.2% on CommonsenseQA with negligible accuracy loss.
- Different models exhibit sensitivity to different perspectives: Qwen3 series heavily favors evidence-based persuasion, while role-play, instruction, and evidence all appear among top candidates for DeepSeek-R1-Distill-Qwen.
- Iterative refinement contributes significantly: Token reduction for DeepSeek-R1-Distill-Qwen-14B improved from 18% to 22%, and for Qwen3-14B from 32% to 37%.

## Highlights & Insights
- The most interesting aspect of the paper is the pivot of persuasive prompting from "attack/jailbreak" contexts to efficiency optimization. It demonstrates that model behavior can be significantly shaped by linguistic persuasion strategies without weight modification.
- Whisper shows strong applicability to closed-source APIs. Many reasoning efficiency methods only work with open-source models, whereas the black-box suffix search can be directly applied to commercial model calls.
- The results suggest that "conciseness" is not just a simple instruction but a behavioral pattern that the model must be persuaded to believe in and execute consistently. Evidence, roles, and context are more effective than bare instructions at altering the model's default long-reasoning habits.
- Such methods remind us that prompt suffixes can forcefully alter reasoning length and style, implying that production systems need to manage potential conflicts between efficiency prompts and safety/compliance prompts.

## Limitations & Future Work
- Open-source model experiments were primarily focused on Qwen3 and DeepSeek-R1-Distill series, excluding larger models like Qwen3-235B-A22B.
- The set of persuasive perspectives is limited; only a few were tested. A more systematic search of grammatical and discourse strategies might yield stronger compression but could also introduce complex safety risks.
- The primary development set is based on mathematical reasoning. Although out-of-domain results were provided, verification is needed for tasks like coding, legal, and medical domains.
- The method relies on development set evaluations where each candidate requires actual model calls; search costs remain a concern for expensive closed-source APIs.
- Certain "threat" or "emotional" prompts may not be appropriate in a product context; future work should explore more neutral and auditable persuasive patterns.

## Related Work & Insights
- **vs. SFT / RL Length Penalty**: Training-based methods can change the model distribution but require compute and data; Whisper is a plug-and-play black-box method that does not modify weights.
- **vs. DEER / Activation Steering**: White-box methods use internal states to early-stop or compress CoT but are inapplicable to closed APIs; Whisper only requires input-output access.
- **vs. BeConcise / Chain-of-Draft**: Simple short-answer instructions usually yield limited compression or harm accuracy; Whisper finds more robust suffixes through automated search and accuracy constraints.
- **Insight**: Reasoning systems can treat "whether long-form thinking is required" as a controllable strategy. Compressing simple samples via Whisper-style suffixes while retaining long reasoning or using verifiers for difficult samples may be more economical than a universal CoT approach.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The application of persuasive prompting to reasoning efficiency is a fresh perspective; the method itself is a lightweight prompt search.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers open/closed source models, multiple benchmarks, and transfer analysis, though more massive models and task domains could be included.
- Writing Quality: ⭐⭐⭐⭐☆ Problem definition is clear and tables are informative, though some persuasive examples require the reader to judge product acceptability.
- Value: ⭐⭐⭐⭐⭐ Extremely practical for LRM applications sensitive to API costs and latency, especially where model weights cannot be modified.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Inducing Overthink: Hierarchical Genetic Algorithm-based DoS Attack on Black-Box Large Language Reasoning Models](../../ICML2026/llm_reasoning/inducing_overthink_hierarchical_genetic_algorithm-based_dos_attack_on_black-box_.md)
- [\[ACL 2026\] ReProbe: Efficient Test-Time Scaling of Multi-Step Reasoning by Probing Internal States of Large Language Models](reprobe_efficient_test-time_scaling_of_multi-step_reasoning_by_probing_internal_.md)
- [\[ACL 2026\] DRP: Distilled Reasoning Pruning with Skill-aware Step Decomposition for Efficient Large Reasoning Models](drp_distilled_reasoning_pruning_with_skill-aware_step_decomposition_for_efficien.md)
- [\[ICML 2026\] Diagnosing Multi-step Reasoning Failures in Black-box LLMs via Stepwise Confidence Attribution](../../ICML2026/llm_reasoning/diagnosing_multi-step_reasoning_failures_in_black-box_llms_via_stepwise_confiden.md)
- [\[ACL 2026\] SeLaR: Selective Latent Reasoning in Large Language Models](selar_selective_latent_reasoning_in_large_language_models.md)

</div>

<!-- RELATED:END -->
