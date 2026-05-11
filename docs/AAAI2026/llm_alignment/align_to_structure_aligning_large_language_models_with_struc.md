---
title: >-
  [Paper Note] Align to Structure: Aligning Large Language Models with Structural Information
description: >-
  [AAAI 2026][LLM Alignment][Structural Alignment] This paper proposes Structural Alignment, a method that integrates linguistic discourse structure frameworks—surface-level text structure scoring and an RST-based discours…
tags:
  - "AAAI 2026"
  - "LLM Alignment"
  - "Structural Alignment"
  - "Discourse Structure"
  - "PPO Reinforcement Learning"
  - "Long-form Text Generation"
  - "Dense Reward"
date: 2026-05-08
content_hash: 735c3e74ec9982b1
---

# Align to Structure: Aligning Large Language Models with Structural Information

**Conference**: AAAI 2026
**arXiv**: [2504.03622v2](https://arxiv.org/abs/2504.03622v2)
**Code**: [https://github.com/minnesotanlp/struct_align](https://github.com/minnesotanlp/struct_align)
**Area**: LLM Alignment
**Keywords**: Structural Alignment, Discourse Structure, PPO Reinforcement Learning, Long-form Text Generation, Dense Reward

## TL;DR
This paper proposes Structural Alignment, a method that integrates linguistic discourse structure frameworks—surface-level text structure scoring and an RST-based discourse motif classifier—into PPO reinforcement learning training, and introduces a discourse motif-based dense reward mechanism. This enables LLMs to generate more coherent, human-like long-form text, outperforming standard RLHF models on academic essay writing and long document summarization tasks.

## Background & Motivation
- Existing LLM alignment methods (RLHF/RLAIF) primarily focus on "helpfulness" and "harmlessness"—i.e., *what* humans prefer—while neglecting *how* humans organize coherent discourse structure.
- Human writers naturally employ textual structures such as "problem-solution," "causal," and "descriptive" patterns to maintain local coherence and logical flow, a form of hierarchical planning that LLMs lack.
- Long-form text generation (>1,000 tokens) suffers from topic drift, shallow cohesion, and insufficient rhetorical consistency; sparse episodic rewards in standard RLHF training lead to difficult credit assignment and training instability.
- Evaluation of long-form text remains challenging due to the absence of large-scale annotated datasets and the inherently subjective nature of quality assessment.

## Core Problem
How can LLMs be aligned not only at the content level with human preferences, but also at the **discourse structure level** with human writing conventions, so as to produce logically coherent and well-organized long-form text? Specific challenges include: (1) how to quantify discourse structure quality and convert it into a trainable reward signal; and (2) how to address the instability caused by sparse rewards in long-sequence PPO training.

## Method

### Overall Architecture
Within an RLAIF+PPO framework using Qwen2-1.5B-Instruct as the policy model, two complementary structural alignment pathways are explored:
1. **Surface-Level Text Structure Scoring (SA_S)**: An external large model (Qwen2-72B-Instruct-AWQ) serves as an evaluator to score surface-level structure of generated text.
2. **Graph-Level Discourse Structure Scoring (SA_G)**: Hierarchical discourse motifs are extracted via RST discourse analysis, and an authorship classifier is trained as a reward model.

In addition, a dense reward mechanism and length-penalty normalization are introduced to enhance training stability for long-sequence PPO.

### Key Designs
1. **Surface-Level Text Structure Evaluator**: Qwen2-72B is used as an RLAIF judge to score generated text on a 0–5 scale across three dimensions: (a) *Logical Flow and Structure*—assessing whether ideas progress logically and the overall organization is coherent; (b) *Hierarchical Organization*—evaluating whether content is structured from general to specific; (c) *Balance and Emphasis*—examining whether key points are highlighted and different arguments are covered proportionally. The final reward is the mean of the three scores.

2. **Graph-Level Discourse Structure Scorer**: Following Kim et al. (2024), a DMRST parser converts text into RST discourse trees, which are transformed into recursive hypergraphs from which discourse motifs—structural feature patterns that distinguish human writing from machine-generated text—are extracted. Motif distributions are concatenated with Longformer's [CLS] embeddings and fed into a binary classifier to determine whether text "reads like human writing." The classifier output serves as the reward signal. Since the RST parser is limited to approximately 512 tokens, long texts are segmented into paragraph-level chunks (400–512 tokens each), parsed independently on non-overlapping segments, and motif distributions are then aggregated.

3. **Dense Reward Shaping**: Standard PPO provides only a single episodic reward at the end of a sequence, making credit assignment difficult for long sequences. In addition to the episodic reward, tokens belonging to "human-distinctive discourse motifs" receive an additional reward of $\frac{1}{2 \cdot \text{num\_tokens}}$. Distinctive motifs are identified at the corpus level using MF-IDF (Motif Frequency-Inverse Document Frequency), selecting motifs exceeding a one-standard-deviation threshold. Rewards are computed at the granularity of EDUs (Elementary Discourse Units, averaging approximately 20 subword tokens).

4. **Length-Penalty Normalization**: To address LLMs' tendency to disregard target length constraints, a proportional penalty is applied when generated length falls short: $S_n = S_o \times [1 - \alpha \times \max(0, \frac{L_d - L_r}{L_d})]$. Target generation length is incrementally increased throughout training.

### Loss & Training
- Standard PPO objective (clipped surrogate objective) with KL coefficient 0.03.
- Training configuration: per-device batch size 2, gradient accumulation 4, local rollout forward batch size 12.
- Generated sequence length: 400–2K tokens, with target length increasing each epoch.
- SA_S uses 8 A100 GPUs running an SGLang-served Qwen2-72B-AWQ evaluator instance, with an additional 8 A100 GPUs for PPO training.
- Two-stage training variants are also explored: SA_S followed by SA_G (SA_S→G) and the reverse (SA_G→S), leveraging the complementarity of the two alignment approaches.

## Key Experimental Results

| Task / Dataset | Metric | SA_G (Graph-level) | SA_S (Surface-level) | Base (Qwen2-1.5B) | RLHF_OA | Notes |
|---|---|---|---|---|---|---|
| GovReport Summarization | ROUGE-1 | **55.86** | 55.45 | 53.21 | 53.25 | Graph-level best |
| GovReport Summarization | ROUGE-2 | **21.72** | 21.43 | 20.13 | 20.25 | +1.59 vs. Base |
| GovReport Summarization | ROUGE-L | **52.81** | 52.30 | 50.39 | 50.47 | +2.42 vs. Base |
| Essay Generation | GPT-4o Pairwise | SA_G > SA_S > RLHF > Base | — | — | — | Graph-level consistently wins pairwise evaluation |

- Text Structure Evaluator validation: Pearson correlations with human scores—coherence 0.47, organization 0.44, balance 0.39, purpose 0.37, variability 0.39; MSE 0.79 (on a 0–4 scale).

### Ablation Study
- **Two-stage training**: SA_S→G and SA_G→S marginally outperform individual SA_S or SA_G, though the improvement is modest relative to the additional computational cost.
- **Complementarity of surface and graph-level scoring**: The two scoring methods show very low correlation (scatter plots reveal near-zero correlation), indicating that they capture distinct aspects of structural information and are highly complementary.
- **Limitations of standard RLHF**: RLHF training with the OpenAssistant reward model not only fails to improve discourse structure quality, but the proportion of human-distinctive motifs in generated text actually declines.
- **Effect of dense rewards**: Under graph-level alignment training, the proportion of human-distinctive motifs stabilizes at a consistently high level after approximately 50 batch steps, whereas standard PPO shows a continuous decline.
- **Effect of surface-level alignment**: Generated texts exhibit more discourse connectives (e.g., *therefore*, *however*, *moreover*), clearer paragraph headings (e.g., *Introduction*, *Claim 1*, *Conclusion*), and a significant increase in Joint and Hyperedge hierarchical relation motifs.

## Highlights & Insights
- **Novel alignment objective**: Aligning to human *writing structure* rather than human *preferences* introduces computational discourse linguistics into LLM alignment, representing a genuinely fresh perspective.
- **Elegant dense reward design**: Token-level reward shaping based on discourse motifs leverages the natural granularity of linguistic structure to address the credit assignment problem in long-sequence PPO—simple yet effective.
- **Two complementary scoring methods**: Surface-level scoring enhances readability and explicit structuring, while graph-level scoring reinforces deep coherence and rhetorical complexity; the two are weakly correlated and mutually complementary.
- **Length-penalty normalization**: A concise solution to the problem of LLMs failing to adhere to target length constraints.
- **RST segmentation extension**: Paragraph-level segmentation and motif aggregation extend the 512-token-limited RST parser to handle long-form texts.

## Limitations & Future Work
- Evaluation is limited to formal essay writing and summarization tasks; creative writing, dialogue, and other informal genres—where discourse structure is more loosely defined and harder to formalize—are not addressed.
- The policy model is restricted to Qwen2 at 1.5B parameters; effectiveness on larger models remains unverified due to computational constraints.
- The RST parser's ~512-token input limit requires segmented processing, which may disrupt global coherence patterns.
- Only the RST discourse framework is employed; alternative formalisms such as SDRT may capture more nuanced relational patterns.
- Essay quality evaluation relies on GPT-4o as a judge; large-scale human evaluation is absent.
- Multi-reward alignment strategies are limited to simple two-stage sequential combinations; more advanced multi-reward approaches (e.g., Pareto optimization) may yield further improvements.

## Related Work & Insights
- **vs. Standard RLHF (OpenAssistant RM)**: Standard RLHF optimizes for generic human preferences without reinforcing discourse organization; in the experiments, the RLHF model shows negligible differences from the base model on essay generation and summarization, and the proportion of human-distinctive motifs even decreases.
- **vs. Kim et al. (2024) Discourse Motifs**: That work (ACL 2024) demonstrates that discourse motifs can distinguish human writing from AI-generated text; this paper innovatively repurposes that detection tool as a reward signal and dense reward source for RL training.
- **vs. Dense Reward methods (Chan et al., 2024; Cao et al., 2024)**: Those works propose general-purpose dense reward methods; this paper's contribution lies in leveraging linguistic structure (discourse motifs + EDU granularity) to provide domain-aware dense rewards that are better suited to long-form text generation.

## Takeaways
- The discourse structure alignment paradigm is transferable to other long-form output scenarios, including academic paper writing, report generation, and code documentation.
- Dense reward design exploiting the task's inherent structural features (discourse trees) exemplifies a broadly applicable "domain-aware reward shaping" philosophy that can be extended to other tasks with intrinsic structural information.
- The low correlation between the two complementary scoring methods suggests that selecting orthogonal reward dimensions in multi-objective alignment may be more valuable than using multiple homogeneous rewards.
- Repurposing a classifier originally designed to "detect AI-generated text" as a signal for "training AI to generate more human-like text" is a noteworthy reverse-application strategy.

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing computational discourse linguistics into LLM alignment is a novel perspective, though the underlying framework (PPO+RLAIF) is relatively standard.
- Experimental Thoroughness: ⭐⭐⭐ Only a 1.5B model is used; evaluation covers only essay writing and summarization; large-scale human evaluation is absent; downstream task diversity is insufficient.
- Writing Quality: ⭐⭐⭐⭐ The paper is clearly structured, method descriptions are detailed, and the linguistic motivation is thoroughly articulated.
- Value: ⭐⭐⭐⭐ Proposes a new paradigm shifting from "preference alignment" to "structure alignment"; the dense reward design is reusable, though practical impact is constrained by model scale and scenario coverage.

## Rating
- Novelty: ⭐⭐⭐⭐ Incorporating RST discourse structure into LLM alignment represents cross-domain innovation; the structure-aware preference data construction approach is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two tasks (essay writing and summarization) with comprehensive coverage via both automatic and human evaluation.
- Writing Quality: ⭐⭐⭐⭐ The integration of linguistic theory and NLP methodology is clearly articulated.
- Value to Me: ⭐⭐⭐ Structure-aware writing alignment offers reference value, though the application scope is relatively narrow.

## Additional Notes
- This work offers inspiration for aligning structured output tasks (e.g., JSON/XML/code generation), and could be combined with tree-structure constraints for finer-grained format control.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Exploring the Effects of Alignment on Numerical Bias in Large Language Models](exploring_the_effects_of_alignment_on_numerical_bias_in_large_language_models.md)
- [\[AAAI 2026\] Margin-aware Preference Optimization for Aligning Diffusion Models without Reference](margin-aware_preference_optimization_for_aligning_diffusion_models_without_refer.md)
- [\[AAAI 2026\] BiasJailbreak: Analyzing Ethical Biases and Jailbreak Vulnerabilities in Large Language Models](biasjailbreakanalyzing_ethical_biases_and_jailbreak_vulnerabilities_in_large_lan.md)
- [\[AAAI 2026\] W2S-AlignTree: Weak-to-Strong Inference-Time Alignment for Large Language Models via Monte Carlo Tree Search](w2s-aligntree_weak-to-strong_inference-time_alignment_for_large_language_models_.md)
- [\[AAAI 2026\] Reducing the Scope of Language Models](reducing_the_scope_of_language_models.md)

</div>

<!-- RELATED:END -->
