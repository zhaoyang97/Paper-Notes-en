---
title: >-
  [Paper Note] What Really Improves Mathematical Reasoning: Structured Reasoning Signals Beyond Pure Code
description: >-
  [ICML2026][LLM Reasoning][Data Mixture] This paper demonstrates through controlled experiments with a 10T-token corpus and MoE models trained from scratch that improvements in complex mathematical reasoning are driven by…
tags:
  - "ICML2026"
  - "LLM Reasoning"
  - "Data Mixture"
  - "Mathematical Reasoning"
  - "Code Pre-training"
  - "MoE"
  - "Cognitive Scaffolding"
date: 2026-05-08
content_hash: f03e2a8097f6cc06
---

# What Really Improves Mathematical Reasoning: Structured Reasoning Signals Beyond Pure Code

**Conference**: ICML2026  
**arXiv**: [2605.19762](https://arxiv.org/abs/2605.19762)  
**Code**: Anonymous repository, public URL not provided  
**Area**: LLM Pre-training / Mathematical Reasoning  
**Keywords**: Data Mixture, Mathematical Reasoning, Code Pre-training, MoE, Cognitive Scaffolding  

## TL;DR
This paper demonstrates through controlled experiments with a 10T-token corpus and MoE models trained from scratch that improvements in complex mathematical reasoning are driven by cross-domain structured reasoning signals—specifically "cognitive scaffolds" that explicitly expose intermediate steps in mathematical corpora—rather than pure executable code itself.

## Background & Motivation
**Background**: Pre-training corpora for modern general-purpose LLMs typically contain a significant proportion of code. Empirical conclusions often suggest that the strict syntax, control flow, and algorithmic structure of code not only improve programming ability but also spill over into mathematical, logical, and scientific reasoning. Another related direction is data mixing and selection, which involves allocating data across different domains (Web, Code, Math, Wikipedia, Books, etc.) under a fixed token budget.

**Limitations of Prior Work**: Many previous studies treat "code" as a coarse-grained entity, often grouping executable code, notebooks, Markdown, HTML/CSS, problem-solving text, and mathematical derivations containing code snippets into the same category. Conclusions that "code improves reasoning" might conflate two different signals: pure programming syntax/executable programs and cross-domain reasoning trajectories where natural language, mathematical symbols, and procedural structures are intertwined.

**Key Challenge**: Under a fixed training budget, adding data from one domain is not a cost-free gain; it displaces tokens from other domains. Pure code might enhance programming but reduce the model's exposure to knowledge-dense or mathematically complex derivations. Mathematical data might enhance competitive programming but weaken comprehensive reasoning tasks. The question shifts from "is code useful" to "which structural signals are useful for which tasks, and at what cost."

**Goal**: The authors aim to re-examine the relationship between code, mathematics, and reasoning using finer-grained data definitions: first by separating Code from Code-NL, then performing fixed-budget ablations on a 10T-token corpus, and finally filtering "cognitive scaffolds" with explicit step-by-step structures from the math domain to observe if they enhance complex mathematical reasoning without significantly harming programming ability.

**Key Insight**: The paper decouples "structure" from "code files." Pure Code is strictly defined as executable functions, scripts, and program fragments, excluding comments; Code-NL retains structured materials mixing code and natural language. Subsequently, a FastText structure classifier is used to identify samples in the math corpus with sub-goals, step-by-step derivations, symbolic manipulations, and verification processes as more direct reasoning scaffolds.

**Core Idea**: Instead of generically increasing the code ratio, it is more effective to increase the density of structured mathematical reasoning samples within a fixed math budget, training the model to solve difficult mathematical problems using visible intermediate reasoning trajectories.

## Method
The methodology focuses on large-scale data causal attribution experiments rather than a new model architecture. The authors first construct a strictly domain-partitioned 10T-token pre-training corpus and train MoE and dense models of various scales to compare data configurations such as full data, w/o code, w/o math, and w/o cognitive scaffolds. A key aspect is the fixed total training tokens: when a domain is removed, the remaining domains are upsampled proportionally, ensuring performance differences reflect data replacement effects rather than changes in training volume.

### Overall Architecture
The workflow consists of four steps. First, data partitioning and cleaning: the corpus is divided into Web, Code, Code-NL, Math, Wikipedia, Books, and Multilingual categories, with quality admission controlled by over 300 metrics. Second, pre-training MoE models from scratch: the core model is a 20-layer autoregressive MoE with a hidden size of 2048, 16 heads, and 16 experts per MoE layer using top-2 routing. Third, fixed-budget ablation: removing pure Code or Math to observe changes across five capability dimensions. Fourth, replacing a portion of ordinary math samples with cognitive scaffolds within the Math domain to test the impact of structured reasoning density, using expert routing distributions to explain how data mixtures alter internal activations.

### Key Designs
1. **Strict Distinction between Code and Code-NL**:
    - **Function**: Separates pure executable programs from mixed-format reasoning materials to avoid misattributing reasoning gains from the latter to code.
    - **Mechanism**: Code primarily comes from repositories like GitHub, requiring an executable code density above a threshold and passing syntax, length, and deduplication filters. Code-NL comes from web pages, notebooks, Q&A, and Markdown, allowing natural language explanations and formulas to interleave with code snippets. During code ablation, pure Code is removed while Code-NL is retained.
    - **Design Motivation**: If Code-NL is counted as code, observed reasoning gains might actually stem from structured explanations and mathematical derivations. This split allows estimation of the marginal contribution of pure code.

2. **Fixed-Budget Data Ablation**:
    - **Function**: Measures the competition and synergy between different data domains under the same training budget.
    - **Mechanism**: Starting from the full corpus, w/o code and w/o math models are trained. Upon removing a domain, the total token count is maintained by proportionally upsampling other domains. Evaluation covers general knowledge, programming, mathematics, comprehensive reasoning, and professional knowledge.
    - **Design Motivation**: In real-world pre-training, the most common constraint is a fixed token budget. This setup exposes negative coupling: adding one domain may improve domain-specific performance but depress other capabilities.

3. **Cognitive Scaffold Filtering and MoE Routing Analysis**:
    - **Function**: Identifies high-density structured reasoning samples within the math domain and checks if such data improves complex math capabilities via stable internal routing.
    - **Mechanism**: A lightweight FastText classifier is trained using 200k code samples as positive labels and 200k non-code samples as negative labels to recognize explicit structural patterns. It is then applied to the Math corpus to select cognitive scaffolds where $f_\theta(x)\geq\tau$. The classifier achieves 0.9696 accuracy on the validation set. Post-hoc statistics show these samples have higher symbolic density, more derivation steps, higher indentation ratios, and longer text lengths.
    - **Design Motivation**: The authors aim to prove that the effective signal is "explicit intermediate reasoning structure" rather than code semantics. MoE routing analysis further shows that scaffolds do not drastically rearrange experts like removing Code/Math does, but rather act as a stable cross-domain signal.

### Loss & Training
The models use standard autoregressive language modeling objectives. MoE training employs dropless routing, load-balancing loss, and router z-loss, with stochastic routing warmup in the early stages: learned routing logits are interpolated with random logits based on $\alpha=\min(t_c/t_w,1)$ to mitigate early expert congestion. Training uses AdamW with a learning rate of $5\times10^{-5}$, 2000 steps of warmup, bfloat16/FP8 mixed precision, and total training of 24,000 iterations, with checkpoints saved every 1200 iterations. Cognitive scaffolds are not scheduled as a new domain but replace ordinary math samples within the fixed Math budget.

## Key Experimental Results

### Main Results
| Research Question | Metric / Dataset | Key Results | Control | Conclusion |
|-------------------|------------------|-------------|---------|------------|
| Does pure code improve math reasoning? | Math Avg. | Full data is 14.38% lower than w/o code | Code-NL held constant | Pure executable code is not a universal math reasoning enhancer |
| Impact of code on complex math tasks | Minerva-Math / OlympiadBench / MATH | -71.53% / -47.16% / -22.64% | w/o code is better | Code competes significantly with math knowledge/derivation budget in difficult tasks |
| Impact of math on programming | CodeForces / LiveCodeBench | +37.11% / +11.26% | w/o math is worse | Math data helps in competitive programming/algorithmic tasks |
| Negative coupling of math data | CruxEval / MBPP | -17.30% / -6.12% | w/o math is better | Math data can interfere with some mixed code reasoning tasks |
| Cognitive scaffolds | Math Avg. | +17.56% | Fixed Math token budget | Structured math samples significantly enhance complex reasoning |

### Ablation Study
| Configuration | Key Metric | Description |
|---------------|------------|-------------|
| Full data (32e) | Math overall 36.20 / Programming overall 26.94 | Full corpus baseline for 32-expert MoE |
| w/o code (32e) | Math overall 38.52 / Programming overall 14.25 | Removing pure code increases math avg but sharply drops programming |
| w/o math (32e) | Math overall 17.71 / Programming overall 24.25 | Removing math causes math collapse; programming is slightly lower than full |
| Scaffold replacement | College Math +30.05%, MATH +23.17%, OlympiadBench +47.78% | Increasing structured sample density in math budget gives max gains on complex tasks |
| Scaffold side effect | GSM8K -6.29%, CMath -2.00%, code benchmarks ~ -1% | Slight competition with simple NL math problems; minimal impact on code |

### Key Findings
- The paper refutes the coarse claim that "pure code naturally improves reasoning," rather than denying the utility of structured data. The truly effective signals come from explicit steps, symbolic manipulation, hierarchical decomposition, and verification.
- Code-NL is the key control variable for resolving discrepancies. Gains observed in prior work likely came from these mixed structured texts (Markdown, notebooks, etc.) rather than executable code.
- Cognitive scaffolds yield significant gains for complex math but slightly hinder simple natural language math (GSM8K, CMath), suggesting "more structure" is not always better.
- MoE routing shows that removing Code or Math causes significant shifts in expert distribution, whereas removing scaffolds causes smaller, more dispersed shifts, supporting their interpretation as stable cross-domain reasoning signals.

## Highlights & Insights
- The paper advances the pre-training data discussion from "which domain ratio is higher" to "which structural features within a domain are effective." This is more actionable than debating code ratios as it guides data selection.
- The fixed-budget design is critical. Data ablations that simply reduce tokens conflate quality with training volume; by keeping total tokens constant, this study captures real-world resource allocation problems.
- The FastText classifier for scaffolds is simple yet effective. Without training complex reward models, the authors learn transferable explicit structures from code samples and project them onto the Math domain.
- Expert routing analysis provides mechanistic evidence beyond downstream scores. While routing patterns are not strictly causal, they show that different data configurations fundamentally change how MoE experts are utilized.

## Limitations & Future Work
- The definition of "cognitive scaffold" remains operational. FastText might learn surface features like formatting, length, or indentation, despite the authors' audit for contamination and post-hoc statistics.
- The paper does not systematically scan scaffold replacement ratios, so it cannot conclude that "higher scaffold ratios are always better." Current findings apply only to the specific replacement setup.
- Computational costs are high, creating a barrier for replication. While 10T tokens and MoE-from-scratch increase credibility, they limit external validation.
- Evaluation focuses on pre-training capabilities; it is yet to be seen if these data effects persist after instruction tuning, RLHF, or in tool-use agent scenarios.

## Related Work & Insights
- **vs. To Code or Not to Code / Code-ratio studies**: Prior works often mix code and structured text. By isolating pure Code, this paper shows reasoning gains likely originate from mixed structured samples.
- **vs. DoReMi / REGMIX**: While data mixing methods focus on domain-level ratios, this paper highlights that instance-level structural features within domains are equally crucial. Cognitive scaffolds could be included as a learnable subdomain in future optimizations.
- **vs. Data Selection Methods**: The scaffold filtering can be viewed as offline data selection for reasoning. Unlike general quality scoring, it emphasizes the visibility of intermediate steps and symbolic processes.
- **Insight**: For LLM pre-training, enhancing math reasoning does not necessarily require increasing the entire Math domain ratio; one can increase the density of "traceable reasoning trajectories" within a fixed budget. Code data should also be distinguished between executable programs, explanations, and annotated derivations.

## Rating
- Novelty: ⭐⭐⭐⭐ The topic has been studied, but the separation of Code/Code-NL and the focus on cognitive scaffolds are highly insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ The combination of a 10T corpus, multiple model scales, fixed-budget ablations, and routing analysis is very robust.
- Writing Quality: ⭐⭐⭐⭐ Clear narrative with sufficient data, though appendix-heavy analysis requires careful attention to relative versus absolute scores.
- Value: ⭐⭐⭐⭐⭐ Highly practical for pre-training data engineering, indicating that "structured reasoning sample density" is more controllable than generic code increases.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Beyond Test-Time Memory: State-Space Optimal Control for LLM Reasoning](beyond_test-time_memory_state-space_optimal_control_for_llm_reasoning.md)
- [\[ICML 2026\] Biases in the Blind Spot: Detecting What LLMs Fail to Mention](biases_in_the_blind_spot_detecting_what_llms_fail_to_mention.md)
- [\[ICML 2026\] Beyond Two-Stage Training: Cooperative SFT and RL for LLM Reasoning](beyond_two-stage_training_cooperative_sft_and_rl_for_llm_reasoning.md)
- [\[ICML 2026\] FloorplanQA: A Benchmark for Spatial Reasoning in LLMs Using Structured Representations](floorplanqa_a_benchmark_for_spatial_reasoning_in_llms_using_structured_represent.md)
- [\[ACL 2026\] LLM Reasoning as Trajectories: Step-Specific Representation Geometry and Correctness Signals](../../ACL2026/llm_reasoning/llm_reasoning_as_trajectories_step-specific_representation_geometry_and_correctn.md)

</div>

<!-- RELATED:END -->
