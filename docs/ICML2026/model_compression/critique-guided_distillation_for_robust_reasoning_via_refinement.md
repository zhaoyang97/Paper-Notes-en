---
title: >-
  [Paper Note] Critique-Guided Distillation for Robust Reasoning via Refinement
description: >-
  [ICML 2026][Model Compression][Knowledge Distillation] Let the student **consume** rather than **generate** the teacher's critique during training—predict the teacher's refined answer conditioned on (prompt…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Knowledge Distillation"
  - "Mathematical Reasoning"
  - "Critique"
  - "Self-Correction"
  - "Supervised Fine-Tuning"
date: 2026-05-08
content_hash: 9cf6fd78f4f99ad3
---

# Critique-Guided Distillation for Robust Reasoning via Refinement

**Conference**: ICML 2026  
**arXiv**: [2505.11628](https://arxiv.org/abs/2505.11628)  
**Code**: Repository not yet released  
**Area**: Model Compression / Knowledge Distillation  
**Keywords**: Knowledge Distillation, Mathematical Reasoning, Critique, Self-Correction, Supervised Fine-Tuning  

## TL;DR
Let the student **consume** rather than **generate** the teacher's critique during training—predict the teacher's refined answer conditioned on (prompt, student's own answer, teacher's critique). During inference, a single prompt produces longer and more accurate reasoning chains without destroying instruction-following capabilities as seen in CFT.

## Background & Motivation

**Background**: The mainstream recipe for distilling strong reasoning capabilities into small models from large teachers is SFT/Distilled-SFT—directly imitating the teacher's gold answer or CoT on the same prompt. A few works (CFT, Self-Refine, Reflexion) attempt to introduce "critique" signals to teach the model self-correction.

**Limitations of Prior Work**: (i) Simple SFT "learns conclusions but not the why", collapsing quickly on OOD and difficult problems; (ii) Methods like Self-Refine/Reflexion run critiques multiple times during inference, doubling inference costs; (iii) Wang 2025's Critique Fine-Tuning (CFT) moves critique generation to training, training the student to "generate critiques"—resulting in severe output-format drift, where LLaMA3.1-8B's IFEval score plummeted from 76.9% to 55.6%, sacrificing a large portion of general capability.

**Key Challenge**: Teaching a model to "where it was wrong and how" via critique is effective, but **teaching a student to output critiques** and **teaching a student to refine answers based on critiques** are two different tasks. The former modifies the student's output distribution and format, while the latter treats the critique as an additional condition during training. Shuffling these two together is why CFT "wins at math but loses at IFEval."

**Goal**: Retain all the benefits of critiques while stripping away the side effects of critique generation, all while maintaining single-pass inference and keeping the model architecture unchanged.

**Key Insight**: Critique should only serve as a "semantic scaffold" during training—it tells the student where the current answer is wrong, but the student's goal is solely to correct that error. During inference, neither the critique nor the student's own draft is provided; the model "internalizes" error-aware reasoning.

**Core Idea**: **Decouple critique consumption from critique generation**. During training, the student sees its own poor answer plus the teacher's critique but is only supervised to predict the teacher's refined answer; during inference, it only takes the prompt as input for single-pass generation.

## Method

### Overall Architecture
CGD is a minimalist pipeline consisting of a three-stage data synthesis process followed by a one-time SFT, without introducing new modules or changing prompt formats: (1) **Student Answer Generation**: Use the untrained $S_{\theta_{\text{init}}}$ to sample a "likely incorrect" initial answer $y' \sim S_{\theta_{\text{init}}}(\cdot \mid x)$ for each prompt $x$; (2) **Critique Generation**: The teacher $T_\phi$ writes a textual critique $c \sim T_\phi(\cdot \mid x, y')$ clearly pointing out where $y'$ is wrong; (3) **Refined Answer Generation**: The teacher produces a gold-standard refinement $\hat{y} \sim T_\phi(\cdot \mid x, y', c)$ based on the full context. Finally, the student is trained once using this quadruple $((x, y', c), \hat{y})$.

### Key Designs

1. **Curriculum anchored on student-specific errors**:

    - Function: Ensures critiques are always targeted at "errors the current student actually makes," rather than general errors imagined by the teacher.
    - Mechanism: $y'$ is not pre-generated once but **sampled fresh for each prompt**—meaning the critique and refined answer are tied to the failure modes of the specific checkpoint $S_{\theta_{\text{init}}}$. This contrasts sharply with standard distilled SFT, where the same teacher solution is fed regardless of the student's specific mistake, effectively creating a "customized curriculum" based on student weaknesses.
    - Design Motivation: Ablations show that when critiques do not match the student's actual errors (e.g., replaced with placeholders or irrelevant critiques), gains shrink significantly. The authors call this "specificity and relevance of feedback" and identify it as the direct driver of CGD's gains.

2. **Quaternary conditions during training, single-pass prompt during inference**:

    - Function: Uses the critique as a "semantic scaffold" exclusive to the training phase, completely erasing it during inference.
    - Mechanism: The training objective is standard NLL, $$\mathcal{L}(\theta) = \mathbb{E}_{(x, y', c, \hat{y})}\big[-\log S_\theta(\hat{y} \mid x, y', c)\big]$$; during inference, the model only sees $x$ and generates the answer in a single forward pass without special tokens or modified templates.
    - Design Motivation: CFT's goal is $-\log S_\theta(c \mid x, y')$, which pulls the student's output distribution toward a critique style, causing general format drift. CGD's goal remains "writing the answer to the prompt," so the output distribution is not contaminated. However, the internal representations learn the mapping from "wrong to right," resulting in CoT chains that are **spontaneously** 4.4× longer (on AIME) during inference even without seeing a critique.

3.  **Fully supervised training objective without RL or extra critics**:

    - Function: Achieves self-correction effects comparable to RL-based systems (SCoRe, RL4F) using the simplest SFT form.
    - Mechanism: The teacher acts as both critic and refiner. Critiques are provided in text rather than scalar rewards, adhering to the observation that "effective feedback must be specific and actionable." Compared to GRACE/QCRD, it requires no discriminator; compared to CTRL/Shepherd, no separate critic model is needed; compared to Self-Refine, no multi-round decoding is required.
    - Design Motivation: This "information-rich training, single-pass inference" paradigm allows the entire training of 100K samples to be completed within 8 GPU-hours on 16 A100s, making it much more practical than RL pipelines.

### Loss & Training
The sole loss is the conditional NLL in Eq. (1). All baselines (SFT, Distilled SFT, CFT) share the same 100K samples, batch size 64, and 1 epoch to align step counts, taking approximately 8 A100·h. Teachers used are LLaMA3.3-70B Instruct (LLaMA family) or S1.1-32B (Qwen family).

## Key Experimental Results

### Main Results

| Student | Method | Math Reasoning Avg ↑ | General Reasoning Avg ↑ | Representative Gain |
|---------|------|---------------------|------------------------|------------|
| LLaMA3.1-8B-Instruct | base | 41.3 | 29.9 | — |
| LLaMA3.1-8B-Instruct | Distilled SFT | 43.7 | 31.9 | — |
| LLaMA3.1-8B-Instruct | CFT | 41.5 | 32.4 | AMC23 22.5 |
| LLaMA3.1-8B-Instruct | **CGD** | **46.9** | **36.7** | **AMC23 37.5 (+15.0)**, OlympiadBench 23.7 (+8.0) |
| S1.1-3B | base | 35.4 | 17.9 | — |
| S1.1-3B | Distilled SFT | 41.7 | 33.5 | — |
| S1.1-3B | CFT | 38.9 | 29.5 | MATH500 49.6 |
| S1.1-3B | **CGD** | **46.1** | **33.4** | **MATH500 61.8 (+12.2)**, Minerva-Math +6.9 |

Cross-family validation: CGD achieved a +22.6% relative gain over the base on Qwen2.5-Math-7B with 8 A100·h training cost.

### Ablation Study

| Dimension | Metric | LLaMA3.1-8B base | + CFT | + CGD | Interpretation |
|---------|------|------------------|-------|-------|------|
| IFEval (Instruction Following) | acc | 76.9 | **55.6 (-21.3)** | ≥76.9 | CFT shows catastrophic degradation; CGD preserves it |
| MUSR / TruthfulQA / BBH / HumanEval | Overall | baseline | Significant drop | Equal or Better | CGD does not damage general capabilities |
| AIME (greedy/Pass@1) | acc | Weak | Weak | Significantly Stronger | Largest gains under low sampling budgets |
| Reasoning Chain Length (AIME) | tokens | 1× | — | **4.4×** | CoT spontaneously lengthens during inference without critiques |

### Key Findings
- **"Specificty" of critique is the causal driver**: Ablations controlling the correlation between critique and actual student errors show that gains shrink significantly when correlation drops—indicating that improvements are not just from "seeing more text," but from specific feedback shaping the learning signal.
- **AMC23 +15.0 / MATH500 +12.2 concentrated in low Pass@k**: This means CGD improves the "reasoning quality of each sample" rather than relying on expanded sample budgets; the fact that Pass@k continues to rise with $k$ shows the distribution has not collapsed.
- **CFT's collapse is objective-driven, not a hyperparameter issue**: The authors clearly state this is a fundamental side effect of changing the training objective to critique generation, which cannot be fixed by tuning—proving the necessity of decoupling.
- **Cross-family transferability**: Improvements of 5-7% in math reasoning were achieved across LLaMA, Qwen, S1.1, Mixtral, and OLMo; CGD on pure math data even transferred to HumanEval code generation.

## Highlights & Insights
- The concept of "decoupling consumption from generation" is highly valuable. Many self-improvement/self-refine works couple "judging" with "outputting the judgment." CGD cleanly keeps "using feedback" and discards "generating feedback," bypassing format drift with zero extra inference overhead.
- Using student-specific errors for the curriculum is conceptually close to the "on-policy" idea in RLHF/DPO, but the implementation is entirely supervised without reward models or sampling loops—a "pseudo-on-policy" approach that yields RL-style benefits at SFT costs.
- The "information-rich training, prompt-only inference" paradigm can be naturally extended: for example, providing plans/scratchpads/tool-call trajectories during training for the student to internalize. CGD solidifies the minimum viable recipe for this category of methods.
- The IFEval 76.9→55.6 data point is very educational—it explains why many "reasoning-enhanced" models perform worse on general chatbot benchmarks and localizes the issue to the training objective itself rather than data scale or learning rate.

## Limitations & Future Work
- Student can be misled when the teacher writes incorrect critiques: The authors acknowledge that LLaMA3.3-70B's critique quality is a bottleneck on hard math, and no robustness curve for critique error rates was provided.
- Evaluations are mainly focused on math/reasoning benchmarks; OOD experiments are limited to HumanEval; multimodal, long-context, and safety scenarios were not touched.
- Training samples contain <0.1% fuzzy overlap with evaluation sets (removed by authors), but the risk of contamination in web-crawled instruction data persists; large gains like +12%/+15% require cautious attribution.
- Direct comparisons with RL self-correction methods (SCoRe, RL4F) are limited in the main text (found in the appendix); using CGD as an initialization for RL still requires further validation.

## Related Work & Insights
- **vs Critique Fine-Tuning (CFT, Wang 2025)**: CFT's goal is to predict the critique; CGD's is to predict the refined answer. Using the same (x, y', c) data but with opposite conditional directions, CGD outperforms CFT by 5-7% in math while avoiding IFEval disasters.
- **vs Self-Refine / Reflexion (Madaan/Shinn 2023)**: Those methods run critiques in an inference loop; CGD incorporates critiques into training data for single-pass inference—resulting in lower latency for the same reasoning budget.
- **vs On-policy distillation (GKD/SKD)**: GKD/SKD use teacher probabilities or rewards as implicit signals; CGD provides explicit semantic signals via text critiques. The two could be combined for further gains.
- **vs SCoRe / RL4F**: RL methods require reward models and sampling cycles; CGD achieves similar self-correction behavior with the same data at the cost of a single SFT.

## Rating
- Novelty: ⭐⭐⭐⭐ "Decoupling consumption from generation" is a clear concept, though the technical implementation is a conditional rewrite of CFT; the SFT framework itself is standard.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five model families, two datasets, math+general+OOD evaluations, and the IFEval counter-example are all present, but critique robustness and RL comparisons are relegated to the appendix.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem motivation is precisely targeted (using CFT's IFEval failure as a selling point). The algorithm is concisely explained with 11 lines of pseudocode and one formula.
- Value: ⭐⭐⭐⭐⭐ +22.6% cross-family gain for 8 GPU-hours, no architectural changes, single-pass inference—this is a plug-and-play strong baseline for distilling reasoning into small/medium models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Toward Understanding Adversarial Distillation: Why Robust Teachers Fail](toward_understanding_adversarial_distillation_why_robust_teachers_fail.md)
- [\[ICLR 2026\] STAR: Similarity-guided Teacher-Assisted Refinement for Super-Tiny Function Calling Models](../../ICLR2026/model_compression/star_similarity-guided_teacher-assisted_refinement_for_super-tiny_function_calli.md)
- [\[AAAI 2026\] Efficient Reasoning for Large Reasoning Language Models via Certainty-Guided Reflection Suppression](../../AAAI2026/model_compression/efficient_reasoning_for_large_reasoning_language_models_via_certainty-guided_ref.md)
- [\[ICML 2026\] T3S: Training Trajectory-Aware Token Selection to Break "Imitation Shock" in Reasoning Distillation](training-trajectory-aware_token_selection.md)
- [\[ICML 2026\] Mind Your Margin and Boundary: Are Your Distilled Datasets Truly Robust?](mind_your_margin_and_boundary_are_your_distilled_datasets_truly_robust.md)

</div>

<!-- RELATED:END -->
