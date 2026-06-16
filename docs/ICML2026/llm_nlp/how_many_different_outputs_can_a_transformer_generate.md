---
title: >-
  [Paper Note] How Many Different Outputs Can a Transformer Generate?
description: >-
  [ICML 2026][LLM (Other)][packing number] Starting from two fundamental architectural facts—finite precision and bounded embedding support—this paper proves that any Transformer can only generate a finite number of "accessible sequences." It provides a tight upper bound where the length of accessible sequences grows linearly with prompt length, after which the
tags:
  - ICML 2026
  - LLM (Other)
  - packing number
  - copying / cramming
date: 2026-05-08
content_hash: dad5a585f524f04d
---
# How Many Different Outputs Can a Transformer Generate?

**Conference**: ICML 2026 Spotlight  
**arXiv**: [2605.22223](https://arxiv.org/abs/2605.22223)  
**Code**: https://github.com/mario-michelessa/transformers_accessibility (Available)  
**Area**: LLM / NLP Theoretical Analysis  
**Keywords**: accessible sequences, packing number, embedding space geometry, copying / cramming, finite precision  

## TL;DR
Starting from two fundamental architectural facts—finite precision and bounded embedding support—this paper proves that any Transformer can only generate a finite number of "accessible sequences." It provides a tight upper bound where the length of accessible sequences grows linearly with prompt length, after which the proportion of accessible sequences decays exponentially at a rate of $1/|V|^n$. Experiments on Pythia, Qwen, Llama, and Gemma verify that the theoretical slope differs from the measured value by only 5–10x.

## Background & Motivation
**Background**: Transformers are the dominant architecture in both NLP and CV. Extensive work on "approximation theory" has proven them to be universal approximators (Yun 2020a/b, Edelman 2022, Kratsios 2022), capable of simulating Turing machines and arbitrary programs (Wei 2022, Giannou 2023). Their expressivity is further enhanced with Chain-of-Thought (CoT). **These theories suggest that Transformers can compute almost anything.**

**Limitations of Prior Work**: However, counter-intuitive failures are frequently observed in practice: "elementary-level" tasks such as long-sequence **copying** (Jelassi 2024), **repetition** (Barbero 2024), and **cramming** (Kuratov 2025) fail once the input length exceeds a critical threshold. These failures are not gradual but **cliff-like**—dropping from near 100% accuracy to 0% past the threshold, regardless of model size or training data.

**Key Challenge**: Universal approximation is an asymptotic result under "continuous space + infinite precision." Real Transformers operate on **finite floating-point precision**, and their internal representations are constrained within **bounded and highly anisotropic** subspaces (Brody 2023, Rudman 2022). These two finitenesses imply that the sets of distinguishable inputs and outputtable sequences must be **finite**, creating a structural gap with the exponential growth of $|V|^n$.

**Goal**: To formalize this gap and answer three specific questions: (i) Given prompt length $m$, what is the maximum length of distinct sequences that can be generated? (ii) Does a prompt-independent hard upper bound exist? (iii) Can this bound be written in a closed form using only architectural parameters (embedding dimension $d$, radius $r$, precision $\varepsilon$, vocabulary $|V|$) to **predict** failure lengths for copying/cramming?

**Key Insight**: By partitioning the last-layer embedding space into $|V|$ argmax cells based on the "most likely next token" (Fig 1), generating a sequence of length $n$ is equivalent to finding a prompt embedding such that greedy decoding falls into the correct cells for $n$ consecutive steps. **Generatability $\iff$ hitting a sufficiently small feasible region.** The number of inputs distinguishable by a finite-precision Transformer is controlled by the **packing number**.

**Core Idea**: Establish the theorem that "number of accessible sequences" $\leq$ "packing number of the embedding support." This converts the upper bound of Transformer generation capacity into a pure geometric quantity. The framework is then extended to infinite prompt lengths using mean-field theory and Wasserstein distance.

## Method

### Overall Architecture
The paper uses a concise logical chain to answer how many different sequences a Transformer can output: by viewing the Transformer as a mapping under finite precision, the number of inputs it can distinguish is finite, and thus its output sequences must also be finite. The analysis starts from architectural facts, translates generation behavior into a geometric problem in embedding space, constrains the total number of accessible sequences using the packing number, derives closed-form formulas for length thresholds, and finally validates these predictions on cramming and copying tasks.

Specifically, the Transformer is formalized as a mapping $\tau:\bigcup_m \mathbb{R}^{d\times m}\to\Delta_{|V|}$. Assuming the last-layer embedding support lies within a ball $E\subset B_d(0,r)$, the space $E$ is partitioned into $|V|$ argmax cells $E_i=\{x:(Fx)_i\geq (Fx)_j\,\forall j\}$ according to the unembedding matrix $F$. The task is to count how many sequences can be hit consecutively by greedy decoding within this partition.

### Key Designs

**1. Geometrized definition of accessible sequences: Mapping "outputting sequence $t$" to "fitting an epsilon-ball in embedding space"**

Previous universal approximation results were based on continuity and infinite precision, failing to explain discrete output failures. The first step here is a change of language: partition the last-layer embedding into $|V|$ cells based on the next-token argmax (Section 3.1, Fig 1). Generating a target sequence $t$ of length $n$ is equivalent to falling into the correct cells for $n$ steps. Stacking these conditions yields $E_t^m\subset B_{d\times m}(0,r)$. Introducing finite precision (Assumption 4.3) requires the Transformer to be constant within any hypercube of side length $\varepsilon$. Thus, "sequence $t$ is accessible" $\iff$ $E_t^m$ contains a ball of radius $\varepsilon/2$ (Definition 4.1). This translates the discrete generation problem into a geometric one—the total number of accessible sequences is bounded by the packing number of the embedding support. Remark 3.2 extends this to stochastic decoding: if greedy is inaccessible, the success rate of any sampling strategy is $<50\%$.

**2. Dual-track packing-number bounds: Constraining "short prompts" and "arbitrarily long prompts"**

With the geometric definition, the number of accessible sequences equals the number of non-overlapping precision balls that fit in the support (the packing number). For short prompts (Thm 4.5 + Cor 4.6), the Transformer $\tau$ can distinguish at most $P(B_{d\times m}(0,r),\|\cdot\|,\varepsilon)\leq (1+2r/\varepsilon)^{dm}$ inputs. Comparing this to the total possible sequences $|V|^n$, it follows that when $n>C\cdot m$ (where $C=d\ln(1+2r/\varepsilon)/\ln|V|$), inaccessible sequences must exist, with the ratio decaying as $O(1/|V|^n)$. This explains why the threshold grows linearly with $m$. To address whether infinite prompts could copy any sequence, the second track (Thm 4.9 + Cor 4.10) uses mean-field theory (Sander 2022) to treat the Transformer as a mapping between probability measures. Under a Wasserstein-based precision assumption (Assumption 4.8), a prompt-independent bound $(e+e(2r)^q/\varepsilon^q)^{(1+2r/\varepsilon)^d}$ is derived. While there is no "hard length limit," the accessible ratio still decays as $O(1/|V|^n)$, meaning long prompts cannot save the copying task. Together, these bounds characterize the **sigmoid shape** observed in cramming/copying: near-total accessibility before a critical length, followed by a sharp drop.

**3. Support refinement + Non-uniform cell volume correction: Tightening worst-case bounds**

The original Cor 4.6 assumes a full ball $B_d(0,r)$ and uniform cell volumes, leading to theoretical slopes 14–20x larger than observed. Two refinement steps reduce this to 5–10x. First, **Support Approximation** (Section 5.2): Since embeddings are highly anisotropic (Rudman 2022), the full ball is replaced with an axis-aligned ellipsoid and a cone. The packing number is recalculated (Appendix F), requiring only 10K random prompts to estimate the shape (Fig 9). Second, **Non-uniform Cell Volumes** (Section 5.3): Common tokens occupy large cells while rare ones occupy small ones. The volume distribution $D=\{|E_t|/|E|\}$ is measured via Monte-Carlo sampling. An $n$-fold product convolution $D^{\otimes n}$ simulates the volume distribution of length-$n$ sequences. The threshold is defined as the point where the median of $D^{\otimes n}$ falls below $1/P(E,\|\cdot\|,\varepsilon)$. Table 1 shows that combining the ellipsoid and non-uniform cell corrections reduces the gap to 5–11x across all 7 models.

### Loss & Training
The paper uses pure theory and black-box probing without training new models. Cramming experiments optimize a soft prompt $Y\in\mathbb{R}^{d\times m}$ using teacher-forcing: $\mathcal{L}(Y;x_{1:n})=-\sum_{i=1}^n\log p_\tau(x_i\mid[Y,x_{1:i-1}])$ with frozen weights. Copying experiments fine-tune models on synthetic strings $x_{1:n}|x_{1:n}$ ($n \leq 50$) until 100% training accuracy or 10K steps, then test exact-match on longer strings.

## Key Experimental Results

### Main Results
Models include Pythia (160M–2.8B), Qwen-2.5 (0.5B / 1.5B), Llama-3.2 (1B / 3B), and Gemma-3 (270M / 1B). Cramming used 20 targets per $(n,m)$ from PG19 text and random tokens.

| Experiment / Model | Key Observation | Value |
|------------|---------|------|
| Cramming (Qwen-2.5-1.5B), fixed m | Sigmoid fit for "Accuracy vs. Length n" | Min $R^2=0.88$ |
| n50(m) Linearity (PG19) | n50 ≈ Cm | $R^2_{\text{PG19}}=0.999$ |
| n50(m) Linearity (Random) | As above, smaller slope | $R^2_{\text{rand}}=0.995$ |
| Copying task, 7 models | Sigmoid-like drop past training length | Median $R^2=0.95$ |
| Theoretical (Ball) / Empirical Slope | Average ratio ≈ 12× | See Ablation |

### Ablation Study (Table 1: Ratio of Theoretical Bound to Empirical Slope)

| Geometric Assumption | Pythia-160M | Pythia-1B | Qwen-0.5B | Qwen-1.5B | Llama-1B | Gemma-270M |
|---------|------------|-----------|-----------|-----------|----------|-----------|
| Ball (Original Cor 4.6) | 9.24 | 7.77 | 14.1 | 20.4 | 14.3 | 11.52 |
| Cone (Minimal Opening) | 9.10 | 7.70 | 14.01 | 20.34 | 13.98 | 11.24 |
| Ellipsoid (Anisotropic) | 7.92 | 6.12 | 10.96 | 15.30 | 11.86 | 11.12 |
| **Ellipsoid + Non-uniform cell** | **6.66** | **4.56** | **7.92** | **10.82** | **10.71** | **8.79** |

### Key Findings
- **Linearity of Slopes**: n50(m) shows a linear fit $R^2 \geq 0.995$ for both PG19 and random strings, validating the $n^\star(m)=Cm$ prediction of Cor 4.6.
- **PG19 vs. Random**: Natural text has more structure and less information density, allowing longer sequences to be "crammed" for the same $m$, consistent with Kuratov 2025.
- **Geometric Refinement**: Moving from a ball to an ellipsoid captures anisotropy, reducing the error ratio by 1–4x. Adding non-uniform cell volumes brings all models within 4.5–10.8x.
- **Copying Cliff**: Accuracy on long strings behaves like a step function (Median $R^2 = 0.95$), validating the exponential decay predicted by Cor 4.10.
- **Generalizability**: Remark 1.1 notes that any architecture with bounded representations and finite precision (e.g., Mamba/SSMs) is subject to the same upper bounds.

## Highlights & Insights
- **Clean Translation**: Mapping generation capacity to the packing number of embedding space provides a natural formal language for these limits.
- **Explanatory Power**: Cor 4.6 and Cor 4.10 provide the first rigorous theoretical explanations for empirical observations in copying/cramming over the past three years.
- **Mean-field Integration**: Replacing discrete prompts with empirical measures $M(X)$ allows the analysis of infinite-length limits via Wasserstein-ε precision.
- **Predictive Utility**: The "geometry + density" correction method can be used as a low-cost tool to estimate the accessibility limits of any fixed Transformer on new tasks.

## Limitations & Future Work
- The upper bound remains 5–10x loose, suggesting that last-layer embeddings may reside on more complex low-dimensional manifolds rather than simple ellipsoids.
- Assumption 4.8 (Wasserstein precision) is stronger than $\ell_\infty$ precision; the alternative "elementary operations" assumption in Appendix E is less intuitive.
- The study focuses on "mechanical" tasks (copying/cramming); reasoning tasks might be limited by different bottlenecks as their target sequences might cluster in smaller subspaces.
- The theory only provides an upper bound; tighter capacity limits may exist depending on cell regularity.

## Related Work & Insights
- **vs Kuratov et al. 2025**: Provides the first rigorous theoretical explanation for their empirical cramming findings.
- **vs Jelassi et al. 2024**: Proves that the failure to copy long sequences is an inevitable consequence of embedding geometry.
- **vs Huang et al. 2025**: Quantifies the "impossibility" of length generalization as a predictable critical length, removing idealized assumptions.
- **vs Chiang 2025 / Strobl et al. 2024**: While those works focus on complexity classes (what can be expressed), this work provides the **counting version**—how many different outputs can be generated.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First closed-form packing-number bound for Transformer accessibility.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive model coverage, though focused on mechanical tasks and sub-10B scales.
- Writing Quality: ⭐⭐⭐⭐ Clear mapping between theory and experiment; intuitive geometric figures.
- Value: ⭐⭐⭐⭐⭐ Provides an architecture-agnostic theory for fundamental Transformer limitations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] One Persona, Many Cues, Different Results: How Sociodemographic Cues Impact LLM Personalization](../../ACL2026/llm_nlp/one_persona_many_cues_different_results_how_sociodemographic_cues_impact_llm_per.md)
- [\[ICML 2026\] "I've Seen How This Goes"：用渐进条件惊奇度刻画 LLM 与人类写作的多样性](ive_seen_how_this_goes_characterizing_diversity_via_progressive_conditional_surp.md)
- [\[ACL 2025\] Can Large Language Models Accurately Generate Answer Keys for Health-related Questions?](../../ACL2025/llm_nlp/can_large_language_models_accurately_generate_answer_keys_for_health-related_que.md)
- [\[ACL 2025\] LLM Meets Scene Graph: Can Large Language Models Understand and Generate Scene Graphs?](../../ACL2025/llm_nlp/llm_meets_scene_graph_can_large_language_models_understand_and_generate_scene_gr.md)
- [\[ACL 2026\] Synthetic Eggs in Many Baskets: The Impact of Synthetic Data Diversity on LLM Fine-Tuning](../../ACL2026/llm_nlp/synthetic_eggs_in_many_baskets_the_impact_of_synthetic_data_diversity_on_llm_fin.md)

</div>

<!-- RELATED:END -->
