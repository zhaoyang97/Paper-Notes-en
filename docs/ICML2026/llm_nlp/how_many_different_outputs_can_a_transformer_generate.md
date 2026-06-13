---
title: >-
  [Paper Note] How Many Different Outputs Can a Transformer Generate?
description: >-
  [ICML 2026][LLM/NLP][Accessible sequences] Starting from two fundamental architectural facts—finite precision and bounded embedding support—this paper proves that any transformer can only generate a finite number of "acc…
tags:
  - "ICML 2026"
  - "LLM/NLP"
  - "Accessible sequences"
  - "packing number"
  - "embedding space geometry"
  - "copying / cramming"
  - "finite precision"
date: 2026-05-08
content_hash: e6f718c25adc8dd5
---

# How Many Different Outputs Can a Transformer Generate?

**Conference**: ICML 2026 Spotlight  
**arXiv**: [2605.22223](https://arxiv.org/abs/2605.22223)  
**Code**: https://github.com/mario-michelessa/transformers_accessibility (Available)  
**Area**: LLM / NLP Theoretical Analysis  
**Keywords**: Accessible sequences, packing number, embedding space geometry, copying / cramming, finite precision  

## TL;DR
Starting from two fundamental architectural facts—finite precision and bounded embedding support—this paper proves that any transformer can only generate a finite number of "accessible sequences." It provides tight upper bounds showing that the length of accessible sequences grows linearly with prompt length, while the proportion of accessible sequences decays exponentially as $1/|V|^n$ beyond a threshold. Theoretical slopes are verified on Pythia, Qwen, Llama, and Gemma using cramming and copying experiments, with empirical results differing from theory by only 5–10x.

## Background & Motivation
**Background**: Transformers are the dominant architecture in both NLP and CV. Extensive "approximation theory" work has proven them to be universal approximators (Yun 2020a/b, Edelman 2022, Kratsios 2022), capable of simulating Turing machines and arbitrary programs (Wei 2022, Giannou 2023). With Chain-of-Thought (CoT), their expressivity is further enhanced. **These theories seem to suggest that transformers can compute almost "anything."**

**Limitations of Prior Work**: However, counter-intuitive failures are repeatedly observed in practice for "elementary" tasks like long-sequence **copying** (Jelassi 2024), **repetition** (Barbero 2024), and **cramming** (Kuratov 2025). Once the input length exceeds a critical threshold, neither larger models nor more training data can resolve the failure. Moreover, the failure is not gradual but **abrupt (cliff-like)**—success is nearly 100% before the threshold and drops almost to 0 after it.

**Key Challenge**: Universal approximation is an asymptotic result under "continuous space + infinite precision." Real transformers operate on **finite floating-point precision**, and their internal representations are empirically constrained to **bounded and highly anisotropic** subspaces (Brody 2023, Rudman 2022). These two limitations imply that the set of distinguishable inputs and output sequences must be a **finite set**, creating a structural gap against the exponential growth of $|V|^n$.

**Goal**: To formalize this gap and answer three specific questions: (i) Given a prompt length $m$, what is the maximum length of different sequences that can be output? (ii) Is there a prompt-independent hard upper bound? (iii) Can this bound be written in a closed-form using only architectural parameters (embedding dimension $d$, radius $r$, precision $\varepsilon$, vocabulary size $|V|$) to **predict** failure lengths for copying/cramming in new models?

**Key Insight**: By partitioning the last-layer embedding space into $|V|$ argmax cells based on the "most likely next token" (Fig 1), generating a sequence of length $n$ is equivalent to finding a prompt embedding such that greedy decoding for $n$ consecutive steps consistently falls into the correct cells. **Gnerability ⟺ Falling into a sufficiently small feasible region.** The number of distinguishable inputs for a finite-precision transformer is governed by the **packing number**.

**Core Idea**: Theoremize that the "number of accessible sequences" is bounded by the "packing number of the embedding support." This converts the upper bound of transformer generation capability into a purely geometric quantity, using mean-field theory and Wasserstein distance to extend the framework to infinite prompt lengths.

## Method

### Overall Architecture
The analytical chain of the paper follows a straight line: "**Architectural facts → Embedding geometry → Packing bound → Accessible sequence bound → Threshold prediction → Experimental validation**":

1. Formalize the transformer as a mapping $\tau:\bigcup_m \mathbb{R}^{d\times m}\to\Delta_{|V|}$ (soft prompt version, including hard prompts as a special case).
2. Assume the last-layer embedding support $E\subset B_d(0,r)$ is partitioned by the unembedding matrix $F$ into $E_i=\{x: (Fx)_i\geq (Fx)_j\,\forall j\}$ (Fig 1 shows a 2D slice visualization on Qwen-2-0.5B).
3. Define **Accessible Sequence**: A sequence $t$ is accessible under precision $\varepsilon$ and prompt length $m$ if there exists $X\in\mathbb{R}^{d\times m}$ such that $B(X,\varepsilon/2)\subset E_t^m$ (i.e., an entire precision ball is mapped to $t$).
4. Prove packing-type upper bounds under two routes: "**finite prompt length**" and "**arbitrary prompt length (mean-field)**."
5. Conduct complementary experiments using cramming (short $m$, measuring linear thresholds) and copying (arbitrary $m$, measuring asymptotic thresholds). Refine crude geometric assumptions into empirical measurements through support refinement and cell volume distribution analysis to reduce the theory/empirical ratio from ~20x to ~5–10x.

### Key Designs

1. **Geometric Definition of Accessible Sequences**:
    - **Function**: Translates the discrete problem of "whether a transformer can output sequence $t$" into a geometric problem of "whether there is an $\varepsilon/2$-neighborhood in the embedding space falling entirely within cell $E_t^m$," allowing the direct application of packing-number tools.
    - **Mechanism**: The last-layer embedding is partitioned into $|V|$ cells $E_i$ based on the argmax of the next token (Definition in Section 3.1, Fig 1 for 2D visualization). For a target sequence of length $n$, the conditions for $n$ consecutive greedy decoding steps are concatenated to form $E_t^m\subset B_{d\times m}(0,r)$. Introducing precision $\varepsilon$ (Assumption 4.3: the transformer is a constant function within each $\varepsilon$-cube), sequence $t$ is accessible iff $E_t^m$ contains a ball of radius $\varepsilon/2$ (Definition 4.1).
    - **Design Motivation**: Previous "universal approximation" results were continuous and failed to explain discrete output failures. This work geometrizes "output = falling into a cell" and "distinguishability = accommodating a precision ball," so the number of accessible sequences is naturally bounded by the packing number. Remark 3.2 extends this to stochastic decoding: greedy inaccessibility implies any stochastic strategy has a success rate $< 50\%$.

2. **Dual-track Packing-number Bounds (finite-m / arbitrary-m)**:
    - **Function**: Provides closed-form upper bounds for the number of accessible sequences and threshold lengths, covering both short prompts and arbitrarily long prompts.
    - **Mechanism**:
        - **Finite prompt length (Thm 4.5 + Cor 4.6)**: $\tau$ distinguishes at most $P(B_{d\times m}(0,r),\|\cdot\|,\varepsilon)\leq (1+2r/\varepsilon)^{dm}$ inputs; thus, the number of accessible sequences is $\leq (1+2r/\varepsilon)^{dm}$. Comparing this to $|V|^n$ yields: once $n > C\cdot m$, where $C = d\ln(1+2r/\varepsilon)/\ln|V|$, inaccessible sequences must exist, and the proportion of accessible sequences decays exponentially as $O(1/|V|^n)$.
        - **Arbitrary prompt length (Thm 4.9 + Cor 4.10)**: Treat transformers as mappings **between probability measures** (mean-field, Sander 2022; attention is permutation equivariant and $L([X,X])_{:,i}=L(X)_{:,i}$). Each prompt $X$ is replaced by an empirical measure $M(X)=\frac1m\sum_i \delta_{X_{:,i}}$. Introducing Wasserstein precision (Assumption 4.8) results in a prompt-independent upper bound $(e+e(2r)^q/\varepsilon^q)^{(1+2r/\varepsilon)^d}$. Here, no "hard length limit" exists, but accessibility still decays as $O(1/|V|^n)$.
    - **Design Motivation**: The first bound explains the linear threshold growth with short prompts, while the second explains why extra-long prompts cannot save copying. Together, they describe the **sigmoid shape** observed in cramming and copying: nearly complete accessibility before the critical length, followed by a $1/|V|^n$ crash. All conclusions are architecture-level properties, agnostic to prompts, training, or compute.

3. **Support Support Refinement + Non-uniform Cell Volume Correction**:
    - **Function**: Replaces two "worst-case" assumptions—(a) embedding support is a full ball $B_d(0,r)$, and (b) each cell $E_t$ has equal volume—with tighter empirical approximations, reducing the theoretical/empirical slope ratio from 14–20x to 5–10x.
    - **Mechanism**:
        - **Support Approximation (Section 5.2)**: Empirical embeddings are small, anisotropic subsets (Rudman 2022). The full ball is replaced with an axis-aligned ellipsoid (radius $r_i$ per dimension) and a cone (minimum opening angle). The packing number is recalculated (Appendix F). Shape parameters are stably estimated using 10K random prompts with length $\ell\approx 1000$ (Fig 9).
        - **Cell Volume Distribution (Section 5.3)**: Frequent tokens occupy larger cells, while rare ones occupy smaller ones. The volume distribution $D=\{|E_t|/|E|\}$ of the next token is measured. An $n$-fold product convolution $D^{\otimes n}$ mimics the volume distribution for sequences of length $n$. The threshold is found where the median of $D^{\otimes n}$ falls below $1/P(E,\|\cdot\|,\varepsilon)$, meaning over half of the sequences are inaccessible (Fig 3). Note that if $D$ is a Dirac delta at $1/|V|$, this reverts to Cor 4.6.
    - **Design Motivation**: The original Cor 4.6 bound relied on worst-case geometry, overestimating slopes. By using "shape + density" corrections from real embeddings, the theoretical values become tight enough to serve as a predictive model—Table 1 shows "Ellipsoid + non-uniform cell" reduces ratios for all 7 models to 5–11x.

### Loss & Training
The paper is purely theoretical with black-box probing; no new models are trained. The cramming experiment only optimizes a soft prompt $Y\in\mathbb{R}^{d\times m}$ of length $m$ using teacher-forcing $\mathcal{L}(Y;x_{1:n})=-\sum_{i=1}^n\log p_\tau(x_i\mid[Y,x_{1:i-1}])$ with frozen weights. The copying experiment fine-tunes on synthetic $x_{1:n}|x_{1:n}$ strings (length $\leq 50$) until 100% accuracy or 10K steps, then tests exact-match on longer strings.

## Key Experimental Results

### Main Results
Models tested include Pythia (160M–2.8B), Qwen-2.5 (0.5B / 1.5B), Llama-3.2 (1B / 3B), and Gemma-3 (270M / 1B). For cramming, 20 targets per $(n,m)$ pair are used to estimate the mean, with targets sourced from PG19 natural text and uniformly sampled random token strings.

| Experiment / Model | Key Observation | Value |
|------------|---------|------|
| Cramming (Qwen-2.5-1.5B), fixed $m$ | Sigmoid fit for "Accuracy vs Length $n$" | Min $R^2=0.88$ |
| $n_{50}(m)$ Linearity (PG19) | $n_{50} \approx Cm$ | $R^2_{\text{PG19}}=0.999$ |
| $n_{50}(m)$ Linearity (random) | Same as above, smaller slope | $R^2_{\text{rand}}=0.995$ |
| Copying task, 7 models | Training length $\leq 50$, cliff-like drop on longer strings | Median Sigmoid $R^2=0.95$ |
| Theoretical Slope (Ball) / Empirical Slope | Avg $\approx 12\times$ across 7 models | See Ablation |

### Ablation Study (Table 1: Ratio of Theoretical Upper Bound to Empirical Slope)

| Geometric Assumption | Pythia-160M | Pythia-410M | Pythia-1B | Qwen-0.5B | Qwen-1.5B | Llama-1B | Gemma-270M |
|---------|------------|------------|-----------|-----------|-----------|----------|-----------|
| Ball (Original Cor 4.6) | 9.24 | 9.79 | 7.77 | 14.1 | 20.4 | 14.3 | 11.52 |
| Cone (Min opening angle) | 9.10 | 9.60 | 7.70 | 14.01 | 20.34 | 13.98 | 11.24 |
| Ellipsoid (Anisotropic) | 7.92 | 8.15 | 6.12 | 10.96 | 15.30 | 11.86 | 11.12 |
| **Ellipsoid + Non-uniform cell** | **6.66** | **5.99** | **4.56** | **7.92** | **10.82** | **10.71** | **8.79** |
| Ellipsoid + variable $\varepsilon$ | 8.65 | 9.83 | 7.71 | 12.32 | 18.81 | 14.63 | 13.42 |

### Key Findings
- **The slope is indeed linear**: $n_{50}(m)$ shows linear fits with $R^2 \geq 0.995$ for both PG19 and random strings, directly validating the $n^\star(m)=Cm$ prediction from Cor 4.6.
- **PG19 slope > random string slope**: Natural text has stronger structure and less effective information, allowing "more" length for the same $m$, consistent with Kuratov 2025.
- **Geometric corrections progressively reduce error**: Shifting from Ball → Cone yields little gain, but Ellipsoid captures anisotropy to reduce the ratio by 1–4x. Adding non-uniform cell volumes further tightens the ratio to 4.5–10.8x. Qwen-2.5-1.5B remains an outlier, suggesting its support shape is more complex.
- **Cliff-like drops in copying**: Long-string accuracy behaves like a step function after 100% training accuracy (median Sigmoid $R^2 = 0.95$), validating the exponential decay beyond prompt-independent thresholds in Cor 4.10.
- **Universal conclusions**: Remark 1.1 notes that any architecture with bounded representations and finite precision (including SSMs like Mamba) is subject to these same bounds; memory vector length $h$ in test-time training can also be predicted.

## Highlights & Insights
- **Geometric translation of capacity**: Representing generation upper bounds as the packing number of embedding space is the most natural formal language for these problems. Transformer "capability" is now equivalent to "whether enough non-overlapping precision balls can fit in the embedding support."
- **Highly explanatory corollaries**: Cor 4.6 explains linear threshold growth for short prompts; Cor 4.10 explains why long prompts don't save copying. These two cover almost all empirical findings on copying/cramming from the last three years.
- **Infinite prompts via mean-field + Wasserstein**: Replacing prompts with empirical measures $M(X)$ allows limits of "infinite length" to gain an analytical precision concept (Wasserstein-$\varepsilon$), extending fixed-$m$ results.
- **Transferable trick**: Using the "unembedding matrix + Monte-Carlo sampling" to measure cell volume distribution $D$ and estimating critical $n$ via $n$-fold convolution is a "data-driven geometric estimation" applicable to any transformer.
- **Agnostic to architecture specifics**: Bounds depend only on $d, r, \varepsilon, |V|$, making them applicable to Pythia and Mamba alike. With closed-form theories, the cost of predicting failure thresholds for new models is essentially zero.

## Limitations & Future Work
- Upper bounds are still loose by 5–10x: Even with ellipsoid and non-uniform corrections, Qwen-1.5B has a 10.8x error, suggesting last-layer embeddings have more complex structures (potential low-dimensional manifolds) than ellipsoids.
- Assumption 4.8 (Wasserstein precision) is significantly stronger than $\ell_\infty$ precision; the "elementary operations" alternative in Appendix E has poor readability, limiting the intuitive appeal of Thm 4.9.
- Experimental focus on "mechanical" tasks (cramming/copying) without touching reasoning tasks, where target sequences might cluster in small subspaces, potentially making accessibility less of a bottleneck.
- Analysis limited to models $\leq 3$B parameters; whether 70B+ models follow the same proportions is unverified. There's also no fine-grained alignment with actual floating-point (FP16/BF16) $\varepsilon$ values.
- Only upper bounds are provided: More restrictive capacity limits than packing might exist, especially if cell shapes are highly irregular.

## Related Work & Insights
- **vs. Kuratov et al. 2025 (Cramming empirical study)**: They first used cramming as an operational definition of accessibility and reported the PG19/random gap; this paper provides the first rigorous theoretical explanation and extends experiments to multiple model families.
- **vs. Jelassi et al. 2024 ("Transformers can't copy")**: They showed empirical copying failure; this paper proves it is an inevitable consequence of embedding geometry, independent of training or size.
- **vs. Huang et al. 2025 (Length generalization framework)**: Huang et al. proved copying impossibility under absolute positional encoding with idealized assumptions; this paper removes those assumptions and quantifies "impossibility" as a predictable threshold.
- **vs. Meyer et al. 2025 (Memory limits)**: Also uses packing number but focuses on memory; this paper pushes the tool to generation capability.
- **vs. Chiang 2025, Strobl et al. 2024 (Formal languages)**: While they analyze which language classes transformers can represent (TC⁰, etc.), this paper is a **counting version** of expressivity—it tells you "how many sequences" can be output rather than just whether a class is possible.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to frame transformer accessible sequence counts as a closed-form packing-number bound.
- Experimental Thoroughness: ⭐⭐⭐⭐ 11 sizes across 4 model families + 2 tasks + 2 target distributions, but restricted to mechanical tasks and below 10B scale.
- Writing Quality: ⭐⭐⭐⭐ Clear correspondence between theorems and experiments; geometric intuition is very accessible; Wasserstein assumptions in the mean-field section are a bit abrupt.
- Value: ⭐⭐⭐⭐⭐ Provides the first architecture-agnostic theory for "why transformers can't copy," with direct implications for SSMs and test-time training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] One Persona, Many Cues, Different Results: How Sociodemographic Cues Impact LLM Personalization](../../ACL2026/llm_nlp/one_persona_many_cues_different_results_how_sociodemographic_cues_impact_llm_per.md)
- [\[ICML 2026\] "I've Seen How This Goes": Characterizing LLM and Human Writing Diversity via Progressive Conditional Surprisal](ive_seen_how_this_goes_characterizing_diversity_via_progressive_conditional_surp.md)
- [\[ACL 2026\] Big AI is Accelerating the Metacrisis: What Can We Do?](../../ACL2026/llm_nlp/big_ai_is_accelerating_the_metacrisis_what_can_we_do.md)
- [\[ICLR 2026\] AssetFormer: Modular 3D Assets Generation with Autoregressive Transformer](../../ICLR2026/llm_nlp/assetformer_modular_3d_assets_generation_with_autoregressive_transformer.md)
- [\[ACL 2026\] Synthetic Eggs in Many Baskets: The Impact of Synthetic Data Diversity on LLM Fine-Tuning](../../ACL2026/llm_nlp/synthetic_eggs_in_many_baskets_the_impact_of_synthetic_data_diversity_on_llm_fin.md)

</div>

<!-- RELATED:END -->
