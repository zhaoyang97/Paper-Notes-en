---
title: >-
  [Paper Note] Position: Let's Develop Data Probes to Fundamentally Understand How Data Affects LLM Performance
description: >-
  [ICML 2026][Interpretability][Data Probes] The authors argue that instead of repeated trial-and-error with large-scale real-world corpora…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Data Probes"
  - "Typical Set"
  - "Markov Chains"
  - "Falsifiable Transfer"
  - "Position Paper"
date: 2026-05-08
content_hash: 164faeaaa8b3dc1d
---

# Position: Let's Develop Data Probes to Fundamentally Understand How Data Affects LLM Performance

**Conference**: ICML 2026  
**arXiv**: [2605.18801](https://arxiv.org/abs/2605.18801)  
**Code**: None (position paper, only provides illustrative experiments with GPT-2 + Markov chains)  
**Area**: Interpretability / LLM Data Science / Information Theoretic Analysis  
**Keywords**: Data Probes, Typical Set, Markov Chains, Falsifiable Transfer, Position Paper

## TL;DR
The authors argue that instead of repeated trial-and-error with large-scale real-world corpora, researchers should design "data probes"—synthetic sequences sampled from **fully known** stochastic processes. By training/fine-tuning LLMs on these and feeding the generated results **back into the known distributions** for likelihood analysis, the question of "what kind of data allows the model to learn what" can be elevated from empirical heuristics to falsifiable scientific propositions.

## Background & Motivation

**Background**: Today, LLM training data involves trillions of tokens. Stages such as data filtering, mixing ratios, and curriculum depend on empirical heuristics (e.g., filtering pipelines of DataComp-LM, FineWeb, DeepSeek) derived from repeated experiments by large organizations on real corpora.

**Limitations of Prior Work**: Such research suffers from three major flaws: (1) Extremely high computational costs, affordable only by a few large organizations; (2) The true distribution of real corpora is **unknown**, making it impossible to calculate the true likelihood of any sequence and thus impossible to judge whether model generation is "over-conservative" or "over-divergent"; (3) Benchmark evaluations only indicate whether a model works, failing to answer **why** specific data leads to good or bad performance.

**Key Challenge**: A gap exists between the theoretical side (analyses using simplified Markov/Transformer models by Makkuva, Rajaraman, etc.) and the practical side (tuning parameters on real data). Theoretical conclusions are too abstract for LLMs, while practical findings are too fragmented and case-specific. Both sides lack a **unified, controllable, and likelihood-computable experimental medium**.

**Goal**: To provide a methodological framework that enables researchers to (a) precisely control data distribution properties (entropy rate, vocabulary size, dependency structure), (b) calculate the likelihood of generated sequences under a known distribution, and (c) transfer falsifiable "claims" from the probe space to the real LLM space.

**Key Insight**: Rather than attempting to characterize real data, one should do the **opposite**: since the real distribution is unlearnable, actively construct a **fully known** distribution as a "reference frame." This inspiration traces back to Shannon’s 1948 assertion: "a sufficiently complex stochastic process can adequately represent a discrete source."

**Core Idea**: Treat **data itself** as a formal object with an explicit probabilistic definition—a data probe $\Pi=(\mathcal{P},\mathcal{M},\mathcal{H},\mathcal{F})$ (generation process, metrics, claims, falsification rules)—paired with a two-layer IV/EV (Internal/External Validity) verification protocol, making "data $\to$ LLM behavior" research as controllable, reproducible, and falsifiable as physics experiments.

## Method

This is a position paper providing a **methodological framework + an illustrative experiment** rather than a single algorithm. The following unfolds across three layers: "Definition—Verification—Application."

### Overall Architecture

The entire pipeline of the data probe methodology follows four steps: (1) Designing a generation process $\mathcal{P}$ with **theoretical interpretations** and controllable knobs (e.g., entropy rate, vocabulary size, dependency order); (2) Sampling training/testing sequences from $\mathcal{P}$ to train a **probe-LLM** (architecture identical to real LLMs, with embeddings adapted to the synthetic vocabulary); (3) Generating new sequences from the probe-LLM under different decoding conditions and feeding them **back to $\mathcal{P}$ to calculate likelihood**, comparing against computable diagnostic metrics (e.g., average NLL, typical set membership); (4) Conducting **direction-consistent** qualitative comparisons on real LLMs (text-LLM, e.g., GPT-2) to determine if a claim is "transferable" or "probe-local."

The input is a causal hypothesis (claim card) pre-declared by the researcher, and the output is a **transfer decision table**: cross-referencing whether internal validity IV(h) holds in the probe space and whether external validity EV(h) holds in the real space. "Successful transfer" requires both to be 1. If IV=1 and EV=0, the conclusion holds only locally in the probe space; if IV=0, the claim is directly falsified.

### Key Designs

1.  **Formalization of Data Probes and Four Admission Criteria**:

    - **Function**: Upgrades "data" from a vague corpus object to a formalizable tuple $\Pi=(\mathcal{P},\mathcal{M},\mathcal{H},\mathcal{F})$, forcing researchers to declare four criteria for a qualified probe.
    - **Mechanism**: $\mathcal{P}$ must be a fully known and samplable generation process (C1); $\mathcal{P}$ must expose **interpretable intervention knobs**, such as entropy rate, vocabulary size, and dependency order (C2); all diagnostic metrics $\mathcal{M}$ must be computable (C3, e.g., average NLL $-\log p(x^n)/n$ is computable because $p$ is known); every claim $h\in\mathcal{H}$ must be paired with **pre-declared falsification conditions** $\mathcal{F}$ (C4). The paper scores six existing types of work (data diversity, data filtering, transfer/ICL, robustness, information theory, mechanistic interpretability) against C1–C4 in Table 3, identifying which criteria each line lacks.
    - **Design Motivation**: Existing works using synthetic data to study LLMs (e.g., Allen-Zhu’s Physics of LLMs, Makkuva’s Markov analysis) struggle to accumulate findings primarily due to the lack of a unified contract on "what constitutes a qualified probe." These four criteria transform this from a "research style" into an **auditable methodology**.

2.  **Markov Chain Probes with Entropy Rate Constraints and Typical Set Interpretation**:

    - **Function**: Serves as the simplest example, "reducing" the complexities of open corpora into a stationary Markov chain with a target entropy rate $H$, then generating training sequences.
    - **Mechanism**: Since direct construction of a Markov chain with entropy rate $H$ is difficult, the authors use **rejection sampling**—randomly generating numerous transition matrices and selecting the one with an entropy rate closest to $H$ as $\mathcal{P}$. The resulting sequences are fed to a GPT-2 small (probe-LLM), with the embedding layer reshaped to the state space size $M=128$. Theoretically, the $\varepsilon$-typical set from information theory $A_\varepsilon^{(n)}=\{x^n: H-\varepsilon\le -\log p(x^n)/n \le H+\varepsilon\}$ provides a three-regime interpretation: average NLL below the lower bound $\to$ "over-conservative" (repetitive degradation); within the band $\to$ "typical"; above the upper bound $\to$ "uncertain" (deviation from training distribution).
    - **Design Motivation**: Markov chains offer analytical expressions for entropy rate, $p(x^n)$ can be calculated via token-wise multiplication, and **length can be extrapolated arbitrarily**. This allowed the authors to verify a non-trivial phenomenon: while training loss is equivalent to $T=1$ sampling, the average NLL distribution **entirely deviates** from the ground-truth Markov chain during long-sequence generation (extrapolating 128 tokens from 1 starting token)—this is the synthetic counterpart to "degradation in long-content generation" in real LLMs.

3.  **Falsifiable IV/EV Two-Layer Transfer Protocol + Reduction Record**:

    - **Function**: Ensures that the leap from "discovering a phenomenon in probe space" to "it also holds in real LLMs" is **structured and falsifiable**, rather than a narrative analogy.
    - **Mechanism**: Every experimental table is accompanied by a *Claim Card*, which must specify the claim, intervention, probe diagnostics, real-side counterparts, pre-declared failure conditions, and current transfer status. Simultaneously, a **reduction record** is mandated—a table listing line-by-line **what factors were removed from the real scenario to obtain the probe, what invariants were kept, expected directions, and conditions that would overturn the findings**. The final decision is $\mathrm{Accept}(h)=1 \iff \mathrm{IV}(h)=1 \land \mathrm{EV}(h)=1$; if IV=1 but EV=0, the conclusion is explicitly labeled "probe-local."
    - **Design Motivation**: This is key to upgrading the methodology from "we saw X with synthetic data" to "we pre-declare: if X doesn't hold, Y is falsified." The authors emphasize that both bottom-up (designing probes from theory) and top-down (reducing real failure cases to probes) paths share the same protocol, preventing synthetic data research from sliding into "synthesizing for synthesis' sake."

### Loss & Training
No new loss functions are used on the training side; standard next-token cross-entropy is employed. The key is that **training data for the probe-LLM is generated online via Markov sampling**—no dataset management is needed, and scale can be expanded at will. The test set is independently sampled from the same Markov chain to avoid contamination. During the generation phase, greedy/temperature sampling ($T\in\{0,1.0,1.3,1.5\}$, etc.) is used for intervention comparison.

## Key Experimental Results

The experiments are a "proof of concept" intended not to beat benchmarks but to demonstrate whether the methodology can **reproduce** known degradation/uncertainty behaviors of real LLMs.

### Main Results: Probe vs. Real LLM Behavior Under Temperature Intervention

| Decoding Method | probe-LLM Avg NLL | Probe Diagnostic | Real GPT-2 Text Behavior | Direction Consistent? |
| :--- | :--- | :--- | :--- | :--- |
| Greedy ($T=0$) | 0.694 | Over-conservative (Below Typical Set) | Repetitive degradation ("a new field of research that has been around for a while" loops) | Yes |
| Sampling $T=1.0$ | 0.866 | Within Typical Set | Fluent, prompt-related | Yes |
| Sampling $T=1.3$ | 0.979 | Within Typical Set | Slightly divergent but readable | Yes |
| Sampling $T=1.5$ | 1.406 | Uncertain (Above Typical Set) | Prompt-detached, information-irrelevant | Yes |

> **Interpretation**: With just a basic Markov probe (entropy $H=1$ bit/token, vocabulary $M=128$) and GPT-2 small, the authors **reproduced** the three-regime degradation (over-conservative $\to$ typical $\to$ uncertain) seen in real LLMs at different temperatures. Furthermore, the **computable** NLL in the probe space is strictly consistent in direction with the **qualitative descriptions** in the real space.

### Ablation Study / Analysis: Comparison of Criteria with Existing Research

| Research Theme (Representative Works) | C1 Known Process | C2 Controllable Knobs | C3 Computable Metrics | C4 Pre-declared Falsification | Contribution of Probe Method |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Data Diversity/Sufficiency (Makkuva 2025, Rajaraman 2024) | ✓ | Partial | ✓ | ✗ | Adds intervention grids + pre-registered failure rules |
| Data Filtering/Curation (Wettig 2024, Penedo 2024) | ✗ | Partial | ✓ | ✗ | Introduces known process generators + transfer decisions |
| Transfer/ICL (Von Oswald 2023, Edelman 2024) | Partial | ✓ | ✓ | ✗ | Maps distribution shifts to source process hypotheses |
| Robustness/Adversarial (Sainz 2023, Shu & Yu 2024) | ✗ | Partial | ✓ | ✗ | Explicit perturbation intensity + falsification thresholds |
| Information Theoretic Understanding (Zekri 2024) | ✓ | Partial | ✓ | ✗ | Standardized intervention knobs |
| Mechanistic Interpretability (Singh 2024, Räuker 2023) | ✗ | Partial | Partial | ✗ | Known structural families + data-to-mechanism attribution |

### Key Findings
- **Training Loss = $T=1$ sampling** only holds **at the single-step level**: when the probe-LLM autoregressively generates 127 tokens from 1 token, the average NLL distribution is significantly **higher** than the Markov ground-truth (meaning generated sequences are more predictable than the true distribution). This is the synthetic counterpart to "LLM long-sequence generation being inferior to humans." The value of this finding is that such a comparison is **impossible** on real data because the true distribution is uncomputable.
- At $T=1.25$, a **bimodal distribution** emerges—most sequences are more predictable than ground-truth, while a few have unusually high NLL, corresponding to the practical experience that "LLMs are usually conservative but occasionally hallucinate."
- The typical set three-regime (over-conservative / typical / uncertain) serves as a **falsifiable diagnostic** that is more actionable than descriptive terms like "repetitive degradation." One can pre-declare that "increasing temperature should monotonically shift regime mass from the lower bound to the upper bound" and define specific counter-example conditions.

## Highlights & Insights
- **Treating the reduction record as a first-class citizen**: Requiring every probe experiment to include a table of "what was removed and what was kept" directly addresses a major weakness in synthetic data research. While researchers previously took the validity of "synthetic simplification" for granted, this paper requires it to be **explicitly documented and vulnerable to counter-evidence**. This approach is transferable to any field utilizing simplified models (e.g., toy RL benchmarks, small-scale diffusion models).
- **NLL is valuable because the distribution is known**: This is the most easily overlooked insight. Typical set analysis is not done on real corpora because the math is hard, but because $p(x^n)$ is **simply uncomputable**. Sacrificing expressivity for a computable ground-truth distribution is the heart of this position.
- The use of a **unified protocol for both bottom-up and top-down paths** is elegant. it avoids the typical slippery slope of synthetic data research: "I designed a toy $\to$ it showed phenomenon X $\to$ I claim large models also have X."

## Limitations & Future Work
- The authors acknowledge that Markov chains are entry-level demonstrations and cannot cover core dimensions of real language like semantics, pragmatics, or world knowledge—a primary point of contention in the *Alternative Views* section.
- The current EV (External Validity) verification is only **qualitatively** aligned, lacking formal statistical transfer tests. The paper leaves this as an open problem for future work.
- The vocabulary size of 128 and tiny state space are vastly different from the 50k+ vocabulary of real LLMs. By the methodology's own standards, one must explicitly declare the hypothesis that "vocabulary scale does not affect the three-regime typical set behavior" in the reduction record, which was not done here.
- A hidden risk: while **pre-registering falsification conditions** improves rigor, it might inadvertently encourage selective reporting—choosing only claims with easy-to-pass IVs. Monitoring for "p-hacking-on-probes" will be necessary.
- Future directions: The authors list PCFG probes (hierarchical grammar + controllable tree depth/branching factor), multilingual/multimodal probes, and "creative" probes (transforming base probes with another stochastic process)—essentially aiming to expand the current "1D entropy rate" into a "multidimensional data property spectrum."

## Related Work & Insights
- **vs. Physics of LLMs (Allen-Zhu et al.)**: Both use synthetic data to study LLMs, but *Physics of LLMs* usually uses **hand-designed** data for specific tasks (knowledge storage, reasoning structures), lacking unified probabilistic definitions for information theoretic analysis. This paper requires explicit distributions and computable likelihood, offering deeper theoretical hooks.
- **vs. Simplified Transformer Theoretical Analysis (Makkuva 2025, Rajaraman 2024, Zekri 2024)**: These provide asymptotic results for Transformer learning behavior on Markov data but use **simplified architectures** without IV/EV protocols. This paper bridges the gap using real GPT-2 and transfer decisions.
- **vs. Data Filtering Practice (Wettig 2024, Penedo 2024, etc.)**: Practical approaches provide heuristics for "what data is useful"; this paper provides a falsifiable framework for **why** it is useful—they are complementary.
- **vs. Mechanistic Interpretability (Singh 2024, Räuker 2023)**: Interpretability asks "how the model computes"; this paper asks "what data leads to this computation." The former is reverse engineering the model; the latter is controllable forward experimentation on data. Combining them could form a causal chain of "Data Property $\to$ Internal Mechanism $\to$ External Behavior."
- **Insight**: For one's own work, any paper using toy/synthetic settings to study LLM phenomena should be mandated to include a reduction record and pre-declared falsification conditions. This significantly lowers the cost of skepticism regarding "whether the toy holds up" and helps clarify the true boundaries of the conclusions.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Elevating "data probes" from fragmented practice to a methodology with four criteria and IV/EV protocols is a genuine paradigmatic contribution, though the individual technical elements (Markov probes, typical set analysis) are not entirely new.
- **Experimental Thoroughness**: ⭐⭐⭐ As a position paper, it only provides a demonstration with GPT-2 small and a single Markov chain. This is sufficient to prove the methodology's feasibility but far from exhaustive; the empirical burden is left to the community.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure; the *Claim Card* and *reduction record* templates are immediately reusable for subsequent work. The diagnosis of existing research in Table 3 is particularly valuable.
- **Value**: ⭐⭐⭐⭐ If the community adopts the C1–C4 + IV/EV protocol, this contract will significantly improve the cumulative nature of the "synthetic data for LLMs" track. Even if not fully adopted, the *reduction record* suggestion is beneficial for any researcher conducting controlled experiments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] GEM: Geometric Entropy Mixing for Optimal LLM Data Curation](gem_geometric_entropy_mixing_for_optimal_llm_data_curation.md)
- [\[ICML 2026\] MiniMax Learning of Interpretable Factored Stochastic Policies from Conjoint Data, with Uncertainty Quantification](minimax_learning_of_interpretable_factored_stochastic_policies_from_conjoint_dat.md)
- [\[AAAI 2026\] Data Whitening Improves Sparse Autoencoder Learning](../../AAAI2026/interpretability/data_whitening_improves_sparse_autoencoder_learning.md)
- [\[ICML 2026\] Position: Ideas Should be the Center of Machine Learning Research](position_ideas_should_be_the_center_of_machine_learning_research.md)
- [\[ACL 2026\] The Impact of Off-Policy Training Data on Probe Generalisation](../../ACL2026/interpretability/the_impact_of_off-policy_training_data_on_probe_generalisation.md)

</div>

<!-- RELATED:END -->
