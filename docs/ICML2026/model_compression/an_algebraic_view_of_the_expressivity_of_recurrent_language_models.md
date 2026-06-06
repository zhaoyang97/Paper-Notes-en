---
title: >-
  [Paper Note] An Algebraic View of the Expressivity of Recurrent Language Models
description: >-
  [ICML2026][Model Compression][Recurrent Language Models] This paper unifies the formal language expressivity of RNNs/SSMs as an algebraic problem: once numerical semantics are fixed…
tags:
  - "ICML2026"
  - "Model Compression"
  - "Recurrent Language Models"
  - "Formal Languages"
  - "Transition Monoids"
  - "Finite Precision"
  - "State Space Models"
date: 2026-05-08
content_hash: c4e3b890ffb9b3c4
---

# An Algebraic View of the Expressivity of Recurrent Language Models

**Conference**: ICML2026  
**arXiv**: [2606.01765](https://arxiv.org/abs/2606.01765)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: Recurrent Language Models, Formal Languages, Transition Monoids, Finite Precision, State Space Models  

## TL;DR
This paper unifies the formal language expressivity of RNNs/SSMs as an algebraic problem: once numerical semantics are fixed, the languages a model can recognize are determined by its hierarchical transition monoid and their wreath products. Furthermore, the same architecture yields entirely different counting capabilities under floating-point versus unsigned integer semantics.

## Background & Motivation
**Background**: In recent years, the analysis of language model expressivity has often studied architectures like RNNs, SSMs, and Transformers as formal language recognizers. The goal is to determine whether they can implement classical computational tasks such as parenthesis matching (dyck languages), modular counting, or finite automata simulation. Theoretical results in this direction often translate into architectural heuristics, such as whether linear RNNs or Mamba-like state-space models can preserve counting information over long sequences.

**Limitations of Prior Work**: Conclusions in existing literature are inconsistent. Some works prove that RNNs possess strong computational power under exact real or rational arithmetic, even reaching Turing completeness. Others prove they can only simulate finite automata under finite-precision or resource-constrained assumptions. The issue is not that one side is "wrong," but that the default arithmetic models, rounding rules, overflow semantics, and evaluation orders differ across these proofs.

**Key Challenge**: Neural network formulas appear to be continuous real-number operations, but real-world deployment always occurs on discrete, finite numerical systems with rounding. If a theoretical proof relies on the associative law in the real field, infinite precision, or arbitrarily rearrangeable algebraic identities, it may not transfer to floating-point implementations. Conversely, stating "finite precision" without specifying exact computational semantics fails to yield reproducible expressivity conclusions.

**Goal**: The authors aim to provide a unified framework that decomposes the expressivity of recurrent language models into three replaceable components: the architecture’s state transition structure, the composition of inter-layer wiring, and the underlying arithmetic semantics. This allows researchers to pinpoint whether conflicts in conclusions arise from the architecture itself or from differing numerical semantics.

**Key Insight**: Starting from monoid theory in automata theory, the paper treats each recurrent core as a finite state transition system and views the hierarchical composition of deep RNNs as the wreath product of transformation monoids. Recognizing a formal language is no longer about directly constructing network parameters, but about determining whether the syntactic monoid of the target language can divide the monoid structure implementable by the model.

**Core Idea**: Replace vague real-valued network formulas with "transition monoids under fixed arithmetic semantics," thereby reducing the expressivity of RNNs/SSMs to a divisibility problem in finite algebra.

## Method
Instead of proposing a new training algorithm, the paper presents an algebraic framework for analyzing the expressivity of recurrent language models. The logic is: first abstract one recurrent module layer as an algebraic core, then abstract multi-layer networks as a cascade of cores; use the wreath product to provide an upper bound on hierarchical transitions, and then use the realized input set to tighten it to transitions reachable by actual wiring; finally, connect the language recognition problem to syntactic monoids to obtain an algebraic criterion for "whether the target language can be recognized."

### Overall Architecture
The input consists of a class of recurrent language models, a finite input alphabet, a fixed encoder, and numerical semantics; the output is not a model prediction but a structured characterization of the family of recognizable languages. The framework defines a state set $Q$, input set $X$, output set $Y$, state transition $f:Q\times X\to Q$, and readout $g:Q\times X\to Y$ for each layer. For each input $x$, the state transition $f_x:Q\to Q$ generates the transition monoid of that layer. Since multi-layer networks use a wiring map to turn the input/output of the previous layer into the input of the next, the global state space is the Cartesian product of the states of each layer.

Subsequently, the paper uses the wreath product to describe hierarchical dependencies: the state of a lower layer affects the input seen by the upper layer at the current timestep. Thus, a deep RNN is not a simple parallel direct product, but a cascade transition system. To avoid counting unreachable inputs as expressivity, the author further defines the reachable input set and the realized transition monoid relative to the input set $T$, retaining only the intra-layer transitions actually activated by the encoder and wiring.

Finally, the author transforms the model into a language acceptor: by appending an accepting core, the acceptance condition becomes a set over the state of the final layer rather than post-processing on temporary readout values. This allows the use of Eilenberg-style algebraic language theory: if a model recognizes language $\mathcal{L}$, then the syntactic monoid of $\mathcal{L}$ must divide the realized wreath product of that model.

### Key Designs
1.  **Algebraic core and transition monoid**:
    *   **Function**: Abstracts each recurrent layer into a finite or discrete state transitioner, preserving the core structure of "input-driven state changes."
    *   **Mechanism**: For a core $\mathfrak{c}=(Q,X,Y,f,g)$, each input $x\in X$ induces a self-map $f_x:Q\to Q$. All such maps generate $M_{\mathfrak{c}}=\langle f_x\mid x\in X\rangle$ under function composition. Readout $g$ does not directly enter the transition monoid, as it determines how states are observed rather than the state dynamics.
    *   **Design Motivation**: This separates "what dynamic information the model can store internally" from "how the answer is finally read out," preventing the expressivity of the decoder from being misattributed to recurrent dynamics.

2.  **Realized wreath product**:
    *   **Function**: Characterizes the global transition structure actually achievable by a multi-layer RNN given a wiring and input alphabet.
    *   **Mechanism**: The ambient upper bound of a deep cascade is the iterated wreath product of each layer's transition monoid. However, this bound allows the upper layer to receive any input in $X_n$. The author defines the layer-input dependency map $\varphi_n^T$, which collects only inputs reaching the $n$-th layer from the first-layer input set $T$. These reachable inputs generate $M_n^T$. The final result is $\mathbb{W}_{\mathcal{R}}^T=(M_1^T,Q_1)\wr\cdots\wr(M_N^T,Q_N)$.
    *   **Design Motivation**: This tightening avoids "phantom expressivity" that is formally allowed by the architecture but never triggered by the wiring, while also allowing local updates to the analysis when the encoder or input distribution changes.

3.  **Incorporating arithmetic semantics into model definition**:
    *   **Function**: Explains why the same RNN/SSM architecture produces contradictory expressivity conclusions in different papers.
    *   **Mechanism**: The author defines the arithmetic model as $\mathfrak{M}=(\mathcal{D},\mathcal{O},\square)$, where $\mathcal{D}$ is the representable range, $\mathcal{O}$ is the set of operations, and $\square$ is the rounding/truncation map. Expressions must be equipped with a fixed evaluation tree, and recurrent updates must satisfy recurrence-consistent evaluation (completing time $t$ before entering $t+1$).
    *   **Design Motivation**: Floating-point addition and multiplication are not associative; compiler or hardware reordering changes results. Without fixing these semantics, the question of "whether a language can be recognized" is not well-defined.

### Loss & Training
This paper does not involve training losses or optimization strategies. It studies expressivity rather than learnability: given a family of architectures and numerical semantics, does there exist a parameterized instance that can recognize the target formal language. The author explicitly states that the framework does not guarantee that these parameters can be automatically found through gradient descent.

## Key Experimental Results

### Main Results
The "main experiments" in this paper are theoretical results and case studies rather than dataset benchmarks. The core result compares the differences in expressivity under different arithmetic models within a single algebraic table.

| Object | Criterion / Result | Conclusion | Impact |
| :--- | :--- | :--- | :--- |
| Single-layer algebraic core | $M_{\mathfrak{c}}=\langle f_x\rangle$ | Intra-layer dynamics are determined by the transition monoid | Transfers architectural capability to monoid problems |
| Deep algebraic RNN | $M_{\mathcal{R}}^T\leq W_{\mathcal{R}}^T$ | Global transitions are embedded in the realized wreath product | Hierarchical composition can be analyzed via wreath products |
| Language acceptor | $M(\mathcal{L})\prec M_{\mathcal{R}^+}^{e(\Sigma)}\leq W_{\mathcal{R}^+}^{e(\Sigma)}$ | Syntactic monoid of the target language must divide the model transition structure | Unified entry point for non-recognizability proofs and recognizable constructions |
| Non-negative diagonal SSM + float | core monoid aperiodic | Cannot implement modular counting requiring non-trivial groups | Corrects over-strong claims that "the same architecture can count" |
| diagonal SSM + unsigned int quantization | Can contain $\mathbb{Z}/2^k\mathbb{Z}$ | Supports structures like even modular counting | Changes in numerical semantics alter expressivity |

### Ablation Study
The following table serves as an arithmetic semantics analysis: keeping the diagonal SSM form essentially the same while only replacing the recurrence multiplier and numerical model to observe the group structures that can appear in the core monoid.

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| Non-negative recurrence + float fp | Only trivial subgroups, core monoid is aperiodic | Non-negative floating-point affine updates are order-preserving maps on finite chains, cannot produce non-trivial cycles |
| Signed multiplier allowed + float fp | $\mathbb{Z}/2\mathbb{Z}$ can appear, but subgroups are at most elementary abelian 2-groups | Negative multipliers bring order-reversing maps, thus at most implementing second-order flip structures |
| Non-negative recurrence + unsigned int $\mathrm{int}^p$ | Can implement $\mathbb{Z}/2^k\mathbb{Z}$ where $k\leq p$ | Wraparound addition $q\mapsto q+1\bmod 2^p$ directly provides cyclic counters |
| Unfixed evaluation order | Expressivity statements are no longer well-defined | Once non-associative float operations are reordered, the single-step recurrence itself may become a different function |

### Key Findings
- The greatest contribution is not a single new theorem, but the separation of "architecture, wiring, and arithmetic semantics," allowing previously conflicting conclusions to be compared within the same coordinate system.
- For finite-precision models, all induced transition monoids are finite; thus, recognizable languages are at most regular. To discuss non-regular capabilities, one must explicitly introduce precision, depth, or external resources that grow with sequence length.
- The diagonal SSM case study is enlightening: the same class of formal recurrence cannot perform even modular counting under non-negative floating-point semantics but can construct counters under unsigned integer wraparound semantics.

## Highlights & Insights
- The paper argues thoroughly that "numerical semantics are part of the model." Many expressivity proofs assume real algebraic identities hold, but rounding, overflow, and non-associativity in real floating-point systems change the transition function itself, which is particularly critical in long-sequence recurrence.
- The realized wreath product is a clean abstraction. It preserves the "lower layer controls the upper layer" hierarchical structure of deep RNNs while avoiding the inclusion of phantom monoids caused by unreachable inputs, making it suitable for precise non-expressibility proofs.
- The design of the acceptance core transforms the decoder from a temporary post-processor into a layer of the network cascade, allowing language recognition to strictly interface with the syntactic monoid. This small change makes the interface between machine-learning style RNNs and classical automata theory more natural.
- Inspiration for practical architecture design: If a task relies on stable counting or group structures, continuous formulas that "look like recurrence" are not enough; one must also check if the deployed numerical types truly support the corresponding algebraic cycles.

## Limitations & Future Work
- This paper analyzes existential expressivity rather than whether parameters can be learned. The fact that an architecture can algebraically express a language does not mean SGD will find the corresponding parameters.
- The framework primarily covers finite-precision semantics and thus falls naturally within the scope of regular languages. For models with precision growing with sequence length, external memory, or dynamic depth, extensions to infinite monoids or resource-sensitive versions are needed.
- The paper focuses on explicit recurrent architectures, particularly RNNs and diagonal SSMs. To incorporate Transformers into the same framework, they must first be formalized as some form of recurrent computation, which is not straightforward for all-attention models.
- The case study mainly re-analyzes known expressivity disputes of diagonal SSMs rather than systematically covering all modern sequence models. Future work could place linear attention, RWKV, RetNet, or chunked state-space implementations into the same algebraic template for comparison.

## Related Work & Insights
- **vs Siegelmann and Sontag-style RNN Turing completeness**: Those results usually depend on exact real numbers or infinite precision assumptions. This paper emphasizes that these assumptions do not automatically transfer to finite-precision deployment; the advantage is more reproducible semantics, at the cost of more conservative conclusions.
- **vs Merrill et al.'s analysis of finite-precision LMs**: Related work has pointed out that finite precision limits expressivity. This paper goes further by requiring the arithmetic model, evaluation order, and transition monoid to be made explicit, thereby writing limitations as checkable algebraic divisibility conditions.
- **vs Sarrof et al.'s results on diagonal SSM counting capabilities**: This paper reproduces and refines the limitations of non-negative diagonal SSMs while demonstrating that the same recurrence family can implement even modular counting under unsigned integer semantics, showing the core of the dispute lies in numerical semantics rather than the "SSM" label.
- **Insight**: When conducting language model theory, one should report implementation semantics with equal importance to architecture: numerical domain, rounding method, overflow, NaN handling, evaluation order, and whether compiler reordering is permitted. Otherwise, expressivity conclusions are likely to hold only for formulas on paper.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Unifying RNN expressivity disputes using monoid divisibility and wreath products provides a very clear theoretical perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ As a theoretical paper, the case studies are sufficient to support the claims; however, it lacks instantiated analysis of more modern architectures.
- Writing Quality: ⭐⭐⭐⭐☆ Rigorous structure and complete definitions, though the heavy algebraic background requires a certain threshold for readers unfamiliar with formal languages.
- Value: ⭐⭐⭐⭐⭐ High long-term reference value for RNN/SSM expressivity, finite-precision theory, and reproducible architectural analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Why Steering Works: Toward a Unified View of Language Model Parameter Dynamics](../../ACL2026/model_compression/why_steering_works_toward_a_unified_view_of_language_model_parameter_dynamics.md)
- [\[ICML 2026\] Procedural Pretraining: Warming Up Language Models with Abstract Data](procedural_pretraining_warming_up_language_models_with_abstract_data.md)
- [\[ICML 2026\] IDLM: Inverse-distilled Diffusion Language Models](idlm_inverse-distilled_diffusion_language_models.md)
- [\[ICML 2026\] Entropy-Aware On-Policy Distillation of Language Models](entropy-aware_on-policy_distillation_of_language_models.md)
- [\[ICML 2026\] WinQ: Accelerating Quantization-Aware Training of Language Models Around Saddle Points](winq_accelerating_quantization-aware_training_of_language_models_around_saddle_p.md)

</div>

<!-- RELATED:END -->
