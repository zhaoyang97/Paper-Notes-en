---
title: >-
  [Paper Note] Position: Ideas Should be the Center of Machine Learning Research
description: >-
  [ICML 2026][Interpretability][Ideas First] The authors propose the "Ideas First" stance: treating "idea $\rightarrow$ observable signature $\rightarrow$ tailored experiment" as the core unit for evaluating machine learni…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Ideas First"
  - "signature"
  - "tailored experiments"
  - "benchmark myth"
  - "computational fairness"
date: 2026-05-08
content_hash: ddbe8dfc4c19f99e
---

# Position: Ideas Should be the Center of Machine Learning Research

**Conference**: ICML 2026  
**arXiv**: [2605.15253](https://arxiv.org/abs/2605.15253)  
**Code**: None  
**Area**: Position Paper / ML Research Methodology / Philosophy of Science  
**Keywords**: Ideas First, signature, tailored experiments, benchmark myth, computational fairness  

## TL;DR
The authors propose the "Ideas First" stance: treating "idea $\rightarrow$ observable signature $\rightarrow$ tailored experiment" as the core unit for evaluating machine learning research. This approach aims to oppose the pursuit of leaderboard numbers or idealized theorems as ends in themselves, thereby bridging the theory-practice gap and lowering the barrier for entry for researchers with limited computational resources.

## Background & Motivation

**Background**: Current ML research is split into two mainstream modes: Mode A (Benchmark-driven Engineering), which defines contributions through a single metric by scaling models, modifying data, or tuning architectures; and Mode B (Idealized Theory), which proves theorems under highly idealized settings such as infinite width, infinitesimal step sizes, or separable data. Both have produced real progress (AlexNet/ResNet/CLIP/GPT-3 vs. NTK/margin bound/benign overfitting), but both increasingly serve as "barriers" rather than "tools."

**Limitations of Prior Work**: (1) **Benchmark myopia** — metric improvements often cannot be attributed to mechanisms, and conclusions become unreadable when multiple changes cancel each other out; (2) **Transfer gap** — idealized theorems rarely provide observable predictions, rendering theory a "post-hoc explanation" rather than a "pre-hoc measurement guide"; (3) **Non-cumulative findings** — exploratory ablations are not anchored to hypotheses, making results difficult to reuse in subsequent work; (4) **Complexity premium** — reviewers equate "complex $\approx$ rigorous," while simple yet sharp ideas are dismissed as "not deep enough"; (5) **Resource asymmetry** — the implicit compute threshold for SOTA excludes researchers without massive clusters ("compute divide" / "Red AI").

**Key Challenge**: The true carrier of scientific value is the *idea* (hypotheses about how learning systems work), whereas the current system treats proxies for value (benchmark scores / abstract theorems) as the value itself. This leads to peer review suppressing both "mechanistic clarification" and "marginalized groups."

**Goal**: (i) Define a framework to translate abstract ideas into falsifiable empirical measurements; (ii) argue that idea-centric evaluation is both scientific and fair; (iii) provide a specific "field guide" for authors and reviewers.

**Key Insight**: Drawing from the "hypothesis-driven" paradigm in physics and biology — instead of asking "what works?" or "what must be true under idealization?", researchers should address the neglected middle question: "If this mechanistic hypothesis were correct, what observable traces should be seen in a real model?"

**Core Idea**: Shift the center of research from "systems/theorems" to the tripartite chain of "idea $\rightarrow$ signature $\rightarrow$ tailored experiment" — benchmarks and theorems are demoted to testing tools rather than evaluation ends.

## Method

As a position paper, the "method" refers to the proposed research framework (Ideas First) and its accompanying guidelines for writing, reviewing, and case analysis. This is organized below as a complete pipeline.

### Overall Architecture

The authors characterize an idea-centric work as a tripartite chain: $\text{idea} \rightarrow \text{signature} \rightarrow \text{tailored experiment}$. This starts from a "hypothesis," translates it into "what measurable traces should be observed in complex models," and then designs experiments specifically to find or falsify those traces. In this view, benchmarks are no longer "podiums" but "microscopes"; theorems are not "verdicts" but "telescopes." Their legitimacy stems from whether they help expose or debunk an idea.

The input is a research concept that has yet to be rigorously evaluated. The intermediate products are (i) a set of explicitly stated observable signatures and (ii) experiments/controls specifically targeting those signatures. The output is a clear judgment on "whether the signal appeared" (present, absent, or partially present suggesting new boundaries) — rather than "I outperformed the SOTA by X%."

### Key Designs

1.  **Idea: A scope-bearing claim**:
    - **Function**: Condenses "vague intuition" into a scientific statement that can be falsified in the future, clarifying "under what settings I believe this holds, and under what settings I admit it might fail."
    - **Mechanism**: A valid idea must satisfy three criteria: (a) describe the mechanism in one or two sentences, (b) provide a scope (range of architectures, data scales, training phases, or theoretical assumptions), and (c) list at least one plausible failure mode. It usually takes shape in simplified settings (single-layer attention / infinite width / synthetic data) where controlled analysis is possible. The value of an idea lies in its conceptual and predictive nature rather than its direct score. The paper uses NTK, implicit max-margin bias, and Mixup as real-world examples to demonstrate this "statement + scope" format.
    - **Design Motivation**: Current reviews often default to "no theorem = no idea" or "no LLM = no idea," which the authors argue mistakes the "carrier" for the "content." Requiring internal scope and failure modes can block both "slogan-style" speculation (Avoid: "letting the idea be defined only via a specific experiment, benchmark gain, or theorem statement") and "high-dimensional theorems with no meaning for real models."

2.  **Signature: Translating abstract mechanisms into measurable traces**:
    - **Function**: If an idea holds in simplified settings and is to be "valid" when transferred to modern complex models, it must first be determined "in what form it will appear" — such as geometric features, trends in training dynamics, causal responses, invariances, thresholds, or characteristic error patterns. The signature is the interface language between the idea and the experiment.
    - **Mechanism**: A good signature answers three questions: (a) what quantity to measure (margin distribution? cosine similarity of a layer? logit curve along interpolation paths?), (b) where to measure it (which layer / training stage / scale segment), and (c) what trend or threshold is expected (usually monotonic? trend weakening after a certain width?). The paper emphasizes that real systems are "noisy and heterogeneous," so signatures must be evaluated "in expectation / as coarse-grained trends," allowing for bounded exceptions. The criterion is whether the "predicted shape is visible after reasonable aggregation," not whether "every point fits." Section 4 provides three examples: the signature of NTK is "predictions/loss trajectories in early training fit linearized models at large widths"; the signature of max-margin is "normalized margin in the penultimate-layer grows monotonically and aligns with the SVM solution after training error hits zero"; the signature of Mixup is "logits are approximately linear along interpolation coefficient $\lambda$, and memorization decreases under label noise."
    - **Design Motivation**: Measuring an idea with vague slogans like "better generalization" is not falsifiable. Requiring authors to pin down the idea into a "shape to look for" during the writing stage provides a target for subsequent experiments and allows others to reproduce or challenge the signal — this is key to converting "exploratory ablation" into "hypothesis-driven tests" and solving the problem of non-cumulative findings.

3.  **Tailored experiment: Experimental design for the purpose of "seeing the signal"**:
    - **Function**: The success criterion for experiments is explicitly shifted from "gaining points" to "clearly seeing (or confirming the absence of, under proper control) the predicted pattern."
    - **Mechanism**: The authors provide a five-step procedure: (i) define the signature as a distinguishable statistic or visualization; (ii) choose "instruments" capable of exposing it (measurement, patching ablations, counterfactuals); (iii) place measurements where the idea predicts the signature will be strongest (specific layers / training windows / scale segments); (iv) sweep the knob the idea claims will modulate the signal and include a negative control where the idea claims "there will be no effect"; (v) report qualitative trends, thresholds, and failure cases that refine the scope. Section 6 uses "Topic Inertia in LLMs" for a hypothetical case study — simplified analysis of single-layer unified-KQ attention suggests "the longer the prompt, the higher the semantic similarity trend between generation and prompt." Length is then swept from 10-200 tokens on LLaMA-2 / GPT-NeoX-20B / MPT-7B, with RNN as a negative control (no attention $\rightarrow$ signature should not appear). The signal was visible on all attention models and absent on RNNs. Section 6.1 also rehearses two types of reviewer counterarguments and provides "defensive responses": (a) To "why not use GPT-5.1 / LongBench?": when the signal is clearly distinguishable at 200 tokens, scaling only increases compute barriers without adding mechanistic insight; (b) To "why does the trend fluctuate instead of being strictly monotonic?": treating signatures as mathematical laws is a misunderstanding; real data is noisy, and the key is that negative controls exclude general artifacts.
    - **Design Motivation**: This step is the framework's most direct response to "complexity premium" and "resource asymmetry." When evaluation criteria shift to "is the mechanism visible," "simple but sharp experiments" automatically gain legitimacy. Work using small models / data / compute is no longer rejected for "lack of scale," while authors are encouraged to report failures and boundaries (refining scope), which is the base of cumulative science.

### Loss & Training

This paper does not involve model training. As a counterpart, the "Field Guide" in Section 5 provides an "Aim / Avoid" checklist for both the author side (Specifying the idea / Defining signatures / Designing tailored experiments) and the reviewer side (Evaluating the idea / signatures / experiments). This can be viewed as an operational guide for translating the framework into the "training objectives" of peer review: reviewers should judge the clarity and scope of the idea, the measurability of the signature, and the alignment of experiments with the signature, rather than defaulting to requests for larger models, more benchmarks, or more comprehensive theorems.

## Key Experimental Results

As a position paper, there are no quantitative results, but the authors support the operability of their framework through three types of "cases + counter-cases." These are organized in the tables below.

### Main Results: Can existing research be restated using the Ideas First framework?

| Case | Idea (Simplified Setting) | Signature (Expected in Complex Models) | Tailored Experiment | Source |
| :--- | :--- | :--- | :--- | :--- |
| NTK Linearized Training | Under infinite width + square loss, GD $\equiv$ kernel regression on NTK frozen at initialization | In early training, network prediction/loss trajectories closely follow the NTK-at-init linearized model; fit improves with width, fails as training progresses | Comparative alignment of "full training vs. linearized prediction" across widths/epochs on CNN/ResNet/WRN using CIFAR | Jacot+2018 / Lee+2019 |
| Implicit Max-margin Bias | Under separable data + exponential-tail loss, GD makes weight directions converge to hard-margin SVM; similar for homogeneous nets | After training error hits zero, penultimate-layer normalized margin continues to rise monotonically; classifier direction aligns with SVM solution | Tracking normalized margin and SVM alignment after interpolation using MLP/CNN on MNIST/CIFAR | Soudry+2018 / Lyu+Li 2020 |
| Mixup (Heuristic) | Decision regions are "straightened" along sample interpolation segments; logits change smoothly with mixing coefficients | Along the path $x_\lambda = \lambda x_i + (1-\lambda) x_j$, logits are approximately linear w.r.t. $\lambda$; memorization weakens under label noise; gradients are smoother between samples | Sweeping $\lambda$ on real vision/speech data; stress tests with corrupted labels; reporting interpolation linearity + anti-memorization | Zhang+2018 |

### Ablation Study: Which pain points of standard modes does this framework solve?

| Standard Mode Pain Point | Standard Mode Symptom | Ideas First Resolution | Manifestation in Case Study |
| :--- | :--- | :--- | :--- |
| Benchmark myopia (Mode A) | Multiple changes entered simultaneously; single numbers lack attribution | Shift evaluation from score differences to "whether the signature is visible / falsified" | Topic Inertia uses similarity trends as the sole main result, reporting no leaderboard |
| Transfer gap (Mode B) | Idealized theorems provide no measurable predictions | Mandate that ideas must be translated into signatures before experimentation | Single-layer unified-KQ attention conclusions translated into "longer prompt $\rightarrow$ similarity trend $\uparrow$" |
| Non-cumulative findings | Exploratory ablations are not anchored to hypotheses | Experimental design targets signatures + mandatory negative controls | RNN baseline serves as a negative control for absence of attention |
| Complexity premium | "Complex $\approx$ rigorous" reviewer bias | Encourage "minimal experiments capable of presenting the signature" | 200 tokens suffice to distinguish signals; no need to chase GPT-5.1 |
| Resource asymmetry | Compute threshold excludes small groups | Explicitly allow clear experiments at a modest scale | All experiments used open-source weights (LLaMA-2 / GPT-NeoX / MPT-7B) |

### Key Findings

- The "operability" of the framework primarily comes from the **signature being a mandatory interface between the idea and the experiment**: ideas without signatures cannot be grounded, and experiments without signature anchors degenerate into aimless ablations — the authors repeatedly emphasize that "the unit of explanation and value is signatures, not ranks."
- The greatest demonstrative value of the Topic Inertia case study lies not in the results themselves, but in the **rehearsed rebuttals and responses in Section 6.1** — it explains the most common reasons for rejection, such as "why not use larger models / more metrics," as specific manifestations of the *complexity premium* and provides two principled counter-strikes: "signatures are already distinguishable at 200 tokens" and "signatures are not mathematical laws."
- The three illustrative examples (NTK / max-margin / Mixup) are already widely accepted works. By *retrospectively* rewriting them into the "idea $\rightarrow$ signature $\rightarrow$ tailored experiment" format, the authors argue that Ideas First **is not a radical new norm, but a formalization and teachable version of paradigms that "good work has always followed."**
- Most notable negative boundary: For the risk of "signatures being unconsciously tuned to match expectations," the framework's antidote is *negative controls + pre-declared scope/failure modes*; the authors admit this is harder to standardize than "strict benchmarks," which is an objection explicitly listed in Section 8.

## Highlights & Insights

- Using "signature" as a mandatory interface between idea and experiment is the sharpest design of this position — it cuts through the common maladies of "claiming contribution through a vague intuition" and "retrofitting stories after a pile of ablations," forcing researchers to commit to "what quantity I intend for others to observe" while stating their hypothesis.
- Explicitly placing "complexity premium" and "resource asymmetry" within epistemology rather than just ethics is a powerful move — the authors argue that "low-compute experiments are justified not as a concession to fairness, but because they possess superior evidentiary power when isolating mechanisms," transforming frugal AI from a "moral slogan" into a "methodological requirement."
- The "rehearsed reviewer rebuttals + principled responses" in Section 6.1 are highly reusable — any author wishing to conduct mechanistic research at a low compute scale can use this template to internalize common queries like "why not scale up" into scope arguments rather than responding passively.

## Limitations & Future Work

- Author-identified (Section 8): (i) Targeted experiments are more prone to "unconscious tuning" to match expectations than broad benchmarks; the framework can only mitigate this via negative controls and pre-registered scopes, lacking institutional guarantees; (ii) this stance will not replace theory or benchmarks, but rather fill the middle layer between "idealized theorems" and "benchmark-driven" research; (iii) in engineering-led ML subfields (e.g., production deployment), the criterion of "signature visibility" may yield to "system utility."
- Additional insights: (i) While the "Aim / Avoid" checklist in the Field Guide is clear, it lacks quantitative or semi-structured evaluation criteria for "signature strength"; implementation in specific review scenarios still relies heavily on individual judgment, risking "old wine in new bottles"; (ii) Topic Inertia serves only as a *hypothetical* case study without real 200-doc datasets or detailed numerical tables, so the evidence for "replicability of the framework" is more rhetorical; (iii) the paper does not discuss how to arbitrate when "signatures compete" — when an idea predicts multiple signatures and some appear while others don't, should the scope be rewritten? This decision criterion remains a gap in the framework.
- Specific improvement ideas: A supplementary "signature card" template (similar to model cards / dataset datasheets) could be introduced for authors to structurally disclose "what to measure, where, expected trends, negative controls, and known failure regimes," turning the current "soft call" into "hard disclosure" to significantly reduce review inconsistency.

## Related Work & Insights

- **vs. Lipton & Steinhardt (2018) / Dehghani et al. (2021) (Benchmark Critique School)**: They documented issues like benchmark lotteries and the illusion of complexity. This paper goes beyond "pointing out problems" by providing a *positive alternative* (idea $\rightarrow$ signature $\rightarrow$ tailored experiment) — this is the primary distinction.
- **vs. Baraniuk et al. (2020) / Drori et al. (2022) (Science of Deep Learning School)**: They advocate for more rigorous scientific methods in studying deep networks. This paper goes a step further methodologically by grounding "science" in the signature interface and explicitly binding "sufficient experimentation" with scope, avoiding the "toy model" criticism often leveled at this school.
- **vs. Strubell et al. (2019) / Ahmed & Wahed (2020) (Green AI / Compute Divide School)**: They oppose Red AI from ethical and environmental perspectives. This paper further embeds "low-compute experiments" into epistemology, making frugal AI not only "allowable" but "the superior form of evidence for certain mechanistic questions."
- **Insights**: (i) When writing papers, authors can explicitly add a "signatures we will look for" section at the end of the Intro to replace traditional "contributions" lists; (ii) before performing an ablation, one should ask "which negative control of which signature of which idea does this correspond to" — any ablation that cannot answer this can be removed; (iii) when reviewing others' papers, "whether the author declared scope and failure modes" can serve as the first filter.

## Rating
- Novelty: ⭐⭐⭐⭐ Each component (critiquing benchmarks, advocating hypothesis-driven work, focusing on compute fairness) is not new individually, but using "signature" as an interface language combined with case studies and defensive rehearsals is a complete and operational integration.
- Experimental Thoroughness: ⭐⭐⭐ As a position paper, it uses three real historical cases (NTK/max-margin/Mixup) + one hypothetical case study (Topic Inertia) to demonstrate groundability, but lacks real small-scale reproduction data to harden the demonstration.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure (critique $\rightarrow$ framework $\rightarrow$ examples $\rightarrow$ field guide $\rightarrow$ case study $\rightarrow$ discussion $\rightarrow$ alternative views). Making "Alternative Views" a standalone section and honestly responding to opposing stances shows commendable rhetorical discipline for a position paper.
- Value: ⭐⭐⭐⭐ For researchers conducting low-compute mechanistic research who are repeatedly rejected for "lack of scale," this paper can almost serve directly as a rebuttal template. For senior PIs and program chairs, it provides a citable roadmap for embedding "frugal AI" into review norms.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Position: Zeroth-Order Optimization in Deep Learning Is Underexplored, Not Underpowered](position_zeroth-order_optimization_in_deep_learning_is_underexplored_not_underpo.md)
- [\[ICLR 2026\] When Machine Learning Gets Personal: Evaluating Prediction and Explanation](../../ICLR2026/interpretability/when_machine_learning_gets_personal_evaluating_prediction_and_explanation.md)
- [\[ICML 2026\] Position: Let's Develop Data Probes to Fundamentally Understand How Data Affects LLM Performance](position_lets_develop_data_probes_to_fundamentally_understand_how_data_affects_l.md)
- [\[ICML 2026\] Provably Learning Attention with Queries](provably_learning_attention_with_queries.md)
- [\[ICML 2026\] Universal 1/3 Time Scaling in Learning Peaked Distributions](universal_one-third_time_scaling_in_learning_peaked_distributions.md)

</div>

<!-- RELATED:END -->
