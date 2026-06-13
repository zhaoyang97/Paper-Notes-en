---
title: >-
  [Paper Note] Position: Adversarial ML for LLMs Is Not Making Any Progress
description: >-
  [ICML 2026][LLM/NLP][Adversarial ML] This position paper argues that adversarial machine learning in the LLM era deals with problems that are "harder to define, harder to solve…
tags:
  - "ICML 2026"
  - "LLM/NLP"
  - "Adversarial ML"
  - "LLM Safety"
  - "Jailbreaking"
  - "Prompt Injection"
  - "Evaluation Reproducibility"
date: 2026-05-08
content_hash: 74b2c914bc6a356a
---

# Position: Adversarial ML for LLMs Is Not Making Any Progress

**Conference**: ICML 2026  
**arXiv**: [2502.02260](https://arxiv.org/abs/2502.02260)  
**Code**: None (Position Paper)  
**Area**: LLM Safety / Adversarial Machine Learning / Evaluation Methodology  
**Keywords**: Adversarial ML, LLM Safety, Jailbreaking, Prompt Injection, Evaluation Reproducibility  

## TL;DR
This position paper argues that adversarial machine learning in the LLM era deals with problems that are "harder to define, harder to solve, and harder to evaluate" than traditional classifier scenarios. Having made slow progress on "toy problems" like $\ell_p$ robustness over the past decade, the field's full shift toward LLMs risks spending another decade without producing measurable or reproducible safety guarantees.

## Background & Motivation

**Background**: Adversarial ML originated from "small problems, large methods"—targets were narrow-task models like spam filters or CNNs on CIFAR/ImageNet. The typical threat model involved adding a perturbation $\ell_p \le \epsilon$ to the input to cause misclassification. Attack targets were optimized using first-order gradients of cross-entropy loss, and defense effectiveness was compared via test accuracy. Even in this ideal setting, the community spent a decade without truly solving $\ell_p$-bounded robustness, as numerous published empirical defenses were eventually broken by adaptive attacks (Carlini & Wagner 2017, Tramèr et al. 2020).

**Limitations of Prior Work**: As research focus shifts to LLMs, "safety" is no longer a formally defined task. Developers care about abstract properties like Helpfulness, Honesty, and Harmlessness (HHH); attackers aim to force "harmful" outputs; and threat models have expanded from "small perturbations" to "arbitrary prompts + fine-tuning + pruning." Sub-problems such as jailbreaking, prompt injection, unlearning, and membership inference simultaneously face three dilemmas: (a) attack success is difficult to determine, relying heavily on self-referential "LLM-as-judge" evaluations; (b) the attack search space is discrete, unbounded, and non-differentiable, with automated attacks generally underperforming manual red teaming; (c) the primary systems under test are closed-source, continuously updated APIs, making results impossible to reproduce.

**Key Challenge**: Traditional adversarial ML could be considered a "science" because the $\ell_p$ ball and classification accuracy provided a simplified but precisely defined, optimizable, and reproducible "necessary condition." In pursuing "real-world threats," LLM safety research proactively abandoned this formal framework without providing new measurable substitutes or certified defenses. Consequently, the community has developed a systemic illusion of "perceived progress" (new models being harder to jailbreak) while there is "no actual progress" (worst-case failure rates remain near 100%).

**Goal**: Systematically analyze the additional difficulties in LLM-era adversarial ML compared to traditional settings across three dimensions—defining, solving, and evaluating—and empirically demonstrate how these difficulties hinder cumulative scientific progress through six sub-field case studies (jailbreak, un-finetunable, poisoning, prompt injection, membership inference, unlearning).

**Key Insight**: The authors do not deny that LLM safety is a real problem but insist on distinguishing between "researching real-world security vulnerabilities" and "advancing the scientific understanding of adversarial ML." The latter must be built upon formalized, reproducible toy problems; if scaled-down sub-problems cannot be solved, "progress" on the fuzzy overall problem remains unfalsifiable.

**Core Idea**: In short, "solve definable sub-problems first before discussing safety." The authors call for the community to define minimal formal versions for each LLM safety direction—analogous to $\ell_p$-bounded perturbations—otherwise, looking back in ten years, we will still be unable to answer "how much have we actually progressed."

## Method

As a position paper, there are no traditional algorithms or training procedures. The "method" is an analytical framework for diagnosing the health of adversarial ML research, consisting of "Three Dimensions × Sub-challenges," using six LLM safety sub-domains as "cases" for comparative examination.

### Overall Architecture

The authors abstract the adversarial ML research process into a closed loop: "Define Problem → Solve Problem → Evaluate Results." Each stage identifies several sub-challenges that have significantly worsened since the classifier era. A challenge matrix (Table 1 in the paper) maps the six sub-domains to these challenges, revealing which directions have "collapsed across all dimensions" and which still have "one or two points of leverage." The diagnosis concludes that jailbreaking, prompt injection, poisoning, and unlearning are worse than traditional adversarial examples in almost all dimensions, while membership inference and un-finetunable models particularly collapse in the evaluation dimension.

### Key Designs

1. **Three Collapse Points in the "Defining" Dimension**:

    - **Function**: Characterizes the degradation of problem well-definedness from traditional to LLM settings.
    - **Mechanism**: The first collapse is the determination of attack success—classification only requires checking labels, while LLM "harmfulness" cannot be formalized, forcing the community back to "LLM-as-judge" oversight with circular dependencies. The second collapse is the boundary of the attack space—classifiers have geometric constraints like $\|x' - x\|_p \le \epsilon$, whereas any LLM input might trigger unsafe output; most jailbreak/prompt injection papers default to "unbounded" threat models and even grant attackers the ability to modify the model via fine-tuning or pruning. The third collapse is the boundary of training data—traditional IID train/test splits fail on trillion-token corpora, causing membership inference and unlearning to degrade from "containing a specific point" to "containing a specific concept," where the identity of samples loses meaning.
    - **Design Motivation**: Definition is the minimum threshold for science. If "what counts as a successful attack," "what counts as a legitimate attack," or "what counts as a training member" cannot be clearly stated, the "X% improvement" in subsequent papers loses its baseline for comparison.

2. **Two Collapse Points in the "Solving" Dimension**:

    - **Function**: Characterizes the degradation from differentiable optimization to ad-hoc empirical attacks.
    - **Mechanism**: The first collapse in solving is the attack search—in classifier scenarios, input-gradient attacks like PGD/CW consistently outperform humans; attackers simply move toward $\nabla_x \mathcal{L}$. LLMs' discrete token space renders gradient methods less effective (representative methods like GCG generate gibberish strings with little improvement over random search), while truly strong attacks like persona modulation, multi-turn dialogues, and social engineering rely entirely on manual red teaming, meaning "worst-case performance" cannot be computationally approximated. The second collapse is the defense principle—classifier scenarios have certified defenses like randomized smoothing and empirical defenses with formal objectives like adversarial training. LLM defenses are almost entirely (i) adversarial fine-tuning against known attacks, (ii) latent space virtual adversarial training, (iii) external LLM classifiers like Llama Guard, or (iv) input random preprocessing—all of which lack formal characterization of "what is being defended" and have been repeatedly bypassed by new attacks (e.g., Łucki et al. 2024).
    - **Design Motivation**: The key metric for whether adversarial ML is "solving the problem" is whether the system holds up under the strongest attack; when the strongest attack relies on humans and defenses cannot be formalized, the solving process becomes a subjective guessing game that fails to converge.

3. **Two Collapse Points in the "Evaluating" Dimension**:

    - **Function**: Characterizes the degradation of harm metrics and benchmark reproducibility.
    - **Mechanism**: The first collapse in evaluation is the measurement of harm and utility—traditional tasks use misclassification rates and clean accuracy to measure attack and utility simultaneously. LLMs must rely on LLM-as-judge to determine if output is harmful, but the judge itself can be attacked by prompts (Mangaokar et al. 2024), may misjudge "any non-refusal" as success (Souly et al. 2024), and exhibits pro-defense bias when evaluating defenses based on similar discriminators. Simultaneously, there is no standard answer for "is the model still useful," as a trivial defense that refuses all queries would be perfect on safety but destroy all utility. The second collapse is reproducibility—major targets like GPT-4 or Claude are closed-source and updated silently; an attack prompt may fail a week later. Success rates reported in many papers cannot be independently verified, essentially shooting at a moving target, preventing the community from accumulating comparable results over time.
    - **Design Motivation**: The scientific community can only accumulate progress on reproducible benchmarks. When evaluation is neither objective nor stable, "this year's SOTA" is just a paper title rather than proof of a safer system.

### Loss & Training
This paper does not involve traditional training objectives. The authors' "normative recommendation" is equivalent to a meta-loss: explicitly categorize papers as either "studying real-world vulnerabilities" or "advancing the scientific understanding of adversarial ML." The former allows fuzzy evaluation but must emphasize specific harms, while the latter must restrict itself to formalized toy sub-problems (e.g., fixed-length suffix jailbreaks, bounded edit-distance prompt injections) for adaptive evaluation; otherwise, it should not be viewed as a scientific contribution.

## Key Experimental Results

As a position paper, there are no quantitative comparative experiments. The authors substitute a "main table" with six sub-domain case studies and one challenge matrix. The following summarizes Table 1 and Section 3 of the paper.

### Main Results: Collapse Matrix of 6 Sub-domains × 7 Sub-challenges

| Sub-domain | Define Success | Unbounded Space | Fuzzy Data Boundary | Hard Attack Search | No Defense Principle | Utility Hard to Measure | Poor Reproducibility |
|---|---|---|---|---|---|---|---|
| Jailbreaks | ✓ | ✓ | — | ✓ | ✓ | ✓ | ✓ |
| Un-finetunable Models | ✓ | ✓ | — | ✓ | ✓ | ✓ | — |
| Poisoning & Backdoors | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Prompt Injections | ✓ | ✓ | — | ✓ | ✓ | ✓ | ✓ |
| Membership Inference | ✓ | — | ✓ | — | — | ✓ | — |
| Unlearning | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |

Interpretation: ✓ indicates that the sub-domain has been significantly worsened by LLM-ification in that dimension, losing its original scientific leverage compared to the classifier era. All six directions have collapsed in at least four dimensions, with jailbreaking, poisoning, prompt injection, and unlearning collapsing in nearly all dimensions.

### Ablation Study: Key Capabilities Comparison (LLM Era vs. Classifier Era)

| Setting | Attack Goal | Attack Space | Source of Strongest Attack | Evaluator | Reproducibility |
|---|---|---|---|---|---|
| Classic $\ell_p$ Adv. Examples | Misclassification (Clear) | $\|x'-x\|_p \le \epsilon$ | White-box PGD/CW (Auto) | Test Accuracy | Public Data + Weights |
| LLM Jailbreaking | Output "Harmful" (Subjective) | Arbitrary tokens + Fine-tune/Pruning | Human Red Team (Manual) | LLM-as-judge | Closed API Updates |
| LLM Unlearning | Erase "A Concept" | Arbitrary prompts + White-box intervention | Adaptive fine-tuning | Hard to isolate utility | Retraining infeasible |

### Key Findings

- The most critical degradation is not that the "problem is harder," but that the "standard of success has vanished"—the circular dependency of LLM-as-judge allows both attackers and defenders to game scores by attacking the judge.
- Counter-intuitively, manual attacks on LLMs remain stronger than automated optimization over the long term (GCG results are close to simple random search). This contrasts with the image adversarial era where automated white-box attacks dominated, implying that "worst-case" has no computable upper bound.
- Newer models being harder to jailbreak does not equal "progress in safety"—evaluation capabilities themselves are degrading, and it is more likely that the tools to find failure cases are being lost.
- Using a evaluator from the same family as the defense leads to artificially high scores (e.g., output-filtering defenses in Liu et al. 2024), constituting structural benchmark contamination.

## Highlights & Insights

- The "Defining—Solving—Evaluating" framework and the sub-challenge matrix unify scattered pain points from jailbreaking, unlearning, and MI communities into a single "physical exam." This provides a rare horizontal diagnostic perspective to check if a safety paper is "vulnerable" in specific dimensions.
- The distinction between "studying real-world vulnerabilities" and "advancing scientific understanding" is highly practical—it separates "demo attacks on GPT-4" from "proving a defense's robust radius," preventing researchers from criticizing one type of work using the metric of the other.
- Observations regarding the circular dependency of LLM-as-judge and evaluator-defense bias highlight blind spots in current safety benchmarks that are widely used but rarely formalized. These insights are transferable to RLHF evaluation and red team benchmark design.
- Adapting the "necessary condition" approach from $\ell_p$ balls to LLM safety—such as "detectability of fixed-length suffix jailbreaks" or "defensibility of bounded sentence modification in prompt injection"—provides a grounded research agenda capable of incubating formalized sub-tasks.

## Limitations & Future Work

- The authors acknowledge counterarguments: increased complexity might be the price of "finally solving real problems." Jailbreaking's "never output X regardless of context" might actually be simpler than "correctly classify a guacamole vs. a cat misclassified as guacamole." The paper’s response (citing examples where representation engineering was also broken) is qualitative and lacks quantitative criteria to distinguish between "temporarily unsolved" and "unsolvable in principle."
- The paper is primarily "critique + appeal" and does not propose specific LLM safety toy benchmarks. Readers might ask what should replace HarmBench or JailbreakBench; the authors suggest a direction ("formalized sub-tasks like $\ell_p$") but offer no candidate designs.
- Coverage is biased toward jailbreaking/unlearning, with LLM agent safety (tool calling, multi-agent protocol attacks) only briefly mentioned in the prompt injection section. The analysis of degradation in the agentic world is insufficient.
- The article assumes "formalization = cumulative science," but formalization does not automatically equal "closeness to real-world harm." Whether a decade of $\ell_p$ robustness research meaningfully improved the safety of real-world facial recognition systems remains an open question, making its use as a template for LLM safety debatable.
- **Future Directions**: Propose a minimal formal benchmark for each sub-field (e.g., "jailbreak rate for suffixes of length $\le k$") with associated adaptive evaluation protocols to make the paper's call to action operational.

## Related Work & Insights

- **vs. Carlini & Wagner 2017 / Tramèr et al. 2020 on "Broken Defenses"**: These are the classic warnings from the image era. This paper follows the tradition that "empirical defenses must withstand adaptive attacks to be valid," extending it to LLMs where even what constitutes an adaptive attack lacks consensus.
- **vs. HarmBench / JailbreakBench**: These works attempt to standardize "harm scores," whereas this paper points out the circular dependency of standardization itself. It does not deny their short-term necessity but calls for additional formalized toy sub-tasks as scientific coordinates.
- **vs. Representation Engineering / Circuit Breakers**: The authors compare these "latent direction identification" defenses to detection-based defenses in the image era, warning that detection methods typically fail collectively against new attacks. The insight is that any defense relying on implicit representations requires rigorous adaptive evaluation.
- **vs. Cooper et al. 2024 (Unlearning Position Paper)**: Both are pessimistic about LLM unlearning. This paper supplements the argument using the adversarial ML triad, highlighting that "concept-level unlearning" and "sample-level unlearning" are conflated and share failure modes with jailbreaking and prompt injection.
- **Insight**: Future papers should explicitly state whether they belong to "vulnerability demos" or "scientific research." The latter should include a formal definition + an adaptive attack protocol to avoid being dismissed as "fuzzy."

## Rating
- **Novelty**: ⭐⭐⭐⭐ The framework (triad + matrix) is highly integrated, though specific views have existed separately in sub-communities.
- **Experimental Thoroughness**: ⭐⭐⭐ As a position paper, there are no experiments. The six cases cover typical directions but vary in depth; jailbreaking/unlearning are deep, while poisoning/MI are more of an overview.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear argumentation with both sides represented (Section 4 explores opposing views). A rare "constructively pessimistic" position paper.
- **Value**: ⭐⭐⭐⭐⭐ Provides a mirror for the LLM safety community, offering substantial guidance for peer review standards and future research directions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] When Stability Fails: Hidden Failure Modes of LLMs in Data-Constrained Scientific Decision-Making](../../ICLR2026/llm_nlp/when_stability_fails_hidden_failure_modes_of_llms_in_data-constrained_scientific.md)
- [\[ICML 2026\] Position: The Turing-Completeness of Autoregressive Transformers Relies Heavily on Context Management](position_the_turing-completeness_of_autoregressive_transformers_relies_heavily_o.md)
- [\[ICML 2026\] Express Your Doubts: Probabilistic World Modeling Should Not Be Based on Token logprobs](express_your_doubts_--_probabilistic_world_modeling_should_not_be_based_on_token.md)
- [\[ICML 2026\] Differential Syntactic and Semantic Encoding in LLMs](differential_syntactic_and_semantic_encoding_in_llms.md)
- [\[AAAI 2026\] Position on LLM-Assisted Peer Review: Addressing Reviewer Gap through Mentoring and Feedback](../../AAAI2026/llm_nlp/position_on_llm-assisted_peer_review_addressing_reviewer_gap_through_mentoring_a.md)

</div>

<!-- RELATED:END -->
