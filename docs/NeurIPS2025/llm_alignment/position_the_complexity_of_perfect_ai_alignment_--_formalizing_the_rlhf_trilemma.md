---
title: >-
  [Paper Note] Position: The Complexity of Perfect AI Alignment -- Formalizing the RLHF Trilemma
description: >-
  [NeurIPS 2025][LLM Alignment][AI alignment] This paper formalizes the recurring safety–fairness–efficiency tensions in RLHF as an "alignment trilemma": it proves that no RLHF system can simultaneously satisfy $\varepsilon$-representativeness (faithfully reflecting diverse values), polynomial tractability (computational feasibility), and $\delta$-robustness (resistance to adversarial attacks), thereby providing a unified complexity-theoretic explanation for pathological phenomena such as preference collapse and sycophancy observed in current RLHF systems.
tags:
  - NeurIPS 2025
  - LLM Alignment
  - AI alignment
  - RLHF
  - trilemma
  - social choice theory
  - formal analysis
date: 2026-05-08
content_hash: 27a3478e43bb6546
---

# Position: The Complexity of Perfect AI Alignment -- Formalizing the RLHF Trilemma

**Conference**: NeurIPS 2025
**arXiv**: [2511.19504](https://arxiv.org/abs/2511.19504)
**Code**: None
**Area**: LLM Alignment
**Keywords**: AI alignment, RLHF, trilemma, social choice theory, formal analysis

## TL;DR

This paper formalizes the recurring safety–fairness–efficiency tensions in RLHF as an "alignment trilemma": it proves that no RLHF system can simultaneously satisfy $\varepsilon$-representativeness (faithfully reflecting diverse values), polynomial tractability (computational feasibility), and $\delta$-robustness (resistance to adversarial attacks), thereby providing a unified complexity-theoretic explanation for pathological phenomena such as preference collapse and sycophancy observed in current RLHF systems.

## Background & Motivation

- **Root Cause**: RLHF has become the dominant paradigm for aligning LLMs, yet three recurring failure modes persist in practice:
    - **Bias amplification**: RLHF models assign >99% probability weight to majority-group opinions, systematically erasing minority perspectives.
    - **Sycophantic behavior**: RLHF-trained assistants sacrifice truthfulness to cater to users' false beliefs (Sharma et al., 2024).
    - **Preference collapse**: A single scalar reward model is theoretically incapable of capturing multimodal human preferences (Chakraborty et al., 2024).
- **Limitations of Prior Work**: These failures are not engineering bugs but **computational inevitabilities**—current RLHF pipelines rely on only $10^3$–$10^4$ preference samples from homogeneous annotators (predominantly WEIRD populations), whereas genuine global representativeness requires $10^7$–$10^8$ samples.
- **State of the Field**: Existing patches (fairness regularization, adversarial training, post-hoc calibration) repeatedly hit the same ceiling, yet a unified theoretical explanation for why all fixes trade off along the same boundary remains absent.
- **Paper Goals**: As a position paper, this work shifts the discourse from "how to fix RLHF" to "which trade-offs we are willing to accept."

## Method

### Overall Architecture

The paper draws on **complexity theory + statistical learning theory + robust optimization** to formally define three desiderata for alignment and prove that they form an impossibility triangle.

| Property | Formal Definition | Intuition |
|---|---|---|
| $\varepsilon$-Representativeness | $\|\mathbb{E}_{h \sim \mathcal{H}}[V_h(\pi)] - \hat{V}(\pi)\| \leq \varepsilon$ | The reward model faithfully reflects diverse human preferences. |
| Polynomial Tractability | Sample complexity $m = \text{poly}(d, 1/\varepsilon, \log(1/\delta))$; computational complexity $\mathcal{O}(\text{poly}(m,d))$ | Gradient-based optimization completes in polynomial time. |
| $\delta$-Robustness | $\mathbb{P}_{a \sim \mathcal{A}}[\mathbb{E}_{h \sim \mathcal{H}}[V_h(\pi;a)] \geq V_{\min}] \geq 1 - \delta$ | Acceptable performance is maintained under distribution shift and adversarial attacks. |

**Core claim (informal)**: For sufficiently large populations $|\mathcal{H}| \to \infty$ and sufficiently rich adversarial spaces $|\mathcal{A}| \to \infty$, **any polynomially tractable alignment procedure cannot simultaneously achieve small $\varepsilon$ and small $\delta$**.

### Key Designs

1. **Formalization of the RLHF Trilemma**:
    - Design choices in the three-stage RLHF pipeline (SFT → reward modeling → PPO policy optimization) are mapped to the three vertices of the trilemma.
    - The aggregation mechanism in reward modeling, $r_\phi(\tau) \approx \sum_{i=1}^m w_i r_{\phi,i}(\tau)$ (with $w_i \propto \text{agreement}_i$), inherently amplifies majority views through weighted averaging.
    - The KL penalty $\beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$ constrains policy exploration, enhancing robustness at the cost of suppressing minority preferences.

2. **Impossibility Theorem**:
    - **Complexity lower bound**: Joint $(\varepsilon, \delta)$-alignment requires $\Omega(\kappa \cdot 2^{d_{\text{context}}} / (\varepsilon^2 n \delta))$ operations.
    - When the context dimension satisfies $d_{\text{context}} = \omega(\log n)$, this bound is **super-polynomial**.
    - Concretely, achieving global-scale alignment with $\varepsilon \leq 0.01$ (representativeness) and $\delta \leq 0.001$ (robustness) requires $\Omega(2^{d_{\text{context}}})$ operations.
    - **Key insight**: The growth rate of context dimensionality (culture, language, scenario, etc.) outpaces the scaling of computational resources.

3. **Alignment Complexity Analysis**—three dual-property sacrifice modes:

    | Preserved Properties | Sacrificed Property | Practical Manifestation | Current RLHF Status |
    |---|---|---|---|
    | Tractability + Robustness | **Representativeness** | Homogeneous annotators, majority voting, KL constraint → minority groups ignored | ε > 0.3–0.5; **dominant current practice** |
    | Representativeness + Tractability | **Robustness** | Expanding annotator diversity → adversarial poisoning with α ≈ 0.05 suffices to compromise the system | δ → 1 |
    | Representativeness + Robustness | **Tractability** | Minimax optimization $\max_\pi \min_a \mathbb{E}_h[V_h(\pi;a)]$ → NP-hard | Samples required: $\Omega(|\mathcal{A}| \cdot |\mathcal{H}| / \varepsilon^2)$ |

### Loss & Training

This paper is a theoretical position paper and proposes no new training algorithm. It analyzes the following key formulas from existing RLHF work:

- **Reward model loss**: $\mathcal{L}(\phi) = -\sum_{(a,b)} \log \sigma(r_\phi(\tau_a) - r_\phi(\tau_b))$ (Bradley–Terry model)
- **Policy optimization objective**: $\theta^* = \arg\max_\theta \{ \mathbb{E}_{\tau \sim \pi_\theta}[r_\phi(\tau)] - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) \}$
- **Diversity-expanded loss** (an attempt to broaden representativeness): $\mathcal{L}_{\text{diverse}}(\phi) = \sum_{g=1}^G w_g \sum_{(x_i, y_i^{\text{pref}}) \in D_g} -\log P_\phi(y_i^{\text{pref}} | x_i, \text{context}_g)$
- **Theoretically optimal objective** (intractable): $\pi^* = \arg\max_{\pi} \min_{a \in \mathcal{A}} \mathbb{E}_{h \sim \mathcal{H}}[V_h(\pi; \text{context}, t, a)]$

## Key Experimental Results

### Main Results

This paper is a position paper with no conventional experiments. The core "data" derive from complexity-theoretic analysis and quantitative comparison with existing RLHF practice:

| Metric | Current RLHF Practice | Theoretically Required | Gap |
|---|---|---|---|
| Preference samples | $10^3$–$10^4$ | $10^7$–$10^8$ (global representativeness) | 1000–10000× |
| Annotator diversity | Predominantly WEIRD populations | 180+ countries/cultures | Systematically absent |
| Representativeness ε | 0.3–0.5 | ≤ 0.01 | 30–50× |
| Robustness δ | 0.1–0.2 | ≤ 0.001 | 100–200× |
| Joint alignment compute | poly(n) | $\Omega(2^{d_{\text{context}}})$ | Super-polynomial |

### Ablation Study

No conventional ablation study. The paper explores the trade-off space by analyzing three "relaxation strategies":

| Relaxation Strategy | Approach | Cost | Applicable Scenario |
|---|---|---|---|
| Constrained representativeness | Focus on $K \approx 30$ core human-rights values vs. $K \approx 10^6$ cultural preference dimensions | Non-core cultural differences are ignored | General-purpose deployment |
| Bounded robustness scope | Test only $10^2$ realistic scenarios vs. $2^{100}$ theoretically possible cases | Cannot defend against unknown threats | Low-to-medium risk applications |
| Accepting super-polynomial cost | Train a single highly reliable system on $10^9$ samples | Extremely high computational cost | High-stakes domains: medical, legal, military |

### Key Findings

- **Scaling Wall**: When population size $n \gtrsim 10^6$ and context dimension $d_{\text{context}} \gtrsim 50$, a **phase transition** emerges—the computational demand for joint alignment jumps from polynomial to super-polynomial.
- **Brute-force scaling is ineffective**: A $10\times$ or $100\times$ increase in compute or data does **not** yield proportional gains in fairness and robustness, because the rate at which heterogeneity introduces adversarial attack surface exceeds the rate at which robustness scales.
- **Exponential leverage from dimensionality reduction**: Reducing effective $d_{\text{context}}$ by $2\times$ is equivalent to reducing computational cost by $10^9\times$, making algorithmic breakthroughs far more valuable than brute-force scaling.

## Highlights & Insights

- **Strong unifying explanatory power**: A single trilemma framework explains three seemingly distinct RLHF pathologies—preference collapse, sycophancy, and bias amplification—revealing them as different symptoms of the same computational bottleneck.
- **Reframing the problem**: Shifting from "how to fix RLHF" to "what we are willing to sacrifice" represents a paradigm shift with significant implications for industrial practice.
- **Quantified gap**: The paper clearly demonstrates the chasm between current practice ($10^3$ samples, $\varepsilon > 0.3$) and theoretical requirements ($10^7$ samples, $\varepsilon \leq 0.01$).
- **Actionable recommendations**: Three strategic relaxation directions are proposed (constraining representativeness, bounding robustness scope, accepting high cost) rather than vague calls for "further research."
- **Valuable research directions**: Modular value architectures, active learning of disagreement regions, and structured robustness constraints are identified as practically feasible avenues.

## Limitations & Future Work

- **Worst-case analysis**: The complexity lower bounds rely on worst-case reasoning and may overestimate actual costs in specific alignment scenarios; average-case complexity could be substantially lower.
- **Lack of empirical validation**: As a position paper, no experiments are conducted to locate current RLHF models in the trilemma space.
- **Missing quantitative thresholds**: The paper does not specify concrete thresholds for "sufficient" representativeness or robustness, leaving deployment decisions to case-by-case judgment.
- **Scope limited to RLHF**: Whether alternative alignment paradigms—Constitutional AI, Debate, DPO—face analogous trilemmas is not analyzed (the authors conjecture affirmatively but offer no proof).
- **Risk of misuse**: The impossibility result could be invoked by developers to excuse inadequate alignment efforts under the reasoning that "perfect alignment is impossible anyway."
- **Quality of approximate solutions unaddressed**: Under a relaxed standard of $\varepsilon = 0.1$ rather than $0.01$, it remains unclear whether the trilemma still holds and how the complexity changes.

## Related Work & Insights

- **MaxMin-RLHF** (Chakraborty et al., 2024): Proves the impossibility of a single reward model capturing multimodal preferences and proposes a mixed-reward approach optimizing worst-group utility—this paper generalizes that local result into a comprehensive trilemma.
- **Sycophancy research** (Sharma et al., 2024): Empirically demonstrates that RLHF induces sycophantic behavior—this paper interprets the phenomenon as an inevitable consequence of trading robustness for tractability.
- **Social choice theory**: The spirit of Arrow's impossibility theorem is extended here—from "no perfect voting rule exists" to "no perfect alignment procedure exists."
- **Insights**: For alignment research teams, investing in algorithmic innovations that reduce the effective dimensionality of $d_{\text{context}}$ (e.g., hierarchical value modeling, modular cultural adaptation) may be more productive than continuing to patch the RLHF pipeline.

## Rating

| Dimension | Score (1–10) | Remarks |
|---|:-:|---|
| Novelty | 8 | First work to unify multiple RLHF pathologies under a single complexity-theoretic framework with an impossibility proof. |
| Theoretical Depth | 7 | Formal definitions are clear and complexity lower bounds are convincing, but proofs rely on worst-case assumptions and tight bounds are absent. |
| Experimental Thoroughness | 6 | Strategic relaxation directions offer practical guidance, but concrete algorithms and empirical validation are lacking. |
| Writing Quality | 8 | Logic is clear and structure is well-organized; the transition from intuition to formalization is smooth and natural. |
| **Overall** | **7** | An excellent theoretical position paper that provides a valuable conceptual framework and complexity-theoretic perspective for alignment research, though follow-up empirical work is needed to ground the findings. |

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] A Systematic Evaluation of Preference Aggregation in Federated RLHF for Pluralistic Alignment of LLMs](a_systematic_evaluation_of_preference_aggregation_in_federated_rlhf_for_pluralis.md)
- [\[ICLR 2026\] Beyond RLHF and NLHF: Population-Proportional Alignment under an Axiomatic Framework](../../ICLR2026/llm_alignment/beyond_rlhf_and_nlhf_population-proportional_alignment_under_an_axiomatic_framew.md)
- [\[NeurIPS 2025\] Greedy Sampling Is Provably Efficient for RLHF](greedy_sampling_is_provably_efficient_for_rlhf.md)
- [\[NeurIPS 2025\] Provably Efficient Online RLHF with One-Pass Reward Modeling](provably_efficient_online_rlhf_with_one-pass_reward_modeling.md)
- [\[NeurIPS 2025\] Inference-time Alignment in Continuous Space](inference-time_alignment_in_continuous_space.md)

<!-- RELATED:END -->
