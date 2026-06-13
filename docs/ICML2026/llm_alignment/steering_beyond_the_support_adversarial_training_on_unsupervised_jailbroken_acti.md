---
title: >-
  [Paper Note] Steering Beyond the Support: Adversarial Training on Unsupervised Jailbroken Activation Simulation
description: >-
  [ICML 2026][LLM Alignment][safety steering] The paper addresses the failure of supervised safety steering on unseen jailbreak attacks. It proposes "unsupervised latent direction discovery + bi-level adversarial training"…
tags:
  - "ICML 2026"
  - "LLM Alignment"
  - "safety steering"
  - "jailbreak defense"
  - "unsupervised latent direction"
  - "bi-level adversarial training"
  - "optimal transport"
date: 2026-05-08
content_hash: d642059520ac2473
---

# Steering Beyond the Support: Adversarial Training on Unsupervised Jailbroken Activation Simulation

**Conference**: ICML 2026  
**arXiv**: [2605.24535](https://arxiv.org/abs/2605.24535)  
**Code**: None  
**Area**: LLM Safety / Jailbreak Defense / Activation Steering / Adversarial Training  
**Keywords**: safety steering, jailbreak defense, unsupervised latent direction, bi-level adversarial training, optimal transport

## TL;DR
The paper addresses the failure of supervised safety steering on unseen jailbreak attacks. It proposes "unsupervised latent direction discovery + bi-level adversarial training" to simulate novel jailbroken states from scratch in the activation space. These simulated states are treated as adversarial examples to train an OT potential function (whose gradient constitutes a spatially-varying steering field). Across three LLMs and six classic jailbreak categories, the method reduces attack success rates to mostly $<5\%$ with minimal impact on benign utility.

## Background & Motivation

**Background**: Safety steering is currently a mainstream lightweight solution for LLM jailbreak defense. It avoids retraining the model by performing test-time interventions on hidden activations (e.g., adding refusal directions linearly, conditional steering, null-space constraints). It aims to push activations that trigger harmful responses back into refusal regions while preserving the functionality of benign inputs.

**Limitations of Prior Work**: All such methods are **supervised**, requiring prior collection of paired "benign / known jailbreak" activations as training support. However, real-world jailbreaks evolve constantly (GCG suffixes, AutoDAN, PAIR, TAP, FewShot, etc.), and training sets represent only the tip of the iceberg. Empirical evidence using AlphaSteer on Mistral-v2-7B (Tab. 1) shows that when trained on only a subset of jailbreak families, the SR on in-distribution attacks can be reduced to 1.8–8.4, but it immediately rebounds to $30\%+$ for unseen families. Supervised steering essentially performs local corrections around observed jailbreaks.

**Key Challenge**: Training support is **static and limited**, whereas true jailbroken activation distributions are **open and OOD**. One must either use global intervention (which hurts benign utility) or conditional intervention (which fails to generalize), creating a fundamental trade-off.

**Goal**: (1) Expand training support zero-shot without relying on jailbreak labels; (2) Achieve near-zero intervention on benign activations and strong intervention on jailbroken ones; (3) Allow the steering mechanism to evolve continuously to keep pace with unseen attacks.

**Key Insight**: The authors notice *unsupervised latent direction discovery* (ULDD; Mack & Turner 2024)—using only unlabeled prompts can identify a set of directions $V \in \mathbb{R}^{d\times K}$ in shallow layers. When injected, these can **causally** alter deep-layer activations and induce various behaviors (changing language, tone, or even accidentally triggering jailbreaks). Tab. 2 shows these directions can generate dozens to hundreds of successful "simulated jailbreaks" on LLaMA-3-8B / Mistral-v2-7B / Qwen-2.5-7B, with high diversity (cosine similarity $<0.11$).

**Core Idea**: Use ULDD to **extrapolate** OOD "pseudo-jailbroken activations" from refusal-state activations. This extrapolation process is embedded into bi-level adversarial training: the inner loop continuously generates pseudo-jailbreaks that are hardest for the current steering to mitigate, while the outer loop trains the steering to suppress these pseudo-jailbreaks, thereby **forcing the simulated support to gradually approximate the true jailbreak subspace**.

## Method

The overall method comprises three components: (i) upgrading fixed-vector steering to **gradient field steering**; (ii) **simulating** OOD jailbreak activations from refusal activations using ULDD; and (iii) using **bi-level adversarial training** to drive mutual optimization.

### Overall Architecture

Let the hidden state of an aligned LLM $F$ at layer $\ell$ be $h_\ell(x) \in \mathbb{R}^d$. The training phase maintains:

- Benign activations $h_b$ (500 AlpacaEval samples)
- Refusal-state activations $h_r$ (1500 direct harmful requests + OR-Bench boundaries)
- Learned parameters $\phi$: A small MLP potential function $f_\phi: \mathbb{R}^d \to \mathbb{R}$
- Learned parameters $V \in \mathbb{R}^{d \times K}$: $K$ ULDD latent directions

Each outer step begins with the **inner loop**: Fix $\phi$, and adversarially optimize $V$ so that pseudo-jailbreaks $h_j^{\text{adv}} = h_r + R v$ (where $v$ is sampled from columns of $V$ and $R$ is a preset magnitude) are "difficult to guide" under the current potential function. Then, the **outer loop**: Update $\phi$ using these new $h_j^{\text{adv}}$ to satisfy three properties—universal steerability, benign zero-guidance, and jailbreak strong-guidance.

The testing phase requires no jailbreak labels: For any input $x$, take activation $h$ at layer $\ell$, and perform $K$ steps of gradient ascent along $\nabla_h f_\phi(h)$: $h^{(k+1)} = h^{(k)} + \eta \nabla_h f_\phi(h^{(k)})$. Feed the guided activation back for the forward pass.

### Key Designs

1.  **OT Potential + Gradient Steering Field (Replacing fixed refusal vectors)**:
    - **Function**: Upgrades steering from "global linear addition of a vector" to an "input-dependent non-linear field," allowing for localized treatment of benign vs. jailbroken regions.
    - **Mechanism**: Models the transport from "jailbroken distribution $\mu \to$ refusal distribution $\nu$" as a minimum-cost Wasserstein-1 transport. Using Kantorovich-Rubinstein duality $W_1(\mu,\nu) = \sup_{\|f\|_L \le 1}(\mathbb{E}_\mu[f] - \mathbb{E}_\nu[f])$, $f_\phi$ is trained to take large values on refusal activations and small values elsewhere. Its gradient $v_\phi(h) = \nabla_h f_\phi(h)$ naturally points in the direction that "pushes $h$ toward the refusal region." Lipschitz constraints are enforced via WGAN-GP gradient penalties on interpolated points $\hat h = \epsilon h_r + (1-\epsilon) h_-$ as $L_{\text{GP}} = \mathbb{E}[\text{ReLU}(\|\nabla \hat h f_\phi\|_2 - 1)]$.
    - **Design Motivation**: Fixed-vector steering is a special case when $f$ is quadratic, limiting expressiveness. To achieve "benign zero-intervention + jailbroken strong-intervention," the field must be spatially variable, necessitating an MLP-based potential function. Ablations in Appendix G verify that MLP potentials outperform first-order or higher-order linear steering.

2.  **Outer Loss Constrained by Three Properties**:
    - **Function**: Ensures $f_\phi$ satisfies universal steerability ($L_g$), benign zero-guidance ($L_b$), and jailbreak strong-guidance ($L_j$).
    - **Mechanism**: $L_g = L_{\text{OT}} + \lambda_{\text{GP}} L_{\text{GP}}$, where $L_{\text{OT}} = -(\mathbb{E}_{h_r}[f_\phi(h_r)] - \mathbb{E}_{h_-}[f_\phi(h_-)])$ maximizes the potential difference between refusal and other activations ($h_-$ includes both benign and adversarial jailbroken). Benign zero-guidance uses a gradient norm penalty $L_b = \mathbb{R}_{h_b}[\|\nabla_h f_\phi(h_b)\|_2^2]$ with a large weight to force "gradient $\approx 0$." Jailbreak strong-guidance conversely uses $L_j = -\mathbb{E}_{h_j}[\|\nabla_h f_\phi(h_j)\|_2^2]$ to encourage high gradients near pseudo-jailbreak activations, ensuring efficient pull-back to the refusal zone via gradient ascent.
    - **Design Motivation**: OT-dual alone only ensures a "global flow toward refusal" but cannot distinguish "benign" from "jailbroken." Treating the gradient norm as a local intensity knob (suppressing one, enhancing the other) allows spatially differentiated control within the same field—a capability missing in fixed-vector steering.

3.  **Bi-level Adversarial Training: Online Generation of Hardest Pseudo-jailbreaks**:
    - **Function**: Evolves training support alongside $f_\phi$; pseudo-jailbreaks migrate to wherever the potential field is weakest.
    - **Mechanism**: The inner loop fixes $\phi$ and updates $V$ to maximize $L_j(h_j^{\text{adv}}; f_\phi) + \gamma L_{\text{ULDD}}(h_r)$, where $h_j^{\text{adv}} = h_r + R v, v \sim V$. The ULDD loss $L_{\text{ULDD}} = \mathbb{E}_{u\in U, v\in V}[\langle u, \Delta h_t(v)\rangle] - \lambda(\|U^\top U - I\|^2 + \|V^\top V - I\|^2)$ ensures directions induce significant and diverse semantic changes. In the inner loop, $L_j$ is maximized (generating harder pseudo-jailbreaks), while in the outer loop, it is minimized (mitigating them), forming a minimax game.
    - **Design Motivation**: The root cause of supervised steering failure is that training support does not cover the real jailbreak subspace. By letting the inner loop actively find blind spots in the field and the outer loop fill them, the subspace coverage metric (Eq. 18-19, using PCA projection energy ratio $\text{Cov}_t^a(h) = \|P_a h\|^2 / \|h\|^2$) increases monotonically during training, correlating with monotonically decreasing Avg. SR (Fig. 6-7). This explicitly links "coverage growth" to "defense enhancement."

### Loss & Training

The complete bi-level minimax formulation:

- **Inner**: $V \in \arg\max_V [L_j(h_j^{\text{adv}}; f_\phi) + \gamma L_{\text{ULDD}}(h_r)]$, where $h_j^{\text{adv}} = h_r + R v, v \sim V$;
- **Outer**: $\phi \in \arg\min_\phi L_g(h_b, h_r, h_j^{\text{adv}}; f_\phi) + \lambda_1 L_b(h_b; f_\phi) + \lambda_2 L_j(h_j^{\text{adv}}; f_\phi)$.

Data: 500 AlpacaEval (benign) + 500 OR-Bench (borderline) + 1000 AdvBench $\cup$ OR-Bench-toxic (deduplicated Harmbench). A separate $f_\phi$ is trained for each of the three models.

## Key Experimental Results

### Main Results

Evaluated on Harmbench across six jailbreak types (GCG / AutoDAN / GPTFuzz / PAIR / TAP / FewShot) using StrongReject (SR ↓) (Selected from Tab. 3):

| Model | Baseline | + CB | + LAT | + ROSI | + AlphaSteer | + Ours |
|------|------|------|-------|--------|--------------|--------|
| LLaMA-3-8B Safety Avg | 14.93 | **3.85** | 5.05 | 6.04 | 5.38 | 4.01 |
| Mistral-v2-7B Safety Avg | 59.72 | 5.42 | 6.70 | 8.47 | 5.86 | **5.26** |
| Qwen-2.5-7B Safety Avg | 33.27 | 5.70 | 5.82 | 7.80 | 4.48 | **4.46** |

Utility Persistence (Tab. 4, Utility = (ARC + TruthfulQA + GSM8K)/3 − OR-FPR, higher is better):

| Model | Baseline | + CB | + LAT | + ROSI | + AlphaSteer | + Ours |
|------|------|------|-------|--------|--------------|--------|
| LLaMA-3-8B Utility | 56.5 | 14.0 | 23.5 | 30.1 | 37.9 | **46.5** |
| Mistral-v2-7B Utility | 56.4 | −26.0 | 24.5 | 20.7 | 31.4 | **36.6** |
| Qwen-2.5-7B Utility | 74.8 | 40.3 | 46.8 | 42.2 | 49.8 | **53.5** |

CB and LAT offer strong defense but severe over-refusal (OR-FPR $83.3\%$ on Mistral, yielding negative utility). Ours preserves most base capabilities while keeping OR-FPR between $16–23\%$, leading in overall utility.

### Ablation Study

| Configuration | Mistral SR ↓ | Description |
|------|--------------|------|
| Baseline Model | 63.34 | No defense |
| Ours (Non-adaptive attack) | 7.10 | Standard GCG, etc. |
| Adaptive GCG | 12.55 | Attacker optimizes suffix against the defended model |
| Steering-aware GCG | 15.54 | Attacker minimizes $\|\nabla f_\phi\|$ to bypass gradient field |

| Training Strategy | Trend | Description |
|----------|------|------|
| Targeted AT (Pseudo-jailbreaks start with "sure, here is...")| Coverage plateaus early, Avg. SR stalls at high levels | Target prefix constrains adversarial activation diversity |
| Unsupervised AT (Ours) | Coverage rises steadily across 6 families, Avg. SR decreases | ULDD diversity brings broad coverage |
| Without AT | Coverage static, limited Avg. SR improvement | Bi-level AT is the primary performance driver |

### Key Findings

- **Subspace coverage is tightly coupled with defense strength** (Fig. 6-7): The rising coverage curve and falling Avg. SR curve are almost mirror images. The authors use the "energy ratio of simulated activations projected onto real subspaces" as an interpretable proxy for training progress.
- **Performance collapses without bi-level AT**: The outer objective alone is insufficient, indicating that the inner loop's active "vulnerability searching" is crucial.
- **Robustness against adaptive attacks**: Even when attackers explicitly optimize to reduce the steering field's gradient (steering-aware GCG), SR only rises to 15.54 (baseline 63.34), showing the gradient field is harder to bypass than fixed-vector steering.
- **OR-Bench compatibility**: CB/LAT perform poorly on utility because they "suppress all." By explicitly forcing benign gradients to zero with large weight $\lambda_1$, this method achieves much lower OR-FPR than other strong defenses.

## Highlights & Insights

- Re-purposes *Unsupervised Latent Direction Discovery*—typically an interpretability/behavioral probe tool—as a "zero-cost jailbreak simulator." Creating OOD attack samples in activation space without labels is the paper's most ingenious shift.
- Uses a gradient field rather than a fixed vector for steering, decoupling "whether to guide" and "how much to guide" into two knobs: "potential difference" and "gradient norm." This allows benign zero-guidance and jailbreak strong-guidance to co-exist in the same $f_\phi$.
- Subspace coverage is a valuable metric: Future adversarial work regarding "training support vs. real distributions" can use the "PCA projection energy ratio" to quantify coverage growth as a mechanistic explanation.
- This framework essentially transplants the WGAN dual perspective into activation space, replacing input-level noise perturbations with latent-direction linear extrapolations—a paradigm better suited for internal LLM representations.

## Limitations & Future Work

- Simulation capability is limited by "linear latent direction extrapolation": The authors admit that non-natural language jailbreaks like base64 or ciphers might fall into regions unreachable by linear ULDD. Future work could explore non-linear transformations or gradient-based searches.
- Validation was limited to 7-8B models. Whether potential function capacity scales to 70B+ models or how training costs evolve is unclear.
- Inference requires $K$ gradient ascent steps (unspecified $K$ in summary), increasing forward latency compared to one-shot linear steering.
- Sensitivity to hyperparameters like weights $\lambda_1, \lambda_2$, ULDD dimension $K$, and extrapolation magnitude $R$ requires more extensive scanning.
- Evaluator bias: Relies on StrongReject (a fine-tuned Gemma-2B) and GPT-4 for OR-FPR; potential evaluator bias was not deeply discussed.

## Related Work & Insights

- **vs. Refusal Direction (Arditi et al., 2024) / AlphaSteer (Sheng et al., 2025)**: Both belong to the linear refusal steering family. AlphaSteer adds null-space constraints for utility but remains a **global, single-direction, supervised** method. This work upgrades to an MLP gradient field + unsupervised adversarial training.
- **vs. Conditional Activation Steering (Lee et al., 2024) / JBShield (Zhang et al., 2025)**: These classify inputs to apply selective steering but still require labeled jailbreaks. This work uses ULDD to simulate pseudo-jailbreaks, bypassing labeling entirely.
- **vs. Circuit Breaker (Zou et al., 2024) / LAT (Sheshadri et al., 2024)**: Both are representation-level safety engineering. CB disrupts harmful activations, while LAT performs local adversarial robustness near refusal states. Both cause severe over-refusal (OR-FPR 30–80%). This method suppresses OR-FPR to 16–23% via explicit benign zero-gradient optimization.
- **vs. Mack & Turner (2024) Unsupervised Steering**: They use ULDD for behavioral guidance/probing; this paper uses it inversely to generate adversarial samples for safety training, demonstrating a new application for ULDD in safety and unlearning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Short-length Adversarial Training Helps LLMs Defend Long-length Jailbreak Attacks](../../NeurIPS2025/llm_alignment/short-length_adversarial_training_helps_llms_defend_long-length_jailbreak_attack.md)
- [\[ICML 2026\] Adaptive Probe-based Steering for Robust LLM Jailbreaking](adaptive_probe-based_steering_for_robust_llm_jailbreaking.md)
- [\[ICLR 2026\] AlphaSteer: Learning Refusal Steering with Principled Null-Space Constraint](../../ICLR2026/llm_alignment/alphasteer_learning_refusal_steering_with_principled_null-space_constraint.md)
- [\[ICML 2026\] Consistency Training Can Entrench Misalignment](consistency_training_can_entrench_misalignment.md)
- [\[CVPR 2026\] Principled Steering via Null-space Projection for Jailbreak Defense in Vision-Language Models](../../CVPR2026/llm_alignment/principled_steering_via_null-space_projection_for_jailbreak_defense_in_vision-la.md)

</div>

<!-- RELATED:END -->
