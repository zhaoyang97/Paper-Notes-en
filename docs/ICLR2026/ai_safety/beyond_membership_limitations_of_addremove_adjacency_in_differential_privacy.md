---
title: >-
  [Paper Note] Beyond Membership: Limitations of Add/Remove Adjacency in Differential Privacy
description: >-
  [ICLR 2026][AI Safety][Differential Privacy] The paper argues that the **add/remove adjacency** used by mainstream DP libraries only protects "whether a member is in the training set." For attacks aiming to "infer attributes/labels of samples known to be in the set," it only provides much weaker protection under **substitute adjacency**. The authors design a canary auditing toolkit for substitute adjacency, empirically demonstrating that privacy leakage can **exceed the $\var…
tags:
  - "ICLR 2026"
  - "AI Safety"
  - "Differential Privacy"
  - "add/remove adjacency"
  - "substitute adjacency"
  - "attribute inference"
  - "DP-SGD"
  - "privacy auditing"
  - "canary"
date: 2026-05-08
content_hash: 49e30b86c071a620
---

# Beyond Membership: Limitations of Add/Remove Adjacency in Differential Privacy

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=C4jAhm8L1V](https://openreview.net/forum?id=C4jAhm8L1V)  
**Code**: Provided in supplementary materials (prv-accountant for substitute adjacency)  
**Area**: Differential Privacy / Privacy Auditing / AI Safety  
**Keywords**: Differential Privacy, add/remove adjacency, substitute adjacency, attribute inference, DP-SGD, privacy auditing, canary  

## TL;DR
The paper argues that the **add/remove adjacency** used by mainstream DP libraries only protects "whether a member is in the training set." For attacks aiming to "infer attributes/labels of samples known to be in the set," it only provides much weaker protection under **substitute adjacency**. The authors design a canary auditing toolkit for substitute adjacency, empirically demonstrating that privacy leakage can **exceed the $\varepsilon_{AR}$ upper bound** reported by add/remove accountants, while closely matching the $\varepsilon_S$ predicted by substitute accountants.

## Background & Motivation
**Background** —— Differential Privacy (DP) constrains "an adversary's ability to distinguish two adjacent datasets" into an $(\varepsilon,\delta)$ upper bound. The definition of "adjacent" (adjacency relation) determines the privacy semantics. The de facto standard in deep learning is DP-SGD + **add/remove adjacency** ($D'$ is obtained by adding or removing one record from $D$). Almost all mainstream libraries like Opacus and prv-accountant default to this, originally designed to defend against **membership inference** (whether an individual participated in training).

**Limitations of Prior Work** —— In many real-world scenarios, the protection goal is not "inclusion in the training set," but rather **attributes or labels of a record known to be in the training set**: label privacy in supervised fine-tuning, inference of sensitive user attributes, etc. The threat model for such attacks corresponds to **substitute adjacency** ($D'$ replaces a record $z$ in $D$ with $z'$). The add/remove upper bound only provides "indirect and loose" protection for this.

**Key Challenge** —— While $\varepsilon_{AR}$ from add/remove can derive a substitute $\varepsilon_S$, it comes at the cost of significantly loosened parameters (the group privacy theorem gives $\varepsilon_S = 2\varepsilon_{AR}$). Practitioners observing the $\varepsilon_{AR}$ reported by libraries may **falsely assume attribute privacy is equally strong**—a systematic "privacy overestimation." However, tools to quantify and audit this gap are lacking.

**Goal** —— To prove and empirically measure that when the protection target is per-record attributes rather than membership, add/remove accountants **overestimate protection strength**; and to provide tight auditing methods under substitute adjacency.

**Core Idea**: **"Record Substitution instead of Add/Remove" worst-case canary** —— Instead of auditing by "adding/removing one canary," a pair of opposing canaries $(z, z')$ with norms reaching the clipping bound $C$ are constructed for "substitution." This forces the true leakage of substitute adjacency even under a hidden-state threat model.

## Method

### Overall Architecture
The method consists of two parts: **(1) Auditing Protocol**—organizing "challenger training, adversary canary construction, and adversary differentiation" into a membership game repeatable $R$ times (Algorithm 1), using $\mu$-GDP to convert the differentiation success rate into an empirical $\varepsilon$ lower bound; **(2) Canary Construction**—providing 5 scenarios S1–S5 (gradient space vs. input space × different prior knowledge) according to adversary strength, each paired with a canary construction algorithm. These components combine to answer "how much DP-SGD actually leaks under substitute adjacency."

```mermaid
flowchart TD
    A["Target Record z"] --> B{Adversary can modify gradient?}
    B -->|Yes · Gradient Space| C["S1 Worst-case dataset canary<br/>S2 Worst-gradient canary (hidden-state)"]
    B -->|No · Input Space| D["S3 Complementary input / S4 Mislabeling / S5 Natural adversarial samples"]
    C --> E["Train DP-SGD R times<br/>Randomly use z or z'"]
    D --> E
    E --> F["Divergence Score<br/>logit(z)-logit(z') or (gz/C)·(θT-θ0)"]
    F --> G["Clopper-Pearson est. FPR/FNR<br/>→ μ-GDP → Empirical εS lower bound"]
    G --> H{εS(Audit) vs εAR(Accountant)?}
    H --> I["Exceeds εAR, matches εS<br/>⇒ add/remove overestimates attribute protection"]
```

### Key Designs

**1. Substitute worst-case canary: Turning "add/remove" into "hedging."** In add/remove auditing, the canary is **present/absent** in one of the two datasets. Substitute auditing requires the canary to be **present in both, but opposite**. The authors construct $z$ such that its gradient $\|g_z\|=C$ hits the clipping bound throughout training, and replace $z$ with $z'$ such that $\|g_{z'}\|=C$ with a **precisely opposite direction**, while other samples contribute zero gradients, maximizing distinguishability. Crucially, unlike Nasr et al. (2021), Ours **does not assume a learning rate of 0 for non-canary steps**, thus realistically accounting for "canary dilution by subsampling" in the noise, reflecting true DP-SGD dynamics. Given sampling rate $q$, the number of times a canary is sampled within $T$ steps follows $B\sim\text{Binomial}(T,q)$. The conditional distribution of the accumulated gradient is $\Pr(g_T\mid B=k)\sim\mathcal{N}(\pm kC,\,T\sigma^2C^2)$. Summing over $k$ yields the marginal distribution:

$$\Pr(g_T\mid D \text{ or } D') = \sum_{k=0}^{T}\binom{T}{k}q^k(1-q)^{T-k}\,\mathcal{N}\!\big(g_T;\pm kC,\,T\sigma^2C^2\big),$$

The adversary uses $\log\Pr(g_T\mid D)-\log\Pr(g_T\mid D')$ as the divergence score to extract the tightest empirical $\varepsilon_S$ lower bound under substitute adjacency.

**2. Hidden-state canary family: Receding from gradient space to input space.** Real adversaries usually cannot access intermediate models and can only observe the final model, often modifying inputs rather than gradients. The authors arrange canaries into a "decreasing capability" spectrum: **S2 Worst-gradient canary** (Algorithm 2)—selecting the parameter dimension $j^\*=\arg\min_j S_j$ with the smallest magnitude change during training ($S_j$ is cumulative $|\theta^j_{t+1}-\theta^j_t|$), assigning $\pm C$ only to that dimension and zeroing others. This ensures $\|g_z\|=\|g_{z'}\|=C$ in opposite directions, using $\theta_T-\theta_0$ as the score (suitable for Federated Learning auditing); **S3 Complementary input canary** (Algorithm 3)—using a non-DP reference model to optimize $x'$ to minimize the cosine similarity between $g_{z'}$ and $g_z$, while MSE constraints keep the scales similar, optimized via gradient descent on $L_{\text{cosim}}+L_{\text{MSE}}$; **S4 Mislabeling canary** (Algorithm 4)—fixing input $x$ and selecting $y'$ in the label space that maximizes gradient "hedging"; **S5 Natural adversarial canary** (Algorithm 5)—selecting a real sample from an auxiliary set $D_{aux}$ with the minimum gradient cosine similarity to $z$. All input space scenarios use $\text{logit}(z;\theta_T)-\text{logit}(z';\theta_T)$ as the score, corresponding to real threats like data poisoning or label inference.

**3. Group privacy conversion is only a loose bound; use substitute accountants instead.** In practice, substitution is often treated as a composite of "one add + one remove." From group privacy (Dwork & Roth, 2014), Theorem 4.1 states: an algorithm satisfying $(\varepsilon_{AR},\delta_{AR},\sim_{AR})$-DP is $(\varepsilon_S,\delta_S,\sim_S)$-DP, where $\varepsilon_S=2\varepsilon_{AR},\ \delta_S=(1+e^{\varepsilon_{AR}})\delta_{AR}$. However, this conversion is algorithm-agnostic and overly conservative. Ours proposes that for algorithms characterized by Privacy Loss Random Variables (PRV) or Privacy Loss Distributions (PLD) (e.g., Poisson subsampled DP-SGD), directly calculating the privacy curve for substitute adjacency using a **numerical accountant** is much tighter. The authors adapted Microsoft's prv-accountant for substitute adjacency, finding that $\varepsilon_S$(Accounting) is significantly tighter than $\varepsilon_S$(Group Privacy)—this tight curve is the "ground truth" for auditing. The paper also warns that Theorem 4.1 assumes proportional scaling of $\delta$; if $\delta$ is fixed, $\varepsilon_S$ may even exceed $\varepsilon_{AR}$.

**Rating conversion ($\mu$-GDP):** Auditing is standardized via Gaussian DP. From the confusion matrix of $R$ games, FPR/FNR are estimated using Clopper–Pearson ($\alpha=0.05$), then $\mu_{\text{lower}}=\Phi^{-1}(1-\text{FPR})-\Phi^{-1}(\text{FNR})$ is calculated. Finally, the $\delta(\varepsilon)=\Phi(-\frac{\varepsilon}{\mu}+\frac{\mu}{2})-e^{\varepsilon}\Phi(-\frac{\varepsilon}{\mu}-\frac{\mu}{2})$ formula from Dong et al. (2019) maps the $\mu$ lower bound to an $\varepsilon$ lower bound.

## Key Experimental Results

### Main Results

| Setting | Model / Data | Key Findings |
|------|-------------|----------|
| Worst-case dataset canary (S1) | Synthetic worst-case, $\delta=10^{-5},C=1,T=500$, $R=25\text{K}$ | $\varepsilon$(Audit) **exceeds** $\varepsilon_{AR}$, closely matching $\varepsilon_S$(Accounting); $\varepsilon_S$(Accounting) is tighter than $\varepsilon_S$(Group Privacy) |
| Natural data fine-tuning | ViT-B-16(IN21K) last layer + CIFAR10(500) | All S2–S5 **break** $\varepsilon_{AR}$ at large $q$; Gradient canary (S2) is the tightest |
| Text fine-tuning | Sentence-BERT linear head + SST-2(5K) | Gradient canary auditing is tight; Input canaries also exceed $\varepsilon_{AR}$ |
| Training from scratch | 3-layer MLP + Purchase100(50K), DP-Adam | Input canary auditing is weak and **does not** exceed $\varepsilon_{AR}$; however, gradient canary still matches $\varepsilon_S$ |

### Ablation Study

| Factor | Impact |
|------|------|
| Sampling rate $q$ (1.0 / 0.25 / 0.0625) | Smaller $q$ "dilutes" the canary, loosening auditing; Input space canaries are most affected |
| Clipping bound $C$ / Training steps $T$ / Learning rate $\eta$ (Appendix A2–A4) | Input space canaries are sensitive to these; auditing effectiveness decreases in later training steps |
| Canary effect on model utility | Minimal impact (Appendix A1), no loss in utility |

### Key Findings
- **Gradient canary is the "Gold Standard"**: Whether in fine-tuning or training from scratch, $\varepsilon$(Audit) closely matches the substitute accountant $\varepsilon_S$, consistently breaking the add/remove upper bound.
- **Fine-tuning is riskier than training from scratch**: Fine-tuned models are more vulnerable to input space canaries—a single poisoned sample in fine-tuning data can cause leakage to exceed $\varepsilon_{AR}$, especially in commonly used large $q$ ranges.
- **Training from scratch is more "robust"**: Non-convex optimization combined with DP-Adam and subsampling renders input canaries less effective. In this case, add/remove is sufficient—indicating that risk is highly dependent on the training paradigm.

## Highlights & Insights
- **Conceptual "Emperor’s New Clothes"**: Reveals a widely ignored semantic mismatch—libraries report $\varepsilon_{AR}$, but the attribute/label privacy users actually desire is governed by substitute adjacency, with a significant gap between the two.
- **Auditing instead of just proving**: Moves beyond the theoretical fact that "add/remove ⇒ loose substitute bound" by creating attacks that **empirically break** $\varepsilon_{AR}$, turning an abstract gap into measurable empirical leakage.
- **Complete spectrum of threat models**: S1–S5 cover from "strongest adversary who can modify gradients" to "realistic adversary who can only poison a single natural sample," with priors for each scenario clearly mapped (Table 2).
- **Honest auditing modeling**: Explicitly avoids the simplification of "zero learning rate for non-canary steps," realistically incorporating subsampling dilution to make conclusions more credible.

## Limitations & Future Work
- **Limitations**: Experiments focus on small-scale fine-tuning (last layer/linear head) and small-to-medium datasets (CIFAR10 500 samples, SST-2 5K, Purchase100), not covering the true scale of full fine-tuning or LLM SFT; input space canaries fail under training from scratch + DP-Adam, suggesting conclusions are paradigm-sensitive.
- **Future Work**: Extending substitute auditing to more realistic black-box/query access; studying accountants or mechanisms that balance utility and block attribute leakage at large $q$; integrating substitute accountants as a default option in mainstream DP libraries.
- **Goals**: Calling for practitioners to directly report $\varepsilon_S$ for substitute adjacency when the protection target is attributes/labels, rather than relying on add/remove, to avoid systematic privacy overestimation.

## Related Work & Insights
- **DP Auditing Spectrum**: Jayaraman & Evans (2019) first revealed the gap between empirical leakage and theoretical bounds. Nasr et al. (2021/2023), Steinke et al. (2023), and Cebere et al. (2025) advanced worst-case canaries and realistic threat models—Ours shifts this line from add/remove to substitute, utilizing Cebere's minimal parameter update idea for gradient canaries.
- **Adjacency and Group Privacy**: The zero-out by Kairouz et al. (2021) and the group privacy theorem by Dwork & Roth (2014) are sources for Theorem 4.1. The paper notes these are too loose for conversion and should be replaced by PRV/PLD numerical accountants (Gopi et al., 2021).
- **$\mu$-GDP Auditing**: Gaussian DP by Dong et al. (2019) and the $\mu$-GDP auditing method by Nasr et al. (2023) serve as the foundation for empirical $\varepsilon$ estimation.
- **Insight**: The choice of adjacency relation is not a mere theoretical detail; it directly affects whether reported privacy numbers correspond to the attacks one actually intends to prevent. Privacy systems should first align "Protection Target ↔ Adjacency Definition ↔ Accountant."

## Rating
- **Novelty**: ⭐⭐⭐⭐ —— Not a new mechanism, but identifies and quantifies a widely ignored semantic mismatch, supplemented by a 5-scenario substitute canary auditing toolkit.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ —— Covers image/text/tabular modalities, fine-tuning, and training from scratch, including sensitivity to $q/C/T/\eta$ and utility checks; however, the scale is relatively small.
- **Writing Quality**: ⭐⭐⭐⭐ —— Definitions, theorems, algorithms, and charts are clearly layered; the threat model prior table (Table 2) is very helpful.
- **Value**: ⭐⭐⭐⭐ —— A direct, actionable warning for practitioners using DP libraries for attribute/label privacy, potentially driving libraries to report substitute adjacency by default.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Privacy Beyond Pixels: Latent Anonymization for Privacy-Preserving Video Understanding](privacy_beyond_pixels_latent_anonymization_for_privacy-preserving_video_understa.md)
- [\[ICLR 2026\] INO-SGD: Addressing Utility Imbalance under Individualized Differential Privacy](ino-sgd_addressing_utility_imbalance_under_individualized_differential_privacy.md)
- [\[ICLR 2026\] Membership Privacy Risks of Sharpness Aware Minimization](sam_membership_privacy_risks.md)
- [\[ICLR 2026\] Exponential-Wrapped Mechanisms: Differential Privacy on Hadamard Manifolds Made Practical](exponential-wrapped_mechanisms_differential_privacy_on_hadamard_manifolds_made_p.md)
- [\[ICLR 2026\] Optimizing Canaries for Privacy Auditing with Metagradient Descent](optimizing_canaries_for_privacy_auditing_with_metagradient_descent.md)

</div>

<!-- RELATED:END -->
