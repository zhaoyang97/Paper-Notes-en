---
title: >-
  ICML2026 Learning Theory Papers · 45 Notes
description: >-
  45 ICML2026 papers in the Learning Theory area, covering Adversarial Robustness, LLM, Compression, Layout & Composition and more. Each note has TL;DR, motivation, method, experiments, highlights, and limitations — 5-minute reads of core ideas.
tags:
  - "ICML2026"
  - "Learning Theory"
  - "AI paper notes"
  - "paper summaries"
  - "Adversarial Robustness"
  - "LLM"
  - "Compression"
  - "Layout & Composition"
item_list:
  - u: "a_perturbation_approach_to_unconstrained_linear_bandits/"
    t: "A Perturbation Approach to Unconstrained Linear Bandits"
  - u: "active_learning_with_low-rank_structure_for_data_selection/"
    t: "Active Learning with Low-Rank Structure for Data Selection"
  - u: "ai4slt_empirical_processes_in_lean_4_for_formal_statistical_learning_theory/"
    t: "AI4SLT: Empirical Processes in Lean 4 for Formal Statistical Learning Theory"
  - u: "asymptotic_optimality_of_the_high-dimensional_gaussian_mechanism_and_improved_lo/"
    t: "Asymptotic Optimality of the High-Dimensional Gaussian Mechanism and Improved Low-Dimensional Mechanisms for Differential Privacy"
  - u: "bandit_social_learning_with_exploration_episodes/"
    t: "Bandit Social Learning with Exploration Episodes"
  - u: "catastrophic_forgetting_is_low-rank_a_function-space_theory_for_continual_adapta/"
    t: "Catastrophic Forgetting is Low-Rank: A Function-Space Theory for Continual Adaptation"
  - u: "conditional_krr_injecting_unpenalized_features_into_kernel_methods_with_applicat/"
    t: "Conditional KRR: Injecting Unpenalized Features into Kernel Methods with Applications to Kernel Thresholding"
  - u: "core-mtl_rethinking_gradient_balancing_via_causal_orthogonal_representations/"
    t: "CORE-MTL: Rethinking Gradient Balancing via Causal Orthogonal Representations"
  - u: "correcting_split_selection_in_online_decision_trees_via_anytime-valid_inference/"
    t: "Correcting Split Selection in Online Decision Trees via Anytime-Valid Inference"
  - u: "cutting_llm_evaluation_costs_with_sysrs_a_bandit_algorithm_that_provably_exploit/"
    t: "Cutting LLM Evaluation Costs with SySRs: A Bandit Algorithm that Provably Exploits Model Similarity"
  - u: "efficiently_learning_drifting_halfspaces_with_massart_noise/"
    t: "Efficiently Learning Drifting Halfspaces with Massart Noise"
  - u: "enhancing_conformal_prediction_via_class_similarity/"
    t: "Enhancing Conformal Prediction via Class Similarity"
  - u: "estimating_correlation_clustering_cost_in_node-arrival_stream/"
    t: "Estimating Correlation Clustering Cost in Node-Arrival Stream"
  - u: "expectation_consistency_loss_rethink_confidence_calibration_under_covariate_shif/"
    t: "Expectation Consistency Loss: Rethink Confidence Calibration under Covariate Shift"
  - u: "finite-width_neural_tangent_kernels_from_feynman_diagrams/"
    t: "Finite-Width Neural Tangent Kernels from Feynman Diagrams"
  - u: "formalizing_learning_from_language_feedback_with_provable_guarantees/"
    t: "Formalizing Learning from Language Feedback with Provable Guarantees"
  - u: "geometric_and_stochastic_analysis_of_discontinuities_in_sparse_mixture-of-expert/"
    t: "Geometric and Stochastic Analysis of Discontinuities in Sparse Mixture-of-Experts"
  - u: "is_spurious_correlation_removal_always_learnable/"
    t: "Is Spurious Correlation Removal Always Learnable?"
  - u: "learning_credal_ensembles_via_distributionally_robust_optimization/"
    t: "Learning Credal Ensembles via Distributionally Robust Optimization"
  - u: "matroid_algorithms_under_size-sensitive_independence_oracles/"
    t: "Matroid Algorithms Under Size-Sensitive Independence Oracles"
  - u: "mmd-balls_as_credal_sets_a_pac-bayesian_framework_for_epistemic_uncertainty_in_t/"
    t: "MMD-Balls as Credal Sets: A PAC-Bayesian Framework for Epistemic Uncertainty in Test-Time Adaptation"
  - u: "multi-task_linear_regression_without_eigenvalue_lower_bounds_adaptivity_robustne/"
    t: "Multi-task Linear Regression without Eigenvalue Lower Bounds: Adaptivity, Robustness and Safety"
  - u: "on_regret_bounds_of_thompson_sampling_for_bayesian_optimization/"
    t: "On Regret Bounds of Thompson Sampling for Bayesian Optimization"
  - u: "on_the_learnability_of_test-time_adaptation_a_recovery_complexity_perspective/"
    t: "On the Learnability of Test-Time Adaptation: A Recovery Complexity Perspective"
  - u: "on_the_robustness_of_langevin_dynamics_to_score_function_error/"
    t: "On the Robustness of Langevin Dynamics to Score Function Error"
  - u: "online_learning_with_recency_algorithms_for_sliding-window_streaming_multi-armed/"
    t: "Online Learning with Recency: Algorithms for Sliding-window Streaming Multi-armed Bandits"
  - u: "optimal_design_for_multinomial_logit_model_with_applications_to_best_assortment_/"
    t: "Optimal Design for Multinomial Logit Model with Applications to Best Assortment Identification"
  - u: "parsimonious_learning-augmented_online_metric_matching/"
    t: "Parsimonious Learning-Augmented Online Metric Matching"
  - u: "performative_learning_theory/"
    t: "Performative Learning Theory"
  - u: "provably_data-driven_multiple_hyper-parameter_tuning_with_structured_loss_functi/"
    t: "Provably Data-driven Multiple Hyper-parameter Tuning with Structured Loss Function"
item_total: 45
---

<!-- Auto-generated by src/gen_blog_index.py --lang en -->
# 📐 Learning Theory

**🧪 ICML2026** · **45** paper notes

📌 **Same area in other venues:** [🔬 ICLR2026 (293)](../../ICLR2026/learning_theory/index.md) · [🤖 AAAI2026 (3)](../../AAAI2026/learning_theory/index.md) · [🧠 NeurIPS2025 (25)](../../NeurIPS2025/learning_theory/index.md)

🔥 **Top topics:** Adversarial Robustness ×5

**[A Perturbation Approach to Unconstrained Linear Bandits](a_perturbation_approach_to_unconstrained_linear_bandits.md)**

:   This paper revisits the perturbation-based bandit linear optimization approach by Abernethy et al., proposing the PABLO reduction. This reduction transforms unconstrained linear bandits into a problem that can call any OLO subroutine, thereby obtaining comparator-adaptive static/dynamic regret, high-probability bounds, and discussions on various lower bounds.

**[Active Learning with Low-Rank Structure for Data Selection](active_learning_with_low-rank_structure_for_data_selection.md)**

:   Addressing the mismatch where existing coreset methods assume geometric clustering while modern datasets exhibit global algebraic (low-rank) structures, this paper proposes a data selection framework based on low-rank approximation and residual sensitivity sampling. Using a weighted subset of size $\tilde{O}(k+1/\varepsilon^2)$, the method approximates the full average loss to a $(1\pm\varepsilon)$ relative error (with an additive term proportional to the optimal rank-$k$ approximation cost $\Phi_k$). It outperforms uniform and cluster-based sampling on tabular data and Llama3-8B / Qwen2.5-3B fine-tuning.

**[AI4SLT: Empirical Processes in Lean 4 for Formal Statistical Learning Theory](ai4slt_empirical_processes_in_lean_4_for_formal_statistical_learning_theory.md)**

:   This work presents the first systematic formalization of "Empirical Process-based Statistical Learning Theory (SLT)" from scratch in Lean 4. It fills gaps in Mathlib by implementing Gaussian Lipschitz concentration, the Dudley entropy integral theorem, and sharp rates for least squares regression (including $\ell_1$ constraints). The project consists of approximately 30,000 lines of Lean code without `sorry` or `axiom`, completed through a human-AI collaborative paradigm where humans designed proof strategies and agents (Claude Code + Opus-4.5) executed tactical proofs.

**[Asymptotic Optimality of the High-Dimensional Gaussian Mechanism and Improved Low-Dimensional Mechanisms for Differential Privacy](asymptotic_optimality_of_the_high-dimensional_gaussian_mechanism_and_improved_lo.md)**

:   This theoretical paper answers two long-standing open questions: whether the Gaussian mechanism is the optimal choice for additive noise differential privacy in high dimensions (Ans: as the dimension $T\to\infty$, no additive noise can asymptotically outperform the Gaussian at a fixed mean squared error), and whether there exist mechanisms superior to both Gaussian and $\ell_2$ mechanisms in low dimensions (Ans: yes—the authors propose a three-parameter family of Spherical Generalized Gamma noise, which reduces MSE by up to 15% in certain low-dimensional settings, and they provide tight composition guarantees for this family, resolving an open question by Joseph et al. regarding the $\ell_2$ mechanism).

**[Bandit Social Learning with Exploration Episodes](bandit_social_learning_with_exploration_episodes.md)**

:   This paper investigates the social learning dynamics of bandits where "each selfish agent controls a short sequence of decisions (episode)." It proves that even if agents spontaneously explore within their own episodes, **exploration at the aggregate level still fails**. For any episode length $m \geq 2$ and any aggregate utility function $f$ (such as sum, max, or min), learning failure occurs with positive probability, leading to **linear growth** of Bayesian regret over time.

**[Catastrophic Forgetting is Low-Rank: A Function-Space Theory for Continual Adaptation](catastrophic_forgetting_is_low-rank_a_function-space_theory_for_continual_adapta.md)**

:   Instead of treating catastrophic forgetting as "parameter drift," this work provides a closed-form characterization in function space under the NTK framework: new task training drags old task predictions away via the cross-task kernel $K_{AB}$, and this "forgetting vector" is precisely predictable before training. This vector concentrates on an extremely small number of eigenmodes of the old task kernel $K_{AA}$ (1–6 modes carry 50–90% of the forgetting energy), explaining why parameter-space regularizers fail on shared-head benchmarks and leading to a spectral regularization method that protects only the vulnerable subspace.

**[Conditional KRR: Injecting Unpenalized Features into Kernel Methods with Applications to Kernel Thresholding](conditional_krr_injecting_unpenalized_features_into_kernel_methods_with_applicat.md)**

:   This paper proposes a Conditional Kernel Ridge Regression (Conditional KRR) framework that injects a set of unpenalized features into kernel methods. By reducing it to a standard KRR via a residual kernel, the authors prove a reduction cost of $\mathcal{O}(1/\sqrt{N})$ and verify sufficient conditions where Conditional KRR outperforms standard KRR under both hard thresholding (top-k eigenfunctions) and soft thresholding (random Gaussian features) settings.

**[CORE-MTL: Rethinking Gradient Balancing via Causal Orthogonal Representations](core-mtl_rethinking_gradient_balancing_via_causal_orthogonal_representations.md)**

:   The authors reattribute the root cause of "negative transfer" in Multi-Task Learning (MTL) from "gradient conflict" to the "entanglement of semantics and noise in shared representations." They propose CORE-MTL: a dual-stream encoder splits representations into semantic $\hat{Z}_s$ and residual $\hat{Z}_r$, implementing "causal orthogonality" through CKA independence constraints, counterfactual style replacement, and inverse rendering reconstruction. Theoretically, it provides a tighter OOD upper bound than gradient balancing; experimentally, it outperforms ten baselines including PCGrad, GradNorm, STCH, and FairGrad on NYUv2/Cityscapes (ID) and GTA5→Cityscapes/Cityscapes-C (OOD) settings.

**[Correcting Split Selection in Online Decision Trees via Anytime-Valid Inference](correcting_split_selection_in_online_decision_trees_via_anytime-valid_inference.md)**

:   The authors point out that the "fixed sample size" concentration inequalities used by the classic Hoeffding Tree (HT) for splitting on data streams are violated by its own "data-dependent stopping rule." They reformulate the split criterion using testing-by-betting + Universal Portfolio, allowing both single trees and Adaptive Random Forests to maintain controlled Type-I errors at any stopping time, while achieving higher accuracy and smaller tree sizes across 12 real-world streams.

**[Cutting LLM Evaluation Costs with SySRs: A Bandit Algorithm that Provably Exploits Model Similarity](cutting_llm_evaluation_costs_with_sysrs_a_bandit_algorithm_that_provably_exploit.md)**

:   To reduce evaluation budgets when "selecting the best model," the authors transform the classic Successive Rejects bandit algorithm into a "synchronized" version called SySRs. By evaluating all surviving models on the same batch of test samples in each stage, the algorithm exploits inter-model correlations similarly to paired testing. This results in a hyperparameter-free best-arm identification algorithm with an error bound that tightens as model correlation increases. On 15 standard benchmarks, it reliably selects the optimal model using $\le 35\%$ of model-sample pairs, outperforming existing methods.

**[Efficiently Learning Drifting Halfspaces with Massart Noise](efficiently_learning_drifting_halfspaces_with_massart_noise.md)**

:   In online learning scenarios where the distribution drifts over time and labels are corrupted by Massart noise, this paper provides the first **polynomial-time** algorithm for learning $\gamma$-margin halfspaces with an error of $\eta+\tilde{O}(\Delta^{1/3}/\gamma)$. Using low-degree polynomial lower bounds, it further demonstrates that the $\Delta^{1/3}$ exponent is unavoidable for efficient algorithms, thereby revealing an information-computation gap.

**[Enhancing Conformal Prediction via Class Similarity](enhancing_conformal_prediction_via_class_similarity.md)**

:   This paper incorporates an "out-of-group penalty" into any arbitrary Conformal Prediction (CP) scoring function, penalizing candidate labels that belong to different semantic groups than the top-1 predicted class. It theoretically demonstrates that this penalty reduces the number of semantic groups in the prediction set while maintaining coverage and **unexpectedly shrinking the average prediction set size**. Furthermore, a model-adaptive variant is proposed that constructs the class similarity matrix directly from model features without requiring manual semantic partitions.

**[Estimating Correlation Clustering Cost in Node-Arrival Stream](estimating_correlation_clustering_cost_in_node-arrival_stream.md)**

:   This paper investigates the problem of approximating correlation clustering cost under the "node-arrival" streaming model. The authors propose the C4Approx algorithm, which utilizes **sublinear** space of $O(n^{(3+\alpha)/4}\log n)$ words and a constant number of passes to achieve an $(O(1), n^{1-\alpha})$-approximation. They also provide two matching lower bounds proving that multiple passes and additive error are both inevitable. On real-world datasets, the algorithm achieves performance comparable to the Pivot algorithm while storing only 2% of nodes.

**[Expectation Consistency Loss: Rethink Confidence Calibration under Covariate Shift](expectation_consistency_loss_rethink_confidence_calibration_under_covariate_shif.md)**

:   ECL demonstrates that full alignment of input distributions $P_s(X) = P_t(X)$ is not a necessary condition for calibration under covariate shift. Instead, it is sufficient that the "conditional expectation of $P(Y_k=1|X)$ on each confidence level set is consistent across domains." Based on this, the authors construct ECL, a differentiable loss with unbiased mini-batch gradients that is universal for canonical, class-wise, and top-label calibration.

**[Finite-Width Neural Tangent Kernels from Feynman Diagrams](finite-width_neural_tangent_kernels_from_feynman_diagrams.md)**

:   This paper adapts Feynman diagrams from quantum field theory to neural network analysis, providing a graphical framework of rules for the "finite-width statistical corrections of NTK." This transforms extremely tedious layer-wise recursive derivations into a "draw and translate" process. It proves the critical stability of NTK and demonstrates that scale-invariant activations like ReLU have no finite-width corrections on the diagonal. Numerically, the results align with sampled networks at widths $n \gtrsim 20$.

**[Formalizing Learning from Language Feedback with Provable Guarantees](formalizing_learning_from_language_feedback_with_provable_guarantees.md)**

:   This paper establishes the first formal framework for "Learning from Language Feedback" (LLF), a common but theoretically underspecified decision-making paradigm for LLM agents. Under a setting where rewards are latent, the authors provide sufficient assumptions for learnability, introduce "transfer eluder dimension" to characterize task difficulty, prove that rich language feedback can be exponentially faster than reward learning, and propose HELiX, a no-regret algorithm with provable guarantees (consistently outperforming CoT prompt baselines on Battleship and Minesweeper).

**[Geometric and Stochastic Analysis of Discontinuities in Sparse Mixture-of-Experts](geometric_and_stochastic_analysis_of_discontinuities_in_sparse_mixture-of-expert.md)**

:   This paper presents the first rigorous geometric and stochastic analysis of the "input-output mapping discontinuity" caused by Top-$k$ routing in Sparse Mixture-of-Experts (SMoE). By classifying discontinuity surfaces based on the "number of tied experts," the authors prove that order-1 surfaces occupy almost all "near-discontinuity" volume, while higher-order volumes are negligible. Using diffusion processes, they demonstrate that random perturbations almost surely hit order-1 surfaces first. Based on these insights, a plug-and-play $\ell_\infty$ local smoothing mechanism, SmoothSMoE, is proposed. It restores continuity to the SMoE mapping and improves performance in language and vision tasks with near-zero additional computational overhead.

**[Is Spurious Correlation Removal Always Learnable?](is_spurious_correlation_removal_always_learnable.md)**

:   This paper demonstrates that removing spurious correlations may be "computationally non-learnable" even when the invariant structure is "statistically identifiable" in ideal scenarios. It proves the existence of a family of multi-environment instances where brute-force search recovers the invariant direction with polynomial samples, but any polynomial-time algorithm achieving constant precision would resolve a widely believed hard sparse recovery problem. Simultaneously, the paper characterizes identifiability, minimax rates, and sample complexity phase transitions using an "environmental diversity" parameter $\gamma$.

**[Learning Credal Ensembles via Distributionally Robust Optimization](learning_credal_ensembles_via_distributionally_robust_optimization.md)**

:   CreDRO redefines "epistemic uncertainty" (EU) as **disagreement between models under different training-test distribution shift hypotheses**. Using Distributionally Robust Optimization (DRO), it assigns varying shift intensities to train ensemble members. Their softmax outputs are transformed into class probability intervals to form a box credal set for quantifying uncertainty, consistently outperforming existing credal methods in OOD detection and medical selective classification.

**[Matroid Algorithms Under Size-Sensitive Independence Oracles](matroid_algorithms_under_size-sensitive_independence_oracles.md)**

:   The authors propose a "size-sensitive matroid oracle" model where the query cost grows linearly with the size of the query set. They prove that under this model, the optimal query costs for finding a basis, estimating the rank, and estimating the partition number are all $\tilde{\Theta}(n^2)$. Furthermore, for matroids with a bounded girth $c$, they provide a maximum weight basis algorithm with a complexity of $\mathcal{O}(n^{2-1/c}\log n)$, breaking the quadratic lower bound.

**[MMD-Balls as Credal Sets: A PAC-Bayesian Framework for Epistemic Uncertainty in Test-Time Adaptation](mmd-balls_as_credal_sets_a_pac-bayesian_framework_for_epistemic_uncertainty_in_t.md)**

:   The paper provides the first PAC-Bayes upper bound for test-time adaptation in the form of "target risk $\le$ source empirical risk + KL complexity + MMD distribution shift." It interprets MMD-balls as credal sets in the sense of Walley, naturally separating aleatoric and epistemic uncertainty via "upper/lower risk intervals," and providing computable criteria for "when to adapt and when to abstain."

**[Multi-task Linear Regression without Eigenvalue Lower Bounds: Adaptivity, Robustness and Safety](multi-task_linear_regression_without_eigenvalue_lower_bounds_adaptivity_robustne.md)**

:   This paper proposes a robust multi-task linear regression estimator using $\|\theta_j-\beta\|_{\bm\Sigma_j}$ (matrix-weighted norm) as a regularization term. It replaces the rigid "minimum eigenvalue $\Omega(1)$ of the second moment for each task" assumption found in prior work with a relative "balance constant" $B$. This provides minimax rates, adaptivity, and safety guarantees that fall back to Independent Task Learning (ITL) in high-dimensional scenarios involving ill-conditioned, low-rank, or outlier-contaminated tasks.

**[On Regret Bounds of Thompson Sampling for Bayesian Optimization](on_regret_bounds_of_thompson_sampling_for_bayesian_optimization.md)**

:   This paper systematically completes the regret analysis of Gaussian Process Thompson Sampling (GP-TS) in the Bayesian setting: it first constructs a counter-example proving that GP-TS can only achieve polynomial dependence on the failure probability $\delta$ (cannot reach $\log(1/\delta)$), then provides a second-moment upper bound for cumulative regret that tightens the $\delta$ dependence by $1/\sqrt{\delta}$ times, the first polylogarithmic upper bound for expected lenient regret, and a high-probability regret bound of $\tilde O(\sqrt T)$ under relaxed Matérn conditions, bringing the theoretical guarantees of GP-TS essentially on par with the well-studied GP-UCB.

**[On the Learnability of Test-Time Adaptation: A Recovery Complexity Perspective](on_the_learnability_of_test-time_adaptation_a_recovery_complexity_perspective.md)**

:   This paper establishes the first theoretical framework for Test-Time Adaptation (TTA) learnability by introducing $(\epsilon, \delta)$-Recovery Complexity to measure the time required to reduce excess risk to $\epsilon$ after a distribution shift. By extending local recovery to non-stationary test streams via $(\epsilon, \rho)$-TTA Learnability and deriving matching minimax upper/lower bounds, the work reveals the intrinsic "adaptation speed vs. information constraint" tradeoff in TTA.

**[On the Robustness of Langevin Dynamics to Score Function Error](on_the_robustness_of_langevin_dynamics_to_score_function_error.md)**

:   This paper proves a counter-intuitive negative result: even when the $L^2$ (or even $L^p$) estimation error of the score function is arbitrarily small, Langevin dynamics in high dimensions may fail to sample from the target distribution in any polynomial time (with a Total Variation distance as high as $1-e^{-\Omega(d)}$). Conversely, diffusion models succeed in polynomial time under similar conditions—arguing from a new perspective that "diffusion models are more reliable than Langevin dynamics" and providing a practical warning: when using data initialization, one must use fresh samples not involved in training the score.

**[Online Learning with Recency: Algorithms for Sliding-window Streaming Multi-armed Bandits](online_learning_with_recency_algorithms_for_sliding-window_streaming_multi-armed.md)**

:   This paper introduces the "recency effect" into streaming multi-armed bandits (MABs) by proposing a **sliding-window streaming MAB** model—where only the most recent $W$ arms are valid. It systematically characterizes the memory complexity bounds for pure exploration and regret minimization: exact identification of the optimal arm requires $\Omega(W)$ memory (essentially storing the entire window), while findig an $\varepsilon$-optimal arm requires only $O(1/\varepsilon)$ memory. Furthermore, regret minimization exhibits a sharp phase transition at $\Theta(W)$ memory.

**[Optimal Design for Multinomial Logit Model with Applications to Best Assortment Identification](optimal_design_for_multinomial_logit_model_with_applications_to_best_assortment_.md)**

:   This work provides the first **computationally feasible** G-optimal experimental design in combinatorial MNL bandit action spaces—reformulating the Frank–Wolfe Linear Maximization Oracle (LMO) as a 0–1 MILP or a polynomial-time Schur complement relaxation—and construct the first best assortment identification algorithm for "linear utility + non-uniform revenues" with sample complexity $\tilde{\mathcal{O}}(d\log N / \Delta^2)$.

**[Parsimonious Learning-Augmented Online Metric Matching](parsimonious_learning-augmented_online_metric_matching.md)**

:   This paper addresses an open problem posed by Im et al. (2022) by bringing "action-predicted" Online Metric Matching (OMM) into a "parsimonious prediction" framework—where predictions are expensive and provided only every $k$ steps. Using the Follow-the-Prediction (FtP) framework combined with a meta-algorithm that automatically fills in "virtual predictions," the authors provide deterministic and randomized competitive ratio upper bounds that essentially match existing lower bounds.

**[Performative Learning Theory](performative_learning_theory.md)**

:   This paper embeds the "performative prediction" phenomenon—where predictions change the very outcomes they intend to forecast—into statistical learning theory for the first time. It proves upper bounds for generalization error, generalization gap, and excess risk under three scenarios: sample-only, population-only, and joint performative perturbations. The work reveals a fundamental tradeoff between "changing the world" and "learning from the world," as well as an "empirical echo chamber" formed by self-negating populations and self-fulfilling samples in the worst case.

**[Provably Data-driven Multiple Hyper-parameter Tuning with Structured Loss Function](provably_data-driven_multiple_hyper-parameter_tuning_with_structured_loss_functi.md)**

:   This paper utilizes "real algebraic geometry + first-order logic quantifier elimination" to provide the first provable generalization bound for multi-dimensional hyper-parameter tuning. It extends the Balcan 2025 framework, which was limited to scalar hyper-parameters, to arbitrary $p$-dimensions, bi-level validation loss, and approximate inner-level optimization, while providing the first matching lower bound.

**[Quantum Algorithms for Triangle Cut Sparsification](quantum_algorithms_for_triangle_cut_sparsification.md)**

:   This work designs quantum algorithms for "triangle cut sparsification": it first provides the first provably accelerated quantum triangle listing algorithm (integrating heavy-light vertex partitioning, quantum walk, and Grover search, taking the best of the three). It then embeds this into the motif sparsification framework of Kapralov et al. and applies quantum acceleration to post-processing sampling. This allows the construction of an $\varepsilon$-triangle cut sparsifier of size $\widetilde{O}(n/\varepsilon^2)$ with an additional overhead of $\widetilde{O}(\sqrt{mn}/\varepsilon)$, accompanied by a matching lower bound of $\Omega(n/\varepsilon^2)$.

**[Realizable Bayes-Consistency for General Metric Losses](realizable_bayes-consistency_for_general_metric_losses.md)**

:   This paper provides a sharp characterization of the open problem regarding when a hypothesis class $\mathcal{H}$ admits a distribution-free strong universal Bayes-consistent learning algorithm under general (possibly unbounded) metric losses in the realizable setting. The necessary and sufficient condition is that $\mathcal{H}$ does not contain a new combinatorial obstacle called an "unbounded-gap Littlestone tree."

**[Revenue Guarantees of No-Swap-Regret Dynamics in First Price Auctions](revenue_guarantees_of_no-swap-regret_dynamics_in_first_price_auctions.md)**

:   This paper proves that in discrete first-price auctions, the revenue of any $\epsilon$-approximate correlated equilibrium is at least $v_2-\Theta(1/k)-\Theta(\epsilon k^2)$ (where $v_2$ is the second-highest valuation). This provides the first **polynomial convergence rate** for the revenue of no-swap-regret bidders in first-price auctions—by using an optimal $O(\sqrt{kT})$ swap-regret algorithm, only $O(k^5/\epsilon^2)$ rounds are required for the time-average revenue to approach the second-highest valuation, significantly improving upon the previous quasi-polynomial bound of $k^{O(\log k)}$.

**[Robustness of Mixtures of Experts to Feature Noise](robustness_of_mixtures_of_experts_to_feature_noise.md)**

:   Under the fair "iso-parameter" setting, this paper demonstrates using a block-diagonal noisy linear regression model that MoE's sparse expert activation acts as a noise filter. This allows it to achieve lower generalization error, stronger perturbation robustness, and faster convergence compared to a dense model of equal size under feature noise.

**[Semi-Supervised Noise Adaptation: Transferring Knowledge from Noise Domain](semi-supervised_noise_adaptation_transferring_knowledge_from_noise_domain.md)**

:   The authors treat a "synthetic domain generated from Gaussian noise" as a surrogate source domain in semi-supervised transfer learning. They demonstrate that such "non-semantic but discriminatively structured" noise provides a quantifiable improvement in generalization bounds for the target domain. They introduce the Noise Adaptation Framework (NAF) to jointly optimize risks and distribution discrepancies across both domains, achieving a 12.35% improvement over ERM on 4-shot ResNet-18 for CIFAR-10.

**[Sequential Kernel-based Conditional Independence Testing via Adaptive Betting](sequential_kernel-based_conditional_independence_testing_via_adaptive_betting.md)**

:   SKCI proposes a sequential (anytime-valid) conditional independence test: it applies "testing-by-betting" to a **self-normalized kernel conditional independence (KCI) statistic**, coupled with a "truncation + shifting" Gaussian approximation calibration. This ensures that even when the conditional distribution $P_{A\mid C}$ in the Model-X assumption must be estimated online (rather than being exactly known), the Type I error inflation remains minor while maintaining high power—outperforming existing sequential Model-X methods on high-dimensional synthetic benchmarks and real-world fairness auditing tasks.

**[Simple Algorithms for Bad Triangle Transversals with Applications to Correlation Clustering](simple_algorithms_for_bad_triangle_transversals_with_applications_to_correlation.md)**

:   This paper provides two simple 2-approximation algorithms for the "Bad Triangle Transversal" (BTT) problem on signed graphs that require only a single LP solve. It proves a unified NP-hard inapproximability bound of $\tfrac{2137}{2136}$ for BTT, Correlation Clustering (CC), MinSTC, and Cluster Deletion on complete graphs. Additionally, it constructs a new pivot procedure to convert any feasible BTT cover into a clustering with at most $\tfrac{3}{2}|F|$ errors, tightening the gap between BTT and CC optima from 2 to $3/2$.

**[Task-Restricted Symmetries in Recurrent Weight Space](task-restricted_symmetries_in_recurrent_weight_space.md)**

:   The paper uses "ordered real Schur coordinates" to decompose the recurrent matrix of a trained single-layer tanh RNN into spectral blocks and non-normal coupling blocks. Through structured ablation by zeroing out blocks, it reveals that certain non-normal couplings can be removed with almost no impact on task behavior (approximate functional invariance), while others are task-critical directions. This profile of "removable/non-removable" components varies across tasks and training solutions, rather than being a universal symmetry of the recurrent weight space.

**[The Data Manifold under the Microscope](the_data_manifold_under_the_microscope.md)**

:   Addressing the gap where "manifold fitting theory's generalization/approximation bounds are nearly unverifiable on real data," this paper proposes a **controllable geometric benchmark framework**. By recreating datasets like dSprites and COIL-20 as low-dimensional manifolds sampled on **dense regular grids** along transformation axes, and using **finite difference geometric estimators**, quantities like curvature, reach, and volume can be calculated with near-ground-truth precision under low intrinsic dimensions. This allows for the empirical calibration of manifold fitting bounds from Genovese, Fefferman, and others in a "known ground truth" sandbox.

**[Towards Optimal Robustness in Learning-Augmented Paging](towards_optimal_robustness_in_learning-augmented_paging.md)**

:   This paper proposes a unified "Relative Prediction Budget" (RPB) perspective for randomized online paging with predictions. Based on OnlineMin, the RPB-OnOPT framework is designed, pushing the provable robust competitive ratio from the existing $2H_k+O(1)$ to $H_k+O(1)$, which is close to the information-theoretic lower bound, while maintaining 1-consistency.

**[Tree-Structured Orthonormal Decomposition of the Aitchison Simplex](tree-structured_orthonormal_decomposition_of_the_aitchison_simplex.md)**

:   PolyILR constructs a **canonical, complete, and orthonormal** Aitchison simplex coordinate system for any tree structure (including polytomies). Each internal node contributes $k_u-1$ contrast coordinates through weighted inner products, Helmert contrasts, and expansion by subtree size, ensuring a valid isometric ILR basis where every coordinate corresponds to a specific location on the tree.

**[Two-Layer Linear Auto-Regressive Models Estimate Latent States](two-layer_linear_auto-regressive_models_estimate_latent_states.md)**

:   This paper theoretically proves that training a **two-layer linear autoregressive model** using empirical risk minimization on data from a partially observable linear dynamical system results in the hidden layer activations spontaneously approximating (up to a similarity transform) the optimal latent state estimates provided by a Kalman filter. The model learns filtering "end-to-end" without being informed of system parameters or states, providing triple finite-sample guarantees for prediction, parameters, and state recovery.

**[Understanding the Parameter Space Geometry of Transformers Encoding Boolean Functions](understanding_the_parameter_space_geometry_of_transformers_encoding_boolean_func.md)**

:   This paper explains why Transformers fail to learn "sensitive" Boolean functions like Parity from the perspective of **parameter space geometry**. It proves that a randomly initialized Transformer almost surely computes functions containing a large number of strings with zero sensitivity. The parameters corresponding to functions like Parity or First, which lack zero-sensitivity strings, constitute only a **Lebesgue measure zero** subset of the parameter space. Consequently, random initialization almost certainly misses these functions, rendering them provably unlearnable.

**[Unraveling Syntax: Language Modeling and the Substructure of Grammars](unraveling_syntax_language_modeling_and_the_substructure_of_grammars.md)**

:   This paper establishes a foundational set of theorems linking "language modeling loss" to the "substructures of Context-Free Grammars (CFG)," proving that the KL divergence of language modeling can be linearly decomposed recursively along the subgrammar hierarchy. Through training small transformers on synthetic PCFGs, it is discovered that models learn all subgrammars **in parallel** (unlike children who master simple structures first). While PCFG subgrammar pre-training primarily benefits models that are small relative to the grammar complexity, it consistently aligns internal representations more closely with the grammar's sub-structures.

**[When Sample Selection Bias Precipitates Model Collapse](when_sample_selection_bias_precipitates_model_collapse.md)**

:   This paper demonstrates that in low-resource and data-island scenarios, data selection—widely regarded as a "remedy" for model collapse—actually accelerates it. Since each verifier only observes a biased local slice of the target manifold, they prioritize samples matching local references and prune globally relevant tail modes, theoretically collapsing variance to point masses at a power-law rate. The authors propose constructing Wasserstein proxy references (geodesic interpolation/barycenters) across multiple islands to enable collaborative selection without sharing raw data.
