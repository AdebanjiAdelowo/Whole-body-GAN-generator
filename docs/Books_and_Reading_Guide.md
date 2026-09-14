---
title: "Books and Reading Guide"
subtitle: "A Structured Study Roadmap for the InsetGAN Whole-Body Generation Project"
author: "Adebanji Oluwatimileyin Adelowo"
date: "2026"
toc: true
toc-depth: 3
number-sections: true
geometry: margin=1in
fontsize: 11pt
linkcolor: blue
urlcolor: blue
---

# How to Use This Guide

## Purpose

This document is a **study roadmap**, not a bibliography. Its companion,
`InsetGAN_Technical_Documentation.md`, is a self-contained technical account of
the Whole-Body GAN Generator system — a full-body human image generator that
composites a StyleGAN2-FFHQ face into a StyleGAN-Human body through InsetGAN's
joint latent optimisation, with ReStyle and PTI supplying the inversion of a
real user photograph.

That document cites 45 papers. Papers are the right medium for *what was
discovered*; they are a poor medium for *learning the machinery that makes the
discovery legible*. A reader who has not previously derived the Jensen--Shannon
form of the GAN objective, or who has not met block coordinate descent, or who
does not know why an SVM normal vector defines a semantic direction, will find
the papers opaque no matter how carefully they are written.

Every book below was chosen because it closes a specific gap of that kind, and
each entry states exactly which section of the technical document it supports
and which chapters to read. Books that are merely famous but do not serve this
project are deliberately excluded.

## The twelve books at a glance

| # | Book | Level | Role |
|---|---|---|---|
| 1 | Deisenroth, Faisal & Ong — *Mathematics for Machine Learning* | Foundational | Linear algebra, SVD, SVMs, optimisation |
| 2 | Goodfellow, Bengio & Courville — *Deep Learning* | Foundational | Canonical DL reference; original GAN authors |
| 3 | Casella & Berger — *Statistical Inference* | Foundational | Distributions, divergence, estimation |
| 4 | Bishop & Bishop — *Deep Learning: Foundations and Concepts* | Core | Modern, mathematically complete DL text |
| 5 | Prince — *Understanding Deep Learning* | Core | Clearest modern GAN/diffusion exposition |
| 6 | Szeliski — *Computer Vision: Algorithms and Applications* | Core | Image formation, alignment, compositing |
| 7 | Tomczak — *Deep Generative Modeling* | Core | Unified probabilistic view of generative models |
| 8 | Foster — *Generative Deep Learning* | Practical | Hands-on GAN/VAE/diffusion implementation |
| 9 | Langr & Bok — *GANs in Action* | Practical | GAN training dynamics and the ProGAN lineage |
| 10 | Boyd & Vandenberghe — *Convex Optimization* | Supplementary | Gradient methods, regularisation, alternating minimisation |
| 11 | Nocedal & Wright — *Numerical Optimization* | Supplementary | Line search, convergence, quasi-Newton |
| 12 | Murphy — *Probabilistic Machine Learning: Advanced Topics* | Advanced | Research-level reference on deep generative models |

## Classification key

- **Foundational** — prerequisite machinery. Read (or confirm you already know)
  before the technical document will make sense.
- **Core** — directly teaches the concepts the technical document uses.
- **Practical** — implementation-level understanding; read alongside the code.
- **Supplementary** — deepens a specific aspect, usually optimisation.
- **Advanced** — research-level reference for going beyond the project.

# Foundational Texts

## Mathematics for Machine Learning

**Full title:** *Mathematics for Machine Learning*
**Authors:** Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong
**Edition:** 1st **Publisher:** Cambridge University Press **Year:** 2020
**ISBN:** 978-1-108-45514-5 (paperback); 978-1-108-47004-9 (hardback)
**Publisher page:** <https://www.cambridge.org/9781108470049>
**Freely available:** <https://mml-book.github.io>
**Level:** Undergraduate to early graduate
**Classification:** Foundational

**Why it is relevant to this project.** This project's entire method is linear
algebra acting on a $512$-dimensional style space. Two of the three attribute
editing techniques the repository ships are, mathematically, textbook
procedures from this book applied to a GAN latent space — and this is the only
listed text that derives both.

**Specific topics it covers that appear in the technical document:**

- **Support vector machines** are exactly the mechanism of InterFaceGAN
  (technical document §11.2, reference [29]). InterFaceGAN fits a linear SVM
  separating latents labelled positive and negative for an attribute and uses
  the unit normal $n$ of the separating hyperplane as the edit direction,
  $w' = w + \alpha n$. Chapter 12 derives that hyperplane and its normal from
  first principles.
- **Eigendecomposition and SVD** are exactly the mechanism of SeFa (§11.2,
  reference [30]), which discovers edit directions in closed form as the top
  eigenvectors of $A^{\mathsf{T}}A$ for the first affine layer weight $A$.
  Chapter 4 is the required background.
- **Projections onto subspaces** underpin InterFaceGAN's conditional
  manipulation, which removes an interfering attribute by projecting out its
  normal, $n_1 - (n_1^{\mathsf{T}} n_2) n_2$ (§11.2).
- **Gradient descent and Lagrange multipliers** (Chapter 7) are the substrate
  of every optimisation in the document.
- **Vector calculus and the Jacobian** (Chapter 5) formalise
  $\partial G(w)/\partial w$, the generator Jacobian whose role is the crux of
  §9.9 — the explanation of why gradients can flow through two frozen,
  independently trained generators.

**Chapters to study:** 2 (Linear Algebra), 3 (Analytic Geometry), 4 (Matrix
Decompositions) — essential for SeFa; 5 (Vector Calculus) — essential for the
Jacobian argument; 7 (Continuous Optimization); 12 (Classification with SVMs)
— essential for InterFaceGAN.

**Connection to implementation.** Chapter 4 explains what
`edit/edit_helper.py`'s `encoder_sefa` is doing when it factorises a weight
matrix; Chapter 12 explains what `encoder_ifg` is doing with a precomputed
direction vector.

**Relationship to the cited papers.** Prerequisite for references [29]
(InterFaceGAN), [30] (SeFa) and [31] (StyleSpace).

## Deep Learning

**Full title:** *Deep Learning*
**Authors:** Ian Goodfellow, Yoshua Bengio, Aaron Courville
**Edition:** 1st **Publisher:** MIT Press **Year:** 2016
**ISBN:** 978-0-262-03561-3
**Publisher page:** <https://mitpress.mit.edu/9780262035613/deep-learning>
**Freely available:** <https://www.deeplearningbook.org>
**Level:** Graduate
**Classification:** Foundational

**Why it is relevant to this project.** Written by the first author of the
original GAN paper — reference [1] in the technical document — two years after
that paper. Section 20.10.4 is Goodfellow's own textbook treatment of the
method, and it is the best available statement of *why* the non-saturating loss
is used, which the technical document derives in §3.1.

**Specific topics it covers that appear in the technical document:**

- **The adversarial objective and its variants** (§20.10.4) — the minimax game,
  the optimal discriminator, and the saturating/non-saturating distinction
  derived in technical document §3.1.
- **Regularisation** (Chapter 7) — the conceptual basis for every regulariser
  in the project: InsetGAN's $\mathcal{L}_R$ mean-latent prior (§9.5), PTI's
  locality term (§8.3), and ReStyle's $\mathcal{W}$-norm loss (§7.4).
- **Optimization for training deep models** (Chapter 8) — including Adam, which
  is the optimiser `insetgan.py` instantiates twice, once per coordinate block
  (§10.6).
- **Convolutional networks** (Chapter 9) — required to follow the weight
  demodulation derivation in §6.3, which reasons about the variance of a
  convolution output.
- **Autoencoders and VAEs** (Chapters 14 and 20.10.3) — background for §4.1.
- **Representation learning** (Chapter 15) — the general framing behind
  disentanglement and the $\mathcal{Z}$ versus $\mathcal{W}$ argument in §3.2.

**Chapters to study:** 5 (Machine Learning Basics), 7 (Regularization),
8 (Optimization), 9 (Convolutional Networks), 15 (Representation Learning),
20.10 (Generative Models — especially 20.10.4 on GANs).

**Caveat.** Published in 2016, it predates StyleGAN entirely. Use it for
fundamentals and for the GAN objective; use books 4, 5 and 7 for anything
architecture-specific.

**Relationship to the cited papers.** Directly expands reference [1]
(Goodfellow et al., 2014) and provides background for [3] (DCGAN) and
[4] (WGAN).

## Statistical Inference

**Full title:** *Statistical Inference*
**Authors:** George Casella, Roger L. Berger
**Edition:** 2nd **Publisher:** Duxbury / Cengage Learning **Year:** 2002
**ISBN:** 978-0-534-24312-8
**Level:** Graduate
**Classification:** Foundational (optional, depending on background)

**Why it is relevant to this project.** The technical document's §3.1 shows
that the adversarial game, played to optimality, minimises the Jensen--Shannon
divergence between $p_{\text{data}}$ and $p_g$. That derivation is a
*distribution-matching* argument: it manipulates densities, expectations, and a
divergence, and concludes with a statement about when two distributions are
equal. A reader without solid measure-free probability will follow the algebra
but miss the meaning.

**Specific topics it covers that appear in the technical document:**

- **Probability theory, densities and transformations** (Chapters 1--4) — the
  machinery behind $p_z$, $p_g$, $p_{\text{data}}$ and the change of variables
  the generator performs when it "warps" an isotropic Gaussian onto the data
  manifold (§3.2).
- **Expectation and convergence** (Chapter 5) — required to read
  $\mathbb{E}_{z \sim p_z}[f(z)]$, the definition of the mean latent
  $\bar{w}$ underlying the truncation trick (§3.4).
- **Point estimation and loss functions** (Chapter 7) — why an $L_2$ objective
  recovers a conditional mean (hence blurry VAE samples, §4.1) while an $L_1$
  objective recovers a conditional median and is more robust to outliers. The
  latter is precisely the InsetGAN authors' stated reason for preferring $L_1$
  (§9.3).

**Chapters to study:** 1--2 (Probability, Transformations), 4 (Multiple Random
Variables), 5 (Properties of a Random Sample), 7 (Point Estimation).

**Substitution note.** If you have had a rigorous probability course, skip
this. Book 12 (Murphy) covers the same ground in an ML-native idiom, and Book 1
Chapter 6 gives a compressed version. This entry earns its place only for
readers who need the $L_1$-versus-$L_2$ estimator argument made rigorously.

**Relationship to the cited papers.** Supports the derivation attributed to
reference [1] and the design choice reported in reference [15] (InsetGAN).

# Core Texts

## Deep Learning: Foundations and Concepts

**Full title:** *Deep Learning: Foundations and Concepts*
**Authors:** Christopher M. Bishop, Hugh Bishop
**Edition:** 1st **Publisher:** Springer **Year:** 2024 (published November 2023)
**ISBN:** 978-3-031-45467-7 (hardcover); 978-3-031-45468-4 (eBook)
**DOI:** 10.1007/978-3-031-45468-4
**Publisher page:** <https://link.springer.com/book/10.1007/978-3-031-45468-4>
**Book site:** <https://www.bishopbook.com>
**Level:** Graduate
**Classification:** Core

**Why it is relevant to this project.** This is the single best modern
replacement for Goodfellow et al. for this project's purposes: it is
mathematically complete in Bishop's characteristic style, and — unlike the 2016
text — it covers GANs, normalising flows and diffusion models in one consistent
probabilistic framework. That matters here because the technical document ends
(§13) by analysing this project's replacement by a diffusion system, and Bishop
lets you hold both paradigms in the same notation.

**Specific topics it covers that appear in the technical document:**

- **Generative adversarial networks** — the adversarial objective, training
  dynamics, mode collapse and the practical difficulties catalogued in §4.2.
- **Continuous latent variables and VAEs** — the ELBO derivation reproduced in
  §4.1.
- **Normalisation layers** — background for AdaIN (§3.3) and for understanding
  what weight demodulation replaces (§6.3).
- **Diffusion models** — the foundation for §13.1's comparison of the 2022 GAN
  system against its 2026 diffusion successor (references [32], [33], [34]).
- **Gradient descent and backpropagation** — presented rigorously enough to
  support §9.9's Jacobian argument.

**Chapters to study:** 6--7 (Deep Neural Networks, Gradient Descent),
9 (Regularization), 10 (Convolutional Networks), 17 (Generative Adversarial
Networks), 19 (Autoencoders / latent variable models), 20 (Diffusion Models).

**Connection to implementation.** Bishop's treatment of normalisation makes
clear why AdaIN's *data-dependent* normalisation gives the generator an
exploitable degree of freedom, which is the root cause of the droplet artefact
and the entire motivation for the demodulation formula derived in §6.3 and
quoted from `edit/edit_helper.py`.

**Relationship to the cited papers.** Best single companion to references [6]
and [7] (StyleGAN, StyleGAN2) and to [32]--[34] (diffusion lineage).

## Understanding Deep Learning

**Full title:** *Understanding Deep Learning*
**Author:** Simon J. D. Prince
**Edition:** 1st **Publisher:** MIT Press **Year:** 2023
**ISBN:** 978-0-262-04864-4
**Publisher page:** <https://mitpress.mit.edu/9780262048644/understanding-deep-learning>
**Freely available:** <https://udlbook.github.io/udlbook/>
**Level:** Advanced undergraduate to graduate
**Classification:** Core

**Why it is relevant to this project.** Prince's Chapter 15 is, in this
author's assessment, the clearest short exposition of GANs in any textbook, and
Chapter 15 specifically discusses **StyleGAN** and **progressive growing** —
the two architectural ideas the technical document identifies in §4.3 as the
discontinuity that made everything downstream possible. Very few textbooks go
that far into the StyleGAN family.

**Specific topics it covers that appear in the technical document:**

- **GANs, DCGAN, progressive growing, StyleGAN** (Chapter 15) — covering
  §4.3, §5.1 and §5.2 of the technical document almost exactly.
- **Mode collapse and training instability** (Chapter 15) — the failure modes
  catalogued in §4.2.
- **VAEs** (Chapter 17) — §4.1.
- **Diffusion models** (Chapter 18) — §13.1.
- **Regularisation** (Chapter 9) and **normalisation** (Chapter 11) — §3.3 and
  §6.3.

**Chapters to study:** 10--11 (Convolutional Networks, Residual Networks and
Normalization), 15 (Generative Adversarial Networks) — the single most
important chapter in this guide, 17 (Variational Autoencoders),
18 (Diffusion Models).

**Practical note.** The book is free from the author's site, includes Python
notebooks per chapter, and its figures are unusually good at conveying latent
space geometry — worth reading Chapter 15's figures before attempting §3.2 of
the technical document.

**Relationship to the cited papers.** The best textbook bridge to references
[3], [5], [6], [7] (DCGAN through StyleGAN2).

## Computer Vision: Algorithms and Applications

**Full title:** *Computer Vision: Algorithms and Applications*
**Author:** Richard Szeliski
**Edition:** 2nd **Publisher:** Springer (Texts in Computer Science) **Year:** 2022
**ISBN:** 978-3-030-34371-2 (hardcover); 978-3-030-34372-9 (eBook)
**DOI:** 10.1007/978-3-030-34372-9
**Publisher page:** <https://link.springer.com/book/10.1007/978-3-030-34372-9>
**Freely available:** <https://szeliski.org/Book/>
**Level:** Graduate
**Classification:** Core

**Why it is relevant to this project.** InsetGAN is, at bottom, an **image
compositing** method. It just performs the compositing in latent space instead
of pixel space. To appreciate why that reformulation is clever, you need to
know what the classical pixel-space alternatives are and why they are
unsatisfying here — and this is the standard reference for exactly that
material.

**Specific topics it covers that appear in the technical document:**

- **Image blending and compositing** — feathering, Laplacian pyramid blending
  and Poisson image editing. Technical document §9.4 notes that InsetGAN uses
  *none* of these: the composite is a hard rectangular replacement, and
  seamlessness is instead obtained by constraining an 8-pixel border in latent
  space. That contrast is only meaningful if you know the alternatives.
- **Image alignment and warping** — the similarity transform underlying the
  FFHQ face alignment that §10.4 identifies as a hard prerequisite for
  inversion, and the pose-based body alignment central to StyleGAN-Human
  (reference [13], §5.3).
- **Feature detection and face detection** — the dlib CNN detector and 68-point
  landmark predictor used throughout the pipeline (reference [39]).
- **Image processing and convolution** (Chapter 3) — the operator whose output
  variance is computed in the demodulation derivation (§6.3).
- **Resampling and interpolation** — what `F.interpolate(..., mode='area')` is
  doing in `loss_coarse` and in the final crop resize (§10.6).

**Chapters to study:** 2 (Image Formation), 3 (Image Processing — especially
pyramids and compositing), 5--6 (Deep Learning, Recognition) for context,
8 (Motion Estimation) for image alignment.

**Relationship to the cited papers.** Background for references [13]
(StyleGAN-Human's alignment-centric thesis), [17] (CoModGAN inpainting
baseline), [28] (FaceNet) and [39] (dlib).

## Deep Generative Modeling

**Full title:** *Deep Generative Modeling*
**Author:** Jakub M. Tomczak
**Edition:** 2nd **Publisher:** Springer **Year:** 2024
**ISBN:** 978-3-031-64086-5 (hardcover); 978-3-031-64087-2 (eBook)
**DOI:** 10.1007/978-3-031-64087-2
**Publisher page:** <https://link.springer.com/book/10.1007/978-3-031-64087-2>
**Level:** Graduate
**Classification:** Core

**Why it is relevant to this project.** This is the only book in the list
devoted *entirely* to generative modelling, and it presents autoregressive
models, flows, VAEs, GANs, energy-based models and diffusion in a single
probabilistic frame. The technical document's §4 and §13 both make comparative
claims across these families — why VAEs blur, why GANs won at $1024^2$ in 2022,
why diffusion overtook them — and Tomczak supplies the common vocabulary those
comparisons require.

**Specific topics it covers that appear in the technical document:**

- **Why likelihood-based models blur and adversarial models do not** — the
  argument in §4.1 justifying a GAN synthesis backbone with a *separately
  trained encoder* for inversion, rather than a VAE.
- **Latent variable models and the structure of latent spaces** — §3.2.
- **GANs as implicit generative models** — the fact that $p_g$ is defined only
  through the sampling procedure, which is why inversion (§3.5) is a genuine
  research problem rather than a lookup.
- **Diffusion and score-based models** — §13.1.
- **Hybrid modelling** — relevant to the idea, central to InsetGAN, of
  combining several trained models rather than training one.

**Chapters to study:** the introductory chapter on the probabilistic framing;
the VAE chapter; the GAN chapter; the diffusion chapter; and the concluding
chapters on combining model families.

**Relationship to the cited papers.** A unifying companion to references [1],
[2], [32] and [33].

# Practical Texts

## Generative Deep Learning

**Full title:** *Generative Deep Learning: Teaching Machines to Paint, Write,
Compose, and Play*
**Author:** David Foster
**Edition:** 2nd **Publisher:** O'Reilly Media **Year:** 2023
**ISBN:** 978-1-098-13418-1 (print); 978-1-098-13414-3 (digital)
**Publisher page:** <https://www.oreilly.com/library/view/generative-deep-learning/9781098134174/>
**Level:** Practitioner / intermediate
**Classification:** Practical

**Why it is relevant to this project.** The second edition covers VAEs, GANs
(including WGAN-GP, **ProGAN and StyleGAN**), transformers, and diffusion
models with runnable code. Its StyleGAN treatment is the fastest route to an
operational understanding of the mapping network, AdaIN injection and style
mixing described in technical document §6, and its diffusion chapter directly
supports §13.

**Specific topics it covers that appear in the technical document:**

- **StyleGAN and StyleGAN2 architecture** — the mapping network $\mathcal{Z}
  \to \mathcal{W}$, per-layer style injection, the truncation trick and style
  mixing (§3.2--3.4, §6.1--6.5).
- **GAN training dynamics** — mode collapse, discriminator/generator balance
  (§4.2).
- **VAEs** — §4.1.
- **Diffusion models** — §13.1, and the IP-Adapter/SDXL successor discussion
  (references [33], [34], [35]).

**Chapters to study:** the VAE chapter; the GAN chapters (including WGAN-GP);
the chapter covering ProGAN/StyleGAN/StyleGAN2 — most relevant to §6; the
diffusion chapter — most relevant to §13.

**How it connects to the implementation.** Reading Foster's StyleGAN chapter
before opening `torch_utils/models.py` in this repository converts an opaque
generator class into a recognisable structure: you will be able to point at the
mapping MLP (`n_mlp: 8`), the per-layer affine transforms, the modulated
convolutions and the `toRGB` skip branches described in §6.2.

**Caveat.** Implementations are in TensorFlow/Keras while this project is
PyTorch. Read it for architecture, not for API.

## GANs in Action

**Full title:** *GANs in Action: Deep Learning with Generative Adversarial Networks*
**Authors:** Jakub Langr, Vladimir Bok
**Edition:** 1st **Publisher:** Manning Publications **Year:** 2019
**ISBN:** 978-1-617-29556-0
**Publisher page:** <https://www.manning.com/books/gans-in-action>
**Code:** <https://github.com/GANs-in-Action/gans-in-action>
**Level:** Practitioner / intermediate
**Classification:** Practical

**Why it is relevant to this project.** This is the listed book most focused on
**GAN training dynamics and optimisation as a practical matter** — the
coordinator's specific criterion. Chapter 6, "Progressing with GANs," is a
dedicated treatment of Progressive GAN, which technical document §4.3 and §5.1
identify as the breakthrough that made $1024 \times 1024$ synthesis feasible
and which StyleGAN inherited.

**Specific topics it covers that appear in the technical document:**

- **Progressive growing** (Chapter 6) — reference [5], discussed in §4.3 and
  §5.1, including why StyleGAN2 later *removed* it to eliminate phase artefacts
  (§5.2).
- **Training dynamics, mode collapse, Nash equilibrium** — the practical
  failure modes of §4.2, treated at length rather than in passing.
- **DCGAN** (reference [3]) — the architectural recipe described in §4.2.
- **Evaluation of GANs** — Inception Score and FID (reference [12]). This
  supports one of the technical document's more important methodological
  points, in §13.6: FID is more sensitive to diversity than to per-image
  quality, which is why InsetGAN's face-refinement FID barely moves ($25.33$
  versus $26.67$) despite a 98% human preference.

**Chapters to study:** 2--3 (Autoencoders, first GAN), 4 (DCGAN), 5 (Training
and common challenges) — the core of the book for this project, 6 (Progressive
GAN), and the evaluation chapter.

**Caveat.** Published 2019, Keras-based, and predates StyleGAN2. Read it for
training dynamics and ProGAN; do not rely on it for anything after 2019.

**Relationship to the cited papers.** Expands references [3], [5] and [12].

# Supplementary: Optimisation

Both books in this section address the same gap. The technical document's core
method — InsetGAN's joint optimisation (§9) — is a **nonconvex, alternating,
gradient-based optimisation over two latent codes with a moving constraint
region**. No convex-optimisation text covers that problem directly, and this
guide does not pretend otherwise. What these books supply is the vocabulary and
the intuition for the pieces: what a gradient step does, what regularisation
buys, why alternating minimisation is a reasonable strategy, and how learning
rate schedules affect convergence.

## Convex Optimization

**Full title:** *Convex Optimization*
**Authors:** Stephen Boyd, Lieven Vandenberghe
**Edition:** 1st **Publisher:** Cambridge University Press **Year:** 2004
**ISBN:** 978-0-521-83378-3
**Publisher page:** <https://www.cambridge.org/9780521833783>
**Freely available:** <https://web.stanford.edu/~boyd/cvxbook/>
**Level:** Graduate
**Classification:** Supplementary

**Why it is relevant to this project — stated precisely.** The InsetGAN
objective is *not* convex, and nothing in this book proves anything about it.
Include it only for these specific chapters, each of which maps to a concrete
construct in the technical document:

- **Chapter 9 (Unconstrained minimization)** — gradient descent, descent
  directions, step-size selection and convergence behaviour. This is the
  formal backing for §9.9's claim that gradient descent on $w$ is well behaved,
  and for §5.2's observation that StyleGAN2's path-length regularisation makes
  the parameterisation *better conditioned* — conditioning being a concept this
  chapter defines precisely.
- **Chapter 6 (Approximation and fitting)** — regularised approximation and the
  $\ell_1$-versus-$\ell_2$ trade-off, including robustness to outliers. This is
  the formal version of the InsetGAN authors' empirical preference for $L_1$
  (§9.3), and of the penalty structure of $\mathcal{L}_R$ (§9.5).
- **Section 4.2 and the discussion of alternating minimisation / block
  coordinate methods** — the structure of InsetGAN's alternating schedule
  (§9.8), where $w_A$ is frozen while $w_B$ is optimised and vice versa. Each
  *block* subproblem is far better behaved than the joint problem, which is
  precisely the argument the technical document makes.

**Chapters to study:** 2--3 (Convex sets and functions) for vocabulary,
6 (Approximation and fitting), 9 (Unconstrained minimization). Chapters 4, 5,
10 and 11 are not needed for this project.

**Honest limitation.** If your goal is only to understand the technical
document, Book 1 Chapter 7 plus Book 2 Chapter 8 may suffice. Read Boyd if you
want the underlying theory of why gradient methods behave as they do.

## Numerical Optimization

**Full title:** *Numerical Optimization*
**Authors:** Jorge Nocedal, Stephen J. Wright
**Edition:** 2nd **Publisher:** Springer (Springer Series in Operations Research
and Financial Engineering) **Year:** 2006
**ISBN:** 978-0-387-30303-1
**DOI:** 10.1007/978-0-387-40065-5
**Publisher page:** <https://link.springer.com/book/10.1007/978-0-387-40065-5>
**Level:** Graduate
**Classification:** Supplementary

**Why it is relevant to this project.** Where Boyd is theory for convex
problems, Nocedal and Wright is the standard reference for **nonconvex,
iterative numerical optimisation** — which is what InsetGAN, PTI's inversion
stage and the StyleGAN2 projector all actually are.

**Specific topics it covers that appear in the technical document:**

- **Line search methods and step-length selection** (Chapters 3) — the
  conceptual home of the learning-rate schedule derived in §10.6, with its
  linear warm-up over the first 5% of steps and cosine ramp-down over the final
  25%.
- **Convergence of iterative methods and fixed-point iteration** — the formal
  framing of ReStyle's recurrence $w_{t+1} = w_t + E(x \,\|\, G(w_t))$ (§7.2),
  which the technical document identifies as a fixed-point iteration whose
  fixed point satisfies $E(x \,\|\, G(w^\star)) = 0$.
- **Quasi-Newton and momentum-adjacent methods** — background for Adam, used
  with $\beta = (0.9, 0.999)$ in both InsetGAN optimisers (§10.6).
- **Nonlinear least squares** — the general shape of every reconstruction loss
  in §7, §8 and §9.

**Chapters to study:** 2 (Fundamentals of Unconstrained Optimization),
3 (Line Search Methods) — most directly relevant, 6 (Quasi-Newton Methods),
10 (Least-Squares Problems).

**Relationship to the cited papers.** Supports the optimisation procedures in
references [19] (Image2StyleGAN), [23] (PTI) and [15] (InsetGAN).

# Advanced Reference

## Probabilistic Machine Learning: Advanced Topics

**Full title:** *Probabilistic Machine Learning: Advanced Topics*
**Author:** Kevin P. Murphy
**Edition:** 1st **Publisher:** MIT Press (Adaptive Computation and Machine
Learning series) **Year:** 2023
**ISBN:** 978-0-262-04843-9
**Publisher page:** <https://mitpress.mit.edu/9780262048439/>
**Freely available (draft):** <https://probml.github.io/pml-book/book2.html>
**Level:** Research
**Classification:** Advanced

**Why it is relevant to this project.** At roughly 1{,}360 pages this is a
reference, not a read-through. Its value here is that it treats **deep
generative models** — VAEs, GANs, normalising flows, diffusion, energy-based
models — at research depth and in one notation, and it covers two topics no
other book in this list handles well: **amortised inference** and
**representation learning / disentanglement**.

**Specific topics it covers that appear in the technical document:**

- **Amortised inference** — the exact concept behind encoder-based GAN
  inversion. pSp, e4e and ReStyle (references [20], [21], [22]) all *amortise*
  a per-image optimisation into a learned feed-forward network. Technical
  document §5.4 describes this transition without naming the general principle;
  Murphy names and formalises it.
- **Disentangled representations** — the formal treatment of the property
  §3.2 attributes to $\mathcal{W}$ relative to $\mathcal{Z}$.
- **GANs, divergence minimisation and $f$-divergences** — generalising the
  Jensen--Shannon result derived in §3.1.
- **Diffusion and score-based models** — the deepest treatment in this list,
  supporting §13.1.
- **Model combination** — the general principle InsetGAN instantiates.

**Chapters to study (as reference, not linearly):** the deep generative models
part in full — the VAE chapters, the GAN chapter, the diffusion chapter — plus
the representation learning chapter.

**Relationship to the cited papers.** The best single reference for going
*beyond* the technical document: references [18] (GAN Inversion survey), [32],
[33] and [36] (EG3D) all become more tractable after the relevant Murphy
chapters.

# Reading Roadmap

## Recommended order

The guide assumes you want to reach a working understanding of the technical
document and then of the code. Four tracks are given; pick by background.

**Track A — Newcomer to deep generative modelling (roughly 3--4 months)**

1. Book 1 (*Mathematics for ML*), Chapters 2--5, 7 — build the linear algebra
   and calculus base.
2. Book 5 (*Understanding Deep Learning*), Chapters 10--11, then 15 — get GANs
   and StyleGAN in the clearest available form.
3. Read technical document §1--§3 — problem definition and mathematical
   foundations.
4. Book 2 (*Deep Learning*), Chapters 7--9, 20.10 — regularisation,
   optimisation, convolutions, the GAN objective.
5. Read technical document §4--§6 — generative lineage and StyleGAN2 internals.
6. Book 8 (*Generative Deep Learning*), StyleGAN and GAN chapters — make it
   concrete with code.
7. Read technical document §7--§9 — inversion, PTI, and the InsetGAN objective.
8. Book 1 Chapters 4 and 12 — SeFa and InterFaceGAN prerequisites.
9. Read technical document §10--§14 with the repository open alongside.

**Track B — Comfortable with deep learning, new to GANs (4--6 weeks)**

1. Book 5, Chapter 15 (GANs).
2. Book 9 (*GANs in Action*), Chapters 4--6 — DCGAN, training dynamics,
   progressive growing.
3. Technical document §3--§6.
4. Book 4 (*Bishop & Bishop*), Chapter 17, plus the normalisation material.
5. Technical document §7--§9, then Book 11 (*Numerical Optimization*)
   Chapters 2--3 for the optimisation machinery.
6. Technical document §10--§14.

**Track C — Focused on the InsetGAN method itself (1--2 weeks)**

1. Book 5, Chapter 15 — StyleGAN context.
2. Technical document §3.2 ($\mathcal{Z}$/$\mathcal{W}$/$\mathcal{W}+$) and
   §6 (StyleGAN2).
3. Book 6 (*Szeliski*), Chapter 3 compositing sections — understand the
   pixel-space alternatives InsetGAN declines to use.
4. Book 10 (*Boyd*), Chapter 9 and the alternating-minimisation discussion.
5. Technical document §9 in full, then read `insetgan.py` alongside §10.6.

**Track D — Extending the project (ongoing)**

1. Book 7 (*Tomczak*) end to end — unified view of model families.
2. Book 12 (*Murphy*) — amortised inference and representation learning
   chapters.
3. Book 4, Chapter 20 (diffusion) and Book 5, Chapter 18.
4. Technical document §13, then the successor repository.

## Minimum viable path

If you read only three things: Book 5 Chapter 15, technical document §9, and
`insetgan.py`. That triple is sufficient to understand what the project does
and why it works.

# Coverage Matrix

Each row is a section of `InsetGAN_Technical_Documentation.md`; each entry
names the books that support it. This table is the consistency check between
the two documents.

| Technical document section | Supporting books |
|---|---|
| §2 Problem definition | 5, 7 |
| §3.1 Adversarial objective, minimax, JSD | 2 (20.10), 3, 4, 5 (15), 12 |
| §3.2 $\mathcal{Z}$ / $\mathcal{W}$ / $\mathcal{W}+$, disentanglement | 2 (15), 5 (15), 8, 12 |
| §3.3 AdaIN and style modulation | 4, 5 (11), 8 |
| §3.4 Truncation trick | 5 (15), 8, 9 |
| §3.5 GAN inversion as optimisation | 10 (9), 11 (2--3), 12 |
| §3.6 Perceptual, LPIPS and identity losses | 2 (9), 6 |
| §4.1 VAEs | 2 (14, 20.10.3), 4 (19), 5 (17), 7 |
| §4.2 Early GAN failure modes | 5 (15), 9 (5), 4 (17) |
| §4.3 Progressive growing, StyleGAN leap | 5 (15), 9 (6), 8 |
| §5 Research evolution (DCGAN to compositing) | 5 (15), 8, 9 |
| §6.1--6.2 Mapping and synthesis networks | 8, 5 (15), 4 |
| §6.3 Weight demodulation derivation | 2 (9), 4, 6 (3) |
| §6.5 Style mixing | 8, 5 (15) |
| §7 ReStyle iterative refinement | 11 (2--3), 12 (amortised inference) |
| §8 Pivotal Tuning Inversion | 2 (7), 10 (6), 11 (3) |
| §9.3--9.7 InsetGAN loss decomposition | 3 (7), 6 (3), 10 (6) |
| §9.8 Alternating optimisation schedule | 10 (9), 11 (2--3) |
| §9.9 Gradients through frozen generators | 1 (5), 4, 10 (9) |
| §10 System and code implementation | 6, 8 |
| §11.2 Attribute editing (InterFaceGAN, SeFa, StyleSpace) | 1 (4, 12), 8 |
| §11.3 Style mixing experiments | 8, 5 (15) |
| §12 Limitations, FID critique | 9 (evaluation), 12 |
| §13 GAN-to-diffusion transition | 4 (20), 5 (18), 7, 8, 12 |
| §13.3 3D-aware generation | 6, 12 |

## Gaps this guide does not close

Stated explicitly, because a reference document should be honest about its
limits:

- **No book covers InsetGAN itself.** The method is from 2022 and appears in no
  textbook. Reference [15] and technical document §9 are the only treatments.
- **No book covers ReStyle or PTI.** These are 2021--2022 results; references
  [22] and [23] with technical document §7--§8 are the sources.
- **No book covers StyleGAN-Human.** Reference [13] and §5.3.
- **Deployment, serving and mobile integration** are outside all twelve books.
  Technical document §10.8--§10.9 and §12.6--§12.7 stand alone, including the
  security findings, which are engineering practice rather than research
  content.
- **IP-Adapter and SDXL** (references [34], [35]) postdate every book here
  except in general diffusion terms.

# Bibliographic Summary

1. Deisenroth, M. P., Faisal, A. A., and Ong, C. S. (2020). *Mathematics for
   Machine Learning.* Cambridge University Press.
   ISBN 978-1-108-45514-5. <https://mml-book.github.io>

2. Goodfellow, I., Bengio, Y., and Courville, A. (2016). *Deep Learning.*
   MIT Press. ISBN 978-0-262-03561-3. <https://www.deeplearningbook.org>

3. Casella, G., and Berger, R. L. (2002). *Statistical Inference*, 2nd ed.
   Duxbury / Cengage Learning. ISBN 978-0-534-24312-8.

4. Bishop, C. M., and Bishop, H. (2024). *Deep Learning: Foundations and
   Concepts.* Springer. ISBN 978-3-031-45467-7.
   DOI: 10.1007/978-3-031-45468-4.

5. Prince, S. J. D. (2023). *Understanding Deep Learning.* MIT Press.
   ISBN 978-0-262-04864-4. <https://udlbook.github.io/udlbook/>

6. Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*, 2nd ed.
   Springer, Texts in Computer Science. ISBN 978-3-030-34371-2.
   DOI: 10.1007/978-3-030-34372-9. <https://szeliski.org/Book/>

7. Tomczak, J. M. (2024). *Deep Generative Modeling*, 2nd ed. Springer.
   ISBN 978-3-031-64086-5. DOI: 10.1007/978-3-031-64087-2.

8. Foster, D. (2023). *Generative Deep Learning: Teaching Machines to Paint,
   Write, Compose, and Play*, 2nd ed. O'Reilly Media.
   ISBN 978-1-098-13418-1.

9. Langr, J., and Bok, V. (2019). *GANs in Action: Deep Learning with
   Generative Adversarial Networks.* Manning Publications.
   ISBN 978-1-617-29556-0.

10. Boyd, S., and Vandenberghe, L. (2004). *Convex Optimization.* Cambridge
    University Press. ISBN 978-0-521-83378-3.
    <https://web.stanford.edu/~boyd/cvxbook/>

11. Nocedal, J., and Wright, S. J. (2006). *Numerical Optimization*, 2nd ed.
    Springer. ISBN 978-0-387-30303-1. DOI: 10.1007/978-0-387-40065-5.

12. Murphy, K. P. (2023). *Probabilistic Machine Learning: Advanced Topics.*
    MIT Press. ISBN 978-0-262-04843-9.
    <https://probml.github.io/pml-book/book2.html>
