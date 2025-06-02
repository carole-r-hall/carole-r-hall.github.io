---
layout: page
title: Procrustes alignment
permalink: /procrustes-alignment/
---

The term _Procrustean_ describes something that ignores differences or special circumstances, or something known to enforce uniformity without flexibility. Greek mythology has stories of a bandit from Attica by the name of Procrustes (also called Prokoptas, Damastes, or Polypemon) who stretched or cut limbs off of people in an effort to get them to fit into an iron bed. Hence, why Procrustean now means the quality of forcing things to fit a desired mould, often without remorse or consideration. Procrustes was a son of Poseidon, and had a bed which he would offer to travelers coming through the road between Athens and Eleusis ("the sacred way"). Travelers who stayed in his bed were stretched if they were shorter than the bed, or their legs were cut off if they were larger than the bed, and no one ever was the perfect size. In fact, in Edgar Allan Poe referenced the Procrustean bed in his short story ["The Purloined Letter"](https://poestories.com/read/purloined), describing the firm method of investigation performed by the Parisian police. 

The first mathematical Procrustes problem was posed by Hurley and Catell in 1962, seeking to find a way to make a matrix $\boldsymbol{A}$ resemble a target matrix $\boldsymbol{B}$ (see [here](https://onlinelibrary.wiley.com/doi/10.1002/bs.3830070216) for that original paper). The authors here make reference to previously developed methods for aligning matrices, listing Tucker (Ledyard Tucker&mdash;of [Tucker's congruence coefficient](https://en.wikipedia.org/wiki/Congruence_coefficient)), Cattell (see [this study on factor analysis in personality tests](https://psycnet.apa.org/record/1956-02450-001)), and Ahmavaara ([Yrjö Ahmavaara](https://fi.wikipedia.org/wiki/Yrj%C3%B6_Ahmavaara), a multidisciplinaire and professor of theoretical physics, psychology, and mathematics who published the book Introduction to Factor Analysis&mdash;it doesn't seem likely that I can find a way to access this book, but if you look [here](https://www.cambridge.org/core/journals/psychometrika/article/abs/y-ahmavaara-and-t-markkanen-the-unified-factor-model-its-position-in-psychometric-theory-and-application-to-sociological-alcohol-study-vol-7-helsinki-the-finnish-foundation-for-alcohol-studies-1958-pp-187-stockholm-almqvist-and-wiksell-distributors/8FE3C993BEDABD9C603D67B38FD63A7C) there's some info on it). All of these players were big in a subject called [factor analysis](https://www.hawaii.edu/powerkills/UFA.HTM), a statistical method used to model observed variables as linear combinations of latent (or unobersved) factors. Formally, say we have a data matrix $X\in\mathbb{R}^{n\times{p}), with $n$ observations and $p$ measured variables. If we want to explain the covariance structure of $X$ using fewer unobserved variables, we have 

$$
X \approx FL^{\intercal} + \epsilon
$$

where $F\in\mathbb{R}^{n\times{k}}$ are the factor scores or latent variables, $L\in\mathbb{R}^{p\times{k}} are the factor loadings (i.e., how much a given variable is related to a latent variable), and $\epsilon$ are the residuals (also known as error or noise). Factor analysis, interestingly, is heavily used in psychology, often being used to determine things like intelligence or personality traits from questionnaires or test data. Specifically, this arises due to _rotational indeterminacy_ or solutions not being uniquely determined in factor loading matrices; i.e., 

$$
X \approx FL^{\intercal} \implies X \approx (FR^{\intercal})(LR^{\intercal})^{\intercal} = F'L'^{\intercal}
$$

where $R\in\mathbb{R}^{k\times{k}}$ is an orthogonal matrix. Let's run through a small example. Let's say we have a longitudinal study on personality development between year 1 (point A) and year 5 (point B), and we want to know whether the underlying latent traits (in a psychology study this might be something like Neuroticism) remain similar over the study's time period. However, factor analysis doesn't care about rotation, so the loading matrices from year 1, $L_1$, and year 2, $L_2$, are arbitrarily ordered/oriented. Even if the psychological structure we care about is stable through time, the matrices may look different just because of rotational indeterminacy. For example, in year 1, question 1 of the study may have the loadings (0.85, 0.10, 0.00), while in year 5, these loadings may be (0.05, 0.87, 0.02). These loadings are similar, just in different orders. Procrustes helps solve this problem and make the loading comparison meaningful. 

Anyway, back to Procrustes. Procrustes analysis is a geometric method used to compare two configurations of points, often used to align sets of landmarks by finding the ideal rotation transformation to match one configuration to the other. We solve the orthogonal procsustes problem, which is presented as:

$$
\min_{R\in{O}(k)}\VertAR - B\Vert_F
$$

Here, we're attemptping to align matrices $A$ and $B$ of comparable shapes. 

