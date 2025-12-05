# Methodology: Signal-Preserving Foreground Removal via Inter-Scale Learning

## Introduction to Signal-Preserving Machine Learning Framework

The Internal Linear Combination (ILC) method has been successfully applied to many CMB datasets for component separation. However, many real-world foreground signals exhibit non-Gaussian and anisotropic properties that are not fully captured by linear methods. This motivates the application of machine learning (ML) solutions, which can learn non-linear mappings between observables. However, a key challenge in applying ML to CMB component separation is ensuring that the primary CMB signal remains unbiased and preserved throughout the reconstruction process.

Following the framework introduced by McCarthy et al. (see Section~\ref{sec:signal_preserving_vs_direct} in \cite{mccarthy2024}), we adopt a signal-preserving approach that guarantees unbiased reconstruction of the component of interest (COI). The core principle is to construct neural network inputs that are explicitly signal-free, ensuring that the network cannot learn or bias the primary CMB signal from imperfect simulations or modeling misspecifications.

Following the McCarthy framework, we use the following observation model for multi-frequency CMB maps. In this work, we work with only B-mode polarization maps:
\begin{equation}
B_i(\hat{\mathbf{n}}) = a_i S^{\mathrm{coi}}(\hat{\mathbf{n}}) + F_i(\hat{\mathbf{n}}) + N_i(\hat{\mathbf{n}}) \label{eq:general_obs_model}
\end{equation}
where $i$ indicates a frequency channel, $\hat{\mathbf{n}}$ is a unit vector on the sphere, $S^{\mathrm{coi}}(\hat{\mathbf{n}})$ is the component of interest, $a_i$ is the spectral energy distribution (SED) vector describing the frequency dependence of the COI at channel $i$, and $F_i(\hat{\mathbf{n}})$ are foregrounds and noise respectively, which are uncorrelated with $S^{\mathrm{coi}}(\hat{\mathbf{n}})$. For the CMB in CMB temperature units, $a_i = 1$ for all frequencies since the CMB follows a blackbody spectrum. For primordial CMB B-modes, which follow a blackbody spectrum, we have $a_i = 1$ for all frequencies when working in appropriate polarization units.

The goal is to design a neural network $\mathcal{N}$ to learn a mapping between signal free inputs and outputs that can be used to separate and remove unwanted components from the contaminated maps and recover the COI. The key to constructing a neural network that obeys signal-preservation is two-fold:

1. **Signal-free inputs**: First, the network inputs are constructed to be uncorrelated with the COI. This ensures the network cannot learn anything about $S_L$ from the inputs themselves and thereby cannot bias the primary CMB signal from e.g. modeling misspecification of imperfect simulations. However, the inputs need to be correlated with the components (i.e foregrounds) that one wishes to remove. Overall, this constraint gives rise to a network that is a function of only signal-free information.

2. **Signal-free targets**: Second, the network is constrained to predict components that are uncorrelated with the COI, so that subtracting the prediction from the contaminated map does not inadvertently remove or bias $S_L$.

This dual constraint ensures that even if the network's learned mapping is imperfect or if simulations contain modeling errors, the primary CMB signal remains unbiased. In the following sections, we describe the various configurations of this core framework to perform CMB reconstruction. In all cases discussed in this work, the component of interest is the large-scale primary CMB B-mode polarization, denoted by $S^{\mathrm{coi}}_L(\hat{\mathbf{n}})$.

##  Inter-Scale Learning: Small-scale B-modes Input

We first specialize this framework to the case of component separation via inter-scale learning, focusing on the reconstruction of large-scale primordial B-mode polarization maps from small-scale B-mode polarization inputs. This configuration exploits the correlation between small-scale and large-scale foreground structures. We emphasize that this method operates on single-frequency maps decomposed by angular scale. As a result, the frequency-dependent SED factor $a_i$ does not explicitly appear in our scale-decomposed observation model (Equation~\eqref{eq:B_large}) for this case.

We consider B-mode polarization maps $B(\hat{\mathbf{n}})$ that contain both the primary CMB signal $S(\hat{\mathbf{n}})$ and foreground contamination $F(\hat{\mathbf{n}})$:
\begin{equation}
B(\hat{\mathbf{n}}) = S(\hat{\mathbf{n}}) + F(\hat{\mathbf{n}}) \label{eq:B_obs_model}
\end{equation}

We decompose these maps into large-scale ($\ell < 200$) and small-scale ($\ell > 200$) components:
\begin{align}
B_L(\hat{\mathbf{n}}) &= S_L(\hat{\mathbf{n}}) + F_L(\hat{\mathbf{n}}) \label{eq:B_large} \\
B_S(\hat{\mathbf{n}}) &= S_S(\hat{\mathbf{n}}) + F_S(\hat{\mathbf{n}}) \label{eq:B_small}
\end{align}

Notice that this setup obeys the two-fold constraint previously described because while the primary CMB components $S_L$ and $S_S$ are statistically independent (as expected for a Gaussian random field), the foreground components $F_L$ and $F_S$ are correlated due to the underlying physical processes that couple them in the Galactic field.


----
We design a neural network $\mathcal{N}$ to learn a mapping from signal-free inputs to foreground predictions. The signal-preservation property is achieved through two complementary mechanisms:

1. **Signal-free inputs**: The network inputs are constructed to be uncorrelated with the COI, ensuring the network cannot learn anything about $S_L$ from the inputs themselves.

2. **Signal-free targets**: The network is trained to predict foreground components that are uncorrelated with the COI, so that subtracting the prediction from the contaminated map does not inadvertently remove or bias $S_L$.

This dual constraint ensures that even if the network's learned mapping is imperfect or if simulations contain modeling errors, the primary CMB signal remains unbiased.

## Scenario 1: Inter-Scale Correlation Learning

In the most basic scenario, we use small-scale B-mode maps $B_S(\hat{\mathbf{n}})$ as input to predict large-scale foregrounds $F_L(\hat{\mathbf{n}})$. During training, we remove the small-scale primary CMB component $S_S(\hat{\mathbf{n}})$ from the network input. Given that the CMB is a Gaussian random field, $S_S(\hat{\mathbf{n}})$ is statistically independent of both $F_S(\hat{\mathbf{n}})$ and $F_L(\hat{\mathbf{n}})$, and would act as a noise-like contaminant if included, potentially hindering the network's ability to learn the foreground inter-scale correlations. Isolating $F_S(\hat{\mathbf{n}})$ as the input focuses the network on the characteristics of the foregrounds, which are the target of the reconstruction.

The neural network prediction of the large-scale foregrounds is thus a function solely of the small-scale foregrounds:
\begin{equation}
\hat{F}_L(\hat{\mathbf{n}}) = \mathcal{N}(B_S(\hat{\mathbf{n}})) = \mathcal{N}(F_S(\hat{\mathbf{n}})) \label{eq:F_L_prediction}
\end{equation}

We then subtract this prediction from the contaminated large-scale map to obtain the neural network's reconstruction:
\begin{equation}
\hat{B}_L(\hat{\mathbf{n}}) = B_L(\hat{\mathbf{n}}) - \hat{F}_L(\hat{\mathbf{n}}) = S_L(\hat{\mathbf{n}}) + F_L(\hat{\mathbf{n}}) - \mathcal{N}(F_S(\hat{\mathbf{n}})) \label{eq:B_L_reconstruction}
\end{equation}

This approach investigates the extent to which correlations exist between $F_S$ and $F_L$, and whether these correlations can be captured by a non-linear mapping learned from data.

## Scenario 2: Augmented Inputs with Temperature and E-Mode Information

To improve the foreground prediction, we augment the inputs with additional information that is uncorrelated with the COI but potentially correlated with the foregrounds. Specifically, we include CMB temperature maps $T(\hat{\mathbf{n}})$ and E-mode polarization maps $E(\hat{\mathbf{n}})$:
\begin{align}
T(\hat{\mathbf{n}}) &= S_T(\hat{\mathbf{n}}) + F_T(\hat{\mathbf{n}}) \label{eq:T_map} \\
E(\hat{\mathbf{n}}) &= S_E(\hat{\mathbf{n}}) + F_E(\hat{\mathbf{n}}) \label{eq:E_map}
\end{align}

### Motivation for Temperature and E-Mode Augmentation
<to be filled>

### Signal Preservation with Augmented Inputs

Critically, this augmentation strategy maintains signal preservation for the large-scale B-mode CMB signal $S_L^B(\hat{\mathbf{n}})$. This relies on the statistical properties of the primordial CMB: in standard inflationary cosmology, the temperature ($S_T$), E-mode ($S_E$), and primordial B-mode ($S^B$) components are statistically independent Gaussian random fields. Therefore, providing the full $T(\hat{\mathbf{n}})$ and $E(\hat{\mathbf{n}})$ maps (which contain $S_T$ and $S_E$ across all scales) as input does not introduce information that is correlated with the large-scale primordial B-mode signal $S_L^B$ that we aim to preserve.

During training, we discard the primary CMB components $S_T$ and $S_E$ from the temperature and E-mode inputs, as they act as uncorrelated random noise that is not correlated with either the foregrounds or the COI. The network's prediction $\hat{F}_L^B(\hat{\mathbf{n}})$ remains statistically independent of $S_L^B(\hat{\mathbf{n}})$, ensuring that:
\begin{equation}
\langle \hat{B}_L(\hat{\mathbf{n}}) S_L^B(\hat{\mathbf{n}}) \rangle = \langle S_L^B(\hat{\mathbf{n}}) S_L^B(\hat{\mathbf{n}}) \rangle \label{eq:signal_preservation}
\end{equation}

The augmented network prediction is:
\begin{equation}
\hat{F}_L^B(\hat{\mathbf{n}}) = \mathcal{N}_{\text{aug}}(B_S(\hat{\mathbf{n}}), T(\hat{\mathbf{n}}), E(\hat{\mathbf{n}})) = \mathcal{N}_{\text{aug}}(F_S^B(\hat{\mathbf{n}}), F_T(\hat{\mathbf{n}}), F_E(\hat{\mathbf{n}})) \label{eq:F_L_aug_prediction}
\end{equation}

And the final cleaned reconstruction is:
\begin{equation}
\hat{B}_L(\hat{\mathbf{n}}) = B_L(\hat{\mathbf{n}}) - \hat{F}_L^B(\hat{\mathbf{n}}) = S_L(\hat{\mathbf{n}}) + F_L(\hat{\mathbf{n}}) - \mathcal{N}_{\text{aug}}(F_S^B(\hat{\mathbf{n}}), F_T(\hat{\mathbf{n}}), F_E(\hat{\mathbf{n}})) \label{eq:B_L_aug_reconstruction}
\end{equation}

## Summary

Our methodology extends the signal-preserving ML framework of McCarthy et al. to the case of inter-scale learning for B-mode polarization foreground removal. By exploiting correlations between small-scale and large-scale foreground structures, and augmenting with temperature and E-mode information, we aim to improve upon traditional ILC methods while maintaining strict signal preservation guarantees. The approach ensures that the primary CMB B-mode signal remains unbiased, even in the presence of imperfect simulations or modeling assumptions, by constraining the network to operate only on signal-free inputs and learn signal-free targets.
