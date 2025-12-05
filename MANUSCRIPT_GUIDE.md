# Journal Manuscript Preparation Guide

## Overview

I've created a journal manuscript (`journal_manuscript.tex`) based on your previous work and new results. This document provides guidance on finalizing the manuscript for submission.

## What's Included

The manuscript includes:

1. **Complete structure** following standard journal paper format:
   - Abstract
   - Introduction (with scientific motivation)
   - Method (inter-scale framework, network architecture, training)
   - Data (DustFilaments simulations)
   - Results (comprehensive evaluation)
   - Discussion (findings, limitations, future work)
   - Conclusions

2. **Figure placeholders** with detailed captions based on `figure_descriptions.md`:
   - Figure 1: CMB Reconstruction Analysis (1×6 panels)
   - Figure 2: Foreground Reconstruction Analysis (2×3 panels)
   - Figure 3: Input Channel Visualization (1×3 panels)
   - Figure 4: MSE Progression Comparison
   - Figure 5: Spatial Correlation Comparison

3. **Quantitative results** integrated from your evaluation:
   - Spatial correlations: 0.45 ± 0.29
   - Cross-power spectrum correlations: 0.49
   - MSE: 3.5 × 10⁻⁴
   - CMB reconstruction improvements: 60% MSE reduction
   - Signal preservation validation

## Next Steps

### 1. Update Figure Paths

The manuscript currently has placeholder paths for figures. Update them to point to your actual figure files:

```latex
% Current placeholder:
\includegraphics[width=1.0\textwidth]{Figures/cmb_reconstructions/cmb_reconstruction_sample_0.png}

% Update to your actual paths, e.g.:
\includegraphics[width=1.0\textwidth]{/scratch/gpfs/JDUNKLEY/hshao/old_data/.../cmb_reconstruction_sample_0.png}
```

### 2. Add Specific Quantitative Results

Review your evaluation results and add specific numbers where marked with placeholders:

- **MSE comparison analysis**: Add the percentage of patches where `mse_target_vs_pred < mse_ilc_vs_cmb`
- **Mean improvement ratio**: Add the ratio `mse_ilc_vs_cmb / mse_target_vs_pred`
- **Percentile distributions**: Add specific percentile values if available
- **Per-patch statistics**: Add any additional statistics from your analysis

### 3. Complete Bibliography

Ensure all citations in `references.bib` are complete and properly formatted. The manuscript references:
- Previous work (PartIII_Final_Report)
- DustFilaments paper
- Standard CMB/foreground references
- ML/UNet references

### 4. Add Funding and Acknowledgments

Update the acknowledgments section with:
- Actual funding sources
- Computing resources used
- Collaborators and contributors

### 5. Review and Refine

- **Abstract**: Ensure it captures all key findings (currently ~250 words, adjust for journal requirements)
- **Introduction**: Verify it properly motivates the work and places it in context
- **Method**: Check that all technical details are accurate
- **Results**: Ensure all quantitative claims are supported by your data
- **Discussion**: Add any additional insights or limitations you've discovered

### 6. Journal-Specific Formatting

Before submission, check the target journal's requirements:
- **Length limits**: Some journals have strict page limits
- **Figure requirements**: Resolution, format (PDF, PNG, etc.)
- **Reference style**: May need to adjust citation format
- **Supplementary material**: Decide what goes in main text vs. supplement

## Key Strengths of This Manuscript

1. **Clear signal preservation guarantee**: The theoretical foundation is well-established
2. **Comprehensive evaluation**: Multiple metrics (spatial, harmonic, MSE, null tests)
3. **Statistical validation**: Null hypothesis testing provides rigor
4. **Practical utility**: Direct comparison with observational contamination levels
5. **Multi-channel enhancement**: Shows benefit of T,E,B inputs

## Areas That May Need Expansion

1. **Comparison with ILC**: Could add more detailed comparison with multi-frequency ILC
2. **Computational efficiency**: Could discuss training/inference time
3. **Robustness tests**: Could add more on generalization to other models
4. **Error analysis**: Could add more detailed breakdown of failure modes
5. **Physical interpretation**: Could expand discussion of what the network learns

## Figure Organization

Your figures should be organized as:

```
Figures/
├── cmb_reconstructions/
│   └── cmb_reconstruction_sample_*.png
├── fg_reconstructions/
│   └── fg_reconstruction_sample_*.png
├── input_channels/
│   └── input_channels_sample_*.png
└── model_comparison/
    ├── mse_progression.png
    └── correlation_comparison.png
```

## Compilation

To compile the manuscript:

```bash
cd /scratch/gpfs/hshao/ILC_ML/PartIII___Final_Report__long_
pdflatex journal_manuscript.tex
bibtex journal_manuscript
pdflatex journal_manuscript.tex
pdflatex journal_manuscript.tex
```

## Suggested Journals

Based on the content, suitable journals include:
- **Physical Review D** (cosmology focus)
- **Monthly Notices of the Royal Astronomical Society** (astronomy focus)
- **The Astrophysical Journal** (astrophysics focus)
- **Machine Learning: Science and Technology** (ML applications)

## Questions to Consider

1. Should you include the ILC-ML hybrid results from your previous work as a comparison?
2. Do you want to include the diffusion model results, or focus only on UNet?
3. Should you add a section on computational requirements?
4. Do you want to include code/data availability statements?

## Contact

If you need help with:
- Adding specific quantitative results
- Refining particular sections
- Formatting for a specific journal
- Creating additional figures or tables

Let me know and I can help update the manuscript accordingly.

