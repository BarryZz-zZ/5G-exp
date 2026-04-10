# Further Discussion: Why no clear correlation peak with large frequency offset, and how to fix it

## Problem
For `recv_signal_freq_offset`, direct time-domain correlation with PSS templates may show weak or unclear peaks.

Reason:
- If received signal is `r[n] = s[n-n0] * exp(j*2*pi*Df*n/Fs) + w[n]`, the unknown carrier frequency offset `Df` causes fast phase rotation.
- Correlation assumes phase-consistent matching between `r[n]` and template `s[n]`.
- With large `Df`, phase rotates significantly over the 256-sample template, so coherent accumulation is destroyed and peak collapses.

## Practical solution used in the MATLAB code
The script `run_5g_experiment.m` implements a 3-step method:

1. Rough TOA by raw `xcorr`  
Use each reference template once to get a coarse peak location.

2. Estimate frequency offset (CFO)  
At the rough TOA segment, compute:
`p[n] = r_seg[n] * conj(s[n])`  
Then estimate average phase increment:
`dphi = angle(sum(conj(p[n])*p[n+1]))`  
`Df_est = dphi * Fs / (2*pi)`

3. Compensate CFO and correlate again  
`r_comp[n] = r[n] * exp(-j*2*pi*Df_est*n/Fs)`  
Then run `xcorr` again. Correlation peak becomes much clearer and PSS/TOA detection is more stable.

## Why this works
- After compensation, template and received segment are phase-aligned.
- Correlation changes from non-coherent-like behavior back to coherent accumulation.
- Peak-to-background ratio increases, so TOA and PSS decisions are more reliable.

## Additional methods (if offset is even larger)
- Coarse frequency grid search + correlation (`Df` sweep, choose max peak)
- Two-stage synchronization: coarse CFO -> fine CFO (PLL / pilot-aided)
- Non-coherent accumulation as a fallback when phase cannot be perfectly aligned
- Joint TOA-CFO estimation (2D search), better performance but higher complexity

## Conclusion for the discussion question
When `recv_signal_freq_offset` gives no obvious correlation peak, the key is to **estimate and compensate frequency offset before final coherent correlation**.  
This is the direct and effective fix for the issue raised in the PPT.
