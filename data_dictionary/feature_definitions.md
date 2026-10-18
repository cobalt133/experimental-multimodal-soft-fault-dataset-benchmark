# Structured feature data dictionary

This dictionary defines the exported values, their inputs, normalization, and
units. The public `data/structured_descriptors_34_all_captures.csv` contains two ID
columns and 34 numeric features. The canonical **32-feature** input
retains the same values and column order after excluding `amp_mean_x` and
`amp_mean_y`; it does not re-extract features from a different signal stage.
The column inventory is provided in `feature_dictionary.csv`.

## Signal inputs and shared notation

For a capture with four real channels, define

$$
I_1=\mathrm{ch}_1,\quad Q_1=-\mathrm{ch}_2,\quad
I_2=\mathrm{ch}_3,\quad Q_2=-\mathrm{ch}_4,\qquad
X=I_1+jQ_1,\quad Y=I_2+jQ_2.
$$

All means below are arithmetic means. Set $\epsilon=10^{-12}$ and

$$
p_x=\operatorname{mean}|X|^2+\epsilon,\quad
p_y=\operatorname{mean}|Y|^2+\epsilon,\quad
X_n=X/\sqrt{p_x},\quad Y_n=Y/\sqrt{p_y},\quad
A_x=|X_n|^2,\quad A_y=|Y_n|^2,\quad Z=(X_n+Y_n)/2.
$$

Branch powers are measured over the **full capture**, before cropping for the
spectrum. All descriptors use pre-compensation receiver captures at 256 GSa/s.
Branch power normalization is per capture and uses no labels or dataset-wide
statistics. Model-side scaling, if used, is a separate step.

`amp_*` describes **intensity** (squared magnitude), despite the historical
"amp" name. $Z$ is a coherent arithmetic mixture of the normalized complex
branches. Its spectrum includes cross terms; it is not the average of the two
polarization power spectra. The six scalar `psd_*` columns all use $Z$.

## Six spectral features

### Common spectral calculation

1. Take the first $M=8192$ samples of $Z$. Short inputs are right-padded with
   complex zeros to 8192 samples (the exporter uses a `complex64` padding array).
2. Multiply by the symmetric Hann window
   $w[n]=0.5-0.5\cos(2\pi n/(M-1))$, $n=0,\ldots,M-1$ (`np.hanning`).
3. Compute $F=\operatorname{fftshift}(\operatorname{FFT}(wZ))$,
   $P_k=|F_k|^2+\epsilon$, and $D_k=10\log_{10}P_k$.
4. Use the coordinate $f_k=-0.5+k/M$, $k=0,\ldots,M-1$, and, for the
   centroid only, $\pi_k=P_k/\sum_jP_j$.

There is no signal-mean removal, passband selection, spectral smoothing, or
Welch averaging. FFT power is not divided by FFT length, window energy, or
sampling rate. The historical `psd` names therefore refer to FFT-power
descriptors rather than a calibrated power spectral density in power/Hz.
The epsilon floor remains in every bin.

The frequency coordinate is **cycles per input sample**, equivalently frequency
divided by the input sampling rate $F_s$. It is normalized by $F_s$, not by the
Nyquist frequency and not by the baud rate. With $F_s=256$ GSa/s, the grid spans
$-128$ GHz through $128\,\mathrm{GHz}-31.25$ MHz; the spacing is 31.25 MHz.
The 8192-sample segment lasts 32 ns at this rate. The frequency conversion uses the configured 256-GSa/s acquisition rate.

| Column | Physical/statistical meaning | Input and normalization | Unit / coordinate | Exact calculation and range |
|---|---|---|---|---|
| `psd_centroid_norm` | Signed center of FFT-power mass; describes spectral displacement in the acquisition coordinate. | $Z$; weights $\pi_k$ sum to one across all bins. | Cycles/sample ($f/F_s$). | $\sum_k f_k\pi_k$. Within the grid bounds $[-0.5,\,0.5-1/M]$. Multiply by $F_s$ for Hz. |
| `psd_bw3_norm` | Total frequency span of all bins within 3 dB of the global peak. | $Z$; $D_k$ compared with the global maximum of $D$. | Bandwidth divided by $F_s$. | For $S_3=\{k:D_k\geq\max_jD_j-3\}$, return $\max_{k\in S_3}f_k-\min_{k\in S_3}f_k$. Empty mask returns zero; one bin also gives zero. Range $[0,\,1-1/M]$. |
| `psd_bw10_norm` | Total frequency span of all bins within 10 dB of the global peak. | Same calculation as above with a 10 dB threshold. | Bandwidth divided by $F_s$. | For $S_{10}=\{k:D_k\geq\max_jD_j-10\}$, return $\max_{k\in S_{10}}f_k-\min_{k\in S_{10}}f_k$, or zero for an empty mask. Range $[0,\,1-1/M]$. |
| `psd_lr_diff_db` | Negative-frequency versus nonnegative-frequency power imbalance. Positive means the negative half has more power. | $Z$; sums of $P_k$ over the two FFT halves. | dB power ratio. | $10\log_{10}[(\sum_{k=0}^{M/2-1}P_k+\epsilon)/(\sum_{k=M/2}^{M-1}P_k+\epsilon)]$. DC belongs to the right half; the negative Nyquist bin belongs to the left. No fixed signal-independent finite bounds. |
| `psd_flatness_db` | Geometric-to-arithmetic mean FFT-power ratio; values nearer zero indicate a flatter full-grid power distribution. | $Z$; all 8192 $P_k$, including out-of-band bins and the epsilon floor. | dB ratio. | $10\log_{10}\{\exp[M^{-1}\sum_k\ln P_k]/[M^{-1}\sum_kP_k]\}$. Natural logarithm inside the geometric mean. Nonpositive mathematically, up to rounding; no fixed finite lower bound. |
| `psd_ripple_std_db` | Full-grid spread of log FFT power. This is not a passband-only ripple or peak-to-peak ripple measure. | $Z$; subtract the mean of all 8192 $D_k$. | dB. | `np.std(D - np.mean(D))`, i.e. $\sqrt{M^{-1}\sum_k(D_k-\bar D)^2}$ with `ddof=0`. Nonnegative; no fixed finite upper bound. No fitted-envelope subtraction. |

Both bandwidths use **global extreme selected bins**, including separated peaks
and gaps. The implementation does not restrict the mask to a connected lobe,
interpolate threshold crossings, count selected bins, add one bin width, or
wrap across the FFT boundary. Thus these columns should not be interpreted as
an interpolated optical-filter passband width. Flatness and ripple use the
entire FFT grid, not the 3/10-dB mask or an independently defined occupied band.
Their values can depend on the spectral floor and out-of-band content.

For an all-zero input, the epsilon floor makes $P$ constant: both bandwidths
equal $1-1/M$, flatness and ripple are zero up to rounding, and the centroid is
$-1/(2M)$ because the discrete grid includes $-0.5$ but excludes $+0.5$.
These are the exported edge-case values, not an inferred occupied bandwidth.

### Relation to the public PSD tensor

The six scalar features above are computed from the **8192-bin mixed spectrum**,
before pooling. Independently, the exporter constructs `psd_seq` in the order
`[X_n, Y_n, Z]`. For each branch, it partitions $D$ into 512 consecutive groups
of 16 bins and stores each group's **arithmetic mean in dB** as `float32`.
It does not average linear power and then take a logarithm. Recomputing the six
scalars from the pooled tensor will in general give different values.

## Other structured columns

The following definitions use full-capture sequences. In the table,
$\mu(v)=\operatorname{mean}(v)$ and
$s(v)=\sqrt{\operatorname{mean}[(v-\mu(v))^2]}$ (`ddof=0`). Define

$$
\operatorname{skew}(v)=\operatorname{mean}[(v-\mu(v))^3]/s(v)^3,\qquad
\operatorname{kurt}(v)=\operatorname{mean}[(v-\mu(v))^4]/s(v)^4.
$$

The exporter returns zero for either standardized moment when $s(v)<10^{-12}$.
Kurtosis is the fourth standardized moment, **without subtracting three** and
without a finite-sample bias correction. `corr(a,b)` is Pearson correlation
(`np.corrcoef`); it returns zero when either input's population standard
deviation is below $10^{-12}$. All correlations are dimensionless in $[-1,1]$
up to numerical precision, with the stated zero fallback.

| Columns (one scalar per listed name) | Physical/statistical meaning | Input / normalization | Formula / rule | Units |
|---|---|---|---|---|
| `sample_id`, `record_id` | Capture identifiers for joining feature, condition, and tensor tables. | Public metadata. | Strings; excluded from model feature inputs. | None. |
| `power_x_db`, `power_y_db` | Mean electrical squared magnitude of each polarization before branch normalization. | Raw $X$, $Y$; epsilon added to each full-capture mean power. | $10\log_{10}p_x$, $10\log_{10}p_y$. | Relative dB in acquisition squared-amplitude units, not calibrated dBm. |
| `power_total_db` | Sum of the two pre-normalization branch powers. | $p_x+p_y$; contains both branch epsilon terms. | $10\log_{10}(p_x+p_y)$. | Same relative dB convention. |
| `crest_i1`, `crest_q1`, `crest_i2`, `crest_q2` | Peak-to-RMS ratio of each real channel. | Respectively $I_1,Q_1,I_2,Q_2$; raw channel scale, no mean removal. | $\max\lvert v\rvert/[\sqrt{\operatorname{mean}(v^2)}+\epsilon]$. | Dimensionless ratio. |
| `skew_i1`, `skew_q1`, `skew_i2`, `skew_q2` | Asymmetry of the channel-value distribution. | Respectively $I_1,Q_1,I_2,Q_2$; centered and divided by its own standard deviation. | $\operatorname{skew}(v)$ as defined above. | Dimensionless. |
| `kurt_i1`, `kurt_q1`, `kurt_i2`, `kurt_q2` | Fourth standardized moment, describing tails and concentration. | Respectively $I_1,Q_1,I_2,Q_2$; same centering/scaling. | $\operatorname{kurt}(v)$ as defined above. | Dimensionless; not excess kurtosis. |
| `corr_i1_q1`, `corr_i2_q2` | Within-polarization real-channel linear dependence. | Raw $(I_1,Q_1)$ and $(I_2,Q_2)$ respectively. | $\operatorname{corr}(I_1,Q_1)$, $\operatorname{corr}(I_2,Q_2)$. | Dimensionless. |
| `corr_i1_i2`, `corr_q1_q2` | Cross-polarization linear dependence of corresponding real channels. | Raw $(I_1,I_2)$ and $(Q_1,Q_2)$ respectively. | $\operatorname{corr}(I_1,I_2)$, $\operatorname{corr}(Q_1,Q_2)$. | Dimensionless. |
| `corr_amp_xy` | Cross-polarization intensity linear dependence. | Normalized $A_x$, $A_y$. | $\operatorname{corr}(A_x,A_y)$. | Dimensionless. |
| `amp_mean_x`, `amp_mean_y` | Legacy normalized-intensity means; excluded from the canonical 32-feature input. | Respectively $A_x,A_y$. | $\operatorname{mean}A_x=(p_x-\epsilon)/p_x$ and analogously for Y, up to floating-point rounding. Nearly one; epsilon-sensitive redundant power proxies. | Dimensionless. |
| `amp_std_x`, `amp_std_y` | Dispersion of normalized intensity. | Respectively $A_x,A_y$; population standard deviation. | $s(A_x)$, $s(A_y)$. | Dimensionless normalized intensity. |
| `amp_skew_x`, `amp_skew_y` | Asymmetry of normalized-intensity distributions. | Respectively $A_x,A_y$; standardized moments. | $\operatorname{skew}(A_x)$, $\operatorname{skew}(A_y)$. | Dimensionless. |
| `amp_kurt_x`, `amp_kurt_y` | Fourth standardized moment of normalized intensity. | Respectively $A_x,A_y$; standardized moments. | $\operatorname{kurt}(A_x)$, $\operatorname{kurt}(A_y)$. | Dimensionless; not excess kurtosis. |

These are statistical descriptors of a receiver capture. They are not direct
measurements of a component's transfer function, optical power in dBm, OSNR, or
demodulated quality metrics.
