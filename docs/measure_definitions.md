# Measurement definitions

## Spectral participation ratio

The current `P_C` uses the row-normalized incoming-weight matrix
`W_in_normalized` with shape `(hidden units, input features)`. Its spectral
values are the singular values `s_i` of that matrix, and its effective
dimension is
`(sum(s_i^2))^2 / sum(s_i^4)`.

`R_C` uses the centered post-activation matrix with shape
`(reference examples, hidden units)`. Its spectral values are the eigenvalues
of the activation covariance, proportional to `s_i^2` for the centered
activation matrix. It therefore uses the same energy participation ratio.
The previous code instead used `(sum(s_i))^2 / sum(s_i^2)` for `P_C`, while
`R_C` used `(sum(s_i^2))^2 / sum(s_i^4)`. Repository comments and docs gave
no rationale for the mismatch; it was treated as an inconsistency and fixed.
New outputs identify this convention with metric-definition version `2`;
historical outputs are not migrated or relabelled.

## Centered `F_C`

For each fixed ordered reference sample, compute its post-nonlinearity mean
`mu` and centered PCA directions `D`. The intervention is
`mu + (X - mu) - (X - mu) D.T D`, preserving the reference mean while removing
the selected centered component. Random-control directions use the same
intervention. New output records identify it as
`centered_pca_projection`, save the mean and covariance spectrum, and include
the sample identity and size.

Temporal measurements use fixed ordered train and validation samples.
The final summary uses the fixed ordered `D_ref` test sample. The combined
`P_L` display remains the average of separately saved incoming and outgoing
cosine-similarity matrices.
