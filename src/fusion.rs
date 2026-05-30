use rayon::prelude::*;

pub struct FusedLinearReLU {
    pub in_features: usize,
    pub out_features: usize,
}

impl FusedLinearReLU {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            in_features,
            out_features,
        }
    }

    /// Single-threaded register-localized execution block (Ideal for Lightweight Experts)
    pub fn forward(
        &self,
        input: &[f32],
        weights: &[f32],
        bias: &[f32],
        output: &mut [f32],
    ) {
        assert_eq!(input.len(), self.in_features);
        assert_eq!(weights.len(), self.in_features * self.out_features);
        assert_eq!(bias.len(), self.out_features);
        assert_eq!(output.len(), self.out_features);

        for row in 0..self.out_features {
            let mut acc = bias[row];
            let weight_row_offset = row * self.in_features;

            for col in 0..self.in_features {
                acc += input[col] * weights[weight_row_offset + col];
            }
            output[row] = if acc > 0.0 { acc } else { 0.0 };
        }
    }

    /// Multi-threaded parallel execution block via Rayon (Ideal for Dense Experts)
    /// Splits rows across hardware processor cores to accelerate massive matrices.
    pub fn forward_parallel(
        &self,
        input: &[f32],
        weights: &[f32],
        bias: &[f32],
        output: &mut [f32],
    ) {
        assert_eq!(input.len(), self.in_features);
        assert_eq!(weights.len(), self.in_features * self.out_features);
        assert_eq!(bias.len(), self.out_features);
        assert_eq!(output.len(), self.out_features);

        // Process rows in parallel across available edge processor cores
        output.par_iter_mut().enumerate().for_each(|(row, out_val)| {
            let mut acc = bias[row];
            let weight_row_offset = row * self.in_features;

            for col in 0..self.in_features {
                acc += input[col] * weights[weight_row_offset + col];
            }
            *out_val = if acc > 0.0 { acc } else { 0.0 };
        });
    }
}