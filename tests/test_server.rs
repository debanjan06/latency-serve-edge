use std::fs::File;
use std::io::Write;
use latency_serve_edge::memory::ZeroCopyTensorReader;
use latency_serve_edge::fusion::FusedLinearReLU;
use latency_serve_edge::routing::{ScenarioAwareRouter, InferenceRoute};

#[test]
fn test_zero_copy_alignment_and_slicing() {
    let test_file_path = "test_tensor_bounds.bin";
    
    // 1. Create a dummy file containing 4 specific float elements (16 bytes)
    let expected_data: Vec<f32> = vec![1.25, -2.5, 3.75, 0.0];
    let mut file = File::create(test_file_path).unwrap();
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            expected_data.as_ptr() as *const u8,
            expected_data.len() * 4,
        )
    };
    file.write_all(bytes).unwrap();

    // 2. Map file into memory space
    let reader = ZeroCopyTensorReader::new(test_file_path).unwrap();
    assert_eq!(reader.total_elements, 4);
    
    let mapped_slice = reader.as_slice();
    assert_eq!(mapped_slice[0], 1.25);
    assert_eq!(mapped_slice[1], -2.5);
    assert_eq!(mapped_slice[2], 3.75);
    assert_eq!(mapped_slice[3], 0.0);

    // Clean up temporary tracking files
    std::fs::remove_file(test_file_path).unwrap();
}

#[test]
fn test_register_fused_linear_relu_math() {
    // Set up a simple 2x2 layer configuration
    let in_features = 2;
    let out_features = 2;
    let layer = FusedLinearReLU::new(in_features, out_features);

    let input = vec![1.0, 2.0];
    let weights = vec![
        0.5,  2.0,  // Row 0 weights
       -1.0,  0.5,  // Row 1 weights
    ];
    let bias = vec![0.1, -5.0];
    let mut output = vec![0.0; out_features];

    // Execution Calculations:
    // Row 0 Accumulator = 0.1 + (1.0 * 0.5) + (2.0 * 2.0) = 4.6 (Passes ReLU)
    // Row 1 Accumulator = -5.0 + (1.0 * -1.0) + (2.0 * 0.5) = -5.0 (Clamped to 0.0 by ReLU)
    layer.forward(&input, &weights, &bias, &mut output);

    assert_eq!(output[0], 4.6);
    assert_eq!(output[1], 0.0);
}

#[test]
fn test_scenario_aware_routing_boundaries() {
    let router = ScenarioAwareRouter::new(1.5);

    // Variance under 0.25 threshold maps to the lightweight path
    match router.route(0.15) {
        InferenceRoute::LightweightExpert => {}
        InferenceRoute::DenseExpert => panic!("Expected LightweightExpert route configuration deviation"),
    }

    // Variance equal to or above 0.25 threshold maps to the dense path
    match router.route(0.35) {
        InferenceRoute::DenseExpert => {}
        InferenceRoute::LightweightExpert => panic!("Expected DenseExpert route configuration deviation"),
    }
}