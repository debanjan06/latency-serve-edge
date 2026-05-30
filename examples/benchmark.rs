use std::fs::File;
use std::io::Write;
use std::time::Instant;
use latency_serve_edge::memory::ZeroCopyTensorReader;
use latency_serve_edge::fusion::FusedLinearReLU;
use latency_serve_edge::routing::{ScenarioAwareRouter, InferenceRoute};

fn create_mock_weights_file(path: &str, num_elements: usize) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    // Populate dummy weight values for structured processing simulation
    let weights = vec![0.15f32; num_elements];
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            weights.as_ptr() as *const u8,
            weights.len() * 4,
        )
    };
    file.write_all(bytes)?;
    Ok(())
}

fn main() -> std::io::Result<()> {
    println!("=====================================================");
    println!("    LATENCYSERVE-EDGE INFRASTRUCTURE BENCHMARK       ");
    println!("=====================================================");

    let mock_weights_path = "mock_edge_weights.bin";
    
    // Scale matrix dimensions up to give multi-threading room to breathe
    let in_features = 2048;
    let out_features = 1024;
    let total_weights = in_features * out_features;

    // Generate parameters block directly on disk
    create_mock_weights_file(mock_weights_path, total_weights)?;

    // 1. Initialize Zero-Copy Virtual Memory Mapping
    let reader = ZeroCopyTensorReader::new(mock_weights_path)?;
    let weight_slice = reader.as_slice();
    println!("-> Memory-Mapped Tensor Weights File Successfully.");
    println!("   Total Parameter Array Size: {} elements", weight_slice.len());

    // 2. Setup Operators and Shared Pre-allocated Accumulator Buffers
    let layer = FusedLinearReLU::new(in_features, out_features);
    let mock_input = vec![1.0f32; in_features];
    let mock_bias = vec![0.05f32; out_features];
    let mut output_buffer = vec![0.0f32; out_features];

    // 3. Initialize Scenario-Aware Routing Rules (1.5ms operational target)
    let router = ScenarioAwareRouter::new(1.5);
    
    // ---------------------------------------------------------------
    // [Scenario A] Low Structural Complexity Input Frame Processing
    // ---------------------------------------------------------------
    let sample_variance_low = 0.12f32; 
    println!("\n[Scenario A] Processing incoming frame with low variance ({})", sample_variance_low);
    
    let start_route_a = Instant::now();
    match router.route(sample_variance_low) {
        InferenceRoute::LightweightExpert => {
            println!("   -> Router assigned task to: Lightweight Expert");
            // Run register-localized execution on a single core for optimal power efficiency
            layer.forward(&mock_input, weight_slice, &mock_bias, &mut output_buffer);
        }
        InferenceRoute::DenseExpert => {}
    }
    let duration_a = start_route_a.elapsed();
    println!("   -> Fused Single-Thread Execution Completed in: {:?}", duration_a);

    // ---------------------------------------------------------------
    // [Scenario B] High Structural Complexity Input Frame Processing
    // ---------------------------------------------------------------
    let sample_variance_high = 0.45f32;
    println!("\n[Scenario B] Processing incoming frame with high variance ({})", sample_variance_high);
    
    let start_route_b = Instant::now();
    match router.route(sample_variance_high) {
        InferenceRoute::LightweightExpert => {}
        InferenceRoute::DenseExpert => {
            println!("   -> Router assigned task to: Dense Expert");
            // Invoke the Rayon work-stealing parallel engine to spread rows over all logical threads
            layer.forward_parallel(&mock_input, weight_slice, &mock_bias, &mut output_buffer);
        }
    }
    let duration_b = start_route_b.elapsed();
    println!("   -> Fused Multi-Thread Parallel Execution Completed in: {:?}", duration_b);

    // Wipe the temporary mock binary parameters from workspace disk
    std::fs::remove_file(mock_weights_path)?;
    println!("\n=====================================================");
    Ok(())
}