pub enum InferenceRoute {
    LightweightExpert,
    DenseExpert,
}

pub struct ScenarioAwareRouter {
    pub latency_threshold_ms: f64,
}

impl ScenarioAwareRouter {
    pub fn new(latency_threshold_ms: f64) -> Self {
        Self { latency_threshold_ms }
    }

    /// inspects incoming spatial characteristics to dynamically assign the inference route
    pub fn route(&self, input_variance: f32) -> InferenceRoute {
        // If spatial variance is low, route to the fast, low-power lightweight model.
        // If data complexity is high, invoke the high-capacity expert.
        if input_variance < 0.25 {
            InferenceRoute::LightweightExpert
        } else {
            InferenceRoute::DenseExpert
        }
    }
}