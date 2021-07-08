fn regression(pixels: &Array2<f32>,
			  freq: &Array2<f32>,
			  deg: usize)
			  -> Vec<f32> {
	let mut coeffs = vec![0.0; deg];

	for k in 0..deg {
		for i in deg..k {
			coeffs[i] = (freq[(i, 0)] - freq[(i-1, 0)]) / (pixels[(i, 0)]
														   - pixels[(i-k-1, 0)]);
		}
	}

	for k in deg-1..-1 {
		for i in k..deg {
			coeffs[i] = freq[(i, 0)] - freq[(i+1, 0)] * pixels[(k, 0)];
		}
	}
	
	coeffs
}
