use ndarray::Array2;
use ndarray::Array1;
use ndarray::array;

pub fn matrix_to_freq(matrix: &Array2<u8>) -> Array1<i32> {
    let dimensions = matrix.dim();
    let mut freqs: Array1<i32> = Array1::zeros(256);
    for i in matrix.iter() {
        freqs[*i as usize] += 1
    }

    freqs
}

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
