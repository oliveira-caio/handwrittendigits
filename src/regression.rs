use ndarray::Array2;
use ndarray::Array1;
use ndarray::array;
use ndarray_linalg::solve::Inverse;

pub fn matrix_to_freq(matrix: &Array2<u8>) -> Array1<i32> {
    let dimensions = matrix.dim();
    let mut freqs: Array1<i32> = Array1::zeros(256);
    for i in matrix.iter() {
        freqs[*i as usize] += 1
    }

    freqs
}

pub fn inverte(matrix: &Array2<f32>) -> Array2<f32> {
    matrix.inv().unwrap()
}
