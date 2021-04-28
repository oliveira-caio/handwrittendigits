use crate::loadmnist::MnistImage;
use ndarray::Array2;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use rand::seq::SliceRandom;
use std::f32::consts::E;
mod loadmnist;

#[derive(Debug)]
struct Network {
    num_layers: usize,
    sizes: Vec<usize>,
    biases: Vec<Array2<f32>>,
    weights: Vec<Array2<f32>>,
}

impl Network {
    fn new(sizes: Vec<usize>) -> Network {
        let num_layers = sizes.len();
        let sizes = sizes;
        let mut biases = Vec::new();
        let mut weights = Vec::new();

        for i in 1..sizes.len() {
            biases.push(vec_random(sizes[i]));
            weights.push(matrix_random(sizes[i], sizes[i - 1]));
        }

        Network {
            num_layers,
            sizes,
            biases,
            weights,
        }
    }

    fn feedforward(&self, vector: &Array2<f32>) -> Array2<f32> {
        let mut result = vector.clone();

        for (b, w) in self.biases.iter().zip(self.weights.iter()) {
            result = vec_sigmoid(&(w.dot(&result) + b));
        }

        result
    }

    fn evaluate(&self, test_data: &Vec<MnistImage>) -> usize {
        let mut test_results = Vec::new();
        let mut sum = 0;

        for t in test_data {
            test_results.push((self.feedforward(&t.image), t.classification));
        }

        for (u, v) in test_results {
            if argmax(&u) == v {
                sum += 1;
            }
        }

        sum
    }

    fn cost_derivative(
        &self,
        output_activations: &Array2<f32>,
        output: &Array2<f32>,
    ) -> Array2<f32> {
        output_activations - output
    }

    fn sgd(
        &mut self,
        training_data: &Vec<MnistImage>,
        epochs: usize,
        mini_batch_size: usize,
        eta: f32,
        test_data: Option<&Vec<MnistImage>>,
    ) {
        let mut i: usize = 0;
        let mut mini_batches: Vec<Vec<(Array2<f32>, Array2<f32>)>> = Vec::new();
        let mut converted_training = Vec::new();

        for t in training_data {
            converted_training.push((t.image.clone(), u8_to_array2(t.classification)));
        }

        for j in 0..epochs {
            let shuffled = my_shuffle(&converted_training);

            while i < shuffled.len() {
                mini_batches.push(shuffled[i..i + mini_batch_size].to_vec());
                i += mini_batch_size;
            }

            for mini_batch in mini_batches.iter() {
                self.update_mini_batch(mini_batch, eta);
            }

            match test_data {
                Some(x) => println!("Epoch {}: {}/{}", j, self.evaluate(x), x.len()),
                None => println!("Epoch {} complete", j),
            }
        }
    }

    fn update_mini_batch(&mut self, mini_batch: &Vec<(Array2<f32>, Array2<f32>)>, eta: f32) {
        let mut nabla_b: Vec<Array2<f32>> = Vec::new();
        let mut nabla_w: Vec<Array2<f32>> = Vec::new();
        let nbatch = mini_batch.len() as f32;

        for b in self.biases.iter() {
            nabla_b.push(Array2::zeros(b.dim()));
        }

        for w in self.weights.iter() {
            nabla_w.push(Array2::zeros(w.dim()));
        }

        for (x, y) in mini_batch.iter() {
            let (delta_nabla_b, delta_nabla_w) = self.backprop(x, y);

            for (nb, dnb) in nabla_b.iter_mut().zip(delta_nabla_b.iter()) {
                *nb += dnb;
            }

            for (nw, dnw) in nabla_w.iter_mut().zip(delta_nabla_w.iter()) {
                *nw += dnw;
            }
        }

        for (b, nb) in self.biases.iter_mut().zip(nabla_b.iter_mut()) {
            *b -= &nb.mapv(|x| x * eta / nbatch)
        }

        for (w, nw) in self.weights.iter_mut().zip(nabla_w.iter_mut()) {
            *w -= &nw.mapv(|x| x * eta / nbatch)
        }
    }

    fn backprop(
        &self,
        activation: &Array2<f32>,
        output: &Array2<f32>,
    ) -> (Vec<Array2<f32>>, Vec<Array2<f32>>) {
        let mut nabla_b: Vec<Array2<f32>> = Vec::new();
        let mut nabla_w: Vec<Array2<f32>> = Vec::new();
        let mut activations = vec![activation.clone()];
        let mut zs = Vec::new();
        let mut z = activation.clone();

        for (b, w) in self.biases.iter().zip(self.weights.iter()) {
            z = w.dot(&z) + b;
            zs.push(z.clone());
            activations.push(vec_sigmoid(&z));
            z = vec_sigmoid(&z);
        }

        let asize = activations.len();
        let mut delta: Array2<f32> = self.cost_derivative(&activations[asize - 1], &output)
            * sigmoid_prime(&zs[zs.len() - 1]);
        nabla_b.push(delta.clone());
        nabla_w.push(delta.dot(&(activations[asize - 2]).t()));

        for l in 2..self.num_layers {
            z = zs[zs.len() - l].clone();
            let sp = sigmoid_prime(&z);
            delta = (self.weights[self.weights.len() - l + 1].t()).dot(&delta) * sp;
            nabla_b.push(delta.clone());
            nabla_w.push(delta.dot(&(activations[asize - l - 1]).t()));
        }

        nabla_b.reverse();
        nabla_w.reverse();

        (nabla_b, nabla_w)
    }
}

fn argmax(vector: &Array2<f32>) -> u8 {
    let mut x = vector[(0, 0)];
    let mut index = 0;

    for i in 0..vector.len() {
        if vector[(i, 0)] > x {
            index = i;
            x = vector[(i, 0)];
        }
    }

    index as u8
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + f32::powf(E, -x))
}

fn vec_sigmoid(vector: &Array2<f32>) -> Array2<f32> {
    vector.mapv(|x| sigmoid(x))
}

fn sigmoid_prime(vector: &Array2<f32>) -> Array2<f32> {
    vector.mapv(|x| sigmoid(x) * (1.0 - sigmoid(x)))
}

fn vec_random(size: usize) -> Array2<f32> {
    Array2::random((size, 1), StandardNormal)
}

fn matrix_random(rows: usize, columns: usize) -> Array2<f32> {
    Array2::random((rows, columns), StandardNormal)
}

fn my_shuffle<T: Clone>(vector: &[T]) -> Vec<T> {
    let mut shuffled = Vec::new();
    let mut aux: Vec<usize> = (0..vector.len()).collect();

    aux.shuffle(&mut rand::thread_rng());

    for i in 0..aux.len() {
        shuffled.push(vector[aux[i]].clone());
    }

    shuffled
}

fn u8_to_array2(number: u8) -> Array2<f32> {
    let mut classification = Array2::zeros((10, 1));
    classification[(number as usize, 0)] = 1.0;
    classification
}

fn main() {
    let net_sizes: Vec<usize> = vec![784, 30, 10];
    let mut net = Network::new(net_sizes);
    let training_data = loadmnist::load_data("train").unwrap();
    let test_data = loadmnist::load_data("t10k").unwrap();

    net.sgd(&training_data, 30, 10, 3.0, Some(&test_data));
}
