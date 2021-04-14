use rand::seq::SliceRandom;
use std::f32::consts::E;
use ndarray::{array, Array, Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use std::io::{Cursor, Read};
use flate2;
use flate2::read::GzDecoder;
use std::fs::File;
use byteorder::BigEndian;
use byteorder::ReadBytesExt;
use std::time::Instant;

#[derive(Debug)]
struct Network {
	num_layers: u16,
	sizes: Vec<u16>,
	biases: Vec<Array1<f32>>,
	weights: Vec<Array2<f32>>,
}

#[derive(Debug)]
struct MnistData {
	sizes: Vec<i32>,
	data: Vec<u8>,
}

#[derive(Debug)]
pub struct MnistImage {
	pub image: Array2<f64>,
	pub classification: u8,
}

pub fn load_data(dataset_name: &str)
				 -> Result<Vec<MnistImage>, std::io::Error> {
	let filename = format!("data/{}-labels-idx1-ubyte.gz", dataset_name);
	let label_data = &MnistData::new(&(File::open(filename))?)?;
	let filename = format!("data/{}-images-idx3-ubyte.gz", dataset_name);
	let images_data = &MnistData::new(&(File::open(filename))?)?;
	let mut images: Vec<Array2<f64>> = Vec::new();
	let image_shape = (images_data.sizes[1] * images_data.sizes[2]) as usize;
	
	for i in 0..images_data.sizes[0] as usize {
		let start = i * image_shape;
		let image_data = images_data.data[start..(start + image_shape)].to_vec();
		let image_data: Vec<f64> = image_data.into_iter().map(|x| x as f64 / 255.).collect();
		images.push(Array2::from_shape_vec((image_shape, 1), image_data).unwrap());		
	}

	let classifications: Vec<u8> = label_data.data.clone();
	let mut ret: Vec<MnistImage> = Vec::new();
	
	for (image, classification) in images.into_iter().zip(classifications.into_iter()) {
		ret.push(MnistImage {
			image,
			classification,
		})		
	}

	Ok(ret)
}

impl MnistData {
	fn new(f: &File) -> Result<MnistData, std::io::Error> {
		let mut gz = GzDecoder::new(f);
		let mut contents: Vec<u8> = Vec::new();
		gz.read_to_end(&mut contents)?;
		let mut r = Cursor::new(&contents);

		let magic_number = r.read_i32::<BigEndian>()?;

		let mut sizes: Vec<i32> = Vec::new();
		let mut data: Vec<u8> = Vec::new();

		match magic_number {
			2049 => {
				sizes.push(r.read_i32::<BigEndian>()?);
			}
			2051 => {
				sizes.push(r.read_i32::<BigEndian>()?);
				sizes.push(r.read_i32::<BigEndian>()?);
				sizes.push(r.read_i32::<BigEndian>()?);
			}
			_ => panic!(),
		}

		r.read_to_end(&mut data)?;

		Ok(MnistData { sizes, data })
	}
}

impl Network {
	fn new(sizes: Vec<u16>) -> Network {
		let num_layers = sizes.len() as u16;
		let sizes = sizes;
		let mut biases = Vec::new();
		let mut weights = Vec::new();

		for y in sizes[1..].iter() {
			biases.push(aleatory_vec(*y as usize));
		}

		for (y,x) in sizes[1..].iter().zip(sizes[..sizes.len() - 1].iter()) {
			weights.push(aleatory_matrix(*y as usize, *x as usize));
		}

		Network { num_layers, sizes, biases, weights }
    }

	fn feedforward(&self, vector: Array1<f32>) -> Array1<f32> {
		let mut result = vector;

		for (b, w) in self.biases.iter().zip(self.weights.iter()) {
			result = vec_sigmoid(&(w.dot(&result) + b));
		}
		
		result
	}

	fn sgd(&mut self, training_data: &Vec<(Array1<f32>, Array1<f32>)>,
		   epochs: usize,
		   mini_batch_size: usize, eta: f32,
		   test_data:Option<&Vec<(Array1<f32>,Array1<f32>)>>) {
		let mut i: usize = 0;
		let mut mini_batches: Vec<Vec<(Array1<f32>, Array1<f32>)>> = Vec::new();

		for j in 0..epochs {
            println!("Shuffling");
            let shuffled = my_shuffle(&training_data);
			while i < training_data.len() {
				mini_batches.push(
					shuffled[i..i+mini_batch_size].to_vec()
				);
				i += mini_batch_size;
			}

            println!("Mini batches");
            let mut current_iter = 0;
            let start = Instant::now();
			for mini_batch in mini_batches.iter() {
				Network::update_mini_batch(self, mini_batch, eta);
                let duration = start.elapsed();
                println!("Mini batch {}/{} done, {:?} elapsed", current_iter, mini_batches.len(), duration);
                current_iter += 1
			}

			match test_data {
				Some(x) => println!("Epoch {}: {}/{}",
									j, self.evaluate(x.to_vec()), x.len()),
				None => println!("Epoch {} complete", j)
			}
		}
	}

	fn evaluate(&self, test_data: Vec<(Array1<f32>, Array1<f32>)>) -> u16 {
		let mut test_results = Vec::new();
		let mut sum = 0;

		for (u, v) in test_data {
			let x = Network::feedforward(&self, u);
			test_results.push((x,v));
		}

		for (u,v) in test_results.iter() {
       		if argmax(&u) == argmax(&v) {
				sum += 1;
			}
		}

		sum
	}

	fn update_mini_batch(&mut self,
						 mini_batch: &Vec<(Array1<f32>,Array1<f32>)>,
						 eta: f32) {
		let mut nabla_b = Vec::new();
		let mut nabla_w = Vec::new();
		
		for b in self.biases {
			nabla_b.push(Array::zeros(b.shape()));
		}

		for w in self.weights {
			nabla_w.push(Array::zeros(w.shape()));
		}

		for (x,y) in mini_batch.iter() {
			let (delta_nabla_b, delta_nabla_w) = Network::backprop(self, x, y);
			
			for i in 0..delta_nabla_b.len() {
				nabla_b[i] = nabla_b[i] + delta_nabla_b[i];
			}
			
			for i in 0..nabla_w.len() {
				nabla_w[i] = nabla_w[i] + delta_nabla_w[i];
			}
		}

		for i in 0..self.biases.len() {
			self.biases[i] = self.biases[i] - nabla_b[i] * eta / (mini_batch.len() as f32);
		}

		for i in 0..self.weights.len() {
			self.weights[i] = self.weights[i] - nabla_w[i] * eta / (mini_batch.len() as f32);
		}
	}
	
	fn backprop(&self, activation: &Array1<f32>, output: &Array1<f32>) ->
		(Vec<Array1<f32>>, Vec<Array2<f32>>) {
			let mut nabla_b: Vec<Array1<f32>> = Vec::new();
			let mut nabla_w: Vec<Array2<f32>> = Vec::new();
			let mut activations = vec![activation.clone()];
			let mut zs = Vec::new();
			let mut z = activation.clone();
			
			for (b,w) in self.biases.iter().zip(self.weights.iter()) {
				z = w.dot(&z) + b;
				zs.push(z.clone());
				activations.push(vec_sigmoid(&z));
				z = vec_sigmoid(&z);
			}
			
			let mut delta: Array1<f32> = Network::cost_derivative(
				&activations[activations.len() - 1],
				&output
			) * sigmoid_prime(&zs[zs.len() - 1]);
			nabla_b.push(delta.clone());
			nabla_w.push(delta.dot(&array![activations[activations.len() - 2]]));
			
			for l in 2..self.num_layers {
				z = zs[zs.len() - l as usize].clone();
				let sp = sigmoid_prime(&z);
				delta = self.weights[self.weights.len() - (l as usize) + 1].dot(&delta) * sp;
				nabla_b.push(delta.clone());
				nabla_w.push(delta.dot(&array![activations[activations.len() - (l as usize) - 1]]);
			}

			nabla_b.reverse();
			nabla_w.reverse();
			
			(nabla_b, nabla_w)
		}
	
	fn cost_derivative(output_activations: &Array1<f32>, output: &Array1<f32>) -> Array1<f32> {
		output_activations.iter().zip(output.iter()).map(|(&u, &v)| u-v).collect()
	}
}	

fn argmax(v: &Array1<f32>) -> usize {
	let x = v[0];
	let mut index = 0;

	for i in 0..v.len() {
		if v[i] > x {
			index = i;
		}
	}
	
	index
}

fn sigmoid(x: &f32) -> f32 {
	1.0 / (1.0 + f32::powf(E,-x))
}

fn vec_sigmoid(vetor: &Array1<f32>) -> Array1<f32> {
	vetor.map(|&x| sigmoid(&x))
}

fn sigmoid_prime(vetor: &Array1<f32>) -> Array1<f32> {
	vetor.mapv(|x| sigmoid(&x)*(1.0-sigmoid(&x)))
}

fn aleatory_vec(tamanho: usize) -> Array1<f32> {
	Array::random(tamanho, Uniform::new(-1., 1.))
}

fn aleatory_matrix(linhas: usize, colunas: usize) -> Array2<f32> {
	Array::random((linhas, colunas), Uniform::new(-1., 1.))
}

fn my_shuffle<T: Clone>(vetor: &[T]) -> Vec<T> {
	let mut shuffled = Vec::new();
	let mut aux: Vec<usize> = (0..vetor.len()).collect();

	aux.shuffle(&mut rand::thread_rng());

	for i in 0..aux.len() {
		shuffled.push(vetor[aux[i]].clone());
	}

	shuffled
}

fn main() {
    let net_sizes: Array1<u16> = array![784,30,10];
    let mut net = Network::new(net_sizes);

    let train = load_data("train").unwrap();
    let mut training_data = Vec::new();
    for i in 0..train.len() {
        let input_iter = train[i].image.into_iter();
        let mut input: Vec<f32> = Vec::new();
        for x in input_iter {
            input.push(*x as f32);
        }
        let mut output: Vec<f32> = vec![0.0; 10];
        output[train[i].classification as usize] = 1.0;
        training_data.push((input, output))
    }

    let test = load_data("t10k").unwrap();
    let mut test_data = Vec::new();
    for i in 0..test.len() {
        let input_iter = test[i].image.into_iter();
        let mut input: Vec<f32> = Vec::new();
        for x in input_iter {
            input.push(*x as f32);
        }
        let mut output: Vec<f32> = vec![0.0; 10];
        output[test[i].classification as usize] = 1.0;
        test_data.push((input, output))
    }

	println!("Started.");
    net.sgd(&training_data, 30, 100, 3.0, Some(&test_data));
    let acc = net.evaluate(training_data);
    println!("{:?}", acc);
}
