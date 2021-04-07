use rand::seq::SliceRandom;
use rand::Rng;
use std::f32::consts::E;
use ndarray::Array2;
use std::io::{Cursor, Read};
use flate2;
use flate2::read::GzDecoder;
use std::fs::File;
use byteorder::BigEndian;
use byteorder::ReadBytesExt;

#[derive(Debug)]
struct Network {
	num_layers: u16,
	sizes: Vec<u16>,
	biases: Vec<Vec<f32>>,
	weights: Vec<Vec<Vec<f32>>>,
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

pub fn load_data(dataset_name: &str) -> Result<Vec<MnistImage>, std::io::Error> {
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
			biases.push(vetor_aleatorio(*y as usize));
		}

		for (y,x) in sizes[1..].iter().zip(sizes[..sizes.len() - 1].iter()) {
			weights.push(matriz_aleatoria(*y as usize, *x as usize));
		}

		Network { num_layers, sizes, biases, weights }
    }

	fn feedforward(&self, vetor: Vec<f32>) -> Vec<f32> {
		let mut result = vetor;

		for (b, w) in self.biases.iter().zip(self.weights.iter()) {
			result = vec_sigmoid(&dot(w, &result)
								 .iter()
								 .zip(b.iter())
								 .map(|(&u, &v)| u+v)
								 .collect());
		}
		println!("{:?}", result);
		result
	}

	fn sgd(&mut self, training_data: &Vec<(Vec<f32>, Vec<f32>)>, epochs: usize,
		   mini_batch_size: usize, eta: f32,
		   test_data:Option<&Vec<(Vec<f32>,Vec<f32>)>>) {
		let mut i: usize = 0;
		let mut mini_batches: Vec<Vec<(Vec<f32>, Vec<f32>)>> = Vec::new();

		for j in 0..epochs {
			while i < training_data.len() {
				mini_batches.push(
					my_shuffle(&training_data)[i..i+mini_batch_size].to_vec()
				);
				i += mini_batch_size;
			}

			for mini_batch in mini_batches.iter() {
				Network::update_mini_batch(self, mini_batch, eta);
			}

			match test_data {
				Some(x) => println!("Epoch {}: {}/{} \n {:?}",
									j, self.evaluate(x.to_vec()), x.len(), training_data[j].1),
				None => println!("Epoch {} complete", j)
			}
		}
	}

	fn evaluate(&self, test_data: Vec<(Vec<f32>, Vec<f32>)>) -> u16 {
		let mut test_results = Vec::new();
		let mut soma = 0;

		for i in 0..test_data.len() {
            let a = &test_data[i].0;
            let b = &test_data[i].1;
			let x = Network::feedforward(&self, a.to_vec());
			test_results.push((x,b));
		}

		for (u,v) in test_results.iter() {
       		if argmax(&u) == argmax(&v) {
				soma += 1;
			}
		}

		soma
	}

	fn update_mini_batch(&mut self, mini_batch: &Vec<(Vec<f32>,Vec<f32>)>, eta: f32) {
		let mut nabla_b = Vec::new();
		let mut nabla_w = Vec::new();
		
		for i in 0..self.biases.len() {
			nabla_b.push(vec![0.0; (self.biases[i]).len()]);
		}
		
		for i in 0..self.weights.len() {
			nabla_w.push(vec![vec![0.0; self.weights[i][0].len()];
							  self.weights[i].len()]);
		}
		
		for (x,y) in mini_batch.iter() {
			let (delta_nabla_b, delta_nabla_w) = Network::backprop(self, x, y);
			
			for i in 0..delta_nabla_b.len() {
				nabla_b[i] = soma_vetores(&nabla_b[i], &delta_nabla_b[i]);
			}
			
			for i in 0..nabla_w.len() {
				nabla_w[i] = soma_matrizes(&nabla_w[i], &delta_nabla_w[i]);
			}
		}


		for i in 0..self.biases.len() {
			for j in 0..self.biases[i].len() {
				self.biases[i][j] = self.biases[i][j] - nabla_b[i][j] * eta / (mini_batch.len() as f32);
			}
		}

		for i in 0..self.weights.len() {
			for j in 0..self.weights[i].len() {
				for k in 0..self.weights[i][j].len() {
					self.weights[i][j][k] = self.weights[i][j][k] - nabla_w[i][j][k] * eta / (mini_batch.len() as f32);
				}
			}
		}
	}
	
	fn backprop(&self, activation: &Vec<f32>, y: &Vec<f32>) ->
		(Vec<Vec<f32>>, Vec<Vec<Vec<f32>>>) {
			let mut nabla_b: Vec<Vec<f32>> = Vec::new();
			let mut nabla_w: Vec<Vec<Vec<f32>>> = Vec::new();
			let mut activations = vec![activation.clone()];
			let mut zs = Vec::new();
			let mut z = activation.clone();
			
			for (b,w) in self.biases.iter().zip(self.weights.iter()) {
				z = dot(w, &z)
					.iter()
					.zip(b.iter())
					.map(|(&u, &v)| u+v)
					.collect();
				zs.push(z.clone());
				activations.push(vec_sigmoid(&z));
				z = vec_sigmoid(&z);
			}
			
			let mut delta: Vec<f32> = Network::cost_derivative(&activations[activations.len() - 1], &y)
				.iter()
				.zip(sigmoid_prime(&zs[zs.len() - 1]).iter())
				.map(|(&u,&v)| u*v)
				.collect();
			nabla_b.push(delta.clone());
			nabla_w.push(multiplica_matrizes(&transpose(&vec![delta.clone()]),
										&vec![activations[activations.len() - 2].clone()]));
			
			for l in 2..self.num_layers {
				z = zs[zs.len() - l as usize].clone();
				let sp = sigmoid_prime(&z);
				delta = dot(&transpose(&self.weights[self.weights.len() - (l as usize) + 1]),
							&delta)
					.iter()
					.zip(sp.iter())
					.map(|(&u,&v)| u*v)
					.collect();
				nabla_b.push(delta.clone());
				nabla_w.push(multiplica_matrizes(&transpose(&vec![delta.clone()]),
												 &vec![activations[activations.len() - (l as usize) - 1].clone()]));
			}

			nabla_b.reverse();
			nabla_w.reverse();
			
			(nabla_b, nabla_w)
		}
	
	fn cost_derivative(output_activations: &Vec<f32>, y: &Vec<f32>) -> Vec<f32> {
		output_activations.iter().zip(y.iter()).map(|(&u, &v)| u-v).collect()
	}
}	

fn multiplica_matrizes(matriz1: &Vec<Vec<f32>>, matriz2: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
	let mut aux = 0.0;
	let mut matriz3 = vec![vec![0.0; matriz2[0].len()]; matriz1.len()];

	for i in 0..matriz1.len() {
		for j in 0..matriz2[0].len() {
			for k in 0..matriz2.len() {
				aux += matriz1[i][k] * matriz2[k][j];
			}
			matriz3[i][j] = aux;
			aux = 0.0;
		}
	}

	matriz3
}

fn argmax(v: &Vec<f32>) -> usize {
	let x = v[0];
	let mut index = 0;

	for i in 0..v.len() {
		if v[i] > x {
			index = i;
		}
	}
	index
}

fn soma_vetores(vetor1: &Vec<f32>, vetor2: &Vec<f32>) -> Vec<f32> {
	vetor1.iter().zip(vetor2.iter()).map(|(&u, &v)| u+v).collect()
}

fn soma_matrizes(matriz1: &Vec<Vec<f32>>, matriz2: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
	let mut matriz3: Vec<Vec<f32>> = vec![vec![0.0; matriz1[0].len()]; matriz1.len()];
	
	for i in 0..matriz1.len() {
		for j in 0..matriz1[0].len() {
			matriz3[i][j] = matriz1[i][j] + matriz2[i][j];
		}
	}
	
	matriz3
}

fn transpose(matriz: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
	let mut transposta = vec![vec![0.0; matriz.len()]; matriz[0].len()];

	for i in 0..matriz.len() {
		for j in 0..matriz[i].len() {
			transposta[j][i] = matriz[i][j];
		}
	}

	transposta
}

fn dot(matriz: &Vec<Vec<f32>>, vetor: &Vec<f32>) -> Vec<f32> {
	let mut resultado = vec![0.0; matriz.len()];
	
	for i in 0..matriz.len() {
		for j in 0..vetor.len() {
			resultado[i] += matriz[i][j] * vetor[j];
		}
	}

	resultado
}

fn sigmoid(x: &f32) -> f32 {
	1.0 / (1.0 + f32::powf(E,-x))
}

fn vec_sigmoid(vetor: &Vec<f32>) -> Vec<f32> {
	vetor.iter().map(|&x| sigmoid(&x)).collect()
}

fn sigmoid_prime(vetor: &Vec<f32>) -> Vec<f32> {
	vetor.iter().map(|&x| sigmoid(&x)*(1.0-sigmoid(&x))).collect()
}

fn vetor_aleatorio(tamanho: usize) -> Vec<f32> {
	(0..tamanho).map(|_| rand::thread_rng().gen_range(-1.0..1.0)).collect()
}

fn matriz_aleatoria(linhas: usize, colunas: usize) -> Vec<Vec<f32>> {
	let mut matriz = vec![vec![0.0; colunas]; linhas];

	for i in 0..linhas {
		matriz[i] = vetor_aleatorio(colunas);
	}

	matriz
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
    let net_sizes: Vec<u16> = vec![784,30,10];
    let mut net = Network::new(net_sizes);

    let train = load_data("t10k").unwrap();
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

	println!("Started.");
    net.sgd(&training_data, 30, 10, 3.0, Some(&training_data));
    let acc = net.evaluate(training_data);
    println!("{:?}", acc);
}
