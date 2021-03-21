use rand::seq::SliceRandom;
use rand::Rng;
use std::f32::consts::E;

#[derive(Debug)]
struct Network {
	num_layers: u8,
	sizes: Vec<u8>,
	biases: Vec<Vec<f32>>,
	weights: Vec<Vec<Vec<f32>>>,
}

impl Network {
	fn new(sizes: Vec<u8>) -> Network {
		let num_layers = sizes.len() as u8;
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

	fn feedforward(&self, vetor: &Vec<f32>) -> Vec<f32> {
		let mut result = Vec::new();

		for (b, w) in self.biases.iter().zip(self.weights.iter()) {
			result = vec_sigmoid(&dot(w, &result)
								 .iter()
								 .zip(b.iter())
								 .map(|(&u, &v)| u+v)
								 .collect());
		}

		result
	}

	fn sgd(&mut self, training_data: &Vec<(Vec<f32>, Vec<f32>)>, epochs: usize,
		   mini_batch_size: usize, eta: f32) {
		let mut i: usize = 0;
		let mut mini_batches: Vec<Vec<(Vec<f32>, Vec<f32>)>> = Vec::new();
		let mut training_data_shuffled = Vec::new();

		for j in 0..epochs {
			training_data_shuffled = my_shuffle(&training_data);
			
			while i < training_data.len() {
				mini_batches.push(training_data_shuffled[i..i+mini_batch_size].to_vec());
				i += mini_batch_size;
			}

			for mini_batch in mini_batches.iter() {
				Network::update_mini_batch(self, mini_batch, eta);
			}

			println!("Epoch {} complete", j);
		}
	}

	fn evaluate(&self, test_data: (Vec<f32>, Vec<f32>)) -> u16 {
		let mut test_results = Vec::new();
		let mut soma = 0;
		let (mut a, b) = test_data;

		for i in 0..a.len() {
			let x = argmax(&Network::feedforward(&self, &mut a));
			test_results.push((x,b[i]));
		}

		for (u,v) in test_results.iter() {
			if *u == *v {
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
			nabla_w.push(vec![Vec::new(); (self.weights[i]).len()]);
			for j in 0..nabla_w[i].len() {
				nabla_w[i].push(vec![0.0; (self.weights[i][j]).len()]);
			}
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
				self.biases[i][j] = self.biases[i][j] - self.biases[i][j] * eta / (mini_batch.len() as f32);
			}
		}

		for i in 0..self.weights.len() {
			for j in 0..self.weights[i].len() {
				for k in 0..self.weights[i][j].len() {
					self.weights[i][j][k] = self.weights[i][j][k] - self.weights[i][j][k] * eta / (mini_batch.len() as f32);
				}
			}
		}
	}
	
	fn backprop(&self, activation: &Vec<f32>, y: &Vec<f32>) ->
		(Vec<Vec<f32>>, Vec<Vec<Vec<f32>>>) {
			let mut nabla_b: Vec<Vec<f32>> = Vec::new();
			let mut nabla_w: Vec<Vec<Vec<f32>>> = vec![vec![vec![0.0]]; self.weights.len()];
			let mut activations = vec![activation.clone()];
			let mut zs = Vec::new();
			let mut z = Vec::new();
			
			for (b,w) in self.biases.iter().zip(self.weights.iter()) {
 				z = dot(w, &activation)
					.iter()
					.zip(b.iter())
					.map(|(&u, &v)| u+v)
					.collect();
				zs.push(z.clone());
				activations.push(vec_sigmoid(&z));
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
				z = zs[zs.len() - 1].clone();
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

			(inverte_vetor(&nabla_b), inverte_vetor(&nabla_w))
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

fn inverte_vetor<T: Clone>(vetor: &Vec<T>) -> Vec<T> {
	vetor.iter().rev().cloned().collect()
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

fn argmax(vetor: &Vec<f32>) -> f32 {
	let mut max: f32 = vetor[0];
	
	for x in vetor.iter() {
		if max < *x {
			max = *x;
		}
	}

	max
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

fn my_shuffle<T: Clone>(vetor: &Vec<T>) -> Vec<T> {
	let mut shuffled = Vec::new();
	let mut aux: Vec<usize> = (0..vetor.len()).collect();

	aux.shuffle(&mut rand::thread_rng());

	for i in 0..aux.len() {
		shuffled.push(vetor[aux[i]].clone());
	}

	shuffled
}

fn main() {
    let teste: Vec<u8> = vec![2,3,1];
    let mut net = Network::new(teste);
    let mut training_data = Vec::new();
	let mut rng = rand::thread_rng();

    for i in 0..100 {
        let input: Vec<f32> = (0..2).map(|_| rng.gen_range(-0.0..1.0)).collect();
        let output: Vec<f32> = (0..1).map(|_| rng.gen_range(-0.0..1.0)).collect();
        training_data.push((input, output))
    }

    net.sgd(&mut training_data,3,100,3.0);
}
