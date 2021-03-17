use rand::Rng;
use std::f64::consts::E;

#[derive(Debug)]
struct Network {
    num_layers: u32,
    sizes: Vec<u64>,
    biases: Vec<Vec<f64>>,
    weights: Vec<Vec<Vec<f64>>>,
}

impl Network {
    fn new(sizes: Vec<u64>) -> Network {
		let num_layers = sizes.len() as u32;
		let sizes = sizes;
		let mut biases = Vec::new();
		let mut weights = Vec::new();
		let x = &sizes[..sizes.len()-1];
		let y = &sizes[1..];
		
		for i in y.iter() {
			let layer = (0..*i).map(|_| rand::thread_rng().gen_range(-1.0,1.0)).collect();
			biases.push(layer);
		}
		
		for i in 0..y.len() {
			let mut linha = Vec::new();
			for _ in 0..y[i] {
				let coluna = (0..x[i]).map(|_| rand::thread_rng().gen_range(-1.0,1.0)).collect();
				linha.push(coluna);
			}
			weights.push(linha);
		}
		
		Network { num_layers, sizes, biases, weights }
    }

	// Return the output of the network if ``a`` is input.
	fn feedforward(&self, vetor: &Vec<f64>) -> Vec<f64> {
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

	fn sgd(&mut self, training_data: Vec<(Vec<f64>, Vec<f64>)>, epochs: u8, mini_batch_size: u8, eta: f64) {
		// let mut rng = thread_rng();
		let mut i: usize = 0;
		let mut mini_batches: Vec<Vec<(Vec<f64>, Vec<f64>)>> = Vec::new();
		
		for j in 0..(epochs as usize) {
			// training_data.shuffle(&mut rng);

			while i < training_data.len() {
				mini_batches.push(training_data[i..i+mini_batch_size as usize].to_vec());
				i += mini_batch_size as usize;
			}
			
			for mini_batch in mini_batches.iter() {
				Network::update_mini_batch(self, mini_batch, eta);
			}
		}
	 }

	// """Return the number of test inputs for which the neural
    //     network outputs the correct result. Note that the neural
    //     network's output is assumed to be the index of whichever
    //     neuron in the final layer has the highest activation."""
	fn evaluate(&self, test_data: (Vec<f64>, Vec<f64>)) -> u64 {
		let mut test_results = Vec::new();
		let mut soma = 0;
		let mut a = test_data.0;
		let b = test_data.1;
		
		for i in 0..a.len() {
			let x = argmax(&Network::feedforward(&self, &mut a));
			test_results.push((x,b[i]));
		}
		
		for i in 0..a.len() {
			if a[i] == b[i] {
				soma += 1;
			}
		}
		soma
	}

	fn update_mini_batch(&mut self, mini_batch: &Vec<(Vec<f64>,Vec<f64>)>, eta: f64) {
		let mut nabla_b = Vec::new();
		let mut nabla_w = Vec::new();
		
		for i in 0..self.biases.len() {
			nabla_b[i] = vec![0.0; (self.biases[i]).len()];
		}
		
		for i in 0..self.weights.len() {
			nabla_w[i] = vec![vec![0.0]; (self.weights[i]).len()];
			for j in 0..nabla_w[i].len() {
				nabla_w[i][j] = vec![0.0; (self.weights[i][j]).len()];
			}
		}
		
		for (x, y) in mini_batch.iter() {
			let (delta_nabla_b, delta_nabla_w) = Network::backprop(self, x, y);
			
			for i in 0..delta_nabla_b.len() {
				nabla_b[i] = soma_vetores(&nabla_b[i], &delta_nabla_b[i]);
			}
			
			for i in 0..delta_nabla_w.len() {
				nabla_w[i] = soma_matrizes(&nabla_w[i], &delta_nabla_w[i]);
			}
		}

		for i in 0..self.biases.len() {
			for j in 0..self.biases[i].len() {
				self.biases[i][j] = self.biases[i][j] - self.biases[i][j]*eta/(mini_batch.len() as f64);
			}
		}

		for i in 0..self.weights.len() {
			for j in 0..self.weights[i].len() {
				for k in 0..self.weights[i][j].len() {
					self.weights[i][j][k] = self.weights[i][j][k] - self.weights[i][j][k]*eta/(mini_batch.len() as f64);
				}
			}
		}
	}
	
	fn backprop(&self, activation: &Vec<f64>, y: &Vec<f64>) ->
		(Vec<Vec<f64>>, Vec<Vec<Vec<f64>>>) {
			let mut nabla_b: Vec<Vec<f64>> = Vec::new();
			let mut nabla_w: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0]]; self.weights.len()];
			let mut activations = vec![activation.clone()];
			let mut zs = Vec::new();
			let mut z: Vec<f64> = Vec::new();
			
			for (b,w) in self.biases.iter().zip(self.weights.iter()) {
				z = dot(w, &activation).iter().zip(b.iter()).map(|(&u, &v)| u+v).collect();
				zs.push(z.clone());
				activations.push(vec_sigmoid(&mut z));
			}
			
			let mut delta: Vec<f64> = Network::cost_derivative(&activations[activations.len() - 1], &y)
				.iter()
				.zip(sigmoid_prime(&zs[zs.len() - 1 as usize]).iter())
				.map(|(&u,&v)| u*v)
				.collect();
			nabla_b.push(delta.clone());
			nabla_w.push(
				multiplica_matriz(&vec![delta.clone()],
								  &transpose(
									  &vec![activations[activations.len() - 2 as usize].clone()])).clone());
			
			for l in 2..self.num_layers {
				z = zs[zs.len() - 1 as usize].clone();
				let sp = sigmoid_prime(&z);
				delta = dot(&transpose(&self.weights[1-l as usize]), &delta)
					.iter()
					.zip(sp.iter())
					.map(|(&u,&v)| u*v)
					.collect();
				nabla_b.push(delta.clone());
				nabla_w.push(
				multiplica_matriz(&vec![delta.clone()],
								  &transpose(
									  &vec![activations[activations.len() - l as usize - 1].clone()])).clone());
			}

			let nabla_b = inverte_vetor(&nabla_b);
			let nabla_w = inverte_vetor(&nabla_w);
			
			(nabla_b,nabla_w)
				
		}
	
	fn cost_derivative(output_activations: &Vec<f64>, y: &Vec<f64>) -> Vec<f64> {
		output_activations.iter().zip(y.iter()).map(|(&u, &v)| u-v).collect()
	}
}

fn multiplica_matriz(matriz1: &Vec<Vec<f64>>, matriz2: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
	let mut aux = 0.0;
	let mut matriz3 = vec![vec![0.0]];
	
	for i in 0..matriz1.len() { // 2
		for j in 0..matriz1[i].len() { // 3
			for k in 0..matriz1[i].len() { // 3
				aux += matriz1[i][k]*matriz2[k][j];
			}
			matriz3[i][j] = aux;
		}
	}

	matriz3
}

fn soma_vetores(vetor1: &Vec<f64>, vetor2: &Vec<f64>) -> Vec<f64> {
	vetor1.iter().zip(vetor2.iter()).map(|(&u, &v)| u+v).collect()
}

fn soma_matrizes(matriz1: &Vec<Vec<f64>>, matriz2: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
	let mut matriz3: Vec<Vec<f64>> = Vec::new();
	
	for i in 0..matriz1.len() {
		for j in 0..matriz1[i].len() {
			matriz3[i][j] = matriz1[i][j] + matriz2[i][j];
		}
	}
	matriz3
}

fn inverte_vetor<T: Clone>(vetor: &Vec<T>) -> Vec<T> {
	vetor.iter().rev().cloned().collect()
}

fn transpose(matriz: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
	let mut transposta = vec![vec![0.0]];

	for i in 0..matriz.len() {
		for j in 0..matriz[i].len() {
			transposta[i][j] = matriz[j][i];
		}
	}

	transposta
}

fn dot(matriz: &Vec<Vec<f64>>, vetor: &Vec<f64>) -> Vec<f64> {
	let mut resultado = vec![0.0; vetor.len()];
	
	for i in 0..matriz.len() {
		for j in 0..matriz[i].len() {
			resultado[i] += matriz[i][j] * vetor[j];
		}
	}
	
	resultado
}

fn argmax(vetor: &Vec<f64>) -> f64 {
	let mut max: f64 = vetor[0];
	
	for x in vetor.iter() {
		if max < *x {
			max = *x;
		}
	}
	
	max
}

fn sigmoid(x: &f64) -> f64 {
	1.0 / (1.0 + f64::powf(E,-x))
}

fn vec_sigmoid(vetor: &Vec<f64>) -> Vec<f64> {
	vetor.iter().map(|&x| sigmoid(&x)).collect()
}

fn sigmoid_prime(vetor: &Vec<f64>) -> Vec<f64> {
	vetor.iter().map(|&x| sigmoid(&x)*(1.0-sigmoid(&x))).collect()
}

fn main() {
    let teste: Vec<u64> = vec![2,3,1];
    let net = Network::new(teste);
	
	println!("{:#?}, {}", net.weights, net.weights[1][0][2]);
}
