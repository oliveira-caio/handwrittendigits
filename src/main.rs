use rand::Rng;

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
}

fn main() {
    let teste = vec![2,3,1];
    let net = Network::new(teste);
    println!("{:#?}, {}", net.weights, net.weights[1][0][2]);
}
