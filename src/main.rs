use byteorder::BigEndian;
use byteorder::ReadBytesExt;
use flate2;
use flate2::read::GzDecoder;
use ndarray::Array2;
use ndarray_rand::RandomExt;
use rand::seq::SliceRandom;
use std::f32::consts::E;
use std::fs::File;
use std::io::{Cursor, Read};
use ndarray_rand::rand_distr::StandardNormal;

#[derive(Debug)]
struct Network {
    num_layers: usize,
    sizes: Vec<usize>,
    biases: Vec<Array2<f32>>,
    weights: Vec<Array2<f32>>,
}

#[derive(Debug)]
struct MnistData {
    sizes: Vec<i32>,
    data: Vec<u8>,
}

#[derive(Debug)]
pub struct MnistImage {
    pub image: Array2<f32>,
    pub classification: u8,
}

pub fn load_data(dataset_name: &str) -> Result<Vec<MnistImage>, std::io::Error> {
    let filename = format!("data/{}-labels-idx1-ubyte.gz", dataset_name);
    let label_data = &MnistData::new(&(File::open(filename))?)?;
    let filename = format!("data/{}-images-idx3-ubyte.gz", dataset_name);
    let images_data = &MnistData::new(&(File::open(filename))?)?;
    let mut images: Vec<Array2<f32>> = Vec::new();
    let image_shape = (images_data.sizes[1] * images_data.sizes[2]) as usize;

    for i in 0..images_data.sizes[0] as usize {
        let start = i * image_shape;
        let image_data = images_data.data[start..(start + image_shape)].to_vec();
        let image_data: Vec<f32> = image_data.into_iter().map(|x| x as f32 / 255.).collect();
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
    fn new(sizes: Vec<usize>) -> Network {
        let num_layers = sizes.len();
        let sizes = sizes;
        let mut biases = Vec::new();
        let mut weights = Vec::new();

		for i in 1..sizes.len() {
			biases.push(vec_random(sizes[i]));
			weights.push(matrix_random(sizes[i], sizes[i-1]));
		}
		
        // for y in sizes[1..].iter() {
        //     biases.push(vec_random(*y as usize));
        // }

        // for (y, x) in sizes[1..].iter().zip(sizes[..sizes.len() - 1].iter()) {
        //     weights.push(matrix_random(*y as usize, *x as usize));
        // }

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

	fn evaluate(&self, test_data: &Vec<(Array2<f32>, Array2<f32>)>) -> usize {
        let mut test_results = Vec::new();
        let mut sum = 0;

        for (u, v) in test_data {
            test_results.push((self.feedforward(u), v));
        }

        for (u, v) in test_results {
            if argmax(&u) == argmax(&v) {
                sum += 1;
            }
        }

        sum
	}
	
    fn cost_derivative(&self, output_activations: &Array2<f32>,
					   output: &Array2<f32>) -> Array2<f32> {
        output_activations - output
    }

    fn sgd(
        &mut self,
        training_data: &Vec<(Array2<f32>, Array2<f32>)>,
        epochs: usize,
        mini_batch_size: usize,
        eta: f32,
        test_data: Option<&Vec<(Array2<f32>, Array2<f32>)>>,
    ) {
        let mut i: usize = 0;
        let mut mini_batches: Vec<Vec<(Array2<f32>, Array2<f32>)>> = Vec::new();

        for j in 0..epochs {
            let shuffled = my_shuffle(&training_data);
            while i < training_data.len() {
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
			
			for (b, nb) in nabla_b.iter_mut().zip(delta_nabla_b.iter()) {
        		*b += nb;
			}

			for (w, nw) in nabla_w.iter_mut().zip(delta_nabla_w.iter()) {
        		*w += nw;
			}			
        }

        for (b, nb) in self.biases.iter_mut().zip(nabla_b.iter()) {
        	*b -= &nb.mapv(|x| x * eta / nbatch)
        }

        for (w, nw) in self.weights.iter_mut().zip(nabla_w.iter()) {
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
        let mut delta: Array2<f32> =
            self.cost_derivative(&activations[asize - 1], &output)
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

fn argmax(v: &Array2<f32>) -> (usize, usize) {
    let x = v[(0, 0)];
    let mut index = 0;

    for i in 0..v.len() {
        if v[(i, 0)] > x {
            index = i;
        }
    }

    (index, 0)
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

fn vec_random(tamanho: usize) -> Array2<f32> {
    Array2::random((tamanho, 1), StandardNormal)
}

fn matrix_random(linhas: usize, colunas: usize) -> Array2<f32> {
	Array2::random((linhas, colunas), StandardNormal)	
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
    let net_sizes: Vec<usize> = vec![784, 30, 10];
    let mut net = Network::new(net_sizes);

    let train = load_data("train").unwrap();
    let mut training_data = Vec::new();
    for s in train.iter() {
        let mut output = Array2::zeros((10, 1));
        output[(s.classification as usize, 0)] = 1.0;
        training_data.push((s.image.clone(), output))
    }

    let test = load_data("t10k").unwrap();
    let mut test_data = Vec::new();
    for s in test.iter() {
        let mut output = Array2::zeros((10, 1));
        output[(s.classification as usize, 0)] = 1.0;
        test_data.push((s.image.clone(), output))
    }

    net.sgd(&training_data, 30, 20, 3.0, Some(&test_data));
    // let acc = net.evaluate(training_data);
    // println!("{:?}", acc);
}
