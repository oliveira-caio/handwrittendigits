use image;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use image::imageops;
use ndarray::Array2;
use rand::Rng;

fn contrast(img: &Array2<f32>, intensity: f32) -> Array2<f32> {
    let mut rng = rand::thread_rng();
    let displacement = rng.gen_range(0.0..intensity);
    let scaled_image = (img*(1.0-intensity)) + (displacement)/2.0;

    scaled_image
}

fn jitter(img: &Array2<f32>, intensity: f32) -> Array2<f32> {
    let mut random_matrix:Array2<f32> = Array2::random(img.raw_dim(), StandardNormal);
    random_matrix = random_matrix.clone() + (1.0 - random_matrix)*(1.0-intensity);
    let jitter_img = (img.dot(&random_matrix)) + (-random_matrix.clone()+1.0)/2.0;

    jitter_img
}

fn crop(image: &Array2<f32>) -> Array2<f32> {
    let n = (image.shape()[0]
            as f64).sqrt() as i32 as usize;
    //let crop_matrix: Array2<f32> = Array2::ones((n, n));
    let x = rand::thread_rng().gen_range(10..20);
    let y = rand::thread_rng().gen_range(10..20);
    let size: usize = rand::thread_rng().gen_range(2..5);
    let mut copy = image.clone().into_shape((n, n)).unwrap();
    //copy = copy * crop_matrix;

    for i in 0..size {
        for j in 0..size {
            copy[(x+i,y+j)] = 0.0;
        }
    }

    copy.into_shape((n*n, 1)).unwrap()
}

pub fn transform(image: &Array2<f32>) -> Array2<f32> {
    let mut rng = rand::thread_rng();
    let contrast_intensity = rng.gen_range(0.0..0.6);
    let jitter_intensity = rng.gen_range(0.0..0.4);

    let new_image = jitter(&contrast(&crop(image), contrast_intensity), jitter_intensity);

    new_image
}
