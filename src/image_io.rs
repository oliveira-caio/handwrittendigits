use image;
use ndarray::Array2;
use image::imageops;

pub fn to_image(arr: &Array2<f32>, path: &str) { 
    let size = (arr.shape()[0]
                as f64).sqrt() as i32 as usize;
    let new_arr = arr.clone().into_shape((size, size)).unwrap().reversed_axes() * 255.0;

    let mut img = image::ImageBuffer::new(size as u32, size as u32);
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        *pixel = image::Luma([
            (new_arr[[x as usize, y as usize]]) as u8
        ]);
    }
    imageops::invert(&mut img);
    img.save(path).unwrap();
}

pub fn read_image_standard(path: &str) -> Array2<f32> {
    let img  = image::open(path).unwrap().to_luma8();
    let img2 = img.into_vec();
    let mut aux: Array2<f32> = Array2::zeros((784,1));
    for j in 0..784 {
        aux[(j,0)] = 1.0 - (img2[j] as f32 / 255.0);
    }

    aux
}

pub fn read_image(path: &str) -> Array2<u8> {
    let img  = image::open(path).unwrap().to_luma8();
    let img2 = img.into_vec();
    let mut aux: Array2<u8> = Array2::zeros((784,1));
    for j in 0..784 {
        aux[(j,0)] = 255 - img2[j];
    }

    aux
}

pub fn read_image_square_standard(path: &str) -> Array2<f32> {
    let mut img  = image::open(path).unwrap().to_luma8();
    let n = (img.len() as f64).sqrt() as i32 as usize;
    let img2 = img.into_vec();
    let mut aux = Array2::zeros((n,n));
    for i in 0..n {
        for j in 0..n {
            aux[(i,j)] = 1.0 - (img2[i*n + j] as f32 / 255.0);
        }
    }

    aux
}

pub fn read_image_square(path: &str) -> Array2<u8> {
    let mut img  = image::open(path).unwrap().to_luma8();
    let n = (img.len() as f64).sqrt() as i32 as usize;
    let img2 = img.into_vec();
    let mut aux: Array2<u8> = Array2::zeros((n,n));
    for i in 0..n {
        for j in 0..n {
            aux[(i,j)] = 255 - img2[i*n + j];
        }
    }

    aux
}
