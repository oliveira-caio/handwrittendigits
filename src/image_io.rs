use image;
use image::{GenericImage, GenericImageView, ImageBuffer, RgbImage, imageops};
use ndarray::Array2;
use ndarray_image;

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

/*
pub fn to_vec(path: &str) {
    let img = ndarray_image::open_gray_image(path).unwrap(); 
    let arr: Array<f32>img.into(Array2)/255.0
}
*/

