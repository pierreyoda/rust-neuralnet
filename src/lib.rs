#[cfg(test)]
#[macro_use]
extern crate approx;
extern crate ndarray;
extern crate rand;

pub type Float = f64;

pub mod activation;
pub mod layer;
pub mod network;
mod utils;
