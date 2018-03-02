#[cfg(test)]
#[macro_use]
extern crate approx;
#[macro_use]
extern crate ndarray;
extern crate rand;

pub type Float = f64;

pub mod activation;
pub mod layer;
pub mod network;
mod utils;
