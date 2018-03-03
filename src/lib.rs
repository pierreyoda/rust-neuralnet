#[cfg(test)]
#[macro_use]
extern crate approx;
extern crate ndarray;
extern crate rand;

pub type Float = f64;
pub type ResultString<T> = Result<T, String>;

pub mod activation;
pub mod builder;
pub mod layer;
pub mod network;
pub mod training;
mod utils;
