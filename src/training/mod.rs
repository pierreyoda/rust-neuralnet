use super::{Float, ResultString};

mod sample;
mod trainer;

pub use self::sample::{Sample, prepare_dataset};
pub use self::trainer::{Trainer, TrainerHaltCondition};
