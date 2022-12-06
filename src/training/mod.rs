use super::{Float, ResultString};

mod sample;
mod trainer;

pub use self::sample::{prepare_dataset, Sample};
pub use self::trainer::{Trainer, TrainerHaltCondition};
