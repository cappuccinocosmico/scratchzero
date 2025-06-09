#![feature(generic_const_exprs)]

use crate::mnist::layers::dyn_sequential::make_test_sequential;
mod mnist;

fn main() {
    println!("Hello, world!");
    let _seq = make_test_sequential();
    println!("Successfully initialized sequential")
}
