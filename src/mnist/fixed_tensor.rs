use const_for::const_for;
use rand::random_range;

type DimSizeType = usize;
type flt = f32;

const fn dims_size<const n: usize>(dims: [usize; n]) -> usize {
    let mut res = 0;
    const_for!( i in 0..dims.len() => {
        res = res * dims[i];
    });
    res
}

const fn matsize(a: usize, b: usize) -> usize {
    return a * b;
}

// I want you to get around this by creating a macro: define_tensor(num)
// That will make a generic tensor of a specific dimension like so:
// define_tensor_type!(2)
// expands to:
struct Tensor_2<const A: usize, const B: usize>
where
    [flt; A * B]: Sized,
{
    values: [flt; A * B],
}
impl<const A: usize, const B: usize> Tensor_2<A, B>
where
    [flt; A * B]: Sized,
{
    const SIZE: usize = A * B;

    fn zeros() -> Self {
        Self {
            values: [0.0; A * B],
        }
    }

    fn random() -> Self {
        let mut values = [0.0; A * B];
        for i in 0..values.len() {
            values[i] = random_range(-1.0..1.0)
        }
        return Self { values };
    }
    fn add(&self, other: &Self) -> Self {
        let mut values = [0.0; A * B];
        for i in 0..values.len() {
            values[i] = self.values[i] + other.values[i]
        }
        return Self { values };
    }
}
// For any integer from 1 to 10.

// In order to get around the bugs I had for the previous implementation
// 1. the type of const parameters must not depend on other generic parameters
//    the type must not depend on the parameter `DIM` [E0770]
// 2. `[usize; DIM]` is forbidden as the type of a const generic parameter
//    the only supported types are integers, `bool`, and `char`
// struct FixedTensor<const DIM: usize, const DIMS: [usize; DIM]> {
//     values: [flt; dims_size(&DIMS)],
// }
//
// impl<const DIM: usize, const DIMS: [usize; DIM]> FixedTensor<DIM, DIMS> {
//     const fn size() -> usize {
//         dims_size(&DIMS)
//     }
//     fn zeros() -> Self {
//         Self {
//             values: [0.0; dims_size(&DIMS)],
//         }
//     }
//
//     fn random() -> Self {
//         let mut values: [flt; dims_size(&DIMS)];
//         for i in 0..values.len() {
//             values[i] = random_range(-1.0..1.0)
//         }
//         return Self { values };
//     }
//
//     fn add(&self, other: &Self) -> Self {
//         let mut values: [flt; dims_size(&DIMS)];
//         for i in 0..values.len() {
//             values[i] = self.values[i] + other.values[i]
//         }
//         return Self { values };
//     }
// }
