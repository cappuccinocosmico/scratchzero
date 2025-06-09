type DimSizeType = usize;
type flt = f32;

const fn dims_product(dims: &[usize]) -> usize {
    let mut res = 0;
    for i in dims {
        res = res + i;
    }
    res
}

struct FixedTensor<const DIM: usize, const DIMS: [usize; DIM]> {
    values: [flt; dims_product(&DIMS)],
}

impl<const DIM: usize, const DIMS: [usize; DIM]> FixedTensor<DIM, DIMS> {
    const fn zeros() -> Box<Self> {
        Box::new(Self {
            values: [0.0; dims_product(&DIMS)],
        })
    }
}
