// Rust 1.74 predates array::each_ref/each_mut. These safe adapters borrow
// elements in index order without cloning, moving or allocating their contents.
trait ArrayBorrows<T, const N: usize> {
    fn array_refs(&self) -> [&T; N];
    fn array_muts(&mut self) -> [&mut T; N];
}

impl<T, const N: usize> ArrayBorrows<T, N> for [T; N] {
    fn array_refs(&self) -> [&T; N] {
        std::array::from_fn(|index| &self[index])
    }

    fn array_muts(&mut self) -> [&mut T; N] {
        let mut elements = self.iter_mut();
        std::array::from_fn(|_| elements.next().expect("array iterator has N elements"))
    }
}

