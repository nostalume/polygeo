//! Generic owned `NumPy` allocation and checked native-index transfer.

use numpy::ndarray::Dimension;
use numpy::{Element, PyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArrayDyn};
use polygeo_core::topology::TopologyError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::topology::topology_error;

fn filled_array<T, D>(
    array: Bound<'_, PyArray<T, D>>,
    fill: impl FnOnce(&mut [T]) -> Result<(), TopologyError>,
) -> PyResult<Py<PyAny>>
where
    T: Element,
    D: Dimension,
{
    let mut writable = array
        .try_readwrite()
        .map_err(|_| topology_error(TopologyError::InternalInvariant))?;
    let output = writable
        .as_slice_mut()
        .map_err(|_| topology_error(TopologyError::InternalInvariant))?;
    fill(output).map_err(topology_error)?;
    drop(writable);
    Ok(array.unbind().into_any())
}

pub(crate) fn filled_array_1d<T>(
    py: Python<'_>,
    length: usize,
    fill: impl FnOnce(&mut [T]) -> Result<(), TopologyError>,
) -> PyResult<Py<PyAny>>
where
    T: Element,
{
    filled_array(PyArray1::<T>::zeros(py, length, false), fill)
}

pub(crate) fn filled_array_2d<T>(
    py: Python<'_>,
    rows: usize,
    columns: usize,
    fill: impl FnOnce(&mut [T]) -> Result<(), TopologyError>,
) -> PyResult<Py<PyAny>>
where
    T: Element,
{
    filled_array(PyArray2::<T>::zeros(py, [rows, columns], false), fill)
}

pub(crate) fn fill_indices<T>(
    values: impl IntoIterator<Item = usize>,
    output: &mut [T],
) -> Result<(), TopologyError>
where
    T: TryFrom<usize>,
{
    for (target, value) in output.iter_mut().zip(values) {
        *target = T::try_from(value).map_err(|_| TopologyError::IndexOverflow)?;
    }
    Ok(())
}

pub(crate) fn copy_indices<T>(
    value: &Bound<'_, PyAny>,
    mut convert: impl FnMut(T) -> Result<usize, TopologyError>,
) -> PyResult<Vec<usize>>
where
    T: Element + Copy,
{
    let array = value.extract::<PyReadonlyArrayDyn<'_, T>>()?;
    let view = array.as_array();
    let mut output = Vec::new();
    output
        .try_reserve_exact(view.len())
        .map_err(|_| topology_error(TopologyError::Allocation))?;
    for value in view.iter().copied() {
        output.push(convert(value).map_err(topology_error)?);
    }
    Ok(output)
}
