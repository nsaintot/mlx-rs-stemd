use std::ffi::CStr;

use mlx_internal_macros::{default_device, generate_macro};

use crate::{
    error::{Exception, Result},
    utils::{guard::Guarded, VectorArray},
    Array, Stream,
};

/// mlx-c 0.6.0 made `group_size` and `bits` optional and added a `mode` string.
/// "affine" is MLX's default and the original scheme.
const AFFINE_MODE: &CStr = c"affine";

fn opt_int(value: i32) -> mlx_sys::mlx_optional_int {
    mlx_sys::mlx_optional_int {
        value,
        has_value: true,
    }
}

/// The absent value for mlx-c's nullable array parameters.
fn null_array() -> mlx_sys::mlx_array {
    mlx_sys::mlx_array {
        ctx: std::ptr::null_mut(),
    }
}

fn no_dtype() -> mlx_sys::mlx_optional_dtype {
    mlx_sys::mlx_optional_dtype {
        value: 0,
        has_value: false,
    }
}

/// Quantize the matrix `w` using `bits` bits per element.
///
/// Note, every `group_size` elements in a row of `w` are quantized together. Hence, number of
/// columns of `w` should be divisible by `group_size`. In particular, the rows of `w` are divided
/// into groups of size `group_size` which are quantized together.
///
/// > `quantized` currently only supports 2D inputs with dimensions which are multiples of 32
///
/// For details, please see [this
/// documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html)
///
/// # Params
///
/// - `w`: The input matrix
/// - `group_size`: The size of the group in `w` that shares a scale and bias. (default: `64`)
/// - `bits`: The number of bits occupied by each element of w in the returned quantized matrix.
///   (default: 4)
#[generate_macro]
#[default_device]
pub fn quantize_device(
    w: impl AsRef<Array>,
    #[optional] group_size: impl Into<Option<i32>>,
    #[optional] bits: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<(Array, Array, Array)> {
    let group_size = group_size.into().unwrap_or(64);
    let bits = bits.into().unwrap_or(4);

    // mlx-c 0.6.0 returns one vector rather than three out-parameters, since a
    // mode may produce a different count. "affine" produces three: packed
    // weights, scales, biases.
    let packed = VectorArray::try_from_op(|res| unsafe {
        mlx_sys::mlx_quantize(
            res,
            w.as_ref().as_ptr(),
            opt_int(group_size),
            opt_int(bits),
            AFFINE_MODE.as_ptr(),
            null_array(),
            stream.as_ref().as_ptr(),
        )
    })?;

    let mut values: Vec<Array> = packed.try_into_values()?;
    if values.len() != 3 {
        return Err(Exception::custom(format!(
            "mlx_quantize returned {} arrays, expected 3",
            values.len()
        )));
    }
    let biases = values.pop().unwrap();
    let scales = values.pop().unwrap();
    let wq = values.pop().unwrap();
    Ok((wq, scales, biases))
}

/// Perform the matrix multiplication with the quantized matrix `w`. The quantization uses one
/// floating point scale and bias per `group_size` of elements. Each element in `w` takes `bits`
/// bits and is packed in an unsigned 32 bit integer.
#[allow(clippy::too_many_arguments)]
#[generate_macro]
#[default_device]
pub fn quantized_matmul_device(
    x: impl AsRef<Array>,
    w: impl AsRef<Array>,
    scales: impl AsRef<Array>,
    biases: impl AsRef<Array>,
    #[optional] transpose: impl Into<Option<bool>>,
    #[optional] group_size: impl Into<Option<i32>>,
    #[optional] bits: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let transpose = transpose.into().unwrap_or(false);
    let group_size = group_size.into().unwrap_or(64);
    let bits = bits.into().unwrap_or(4);

    <Array as Guarded>::try_from_op(|res| unsafe {
        mlx_sys::mlx_quantized_matmul(
            res,
            x.as_ref().as_ptr(),
            w.as_ref().as_ptr(),
            scales.as_ref().as_ptr(),
            biases.as_ref().as_ptr(),
            transpose,
            opt_int(group_size),
            opt_int(bits),
            AFFINE_MODE.as_ptr(),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Dequantize the matrix `w` using the provided `scales` and `biases` and the `group_size` and
/// `bits` configuration.
///
/// For details, please see [this
/// documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.dequantize.html)
#[generate_macro]
#[default_device]
pub fn dequantize_device(
    w: impl AsRef<Array>,
    scales: impl AsRef<Array>,
    biases: impl AsRef<Array>,
    #[optional] group_size: impl Into<Option<i32>>,
    #[optional] bits: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let group_size = group_size.into().unwrap_or(64);
    let bits = bits.into().unwrap_or(4);

    <Array as Guarded>::try_from_op(|res| unsafe {
        mlx_sys::mlx_dequantize(
            res,
            w.as_ref().as_ptr(),
            scales.as_ref().as_ptr(),
            biases.as_ref().as_ptr(),
            opt_int(group_size),
            opt_int(bits),
            AFFINE_MODE.as_ptr(),
            null_array(),
            no_dtype(),
            stream.as_ref().as_ptr(),
        )
    })
}

#[cfg(test)]
mod tests {
    use crate::{
        ops::{dequantize, expand_dims, quantize},
        Array,
    };

    #[test]
    fn test_quantize_dequantize() {
        let x1 = Array::ones::<f32>(&[128, 1]).unwrap();
        let x2 = expand_dims(Array::arange::<_, f32>(0, 512, None).unwrap(), 0).unwrap();
        let x = x1 * x2;

        for i in [2, 4, 8].iter() {
            let el_per_int = 32 / i;
            let (x_q, scales, biases) = quantize(&x, 128, *i).unwrap();
            assert_eq!(x_q.shape(), [128, 512 / el_per_int]);
            assert_eq!(scales.shape(), [128, 4]);
            assert_eq!(biases.shape(), [128, 4]);

            let x_hat = dequantize(&x_q, &scales, &biases, 128, *i).unwrap();
            let max_diff = ((&x - &x_hat).abs().unwrap().max(None).unwrap()).item::<f32>();
            assert!(max_diff <= 127.0 / (1 << i) as f32);
        }
    }
}
