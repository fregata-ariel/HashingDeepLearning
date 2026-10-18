#pragma once
/** Contiguous CPU tensor ABI guard. Reject before raw-pointer access/writes.
 * Owner must outlive every borrowed view and derived alias. Direct Python
 * getter results retain the owner through pybind keep_alive; clone for escape.
 * TRACE_TEST_ID: MAGICPIG-TORCH-CPU.
 */
inline void magicpig_tensor(const torch::Tensor& tensor, torch::ScalarType dtype) {
    TORCH_CHECK(tensor.is_cpu() && tensor.is_contiguous(), "requires contiguous CPU tensor");
    TORCH_CHECK(tensor.scalar_type() == dtype, "unexpected tensor dtype");
}
