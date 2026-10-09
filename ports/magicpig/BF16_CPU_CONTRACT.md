# BF16 CPU boundary validation (#58)

The `bf16_model` suite executes the unchanged selected `wv_kernel` using software
FP32 lanes and explicit conversion substitutes. The independently gated native
`bf16_tile` suite executes the unchanged `qk_kernel_bf16_impl`. Both are small
deterministic CPU fixtures; GPU experiments remain postponed.

Run `python3 tools/check_magicpig_cpu.py --suite bf16_model` for the portable
model, or add `--native` to request the separately gated native BF16 fixture.
Native BF16 requires confirmed AVX512F/BW/BF16 and GCC >=11 before compiling and
starting the SIMD executable. It does not require FMA. An ineligible CPU is an
explicit skip, rather than numerical success. The default driver enrolls the
portable fixture; native execution remains opt-in.

## Conversion policy and independent references

The pinned `library/sparse_attention/3rdparty/FBGEMM/src/` converter
`FbgemmBfloat16ConvertAvx512.cc` implements `(raw_uint32_bits + 0x8000) >> 16`.
For finite midpoint inputs this rounds away from zero, rather than ties to
even. The fixture labels this as a software model: the actual FBGEMM converter,
dispatch and Torch extension do not execute. Each selected WV call converts
exactly 16 values per block, so the archived converter's AVX2 remainder path is
outside this model's claim.

Twenty-two literal hexadecimal expectations cover both signed zeros, values
around positive/negative midpoints, normal/subnormal boundaries, maximum
finite values, infinities and selected NaN payloads. The historical raw-bit
operation can turn a small NaN payload into infinity; a maximal negative NaN
bit pattern wraps to positive zero. Those are recorded source behavior, not a
recommended NaN-conversion policy. Eight independent arithmetic decoding cases
also verify the placement of BF16 storage bits.

The finite output-rounding oracle derives the representable step from the
binary exponent, uses an arithmetic half-step rule and then extracts the
already-rounded value. It does not reuse the converter's raw-bit addition.
FP64 weighted sums decode quantized values with sign/exponent/mantissa
arithmetic, independently of the adapter's bit placement.

## Weighted-value model

Twenty-four combinations cross dimensions 16,32,128 and nnz
0,1,2,15,16,17,31,32; 1,408 output coordinates are checked. Output canaries guard
the exact supported head dimension. The adapter observes the FP32 accumulator
before output conversion. Its source-order FMA result must agree bitwise with
an independent scalar same-order FMA computation and remain within
`(nnz+1)*float_epsilon*sum_abs + 1e-12` of the FP64 sum on the identical quantized
inputs. Output BF16 bits must match the separate arithmetic rounding oracle.

Input quantization, FP32 accumulation and final BF16 rounding are different
effects. Their checks do not attribute any error to the LSH sampling estimator.
For the fixed ordinary inputs, maximum input-quantization contribution is about
0.000113079 and maximum FP32 accumulation error is about 1.11961e-8. Separate
`2^24,+1,-2^24` and exact midpoint witnesses demonstrate cancellation and the
historical tie rule. These selected-body tests use supplied weights and do not
execute the probability correction or Softmax path.

## Actual native BF16 QK

The expanded native fixture retains `MAGICPIG-BF16-SELECTED-CPU` and adds
`MAGICPIG-BF16-BOUNDARY-CPU`. Eighteen supported cases cross dimensions 32,64,128,
nnz 0,16,32 and exactly representable/nontrivial input families. Twenty-four
additional partial-tile cases use nnz 1,15,17,31 with initialized physical ID
and score capacity rounded up to 16. There are 864 score checks across both
groups, using FP64 dots of the same quantized inputs and the bound
`(head_dim+2)*float_epsilon*(sum_abs+0.75) + 1e-6`.

The source computes complete 16-row tiles even for a partial logical nnz and
zeroes only logical score entries. The padded fixture explicitly checks that
preexisting 0.75 padding receives the extra dot products; guard entries outside
physical capacity remain unchanged. This reproduces the current padded-buffer
contract and does not establish support for arrays allocated at exact logical
nnz. Empty QK uses valid storage and performs no score writes; it says nothing
about empty Softmax, whose domain is tested separately.

Three safe all-ones cases at dimensions 16,48,80 reproduce the source's
`floor(head_dim/32)*32` result, differing from the complete dot product. These
are unsupported-dimension defect characterizations. The ordinary native inputs
show maximum accumulator error about 7.45058e-8 and input-quantization
contribution about 0.000169936. Neither padded tails nor silent dimension
truncation is reported as unrestricted QK correctness.

The plain and local ASan/UBSan standalone runs pass both suites. Local process
inspection restrictions require the explicit leak-check disable option; hosted
CI must keep LeakSanitizer enabled. Full Torch/FBGEMM linkage, native conversion
execution, BF16 server dispatch and model-level approximation quality remain
outside these fixtures.
