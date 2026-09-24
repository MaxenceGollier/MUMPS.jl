# this file contains a bunch of shortcuts for manipulating ICNTL.

# export set_error_stream!, set_diagnostics_stream!, set_info_stream!, set_print_level!,
# suppress_printing!, toggle_printing!, suppress_display!, toggle_display!,
# sparse_matrix!, dense_matrix!,
# sparse_rhs!, dense_rhs!,
# toggle_null_pivot!

const ICNTL_DEFAULT = (
  6,    # 1:  output stream for error messages
  0,    # 2:  output stream for diagnostics/statistics/warnings
  6,    # 3:  output stream for global info
  2,    # 4:  level of printing
  0,    # 5:  matrix input format (0 = assembled)
  7,    # 6:  permutation/scaling (7 = automatic)
  7,    # 7:  ordering (7 = automatic)
  77,   # 8:  scaling strategy (77 = automatic)
  1,    # 9:  solve Ax = b (otherwise Aᵀx = b)
  0,    # 10: iterative refinement steps
  0,    # 11: error analysis
  0,    # 12: ordering strategy for symmetric matrices
  0,    # 13: ScaLAPACK on root node
  20,   # 14: % increase of estimated working space
  0,    # 15: analysis by blocks
  0,    # 16: number of OpenMP threads (0 = unchanged)
  0,    # 17: MPI processes used as OpenMP resources (MUMPS ≥ 5.9)
  0,    # 18: distributed input matrix
  0,    # 19: Schur complement
  0,    # 20: rhs format (0 = dense, centralized)
  0,    # 21: solution distribution
  0,    # 22: out-of-core
  0,    # 23: max working memory per process
  0,    # 24: null pivot detection
  0,    # 25: null space basis
  0,    # 26: Schur solution phase
  -8,   # 27: blocking size for multiple rhs
  0,    # 28: sequential/parallel analysis (0 = automatic)
  0,    # 29: parallel ordering tool (0 = automatic)
  0,    # 30: entries of A⁻¹
  0,    # 31: discard factors
  0,    # 32: forward elimination during factorization
  0,    # 33: determinant
  0,    # 34: OOC files with save/restore
  0,    # 35: Block Low-Rank
  1,    # 36: BLR variant (1 = UCFS, default since 5.9)
  0,    # 37: BLR compression of contribution blocks
  600,  # 38: estimated compression rate of LU factors (per mille)
  500,  # 39: estimated compression rate of contribution blocks (per mille)
  0,    # 40: mixed/adaptive precision BLR (MUMPS ≥ 5.9)
  0,    # 41: not used
  0,    # 42: not used
  0,    # 43: not used
  0,    # 44: not used
  0,    # 45: not used
  0,    # 46: not used
  0,    # 47: single precision factorization in double instance (MUMPS ≥ 5.9)
  1,    # 48: tree parallelism (L0-threads)
  0,    # 49: compact workarray S
  0,    # 50: not used
  0,    # 51: GPU offload (MUMPS ≥ 5.9)
  0,    # 52: not used
  0,    # 53: not used
  0,    # 54: not used
  0,    # 55: not used
  0,    # 56: rank-revealing factorization
  0,    # 57: not used
  2,    # 58: symbolic factorization (2 = column counts)
  0,    # 59: not used
  0,    # 60: not used
)
"""
    default_icntl!(mumps)

reset ICNTL to its default
"""
function default_icntl!(mumps::Mumps)
  mumps.icntl = ICNTL_DEFAULT
  return nothing
end

set_error_stream!(mumps::Mumps, i) = begin
  set_icntl!(mumps, 1, i; displaylevel = 0)
  return mumps
end
set_diagnostics_stream!(mumps::Mumps, i) = begin
  set_icntl!(mumps, 2, i; displaylevel = 0)
  return mumps
end
set_info_stream!(mumps::Mumps, i) = begin
  set_icntl!(mumps, 3, i; displaylevel = 0)
  return mumps
end
set_print_level!(mumps::Mumps, i) = begin
  set_icntl!(mumps, 4, i; displaylevel = 0)
  return mumps
end

suppress_printing!(mumps::Mumps) = begin
  set_print_level!(mumps, 1)
  return mumps
end
toggle_printing!(mumps::Mumps) = begin
  set_print_level!(mumps, mod1(mumps.icntl[4] + 1, 2))
  return mumps
end
suppress_display! = suppress_printing!
toggle_display! = toggle_printing!

sparse_matrix!(mumps::Mumps) = begin
  set_icntl!(mumps, 5, 0; displaylevel = 0)
  return mumps
end
dense_matrix!(mumps::Mumps) = begin
  set_icntl!(mumps, 5, 1; displaylevel = 0)
  return mumps
end

LinearAlgebra.transpose!(mumps::Mumps) = begin
  set_icntl!(mumps, 9, mod(mumps.icntl[9] + 1, 2); displaylevel = 0)
  return mumps
end

sparse_rhs!(mumps::Mumps) = begin
  set_icntl!(mumps, 20, 1; displaylevel = 0)
  return mumps
end
dense_rhs!(mumps::Mumps) = begin
  set_icntl!(mumps, 20, 0; displaylevel = 0)
  return mumps
end

toggle_null_pivot!(mumps::Mumps) = begin
  set_icntl!(mumps, 24, mod(mumps.icntl[24] + 1, 2); displaylevel = 0)
  return mumps
end

set_num_threads!(mumps::Mumps, n::Integer) = begin
  set_icntl!(mumps, 16, n; displaylevel = 0)
  return mumps
end

toggle_tree_parallelism!(mumps::Mumps) = begin
  set_icntl!(mumps, 48, mod(mumps.icntl[48] + 1, 2); displaylevel = 0)
  return mumps
end

toggle_rank_revealing!(mumps::Mumps) = begin
  set_icntl!(mumps, 56, mod(mumps.icntl[56] + 1, 2); displaylevel = 0)
  return mumps
end

"""
    set_blr!(mumps, tol; mode=1)

Activate the Block Low-Rank factorization (ICNTL(35)=`mode`) with dropping parameter
CNTL(7)=`tol`. Use `mode=0` to deactivate it.
"""
function set_blr!(mumps::Mumps, tol::AbstractFloat; mode::Integer = 1)
  0 ≤ mode ≤ 3 || throw(MUMPSException("ICNTL(35) must be 0, 1, 2 or 3, got $mode"))
  set_icntl!(mumps, 35, mode; displaylevel = 0)
  set_cntl!(mumps, 7, tol; displaylevel = 0)
  return mumps
end

"""
    single_precision_factorization!(mumps, flag=true)

Perform the factorization in single precision within a double precision instance
(ICNTL(47), MUMPS ≥ 5.9). Iterative refinement (ICNTL(10)) is recommended to recover
double precision accuracy.
"""
function single_precision_factorization!(mumps::Mumps{TC, TR}, flag::Bool = true) where {TC, TR}
  TR == Float64 ||
    throw(MUMPSException("ICNTL(47) requires a Float64 or ComplexF64 instance, got $TC"))
  set_icntl!(mumps, 47, flag ? 1 : 0; displaylevel = 0)
  return mumps
end
