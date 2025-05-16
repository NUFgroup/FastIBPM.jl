using FastIBPM: DeltaYang3S
using Test
import CUDA, AMDGPU

include("tests.jl")

arrays = [Array]
CUDA.functional() && push!(arrays, CUDA.CuArray)
AMDGPU.functional() && push!(arrays, AMDGPU.ROCArray)

@info "Functional backends: $arrays"

@testset "FastIBPM.jl" verbose = true begin
    @testset "utils" Tests.test_utils()
    @testset "problems" Tests.test_problems()
    @testset "operators" begin
        @testset "$δ" for δ in (DeltaYang3S(),)
            Tests.test_delta_func(δ)
        end
    end
    @testset "operators $(nameof(array))" for array in arrays
        @testset "@loop" Tests.test_loop(array)
        @testset "FFT R2R" Tests.test_fft_r2r(array)
        @testset "$(replace(string(nameof(test)), r"^test_" => ""))" for test in [
            Tests.test_nonlinear
            Tests.test_rot
            Tests.test_curl
            Tests.test_laplacian_inv
            Tests.test_multidomain_coarsen
            Tests.test_multidomain_interpolate
            Tests.test_set_boundary
            Tests.test_multidomain_poisson
            Tests.test_regularization
            Tests.test_cnab
            Tests.test_cnab_io
        ]
            @testset "$(nd)D" for nd in (2, 3)
                test(array, Val(nd))
            end
        end
    end
end
