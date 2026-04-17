
include("tests/$TEST_BACKEND/access_fences.jl")
include("tests/$TEST_BACKEND/vectorization_test.jl")

include("tests/shfl.jl")
include("tests/vectorization_custom_test.jl")

# Memory ordering tests: Enable by launching julia with 'TEST_MEMORY_ORDERING=true julia'
if get(ENV, "TEST_MEMORY_ORDERING", "false") == "true"
    if Base.JLOptions().check_bounds != 0
        @warn """
            Bounds checking not set to 'auto' (current value: $(Base.JLOptions().check_bounds))
                     Memory ordering tests may show inaccurate results and are skipped
                     Run with `TEST_MEMORY_ORDERING=true julia --project -e 'using Pkg; Pkg.test(julia_args=[\"--check-bounds=auto\"])'`
            """
    else
        include("tests/memory_ordering.jl")
    end
end
