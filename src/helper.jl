"""
    _llvm_barrier()

Insert an LLVM optimization barrier that prevents memory operation reordering.

This function inserts a compiler-level memory clobber hint that prevents LLVM from
reordering or merging load/store operations across it. It generates no actual machine
instructions and has zero runtime cost—it only affects compiler optimizations.

Useful for preventing aggressive load merging that can break vectorization patterns
or for maintaining separation between independent memory operations.
"""
@inline function _llvm_barrier()
    Base.llvmcall("""
        call void asm sideeffect "", "~{memory}"()
        ret void
        """, Nothing, Tuple{})
end