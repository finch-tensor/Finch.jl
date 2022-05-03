using Finch.IndexNotation
using Finch.IndexNotation: call_instance, assign_instance, access_instance, value_instance, name_instance, loop_instance, with_instance, label_instance, walk_instance

@testset "Parse" begin
    @test @index_program_instance(:f(:B[i::walk, k] * :C[k, j]^3, 42)) ==
        call_instance(
            value_instance(:f),
            call_instance(
                label_instance(:*, value_instance(*)),
                access_instance(
                    value_instance(:B),
                    Read(),
                    walk_instance(:i),
                    name_instance(:k)),
                call_instance(
                    label_instance(:^, value_instance(^)),
                    access_instance(
                        value_instance(:C),
                        Read(),
                        name_instance(:k),
                        name_instance(:j)),
                    value_instance(3))),
            value_instance(42))

    @test Finch.virtualize(:ex, typeof(@index_program_instance(:f(:B[i::walk, k] * :C[k, j]^3, 42))), Finch.LowerJulia()) ==
        call(:f, 
            call(*,
                access(:B, Read(), Walk(:i), Name(:k)),
                call(^,
                    access(:C, Read(), Name(:k), Name(:j)),
                    3)),
            42) 

    @test Finch.virtualize(:ex, typeof(@index_program_instance((:A[] = 1; :B[] = 2))), Finch.LowerJulia()) ==
        multi(
            assign(
                access(:A, Read()), 1),
                assign(
                    access(:B, Read()),
                    2))

    @test @index_program(@loop i :A[i] += :B[i] * i) ==
        loop(Name(:i),
            assign(
                access(:A,Update(), Name(:i)),
                +,
                call(*,
                    access(:B, Read(), Name(:i)),
                    Name(:i))))

    @test @index_program_instance(@loop i :A[i] += :B[i] * i) ==
        loop_instance(
            name_instance(:i),
            assign_instance(
                access_instance(
                    value_instance(:A),
                    Update(),
                    name_instance(:i)),
                label_instance(:+, value_instance(+)),
                call_instance(
                    label_instance(:*, value_instance(*)),
                    access_instance(
                        value_instance(:B),
                        Read(),
                        name_instance(:i)),
                    name_instance(:i))))

    @test @index_program(@loop i :A[i] <<(+)>>= :B[i] * i) ==
        loop(Name(:i),
            assign(
                access(:A,Update(), Name(:i)),
                +,
                call(*,
                    access(:B, Read(), Name(:i)),
                    Name(:i))))

    @test @index_program_instance(@loop i :A[i] <<(+)>>= :B[i] * i) ==
        loop_instance(
            name_instance(:i),
            assign_instance(
                access_instance(
                    value_instance(:A),
                    Update(),
                    name_instance(:i)),
                label_instance(:+, value_instance(+)),
                call_instance(
                    label_instance(:*, value_instance(*)),
                    access_instance(
                        value_instance(:B),
                        Read(),
                        name_instance(:i)),
                    name_instance(:i))))

    @test @index_program(:A[i] += i < j < k) ==
        assign(
            access(:A, Update(), Name(:i)),
            +,
            call(&,
                call(<, Name(:i), Name(:j)),
                call(<, Name(:j), Name(:k))))

    @test @index_program(:A[i] = i == j && k < l) ==
        assign(
            access(:A, Write(), Name(:i)),
            nothing,
            call(&,
                call(==, Name(:i), Name(:j)),
                call(<, Name(:k), Name(:l))))

    #call(:f, call(*, access_instance(:B, Read(), Name(:i), Name(:k)), call(^, access_instance(:C, Read(), Name(:k), Name(:j)), 3)), 42)
    #call(:f, call(*, access(:B, Read(), Name(:i), Name(:k)), call(^, access(:C, Read(), Name(:k), Name(:j)), 3)), 42)

    #@test @i(
    #    @loop i (
    #        @loop j :A[i, j] += :w[j]
    #    ) where (
    #        @loop j k :w[j] += :B[i, k] * :C[k, j]
    #    )
    #) ==
    #loop(Name(:i), with(loop(Name(:j), assign(access(:A, Update(), Name(:i), Name(:j)), +, access(:w, Read(), Name(:j)))), loop(Name(:j), Name(:k), assign(access(:w, Update(), Name(:j)), +, call(*, access(:B, Read(), Name(:i), Name(:k)), access(:C, Read(), Name(:k), Name(:j)))))))
end