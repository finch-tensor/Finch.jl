@testitem "continuous" setup = [CheckOutput] skip = true begin
    include("../data/continuous_data.jl")
    let
        s1 = Scalar(0)
        x = Tensor(SparseRunList{Limit{Float32}}(Element(0)), 10)
        @finch mode = :fast (x[3] = 1)
        @finch mode = :fast (
            for i in realextent(5 + Eps, 7 - Eps)
                x[~i] = 1
            end
        )
        @finch mode = :fast (
            for i in realextent(8, 9 + Eps)
                x[~i] = 1
            end
        )

        @finch mode = :fast (
            for i in _
                s1[] += x[i] * d(i)
            end
        )
        @test s1.val == 3
    end

    let
        s1 = Scalar(0)
        x = Tensor(SparseRunList{Limit{Float32}}(SparseList(Element(0))), 10, 10)
        a = [1, 4, 8]
        @finch mode = :fast (
            for i in realextent(2, 4 - Eps)
                for j in extent(1, 3)
                    x[a[j], ~i] = 1
                end
            end
        )
        @finch mode = :fast (
            for i in realextent(6 + Eps, 10 - Eps)
                x[2, ~i] = 1
            end
        )

        @finch mode = :fast (
            for i in _
                for j in _
                    s1[] += x[j, i] * d(i)
                end
            end
        )
        @test s1.val == 10
    end

    using StatsBase
    using Random
    using StableRNGs

    Shape = 100 #1000
    NumItvl = 10 #100

    # Generating Random Intervals
    #v = sort(sample(StableRNG(1234), 1:Shape, NumItvl*2, replace=false))
    v = [2, 6, 9, 16, 26, 30, 37, 40, 41, 56, 68, 69, 70, 78, 86, 87, 88, 89, 92, 96]
    #w = sort(sample(StableRNG(4321), 1:Shape, NumItvl*2, replace=false))
    w = [2, 3, 6, 8, 12, 18, 22, 27, 29, 40, 42, 45, 47, 67, 73, 79, 83, 84, 87, 98]

    s1, e1 = v[begin:2:end], v[(begin + 1):2:end]
    s2, e2 = w[begin:2:end], w[(begin + 1):2:end]

    let
        x = Tensor(
            SparseList{Float32}(
                Element{0}(fill(1, 2 * NumItvl)), Shape, [1, 2 * NumItvl + 1], v
            ),
        )
        y = Tensor(
            SparseRunList{Float32}(
                Element{0}(fill(1, 2 * NumItvl)), Shape, [1, 2 * NumItvl + 1], v, v
            ),
        )
        z1 = Scalar(0)
        z2 = Scalar(0)

        io = IOBuffer()
        @repl io @finch_code (z1 .= 0;
        for i in _
            z1[] += x[i]
        end)
        @repl io @finch (z1 .= 0;
        for i in _
            z1[] += x[i]
        end)
        @test check_output("continuous/pinpoint_sl.txt", String(take!(io)))

        @repl io @finch_code (z2 .= 0;
        for i in _
            z2[] += y[i]
        end)
        @repl io @finch (z2 .= 0;
        for i in _
            z2[] += y[i]
        end)
        @test check_output("continuous/pinpoint_rle.txt", String(take!(io)))

        @test z1.val == z2.val

        @finch (z1 .= 0;
        for i in _
            z1[] += x[i] * d(i)
        end)
        @finch (z2 .= 0;
        for i in _
            z2[] += y[i] * d(i)
        end)
        @test z1.val == 0 && z2.val == 0
    end

    let
        x = Tensor(
            SparseList{Float32}(
                Element{0}(fill(1, 2 * NumItvl)), Shape, [1, 2 * NumItvl + 1], v
            ),
        )
        y = Tensor(
            SparseRunList{Float32}(
                Element{0}(fill(1, 2 * NumItvl)), Shape, [1, 2 * NumItvl + 1], v, v
            ),
        )
        z1 = Scalar(0)
        z2 = Scalar(0)

        io = IOBuffer()
        @repl io @finch_code (z1 .= 0;
        for i in _
            z1[] += x[2 * i + 10]
        end)
        @repl io @finch (z1 .= 0;
        for i in _
            z1[] += x[2 * i + 10]
        end)
        @test check_output("continuous/affine_sl.txt", String(take!(io)))

        @repl io @finch_code (z1 .= 0;
        for i in _
            z2[] += y[2 * i + 10]
        end)
        @repl io @finch (z1 .= 0;
        for i in _
            z2[] += y[2 * i + 10]
        end)
        @test check_output("continuous/affine_rle.txt", String(take!(io)))

        @test z1.val == z2.val

        @finch (z1 .= 0;
        for i in realextent(3, 15)
            z1[] += coalesce(x[~(2 * i + 10)], 0)
        end)
        @finch (z2 .= 0;
        for i in realextent(3, 15)
            z2[] += coalesce(y[~(2 * i + 10)], 0)
        end)
        @test z1.val == 5 && z2.val == 5
    end

    let
        x = Tensor(
            SparseRunList{Float32}(
                Element{0}(fill(1, NumItvl)), Shape, [1, NumItvl + 1], s1, e1
            ),
        )
        y = Tensor(
            SparseRunList{Float32}(
                Element{0}(fill(1, NumItvl)), Shape, [1, NumItvl + 1], s2, e2
            ),
        )
        z = Tensor(SparseRunList{Limit{Float32}}(Element{0}()))
        s = Scalar(0)

        io = IOBuffer()
        @repl io @finch_code (z .= 0;
        for i in _
            z[i] += x[i] * y[i]
        end)
        @repl io @finch (z .= 0;
        for i in _
            z[i] += x[i] * y[i]
        end)
        @test check_output("continuous/intersect.txt", String(take!(io)))

        @repl io @finch_code (z .= 0;
        for i in _
            z[i] += x[i] + y[i]
        end)
        @repl io @finch (z .= 0;
        for i in _
            z[i] += x[i] + y[i]
        end)
        @test check_output("continuous/union.txt", String(take!(io)))

        @repl io @finch_code (s .= 0;
        for i in _
            s[] += (x[i] * y[i])
        end)
        @repl io @finch (s .= 0;
        for i in _
            s[] += (x[i] * y[i])
        end)
        @test check_output("continuous/intersect_counting.txt", String(take!(io)))

        @repl io @finch_code (s .= 0;
        for i in _
            s[] += (x[i] * y[i]) * d(i)
        end)
        @repl io @finch (s .= 0;
        for i in _
            s[] += (x[i] * y[i]) * d(i)
        end)
        @test check_output("continuous/intersect_lebesgue.txt", String(take!(io)))

        @repl io @finch_code (s .= 0;
        for i in _
            s[] += (x[i] + y[i])
        end)
        @repl io @finch (s .= 0;
        for i in _
            s[] += (x[i] + y[i])
        end)
        @test check_output("continuous/union_counting.txt", String(take!(io)))

        @repl io @finch_code (s .= 0;
        for i in _
            s[] += (x[i] + y[i]) * d(i)
        end)
        @repl io @finch (s .= 0;
        for i in _
            s[] += (x[i] + y[i]) * d(i)
        end)
        @test check_output("continuous/union_lebesgue.txt", String(take!(io)))
    end

    let
        colptr = [1 + NumItvl * i for i in 0:NumItvl]
        #endpoint1 = vcat([sort(sample(StableRNG(i), 1:Shape, NumItvl*2, replace=false)) for i in 1:NumItvl]...)
        #endpoint2 = vcat([sort(sample(StableRNG(i+NumItvl), 1:Shape, NumItvl*2, replace=false)) for i in 1:NumItvl]...)
        endpoint1 = [
            17,
            24,
            27,
            30,
            34,
            36,
            37,
            38,
            47,
            52,
            55,
            56,
            59,
            65,
            78,
            84,
            91,
            92,
            94,
            96,
            9,
            10,
            13,
            17,
            20,
            23,
            27,
            29,
            39,
            42,
            46,
            47,
            53,
            54,
            62,
            68,
            69,
            70,
            81,
            93,
            1,
            2,
            5,
            10,
            12,
            18,
            19,
            25,
            32,
            33,
            34,
            38,
            61,
            62,
            63,
            65,
            73,
            80,
            93,
            98,
            2,
            5,
            18,
            19,
            20,
            23,
            26,
            41,
            45,
            50,
            61,
            74,
            79,
            81,
            83,
            84,
            90,
            94,
            95,
            96,
            7,
            9,
            14,
            15,
            33,
            39,
            40,
            45,
            52,
            53,
            57,
            62,
            74,
            83,
            89,
            90,
            93,
            95,
            98,
            99,
            6,
            7,
            10,
            15,
            22,
            25,
            32,
            33,
            34,
            44,
            46,
            55,
            56,
            59,
            63,
            69,
            80,
            95,
            97,
            99,
            3,
            11,
            15,
            17,
            18,
            25,
            30,
            33,
            34,
            38,
            39,
            44,
            55,
            67,
            71,
            74,
            80,
            86,
            92,
            98,
            4,
            12,
            15,
            17,
            19,
            20,
            37,
            40,
            50,
            51,
            59,
            61,
            66,
            68,
            84,
            90,
            92,
            94,
            99,
            100,
            3,
            11,
            15,
            21,
            22,
            32,
            35,
            38,
            48,
            52,
            54,
            65,
            67,
            70,
            71,
            78,
            86,
            89,
            95,
            96,
            3,
            10,
            16,
            19,
            26,
            39,
            46,
            47,
            48,
            59,
            60,
            64,
            67,
            68,
            73,
            87,
            90,
            92,
            94,
            97,
        ]
        endpoint2 = [
            11,
            13,
            17,
            18,
            19,
            26,
            27,
            38,
            39,
            40,
            43,
            55,
            65,
            68,
            72,
            73,
            74,
            88,
            95,
            98,
            2,
            4,
            12,
            13,
            20,
            21,
            33,
            37,
            38,
            43,
            44,
            49,
            55,
            72,
            73,
            75,
            78,
            81,
            85,
            96,
            1,
            2,
            4,
            6,
            7,
            13,
            23,
            29,
            30,
            39,
            42,
            46,
            47,
            55,
            61,
            68,
            91,
            93,
            95,
            98,
            8,
            10,
            15,
            23,
            24,
            27,
            30,
            45,
            49,
            54,
            57,
            60,
            63,
            73,
            74,
            75,
            81,
            82,
            83,
            97,
            2,
            4,
            19,
            24,
            32,
            36,
            38,
            43,
            45,
            51,
            52,
            73,
            74,
            77,
            78,
            80,
            82,
            92,
            94,
            100,
            2,
            8,
            9,
            16,
            17,
            19,
            21,
            22,
            25,
            27,
            29,
            35,
            37,
            42,
            48,
            49,
            72,
            75,
            90,
            98,
            11,
            18,
            19,
            22,
            26,
            31,
            35,
            40,
            44,
            47,
            53,
            54,
            59,
            64,
            66,
            73,
            77,
            81,
            87,
            95,
            1,
            5,
            7,
            8,
            17,
            18,
            20,
            23,
            28,
            32,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            82,
            91,
            1,
            5,
            14,
            16,
            22,
            25,
            35,
            38,
            40,
            55,
            57,
            61,
            67,
            71,
            76,
            79,
            80,
            86,
            97,
            99,
            2,
            5,
            6,
            7,
            14,
            24,
            27,
            32,
            34,
            38,
            48,
            54,
            55,
            65,
            72,
            73,
            81,
            82,
            91,
            96,
        ]
        col_s1, col_e1 = endpoint1[begin:2:end], endpoint1[(begin + 1):2:end]
        col_s2, col_e2 = endpoint2[begin:2:end], endpoint2[(begin + 1):2:end]

        x = Tensor(
            SparseRunList{Float32}(
                SparseRunList{Float32}(
                    Element{0}(fill(1, NumItvl * NumItvl)), Shape, colptr, col_s1, col_e1
                ),
                Shape,
                [1, NumItvl + 1],
                s1,
                e1,
            ),
        )
        y = Tensor(
            SparseRunList{Float32}(
                SparseRunList{Float32}(
                    Element{0}(fill(1, NumItvl * NumItvl)), Shape, colptr, col_s2, col_e2
                ),
                Shape,
                [1, NumItvl + 1],
                s2,
                e2,
            ),
        )
        z = Tensor(
            SparseRunList{Limit{Float32}}(SparseRunList{Limit{Float32}}(Element{0}()))
        )
        s = Scalar(0)

        io = IOBuffer()
        @repl io @finch_code (z .= 0;
        for i in _, j in _
            z[j, i] += x[j, i] * y[j, i]
        end)
        @repl io @finch (z .= 0;
        for i in _, j in _
            z[j, i] += x[j, i] * y[j, i]
        end)
        @test check_output("continuous/2d_intersect.txt", String(take!(io)))

        @repl io @finch_code (z .= 0;
        for i in _, j in _
            z[j, i] += x[j, i] + y[j, i]
        end)
        @repl io @finch (z .= 0;
        for i in _, j in _
            z[j, i] += x[j, i] + y[j, i]
        end)
        @test check_output("continuous/2d_union.txt", String(take!(io)))

        @repl io @finch_code (s .= 0;
        for i in _, j in _
            s[] += (x[j, i] * y[j, i]) * d(i, j)
        end)
        @repl io @finch (s .= 0;
        for i in _, j in _
            s[] += (x[j, i] * y[j, i]) * d(i, j)
        end)
        @test check_output("continuous/2d_intersect_lebesgue.txt", String(take!(io)))

        @repl io @finch_code (s .= 0;
        for i in _, j in _
            s[] += (x[j, i] * y[j, i])
        end)
        @repl io @finch (s .= 0;
        for i in _, j in _
            s[] += (x[j, i] * y[j, i])
        end)
        @test check_output("continuous/2d_intersect_counting.txt", String(take!(io)))

        @repl io @finch_code (s .= 0;
        for i in _, j in _
            s[] += (x[j, i] + y[j, i])
        end)
        @repl io @finch (s .= 0;
        for i in _, j in _
            s[] += (x[j, i] + y[j, i])
        end)
        @test check_output("continuous/2d_union_counting.txt", String(take!(io)))

        @repl io @finch_code (s .= 0;
        for i in _, j in _
            s[] += (x[j, i] + y[j, i]) * d(i, j)
        end)
        @repl io @finch (s .= 0;
        for i in _, j in _
            s[] += (x[j, i] + y[j, i]) * d(i, j)
        end)
        @test check_output("continuous/2d_union_lebesgue.txt", String(take!(io)))

        sum1 = Scalar(0)
        sum2 = Scalar(0)
        @repl io @finch (sum1 .= 0;
        for i in _, j in _
            sum1[] += x[j, i] * d(i, j)
        end)
        @repl io @finch (sum2 .= 0;
        for i in _, j in _
            sum2[] += y[j, i] * d(i, j)
        end)
        @test s.val == sum1.val + sum2.val
    end

    let
        colptr = [1 + NumItvl * i for i in 0:NumItvl]
        #endpoint1 = vcat([sort(sample(StableRNG(i), 1:Shape, NumItvl*2, replace=false)) for i in 1:NumItvl]...)
        endpoint1 = [
            17,
            24,
            27,
            30,
            34,
            36,
            37,
            38,
            47,
            52,
            55,
            56,
            59,
            65,
            78,
            84,
            91,
            92,
            94,
            96,
            9,
            10,
            13,
            17,
            20,
            23,
            27,
            29,
            39,
            42,
            46,
            47,
            53,
            54,
            62,
            68,
            69,
            70,
            81,
            93,
            1,
            2,
            5,
            10,
            12,
            18,
            19,
            25,
            32,
            33,
            34,
            38,
            61,
            62,
            63,
            65,
            73,
            80,
            93,
            98,
            2,
            5,
            18,
            19,
            20,
            23,
            26,
            41,
            45,
            50,
            61,
            74,
            79,
            81,
            83,
            84,
            90,
            94,
            95,
            96,
            7,
            9,
            14,
            15,
            33,
            39,
            40,
            45,
            52,
            53,
            57,
            62,
            74,
            83,
            89,
            90,
            93,
            95,
            98,
            99,
            6,
            7,
            10,
            15,
            22,
            25,
            32,
            33,
            34,
            44,
            46,
            55,
            56,
            59,
            63,
            69,
            80,
            95,
            97,
            99,
            3,
            11,
            15,
            17,
            18,
            25,
            30,
            33,
            34,
            38,
            39,
            44,
            55,
            67,
            71,
            74,
            80,
            86,
            92,
            98,
            4,
            12,
            15,
            17,
            19,
            20,
            37,
            40,
            50,
            51,
            59,
            61,
            66,
            68,
            84,
            90,
            92,
            94,
            99,
            100,
            3,
            11,
            15,
            21,
            22,
            32,
            35,
            38,
            48,
            52,
            54,
            65,
            67,
            70,
            71,
            78,
            86,
            89,
            95,
            96,
            3,
            10,
            16,
            19,
            26,
            39,
            46,
            47,
            48,
            59,
            60,
            64,
            67,
            68,
            73,
            87,
            90,
            92,
            94,
            97,
        ]
        col_s1, col_e1 = endpoint1[begin:2:end], endpoint1[(begin + 1):2:end]

        x1 = Tensor(
            SparseList{Float32}(
                SparseRunList{Float32}(
                    Element{0}(fill(1, NumItvl * NumItvl)), Shape, colptr, col_s1, col_e1
                ),
                Shape,
                [1, NumItvl + 1],
                e1,
            ),
        )
        x2 = Tensor(
            SparseRunList{Float32}(
                SparseRunList{Float32}(
                    Element{0}(fill(1, NumItvl * NumItvl)), Shape, colptr, col_s1, col_e1
                ),
                Shape,
                [1, NumItvl + 1],
                e1,
                e1,
            ),
        )
        y = Tensor(
            SparseRunList{Float32}(
                Element{0}(fill(1, NumItvl)), Shape, [1, NumItvl + 1], s2, e2
            ),
        )

        s1 = Scalar(0)
        s2 = Scalar(0)

        io = IOBuffer()
        @repl io @finch_code (s1 .= 0;
        for i in _, j in _
            s1[] += x1[j, i] * y[j] * d(j)
        end)
        @repl io @finch (s1 .= 0;
        for i in _, j in _
            s1[] += x1[j, i] * y[j] * d(j)
        end)
        @test check_output("continuous/2d_itvl_sum_sl.txt", String(take!(io)))

        @repl io @finch_code (s2 .= 0;
        for i in _, j in _
            s2[] += x1[j, i] * y[j] * d(j)
        end)
        @repl io @finch (s2 .= 0;
        for i in _, j in _
            s2[] += x1[j, i] * y[j] * d(j)
        end)
        @test check_output("continuous/2d_itvl_sum_rle.txt", String(take!(io)))

        @test s1.val == s2.val
    end
end
