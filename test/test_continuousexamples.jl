@testset "continuous examples" begin
    @info "Testing Continuous Finch Examples"

    @testset "openclosed + openclosed" begin
        A_left = [1.1, 6.6, 9.9]
        A_right = [3.3, 7.7, 11.11]
        A_val = [1, 2, 3]
        A = Tensor(SparseRunList{Float64}(Element(0, A_val), 12.0, [1, 4], Finch.PlusEpsVector(A_left), A_right, Element(0, Int[])))
        B_left = [2.2, 5.5, 8.8]
        B_right = [3.3, 8.8, 9.9]
        B_val = [1, 1, 2]
        B = Tensor(SparseRunList{Float64}(Element(0, B_val), 12.0, [1, 4], Finch.PlusEpsVector(B_left), B_right, Element(0, Int[])))
        C = Tensor(SparseRunList{Float64}(Element(0), 12.0, [1], Finch.PlusEpsVector(Float64[]), Float64[], Element(0, Int[])))
        @finch begin
            C .= 0
            for i = _
                C[i] += A[i] + B[i]
            end
        end
        C_ref = Tensor(SparseRunList{Float64}(Element(0, [1, 2, 1, 3, 1, 2, 3]), 12.0, [1, 8], Finch.PlusEpsVector([1.1, 2.2, 5.5, 6.6, 7.7, 8.8, 9.9]), [2.2, 3.3, 6.6, 7.7, 8.8, 9.9, 11.11], Element(0, Int[])))
        @test Structure(C) == Structure(C_ref)
    end

    @testset "2D Box Search" begin
        # Load 2d Points
        point = [Pinpoints2D[i,:] .+ 1e7 for i in 1:size(Pinpoints2D,1)]
        point = sort(point)
        # Load 2d Box Query
        query = [QueryBox2D[i,:] .+ 1e7 for i in 1:size(QueryBox2D,1)]
        answer = [49, 11, 21, 18, 22, 18, 7, 95, 0, 19]
        radanswer = [73, 13, 27, 25, 37, 39, 11, 140, 7, 26]

        # Setting up Point Fiber
        shape = Float64(2e7)
        shape_id = length(point)
        point_x = (xy->xy[1]).(point)
        point_y = (xy->xy[2]).(point)
        point_id = collect(Int64, 1:length(point_y))
        point_ptr_x = [1,length(point_x)+1]
        point_ptr_y = collect(Int64, 1:length(point_y)+1)
        points = Tensor(SparseList{Float64}(SparseList{Float64}(SparsePoint{Int64}(Pattern(),
                                                              shape_id,
                                                              point_id),
                                          shape,
                                          point_ptr_y,
                                          point_y),
                      shape,
                      point_ptr_x,
                      point_x))

        # Setting up Box Fiber
        box_x_start = [query[1][1]]
        box_y_start = [query[1][2]]
        box_x_stop = [query[1][3]]
        box_y_stop = [query[1][4]]
        box = Tensor(SparseInterval{Float64}(SparseInterval{Float64}(Pattern(),
                                          shape,
                                          box_y_start,
                                          box_y_stop),
                       shape,
                       box_x_start,
                       box_x_stop))

        output = Tensor(SparseByteMap{Int64}(Pattern(), shape_id))

        def = @finch_kernel mode=:fast function rangequery(output, box, points)
            output .= false
            for x=_, y=_, id=_
                output[id] |= box[y,x] && points[id,y,x]
            end
        end

        radius=ox=oy=0.0 #placeholder
        def2 = @finch_kernel mode=:fast function radiusquery(output, points, radius, ox, oy)
            output .= false
            for x=realextent(ox-radius,ox+radius), y=realextent(oy-radius,oy+radius)
                if (x-ox)^2 + (y-oy)^2 <= radius^2
                    for id=_
                        output[id] |= coalesce(points[id,~y,~x], false)
                    end
                end
            end
        end

        eval(def)
        eval(def2)


        for i=1:length(query)
            box_x_start = [query[i][1]]
            box_y_start = [query[i][2]]
            box_x_stop = [query[i][3]]
            box_y_stop = [query[i][4]]
            box = Tensor(SparseInterval{Float64}(SparseInterval{Float64}(Pattern(),
                                       shape,
                                       box_y_start,
                                       box_y_stop),
                    shape,
                    box_x_start,
                    box_x_stop))

            output = Tensor(SparseByteMap{Int64}(Pattern(), shape_id))
            rangequery(output, box, points)
            count = Scalar(0)
            @finch begin
                for id=_
                    if output[id]
                        count[] += 1
                    end
                end
            end
            @test count.val == answer[i]

            output = Tensor(SparseByteMap{Int64}(Pattern(), shape_id))
            ox = (query[i][1] + query[i][3]) / 2.0
            oy = (query[i][2] + query[i][4]) / 2.0
            radius = sqrt((query[i][1]-query[i][3])^2 + (query[i][2]-query[i][4])^2) / 2.0
            radiusquery(output, points, radius, ox, oy)
            count = Scalar(0)
            @finch begin
                for id=_
                    if output[id]
                        count[] += 1
                    end
                end
            end
            @test count.val == radanswer[i]

        end
    end


    @testset "Trilinear Interpolation on Sampled Ray" begin
        output = Tensor(SparseList(Dense(Element(Float32(0.0)))), 16, 100)
        grid = Tensor(SparseRunList{Float64}(SparseRunList{Float64}(SparseRunList{Float64}(Dense(Element(0))))), 16,16,16,27)
        timeray = Tensor(SparseInterval{Int64}(Pattern()), 100)
        @finch begin
            grid .= 0
            for i=realextent(4.0,12.0),j=realextent(4.0,12.0),k=realextent(4.0,12.0)
                for c=_
                    grid[c,~k,~j,~i] = 1.0
                end
            end
        end

        @finch begin
            timeray .= 0
            for i=extent(23,69)
               timeray[i] = true
            end
        end

        dx=dy=dz=0.1
        ox=oy=oz=0.1

        #Main Kernel
        @finch mode=:fast begin
             output .= 0
             for t=_
                 if timeray[t]
                     for i=realextent(0.0,1.0), j=realextent(0.0,1.0), k=realextent(0.0,1.0)
                         for c = _
                             output[c,t] += coalesce(grid[c,(k+dz*t+oz),(j+dy*t+oy),(i+dx*t+ox)],0) * d(i) * d(j) * d(k)
                         end
                     end
                 end
             end
         end

        res = Scalar(0.0)
        @finch begin
            for i=_, c=_
                res[] += output[c,i]
            end
        end

        @test abs(res.val - 528.4) < 1e-4
    end
end
