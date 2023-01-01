function thin_plate_spline(x::Vector{Float64})
    # Compute the Euclidean distance between each pair of points
    d = pdist(x)

    # Compute the thin plate spline function
    tps = zeros(length(x))
    for i = 1:length(x)
        for j = 1:length(x)
            if d[i, j] > 0
                tps[i] += d[i, j] ^ 2 * log(d[i, j])
            end
        end
    end
    return tps
end