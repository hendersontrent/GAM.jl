function pdist(x::Vector{Float64})
    # Compute the number of elements
    n = length(x)

    # Initialize the distance vector
    d = zeros(n, n)

    # Iterate over the elements
    for i = 1:n
        # Iterate over the elements again
        for j = 1:n
            # Compute the Euclidean distance between the i-th and j-th elements
            d[i, j] = sqrt((x[i] - x[j])^2)
        end
    end

    return d
end