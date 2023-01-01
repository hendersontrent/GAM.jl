function tprs_smoothness_penalty(tprs_basis::Matrix{Float64}, k::Int)
    # Compute the smoothness penalty as the trace of the product of the first k rows of the TPRS basis matrix with the transpose of the last n - k rows
    
    #return tr(tprs_basis[1:k, :] * tprs_basis[k + 1:end, :]')
    tr(tprs_basis[1:k, :]' * tprs_basis[1:k, :])
end