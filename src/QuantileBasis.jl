function QuantileBasis(x::AbstractVector, n_knots::Int64, order::Int64)

    # Build a list of the Knots (breakpoints)
    KnotsList = quantile(x, range(0, 1; length=n_knots));

    # Define the Basis Object
    Basis = BSplineBasis(order, KnotsList);

    return Basis
end