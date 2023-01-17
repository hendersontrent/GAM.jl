function parse_smooth_term(term::AbstractString)
    if occursin(r"s\((.*?)\)", term)
        # Extract arguments within parentheses
        mymatch = match(r"s\((.*)\)", term)
        smooth_arguments = split(mymatch.captures[1], ',')
        smooth_kwargs = Dict()
        for i in 2:length(smooth_arguments)
            # Split on "=" to get key-value pairs
            kv = split(smooth_arguments[i], '=')
            if length(kv) == 2
                # If key-value pair has argument name and value
                if kv[1] == "k"
                    smooth_kwargs[:k] = parse(Int64, kv[2])
                elseif kv[1] == "degree"
                    smooth_kwargs[:degree] = parse(Int64, kv[2])
                end
            else
                # If no argument name is provided, use the index as the argument name
                if i == 2
                    smooth_kwargs[:k] = parse(Symbol, kv[1])
                elseif i == 3
                    smooth_kwargs[:degree] = parse(Symbol, kv[1])
                end
            end
        end
        # Set default values for missing arguments
        if !haskey(smooth_kwargs, :k)
            smooth_kwargs[:k] = 10
        end
        if !haskey(smooth_kwargs, :degree)
            smooth_kwargs[:degree] = 3
        end
        # Return values of supplied arguments
        return [smooth_kwargs[:k], smooth_kwargs[:degree]]
    end
end