function starts(s::AbstractString, t::AbstractString)
    s[1:length(t)] == t
end