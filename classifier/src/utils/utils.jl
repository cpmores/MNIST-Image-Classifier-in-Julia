module Utils
export format_time
export TestResult, updateResult, save_to_csv

using CSV, DataFrames

mutable struct TestResult
    indices::Vector{Int}
    y_trues::Vector{UInt8}
    y_preds::Vector{UInt8}

    function TestResult() 
        new(Vector{Int}(), Vector{UInt8}(), Vector{UInt8}())
    end
end

function updateResult(res::TestResult, indice, y_true, y_pred)
    push!(res.indices, indice)
    push!(res.y_trues, y_true)
    push!(res.y_preds, y_pred)
end

function save_to_csv(result::TestResult, filename::String="test_results.csv")
    @assert length(result.indices) == length(result.y_trues) == length(result.y_preds)
    
    df = DataFrame(
        index = result.indices,
        true_label = result.y_trues,
        predicted_label = result.y_preds,
        is_correct = result.y_trues .== result.y_preds
    )
    
    CSV.write(filename, df)
    println("   Saved $(nrow(df)) results to $filename")
    return df
end

function format_time(seconds::Float64)
    if seconds < 60
        return "$(round(seconds, digits=2)) seconds"
    elseif seconds < 3600
        minutes = floor(Int, seconds / 60)
        secs = round(seconds % 60, digits=2)
        return "$minutes minutes $secs seconds"
    else
        hours = floor(Int, seconds / 3600)
        minutes = floor(Int, (seconds % 3600) / 60)
        secs = round(seconds % 60, digits=2)
        return "$hours hours $minutes minutes $secs seconds"
    end
end

end