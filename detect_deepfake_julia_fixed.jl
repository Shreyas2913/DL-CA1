using JSON3
using Statistics

# File paths
metadata_file = "DFFD_metadata_fully_fixed.json"
predictions_file = "predictions.json"

# Load JSON
metadata = JSON3.read(open(metadata_file, "r"))
predictions = JSON3.read(open(predictions_file, "r"))

# Convert keys to string
predictions = Dict(String(k) => v for (k, v) in predictions)
metadata = Dict(String(k) => v for (k, v) in metadata)

# Collect labels and scores
y_true = Int[]
y_score = Float64[]

for (filename, prob) in predictions
    base = replace(filename, r"__+frame\d+\.jpg" => "")
    video_key = base * ".mp4"

    if haskey(metadata, video_key)
        label = lowercase(metadata[video_key]["label"])
        label_num = label == "real" ? 0 : 1
        push!(y_true, label_num)
        push!(y_score, prob)
    else
        @warn " No metadata for $video_key"
    end
end

# Accuracy calculation
predicted = [p > 0.5 ? 1 : 0 for p in y_score]
accuracy = sum(y_true[i] == predicted[i] for i in 1:length(y_true)) / length(y_true)

# Manual AUC calculation (simple implementation)
function compute_auc(y_true, y_score)
    combined = collect(zip(y_score, y_true))
    sorted = sort(combined, by = x -> -x[1])
    tp = 0
    fp = 0
    tpr = Float64[]
    fpr = Float64[]
    pos = count(==(1), y_true)
    neg = count(==(0), y_true)
    for (score, label) in sorted
        if label == 1
            tp += 1
        else
            fp += 1
        end
        push!(tpr, tp / pos)
        push!(fpr, fp / neg)
    end
    # Trapezoidal integration
    auc = sum((fpr[i+1] - fpr[i]) * (tpr[i+1] + tpr[i]) / 2 for i in 1:length(tpr)-1)
    return auc
end

auc_val = compute_auc(y_true, y_score)

# Print final results
println("Accuracy: ", round(accuracy, digits=4))
println(" AUC: ", round(auc_val, digits=4))



