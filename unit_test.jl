# HUNG NGOC PHAT - 19120615
# This is a module to perform UNIT TESTING 
# To determine if the important functions (such as entropy and informatic_gain) are working correctly

include("decisiontree.jl")

dataset = read_dataset("PlayTennis.csv")
# Testing dataset_query
test1 = dataset_query(row -> row[1] == "Sunny")
# Are all queried rows has "Sunny" as the first feature?
@assert all(dataset[test1, 1] .== "Sunny")
# Are all the other rows does not have "Sunny" as the first feature?
@assert all(dataset[setdiff(Set(1:size(dataset)[1]), test1) |> collect, 1] .!= "Sunny")
println("Test 1 passed.")

# Testing entropy
root_node = root_node_dataset(dataset)
@assert round(entropy(root_node), digits=3) == 0.940
println("Test 2 passed.")

# Testing information_gain
# These values were calculated manually
@assert round(information_gain(root_node, 1), digits=3) == 0.247
@assert round(information_gain(root_node, 4), digits=3) == 0.048
println("Test 3 passed.")

# Testing split_node_by_attr
for attr in 1:size(dataset)[2]
    children = split_node_by_attr(root_node, attr)
    for child in children
        @assert length(unique(dataset[child.items, attr])) == 1
        @assert issetequal(child.candidate_attrs, setdiff(root_node.candidate_attrs, [attr]))
    end
end
println("Test 4 passed.")

# Testing check_purity
for class in unique(dataset[:, end])
    node = Node()
    node.items = dataset_query(row -> row[end] == class)
    is_pure, pure_class = check_purity(node)
    @assert is_pure
    @assert pure_class == class
end
println("Test 5 passed.")
