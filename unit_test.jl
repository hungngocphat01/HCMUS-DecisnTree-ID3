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
@assert round(information_gain(root_node, 1), digits=3) == 0.247
@assert round(information_gain(root_node, 4), digits=3) == 0.048
println("Test 3 passed.")