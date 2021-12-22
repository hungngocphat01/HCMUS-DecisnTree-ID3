# HUNG NGOC PHAT - 19120615
# This is a module to perform UNIT TESTING 
# To determine if the important functions (such as entropy and informatic_gain) are working correctly

include("decisiontree.jl")

numerical_mode = true 
dataset = read_dataset("iris.csv")

# Testing dataquery 
test1a = dataset_query(row -> row[1] <= 3)
test1b = dataset_query(row -> row[1] > 3)
@assert length(test1a) + length(test1b) == size(dataset)[1]
@assert all(dataset[test1a, 1] .<= 3)
@assert all(dataset[test1b, 1] .> 3)
println("Test 1 passed.")

# Testing split_node_by_attr
root_node = root_node_dataset(dataset)
children = split_node_by_attr(root_node, 1)
@assert all(dataset[children[1].items, 1] .<= 5.5)
@assert all(dataset[children[2].items, 1] .> 5.5)
println("Test 2 passed.")

# MASTER TEST (accuracy should be 100%)
fit(root_node)
println("Overall accuracy: ", evaluate(root_node, dataset))