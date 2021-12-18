include("decisiontree.jl")

function main()
    # Ensure that we are writing to the GLOBAL dataset 
    # so that other functions can access this very dataset instance, too
    global verbose = true
    global dataset = read_dataset("PlayTennis.csv")
    root_node = root_node_dataset(dataset)
    fit(root_node, dataset)
end
main()