include("decisiontree.jl")
using Random

# Function to randomly shuffle the rows of the dataset, then split it accordingly to a certain ratio
function train_test_split(dataset, train_ratio::Float64)
    shuffled_dataset = dataset[shuffle(1:end), :]
    split_point = floor(Integer, train_ratio * size(dataset)[1])
    # return train, test
    return shuffled_dataset[1:split_point, :], shuffled_dataset[split_point+1:end, :]
end

function main()
    # global verbose = true           # debugging is disabled when submitted to moodle for grading
    orig_dataset = read_dataset("iris.csv")
    test_set, train_set = train_test_split(orig_dataset, 2/3)

    # Ensure that we are writing to the GLOBAL dataset 
    #   so that other functions can access this dataset instance, too
    global dataset = train_set
    global numerical_mode = true 
    
    decision_model = root_node_dataset(train_set)
    fit(decision_model)
    # Calculating accuracy for test and train set 
    train_acc = evaluate(decision_model, train_set)
    test_acc = evaluate(decision_model, test_set)
    println("Train accuracy: ", train_acc)
    println("Test accuracy: ", test_acc)
end
main()