# HUNG NGOC PHAT - 19120615
using DelimitedFiles

# Global dataset (initialized as null at the moment)
dataset = nothing

# Data structure to represent a decision tree node
mutable struct Node
    attribute::Integer        # The attribute which is used to classify at this node 
    items::Vector           # List of items which is classified in this node
    children::Vector        # List of links to the children of this node 
    candidate_attrs::Vector # List of candidate attributes for this node
end

Node() = Node(-1, Vector(), Vector(), Vector())   # Default constructor


# Function to construct a root node from a dataset
function root_node_dataset(dataset)
    root_node = Node() 
    root_node.items = 1:size(dataset)[1]
    return root_node
end

# Function to query the dataset
# Returns a LIST OF INDICES of rows which satisfy the query 
# func is a boolean FUNCTION which receives 1 param: each row of the dataset
function dataset_query(func)
    return Vector([i for i in 1:size(dataset)[1] if func(dataset[i, :])])
end


# Function to calculate entropy of a node
function entropy(node::Node)
    # E(S) = -sum(pi * log(pi))
    # Get all classes from the current node 
    classes = unique(dataset[node.items, end])
    if length(classes) <= 1
        return 0 
    end

    # Dictionary to count the frequency of each class
    counter = Dict(c => 0 for c in classes)
    # For each row in the dataset
    for row in eachrow(dataset[node.items, :])
        class = row[end]
        counter[class] += 1
    end
    # Extract frequencies from counter
    frequencies = values(counter) ./ length(node.items)
    # Calculate "actual" entropy
    return -sum(frequencies .* log2.(frequencies))
end

# Function to calculate information gain of a node and an attribute
function information_gain(node::Node, attribute::Integer)
    # List of values of this attribute
    values = unique(dataset[node.items, attribute]) # Set of "A"

    # Gain(S, A) = Entropy(S) - sum{(|Sv|/|S|) * Entropy(Sv) | v in A}
    gain = entropy(node)
    for value in values 
        temp_node = Node()
        # Get all rows whose value of "attribute" is "v"
        # This is the "anonymous function" syntax like in javascript/C#
        temp_node.items = dataset_query(row -> row[attribute] == value)
        temp_entropy =  entropy(temp_node)
        # display(dataset[temp_node.items, :])
        # println((length(temp_node.items), temp_entropy))
        gain -= length(temp_node.items) / length(node.items) * temp_entropy
    end
    return gain
end    

# Function to "train" a decision tree model
# Here, the initial model should be an EMPTY NODE
# The LAST column will ALWAYS been the class column
function fit(root_node::Node, db)
    
    return root_node
end

# Read the dataset from disk 
function read_dataset(filename::String, delim::Char = ',')
    return readdlm(filename, delim)
end


function main()
    global dataset
    dataset = read_dataset("PlayTennis.csv")
end
main()