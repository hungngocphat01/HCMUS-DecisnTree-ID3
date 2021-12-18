# HUNG NGOC PHAT - 19120615
using DelimitedFiles
using Printf

# Global dataset (initialized as null at the moment)
dataset = nothing

# Debug flag 
verbose = false

# Macro to debug printing
function debug_print(args...)
    if verbose
        println(args...)
    end
end

# Data structure to represent a decision tree node
mutable struct Node
    decision::Any           # Decision value at this node (IF it is a LEAF node)
    attribute::Integer      # The attribute which is used to classify at this node 
    items::Vector           # List of items which is classified in this node
    children::Vector        # List of links to the children of this node
    candidate_attrs::Vector # List of candidate attributes for this node
    jump_condition::Union{Missing, Function}    
                            # Pointer to the function which evaluates the jump condition to this node 
                            # e.g. dataset[someattribute] == somevalue
                            # By using a function pointer, we can handle continuous data
    jump_condition_str::String
                            # The representation string of the condition (for debugging purposes)
end

# Default constructor (initialize everything as empty)
Node() = Node(missing, -1, Vector(), Vector(), Vector(), missing, "") 

# Function to construct a root node from a dataset
function root_node_dataset(dataset)
    root_node = Node() 
    root_node.items = 1:size(dataset)[1]
    root_node.candidate_attrs = 1:(size(dataset)[2] - 1) # remove last column (class column)
    return root_node
end

# Function to query the dataset
# Returns a LIST OF INDICES of rows which satisfy the query 
# func is a boolean FUNCTION which receives 1 param: each row of the dataset
function dataset_query(query_func)
    return [i for i in 1:size(dataset)[1] if query_func(dataset[i, :])]
end


# Return a LIST of child nodes which is splitted from a node respecting to an attribute 
# Each child node corresponds to a value of the attribute
# Only works with CATEGORICAL columns
function split_node_by_attr(node::Node, attr::Integer)
    values = unique(dataset[node.items, attr])
    children = []

    for value in values
        child = Node()
        # Candidate testing expression
        child.jump_condition = (row -> row[attr] == value)
        child.jump_condition_str = value
        # Candidate attributes for the child node will not contain the current split attribute
        # `collect` is a julia function to convert any `iterable` to a vector/array
        child.candidate_attrs = setdiff(node.candidate_attrs, [attr]) |> collect
        # Items for this child node = items in the current node, which satisfy the value condition at attr
        child.items = intersect(node.items, dataset_query(child.jump_condition)) |> collect
        # Add the test condition for this children 
        push!(children, child)
    end
    return children
end

# Check if all items in this node belongs to only one class 
# Also returns that class in case the node is pure
# Does not check for Entropy == 0 due to floating point inaccuracy
function check_purity(node::Node)
    classes = unique(dataset[node.items, end])
    is_pure = length(classes) == 1
    if ~is_pure 
        return (false, missing)
    end 
    return (true, first(classes))
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

# Function to calculate information gain of a node against an attribute
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
# The initial model should be a ROOT NODE (already populated with all items)
# The LAST column will ALWAYS been the class column
function fit(node::Node, db)
    debug_print("\n--------------------------------------------")
    debug_print("Jump condition: ", node.jump_condition_str)
    # If this node is a leaf node, do nothing
    if ~ismissing(node.decision)
        debug_print("Leaf node: ", node.decision)
        return
    end

    # Calculate information gain of current node against all candidate attributes
    gains = Dict(attr => information_gain(node, attr) for attr in node.candidate_attrs)
    debug_print("Gains: ", gains)

    # Find the attribute with best gain (i enjoy functional programming <(") so i wrote it like this)
    # keys() and values() are guaranteed to be consistently ordered, at the current version of julia (1.6.4)
    best_attr = collect(keys(gains))[argmin(values(gains) |> collect)]
    debug_print("Best attr: ", best_attr)

    node.attribute = best_attr
    node.children = split_node_by_attr(node, best_attr)
    debug_print("Built ", length(node.children), " children branches")

    # Recursively build the tree for each child of the node 
    for child in node.children
        # If node is pure, do not check further
        is_pure, decsn_class = check_purity(child)
        if is_pure
            child.decision = decsn_class
        else 
            # Else, recursively build its subtree
            fit(child, dataset)
        end
    end
    debug_print("Going up")
end

# Read the dataset from disk 
function read_dataset(filename::String, delim::Char = ',')
    return readdlm(filename, delim)
end