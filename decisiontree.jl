###################################
# HUNG NGOC PHAT - 19120615       #
# FIT@VNU-HCMUS, December 2021    #
# ID3 ALGORITHM W/ NUMERICAL COLS #
###################################

# Standard Julia library for handing CSV-like files
using DelimitedFiles
using Printf

# Global dataset (initialized as null at the moment)
dataset = missing

# If dataset is numerical (this program works with both categorical & numerical data)
numerical_mode = false

# Debug flag 
verbose = false

# Function for debug printing
function debug_print(args...)
    if verbose
        println(args...)
    end
end

# Function to get the key of the maximum/minimum value of a dictionary 
# cmpr_func SHOULD EITHER BE argmin or argmax
function get_extreme_key(dict::Dict, cmpr_func::Function)
    # A functional programming way (not syntatically aethestic, but offer good performance)
    # Note: at the moment (Julia 1.6.4), keys() and values() are guaranteed to be consistent in key-value order.
    return collect(keys(dict))[cmpr_func(values(dict) |> collect)]
end

# Data structure to represent a decision tree node
mutable struct Node
    decision::Any           # Decision value at this node (IF it is a LEAF node)
    items::Vector           # List of items (index) which belongs to this node
    children::Vector        # List of links to the children of this node
    candidate_attrs::Vector # List of candidate attributes for this node
    numerical_split_point::Float64 
                            # The point to split this node (if numerical) into two halves
    jump_condition::Union{Missing, Function}    
                            # Pointer to the function which evaluates the jump condition to this node 
                            #   e.g. somerow -> (somerow[someattribute] == somevalue)
                            #   By using a function pointer, we can handle continuous data
    jump_condition_str::String
                            # The string representation of the condition (for DEBUGGING purposes)
    depth::Integer          # To perform cutting (if needed) to tackle overfitting
                            #   For this project, it is only a DEBUGGING parameter
    attribute::Integer      # The attribute which is used to classify at this node (for debugging only)
end

# Default constructor (initialize everything as empty)
Node() = Node(missing, Vector(), Vector(), Vector(), Inf, missing, "", -1, -1) 

# Function to construct a root node from a dataset
function root_node_dataset(dataset)
    root_node = Node() 
    root_node.items = 1:size(dataset)[1]
    root_node.candidate_attrs = 1:(size(dataset)[2] - 1) # remove last column (class column)
    root_node.depth = 1
    return root_node
end

# Function to query the dataset
# Returns a LIST OF INDICES of rows which satisfy the query 
# func is a boolean FUNCTION which receives 1 param: each row of the dataset
# Example usage: dataset_query(row -> row[1] == "Sunny")
function dataset_query(query_func)
    return [i for i in 1:size(dataset)[1] if query_func(dataset[i, :])]
end


# Return a LIST of child nodes which is splitted from a node respecting to an attribute 
# Each child node corresponds to a value of the attribute
function split_node_by_attr(node::Node, attr::Integer)
    # PROCESS FOR NUMERICAL DATA 
    if numerical_mode
        best_split, _ = get_best_numerical_split(node, attr)
        node.numerical_split_point = best_split
        left_node = Node()
        right_node = Node() 

        left_node.jump_condition = (row -> row[attr] <= best_split)
        left_node.jump_condition_str = @sprintf("attr %d <= %f", attr, best_split)
        right_node.jump_condition = (row -> row[attr] > best_split)
        right_node.jump_condition_str = @sprintf("attr %d > %f", attr, best_split)

        left_node.candidate_attrs = setdiff(node.candidate_attrs, [attr]) |> collect
        right_node.candidate_attrs = copy(left_node.candidate_attrs)

        left_node.depth = node.depth + 1
        right_node.depth = node.depth + 1

        left_node.items = (intersect(dataset_query(left_node.jump_condition), node.items) |> collect)
        right_node.items = (setdiff(node.items, left_node.items) |> collect) # reduce computational cost 
        return [left_node, right_node]
    end

    # PROCESS FOR CATEGORICAL DATA
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
        child.depth = node.depth + 1

        push!(children, child)
    end
    return children
end


# Check if all items in this node belongs to only one class, OR there is no more attribute to choose from
# Also returns that class in case the node is pure
# Does not check for Entropy == 0 due to floating point inaccuracy
function check_purity(node::Node)
    class_col = dataset[node.items, end]
    classes = unique(class_col)
    is_pure = length(classes) == 1
    
    if is_pure 
        return (true, first(classes))
    end 

    # If node is not pure, but there is no more attribute to split
    if length(node.candidate_attrs) == 0
        counter = Dict(class => count(r -> r == class, class_col) for class in classes)
        # Get the major class in this node
        major_class = get_extreme_key(counter, argmax)
        # Pretend to be pure
        return (true, major_class)
    end

    # If node is not pure and is divisible
    return (false, missing)
end

# Function to calculate entropy of a node
# Only works with CATEGORICAL DATA
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
        row_class = row[end]
        counter[row_class] += 1
    end
    # Extract frequencies from counter
    frequencies = values(counter) ./ length(node.items)
    # Calculate "actual" entropy
    return -sum(frequencies .* log2.(frequencies))
end

# Function to find and set the best numerical split point of a particular column on a node (subset of database)
function get_best_numerical_split(node::Node, attr::Integer)
    # Get all unique values of the column
    values = unique(dataset[node.items, attr])
    
    weighted_entropies = Dict()
    # For each value of the column
    for value in values
        # Split the dataset into 2 halves
        left_node = Node()      # items having row[col] <= value 
        right_node = Node()     # items having row[col] > value

        left_node.items = (intersect(dataset_query(row -> row[attr] <= value), node.items) |> collect)
        right_node.items = (setdiff(node.items, left_node.items) |> collect) # reduce computational cost 
        
        # Calculate weighted entropy
        e_left = entropy(left_node)
        e_right = entropy(right_node)
        we_left = length(left_node.items) / length(node.items) * e_left
        we_right = length(right_node.items) / length(node.items) * e_right
        weighted_entropies[value] = we_left + we_right
    end
    # Best split: least entropy
    best_split = get_extreme_key(weighted_entropies, argmin)
    best_entropy = weighted_entropies[best_split]
    return (best_split, best_entropy)
end

# Function to calculate information gain of a node against an attribute
# Only works with CATEGORICAL DATA!!
function information_gain(node::Node, attribute::Integer)
    # PROCESS NUMERICAL DATA
    if numerical_mode
        _, best_entropy = get_best_numerical_split(node, attribute)
        return entropy(node) - best_entropy 
    end

    # PROCESS CATEGORICAL DATA
    # List of values of this attribute
    values = unique(dataset[node.items, attribute]) # Set of "A"

    # Gain(S, A) = Entropy(S) - sum{(|Sv|/|S|) * Entropy(Sv) | v in A}
    weighted_entropy = 0
    for value in values 
        temp_node = Node()
        # Get all rows whose value of "attribute" is "v"
        temp_node.items = (intersect(dataset_query(row -> row[attribute] == value), node.items) |> collect) 
        temp_entropy =  entropy(temp_node)
        # display(dataset[temp_node.items, :])
        # println((length(temp_node.items), temp_entropy))
        weighted_entropy += length(temp_node.items) / length(node.items) * temp_entropy
    end
    return entropy(node) - weighted_entropy
end

# Function to "train" a decision tree model
# The initial model should be a ROOT NODE (already populated with all items)
# The LAST column must ALWAYS be the class column
function fit(node::Node)
    debug_print("\n--------------------------------------------")
    debug_print("Jump condition: ", node.jump_condition_str)
    debug_print("Depth: ", node.depth)
    # If this node is a leaf node, do nothing
    if ~ismissing(node.decision)
        debug_print("Leaf node: ", node.decision)
        return
    end

    # Calculate information gain of current node against all candidate attributes
    gains = Dict(attr => information_gain(node, attr) for attr in node.candidate_attrs)
    debug_print("Gains: ", gains)

    # Find the attribute with best (highest) gain
    best_attr = get_extreme_key(gains, argmax)
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
        end
        # Else, recursively build its subtree
        fit(child)
    end
    debug_print("Going up")
end

# Function to predict the class for a given datapoint 
# node should be the root node of a "trained" tree
# The class column (if present) will not be taken into account
function predict(node::Node, row)
    # If is leaf node: return the class 
    if ~ismissing(node.decision)
        debug_print("Leaf node met!")
        return node.decision
    end
    debug_print("Jumping into node with attribute: ", node.attribute)
    debug_print("Jump condition: ", node.jump_condition_str)
    
    # Test for each child branch of the tree
    for child in node.children
        # If this datapoint satisfies one of the child: jump into that child
        if child.jump_condition(row)
            return predict(child, row)
        end
    end
end


# Function to evaluate the model 
# Returns the percentage [0, 1] of correcly labeled datapoints
# Test set should be in the SAME FORMAT as dataset (class column must be included)
function evaluate(node::Node, testset)
    # Number of rows 
    n = size(testset)[1]
    # Number of correcly labeled datapoints 
    n_correct = 0 
    for row in eachrow(testset)
        c = predict(node, row)
        if c == row[end]
            n_correct += 1
        end
    end
    return n_correct / n
end

# Read the dataset from disk 
function read_dataset(filename::String, delim::Char = ',')
    ds = readdlm(filename, delim)
    return ds
end