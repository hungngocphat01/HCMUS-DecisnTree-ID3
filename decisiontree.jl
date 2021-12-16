using DelimitedFiles

# Global dataset (initialized as null at the moment)
dataset = nothing

# Data structure to represent a decision tree node
mutable struct Node
    attribute::Int32        # The attribute which is used to classify at this node 
    items::Vector           # List of items which is classified in this node
    children::Vector        # List of links to the children of this node 
    candidate_attrs::Vector # List of candidate attributes for this node
end

Node() = Node(-1, Vector(), Vector(), Vector())   # Default constructor

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
    for row in eachrow(dataset)
        class = row[end]
        counter[class] += 1
    end
    # Extract frequencies from counter
    frequencies = values(counter)
    # Calculate "actual" entropy
    return sum(frequencies .* log2.(frequencies))
end

# Function to calculate information gain of a node and an attribute
function information_gain(node::Node, attribute::Int32)
    # List of values of this attribute
    values = unique(dataset[node.items, attribute])
    # Gain(S, A) = Entropy(S) - sum{(|Sv|/|S|) * Entropy(Sv) | v in A}
    
end    

function build_tree(db)
    # Khởi tạo node gốc với mọi record, các thuộc tính cần xét là mọi thuộc tính
    root_node = Node()
    root_node.items = Vector(1:length(db))
    root_node.candidate_attrs = Vector(1:length(db[0]))
    
    return root_node
end

# Hàm đọc dataset từ đĩa
function read_dataset(filename::String, delim::Char = ',')
    return readdlm(filename, delim)
end

# Hàm main
function main()
    read_dataset("PlayTennis.csv")
end
main()