# pseudo code

## build KD-Tree

```
function BuildKDTree(points, depth)
    if points is empty
        return null
    
    k = number of dimensions
    axis = depth mod k
    
    // Sort points along the current axis
    sort points by their value in the chosen axis
    
    median_index = floor(length(points) / 2)
    median_point = points[median_index]
    
    // Create node with the median point
    node = new Node(
        point = median_point,
        axis = axis,
        left = BuildKDTree(points[0:median_index-1], depth+1),
        right = BuildKDTree(points[median_index+1:end], depth+1)
    )
    
    return node
```

## Nearest Neighbor Search

```
function NearestNeighbor(query_point)
    best_point = null
    best_distance = infinity
    
    function Search(node, depth)
        if node is null
            return
        
        // Current dimension for comparison
        axis = depth mod k
        
        // Calculate distance to current node's point
        current_distance = EuclideanDistance(query_point, node.point)
        
        // Update best if current is closer
        if current_distance < best_distance
            best_point = node.point
            best_distance = current_distance
        
        // Determine which subtree to search first
        if query_point[axis] < node.point[axis]
            first_subtree = node.left
            second_subtree = node.right
        else
            first_subtree = node.right
            second_subtree = node.left
        
        // Search the closer subtree first
        Search(first_subtree, depth+1)
        
        // Calculate distance to the splitting plane
        plane_distance = (query_point[axis] - node.point[axis])²
        
        // Only search other subtree if it could contain a closer point
        if plane_distance < best_distance
            Search(second_subtree, depth+1)
    
    Search(root, 0)
    return (best_point, best_distance)
```

## k-Nearest Neighbor Search

```
function KNearestNeighbors(query_point, k)
    // Max heap to store k nearest points (negative distance for max-heap behavior)
    nearest = empty max-heap
    
    function Search(node, depth)
        if node is null
            return
        
        axis = depth mod k
        
        // Calculate distance to current node
        current_distance = EuclideanDistance(query_point, node.point)
        
        // Add to heap if we have fewer than k points or if closer than furthest in heap
        if size(nearest) < k or current_distance < -top(nearest).distance
            if size(nearest) == k
                remove top of heap
            add (-current_distance, node.point) to heap
        
        // Determine which subtree to search first
        if query_point[axis] < node.point[axis]
            first_subtree = node.left
            second_subtree = node.right
        else
            first_subtree = node.right
            second_subtree = node.left
        
        // Search closer subtree first
        Search(first_subtree, depth+1)
        
        // Calculate distance to splitting plane
        plane_distance = abs(query_point[axis] - node.point[axis])
        
        // Search other subtree if it could contain points closer than our furthest neighbor
        if size(nearest) < k or plane_distance < -top(nearest).distance
            Search(second_subtree, depth+1)
    
    Search(root, 0)
    
    // Convert heap to sorted list by distance
    result = empty list
    while nearest is not empty
        (neg_dist, point) = pop from heap
        add (point, -neg_dist) to result
    
    return reverse(result)  // Return in ascending order by distance
```

## Range Search

```
function RangeSearch(lower_bound, upper_bound)
    result = empty list
    
    function Search(node, depth)
        if node is null
            return
        
        // Check if current point is in range
        in_range = true
        for i = 0 to k-1
            if node.point[i] < lower_bound[i] or node.point[i] > upper_bound[i]
                in_range = false
                break
        
        if in_range
            add node.point to result
        
        axis = depth mod k
        
        // Search left subtree if it could contain points in range
        if node.left is not null and lower_bound[axis] <= node.point[axis]
            Search(node.left, depth+1)
        
        // Search right subtree if it could contain points in range
        if node.right is not null and upper_bound[axis] >= node.point[axis]
            Search(node.right, depth+1)
    
    Search(root, 0)
    return result
```