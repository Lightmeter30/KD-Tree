import numpy as np
import heapq
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from myTime import time_decorator
class KDTree:
    """
    K-dimensional tree implementation for efficient spatial searches with visualization.
    """
    
    class Node:
        """
        Node in the KD-Tree
        """
        def __init__(self, point, left=None, right=None, axis=None):
            self.point = point  # k-dimensional point
            self.left = left    # left child
            self.right = right  # right child
            self.axis = axis    # split axis
    
    def __init__(self, points):
        """
        Initialize KD-Tree with a list of points.
        
        Args:
            points: List of points where each point is a list or array of k dimensions.
        """
        if not isinstance(points, np.ndarray):
            points = np.array(points)
        
        if len(points) == 0:
            self.root = None
            return
            
        self.k = points.shape[1]  # number of dimensions
        self.all_points = points  # store all points for visualization
        self.root = self.build_tree(points, 0)
    
    def find_highest_variance_axis(self, points):
        """
        Find the axis (dimension) with the highest variance.
        
        Args:
            points: Array of points to analyze.
            
        Returns:
            Index of the axis with highest variance.
        """
        # Calculate variance along each dimension
        variances = np.var(points, axis=0)
        
        # Return the index of the dimension with highest variance
        return np.argmax(variances)
    
    @time_decorator
    def build_tree(self, points, depth):
        """
        Recursively build the KD-Tree.
        
        Args:
            points: Array of points to build the tree from.
            depth: Current depth in the tree.
            
        Returns:
            Root node of the subtree.
        """
        if len(points) == 0:
            return None
            
        # Select axis based on depth so that axis cycles through all dimensions
        axis = self.find_highest_variance_axis(points=points)
        # axis = depth % self.k
        
        # Sort points along the selected axis
        points = points[points[:, axis].argsort()]
        
        # Get the median point
        median_idx = len(points) // 2
        
        # Create node and recursively construct subtrees
        node = self.Node(
            point=points[median_idx],
            axis=axis,
            left=self.build_tree(points[:median_idx], depth + 1),
            right=self.build_tree(points[median_idx+1:], depth + 1)
        )
        
        return node
    
    @time_decorator
    def nearest_neighbor(self, query_point):
        """
        Find the nearest neighbor to the query point.
        
        Args:
            query_point: The point to find the nearest neighbor for.
            
        Returns:
            Tuple of (nearest point, distance)
        """
        if self.root is None:
            return None, float('inf')
            
        if not isinstance(query_point, np.ndarray):
            query_point = np.array(query_point)
            
        best = [None, float('inf')]  # [nearest_point, nearest_distance]
        search_path = []  # Store nodes visited for visualization
        
        def _search(node):
            if node is None:
                return
                
            search_path.append(node)
            
            # Compute current distance
            current_distance = np.sum((query_point - node.point) ** 2)
            
            # Update best if current point is closer
            if current_distance < best[1]:
                best[0] = node.point
                best[1] = current_distance
            
            # Decide which subtree to search first based on query point position
            if query_point[node.axis] < node.point[node.axis]:
                first, second = node.left, node.right
            else:
                first, second = node.right, node.left
                
            # Search the most promising subtree first
            _search(first)
            
            # Check if we need to search the other subtree
            # If the distance to the splitting plane is greater than the current best distance,
            # we don't need to search the other subtree
            if (query_point[node.axis] - node.point[node.axis])**2 < best[1]:
                _search(second)
        
        _search(self.root)
        print(f"Nearest point to {query_point} is {best[0]} with distance {np.sqrt(best[1]):.2f}")
        return tuple(best), search_path
    
    @time_decorator
    def k_nearest_neighbors(self, query_point, k=1):
        """
        Find the k nearest neighbors to the query point.
        
        Args:
            query_point: The point to find the nearest neighbors for.
            k: Number of nearest neighbors to find.
            
        Returns:
            List of k nearest points and their distances.
        """
        if self.root is None:
            return []
            
        if not isinstance(query_point, np.ndarray):
            query_point = np.array(query_point)
            
        # Use a max heap to keep track of the k nearest neighbors
        nearest = []  # (negative distance, point) pairs
        search_path = []  # Store nodes visited for visualization
        
        def _search(node):
            if node is None:
                return
                
            search_path.append(node)
            
            # Compute current distance
            current_distance = np.sum((query_point - node.point) ** 2)
            
            # If we have less than k points or current point is closer than the furthest point in our heap
            if len(nearest) < k or -nearest[0][0] > current_distance:
                # Add current point to heap
                if len(nearest) == k:
                    heapq.heappushpop(nearest, (-current_distance, tuple(node.point)))
                else:
                    heapq.heappush(nearest, (-current_distance, tuple(node.point)))
            
            # Decide which subtree to search first based on query point position
            if query_point[node.axis] < node.point[node.axis]:
                first, second = node.left, node.right
            else:
                first, second = node.right, node.left
                
            # Search the most promising subtree first
            _search(first)
            
            # Check if we need to search the other subtree
            # If the distance to the splitting plane is greater than the furthest point in our heap,
            # we don't need to search the other subtree
            if len(nearest) < k or abs(query_point[node.axis] - node.point[node.axis]) ** 2 < -nearest[0][0]:
                _search(second)
        
        _search(self.root)
        
        # Convert heap to sorted list of (point, distance) pairs
        result = [(point, -dist) for dist, point in sorted(nearest, reverse=True)]
        print(f"{k} nearest points to {query_point}:")
        for i, (point, dist) in enumerate(result):
            print(f"{i+1}: {point} with distance {np.sqrt(dist):.2f}")
        return result, search_path
    
    @time_decorator
    def range_search(self, lower_bound, upper_bound):
        """
        Find all points within the given range.
        
        Args:
            lower_bound: Lower bound of the range (inclusive).
            upper_bound: Upper bound of the range (inclusive).
            
        Returns:
            List of points within the range.
        """
        if self.root is None:
            return []
            
        if not isinstance(lower_bound, np.ndarray):
            lower_bound = np.array(lower_bound)
        if not isinstance(upper_bound, np.ndarray):
            upper_bound = np.array(upper_bound)
        if not np.all(lower_bound <= upper_bound):
            raise ValueError("Invalid range: lower_bound must be less than or equal to upper_bound in all dimensions")
        
        result = []
        search_path = []  # Store nodes visited for visualization
        
        def _search(node):
            if node is None:
                return
                
            search_path.append(node)
                
            # Check if the current point is within the range
            if np.all(lower_bound <= node.point) and np.all(node.point <= upper_bound):
                result.append(node.point)
            
            # Check if the left subtree needs to be searched
            if node.left is not None and lower_bound[node.axis] <= node.point[node.axis]:
                _search(node.left)
                
            # Check if the right subtree needs to be searched
            if node.right is not None and upper_bound[node.axis] >= node.point[node.axis]:
                _search(node.right)
        
        _search(self.root)
        print(f"Points in range {lower_bound} to {upper_bound}:")
        for point in result:
            print(f"{point}")
        return result, search_path
    
    def visualize_tree(self, bounds=None, ax=None, depth=0, node=None):
        """
        Visualize the KD-Tree structure (currently supports 2D only).
        
        Args:
            bounds: The bounds of the current region (min_x, min_y, max_x, max_y).
            ax: Matplotlib axis to plot on.
            depth: Current depth in the tree.
            node: Current node to visualize.
        """
        if self.k != 2:
            raise ValueError("Visualization is only supported for 2D trees")
            
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
            
        if bounds is None:
            # Determine bounds from all points
            min_vals = np.min(self.all_points, axis=0)
            max_vals = np.max(self.all_points, axis=0)
            # Add some padding
            padding = (max_vals - min_vals) * 0.1
            bounds = (min_vals[0] - padding[0], min_vals[1] - padding[1],
                    max_vals[0] + padding[0], max_vals[1] + padding[1])
        if node is None:
            node = self.root
            # Plot all points
            ax.scatter(self.all_points[:, 0], self.all_points[:, 1], c='red', s=30, label='Points')
            
        if node is None:
            return
            
        # Draw the splitting line
        axis = node.axis
        if axis == 0:  # Vertical line (split on x-axis)
            y_min = bounds[1]
            y_max = bounds[3]
            if y_max <= y_min:
                # get the global y bounds
                y_min -= 0.5
                y_max += 0.5
                # Add padding
                padding = (y_max - y_min) * 0.1
                y_min -= padding
                y_max += padding
            ax.plot([node.point[0], node.point[0]], [y_min, y_max], 'r-', alpha=0.5)
            
            # Recurse to left and right subtrees with updated bounds
            if node.left:
                new_bounds = (bounds[0], bounds[1], node.point[0], bounds[3])
                self.visualize_tree(new_bounds, ax, depth + 1, node.left)
            if node.right:
                new_bounds = (node.point[0], bounds[1], bounds[2], bounds[3])
                self.visualize_tree(new_bounds, ax, depth + 1, node.right)
                
        else:  # Horizontal line (split on y-axis)
            x_min = bounds[0]
            x_max = bounds[2]
            if x_max <= x_min:
                # get the global x bounds
                x_min -= 0.5
                x_max += 0.5
                # Add padding
                padding = (x_max - x_min) * 0.1
                x_min -= padding
                x_max += padding
            ax.plot([x_min, x_max], [node.point[1], node.point[1]], 'g-', alpha=0.5)
            
            # Recurse to left and right subtrees with updated bounds
            if node.left:
                new_bounds = (bounds[0], bounds[1], bounds[2], node.point[1])
                self.visualize_tree(new_bounds, ax, depth + 1, node.left)
            if node.right:
                new_bounds = (bounds[0], node.point[1], bounds[2], bounds[3])
                self.visualize_tree(new_bounds, ax, depth + 1, node.right)
                
        # Highlight this node
        ax.scatter(node.point[0], node.point[1], c='red', s=50)
            
        if depth == 0:  # Only for the initial call
            ax.set_xlim(bounds[0], bounds[2])
            ax.set_ylim(bounds[1], bounds[3])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title('KD-Tree Visualization')
            ax.legend()
            
        return ax
    
    def visualize_nearest_neighbor(self, query_point, ax=None):
        """
        Visualize the nearest neighbor search process (supports 2D points only).
        
        Args:
            query_point: Query point to find nearest neighbor for.
            ax: Matplotlib axis to plot on.
        """
        if self.k != 2:
            raise ValueError("Visualization is only supported for 2D trees")
            
        if not isinstance(query_point, np.ndarray):
            query_point = np.array(query_point)
            
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
            
        # First visualize the tree structure
        self.visualize_tree(ax=ax)
        
        # Perform nearest neighbor search and get the search path
        (nearest_point, distance), search_path = self.nearest_neighbor(query_point)
        
        # Plot query point
        ax.scatter(query_point[0], query_point[1], c='purple', s=100, marker='*', label='Query Point')
        
        # Plot nearest point
        ax.scatter(nearest_point[0], nearest_point[1], c='blue', s=80, marker='o', label='Nearest Point')
        
        # Draw a circle with radius equal to the distance to the nearest neighbor
        circle = plt.Circle((query_point[0], query_point[1]), distance, color='purple', fill=False, alpha=0.5)
        ax.add_patch(circle)
        
        # Draw line connecting query point to nearest point
        ax.plot([query_point[0], nearest_point[0]], [query_point[1], nearest_point[1]], 'k--', alpha=0.7)
        
        # Highlight the search path
        for i, node in enumerate(search_path):
            alpha = 0.3 + 0.7 * (i / len(search_path))  # Fade in nodes visited later
            ax.scatter(node.point[0], node.point[1], c='green', s=70, alpha=alpha, edgecolors='black')
            
        ax.set_title(f'Nearest Neighbor Search for {query_point}')
        ax.legend()
        
        return ax
    
    def visualize_k_nearest_neighbors(self, query_point, k, ax=None):
        """
        Visualize the k nearest neighbors search process.
        
        Args:
            query_point: Query point to find nearest neighbors for.
            k: Number of nearest neighbors to find.
            ax: Matplotlib axis to plot on.
        """
        if self.k != 2:
            raise ValueError("Visualization is only supported for 2D trees")
            
        if not isinstance(query_point, np.ndarray):
            query_point = np.array(query_point)
            
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
            
        # First visualize the tree structure
        self.visualize_tree(ax=ax)
        
        # Perform k nearest neighbors search and get the search path
        k_nearest, search_path = self.k_nearest_neighbors(query_point, k)
        
        # Plot query point
        ax.scatter(query_point[0], query_point[1], c='purple', s=100, marker='*', label='Query Point')
        
        # Plot k nearest points
        for i, (point, distance) in enumerate(k_nearest):
            ax.scatter(point[0], point[1], c='blue', s=80, marker='o', alpha=0.7, 
                       label=f'Nearest {i+1}' if i == 0 else "")
            
            # Draw line connecting query point to nearest point
            ax.plot([query_point[0], point[0]], [query_point[1], point[1]], 'k--', alpha=0.5)
            
        # Draw a circle with radius equal to the distance to the farthest of the k nearest neighbors
        max_distance = k_nearest[-1][1]
        circle = plt.Circle((query_point[0], query_point[1]), max_distance, color='purple', fill=False, alpha=0.5)
        ax.add_patch(circle)
        
        # Highlight the search path
        for i, node in enumerate(search_path):
            alpha = 0.3 + 0.7 * (i / len(search_path))  # Fade in nodes visited later
            ax.scatter(node.point[0], node.point[1], c='green', s=70, alpha=alpha, edgecolors='black')
            
        ax.set_title(f'{k} Nearest Neighbors Search for {query_point}')
        ax.legend()
        
        return ax
    
    def visualize_range_search(self, lower_bound, upper_bound, ax=None):
        """
        Visualize the range search process.
        
        Args:
            lower_bound: Lower bound of the range.
            upper_bound: Upper bound of the range.
            ax: Matplotlib axis to plot on.
        """
        if self.k != 2:
            raise ValueError("Visualization is only supported for 2D trees")
            
        if not isinstance(lower_bound, np.ndarray):
            lower_bound = np.array(lower_bound)
        if not isinstance(upper_bound, np.ndarray):
            upper_bound = np.array(upper_bound)
            
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
            
        # First visualize the tree structure
        self.visualize_tree(ax=ax)
        
        # Perform range search and get the search path
        points_in_range, search_path = self.range_search(lower_bound, upper_bound)
        
        # Draw the range rectangle
        rect_width = upper_bound[0] - lower_bound[0]
        rect_height = upper_bound[1] - lower_bound[1]
        rect = Rectangle((lower_bound[0], lower_bound[1]), rect_width, rect_height, 
                        linewidth=2, edgecolor='purple', facecolor='purple', alpha=0.2,
                        label='Search Range')
        ax.add_patch(rect)
        
        # Plot points in range
        for point in points_in_range:
            ax.scatter(point[0], point[1], c='blue', s=80, marker='o')
            
        # Highlight the search path
        for i, node in enumerate(search_path):
            alpha = 0.3 + 0.7 * (i / len(search_path))  # Fade in nodes visited later
            ax.scatter(node.point[0], node.point[1], c='green', s=70, alpha=alpha, edgecolors='black')
            
        ax.set_title(f'Range Search from {lower_bound} to {upper_bound}')
        ax.legend()
        
        return ax
    
    def __str__(self):
        """
        Return a string representation of the tree.
        """
        if self.root is None:
            return "Empty KD-Tree"
            
        result = []
        
        def _traverse(node, depth=0, prefix="Root: "):
            if node is None:
                return
                
            indent = "  " * depth
            result.append(f"{indent}{prefix}{node.point} (axis={node.axis})")
            
            _traverse(node.left, depth + 1, "L: ")
            _traverse(node.right, depth + 1, "R: ")
            
        _traverse(self.root)
        return "\n".join(result)