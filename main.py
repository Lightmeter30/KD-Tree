import numpy as np
import matplotlib.pyplot as plt

from kdTree import KDTree

def gene_data(low, high, ndata, ndim):
    a = low # a is the lower bound
    b = high # b is the upper bound
    return a + (b - a) * np.random.rand(ndata*ndim).reshape( (ndata,ndim) )

def gene_special_data():
    res = []
    for i in range(20):
        res.append([-i, 2])
    return np.array(res)

def data_2D_test():
    points = gene_data(-100, 100, 1000, 2)
    # Create a figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    kdtree = KDTree(points)
    # Subplot 1: Visualize tree structure
    kdtree.visualize_tree(ax=axes[0][0], node=kdtree.root)
    axes[0][0].set_title('KD-Tree Structure')

    # Subplot 2: Visualize nearest neighbor search
    query_point = np.array([14, 15])
    kdtree.visualize_nearest_neighbor(query_point, ax=axes[0][1])
    axes[0][1].set_title(f'Nearest Neighbor Search for {query_point}')
    # Subplot 3: Visualize k-nearest neighbors search
    kdtree.visualize_k_nearest_neighbors(query_point, k=3, ax=axes[1][0])
    axes[1][0].set_title(f'3 Nearest Neighbors Search for {query_point}')

    # Subplot 4: Visualize range search
    lower = np.array([13, 13])
    upper = np.array([57, 57])
    kdtree.visualize_range_search(lower, upper, ax=axes[1][1])
    axes[1][1].set_title(f'Range Search from {lower} to {upper}')

    # Adjust layout and show
    plt.tight_layout()

    # Example of how to save the visualization to a file
    fig.savefig('kdtree_visualization.png', dpi=300, bbox_inches='tight')

    # Additional example: visualize individual search operations
    plt.figure(figsize=(10, 10))
    query_point = np.array([6, 2])
    kdtree.visualize_nearest_neighbor(query_point)
    plt.title(f'Nearest Neighbor Search for {query_point}')
    plt.tight_layout()

def data_2D_special_test():
    points = gene_special_data()
    # Create a figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    kdtree = KDTree(points)
    # Subplot 1: Visualize tree structure
    kdtree.visualize_tree(ax=axes[0][0], node=kdtree.root)
    axes[0][0].set_title('KD-Tree Structure')

    # Subplot 2: Visualize nearest neighbor search
    query_point = np.array([14, 15])
    kdtree.visualize_nearest_neighbor(query_point, ax=axes[0][1])
    axes[0][1].set_title(f'Nearest Neighbor Search for {query_point}')
    # Subplot 3: Visualize k-nearest neighbors search
    kdtree.visualize_k_nearest_neighbors(query_point, k=3, ax=axes[1][0])
    axes[1][0].set_title(f'3 Nearest Neighbors Search for {query_point}')

    # Subplot 4: Visualize range search
    lower = np.array([13, 13])
    upper = np.array([57, 57])
    kdtree.visualize_range_search(lower, upper, ax=axes[1][1])
    axes[1][1].set_title(f'Range Search from {lower} to {upper}')

    # Adjust layout and show
    plt.tight_layout()

    # Example of how to save the visualization to a file
    fig.savefig('kdtree_special_visualization.png', dpi=300, bbox_inches='tight')

    # Additional example: visualize individual search operations
    plt.figure(figsize=(10, 10))
    query_point = np.array([6, 2])
    kdtree.visualize_nearest_neighbor(query_point)
    plt.title(f'Nearest Neighbor Search for {query_point}')
    plt.tight_layout()

def data_3D_test():
    # Create sample points (2D for visualization)
    points = gene_data(-100, 100, 1000, 3)
    # points = gene_spetical_data()

    # Build KD-Tree
    kdtree = KDTree(points)
    
    # Print tree structure
    print("KD-Tree Structure:")
    print(kdtree)
    
    # 3 dimensional data
    query_point = np.array([4.5, 5.5, 6.5])
    (nearest_point, distance), search_path = kdtree.nearest_neighbor(query_point)
    
    k = 3 # find k nearest neighbors
    k_nearest_points, search_path = kdtree.k_nearest_neighbors(query_point, k)
    
    # Range search
    lower = np.array([3, 3, 3])
    upper = np.array([7, 7, 7])
    range_points, search_path = kdtree.range_search(lower, upper)
# Example usage
if __name__ == "__main__":
    data_2D_special_test()
    data_2D_test()
    data_3D_test()