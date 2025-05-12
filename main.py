import numpy as np
import matplotlib.pyplot as plt
import time
from kdTree import KDTree
from brute import BruteForceSearch

def gene_data(low, high, ndata, ndim):
    a = low # a is the lower bound
    b = high # b is the upper bound
    return a + (b - a) * np.random.rand(ndata*ndim).reshape( (ndata,ndim) )

def gene_special_data():
    res = []
    for i in range(20):
        res.append([-i, 2])
    return np.array(res)

def data_2D_test(points):
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
    query_point = np.array([-5.5, 2.02])
    kdtree.visualize_nearest_neighbor(query_point, ax=axes[0][1])
    axes[0][1].set_title(f'Nearest Neighbor Search for {query_point}')
    # Subplot 3: Visualize k-nearest neighbors search
    kdtree.visualize_k_nearest_neighbors(query_point, k=3, ax=axes[1][0])
    axes[1][0].set_title(f'3 Nearest Neighbors Search for {query_point}')

    # Subplot 4: Visualize range search
    lower = np.array([-5.5, 1.93])
    upper = np.array([-1.1, 2.03])
    kdtree.visualize_range_search(lower, upper, ax=axes[1][1])
    axes[1][1].set_title(f'Range Search from {lower} to {upper}')

    # Adjust layout and show
    plt.tight_layout()

    # Example of how to save the visualization to a file
    fig.savefig('kdtree_special_visualization.png', dpi=300, bbox_inches='tight')

def data_3D_test(points):
    # Create sample points (2D for visualization)
    # points = gene_spetical_data()

    # Build KD-Tree
    kdtree = KDTree(points)
    
    # Print tree structure
    # print("KD-Tree Structure:")
    # print(kdtree)
    
    # 3 dimensional data
    query_point = np.array([45, 55, 65])
    (nearest_point, distance), search_path = kdtree.nearest_neighbor(query_point)
    
    k = 3 # find k nearest neighbors
    k_nearest_points, search_path = kdtree.k_nearest_neighbors(query_point, k)
    
    # Range search
    lower = np.array([30, 30, 30])
    upper = np.array([35, 35, 35])
    range_points, search_path = kdtree.range_search(lower, upper)

def performance_analysis(points, num_queries=100, filename="performance_comparison.png"):
    kdtree = KDTree(points)
    brute_force = BruteForceSearch(points)
    k = 3  # for k-nearest neighbors
    
    # Generate random query points
    query_points = gene_data(-100, 100, num_queries, points.shape[1])
    
    # Initialize timing results
    nn_times = {'kdtree': [], 'brute_force': []}
    knn_times = {'kdtree': [], 'brute_force': []}
    range_times = {'kdtree': [], 'brute_force': []}
    
    # Initialize correctness counters
    nn_correct = 0
    knn_correct = 0
    range_correct = 0
    
    for query_point in query_points:
        # Nearest Neighbor
        start_time = time.time()
        nn_kdtree, _ = kdtree.nearest_neighbor(query_point)
        nn_times['kdtree'].append(time.time() - start_time)
        
        start_time = time.time()
        nn_brute, _ = brute_force.nearest_neighbor(query_point)
        nn_times['brute_force'].append(time.time() - start_time)
        
        if np.array_equal(nn_kdtree[0], nn_brute[0]):
            nn_correct += 1
            
        # K-Nearest Neighbors
        start_time = time.time()
        knn_kdtree, _ = kdtree.k_nearest_neighbors(query_point, k)
        knn_times['kdtree'].append(time.time() - start_time)
        
        start_time = time.time()
        knn_brute, _ = brute_force.k_nearest_neighbors(query_point, k)
        knn_times['brute_force'].append(time.time() - start_time)
        
        if all(np.array_equal(kdtree_point[0], brute_point[0]) 
               for kdtree_point, brute_point in zip(knn_kdtree, knn_brute)):
            knn_correct += 1
            
        # Range Search
        lower = query_point - 10
        upper = query_point + 10
        
        start_time = time.time()
        range_kdtree, _ = kdtree.range_search(lower, upper)
        range_times['kdtree'].append(time.time() - start_time)
        
        start_time = time.time()
        range_brute, _ = brute_force.range_search(lower, upper)
        range_times['brute_force'].append(time.time() - start_time)
        
        if len(range_kdtree) == len(range_brute) and \
           all(any(np.array_equal(p1, p2) for p2 in range_brute) 
               for p1 in range_kdtree):
            range_correct += 1
    
    # Print results
    print("\nPerformance Analysis Results:")
    print("-" * 50)
    
    print("\nNearest Neighbor Search:")
    print(f"KD-Tree avg time: {np.mean(nn_times['kdtree']):.6f} seconds")
    print(f"Brute Force avg time: {np.mean(nn_times['brute_force']):.6f} seconds")
    print(f"Speedup: {np.mean(nn_times['brute_force']) / np.mean(nn_times['kdtree']):.2f}x")
    print(f"Correctness: {nn_correct}/{num_queries} ({nn_correct/num_queries*100:.1f}%)")
    
    print("\nK-Nearest Neighbors Search:")
    print(f"KD-Tree avg time: {np.mean(knn_times['kdtree']):.6f} seconds")
    print(f"Brute Force avg time: {np.mean(knn_times['brute_force']):.6f} seconds")
    print(f"Speedup: {np.mean(knn_times['brute_force']) / np.mean(knn_times['kdtree']):.2f}x")
    print(f"Correctness: {knn_correct}/{num_queries} ({knn_correct/num_queries*100:.1f}%)")
    
    print("\nRange Search:")
    print(f"KD-Tree avg time: {np.mean(range_times['kdtree']):.6f} seconds")
    print(f"Brute Force avg time: {np.mean(range_times['brute_force']):.6f} seconds")
    print(f"Speedup: {np.mean(range_times['brute_force']) / np.mean(range_times['kdtree']):.2f}x")
    print(f"Correctness: {range_correct}/{num_queries} ({range_correct/num_queries*100:.1f}%)")
    
    # Plot performance comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Nearest Neighbor
    axes[0].boxplot([nn_times['kdtree'], nn_times['brute_force']], 
                    labels=['KD-Tree', 'Brute Force'])
    axes[0].set_title('Nearest Neighbor Search')
    axes[0].set_ylabel('Time (seconds)')
    
    # K-Nearest Neighbors
    axes[1].boxplot([knn_times['kdtree'], knn_times['brute_force']], 
                    labels=['KD-Tree', 'Brute Force'])
    axes[1].set_title('K-Nearest Neighbors Search')
    axes[1].set_ylabel('Time (seconds)')
    
    # Range Search
    axes[2].boxplot([range_times['kdtree'], range_times['brute_force']], 
                    labels=['KD-Tree', 'Brute Force'])
    axes[2].set_title('Range Search')
    axes[2].set_ylabel('Time (seconds)')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # special 2D data
    data_2D_special_test()
    
    # 2D data
    points = gene_data(-100, 100, 1000, 2)
    data_2D_test(points)
    
    # # Performance analysis for 2D data
    print("\nPerforming performance analysis for 2D data...")
    performance_analysis(points, filename="performance_comparison_2D.png")
    
    # 4D data
    points = gene_data(-50, 50, 10000, 4)
    
    # Performance analysis for 4D data
    print("\nPerforming performance analysis for 4D data...")
    performance_analysis(points, filename="performance_comparison_4D.png")
    
    # 10D data
    points = gene_data(-50, 50, 100000, 10)
    
    # Performance analysis for 4D data
    print("\nPerforming performance analysis for 10D data...")
    performance_analysis(points, filename="performance_comparison_10D.png")