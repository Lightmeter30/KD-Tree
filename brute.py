import numpy as np
import time
from myTime import time_decorator

class BruteForceSearch:
    
    def __init__(self, points):
        if not isinstance(points, np.ndarray):
            points = np.array(points)
        self.points = points
    
    @time_decorator
    def nearest_neighbor(self, query_point):
        if not isinstance(query_point, np.ndarray):
            query_point = np.array(query_point)
            
        min_dist = float('inf')
        nearest_point = None
        
        for point in self.points:
            dist = np.sum((query_point - point) ** 2)
            if dist < min_dist:
                min_dist = dist
                nearest_point = point
                
        return (nearest_point, min_dist), []
    
    @time_decorator
    def k_nearest_neighbors(self, query_point, k=1):
        if not isinstance(query_point, np.ndarray):
            query_point = np.array(query_point)
            
        # Initialize with k infinity distances
        k_nearest = [(float('inf'), None) for _ in range(k)]
        
        for point in self.points:
            dist = np.sum((query_point - point) ** 2)
            
            # If current point is closer than the furthest in k_nearest
            if dist < k_nearest[-1][0]:
                # Find the correct position to insert
                for i in range(k):
                    if dist < k_nearest[i][0]:
                        # Shift elements to make space
                        k_nearest[i+1:] = k_nearest[i:-1]
                        # Insert new point
                        k_nearest[i] = (dist, tuple(point))
                        break
        
        # Convert to list of (point, distance) pairs
        result = [(point, dist) for dist, point in k_nearest if point is not None]
        return result, []
    
    @time_decorator
    def range_search(self, lower_bound, upper_bound):
        if not isinstance(lower_bound, np.ndarray):
            lower_bound = np.array(lower_bound)
        if not isinstance(upper_bound, np.ndarray):
            upper_bound = np.array(upper_bound)
            
        result = []
        for point in self.points:
            if np.all(lower_bound <= point) and np.all(point <= upper_bound):
                result.append(point)
                
        return result, []
