import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from day_8_cosine_similarity.practice import VectorTuple

class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def distance_to_origin(self):
        vector_o = tuple([0, 0])
        normal_distance = sum((point - vector_o[i])**2 for i, point in enumerate([self.x, self.y]))

        return VectorTuple.square_root(normal_distance)

def main():
    point_a = Point(1, 2)
    point_b = Point(4, 5)
    
    print(point_a.distance_to_origin())
    print(point_b.distance_to_origin())

if __name__ == "__main__":
    main()
        