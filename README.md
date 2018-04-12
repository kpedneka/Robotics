# Robotics
Robotics machine coding challenges

### 01 - Trilateration
Used the Numpy library to triangulate a location.
Input to the program is in the format: x-coordinate, y-coordinate, distance.
Uses linear algebra concepts to calculate triangulated location.

### 02 - 2D Kalman Filter
Implemented a Kalman filter to reduce discrepencies in data.
Sample output of the program:
Note: green is prediction, red is observed value
![kalman filter prediction](https://github.com/kpedneka/Robotics/blob/master/02/Kalman_Filter.png?raw=true)

### 03 - Rapidly-exploring Random Trees
Implemented collision free motion planning via RRT. The path generated was a result of generating random coordinates, k-nn search, and shortest path algorithms.

Sample output from program:
![RRT result](https://github.com/kpedneka/Robotics/blob/master/03/figure_1.png?raw=true)