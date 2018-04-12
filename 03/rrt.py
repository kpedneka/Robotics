import sys

import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
import math

'''
Set up matplotlib to create a plot with an empty square
'''
def setupPlot():
    fig = plt.figure(num=None, figsize=(5, 5), dpi=120, facecolor='w', edgecolor='k')
    plt.autoscale(False)
    plt.axis('off')
    ax = fig.add_subplot(1,1,1)
    ax.set_axis_off()
    ax.add_patch(patches.Rectangle(
        (0,0),   # (x,y)
        1,          # width
        1,          # height
        fill=False
        ))
    return fig, ax

'''
Make a patch for a single pology 
'''
def createPolygonPatch(polygon, color):
    verts = []
    codes= []
    for v in range(0, len(polygon)):
        xy = polygon[v]
        verts.append((xy[0]/10., xy[1]/10.))
        if v == 0:
            codes.append(Path.MOVETO)
        else:
            codes.append(Path.LINETO)
    verts.append(verts[0])
    codes.append(Path.CLOSEPOLY)
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor=color, lw=1, alpha=0.7)
    return patch
    

'''
Render the problem  
'''
def drawProblem(robotStart, robotGoal, polygons):
    fig, ax = setupPlot()
    patch = createPolygonPatch(robotStart, 'green')
    ax.add_patch(patch)    
    patch = createPolygonPatch(robotGoal, 'red')
    ax.add_patch(patch)    
    for p in range(0, len(polygons)):
        patch = createPolygonPatch(polygons[p], 'gray')
        ax.add_patch(patch)  
    #remove before submitting
    plt.show()

'''
Grow a simple RRT 
'''
def growSimpleRRT(points):
    newPoints = dict()
    adjListMap = dict()

    # This array allows us to quickly find 1-NN
    pointArr = []
    pointArr.append(points[1])
    # add root point to newPoints and [] to it's adjacency list index
    count = 1
    newPoints[count] = points[1]
    adjListMap[count] = []

    for i in range (2, len(points)+1):
        # find nearest neighbor in existing tree
        NN = findNearest(points[i], pointArr)
        # create an edge for all the neighbors of NN in adjacency list
        key = newPoints.keys()[newPoints.values().index(NN)]
        # vertex has no neighbors at all (isolated vertex)
        if len(adjListMap[key]) == 0:
            # print "no neighbors", key
            # add points[i] to newPoints and make both these vertices neighbors of each other
            count = count + 1
            newPoints[count] = points[i]
            adjListMap[key] = [count]
            adjListMap[count] = [key]
            pointArr.append(points[i])
        else:
            x, y = -1, -1
            dist = 100
            for vertex in adjListMap[key]:
                newx, newy = Nearest_Point([NN, newPoints[vertex]], points[i])
                newdist = np.linalg.norm(np.array((newx, newy))-np.array(points[i]))
                if newdist < dist:
                    dist = newdist
                    x, y = newx, newy
            # add x, y as a point on RRT
            if x >= 0 and y >= 0:
                try:
                    checkKey = newPoints.keys()[newPoints.values().index((x,y))]
                except:
                    newPoints[count+1] = (x, y)
                    count = count + 1
                    checkKey = count
                newPoints[count+1] = points[i]
                pointArr.append(points[i])
                pointArr.append((x,y))
                # update adjListMap for all points
                adjListMap[checkKey] = [key, vertex, count+1]
                adjListMap[count+1] =  [checkKey]
                # remove vertex from NN's adjlist, add (x, y) only if (x, y) is not NN itself
                if x != NN[0] and y != NN[1]:
                    arr = adjListMap[key]
                    arr.remove(vertex)
                    arr.append(checkKey)
                    adjListMap[key] = arr
                else:
                    arr = adjListMap[key]
                    arr.append(checkKey)
                    adjListMap[key] = arr
                # remove NN from vertex's adjlist, add (x, y)
                arr = adjListMap[vertex]
                arr.remove(key)
                arr.append(checkKey)
                adjListMap[vertex] = arr
                # update count for indexes of newPoints
                count = count+1

    return newPoints, adjListMap


'''
Helper function to find the nearest distance point between a line segment and a point
'''
def Nearest_Point(segment, point):
    x1 = segment[0][0]
    y1 = segment[0][1]
    x2 = segment[1][0]
    y2 = segment[1][1]
    px = point[0]
    py = point[1]

    diff_x = x2 - x1
    diff_y = y2 - y1

    if (diff_x == 0) and (diff_y == 0):
	return -1, -1
    slope = ((px - x1) * diff_x + (py - y1) * diff_y) / (diff_x * diff_x + diff_y * diff_y)

    if slope < 0:
	return x1, y1
    elif slope > 1:
	return x2, y2
    else:
	return x1 + slope * diff_x , y1 + slope * diff_y 
'''
Helper function for growSimpleRRT to find nearest point
'''
def findNearest(x, D):
    sqd = [np.linalg.norm(np.array(a)-np.array(x)) for a in D]
    idx = sqd.index(min(sqd))
    # return the indexes of K nearest neighbours
    return D[idx]

'''
Perform basic search 
'''
def basicSearch(tree, start, goal):
    path = []
    # Your code goes here. As the result, the function should
    # return a list of vertex labels, e.g.
    #
    # path = [23, 15, 9, ..., 37]
    #
    # in which 23 would be the label for the start and 37 the
    # label for the goal.
    if start==goal:
	return path.append(start)

    queue = [[start]]
    done = []
    while queue:
	path = queue.pop(0)
	node = path[-1]
	if node == goal:
	    return path
	else:
	    if node not in done:
	        child = tree[node]
	        for item in child:
	            newlist = list(path)
	            newlist.append(item)
	            queue.append(newlist)
		done.append(node)
    return []

'''
Display the RRT and Path
'''
def displayRRTandPath(points, tree, path, robotStart = None, robotGoal = None, polygons = None):
    
    # Your code goes here
    # You could start by copying code from the function
    # drawProblem and modify it to do what you need.
    # You should draw the problem when applicable.
    print "robot start is ", robotStart
    fig, ax = setupPlot()
    patch = createPolygonPatch(robotStart, 'green')
    ax.add_patch(patch)
    patch = createPolygonPatch(robotGoal, 'red')
    ax.add_patch(patch)
    
    if polygons != None:
        for p in range(0, len(polygons)):
            patch = createPolygonPatch(polygons[p], 'gray')
            ax.add_patch(patch)

    for k, v in points.iteritems():
        for n in tree[k]:
            x = [v[0]/10., points[n][0]/10.]
            y = [v[1]/10., points[n][1]/10.]
            l = Line2D(x,y, marker=".", linestyle="solid", color="black")
            ax.add_line(l)
    if path != None:
        for i in range(0, len(path)-1):
            p1 = points[path[i]]
            p2 = points[path[i+1]]
            x = [p1[0]/10., p2[0]/10.]
            y = [p1[1]/10., p2[1]/10.]
            l = Line2D(x,y, marker=".", linestyle="solid", color="orange")
            ax.add_line(l)

    plt.show()

'''
Compute center of the robot
'''
def centerOfRobot(robot):
    return robot[1][0], robot[0][1]
'''
Translate robot to a point
'''
def translateRobot(robot, point):
    x_init, y_init = centerOfRobot(robot)
    x_diff = point[0]-x_init
    y_diff = point[1]-y_init
    for i in range(len(robot)):
        x = robot[i][0]
        y = robot[i][1]
        robot[i] = (x+x_diff, y+y_diff)
    return robot
'''
Collision checking
'''
def isCollisionFree(robot, point, obstacles):
    print point
    robot = translateRobot(robot, point)
    if outOfBounds(robot):
        return True
    # Your code goes here.
    for i in range(0,len(robot)):
        p1 = robot[i]
        p2 = robot[i - 1]
        for obstacle in obstacles:
            for i in range(0, len(obstacle)):
                o2 = obstacle[i]
                o1 = obstacle[i - 1]
                x, y = findIntersection([o1,o2],[p1,p2])
                if x > 0 and y > 0 or inside(getmidpoint(p1,p2), obstacles) == True:
                    return False
    return True
'''
Detect collision with boundary
'''
def outOfBounds(robot):
    for point in robot:
        if point[0] >= 10 or point[1] >= 10:
            print "out of bounds"
            return True
    return False
'''
Helper function to calculate midpoint
'''
def getmidpoint(point1, point2):
    return (point1[0]+point2[0])/2, (point1[1]+point2[1])/2

'''
Helper function for isCollisionFree
'''
def findIntersection(line1, line2):
    #print line1, line2
    # Now check for intersection point
    pt1 = line1[0]; pt2 = line1[1]; ptA = line2[0]; ptB = line2[1]
    DET_TOLERANCE = 0.00000001
    x1, y1 = pt1;   x2, y2 = pt2
    dx1 = x2 - x1;  dy1 = y2 - y1

    x, y = ptA;   xB, yB = ptB;
    dx = xB - x;  dy = yB - y;

    DET = (-dx1 * dy + dy1 * dx)
    if math.fabs(DET) < DET_TOLERANCE: return 0,0

    # now, the determinant should be OK
    DETinv = 1.0/DET

    # find the scalar amount along the "self" segment
    r = DETinv * (-dy  * (x-x1) +  dx * (y-y1))

    # find the scalar amount along the input line
    s = DETinv * (-dy1 * (x-x1) + dx1 * (y-y1))
    # return the average of the two descriptions
    xi = (x1 + r*dx1 + x + s*dx)/2.0
    yi = (y1 + r*dy1 + y + s*dy)/2.0
    #print xi, yi
    maxx1 = max(line1[0][0], line1[1][0]); maxx2 = max(line2[0][0], line2[1][0])
    maxy1 = max(line1[0][1], line1[1][1]); maxy2 = max(line2[0][1], line2[1][1])
    minx1 = min(line1[0][0], line1[1][0]); minx2 = min(line2[0][0], line2[1][0])
    miny1 = min(line1[0][1], line1[1][1]); miny2 = min(line2[0][1], line2[1][1])
    if minx1 <= xi <= maxx1 and miny1 <= yi <= maxy1:
        if minx2 <= xi <= maxx2 and miny2 <= yi <= maxy2:
            return xi, yi
        else:
            #print "out of range for line 2 ", line2
            return 0, 0
    else:
        #print "out of range for line 1 ", line1
        return 0, 0
'''
Helper function to check if robot is inside obstacle
'''
def inside(midpoint, polygons):
    for polygon in polygons:
	path = Path(polygon)
	if path.contains_point(midpoint):
	    return True
    return False
'''
Helper Function to check the robot collision on the way
'''
def recursive(robot,point1, point2, obstacles):
   x_f = (point2[0] - point1[0])/20
   y_f = (point2[1] - point1[1])/20
   for i in range(0,20):
	mid = (x1+x_f*i,y1+y_f*i)
	if isCollisionFree(robot, mid, obstacles) == False:
	    return False
   return True
'''
Helper function grow the Full tree
'''
def growfullRRT(points, obstacles, start, goal, robot):
    newPoints = dict()
    adjListMap = dict()
    startr = translateRobot(robot, start)
    goalr = translateRobot(robot, goal)
    Gconnected = False
    Sconnected = False
    startkey = 0
    goalkey = -1
    # This array allows us to quickly find 1-NN
    pointArr = []
    pointArr.append(points[1])
    print "Start", start, "Goal", goal
    # add root point to newPoints and [] to it's adjacency list index
    count = 1
    newPoints[count] = points[1]
    adjListMap[count] = []

    for i in range (2, len(points)+1):

	if (i%5 == 0) and Sconnected == False:
	    print "working on start", i
	    SN = findNearest(start, pointArr)
	    key = newPoints.keys()[newPoints.values().index(SN)]
	    points[i] = start
	elif(i%6 == 0) and Gconnected == False:
	    print "working on goal",i
	    GN = findNearest(goal,pointArr)
	    key = newPoints.keys()[newPoints.values().index(GN)]
	    points[i] = goal
	else:
           # find nearest neighbor in existing tree
           NN = findNearest(points[i], pointArr)
           # create an edge for all the neighbors of NN in adjacency list
           key = newPoints.keys()[newPoints.values().index(NN)]
           # vertex has no neighbors at all (isolated vertex)
        if len(adjListMap[key]) == 0:
            #print "no neighbors", key
            # add points[i] to newPoints and make both these vertices neighbors of each other
	    enter = False
            for obstacle in obstacles:
		for q in range(0, len(obstacle)):
		    o2 = obstacle[q]
		    o1 = obstacle[q-1]
		    c1, c2 = findIntersection([o2,o1],[points[i],points[1]])
		    if c1 > 0 and c2 > 0:
			enter = True 
	    if enter == False and recursive(robot,points[i],points[1],obstacles):
		print
		print
		print "intersection points", c1,c2
		print "vertices are", points[i],points[1]
		count = count + 1 
        	newPoints[count] = points[i]
            	adjListMap[key] = [count]
            	adjListMap[count] = [key]
            	pointArr.append(points[i])
		print "newPoints are ", newPoints
		print "Adjacent List is", adjListMap
		print 
		print
		#displayRRTandPath(newPoints, adjListMap, None, startr, goalr, obstacles)
		#exit()
        else:
            x, y = -1, -1
            dist = 100
	    intersection = False
            for vertex in adjListMap[key]:
                newx, newy = Nearest_Point([NN, newPoints[vertex]], points[i])
                newdist = np.linalg.norm(np.array((newx, newy))-np.array(points[i]))
        	for obstacle in obstacles:
            	    for k in range(0, len(obstacle)):
                	o2 = obstacle[k]
                	o1 = obstacle[k - 1]

			#coll = recursive(robot,(newx,newy), points[i], obstacle)
			#print coll
			#print "Points", points[i],"New point on line", newx, newy
			c1, c2 = findIntersection([o1,o2], [(newx,newy), points[i]])
			#print "Points of intersection",c1,c2
			if c1 > 0 and c2 > 0:
			    intersection = True
			    print c1, c2
			    print "Intersecting the line is True"
			    
			#print "Lines don't collide"
                if newdist < dist and intersection == False:
		    coll = recursive(robot,(newx,newy), points[i], obstacles)
		    print coll
		    if coll == True:
                        dist = newdist
                        x, y = newx, newy
            # add x, y as a point on RRT
            if x >= 0 and y >= 0:
                newPoints[count+1] = (x, y)
                newPoints[count+2] = points[i]
                pointArr.append(points[i])
                pointArr.append((x,y))
                # update adjListMap for all points
                adjListMap[count+1] = [key, vertex, count+2]

                adjListMap[count+2] =  [count+1]
                # remove vertex from NN's adjlist, add (x, y) only if (x, y) is not NN itself
                if x != NN[0] and y != NN[1]:
                    arr = adjListMap[key]
                    arr.remove(vertex)
                    arr.append(count+1)
                    adjListMap[key] = arr
                else:

                    arr = adjListMap[key]
                    arr.append(count+1)
                    adjListMap[key] = arr
                # remove NN from vertex's adjlist, add (x, y)
                arr = adjListMap[vertex]
                if key in arr:
	            arr.remove(key)
                arr.append(count+1)
                adjListMap[vertex] = arr
                # update count for indexes of newPoints
                count = count+2
		if(i%5 == 0) and Sconnected == False:
		    Sconnected = True
		    startkey = count
		    print "Start Added"
		if(i%6 == 0) and Gconnected == False:
		   Gconnected = True
		   goalkey = count
		   print "Gaol Added"
	if Sconnected and Gconnected:
	    print "Start and Goal Added"
	    return startkey, goalkey, newPoints, adjListMap

    return startkey, goalkey, newPoints, adjListMap

'''
The full RRT algorithm
'''

def RRT(robot, obstacles, startPoint, goalPoint):

    points = dict()
    tree = dict()
    path = []
    # Your code goes here.
    j = 1
    randompoints = dict()
    while (len(path) == 0):
        N = 50
        for i in range(1, N):
            x = np.random.uniform(0.0,10.0)
            y = np.random.uniform(0.0,10.0)
	    print "x and y",x,y
            p = (x,y)
            if not inside((float("{0:.2f}".format(p[0])),float("{0:.2f}".format(p[1]))),obstacles) and isCollisionFree(robot, p, obstacles):
                randompoints[j] = (float("{0:.2f}".format(p[0])),float("{0:.2f}".format(p[1])))
                j = j+1
        # The Grow Tree 
        startkey, goalkey, points, tree = growfullRRT(randompoints, obstacles, startPoint, goalPoint, robot)
        print len(tree)
        print startkey,goalkey
        if(len(tree) > 3):
            try:
                path = basicSearch(tree, startkey, goalkey)
                print "path is ", path
            except:
                continue
        else:
            continue

    return points, tree, path

if __name__ == "__main__":
    
    # Retrive file name for input data
    if(len(sys.argv) < 6):
        print "Five arguments required: python spr.py [env-file] [x1] [y1] [x2] [y2]"
        exit()
    
    filename = sys.argv[1]
    x1 = float(sys.argv[2])
    y1 = float(sys.argv[3])
    x2 = float(sys.argv[4])
    y2 = float(sys.argv[5])

    # Read data and parse polygons
    lines = [line.rstrip('\n') for line in open(filename)]
    robot = []
    obstacles = []
    for line in range(0, len(lines)):
        xys = lines[line].split(';')
        polygon = []
        for p in range(0, len(xys)):
            xy = xys[p].split(',')
            polygon.append((float(xy[0]), float(xy[1])))
        if line == 0 :
            robot = polygon
        else:
            obstacles.append(polygon)

    # Print out the data
    print "Robot:"
    print str(robot)
    print "Pologonal obstacles:"
    for p in range(0, len(obstacles)):
        print str(obstacles[p])
    print ""

    # Visualize
    robotStart = []
    robotGoal = []

    def start((x,y)):
        return (x+x1, y+y1)
    def goal((x,y)):
        return (x+x2, y+y2)
    robotStart = map(start, robot)
    robotGoal = map(goal, robot)
    drawProblem(robotStart, robotGoal, obstacles)

    # Example points for calling growSimpleRRT
    # You should expect many mroe points, e.g., 200-500
    points = dict()
    points[1] = (5, 5)
    points[2] = (7, 8.2)
    points[3] = (6.5, 5.2)
    points[4] = (0.3, 4)
    points[5] = (6, 3.7)
    points[6] = (9.7, 6.4)
    points[7] = (4.4, 2.8)
    points[8] = (9.1, 3.1)
    points[9] = (8.1, 6.5)
    points[10] = (0.7, 5.4)
    points[11] = (5.1, 3.9)
    points[12] = (2, 6)
    points[13] = (0.5, 6.7)
    points[14] = (8.3, 2.1)
    points[15] = (7.7, 6.3)
    points[16] = (7.9, 5)
    points[17] = (4.8, 6.1)
    points[18] = (3.2, 9.3)
    points[19] = (7.3, 5.8)
    points[20] = (9, 0.6)

    # Printing the points
    print "" 
    print "The input points are:"
    print str(points)
    print ""
    
    points, adjListMap = growSimpleRRT(points)

    # Search for a solution  
    path = basicSearch(adjListMap, 1, 24)

    print "path done, ",path
    # Your visualization code 
    displayRRTandPath(points, adjListMap, path, translateRobot(robot, points[1]), translateRobot(robot, points[24])) 

    # Solve a real RRT problem
    points, adjListMap, path = RRT(robot, obstacles, (x1, y1), (x2, y2))
    
    # Your visualization code 
    displayRRTandPath(points, adjListMap, path, robotStart, robotGoal, obstacles) 
