import sys, math, numpy

def trilaterate3D(distances):
    origin = [0.,0.,0.,0.]
    #get the offset to move first point to origin
    translate = [a_i - b_i for a_i, b_i in zip(origin, distances[0])]
    #ensure that the distance never gets affected
    translate[3] = 0
    #translate all points, first point is on origin
    for i in range(4):
        distances[i] = map(sum, zip(distances[i], translate))
    #we want the rotation matrix (around z-axis) with angle pi + arc tan (y,x) for the second point
    #this would ensure first point is origin and second point is on the x axis
    theta = math.atan2(distances[1][1], distances[1][0]) + math.pi
    print "theta is: ", theta
    rotationZ = [[math.cos(theta), -math.sin(theta), 0, 0],
                [math.sin(theta), math.cos(theta), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]]
    print"rotation matrix:\n",(numpy.matrix(rotationZ))
    #perform rotation
    distances = numpy.dot(rotationZ, distances)
    print "after rotation:\n",numpy.matrix(distances)
    return [0.,0.,0.,0.]

if __name__ == "__main__":
    
    # Retrive file name for input data
    if(len(sys.argv) == 1):
        print "Please enter data file name."
        exit()
    
    filename = sys.argv[1]

    # Read data
    lines = [line.rstrip('\n') for line in open(filename)]
    distances = []
    for line in range(0, len(lines)):
        distances.append(map(float, lines[line].split(' ')))

    # Print out the data
    print "The input four points and distances, in the format of [x, y, z, d], are:"
    for p in range(0, len(distances)):
        print distances[p] 

    # Call the function and compute the location 
    location = trilaterate3D(distances)
    print 
    print "The location of the point is: " + str(location)
