import sys, math, numpy

def trilaterate3D(distances):
    origin = [0.,0.,0.,0.]
    # convert to numpy array for greater manipulation
    coord = numpy.array(distances)
    # keep a copy of distances
    d = coord[:,3]
    # delete distances from coord matrix
    coord = numpy.delete(coord, 3, axis=1)
    # get the offset to move first point to origin
    translate = [a_i - b_i for a_i, b_i in zip(origin, coord[0])]
    # translate all points, first point is on origin
    for i in range(4):
        coord[i] = map(sum, zip(coord[i], translate))
    # we want the rotation matrix (around z-axis) with angle pi + arc tan (y,x) for the second point
    # this would ensure first point is origin and second point is on the x axis
    theta = 2*math.pi - math.atan2(coord[1][1], coord[1][0])
    print
    print "theta is: ", theta
    rotationZ = [[math.cos(theta), -math.sin(theta), 0],
                [math.sin(theta), math.cos(theta), 0],
                [0, 0, 1]]
    print
    print "rotation matrix:\n",(numpy.matrix(rotationZ))
    # perform rotation
    coord = numpy.transpose(coord)
    print
    print "rotated coordinates (columns are x,y,z): \n", numpy.matrix(coord)
    # rotate the array
    coord = numpy.dot(rotationZ, coord)
    print
    numpy.set_printoptions(suppress=True)
    print "after rotation(column are x,y,z):\n",numpy.matrix(coord)
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
