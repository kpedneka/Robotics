import sys, math, numpy

def trilaterate3D(distances):
    origin = [0.,0.,0.]
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
    print "\ntheta is: ", numpy.degrees(theta)
    rotationZ = [[math.cos(theta), -math.sin(theta), 0],
                [math.sin(theta), math.cos(theta), 0],
                [0, 0, 1]]
    # perform rotation
    coord = numpy.transpose(coord)
    # rotate the array
    coord = numpy.transpose(numpy.dot(rotationZ, coord))

    numpy.set_printoptions(suppress=True)
    print "\nafter rotation(rows are x,y,z):\n",numpy.matrix(coord)

    # x = ( a^2 + d^2 - b^2 ) / 2d
    x = ( math.pow(d[0],2) + math.pow(coord[1][0],2) - math.pow(d[1],2) ) / ( 2*coord[1][0] )
    y = math.sqrt((d[0]**2 - x**2))
    z = math.sqrt(d[0]**2 - x**2 - y**2)
    #need to unrotate, untranslate x and y to get back what we want
    theta2 =  2*math.pi - theta
    rotationZ = [[math.cos(theta2), -math.sin(theta2), 0],
                [math.sin(theta2), math.cos(theta2), 0],
                [0, 0, 1]]
    print "\ntheta is:\t", numpy.degrees(theta2)
    unrotate = numpy.dot(rotationZ, numpy.transpose([x,y,0]))
    untranslate = [a_i - b_i for a_i, b_i in zip(unrotate, translate)]
    x = untranslate[0]
    y = untranslate[1]
    z = untranslate[2]
    return [x,y,z]

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
