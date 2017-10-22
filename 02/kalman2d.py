import sys, matplotlib.pyplot as plt, numpy as np

def kalman(data):
    x_guess = np.matrix([[x10], [x20]])
    P0_guess = np.matrix(np.identity(2)) * scaler

    # these are the variables we want to keep track of

    A = np.identity(2)
    B = np.identity(2)
    H = np.identity(2)
    Q = np.matrix([[0.0001, 0.00002], [0.00002, 0.0001]])
    R = np.matrix([[0.01, 0.005], [0.005, 0.02]])
    kalman_x = []
    kalman_y = []
    u = []
    z = []
    x_hat = []
    x_hat_pred = []
    p_pred = []
    K = []
    P = []
    z_1 = []
    z_2 = []
    for line in range(0, len(data)):
        u.append(np.matrix([data[line][0], data[line][1]]))
        z.append(np.matrix([data[line][2], data[line][3]]))
        z_1.append(float(data[line][2]))
        z_2.append(float(data[line][3]))

    # first iteration uses our guess values for x1, x2, p0

    x_hat_pred.append(np.dot(A, x_guess) + np.dot(B, u[0].T)) # need to double check math here
    p_pred.append(np.dot(A, np.dot(P0_guess, A.T)) + Q)
    K.append(np.divide(np.dot(p_pred[0], H.T), np.dot(H, np.dot(p_pred[0], H.T) + R)))
    P.append(np.dot((np.identity(2) - np.dot(K[0], H)), p_pred[0]))
    x_hat.append(x_hat_pred[0] + np.dot(K[0], (z[0] - np.dot(H, x_hat_pred[0]))))

    for k in range(1, len(data)):
        x_hat_pred.append(np.dot(A, x_hat[k-1]) + np.dot(B, u[k-1].T)) # need to double check math here
        p_pred.append(np.dot(A, np.dot(P[k-1], A.T)) + Q)
        K.append(np.divide(np.dot(p_pred[k], H.T), np.dot(H, np.dot(p_pred[k], H.T) + R)))
        P.append(np.dot((np.identity(2) - np.dot(K[k], H)), p_pred[k]))
        x_hat.append(x_hat_pred[k] + np.dot(K[k], (z[k] - np.dot(H, x_hat_pred[k]))))

    x_hat = np.array(x_hat)
    #print x_hat
    for item in x_hat:
        print item[0][0]
        kalman_x.append(item[0][0])
        print item[1][1]
        kalman_y.append(item[1][1])

    plt.plot(z_1,z_2,'r-')
    plt.plot(kalman_x,kalman_y,'g-')
    plt.show()

if __name__ == "__main__":

    

    # Retrive file name for input data

    if(len(sys.argv) < 5):

        print "Four arguments required: python kalman2d.py [datafile] [x1] [x2] [lambda]"

        exit()

    

    filename = sys.argv[1]

    x10 = float(sys.argv[2])

    x20 = float(sys.argv[3])

    scaler = float(sys.argv[4])



    # Read data

    lines = [line.rstrip('\n') for line in open(filename)]

    data = []

    for line in range(0, len(lines)):

        data.append(map(float, lines[line].split(' ')))



    # Print out the data

    print "The input data points in the format of 'k [u1, u2, z1, z2]', are:"

    for it in range(0, len(data)):

        print str(it + 1) + " " + str(data[it])

    print "\n\n\n"

    kalman(data)

