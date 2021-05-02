import numpy as np
import csv
import math
import math
from math import floor
from random import randint
from decimal import Decimal
from datetime import datetime
from random import seed
from random import random
from numpy.linalg import pinv
from matplotlib import pyplot

class OptTechnique:
    def __init__(self):
        self.xmax = []
        self.xmin = []
        self.ymax = 0.00
        self.ymin = 0.00
        
def createTrainingMatrix(self):    
        # load the dataset from the CSV file
        reader = csv.reader(open("C:/Users/CONALDES/Documents/malawiFlood/Train.csv", "r"), delimiter=",")

        xx = list(reader)
        xxln = len(xx)
        
        allrecs = []
        for row in range(1, xxln):
            fields = []
            recln = len(xx[row])
            for i in range(0, recln):
                fields.append(xx[row][i])    
            allrecs.append(fields)

        '''        
        x = []
        y = []
        squreid = []
        xxln = len(allrecs)
        for row in range(1, xxln):
            temp = []
            rowln = len(allrecs[row])
            for i in range(0, rowln):
                print('i, (i != 2 and i != 39): ' + str(i) + ', ' + str((i != 2 and i != 39)))
                if((i != 2 and i != 39) == True):
                    temp.append(float(allrecs[row][i]))
                elif(i == 2):
                    y.append(float(allrecs[row][i]))
                elif(i == 39):
                    squreid.append(float(allrecs[row][i]))
                    
            x.append(temp)
        '''    
                       
        #xln = len(x)
        x = np.array(allrecs)
        target = x[:,2]
        y = np.vstack(target).astype("float")
        longtde = x[:,0]
        longtde = np.vstack(longtde).astype("float")
        latitde = x[:,1]
        latitde = np.vstack(latitde).astype("float")
        elevton = x[:,3]
        elevton = np.vstack(elevton).astype("float")
        lctmode = x[:,38]
        lctmode = np.vstack(lctmode).astype("float")
        precp15 = x[:,4:20]
        p15row, p15col = precp15.shape        
        precip_15 = []
        for i in range(0, p15row):
            temp = []
            for j in range(0, p15col):
                temp.append(precp15[i][j])
            precip_15.append(temp)
        precip_15 = np.array(precip_15).astype("float")

        precp19 = x[:,21:37]
        p19row, p19col = precp19.shape        
        precip_19 = []
        for i in range(0, p19row):
            temp = []
            for j in range(0, p19col):
                temp.append(precp19[i][j])
            precip_19.append(temp)
        precip_19 = np.array(precip_19).astype("float")

        squreid = x[:,39]
        squreid = np.vstack(squreid)

        print('longtde: ' + str(longtde))
        print('latitde: ' + str(latitde))
        print('target: ' + str(y))
        print('elevton: ' + str(elevton))
        print('precip_15: ' + str(precip_15))
        print('precip_19: ' + str(precip_19))
        print('lctmode: ' + str(lctmode))
        print('squreid: ' + str(squreid))

        print('longtde.shape: ' + str(longtde.shape))
        print('latitde.shape: ' + str(latitde.shape))
        print('target.shape: ' + str(y.shape))
        print('elevton.shape: ' + str(elevton.shape))
        print('precip_15.shape: ' + str(precip_15.shape))
        print('precip_19.shape: ' + str(precip_19.shape))
        print('lctmode.shape: ' + str(lctmode.shape))
        print('squreid.shape: ' + str(squreid.shape))
        
        input('Press enter to continue......')
        
        x = np.concatenate((longtde, latitde, elevton, precip_15, lctmode), axis=1)
        x_test = np.concatenate((longtde, latitde, elevton, precip_19, lctmode), axis=1)
                
        xrow, xcol = x.shape
        print('@@@@@@@@@@@ Un-normalised data @@@@@@@@@@@')
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print('x: ' + str(x))
        print('                           ')
        print('y: ' + str(y))
        print('x.shape: ' + str(x.shape))
        
        xmax = []
        xmin = []
        for k in range(0, xcol):
            xmax.append(np.max(x[:,k]))
            xmin.append(np.min(x[:,k]))
        print('xmax: ' + str(xmax))
        print('xmin: ' + str(xmin))
        ymax = np.max(y)
        ymin = np.min(y)
        
        xrows, xcols = x.shape
        for c in range(xrows):
            for i in range(xcols): 
                x[c][i] = (x[c][i] - xmin[i])/(xmax[i] - xmin[i])                

        yln = len(y)  
        for c in range(yln):
            y[c] = (y[c] - ymin)/(ymax - ymin)

        print('                           ')    
        print('@@@@@@@@@@@ Normalised data @@@@@@@@@@@')
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print('x: ' + str(x))
        print('                           ')
        print('y: ' + str(y))
       
        saveDataSetMetadata(self,xmax,xmin,ymax,ymin)
        return x, y, x_test, squreid

def createTestingMatrix(self):    
        # load the dataset from the CSV file
        reader = csv.reader(open("C:/Users/User/Documents/malawiFlood/Train.csv", "r"), delimiter=",")
        xx = list(reader)
        xxln = len(xx)
        x_test = []
        squareid = []
        for row in range(1, xxln):
            fields = []
            recln = len(xx[row])
            for i in range(0, recln):
                if i == 0 or i == 1 or i == 3 or i == 23 or i == 24 or i == 25 or i == 26 or i == 27 or i == 38 or i == 39:
                    if i == 39:
                        squareid.append(xx[row][i])
                    else:
                       fields.append(xx[row][i]) 
            x_test.append(fields)
        xln = len(x_test)
        x_test = np.array(x_test).astype("float")
        squareid = np.array(squareid)
        long = x_test[:,0]
        long = np.vstack(long)
        lati = x_test[:,1]
        lati = np.vstack(lati)
        elev = x_test[:,2]
        elev = np.vstack(elev)
        prec = x_test[:,7]
        prec = np.vstack(prec)
        '''
        precip = x_test[:,3:7]
        prow, pcol = precip.shape
        prec = []
        for i in range(0, prow):
            temp = 0
            for j in range(0, pcol):
                temp = temp + precip[i][j]
            prec.append(temp)
        prec = np.vstack(prec)
        '''
        lctmd = x_test[:,8]
        lctmd = np.vstack(lctmd)

        x_test = np.concatenate((long, lati, elev, prec, lctmd), axis=1)
        #x_test = np.concatenate((long, lati, elev, prec, lctmd), axis=1)
        xrow, xcol = x_test.shape
        #print("lcType1Mode.shape: " + str(lcType1Mode.shape))
        #print("dataX19.shape: " + str(dataX19.shape))
        return x_test, squareid

def saveDataSetMetadata(self,xmax,xmin,ymax,ymin):        
    self.xmax = xmax
    self.xmin = xmin
    self.ymax = ymax
    self.ymin = ymin
   
def predict_targets(self, testdata, coefs):
    tdln = len(testdata)    
    xtest = []
    
    for i in range(0, tdln):        
        temp = (testdata[i] - self.xmin[i])/(self.xmax[i] - self.xmin[i])
        xtest.append(temp)
                
    predy = 0.00
    cfln = len(coefs)
    #print("len(xtest) = " + str(len(xtest)))
    #print("len(coefs) = " + str(len(coefs)))
    predy = predy + coefs[0]
    for i in range(0, (cfln - 1)):
        predy = predy + xtest[i]*coefs[i + 1]    # xtest[i] i = 0 to 46 and coefs[i] i = 1 to 47
        
    predy = (predy*(self.ymax - self.ymin)) + self.ymin
    return predy

def roundup(a, digits=0):
    #n = 10**-digits
    #return round(math.ceil(a / n) * n, digits)
    return round(a, digits)

'''
def invert(a):
    n, acol = a.shape
    x = np.zeros((n, n))
    b = np.zeros((n, n))
    #index = []
    for i in range(n): 
        b[i][i] = 1;
 
    # Transform the matrix into an upper triangle
    a, index = gaussian(a)
    #a, index = gaussian(a, index)
    lstln = len(index)
    print("invert a.shape = " + str(a.shape))
    print("len(index) = " + str(len(index)))
    # Update the matrix b[i][j] using the ratios stored
    for i in range(0, n-1):
        for j in range(i+1, n):
            for k in range(0,n):
                b[index[j]][k] = b[index[j]][k] - a[index[j]][i]*b[index[i]][k]
 
    # Perform backward substitutions
    for i in range(n):
        x[n-1][i] = b[index[n-1]][i]/a[index[n-1]][n-1]
        #for l in range(len(nn_structure), 0, -1):
        for j in range(n-2, 0, -1):
        
            x[j][i] = b[index[j]][i]
            for k in range(j+1, n):
                x[j][i] = x[j][i] - a[index[j]][k]*x[k][i]
            
            x[j][i] = x[j][i]/a[index[j]][j]
        
    return x

# Method to carry out the partial-pivoting Gaussian
# elimination.  Here index[] stores pivoting order.
# def gaussian(a, index):
def gaussian(a):
    #n = len(index)
    index = []
    n, acl = a.shape
    c = []
    print("gaussian a.shape = " + str(a.shape))
    # Initialize the index
    for i in range(n): 
        index.append(i)
 
    # Find the rescaling factors, one from each row
    for i in range(n):
        c1 = 0;
        for j in range(n):
            c0 = math.fabs(a[i][j])
            if (c0 > c1):
                c1 = c0
        
        c.append(c1)
    
 
    # Search the pivoting element from each column
    k = 0
    for j in range(n-1):
        pi1 = 0
        for i in range(j, n):
            pi0 = math.fabs(a[index[i]][j])
            pi0 = pi0/c[index[i]]
            if (pi0 > pi1):
                pi1 = pi0
                k = i        
 
        # Interchange rows according to the pivoting order
        itmp = index[j]
        index[j] = index[k]
        index[k] = itmp
        for i in range(j+1, n):
            pj = a[index[i]][j]/a[index[j]][j]
 
            # Record pivoting ratios below the diagonal
            a[index[i]][j] = pj
 
            # Modify other elements accordingly
            for l in range(j+1, n):
                a[index[i]][l] = a[index[i]][l] - pj*a[index[j]][l]
    return a, index
'''

def simulateModel(self,x,y,x_data):
    # current date and time
    print('                           ')    
    print('@@@@@@@@@@@ Model Simulation @@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    now0 = datetime.now()
    timestamp0 = datetime.timestamp(now0)
    brow, bcol = x.shape
    m = np.ones((brow,1))
    #x = np.array(x_train)            # x
    #print('x: ' + str(x))
    #print('x.shape: ' + str(x.shape))
    x = np.concatenate((m,x),axis=1)

    print('x (1 column added): ' + str(x))

    '''
    print('x (1 column added): ' + str(x))
    d = x.T                           # x.T     
    print('x.T: ' + str(d))
    #print('b.T.shape: ' + str(d.shape))
    e = np.linalg.inv(np.matmul(d,x)) # inverse(x.Tx)
    print('Inverse of (x.T)x: ' + str(e))
    #print('e.shape: ' + str(e.shape))
    #y1 = np.array(y1)
    #print('y.shape: ' + str(y.shape))
    #print('y: ' + str(y))
    #y1 = y1.T
    #print('y1.T.shape: ' + str(y1.shape))

    f = np.matmul(e,d)                # inverse(x.Tx)x
    c = np.matmul(f,y)              # (inverse(x.Tx)x)y
    
    #print('f.shape: ' + str(f.shape))
    print('Inverse of (x.T)x) multiplied by x: ' + str(f))
    #print('c.shape: ' + str(c.shape))
    '''
    #c, residuals, rank, s = lstsq(x, y)
    c = pinv(x).dot(y)
    print('Coefficients: ' + str(c))
    print('len(Coefficients): ' + str(len(c)))
    
    # predict using coefficients
    #x_data, wards = createTestingMatrix(self)
        
    #x_data, wards = createTestingMatrix(self)
    #print("x_data.shape: " + str(x_data.shape))
    #xdtrow, xdtcol = x_data.shape
    #print("xdtrow, xdtcol: " + str(xdtrow) + ", " + str(xdtcol))
    xdtln = len(x_data)
    print('len(x_data): ' + str(len(x_data)))
    print('x_data: ' + str(x_data))
    predicted_targets = np.zeros((xdtln, 1))
    #for i in range(0, xdtln):
    #    print(str(wards[i]))  
    for i in range(0, xdtln):
        pred_targets = predict_targets(nn, x_data[i], c)
        if pred_targets < 0:
            pred_targets = 0.00

        #print(str(pred_targets))    
        predicted_targets[i][0] = pred_targets
    now1 = datetime.now()
    timestamp1 = datetime.timestamp(now1)
    time_elapsed = timestamp1 - timestamp0
    print('Time elapsed for computations: ' + str(time_elapsed) + 'seconds')
    return predicted_targets

nn = OptTechnique()

dataTrainX, dataTrainY, x_data, wardids = createTrainingMatrix(nn)
#predicted_targets, wardids = simulateModel(nn,dataTrainX,dataTrainY)
predicted_targets = simulateModel(nn,dataTrainX,dataTrainY, x_data)
wardids = np.vstack(wardids)
#print("wardids.shape: " + str(wardids.shape))
#print("predicted_targets.shape: " + str(predicted_targets.shape))
wardids_targets_arr = np.concatenate((wardids, predicted_targets), axis=1)
#print("wardids_targets_arr: " + str(wardids_targets_arr.shape))
with open("C:/Users/CONALDES/Documents/malawiFlood/ConaldesSubmission.csv", "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Square_ID", "target_2019"]) 
    for row in wardids_targets_arr:    
        l = list(row)    
        writer.writerow(l)

print("                                          ")
print("### C:/Users/CONALDES/Documents/malawiFlood/ConaldesSubmission.csv contains results ###") 
