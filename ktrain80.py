import time
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.inf, suppress=True)
import pandas as pd
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# For plotting
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


# Model Libraries
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.callbacks import TensorBoard, EarlyStopping
from keras import optimizers,regularizers
from keras import backend as K




startTime = time.time()

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Data and model checkpoints directories

parser.add_argument("--l", type=int, default=20,help="Size of the lattice")

parser.add_argument("--neurons", type=int, default=2,help="Hidden Layer Neurons")

parser.add_argument("--epochs", type=int, default=1,help="How many epochs to run the network. An epoch is a full pass through the training data.")

parser.add_argument("--batchSize", type=int, default=1024,help="batchSize")

parser.add_argument("--lamb", type=float, default=1e-3,help="Lambda")

parser.add_argument("--lRate", type=float, default=1e-3,help="learningRate")

parser.add_argument("--aData", type=int, default=545000,help="Amount of training data to use from the loaded file. Leave empty for full data.")

args = parser.parse_args()





# ----------------------------------------- Network Configuration -------------------------------------

L = args.l
num_classes = 1
neurons = args.neurons
epochs = args.epochs
batchSize = args.batchSize
lamb = args.lamb
cost_function = 'binary_crossentropy'
learning_rate = args.lRate
dataKind = "hex"

# Graph Parameters
ylim = 0.4
color_min_and_max = 0.20

# ---------------------------------------------- Load Data -------------------------------------------



#Import the data
print("Loading data...")
myPath = os.path.abspath(os.path.dirname(__file__))

graphDataPath = os.path.join(myPath, "../train_data/data_"+ dataKind + "/data_" + str(L) + "_5000new" + dataKind + ".txt")

path = os.path.join(myPath, "../train_data/data_"+ dataKind + "/data_" + str(L) + "_3000" + dataKind + ".npy")


if neurons == 2:
    magnetizationGraph = os.path.join(myPath, "../train_data/data_"+ dataKind + "_not_used/data_" + str(L) + "_8400" + dataKind + ".npy")


xTrain = np.load(path)



#for num class = 1
yTrain = np.zeros((xTrain.shape[0], 1))
# yTrain[245000:,:]= 1
yTrain[147000:,:]= 1

print("Data loaded")


xTrain, xTest, yTrain, yTest = train_test_split(xTrain, yTrain, test_size=0.2)

xTrain, xVal, yTrain, yVal = train_test_split(xTrain, yTrain, test_size=0.25)



nowTime = time.strftime(dataKind + "-L" + str(L) + "-E" + str(epochs) + "-%I.%M-%b_%d-" + str(len(xTrain)))

os.makedirs("logs/{}".format(nowTime))
tensorboard = TensorBoard(log_dir="logs/{}".format(nowTime))






# ------------------------------------------ The Model -------------------------------------

#Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = neurons, kernel_initializer = 'truncated_normal', kernel_regularizer=regularizers.l2(lamb), activation = 'sigmoid', input_dim = (L*L)))

# Adding the output layer
classifier.add(Dense(units = num_classes, kernel_initializer = 'truncated_normal', kernel_regularizer=regularizers.l2(lamb), activation = 'sigmoid'))

adam = optimizers.adam(lr=learning_rate)

# Compiling the ANN
classifier.compile(optimizer = adam, loss = cost_function, metrics = ["accuracy"])



# --------------------------------------------------------------------------------------------------------------



if neurons == 2:
    print("Loading Magnetization Graph Data...")
    graphData = np.load(magnetizationGraph)
    print("Graph Data Loaded")

    # with a Sequential model
    hiddenLayerWeight1 = classifier.layers[0].get_weights()[0][:,0]
    hiddenLeyerBias1 = classifier.layers[0].get_weights()[1][0]

    hiddenLayerWeight2 = classifier.layers[0].get_weights()[0][:,1]
    hiddenLeyerBias2 = classifier.layers[0].get_weights()[1][1]

    wxb1 = np.dot(graphData,hiddenLayerWeight1) + hiddenLeyerBias1
    wxb2 = np.dot(graphData,hiddenLayerWeight2) + hiddenLeyerBias2

    xAxis = np.sum(graphData, axis=1)/(L*L)

    f = open("logs/"+ str(nowTime) + "/WXB and m(x).txt","w")
    f.write("wxb1:\n" + str(wxb1) + "\nwxb2:\n" + str(wxb2))
    f.write("\n\n\nm(x):\n" + str(xAxis))
    f.close()

    plt.ylabel("W x + b", fontsize=15)
    plt.xlabel("m(x)", fontsize=15,labelpad=0)
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')

    # Plot the X and Y values
    plt.plot(xAxis,wxb1,linestyle="",marker=".",markersize=0.005)
    plt.plot(xAxis,wxb2,linestyle="",marker=".",markersize=0.005, color="red")

    # Save Graph PDF
    plt.savefig("logs/"+ str(nowTime) + "/Magnetization.pdf")
    plt.clf()
    graphData = None
# --------------------------------------------------------------------------------------------------------------



# Saving Model Weights and Bias in txt format
hiddenLayerWeights = classifier.layers[0].get_weights()[0]
hiddenLeyerBias = classifier.layers[0].get_weights()[1]

outputLayerWeights = classifier.layers[1].get_weights()[0]
outputLeyerBias = classifier.layers[1].get_weights()[1]

f = open("logs/"+ str(nowTime) + "/Weights and Bias Before.txt","w")
f.write("Bias:\n" + str(hiddenLeyerBias) + "\nWeights:\n" + str(hiddenLayerWeights))
f.write("\nBias:\n" + str(outputLeyerBias) + "\nWeights:\n" + str(outputLayerWeights))
f.close()

hiddenLayerWeights, hiddenLeyerBias, outputLayerWeights, outputLeyerBias = None, None, None, None

# Fitting the ANN to the Training set
classifier.fit(xTrain, yTrain, batch_size = batchSize, epochs = epochs, validation_data=(xVal, yVal), verbose=1 , callbacks=[tensorboard])

# Saving the model
classifier.save("logs/"+ str(nowTime) + "/model.h5")
# --------------------------------------------------------------------------------------------------------------


if neurons == 2:
    print("Loading Magnetization Graph Data...")
    graphData = np.load(magnetizationGraph)
    print("Graph Data Loaded")

    # with a Sequential model
    hiddenLayerWeight1 = classifier.layers[0].get_weights()[0][:,0]
    hiddenLeyerBias1 = classifier.layers[0].get_weights()[1][0]

    hiddenLayerWeight2 = classifier.layers[0].get_weights()[0][:,1]
    hiddenLeyerBias2 = classifier.layers[0].get_weights()[1][1]

    wxb1 = np.dot(graphData,hiddenLayerWeight1) + hiddenLeyerBias1
    wxb2 = np.dot(graphData,hiddenLayerWeight2) + hiddenLeyerBias2

    xAxis = np.sum(graphData, axis=1)/(L*L)

    f = open("logs/"+ str(nowTime) + "/WXB and m(x) Trained.txt","w")
    f.write("wxb1:\n" + str(wxb1) + "\nwxb2:\n" + str(wxb2))
    f.write("\n\n\nm(x):\n" + str(xAxis))
    f.close()

    plt.ylabel("W x + b", fontsize=15)
    plt.xlabel("m(x)", fontsize=15,labelpad=0)
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')

    # Plot the X and Y values
    plt.plot(xAxis,wxb1,linestyle="",marker=".",markersize=0.005)
    plt.plot(xAxis,wxb2,linestyle="",marker=".",markersize=0.005, color="red")

    # Save Graph PDF
    plt.savefig("logs/"+ str(nowTime) + "/Magnetization Trained.pdf")
    plt.clf()
    graphData = None



# --------------------------------------------------------------------------------------------------------------


hours, rem = divmod(time.time() - startTime, 3600)
minutes, seconds = divmod(rem, 60)
timeTaken = "{:0>2}.{:0>2}.{:05.2f}".format(int(hours),int(minutes),seconds)

# View accuracy of testing data
test = classifier.evaluate(xTest, yTest, verbose=0)
accuracy = test[1]
print("Test Accuracy: ", accuracy)

# Saving Model Weights and Bias in txt format
hiddenLayerWeights = classifier.layers[0].get_weights()[0]
hiddenLeyerBias = classifier.layers[0].get_weights()[1]

outputLayerWeights = classifier.layers[1].get_weights()[0]
outputLeyerBias = classifier.layers[1].get_weights()[1]

f = open("logs/"+ str(nowTime) + "/Weights and Bias After.txt","w")
f.write("Bias:\n" + str(hiddenLeyerBias) + "\nWeights:\n" + str(hiddenLayerWeights))
f.write("\nBias:\n" + str(outputLeyerBias) + "\nWeights:\n" + str(outputLayerWeights))
f.close()


# Saving Model Parameters in txt format
f = open("logs/"+ str(nowTime) + "/Parameters.txt","w")
f.write("L," + str(L) + "\nNeurons," + str(neurons)+ "\nEpoch," + str(epochs)+ "\nlamb," + str(lamb)+
        "\nCost_Function," + str(cost_function) +"\nLearning_rate," + str(learning_rate) +"\nNum_Classes," + str(num_classes)+
        "\nLen_training," + str(len(xTrain))+"\nLen_testing," + str(len(xTest)) +"\nLen_validation," + str(len(xVal))+
        "\nTest_Accuracy," + str(accuracy) + "\nTime_taken," + timeTaken)

f.close()


# --------------------------------------------------------------------------------------------------------------

if neurons ==2:
    # Figure containing Neuron 1 & 2 weights
    fig = plt.figure()

    plot1 = fig.add_subplot(311)
    axes = plt.gca()
    axes.set_ylim([-ylim,ylim])
    plt.title("Weights vs N\nEpochs: " + str(epochs))
    plot1.plot(list(range(L*L)), hiddenLayerWeights[:,0], label="Neuron 1")

    plot2 = fig.add_subplot(312)
    axes = plt.gca()
    axes.set_ylim([-ylim,ylim])
    plt.ylabel("Weights")
    plot2.plot(list(range(L*L)), hiddenLayerWeights[:,1], label="Neuron 2")

    plot3 = fig.add_subplot(313)
    axes = plt.gca()
    axes.set_ylim([-ylim,ylim])
    axes.set_xlim([0,L*L])
    plt.xlabel("N")
    plot3.plot(list(range(L*L)), hiddenLayerWeights[:,1], label="Neuron 2")
    plot3.plot(list(range(L*L)), hiddenLayerWeights[:,0], label="Neuron 1")
    plt.legend()

    fig.savefig("logs/"+ str(nowTime) + "/weights_vs_n.pdf")

    # Figure containing weights from one neuron color coded
    fig = plt.figure()

    plot1 = fig.add_subplot(211)
    plt.title(nowTime + "\n\nNeuron 1")
    zvals = np.reshape(hiddenLayerWeights[:,0], (L,L))
    # make a color map of fixed colors
    cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',['blue','black','red'],256)
    # tell imshow about color map so that only set colors are used
    img2 = plt.imshow(zvals,interpolation='nearest', cmap = cmap2, origin='lower', vmin=-color_min_and_max, vmax=color_min_and_max)
    # make a color bar
    plt.colorbar(img2,cmap=cmap2)

    plot1 = fig.add_subplot(212)
    plt.title("Neuron 2")
    zvals = np.reshape(hiddenLayerWeights[:,1], (L,L))
    # make a color map
    cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',['blue','black','red'],256)
    # tell imshow about color map so that only set colors are used
    img2 = plt.imshow(zvals,interpolation='nearest', cmap = cmap2, origin='lower', vmin=-color_min_and_max, vmax=color_min_and_max)
    # make a color bar
    plt.colorbar(img2,cmap=cmap2)

    plt.subplots_adjust(hspace=0.4)

    fig.savefig("logs/"+ str(nowTime) + "/weights_color.pdf")
    plt.clf()

# --------------------------------------------------------------------------------------------------------------
xTrain, xTest, yTrain, yTest, xVal, yVal = None, None, None, None, None, None


#620000 at 0.01 step from 0.5 to 2.08
#1.58

print("Loading Graph Data...")
graphData = np.loadtxt(graphDataPath)
print("Graph Data Loaded")
graphDataOutput = classifier.predict(graphData)

# temperature = np.linspace(0.82,2.08,127)
# temperature = np.delete(temperature,[31,63,95])

temperatures = np.linspace(0.3,3,109)
temperatures = np.delete(temperatures,[25,26,27,53,54,55,81,82,83])

# #Calculate average value per each temperature
# average = 0
# yAxis = []
# for index in range(len(graphDataOutput)):
#     average += graphDataOutput[index][0]
#     if ((index+1) % 3000) == 0:
#         yAxis.append(average/3000)
#         average = 0

#Calculate average value per each temperature
average = 0
yAxis = []
for index in range(len(graphDataOutput)):
    average += graphDataOutput[index][0]
    if ((index+1) % 5000) == 0:
        yAxis.append(average/5000)
        average = 0



yAxis = np.array(yAxis)
np.savetxt("logs/"+ str(nowTime) + "/yAxis.txt",yAxis)


# Plot the X and Y values
plt.plot(temperatures,yAxis,marker="D",markersize=5,linewidth=1,label="High-T Neuron")

# Add axis labels
plt.ylabel("Average output layer", fontsize=15)
plt.xlabel("T", fontsize=15,labelpad=0)
plt.xlim([0,4])

# Plot theoretical Tc line
y=[0,1]
if dataKind == "square":
    x=[2.26918,2.26918]
    plt.plot(x, y,color="#FFA500",linewidth=2, label="Actual T = 2.269")
elif dataKind == "hex":
    x=[1.519,1.519]
    plt.plot(x, y,color="#FFA500",linewidth=2, label="Actual T = 1.519")
else:
    x=[0,0]

# Calculate predicted Tc
interp = np.interp(0.5, yAxis, temperatures)

# Plot predicted Tc line
plt.plot([interp,interp], y,color="#008000",linewidth=2, label="Predicted T = {0:.3f}".format(interp))

# Add a legend
leg = plt.legend(loc="best",numpoints=1,markerscale=1.0,fontsize=12,labelspacing=0.1)

# Save Graph PDF
plt.savefig("logs/"+ str(nowTime) + "/neuralnetoutput.pdf")
