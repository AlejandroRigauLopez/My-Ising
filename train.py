from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt #2.1
import tensorflow as tf #1.8
import numpy as np
import argparse
import time
import os

startTime = time.time()

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Data and model checkpoints directories
parser.add_argument("--epochs", type=int, default=1,help="How many epochs to run the network. An epoch is a full pass through the training data.")

parser.add_argument("--aData", type=int, default=545000,help="Amount of training data to use from the loaded file. Leave empty for full data.")

parser.add_argument("--l", type=int, default=10,help="Size of the lattice")


args = parser.parse_args()



#Configuration
L = args.l
dataKind = "hex"
epochs = args.epochs
neurons = 2
lamb = 0.001
learningRate = 1e-4
batchSize = 1024

# Data contains 545000 examples and 109 temperatures (0.3 to 3) on 0.025 temperature steps making it 5000 examples per temperature
# 245000 Configurations are below Tc [0] and 300000 are above Tc [1]


#Import the data
print("Loading data...")
myPath = os.path.abspath(os.path.dirname(__file__))

path = os.path.join(myPath, "../train_data/data_"+ dataKind + "/data_" + str(L) + "_5000new" + dataKind + "test.npy")
graphDataPath = os.path.join(myPath, "../train_data/data_"+ dataKind + "/data_" + str(L) + "_8400" + dataKind + "test.npy")
# graphDataPath = os.path.join(myPath, "../train_data/data_"+ dataKind + "/data_" + str(L) + "_3000" + dataKind + ".npy")

xTrainGraph = np.load(graphDataPath)
xTrain = np.load(path)

yTrain = np.zeros((xTrain.shape[0], 1))
yTrain[245000:,:]= 1

print("Data loaded")





#Split the data Randomly
xTrain, xTest, yTrain, yTest = train_test_split(xTrain, yTrain, test_size=0.1)

xTrain = xTrain[:args.aData]
yTrain = yTrain[:args.aData]

print("Length train: ",len(xTrain))
print("Length test: ",len(xTest))




# Create log files
nowTime = time.strftime(dataKind + "-L" + str(L) + "-E" + str(epochs) + "-%I.%M- %b_%d-" + str(len(xTrain)))
logPath = os.path.join(myPath, "logs/",nowTime)
os.makedirs(logPath)
os.makedirs(logPath + "/model")

inputs = tf.placeholder(tf.float32, shape=[None, L*L])
labels = tf.placeholder(tf.float32, shape=[None, 1])


weights = np.ones((L*L, neurons), dtype="float32")
weights[:,0]= 1/(L*L)
weights[:,1]= -1/(L*L)


# The model
# hiddenLayer = {"weights": tf.get_variable("W1", trainable=False, initializer=weights),
#                "biases": tf.get_variable("B1", shape=[neurons], initializer=tf.contrib.layers.xavier_initializer())}

hiddenLayer = {"weights": tf.get_variable("W1", shape=[L*L, neurons], initializer=tf.contrib.layers.xavier_initializer()),
               "biases": tf.get_variable("B1", shape=[neurons], initializer=tf.contrib.layers.xavier_initializer())}

outputLayer = {"weights": tf.get_variable("WO", shape=[neurons, 1], initializer=tf.contrib.layers.xavier_initializer()),
               "biases": tf.get_variable("BO", shape=[1], initializer=tf.contrib.layers.xavier_initializer())}


hiddenLayerOutput = tf.nn.sigmoid(tf.add(tf.matmul(inputs, hiddenLayer["weights"]), hiddenLayer["biases"]))
output = tf.add(tf.matmul(hiddenLayerOutput, outputLayer["weights"]), outputLayer["biases"])


# Let x = logits, z = labels
# Cost function: z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
# crossEntropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=output))

crossEntropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=output))

# regularizers = tf.nn.l2_loss(hiddenLayer["weights"]) + tf.nn.l2_loss(outputLayer["weights"])
#
# crossEntropy = tf.reduce_mean(crossEntropy + (lamb * regularizers))


# Minimises the error function
global_step = tf.Variable(0, trainable=False)
#
# learningRate = tf.train.exponential_decay(learningRate, global_step, len(xTrain), 0.95, staircase=True)

optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(crossEntropy, global_step=global_step)
# optimizer = tf.contrib.opt.NadamOptimizer(learning_rate=learningRate).minimize(crossEntropy, global_step=global_step)

# Add ops to save and restore all the variables.
# saver = tf.train.Saver()

# Configuration for Session preventing full allocation of memory
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

with tf.Session(config=config) as sess:

    sess.run(tf.global_variables_initializer())

    # Save weights and bias before training
    f = open(logPath + "/Weights and Bias Before.txt","w")
    f.write("Bias:\n" + str(sess.run(hiddenLayer["biases"])) + "\nWeights:\n" + str(sess.run(hiddenLayer["weights"])))
    f.write("\nBias 2:\n" + str(sess.run(outputLayer["biases"])) + "\nWeights 2:\n" + str(sess.run(outputLayer["weights"])))
    f.close()


    lossGraph = list()

    # Training Loop
    for epoch in range(epochs):

        print("-------------------------------------------------")
        print("Training Epoch " + str(epoch+1) + "/" + str(epochs))

        totalBatch = int(len(xTrain) / batchSize)
        xBatches = np.array_split(xTrain, totalBatch)
        yBatches = np.array_split(yTrain, totalBatch)

        for step in range(totalBatch):

            _, loss = sess.run([optimizer, crossEntropy], feed_dict={inputs: xBatches[step], labels: yBatches[step]})


            # Print 4 times every epoch
            if totalBatch != 1 and step % int(totalBatch/4) == 0:
                print("Step:" + str(step)+ "/" + str(totalBatch) + "\tLoss: " + str(loss))

        lossGraph.append(loss)

        xTrain, _, yTrain, _ = train_test_split(xTrain, yTrain, test_size=0)

        print("Final Epoch Loss: " + str(loss))
        # Save model
        # saver.save(sess, logPath + "/model/model.ckpt")

        # Accuracy for each epoch
        predicted = tf.nn.sigmoid(output)
        correct_prediction = tf.equal(tf.round(predicted), labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Epoch Accuracy: " + str(sess.run(accuracy, feed_dict={inputs: xTest, labels: yTest})*100) + "%")



    #Calculating final accuracy
    predicted = tf.nn.sigmoid(output)
    correct_prediction = tf.equal(tf.round(predicted), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Final Accuracy: " + str(sess.run(accuracy, feed_dict={inputs: xTest, labels: yTest})*100) + "%")


    #Save the weights and bias after training
    f = open(logPath + "/Weights and Bias After.txt","w")
    f.write("Bias:\n" + str(sess.run(hiddenLayer["biases"])) + "\nWeights:\n" + str(sess.run(hiddenLayer["weights"])))
    f.write("\nBias 2:\n" + str(sess.run(outputLayer["biases"])) + "\nWeights 2:\n" + str(sess.run(outputLayer["weights"])))
    f.close()

    # Save parameters
    f = open(logPath + "/Parameters.txt","w")
    f.write("L," + str(L) + "\nNeurons," + str(neurons)+ "\nEpoch," + str(epochs)+"\nTest_Accuracy," + str(sess.run(accuracy, feed_dict={inputs: xTest, labels: yTest})*100) + "%")
    f.close()

    # Save temperature list
    if dataKind == "square":
        temperatures = np.linspace(0.8,3.5,28)

    if dataKind == "hex":
        temperatures = np.linspace(0.3,3,28)
        # temperatures = np.linspace(0.3,3,109)

    np.savetxt(logPath + "/Temperatures.txt",temperatures)


    # Full graph data output
    graphDataOutput = tf.nn.sigmoid(output)
    graphDataOutput = sess.run(graphDataOutput, feed_dict={inputs: xTrainGraph})
    print(graphDataOutput)
    np.set_printoptions(threshold=np.inf)
    np.savetxt(logPath + "/Test.txt",graphDataOutput)


# Loss Graph and Loss file
plt.plot(lossGraph)
plt.savefig(logPath + "/lossGraph.pdf")
plt.clf()

f = open(logPath + "/Loss.txt","w")
f.write(str(lossGraph))
f.close()

#Calculate average value per each temperature
average = 0
yAxis = []
for index in range(len(graphDataOutput)):
    average += graphDataOutput[index][0]
    if ((index+1) % 300) == 0:
        yAxis.append(average/300)
    # if ((index+1) % 3000) == 0:
    #     yAxis.append(average/3000)
        average = 0
np.savetxt(logPath + "/Y_axis.txt",yAxis)


# Plot the X and Y values
c = plt.cm.spectral((1)/4.,1)
plt.plot(temperatures,yAxis,marker="D",markersize=5,color=c,linewidth=1,label="High-T Neuron")

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
plt.savefig(logPath + "/neuralnetoutput.pdf")

hours, rem = divmod(time.time() - startTime, 3600)
minutes, seconds = divmod(rem, 60)
timeTaken = "{:0>2}.{:0>2}.{:05.2f}".format(int(hours),int(minutes),seconds)

f = open(logPath + "/Pred_T_{0:.3f}_&_Runtime{1}.txt".format(interp,timeTaken),"a")
f.write("\nRuntime = " + str(timeTaken))
f.write("\nPredicted T = {}".format(interp))
f.close()
