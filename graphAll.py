import numpy as np
# For plotting
import matplotlib as mpl
import matplotlib.pyplot as plt

import os

myPath = os.path.abspath(os.path.dirname(__file__))

# Plot theoretical Tc line
y=[0,1]
x=[1.519,1.519]
plt.plot(x, y,color="#FFA500",linewidth=1, label="Actual T = 1.519")

temperatures = np.linspace(0.3,3,109)


path = os.path.join(myPath, "logs/hex-L20-E100-10.55-Mar_18-196200/yAxis.txt")
yAxis = np.loadtxt(path)
# Plot the X and Y values
plt.plot(temperatures,yAxis,marker="D",color="#008000",markersize=1,linewidth=2,label="L = 20")
# Calculate predicted Tc
interp = np.interp(0.5, yAxis, temperatures)
# Plot predicted Tc line
plt.plot([interp,interp], y,color="#008000",linewidth=0, label="Predicted T = {0:.3f}".format(interp))

path = os.path.join(myPath, "logs/hex-L40-E100-10.57-Mar_18-196200/yAxis.txt")
yAxis = np.loadtxt(path)
# Plot the X and Y values
plt.plot(temperatures,yAxis,marker="D",color="#922B21",markersize=1,linewidth=2,label="L = 40")
# Calculate predicted Tc
interp = np.interp(0.5, yAxis, temperatures)
# Plot predicted Tc line
plt.plot([interp,interp], y,color="#922B21",linewidth=0, label="Predicted T = {0:.3f}".format(interp))

path = os.path.join(myPath, "logs/hex-L60-E100-11.06-Mar_18-196200/yAxis.txt")
yAxis = np.loadtxt(path)
# Plot the X and Y values
plt.plot(temperatures,yAxis,marker="D",color="#2874A6",markersize=1,linewidth=2,label="L = 60")
# Calculate predicted Tc
interp = np.interp(0.5, yAxis, temperatures)
# Plot predicted Tc line
plt.plot([interp,interp], y,color="#2874A6",linewidth=0, label="Predicted T = {0:.3f}".format(interp))


temperatures = np.delete(temperatures,[25,26,27,53,54,55,81,82,83])
path = os.path.join(myPath, "logs/hex-L80-E100-08.11-Mar_22-196200/yAxis.txt")
yAxis = np.loadtxt(path)
# Plot the X and Y values
plt.plot(temperatures,yAxis,marker="D",color="#244446",markersize=1,linewidth=2,label="L = 80")
# Calculate predicted Tc
interp = np.interp(0.5, yAxis, temperatures)
# Plot predicted Tc line
plt.plot([interp,interp], y,color="#2874A6",linewidth=0, label="Predicted T = {0:.3f}".format(interp))

# Add axis labels
plt.ylabel("Average output layer", fontsize=15)
plt.xlabel("T", fontsize=15,labelpad=0)
plt.xlim([0,4])



# Add a legend
leg = plt.legend(loc="best",numpoints=1,markerscale=1.0,fontsize=12,labelspacing=0.1)

plt.show()
