import numpy as np
import os


#Configuration
LL=[80]
data_kind = "hex"

for L in LL:
    #Import the data
    print("Loading data...")
    my_path = os.path.abspath(os.path.dirname(__file__))

    path = os.path.join(my_path, "../train_data/data_"+ data_kind + "/data" + str(L) + "_5000" + data_kind + ".txt")

    X_train = np.loadtxt(path)

    path = os.path.join(my_path, "../train_data/data_"+ data_kind + "/data" + str(L) + "_5000" + data_kind + ".npy")

    print("Data loaded")


    np.save(path, X_train)

    print("Data Saved Successfully")
