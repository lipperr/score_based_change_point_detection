import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class WISDM:
    def __init__(self, dim=3, bandwidth=20, part="val", save_path="results/", path="data/WISDM/sample_0.csv"):
        self.type = "sequential"
        self.name = "wisdm"
        self.part = part
        self.bandwidth = bandwidth
        self.save_path = save_path
        self.dim = dim
        df = pd.read_csv(path)
        self.raw_data = np.array(df[['X1', 'X2', 'X3']])
        self.raw_labels = np.array(df['Label'])
        self.new_labels = np.empty(0)
        self.change_points = []
        self.data = np.empty((0, self.dim))

        self.preprocess()

    def __call__(self):
       pass
    
    def preprocess(self):
        N = self.raw_data.shape[0] // self.bandwidth
        for t in range(N):
                
            self.data = np.append(self.data, np.mean(self.raw_data[self.bandwidth * t : self.bandwidth * (t + 1)], axis=0).reshape(1, -1), axis=0)
            self.new_labels = np.append(self.new_labels, np.sum(self.raw_labels[self.bandwidth * t : self.bandwidth * (t + 1)]))
            
        self.change_points = np.where(self.new_labels)[0]
        print("Change points after preprocessing: ", self.change_points[:-1])

        # Split the data into the stationary part, validation part, and test part

        # Stationary part: four parts of the time series without the change points.
        # Used to tune the threshold
        self.data_stationary = [self.data[:180], self.data[181:360], self.data[361:540], self.data[541:720]]
        scale = np.max([np.max(self.data_stationary[i], axis=0) for i in range(len(self.data_stationary))], axis=0)
        self.data = self.data / scale

        # Validation part: a part with several change points to tune the hyperparameters
        val_start = 0
        val_end = 1670
        self.data_val = self.data[val_start:val_end] 
        self.change_points_val = self.change_points[self.change_points < val_end] - val_start
        self.change_points_val = self.change_points_val[self.change_points_val > 0]
        print('Validation change points:', self.change_points_val)

        # Test part: check the performance of the procedures
        test_start = 1670
        test_end = 3060
        self.data_test = self.data[test_start:test_end] 
        self.change_points_test = self.change_points[self.change_points < test_end] - test_start
        self.change_points_test = self.change_points_test[self.change_points_test > 0]
        print('Test change points:', self.change_points_test)


    def get_data_stat(self):
        return self.data_stationary

    def data_cp_len(self):
        return 1

    def get_data_cp(self):
        if self.part == "val":
            return self.data_val
        elif self.part == "test":
            return self.data_test
        else:
            return self.data
    
    def get_cps(self):
        if self.part == "val":
            return self.change_points_val
        elif self.part == "test":
            return self.change_points_test
        else:
            return self.change_points

    def get_cps_plot(self):
        return self.get_cps()

    def display_data(self, save_path_plot="results/wisdm.png"):
        fig = plt.figure(figsize=(14, 5))

        gs = GridSpec(12, 1, figure=fig)  
        val_start = 0
        val_end = 1670
        test_start = 1670
        test_end = 3060

        ax1 = fig.add_subplot(gs[0:4, 0])
        ax1.plot(np.arange(val_end), self.data_val[:, 0], c='#bb20c1')
        ax1.plot(np.arange(test_start, test_end), self.data_test[:, 0], c='#f171f6')
        ax1.plot(0, lw=0, label="1st component")
        ax1.legend(loc='upper right', fontsize=16)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Signal')
        for i in range(len(self.change_points)):
            ax1.axvline(self.change_points[i], c='#356258', ls=':')

        ax2 = fig.add_subplot(gs[4:8, 0])
        ax2.plot(np.arange(val_end), self.data_val[:, 1], c='#192586')
        ax2.plot(np.arange(test_start, test_end), self.data_test[:, 1], c='#8993e9')
        ax2.plot(0, lw=0, label="2nd component")
        ax2.legend(loc='upper right', fontsize=16)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Signal')
        for i in range(len(self.change_points)):
            ax2.axvline(self.change_points[i], c='#356258', ls=':')


        ax3 = fig.add_subplot(gs[8:12, 0])
        ax3.plot(np.arange(val_end), self.data_val[:, 2], c='#0098a6')
        ax3.plot(np.arange(test_start, test_end), self.data_test[:, 2], c='#05ebff')
        ax3.plot(0, lw=0, label="3rd component")
        ax3.legend(loc='upper right', fontsize=16)
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Signal')
        for i in range(len(self.change_points)):
            ax3.axvline(self.change_points[i], c='#356258', ls=':')

        plt.savefig(save_path_plot)
        plt.close()