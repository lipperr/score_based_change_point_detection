import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from sklearn.preprocessing import StandardScaler

plot_names = {"clean": "CENSREC-1-C CLEAN", "snr20": "CENSREC-1-C SNR=20", "snr15": "CENSREC-1-C SNR=15"}

class CENSREC:
    def __init__(self, part="", data_part="clean", dim=1, path="data/CENSREC/MAH_", save_path = "results/"):
        self.name = "censrec"
        self.part = part
        self.dim = dim
        self.save_path = save_path
        self.data_part = data_part # clean, snr15, snr20
        self.type = "independent"
        full_path = path+data_part+".wav"
        input_data = read(full_path)
        audio = input_data[1]       
        # Data preprocessing

        # Scaling
        scaler = StandardScaler()
        audio_scaled = scaler.fit_transform(audio.reshape(-1, 1)) 
        audio_scaled = audio_scaled.reshape(-1)

        # Reduce the data, averaging over 10 values
        bandwidth = 10
        N = audio_scaled.shape[0] // bandwidth

        self.data = np.empty(0)

        for t in range(N):
                
            self.data = np.append(self.data, np.mean(audio_scaled[bandwidth * t : bandwidth * (t + 1)]))

        self.change_points = [410, 2835, 6147, 7891, 10347, 12974, 15967, 18682, 21128, 23620]
        self.change_points = np.array(self.change_points)
        
        # Split the data into the stationary part, validation part, and test part

        # Stationary part: a part of the time series without the change points.
        # Used to tune the threshold
        self.data_stationary = [self.data[:390]]
        
        # Validation part: a part with several change points to tune the hyperparameters
        val_start = [300, 2720, 6050, 7800]
        val_end = [500, 2920, 6250, 8000]
        self.data_val = [self.data[val_start[i]:val_end[i]] for i in range(len(val_start))]
        # Change points on the validation set
        change_points_val = self.change_points[self.change_points < val_end[-1]] - val_start
        self.change_points_val = change_points_val[change_points_val > 0]
        print('Validation change points:', self.change_points_val)


        # Test part: check the performance of the procedures
        test_start = [10250, 12870, 15850, 18580, 21020, 23520]
        test_end = [10450, 13070, 16050, 18780, 21220, 23720]
        self.data_test = [self.data[test_start[i]:test_end[i]] for i in range(len(test_start))]
        # Change points on the test set
        change_points_test = self.change_points[self.change_points < test_end[-1]] 
        self.change_points_test = change_points_test[change_points_test > test_start[0]] - test_start
        print('Test change points:', self.change_points_test)

        # Validation change points for plot
        self.change_points_val_plot = []
        l = 0
        for i in range(len(change_points_val)):
            self.change_points_val_plot += [change_points_val[i] + l]
            l += self.data_val[0].shape[0]

        # Validation data for plot
        self.data_val_plot = np.hstack([self.data_val[i] for i in range(len(change_points_val))])

        # Test change points for plot
        self.change_points_test_plot = []
        l = 0
        for i in range(len(self.change_points_test)):
            self.change_points_test_plot += [self.change_points_test[i] + l]
            l += self.data_test[0].shape[0]

        # Test data for plot
        self.data_test_plot = np.hstack([self.data_test[i] for i in range(len(self.change_points_test))])



    def get_data_stat(self):
        return self.data_stationary

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
        if self.part == "val":
            return self.change_points_val_plot
        elif self.part == "test":
            return self.change_points_test_plot
        else:
            return self.change_points
    
    def display_data(self, save_path_plot="results/censrec.png"):
        plt.figure(figsize=(15, 4))

        for c in self.change_points:
            plt.axvline(c, c='m', ls=':')
        plt.plot(np.arange(len(self.data)), self.data, c='b')

        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.title(label=f'{plot_names[self.data_part]} {self.part}', fontsize=24)
        plt.xlabel('Time', fontsize=18)
        plt.ylabel('Signal', fontsize=18)
        plt.grid()
        plt.tight_layout()
        plt.savefig(save_path_plot)
        plt.close()