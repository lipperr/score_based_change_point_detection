import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from matplotlib.gridspec import GridSpec


class Occupancy:
    def __init__(self, part="val", dim=4, path="data/occupancy/occupancy.json", annotations_path='data/occupancy/annotations.json', save_path = "results/"):
        self.type = "sequential"
        self.name="occupancy"
        self.part = part
        self.dim=dim
        self.path = path
        self.save_path = save_path

        self.annotations_path = annotations_path
        data, title, not_parsed_cp = self.load_data(self.path, self.annotations_path, cp=True)
        data.drop(columns=['x'], inplace=True)
        
        myDict = self.parse_cp(data, not_parsed_cp)
        self.merge_keys(myDict, eps=2)
        change_points = self.filter_keys(myDict, cp_treshold=2)
        change_points.append(325)
        self.change_points = np.sort(np.array(change_points))
        print("Change points after preprocessing: ", self.change_points)


        self.df = self.change_series(data, type_change="log_diff")
        # Split the data into the stationary part, validation part, and test part

        # Stationary part: three parts of the time series without the change points.
        # Used to tune the threshold
        self.data = self.df.values
        self.data_stationary = [self.data[:40, :], self.data[100:130], self.data[270:310]]

        # Normalization
        scale = np.max([np.max(self.data_stationary[i], axis=0) for i in range(len(self.data_stationary))], axis=0)
        scale[2] = 10
        self.data = self.data / scale

        # Validation part: a part with several change points to tune the hyperparameters
        val_start = 0
        val_end = 300
        self.data_val = self.data[val_start:val_end]
        # Change points on the validation set
        change_points_val = self.change_points[self.change_points < val_end] - val_start
        self.change_points_val = change_points_val[change_points_val > 0]
        print('Validation change points:', change_points_val)

        
        # Test part: check the performance of the procedures
        test_start = 300
        self.data_test = self.data[test_start:]
        # Change points on the test set
        change_points_test = self.change_points - test_start
        self.change_points_test = change_points_test[change_points_test > 0]
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


    def change_series(self, df, type_change=None):
        if type_change is None:
            return df
        
        cols = df.columns[1:]
        new_df = pd.DataFrame(columns=['x'].extend(cols))
        new_df['x'] = df['x'][:-1]
        for c in cols:
            if type_change == 'log_simple':
                new_df[c] = (np.log1p(np.roll(df[c], -1)) - np.log1p(df[c]))[:-1]
            elif type_change == 'lin':
                new_df[c] = (np.roll(df[c], -1) - df[c])[:-1]
            else:
                new_df[c] = np.log1p(abs(np.roll(df[c], -1) -df[c]) / abs(df[c] + 1))[:-1]

        return new_df
    
    def load_data(self, filename, annotations_filename, cp=False):
        with open(filename, "rb") as fid:
            data = json.load(fid)
        title = data["name"]
        y = data["series"][0]["raw"]
        if "time" in data and "format" in data["time"]:
            try:
                x = pd.to_datetime(
                    data["time"]["raw"], format=data["time"]["format"]
                )
            except ValueError:
                x = list(range(1, len(y) + 1))
        else:
            x = list(range(1, len(y) + 1))
        as_dict = {"x": x}
        for idx, series in enumerate(data["series"]):
            as_dict["y" + str(idx)] = series["raw"]

        df = pd.DataFrame(as_dict)
        if cp:
            with open(annotations_filename) as json_file:
                changepoints_dict = json.load(json_file)
        
            changepoints = changepoints_dict[title]
            return df, title, changepoints
        
        return df, title
    
    def display_data(self, save_path_plot="results/occupancy.png"):

        fig = plt.figure(figsize=(14, 5))

        gs = GridSpec(16, 1, figure=fig)  
        val_start = 0
        val_end = 300
        test_start = 300
        test_end = self.data.shape[0]

        ax1 = fig.add_subplot(gs[0:4, 0])
        ax1.plot(np.arange(val_end), self.data_val[:, 0], c='#bb20c1')
        ax1.plot(np.arange(test_start, test_end), self.data_test[:, 0], c='#f171f6')
        ax1.plot(0, lw=0, label="1st component")
        ax1.legend(loc='upper left', fontsize=14)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Signal')
        for i in range(len(self.change_points)):
            ax1.axvline(self.change_points[i], c='#356258', ls=':')

        ax2 = fig.add_subplot(gs[4:8, 0])
        ax2.plot(np.arange(val_end), self.data_val[:, 1], c='#192586')
        ax2.plot(np.arange(test_start, test_end), self.data_test[:, 1], c='#8993e9')
        ax2.plot(0, lw=0, label="2nd component")
        ax2.legend(loc='upper left', fontsize=14)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Signal')
        for i in range(len(self.change_points)):
            ax2.axvline(self.change_points[i], c='#356258', ls=':')


        ax3 = fig.add_subplot(gs[8:12, 0])
        ax3.plot(np.arange(val_end), self.data_val[:, 2], c='#0098a6')
        ax3.plot(np.arange(test_start, test_end), self.data_test[:, 2], c='#05ebff')
        ax3.plot(0, lw=0, label="3rd component")
        ax3.legend(loc='upper left', fontsize=14)
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Signal')
        for i in range(len(self.change_points)):
            ax3.axvline(self.change_points[i], c='#356258', ls=':')

        ax4 = fig.add_subplot(gs[12:16, 0])
        ax4.plot(np.arange(val_end), self.data_val[:, 3], c="#89480b")
        ax4.plot(np.arange(test_start, test_end), self.data_test[:, 3], c="#e2a46d")
        ax4.plot(0, lw=0, label="4th component")
        ax4.legend(loc='upper left', fontsize=14)
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Signal')
        for i in range(len(self.change_points)):
            ax4.axvline(self.change_points[i], c='#356258', ls=':')

        plt.savefig(save_path_plot)
        plt.close()

    def parse_cp(self, df, cp):
        freq_dict = {}
        for k, v in cp.items():
            for key in v:
                freq_dict[key] = freq_dict.get(key, 0) + 1
        
        myKeys = list(freq_dict.keys())
        myKeys.sort()
        sorted_dict = {i: freq_dict[i] for i in myKeys}
        return sorted_dict

    def merge_keys(self, sorted_dict, eps=2): # inplace

        flag = 0
        keys = list(sorted_dict.keys())
        for key in keys:
            if flag == 0:
                flag = 1
                prev_key = key
                key_to_write = key
                continue
            
            if key - prev_key <= eps:
                sorted_dict[key_to_write] += sorted_dict[key]
                _ = sorted_dict.pop(key, 0)
            else:
                key_to_write = key
            
            prev_key = key

    def filter_keys(self, sorted_dict, cp_treshold=2):
        cps = []
        for k, v in sorted_dict.items():
            if v >= cp_treshold:
                cps.append(k)
        return cps
    
    def change_series(self, df, type_change=None):
        if type_change is None:
            return df
        new_df = pd.DataFrame(columns=df.columns)
        for c in df.columns:
            if type_change == 'log_simple':
                new_df[c] = (np.log1p(np.roll(df[c], -1)) - np.log1p(df[c]))[:-1]
            elif type_change == 'lin':
                new_df[c] = np.abs((np.roll(df[c], -1) - df[c]))[:-1]
            elif type_change == 'log_diff':
                new_df[c] = np.log1p(abs(np.roll(df[c], -1) -df[c]) / abs(np.roll(df[c], -1) + 1))[:-1]
                new_df.iloc[:10, :] = 0
            else:
                feature = abs(np.roll(df[c], -1) -df[c]) / abs(df[c] + 1) # diff normed
                new_df[c] = feature[:-1]
        return new_df