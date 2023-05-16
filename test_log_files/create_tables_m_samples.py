from cProfile import label
from codecs import ignore_errors
import os
from pickle import TRUE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

directory_list = ["test_log_files/opt/best", "test_log_files/opt/best_wt", "test_log_files/opt/last"]
for directory in directory_list:
    columns = []
    n_files_per_model = 0
    for dirname in os.scandir(directory):
        print("Creating colums")
        if dirname.is_dir() and dirname.name not in 'old':
            print(dirname.name)
            for filename in os.scandir(dirname):
                if not filename.is_dir():
                    column_name = float(filename.name[8:11])
                    if column_name not in columns:
                        columns.append(column_name)
                    n_files_per_model += 1
            columns = sorted(columns)
            columns.insert(0, "model")
            print(f'colums = {columns}')
            break

    waiting_time_table = pd.DataFrame(columns=columns)
    time_loss_table = pd.DataFrame(columns=columns)
    depart_delay_table = pd.DataFrame(columns=columns)
    ending_time_table = pd.DataFrame(columns=columns)
    waiting_vehicles_table = pd.DataFrame(columns=columns)

    for dirname in os.scandir(directory):
        if dirname.is_dir() and dirname.name not in 'old':
            print(f"Creating tables of {dirname.name}")
            array_column_idx_prev = 99999999
            n_scenarios = len(columns) - 1
            n_files_per_model = 0
            for filename in os.scandir(dirname):
                if not filename.is_dir():
                    n_files_per_model += 1
            m_samples = int(n_files_per_model / n_scenarios)
            waiting_time_file_array = np.zeros((m_samples, n_scenarios))
            time_loss_file_array = np.zeros((m_samples, n_scenarios))
            depart_delay_file_array = np.zeros((m_samples, n_scenarios))
            ending_time_file_array = np.zeros((m_samples, n_scenarios))
            waiting_vehicles_file_array = np.zeros((m_samples, n_scenarios))

            for filename in os.scandir(dirname):
                if not filename.is_dir():
                    column_name = float(filename.name[8:11])
                    if column_name in columns:
                        array_column_idx = columns.index(column_name) - 1  # -1 por la columna "modelo" que se elimina
                        for simb in range(len(filename.name)):
                            if filename.name[-simb] == "_":
                                print(f"filename:{filename}, {filename.name[-simb + 1:-4]}")
                                array_row_idx = int(filename.name[
                                                    -simb + 1:-4]) - 1  # -1 porque los archivos estan enumerados desde el 1 en adelante
                                break

                        with open(filename) as f:
                            content = f.readlines()
                            ready_to_load = False
                            for line_idx in range(len(content)):
                                if content[line_idx][0:10] == "Statistics":
                                    ready_to_load = True
                                if "WaitingTime" in content[line_idx] and ready_to_load:
                                    waiting_time_file_array[array_row_idx, array_column_idx] = content[line_idx][14:-1]
                                    time_loss_file_array[array_row_idx, array_column_idx] = content[line_idx + 1][11:-1]
                                    ready_to_load = False
                                    break
                                if content[line_idx][:24] == "Simulation ended at time":
                                    ending_time_file_array[array_row_idx, array_column_idx] = int(content[line_idx][26:-4])
                                if content[line_idx][0:8] == "Vehicles":
                                    waiting_vehicles_file_array[array_row_idx, array_column_idx] = int(
                                        content[line_idx + 3][10:-1])
                                if "DepartDelay" in content[line_idx]:
                                    depart_delay_file_array[array_row_idx, array_column_idx] = float(
                                        content[line_idx][14:-1])
                                    break

            waiting_time_file_array = np.mean(waiting_time_file_array, 0)
            time_loss_file_array = np.mean(time_loss_file_array, 0)
            depart_delay_file_array = np.mean(depart_delay_file_array, 0)
            ending_time_file_array = np.mean(ending_time_file_array, 0)
            waiting_vehicles_file_array = np.mean(waiting_vehicles_file_array, 0)
            waiting_time_row = {}
            time_loss_row = {}
            depart_delay_row = {}
            ending_time_row = {}
            waiting_vehicles_row = {}
            for column in range(len(columns)):
                if columns[column] == "model":
                    waiting_time_row[columns[column]] = dirname.name
                    time_loss_row[columns[column]] = dirname.name
                    depart_delay_row[columns[column]] = dirname.name
                    ending_time_row[columns[column]] = dirname.name
                    waiting_vehicles_row[columns[column]] = dirname.name
                else:
                    waiting_time_row[columns[column]] = waiting_time_file_array[column - 1]
                    time_loss_row[columns[column]] = time_loss_file_array[column - 1]
                    depart_delay_row[columns[column]] = depart_delay_file_array[column - 1]
                    ending_time_row[columns[column]] = ending_time_file_array[column - 1]
                    waiting_vehicles_row[columns[column]] = waiting_vehicles_file_array[column - 1]

            waiting_time_table = waiting_time_table.append(waiting_time_row, ignore_index=True)
            time_loss_table = time_loss_table.append(time_loss_row, ignore_index=True)
            depart_delay_table = depart_delay_table.append(depart_delay_row, ignore_index=True)
            ending_time_table = ending_time_table.append(ending_time_row, ignore_index=True)
            waiting_vehicles_table = waiting_vehicles_table.append(waiting_vehicles_row, ignore_index=True)

    waiting_time_table.set_index("model", inplace=True)
    time_loss_table.set_index("model", inplace=True)
    depart_delay_table.set_index("model", inplace=True)
    ending_time_table.set_index("model", inplace=True)
    waiting_vehicles_table.set_index("model", inplace=True)
    print(waiting_time_table)
    print(time_loss_table)
    print(depart_delay_table)
    print(ending_time_table)
    print(waiting_vehicles_table)

    waiting_time_table.to_csv(f"{directory}/waiting_time_table.csv")
    time_loss_table.to_csv(f"{directory}/time_loss_table.csv")
    depart_delay_table.to_csv(f"{directory}/depart_delay_table.csv")
    ending_time_table.to_csv(f"{directory}/ending_time_table.csv")
    waiting_vehicles_table.to_csv(f"{directory}/waiting_vehicles_table.csv")

    # # *****************************************************************
    # # ****************************** Plot *****************************
    # # *****************************************************************
    # waiting_time_array = waiting_time_table.to_numpy()
    # time_loss_array = time_loss_table.to_numpy()
    # depart_delay_array = depart_delay_table.to_numpy()
    # ending_time__array = ending_time_table.to_numpy()
    # waiting_vehicles_array = waiting_vehicles_table.to_numpy()
    # index_list = list(waiting_time_table.index)
    # # print(waiting_time_array)
    # # print(f"index_list = {index_list}")

    # x = np.arange(0.5, 1.3, 0.1)
    # plt.figure()
    # # Add Title
    # plt.title("Test VAriables VS Congestion Level")

    # # Data Coordinates
    # plt.subplot(311)
    # for row_idx in range(len(waiting_time_array)):
    #     plt.plot(x, waiting_time_array[row_idx].astype(float), label=index_list[row_idx])

    # # Add Axes Labels
    # plt.xlabel("congestion level")
    # plt.ylabel("waiting time (s)")
    # plt.legend()

    # # Data Coordinates
    # plt.subplot(312)
    # for row_idx in range(len(time_loss_array)):
    #     plt.plot(x, time_loss_array[row_idx].astype(float), label=index_list[row_idx])

    # # Add Axes Labels
    # plt.xlabel("congestion level")
    # plt.ylabel("time loss (s)")
    # plt.legend()

    # # Data Coordinates
    # plt.subplot(313)
    # for row_idx in range(len(depart_delay_array)):
    #     plt.plot(x, depart_delay_array[row_idx].astype(float), label=index_list[row_idx])

    # # Add Axes Labels
    # plt.xlabel("congestion level")
    # plt.ylabel("depart delay (s)")
    # plt.legend()

    # # Display
    # plt.show()