import os
import sys
import argparse
import traci
import sumolib

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import math
import pandas as pd

LIBSUMO = 'LIBSUMO_AS_TRACI' in os.environ


def generate_ids_file(detector_ids, data_folder, suffix):
    with open(f'{data_folder}graph_sensor_ids_{suffix}.txt', 'w') as f:
        s = ''
        for id in detector_ids:
            s += f'{id},'
        print(f'id_list: {s[:-1]}')
        f.write(s[:-1])


def get_distance(cord0, cord1, in_flag):
    # distance = math.sqrt((cord0[0] - cord1[0]) ** 2 + (cord0[1] - cord1[1]) ** 2)
    # TODO: generar un valor de distancia más significativo. Por ejemplo un valor asociado al tamaño de las lanes y
    #       a la dirección de la lane
    distance = 0
    return distance


def generate_distance_file(lanes_dict, data_folder, suffix):
    output_filename = f'{data_folder}distances_{suffix}.csv'
    distances = pd.DataFrame(columns=['from', 'to', 'cost'])
    for lane0 in lanes_dict:
        out_lanes0 = [lane[0] for lane in conn.lane.getLinks(lane0)]
        for lane1 in lanes_dict:
            out_lanes1 = [lane[0] for lane in conn.lane.getLinks(lane1)]
            if (lane1 in out_lanes0):
                in_flag= False
                distance = get_distance(lane0, lane1, in_flag)
            elif (lane0 in out_lanes1):
                in_flag = True
                distance = get_distance(lane0, lane1, in_flag)
            elif (lane0 == lane1):
                distance = 0
            else:
                distance = 1e9
            distances = distances.append(pd.DataFrame([[lanes_dict[lane0], lanes_dict[lane1], distance]],
                             columns=['from', 'to', 'cost']))

    print(distances.head(30))

    distances.to_csv(output_filename, index=False)


if __name__ == '__main__':
    ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '', '..'))
    os.chdir(ROOT_DIR)
    print(f'root directory: {os.getcwd()}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--sumocfg_filename', type=str, default='data/sumo_optimal/optimal.sumocfg',
                        help='.sumocfg path/filename')
    parser.add_argument('--output_folder', type=str, default='data/sensor_graph/',
                        help='Folder to save the output files: ids, locations and distances')
    parser.add_argument('--files_suffix', type=str, default='opt',
                        help='Suffix to identify the files')
    args = parser.parse_args()

    suffix = args.files_suffix
    data_folder = args.output_folder
    label = str(0)
    sumocfg = args.sumocfg_filename
    if LIBSUMO:
        traci.start([sumolib.checkBinary('sumo'), '-c', sumocfg])  # Start only to retrieve traffic light information
        conn = traci
    else:
        traci.start([sumolib.checkBinary('sumo'), '-c', sumocfg], label='init_connection ' + label)
        conn = traci.getConnection('init_connection ' + label)

    # generate ids file
    detector_ids = list(conn.lanearea.getIDList())
    generate_ids_file(detector_ids, data_folder, suffix)

    # generate distance file
    lane_ids_detector = {conn.lanearea.getLaneID(det): det for det in detector_ids}  
    generate_distance_file(lane_ids_detector, data_folder, suffix)


