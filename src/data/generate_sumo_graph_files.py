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


def generate_ids_file(ts_ids, data_folder, suffix):
    with open(f'{data_folder}graph_sensor_ids_{suffix}.txt', 'w') as f:
        s = ''
        for id in ts_ids:
            s += f'{id},'
        print(f'id_list: {s[:-2]}')
        f.write(s[:-2])
    
    
def generate_locations_file(jun_pos_dic, data_folder, suffix):
    locations = pd.DataFrame(columns=['sensor_id', 'latitude', 'longitude'])
    for key, value in jun_pos_dic.items():
        locations = locations.append(pd.DataFrame([[key, value[0], value[1]]],
                                                  columns=['sensor_id', 'latitude', 'longitude']))

    locations_file_directory=f'{data_folder}graph_sensor_locations_{suffix}.csv'
    locations.to_csv(locations_file_directory, index=False)
    return locations_file_directory
    

def get_distance(cord0, cord1):
    distance = math.sqrt((cord0[0] - cord1[0])**2 + (cord0[1] - cord1[1])**2)
    return distance

    
def generate_distance_file(location_df_filename, data_folder, suffix):
    output_filename = f'{data_folder}distances_{suffix}.csv'
    df = pd.read_csv(location_df_filename)

    distances = pd.DataFrame(columns=['from','to','cost'])
    for index0, row0 in df.iterrows():
        cord0 = (df['latitude'][index0], df['longitude'][index0])
        for index1, row1 in df.iterrows():
            cord1 = (df['latitude'][index1], df['longitude'][index1])
            distances = distances.append(pd.DataFrame([[row0['sensor_id'], row1['sensor_id'], get_distance(cord0, cord1)]],
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
        traci.start([sumolib.checkBinary('sumo'), '-c', sumocfg], label='init_connection ' +label)
        conn = traci.getConnection('init_connection ' +label)
    
    # generate ids file
    ts_ids = list(conn.trafficlight.getIDList())
    generate_ids_file(ts_ids, data_folder, suffix)

    # generate locations file
    jun_ids = list(conn.junction.getIDList())
    jun_ids_with_ts = {}
    for jun in jun_ids:
        for ts in ts_ids:
            if jun in ts:
                jun_ids_with_ts[jun] = ts
    print(jun_ids_with_ts)
    
    jun_pos_dic = {jun_ids_with_ts[jun]:conn.junction.getPosition(jun) for jun in jun_ids_with_ts}
    locations_file_directory = generate_locations_file(jun_pos_dic, data_folder, suffix)
    
    # generate distance file    
    generate_distance_file(locations_file_directory, data_folder, suffix)


