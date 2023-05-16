import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci
import sumolib
import subprocess

LIBSUMO = 'LIBSUMO_AS_TRACI' in os.environ


def start_simulation(params):
    sumo_cmd = [params["sumo_binary"],
                '-c', params["sumocfg_path"],
                '--max-depart-delay', "1000000",
                '--waiting-time-memory', str(params['max_sim_time']),
                '--time-to-teleport', str(params['time_to_teleport']),
                "--no-step-log", "true", 
                "--no-warnings", str(not params['sumo_warnings']).lower(),
                "--num-clients", str(1),

                "--duration-log.statistics",
                "--verbose",
                "--log", f"{params['test_log_folder']}/verbose_{params['mult_factor']}.xml",
                #"--summary", f"{self.test_log_folder}/summary_{(self.mult_factor - 0.1)}.xml",
                #"--emission-output", f"{self.test_log_folder}/emission_{(self.mult_factor - 0.1)}.xml",
                ]

    if params['sumo_seed'] == 'random':
        sumo_cmd.append('--random')
    else:
        sumo_cmd.extend(['--seed', str(params['sumo_seed'])])
    if params['use_gui']:
        sumo_cmd.extend(['--start', '--quit-on-end'])
    
    traci_port = sumolib.miscutils.getFreeSocketPort()

    if LIBSUMO:
        traci.start(sumo_cmd)
        sumo = traci
    else:
        label = '0'
        traci.start(
            sumo_cmd, 
            label=label, 
            port=traci_port,
            numRetries=240,
            verbose=False, 
            stdout=open(os.devnull, "w")
        )
        sumo = traci.getConnection(label)
        
    if params['use_gui']:
        sumo.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")

    return sumo

def load_test_route(mult_factor, seed, sumo_folder):
    N_HOURS = 1
    DEPART_DURATION = 3600 * N_HOURS
    MAX_N_BIKE = 400 * N_HOURS
    MAX_N_MOTO = 360 * N_HOURS
    MAX_N_VEH = 1000 * N_HOURS
    MAX_N_BUS = 240 * N_HOURS
    MAX_N_TRUCK = 200 * N_HOURS

    vehicle_dict = {'bicycle':
                        {'prefix': 'bike', 'fringe_factor': 2, 'min_dist': 50,
                            'max_dist': 800, 'veh_p': DEPART_DURATION / (MAX_N_BIKE * mult_factor)},
                    'motorcycle':
                        {'prefix': 'moto', 'fringe_factor': 2, 'min_dist': 100, 'max_dist': 800,
                            'veh_p': DEPART_DURATION / (MAX_N_MOTO * mult_factor)},
                    'passenger':
                        {'prefix': 'veh', 'fringe_factor': 5, 'min_dist': 200, 'max_dist': 0,
                            'veh_p': DEPART_DURATION / (MAX_N_VEH * mult_factor)},
                    'bus':
                        {'prefix': 'bus', 'fringe_factor': 5, 'min_dist': 400, 'max_dist': 0,
                            'veh_p': DEPART_DURATION / (MAX_N_BUS * mult_factor)},
                    'truck':
                        {'prefix': 'truck', 'fringe_factor': 5, 'min_dist': 400, 'max_dist': 0,
                            'veh_p': DEPART_DURATION / (MAX_N_TRUCK * mult_factor)}
                    }

    for veh_name in vehicle_dict:
        subprocess_command = [sys.executable, f"{tools}/randomTrips.py",
                                '-n', f'{sumo_folder}/optimal.net.xml',
                                '--seed', str(seed),
                                '--fringe-factor', f'{vehicle_dict[veh_name]["fringe_factor"]}',
                                '-p', str(vehicle_dict[veh_name]["veh_p"]),
                                '-o', f'{sumo_folder}/optimal.{veh_name}.trips.xml',
                                '-e', str(DEPART_DURATION),
                                '-r', f'{sumo_folder}/optimal.{veh_name}.rou.xml',
                                '--vehicle-class', f'{veh_name}',
                                '--vclass', f'{veh_name}',
                                '--prefix', f'{vehicle_dict[veh_name]["prefix"]}',
                                #'--min-distance', f'{vehicle_dict[veh_name]["min_dist"]}',
                                #'--max-distance', f'{vehicle_dict[veh_name]["max_dist"]}',
                                '--fringe-start-attributes', "departSpeed='max'",
                                '--trip-attributes', "departLane='best'",
                                '--validate', '--remove-loops']
        if veh_name == 'passenger':
            subprocess_command.append('--lanes')
        print(f"Generating routes: seed: {seed}, mult_factor {mult_factor}, vehicle {veh_name}")
        return_code=subprocess.check_output(subprocess_command, timeout=10)
    return

def build_phases(sumo):
    ts_ids = list(sumo.trafficlight.getIDList())
    cycle_time = 120
    yellow_time = 5
    red_time = 2
    max_green = 100
    min_green = 5

    for ts_id in ts_ids:
        # sumo_phases = sumo.trafficlight.getAllProgramLogics(ts_id)[0].phases
        # num_sumo_phases = len(sumo_phases)  # there are 12 sumo phases = 4 phases * 3 changes (green, yellow & all red)
        # num_model_phases = 0 # Number of green phases == number of phases (green+yellow) divided by 2
        # sumo_green_phases_idx = [0] * num_sumo_phases
        # sumo_yellow_phases_idx = [0] * num_sumo_phases
        # for i, p in enumerate(sumo_phases): 
        #     if ('g' in p.state) or ('G' in p.state):
        #         num_model_phases += 1
        #         sumo_green_phases_idx[i] = 1
        #         p.maxDur = max_green
        #         p.minDur = min_green
        #     if ('y' in p.state) or ('Y' in p.state): 
        #         sumo_yellow_phases_idx[i] = 1
        #         p.duration = yellow_time

        # cycle_time_green = cycle_time - (num_model_phases * (yellow_time + red_time + min_green))
        # for i, phase_is_green in enumerate(sumo_green_phases_idx):
        #     if phase_is_green:
        #         sumo_phases[i].duration = int(min_green + (cycle_time_green/num_model_phases))

        # programs = sumo.trafficlight.getAllProgramLogics(ts_id)
        # logic = programs[0]
        # logic.type = 0
        # logic.phases = sumo_phases 
        # sumo.trafficlight.setProgramLogic(ts_id, logic)
        current_sumo_phase = 3
        sumo.trafficlight.setPhase(ts_id, current_sumo_phase)        

        return

def run(params):
    if params['gen_routes']:
        load_test_route(params['mult_factor'], params['route_seed'], params['sumocfg_folder'])
    else:
        print("************************* ROUTES are loaded, but NOT Generated! *************************")
    sumo = start_simulation(params)
    build_phases(sumo)
    while (sumo.simulation.getMinExpectedNumber() > 0) or (sumo.simulation.getTime() >= params["max_sim_time"]-1):
        sumo.simulationStep()
    traci.close()
    sumo = None


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sumocfg_folder', type=str, default='src/data/sumo_optimal/sumo_controlers', help='sumoconfig file folder')
    parser.add_argument('--sumocfg_name', type=str, default='timed', help='sumoconfig file name')
    parser.add_argument('--test_log_folder', type=str, default='test_log_files/sumo_controlers', help='test log folder name')
    
    parser.add_argument('--max_sim_time', type=int, default=7200)
    parser.add_argument('--time_to_teleport', type=int, default=-1)

    parser.add_argument('--sumo_seed', type=int, default=5)
    parser.add_argument('--route_seed', type=int, default=5)    
    parser.add_argument('--mult_factor', type=float, default=0.7)

    parser.add_argument('--use_gui', action='store_true', default=False)
    parser.add_argument('--gen_routes', action='store_true', default=False)    
    parser.add_argument('--sumo_warnings', action='store_true', default=False)
    
    args = parser.parse_args()
    # convert to dictionary
    params = vars(args)

    params['sumocfg_path'] = f"{params['sumocfg_folder']}/{params['sumocfg_name']}.sumocfg"
    params['test_log_folder'] = f"{params['test_log_folder']}/{params['sumocfg_name']}"
    if not(os.path.exists(params['test_log_folder'])):
        os.makedirs(params['test_log_folder'])

    if params["use_gui"]:
        params["sumo_binary"] = sumolib.checkBinary('sumo-gui')
    else:
        params["sumo_binary"] = sumolib.checkBinary('sumo')


    run(params)


    print('*********************************\n FINISH sumo_controler_test.py\n*********************************')