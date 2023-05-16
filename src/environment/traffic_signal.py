from cmath import isnan
from doctest import FAIL_FAST
import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci
import sumolib
import numpy as np
from gym import spaces


class TrafficSignal:
    """
    This class represents a Traffic Signal of an intersection
    It is responsible for retrieving information and changing the traffic phase using Traci API

    IMPORTANT: It assumes that the traffic phases defined in the .net file are of the form:
        [green_phase, yellow_phase, green_phase, yellow_phase, ...]
    Currently it is not supporting all-red phases (but should be easy to implement it).

    Default observation space is a vector R^(#greenPhases + 2 * #lanes)
    s = [current phase one-hot encoded, density for each lane, queue for each lane]
    You can change this by modifing self.observation_space and the method _compute_observations()

    Action space is which green phase is going to be open for the next delta_time seconds
    """
    def __init__(self, env, ts_id, delta_time, cycle_time, yellow_time, min_green, max_green, begin_time, sumo, red_time=2, max_time_change=5):
        self.id = ts_id
        self.env = env
        self.delta_time = delta_time
        self.cycle_time = cycle_time
        self.yellow_time = yellow_time
        self.red_time = red_time
        self.min_green = min_green
        self.max_green = max_green
        self.green_model_phase = None
        self.max_time_change=max_time_change
        self.next_action_time = begin_time
        self.last_measure = 0.0
        self.last_reward = None
        self.sumo = sumo
        # intersection model variables
        self.model_phases_to_mov = {0:(0, 4), 1:(0, 5), 2:(1, 4), 3:(1, 5), 4:(2, 6), 5:(2, 7), 6:(3, 6), 7:(3, 7)}
        self.model_dir_to_mov = {'es':0, 'we':1, 'sw':2, 'ns':3, 'wn':4, 'ew':5, 'ne':6, 'sn':7}
        net = sumolib.net.readNet(self.env._sumonet)

        self.lanes = list(
            dict.fromkeys(self.sumo.trafficlight.getControlledLanes(self.id)))  # Remove duplicates and keep order
        self.out_lanes = [link[0][1] for link in self.sumo.trafficlight.getControlledLinks(self.id) if link]
        self.out_lanes = list(set(self.out_lanes))
        self.lanes_lenght = {lane: self.sumo.lane.getLength(lane) for lane in self.lanes}

        e2det_list = list(self.sumo.lanearea.getIDList())
        e2det_list.sort()
        self.detectors = [detector for detector in e2det_list if self.sumo.lanearea.getLaneID(detector) in self.lanes]
        self.out_detectors = [detector for detector in e2det_list if self.sumo.lanearea.getLaneID(detector) in self.out_lanes]
        self.detectors_lenght = {detector: self.sumo.lanearea.getLength(detector) for detector in self.detectors}
        self.num_detectors=len(self.detectors)

        # Configure the tls phases and create its global variables
        self.build_phases(net)

        # Set TLS model_phases_idx (mask)
        self.model_phases_idx = self._get_tls_model_phases_idx(net)  # indices de fases usadas por este modelo de intersección hot encode

        # Set green_phase
        second_green_phase = False
        for i in range(len(self.model_phases_idx)):
            if self.model_phases_idx[i]:
                if second_green_phase:
                    self.green_model_phase = i  # second green phase of the cycle
                    break
                second_green_phase = True

        # Set sumo_phase_idx_to_splits
        self.sumo_phase_idx_to_splits = self._gen_sumo_phase_idx_to_splits()

        # observation variables
        var_cycles = 1  # CAMBIADO: Antes 2. Ahora: solo un ciclo atras observo
        var_buff_len = var_cycles * cycle_time
        self.density = {detector: np.empty(var_buff_len) for detector in self.detectors}
        self.queue = {detector: np.empty([var_buff_len]) for detector in self.detectors}
        self.occupancy = {detector: np.empty([var_buff_len]) for detector in self.detectors}
        self.speed = {detector: np.empty([var_buff_len]) for detector in self.detectors}
        # New reward variable throughput
        self.first_time_green = False
        self.throughput = {detector: 0 for detector in self.detectors}
        self.veh_left = {detector: 0 for detector in self.detectors}
        self.time_green = {detector: 0 for detector in self.detectors}
        self.init_veh_occupancy = {detector: 0 for detector in self.detectors}
        self.init_veh_number = {detector: 0 for detector in self.detectors}
        self.veh_ids = {detector: [] for detector in self.detectors}
        # prediction input buffer
        self.pred_input = {detector: np.zeros((12,1)) for detector in self.detectors}
        self.pred_in_speed = {detector: [] for detector in self.detectors}
        
        for detector in self.detectors:
            self.density[detector][:] = np.nan
            self.queue[detector][:] = np.nan
            self.occupancy[detector][:] = np.nan
            self.speed[detector][:] = np.nan

        # model movements to detectors
        self.mov_to_detector = self._get_mov_to_detector(net)

        # Observation Space
        self.num_obs = 6  # detector_active, split, density, quee, occup, speed
        self.observation_space = spaces.Box(
            low=np.zeros((self.num_detectors, self.num_obs), dtype=np.float32),
            high=np.ones((self.num_detectors, self.num_obs), dtype=np.float32))
        # self.discrete_observation_space = spaces.Tuple((
        #     spaces.Discrete(self.num_detectors),  # Green Phase
        #     *(spaces.Discrete(10) for _ in range(self.num_obs))  # Density, stopped-density, occupancy and speed for each detector
        # ))

        # Action Space
        self.action_space = spaces.Box(
            low= -self.max_time_change * np.ones(self.num_model_phases, dtype=np.float32),  # self.num_model_phases set in build_phases()
            high= self.max_time_change * np.ones(self.num_model_phases, dtype=np.float32)
        )
        # self.discrete_action_space = spaces.Discrete(self.num_model_phases)

        self.last_splits = np.ones(self.num_detectors)/self.num_model_phases  # split for each detector


    def build_phases(self, net):
        sumo_phases = self.sumo.trafficlight.getAllProgramLogics(self.id)[0].phases

        num_sumo_phases = len(sumo_phases)  # there are 12 sumo phases = 4 phases * 3 changes (green, yellow & all red)
        self.num_model_phases = 0 # Number of green phases == number of phases (green+yellow) divided by 2
        self.sumo_green_phases_idx = [0] * num_sumo_phases
        self.sumo_yellow_phases_idx = [0] * num_sumo_phases
        for i, p in enumerate(sumo_phases): 
            if ('g' in p.state) or ('G' in p.state):
                self.num_model_phases += 1
                self.sumo_green_phases_idx[i] = 1
                p.maxDur = self.max_green
                p.minDur = self.min_green
            if ('y' in p.state) or ('Y' in p.state): 
                self.sumo_yellow_phases_idx[i] = 1
                p.duration = self.yellow_time

        self.cycle_time_green = self.cycle_time - (self.num_model_phases * (self.yellow_time + self.red_time + self.min_green))
        for i, phase_is_green in enumerate(self.sumo_green_phases_idx):
            if phase_is_green:
                sumo_phases[i].duration = int(self.min_green + (self.cycle_time_green/self.num_model_phases))

        programs = self.sumo.trafficlight.getAllProgramLogics(self.id)
        self.logic = programs[0]
        self.logic.type = 0
        self.logic.phases = sumo_phases 
        self.sumo.trafficlight.setProgramLogic(self.id, self.logic)
        self.current_sumo_phase = 3
        self.sumo.trafficlight.setPhase(self.id, self.current_sumo_phase)        
        # self._phase_idx_to_action()
        return

    @property
    def time_to_act(self):
        return self.next_action_time == self.env.sim_step


    def update(self):
        first_green_measured = False

        sumo_phase_idx = self.sumo.trafficlight.getPhase(self.id)
        self.current_sumo_phase = sumo_phase_idx  # currently not used
        phase_is_green = self.sumo_green_phases_idx[sumo_phase_idx]
        phase_is_yellow = self.sumo_yellow_phases_idx[sumo_phase_idx]

        if phase_is_green:
            model_phase_idx = self.sumo_phase_idx_to_splits[sumo_phase_idx]
            active_movements = self.model_phases_to_mov[model_phase_idx]
            self.green_model_phase = model_phase_idx
            active_detectors = [self.mov_to_detector[mov] for mov in active_movements]
        elif phase_is_yellow:
            self.green_model_phase = None
            if not self.first_time_green:
                self.first_time_green = True
        else:
            pass
            
        # update variables buffers
        for detector in self.detectors:
            if phase_is_green and (detector in active_detectors):
                self.density[detector] = np.append(self.density[detector][1:], self.get_detector_density(detector))
                self.queue[detector] = np.append(self.queue[detector][1:], self.get_detector_queue(detector))
                self.occupancy[detector] = np.append(self.occupancy[detector][1:], self.get_detector_occupancy(detector))
                self.speed[detector] = np.append(self.speed[detector][1:], self.get_detector_speed(detector))
                self.pred_in_speed[detector].append(self.get_detector_speed_pred(detector))

                if self.first_time_green:
                    self.veh_left[detector] = 0
                    self.time_green[detector] = 1
                    self.init_veh_occupancy[detector] = self.occupancy[detector][-1]*100 + 1e-9 # TODO: Can bedensity or occupancy
                    self.init_veh_number[detector] = len(self.veh_ids[detector])  # TODO: Can also be: self.get_detector_veh_number(detector)
                    first_green_measured = True
                else:
                    self.veh_left[detector] = self.veh_left[detector] + self.get_detector_veh_left(detector)
                    self.time_green[detector] += 1
            else:
                self.density[detector] = np.append(self.density[detector][1:], np.nan)
                self.queue[detector] = np.append(self.queue[detector][1:], np.nan)
                self.occupancy[detector] = np.append(self.occupancy[detector][1:], np.nan)
                self.speed[detector] = np.append(self.speed[detector][1:], np.nan)
                self.veh_ids[detector] = self.sumo.lanearea.getLastStepVehicleIDs(detector)
        
        # Reset first_green_measured if the variables of the green phase detectors was initialized
        if first_green_measured:
            self.first_time_green = False


    def set_tls_splits(self, action):
        """
        Sets what will be the next green phase and sets yellow phase if the next phase is different than the current

        :param new_phase: (int) Number between [0..num_green_phases]
        """

        splits = self._get_act_to_split(action)
        self.last_splits = splits  # currently not used
        cycle_duration = 0
        max_split = 0
        for idx, phase_is_green in enumerate(self.sumo_green_phases_idx):
            if phase_is_green:
                split_idx = self.sumo_phase_idx_to_splits[idx]
                new_duration = self.min_green + (self.cycle_time_green * splits[split_idx])
                self.logic.phases[idx].duration = int(new_duration)
                cycle_duration += int(new_duration - self.min_green)

                if splits[split_idx] > max_split:
                    max_split = splits[split_idx]
                    max_green_idx = idx
        # Asegurar de que los ciclos sean de igual tamaño sumando el restante de redondear al ultimo green
        self.logic.phases[max_green_idx].duration += self.cycle_time_green - cycle_duration

        self.sumo.trafficlight.setProgramLogic(self.id, self.logic)
        fisrt_idx = 3  # it has to be the second green phase
        self.sumo.trafficlight.setPhase(self.id, fisrt_idx)

        self.next_action_time = self.env.sim_step + self.cycle_time
        return

    
    def compute_observation(self):
        detector_active = self.model_phases_to_mov[self.green_model_phase] if self.green_model_phase is not None else (1000, 1000)
        detector_active_id = [1 if i in detector_active else 0 for i in range(self.num_detectors)]  # one-hot encoding
        
        splits = [split/self.cycle_time_green for split in self.time_green.values()]
        
        density = [np.nanmean(self.density[detector]) if not np.isnan(np.nanmean(self.density[detector])) else 0 for detector in self.detectors]
        queue = [np.nanmean(self.queue[detector]) if not np.isnan(np.nanmean(self.queue[detector])) else 0 for detector in self.detectors]
        occupancy = [np.nanmean(self.occupancy[detector]) if not np.isnan(np.nanmean(self.occupancy[detector])) else 0 for detector in self.detectors]
        speed = [np.nanmean(self.speed[detector]) if not np.isnan(np.nanmean(self.speed[detector])) else 1 for detector in self.detectors]  # "1" because it is the max value when there is no car

        obs = np.stack([detector_active_id, splits, density, queue, occupancy, speed], -1)
        assert obs.shape == self.observation_space.shape
        return obs

    def compute_predicton_buffer(self):
        self.pred_input.update({detector:np.append(self.pred_input[detector][1:], np.nanmean(self.pred_in_speed[detector])) if not np.isnan(np.nanmean(self.pred_in_speed[detector]))
                            else np.append(self.pred_input[detector][1:], 0) 
                            for detector in self.detectors})
        self.pred_in_speed = {detector: [] for detector in self.detectors}
        pred_input = np.array([self.pred_input[detector] for detector in self.detectors ])
        return pred_input

    def compute_reward(self):
        self.last_reward = self._veh_difference() * 100
        return self.last_reward

    # RW4
    def _speed_reward(self):
        speed = np.mean([np.nanmean(self.speed[detector]) if not np.isnan(np.nanmean(self.speed[detector])) else 1 for detector in self.detectors])
        return speed

    # RW4.1
    def _speed_reward_modified(self):
        inertia_weight = 0.8
        speed = np.mean([np.nanmean(self.speed[detector]) if not np.isnan(np.nanmean(self.speed[detector])) else 1 for detector in self.detectors])
        rw = (1 - inertia_weight)*speed + inertia_weight/(1+(max(self.time_green.values())-min(self.time_green.values()))/self.cycle_time_green)
        return rw - 0.5

    # RW4.2
    def _speed_reward_modified_2(self):
        inertia_weight = 0.5
        speed = np.mean([np.nanmean(self.speed[detector]) if not np.isnan(np.nanmean(self.speed[detector])) else 1 for detector in self.detectors])
        rw = (1 - inertia_weight)*speed + inertia_weight/(1+np.std(list(self.time_green.values()))/self.cycle_time_green)
        return rw

    # RW0
    def _throughput_reward(self):
        throughput = np.mean([self.veh_left[detector]/self.time_green[detector] for detector in self.detectors])
        return throughput

    # RW1
    def _veh_left_reward(self):
        # veh_left = np.mean([self.veh_left[detector] for detector in self.detectors])
        veh_left = np.sum([self.veh_left[detector] for detector in self.detectors])
        return veh_left

    # RW2
    def _veh_left_rate_reward(self):
        veh_left = np.mean([self.veh_left[detector] for detector in self.detectors])
        init_veh_number = np.mean([self.init_veh_number[detector] for detector in self.detectors])
        veh_left_rate = veh_left/init_veh_number if init_veh_number > 0.0 else 1.0
        return veh_left_rate  # min(1.0, veh_left_rate)

    # RW5
    def _veh_left_rate_prio_reward(self):
        veh_left_prio = np.sum([min(self.veh_left[detector]*self.init_veh_number[detector], np.square(self.init_veh_number[detector])) for detector in self.detectors])
        init_veh_number_square = np.sum([np.square(self.init_veh_number[detector]) for detector in self.detectors])
        veh_left_rate_prio = veh_left_prio/init_veh_number_square if init_veh_number_square > 0.0 else 1.0
        return veh_left_rate_prio  # min(1.0, veh_left_rate_prio)

    # RW7
    def _momentum_reward(self):
        speed_prio = np.sum([np.nanmean(self.speed[detector])*self.init_veh_number[detector] if not np.isnan(np.nanmean(self.speed[detector])) else self.init_veh_number[detector] for detector in self.detectors])
        total_init_veh_number = np.sum([self.init_veh_number[detector] for detector in self.detectors])
        momentum = speed_prio/total_init_veh_number if total_init_veh_number > 0.0 else 1.0
        return momentum # min(1.0, momentum)

    def _veh_difference(self):
        veh_left = np.sum([self.veh_left[detector] for detector in self.detectors])
        init_veh_number = np.sum([self.init_veh_number[detector] for detector in self.detectors])
        veh_difference = veh_left - init_veh_number
        return veh_difference


    # def _pressure_reward(self):
    #     return -self.get_pressure()

    # def _queue_average_reward(self):
    #     new_average = np.mean(self.get_stopped_vehicles_num())
    #     reward = self.last_measure - new_average
    #     self.last_measure = new_average
    #     return reward

    # def _queue_reward(self):
    #     return - (sum(self.get_stopped_vehicles_num())) ** 2

    # def _waiting_time_reward(self):
    #     ts_wait = sum(self.get_waiting_time_per_lane()) / 100.0
    #     reward = self.last_measure - ts_wait
    #     self.last_measure = ts_wait
    #     return reward

    # def _waiting_time_reward2(self):
    #     ts_wait = sum(self.get_waiting_time())
    #     self.last_measure = ts_wait
    #     if ts_wait == 0:
    #         reward = 1.0
    #     else:
    #         reward = 1.0 / ts_wait
    #     return reward

    # def _waiting_time_reward3(self):
    #     ts_wait = sum(self.get_waiting_time())
    #     reward = -ts_wait
    #     self.last_measure = ts_wait
    #     return reward

    def get_waiting_time_per_detector(self):
        wait_time_per_detector = []
        for detector in self.detectors:
            veh_list = self.sumo.lanearea.getLastStepVehicleIDs(detector)
            wait_time = 0.0
            for veh in veh_list:
                veh_detector = self.sumo.vehicle.getLaneID(veh)
                acc = self.sumo.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_detector: acc}
                else:
                    self.env.vehicles[veh][veh_detector] = acc - sum(
                        [self.env.vehicles[veh][detector] for detector in self.env.vehicles[veh].keys() if detector != veh_detector])
                wait_time += self.env.vehicles[veh][veh_detector]
            wait_time_per_detector.append(wait_time)
        return wait_time_per_detector

    # def get_waiting_time_per_lane(self):
    #     wait_time_per_lane = []
    #     for lane in self.lanes:
    #         veh_list = self.sumo.lane.getLastStepVehicleIDs(lane)
    #         wait_time = 0.0
    #         for veh in veh_list:
    #             veh_lane = self.sumo.vehicle.getLaneID(veh)
    #             acc = self.sumo.vehicle.getAccumulatedWaitingTime(veh)
    #             if veh not in self.env.vehicles:
    #                 self.env.vehicles[veh] = {veh_lane: acc}
    #             else:
    #                 self.env.vehicles[veh][veh_lane] = acc - sum([self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane])
    #             wait_time += self.env.vehicles[veh][veh_lane]
    #         wait_time_per_lane.append(wait_time)
    #     return wait_time_per_lane

    # def get_pressure(self):
    #     return abs(sum(self.sumo.lanearea.getLastStepVehicleNumber(detector) for detector in self.detectors) - sum(
    #         self.sumo.lanearea.getLastStepVehicleNumber(detector) for detector in self.out_detectors))

    # def get_out_detectors_density(self):
    #     vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
    #     return [min(1.0, self.sumo.lanearea.getLastStepVehicleNumber(detector) / (
    #                 self.sumo.lanearea.getLength(detector) / vehicle_size_min_gap)) for detector in self.out_detectors]

    # # ***** get observations of ALL detectors
    # def get_detectors_density(self):
    #     vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
    #     return [min(1.0, self.sumo.lanearea.getLastStepVehicleNumber(detector) / (self.detectors_lenght[detector] / vehicle_size_min_gap))
    #             for detector in self.detectors]

    # def get_detectors_queue(self):
    #     vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
    #     return [min(1.0, self.sumo.lanearea.getLastStepHaltingNumber(detector) / (self.detectors_lenght[detector] / vehicle_size_min_gap))
    #             for detector in self.detectors]

    # def get_detectors_jam(self):
    #     vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
    #     return [min(1.0, self.sumo.lanearea.getJamLengthVehicle(detector) / (self.detectors_lenght[detector] / vehicle_size_min_gap))
    #             for detector in self.detectors]

    # def get_detectors_occupancy(self):
    #     return [self.sumo.lanearea.getLastStepOccupancy(detector) / 100 for detector in self.detectors]

    # def get_detectors_speed(self):
    #     # TODO: Colocar el max_speed correcto
    #     max_speed = 13.89
    #     return [self.sumo.lanearea.getLastStepMeanSpeed(detector) / max_speed for detector in self.detectors]

    # # ***** 
    
    # ***** get observations of one detector
    def get_detector_density(self, detector):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        return min(1.0, self.sumo.lanearea.getLastStepVehicleNumber(detector) / (self.detectors_lenght[detector] / vehicle_size_min_gap))

    def get_detector_queue(self, detector):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        return min(1.0, self.sumo.lanearea.getLastStepHaltingNumber(detector) / (self.detectors_lenght[detector] / vehicle_size_min_gap))

    # def get_detector_jam(self, detector):
    #     vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
    #     return min(1.0, self.sumo.lanearea.getJamLengthVehicle(detector) / (self.detectors_lenght[detector] / vehicle_size_min_gap))

    def get_detector_occupancy(self, detector):
        return min(1.0, self.sumo.lanearea.getLastStepOccupancy(detector) / 100)

    def get_detector_speed(self, detector):
        max_speed = 13.89
        speed = self.sumo.lanearea.getLastStepMeanSpeed(detector)
        # speed = speed if speed > 0 else max_speed
        # speed = speed if speed > 0 else 0
        speed = speed if speed > 0 else np.nan
        return min(1.0, speed / max_speed) if not np.isnan(speed) else np.nan
    
    def get_detector_speed_pred(self, detector):
        max_speed = 13.89
        speed = self.sumo.lanearea.getLastStepMeanSpeed(detector)
        speed = speed if speed >= 0 else max_speed
        return min(1.0, speed / max_speed)

    def get_detector_veh_left(self, detector):
        veh_list = self.sumo.lanearea.getLastStepVehicleIDs(detector)
        # vehicle_num = self.sumo.lanearea.getLastStepVehicleNumber(detector)
        veh_left = 0
        for veh in self.veh_ids[detector]:
            if veh not in veh_list:
                veh_left+=1
        
        self.veh_ids[detector] = veh_list

        return veh_left
    
    def get_detector_veh_number(self, detector):
        return self.sumo.lanearea.getLastStepVehicleNumber(detector)

    # ***** 

    def get_total_queued(self):
        return sum([self.sumo.lanearea.getLastStepHaltingNumber(detector) for detector in self.detectors])

    # def _get_veh_list(self):
    #     veh_list = []
    #     for detector in self.detectors:
    #         veh_list += self.sumo.lanearea.getLastStepVehicleIDs(detector)
    #     return veh_list

    # # ****** NO SE USA ******
    # def _phase_idx_to_action(self): 

    #     d = {}
    #     a = 0
    #     for i, phase in enumerate(self.sumo_green_phases_idx):
    #         if phase: 
    #             d[i] = a
    #             a +=1
        
    #     self.phase_idx_to_action = d


    def _gen_sumo_phase_idx_to_splits(self):

        d_green = {}
        a = 0
        for i, phase in enumerate(self.sumo_green_phases_idx):
            if phase:
                d_green[a] = i
                a +=1

        d_model = {}
        a = 0
        for i, phase in enumerate(self.model_phases_idx):
            if phase:
                d_model[a] = i
                a +=1

        d_final = {}
        for key in d_model:
            d_final[d_green[key]] = d_model[key]
        return d_final

    def _get_tls_model_phases_idx(self, net):  # mask
        # model_phases_idx = [1,0,0,1,1,0,0,1]
        controlled_links = self.sumo.trafficlight.getControlledLinks(self.id)
        lanes_coord = {lane: net.getLane(lane).getShape(True) for lane in self.lanes}
        out_lanes_coord = {lane: net.getLane(lane).getShape(True) for lane in self.out_lanes}
        lanes_coord.update(out_lanes_coord)
        junction_coord = lanes_coord[self.lanes[0]][3]
        lanes_cardinal = {}
        for lane in lanes_coord:
            if lanes_coord[lane][1][0] == lanes_coord[lane][2][0]:
                # orientation = 'vertical'
                if lanes_coord[lane][1][1] > junction_coord[1]:
                    lanes_cardinal[lane] = 'north'
                else:
                    lanes_cardinal[lane] = 'south'
            elif lanes_coord[lane][1][1] == lanes_coord[lane][2][1]:
                # orientation = 'horizontal'
                if lanes_coord[lane][1][0] > junction_coord[0]:
                    lanes_cardinal[lane] = 'east'
                else:
                    lanes_cardinal[lane] = 'west'
            else:
                print(f'orientation Error')

        controlled_links_direction = {}
        for idx, link in enumerate(controlled_links):
            controlled_links_direction[idx] = f'{lanes_cardinal[link[0][0]][0]}{lanes_cardinal[link[0][1]][0]}'

        tls_moves_list = []
        for idx_sumo_p, phase in enumerate(self.logic.phases):
            moves = []
            for idx_l, link in enumerate(phase.state):
                if (link == 'G' or link == 'g') and (controlled_links_direction[idx_l] in self.model_dir_to_mov):
                    moves.append(self.model_dir_to_mov[controlled_links_direction[idx_l]])
            if moves:
                moves.sort()
                tls_moves_list.append(tuple(dict.fromkeys(moves)))

        model_phases_idx = [0, 0, 0, 0, 0, 0, 0, 0]
        model_moves_to_phase = {v: k for k, v in self.model_phases_to_mov.items()}
        for move in tls_moves_list:
            model_phases_idx[model_moves_to_phase[move]] = 1

        return model_phases_idx

    def _get_mov_to_detector(self, net):
        lane_to_detector = {self.sumo.lanearea.getLaneID(detector): detector for detector in self.detectors}
        lanes_coord = {lane: net.getLane(lane).getShape(True) for lane in self.lanes}
        out_lanes_coord = {lane: net.getLane(lane).getShape(True) for lane in self.out_lanes}
        lanes_coord.update(out_lanes_coord)
        junction_coord = lanes_coord[self.lanes[0]][3]
        lanes_cardinal = {}
        for lane in lanes_coord:
            if lanes_coord[lane][1][0] == lanes_coord[lane][2][0]:
                # orientation = 'vertical'
                if lanes_coord[lane][1][1] > junction_coord[1]:
                    lanes_cardinal[lane] = 'north'
                else:
                    lanes_cardinal[lane] = 'south'
            elif lanes_coord[lane][1][1] == lanes_coord[lane][2][1]:
                # orientation = 'horizontal'
                if lanes_coord[lane][1][0] > junction_coord[0]:
                    lanes_cardinal[lane] = 'east'
                else:
                    lanes_cardinal[lane] = 'west'
            else:
                print(f'orientation Error')

        directions = self.model_dir_to_mov.keys()
        mov_to_detector = {}
        for lane in lane_to_detector:
            out_links = self.sumo.lane.getLinks(lane)
            for out_link in out_links:
                dir = (lanes_cardinal[lane][0] + lanes_cardinal[out_link[0]][0])
                if dir in directions:
                    mov_to_detector[self.model_dir_to_mov[dir]] = lane_to_detector[lane]
                    continue

        return mov_to_detector

    def softmax(self, x):        
        f_x = np.exp(x) / np.sum(np.exp(x))
        return f_x

    def softmax2(self, x):
        y = np.exp(x - np.max(x))
        f_x = y / np.sum(np.exp(x))
        return f_x

    def _get_act_to_split(self, act):
        act = self.softmax(act)
        split = np.zeros(len(self.model_phases_idx))
        act_idx=0
        for phase_idx in range(len(self.model_phases_idx)):
            if self.model_phases_idx[phase_idx] == 1:
                split[phase_idx]=act[act_idx]
                act_idx+=1
        return split