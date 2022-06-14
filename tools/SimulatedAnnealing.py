import copy
import math
import torch
from siamfc import TrackerSiamFC
from copy import deepcopy
from simanneal import Annealer
import sqlite3
# from networks import LaneNet3D
# from tools.utils import *
import random
import glob
from train import train_sim
from test_module import test_tracker
# from tools import eval_lane_tusimple, eval_3D_lane
import os
# from modeltester import arch_tester


class SimAnealler(Annealer):
    def __init__(self, state, init_model):
        super(SimAnealler, self).__init__(state)
        self.stage = 0
        self.state = state
        self.last_loss = torch.inf
        self.eval_state = None
        self.last_e = torch.inf
        self.last_avg_time = torch.inf
        self.last_arch = [torch.ones((4, 2), dtype=torch.int64), torch.ones((4, 4, 2), dtype=torch.int64)]
        self.last_model = deepcopy(init_model)
        del init_model
        self.last_db_entry = None
        self.num = 132
        self.start = deepcopy(state)
        self.num_params = torch.inf
        self.best = math.inf
        self.best_v_loss = math.inf
        self.ao = 0
        self.sr = 0
        # self.precision_score = 0
        self.speed_fps = torch.inf
        # self.change = {'F': 0, 'invertd': 0, 'layer': 0}
        self.path = r'sim_results'
        self.db = self.path + '/bests_final_new_energy.db'
        self.generated_states = [deepcopy(init_state)]
        self.fps = torch.inf
        self.energy_penalty_term = 1.25
        #self.create_database()

    def create_database(self):
        try:
            os.remove(self.db)
        except OSError:
            pass

        # got10k
        conn = sqlite3.connect(self.db)
        c = conn.cursor()
        c.execute('''CREATE TABLE bests
                                     (num int, arc text, train_loss real,num_params real ,energy real,
                                     ao real, sr real, speed_fps real)''')
        conn.commit()
        c = conn.cursor()
        c.execute('''CREATE TABLE _all_
                                     (num int, arc text, train_loss real,num_params real ,energy real,
                                     ao real, sr real, speed_fps real)''')
        conn.commit()
        conn.close()
        # otb15
        # conn = sqlite3.connect(self.db)
        # c = conn.cursor()
        # c.execute('''CREATE TABLE bests
        #                                      (num int, arc text, train_loss real,success_rate real ,energy real,
        #                                      success_score real, precision_score real, speed_fps real)''')
        # conn.commit()
        # c = conn.cursor()
        # c.execute('''CREATE TABLE _all_
        #                                      (num int, arc text, train_loss real,success_rate real ,energy real,
        #                                      success_score real, precision_score real, speed_fps real)''')
        # conn.commit()
        # conn.close()

    def check_similarity(self, new_state):
        for g_s in self.generated_states:
            if torch.equal(g_s[0], new_state[0]):
                return True
        self.generated_states.append(deepcopy(new_state))
        return False

    def move(self):

        if self.stage == 0:
            self.stage = 1
            change_row = random.randint(0, 3)
            change_ex = random.choice([True, False])
            if change_ex:
                sum_change_row = (self.state[0][change_row] == 1).sum()
                layers_num = random.randint(0, 3)
                while layers_num == sum_change_row:
                    layers_num = random.randint(0, 3)

                self.state[0][change_row] *= 0
                for i in range(layers_num):
                    self.state[0][change_row][i] = 1
            # change_row = random.randint(0, 3)
            # change_col = random.randint(0, 1)
            # change_ex = random.choice([True, False])
            # if change_ex:
            #     if self.state[0][change_row][change_col] == 0:
            #         # input_ = random.randint(0, 1)
            #         self.state[0][change_row][change_col] = 1
            #         # self.state[0][change_row][change_col] = input_
            #     else:
            #         self.state[0][change_row][change_col] = 0
            # else:
            #     input_ = random.randint(0, 1)
            #     self.state[0][change_row][change_col] = input_
        # if self.stage == 0:
        #     self.stage = 1
        #     while_loop = True
        #     while while_loop:
        #         change_row = random.randint(0, 3)
        #         change_col = random.randint(0, 3)
        #         input_ = random.randint(0, 3)
        #         if self.state[0][change_row][change_col] != input_:
        #             if change_col == 2:
        #                 if (self.state[0][change_row][1] > input_) and (self.state[0][change_row][0] < input_):
        #                     nn_state = deepcopy(self.state)
        #                     nn_state[0][change_row][change_col] = input_
        #                     # print(while_loop)
        #                     # while_loop = self.check_similarity(new_state=nn_state)
        #                     while_loop = False
        #             elif change_col == 1:
        #                 if (self.state[0][change_row][0] < input_) and (self.state[0][change_row][2] < input_):
        #                     nn_state = deepcopy(self.state)
        #                     nn_state[0][change_row][change_col] = input_
        #                     while_loop = False
        #                     # while_loop = self.check_similarity(new_state=nn_state)
        #                     # print(while_loop)
        #             elif change_col == 0:
        #                 if (self.state[0][change_row][1] > input_) and (self.state[0][change_row][2] > input_):
        #                     nn_state = deepcopy(self.state)
        #                     nn_state[0][change_row][change_col] = input_
        #                     while_loop = False
        #                     # while_loop = self.check_similarity(new_state=nn_state)
        #                     # print(while_loop)
        #             elif (change_col == 3) and (input_ < 3):
        #                 nn_state = deepcopy(self.state)
        #                 nn_state[0][change_row][change_col] = input_
        #                 while_loop = False
        #                 # while_loop = self.check_similarity(new_state=nn_state)
        #         else:
        #             while_loop = True
        #     self.state = nn_state

        elif self.stage == 1:
            self.stage = 0
            change_row = random.randint(0, 3)
            change_col = random.randint(0, 2)
            change_ex = random.choice([True, False])
            if change_ex:
                if self.state[1][change_row][change_col][0] == 0:
                    input_ = random.randint(0, 3)
                    self.state[1][change_row][change_col][0] = 1
                    self.state[1][change_row][change_col][1] = input_
                else:
                    self.state[1][change_row][change_col][0] = 0
            else:
                input_ = random.randint(0, 3)
                self.state[1][change_row][change_col][1] = input_

        return self.energy()

    def energy(self):

        if torch.equal(self.state[0], self.last_arch[0]) and \
                torch.equal(self.state[1], self.last_arch[1]):

            arch_loss = self.last_loss
            avg_infer_time = self.last_avg_time
            e = self.last_e * 1.1

        else:
            tracker, arch_loss, self.num_params = train_sim(self.state, self.last_model, self.num)
            self.ao, self.sr,  self.speed_fps = test_tracker(self.state, self.num)
            self.last_model = deepcopy(tracker)
            avg_infer_time = self.speed_fps
            alpha = 0.75
            # e = (alpha * (((1 / self.ao) + (1 / self.sr)) + (1 - alpha) * (100 / self.speed_fps)) * max(1, (
            #             (100 / self.speed_fps) - self.energy_penalty_term))) / 1000
            e = (alpha * ((1 / self.ao) + (1 / self.sr)) + ((1 - alpha) * (1 / self.speed_fps))) / 1000
            self.last_e = e
            del tracker.net
            del tracker
            torch.cuda.empty_cache()

        state_string = str(self.state)

        conn = sqlite3.connect(self.db)
        c = conn.cursor()
        c.execute('''INSERT INTO _all_ VALUES (?,?,?,?,?,?,?,?)''',
                  [self.num, state_string, arch_loss, self.num_params, e, self.ao, self.sr,
                   self.speed_fps])
        conn.commit()
        conn.close()

        if e < self.best:
            conn = sqlite3.connect(self.db)
            c = conn.cursor()
            c.execute('''INSERT INTO bests VALUES (?,?,?,?,?,?,?,?)''',
                      [self.num, state_string, arch_loss, self.num_params, e, self.ao, self.sr,
                       self.speed_fps])
            conn.commit()
            conn.close()
            self.best = e
            self.best_energy = e
            self.best_state = copy.deepcopy(self.state)
        self.num = self.num + 1


        # self.last_db_entry = copy.deepcopy(db_entry)
        self.last_arch = copy.deepcopy(self.state)
        self.last_loss = deepcopy(arch_loss)
        self.last_avg_time = avg_infer_time
        if self.best_v_loss > arch_loss:
            self.best_v_loss = arch_loss

        # self.eval_state = eval_state

        e = math.inf
        return e


def Sim_Annealer(init_state, init_model):  # initialize Simulated annealing
    tsp = SimAnealler(init_state, init_model)
    # tsp.Tmax = 25000.0
    # tsp.Tmin = 25.0
    tsp.steps = 300
    tsp.copy_strategy = "deepcopy"
    state, e = tsp.anneal()
    return state


if __name__ == '__main__':
    backbone_state = torch.LongTensor([[0, 0, 0],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 0]])
    # print(backbone_state.shape)
    fusion_state = torch.LongTensor([[[0, 3],
         [1, 3],
         [0, 3],
         [1, 0]],

        [[1, 2],
         [1, 2],
         [1, 2],
         [0, 0]],

        [[0, 2],
         [0, 1],
         [0, 3],
         [1, 0]],

        [[0, 2],
         [0, 1],
         [0, 1],
         [1, 0]]])
    # print(fusion
    # _state.shape)

    # backbone_state = torch.LongTensor([[0, 0, 0, 0],
    #                                    [1, 1, 1, 1],
    #                                    [2, 2, 2, 2],
    #                                    [3, 3, 3, 2]])
    # fusion_state = torch.LongTensor([[[1, 3],
    #                                  [1, 3],
    #                                  [0, 3],
    #                                  [1, 0]],
    #
    #                                 [[1, 2],
    #                                  [1, 2],
    #                                  [1, 2],
    #                                  [0, 0]],
    #
    #                                 [[0, 2],
    #                                  [0, 1],
    #                                  [0, 3],
    #                                  [1, 0]],
    #
    #                                 [[0, 2],
    #                                  [0, 1],
    #                                  [0, 1],
    #                                  [1, 0]]])

    init_state = [backbone_state, fusion_state]
    net_path = r'/home/sadegh/PycharmProjects/siam-cuda01/tools/pretrained/Architecture_number_131.pth'
    init_model = TrackerSiamFC(state=init_state, net_path=net_path)
    # print(init_model.net.parameters)
    SA_state = Sim_Annealer(init_state, init_model)
