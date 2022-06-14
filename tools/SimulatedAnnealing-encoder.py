import copy
import math
import torch
from siamfc import TrackerSiamFC
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


class SimAnealler(Annealer):
    def __init__(self, state, init_model):
        super(SimAnealler, self).__init__(state)
        self.stage = 1
        self.last_loss = math.inf
        self.eval_state = None
        self.last_e = math.inf
        self.last_avg_time = math.inf
        self.last_arch = [torch.ones((4, 2), dtype=torch.int64), torch.ones((4, 4, 2), dtype=torch.int64)]
        self.last_model = init_model
        self.last_db_entry = None
        self.num = 0
        self.best = math.inf
        self.best_v_loss = math.inf
        self.ao = 0
        self.sr = 0
        # self.precision_score = 0
        self.speed_fps = 0
        self.change = {'F': 0, 'invertd': 0, 'layer': 0}
        self.path = r'sim_results'
        self.db = self.path + '/bests.db'
        self.create_database()

    def create_database(self):
        try:
            os.remove(self.db)
        except OSError:
            pass

        # got10k
        conn = sqlite3.connect(self.db)
        c = conn.cursor()
        c.execute('''CREATE TABLE bests
                                     (num int, arc text, train_loss real,avg_infer_time real ,energy real,
                                     ao real, sr real, speed_fps real)''')
        conn.commit()
        c = conn.cursor()
        c.execute('''CREATE TABLE _all_
                                     (num int, arc text, train_loss real,avg_infer_time real ,energy real,
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

    def move(self):
        if self.stage == 0:
            self.stage = 1
            change_row = random.randint(0, 3)
            change_col = random.randint(0, 3)
            input_ = random.randint(0, 3)
            while self.state[0][change_row][change_col] == input_:
                input_ = random.randint(0, 3)
            self.state[0][change_row][change_col] = input_

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
            # db_entry = self.last_db_entry
            # eval_state = self.eval_state

        else:

            tracker, arch_loss = train_sim(self.state, self.last_model, self.num)

            self.ao, self.sr, self.speed_fps = test_tracker(self.state, self.num)
            self.last_model = tracker
            avg_infer_time = self.speed_fps
            e = 1 / (self.ao * avg_infer_time * self.sr)
            self.last_e = e
            del tracker.net
            del tracker
            torch.cuda.empty_cache()
        state_string = str(self.state)

        conn = sqlite3.connect(self.db)
        c = conn.cursor()
        c.execute('''INSERT INTO _all_ VALUES (?,?,?,?,?,?,?,?)''',
                  [self.num, state_string, arch_loss, avg_infer_time, e, self.ao, self.sr, self.speed_fps])
        conn.commit()
        conn.close()

        if e < self.best:
            conn = sqlite3.connect(self.db)
            c = conn.cursor()
            c.execute('''INSERT INTO bests VALUES (?,?,?,?,?,?,?,?)''',
                      [self.num, state_string, arch_loss, avg_infer_time, e, self.ao, self.sr,
                       self.speed_fps])
            conn.commit()
            conn.close()
            self.best = e
        self.num = self.num + 1


        # self.last_db_entry = copy.deepcopy(db_entry)
        self.last_arch = copy.deepcopy(self.state)
        self.last_loss = arch_loss
        self.last_avg_time = avg_infer_time
        # self.eval_state = eval_state

        e = math.inf

        return e


def Sim_Annealer(init_state, init_model):  # initialize Simulated annealing
    tsp = SimAnealler(init_state, init_model)
    tsp.Tmax = 25000.0
    tsp.Tmin = 25.0
    tsp.copy_strategy = "deepcopy"
    state, e = tsp.anneal()
    return state


if __name__ == '__main__':
    backbone_state = torch.LongTensor([[0, 2, 1, 1],
                                       [0, 2, 1, 1],
                                       [0, 2, 1, 1],
                                       [0, 2, 1, 1]])
    fusion_state = torch.LongTensor([[[0, 3],
                                      [0, 3],
                                      [1, 3],
                                      [0, 2]],

                                     [[0, 0],
                                      [1, 0],
                                      [0, 2],
                                      [0, 0]],

                                     [[0, 3],
                                      [0, 1],
                                      [0, 3],
                                      [1, 0]],

                                     [[0, 3],
                                      [0, 1],
                                      [0, 2],
                                      [1, 0]]])

    init_state = [backbone_state, fusion_state]

    init_model = TrackerSiamFC(state=init_state)

    SA_state = Sim_Annealer(init_state, init_model)
