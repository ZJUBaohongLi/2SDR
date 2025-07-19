import traceback
import numpy as np
from models.TSDR.trainer import TSDRTrainer
from config import Config


if __name__ == '__main__':
    try:
        res_list = []
        for idx in range(Config.experiment_num):
            model = TSDRTrainer()
            res = model.train()
            res_list.append(res)
        res_list = np.array(res_list)
        bias = np.abs(np.mean(res_list, axis=0)).reshape(-1, 1)
        sd = np.std(res_list, axis=0).reshape(-1, 1)
        np.savetxt('res/result.txt', np.concatenate((bias, sd), 0))
    except Exception as e:
        print('Exception: ' + str(e))
        print()
        traceback.print_exc()
