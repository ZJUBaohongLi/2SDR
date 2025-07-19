import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from config import Config

sample_num = Config.sample_num


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logistic(x):
    res = np.where(x >= 0.5, 1, 0)
    return res


def classify(x):
    res = np.where(x - np.mean(x) >= 0, 1, 0)
    return res


def generate_confounds_high_dim(dim=100):
    w = np.array([5 for i in range(dim)]).reshape(-1, 1)
    delta = np.array(
        [5 / int(dim * 0.4) for i in range(int(dim * 0.4))] + [5 / int(dim * 0.1) for i in range(int(dim * 0.1))] +
        [-3 / int(dim * 0.1) for i in range(int(dim * 0.1))] + [0 for i in range(int(dim * 0.2))] +
        [0 for i in range(int(dim * 0.2))]).reshape(-1, 1)
    beta = np.array([-0.5 / int(dim * 0.4) for i in range(int(dim * 0.4))] + [0 for i in range(int(dim * 0.1))] +
                    [0.5 / int(dim * 0.1) for i in range(int(dim * 0.1))] + [-0.3 / int(dim * 0.2) for i in
                                                                             range(int(dim * 0.2))] +
                    [0 for i in range(int(dim * 0.2))]).reshape(-1, 1)
    x1 = np.random.binomial(1, 0.5, (sample_num, 1))
    x22 = np.random.randn(sample_num, int(dim * 0.1)) + 1
    x21 = np.random.randn(sample_num, int(dim * 0.4)) + 1
    convert_mat_14 = np.random.randn(int(dim * 0.4), int(dim * 0.2))
    x24 = x21.dot(convert_mat_14) + np.random.randn(sample_num, int(dim * 0.2)) * 0.1
    x25 = np.random.randn(sample_num, int(dim * 0.2)) + 1
    convert_mat_53 = np.random.randn(int(dim * 0.2), int(dim * 0.1))
    x23 = x25.dot(convert_mat_53) + np.random.randn(sample_num, int(dim * 0.1)) * 0.1
    x2 = np.concatenate((x21, x22, x23, x24, x25), axis=1)
    y = 5 + np.dot(x2, w) + x1 * np.dot(x2, delta) + np.random.randn(
        sample_num, 1)
    ground_truth = np.dot(x2, delta)
    rct_x1 = np.random.binomial(1, 0.5, (sample_num * 10, 1))
    rct_x22 = np.random.randn(sample_num * 10, int(dim * 0.1)) + 1
    rct_x21 = np.random.randn(sample_num * 10, int(dim * 0.4)) + 1
    rct_x24 = rct_x21.dot(convert_mat_14) + np.random.randn(sample_num * 10, int(dim * 0.2)) * 0.1
    rct_x25 = np.random.randn(sample_num * 10, int(dim * 0.2)) + 1
    rct_x23 = rct_x25.dot(convert_mat_53) + np.random.randn(sample_num * 10, int(dim * 0.1)) * 0.1
    rct_x2 = np.concatenate((rct_x21, rct_x22, rct_x23, rct_x24, rct_x25), axis=1)
    rct_y = 5 + np.dot(rct_x2, w) + rct_x1 * np.dot(rct_x2, delta) + np.random.randn(
        sample_num * 10, 1)
    rct_gt = np.dot(rct_x2, delta)
    rct_s1 = sigmoid(np.dot(rct_x2, beta) + np.random.randn(
        sample_num * 10, 1))
    for i in range(len(rct_s1)):
        rct_s1[i, 0] = np.random.binomial(1, rct_s1[i, 0], 1)
    selected_index = np.where(rct_s1.reshape(-1) == 1)[0]
    new_selected_index = np.random.permutation(selected_index)
    new_x1 = rct_x1[new_selected_index]
    new_x2 = rct_x2[new_selected_index]
    new_y = rct_y[new_selected_index]
    new_gt = rct_gt[new_selected_index]
    np_data = np.concatenate((x1, y, x2, ground_truth), axis=1)
    pd_column = ['X1', 'Y'] + ['X2_' + str(i) for i in range(1, dim + 1)] + ['GT']
    pd_data = pd.DataFrame(np_data, columns=pd_column)
    pd_data.to_csv('data/highdim_confound_data.csv')
    np_data = np.concatenate((new_x1[:sample_num // Config.obs_ratio], new_y[:sample_num // Config.obs_ratio],
                              new_x2[:sample_num // Config.obs_ratio], new_gt[:sample_num // Config.obs_ratio]),
                             axis=1)
    pd_column = ['X1', 'Y'] + ['X2_' + str(i) for i in range(1, dim + 1)] + ['GT']
    pd_data = pd.DataFrame(np_data, columns=pd_column)
    pd_data.to_csv('data/highdim_rct_data.csv')


def generate_confounds_low_dim():
    w = np.array([5 for i in range(5)]).reshape(-1, 1)
    delta = np.array([5, 5, -3, 0, 0]).reshape(-1, 1)
    beta = np.array([-0.5, 0, 0.5, -0.3, 0]).reshape(-1, 1)
    x1 = np.random.binomial(1, 0.5, (sample_num, 1))
    x22 = np.random.randn(sample_num, 1) + 1
    x21, x24 = np.random.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], sample_num).T + 1
    x21 = x21.reshape(-1, 1)
    x24 = x24.reshape(-1, 1)
    x23, x25 = np.random.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], sample_num).T + 1
    x23 = x23.reshape(-1, 1)
    x25 = x25.reshape(-1, 1)
    x2 = np.concatenate((x21, x22, x23, x24, x25), axis=1)
    y = 5 + np.dot(x2, w) + x1 * np.dot(x2, delta) + np.random.randn(
        sample_num, 1)
    ground_truth = np.dot(x2, delta)

    rct_x1 = np.random.binomial(1, 0.5, (sample_num * 10, 1))
    rct_x22 = np.random.randn(sample_num * 10, 1) + 1
    rct_x21, rct_x24 = np.random.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], sample_num * 10).T + 1
    rct_x21 = rct_x21.reshape(-1, 1)
    rct_x24 = rct_x24.reshape(-1, 1)
    rct_x23, rct_x25 = np.random.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]],
                                                     sample_num * 10).T + 1
    rct_x23 = rct_x23.reshape(-1, 1)
    rct_x25 = rct_x25.reshape(-1, 1)
    rct_x2 = np.concatenate((rct_x21, rct_x22, rct_x23, rct_x24, rct_x25), axis=1)
    rct_y = 5 + np.dot(rct_x2, w) + rct_x1 * np.dot(rct_x2, delta) + np.random.randn(
        sample_num * 10, 1)
    rct_gt = np.dot(rct_x2, delta)
    rct_s1 = sigmoid(np.dot(rct_x2, beta)+ np.random.randn(
        sample_num * 10, 1))
    for i in range(len(rct_s1)):
        rct_s1[i, 0] = np.random.binomial(1, rct_s1[i, 0], 1)
    selected_index = np.where(rct_s1.reshape(-1) == 1)[0]
    new_selected_index = np.random.permutation(selected_index)
    new_x1 = rct_x1[new_selected_index]
    new_x2 = rct_x2[new_selected_index]
    new_y = rct_y[new_selected_index]
    new_gt = rct_gt[new_selected_index]
    np_data = np.concatenate((x1, y, x2, ground_truth), axis=1)
    pd_column = ['X1', 'Y'] + ['X2_' + str(i) for i in range(1, Config.confounds_num + 1)] + ['GT']
    pd_data = pd.DataFrame(np_data, columns=pd_column)
    pd_data.to_csv('data/simulate_confound_data.csv')
    np_data = np.concatenate((new_x1[:sample_num // Config.obs_ratio], new_y[:sample_num // Config.obs_ratio],
                              new_x2[:sample_num // Config.obs_ratio], new_gt[:sample_num // Config.obs_ratio]),
                             axis=1)
    pd_column = ['X1', 'Y'] + ['X2_' + str(i) for i in range(1, Config.confounds_num + 1)] + ['GT']
    pd_data = pd.DataFrame(np_data, columns=pd_column)
    pd_data.to_csv('data/simulate_rct_data.csv')


def simulation():
    generate_confounds_low_dim()
    data = pd.read_csv('data/simulate_confound_data.csv')
    t = np.array(data['X1']).reshape((-1, 1))
    y = np.array(data['Y']).reshape((-1, 1))
    if Config.setting == 1:
        x = np.array(data[['X2_1', 'X2_4', 'X2_5']])
        xo = np.array(data['X2_2']).reshape((-1, 1))
        xr = np.array(data['X2_3']).reshape((-1, 1))
    elif Config.setting == 2:
        x = np.array(data[['X2_1', 'X2_4', 'X2_5']])
        xr = np.array(data['X2_2']).reshape((-1, 1))
        xo = np.array(data['X2_3']).reshape((-1, 1))
    gt = np.array(data['GT']).reshape((-1, 1))
    t_train, t_test, y_train, y_test, x_train, x_test, xr_train, xr_test, xo_train, xo_test, gt_train, gt_test = train_test_split(
        t, y, x, xr, xo, gt, test_size=0.50)
    t_test, t_val, y_test, y_val, x_test, x_val, xr_test, xr_val, xo_test, xo_val, gt_test, gt_val = train_test_split(
        t_test, y_test, x_test, xr_test, xo_test, gt_test, test_size=0.50)
    obs_data_train = {
        't': t_train,
        'x': x_train,
        'y': y_train,
        'xr': xr_train,
        'xo': xo_train,
        'gt': gt_train
    }
    obs_data_val = {
        't': t_val,
        'x': x_val,
        'y': y_val,
        'xr': xr_val,
        'xo': xo_val,
        'gt': gt_val
    }
    obs_data_test = {
        't': t_test,
        'x': x_test,
        'y': y_test,
        'xr': xr_test,
        'xo': xo_test,
        'gt': gt_test
    }
    data = pd.read_csv('data/simulate_rct_data.csv')
    t = np.array(data['X1']).reshape((-1, 1))
    y = np.array(data['Y']).reshape((-1, 1))
    if Config.setting == 1:
        x = np.array(data[['X2_1', 'X2_4', 'X2_5']])
        xr = np.array(data['X2_3']).reshape((-1, 1))
        xo = np.array(data['X2_2']).reshape((-1, 1))
    elif Config.setting == 2:
        x = np.array(data[['X2_1', 'X2_4', 'X2_5']])
        xr = np.array(data['X2_2']).reshape((-1, 1))
        xo = np.array(data['X2_3']).reshape((-1, 1))
    gt = np.array(data['GT']).reshape((-1, 1))
    t_train, t_test, y_train, y_test, x_train, x_test, xr_train, xr_test, xo_train, xo_test, gt_train, gt_test = train_test_split(
        t, y, x, xr, xo, gt, test_size=0.50)
    t_test, t_val, y_test, y_val, x_test, x_val, xr_test, xr_val, xo_test, xo_val, gt_test, gt_val = train_test_split(
        t_test, y_test, x_test, xr_test, xo_test, gt_test, test_size=0.50)
    rct_data_train = {
        't': t_train,
        'y': y_train,
        'x': x_train,
        'xr': xr_train,
        'xo': xo_train,
        'gt': gt_train
    }
    rct_data_val = {
        't': t_val,
        'y': y_val,
        'x': x_val,
        'xr': xr_val,
        'xo': xo_val,
        'gt': gt_val
    }
    rct_data_test = {
        't': t_test,
        'y': y_test,
        'x': x_test,
        'xr': xr_test,
        'xo': xo_test,
        'gt': gt_test
    }
    return obs_data_train, obs_data_val, obs_data_test, rct_data_train, rct_data_val, rct_data_test


def simulation_high_dim():
    generate_confounds_high_dim(Config.confounds_num)
    data = pd.read_csv('data/highdim_confound_data.csv')
    t = np.array(data['X1']).reshape((-1, 1))
    y = np.array(data['Y']).reshape((-1, 1))
    if Config.setting == 1:
        x = np.array(data[['X2_' + str(i) for i in range(1, int(Config.confounds_num * 0.4) + 1)] +
                          ['X2_' + str(i) for i in
                           range(int(Config.confounds_num * 0.6) + 1, Config.confounds_num + 1)]])
        xo = np.array(data[['X2_' + str(i) for i in
                            range(int(Config.confounds_num * 0.4) + 1, int(Config.confounds_num * 0.5) + 1)]])
        xr = np.array(data[['X2_' + str(i) for i in
                            range(int(Config.confounds_num * 0.5) + 1, int(Config.confounds_num * 0.6) + 1)]])
    elif Config.setting == 2:
        x = np.array(data[['X2_' + str(i) for i in range(1, int(Config.confounds_num * 0.4) + 1)] +
                          ['X2_' + str(i) for i in
                           range(int(Config.confounds_num * 0.6) + 1, Config.confounds_num + 1)]])
        xr = np.array(data[['X2_' + str(i) for i in
                            range(int(Config.confounds_num * 0.4) + 1, int(Config.confounds_num * 0.5) + 1)]])
        xo = np.array(data[['X2_' + str(i) for i in
                            range(int(Config.confounds_num * 0.5) + 1, int(Config.confounds_num * 0.6) + 1)]])
    gt = np.array(data['GT']).reshape((-1, 1))
    t_train, t_test, y_train, y_test, x_train, x_test, xr_train, xr_test, xo_train, xo_test, gt_train, gt_test = train_test_split(
        t, y, x, xr, xo, gt, test_size=0.50)
    t_test, t_val, y_test, y_val, x_test, x_val, xr_test, xr_val, xo_test, xo_val, gt_test, gt_val = train_test_split(
        t_test, y_test, x_test, xr_test, xo_test, gt_test, test_size=0.50)
    obs_data_train = {
        't': t_train,
        'y': y_train,
        'x': x_train,
        'xr': xr_train,
        'xo': xo_train,
        'gt': gt_train
    }
    obs_data_val = {
        't': t_val,
        'y': y_val,
        'x': x_val,
        'xr': xr_val,
        'xo': xo_val,
        'gt': gt_val
    }
    obs_data_test = {
        't': t_test,
        'y': y_test,
        'x': x_test,
        'xr': xr_test,
        'xo': xo_test,
        'gt': gt_test
    }
    data = pd.read_csv('data/highdim_rct_data.csv')
    t = np.array(data['X1']).reshape((-1, 1))
    y = np.array(data['Y']).reshape((-1, 1))
    if Config.setting == 1:
        x = np.array(data[['X2_' + str(i) for i in range(1, int(Config.confounds_num * 0.4) + 1)] +
                          ['X2_' + str(i) for i in
                           range(int(Config.confounds_num * 0.6) + 1, Config.confounds_num + 1)]])
        xo = np.array(data[['X2_' + str(i) for i in
                            range(int(Config.confounds_num * 0.4) + 1, int(Config.confounds_num * 0.5) + 1)]])
        xr = np.array(data[['X2_' + str(i) for i in
                            range(int(Config.confounds_num * 0.5) + 1, int(Config.confounds_num * 0.6) + 1)]])
    elif Config.setting == 2:
        x = np.array(data[['X2_' + str(i) for i in range(1, int(Config.confounds_num * 0.4) + 1)] +
                          ['X2_' + str(i) for i in
                           range(int(Config.confounds_num * 0.6) + 1, Config.confounds_num + 1)]])
        xr = np.array(data[['X2_' + str(i) for i in
                            range(int(Config.confounds_num * 0.4) + 1, int(Config.confounds_num * 0.5) + 1)]])
        xo = np.array(data[['X2_' + str(i) for i in
                            range(int(Config.confounds_num * 0.5) + 1, int(Config.confounds_num * 0.6) + 1)]])
    gt = np.array(data['GT']).reshape((-1, 1))
    t_train, t_test, y_train, y_test, x_train, x_test, xr_train, xr_test, xo_train, xo_test, gt_train, gt_test = train_test_split(
        t, y, x, xr, xo, gt, test_size=0.50)
    t_test, t_val, y_test, y_val, x_test, x_val, xr_test, xr_val, xo_test, xo_val, gt_test, gt_val = train_test_split(
        t_test, y_test, x_test, xr_test, xo_test, gt_test, test_size=0.50)
    rct_data_train = {
        't': t_train,
        'y': y_train,
        'x': x_train,
        'xr': xr_train,
        'xo': xo_train,
        'gt': gt_train
    }
    rct_data_val = {
        't': t_val,
        'y': y_val,
        'x': x_val,
        'xr': xr_val,
        'xo': xo_val,
        'gt': gt_val
    }
    rct_data_test = {
        't': t_test,
        'y': y_test,
        'x': x_test,
        'xr': xr_test,
        'xo': xo_test,
        'gt': gt_test
    }
    return obs_data_train, obs_data_val, obs_data_test, rct_data_train, rct_data_val, rct_data_test
