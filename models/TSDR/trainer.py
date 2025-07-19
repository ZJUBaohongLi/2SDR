import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KernelDensity
from data_generation import simulation
from models.TSDR.q_func import QFunc
from models.TSDR.selection_net import SelectionNet
from models.TSDR.or_net import ORNet
from causallearn.utils.cit import CIT
from config import Config


class QDataset(Dataset):
    def __init__(self, x, z, xr, s):
        self.x = x
        self.z = z
        self.s = s
        self.xr = xr

    def __getitem__(self, item):
        return self.x[item], self.z[item], self.xr[item], self.s[item]

    def __len__(self):
        return len(self.s)


class SelectionDataset(Dataset):
    def __init__(self, x, s):
        self.x = x
        self.s = s

    def __getitem__(self, item):
        return self.x[item], self.s[item]

    def __len__(self):
        return len(self.s)


def train_logit(x, y, x_val, y_val, pos_weight):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    selection_model = SelectionNet(x.shape[1], 2, 4).to(device)
    opt_selection = torch.optim.Adam(selection_model.parameters(), lr=Config.q_lr)
    selection_train_dataset = SelectionDataset(x, y)
    selection_train_dataloader = DataLoader(selection_train_dataset, batch_size=Config.selection_batchsize,
                                            shuffle=True)
    selection_val_dataset = SelectionDataset(x_val, y_val)
    selection_val_dataloader = DataLoader(selection_val_dataset, batch_size=Config.selection_batchsize // 3,
                                          shuffle=True)
    bce_func = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))
    best_val_loss = None
    counter = 0
    for e in range(Config.selection_epochs):
        selection_model.train()
        loss_selection_sum = 0
        for batch in selection_train_dataloader:
            x, s = batch
            x = x.to(device)
            s = s.to(device)
            s_hat = selection_model(x)
            loss_selection = bce_func(s_hat.float(), s.float())
            opt_selection.zero_grad()
            loss_selection.backward()
            opt_selection.step()
            loss_selection_sum += loss_selection
        selection_model.eval()
        loss_selection_sum = 0
        with torch.no_grad():
            for index, batch in enumerate(selection_val_dataloader):
                x, s = batch
                x = x.to(device)
                s = s.to(device)
                s_hat = selection_model(x)
                loss_selection = bce_func(s_hat.float(), s.float())
                loss_selection_sum += loss_selection
            if best_val_loss is None:
                best_val_loss = loss_selection_sum
            elif best_val_loss < loss_selection_sum:
                counter += 1
                if counter >= 5:
                    break
            else:
                best_val_loss = loss_selection_sum
    return selection_model


def train_ols(x, y):
    ols_model = LinearRegression()
    y_model = ols_model.fit(x, y)
    return y_model


def test_selection(z_train, x_train, xr_train, s_train, z_val, x_val, s_val):
    res = False
    cit_obj = CIT(np.concatenate((z_train, xr_train, x_train), axis=1), "rcit")
    p_val = cit_obj([i for i in range(z_train.shape[1])],
                    [i for i in range(z_train.shape[1], z_train.shape[1] + xr_train.shape[1])],
                    [i for i in range(z_train.shape[1] + xr_train.shape[1],
                                      z_train.shape[1] + xr_train.shape[1] + x_train.shape[1])])
    if p_val < Config.cit_threshold:
        res = True
    if not res:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        q_model = QFunc(x_train.shape[1] + xr_train.shape[1], z_train.shape[1]).to(device)
        opt_q = torch.optim.Adam(q_model.parameters(), lr=Config.q_lr)
        q_train_dataset = QDataset(x_train, z_train, xr_train, s_train)
        q_train_dataloader = DataLoader(q_train_dataset, batch_size=Config.q_batchsize, shuffle=True)
        q_val_dataset = QDataset(x_val, z_val, np.zeros((x_val.shape[0], xr_train.shape[1])), s_val)
        q_val_dataloader = DataLoader(q_val_dataset, batch_size=Config.q_batchsize // 3, shuffle=True)
        for e in range(Config.q_epochs):
            q_model.train()
            loss_q_sum = 0
            for batch in q_train_dataloader:
                xall, zall, xrall, sall = batch
                xrall = xrall.to(device)
                xall = xall.to(device)
                zall = zall.to(device)
                sall = sall.to(device)
                sall = torch.squeeze(sall)
                xr = xrall[sall == 1]
                x = xall[sall == 1]
                z = zall[sall == 1]
                zus = zall[sall == 0]
                xus = xall[sall == 0]
                q_hat = q_model(torch.cat((x, xr), 1))
                loss_q = torch.mean((1 / (q_hat + 1e-4) - 1) * torch.cat((x, z), 1)) - torch.mean(
                    torch.cat((xus, zus), 1))
                loss_q = torch.sqrt(loss_q * loss_q)
                opt_q.zero_grad()
                loss_q.backward()
                opt_q.step()
                loss_q_sum += loss_q
            q_model.eval()
            loss_q_sum = 0
            with torch.no_grad():
                for index, batch in enumerate(q_val_dataloader):
                    xall, zall, xrall, sall = batch
                    xrall = xrall.to(device)
                    xall = xall.to(device)
                    zall = zall.to(device)
                    sall = sall.to(device)
                    sall = torch.squeeze(sall)
                    xr = xrall[sall == 1]
                    x = xall[sall == 1]
                    z = zall[sall == 1]
                    zus = zall[sall == 0]
                    xus = xall[sall == 0]
                    q_hat = q_model(torch.cat((x, xr), 1))
                    loss_q = torch.mean((1 / (q_hat + 1e-4) - 1) * torch.cat((x, z), 1)) - torch.mean(
                        torch.cat((xus, zus), 1))
                    loss_q = torch.sqrt(loss_q * loss_q)
                    loss_q_sum += loss_q
            if loss_q_sum / (index + 1) <= Config.s_threshold:
                res = True
                break
    return res


def get_weight(selection_net, x):
    x = torch.tensor(x, dtype=torch.float32)
    selection_net.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        x = x.to(device)
        p_hat = selection_net(x)
        p_hat = torch.sigmoid(p_hat)
        p_hat = p_hat.detach().cpu().numpy()
    return p_hat.reshape(-1, 1)


def conditional_kde(kde_joint, kde_marginal, joint, marginal):
    pred_joint = kde_joint.score_samples(joint).reshape(-1, 1)
    pred_marginal = kde_marginal.score_samples(marginal).reshape(-1, 1)
    return np.exp(pred_joint - pred_marginal)


def fit_odds(z, x, xr, zus, xus, x_val, z_val, xr_val):
    # f(Z | X, S)
    kde_xz1 = KernelDensity(kernel='gaussian')
    kde_xz1.fit(np.concatenate((x, z), axis=1))
    kde_x1 = KernelDensity(kernel='gaussian')
    kde_x1.fit(x)
    kde_xz0 = KernelDensity(kernel='gaussian')
    kde_xz0.fit(np.concatenate((xus, zus), axis=1))
    kde_x0 = KernelDensity(kernel='gaussian')
    kde_x0.fit(xus)
    # E[OR(X,Y)|X, Z, S=1] = f(Z | X, S=1) / f(Z | X, S=0)
    pred_z0 = conditional_kde(kde_xz0, kde_x0, np.concatenate((x, z), axis=1), x)
    pred_z1 = conditional_kde(kde_xz1, kde_x1, np.concatenate((x, z), axis=1), x)
    or_true = pred_z0 / pred_z1
    pred_s0_val = conditional_kde(kde_xz0, kde_x0, np.concatenate((x_val, z_val), axis=1), x_val)
    pred_s1_val = conditional_kde(kde_xz1, kde_x1, np.concatenate((x_val, z_val), axis=1), x_val)
    or_true_val = pred_s0_val / pred_s1_val
    # OR~(X,Y)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    or_model = ORNet(x.shape[1] + xr.shape[1], 2, 4).to(device)
    opt_or = torch.optim.Adam(or_model.parameters(), lr=Config.or_lr)
    or_train_dataset = SelectionDataset(np.concatenate((x, xr), axis=1), or_true)
    or_train_dataloader = DataLoader(or_train_dataset, batch_size=Config.or_batchsize,
                                     shuffle=True)
    or_val_dataset = SelectionDataset(np.concatenate((x_val, xr_val), axis=1), or_true_val)
    or_val_dataloader = DataLoader(or_val_dataset, batch_size=Config.or_batchsize // 3,
                                   shuffle=True)
    mse_func = nn.MSELoss()
    best_val_loss = None
    counter = 0
    for e in range(Config.or_epochs):
        or_model.train()
        loss_or_sum = 0
        for batch in or_train_dataloader:
            feature, odds = batch
            feature = feature.to(device)
            odds = odds.to(device)
            y_hat = or_model(feature)
            loss_or = mse_func(y_hat.float(), torch.relu(odds).float())
            opt_or.zero_grad()
            loss_or.backward()
            opt_or.step()
            loss_or_sum += loss_or
        or_model.eval()
        loss_or_sum = 0
        with torch.no_grad():
            for index, batch in enumerate(or_val_dataloader):
                feature, odds = batch
                feature = feature.to(device)
                odds = odds.to(device)
                y_hat = or_model(feature)
                loss_or = mse_func(y_hat.float(), torch.relu(odds).float())
                loss_or_sum += loss_or
            if best_val_loss is None:
                best_val_loss = loss_or_sum
            elif best_val_loss < loss_or_sum:
                counter += 1
                if counter >= 5:
                    break
            else:
                best_val_loss = loss_or_sum
    return or_model


def predict_odds(or_model, x, xr):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    feature = torch.tensor(np.concatenate((x, xr), axis=1), dtype=torch.float32)
    or_model.eval()
    with torch.no_grad():
        feature = feature.to(device)
        gen_or_tilde = or_model(feature)
        or_hat = gen_or_tilde.detach().cpu().numpy()
    return or_hat.reshape(-1, 1)


class TSDRTrainer:
    def __init__(self):
        self._biased_model_t: LinearRegression = LinearRegression()
        self._biased_model_c: LinearRegression = LinearRegression()
        self._xr_model: LinearRegression = LinearRegression()
        self._xo_model: LinearRegression = LinearRegression()
        self.scaler: StandardScaler = StandardScaler()
        self.scaler_xr: StandardScaler = StandardScaler()
        self.scaler_xo: StandardScaler = StandardScaler()
        self.t_s = None
        self.y_s = None
        self.x_s = None
        self.x_us = None
        self.xr_s = None
        self.xo_us = None
        self.s1_s = None
        self.s1_us = None

    def train(self):
        obs_data_train, obs_data_val, obs_data_test, rct_data_train, rct_data_val, rct_data_test = simulation()
        self.t_s = rct_data_train['t']
        self.y_s = rct_data_train['y']
        self.x_s = rct_data_train['x']
        self.x_us = obs_data_train['x']
        if Config.setting == 1:
            self.xr_s = rct_data_train['xr']
        else:
            self.xo_us = obs_data_train['xo']
        self.s1_s = np.ones(self.t_s.shape)
        self.s1_us = np.zeros((self.x_us.shape[0], 1))
        xs_val = rct_data_val['x']
        xus_val = obs_data_val['x']
        x_val = np.concatenate((rct_data_val['x'], obs_data_val['x']), axis=0)

        self.scaler.fit(np.concatenate((self.x_s, self.x_us), axis=0))
        x_s_tr = self.scaler.transform(self.x_s)
        if Config.setting == 1:
            xr_s_tr = self.scaler_xr.fit_transform(self.xr_s)
            xr_val = rct_data_val['xr']
            xr_val_tr = self.scaler_xr.transform(xr_val)
        else:
            xo_us_tr = self.scaler_xo.fit_transform(self.xo_us)
            xo_val = obs_data_val['xo']
            xo_val_tr = self.scaler_xo.transform(xo_val)
        x_us_tr = self.scaler.transform(self.x_us)
        x_val_tr = self.scaler.transform(x_val)
        xs_val_tr = self.scaler.transform(xs_val)
        xus_val_tr = self.scaler.transform(xus_val)

        # select Z
        gb_regressor = GradientBoostingRegressor()
        multioutput_regressor = MultiOutputRegressor(gb_regressor)
        if Config.setting == 1:
            multioutput_regressor.fit(x_s_tr, xr_s_tr)
            output_num = xr_s_tr.shape[1]
        else:
            multioutput_regressor.fit(x_us_tr, xo_us_tr)
            output_num = xo_us_tr.shape[1]
        selected_index = set(range(x_s_tr.shape[1]))
        for i in range(output_num):
            selector = SelectFromModel(multioutput_regressor.estimators_[i], threshold=Config.y_threshold)
            if Config.setting == 1:
                selector.fit(x_s_tr, xr_s_tr[:, i])
            else:
                selector.fit(x_us_tr, xo_us_tr[:, i])
            selected_feature = set(selector.get_support(indices=True))
            selected_index &= selected_feature
        final_z_index = []
        for i in selected_index:
            z_s_curr = x_s_tr[:, i].reshape(-1, 1)
            x_s_curr = x_s_tr[:, np.arange(x_s_tr.shape[1]) != i]
            if x_s_curr.ndim == 1:
                x_s_curr = x_s_curr.reshape(-1, 1)
            z_us_curr = x_us_tr[:, i].reshape(-1, 1)
            x_us_curr = x_us_tr[:, np.arange(x_us_tr.shape[1]) != i]
            if x_us_curr.ndim == 1:
                x_us_curr = x_s_curr.reshape(-1, 1)
            z_curr = np.concatenate((z_s_curr, z_us_curr), axis=0)
            x_curr = np.concatenate((x_s_curr, x_us_curr), axis=0)
            z_val_curr = x_val_tr[:, i].reshape(-1, 1)
            x_val_curr = x_val_tr[:, np.arange(x_val_tr.shape[1]) != i]
            if Config.setting == 1:
                s_train = np.concatenate((self.s1_s, self.s1_us), axis=0)
                s_val = np.concatenate(
                    (np.ones((rct_data_val['x'].shape[0], 1)), np.zeros((obs_data_val['x'].shape[0], 1))),
                    axis=0)
                passed = test_selection(z_curr, x_curr,
                                        np.concatenate((xr_s_tr, np.zeros((x_us_tr.shape[0], xr_s_tr.shape[1]))),
                                                       axis=0),
                                        s_train,
                                        z_val_curr, x_val_curr, s_val)
            else:
                s_train = np.concatenate((1 - self.s1_s, 1 - self.s1_us), axis=0)
                s_val = np.concatenate(
                    (np.zeros((rct_data_val['x'].shape[0], 1)), np.ones((obs_data_val['x'].shape[0], 1))),
                    axis=0)
                passed = test_selection(z_curr, x_curr,
                                        np.concatenate((np.zeros((x_s_tr.shape[0], xo_us_tr.shape[1])), xo_us_tr),
                                                       axis=0),
                                        s_train,
                                        z_val_curr, x_val_curr, s_val)
            if passed:
                final_z_index.append(i)
                if len(final_z_index) >= output_num:
                    break
        if len(final_z_index) < 1:
            raise ValueError("No valid shadow variable exists, please check your data or adjust the threshold.")
        final_zs = x_s_tr[:, final_z_index]
        final_xs = x_s_tr[:, np.setdiff1d(np.arange(x_s_tr.shape[1]), final_z_index)]
        final_zus = x_us_tr[:, final_z_index]
        final_xus = x_us_tr[:, np.setdiff1d(np.arange(x_us_tr.shape[1]), final_z_index)]
        if len(final_zs.shape) == 1:
            final_zs = final_zs.reshape(-1, 1)
        if len(final_zus.shape) == 1:
            final_zus = final_zus.reshape(-1, 1)
        if len(final_xs.shape) == 1:
            final_xs = final_xs.reshape(-1, 1)
        if len(final_xus.shape) == 1:
            final_xus = final_xus.reshape(-1, 1)
        # OR(X,Xr)
        if Config.setting == 1:
            or_model = fit_odds(final_zs, final_xs, xr_s_tr, final_zus, final_xus,
                                xs_val_tr[:, np.setdiff1d(np.arange(xs_val_tr.shape[1]), final_z_index)],
                                xs_val_tr[:, final_z_index],
                                xr_val_tr)
            pred_or_tilde = predict_odds(or_model, final_xs, xr_s_tr)
        else:
            or_model = fit_odds(final_zus, final_xus, xo_us_tr, final_zs, final_xs,
                                xus_val_tr[:, np.setdiff1d(np.arange(xus_val_tr.shape[1]), final_z_index)],
                                xus_val_tr[:, final_z_index],
                                xo_val_tr)
            pred_or_tilde_us = predict_odds(or_model, final_xus, xo_us_tr)
        # f(S | X)
        s_model = train_logit(np.concatenate((final_xs, final_xus), axis=0),
                              np.concatenate((np.ones((final_xs.shape[0], 1)), np.zeros((final_xus.shape[0], 1))),
                                             axis=0),
                              x_val_tr[:, np.setdiff1d(np.arange(x_val_tr.shape[1]), final_z_index)],
                              np.concatenate((np.ones((xs_val.shape[0], 1)), np.zeros((xus_val.shape[0], 1))),
                                             axis=0),
                              final_xus.shape[0] / final_xs.shape[0])
        # 1 / f(S | X Xr)
        if Config.setting == 1:
            s1_pred = get_weight(s_model, final_xs)
            s0_pred = 1 - s1_pred
            propensity = 1 / (s0_pred * pred_or_tilde / s1_pred + 1)
            w_s = (1 - propensity) / propensity
        else:
            s1_pred = get_weight(s_model, final_xus)
            s0_pred = 1 - s1_pred
            propensity = 1 / (s0_pred * pred_or_tilde_us / s1_pred + 1)
            w_us = propensity / (1 - propensity + 1e-5)
        # f(Xr | X, Z, S=1)
        if Config.setting == 1:
            biased_xr_model = train_ols(x_s_tr, xr_s_tr)
            # f(Xr | X, Z, S=0)
            pred_xr_s = biased_xr_model.predict(x_s_tr)
            if len(pred_xr_s.shape) == 1:
                pred_xr_s = pred_xr_s.reshape(-1, 1)
            pred_or_tilde0 = predict_odds(or_model, final_xs, np.zeros(xr_s_tr.shape))
            pred_or_mean = predict_odds(or_model, final_xs, pred_xr_s)
            pred_xr_us = pred_xr_s * pred_or_tilde / (
                    pred_or_tilde0 * pred_or_mean)
            xr_model = train_ols(x_s_tr, pred_xr_us)
            pred_dr_xr_s = xr_model.predict(x_s_tr)
            pred_dr_xr_us = xr_model.predict(x_us_tr)
            if len(pred_dr_xr_s.shape) == 1:
                pred_dr_xr_s = pred_dr_xr_s.reshape(-1, 1)
            if len(pred_dr_xr_us.shape) == 1:
                pred_dr_xr_us = pred_dr_xr_us.reshape(-1, 1)
            pred_dr_xr = np.concatenate((w_s * (xr_s_tr - pred_dr_xr_s) + pred_dr_xr_s, pred_dr_xr_us), axis=0)
            self._xr_model = train_ols(np.concatenate((x_s_tr, x_us_tr), axis=0), pred_dr_xr)
        else:
            biased_xo_model = train_ols(x_us_tr, xo_us_tr)
            # f(Xr | X, Z, S=0)
            pred_xo_us = biased_xo_model.predict(x_us_tr)
            if len(pred_xo_us.shape) == 1:
                pred_xo_us = pred_xo_us.reshape(-1, 1)
            pred_or_tilde0 = predict_odds(or_model, final_xus, np.zeros(xo_us_tr.shape))
            pred_or_mean = predict_odds(or_model, final_xus, pred_xo_us)
            pred_xo_s = pred_xo_us * pred_or_tilde_us / (
                    pred_or_tilde0 * pred_or_mean)
            xo_model = train_ols(x_us_tr, pred_xo_s)
            pred_dr_xo_s = xo_model.predict(x_s_tr)
            pred_dr_xo_us = xo_model.predict(x_us_tr)
            if len(pred_dr_xo_s.shape) == 1:
                pred_dr_xo_s = pred_dr_xo_s.reshape(-1, 1)
            if len(pred_dr_xo_us.shape) == 1:
                pred_dr_xo_us = pred_dr_xo_us.reshape(-1, 1)
            pred_dr_xo = np.concatenate((pred_dr_xo_s, w_us * (xo_us_tr - pred_dr_xo_us) + pred_dr_xo_us), axis=0)
            self._xo_model = train_ols(np.concatenate((x_s_tr, x_us_tr), axis=0), pred_dr_xo)
        # CATE estimation
        t_index = np.where(self.t_s == 1)[0]
        c_index = np.where(self.t_s == 0)[0]
        if Config.setting == 1:
            x_regression = np.concatenate((x_s_tr[t_index], xr_s_tr[t_index]), axis=1)
        else:
            xo_s_tr = self._xo_model.predict(x_s_tr)
            if len(xo_s_tr.shape) == 1:
                xo_s_tr = xo_s_tr.reshape(-1, 1)
            x_regression = np.concatenate((x_s_tr[t_index], xo_s_tr[t_index]), axis=1)
        self._biased_model_t = train_ols(x_regression, self.y_s[t_index])
        if Config.setting == 1:
            x_regression = np.concatenate((x_s_tr[c_index], xr_s_tr[c_index]), axis=1)
        else:
            pred_or_tilde = predict_odds(or_model, final_xs, xo_s_tr)
            s1_pred = get_weight(s_model, final_xs)
            s0_pred = 1 - s1_pred
            propensity = 1 / (s0_pred * pred_or_tilde / s1_pred + 1)
            w_s = (1 - propensity) / (propensity + 1e-5)
            x_regression = np.concatenate((x_s_tr[c_index], xo_s_tr[c_index]), axis=1)
        self._biased_model_c = train_ols(x_regression, self.y_s[c_index])

        x = self.scaler.transform(obs_data_test['x'])
        if Config.setting == 1:
            xr = self._xr_model.predict(x)
            if len(xr.shape) == 1:
                xr = xr.reshape(-1, 1)
            unselected_y_pre = self._biased_model_t.predict(np.concatenate((x, xr), axis=1)).reshape(-1, 1)
            unselected_y_pre_c = self._biased_model_c.predict(np.concatenate((x, xr), axis=1)).reshape(-1, 1)
        else:
            xo = self.scaler_xo.transform(obs_data_test['xo'])
            unselected_y_pre = self._biased_model_t.predict(np.concatenate((x, xo), axis=1)).reshape(-1, 1)
            unselected_y_pre_c = self._biased_model_c.predict(np.concatenate((x, xo), axis=1)).reshape(-1, 1)
        biased_unselected_ite = unselected_y_pre - unselected_y_pre_c

        if Config.setting == 1:
            y_pre = self._biased_model_t.predict(np.concatenate((x_s_tr, xr_s_tr), axis=1)).reshape(-1, 1)
            y_pre_c = self._biased_model_c.predict(np.concatenate((x_s_tr, xr_s_tr), axis=1)).reshape(-1, 1)
        else:
            y_pre = self._biased_model_t.predict(np.concatenate((x_s_tr, xo_s_tr), axis=1)).reshape(-1, 1)
            y_pre_c = self._biased_model_c.predict(np.concatenate((x_s_tr, xo_s_tr), axis=1)).reshape(-1, 1)
        mae = np.abs((np.sum(2 * w_s[t_index] * (self.y_s[t_index] - y_pre[t_index]))
                      - np.sum(2 * w_s[c_index] * (self.y_s[c_index] - y_pre_c[c_index]))) /
                     obs_data_test['gt'].shape[0]
                     + np.mean(biased_unselected_ite) - np.mean(obs_data_test['gt']))
        return mae
