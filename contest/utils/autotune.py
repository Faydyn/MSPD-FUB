import pandas as pd
import random
import signal
from sklearn.metrics import mean_squared_error

# contestants are only allowed to modify the following function!
# input: a) number of coordinates(N: 10, 15, 20, ... 50),
#        b) objective mode (objectiveN: 1,2,3)
#        c) input coordinates dataframe (inputDf),
# output: the list that has the index of sources

######## YOU CAN ONLY MODIFY FROM HERE UNTIL THE NEXT ##.. COMMENTS ###########

# Imports
import sys
import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import numpy as np
from sklearn import cluster
import tensorflow as tf
tf.get_logger().setLevel('FATAL')

from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.initializers import glorot_uniform

from spektral.data import Graph, Dataset, BatchLoader
from spektral.layers import GCNConv, GlobalMaxPool, GlobalAvgPool, GraphMasking


# Magic Constants
N_MAX: int = 50
MAX_SOURCES: int = 3

EPSILON: float = 0.0000001

BATCH_SIZE: int = 128
NODE_FEATURES: int = 13
Y: np.ndarray = np.array([0.0], dtype=np.float32)

LOADER_SETTINGS = {
    'batch_size': BATCH_SIZE,
    'shuffle': False,
    'mask': True
}

# Hyper Params
# OPTIMIZED
# MinMax Treshold Filter (obj2)
MIN_DIST_TWO_SRC: float = 180.0 / 2002.0  # normed for a max dist of 1
MIN_DIST_THREE_SRC: float = 185.0 / 2002.0
MAX_DIST_MULTI_SRC: float = 1300.0 / 2002.0
# 2nd Filter (Skew approx, obj1/2)
#MIN_SKEW_INCLUDE_RATIO = np.array([1.5, 1.15, 100.0])
#INC_SKEW = np.array([1 / 25, 1 / 400, 0.0])
MIN_SKEW_INCLUDE_RATIO = {
        10: np.array([1.75, 1.29, 2.2]),
        15: np.array([2.05, 1.1875, 2.0]),
        25: np.array([2.4, 1.35, 2.0]),
        30: np.array([2.3, 1.45, 2.2]),
        40: np.array([2.3, 1.6, 2.1]),
        45: np.array([2.3, 1.55, 2.2]),
        50: np.array([2.3, 1.5, 2.1]),
}

# Weight Intensity (from Skew Weights)
#SCALE_FAC_START = np.array([0.73, 0.79, 0.975])
#INC_PER_N = np.array([1 / 200, 1 / 700, 1 / 1000])
SCALE_FAC_START = {
        10: np.array([0.835, 0.77, 0.995]),
        15: np.array([0.82, 0.83, 0.98]),
        25: np.array([0.86, 0.75, 0.995]),
        30: np.array([0.91, 0.69, 0.985]),
        40: np.array([0.94, 0.71, 0.98]),
        45: np.array([0.93, 0.72, 0.99]),
        50: np.array([0.945, 0.7, 0.985]),
}

CLUSTER_AMOUNT = {
        10: np.array([3, 0, 4]),
        15: np.array([5, 0, 6]),
        25: np.array([8, 0, 7]),
        30: np.array([6, 0, 9]),
        40: np.array([8, 0, 8]),
        45: np.array([6, 0, 7]),
        50: np.array([6, 0, 6]),
}


# UNOPTIMIZED
# Region Threshold (obj1/3)
CENTER_MID: float = 280.0 / 1001.0
BETWEEN_LOW: float = 360.0 / 1001.0
BETWEEN_MID: float = 495.0 / 1001.0
BETWEEN_HIGH: float = 640.0 / 1001.0
MID_CORNER: float = 780.0 / 1001.0

# Region Filter Thresholds (obj1/3)
dist_threshold_1 = np.array([450.0, 500.0, 500.0, 550.0, 400.0, 600.0]) / 1001.0
x_center_1 = np.array([450.0, 550.0, 350.0, 325.0, 250.0, 275.0]) / 1001.0
y_center_1 = np.array([375.0, 350.0, 280.0, 180.0, 160.0, 300.0]) / 1001.0
include_obj1 = np.array([
    [False, True, True, True, True, True],
    [True, True, True, True, True, False],
    [True, True, True, True, True, False]])

dist_threshold_3 = np.array([550.0, 600.0, 600.0, 600.0, 500.0, 700.0]) / 1001.0
x_center_3 = np.array([450.0, 550.0, 350.0, 325.0, 250.0, 325.0]) / 1001.0
y_center_3 = np.array([350.0, 300.0, 280.0, 180.0, 160.0, 300.0]) / 1001.0
include_obj3 = np.array([
    [False, True, True, True, True, True],
    [True, True, True, True, True, True],
    [True, True, True, True, True, False]])


# Graph Neural Net
class MSPDContestModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(kwargs)
        self.mask = GraphMasking()
        self.conv = GCNConv(32, mode='batch', kernel_initializer=glorot_uniform())
        self.flatten_avg = GlobalAvgPool()
        self.flatten_max = GlobalMaxPool()
        self.flat = Flatten()
        self.dense = Dense(512, 'relu')
        self.last = Dense(1, activation='linear')

    def call(self, inputs, **kwargs):
        x, a = inputs
        x = self.mask(x)
        x = self.conv([x, a])
        out_avg_pool = self.flatten_avg(x)
        out_max_pool = self.flatten_max(x)
        out = self.flat(tf.concat([out_avg_pool, out_max_pool], axis=1))
        out = self.dense(out)
        out = self.last(out)
        return out


# Graph Datasets
class MSPDDummy(Dataset):
    def __init__(self, **kwargs):
        X: np.ndarray = np.zeros((N_MAX, NODE_FEATURES), dtype=np.float32)
        A: np.ndarray = np.zeros((N_MAX, N_MAX), dtype=np.float32)
        self.G: Graph = Graph(x=X, a=A, y=Y)
        super().__init__(**kwargs)

    def read(self) -> [Graph]:
        return [self.G] * BATCH_SIZE


class MSPDContestNet(Dataset):
    def __init__(self, **kwargs):
        self.WEIGHTS = []
        super().__init__(**kwargs)

    def filter_df(self) -> pd.DataFrame:
        filtered_src_combs = self.prep_df()
        df: np.ndarray = np.zeros((len(filtered_src_combs), self.N),
                                  dtype=np.float32)
        for i, src_ids in enumerate(filtered_src_combs):
            for j in src_ids:
                df[i][j] = 1

        return pd.DataFrame(df)

    def scale_fac(self):
        return 1 - SCALE_FAC_START[self.N][self.obj - 1]


    def calc_weights(self, src_ids):
        mins = np.zeros(len(src_ids))
        maxs = np.zeros(len(src_ids))
        for src_id_enum, tup in enumerate(src_ids):
            mins[src_id_enum] = np.min([self.adj_mat[0][idx] for idx in tup])
            maxs[src_id_enum] = np.max(
                    np.min([self.adj_mat[idx][1:] + self.adj_mat[0][idx] for idx in tup], axis=0)
            )

        min_skews = maxs - mins
        best_min_skew = min_skews[np.argmin(min_skews)]
        threshold = best_min_skew * (MIN_SKEW_INCLUDE_RATIO[self.N][self.obj - 1])
        filtered_src_enum = np.where(min_skews < threshold)[0]
        self.WEIGHTS = np.array([
            min_skews[i] for i in filtered_src_enum
        ]).reshape((len(filtered_src_enum), 1))

        # Norm weights
        self.WEIGHTS -= np.min(self.WEIGHTS)
        if np.max(self.WEIGHTS) > 0:
            self.WEIGHTS /= np.max(self.WEIGHTS)  # [0,1]
            self.WEIGHTS *= self.scale_fac()      # [0,x]
            self.WEIGHTS = 1 - self.WEIGHTS       # [1-x,1] as weight
        else:
            self.WEIGHTS += 1

        return [src_ids[src_id_enum] for src_id_enum in filtered_src_enum]


    def prep_df(self):
        if self.obj == 2:
            src_ids = [(i,) for i in range(1, self.N)]
            src_ids += [(i, j) for i in range(1, self.N)
                        for j in range(i + 1, self.N)
                        if MAX_DIST_MULTI_SRC > self.adj_mat[i][j] > MIN_DIST_TWO_SRC]

            if self.N < 25:  # combinatorical reasons
                src_ids += [(i, j, k) for i in range(1, self.N)
                            for j in range(i + 1, self.N) for k in
                            range(j + 1, self.N)
                            if MAX_DIST_MULTI_SRC > self.adj_mat[i][j] > MIN_DIST_THREE_SRC
                            and MAX_DIST_MULTI_SRC > self.adj_mat[i][k] > MIN_DIST_THREE_SRC
                            and MAX_DIST_MULTI_SRC > self.adj_mat[j][k] > MIN_DIST_THREE_SRC]

            return self.calc_weights(src_ids)

        if self.obj == 1:
            d = dist_threshold_1[self.area_num]
            x = x_center_1[self.area_num]
            y = y_center_1[self.area_num]
            include_1 = include_obj1[0][self.area_num]
            include_2 = include_obj1[1][self.area_num]
            include_3 = include_obj1[2][self.area_num]
        else:  # self.obj == 3
            d = dist_threshold_3[self.area_num]
            x = x_center_3[self.area_num]
            y = y_center_3[self.area_num]
            include_1 = include_obj3[0][self.area_num]
            include_2 = include_obj3[1][self.area_num]
            include_3 = include_obj3[2][self.area_num]

        abs_diffs = np.abs(self.xs - x) + np.abs( self.ys - y)  # from x_center,y_center

        size = 0
        while size < 4:
            new_vertices = np.where(abs_diffs[1:] < d)[0] + 1
            new_xs = [self.xs[i] for i in new_vertices]
            new_ys = [self.ys[i] for i in new_vertices]

            size = len(new_xs)
            d += 100

        clusterr = self.make_clusters(new_xs, new_ys)

        new_dict = {i: idx for i, idx in enumerate(new_vertices)}
        filtered_vertices = []

        if include_1:
            for i, _ in enumerate(new_vertices):
                filtered_vertices.append((new_dict[i],))

        coms = np.abs(new_xs - np.mean(new_xs)) + np.abs(
            new_ys - np.mean(new_ys))
        idx_com_src = new_dict[np.argmin(coms)] if len(coms) > 1 else (
                np.argmin(self.node_feats[1:, 2]) + 1)

        if (idx_com_src,) not in filtered_vertices:  # center of mass
            filtered_vertices.append((idx_com_src,))

        if include_2:
            for i, _ in enumerate(new_vertices):
                for j, _ in enumerate(new_vertices[i + 1:], start=i + 1):
                    if clusterr[i] != clusterr[j]:
                        filtered_vertices.append((new_dict[i], new_dict[j]))

        if include_3:
            for i, _ in enumerate(new_vertices):
                for j, _ in enumerate(new_vertices[i + 1:], start=i + 1):
                    for k, _ in enumerate(new_vertices[j + 1:], start=j + 1):
                        if clusterr[i] != clusterr[j] and clusterr[i] != \
                                clusterr[k] and clusterr[j] != clusterr[k]:
                            filtered_vertices.append(
                                (new_dict[i], new_dict[j], new_dict[k]))


        return self.calc_weights(filtered_vertices)

    def read(self) -> [Graph]:
        def row_to_graph(one_hot: np.ndarray) -> Graph:
            A = np.copy(self.adj_mat)
            X = np.copy(self.node_feats)

            for j in range(self.N):
                if not one_hot[j]:  # Remove root -> non-src
                    A[0][j] = 0

            X[:, 8] = np.mean(A, axis=0)
            X[:, 9] = np.max(A, axis=0)
            X[:, 10] = np.mean(A, axis=1)
            X[:, 11] = np.max(A, axis=1)
            X[:, 12] = one_hot
            return Graph(x=X, y=Y, a=A)

        return [row_to_graph(np.array(row)) for _, row in
                self.filter_df().iterrows()]

    def make_clusters(self, xs: np.ndarray, ys: np.ndarray) -> [int]:
        k = min(CLUSTER_AMOUNT[self.N][self.obj - 1], len(xs))
        model = cluster.AgglomerativeClustering(n_clusters=k, linkage='single')
        return model.fit([[x, y] for x, y in zip(xs, ys)]).labels_

class ContestModel:
    def __init__(self):
        # Default init params
        self.N: int = -1
        self.net_idx: int = -1
        self.obj: int = -1
        self.models = self.init_models()

        self.xs = None
        self.ys = None
        self.adj_mat = None
        self.node_feats = None
        self.area_num = -1

    def load_result(self, *args) -> [int]:
        self.check_reload(*args)

        # create all subgraphs from remaining source combinations
        mspd_net: Dataset = MSPDContestNet(**self.__dict__)
        loader: BatchLoader = BatchLoader(mspd_net, **LOADER_SETTINGS)

        # make and return prediction (array of W+S for each subgraph)
        pred: np.ndarray = self.models[self.obj].predict(
            loader.load(), steps=loader.steps_per_epoch, verbose=0)

        pred = pred * mspd_net.WEIGHTS
        max_idx: int = np.argmax(pred)
        best_one_hot: np.ndarray = mspd_net[max_idx].x[:, 12]
        src_comb: np.ndarray = np.where(best_one_hot == 1)[0]

        return list(src_comb)[:MAX_SOURCES]  # safety slice (in case of bugs)

    def check_reload(self, N: int, obj: int, input_df: pd.DataFrame) -> None:
        #if self.N == 10 and obj == 1:
        #    self.obj = 2  # skew more important
        #else:
        self.obj = obj

        input_arr: np.ndarray = np.array(input_df.iloc[0])
        net_idx: int = input_arr[0]

        if self.net_idx != net_idx:  # only recalc every after all 3 objs
            if net_idx == 0 or net_idx == 300:  # means new (larger) N than before
                self.N = N
                self.adj_mat = np.zeros((self.N, self.N))
                self.node_feats = np.zeros((self.N, NODE_FEATURES),
                                           dtype=np.float32)

                # some fine-tuning
                if self.N >= 40:  # never do 3-src
                    include_obj1[2] = False
                    include_obj3[2] = False

            self.net_idx = net_idx

            self.reload_data(input_arr[1:])
            self.norm_data()
            self.calc_features()

    def reload_data(self, data: np.ndarray) -> None:
        self.xs = data[::2]
        self.ys = data[1::2]

        # calc adj_mat with unnormed data (adj_mat is then normed indiv also)
        for i in range(self.N):
            for j in range(i + 1, self.N):  # use symmetry of matrix
                d = abs(self.xs[i] - self.xs[j]) + abs(self.ys[i] - self.ys[j])
                self.adj_mat[i][j] = d
                self.adj_mat[j][i] = d

    def norm_data(self) -> None:
        self.adj_mat /= 2002

        self.xs = (self.xs - 1) / 1001 - 0.5
        self.ys = (self.ys - 1) / 1001 - 0.5

        if self.xs[0] > 0:
            self.xs = -self.xs
        if self.ys[0] > 0:
            self.ys = -self.ys
        if self.ys[0] > self.xs[0]:
            self.xs, self.ys = self.ys, self.xs

        self.xs += 0.5
        self.ys += 0.5

    def calc_features(self) -> None:
        avg_x = np.mean(self.xs)
        avg_y = np.mean(self.ys)

        diff_com_xs = self.xs - avg_x
        diff_com_ys = self.ys - avg_y
        com_dist = np.abs(diff_com_xs) + np.abs(diff_com_ys)
        root_dist = np.abs(self.xs - self.xs[0]) + np.abs(self.ys - self.ys[0])

        denom = diff_com_xs
        for zero_i in np.where(denom == 0)[0]:
            denom[zero_i] = EPSILON
        ratio = diff_com_ys / denom

        abs_ratio = np.abs(ratio)
        pos_ratio = np.array(ratio >= 0, dtype=np.float32)
        gt1_ratio = np.array(abs_ratio >= 1, dtype=np.float32)

        lt1_ratio = 1 - gt1_ratio
        abs_ratio = lt1_ratio * abs_ratio + \
                    gt1_ratio * (1 / (abs_ratio + EPSILON))
        abs_ratio -= 0.5
        vert_half = np.array(diff_com_xs > 0, dtype=np.float32)

        self.node_feats[:, 0] = self.xs
        self.node_feats[:, 1] = self.ys
        self.node_feats[:, 2] = com_dist / 2
        self.node_feats[:, 3] = root_dist / 2
        self.node_feats[:, 4] = abs_ratio
        self.node_feats[:, 5] = pos_ratio
        self.node_feats[:, 6] = gt1_ratio
        self.node_feats[:, 7] = vert_half

        root_dist_from_center = 1 - self.xs[0] - self.ys[0]

        if root_dist_from_center < CENTER_MID:
            self.area_num = 0
        elif root_dist_from_center < BETWEEN_LOW:
            self.area_num = 1
        elif root_dist_from_center < BETWEEN_MID:
            self.area_num = 2
        elif root_dist_from_center < BETWEEN_HIGH:
            self.area_num = 3
        elif root_dist_from_center < MID_CORNER:
            self.area_num = 4
        else:
            self.area_num = 5


    def init_models(self):
        # Load model (with dummy structure)
        shaper: BatchLoader = BatchLoader(MSPDDummy(), **LOADER_SETTINGS)
        batch = None
        for b in shaper:
            batch = b  # to get matching shapes for building the model
            break

        models = {i: MSPDContestModel(training=False) for i in range(1, 4)}
        for i, model in models.items():
            model.compile(optimizer='adam', loss='mse')
            model([batch[0][0], batch[0][1]])
            model.load_weights(f'weights/model_{i}.h5')
        return models


contest_model = ContestModel()


def Inference(N, objectiveN, inputDf):
    return contest_model.load_result(N, objectiveN, inputDf)


######### DO NOT MODIFY FROM HERE ONWARD ############

# runtime exceed handler
def Handler(signum, frame):
    print("Exceed 10s Runtime Limit")
    raise Exception("Runtime Limit Exceeds")


# input: list of sources
# output: corresponding index from sourceDataFrame
def GetResultIdx(resultIdxList, sourceDf):
    N = len(sourceDf.columns) - 1
    # error handling
    if len(resultIdxList) >= 4:
        print("Error: found #sources >=4. #sources are limited <= 3")
        exit(1)
    if len(resultIdxList) != 0 and (
            min(resultIdxList) <= 0 or max(resultIdxList) >= N):
        print(
            "Error: index exceeded expected ranges. Please double-check your indices.")
        print("       Expected index range is 1~N-1, but range is min=",
              min(resultIdxList), "max=", max(resultIdxList))
        exit(1)

    sourceArr = [0] * N
    for val in resultIdxList:
        sourceArr[val] = 1

    mask = True
    for i, val in enumerate(sourceArr):
        mask = (mask & (sourceDf['%d' % (i)] == val))

    # return sourceIdx value from the source dataframe
    return sourceDf.loc[mask]["sourceIdx"].values[0]


# signal setup for maximum runtime limit



def eval_contest():
    listK = [1, 1, 1, 1, 2, 2, 2]
    MSEs = []

    # for various N
    for n in [10, 15,25, 30, 40, 45, 50]:


        #if n == 10:
        #  dataObjDf = pd.read_csv("testcases/TRAIN_data_obj_stt_%d.csv.gz" %(n), compression="gzip").head(130*600)
        #  inputDf = pd.read_csv("testcases/TRAIN_input_stt_%d.csv.gz" % (n), compression="gzip").head(600)
        #
        #elif n == 15:
        if n <= 15:
          dataObjDf = pd.read_csv("testcases/data_obj_stt_%d.csv.gz" %(n), compression="gzip")
          inputDf = pd.read_csv("testcases/input_stt_%d.csv.gz" % (n), compression="gzip")
        else:
          dataObjDf = pd.read_csv("testcases/data_obj_stt_%d.csv" %(n))
          inputDf = pd.read_csv("testcases/input_stt_%d.csv.gz" % (n), compression="gzip")

        sourceDf = pd.read_csv("testcases/sources_stt_%d.csv.gz" % (n), compression="gzip")

        # save predicted result
        # first index: N
        # second index: netIdx
        predictArr = [[], [], []]

        # there will be hidden testcases (200 more nets)
        offset = 0 if n == 10 else 0
        for netIdx in range(0, 300):
            netInputDf = inputDf.loc[inputDf['netIdx'] == (netIdx + offset)]
            netDataObjDf = dataObjDf.loc[dataObjDf['netIdx'] == (netIdx + offset)]
            # for each objective (1,2,3)
            for objectiveN in range(1, 4):
                # for runtime limit - 10 seconds

                # call the Inference function
                isFailed = False
                #try:
                if objectiveN != 2:
                    resultIdxList = Inference(n, objectiveN, netInputDf)
                else:
                    resultIdxList = [1]
                #except Exception as e:
                #    print( "Warning: Runtime Limit Exceeded. Penalty will be applied")
                #    isFailed = True

                if isFailed == False:
                    # retrieve sourceIdx using sourceDataFrame
                    resultIdx = GetResultIdx(resultIdxList, sourceDf)

                    predictedObj = \
                        netDataObjDf.loc[netDataObjDf['sourceIdx'] == resultIdx][
                            'obj%d' % (objectiveN)].values[0]
                    predictArr[objectiveN - 1].append(predictedObj)
                else:
                    # append "-1" flag variable - runtime exceeded
                    predictArr[objectiveN - 1].append(-1)

            # extract the best cost value
            bestCostValue = [netDataObjDf['obj%d' % (i)].min() for i in range(1, 4)]

            # take the "ratio" instead of raw value
            # predictedValue /= bestCostValue
            for i, val in enumerate(bestCostValue):
                # normal predicted case
                if predictArr[i][-1] != -1:
                    predictArr[i][-1] /= val
                # runtime exceeded case
                else:
                    predictArr[i][-1] = 1.5

        # best cost is normalized as "1"
        bestCostArr = [[1] * len(predictArr[0]) for _ in range(3)]

        MSE = [mean_squared_error(bestCostArr[i], predictArr[i]) for i in range(3)]
        print("    for n =", n, ", MSE(mean squared error) of three objectives =", MSE)
        MSEs.append(MSE)

    evalMetric = 0
    for k, MSE in zip(listK, MSEs):
        evalMetric += k * sum(MSE)

    print("    EvalMetric: ", evalMetric)
    return evalMetric

best_dict = {
    'metric' : 1000,
    'c' : None,
}

def auto_tune(c):
    global CLUSTER_AMOUNT
    CLUSTER_AMOUNT = c

    contest_model = ContestModel()
    new_metric = eval_contest()

    if new_metric < best_dict['metric']:
        print('new best!')
        best_dict['metric'] = new_metric
        best_dict['c'] = c
        print(list(best_dict.items()))

# (45,1)

C =[
        {
        10: np.array([4, 0, 4]),
        15: np.array([5, 0, 5]),
        25: np.array([6, 0, 6]),
        30: np.array([6, 0, 6]),
        40: np.array([6, 0, 6]),
        45: np.array([6, 0, 6]),
        50: np.array([6, 0, 6]),
        },
        {
        10: np.array([3, 0, 4]),
        15: np.array([5, 0, 6]),
        25: np.array([8, 0, 7]),
        30: np.array([6, 0, 9]),
        40: np.array([8, 0, 8]),
        45: np.array([6, 0, 7]),
        50: np.array([6, 0, 6]),
        }
    ]
for c in C:
        auto_tune(c)

print()
for k,v in best_dict.items():
    print(k," : ",  v)
sys.exit(0)
