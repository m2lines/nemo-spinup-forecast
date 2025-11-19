# Adapted from code by Maud Tissot (Spinup-NEMO)
# Original source: https://github.com/maudtst/Spinup-NEMO
# Licensed under the MIT License
#
# Modifications in this version by ICCS, 2025
import random
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    RationalQuadratic,
    WhiteKernel,
)  # , Matérn


# NOT UP-TO-DATE WITH PREDICTION CLASS


class optimization:
    def __init__(
        self,
        ids,
        ratio,
        ncomp,
        var,
        steps=1,
        min_test=50,
        min_train=50,
        kernels=None,
        trunc=None,
    ):
        random.shuffle(ids)
        i = int(len(ids) * ratio)
        self.ids_eval = ids[:i]
        self.ids_test = ids[i:]
        self.simu_eval = [TS(id_, var) for id_ in ids[:i]]
        self.simu_test = [TS(id_, var) for id_ in ids[i:]]
        self.ncomp = ncomp
        self.var = var
        self.min_train = min_train
        self.min_test = min_test
        self.steps = steps
        self.trunc = trunc
        self.kernels = [RBF(), RationalQuadratic()] if kernels is None else kernels
        self.seed = random.randint(1, 200)

    ###___get all technique___###

    def getAllGP(self, n=4, r=RBF()):
        kernels = self.kernelCombination(r, n)
        listGP = []
        for kernel in np.array(kernels).reshape(-1):
            kernel = kernel + WhiteKernel()
            listGP.append(
                GaussianProcessRegressor(
                    kernel=kernel, normalize_y=False, n_restarts_optimizer=0
                )
            )
        self.techniques = listGP

    def kernelCombination(self, r=RBF(), n=4):
        k = self.kernels
        if n == 1:
            return r
        else:
            return (
                self.kernelCombination(r + k[0], n=n - 1),
                self.kernelCombination(r * k[0], n=n - 1),
                self.kernelCombination(r + k[1], n=n - 1),
                self.kernelCombination(r * k[1], n=n - 1),
            )

    ###___evaluate current technique___###

    def evaluateCurrentProcess(self):
        random.seed(self.seed)
        results_eval = []
        print("evaluation : ")
        for simu in self.simu_eval:
            print(f"Processing simulation {simu.id}")
            if self.min_train < len(simu) - self.min_test:
                train_lens = np.arange(
                    self.min_train, len(simu) - self.min_test, self.steps
                )
                results_eval.append(
                    simu.evaluateModel(self.ncomp, train_lens, f"{self.var}-1", jobs=15)
                )
                if self.trunc is None:
                    results_eval[-1] = np.sum(result[-1])
                else:
                    results_eval[-1] = np.sum(
                        [min(val, self.trunc) for val in results_eval[-1]]
                    )
        results_test = []
        print("\ntest : ")
        for simu in self.simu_test:
            print(f"Processing simulation {simu.id}")
            if self.min_train < len(simu) - self.min_test:
                train_lens = np.arange(
                    self.min_train, len(simu) - self.min_test, self.steps
                )
                results_test.append(
                    simu.evaluateModel(self.ncomp, train_lens, f"{self.var}-1", jobs=15)
                )
                if self.trunc is None:
                    results_test[-1] = np.sum(results_test[-1])
                else:
                    results_test[-1] = np.sum(
                        [min(val, self.trunc) for val in results_test[-1]]
                    )
        self.current_score_eval = np.sum(results_eval)
        self.current_score_test = np.sum(results_test)

    ###___evaluate technique___###

    # this methode should be changed to rmse with raw simulation
    def evaluateProcess(self, simu, train_lens, process):
        currenttechnique = simu.technique
        simu.technique = process
        print("-", end="")
        test = simu.evaluateModel(self.ncomp, train_lens, f"{self.var}-1", jobs=15)
        simu.technique = currenttechnique
        if self.trunc is None:
            return np.sum(test)
        else:
            return np.sum([min(val, self.trunc) for val in test])

    def evaluateKernels(self):
        random.seed(self.seed)
        results = []
        for simu in self.simu_eval:
            print(f"Processing simulation {simu.id} ", end="")
            if self.min_train < len(simu) - self.min_test:
                train_lens = np.arange(
                    self.min_train, len(simu) - self.min_test, self.steps
                )
                results.append(
                    [
                        self.evaluateProcess(simu, train_lens, process)
                        for process in self.techniques
                    ]
                )
                print("", end="\n")
        results = [
            (process, score)
            for process, score in zip(self.techniques, np.sum(results, axis=0))
        ]
        self.scores_eval = sorted(results, key=lambda item: item[1], reverse=True)

    ###___Select on test___###

    def testKernels(self):
        random.seed(self.seed)
        techniques_test = [
            process
            for process, score in self.scores_eval
            if score > self.current_score_eval
        ]
        results = []
        for simu in self.simu_test:
            print(f"Processing simulation {simu.id}", end="")
            if self.min_train < len(simu) - self.min_test:
                train_lens = np.arange(
                    self.min_train, len(simu) - self.min_test, self.steps
                )
                results.append(
                    [
                        self.evaluateProcess(simu, train_lens, process)
                        for process in techniques_test
                    ]
                )
                print("", end="\n")
        results = [
            (process, score)
            for process, score in zip(techniques_test, np.sum(results, axis=0))
        ]
        self.scores_test = sorted(results, key=lambda item: item[1], reverse=True)
