def DfsPredict(row):
    if row["MCV"] <= 80:
        if row["Sex"] > 0:
            if DF1(row) > 0.49:
                return 1
            else:
                return 0
        else:
            if DF2(row) > 0.74:
                return 1
            else:
                return 0
    else:
        if row["RDW-CV"] / row["RBC"] <= 3.54:
            if row["Sex"] > 0:
                if DF3(row) > -0.01:
                    return 1
                else:
                    return 0
            else:
                if DF4(row) > 0.28:
                    return 1
                else:
                    return 0
        else:
            return 0


def DfsPredictMale(row):
    if row["MCV"] <= 80:
        if DF2(row) > 0.74:
            return 1
        else:
            return 0
    else:
        if row["RDW-CV"] / row["RBC"] <= 3.54:
            if DF4(row) > 0.28:
                return 1
            else:
                return 0
        else:
            return 0


def DfsPredictFemale(row):
    if row["MCV"] <= 80:
        if DF1(row) > 0.49:
            return 1
        else:
            return 0
    else:
        if row["RDW-CV"] / row["RBC"] <= 3.54:
            if DF3(row) > -0.01:
                return 1
            else:
                return 0
        else:
            return 0


def DF1(row):
    return (
        0.015 * row["RDW-CV"] / row["RBC"] - 0.096 * row["RDW-SD"] / row["RBC"] + 1.29
    )


def DF2(row):
    return -0.025 * row["RDW-SD"] / row["RBC"] - 0.035 * row["MCV"] / row["RBC"] + 1.415


def DF3(row):
    return -0.38 * row["MCH"] - 0.02 * row["MCHC"] + 17.37


def DF4(row):
    return 0.007 * row["MCV"] - 0.113 * row["MCH"] + 2.829


class DFSClassifier:
    def __init__(self, type="all"):
        self.type = type

    def fit(self, X, y):
        pass

    def predict(self, X):
        if self.type == "all":
            return X.apply(lambda row: DfsPredict(row), axis=1)
        elif self.type == "male":
            return X.apply(lambda row: DfsPredictMale(row), axis=1)
        elif self.type == "female":
            return X.apply(lambda row: DfsPredictFemale(row), axis=1)

    def score(self, X, y):
        pass
