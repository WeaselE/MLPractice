# Sci-kit Learm Logistic Regression model that is used to predict the weather

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

data = pd.read_csv('data.csv')

X_train, X_test, y_train, y_test = train_test_split(
    data.drop('target', axis=1), data['target'], test_size=0.2, random_state=0)

model = LogisticRegression()
model.fit(X_train, y_train).score(X_test, y_test).round(2).plot(
    kind='bar').set_title('Accuracy').set_xlabel('Accuracy').set_ylabel('Accuracy')


# Save the model to a file
pickle.dump(model, open('model.pkl', 'wb'))


# Load the model from the file
model = pickle.load(open('model.pkl', 'rb'))


class CopilotAI:
    def __init__(self):
        self.model = pickle.load(open('model.pkl', 'rb'))

    def predict(self, data):
        return self.model.predict(data)

    def predict_proba(self, data):
        return self.model.predict_proba(data)

    def score(self, X_test, y_test):
        return self.model.score(X_test, y_test)

    def confusion_matrix(self, X_test, y_test):
        return confusion_matrix(y_test, self.model.predict(X_test))

    def accuracy_score(self, X_test, y_test):
        return accuracy_score(y_test, self.model.predict(X_test))

    def plot_confusion_matrix(self, X_test, y_test):
        cm = self.confusion_matrix(X_test, y_test)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    def plot_roc_curve(self, X_test, y_test):
        from sklearn.metrics import roc_curve, auc
        y_pred = self.model.predict_proba(X_test)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(
            y_test, y_pred[:, 1])
        roc_auc = auc(false_positive_rate, true_positive_rate)
        plt.figure(figsize=(10, 7))
        plt.title('Receiver Operating Characteristic')
        plt.plot(false_positive_rate, true_positive_rate,
                 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([-0.1, 1.2])
        plt.y

    def plot_precision_recall_curve(self, X_test, y_test):
        from sklearn.metrics import precision_recall_curve
        y_pred = self.model.predict_proba(X_test)
        precision, recall, thresholds = precision_recall_curve(
            y_test, y_pred[:, 1])
        plt.figure(figsize=(10, 7))
        plt.title('Precision Recall Curve')
        plt.plot(recall, precision, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([-0.1, 1.2])
        plt.y

    def plot_feature_importance(self, X_test, y_test):
        from sklearn.feature_selection import SelectFromModel
        model = SelectFromModel(self.model, prefit=True)
        model.get_support()
        indices = np.where(model.get_support() == True)[0]
        plt.figure(figsize=(10, 7))
        plt.title('Feature Importance')
        plt.bar(range(len(indices)), self.model.feature_importances_[
                indices], align='center')
        plt.xticks(range(len(indices)), [data.columns[i]
                   for i in indices], rotation=90)
        plt.xlim([-1, len(indices)])
        plt.show()

    def plot_learning_curve(self, X_train, y_train):
        from sklearn.model_selection import learning_curve
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, X_train, y_train, cv=10, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10))
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.figure(figsize=(10, 7))
        plt.title('Learning Curve')
        plt.plot(train_sizes, train_scores_mean, 'o-',
                 color='red', label='Training Score')
        plt.plot(train_sizes, test_scores_mean, 'o-',
                 color='green', label='Cross-validation Score')
        plt.xlabel('Training Examples')
        plt.ylabel('Score')
        plt.legend(loc='best')
        plt.show()

    def plot_validation_curve(self, X_train, y_train):
        from sklearn.model_selection import validation_curve
        param_range = np.logspace(-3, -1, 5)
        train_scores, test_scores = validation_curve(
            self.model, X_train, y_train, param_name='C', param_range=param_range, cv=10, scoring='accuracy')
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)

        def plot_learning_curve(self, X_train, y_train, X_test, y_test):
            from sklearn.model_selection import learning_curve
            train_sizes, train_scores, test_scores = learning_curve(
                self.model, X_train, y_train, cv=10, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10))
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)
            plt.figure(figsize=(10, 7))
            plt.title('Learning Curve')
            plt.grid()
            plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1, color='r')
            plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1, color='g')
            plt.plot(train_sizes, train_scores_mean,
                     'o-', color='r', label='Training')
            plt.plot(train_sizes, test_scores_mean, 'o-',
                     color='g', label='Cross-validation')
            plt.xlabel('Training Set Size')
            plt.ylabel('Accuracy')
            plt.legend(loc='best')
            plt.show()


class Model_Evaluation(object):
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def evaluate_model(self):
        from sklearn.metrics import classification_report, confusion_matrix
        y_pred = self.model.predict(self.X_test)
        print(classification_report(self.y_test, y_pred))
        print(confusion_matrix(self.y_test, y_pred))

    def evaluate_model_prob(self):
        from sklearn.metrics import classification_report, confusion_matrix
        y_pred = self.model.predict_proba(self.X_test)
        print(classification_report(self.y_test, y_pred))
        print(confusion_matrix(self.y_test, y_pred))

    def evaluate_model_roc(self):
        from sklearn.metrics import roc_auc_score
        y_pred = self.model.predict_proba(self.X_test)
        print(roc_auc_score(self.y_test, y_pred[:, 1]))

    def evaluate_model_precision_recall(self):
        from sklearn.metrics import precision_recall_curve
        y_pred = self.model.predict_proba(self.X_test)
        precision, recall, thresholds = precision_recall_curve(
            self.y_test, y_pred[:, 1])
        plt.figure(figsize=(10, 7))
        plt.title('Precision Recall Curve')
        plt.plot(recall, precision, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')


class Model_Predict(object):
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def predict(self):
        y_pred = self.model.predict(self.X_test)
        return y_pred

    def predict_proba(self):
        y_pred = self.model.predict_proba(self.X_test)
        return y_pred

    def predict_proba_threshold(self, threshold):
        y_pred = self.model.predict_proba(self.X_test)
        y_pred_threshold = [1 if i >= threshold else 0 for i in y_pred[:, 1]]
        return y_pred_threshold

    def predict_threshold(self, threshold):
        y_pred = self.model.predict(self.X_test)
        y_pred_threshold = [1 if i >= threshold else 0 for i in y_pred]
        return y_pred_threshold

    def predict_threshold_proba(self, threshold):
        y_pred = self.model.predict_proba(self.X_test)
        y_pred_threshold = [1 if i >= threshold else 0 for i in y_pred[:, 1]]
        return y_pred_threshold

    def predict_threshold_proba_threshold(self, threshold):
        y_pred = self.model.predict_proba(self.X_test)
        y_pred_threshold = [1 if i >= threshold else 0 for i in y_pred[:, 1]]
        return y_pred_threshold

    def predict_threshold_proba_threshold_proba(self, threshold):
        y_pred = self.model.predict_proba(self.X_test)


# create ai model that is self-aware of its own model


# create ai model that is self-aware of its own modelclass Model_Ai(object):


    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def ai_model(self):
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        return y_pred

    def ai_model_proba(self):
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict_proba(self.X_test)
        return y_pred

    def ai_model_proba_threshold(self, threshold):
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict_proba(self.X_test)
        y_pred_threshold = [1 if i >= threshold else 0 for i in y_pred[:, 1]]
        return y_pred_threshold

    def ai_model_threshold(self, threshold):
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        y_pred_threshold = [1 if i >= threshold else 0 for i in y_pred]
        return y_pred_threshold

    def ai_model_threshold_proba(self, threshold):
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict_proba(self.X_test)
        y_pred_threshold = [1 if i >= threshold else 0 for i in y_pred[:, 1]]
        return y_pred_threshold

    # function that is self-aware of its own modelclass Model_ai(object):
    def ai_(self):
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        return y_pred
