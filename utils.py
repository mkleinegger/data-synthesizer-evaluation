import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

from anonymeter.evaluators import SinglingOutEvaluator
from anonymeter.evaluators import InferenceEvaluator

def split_data(df_train, df_test, nominal_features):
    X_train = df_train.drop('ScoreText', axis=1)
    X_train = pd.get_dummies(X_train, columns=nominal_features)
    y_train = df_train['ScoreText']

    X_test = df_test.drop('ScoreText', axis=1)
    X_test = pd.get_dummies(X_test, columns=nominal_features)
    y_test = df_test['ScoreText']
    
    return X_train, y_train, X_test, y_test

def train_and_evaluate(clf, df_train, df_test, nominal_features, remove_diff_cols=False):
    X_train, y_train, X_test, y_test = split_data(df_train, df_test, nominal_features)
    if remove_diff_cols:
        # Remove columns that are not in the train set
        X_test = X_test[X_train.columns]

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:0.2f}%")

def generate_synthetic_data(synthesizer, df, num_rows=None):
    if num_rows is None:
        num_rows = len(df)

    synthesizer.fit(df)
    return synthesizer.sample(num_rows=num_rows)

def evaluate_privacy_risks(df_orig, df_synth, n_attacks = 1000, n_cols = None):
    if n_cols is None:
        n_cols = len(df_orig.columns)

    evaluator = SinglingOutEvaluator(ori=df_orig, syn=df_synth, n_attacks = n_attacks, n_cols = n_cols)
    try:
        evaluator.evaluate(mode='univariate')
        print(f"Privacy risks concerning the univariate attacks (1 col): ${evaluator.risk()}")

    except RuntimeError as ex: 
        print(f"Singling out evaluation failed with {ex}. Please re-run this cell."
          "For more stable results increase `n_attacks`. Note that this will "
          "make the evaluation slower.")
        
    try:
        evaluator.evaluate(mode='multivariate')
        print(f"Privacy risks concerning the multivariate attacks (${n_cols} col): ${evaluator.risk()}")

    except RuntimeError as ex: 
        print(f"Singling out evaluation failed with {ex}. Please re-run this cell."
          "For more stable results increase `n_attacks`. Note that this will "
          "make the evaluation slower.")

def evaluate_inference_risks(df_orig, df_synth, n_attacks = 1000):
    columns = df_orig.columns
    results = []

    for secret in columns:
        aux_cols = [col for col in columns if col != secret]
        evaluator = InferenceEvaluator(ori=df_orig, syn=df_synth, aux_cols=aux_cols, secret=secret, n_attacks=n_attacks)
        evaluator.evaluate(n_jobs=-2)
        results.append((secret, evaluator.results()))

    visulize_inference_risks(results)


def visulize_inference_risks(results):
    fig, ax = plt.subplots()
    risks = [res[1].risk().value for res in results]
    columns = [res[0] for res in results]

    ax.bar(x=columns, height=risks, alpha=0.5, ecolor='black', capsize=10)

    plt.xticks(rotation=45, ha='right')
    ax.set_ylabel("Measured inference risk")
    _ = ax.set_xlabel("Secret column")