import os
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sdmetrics.single_table import DiscreteKLDivergence, ContinuousKLDivergence, KSComplement, CSTest, SVCDetection, TVComplement, LogisticDetection


from anonymeter.evaluators import SinglingOutEvaluator, InferenceEvaluator


def _split_data(df_train, df_test, nominal_features, target):
    def split_into_X_y(column_transformer, data):
        X, y = data.drop(target, axis=1), data[target]
        X_transformed = column_transformer.transform(X)

        return (X_transformed, y)

    column_transformer = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), nominal_features)]
    )

    column_transformer.fit(df_train)

    X_train, y_train = split_into_X_y(column_transformer, df_train)
    X_test, y_test = split_into_X_y(column_transformer, df_test)

    return X_train, y_train, X_test, y_test


def train_and_evaluate(clf, df_train, df_test, nominal_features, target):
    X_train, y_train, X_test, y_test = _split_data(df_train, df_test, nominal_features, target)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='macro'),
        'recall': recall_score(y_test, y_pred, average='macro'),
        'f1': f1_score(y_test, y_pred, average='macro')
    }


def generate_synthetic_data(synthesizer, df, num_rows=None, fit=False):
    def fit_synthesizer(synthesizer, df):
        os.makedirs('./data/SDV', exist_ok=True)

        synthesizer.fit(df)
        synthesizer.save(filepath=f'./data/SDV/{type(synthesizer).__name__}.pkl')

    try:
        if fit:
            fit_synthesizer(synthesizer, df)
        else:
            synthesizer = synthesizer.load(filepath=f'./data/SDV/{type(synthesizer).__name__}.pkl')
    except:
        fit_synthesizer(synthesizer, df)

    if num_rows is None:
        num_rows = len(df)
    return synthesizer.sample(num_rows=num_rows)


def _evaluate(evaluator, mode):
    try:
        evaluator.evaluate(mode=mode)
        return evaluator.risk()
    except RuntimeError as ex:
        print(f"Singling out evaluation failed with {ex}. Please re-run this cell."
              "For more stable results increase `n_attacks`. Note that this will "
              "make the evaluation slower.")
        return None


def evaluate_privacy_risks(df_orig, df_synth, df_control, n_attacks=1000, n_cols=None):
    if n_cols is None:
        n_cols = len(df_orig.columns)

    return {
        'univariate': _evaluate(SinglingOutEvaluator(ori=df_orig, syn=df_synth, control=df_control, n_attacks=n_attacks), 'univariate'),
        'multivariate': _evaluate(SinglingOutEvaluator(ori=df_orig, syn=df_synth, control=df_control, n_attacks=n_attacks, n_cols=len(df_orig.columns)), 'multivariate')
    }


def evaluate_fidelity(df_orig, df_synth):
    return {
        'CSTest': CSTest.compute(df_orig, df_synth),
        'ContinuousKLDivergence': ContinuousKLDivergence.compute(df_orig, df_synth),
        'DiscreteKLDivergence': DiscreteKLDivergence.compute(df_orig, df_synth),
        'SVCDetection': SVCDetection.compute(df_orig, df_synth),
        'LogisticDetection': LogisticDetection.compute(df_orig, df_synth)
    }


def evaluate_inference_risks(df_orig, df_synth, df_control, n_attacks=1000):
    columns = df_orig.columns
    results = []

    for secret in columns:
        aux_cols = [col for col in columns if col != secret]
        evaluator = InferenceEvaluator(ori=df_orig, syn=df_synth, control=df_control, aux_cols=aux_cols, secret=secret, n_attacks=n_attacks)
        evaluator.evaluate(n_jobs=-2)
        results.append((secret, evaluator.results()))

    print({res[0]: res[1].risk() for res in results})
    visulize_inference_risks(results)


def visulize_inference_risks(results):
    _, ax = plt.subplots()
    risks = [res[1].risk().value for res in results]
    columns = [res[0] for res in results]

    ax.bar(x=columns, height=risks, alpha=0.5, ecolor='black', capsize=10)

    plt.xticks(rotation=45, ha='right')
    ax.set_ylabel("Measured inference risk")
    _ = ax.set_xlabel("Secret column")
