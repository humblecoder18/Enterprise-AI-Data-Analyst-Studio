import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression


class SHAPExplainer:
    def __init__(self, pipeline: Pipeline, X_train: pd.DataFrame):
        if not isinstance(pipeline, Pipeline):
            raise TypeError("Expected a fitted sklearn Pipeline.")

        self.pipeline = pipeline
        self.preprocessor = pipeline.named_steps["preprocessor"]
        self.model = pipeline.named_steps["model"]

        self.X_train = X_train.copy()
        self.X_train_processed = self.preprocessor.transform(self.X_train)

        try:
            self.feature_names = list(self.preprocessor.get_feature_names_out())
        except Exception:
            self.feature_names = [f"Feature_{i}" for i in range(self.X_train_processed.shape[1])]

        # Friendly names for UI
        self.display_feature_names = [
            n.replace("num__", "").replace("cat__", "") for n in self.feature_names
        ]

        if isinstance(self.model, (
            RandomForestClassifier,
            RandomForestRegressor,
            GradientBoostingClassifier,
            GradientBoostingRegressor
        )):
            self.explainer = shap.TreeExplainer(self.model)

        elif isinstance(self.model, (LogisticRegression, LinearRegression)):
            bg = shap.sample(self.X_train_processed,
                             min(100, len(self.X_train_processed)),
                             random_state=42)
            self.explainer = shap.LinearExplainer(self.model, bg)
        else:
            bg = shap.sample(self.X_train_processed,
                             min(100, len(self.X_train_processed)),
                             random_state=42)
            pred = self.model.predict_proba if hasattr(self.model, "predict_proba") else self.model.predict
            self.explainer = shap.Explainer(pred, bg)

    def _transform(self, X):
        Xp = self.preprocessor.transform(X)
        sv = self.explainer(Xp)
        return Xp, sv

    def compute_shap_values(self, X):
        return self._transform(X)[1]

    def feature_importance(self, X):
        _, sv = self._transform(X)
        imp = np.abs(sv.values).mean(axis=0)
        return pd.DataFrame({
            "Feature": self.display_feature_names,
            "Internal Feature": self.feature_names,
            "Importance": imp
        }).sort_values("Importance", ascending=False).reset_index(drop=True)

    def summary_plot(self, X):
        Xp, sv = self._transform(X)
        fig = plt.figure(figsize=(10,6))
        shap.summary_plot(sv.values, Xp, feature_names=self.display_feature_names, show=False)
        return fig

    def bar_plot(self, X):
        df = self.feature_importance(X).head(20)
        fig = plt.figure(figsize=(10,6))
        plt.barh(df["Feature"][::-1], df["Importance"][::-1])
        plt.tight_layout()
        return fig

    def waterfall_plot(self, X, row_index=0):
        _, sv = self._transform(X)
        fig = plt.figure(figsize=(10,6))
        shap.plots.waterfall(sv[row_index], show=False)
        return fig

    def shap_table(self, X, row_index=0):
        _, sv = self._transform(X)
        vals = sv.values[row_index]
        return pd.DataFrame({
            "Feature": self.display_feature_names,
            "Internal Feature": self.feature_names,
            "SHAP Value": vals,
            "Absolute Impact": np.abs(vals)
        }).sort_values("Absolute Impact", ascending=False).reset_index(drop=True)

    def dependence_plot(self, feature, X):
        Xp, sv = self._transform(X)

        if isinstance(feature, str):
            # Accept internal or display name
            if feature in self.feature_names:
                idx = self.feature_names.index(feature)
            elif feature in self.display_feature_names:
                idx = self.display_feature_names.index(feature)
            else:
                matches = [i for i, f in enumerate(self.display_feature_names)
                           if feature.lower() in f.lower()]
                if not matches:
                    raise ValueError(f"Could not find feature named: {feature}")
                idx = matches[0]
        else:
            idx = int(feature)

        fig = plt.figure(figsize=(10,6))
        shap.dependence_plot(
            idx,
            sv.values,
            Xp,
            feature_names=self.display_feature_names,
            show=False
        )
        return fig