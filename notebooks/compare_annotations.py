import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import polars.selectors as cs
    import seaborn as sns

    from sklearn.metrics import (
        average_precision_score,
        roc_auc_score,
    )

    return Path, average_precision_score, np, pl, plt, roc_auc_score, sns


@app.cell
def _(Path):
    data_dir = Path("/orcd/data/manoli/001/rcalef/data/vep_comparisons/variants")

    variants_path = data_dir / "collated_variants.annotated.tsv.gz"
    return (variants_path,)


@app.cell
def _(pl, variants_path):
    df = pl.read_csv(
        variants_path,
        separator="\t",
        null_values="-",
        schema_overrides={
            "multisusie_pip": pl.Float64,
            "eqtl_pip": pl.Float64,
        }
    )

    df.shape
    return (df,)


@app.cell
def _(df):
    df.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Raw annotation comparisons
    """)
    return


@app.cell
def _(df, pl, plt, roc_auc_score, sns):
    annotations = [
        "roulette_mr",
        "gnocchi_z",
        "phylop",
    ]

    label_cols = [
        "pip",
        "label",
    ]

    def raw_score_comparison(
        data: pl.DataFrame,
        dataset: str = "ukbb",
        pos_labels: list[str] = ["high_pip"],
        neg_labels: list[str] = ["low_pip"],
        all_non_null: bool = True,
        refilter_pip: bool = False,
        high_pip: float = 0.5,
        low_pip: float = 0.1,
        pos_name: str = "high PIP",
        neg_name: str = "low PIP",
    ):

        want_data = (
            df
            .rename({f"{dataset}_{col}": col for col in label_cols})
            .filter(pl.col("label").is_not_null())
        )
        if all_non_null:
            want_data = want_data.filter(pl.all_horizontal(pl.col(annotations).is_not_null()))

        by_category = pl.concat((
            want_data.with_columns(biotype=pl.lit("mixed")),
            want_data.filter(pl.col("biotype") == "lncRNA"),
            want_data.filter(pl.col("biotype") == "protein_coding"),
        ))

        print(f"{dataset}: {len(want_data)} rows")


        if refilter_pip:
            assert high_pip >= 0.5
            assert low_pip <= 0.1
            by_category = (
                by_category
                .with_columns(
                    label=(
                        pl.when(pl.col("pip") >= high_pip)
                        .then(pl.lit("high_pip"))
                        .otherwise(
                            pl.when(pl.col("pip") <= low_pip)
                            .then(pl.lit("low_pip"))
                            .otherwise(None),
                        )
                    )
                )
                .filter(pl.col("label").is_not_null())
            )

        long = (
            by_category
            .unpivot(
                on=annotations,
                index=label_cols + ["biotype"],
                variable_name="annotation",
                value_name="score"
            )
            .with_columns(
                label = pl.when(pl.col("label").is_in(pos_labels))
                        .then(pl.lit(pos_name))
                        .otherwise(
                            pl.when(pl.col("label").is_in(neg_labels))
                            .then(pl.lit(neg_name))
                            .otherwise(None),
                        )
            )
        )

        all_aucs = {}
        for biotype in ["mixed", "lncRNA", "protein_coding"]:
            for annot in annotations:
                want_rows = long.filter(
                    pl.col("annotation") == annot,
                    pl.col("score").is_not_null(),
                    pl.col("biotype") == biotype,
                )

                auc = roc_auc_score(
                    y_true=(want_rows.get_column("label") == pos_name).cast(pl.UInt8).to_numpy(),
                    y_score=(want_rows.get_column("score")).to_numpy()
                )
                num_pos = (want_rows.get_column("label") == pos_name).sum()
                num_neg = (want_rows.get_column("label") == neg_name).sum()

                print(f"{dataset}: {biotype} - {annot} ({num_pos} / {num_neg}) -> AUC={auc:0.3f}")
                all_aucs[(biotype, annot)] = auc

        fg = sns.FacetGrid(
            long,
            col="annotation",
            row="biotype",
            sharey=False,
        )
        fg.map_dataframe(
            sns.boxplot,
            x="label",
            y="score"
        )
        for (biotype, annot), ax in fg.axes_dict.items():
            auc = all_aucs[(biotype, annot)]
            ax.set_title(f"biotype={biotype}\nannotation={annot}\nAUC={auc:0.3f}")

        plt.suptitle(f"dataset = {dataset}")
        fg.tight_layout()


        return fg

    return (raw_score_comparison,)


@app.cell
def _(df, raw_score_comparison):
    raw_score_comparison(df, dataset="ukbb", high_pip=0.5, low_pip=0.1)
    return


@app.cell
def _(df, raw_score_comparison):
    raw_score_comparison(df, dataset="multisusie", high_pip=0.5, low_pip=0.1)
    return


@app.cell
def _(df, raw_score_comparison):
    raw_score_comparison(df, dataset="eqtl", high_pip=0.5, low_pip=0.1)
    return


@app.cell
def _(df, raw_score_comparison):
    raw_score_comparison(
        df,
        dataset="clinvar",
        pos_labels=["Pathogenic", "Likely_pathogenic"],
        neg_labels=["Benign", "Likely_benign"],
        pos_name="pathogenic",
        neg_name="benign",
    )
    return


@app.cell
def _(df, pl):
    ukbb_only = df.filter(pl.col("ukbb_pip").is_not_null())
    ukbb_only.shape
    return (ukbb_only,)


@app.cell
def _(pl, sns, ukbb_only):
    low_pip = ukbb_only.filter(pl.col("ukbb_label") == "low_pip")
    normed = (
        ukbb_only
        .filter(pl.col("ukbb_label") == "high_pip")
        .with_columns(
            gnocchi_normed=(pl.col("gnocchi_z") - low_pip.get_column("gnocchi_z").mean()) / low_pip.get_column("gnocchi_z").std(),
            phylop_normed=(pl.col("phylop") - low_pip.get_column("phylop").mean()) / low_pip.get_column("phylop").std(),
            roulette_normed=(pl.col("roulette_mr") - low_pip.get_column("roulette_mr").mean()) / low_pip.get_column("roulette_mr").std(),
        )
    )
    ax = sns.scatterplot(
        x="gnocchi_normed",
        y="phylop_normed",
        marker=".",
        data=normed,
    )
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_xlim()
    pmin = min(xmin, ymin)
    pmax = min(xmax, ymax)
    ax.plot([pmin, pmax], [pmin, pmax], '--')

    ax.plot([])
    return (normed,)


@app.cell
def _(normed, pl):
    (
        normed
        .with_columns(
            phylop_pos=pl.col("phylop_normed") >= 2,
            gnocchi_pos=pl.col("gnocchi_normed") >= 2,
        )
        .with_columns(
            gnocchi_only=pl.col("gnocchi_pos") & ~pl.col("phylop_pos")
        )
        .get_column("gnocchi_only")
        .mean()
    )
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Complementarity
    """)
    return


@app.cell
def _():
    from typing import Any

    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import (
        LogisticRegression,
        LogisticRegressionCV,
    )
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.tree import DecisionTreeClassifier

    return (
        Any,
        DecisionTreeClassifier,
        GradientBoostingClassifier,
        GridSearchCV,
        LogisticRegressionCV,
        RandomForestClassifier,
        StratifiedKFold,
    )


@app.cell
def _(df, pl):
    dataset = "ukbb"
    label_col = f"{dataset}_label"
    pos_labels = ["high_pip"]
    want_df = (
        df
        .filter(
            pl.all_horizontal(pl.col("phylop", "gnocchi_z").is_not_null()),
            pl.col(label_col).is_not_null()
        )
        .with_columns(
            label = pl.when(pl.col(label_col).is_in(pos_labels))
                    .then(1)
                    .otherwise(0)
        )
    )

    want_df.shape
    return (want_df,)


@app.cell
def _(want_df):
    y = want_df.get_column("label")
    y.value_counts()
    return (y,)


@app.cell
def _(want_df):
    X = want_df.select("phylop", "gnocchi_z").to_numpy()
    X.shape
    return (X,)


@app.cell
def _(
    Any,
    GridSearchCV,
    StratifiedKFold,
    average_precision_score,
    np,
    pl,
    roc_auc_score,
):
    def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prefix: str,
    ) -> dict[str, float]:
        return {
            f"{prefix}_auprc": average_precision_score(y_true=y_true, y_score=y_pred),
            f"{prefix}_auroc": roc_auc_score(y_true=y_true, y_score=y_pred),
        }

    def cv_classify(
        X: np.ndarray,
        y: np.ndarray,
        clf,
        grid_search_params: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        all_metrics = []
        if grid_search_params:
            clf = GridSearchCV(clf, grid_search_params, n_jobs=8)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train = X[train_idx]
            X_test = X[test_idx]

            y_train = y[train_idx]
            y_test = y[test_idx]

            clf = clf.fit(X_train, y_train)
            y_pred_train = clf.predict_proba(X_train)[:, 1]
            y_pred_test = clf.predict_proba(X_test)[:, 1]

            metrics = {"fold": fold}
            metrics.update(compute_metrics(y_train, y_pred_train, "train"))
            metrics.update(compute_metrics(y_test, y_pred_test, "test"))
            all_metrics.append(metrics)

        all_metrics = pl.DataFrame(all_metrics)
        print(
            f"AUROC: {all_metrics.get_column("test_auroc").mean():0.3f}",
            f"AUPRC: {all_metrics.get_column("test_auprc").mean():0.3f}",
        )
        return all_metrics

    return (cv_classify,)


@app.cell
def _(LogisticRegressionCV, X, cv_classify, y):
    cv_classify(
        X=X,
        y=y,
        clf=LogisticRegressionCV(
            l1_ratios=(0,),
            scoring="neg_log_loss",
            use_legacy_attributes=False,
        ),
    )
    return


@app.cell
def _(DecisionTreeClassifier, X, cv_classify, y):
    cv_classify(
        X=X,
        y=y,
        grid_search_params={
            "min_samples_leaf": [2, 10, 0.01, 0.05, 0.1],
            "max_depth": [None, 2, 4, 8],
        },
        clf=DecisionTreeClassifier(),
    )
    return


@app.cell
def _(RandomForestClassifier, X, cv_classify, y):
    cv_classify(
        X=X,
        y=y,
        grid_search_params={
            "min_samples_leaf": [2, 10, 0.01, 0.05, 0.1],
            "max_depth": [None, 2, 4, 8],
            "n_estimators": [5, 10, 100],
        },
        clf=RandomForestClassifier(),
    )
    return


@app.cell
def _(GradientBoostingClassifier, X, cv_classify, y):
    cv_classify(
        X=X,
        y=y,
        clf=GradientBoostingClassifier(
            min_samples_leaf=0.05,
        ),
    )
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(clf):
    clf.classes_
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ESM-C scores
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(pl):
    esmc_scores = (
        pl.read_csv(
            "/orcd/data/manoli/001/rcalef/projects/vep_comparisons/scoring/collated_variants.esmc_300m.tsv.gz",
            separator="\t",
        )
        # Original scores are log(ref) - log(alt), so negate
        # so we have "pathogenicity" scores instead of
        # "tolerability" scores.
        .with_columns(
            score=-pl.col("score")
        )
    )
    print(esmc_scores.shape)
    esmc_scores.head()
    return (esmc_scores,)


@app.cell
def _(esmc_scores):
    esmc_scores.get_column("score").is_not_null().value_counts()
    return


@app.cell
def _(df, esmc_scores):
    merged = (
        df.join(esmc_scores, on=["variant", "gene"], how="inner", validate="1:1")
    )
    print(f"{len(df)} -> {len(merged)}")
    return (merged,)


@app.cell
def _(merged):
    merged.head()
    return


@app.cell
def _(merged, pl):
    clinvar_check = (
        merged
        .filter(
            pl.col("clinvar_label").is_not_null(),
            pl.col("biotype") == "protein_coding",
        )
        .with_columns(
            label=pl.when(pl.col("clinvar_label").is_in(["Pathogenic", "Likely_pathogenic"],))
                    .then(pl.lit("pos"))
                    .otherwise(pl.lit("neg")),
        )
        .select("score", "label")
    )
    print(clinvar_check.shape)
    clinvar_check.head()
    return (clinvar_check,)


@app.cell
def _(clinvar_check, pl, roc_auc_score):
    roc_auc_score(
        y_true=(clinvar_check.get_column("label") == "pos").cast(pl.UInt8).to_numpy(),
        y_score=(clinvar_check.get_column("score")).to_numpy()
    )
    return


@app.cell
def _(merged):
    datasets = ["ukbb", "multisusie", "eqtl", "clinvar"]

    for ds in datasets:
        print(ds)
        print(merged.get_column(f"{ds}_label").value_counts())
    return


@app.cell
def _(pl, roc_auc_score, sns):
    def score_comparison(
        data: pl.DataFrame,
        all_non_null: bool = True,
        refilter_pip: bool = False,
        high_pip: float = 0.5,
        low_pip: float = 0.1,
        biotypes: list[str] = ["all", "lncRNA", "protein_coding"],
    ):
        ds_labels = {
            "ukbb": ["high_pip"],
            "multisusie": ["high_pip"],
            "eqtl": ["high_pip"],
            "clinvar": ["Pathogenic", "Likely_pathogenic"],

        }
        label_cols = [f"{ds}_label" for ds in ds_labels]

        if all_non_null:
            data = data.filter(pl.all_horizontal(pl.col(label_cols).is_not_null()))

        by_category = []
        for category in biotypes:
            if category == "all":
                by_category.append(data.with_columns(biotype=pl.lit("all")))
            else:
                by_category.append(data.filter(pl.col("biotype") == category),)
        by_category = pl.concat(by_category)

        # if refilter_pip:
        #     assert high_pip >= 0.5
        #     assert low_pip <= 0.1
        #     by_category = (
        #         by_category
        #         .with_columns(
        #             label=(
        #                 pl.when(pl.col("pip") >= high_pip)
        #                 .then(pl.lit("high_pip"))
        #                 .otherwise(
        #                     pl.when(pl.col("pip") <= low_pip)
        #                     .then(pl.lit("low_pip"))
        #                     .otherwise(None),
        #                 )
        #             )
        #         )
        #         .filter(pl.col("label").is_not_null())
        #     )


        long = []
        for ds, pos_labels in ds_labels.items():
            long.append(
                by_category
                .filter(
                    pl.col(f"{ds}_label").is_not_null(),
                )
                .with_columns(
                    label=pl.when(pl.col(f"{ds}_label").is_in(pos_labels))
                            .then(pl.lit("pos"))
                            .otherwise(pl.lit("neg")),
                    dataset=pl.lit(ds),
                )

                .select("score", "biotype", "label", "dataset")
            )
        long = pl.concat(long, how="vertical")

        all_aucs = {}
        for biotype in biotypes:
            for dataset in ds_labels:
                want_rows = long.filter(
                    pl.col("score").is_not_null(),
                    pl.col("biotype") == biotype,
                    pl.col("dataset") == dataset,
                )
                if len(want_rows) == 0:
                    print(f"{dataset} - no data")
                    continue

                auc = roc_auc_score(
                    y_true=(want_rows.get_column("label") == "pos").cast(pl.UInt8).to_numpy(),
                    y_score=(want_rows.get_column("score")).to_numpy()
                )
                num_pos = (want_rows.get_column("label") == "pos").sum()
                num_neg = (want_rows.get_column("label") != "pos").sum()

                print(f"{dataset}: {biotype} ({num_pos} / {num_neg}) -> AUC={auc:0.3f}")
                all_aucs[(dataset, biotype)] = auc

        fg = sns.FacetGrid(
            long,
            col="dataset",
            row="biotype",
            sharey=False,
        )
        fg.map_dataframe(
            sns.boxplot,
            x="label",
            y="score"
        )

        fg.tight_layout()

        for (biotype, dataset), ax in fg.axes_dict.items():
            auc = all_aucs[(dataset, biotype)]
            ax.set_title(f"dataset={dataset}\nbiotype={biotype}\nAUC={auc:0.3f}")

        return fg

    return (score_comparison,)


@app.cell
def _(merged, score_comparison):
    score_comparison(merged, all_non_null=False, biotypes=["protein_coding"])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
