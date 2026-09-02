import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    from collections import defaultdict
    from functools import partial
    from multiprocessing import Pool
    from pathlib import Path
    from typing import Any, Literal

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import seaborn as sns
    from sklearn.linear_model import (
        LogisticRegressionCV,
    )
    from sklearn.metrics import (
        average_precision_score,
        roc_auc_score,
    )
    from sklearn.model_selection import (
        StratifiedGroupKFold,
        cross_val_predict,
    )
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.utils import resample


    return (
        Any,
        Literal,
        LogisticRegressionCV,
        Path,
        PolynomialFeatures,
        Pool,
        StandardScaler,
        StratifiedGroupKFold,
        average_precision_score,
        cross_val_predict,
        defaultdict,
        make_pipeline,
        mo,
        np,
        partial,
        pl,
        plt,
        resample,
        roc_auc_score,
        sns,
    )


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Raw annotation comparisons
    """)


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


@app.cell
def _(df, raw_score_comparison):
    raw_score_comparison(df, dataset="multisusie", high_pip=0.5, low_pip=0.1)


@app.cell
def _(df, raw_score_comparison):
    raw_score_comparison(df, dataset="eqtl", high_pip=0.5, low_pip=0.1)


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Complementarity

    I test complementarity in three ways:

    1. Compare out-of-fold predictions from each score alone, an additive logistic
       model, and a logistic model with quadratic interactions.
       AUROC measures global ranking, while AUPRC emphasizes the rare high-PIP class.
    2. Bootstrap paired differences from the same out-of-fold predictions to show
       whether any AUROC gain over PhyloP is larger than sampling uncertainty.
    3. Bin both scores into quintiles and inspect the high-PIP rate within each cell.
       This asks whether Gnocchi separates variants that have similar PhyloP scores.

    Each analysis is run for UKBB, MultiSuSiE, and their labeled-row union, both
    overall and separately for protein-coding and lncRNA variants. In the combined
    endpoint, a variant is high PIP if either study calls it high PIP; otherwise all
    available labels must be low PIP. Quantiles and normalization are recomputed
    within every dataset/biotype group.
    """)


@app.cell
def _(
    Any,
    Literal,
    StratifiedGroupKFold,
    average_precision_score,
    cross_val_predict,
    df,
    np,
    pl,
    roc_auc_score,
):
    def prepare_labeled_data(
        data: pl.DataFrame,
        label_columns: list[str],
    ) -> pl.DataFrame:
        """Collapse labels to single binary 0/1 labels."""
        return (
            data
            .filter(
                pl.all_horizontal(pl.col("phylop", "gnocchi_z").is_not_null()),
                pl.any_horizontal(pl.col(label_columns).is_not_null()),
            )
            .with_columns(
                label=pl.any_horizontal(
                    (pl.col(label_columns) == "high_pip").fill_null(False)
                ).cast(pl.Int8),
            )
        )

    def select_biotype_group(
        data: pl.DataFrame,
        biotype: Literal["all", "protein_coding", "lncRNA"],
    ) -> pl.DataFrame:
        return data if biotype == "all" else data.filter(pl.col("biotype") == biotype)

    def cv_eval_models(
        labeled_data: pl.DataFrame,
        models: dict[str, Any],
        cv: StratifiedGroupKFold,
    ) -> tuple[pl.DataFrame, np.ndarray, dict[str,np.ndarray]]:
        y = labeled_data.get_column("label").to_numpy()
        groups = labeled_data.get_column("variant").to_numpy()
        predictions = {}
        metrics = []
        for model_name, (features, model) in models.items():
            prediction = cross_val_predict(
                model,
                labeled_data.select(features).to_numpy(),
                y,
                groups=groups,
                cv=cv,
                method="predict_proba",
                n_jobs=1,
            )[:, 1]
            predictions[model_name] = prediction
            metrics.append({
                "model": model_name,
                "auroc": roc_auc_score(y, prediction),
                "auprc": average_precision_score(y, prediction),
            })
        return pl.DataFrame(metrics), y, predictions

    def eval_all_datasets(
        dataset_specs: dict[str, list[str]],
        want_biotypes: list[str],
        model_specs: dict[str, Any],
        n_folds: int = 5,
    ) -> tuple[pl.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
        complementarity_cv = StratifiedGroupKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=42,
        )

        oof_predictions = {}
        oof_outcomes = {}
        model_metrics = []
        for dataset, label_columns in dataset_specs.items():
            dataset_data = prepare_labeled_data(df, label_columns)
            for biotype in want_biotypes:
                data = select_biotype_group(dataset_data, biotype)
                metrics, _y, _predictions = cv_eval_models(
                    data,
                    model_specs,
                    complementarity_cv,
                )
                key = (dataset, biotype)
                oof_outcomes[key] = _y
                oof_predictions[key] = _predictions
                model_metrics.append(
                    metrics.with_columns(dataset=pl.lit(dataset), biotype=pl.lit(biotype))
                )
        model_metrics = pl.concat(model_metrics).sort(
            "dataset", "biotype", "auroc", descending=[False, False, True]
        )
        return model_metrics, oof_predictions, oof_outcomes


    return eval_all_datasets, prepare_labeled_data, select_biotype_group


@app.cell
def _(df, pl, prepare_labeled_data, select_biotype_group):
    dataset_specs = {
        "UKBB": ["ukbb_label"],
        "UKBB + MultiSuSiE": ["ukbb_label", "multisusie_label"],
        "MultiSuSiE": ["multisusie_label"],
    }
    analysis_biotypes = ["all", "protein_coding", "lncRNA"]

    group_sizes = pl.DataFrame([
        {
            "dataset": _dataset,
            "biotype": _biotype,
            "variants": len(_data),
            "high_pip": _data.get_column("label").sum(),
            "low_pip": len(_data) - _data.get_column("label").sum(),
        }
        for _dataset, _label_columns in dataset_specs.items()
        for _biotype in analysis_biotypes
        for _data in [
            select_biotype_group(
                prepare_labeled_data(df, _label_columns),
                _biotype,
            )
        ]
    ])
    group_sizes
    return analysis_biotypes, dataset_specs


@app.cell
def _(LogisticRegressionCV, PolynomialFeatures, StandardScaler, make_pipeline):
    lr_cv_params = {
        "l1_ratios": (0.0,),
        "scoring": "neg_log_loss",
        "use_legacy_attributes": False,
        "n_jobs": 8,
    }

    linear_model = make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(**lr_cv_params),
    )

    model_specs = {
        "PhyloP": (["phylop"], linear_model),
        "Gnocchi": (["gnocchi_z"], linear_model),
        "additive": (["phylop", "gnocchi_z"], linear_model),
        "interaction": (
            ["phylop", "gnocchi_z"],
            make_pipeline(
                PolynomialFeatures(degree=2, include_bias=False),
                StandardScaler(),
                LogisticRegressionCV(**lr_cv_params),
            ),
        ),
        # Underperforms the additive or interaction models across the board
        # "tree": (
        #     ["phylop", "gnocchi_z"],
        #     make_pipeline(
        #         PolynomialFeatures(degree=2, include_bias=False),
        #         StandardScaler(),
        #         GridSearchCV(
        #             estimator=HistGradientBoostingClassifier(),
        #             param_grid={
        #                 "max_depth": [None, 4, 8],
        #                 "max_iter": [50, 100],
        #             },
        #             n_jobs=8,
        #         ),
        #     ),
        # ),
    }


    return (model_specs,)


@app.cell
def _(dataset_specs, eval_all_datasets, model_specs):
    perf_metrics, all_preds, all_labels = eval_all_datasets(
        dataset_specs=dataset_specs,
        want_biotypes=["all"],
        model_specs=model_specs
    )


@app.cell
def _():
    return


@app.cell
def _(
    Pool,
    average_precision_score,
    dataset_specs,
    defaultdict,
    df,
    np,
    oof_predictions,
    partial,
    pl,
    prepare_complementarity_data,
    resample,
    roc_auc_score,
):

    def sample_metrics(
        data: np.ndarray,
        job_idx: int,
        num_bootstraps: int = 100,
    ) -> dict[str, list[float]]:
        rng = np.random.RandomState(seed=42+job_idx)

        results = defaultdict(list)
        for i in range(num_bootstraps):
            sampled = resample(data, random_state=rng)
            results["auroc"].append(roc_auc_score(sampled[:, 0], sampled[:, 1]))
            results["auprc"].append(average_precision_score(sampled[:, 0], sampled[:, 1]))
        return results



    n_procs = 8
    all_results = []
    paired_differences = []
    for feature_set in ["PhyloP", "Gnocchi", "interaction"]:
        for biotype in ["all", "protein_coding", "lncRNA"]:
            selected_data = prepare_complementarity_data(df, label_columns=dataset_specs["UKBB + MultiSuSiE"])

            preds_w_labels = np.column_stack((
                selected_data.get_column("label").to_numpy(),
                oof_predictions[("UKBB + MultiSuSiE", "all")][feature_set],
            ))
            if biotype != "all":
                want_rows = selected_data.get_column("biotype") == biotype
                preds_w_labels = preds_w_labels[want_rows, :]


            metric_func = partial(sample_metrics, preds_w_labels)
            with Pool(processes=n_procs) as p:
                raw_bootstraps = p.map(metric_func, range(n_procs))

            all_results.append(
                pl.DataFrame({
                    "auroc": [x for vals in raw_bootstraps for x in vals["auroc"]],
                    "auprc": [x for vals in raw_bootstraps for x in vals["auprc"]],
                }).with_columns(
                    features=pl.lit(feature_set),
                    biotype=pl.lit(biotype),
                )
            )

    all_results = pl.concat(all_results)
    all_results.head()
    return all_results, n_procs


@app.cell
def _():
    return


@app.cell
def _(
    Pool,
    dataset_specs,
    df,
    n_procs,
    np,
    oof_predictions,
    partial,
    pl,
    prepare_complementarity_data,
    resample,
    roc_auc_score,
):
    def paired_auc_delta(
        y_pred_ref: np.ndarray,
        y_pred_oth: np.ndarray,
        y_true: np.ndarray,
        job_idx: int,
        num_bootstraps: int = 100,
    ) -> np.ndarray:
        rng = np.random.RandomState(seed=42+job_idx)

        results = []
        for i in range(num_bootstraps):
            y_ref, y_oth, y_t = resample(y_pred_ref, y_pred_oth, y_true, random_state=rng)
            auroc_ref = roc_auc_score(y_t, y_ref)
            auroc_oth = roc_auc_score(y_t, y_oth)
            results.append(auroc_oth - auroc_ref)
        return np.array(results)


    gwas_data = prepare_complementarity_data(df, label_columns=dataset_specs["UKBB + MultiSuSiE"])
    auroc_diffs = []
    for _biotype in ["all", "protein_coding", "lncRNA"]:
        labels = gwas_data.get_column("label").to_numpy()
        phylop_scores = oof_predictions[("UKBB + MultiSuSiE", "all")]["PhyloP"]
        interac_scores = oof_predictions[("UKBB + MultiSuSiE", "all")]["interaction"]

        if _biotype != "all":
            bt_rows = gwas_data.get_column("biotype") == _biotype

            labels = labels[bt_rows]
            phylop_scores = phylop_scores[bt_rows]
            interac_scores = interac_scores[bt_rows]

        _metric_func = partial(paired_auc_delta, phylop_scores, interac_scores, labels)
        with Pool(processes=n_procs) as _p:
            raw_diffs = _p.map(_metric_func, range(n_procs))

        auroc_diffs.append(
            pl.DataFrame({
                "auroc_diff": [x for vals in raw_diffs for x in vals],
            }).with_columns(
                biotype=pl.lit(_biotype),
            )
        )
    auroc_diffs = pl.concat(auroc_diffs)
    auroc_diffs.head()
    return auroc_diffs, paired_auc_delta


@app.cell
def _(auroc_diffs, pl):
    (
        auroc_diffs
        .group_by("biotype")
        .agg(
            ci_lo=pl.col("auroc_diff").quantile(0.025),
            ci_hi=pl.col("auroc_diff").quantile(0.975),
            mid=pl.col("auroc_diff").median(),
        )
    )


@app.cell
def _(all_results, plt, sns):
    sns.barplot(
        x="biotype",
        y="auroc",
        hue="features",
        data=all_results,
    )
    _ = plt.ylim(0.5)
    plt.show()


@app.cell
def _(all_results, sns):
    sns.barplot(
        x="biotype",
        y="auprc",
        hue="features",
        data=all_results,
    )


@app.cell
def _(auroc_diffs, sns):
    sns.barplot(
        x="biotype",
        y="auroc_diff",
        data=auroc_diffs,
        errorbar=("pi", 95)
    )


@app.cell
def _(oof_outcomes, oof_predictions, paired_auc_delta, pl):
    bootstrap_rows = []
    for (_dataset, _biotype), _predictions in oof_predictions.items():
        for _model_name, _prediction in _predictions.items():
            if _model_name != "PhyloP":
                _delta, _low, _high = paired_auc_delta(
                    oof_outcomes[(_dataset, _biotype)],
                    _predictions["PhyloP"],
                    _prediction,
                )
                bootstrap_rows.append({
                    "dataset": _dataset,
                    "biotype": _biotype,
                    "model": _model_name,
                    "delta_auroc": _delta,
                    "ci_low": _low,
                    "ci_high": _high,
                })

    auroc_gains = pl.DataFrame(bootstrap_rows).sort(
        "dataset", "biotype", "delta_auroc", descending=[False, False, True]
    )
    auroc_gains
    return (auroc_gains,)


@app.cell
def _(
    analysis_biotypes,
    dataset_specs,
    df,
    pl,
    plt,
    prepare_complementarity_data,
    select_analysis_group,
    sns,
):
    quintiles = [str(i) for i in range(1, 6)]

    def compute_binned_rates(data, dataset, biotype):
        return (
            data
            .with_columns(
                phylop_bin=pl.col("phylop").qcut(5, labels=quintiles),
                gnocchi_bin=pl.col("gnocchi_z").qcut(5, labels=quintiles),
            )
            .group_by("phylop_bin", "gnocchi_bin")
            .agg(high_pip_rate=pl.col("label").mean())
            .pivot(on="gnocchi_bin", index="phylop_bin", values="high_pip_rate")
            .sort("phylop_bin")
            .select("phylop_bin", *quintiles)
            .with_columns(
                phylop_bin=pl.col("phylop_bin").cast(pl.String),
                dataset=pl.lit(dataset),
                biotype=pl.lit(biotype),
            )
        )

    _all_rates = []
    for _dataset, _label_columns in dataset_specs.items():
        _dataset_data = prepare_complementarity_data(df, _label_columns)
        for _biotype in analysis_biotypes:
            _all_rates.append(compute_binned_rates(
                select_analysis_group(_dataset_data, _biotype),
                _dataset,
                _biotype,
            ))
    binned_rates = pl.concat(_all_rates)

    complementarity_figure, _axes = plt.subplots(3, 3, figsize=(15, 12))
    for _row, _dataset in enumerate(dataset_specs):
        _dataset_rates = binned_rates.filter(pl.col("dataset") == _dataset)
        _vmax = _dataset_rates.select(pl.max_horizontal(*quintiles).max()).item()
        for _column, _biotype in enumerate(analysis_biotypes):
            _rates = _dataset_rates.filter(pl.col("biotype") == _biotype)
            _ax = sns.heatmap(
                _rates.select(quintiles).to_numpy(),
                annot=True,
                fmt=".1%",
                cmap="viridis",
                vmin=0,
                vmax=_vmax,
                cbar=_column == 2,
                xticklabels=quintiles,
                yticklabels=quintiles,
                ax=_axes[_row, _column],
            )
            _ax.set(
                xlabel="Gnocchi quintile",
                ylabel="PhyloP quintile",
                title=f"{_dataset} · {_biotype}",
            )
    complementarity_figure.tight_layout()
    complementarity_figure
    return (binned_rates,)


@app.cell
def _(
    analysis_biotypes,
    dataset_specs,
    df,
    pl,
    prepare_complementarity_data,
    select_analysis_group,
):
    def summarize_extreme_region(data, dataset, biotype):
        low_pip = data.filter(pl.col("label") == 0)
        baseline_rate = data.get_column("label").mean()
        total_high_pip = data.get_column("label").sum()
        normalized = data.with_columns(
            phylop_z=(pl.col("phylop") - low_pip["phylop"].mean())
            / low_pip["phylop"].std(),
            gnocchi_z_normed=(pl.col("gnocchi_z") - low_pip["gnocchi_z"].mean())
            / low_pip["gnocchi_z"].std(),
        ).with_columns(
            gnocchi_only_extreme=(pl.col("gnocchi_z_normed") >= 2)
            & (pl.col("phylop_z") < 2),
        )
        return normalized.filter(pl.col("gnocchi_only_extreme")).select(
            dataset=pl.lit(dataset),
            biotype=pl.lit(biotype),
            variants=pl.len(),
            high_pip=pl.col("label").sum(),
            high_pip_rate=pl.col("label").mean(),
        ).with_columns(
            high_pip_recall=pl.col("high_pip") / total_high_pip,
            enrichment=pl.col("high_pip_rate") / baseline_rate,
        )

    extreme_region_summary = pl.concat([
        summarize_extreme_region(
            select_analysis_group(
                prepare_complementarity_data(df, _label_columns),
                _biotype,
            ),
            _dataset,
            _biotype,
        )
        for _dataset, _label_columns in dataset_specs.items()
        for _biotype in analysis_biotypes
    ])
    extreme_region_summary
    return (extreme_region_summary,)


@app.cell(hide_code=True)
def _(
    auroc_gains,
    binned_rates,
    dataset_specs,
    extreme_region_summary,
    mo,
    model_metrics,
    pl,
):
    _labels = {
        "all": "All variants",
        "protein_coding": "Protein coding",
        "lncRNA": "lncRNA",
    }
    _summary_lines = []
    for _dataset in dataset_specs:
        _summary_lines.append(f"**{_dataset}**")
        for _biotype, _label in _labels.items():
            _want_group = (
                (pl.col("dataset") == _dataset)
                & (pl.col("biotype") == _biotype)
            )
            _phylop = model_metrics.filter(_want_group, pl.col("model") == "PhyloP").row(0, named=True)
            _additive = model_metrics.filter(_want_group, pl.col("model") == "additive").row(0, named=True)
            _interaction = model_metrics.filter(_want_group, pl.col("model") == "interaction").row(0, named=True)
            _gain = auroc_gains.filter(_want_group, pl.col("model") == "additive").row(0, named=True)
            _interaction_gain = auroc_gains.filter(
                _want_group,
                pl.col("model") == "interaction",
            ).row(0, named=True)
            _extreme = extreme_region_summary.filter(_want_group).row(0, named=True)
            _top_phylop = binned_rates.filter(_want_group, pl.col("phylop_bin") == "5")
            _summary_lines.append(
                f"- **{_label}:** PhyloP {_phylop['auroc']:.3f} → additive {_additive['auroc']:.3f} "
                f"(Δ={_gain['delta_auroc']:.3f}, 95% CI {_gain['ci_low']:.3f}–{_gain['ci_high']:.3f}); "
                f"interaction {_interaction['auroc']:.3f} (Δ={_interaction_gain['delta_auroc']:.3f}, "
                f"95% CI {_interaction_gain['ci_low']:.3f}–{_interaction_gain['ci_high']:.3f}). "
                f"AUPRC is {_phylop['auprc']:.3f}/{_additive['auprc']:.3f}/"
                f"{_interaction['auprc']:.3f}, respectively. Top-PhyloP-quintile prevalence "
                f"changes {_top_phylop['1'][0]:.1%} → {_top_phylop['5'][0]:.1%} across "
                f"Gnocchi quintiles; the Gnocchi-only extreme captures "
                f"{_extreme['high_pip_recall']:.1%} of positives at "
                f"{_extreme['enrichment']:.1f}x baseline enrichment."
            )
        _summary_lines.append("")
    _summary = "\n".join(_summary_lines)

    mo.md(f"""
    #### Interpretation

    {_summary}

    A nonlinear gain without a corresponding additive gain suggests conditional
    structure rather than broad independent signal; estimates for the small
    MultiSuSiE and lncRNA groups should be treated as exploratory.

    The original normalized scatter makes the niche look larger because it plots only
    high-PIP variants; low-PIP variants set the normalization but are not shown.
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ESM-C scores
    """)


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


@app.cell
def _(merged):
    datasets = ["ukbb", "multisusie", "eqtl", "clinvar"]

    for ds in datasets:
        print(ds)
        print(merged.get_column(f"{ds}_label").value_counts())


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


if __name__ == "__main__":
    app.run()
