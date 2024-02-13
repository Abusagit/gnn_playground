from typing import Any, Optional, Sequence

import numpy as np

# import polars as pl
# print(pl.__version__)
import pandas as pd
import yt.wrapper as yt
from scipy.sparse import coo_matrix


OptionalColumns = Optional[Sequence[str]]


HASH_COL = "hash"

TARGET_COL_NAME = "is_fraud"
INTERACTION_COUNT_NAME = "interaction_count"
TRUST_COUNT_NAME = "trust_count"
FRAUD_COUNT_NAME = "fraud_count"

TRUST_INDICATORS_NAMES = ["soft_trust", "strong_trust", "has_paid_services", "was_here"]

ID_COLUMN_NAMES = ["userid", "orgid", "date"]


TARGETS_COLUMNS = ID_COLUMN_NAMES + [TARGET_COL_NAME]
TRUST_COLUMNS = ID_COLUMN_NAMES + TRUST_INDICATORS_NAMES


TRAIN_RATIO = 0.8
VALID_RATIO = 1.0 - TRAIN_RATIO


def read_table(mr_table) -> pd.DataFrame:
    rows = list(yt.read_table(mr_table, format="yson", unordered=False))
    df = pd.DataFrame(rows)
    return df  # .drop(["hash"], axis="columns", errors="ignore")  # get rid of auxillary column


def merge_dataframes_by_hash(df_left, df_right):
    return pd.merge(left=df_left, right=df_right, on=HASH_COL)


def get_interactions_dataframes_for_train_and_test(
    df_pure_data_train: pd.DataFrame,
    df_pure_data_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    sr_date_train = pd.to_datetime(df_pure_data_train["date"], format="%Y%m%d")
    sr_date_test = pd.to_datetime(df_pure_data_test["date"], format="%Y%m%d")

    date_delta_test = sr_date_test.max() - sr_date_test.min()

    extra_indices_test = np.where(sr_date_train >= sr_date_train.min() + date_delta_test)[0]

    df_interactions_train = df_pure_data_train
    df_interactions_test = pd.concat(
        [df_pure_data_test, df_pure_data_train.iloc[extra_indices_test]],
        ignore_index=True,
    )

    initial_test_indices = np.arange(len(df_pure_data_test))

    return df_interactions_train, df_interactions_test, initial_test_indices


def find_constant_columns(df):
    constant_column_mask = (df == df.iloc[0]).all(axis=0).values
    constant_column_indices = np.where(constant_column_mask)[0]
    constant_column_names = df.columns[constant_column_indices]

    return constant_column_names


def filter_constant_columns(df, column_names=None):
    if column_names is None:
        column_names = find_constant_columns(df)
    return df.drop(columns=column_names, errors="ignore", axis="columns")


def find_secretly_boolean_columns(df):
    column_names = []
    for column_name in df.columns:
        if not isinstance(df.dtypes[column_name], object):
            continue
        unique_values = list(df[column_name].unique())
        if set(unique_values) == {"true", "false", ""}:
            column_names.append(column_name)

    return column_names


def transform_secretly_boolean_columns(df, column_names=None):
    transform_mapping = {"true": True, "false": False, "": False}

    if column_names is None:
        column_names = find_secretly_boolean_columns(df)

    for column_name in column_names:
        df[column_name] = df[column_name].map(transform_mapping).astype(bool)

    return df


def find_categorical_columns(df):
    categorical_column_names = list(filter(lambda column: isinstance(df.dtypes[column], object), df.columns))

    return categorical_column_names


def filter_categorical_columns(df, column_names=None, exclude_column_names=None):
    if column_names is None:
        column_names = find_categorical_columns(df)

    if exclude_column_names is None:
        exclude_column_names = []

    column_names_to_filter = [column_name for column_name in column_names if column_name not in exclude_column_names]
    return df.drop(columns=column_names_to_filter, errors="ignore", axis="columns")


def project_data_on_feature(df, column_name_to_project, column_names_to_remove=None, count_column_name="count"):
    df_converted = df.drop(columns=column_names_to_remove, errors="ignore").astype(np.int64)
    groups = df_converted.groupby(by=column_name_to_project, sort=False)

    df_projected = groups.mean()  # average across all user's interactions with organizations and other stuff
    # all values are the same in a row, thus, only need arbitrary column:
    df_projected[count_column_name] = groups[df_projected.columns[-1]].count()

    return df_projected


def filter_and_process_columns_and_project_on_users(
    df: pd.DataFrame,
    feature_names: Sequence[str],
    target_names: Sequence[str],
    trust_names: OptionalColumns = None,
    constant_column_names: OptionalColumns = None,
    secretly_boolean_column_names: OptionalColumns = None,
    categorical_column_names: OptionalColumns = None,
    initial_test_indices: Optional[Sequence[int]] = None,
):
    df_features = df[feature_names]
    df_targets = df[target_names]

    if constant_column_names is None:
        constant_column_names = find_constant_columns(df_features)
        secretly_boolean_column_names = find_secretly_boolean_columns(df_features)
        categorical_column_names = find_categorical_columns(df_features)

    df_features_transformed = filter_constant_columns(df_features, constant_column_names)
    df_features_transformed = transform_secretly_boolean_columns(df_features_transformed, secretly_boolean_column_names)
    df_features_transformed = filter_categorical_columns(df_features_transformed, categorical_column_names)

    df_data = merge_dataframes_by_hash(df_left=df_features_transformed, df_right=df_targets)

    colname_to_project = "userid"
    colnames_to_remove = ID_COLUMN_NAMES + [HASH_COL]

    if trust_names is not None:
        df_trust_test = df[trust_names]
        df_data = merge_dataframes_by_hash(df_left=df_data, df_right=df_trust_test)

        _intermediate_merge = pd.merge(
            left=project_data_on_feature(
                df=df_data,
                column_name_to_project=colname_to_project,
                column_names_to_remove=colnames_to_remove,
                count_column_name=INTERACTION_COUNT_NAME,
            ),
            right=project_data_on_feature(
                df=df_trust_test.iloc[initial_test_indices],
                column_name_to_project=colname_to_project,
                column_names_to_remove=colnames_to_remove,
                count_column_name=TRUST_COUNT_NAME,
            ),
            how="left",
            left_index=True,
            right_index=True,
        )

        df_projected = pd.merge(
            left=_intermediate_merge,
            right=project_data_on_feature(
                df=df_trust_test.iloc[initial_test_indices],
                column_name_to_project=colname_to_project,
                column_names_to_remove=colnames_to_remove,
                count_column_name=FRAUD_COUNT_NAME,
            ),
            how="left",
            left_index=True,
            right_index=True,
        )

    else:
        df_projected = project_data_on_feature(
            df=df_data,
            column_name_to_project=colname_to_project,
            column_names_to_remove=colnames_to_remove,
            count_column_name=INTERACTION_COUNT_NAME,
        )

    return df_projected, constant_column_names, secretly_boolean_column_names, categorical_column_names


def filter_data_by_target(df, target_thresholds):
    df_filtered = df.copy()
    normal_threshold, fraud_threshold = target_thresholds

    df_filtered = df_filtered[(df[TARGET_COL_NAME] >= fraud_threshold) | (df[TARGET_COL_NAME] < normal_threshold)]

    return df_filtered


def construct_adjacency_matrix(
    df_edges, user_mapping, organisation_mapping, user_column_name="userid", organisation_column_name="orgid"
):
    num_user_ids = len(user_mapping)
    num_organisation_ids = len(organisation_mapping)

    user_encoder = np.vectorize(lambda id: user_mapping[id])
    organisation_encoder = np.vectorize(lambda id: organisation_mapping[id])

    df_edges_encoded = df_edges.copy()
    df_edges_encoded[user_column_name] = df_edges_encoded[user_column_name].apply(user_encoder)
    df_edges_encoded[organisation_column_name] = df_edges_encoded[organisation_column_name].apply(organisation_encoder)

    incidence_data = np.ones(shape=(len(df_edges_encoded),), dtype=np.int64)

    incidence_row_coordinates = df_edges_encoded[user_column_name].values
    incidence_col_coordinates = df_edges_encoded[organisation_column_name].values

    incidence_matrix_train = coo_matrix(
        (incidence_data, (incidence_row_coordinates, incidence_col_coordinates)),
        shape=(num_user_ids, num_organisation_ids),
    ).tocsr()
    adjacency_matrix_train = incidence_matrix_train @ incidence_matrix_train.T

    return adjacency_matrix_train


def separate_features_for_graph_and_tests(
    df_before_projection: pd.DataFrame,
    df_projected: pd.DataFrame,
    column_names_to_remove: set[str],
    test: bool = False,
    target_thresholds: tuple[float, float] = (0.1, 0.5),
):
    def separate_counts(count_name):
        counts = df_projected[count_name].values
        return counts

    def separate_targets(target_thresholds=[0.1, 0.5]):
        _normal_threshold, fraud_threshold = target_thresholds
        targets = (df_projected[TARGET_COL_NAME].values > fraud_threshold).astype(np.int64)
        return targets

    df_filtered = filter_data_by_target(df_projected, target_thresholds=target_thresholds)

    feature_names = [column_name for column_name in df_filtered.columns if column_name not in column_names_to_remove]

    targets: np.ndarray = separate_targets()
    interactions_counts: np.ndarray = separate_counts(INTERACTION_COUNT_NAME)
    features: np.ndarray = df_projected[feature_names].values
    indices: np.ndarray = df_projected.index.values

    if test:
        trust_counts: np.ndarray = separate_counts(TRUST_COUNT_NAME)
        fraud_counts: np.ndarray = separate_counts(FRAUD_COUNT_NAME)
        trust_indicators: np.ndarray = df_projected[TRUST_INDICATORS_NAMES].values

        update_dict = dict(
            trust_counts=trust_counts,
            fraud_counts=fraud_counts,
            trust_indicators=trust_indicators,
        )
    else:
        update_dict = {}

    df_edges = df_before_projection[ID_COLUMN_NAMES]
    df_edges = df_edges[df_edges["userid"].isin(set(indices))]  # is user had connection with this triplet
    user_mapping = {id: id_encoding for id_encoding, id in enumerate(indices)}
    organisation_mapping = {id: id_encoding for id_encoding, id in enumerate(df_edges["orgid"].unique())}

    adjacency_matrix = construct_adjacency_matrix(
        df_edges=df_edges, user_mapping=user_mapping, organisation_mapping=organisation_mapping
    )

    return dict(
        features=features,
        targets=targets,
        interactions_counts=interactions_counts,
        indices=indices,
        adjacency_matrix=adjacency_matrix,
        user_ids=df_before_projection["userid"].values,
        # df_edges=df_edges,
        # user_mapping=user_mapping,
        # organisation_mapping=organisation_mapping,
    ).update(update_dict)


def prepare_split_indices(num_train_samples, train_ratio):
    indices = np.arange(num_train_samples)
    indices_permuted = np.random.permutation(indices)

    train_size = int(train_ratio * num_train_samples)

    train_indices = indices_permuted[:train_size]
    val_indices = indices_permuted[train_size:]

    return train_indices, val_indices


def convert_split_indices_to_mask(num_samples, split_indices):
    split_mask = np.zeros(shape=(num_samples,), dtype=bool)
    split_mask[split_indices] = True

    return split_mask


def main(
    in1,
    in2,
    in3,
    mr_tables,
    token1=None,
    token2=None,
    param1=None,
    param2=None,
    html_file=None,
):
    print("in1:", in1)
    print("in2:", in2)
    print("in3:", in3)
    print("mr_tables:", mr_tables)

    # read all data from first mr table
    yt.config.config["token"] = token1
    yt.config.config["proxy"]["url"] = mr_tables[0]["cluster"]

    PARAMS_OUTPUT = {}

    if len(mr_tables) == 2:  # TRAINING PHASE
        PARAMS_OUTPUT["mode"] = "train"

        train_table = mr_tables[0]["table"]
        test_table = mr_tables[1]["table"]

        train_df: pd.DataFrame = read_table(train_table)
        test_df: pd.DataFrame = read_table(test_table)

        (
            df_interactions_train,
            df_interactions_test,
            initial_test_indices,
        ) = get_interactions_dataframes_for_train_and_test(train_df, test_df)

        pure_target_names: list[str] = TARGETS_COLUMNS + [HASH_COL]
        pure_trust_names: list[str] = TRUST_COLUMNS + [HASH_COL]
        pure_feature_names: list[str] = (
            list(set(list(train_table.columns)) - set(pure_target_names) - set(pure_trust_names))
            + ID_COLUMN_NAMES
            + [HASH_COL]
        )

        (
            df_train_projected,
            constant_column_names,
            secretly_boolean_column_names,
            categorical_column_names,
        ) = filter_and_process_columns_and_project_on_users(
            df=df_interactions_train,
            initial_test_indices=initial_test_indices,
            target_names=pure_target_names,
            feature_names=pure_feature_names,
        )

        df_test_projected, _, _, _ = filter_and_process_columns_and_project_on_users(
            df=df_interactions_test,
            initial_test_indices=initial_test_indices,
            target_names=pure_target_names,
            feature_names=pure_feature_names,
            trust_names=pure_trust_names,
        )

        column_names_to_remove_train = [INTERACTION_COUNT_NAME, TARGET_COL_NAME]

        column_names_to_remove_test = [
            INTERACTION_COUNT_NAME,
            TRUST_COUNT_NAME,
            FRAUD_COUNT_NAME,
            TARGET_COL_NAME,
        ] + TRUST_INDICATORS_NAMES

        train_data: dict[str, Any] = separate_features_for_graph_and_tests(
            df_projected=df_train_projected,
            column_names_to_remove=column_names_to_remove_train,
            df_before_projection=df_interactions_train,
        )

        test_data: dict[str, Any] = separate_features_for_graph_and_tests(
            df_projected=df_test_projected,
            column_names_to_remove=column_names_to_remove_test,
            test=False,
            df_before_projection=df_interactions_test,
        )

        num_samples_train = len(train_data["features"])
        indices_for_train, indices_for_validation = prepare_split_indices(num_samples_train, TRAIN_RATIO)

        train_mask = convert_split_indices_to_mask(num_samples=num_samples_train, split_indices=indices_for_train)
        val_mask = convert_split_indices_to_mask(num_samples=num_samples_train, split_indices=indices_for_validation)

        # for test, we take only those who has trust indicators
        num_samples_test = len(test_data["features"])
        indices_test = np.where(np.all(~np.isnan(test_data["trust_indicators"]), axis=1))[0]

        test_mask = convert_split_indices_to_mask(num_samples=num_samples_test, split_indices=indices_test)

        masks = dict(
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )

        PARAMS_OUTPUT["train_data"] = train_data
        PARAMS_OUTPUT["test_data"] = test_data
        PARAMS_OUTPUT["masks"] = masks

    else:  # INFERENCE PHASE
        PARAMS_OUTPUT["mode"] = "test"

        test_table = mr_tables[0]["table"]

        test_df: pd.DataFrame = read_table(test_table)

        raise NotImplementedError("Work in progress")

    return PARAMS_OUTPUT
