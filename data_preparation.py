from typing import Any, Optional, Sequence, MutableSet, List, Dict, Tuple

import numpy as np

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


PRESERVE_COLUMNS = TRUST_INDICATORS_NAMES + [HASH_COL] + TARGETS_COLUMNS


FEATURE_TO_DTYPE = {
    "date": object,
    "orgid": np.int64,
    "userid": np.int64,
    "has_reaction": bool,
    "has_review": bool,
    "rating": np.int32,
    "reaction": object,
    "time": np.int32,
    "user_1": np.int32,
    "user_4": np.int32,
    "user_5": object,
    "user_6": object,
    "user_8": object,
    "user_9": np.int32,
    "user_10": np.int32,
    "user_12": np.int32,
    "user_13": object,
    "user_15": np.int32,
    "user_16": bool,
    "user_17": np.int32,
    "user_18": np.int32,
    "user_19": np.int32,
    "user_21": object,
    "user_22": object,
    "user_23": object,
    "user_24": object,
    "user_25": object,
    "user_26": object,
    "user_27": object,
    "user_28": object,
    "user_29": object,
    "user_30": object,
    "user_31": object,
    "user_32": object,
    "user_33": object,
    "user_34": object,
    "user_35": object,
    "user_36": object,
    "user_37": object,
    "user_38": object,
    "user_39": object,
    "user_40": object,
    "user_41": object,
    "user_42": object,
    "user_43": object,
    "user_44": object,
    "user_45": object,
    "user_46": object,
    "user_47": object,
    "user_48": object,
    "user_49": object,
    "user_50": object,
    "user_51": object,
    "user_52": object,
    "user_53": object,
    "user_54": object,
    "user_55": object,
    "user_56": object,
    "user_57": np.float32,
    "user_58": np.float32,
    "user_59": np.float32,
    "user_61": np.int32,
    "user_62": np.int32,
    "user_63": np.int32,
    "user_64": np.int32,
    "user_65": np.int32,
    "user_66": np.float32,
    "user_68": np.int32,
    "user_69": np.int32,
    "user_70": object,
    "user_71": bool,
    "user_72": np.int32,
    "user_73": bool,
    "user_74": bool,
    "user_75": np.int32,
    "user_76": bool,
    "user_77": bool,
    "user_78": bool,
    "user_79": bool,
    "user_80": bool,
    "user_81": bool,
    "user_82": np.int32,
    "user_83": np.int32,
    "user_84": bool,
    "user_85": np.int32,
    "user_86": np.int32,
    "user_87": object,
    "user_88": bool,
    "user_89": np.float32,
    "user_90": np.float32,
    "user_91": np.int32,
    "user_92": np.int32,
    "user_93": np.int32,
    "user_94": np.float32,
    "user_95": np.float32,
    "user_96": np.int32,
    "user_97": np.int32,
    "user_98": np.int32,
    "user_99": np.int32,
    "user_100": np.int32,
    "user_101": np.int32,
    "user_102": np.int32,
    "user_103": np.int32,
    "user_104": np.int32,
    "user_105": np.int32,
    "user_106": np.int32,
    "user_107": np.int32,
    "user_108": object,
    "user_109": bool,
    "user_110": bool,
    "user_111": np.int32,
    "user_112": np.int32,
    "user_113": np.int32,
    "user_114": np.int32,
    "user_115": np.int32,
    "user_116": bool,
    "user_117": np.int32,
    "user_118": np.int32,
    "user_119": np.int32,
    "user_120": np.int32,
    "user_121": np.int32,
    "user_122": np.int32,
    "user_123": bool,
    "user_124": np.int32,
    "user_125": np.int32,
    "user_126": np.int32,
    "user_127": np.int32,
    "user_128": np.int32,
    "user_129": np.int32,
    "user_130": np.int32,
    "user_131": np.int32,
    "user_132": np.int32,
    "user_133": np.int32,
    "user_134": np.int32,
    "user_135": np.int32,
    "user_136": np.int32,
    "user_137": np.int32,
    "user_138": np.int32,
    "user_139": np.int32,
    "user_140": np.int32,
    "user_141": bool,
    "user_142": np.float32,
    "user_143": np.int32,
    "user_144": np.int32,
    "user_146": bool,
    "user_147": bool,
    "user_148": np.int32,
    "user_149": np.int32,
    "user_150": bool,
    "user_151": object,
    "user_152": np.float32,
    "user_153": np.int32,
    "user_154": np.int32,
    "user_155": np.int32,
    "user_156": np.int32,
    "user_157": np.float32,
    "user_158": np.int32,
    "user_159": np.int32,
    "user_160": np.int32,
    "user_161": np.int32,
    "org_2": np.float32,
    "org_3": np.float32,
    "org_4": np.int32,
    "org_5": np.int32,
    "org_8": np.float32,
    "org_9": object,
    "org_10": np.int32,
    "org_11": np.int32,
    "org_12": np.int32,
    "org_13": np.float32,
    "org_14": np.int32,
    "org_15": np.float32,
    "org_16": np.int32,
    "org_17": np.int32,
    "org_18": np.int32,
    "org_19": np.int32,
    "org_20": np.int32,
    "org_21": np.float32,
    "org_22": np.float32,
    "org_23": np.int32,
    "org_24": np.float32,
    "org_25": np.float32,
    "org_26": np.float32,
    "org_27": np.int32,
    "org_28": np.int32,
    "org_29": np.int32,
    "org_30": np.float32,
    "org_31": np.float32,
    "org_32": np.float32,
    "org_33": np.float32,
    "org_34": np.int32,
    "org_35": np.int32,
    "org_36": np.float32,
    "org_37": np.float32,
    "org_38": np.float32,
    "org_39": np.float32,
    "org_40": np.float32,
    "org_41": np.float32,
    "org_42": np.float32,
    "org_43": np.int32,
    "org_44": np.int32,
    "org_45": np.int32,
    "org_46": np.int32,
    "org_47": np.int32,
    "org_48": np.int32,
    "org_49": np.int32,
    "org_50": np.int32,
    "org_51": np.int32,
    "org_52": np.int32,
    "org_53": np.int32,
    "org_54": np.int32,
    "org_55": np.int32,
    "org_56": np.int32,
    "org_57": np.int32,
    "org_58": np.float32,
    "org_59": np.int32,
    "org_60": np.float32,
    "org_61": np.float32,
    "org_62": np.float32,
    "org_63": np.float32,
    "org_64": np.float32,
    "org_65": np.float32,
    "org_66": np.float32,
    "org_67": np.float32,
    "org_68": np.float32,
    "org_69": np.float32,
    "org_70": np.float32,
    "org_71": np.int32,
    "org_72": np.float32,
    "org_73": np.float32,
    "org_74": np.float32,
    "org_75": np.float32,
    "org_76": np.float32,
    "org_77": np.float32,
    "org_78": np.float32,
    "org_79": np.float32,
    "org_80": np.float32,
    "org_81": np.float32,
    "org_82": np.float32,
    "org_84": np.int32,
    "org_85": np.int32,
    "org_86": np.int32,
    "org_87": np.int32,
    "org_88": np.int32,
    "org_89": np.int32,
    "org_90": np.int32,
    "org_91": bool,
    "org_93": object,
    "org_94": np.float32,
    "org_95": np.int32,
    "org_98": np.int32,
    "org_99": np.float32,
    "org_100": np.int32,
    "org_101": object,
    "org_102": object,
    "org_103": np.int32,
    "org_104": np.int32,
    "org_105": np.float32,
    "org_106": np.float32,
    "org_107": np.int32,
    "org_108": np.int32,
    "org_109": np.int32,
    "org_110": np.int32,
    "org_111": np.int32,
    "org_112": np.float32,
    "org_113": np.int32,
    "org_114": np.int32,
    "org_115": np.int32,
    "org_117": np.int32,
    "org_118": bool,
    "org_119": bool,
    "review_2": np.float32,
    "review_3": bool,
    "review_4": bool,
    "review_5": np.int32,
    "review_7": np.int32,
    "review_8": np.int32,
    "review_9": np.int32,
    "review_11": np.int32,
    "review_12": np.int32,
    "review_13": np.float32,
    "review_14": np.float32,
    "review_15": np.float32,
    "review_16": np.int32,
    "review_17": np.float32,
    "review_18": object,
    "review_20": np.int32,
    "review_21": np.int32,
    "review_22": np.int32,
    "review_23": np.int32,
    "review_24": np.int32,
    "review_26": object,
    "review_28": object,
    "review_29": np.int32,
    "review_30": np.float32,
    "review_31": bool,
    "review_32": np.float32,
    "review_36": bool,
    "review_37": bool,
    "soft_trust": bool,
    "strong_trust": bool,
    "was_here": bool,
    "has_paid_services": bool,
    "is_fraud": bool,
    "hash": object,
}


def read_table(mr_table, columns_presented_in_train: OptionalColumns = None) -> pd.DataFrame:
    rows = list(yt.read_table(mr_table, format="yson", unordered=False))
    df = pd.DataFrame(rows)

    feature_dtype_to_value = {
        object: "",
        bool: False,
        np.int32: 0,
        np.float32: 0.0,
        np.int64: 0,
        np.uint64: 0,
    }

    columns_types = {}

    for col in df.columns:
        col_desired_dtype = FEATURE_TO_DTYPE[col]
        nan_val_replacer = feature_dtype_to_value[col_desired_dtype]
        print(col, nan_val_replacer, col_desired_dtype)

        df.loc[df[col].isna(), col] = nan_val_replacer

        columns_types[col] = col_desired_dtype
        # df[col] = df[col].astype(col_desired_dtype)

    met_columns = set(columns_types.keys())

    if len(set(ID_COLUMN_NAMES) & met_columns) != 3:
        raise KeyError(f"Some of the obligatory columns ({ID_COLUMN_NAMES}) are missing!")

    if columns_presented_in_train is not None:
        columns_unpresented_in_df = list(set(columns_presented_in_train) - met_columns)

        print(
            f"Columns which aren't presented in the dataframe but the model was trained using them: {columns_unpresented_in_df}"
        )

        for col in columns_unpresented_in_df:
            col_desired_dtype = FEATURE_TO_DTYPE[col]
            nan_val_replacer = feature_dtype_to_value[col_desired_dtype]
            print(col, nan_val_replacer, col_desired_dtype)
            df[col] = nan_val_replacer

        print("Added these columns as empties (nans) and processed")

    df = df.astype(columns_types)

    # additional check
    float_cols = df.select_dtypes(include=["floating"]).columns
    str_cols = df.select_dtypes(include=["object"]).columns
    int_cols = df.select_dtypes(include=["integer"]).columns
    bool_cols = df.select_dtypes(include=["bool"]).columns

    df.loc[:, float_cols] = df.loc[:, float_cols].fillna(0.0)
    df.loc[:, str_cols] = df.loc[:, str_cols].fillna("")
    df.loc[:, int_cols] = df.loc[:, int_cols].fillna(0)
    df.loc[:, bool_cols] = df.loc[:, bool_cols].fillna(False)

    assert df.isna().sum().sum() == 0

    return df  # .drop(["hash"], axis="columns", errors="ignore")  # get rid of auxillary column


def merge_dataframes(df_left, df_right, by):
    return pd.merge(left=df_left, right=df_right, on=by)


def get_interactions_dataframes_for_train_and_test(
    df_pure_data_train: pd.DataFrame,
    df_pure_data_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
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
    constant_column_names = list(df.columns[constant_column_indices].values)

    return constant_column_names


def filter_constant_columns(df, column_names=None):
    if column_names is None:
        column_names = find_constant_columns(df)
    return df.drop(columns=column_names, errors="ignore", axis="columns")


def find_secretly_boolean_columns(df) -> Tuple[str, Any]:
    column_names_and_true_vals = []
    for column_name in df.columns:
        if not isinstance(df.dtypes[column_name], object):
            continue
        unique_values = list(df[column_name].unique())
        if set(unique_values) == {"true", "false", ""}:
            column_names_and_true_vals.append((column_name, "true"))
        elif len(unique_values) == 2:
            column_names_and_true_vals.append((column_name, unique_values[0]))

    return column_names_and_true_vals


def transform_secretly_boolean_columns(df, column_names_and_true_values=None):
    # transform_mapping = {"true": True, "false": False, "": False}

    if column_names_and_true_values is None:
        column_names_and_true_values = find_secretly_boolean_columns(df)

    for column_name, true_val in column_names_and_true_values:
        transform_mapping = {true_val: True}
        df[column_name] = df[column_name].map(transform_mapping).astype(bool)

    return df


def find_categorical_columns(df):
    categorical_column_names = list(df.select_dtypes(include=["object"]).columns.values)

    return categorical_column_names


def filter_categorical_columns(df, column_names=None, exclude_column_names=None):
    if column_names is None:
        column_names = find_categorical_columns(df)

    if exclude_column_names is None:
        exclude_column_names = []

    column_names_to_filter = [column_name for column_name in column_names if column_name not in exclude_column_names]
    return df.drop(columns=column_names_to_filter, errors="ignore", axis="columns")


def project_data_on_feature(df, column_name_to_project, column_names_to_remove=None, count_column_name="count"):
    df_converted = df.drop(columns=column_names_to_remove, errors="ignore")
    print(f"Columns after projecting on {column_name_to_project}: {df_converted.columns}")
    groups = df_converted.groupby(by=column_name_to_project, sort=False)

    df_projected = groups.mean()  # average across all users interactions with organizations and other stuff
    # all values are the same in a row, thus, only need arbitrary column:
    df_projected[count_column_name] = groups[df_projected.columns[-1]].count()

    return df_projected.copy()


def filter_and_process_columns_and_project_on_users(
    df: pd.DataFrame,
    feature_names: Sequence[str],
    target_names: Sequence[str],
    trust_names: OptionalColumns = None,
    constant_column_names: Optional[Sequence[Tuple[str, Any]]] = None,
    secretly_boolean_column_names_and_true_values: OptionalColumns = None,
    categorical_column_names: OptionalColumns = None,
    initial_test_indices: Optional[Sequence[int]] = None,
):
    df_features = df[feature_names]
    df_targets = df[target_names]

    if constant_column_names is None:
        constant_column_names = find_constant_columns(df_features)
    df_features_transformed = filter_constant_columns(df_features, column_names=constant_column_names)

    if secretly_boolean_column_names_and_true_values is None:
        secretly_boolean_column_names_and_true_values = find_secretly_boolean_columns(df_features_transformed)
    df_features_transformed = transform_secretly_boolean_columns(
        df_features_transformed, column_names_and_true_values=secretly_boolean_column_names_and_true_values
    )

    if categorical_column_names is None:
        categorical_column_names = find_categorical_columns(df_features_transformed)

    df_features_transformed = filter_categorical_columns(
        df_features_transformed, categorical_column_names, exclude_column_names=["date"]
    )

    print(f"Constant columns are: {constant_column_names}")
    print(f"Secretly boolean columns are: {secretly_boolean_column_names_and_true_values}")
    print(f"Categorical columns are: {categorical_column_names} (DATE IS EXCLUDED FROM REMOVING)")

    print(f"\n\n\nFeatures left after transformation:{df_features_transformed.columns=}")

    colname_to_project = "userid"
    colnames_to_remove = ["orgid", "date", "time", HASH_COL]  # ID_COLUMN_NAMES + [HASH_COL]

    # print(f"\n\n\n{df_data.columns=}\n\n\n")
    if trust_names is not None:  # TEST MODE
        # need additional  handling of possible more categorical columns in test data:

        df_trust_test = df[trust_names]
        # df_data = merge_dataframes(df_left=df_data, df_right=df_trust_test, by=ID_COLUMN_NAMES)

        # breakpoint()
        _intermediate_merge = pd.merge(
            left=project_data_on_feature(
                df=df_features_transformed,
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

        # breakpoint()

        df_projected = pd.merge(
            left=_intermediate_merge,
            right=project_data_on_feature(
                df=df_targets.iloc[initial_test_indices],
                column_name_to_project=colname_to_project,
                column_names_to_remove=colnames_to_remove,
                count_column_name=FRAUD_COUNT_NAME,
            ),
            how="left",
            left_index=True,
            right_index=True,
        )

    else:
        df_data = merge_dataframes(df_left=df_features_transformed, df_right=df_targets, by=ID_COLUMN_NAMES)
        df_projected = project_data_on_feature(
            df=df_data,
            column_name_to_project=colname_to_project,
            column_names_to_remove=colnames_to_remove,
            count_column_name=INTERACTION_COUNT_NAME,
        )
    # breakpoint()
    return df_projected, constant_column_names, secretly_boolean_column_names_and_true_values, categorical_column_names


def filter_data_by_target(df, target_thresholds):
    df_filtered = df.copy()
    normal_threshold, fraud_threshold = target_thresholds

    df_filtered = df_filtered[(df[TARGET_COL_NAME] >= fraud_threshold) | (df[TARGET_COL_NAME] < normal_threshold)]

    return df_filtered


def get_everything_for_adjacency_matrix(
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

    row_coords, col_coords = adjacency_matrix_train.nonzero()

    return dict(row_coords=row_coords, col_coords=col_coords)


def separate_features_for_graph_and_tests(
    df_before_projection: pd.DataFrame,
    df_projected: pd.DataFrame,
    column_names_to_remove: MutableSet[str],
    test: bool = False,
    target_thresholds: Optional[Tuple[float, float]] = (0.1, 0.5),
):
    def separate_counts(df, count_name):
        counts = df[count_name].values
        return counts

    def separate_targets(df, target_thresholds=[0.1, 0.5]):
        _normal_threshold, fraud_threshold = target_thresholds
        targets = (df[TARGET_COL_NAME].values > fraud_threshold).astype(np.int64)
        return targets

    if target_thresholds is not None:
        df_filtered = filter_data_by_target(df_projected, target_thresholds=target_thresholds)
    else:
        df_filtered = df_projected

    feature_names = [column_name for column_name in df_filtered.columns if column_name not in column_names_to_remove]

    targets: np.ndarray = separate_targets(df_filtered)
    interactions_counts: np.ndarray = separate_counts(df_filtered, INTERACTION_COUNT_NAME)
    features: np.ndarray = df_filtered[feature_names].values
    indices: np.ndarray = df_filtered.index.values

    if test:
        trust_counts: np.ndarray = separate_counts(df_filtered, TRUST_COUNT_NAME)
        fraud_counts: np.ndarray = separate_counts(df_filtered, FRAUD_COUNT_NAME)
        trust_indicators: np.ndarray = df_filtered[TRUST_INDICATORS_NAMES].values.astype(float)

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

    adjacency_matrix_rows_and_cols = get_everything_for_adjacency_matrix(
        df_edges=df_edges, user_mapping=user_mapping, organisation_mapping=organisation_mapping
    )

    user_ids = df_filtered.index.values

    assert len(features) == len(targets)
    assert len(user_ids) == len(features)

    data_dict = dict(
        features=features,
        targets=targets,
        interactions_counts=interactions_counts,
        indices=indices,
        adjacency_matrix_tools=adjacency_matrix_rows_and_cols,
        user_ids=user_ids,
        feature_names=feature_names,
        **update_dict,
        # df_edges=df_edges,
        # user_mapping=user_mapping,
        # organisation_mapping=organisation_mapping,
    )

    return data_dict


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


def main_prepare_mr_tables(
    mr_tables: List[Dict[str, str]],
    token=None,
    columns_metadata: Optional[Dict[str, List[str]]] = None,
):
    print("mr_tables:", mr_tables)

    # read all data from first mr table
    yt.config.config["token"] = token
    yt.config.config["proxy"]["url"] = mr_tables[0]["cluster"]

    PARAMS_OUTPUT = {}

    if len(mr_tables) == 2:  # TRAINING PHASE
        PARAMS_OUTPUT["mode"] = "train"

        train_table = mr_tables[0]["table"]
        test_table = mr_tables[1]["table"]
        train_df: pd.DataFrame = read_table(train_table)
        print("TESTING")
        test_df: pd.DataFrame = read_table(test_table)
        # breakpoint()

        # train_df.to_csv("train_df.csv")
        # test_df.to_csv("test_df.csv")

        # train_df = pd.read_csv("train_df.csv", index_col=0)
        # test_df = pd.read_csv("test_df.csv", index_col=0)

        print(train_df.dtypes)
        print("Read train and test tables")

        print(train_df.head(), f"{train_df.shape=}, {train_df.columns=}")

        print(test_df.head(), f"{test_df.shape=}, {test_df.columns=}")

        (
            df_interactions_train,
            df_interactions_test,
            initial_test_indices,
        ) = get_interactions_dataframes_for_train_and_test(train_df, test_df)

        print("\n\n====== Estimated interactions between train and test ======\n\n")
        print(df_interactions_train.head(), f"{df_interactions_train.shape=}, {df_interactions_train.columns}")
        print(df_interactions_test.head(), f"{df_interactions_test.shape=}, {df_interactions_test.columns}")

        pure_target_names: list[str] = TARGETS_COLUMNS  # + [HASH_COL]
        pure_trust_names: list[str] = TRUST_COLUMNS  # + [HASH_COL]
        pure_feature_names: list[str] = (
            ##[HASH_COL]
            ID_COLUMN_NAMES + list(set(list(train_df.columns)) - set(pure_target_names) - set(pure_trust_names))
        )

        print(f"{pure_target_names=}\n{pure_trust_names=}\n{pure_feature_names=}")

        (
            df_train_projected,
            constant_column_names,
            secretly_boolean_column_names_and_true_values,
            categorical_column_names,
        ) = filter_and_process_columns_and_project_on_users(
            df=df_interactions_train,
            target_names=pure_target_names,
            feature_names=pure_feature_names,
        )

        df_test_projected, _, _, _ = filter_and_process_columns_and_project_on_users(
            df=df_interactions_test,
            initial_test_indices=initial_test_indices,
            target_names=pure_target_names,
            feature_names=pure_feature_names,
            trust_names=pure_trust_names,
            constant_column_names=constant_column_names,
            secretly_boolean_column_names_and_true_values=secretly_boolean_column_names_and_true_values,
            categorical_column_names=categorical_column_names,
        )

        df_train_projected = df_train_projected.copy()
        df_test_projected = df_test_projected.copy()

        print("\n\n===== Projected data on users =====\n\n")
        # breakpoint()

        _additional_cols_in_test = set(df_test_projected.columns) - set(df_train_projected.columns)
        print(
            f"{df_train_projected.shape=}, {df_test_projected.shape=}\ndf_train_projected has {len(_additional_cols_in_test)} columns: {_additional_cols_in_test}",
            end="\n\n",
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
            test=True,
            df_before_projection=df_interactions_test,
        )
        # breakpoint()

        print("Separated features for graph building")

        features_in_train_df = (
            list(
                set(df_train_projected.columns.values)
                - set(ID_COLUMN_NAMES)
                - set([TARGET_COL_NAME])
                - set(TRUST_COLUMNS)
                - set([INTERACTION_COUNT_NAME])
            )
            + ID_COLUMN_NAMES
        )

        _columns_metadata = dict(
            features_in_train_df=features_in_train_df,
            constant_column_names=constant_column_names,
            secretly_boolean_column_names_and_true_values=secretly_boolean_column_names_and_true_values,
            categorical_column_names=categorical_column_names,
        )

        num_samples_train = len(train_data["features"])
        indices_for_train, indices_for_validation = prepare_split_indices(num_samples_train, TRAIN_RATIO)

        train_mask = convert_split_indices_to_mask(num_samples=num_samples_train, split_indices=indices_for_train)
        val_mask = convert_split_indices_to_mask(num_samples=num_samples_train, split_indices=indices_for_validation)

        # for test, we take only those who has trust indicators
        num_samples_test = len(test_data["features"])
        # breakpoint()
        indices_test = np.where(np.all(~np.isnan(test_data["trust_indicators"]), axis=1))[0]

        test_mask = convert_split_indices_to_mask(num_samples=num_samples_test, split_indices=indices_test)

        masks = dict(
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )

        print("Created train, val, test masks")

        PARAMS_OUTPUT["train_data"] = train_data
        PARAMS_OUTPUT["test_data"] = test_data
        PARAMS_OUTPUT["masks"] = masks
        PARAMS_OUTPUT["columns_metadata"] = _columns_metadata

    else:  # INFERENCE PHASE
        PARAMS_OUTPUT["mode"] = "test"
        test_table = mr_tables[0]["table"]

        features_in_train_df = columns_metadata["features_in_train_df"]
        constant_column_names = columns_metadata["constant_column_names"]
        secretly_boolean_column_names_and_true_values = columns_metadata[
            "secretly_boolean_column_names_and_true_values"
        ]
        categorical_column_names = columns_metadata["categorical_column_names"]

        test_df: pd.DataFrame = read_table(test_table, columns_presented_in_train=features_in_train_df)
        test_df[TARGET_COL_NAME] = False  # NOTE this is placeholder

        test_df = test_df.copy()

        df_test_projected, _, _, _ = filter_and_process_columns_and_project_on_users(
            df=test_df,
            feature_names=features_in_train_df,
            target_names=TARGETS_COLUMNS,
            constant_column_names=constant_column_names,
            secretly_boolean_column_names_and_true_values=secretly_boolean_column_names_and_true_values,
            categorical_column_names=categorical_column_names,
        )

        print(f"Projected inference data on users, {df_test_projected.shape=}")

        df_test_projected = df_test_projected.copy()

        column_names_to_remove_test = [
            INTERACTION_COUNT_NAME,
            TRUST_COUNT_NAME,
            FRAUD_COUNT_NAME,
            TARGET_COL_NAME,
        ] + TRUST_INDICATORS_NAMES

        test_data: dict[str, Any] = separate_features_for_graph_and_tests(
            df_projected=df_test_projected,
            column_names_to_remove=column_names_to_remove_test,
            test=False,
            df_before_projection=test_df,
            target_thresholds=None,
        )

        masks = dict(
            test_mask=np.ones(len(df_test_projected)),
        )

        PARAMS_OUTPUT["test_data"] = test_data
        PARAMS_OUTPUT["masks"] = masks
        PARAMS_OUTPUT["columns_metadata"] = columns_metadata
        
        print("Created inference graph")

    return PARAMS_OUTPUT


if __name__ == "__main__":
    import os

    output = main_prepare_mr_tables(
        mr_tables=[
            {
                "cluster": "hahn",
                "table": "//home/yr/fvelikon/nirvana/c8e0052c-2996-4960-944f-b31fc5a8ca80/output1__gamlfOCgRqyq94sYvTIYtg",
            },
            {
                "cluster": "hahn",
                "table": "//home/yr/fvelikon/nirvana/5c3d379a-9604-4b97-a1d0-719238f7076f/output1__aoSo9RpQSQqumA-qBAjW5w",
            },
        ],
        token=os.environ.get("YT_TOKEN"),
    )
