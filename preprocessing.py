import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

def data_preprocessing():

    # #checking for null values
    # total_rows = len(data)
    # print(data.info())

    features = np.genfromtxt('./fetal_health.csv', missing_values=0, skip_header=1, delimiter=',', dtype=float)

    baseline_value = features[:, 0].reshape(-1, 1)
    accelerations = features[:, 1].reshape(-1, 1)
    fetal_movement = features[:, 2].reshape(-1, 1)
    uterine_contractions = features[:, 3].reshape(-1, 1)
    light_decelerations = features[:, 4].reshape(-1, 1)
    severe_decelerations = features[:, 5].reshape(-1, 1)
    prolongued_decelerations = features[:, 6].reshape(-1, 1)
    abnormal_short_term_variability = features[:, 7].reshape(-1, 1)
    mean_value_of_short_term_variability = features[:, 8].reshape(-1, 1)
    percentage_of_time_with_abnormal_long_term_variability = features[:, 9].reshape(-1, 1)
    mean_value_of_long_term_variability = features[:, 10].reshape(-1, 1)
    histogram_width = features[:, 11].reshape(-1, 1)
    histogram_min = features[:, 12].reshape(-1, 1)
    histogram_max = features[:, 13].reshape(-1, 1)
    histogram_number_of_peaks = features[:, 14].reshape(-1, 1)
    histogram_number_of_zeroes = features[:, 15].reshape(-1, 1)
    histogram_mode = features[:, 16].reshape(-1, 1)
    histogram_mean = features[:, 17].reshape(-1, 1)
    histogram_median = features[:, 18].reshape(-1, 1)
    histogram_variance = features[:, 19].reshape(-1, 1)
    histogram_tendency = features[:, 20].reshape(-1, 1)

    baseline_value_bins = 4
    accelerations_bins = 3
    fetal_movement_bins = 2
    uterine_contractions_bins = 2
    light_decelerations_bins = 3
    severe_decelerations_bins = 2
    prolongued_decelerations_bins = 4 # 4/5
    abnormal_short_term_variability_bins = 4 # 3/4
    mean_value_of_short_term_variability_bins =  4
    percentage_of_time_with_abnormal_long_term_variability_bins = 3 # 3/4
    mean_value_of_long_term_variability_bins = 4
    histogram_width_bins = 3
    histogram_min_bins = 4
    histogram_max_bins = 4
    histogram_number_of_peaks_bins = 3
    histogram_number_of_zeroes_bins = 4
    histogram_mode_bins = 4
    histogram_median_bins = 4
    histogram_variance_bins = 3
    histogram_tendency_bins = 3



    model = KBinsDiscretizer(n_bins=baseline_value_bins, encode='ordinal', strategy='kmeans')
    baseline_value = model.fit_transform(baseline_value)

    model = KBinsDiscretizer(n_bins=accelerations_bins, encode='ordinal', strategy='kmeans')
    accelerations = model.fit_transform(accelerations)

    model = KBinsDiscretizer(n_bins=fetal_movement_bins, encode='ordinal', strategy='kmeans')
    fetal_movement = model.fit_transform(fetal_movement)

    model = KBinsDiscretizer(n_bins=uterine_contractions_bins, encode='ordinal', strategy='kmeans')
    uterine_contractions = model.fit_transform(uterine_contractions)

    model = KBinsDiscretizer(n_bins=light_decelerations_bins, encode='ordinal', strategy='kmeans')
    light_decelerations = model.fit_transform(light_decelerations)

    model = KBinsDiscretizer(n_bins=severe_decelerations_bins, encode='ordinal', strategy='kmeans')
    severe_decelerations = model.fit_transform(severe_decelerations)

    model = KBinsDiscretizer(n_bins=prolongued_decelerations_bins, encode='ordinal', strategy='kmeans')
    prolongued_decelerations = model.fit_transform(prolongued_decelerations)

    model = KBinsDiscretizer(n_bins=abnormal_short_term_variability_bins, encode='ordinal', strategy='kmeans')
    abnormal_short_term_variability = model.fit_transform(abnormal_short_term_variability)

    model = KBinsDiscretizer(n_bins=mean_value_of_short_term_variability_bins, encode='ordinal', strategy='kmeans')
    mean_value_of_short_term_variability = model.fit_transform(mean_value_of_short_term_variability)

    model = KBinsDiscretizer(n_bins=percentage_of_time_with_abnormal_long_term_variability_bins, encode='ordinal', strategy='kmeans')
    percentage_of_time_with_abnormal_long_term_variability = model.fit_transform(percentage_of_time_with_abnormal_long_term_variability)

    model = KBinsDiscretizer(n_bins=mean_value_of_long_term_variability_bins, encode='ordinal', strategy='kmeans')
    mean_value_of_long_term_variability = model.fit_transform(mean_value_of_long_term_variability)

    model = KBinsDiscretizer(n_bins=histogram_width_bins, encode='ordinal', strategy='kmeans')
    histogram_width = model.fit_transform(histogram_width)

    model = KBinsDiscretizer(n_bins=histogram_min_bins, encode='ordinal', strategy='kmeans')
    histogram_min = model.fit_transform(histogram_min)

    model = KBinsDiscretizer(n_bins=histogram_max_bins, encode='ordinal', strategy='kmeans')
    histogram_max = model.fit_transform(histogram_max)

    model = KBinsDiscretizer(n_bins=histogram_number_of_peaks_bins, encode='ordinal', strategy='kmeans')
    histogram_number_of_peaks = model.fit_transform(histogram_number_of_peaks)

    model = KBinsDiscretizer(n_bins=histogram_number_of_zeroes_bins, encode='ordinal', strategy='kmeans')
    histogram_number_of_zeroes = model.fit_transform(histogram_number_of_zeroes)

    model = KBinsDiscretizer(n_bins=histogram_mode_bins, encode='ordinal', strategy='kmeans')
    histogram_mode = model.fit_transform(histogram_mode)

    model = KBinsDiscretizer(n_bins=histogram_median_bins, encode='ordinal', strategy='kmeans')
    histogram_median = model.fit_transform(histogram_median)

    model = KBinsDiscretizer(n_bins=histogram_variance_bins, encode='ordinal', strategy='kmeans')
    histogram_variance = model.fit_transform(histogram_variance)

    model = KBinsDiscretizer(n_bins=histogram_tendency_bins, encode='ordinal', strategy='kmeans')
    histogram_tendency = model.fit_transform(histogram_tendency)

    df = pd.DataFrame()
    df['status'] = features[:, 21].astype(int)
    df['baseline_value'] = baseline_value.astype(int)
    df['accelerations'] = accelerations.astype(int)
    df['fetal_movement'] = fetal_movement.astype(int)
    df['uterine_contractions'] = uterine_contractions.astype(int)
    df['light_decelerations'] = light_decelerations.astype(int)
    df['severe_decelerations'] = severe_decelerations.astype(int)
    df['prolongued_decelerations'] = prolongued_decelerations.astype(int)
    df['abnormal_short_term_variability'] = abnormal_short_term_variability.astype(int)
    df['mean_value_of_short_term_variability'] = mean_value_of_short_term_variability.astype(int)
    df['percentage_of_time_with_abnormal_long_term_variability'] = percentage_of_time_with_abnormal_long_term_variability.astype(int)
    df['mean_value_of_long_term_variability'] = mean_value_of_long_term_variability.astype(int)
    df['histogram_width'] = histogram_width.astype(int)
    df['histogram_min'] = histogram_min.astype(int)
    df['histogram_max'] = histogram_max.astype(int)
    df['histogram_number_of_peaks'] = histogram_number_of_peaks.astype(int)
    df['histogram_number_of_zeroes'] = histogram_number_of_zeroes.astype(int)
    df['histogram_width'] = histogram_width.astype(int)
    df['histogram_mode'] = histogram_mode.astype(int)
    df['histogram_median'] = histogram_median.astype(int)
    df['histogram_variance'] = histogram_variance.astype(int)
    df['histogram_tendency'] = histogram_tendency.astype(int)

    new_file_name = 'fetal_health_discretized.csv'
    df.to_csv('./'+new_file_name,header=True)

    return new_file_name