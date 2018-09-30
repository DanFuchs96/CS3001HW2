#!/usr/bin/env python3
# encoding: utf-8
"""
@author: Daniel Fuchs

CS3001: Data Science - Homework #2 Module (Summary Statistics)
"""


def df_build_summary_stats(list_vals):
    list_vals = sorted(list_vals)
    return [df_max(list_vals),
            df_min(list_vals),
            df_mean(list_vals),
            df_median(list_vals),
            df_deviation(list_vals),
            df_percentile(list_vals, 75),
            df_percentile(list_vals, 25)]


def df_print_summary_stats(list_vals, show_values=True, count_values=False):
    list_vals = sorted(list_vals)
    stats = df_build_summary_stats(list_vals)
    if show_values:
        print("Values:", list_vals)
    if count_values:
        print("Count:", len(list_vals), "Entries")
    print("Max:", stats[0])
    print("Min:", stats[1])
    print("Mean:", stats[2])
    print("Median:", stats[3])
    print("Sample Standard Deviation:", stats[4])
    print("75th Percentile:", stats[5])
    print("25th Percentile:", stats[6])


def df_max(list_vals):
    if not list_vals:
        return None
    else:
        curr_max = list_vals[0]
        for element in list_vals:
            if element > curr_max:
                curr_max = element
        return curr_max


def df_min(list_vals):
    if not list_vals:
        return None
    else:
        curr_min = list_vals[0]
        for element in list_vals:
            if element < curr_min:
                curr_min = element
        return curr_min


def df_mean(list_vals):
    if not list_vals:
        return None
    else:
        mean = 0
        for element in list_vals:
            mean += element
        mean /= len(list_vals)
        return mean


def df_deviation(list_vals):
    if not list_vals:
        return None
    elif len(list_vals) <= 1:
        return 0
    else:
        sum_sqs = 0
        mean = df_mean(list_vals)
        for element in list_vals:
            sum_sqs += ((element - mean) ** 2)
        return (sum_sqs / (len(list_vals) - 1)) ** 0.5
    # Note, this is using the sample standard deviation formula


def df_median(list_vals):
    if not list_vals:
        return None
    else:
        size = len(list_vals)
        mid = size // 2
        if size % 2:
            return list_vals[mid]
        else:
            return df_mean(list_vals[mid-1:mid+1])


def df_percentile(list_vals, percent=None):
    if not list_vals:
        return None
    elif not percent:
        return None
    elif percent != 25 and percent != 50 and percent != 75:
        return None
    else:
        if percent == 50:
            return df_median(list_vals)
        else:
            size = len(list_vals)
            mid = size // 2
            if size % 2:
                if percent == 25:
                    return df_median(list_vals[:mid])
                if percent == 75:
                    return df_median(list_vals[mid+1:])
            else:
                if percent == 25:
                    return df_median(list_vals[:mid])
                if percent == 75:
                    return df_median(list_vals[mid:])


def df_scaled_int_count(list_vals, range_max):
    valid_ints = []
    scaled_vals = []
    for _ in range(range_max + 1):
        valid_ints.append(0)
    norm_max = df_max(list_vals)
    for value in list_vals:
        scaled_vals.append(value * range_max / norm_max)
    for value in scaled_vals:
        valid_ints[round(value)] += 1
    return valid_ints


def df_vec_distance(vec):
    result = 0
    for value in vec:
        result += value ** 2
    return result ** 0.5
