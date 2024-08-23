import numpy as np
import pandas as pd

def construct_fx_v1(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    slope = (y2 - y1) / (x2 - x1)
    bias = y1 - (x1 * slope)

    return (slope, bias)

def compute_y_with_fx_v2(point1, point2, x):
    x1, y1 = point1
    x2, y2 = point2
    alpha = (x2 - x) / (x2 - x1)
    y = alpha * y1 + (1 - alpha) * y2

    return round(y, 2)

def main():
    # In this case x is index of selection point range to form an interpolation
    point1 = (0, 2.2)
    point2 = (2, 2.7)
    x = 1

    slope, bias = construct_fx_v1(point1, point2)
    y_v1 = slope * x + bias
    print(f'y in v1: {y_v1:0.2f}')

    y_v2 = compute_y_with_fx_v2(point1, point2, x)
    print(f'y in v2: {y_v2}')

    s_data = pd.Series([np.nan, np.nan, 1, 6, 3, 8, np.nan, 7, np.nan, 2, np.nan], name="num_dropped")
    s_data_np = np.array(s_data.to_list())
    print(s_data)
    s_data = s_data.interpolate() # write an equation of two points
    print(s_data)

    nan_indices = np.where(np.isnan(s_data_np))[0]
    if len(nan_indices) == len(s_data_np):
        return -1
    
    while len(nan_indices) > 0:
        current_idx = nan_indices[0]
        nan_indices = np.delete(nan_indices, 0)

        nearest_not_nan_idx_on_left = np.where(~np.isnan(s_data_np[:current_idx]))[0]
        nearest_not_nan_idx_on_left = nearest_not_nan_idx_on_left[-1] if np.size(nearest_not_nan_idx_on_left) > 0 else np.nan
        
        nearest_not_nan_idx_on_right = np.where(~np.isnan(s_data_np[current_idx:]))[0]
        nearest_not_nan_idx_on_right = nearest_not_nan_idx_on_right[0] + current_idx if np.size(nearest_not_nan_idx_on_right) > 0 else np.nan

        if current_idx == 0 and not np.isnan(nearest_not_nan_idx_on_right):
            s_data_np[current_idx] = s_data_np[nearest_not_nan_idx_on_right]

        if current_idx == len(s_data_np) - 1 and not np.isnan(nearest_not_nan_idx_on_left):
            s_data_np[current_idx] = s_data_np[nearest_not_nan_idx_on_left]
        
        if not np.isnan(nearest_not_nan_idx_on_right) and not np.isnan(nearest_not_nan_idx_on_left):
            point1 = (nearest_not_nan_idx_on_left, s_data_np[nearest_not_nan_idx_on_left])
            point2 = (nearest_not_nan_idx_on_right, s_data_np[nearest_not_nan_idx_on_right])

            s_data_np[current_idx] = compute_y_with_fx_v2(point1, point2, current_idx)
        
    print(pd.Series(s_data_np, name="num_dropped"))       

if __name__ == "__main__":
    main()
