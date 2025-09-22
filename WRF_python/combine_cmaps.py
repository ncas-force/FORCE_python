def combine_cmaps(cmaps, name="combo", sat_scales=(1.0, 1.0), n=256, exp=4.0, cutoff=0.2, reverse_first=False, portions=(0.666, 0.334)):  # fraction of colors for each section

    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib import colors

    def adjust(cmap, sat_scale, n_colors):
        x = np.linspace(0, 1, n_colors)

        rgb = cmap(x)[:, :3]
        hsv = colors.rgb_to_hsv(rgb)

        # r = ramp scaled to fraction of this section
        r = np.linspace(0, 1, n_colors)

        if sat_scale !=  1.0:

            hsv[:, 1] = np.clip(hsv[:, 1] * (1 - sat_scale), 0, 1)

        return colors.hsv_to_rgb(hsv)

    # compute number of colors for each section
    n_colors_list = [int(np.round(p * n)) for p in portions]

    all_colors = []

    for i, (s, n_sec, cmap) in enumerate(zip(sat_scales, n_colors_list, cmaps)):
        if i == 0 and reverse_first=="True":
            temp_cols = adjust(cmap, s, n_sec)
            all_colors.extend(temp_cols[::-1])
        else:
            all_colors.extend(adjust(cmap, s, n_sec))

    color_subset = all_colors[127:255]

    new_all_colors = np.array([color_subset[int(i * (len(color_subset)-1)/255)] for i in range(256)])

    return LinearSegmentedColormap.from_list(name, new_all_colors)

