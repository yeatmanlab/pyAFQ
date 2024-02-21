from AFQ.viz.utils import COLOR_DICT
import numpy as np
import scipy.stats as stats
import altair as alt


def altair_color_dict(names_to_include=None):
    """
    Given a list of bundle names, return a dictionary of colors for each
    Formatted for Altair.
    """
    altair_cd = dict(COLOR_DICT.copy())
    for key in list(altair_cd.keys()):
        value = altair_cd[key]
        if (names_to_include is None) or (key in names_to_include):
            altair_cd[key] = (
                f"rgb({int(value[0]*255)},"
                f"{int(value[1]*255)},"
                f"{int(value[2]*255)})")
        else:
            del altair_cd[key]
    return altair_cd


def combined_profiles_df_to_altair_df(
        profiles,
        tissue_properties=['dti_fa', 'dti_md']):
    """
    Given a profiles dataframe that is combined
    from many subjects, return a dataframe formatted for Altair.
    """
    profiles = profiles.copy()
    if 'dki_md' in tissue_properties:
        profiles.dki_md = profiles.dki_md * 1000.
    if 'dti_md' in tissue_properties:
        profiles.dti_md = profiles.dti_md * 1000.

    id_vars = ['tractID', 'nodeID', 'subjectID']
    if 'sessionID' in profiles.columns:
        id_vars.append('sessionID')

    profiles = profiles.melt(
        id_vars=id_vars,
        value_vars=tissue_properties,
        var_name='TP',
        value_name='Value')

    # Function to calculate 95% CI using a normal distribution
    def calculate_95CI(x):
        ci = stats.norm.interval(
            0.95, loc=np.mean(x), scale=np.std(x) / np.sqrt(len(x)))
        return ci

    # Group by 'tractID', 'nodeID', 'TP' and apply the aggregation functions
    profiles = profiles.groupby(['tractID', 'nodeID', 'TP'])['Value'].agg(
        mean='mean',
        CI_lower=lambda x: calculate_95CI(x)[0],
        CI_upper=lambda x: calculate_95CI(x)[1],
        IQR_lower=lambda x: x.quantile(0.25),
        IQR_upper=lambda x: x.quantile(0.75)
    ).reset_index()

    def get_hemi(cc):
        if cc == "L":
            return "Left"
        elif cc == "R":
            return "Right"
        else:
            return "Callosal"

    def get_bname(s):
        if s.startswith("Left "):
            return s[5:]
        elif s.startswith("Right "):
            return s[6:]
        return s

    def formal_tp(tp_name):
        return tp_name.upper().replace("_", " ")

    profiles["Hemi"] = profiles["tractID"].apply(lambda x: get_hemi(x[-1]))
    profiles["Bundle Name"] = profiles["tractID"].apply(get_bname)
    profiles["TP"] = profiles["TP"].apply(formal_tp)

    return profiles


def altair_df_to_chart(profiles, position_domain=(20, 80),
                       column_count=1, font_size=20,
                       line_size=10, row_label_angle=90,
                       bundle_list=None,
                       legend_line_size=5,
                       alt_x_kwargs={}, alt_y_kwargs={},
                       **kwargs):
    """
    Given a dataframe formatted for Altair, probably from
    combined_profiles_df_to_altair_df, return a chart.

    Example
    -------
        call_results = results[results.Hemi == "Callosal"]
        stand_results = results[results.Hemi != "Callosal"]
        prof_chart = altair_df_to_chart(call_results)
        prof_chart.save("supp_chart_call.png", dpi=300)
        prof_chart = altair_df_to_chart(stand_results,
            column_count=2, color="Hemi")
        prof_chart.save("supp_chart_stand.png", dpi=300)
    """
    this_cd = altair_color_dict(profiles.tractID.unique())

    alt.data_transformers.disable_max_rows()

    profiles = profiles[np.logical_and(
        profiles.nodeID >= position_domain[0],
        profiles.nodeID < position_domain[1])]

    tp_units = {
        "DKI AWF": "",
        "DKI FA": "",
        "DKI MD": " (µm²/ms)",
        "DKI MK": "",
        "DTI FA": "",
        "DTI MD": " (µm²/ms)"}

    if bundle_list is None:
        bundle_list = profiles["Bundle Name"].unique()

    row_charts = []
    for jj, b_name in enumerate(bundle_list):
        row_dataframe = profiles[profiles["Bundle Name"] == b_name]
        charts = []
        for ii, tp in enumerate(sorted(profiles.TP.unique())):
            this_dataframe = row_dataframe[row_dataframe.TP == tp]
            if jj == 0:
                title_name = tp + tp_units[tp]
            else:
                title_name = ""
            if ii == 0:
                y_axis_title = b_name
            else:
                y_axis_title = ""
            if jj == len(profiles["Bundle Name"].unique()) - 1:
                x_axis_title = "Position (%)"
                useXlab = True
            else:
                x_axis_title = ""
                useXlab = False
            y_kwargs = {
                "scale": alt.Scale(zero=False),
                "title": y_axis_title,
                **alt_y_kwargs}
            x_kwargs = {
                "axis": alt.Axis(title=x_axis_title, labels=useXlab),
                **alt_x_kwargs}
            prof_chart = alt.Chart(
                this_dataframe, title=title_name).mark_line(
                    size=line_size).encode(
                y=alt.Y('mean', **y_kwargs),
                x=alt.X('nodeID', **x_kwargs),
                **kwargs)
            prof_chart = prof_chart + alt.Chart(this_dataframe).mark_line(
                size=line_size, opacity=0.5, strokeDash=[1, 1]).encode(
                y=alt.Y('IQR_lower', **y_kwargs),
                x=alt.X('nodeID', **x_kwargs),
                **kwargs)
            prof_chart = prof_chart + alt.Chart(this_dataframe).mark_line(
                size=line_size, opacity=0.5, strokeDash=[1, 1]).encode(
                y=alt.Y('IQR_upper', **y_kwargs),
                x=alt.X('nodeID', **x_kwargs),
                **kwargs)
            prof_chart = prof_chart + alt.Chart(this_dataframe).mark_line(
                size=line_size, opacity=0.5).encode(
                y=alt.Y('CI_lower', **y_kwargs),
                x=alt.X('nodeID', **x_kwargs),
                **kwargs)
            prof_chart = prof_chart + alt.Chart(this_dataframe).mark_line(
                size=line_size, opacity=0.5).encode(
                y=alt.Y('CI_upper', **y_kwargs),
                x=alt.X('nodeID', **x_kwargs),
                **kwargs)
            charts.append(prof_chart)
        row_charts.append(alt.HConcatChart(hconcat=charts))
    return alt.VConcatChart(vconcat=row_charts).configure_axis(
        labelFontSize=font_size,
        titleFontSize=font_size,
        labelLimit=0
    ).configure_legend(
        labelFontSize=font_size,
        titleFontSize=font_size,
        titleLimit=0,
        labelLimit=0,
        columns=column_count,
        symbolStrokeWidth=legend_line_size * 10,
        symbolSize=legend_line_size * 100,
        orient='right'
    ).configure_title(
        fontSize=font_size
    )
