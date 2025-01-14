#!/usr/bin/env python3

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import colorsys

##############################
# 1) LOAD + MERGE
##############################
def load_k12b_and_merge(filename="k12b.txt"):
    try:
        df = pd.read_csv(filename, sep=",", header=None)
    except FileNotFoundError:
        st.error(f"File '{filename}' not found.")
        st.stop()

    if df.shape[1] != 13:
        st.error(f"Expected 13 columns (Population + 12 numeric).")
        st.stop()

    df.columns = ["Population"] + [f"Comp_{i}" for i in range(1,13)]

    def merge_pop_name(p):
        return p.split("_",1)[0] if "_" in p else p
    df["MergedName"] = df["Population"].apply(merge_pop_name)
    return df

##############################
# 2) RENAME COLUMNS
##############################
K12B_RENAME_MAP = {
    "Comp_1":  "Gedrosia",
    "Comp_2":  "Siberian",
    "Comp_3":  "Northwest African",
    "Comp_4":  "Southeast Asian",
    "Comp_5":  "Atlantic Med",
    "Comp_6":  "North European",
    "Comp_7":  "South Asian",
    "Comp_8":  "East African",
    "Comp_9":  "Southwest Asian",
    "Comp_10": "East Asian",
    "Comp_11": "Caucasus",
    "Comp_12": "Sub-Saharan"
}
ADMI_COLS = list(K12B_RENAME_MAP.values())

##############################
# 3) HELPER FUNCTIONS
##############################
def generate_unique_colors(n):
    """
    Generate 'n' visually distinct colors.
    """
    hues = np.linspace(0, 1, n, endpoint=False)
    colors = []
    for h in hues:
        rgb = colorsys.hsv_to_rgb(h, 0.7, 0.9)  # Saturation=0.7, Value=0.9
        rgb = tuple(int(255 * c) for c in rgb)
        hex_color = '#%02x%02x%02x' % rgb
        colors.append(hex_color)
    return colors

##############################
# 4) MAIN
##############################
def main():
    st.set_page_config(page_title="K12b Parallel Calculator", layout="wide")
    st.title("Parallel Coordinates")

    # A) LOAD + MERGE
    df_raw = load_k12b_and_merge("k12b.txt")
    df_raw.rename(columns=K12B_RENAME_MAP, inplace=True)

    # Ensure ADMI_COLS are numeric
    df_raw[ADMI_COLS] = df_raw[ADMI_COLS].apply(pd.to_numeric, errors='coerce')

    # Handle Missing Values in ADMI_COLS
    imputer = SimpleImputer(strategy='mean')
    df_raw[ADMI_COLS] = imputer.fit_transform(df_raw[ADMI_COLS])

    # B) FILTER MERGEDNAME
    all_merged = sorted(df_raw["MergedName"].unique())
    merged_options = ["All"] + all_merged
    chosen_merged = st.multiselect("Pick ethnic groups (merged dataset) or 'All':", 
                                   merged_options, default=["All"])

    if "All" in chosen_merged:
        df_mid = df_raw.copy()
    else:
        df_mid = df_raw[df_raw["MergedName"].isin(chosen_merged)].copy()

    st.write(f"After Merging Dataset: {df_mid.shape[0]} lines.")
    if df_mid.empty:
        st.warning("No lines remain after MergedName filter.")
        return

    # C) CHOOSE REFERENCE from the **original** (or mid), so we can forcibly re-add it
    possible_refs = sorted(df_mid["Population"].unique())
    chosen_ref = st.selectbox("Reference Pop:", ["<None>"] + possible_refs)

    # Always store the reference row from the **original** dataset, if it exists
    ref_row_df = pd.DataFrame()
    if chosen_ref != "<None>":
        ref_row_df = df_raw.loc[df_raw["Population"] == chosen_ref].copy()
        if ref_row_df.empty:
            st.warning(f"Reference '{chosen_ref}' not found in df_raw?! Maybe typed incorrectly.")
            chosen_ref = "<None>"
        else:
            ref_row_df[ADMI_COLS] = imputer.transform(ref_row_df[ADMI_COLS])

    # D) STD FILTER
    st.subheader("STD Filter")
    std_threshold = st.slider("STD threshold:", 0.0, 5.0, 1.0, 0.1)
    if chosen_ref != "<None>" and std_threshold > 0:
        mid_ref = df_mid.loc[df_mid["Population"] == chosen_ref]
        mid_nonref = df_mid.loc[df_mid["Population"] != chosen_ref]

        if not mid_ref.empty:
            ref_vals = mid_ref.iloc[0]

            # Normalize all components together (min-max normalization)
            min_vals = mid_nonref[ADMI_COLS].min()
            max_vals = mid_nonref[ADMI_COLS].max()
            range_vals = max_vals - min_vals
            range_vals[range_vals == 0] = 1  # Prevent division by zero

            normalized = (mid_nonref[ADMI_COLS] - min_vals) / range_vals
            ref_normalized = (ref_vals[ADMI_COLS] - min_vals) / range_vals

            # Calculate Euclidean distance
            diff = normalized - ref_normalized
            distances = (diff ** 2).sum(axis=1).pow(0.5)  # Updated line

            # Apply threshold
            keep_mask = distances <= std_threshold
            mid_nonref = mid_nonref[keep_mask]

            # Combine filtered non-reference populations with the reference population
            df_mid = pd.concat([mid_nonref, mid_ref], ignore_index=True)

        st.write(f"**After STD filter**: {df_mid.shape[0]} lines remain.")
    else:
        if chosen_ref == "<None>":
            st.warning("No reference population selected.")
        else:
            st.warning("STD threshold is too low.")

    # 12 MIN–MAX SLIDERS
    st.subheader("Min–Max Filters (12 DNA Components)")
    for col in ADMI_COLS:
        if df_mid.empty:
            break  # If df_mid is already empty, no need to proceed
        col_min = df_mid[col].min()
        col_max = df_mid[col].max()
        if pd.isna(col_min) or pd.isna(col_max):
            continue
        slider_min = float(np.floor(col_min))
        slider_max = float(np.ceil(col_max))
        user_range = st.slider(
            f"{col} Range:",
            min_value=0.0,
            max_value=100.0,
            value=(slider_min, slider_max),
            step=1.0
        )
        low_val, high_val = user_range
        # Filter df_mid in place
        df_mid = df_mid[(df_mid[col] >= low_val) & (df_mid[col] <= high_val)]

    if df_mid.empty:
        st.warning("All rows were filtered out by min–max sliders.")
        return

    # E) CLUSTERING
    st.subheader("Clustering")
    kmeans_mode = st.checkbox("Enable K-Means Clustering?", value=False)

    color_label_to_color = {}
    unique_groups = []
    unique_clusters = []
    n_labels = 0  # Total number of unique color labels

    if not kmeans_mode:
        # Handling Non-K-Means Mode
        nonref_df = df_mid.copy()
        if chosen_ref != "<None>":
            nonref_df = df_mid[df_mid["Population"] != chosen_ref].copy()
        else:
            nonref_df = df_mid.copy()

        if chosen_ref == "<None>" and "All" in chosen_merged and len(chosen_merged) == 1:
            # Define a single group for "All"
            unique_groups = ["All"]
            group_to_label = {"All": 0}
            color_label_to_color = {0: "#87CEFA"}  # Choose a distinct color for "All"
            df_mid.loc[nonref_df.index, "ColorLabel"] = 0
            n_labels = 1
        elif not nonref_df.empty:
            unique_groups = sorted(nonref_df["MergedName"].unique())
            group_to_label = {g: i for i, g in enumerate(unique_groups)}
            df_mid.loc[nonref_df.index, "ColorLabel"] = nonref_df["MergedName"].map(group_to_label).fillna(-1).astype(int)
            unique_colors = generate_unique_colors(len(unique_groups))
            color_label_to_color = {i: unique_colors[i] for i in range(len(unique_groups))}
            n_labels = len(unique_groups)
        else:
            df_mid["ColorLabel"] = 0
            color_label_to_color = {0: "#87CEFA"}
            n_labels = 1

    else:
        if df_mid.shape[0] > 1:
            k_n = st.slider("Number of K-Means Clusters:", 2, 20, 3)
            if chosen_ref != "<None>":
                nonref_df = df_mid[df_mid["Population"] != chosen_ref].copy()
            else:
                nonref_df = df_mid.copy()

            if nonref_df.empty:
                st.warning("No non-reference populations available for clustering.")
                df_mid["ColorLabel"] = 0
                color_label_to_color = {0: "#87CEFA"}
                n_labels = 1
            else:
                X_nonref = nonref_df[ADMI_COLS].values
                if np.isnan(X_nonref).any():
                    st.warning("NaN values detected in data. Imputing missing values with column means.")
                    imputer_km = SimpleImputer(strategy='mean')
                    X_nonref = imputer_km.fit_transform(X_nonref)

                km_nonref = KMeans(n_clusters=k_n, n_init="auto", random_state=42)
                km_nonref.fit(X_nonref)
                nonref_df["KmeansCluster"] = km_nonref.labels_
                unique_clusters = sorted(nonref_df["KmeansCluster"].unique())
                cluster_to_label = {cluster: idx for idx, cluster in enumerate(unique_clusters)}

                unique_colors = generate_unique_colors(k_n)
                color_label_to_color = {idx: unique_colors[idx] for idx in range(k_n)}
                n_labels = k_n

                nonref_df["ColorLabel"] = nonref_df["KmeansCluster"].map(cluster_to_label)
                df_mid.loc[nonref_df.index, "ColorLabel"] = nonref_df["ColorLabel"]
        else:
            st.warning("Not enough lines for K-Means clustering.")
            df_mid["ColorLabel"] = 0
            color_label_to_color = {0: "#87CEFA"}
            n_labels = 1

    if chosen_ref != "<None>" and not ref_row_df.empty:
        new_label = n_labels
        df_mid.loc[df_mid["Population"] == chosen_ref, "ColorLabel"] = new_label
        color_label_to_color[new_label] = "rgb(255,255,0)"  # Bright yellow
        n_labels += 1

    combined_df = df_mid.copy()

    unique_color_labels = sorted(color_label_to_color.keys())
    n_unique_labels = len(unique_color_labels)
    if n_unique_labels == 0:
        st.error("No color labels defined.")
        st.stop()

    color_label_to_fraction = {}
    colorscale = []
    for i, label in enumerate(unique_color_labels):
        if n_unique_labels > 1:
            frac = i / (n_unique_labels - 1)
        else:
            frac = 0
        color = color_label_to_color[label]
        colorscale.extend([[frac, color], [frac, color]])
        color_label_to_fraction[label] = frac

    combined_df["ColorNumeric"] = combined_df["ColorLabel"].map(color_label_to_fraction)

    dims = []
    for col in ADMI_COLS:
        dims.append(dict(
            range=[0, 100],
            label=col,
            values=combined_df[col],
            tickvals=[0, 20, 40, 60, 80, 100],
            ticktext=["0", "20", "40", "60", "80", "100"]
        ))

    trace = go.Parcoords(
        line=dict(
            color=combined_df["ColorNumeric"],
            colorscale=colorscale,
            cmin=0,
            cmax=1,
            showscale=False
        ),
        dimensions=dims
    )

    fig = go.Figure(data=[trace])
    fig.update_layout(
        height=700,
        margin=dict(l=50, r=50, b=100, t=100),
        plot_bgcolor="white",
        font=dict(size=12)
    )

    st.plotly_chart(fig, use_container_width=True)

    # I) Show info
    st.subheader("Legend & Active Populations")

    def create_legend_entry(label, color, short_list):
        return f"""<div style='display: flex; align-items: center; margin-bottom: 5px;'>
<div style='background-color:{color}; width: 20px; height: 20px; margin-right: 10px;'></div>
<span><strong>{label}:</strong> {short_list}</span>
</div>"""

    legend_html = ""
    if not kmeans_mode:
        if unique_groups:
            legend_html += "<b>Non-Reference Groups:</b><br>"
            for group in unique_groups:
                label = group_to_label[group]
                color = color_label_to_color[label]
                subpops = nonref_df.loc[nonref_df["MergedName"] == group, "Population"].tolist()
                short_list = ", ".join(subpops)
                legend_html += create_legend_entry(group, color, short_list)
    else:
        if unique_clusters:
            legend_html += "<b>Non-Reference Clusters:</b><br>"
            for cluster in unique_clusters:
                label = cluster_to_label[cluster]
                color = color_label_to_color[label]
                subpops = nonref_df.loc[nonref_df["KmeansCluster"] == cluster, "Population"].tolist()
                short_list = ", ".join(subpops)
                legend_html += create_legend_entry(f"Cluster {cluster}", color, short_list)

    if chosen_ref != "<None>" and not ref_row_df.empty:
        reference_label = new_label
        reference_color = color_label_to_color[reference_label]
        legend_html += "<br><b>Reference Population:</b><br>"
        legend_html += create_legend_entry("Reference", reference_color, chosen_ref)

    st.markdown(legend_html, unsafe_allow_html=True)

    if not kmeans_mode:
        active_pops = sorted(nonref_df["Population"].unique())
    else:
        active_pops = sorted(nonref_df["Population"].unique())

    if chosen_ref != "<None>" and not ref_row_df.empty:
        active_pops.append(chosen_ref)

    st.subheader(f"Currently Active Populations ({len(active_pops)} total)")
    st.text_area("Active Populations", "\n".join(active_pops), height=150)

    st.subheader("Summary Stats")
    if chosen_ref != "<None>" and not ref_row_df.empty:
        summary_df = pd.concat([nonref_df, ref_row_df], ignore_index=True)
    else:
        summary_df = nonref_df.copy()
    st.dataframe(summary_df[ADMI_COLS].describe())

if __name__=="__main__":
    main()
