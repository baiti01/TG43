#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: Ti Bai (part of the code is from Dr. Sean Domal)
# Email: tibaiw@gmail.com
# datetime:3/19/2025

import streamlit as st
import pydicom
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, interp2d


class BrachyDoseCalcultion_TG43():
    def __init__(self, RTPlan_path,
                 source_length,
                 source_data_F_path='Flexisource.xlsx',
                 source_data_g_path='Flexisource.xlsx',
                 dose_rate_constant=1.1):
        self.RTPlan = pydicom.dcmread(RTPlan_path)
        self.souce_length = source_length
        self.radial_dose_function = pd.read_excel(source_data_g_path, sheet_name='g')
        self.anisotropy_function = pd.read_excel(source_data_F_path, sheet_name='F')
        self.standard_dose_template = self.get_standard_dose_template(self.radial_dose_function,
                                                                      self.anisotropy_function,
                                                                      dose_rate_constant,
                                                                      source_length)
        self.dose_interpolator = interp2d(self.standard_dose_template['grid_x'],
                                          self.standard_dose_template['grid_y'],
                                          self.standard_dose_template['template'],
                                          kind='linear')
        self.source_strength = self.RTPlan.SourceSequence[0].ReferenceAirKermaRate
        self.catheters = self.get_catheters()
        self.reference_points = self.get_reference_points()

    def get_point_dose(self, coordinates):
        accumulated_dose = 0
        for current_catheter in self.catheters:
            for index, current_source in current_catheter.iterrows():
                # Calculate relative position of the point to the source
                current_source_center = np.array([current_source['x'], current_source['y'], current_source['z']])
                current_source_vector = np.array(
                    [current_source['source_end_2'] - current_source['source_end_1']]).squeeze()
                current_source_vector_normalized = current_source_vector / np.linalg.norm(current_source_vector)

                # Projection of point vector onto the source vector
                current_point_vector = np.array(coordinates) - current_source_center
                current_source_vector_x_projection = np.dot(current_point_vector, current_source_vector_normalized)
                current_source_vector_y_projection = np.sqrt(
                    np.linalg.norm(current_point_vector) ** 2 - current_source_vector_x_projection ** 2)

                away_distance = current_source_vector_y_projection
                along_distance = current_source_vector_x_projection

                if away_distance < 0:
                    raise ValueError('Invalid point with negative away distance!')

                # Interpolate the dose from the standard dose template
                current_dose_weighted = self.dose_interpolator(along_distance, away_distance)[0] * current_source[
                    'time'] / 3600 * self.source_strength
                accumulated_dose += current_dose_weighted
        return accumulated_dose

    def get_standard_dose_template(self,
                                   radial_dose_function,
                                   anisotropy_function,
                                   dose_rate_constant,
                                   source_length):
        # radial dose
        radial_dose_function = {'grid': radial_dose_function.iloc[:, 0].to_numpy(),
                                'value': radial_dose_function.iloc[:, 1].to_numpy()}

        # anisotropy function
        anisotropy_function = {'grid_theta': anisotropy_function.iloc[:, 0].to_numpy(),
                               'grid_r': anisotropy_function.columns[1:].to_numpy(),
                               'value': anisotropy_function.iloc[:, 1:].to_numpy()}

        # unit dose template grids
        unit_dose_grid_x = list(np.arange(7, 2, -1)) + list(np.arange(2, -2, -0.5)) + list(np.arange(-2, -8, -1))
        unit_dose_grid_y = list(np.arange(0, 1, 0.25)) + list(np.arange(1, 2, 0.5)) + list(np.arange(2, 8, 1))

        # reference geometry factor
        geometry_factor_reference = 2 * np.arctan(source_length / 2 / 1) / source_length

        # construct the standard dose distribution template
        standard_dose_distribution = np.zeros((len(unit_dose_grid_x), len(unit_dose_grid_y)))
        for current_x_index, current_x in enumerate(unit_dose_grid_x):
            for current_y_index, current_y in enumerate(unit_dose_grid_y):
                r = np.sqrt(current_x ** 2 + current_y ** 2)
                if current_x == 0:
                    theta = np.pi / 2
                elif current_y == 0 and current_x < 0:
                    theta = np.pi
                elif current_y == 0 and current_x > 0:
                    theta = 0
                else:
                    theta = np.arctan2(current_y, current_x)

                # Geometry factor calculation
                if theta == 0 or theta == np.pi:
                    geometry_factor = 1 / (r ** 2 - source_length ** 2 / 4)
                else:
                    # Calculate beta angle between vectors to source ends
                    v1 = np.array([current_x + source_length / 2, current_y])
                    v2 = np.array([current_x - source_length / 2, current_y])
                    v1_norm = v1 / np.linalg.norm(v1)
                    v2_norm = v2 / np.linalg.norm(v2)
                    beta = np.arccos(np.dot(v1_norm, v2_norm))
                    geometry_factor = beta / (source_length * r * np.sin(theta))

                g = interp1d(radial_dose_function['grid'],
                             radial_dose_function['value'],
                             kind='linear',
                             bounds_error=False,
                             fill_value='extrapolate')(r)

                F = interp2d(anisotropy_function['grid_r'],
                             anisotropy_function['grid_theta'],
                             anisotropy_function['value'],
                             kind='linear')(r, np.degrees(theta))[0]

                standard_dose_distribution[current_x_index][current_y_index] = min(
                    g * F * dose_rate_constant * geometry_factor / geometry_factor_reference, 3.85e8)
        return {'grid_x': unit_dose_grid_x, "grid_y": unit_dose_grid_y,
                'template': standard_dose_distribution.transpose()}

    def get_source_ends(self, current_point, next_point, source_length_cm):
        direction_vector = next_point - current_point
        unit_vector = direction_vector / np.linalg.norm(direction_vector)
        source_end_1 = current_point + unit_vector * (source_length_cm / 2)
        source_end_2 = current_point - unit_vector * (source_length_cm / 2)
        return source_end_1, source_end_2

    def get_source_ends_for_all_points(self, positions):
        source_ends = []
        for i in range(len(positions)):
            current_point = positions[i][:3]
            if i < len(positions) - 1:
                next_point = positions[i + 1][:3]
            else:
                next_point = positions[i][:3] + (positions[i][:3] - positions[i - 1][:3])
            source_end_1, source_end_2 = self.get_source_ends(current_point, next_point, self.souce_length)
            source_ends.append([source_end_1, source_end_2])
        return source_ends

    def get_position_time(self, current_catheter, current_total_time, current_total_time_weight):
        positions = []
        all_positions_unique = []
        for i, control_point in enumerate(current_catheter):
            raw_time_dwell = control_point.CumulativeTimeWeight - (
                current_catheter[i - 1].CumulativeTimeWeight if i > 0 else 0)
            position = np.array(control_point.ControlPoint3DPosition) / 10
            if raw_time_dwell != 0:
                time_dwell = round(10 * raw_time_dwell * current_total_time / current_total_time_weight, 2)
                positions.append(np.append(position, time_dwell / 10))
        return positions, all_positions_unique

    def get_catheters(self):
        catheters = []
        for channel in self.RTPlan.ApplicationSetupSequence[0].ChannelSequence:
            total_time = channel.ChannelTotalTime
            total_time_weight = channel.FinalCumulativeTimeWeight
            catheter = channel.BrachyControlPointSequence

            positions, all_positions_unique = self.get_position_time(catheter, total_time, total_time_weight)
            source_ends = self.get_source_ends_for_all_points(positions)

            position_df = pd.DataFrame(positions, columns=['x', 'y', 'z', 'time'])
            position_df[['source_end_1', 'source_end_2']] = pd.DataFrame(source_ends, index=position_df.index)
            position_df['allpositions'] = [all_positions_unique] * len(position_df)

            catheters.append(position_df)
        return catheters

    def get_reference_points(self):
        calc_points = []
        points = self.RTPlan.DoseReferenceSequence
        for k in points:
            try:
                calc_point = np.array(k.DoseReferencePointCoordinates) / 10
                calc_points.append((k.DoseReferenceDescription, calc_point, k.TargetPrescriptionDose * 100))
            except AttributeError:
                pass
        return calc_points


# Utility function to safely extract DICOM tag values and convert them to strings
def get_tag_value(element, tag):
    value = getattr(element, tag, "Not Available")
    return str(value)


# Streamlit page configuration and custom styling
st.set_page_config(page_title="HDR Second Calc", page_icon=":hospital:")
st.markdown("""
    <style>
        .stApp { background-color: #ffffff; }
        h1 { color: #0e4f88; text-align: center; }
        .stFileUploader { background-color: #e1e5eb; color: #0e4f88; border-color: #0e4f88; }
        .stButton>button { background-color: #0e4f88; color: #ffffff; border-radius: 5px; border: 1px solid #0e4f88; }
    </style>
    """, unsafe_allow_html=True)

st.title("TG-43 Dose Secondary Calculation")
st.write("Upload a DICOM file to begin the dose calculation process.")

# File uploader
uploaded_file = st.file_uploader("Choose a DICOM file", type=["dcm"])

if uploaded_file is not None:
    # Initialize the dose calculation engine
    dose_engine = BrachyDoseCalcultion_TG43(RTPlan_path=uploaded_file, source_length=0.35)

    # Extract patient information into a dictionary
    patient_info = {
        "PatientName": get_tag_value(dose_engine.RTPlan, "PatientName"),
        "PatientID": get_tag_value(dose_engine.RTPlan, "PatientID"),
        "PatientSex": get_tag_value(dose_engine.RTPlan, "PatientSex"),
        "StudyDate": get_tag_value(dose_engine.RTPlan, "StudyDate")
    }
    patient_df = pd.DataFrame(patient_info.items(), columns=["Attribute", "Value"])

    # Extract source information from the first source in the SourceSequence if available
    if "SourceSequence" in dose_engine.RTPlan:
        source = dose_engine.RTPlan.SourceSequence[0]
        source_info = {
            "SourceIsotopeHalfLife": get_tag_value(source, "SourceIsotopeHalfLife"),
            "SourceIsotopeName": get_tag_value(source, "SourceIsotopeName"),
            "ReferenceAirKermaRate": get_tag_value(source, "ReferenceAirKermaRate"),
            "SourceStrengthReferenceDate": get_tag_value(source, "SourceStrengthReferenceDate")
        }
    else:
        source_info = {
            "SourceIsotopeHalfLife": "Not Available",
            "SourceIsotopeName": "Not Available",
            "ReferenceAirKermaRate": "Not Available",
            "SourceStrengthReferenceDate": "Not Available"
        }
    source_df = pd.DataFrame(source_info.items(), columns=["Attribute", "Value"])

    # Display the patient and source information tables
    st.markdown("### Patient Information")
    st.table(patient_df)

    st.markdown("### Source Information")
    st.table(source_df)

    # Automatically calculate and display dose for reference points
    st.markdown("### Dose Calculation for Reference Points")
    results_data = []
    for current_point_name, current_point_coordinates, current_reference_dose in dose_engine.reference_points:
        current_point_dose = dose_engine.get_point_dose(current_point_coordinates)
        difference = (current_point_dose - current_reference_dose) / current_reference_dose
        results_data.append({
            "Point Name": current_point_name,
            "Position (cm)": f"({current_point_coordinates[0]:.2f}, {current_point_coordinates[1]:.2f}, {current_point_coordinates[2]:.2f})",
            "Calculated Dose (cGy)": f"{current_point_dose:.2f}",
            "Reference Dose (cGy)": f"{current_reference_dose:.2f}",
            "Difference (%)": f"{difference * 100:.2f}%"
        })
    st.table(pd.DataFrame(results_data))

    # Section for custom position dose calculation
    st.markdown("### Calculate Dose at Custom Position")
    col1, col2, col3 = st.columns(3)
    with col1:
        custom_x = st.number_input("X (cm)", value=0.0, format="%.2f")
    with col2:
        custom_y = st.number_input("Y (cm)", value=0.0, format="%.2f")
    with col3:
        custom_z = st.number_input("Z (cm)", value=0.0, format="%.2f")

    if st.button("Calculate Custom Dose"):
        try:
            custom_dose = dose_engine.get_point_dose([custom_x, custom_y, custom_z])
            st.write(
                f"Calculated Dose at ({custom_x:.2f}, {custom_y:.2f}, {custom_z:.2f}) is: **{custom_dose:.2f} cGy**")
        except Exception as e:
            st.error(f"Error in calculation: {e}")

    st.write("Calculation complete. Check the tables above.")
