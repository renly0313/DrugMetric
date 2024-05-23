import streamlit as st
from utils import (
    smiles_to_mol,
    mol_file_to_mol,
    draw_molecule,
    mol_to_tensor_graph,
    get_model_predictions,
    RealQEDscores,
)
from cal_pro import mol_to_feather
from PIL import Image
import os

# Constants
LOGO_PATH = "/home/dell/wangzhen/RealQED(2.17)/web server/complete_logo.png"
PAGE_OPTIONS = ["Predictor", "Model analysis"]

# Custom CSS
st.markdown(
    """
<style>
    .reportview-container {
        background-color: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
    }
    h1 {
        color: #1f77b4;
    }
    h2 {
        color: #1f77b4;
    }
</style>
""",
    unsafe_allow_html=True,
)

def main():
    st.markdown(
        "<h1 style='text-align: center; font-size: 30px;'>ChemAIrank: A Quantitative Drug-likeness Scoring Model Based on Chemical Space Distances for Accelerating Drug Discovery</h1>",
        unsafe_allow_html=True,
    )
    setup_sidebar()
    page = st.sidebar.selectbox("Page Navigation", PAGE_OPTIONS)

    if page == "Predictor":
        process_predictor_page()
    else:
        st.markdown("This page is not implemented yet :no_entry_sign:")

def setup_sidebar():
    st.sidebar.markdown("---")
    image = Image.open(LOGO_PATH)
    st.sidebar.image(
        image, width=150, use_column_width=True, clamp=False, channels="RGB", output_format="auto"
    )
    st.sidebar.write(
        "Created by [AIM](https://www.notion.so/AIM-6291cbd7a40a43e5ad2f942d4fe441f8)"
    )
def handle_inputs():
    valid_molecule = True
    loaded_molecule = None
    selection = None
    submit = None

    st.markdown("Select input molecule.")
    upload_columns = st.columns([2, 1])

    # File upload
    file_upload = upload_columns[0].expander(label="Upload a mol file")
    uploaded_file = file_upload.file_uploader("Choose a mol file", type=['mol'])

    # Smiles input
    smiles_select = upload_columns[0].expander(label="Specify SMILES string")
    smiles_string = smiles_select.text_input('Enter a valid SMILES string.')

    # If both are selected, give the option to swap between them
    if uploaded_file and smiles_string:
        selection = upload_columns[1].radio("Select input option", ["File", "SMILES"])

    if selection:
        if selection == "File":
            loaded_molecule = load_molecule_from_file(uploaded_file)
        elif selection== "SMILES":
            loaded_molecule = smiles_to_mol(smiles_string)
    else:
        if uploaded_file:
            loaded_molecule = load_molecule_from_file(uploaded_file)
        elif smiles_string:
            loaded_molecule = smiles_to_mol(smiles_string)

    # Set validity flag
    if loaded_molecule is None:
            valid_molecule = False
    else:
        valid_molecule = True

    if valid_molecule and loaded_molecule is not None:
        submit = upload_columns[1].button("Get predictions")

    return valid_molecule, loaded_molecule, submit
def get_prediction(loaded_molecule):
    with st.spinner(text="Fetching model prediction..."):
        # Get molecular features
        feather = mol_to_feather(loaded_molecule)
        # Model prediction of drug-likeness scores
        prediction = RealQEDscores(feather)
    return prediction
def show_molecule_image(loaded_molecule):
    pil_img = draw_molecule(loaded_molecule)
    st.image(pil_img)
def process_predictor_page():
    st.markdown("<h2>Input Molecule</h2>", unsafe_allow_html=True)
    valid_molecule, loaded_molecule, submit = handle_inputs()

    if not valid_molecule:
        st.error("This molecule appears to be invalid :no_entry_sign:")
    elif loaded_molecule is not None:
        st.info("This molecule appears to be valid :ballot_box_with_check:")
        show_molecule_image(loaded_molecule)

    if submit:
        prediction = get_prediction(loaded_molecule)
        display_prediction(prediction)

# [Previous functions]

if __name__ == "__main__":
    main()
