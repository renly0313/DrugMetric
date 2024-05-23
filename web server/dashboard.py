import streamlit as st
from utils import (smiles_to_mol, mol_file_to_mol, 
                   draw_molecule, mol_to_tensor_graph, get_model_predictions, RealQEDscores)
from cal_pro import mol_to_feather
from PIL import Image
import os


# ----------- General things
st.markdown("<h1 style='text-align: center; font-size: 25px;'>ChemAIrank: A Quantitative Drug-likeness Scoring Model Based on Chemical Space Distances for Accelerating Drug Discovery</h1>", unsafe_allow_html=True)
valid_molecule = True
loaded_molecule = None
selection = None
submit = None
from PIL import Image
# ----------- Sidebar
page = st.sidebar.selectbox('Page Navigation', ["Predictor", "Model analysis"])

st.sidebar.markdown("""---""")
image = Image.open(r"/home/dell/wangzhen/RealQED(2.17)/web server/complete_logo.png")
st.sidebar.image(image, width=150, use_column_width=True, clamp=False, channels='RGB', output_format='auto')
st.sidebar.write("Created by [AIM](https://www.notion.so/AIM-6291cbd7a40a43e5ad2f942d4fe441f8)")


if page == "Predictor":
    # ----------- Inputs
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
            # Save it as temp file
            temp_filename = "temp.mol"
            with open(temp_filename, "wb") as f:
                f.write(uploaded_file.getbuffer())
            loaded_molecule = mol_file_to_mol(temp_filename)
        elif selection== "SMILES":
            loaded_molecule = smiles_to_mol(smiles_string)
    else:
        if uploaded_file:
            # Save it as temp file
            temp_filename = "temp.mol"
            with open(temp_filename, "wb") as f:
                f.write(uploaded_file.getbuffer())
            loaded_molecule = mol_file_to_mol(temp_filename)
        elif smiles_string:
            loaded_molecule = smiles_to_mol(smiles_string)

    # Set validity flag
    if loaded_molecule is None:
            valid_molecule = False
    else:
        valid_molecule = True

    # Draw if valid
    if not valid_molecule and (smiles_string != "" or uploaded_file is not None):
        st.error("This molecule appears to be invalid :no_entry_sign:")
    if valid_molecule and loaded_molecule is not None:
        st.info("This molecule appears to be valid :ballot_box_with_check:")
        pil_img = draw_molecule(loaded_molecule)
        upload_columns[1].image(pil_img)
        submit = upload_columns[1].button("Get predictions")

    # ----------- Submission
    st.markdown("""---""")
    if submit:
        # with st.spinner(text="Fetching model prediction..."):
        #     # Convert molecule to graph features
        #     graph = mol_to_tensor_graph(loaded_molecule)
        #     # Call model endpoint
        #     prediction = get_model_predictions(graph)

        with st.spinner(text="Fetching model prediction..."):

            feather = mol_to_feather(loaded_molecule)

            prediction = RealQEDscores(feather)
        # ----------- Ouputs
        outputs = st.columns([2, 1])
        outputs[0].markdown("the ChemAIrank score is : ")
        outputs[1].markdown(prediction)
else:
    st.markdown("This page is not implemented yet :no_entry_sign:")




