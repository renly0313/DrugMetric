import nglview   # conda install nglview -c conda-forge if import failure
base_pre = f"/home/dell/wangzhen/TankBind-main/examples/HTVS/"
proteinName = "7kjs"
proteinFile = f"{base_pre}/{proteinName}.pdb"
view = nglview.show_file(nglview.FileStructure(proteinFile), default=False)
view.add_representation('cartoon', selection='protein', color='white')

predictedFile = f'{base_pre}/one_tankbind.sdf'
rdkit = view.add_component(nglview.FileStructure(predictedFile), default=False)
rdkit.add_ball_and_stick(color='red')
