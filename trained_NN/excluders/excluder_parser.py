import numpy as np

def get_camera_ids(excluder, fids):
    if excluder=='VeRi':
        if isinstance(fids, str):
            return fids.split("/")[1].split("_")[1]
        else:
            return np.array([ x.split("/")[1].split("_")[1] for x in fids])
    elif excluder=='VehicleReId':
        if isinstance(fids, str):
            return fids.split("/")[0]
        else:
            return np.array([x.split("/")[0] for x in fids])