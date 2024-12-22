import numpy as np
import utils
import amc_parser
npy_file = "motion_extractions_old/backflip_40/0Tq2Izcbclw_000000_000010/landmarkPositions.npy"
npy_data = np.load(npy_file)

amc_data = amc_parser.load_amc("motion_extractions_old/backflip_40/0Tq2Izcbclw_000000_000010/landmarkPositions.amc")
asf_data = amc_parser.load_asf("motion_extractions_old/backflip_40/0Tq2Izcbclw_000000_000010/landmarkPositions.asf")
print(npy_data.shape)
print(utils.skeleton_edges)