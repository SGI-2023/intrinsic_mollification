import os
import sys
import igl

root_folder = os.getcwd()
sys.path.insert(1, "../")
from bad_meshes import *

input_path = "/home/ikaroruan/Downloads/Obj_Files/Obj_Files/Artist_UVs/Cut/"

def output_result(filename, mesh_array):
    f = open(filename, "w")
    for mesh in mesh_array:
        f.write(mesh)
        f.write("\n")
    f.close()

def main():
    negatives = []
    close_to_zero = []

    print("Checking cotLaplace entries.\n")
    for file in os.listdir(input_path):
        f = os.path.join(input_path, file)
        V, F = igl.read_triangle_mesh(os.path.join(root_folder, ".", f))

        if(F.ndim == 1):
            print("Ignoring following file as Faces has dimension 1:")
            print(f)
            continue

        L = igl.edge_lengths(V, F)
        if cotmatrix_negative_entry(F, L):
            negatives.append(f)
        if cotmatrix_close_to_zero_entry(F, L):
            close_to_zero.append(f)

    print("Writing results.")
    output_result("negativeCotEntries.txt", negatives)
    output_result("closeToZeroCotEntries.txt", close_to_zero)

if __name__ == "__main__":
    main()
