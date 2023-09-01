import os
import sys
import igl
import csv

root_folder = os.getcwd()
sys.path.insert(1, "../")

from Mollification import IntrinsicMollificationConstant
from ex_lscm import *

input_path = "/home/ikaroruan/Downloads/Obj_Files/Obj_Files/Artist_UVs/Cut/"

def main():
    results = []
    results.append(["Filename", "Flipped_Original",
                    "Flipped_Neg", "Flipped_Nzero",
                    "Flipped_mol"])

    print("Checking number of flipped triangles...")
    counter = 1
    for file in os.listdir(input_path):
        f = os.path.join(input_path, file)
        V, F = igl.read_triangle_mesh(os.path.join(root_folder, ".", f))

        print("Current file is: ", os.path.basename(f))
        print("File ", counter, " of ", len(os.listdir(input_path)))
        counter += 1

        if(F.ndim == 1):
            print("Ignoring files with F dimension 1.")
            continue
        
        if(igl.boundary_loop(F).size == 0):
            print("Ignoring file, there is no boundary loop.")
            continue

        if(os.stat(f).st_size/(1024*1024) >= 20):
            print("Ignoring file, too big.")
            continue

        # LSCM: No hacks, no mollification.
        try:
            done, vuv_orig = lscm(V, F,
                              mollified=False,
                              neg_hack=NEG_HACK.NONE,
                              nan_hack=NAN_HACK.NONE,
                              close_zero_hack=CLOSE_TO_ZERO_HACK.NONE)
        except:
            print("Some error ocurred. Ignoring file.")
            continue

        # LSCM: Replace negatives with absolute value.
        try:
            done, vuv_neg = lscm(V, F, 
                             mollified=False,
                             neg_hack=NEG_HACK.TO_ABS,
                             nan_hack=NAN_HACK.NONE,
                             close_zero_hack=CLOSE_TO_ZERO_HACK.NONE)
        except:
            print("Some error ocurred. Ignoring file.")
            continue

        # LSCM: Replace close to zero with zero.
        try:
            done, vuv_nzero = lscm(V, F,
                               mollified=False,
                               neg_hack=NEG_HACK.NONE,
                               nan_hack=NAN_HACK.NONE,
                               close_zero_hack=CLOSE_TO_ZERO_HACK.TO_ZERO)
        except:
            print("Some error ocurred. Ignoring file.")
            continue

        # LSCM: Mollified.
        try:
            done, vuv_mol = lscm(V, F,
                             mollified=True,
                             neg_hack=NEG_HACK.NONE,
                             nan_hack=NAN_HACK.NONE,
                             close_zero_hack=CLOSE_TO_ZERO_HACK.NONE)
        except:
            print("Some error ocurred. Ignoring file.")
            continue

        num_orig = igl.flipped_triangles(vuv_orig, F).size
        num_neg = igl.flipped_triangles(vuv_neg, F).size
        num_nzero = igl.flipped_triangles(vuv_nzero, F).size
        num_mol = igl.flipped_triangles(vuv_mol, F).size

        results.append([os.path.basename(f), num_orig,
                        num_neg, num_nzero, num_mol])

        print("File processed.")

    with open("flippedTrianglesResults.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(results)

        

if __name__ == "__main__":
    main()
