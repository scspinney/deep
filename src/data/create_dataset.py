import os
import glob
import pandas as pd
from argparse import ArgumentParser

"""
There are issues with the matching subject ids in the demo file
and the ones found in the filename for the mri data

e.g. Did not find subject sub-C03 in demo file. Study: ATS104

but I manually went through the list for study ATS104 and found:

subject = subjC03^NEURALMETH

which must be the same subject... Another example:

BISCUE_M87186338_mprage2-289166-1 (mri)
BISCUE_M87186338_mprage-289165-1 (demo)

it looks like the same subject with perhaps a different run number... but hard to tell.

"""

def create_dataset(main_path,demo_path):

    demo = pd.read_excel(demo_path)
    print(demo.head())

    # gather images and check if it exists in demo file
    img_paths = glob.glob(os.path.join(main_path, "**","*.mgz"))
    print(f"Found {len(list(img_paths))} mgz files.")

    found_total = 0
    found_study = {k: 0 for k in demo["Study ID"].unique()}
    study_totals = {k: 0 for k in demo["Study ID"].unique()}
    studies = study_totals.keys()
    subjects_with_no_mri = 0
    subjects = demo["Subject"].tolist()
    not_found_studies = []
    dependents = {'dep': [], 'ind': []}

    for imgp in img_paths:
        bfile = imgp.split('/')[-1][6:-4]

        study = imgp.split('/')[-2]
        found = False
        if study == "ALC109":
            bfile = bfile.split('-')[0]

        elif study == "COC113":
            bfile = bfile.split('-')[-1]

        elif study == "ATS104":
            bfile = bfile.split('-')[-1]
            bfile = f"subj{bfile}^NEURALMETH"

        elif study == "COC123":
            bfile = "-".join(bfile.split('-')[1:3])

        elif study == "COC112":
            bfile = "_".join(bfile.split('_')[1:3])

        for i, sub in enumerate(subjects):
            if str(sub) in bfile or bfile in str(sub):
                # print(f"Sub: {sub}, i: {i}, bfile: {bfile}")
                # subjects = subjects[:i] + subjects[i+1:]
                # print(len(subjects))
                found_total += 1
                # found_study[study]+=1
                found = True
                dep = demo["Dependent any drug"][demo["Subject"] == sub].values
                # print(f"{dep}")
                if dep == 1:
                    dependents["dep"].append(imgp)
                elif dep == 0:
                    dependents["ind"].append(imgp)
                else:
                    print(f"Problem finding dependency information for {imgp}.")

                break
        if not found:
            print(f"Did not find subject {bfile} in demo file. Study: {study}")
            not_found_studies.append(study)
    # study_totals[study]+=1


    print(f"Number of MRI files: {len(img_paths)}, Number matched from the demo file: {found_total}")
    print(f"Number of unmatched MRI files: {len(img_paths) - found_total}")
    print(f"Number of subjects from the demo file with no mri: {subjects_with_no_mri}")
    print(f"List of studies where mri file could not be matched in demo file: {set(not_found_studies)}")

    for study in set(not_found_studies):
        if study not in demo["Study ID"].unique():
            print(f"Study {study} does not appear to exist in the demo file.")

    print("Demographic information on retrieved data:")
    print(f"Number of dependent drug users (any drug): {len(dependents['dep'])}, Number of not dependent: {len(dependents['ind'])}")

    # write out file with dataset split
    outname = os.path.join(main_path, "data_split.csv")

    with open(outname, 'w') as out_file:
        out_file.write("filename,dependent\n")
        for row in dependents['dep']:
            parsed_line = f"{row},1\n"
            out_file.write(parsed_line)
        for row in dependents['ind']:
            parsed_line = f"{row},0\n"
            out_file.write(parsed_line)


if __name__ == '__main__':
# python create_dataset.py --main_path /Users/sean/Projects/MRI_Deep_Learning/Kamran_Montreal_Data_Share --demo_fname Mega-Analysis_demographic_data.xlsx


    parser = ArgumentParser()
    parser.add_argument('--main_path', type=str)
    parser.add_argument('--demo_fname', type=str)

    args = parser.parse_args()

    demo_path = os.path.join(args.main_path, args.demo_fname)
    create_dataset(args.main_path, demo_path)