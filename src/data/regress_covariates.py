from nilearn.image import smooth_img, new_img_like, load_img
import os
import numpy as np
import glob
import pandas as pd


from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression

def multivariate_regression(X,y):
    regr_multireg =  LinearRegression()
    regr_multireg.fit(X, y)
    y_pred = regr_multireg.predict(X)
    residuals = y - y_pred
    return residuals


def load_all_img(data_dir,file_paths):
    data = []
    #N = len(file_paths)
    N=10
    for index, image_path in enumerate(file_paths[:N]):
        print(f"Processing {index/len(file_paths)}...")
        # load image and remove nan and inf values.
        path = os.path.join(data_dir, image_path)
        image = smooth_img(path, fwhm=None)
        data.append(image.get_fdata())
        shape = image.shape
    data_matrix = np.array(data)
    # flatten
    print(f"Old shape: {data_matrix.shape}")
    data_matrix = np.reshape(data_matrix, (N,-1))
    print(f"New shape: {data_matrix.shape}")
    return data_matrix, shape


def save_all_img(data_dir,file_paths,residuals):

    for index, image_path in enumerate(file_paths):
        path = os.path.join(data_dir, image_path)
        image = smooth_img(path, fwhm=None)
        new_path = path.split('.')[0] + "_r.mgz"
        # scale into positive values
        residual_img = residuals[index] - np.min(residuals[index])
        new_image = new_img_like(image, residual_img.astype(np.uint8))
        new_image.to_filename(new_path)


def get_files_and_X(data_dir,label,covariate_names):
    df = pd.read_csv(os.path.join(data_dir, 'data_split.csv'))

    # remove any missing rows
    df = df.replace(r'^\s*$', np.nan, regex=True)
    df = df.dropna(axis=0,how='any')

    file_paths = list(df["filename"].values)
    labels = list(df[label].values)

    for c in covariate_names:
        classes = df[c].unique()
        numbers = np.arange(len(classes))
        df[c] = df[c].replace({c:n for c,n in zip(classes,numbers)})

    covariates = df[covariate_names].to_numpy()
    return file_paths, labels, covariates



def main(test=False):
    # pl.seed_everything(1234)

    # ------------
    # args
    # ------------

    # parser = ArgumentParser()
    # parser.add_argument('--data_dir', type=str,
    #                     default='/Users/sean/Projects/MRI_Deep_Learning/Kamran_Montreal_Data_Share/')
    # parser.add_argument('--batch_size', default=4, type=int)
    # parser.add_argument('--num_classes', type=int, default=2)
    # parser.add_argument('--num_workers', type=int, default=0)
    # parser.add_argument('--format', type=str, default='nifti')
    # parser.add_argument('--test', type=bool, default=True)
    # parser = pl.Trainer.add_argparse_args(parser)
    # # parser = DL1Classifier.add_model_specific_args(parser)
    # args = parser.parse_args()

    #data_dir = '/scratch/spinney/enigma/'
    data_dir = '/Users/sean/Projects/MRI_Deep_Learning/Kamran_Montreal_Data_Share/'
    label = "class"
    covariate_names = ['sex', 'age', 'study']
    file_paths, labels, X = get_files_and_X(data_dir,label,covariate_names)
    #N = len(file_paths)
    N=10
    mask = ''

    y, shape = load_all_img(data_dir, file_paths)
    residuals = multivariate_regression(X[:N], y)

    print(f"Dimensions of flat residuals: {residuals.shape}")
    # reshape
    residuals_3d = np.reshape(residuals,(N,shape[0],shape[1],shape[2]))

    print(f"Dimensions of 3d residuals: {residuals_3d.shape}")

    save_all_img(data_dir,file_paths[:N],residuals_3d)
    print(f"Saved all regressed images with _r.mgz siffix.")

if __name__ == '__main__':
    main(test=True)
