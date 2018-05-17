import os
import requests
import zipfile
import shutil
import matplotlib.pyplot as plt
from scipy.misc import imresize

from constants import root, raw_datafile, processed_datafile, resize_size, results_dir

def download_data():

	zipped_dataset_file = 'celebA.zip'

	if not os.path.isdir(root):
		os.mkdir(root)
	if not os.path.isdir(raw_datafile):
		os.mkdir(raw_datafile)
	if not os.path.isdir(processed_datafile):
		os.mkdir(processed_datafile)
	if not os.path.isdir(processed_datafile + 'real/'):
		os.mkdir(processed_datafile + 'real/')
	if not os.path.isdir(results_dir):
		os.mkdir(results_dir)

	# CelebA dataset on Google Drive
	file_id = '0B7EVK8r0v71pZjFTYXZWM3FlRnM'
	destination = root + zipped_dataset_file

	if not os.path.exists(destination):
		print("Downloading data from Google Drive...")
		download_file_from_google_drive(file_id, destination)

	if len(os.listdir(raw_datafile)) == 0:
		print("Unzipping data file...")
		zip_ref = zipfile.ZipFile(destination, 'r')
		zip_ref.extractall(raw_datafile)
		zip_ref.close()

	if len(os.listdir(raw_datafile)) == 1:
		print("Moving data to proper directory...")
		tmp = raw_datafile + 'img_align_celeba/'
		files = os.listdir(tmp)
		for f in files:
			shutil.move(tmp + f, raw_datafile)
		shutil.rmtree(tmp)

	if len(os.listdir(processed_datafile + 'real/')) == 0:
		print("Preprocessing the data...")

		img_list = os.listdir(raw_datafile)

		for i in range(len(img_list)):
			img = plt.imread(raw_datafile + img_list[i], format='jpeg')
			img = imresize(img, (resize_size, resize_size))

			plt.imsave(fname=processed_datafile + 'real/' + img_list[i], arr=img)

			if (i % 100) == 0:
				print('%d / %d images complete' % (i, len(img_list)), end='\r')
		print()

def download_file_from_google_drive(id, destination):
	URL = "https://docs.google.com/uc?export=download"

	session = requests.Session()

	response = session.get(URL, params = { 'id' : id }, stream = True)
	token = get_confirm_token(response)

	if token:
		params = { 'id' : id, 'confirm' : token }
		response = session.get(URL, params = params, stream = True)

	save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == "__main__":

	download_data()