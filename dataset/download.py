# Dataset downloader
import argparse
import requests
from tqdm import tqdm
import re
import os
from zipfile import ZipFile
from glob import glob
import shutil

import pdb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_directory', type=str, default='./')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    
    print('Downloading data...')
    
    shapefiles = download_shapefiles(args)
    extract_archives(shapefiles, args)
    imgfiles = download_imagery(args)
    extract_archives(imgfiles, args)
    
def download_imagery(args):
    
    imagery_path = os.path.join(args.output_directory, 'imagery')
    url_files = glob(os.path.join(imagery_path, '*.txt'))
    
    # Loop over urls
    output_filenames = []
    for url_file in tqdm(url_files, desc='Top imagery urls'):
        # Create the associated subdirectory
        url_subdir = os.path.join(imagery_path, os.path.splitext(os.path.basename(url_file))[0])
        if not os.path.exists(url_subdir):
            os.makedirs(url_subdir)
        
        with open(url_file, 'r') as url_fp:
            line_count = sum(1 for _ in url_fp)
            url_fp.seek(0)
            for curr_url in tqdm(url_fp, total=line_count, desc=os.path.basename(url_subdir) + ' urls'):
                filename = os.path.join(url_subdir, curr_url.split('/')[-1]).strip()
                output_filenames.append(filename)
                if args.overwrite or not os.path.exists(filename):
                    data = requests.get(curr_url, allow_redirects=True)
                    open(filename, 'wb').write(data.content)
                
    return output_filenames
    
def extract_archives(filenames, args):
    for file in tqdm(filenames, desc='Extracting archives'):
        extractDir = os.path.dirname(file)
        try:
            with ZipFile(file) as currZip:
                currZip.extractall(extractDir)
        except:
            print('Could not extract {}'.format(file))
        
        # Check if we created new subdirectories
        subdirs = next(os.walk(extractDir))[1]
        for subdir in subdirs:
            # Move the files out and delete the subdir
            subdir = os.path.join(extractDir, subdir)
            subdir_files = glob(os.path.join(subdir, '*'))
            for subdir_file in subdir_files:
                output_filename = os.path.join(extractDir, os.path.basename(subdir_file))
                if not os.path.exists(output_filename) or (os.path.exists(output_filename) and args.overwrite):
                    shutil.move(subdir_file, output_filename)
            shutil.rmtree(subdir)
    
def download_shapefiles(args):

    # Create the shapefile output directory if it doesn't exist
    output_directory = os.path.join(args.output_directory, 'shapefiles')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    # Get shapefile .zip urls
    shapefile_top_url = 'https://data.humdata.org/dataset/iom-bangladesh-needs-and-population-monitoring-npm-majhee-blocks-mapping'
    top_html = requests.get(shapefile_top_url)
    
    # Extract .zip download URLs
    matches = re.findall('"(http.*?\.zip)"', top_html.text)
    output_filenames = [];
    for url in tqdm(matches):
        filename = os.path.join(output_directory, url.split('/')[-1])
        output_filenames.append(filename)
        if args.overwrite or not os.path.exists(filename):
            data = requests.get(url, allow_redirects=True)
            open(filename, 'wb').write(data.content)
           
    return output_filenames
    
if __name__ == '__main__':
    main()