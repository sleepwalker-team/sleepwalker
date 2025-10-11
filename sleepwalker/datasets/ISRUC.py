from __future__ import annotations
import argparse
import os
import shutil
import tempfile
import traceback
import rarfile 

import pandas as pd
import requests
from tqdm import tqdm

from sleepwalker.utils import logger

from sleepwalker.datasets.Basedataset import BaseDataset

files = [
    'subgroupI/1.rar',
    'subgroupI/2.rar',
    'subgroupI/3.rar',
    'subgroupI/4.rar',
    'subgroupI/5.rar',
    'subgroupI/6.rar',
    'subgroupI/7.rar',
    'subgroupI/8.rar',
    'subgroupI/9.rar',
    'subgroupI/10.rar',
    'subgroupI/11.rar',
    'subgroupI/12.rar',
    'subgroupI/13.rar',
    'subgroupI/14.rar',
    'subgroupI/15.rar',
    'subgroupI/16.rar',
    'subgroupI/17.rar',
    'subgroupI/18.rar',
    'subgroupI/19.rar',
    'subgroupI/20.rar',
    'subgroupI/21.rar',
    'subgroupI/22.rar',
    'subgroupI/23.rar',
    'subgroupI/24.rar',
    'subgroupI/25.rar',
    'subgroupI/26.rar',
    'subgroupI/27.rar',
    'subgroupI/28.rar',
    'subgroupI/29.rar',
    'subgroupI/30.rar',
    'subgroupI/31.rar',
    'subgroupI/32.rar',
    'subgroupI/33.rar',
    'subgroupI/34.rar',
    'subgroupI/35.rar',
    'subgroupI/36.rar',
    'subgroupI/37.rar',
    'subgroupI/38.rar',
    'subgroupI/39.rar',
    'subgroupI/40.rar',
    'subgroupI/41.rar',
    'subgroupI/42.rar',
    'subgroupI/43.rar',
    'subgroupI/44.rar',
    'subgroupI/45.rar',
    'subgroupI/46.rar',
    'subgroupI/47.rar',
    'subgroupI/48.rar',
    'subgroupI/49.rar',
    'subgroupI/50.rar',
    'subgroupI/51.rar',
    'subgroupI/52.rar',
    'subgroupI/53.rar',
    'subgroupI/54.rar',
    'subgroupI/55.rar',
    'subgroupI/56.rar',
    'subgroupI/57.rar',
    'subgroupI/58.rar',
    'subgroupI/59.rar',
    'subgroupI/60.rar',
    'subgroupI/61.rar',
    'subgroupI/62.rar',
    'subgroupI/63.rar',
    'subgroupI/64.rar',
    'subgroupI/65.rar',
    'subgroupI/66.rar',
    'subgroupI/67.rar',
    'subgroupI/68.rar',
    'subgroupI/69.rar',
    'subgroupI/70.rar',
    'subgroupI/71.rar',
    'subgroupI/72.rar',
    'subgroupI/73.rar',
    'subgroupI/74.rar',
    'subgroupI/75.rar',
    'subgroupI/76.rar',
    'subgroupI/77.rar',
    'subgroupI/78.rar',
    'subgroupI/79.rar',
    'subgroupI/80.rar',
    'subgroupI/81.rar',
    'subgroupI/82.rar',
    'subgroupI/83.rar',
    'subgroupI/84.rar',
    'subgroupI/85.rar',
    'subgroupI/86.rar',
    'subgroupI/87.rar',
    'subgroupI/88.rar',
    'subgroupI/89.rar',
    'subgroupI/90.rar',
    'subgroupI/91.rar',
    'subgroupI/92.rar',
    'subgroupI/93.rar',
    'subgroupI/94.rar',
    'subgroupI/95.rar',
    'subgroupI/96.rar',
    'subgroupI/97.rar',
    'subgroupI/98.rar',
    'subgroupI/99.rar',
    'subgroupI/100.rar',
    'subgroupII/1.rar',
    'subgroupII/2.rar',
    'subgroupII/3.rar',
    'subgroupII/4.rar',
    'subgroupII/5.rar',
    'subgroupII/6.rar',
    'subgroupII/7.rar',
    'subgroupII/8.rar',
    'subgroupIII/1.rar',
    'subgroupIII/2.rar',
    'subgroupIII/3.rar',
    'subgroupIII/4.rar',
    'subgroupIII/5.rar',
    'subgroupIII/6.rar',
    'subgroupIII/7.rar',
    'subgroupIII/8.rar',
    'subgroupIII/9.rar',
    'subgroupIII/10.rar'
]

ISRUC_URL = "http://dataset.isr.uc.pt/ISRUC_Sleep"

def download_and_extract(download_url, out_path, prefix=""):
    response = requests.get(download_url, allow_redirects=True, stream=True)

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    with tqdm(total=total_size, unit="B", unit_scale=True, desc=f"{prefix} downloading {download_url} to {out_path}") as progress_bar:
        if response.ok:
            with open(out_path, "wb") as out_f:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    out_f.write(data)
                # out_f.write(response.content)
        else:
            raise ValueError("Could not download file from URL {}. " "Received HTTP response with status code {}".format(download_url,response.status_code))
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with rarfile.RarFile(out_path) as opened_rar:
            opened_rar.extractall(temp_dir)

            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    output_dir = os.path.dirname(out_path)
                    if file.endswith(".rec"):
                        new_file_path = os.path.join(output_dir, f"{file.split('.rec')[0]}.edf")
                    else: 
                        new_file_path = os.path.join(output_dir, file)
                    shutil.move(file_path, new_file_path)

def download_dataset(out_folder, server_url, file_names):
    os.makedirs(out_folder, exist_ok=True)

    for i, file_name in enumerate(file_names):
        if "subgroupIII" in file_name:
            group = "subgroupIII"
        elif "subgroupII" in file_name:
            group = "subgroupII"
        elif "subgroupI" in file_name:
            group = "subgroupI"
        else:
            raise ValueError(f"Could not find subgroup in file {file_name}. Dont know what to do.")

        os.makedirs(os.path.join(out_folder, group), exist_ok=True)

        out_file_path = os.path.join(os.path.join(out_folder, group), file_name.split("/")[1])
        download_url = server_url + f"/{file_name}"

        download_and_extract(download_url, out_file_path, prefix=f"[{i+1}/{len(file_names)}] ")

def load_dataframe(fpath: str, event_mapping, annotator = "s1", start_date: pd.Timestamp = None):
    try:
        name = os.path.basename(fpath).split(".edf")[0]
        dfs = []
        # expected_header = ["Epoch", "Stage", "SpO2", "HR", "Events", "BPOS", "Txln", "TxEx", "Technote"]
        fname = f"{name}_{annotator}.xlsx"
        try:
            df = pd.read_excel(os.path.join(os.path.dirname(fpath), fname), engine='openpyxl')[["Epoch", "Stage"]].dropna()
        except:
            df = pd.read_excel(os.path.join(os.path.dirname(fpath), fname), engine='openpyxl', header=None, names=["Epoch", "Stage"] + [None] * 100)[["Epoch", "Stage"]].dropna()
        # if any(h not in df.columns for h in expected_header):
        #     raise ValueError(f"Header in {os.path.join(os.path.dirname(fpath), fname)} does not match expected header {expected_header}")
            # df = pd.read_excel(os.path.join(os.path.dirname(fpath), fname), engine='openpyxl', header=None)
            # df.columns = expected_header
            
        df = df.rename(columns={"Stage":"Label"})
        df["Starttime"] = df.apply(lambda row : start_date + pd.to_timedelta(f"{row['Epoch']*30} s"),axis=1)
        df["Endtime"] = df.apply(lambda row : start_date + pd.to_timedelta(f"{row['Epoch']*30} s" + pd.to_timedelta("30s")),axis=1)
        df = df[["Label", "Starttime", "Endtime"]].dropna()
        return df 
    except Exception as e:
        logger.warning(f"Error reading {fpath}. Error was {traceback.format_exc()}")
        raise e

class ISRUC(BaseDataset):
    def __init__(self, 
            annotator = ["1", "2"],
            merge = False, 
            **kwargs
        ): 
        
        if not isinstance(annotator, list):
            annotator = [annotator]
        
        if annotator is None or len(annotator) == 0:
            annotator = ["1", "2"] 
        
        self.annotator = annotator
        self.merge = merge
        
        super().__init__(**kwargs)

    def has_extra_target(self):
        return len(self.annotator) > 1 and not self.merge

    def get_extra_event_df(self, fpath: str, start_date: pd.Timestamp) -> pd.DataFrame:
        if len(self.annotator) <= 1 or self.merge:
            raise ValueError(f"This function should not have been called.")
        else:
            return load_dataframe(fpath, self.event_mapping, self.annotator[1], start_date)

    def get_event_df(self, fpath, start_date):
        if len(self.annotator) > 1 and self.merge:
            df0 = load_dataframe(fpath, self.event_mapping, self.annotator[0], start_date)
            df1 = load_dataframe(fpath, self.event_mapping, self.annotator[1], start_date)
            return pd.merge(df0, df1, on=["Label", "Starttime", "Endtime"],how="inner")
        else:
            if len(self.annotator) > 1:
                return load_dataframe(fpath, self.event_mapping, self.annotator[0], start_date)
            else:
                return load_dataframe(fpath, self.event_mapping, self.annotator, start_date)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download the ISRUC data.')
    parser.add_argument("--out", help='Folder to store the ISRUC dataset.', required=False, default=".")
    args = parser.parse_args()

    logger.info("Downloading ISRUC dataset")
    download_dataset(args.out, ISRUC_URL, files)
