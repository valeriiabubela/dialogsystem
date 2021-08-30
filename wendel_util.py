#all used libs
import json
import sys, os
from datetime import datetime #for automatic updates
import requests #get data from url
#download manager

def file_update(module):
    r_path = os.getcwd()+r'/' #get current working directory
    if os.path.isfile(r_path + module + '.json'): #checks if file exist
        # Get file's Last modification time stamp
        time_stamp_file = os.path.getmtime(r_path + module + '.json')
        # Convert to readable timestamp
        mod_time_stamp = datetime.fromtimestamp(time_stamp_file).strftime('%Y-%m-%d')
        date = datetime.now().strftime('%Y-%m-%d') #get current date
        if mod_time_stamp != date: #compares date of file and current date
            #daily update
            url = 'https://api.corona-zahlen.org/' + module #stores url
            data_get = requests.get(url, allow_redirects=True) #get data
            if data_get.status_code == 200:  #checks if url exist 200 is ok, 404 not found
                print('Web site exists')
                data = open(os.path.join(r_path, (module + '.json')),'wb').write(data_get.content) #overwrite last file
                print('File Updated ')
            else: #if url not exist
                if os.path.isfile(r_path + module + '.json'): #checks if file exist
                    print("Using stored file because website doesn't exist anymore.")
        else: #if timestamp of file and date is the same
            print('Up To Date')

    else: #file not exist do first download
        url = 'https://api.corona-zahlen.org/' + module #stores url
        data_get = requests.get(url, allow_redirects=True) #gets data from url
        if data_get.status_code == 200: #checks if url exist 200 is okay, 404 not found
            print('Web site exists')
            data = open(os.path.join(r_path, (module + '.json')),'wb').write(data_get.content) #write data into url
            # Get file's time stamp 
            time_stamp_file = os.path.getmtime(r_path + module + '.json')
            print('File created ')
        else:
            print('Web site does not exist')
            if os.path.isfile(r_path + module + '.json'): #checks if file exist
                print(str(os.path.isfile(r_path + module + '.json'))+' File exists')
            else:
                print("File doesn't exist and is not downloadable")
                print('Program stopped')
                exit()

