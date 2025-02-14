# data_acquisition.py

from FlightTest import FlightTest
import constants as cs
from mat4py import loadmat
from tqdm import tqdm
import pandas as pd
import os


def generate_tests_csv():
    for test in cs.FLIGHT_TESTS:
        generate_test_csv(test)
        
def generate_test_csv(test):
    for n in tqdm(range(1, cs.NUM_CHANNELS), desc=f"Create csv files for the flight test {test.value}", unit="file"):
        path_channel = os.path.join(cs.PATH_FOLDER_DATASET, test.value, f"Channel{n}.mat")
        df = pd.DataFrame(loadmat(path_channel))
        path_channel_csv = os.path.join(cs.PATH_FOLDER_TIMESERIES, test.value, f"Channel{n}.csv")
        df.to_csv(path_channel_csv, index=False)

def generate_flight_tests():
    flight_tests = []
    for test in cs.FLIGHT_TESTS:
        # path header
        path_header = os.path.join(cs.PATH_FOLDER_DATASET, test.value, cs.FILENAME_HEADER)
        header = loadmat(path_header)['header']
        # create flight test instance
        flight = FlightTest(
            id=test.value,
            version=header['Version'],
            number_of_values=header['NumberOfValues'],
            number_of_channels=header['NumberOfChannels'],
            block_size=header['BlockSize'],
            proc_method=header['ProcMethod'],
            acq_mode=header['AcqMode'],
            center_frequency=header['CenterFrequency'],
            span=header['Span'],
            sample_interval=header['SampleInterval'], 
            percent_real_time=header['PercentRealTime'],
            date=header['Date'],
            time=header['Time'],
            scale_factors=header['ScaleFactors'],
            overloads=header['Overloads'],
            ranges=header['Ranges'],
            names=header['Names'],
            nano_sec_delay=header['NanoSecDelay'],
            units=header['Units'],
            couplings=header['Couplings'],
            cutoff_freq=header['CutoffFreq'],
            timeseries=os.path.join(cs.PATH_FOLDER_TIMESERIES, test.value)
        )
        flight_tests.append(flight)
    return flight_tests




















