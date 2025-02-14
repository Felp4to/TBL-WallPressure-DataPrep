# FlightTest.py

# this class rapresents a flight test including all its features and the corresponding time series

class FlightTest:
    def __init__(self, id, version, number_of_values, number_of_channels, block_size, proc_method,
                 acq_mode, center_frequency, span, sample_interval, percent_real_time,
                 date, time, scale_factors, overloads, ranges, names,
                 nano_sec_delay, units, couplings, cutoff_freq, timeseries):
        self.id = id
        self.version = version
        self.number_of_values = number_of_values
        self.number_of_channels = number_of_channels
        self.block_size = block_size
        self.proc_method = proc_method
        self.acq_mode = acq_mode
        self.center_frequency = center_frequency
        self.span = span
        self.sample_interval = sample_interval
        self.percent_real_time = percent_real_time
        self.date = date
        self.time = time
        self.scale_factors = scale_factors
        self.overloads = overloads
        self.ranges = ranges
        self.names = names
        self.nano_sec_delay = nano_sec_delay
        self.units = units
        self.couplings = couplings
        self.cutoff_freq = cutoff_freq
        self.timeseries = timeseries
    
    def __repr__(self):
        return (f"FlightTest(ID={self.id}, Version={self.version}, NumberOfValues={self.number_of_values}, "
                f"Channels={self.number_of_channels}, BlockSize={self.block_size}, ProcMethod={self.proc_method}, "
                f"AcqMode={self.acq_mode}, CenterFrequency={self.center_frequency}, Span={self.span}, "
                f"SampleInterval={self.sample_interval}, PercentRealTime={self.percent_real_time}, Date={self.date}, "
                f"Time={self.time}, ScaleFactors={self.scale_factors}, Overloads={self.overloads}, "
                f"Ranges={self.ranges}, Names={self.names}, NanoSecDelay={self.nano_sec_delay}, "
                f"Units={self.units}, Couplings={self.couplings}, CutoffFreq={self.cutoff_freq}, timeseries={self.timeseries})")

    
    # Getters and Setters
    def get_id(self):
        return self.id
    
    def set_id(self, id):
        self.id = id
    
    def get_version(self):
        return self.version
    
    def set_version(self, version):
        self.version = version
    
    def get_number_of_values(self):
        return self.number_of_values
    
    def set_number_of_values(self, number_of_values):
        self.number_of_values = number_of_values
    
    def get_number_of_channels(self):
        return self.number_of_channels
    
    def set_number_of_channels(self, number_of_channels):
        self.number_of_channels = number_of_channels
    
    def get_block_size(self):
        return self.block_size
    
    def set_block_size(self, block_size):
        self.block_size = block_size
    
    def get_proc_method(self):
        return self.proc_method
    
    def set_proc_method(self, proc_method):
        self.proc_method = proc_method
    
    def get_acq_mode(self):
        return self.acq_mode
    
    def set_acq_mode(self, acq_mode):
        self.acq_mode = acq_mode
    
    def get_center_frequency(self):
        return self.center_frequency
    
    def set_center_frequency(self, center_frequency):
        self.center_frequency = center_frequency
    
    def get_span(self):
        return self.span
    
    def set_span(self, span):
        self.span = span
    
    def get_sample_interval(self):
        return self.sample_interval
    
    def set_sample_interval(self, sample_interval):
        self.sample_interval = sample_interval
    
    def get_percent_real_time(self):
        return self.percent_real_time
    
    def set_percent_real_time(self, percent_real_time):
        self.percent_real_time = percent_real_time
    
    def get_date(self):
        return self.date
    
    def set_date(self, date):
        self.date = date
    
    def get_time(self):
        return self.time
    
    def set_time(self, time):
        self.time = time
    
    def get_scale_factors(self):
        return self.scale_factors
    
    def set_scale_factors(self, scale_factors):
        self.scale_factors = scale_factors
    
    def get_overloads(self):
        return self.overloads
    
    def set_overloads(self, overloads):
        self.overloads = overloads
    
    def get_ranges(self):
        return self.ranges
    
    def set_ranges(self, ranges):
        self.ranges = ranges
    
    def get_names(self):
        return self.names
    
    def set_names(self, names):
        self.names = names
    
    def get_nano_sec_delay(self):
        return self.nano_sec_delay
    
    def set_nano_sec_delay(self, nano_sec_delay):
        self.nano_sec_delay = nano_sec_delay
    
    def get_units(self):
        return self.units
    
    def set_units(self, units):
        self.units = units
    
    def get_couplings(self):
        return self.couplings
    
    def set_couplings(self, couplings):
        self.couplings = couplings
    
    def get_cutoff_freq(self):
        return self.cutoff_freq
    
    def set_cutoff_freq(self, cutoff_freq):
        self.cutoff_freq = cutoff_freq

    def get_timeseries(self):
        return self.timeseries
    
    def set_timeseries(self, timeseries):
        self.timeseries = timeseries





