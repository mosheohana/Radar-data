import numpy as np

def RMS_computation(signal):

    RMS_signal = []
    sampling_rate = 500   # Hz
    t_len_sec = 5   # In seconds
    segment_length = sampling_rate * t_len_sec
    start_segment_t = 0
    end_segment_t =  sampling_rate*t_len_sec
    n_segments = int(len(signal) / (sampling_rate*t_len_sec))
    for i in range(n_segments):
        current_segment = signal[start_segment_t:end_segment_t]
        square_values = []
        for sig_val in current_segment:
            square_values.append(sig_val**2)
        segment_RMS = np.sqrt(np.mean(square_values))
        RMS_signal.append(segment_RMS)
        
        start_segment_t += segment_length
        end_segment_t += segment_length

    RMS_signal = np.array(RMS_signal)

    return RMS_signal