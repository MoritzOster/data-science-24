from prodetect import prodetect_predict

ok_path = '../../data/OK_Measurements/'
nok_path = '../../data/NOK_Measurements/'

# The ProDetect model takes the path to the AE data file as input

ok_recording1 = '/2024.02.14_22.00.10_Grinding/raw/Sampling2000KHz_AEKi-0.parquet'
nok_recording1 = '2024.02.15_02.27.22_Grinding/raw/Sampling2000KHz_AEKi-0.parquet'
results_ok = prodetect_predict(ok_path + ok_recording1)
results_nok = prodetect_predict(nok_path + nok_recording1)
print ('The ProDetect model predicts the following class for a ok example recording:')
print (results_ok)

print ('The ProDetect model predicts the following class for a nok example recording:')
print (results_nok)