import os
import pickle
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

filePath = r'G:\data\CCPD2019\CCPD2019'
files = os.listdir(filePath)
print(files)
for file in files:
    if 'test' not in file:
        continue
    source = filePath + "\\" + file + "\\" + 'list.pkl'
    fr = open(source, "rb")
    result = pickle.load(fr)
    correct = 0
    car_detect = 0
    brand_detect = 0
    for i in range(len(result)):
        # print(result[i][0][0])
        id = result[i][0][0].split('-')[-3].split('_')
        brands = result[i][1]
        car_brand = ''
        car_brand += provinces[int(id[0])]
        car_brand += alphabets[int(id[1])]
        car_brand += ads[int(id[2])]
        car_brand += ads[int(id[3])]
        car_brand += ads[int(id[4])]
        car_brand += ads[int(id[5])]
        car_brand += ads[int(id[6])]
        # electral
        # car_brand += ads[int(id[7])]

        if car_brand in brands:
            correct += 1
        if len(brands) > 0:
            brand_detect +=1
    print(file)
    print("recognize rate:{}%".format(correct / len(result) * 100.))
    print("detect rate:{}%".format(brand_detect / len(result) * 100.))

"""
ccpd_base_test
recognize rate:80.7%
detect rate:100.0%
ccpd_base_test_test
recognize rate:77.9%
detect rate:99.8%
ccpd_blur_test
recognize rate:18.8%
detect rate:99.9%
ccpd_challenge_test
recognize rate:42.6%
detect rate:100.0%
ccpd_db_test
recognize rate:35.699999999999996%
detect rate:95.5%
ccpd_fn_test
recognize rate:42.6%
detect rate:100.0%
ccpd_rotate_test
recognize rate:23.5%
detect rate:98.8%
ccpd_tilt_test
recognize rate:22.0%
detect rate:95.0%
ccpd_weather_test
recognize rate:83.89999999999999%
detect rate:99.8%
"""