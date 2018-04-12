import json
import sys

test_file_name = sys.argv[1]
gate = float(sys.argv[2])

with open(test_file_name, 'r') as f:
    test_dict = json.loads(f.read())

with open('Task1_out.txt', 'w') as fout:
    fout.write('guid/image,N\n')
    for sample_key, result_list in test_dict.items():
        carnum = 0
        for car_box in result_list:
            if car_box[4] > gate:
                carnum += 1
        fout.write(sample_key)
        fout.write(',')
        fout.write(str(carnum))
        fout.write('\n')