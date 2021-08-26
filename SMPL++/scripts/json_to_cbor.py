import argparse
import os
import json
import cbor2

def validate_json_file_path(path):
    extension = os.path.splitext(path)[1]
    if os.path.exists(path) and extension == '.json':
        return path

def convert(path):
    with open(path, 'rb') as json_file:
        data = json.load(json_file)
    output_path = os.path.splitext(path)[0]+'.cbor'
    with open(output_path, 'wb') as cbor_file:
        cbor2.dump(data, cbor_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",
            help="input json file path",
            required = True,
            type = validate_json_file_path)
    args = parser.parse_args()
    convert(args.input)
