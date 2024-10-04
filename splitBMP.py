import os
import random
import argparse
import cv2
import easyocr
import shutil
import re
# A script to generate training, test, and validation samples. 
# Allow the sets to be constrained by choice of language 
# (Thai, English, or both). 
# Selection of language
languages = {
    'thai': ['th'],
    'english': ['en'],
    'both': ['en', 'th']
}

def get_bmp_files(folder, language, dpi, style):
    bmp_files = []
    # digit_folder_pattern = re.compile(r'\d+')
    for root, _, files in os.walk(folder):
        # if digit_folder_pattern.search(os.path.basename(root)):
        #    continue
        for file in files:
            if file.lower().endswith('.bmp'):
                # Construct the expected path pattern
                if dpi and style and language:
                    # Check if the file path contains the specified dpi and style
                    if dpi in root and style in root and language.title() in root:
                        bmp_files.append(os.path.join(root, file))
                        # print(f"# Reading dpi&&style... {file}", end='\r')
                elif dpi:
                    if dpi in root:
                        bmp_files.append(os.path.join(root, file))
                        # print(f"# Reading dpi... {file}", end='\r')
                elif style:
                    if style in root:
                        bmp_files.append(os.path.join(root, file))
                        # print(f"# Reading style... {file}", end='\r')
                else:
                    bmp_files.append(os.path.join(root, file))
                    # print(f"# Reading all...{file}", end='\r')

                # print(f"# Reading... {file}", end='\r')
    print("\nReading BMP files finished")

    # test whether the file is loading correctly according to dpi and style
    # for file in bmp_files:
    #   print(file)
    return bmp_files

def get_test_bmp_files(folder, language, dpi, style):
    bmp_files = []
    # digit_folder_pattern = re.compile(r'\d+')
    for root, _, files in os.walk(folder):
        # if digit_folder_pattern.search(os.path.basename(root)):
        #    continue
        # print(root)
        for file in files:
            if file.lower().endswith('.bmp'):
                # Construct the expected path pattern
                if dpi and style and language:
                    # Check if the file path contains the specified dpi and style
                    if dpi in root and style in root and language.title() in root:
                        bmp_files.append(os.path.join(root, file))
                        # print(f"# Reading dpi&&style... {file}", end='\r')
                elif dpi:
                    if dpi in root:
                        bmp_files.append(os.path.join(root, file))
                        # print(f"# Reading dpi... {file}", end='\r')
                elif style:
                    if style in root:
                        bmp_files.append(os.path.join(root, file))
                        # print(f"# Reading style... {file}", end='\r')
                else:
                    bmp_files.append(os.path.join(root, file))
                    # print(f"# Reading all...{file}", end='\r')

                # print(f"# Reading... {file}", end='\r')
    print("\nReading test_BMP files finished")

    # test whether the file is loading correctly according to dpi and style
    # for file in bmp_files:
    #   print(file)
    return bmp_files

def generate_samples(input_folder, output_folder, language_choice, dpi, style, test_language, test_dpi, test_style):
    print(input_folder)
    # Get all BMP files
    files = []
    for folder in input_folder:
        files.extend(get_bmp_files(folder, language_choice, dpi, style))
    random.shuffle(files)
    # print(files)
    # Get all test BMP files
    test_files = []
    for folder in input_folder:
        test_files.extend(get_test_bmp_files(folder, test_language, test_dpi, test_style))
    random.shuffle(test_files)
    # print(test_files)
    # Counting the number of samples
    total_files = len(files)
    train_count = int(total_files * 0.7)
    val_count = int(total_files * 0.15)

    # Splitting files into train, test, validation sets
    train_files = files[:train_count]
    val_files = files[train_count:train_count + val_count]
    test_files = test_files[train_count + val_count:]

    print(f"Start samples processing")
    # Processing train val samples
    for file_set, folder_name in zip([train_files, val_files], ['train', 'val']):
        folder_path = os.path.join(output_folder, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        for file in file_set:
            # Save
            bmp_file_path = os.path.join(folder_path, f"{os.path.basename(file).split('.')[0]}.bmp")
            with open(bmp_file_path, 'w', encoding='utf-8') as bmp_file:
                shutil.copy(file, bmp_file_path)
            # print(f"#OCR file{file}", end='\r')
        print(f"\n#Divide samples processing...{folder_name}", end='')
    # Processing test samples    
    for file_set, folder_name in zip([test_files], ['test']):
        folder_path = os.path.join(output_folder, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        for file in file_set:
            # Save
            bmp_file_path = os.path.join(folder_path, f"{os.path.basename(file).split('.')[0]}.bmp")
            with open(bmp_file_path, 'w', encoding='utf-8') as bmp_file:
                shutil.copy(file, bmp_file_path)
            # print(f"#OCR file{file}", end='\r')
        print(f"\n#Test samples processing...{folder_name}", end='')    
    print(f"Generate Success")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate training, testing, and validation samples')
    parser.add_argument('--input_folder', type=str, help='Folder path containing the BMP file')
    parser.add_argument('--output_folder', type=str, help='Output folder path to save the generated samples')
    parser.add_argument('--language', type=str, choices=['thai', 'english', 'both'], help='Select recognition language')
    parser.add_argument('--dpi', type=int, help='DPI of the images to filter')
    parser.add_argument('--style', type=str, help='Font style of the images to filter')
    parser.add_argument('--test_language', type=str, choices=['thai', 'english', 'both'], help='Select recognition language')
    parser.add_argument('--test_dpi', type=int, help='DPI of the images to filter')
    parser.add_argument('--test_style', type=str, help='Font style of the images to filter')

    args = parser.parse_args()
    input_folders = []
    # if args.language == 'thai':
    input_folders.append(args.input_folder)
    # elif args.language == 'english':
    #     input_folders.append(args.english_folder)
    # elif args.language == 'both':
    #     input_folders.append(args.thai_folder)
    #     input_folders.append(args.english_folder)
    print(input_folders)
    print("start generate samples")
    generate_samples(input_folders, args.output_folder, args.language, str(args.dpi), args.style, args.test_language, str(args.test_dpi), args.test_style)
