from utils.annotations import get_missing_video_annotation_data_info

def main():
    missing_data = get_missing_video_annotation_data_info()
    print(missing_data)

if __name__ == "__main__":
    main()