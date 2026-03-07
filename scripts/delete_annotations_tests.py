from utils.annotations import get_missing_video_annotation_data_info
from utils.files import delete_annotations_missing_information

def main():
    missing_data = get_missing_video_annotation_data_info()
    delete_annotations_missing_information(missing_data)

if __name__ == "__main__":
    main()